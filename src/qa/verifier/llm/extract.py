"""
_gpt_extract 包裝（非串流；相容 gpt-4o-mini / gpt-5；含 Debug 落檔；抽取模型固定為 gpt-4o-mini）

重點：
- 抽取一律用 EXTRACT_MODEL（預設 'gpt-4o-mini'），不吃全域 GPT_KWARGS 裡的 model。
- 所有超時用 client.with_options(timeout=...)，抽取 60s。
- 相容處理：移除不支援參數、max_tokens <-> max_completion_tokens 互轉。
- Debug：把每次請求與回覆（含實際使用的 model）寫到 data/interim/verifier/debug/
"""

from __future__ import annotations

import json
import os
import time
from itertools import count
from pathlib import Path
from typing import List, Dict, Any, Tuple

from openai import OpenAIError, APITimeoutError
try:
    from openai import BadRequestError
except Exception:
    BadRequestError = OpenAIError

from .client import client, GPT_KWARGS
from ..core.paths import EXTRACT_PROMPT_PATH

EXTRACTION_PROMPT = EXTRACT_PROMPT_PATH.read_text(encoding="utf-8-sig")
EXTRACTION_TIMEOUT_SEC = 180  # 抽取階段超時（秒）

# ── 抽取模型：預設 gpt-4o-mini，可用環境變數覆寫（export EXTRACT_MODEL=...）
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gpt-4o-mini")

# Debug 相關
DEBUG_DIR = Path("data/interim/verifier/debug")
_DEBUG_BASENAME: str | None = None  # 由 pipeline 設定


def set_debug_basename(name: str | None) -> None:
    global _DEBUG_BASENAME
    if name:
        safe = "".join(c for c in name if c.isalnum() or c in ("-", "_"))
        _DEBUG_BASENAME = safe or "run"
    else:
        _DEBUG_BASENAME = None


# ───────────────── 工具 ─────────────────
def _base_kwargs_for_extract() -> Dict[str, Any]:
    """
    產生本次抽取要用的 kwargs：
    - 以全域 GPT_KWARGS 為底，但**強制覆寫 model=EXTRACT_MODEL**
    - 主動移除在 4o-mini 常見不支援的取樣鍵，避免 400
    """
    kw = dict(GPT_KWARGS)
    kw["model"] = EXTRACT_MODEL
    # 保險：移除常見不支援取樣鍵（即便上游沒塞也無妨）
    for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "stream"):
        kw.pop(k, None)
    return kw


def _make_kwargs_with_token_key(base: Dict[str, Any], token_key: str) -> Dict[str, Any]:
    kw = dict(base) if base else {}
    mct = kw.pop("max_completion_tokens", None)
    mt = kw.pop("max_tokens", None)
    val = mct if token_key == "max_completion_tokens" else mt
    if val is None:
        val = mt if token_key == "max_completion_tokens" else mct
    if val is not None:
        kw[token_key] = val
    return kw


def _msg_of(exc: Exception) -> str:
    return getattr(exc, "message", str(exc))


def _bad_param_from_400(exc: Exception) -> str | None:
    msg = _msg_of(exc)
    keys = (
        "temperature", "top_p", "presence_penalty", "frequency_penalty",
        "seed", "response_format", "max_tokens", "max_completion_tokens",
        "stream"
    )
    if "Unsupported parameter" in msg or "Unsupported value" in msg or "Unrecognized request argument" in msg:
        for k in keys:
            if k in msg:
                return k
    if "'param':" in msg or '"param":' in msg:
        for k in keys:
            if k in msg:
                return k
    return None


def _save_debug(tag: str, base_common: Dict[str, Any], used_kw: Dict[str, Any], resp_obj: Any, content: str) -> None:
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        epoch = int(time.time())
        base = _DEBUG_BASENAME or "run"
        prefix = f"extract_{base}_{tag}_{epoch}"
        (DEBUG_DIR / f"{prefix}.txt").write_text(content or "",
                                                 encoding="utf-8-sig")
        out: Dict[str, Any] = {
            "tag": tag,
            "used_kwargs": used_kw,
            "base_common": {k: ("<omitted-large>" if k == "messages" else v) for k, v in base_common.items()},
            "model": getattr(resp_obj, "model", None),
            "usage": {
                "prompt_tokens": getattr(getattr(resp_obj, "usage", None), "prompt_tokens", None),
                "completion_tokens": getattr(getattr(resp_obj, "usage", None), "completion_tokens", None),
                "total_tokens": getattr(getattr(resp_obj, "usage", None), "total_tokens", None),
            },
            "choices": [],
            "raw_str_head": str(resp_obj)[:2000],
        }
        try:
            chs = getattr(resp_obj, "choices", None)
            if chs:
                for ch in chs:
                    out["choices"].append({
                        "finish_reason": getattr(ch, "finish_reason", None),
                        "message_content": getattr(getattr(ch, "message", None), "content", None),
                    })
        except Exception:
            pass
        (DEBUG_DIR / f"{prefix}.json").write_text(json.dumps(out,
                                                             ensure_ascii=False, indent=2), encoding="utf-8-sig")
    except Exception:
        pass


# ───────────────── Chat 呼叫封裝（非串流 + 相容處理；HTTP 層 timeout） ─────────────────
def _chat_create_with_compat(*, base_common: Dict[str, Any], kw: Dict[str, Any], timeout_s: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    oc = client.with_options(timeout=timeout_s)
    try:
        resp = oc.chat.completions.create(**base_common, **kw)
        return resp, base_common, kw
    except BadRequestError as be:
        bad = _bad_param_from_400(be)
        if bad:
            if bad in kw:
                fixed_kw = dict(kw)
                fixed_kw.pop(bad, None)
                resp = oc.chat.completions.create(**base_common, **fixed_kw)
                return resp, base_common, fixed_kw
            if bad in base_common:
                fixed_base = dict(base_common)
                fixed_base.pop(bad, None)
                resp = oc.chat.completions.create(**fixed_base, **kw)
                return resp, fixed_base, kw
        # 若是 max_tokens 被拒，嘗試轉為 max_completion_tokens
        if "max_tokens" in kw and ("max_tokens" in str(be)):
            fixed_kw = dict(kw)
            val = fixed_kw.pop("max_tokens")
            fixed_kw["max_completion_tokens"] = val
            resp = oc.chat.completions.create(**base_common, **fixed_kw)
            return resp, base_common, fixed_kw
        raise


def _call_once(tag: str, messages: List[dict], *, allow_json_format: bool) -> str:
    base_common: Dict[str, Any] = {"messages": messages}
    if allow_json_format:
        base_common["response_format"] = {
            "type": "json_object"}  # 若不支援會在 compat 移除

    base_kw = _base_kwargs_for_extract()  # ★ 固定抽取模型
    kw = _make_kwargs_with_token_key(base_kw, "max_completion_tokens")
    try:
        resp, used_base, used_kw = _chat_create_with_compat(
            base_common=base_common, kw=kw, timeout_s=EXTRACTION_TIMEOUT_SEC)
    except BadRequestError as be:
        if _bad_param_from_400(be) != "max_completion_tokens":
            raise
        kw = _make_kwargs_with_token_key(base_kw, "max_tokens")
        resp, used_base, used_kw = _chat_create_with_compat(
            base_common=base_common, kw=kw, timeout_s=EXTRACTION_TIMEOUT_SEC)

    content = (resp.choices[0].message.content or "").strip()
    _save_debug(tag, used_base, used_kw, resp, content)
    return content


# ───────────────── 對外主函式 ─────────────────
def extract_entities_relations(text: str) -> str:
    """
    抽取實體與關係，回傳 JSON 字串（非串流；HTTP 層 60s timeout）
    空字串保護：若第一次拿到空，降級再呼叫一次；仍空則回空結構。
    """
    base_user = text
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": base_user},
    ]

    backoff = 5
    for attempt in count(1):
        try:
            content = _call_once("primary", messages, allow_json_format=True)
            if content:
                return content

            print("[WARN] LLM 回傳空字串，改用降級重呼叫一次。", flush=True)
            messages_fallback = [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "system", "content": "若文本沒有可抽取的資訊，請輸出嚴格 JSON：{\"entities\":[],\"relations\":[]}。"},
                {"role": "user", "content": base_user},
            ]
            content2 = _call_once(
                "fallback", messages_fallback, allow_json_format=False)
            if content2:
                return content2

            return '{"entities":[],"relations":[]}'
        except KeyboardInterrupt:
            print("\n[INFO] 使用者中止（KeyboardInterrupt）。")
            raise
        except (APITimeoutError, OpenAIError) as exc:
            print(
                f"[WARN] 抽取失敗（第 {attempt} 次）：{exc} -> {backoff}s 後重試", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        except Exception as exc:
            print(
                f"[WARN] 抽取例外（第 {attempt} 次）：{exc} -> {backoff}s 後重試", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
