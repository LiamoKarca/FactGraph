"""
gpt_judge 包裝（非串流；HTTP 層長超時 + 完整 Debug 落檔）
"""
from __future__ import annotations

import json
import time
from itertools import count
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from openai import OpenAIError, APITimeoutError
try:
    from openai import BadRequestError
except Exception:
    BadRequestError = OpenAIError

from .client import client, GPT_KWARGS
from ..core.paths import JUDGE_PROMPT_PATH

# ───────────────────────── 常數 ─────────────────────────
JUDGE_PROMPT = JUDGE_PROMPT_PATH.read_text(encoding='utf-8-sig')
# 仍保留較長超時；若要調整，可改成讀環境變數
JUDGE_TIMEOUT_SEC = 1200  # 20 分鐘

# Debug 目錄（依需求寫在 processed）
DEBUG_DIR_PROC = Path("data/processed/verifier/debug")
_DEBUG_BASENAME: Optional[str] = None  # 由呼叫端（pipeline）設定


def set_debug_basename(name: Optional[str]) -> None:
    """設定本模組的 debug 檔名前綴（只影響本 judge.py）。"""
    global _DEBUG_BASENAME
    if name:
        safe = "".join(c for c in name if c.isalnum() or c in ("-", "_"))
        _DEBUG_BASENAME = safe or "run"
    else:
        _DEBUG_BASENAME = None


def _now_epoch() -> int:
    return int(time.time())


def _msg_of(exc: Exception) -> str:
    return getattr(exc, "message", str(exc))


def _unsupported_param_name(exc: Exception) -> Optional[str]:
    msg = _msg_of(exc)
    tokens = ("Unsupported parameter",
              "Unsupported value", "'param':", '"param":')
    if any(t in msg for t in tokens):
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty",
                    "seed", "response_format", "max_tokens", "max_completion_tokens", "stream"):
            if key in msg:
                return key
    return None


def _ensure_debug_dir() -> None:
    try:
        DEBUG_DIR_PROC.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _save_json(path: Path, obj: Any) -> None:
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False,
                        indent=2), encoding="utf-8-sig")
    except Exception:
        pass


def _save_text(path: Path, content: str) -> None:
    try:
        path.write_text(content or "", encoding="utf-8-sig")
    except Exception:
        pass


def _prefix(tag: str) -> str:
    base = _DEBUG_BASENAME or "run"
    return f"judge_{base}_{tag}_{_now_epoch()}"


def _dump_request(tag: str, base_common: Dict[str, Any], kw: Dict[str, Any]) -> None:
    _ensure_debug_dir()
    name = _prefix(f"{tag}_request")
    # 為避免太肥，messages 內文不砍；你需要時可直接檢視 prompt 檔
    _save_json(DEBUG_DIR_PROC / f"{name}.json", {
        "tag": tag,
        "used_kwargs": kw,
        "base_common": base_common,
    })


def _dump_prompt(tag: str, prompt_text: str) -> None:
    _ensure_debug_dir()
    name = _prefix(f"{tag}_prompt")
    _save_text(DEBUG_DIR_PROC / f"{name}.txt", prompt_text)


def _dump_response(tag: str, base_common: Dict[str, Any], used_kw: Dict[str, Any],
                   resp_obj: Any, content: str) -> None:
    _ensure_debug_dir()
    name = _prefix(f"{tag}_response")
    # 純文字版本（便於肉眼看）
    _save_text(DEBUG_DIR_PROC / f"{name}.txt", content)

    # 結構化摘要
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

    _save_json(DEBUG_DIR_PROC / f"{name}.json", out)


def _dump_error(tag: str, base_common: Dict[str, Any], used_kw: Dict[str, Any], exc: Exception) -> None:
    _ensure_debug_dir()
    name = _prefix(f"{tag}_error")
    _save_json(DEBUG_DIR_PROC / f"{name}.json", {
        "tag": tag,
        "used_kwargs": used_kw,
        "base_common": {k: ("<omitted-large>" if k == "messages" else v) for k, v in base_common.items()},
        "error_type": type(exc).__name__,
        "error_message": _msg_of(exc),
    })


def _chat_create_with_compat(base_common: Dict[str, Any], kw: Dict[str, Any], timeout_s: int) -> Any:
    oc = client.with_options(timeout=timeout_s)
    try:
        return oc.chat.completions.create(**base_common, **kw)
    except BadRequestError as be:
        bad = _unsupported_param_name(be)
        if bad and bad in kw:
            fixed = dict(kw)
            fixed.pop(bad, None)
            return oc.chat.completions.create(**base_common, **fixed)
        if "max_tokens" in kw and ("max_tokens" in str(be)):
            fixed = dict(kw)
            val = fixed.pop("max_tokens")
            fixed["max_completion_tokens"] = val
            return oc.chat.completions.create(**base_common, **fixed)
        raise


def judge_news_kb(text: str, *, debug_name: Optional[str] = None) -> str:
    """
    以 GPT 生成查核報告；同時把請求/回應/錯誤完整落於 data/processed/verifier/debug。

    Args:
        text: 已拼接的「[原始新聞]\\n...\\n[比對知識]\\n...」
        debug_name: 供檔名前綴辨識（通常是 news_id）
    """
    set_debug_basename(debug_name)

    base_common: Dict[str, Any] = dict(
        messages=[
            {'role': 'system', 'content': JUDGE_PROMPT},
            {'role': 'user', 'content': text}
        ],
    )
    base_kw = dict(GPT_KWARGS)  # 不帶 timeout；timeout 走 HTTP 層

    # 先把 prompt 落檔（確定真的有「要發送的內容」）
    try:
        _dump_prompt("primary", text)
    except Exception:
        pass

    backoff = 5
    for attempt in count(1):
        tag = f"primary_try{attempt}"
        try:
            # 記錄這次準備送出的參數與訊息
            _dump_request(tag, base_common, base_kw)

            # 真正送出
            resp = _chat_create_with_compat(
                base_common=base_common, kw=base_kw, timeout_s=JUDGE_TIMEOUT_SEC)

            content = (resp.choices[0].message.content or "").strip()
            # 記錄回應
            _dump_response(tag, base_common, base_kw, resp, content)
            return content

        except KeyboardInterrupt:
            # 也要留下「中止」的跡象，方便你知道斷在路上
            try:
                _dump_error(f"{tag}_keyboardinterrupt", base_common,
                            base_kw, KeyboardInterrupt("User aborted (Ctrl-C)"))
            except Exception:
                pass
            print("\n[INFO] 使用者中止（KeyboardInterrupt）。")
            raise

        except (OpenAIError, APITimeoutError) as exc:
            # 記錄錯誤，並顯示即將重試
            try:
                _dump_error(tag, base_common, base_kw, exc)
            except Exception:
                pass
            print(f'[WARN] GPT 判斷失敗（第 {attempt} 次）: {exc} -> {backoff}s 後重試')
            try:
                time.sleep(backoff)
            except KeyboardInterrupt:
                try:
                    _dump_error(f"{tag}_keyboardinterrupt", base_common, base_kw, KeyboardInterrupt(
                        "User aborted during backoff"))
                except Exception:
                    pass
                print("\n[INFO] 使用者中止（KeyboardInterrupt）。")
                raise
            backoff = min(backoff * 2, 60)

        except Exception as exc:
            # 其他非預期錯誤也落檔
            try:
                _dump_error(tag, base_common, base_kw, exc)
            except Exception:
                pass
            print(f'[WARN] 非預期例外（第 {attempt} 次）: {exc} -> {backoff}s 後重試')
            try:
                time.sleep(backoff)
            except KeyboardInterrupt:
                try:
                    _dump_error(f"{tag}_keyboardinterrupt", base_common, base_kw, KeyboardInterrupt(
                        "User aborted during backoff"))
                except Exception:
                    pass
                print("\n[INFO] 使用者中止（KeyboardInterrupt）。")
                raise
            backoff = min(backoff * 2, 60)
