# -*- coding: utf-8 -*-
"""
RAG-LLM 設定與工具（僅讀專案根 .env）
- 直接載入專案根目錄的 .env（需至少含 GPT_API 與 GPT_MODEL）
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# .env 載入（專案根目錄）
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_ENV = Path(".env").resolve()
if GLOBAL_ENV.exists():
    load_dotenv(dotenv_path=str(GLOBAL_ENV), override=True)
else:
    raise FileNotFoundError(f"找不到專案根目錄的 .env: {GLOBAL_ENV}")

ENCODING = "utf-8-sig"

# ──────────────────────────────────────────────────────────────────────────────
# 專案路徑
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Paths:
    # 中介/輸入輸出
    USER_INPUT_DIR: Path = Path("data/interim/rag_mode/user-input")
    KEYWORDS_DIR:   Path = Path("data/interim/rag_mode/keywords")
    RAG_OUTDIR:     Path = Path("data/processed/rag_mode")

    # 上游（新聞向量庫 id 檔）
    RAG_ID_DIR:     Path = Path("data/processed/news_merge/rag_storage_id")

    # 提示詞
    PROMPT_LONG:     Path = Path("src/qa/rag_mode/prompts/long.txt")
    PROMPT_SHORT:    Path = Path("src/qa/rag_mode/prompts/short.txt")
    PROMPT_CLASSIFY: Path = Path(
        "src/qa/rag_mode/prompts/classify_content.txt")
    PROMPT_KEYWORDS: Path = Path(
        "src/qa/rag_mode/prompts/keywords_extraction.txt")


@dataclass(frozen=True)
class Models:
    api_key: str = os.getenv("GPT_API") or os.getenv(
        "OPENAI_API") or os.getenv("OPENAI_API_KEY") or ""
    rag_model: str = os.getenv("GPT_MODEL", "gpt-4.1")
    keywords_model: str = os.getenv(
        "KEYWORDS_MODEL", os.getenv("GPT_MODEL", "gpt-4.1"))


PATHS = Paths()
MODELS = Models()

if not MODELS.api_key:
    raise RuntimeError("未設定 API Key：請在專案根目錄 .env 填入 GPT_API 或 OPENAI_API_KEY")
if not MODELS.rag_model:
    raise RuntimeError("未設定 GPT 模型：請在專案根目錄 .env 填入 GPT_MODEL")

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI 客戶端
# ──────────────────────────────────────────────────────────────────────────────


def make_openai_client():
    """Responses / Vector Store 等新 SDK 客戶端"""
    from openai import OpenAI
    return OpenAI(api_key=MODELS.api_key)


def configure_legacy_openai_module():
    """部分舊檔若還使用 openai.ChatCompletion，可呼叫本函式先設定 api_key"""
    import openai  # type: ignore
    openai.api_key = MODELS.api_key
    return openai

# ──────────────────────────────────────────────────────────────────────────────
# I/O 與通用工具
# ──────────────────────────────────────────────────────────────────────────────


def read_text(p: Path) -> str:
    return p.read_text(encoding=ENCODING, errors="ignore")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding=ENCODING)


def read_json(p: Path) -> Any:
    return json.loads(read_text(p))


def write_json(p: Path, obj: Any, *, ensure_ascii: bool = False, indent: int = 2) -> None:
    write_text(p, json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent))


def pick_latest(paths: List[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("空路徑清單：無法挑選最新檔案")
    paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return paths[0]


def auto_find_latest_user_txt() -> Path:
    files = list(PATHS.USER_INPUT_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"找不到使用者輸入檔：{PATHS.USER_INPUT_DIR}/*.txt")
    return pick_latest(files)


def auto_find_annex() -> Path:
    """
    支援兩種命名：
    - *_kw_annex.json（你現行的命名）
    - *_annex.json   （保留相容）
    """
    files = list(PATHS.KEYWORDS_DIR.rglob("*_kw_annex.json"))
    if not files:
        files = list(PATHS.KEYWORDS_DIR.rglob("*_annex.json"))
    if not files:
        raise FileNotFoundError(
            f"找不到 annex 檔：{PATHS.KEYWORDS_DIR}/**/*_kw_annex.json")
    return pick_latest(files)


def latest_keywords_json() -> Path:
    files = list(PATHS.KEYWORDS_DIR.rglob("*_keywords.json"))
    if not files:
        raise FileNotFoundError(
            f"找不到 keywords 檔：{PATHS.KEYWORDS_DIR}/**/*_keywords.json")
    return pick_latest(files)


def get_latest_vector_store_id() -> str:
    """
    若仍需讀本機最新 VS id（例如其他工具腳本），可使用此函式。
    注意：RAG 主流程已改為優先使用線上最新 VS，是否使用本地 id 取決於上層邏輯。
    """
    files = list(PATHS.RAG_ID_DIR.glob("*"))
    if not files:
        raise FileNotFoundError(f"找不到任何 RAG id 檔於 {PATHS.RAG_ID_DIR}")
    latest = pick_latest(files)
    return read_text(latest).strip()

# ──────────────────────────────────────────────────────────────────────────────
# 分類工具：classify_kind（長篇問題 / 短問句）
# ──────────────────────────────────────────────────────────────────────────────


def _heuristic_classify(text: str) -> Tuple[str, str]:
    """
    回傳 (type, reason)：
      type ∈ {"長篇問題", "短問句"}
    """
    t = (text or "").strip()
    if not t:
        return "短問句", "空字串或極短輸入"
    # 啟發式：長度、換行、句點/頓號數量
    long_threshold = 180
    many_breaks = t.count("\n") >= 3
    many_punct = sum(t.count(x)
                     for x in ["。", " ", "，", "、", "；", ".", ",", ";"]) >= 6
    if len(t) >= long_threshold or many_breaks or many_punct:
        return "長篇問題", f"啟發式判斷：len={len(t)}, breaks={t.count('\\n')}, punct>=6={many_punct}"
    return "短問句", f"啟發式判斷：len={len(t)}, breaks={t.count('\\n')}, punct<6"


def classify_kind(client, model: str, user_text: str) -> Tuple[str, str]:
    """
    使用 PROMPT_CLASSIFY 呼叫 Responses 進行分類；若解析失敗則退回啟發式。
    回傳 (type, reason)：
      type ∈ {"長篇問題", "短問句"}
      reason：模型輸出或 fallback 原因
    """
    prompt_path = PATHS.PROMPT_CLASSIFY
    if not prompt_path.exists():
        return _heuristic_classify(user_text)

    template = read_text(prompt_path)
    prompt = template.replace("{user_text}", user_text or "")

    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": prompt}],
        )
        out = getattr(resp, "output_text", "") or ""
        out = out.strip()
        parsed = json.loads(out)
        typ = str(parsed.get("type", "")).strip()
        if typ not in ("長篇問題", "短問句"):
            if "長" in typ:
                typ = "長篇問題"
            elif "短" in typ:
                typ = "短問句"
        if typ in ("長篇問題", "短問句"):
            return typ, "model"
        fallback_type, _ = _heuristic_classify(user_text)
        return fallback_type, f"fallback: unexpected model output: {out[:120]}"
    except Exception as e:
        t, _ = _heuristic_classify(user_text)
        return t, f"fallback: exception {e.__class__.__name__}"


def find_single_txt_or_error(root: Path) -> Path:
    """
    僅允許目錄內存在「唯一」.txt；否則報錯（給 annex 用）。
    """
    files = sorted(root.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"找不到任何 .txt：{root}/*.txt")
    if len(files) > 1:
        names = ", ".join([p.name for p in files])
        raise RuntimeError(
            f"{root} 內存在多個 .txt（{len(files)} 個）：{names}；僅支援單一檔案")
    return files[0]


def ensure_annex_list(keywords_payload: Any, annex_head: dict) -> List[Any]:
    """
    產出 annex 目標結構（list）：
      [
        {"user_question": "..."},
        <keywords_payload 的合理展開>
      ]
    - dict → [head, dict]
    - list(長度=1且首元素為dict) → [head, list[0]]
    - list(其他) → [head] + list
    - 其他型別（str/數值）→ [head, {"raw": 原值}]
    """
    # 標準化 payload
    if isinstance(keywords_payload, dict):
        tail = keywords_payload
        return [annex_head, tail]

    if isinstance(keywords_payload, list):
        if len(keywords_payload) == 1 and isinstance(keywords_payload[0], dict):
            return [annex_head, keywords_payload[0]]
        # 多元素：直接併上
        return [annex_head] + keywords_payload

    # 其餘型別包成 raw
    return [annex_head, {"raw": keywords_payload}]


def build_kw_annex_output_name(kw_path: Path) -> Path:
    """
    由 <stem>_keywords.json 推導成 <stem>_kw_annex.json。
    若不是標準命名，則一律接尾 _kw_annex.json。
    僅回傳「同層」檔名，呼叫端可再轉置到固定輸出資料夾。
    """
    name = kw_path.name
    suffix = "_keywords.json"
    if name.endswith(suffix):
        base = name[: -len(suffix)]
        out_name = f"{base}_kw_annex.json"
    else:
        out_name = f"{kw_path.stem}_kw_annex.json"
    return kw_path.with_name(out_name)
