"""
RAG-LLM 設定與工具（.env 可選；雲端優先讀環境變數）
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

# .env -> 本機開發方便；雲端沒有也沒關係
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # 雲端沒裝也無妨

# ──────────────────────────────────────────────────────────────────────────────
# .env（可選，不存在就略過）
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_ENV = Path(".env").resolve()
if load_dotenv and GLOBAL_ENV.exists():
    load_dotenv(dotenv_path=str(GLOBAL_ENV), override=True)
elif load_dotenv:
    # 允許往上尋找或當前目錄的 .env；若仍沒有就忽略
    load_dotenv(override=True)

ENCODING = "utf-8-sig"

# ──────────────────────────────────────────────────────────────────────────────
# 專案路徑（維持相對目錄，因為 pipeline 會以專案根為 cwd 執行）
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
    # 你現在用的是 GPT_API，這裡同時容忍 OPENAI_API / OPENAI_API_KEY
    api_key: str = (
        os.getenv("GPT_API")
        or os.getenv("OPENAI_API")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    rag_model: str = os.getenv("GPT_MODEL", "gpt-4.1")
    keywords_model: str = os.getenv(
        "KEYWORDS_MODEL", os.getenv("GPT_MODEL", "gpt-4.1"))


PATHS = Paths()
MODELS = Models()

# 雲端沒有 .env 也沒關係；但真的少金鑰才報錯
if not MODELS.api_key:
    raise RuntimeError(
        "未設定 API Key：請在 Cloud Run 服務環境變數或 .env 填入 GPT_API / OPENAI_API_KEY"
    )

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI 客戶端
# ──────────────────────────────────────────────────────────────────────────────


def make_openai_client():
    """OpenAI Responses / Vector Store 等官方 SDK 客戶端"""
    from openai import OpenAI  # pip install openai
    return OpenAI(api_key=MODELS.api_key)


def configure_legacy_openai_module():
    """若舊程式仍用 openai.ChatCompletion，先設定 api_key。"""
    import openai  # type: ignore
    openai.api_key = MODELS.api_key
    return openai

# ──────────────────────────────────────────────────────────────────────────────
# I/O 與通用工具（原樣保留）
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
    files = list(PATHS.RAG_ID_DIR.glob("*"))
    if not files:
        raise FileNotFoundError(f"找不到任何 RAG id 檔於 {PATHS.RAG_ID_DIR}")
    latest = pick_latest(files)
    return read_text(latest).strip()

# ──────────────────────────────────────────────────────────────────────────────
# 分類工具（原樣保留）
# ──────────────────────────────────────────────────────────────────────────────


def _heuristic_classify(text: str) -> Tuple[str, str]:
    t = (text or "").strip()
    if not t:
        return "短問句", "空字串或極短輸入"
    long_threshold = 180
    many_breaks = t.count("\n") >= 3
    many_punct = sum(t.count(x)
                     for x in ["。", " ", "，", "、", "；", ".", ",", ";"]) >= 6
    if len(t) >= long_threshold or many_breaks or many_punct:
        return "長篇問題", f"啟發式判斷：len={len(t)}, breaks={t.count('\\n')}, punct>=6={many_punct}"
    return "短問句", f"啟發式判斷：len={len(t)}, breaks={t.count('\\n')}, punct<6"


def classify_kind(client, model: str, user_text: str) -> Tuple[str, str]:
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
    files = sorted(root.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"找不到任何 .txt：{root}/*.txt")
    if len(files) > 1:
        names = ", ".join([p.name for p in files])
        raise RuntimeError(
            f"{root} 內存在多個 .txt（{len(files)} 個）：{names}；僅支援單一檔案")
    return files[0]


def ensure_annex_list(keywords_payload: Any, annex_head: dict) -> List[Any]:
    if isinstance(keywords_payload, dict):
        return [annex_head, keywords_payload]
    if isinstance(keywords_payload, list):
        if len(keywords_payload) == 1 and isinstance(keywords_payload[0], dict):
            return [annex_head, keywords_payload[0]]
        return [annex_head] + keywords_payload
    return [annex_head, {"raw": keywords_payload}]


def build_kw_annex_output_name(kw_path: Path) -> Path:
    name = kw_path.name
    suffix = "_keywords.json"
    if name.endswith(suffix):
        base = name[: -len(suffix)]
        out_name = f"{base}_kw_annex.json"
    else:
        out_name = f"{kw_path.stem}_kw_annex.json"
    return kw_path.with_name(out_name)
