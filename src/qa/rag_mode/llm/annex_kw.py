from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import (
    PATHS,
    read_text, write_text,
    auto_find_latest_user_txt,  # 全域後備
)


def _latest(glob_iter) -> Optional[Path]:
    try:
        return sorted(glob_iter, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    except Exception:
        return None


def _ensure_list(x: Any):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _normalize_keywords_payload(payload: Any) -> dict:
    """
    將 keywords 檔整理為標準欄位（dict）：
    - 支援 dict / list / str；盡量取出 entities/events/dates/numbers/focus
    - 不存在則給空陣列
    """
    obj = payload
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            obj = {}

    # keywords 可能是 list（例如模型輸出陣列），嘗試抓第一個 dict
    if isinstance(obj, list):
        head = next((x for x in obj if isinstance(x, dict)), {})
        obj = head if isinstance(head, dict) else {}

    if not isinstance(obj, dict):
        obj = {}

    return {
        "entities": _ensure_list(obj.get("entities")),
        "events":   _ensure_list(obj.get("events")),
        "dates":    _ensure_list(obj.get("dates")),
        "numbers":  _ensure_list(obj.get("numbers")),
        "focus":    _ensure_list(obj.get("focus")),
    }


def main() -> None:
    # 1) 讀取「本次 job」的 user 輸入（優先）；否則落回全域最新
    env_user = os.getenv("RAG_USER_FILE")
    if env_user:
        user_path = Path(env_user)
        if not user_path.exists():
            raise SystemExit(f"[annex] RAG_USER_FILE 不存在：{user_path}")
    else:
        user_path = auto_find_latest_user_txt()
    user_txt = (read_text(user_path) or "").strip()
    if not user_txt:
        raise SystemExit("[annex] 使用者輸入為空白")

    # 2) 尋找「本次 job」的 keywords；若無 job，落回全域 keywords 目錄最新
    job_dir = os.getenv("RAG_JOB_DIR")
    if job_dir:
        kw_dir = Path(job_dir) / "keywords"
        kw_path = _latest(kw_dir.glob("*.json")) if kw_dir.exists() else None
    else:
        kw_dir = PATHS.KEYWORDS_DIR
        kw_path = _latest(kw_dir.glob("*_keywords.json"))

    kw_obj: Any = {}
    if kw_path and kw_path.exists():
        try:
            kw_obj = json.loads(read_text(kw_path))
        except Exception:
            kw_obj = read_text(kw_path)  # 交給 normalizer 處理
    # 沒找到 keywords 也不視為錯誤 → clues 全空

    clues = _normalize_keywords_payload(kw_obj)

    # 3) 決定輸出位置與檔名（job 優先）
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = (os.getenv("RAG_JOB_ID") or "").strip()
    if job_dir:
        annex_dir = Path(job_dir) / "annex"
    else:
        annex_dir = PATHS.KEYWORDS_DIR  # 與既有相容
    annex_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{job_id}_{ts}_annex.json" if job_id else f"{ts}_annex.json"
    out_path = annex_dir / out_name

    payload = {
        "user_question": user_txt,
        "entities": clues["entities"],
        "events": clues["events"],
        "dates": clues["dates"],
        "numbers": clues["numbers"],
        "focus": clues["focus"],
    }
    write_text(out_path, json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[annex] Save: {out_path}")

    # 另寫 latest pointer（便於 gpt_rag.py 以 job 搜尋 annex）
    if job_id:
        latest = annex_dir / f"{job_id}.json"
        try:
            write_text(latest, json.dumps(
                payload, ensure_ascii=False, indent=2))
            print(f"[annex] Latest pointer: {latest}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
