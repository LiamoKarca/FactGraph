# src/web/routers/rag_mode.py
from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/rag_mode", tags=["rag_mode"])


@router.post("/query")
async def query_rag(file: UploadFile = File(...)):
    """
    同步版 RAG：
    1) 接收一段文字，寫入 interim/rag_mode/user-input/{YYYY-mm-dd_HHMM}.txt
    2) 呼叫 RAG pipeline
    3) 「點名」讀取 processed/rag_mode/{YYYYMMDD-HHMM}_rag.md
       - 若同分鐘檔不存在，退回以啟動時間為門檻，挑本次請求後產生的 *_rag.md 最新一份
    4) 若內容為空白，回 500（避免寫空結果）
    """
    # 讀取上傳文字
    content_bytes = await file.read()
    try:
        text_content = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text_content = content_bytes.decode("utf-8", errors="ignore")

    # 專案根目錄
    project_root = Path(__file__).resolve().parents[3]

    # 準備 interim 目錄與檔名（給 pipeline 讀最新/唯一檔）
    interim_dir = project_root / "data" / "interim" / "rag_mode" / "user-input"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 清舊檔，避免 pipeline 誤讀
    for old in interim_dir.glob("*.txt"):
        try:
            old.unlink()
        except Exception:
            pass

    tz = ZoneInfo("Asia/Taipei")
    now = datetime.now(tz)
    # 給「輸入檔」用的戳：YYYY-mm-dd_HHMM（你原本就這樣取名）
    stamp_in = now.strftime("%Y-%m-%d_%H%M")
    input_path = interim_dir / f"{stamp_in}.txt"
    input_path.write_text(text_content, encoding="utf-8")

    # 同步呼叫 RAG pipeline（內部會輸出 YYYYMMDD-HHMM_rag.md）
    start_ts = time.time()
    # 這是「期望的輸出檔名戳」：YYYYMMDD-HHMM（與 gpt_rag.py 一致）
    stamp_out = now.strftime("%Y%m%d-%H%M")

    try:
        subprocess.run(
            ["python", "-m", "src.qa.rag_mode.pipeline"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # 清理輸入檔後回錯
        input_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline 執行錯誤：{(e.stderr or e.stdout)[:4000]}",
        )

    # ── 點名讀檔 ──────────────────────────────────────────────────────────────
    processed_dir = project_root / "data" / "processed" / "rag_mode"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1) 首選：同分鐘的精準檔名
    exact_md = processed_dir / f"{stamp_out}_rag.md"
    if exact_md.exists():
        result_md = exact_md.read_text(encoding="utf-8-sig")
    else:
        # 2) 備援：挑「本請求啟動後」新產生的 *_rag.md（避免撈到舊檔）
        candidates = []
        for p in processed_dir.glob("*_rag.md"):
            try:
                if p.stat().st_mtime >= start_ts - 1:  # 小幅緩衝
                    candidates.append(p)
            except FileNotFoundError:
                pass
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            input_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="未找到本次 RAG 產出（.md）")
        result_md = candidates[0].read_text(encoding="utf-8-sig")

    # 內容為空白 → 視為錯誤（交由上層背景任務寫 FAILED/last_error）
    if not result_md.strip():
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="RAG 產出為空，請檢查向量庫或 annex")

    # 清除暫存輸入
    input_path.unlink(missing_ok=True)

    return {"result_md": result_md}
