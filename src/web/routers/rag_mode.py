# src/web/routers/rag_mode.py
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/rag_mode", tags=["rag_mode"])


@router.post("/query")
async def query_rag(file: UploadFile = File(...)):
    """
    同步版 RAG：接收一段文字（檔案），落地到 interim，再呼叫 pipeline，
    最後讀取 processed/rag_mode/*.md 的最新檔回傳給呼叫者。
    （Firestore 的更新交給 main.py 的背景任務處理）
    """
    # 讀取上傳文字
    content_bytes = await file.read()
    try:
        text_content = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text_content = content_bytes.decode("utf-8", errors="ignore")

    # 專案根目錄
    project_root = Path(__file__).resolve().parents[3]

    # 寫入唯一一個 .txt 到 interim（pipeline 習慣讀該目錄最新/唯一檔）
    interim_dir = project_root / "data" / "interim" / "rag_mode" / "user-input"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 保險：清掉舊檔，避免 pipeline 誤讀
    for old in interim_dir.glob("*.txt"):
        try:
            old.unlink()
        except Exception:
            pass

    now = datetime.now(ZoneInfo("Asia/Taipei"))
    stamp = now.strftime("%Y-%m-%d_%H%M")
    input_path = interim_dir / f"{stamp}.txt"
    input_path.write_text(text_content, encoding="utf-8")

    # 呼叫 RAG pipeline（與你原本使用方式一致，不帶參數）
    try:
        subprocess.run(
            ["python", "-m", "src.qa.rag_mode.pipeline"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500, detail=f"RAG pipeline 執行錯誤：{e.stderr or e.stdout}")

    # 讀取 processed 目錄最新的 .md 作為結果
    processed_dir = project_root / "data" / "processed" / "rag_mode"
    md_files = sorted(processed_dir.glob("*.md"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not md_files:
        # 清檔後也要回應
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="未找到 RAG 產出（.md）")

    result_md = md_files[0].read_text(encoding="utf-8-sig")

    # 清除暫存輸入
    input_path.unlink(missing_ok=True)

    return {"result_md": result_md}
