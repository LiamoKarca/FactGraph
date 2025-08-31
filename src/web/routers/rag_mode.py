# src/web/routers/rag_mode.py
from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

router = APIRouter(prefix="/rag_mode", tags=["rag_mode"])

TZ = ZoneInfo("Asia/Taipei")

# ── 正確定位專案根目錄（…/FactGraph）────────────────────────────────────────────
# 本檔位於 …/FactGraph/src/web/routers/rag_mode.py
ROUTER_FILE = Path(__file__).resolve()
BASE_DIR = ROUTER_FILE.parents[1]         # …/FactGraph/src/web
PROJECT_ROOT = BASE_DIR.parent.parent     # …/FactGraph  ← 這裡才是根
# （先前寫成 parents[2] 只會到 …/FactGraph/src，導致找不到 top-level 包 src）


def _job_paths(job_id: str):
    """
    以 job_id 建立獨立工作空間，避免並行互相覆蓋。
    - 使用者輸入：data/interim/rag_mode/jobs/{job_id}/user-input/{ts}.txt
    - 產出：      data/processed/rag_mode/jobs/{job_id}/{job_id}_{ts}_rag.{md,json}
    """
    interim = PROJECT_ROOT / "data" / "interim" / "rag_mode" / "jobs" / job_id
    processed = PROJECT_ROOT / "data" / "processed" / "rag_mode" / "jobs" / job_id
    user_dir = interim / "user-input"
    out_dir = processed
    user_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return user_dir, out_dir


@router.post("/query")
async def query_rag(
    job_id: str = Query(..., description="上游建立並沿用到底的唯一 ID"),
    file: UploadFile = File(...),
):
    """
    1) 以 job_id 建立專屬目錄，寫入使用者輸入
    2) 以環境變數把 job-scoped 參數傳進 pipeline/gpt
    3) 僅回傳本 job 的輸出（不刪輸入；等 /cleanup 被呼叫）
    """
    user_dir, out_dir = _job_paths(job_id)
    ts = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    input_path = user_dir / f"{ts}.txt"

    try:
        content = (await file.read()).decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(400, f"上傳內容讀取失敗：{e}")

    if not content.strip():
        raise HTTPException(400, "上傳內容為空白")

    try:
        input_path.write_text(content, encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"寫入使用者輸入失敗：{e}")

    # ── 準備環境（把專案根加入 PYTHONPATH，並傳遞 job 參數）────────────────────────
    env = os.environ.copy()
    # 確保 'src' 被看見：把專案根（包含 src/ 目錄）放進 PYTHONPATH
    env["PYTHONPATH"] = (
        (str(PROJECT_ROOT) + os.pathsep + env["PYTHONPATH"])
        if env.get("PYTHONPATH")
        else str(PROJECT_ROOT)
    )
    env["RAG_JOB_ID"] = job_id
    env["RAG_USER_FILE"] = str(input_path)
    env["RAG_OUT_DIR"] = str(out_dir)
    env["RAG_JOB_DIR"] = str(user_dir.parent)  # = …/jobs/{job_id}

    try:
        # 僅處理本 job 的資料（pipeline 會偵測 RAG_USER_FILE 跳過全域掃描）
        # 使用當前直譯器啟動，以避免多 Python 版本/venv 不一致
        subprocess.run(
            [sys.executable, "-m", "src.qa.rag_mode.pipeline"],
            cwd=str(PROJECT_ROOT),   # ← 這裡一定要是專案根
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # 不刪輸入；讓上游能排錯
        stderr = (e.stderr or "")[-2000:]
        raise HTTPException(500, f"RAG pipeline 失敗：{stderr or 'no stderr'}")

    # 只抓本 job 的輸出（不會打到其它 job）
    try:
        candidates = sorted(
            out_dir.glob(f"{job_id}_*_rag.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise HTTPException(500, "未找到本次 RAG 產出（.md）")

        result_md = candidates[0].read_text(encoding="utf-8-sig").strip()
        if not result_md:
            raise HTTPException(500, "RAG 產出為空，請檢查向量庫或 annex")
        return {"result_md": result_md}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"讀取 RAG 產出失敗：{e}")


@router.post("/cleanup")
async def cleanup_job(
    job_id: str = Query(..., description="寫入 Firestore 成功後再清理此 job 的原始輸入"),
):
    user_dir, _ = _job_paths(job_id)
    removed = 0
    try:
        if user_dir.exists():
            for p in user_dir.glob("*"):
                try:
                    p.unlink(missing_ok=True)
                    removed += 1
                except Exception:
                    pass
        return {"job_id": job_id, "removed_files": removed}
    except Exception as e:
        # 清理失敗不視為致命；上游可忽略
        raise HTTPException(500, f"清理失敗：{e}")
