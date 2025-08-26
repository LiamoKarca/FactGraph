"""
Pipeline：
1) 執行 merge.py  → 輸出 data/processed/news_merge/cna_ey_moi_pts.json
2) 執行 dedup.py  → 輸出 data/processed/news_merge/cna_ey_moi_pts_DEDUP.json
3) 執行 upload_storage.py → 讀取 *_DEDUP.json 上傳至 RAG storage，
   並於 data/processed/news_merge/rag_storage_id 寫入「yyyy-mm-dd-hhmm」命名的 id 檔
4) 僅當「第 3 步確實成功且新增了 *新的* id 檔」時，才執行 delete_storage.py 刪除舊向量庫

說明：
- 本版對「是否真的上傳成功」做雙保險判定：
  (A) 上傳腳本 returncode==0
  (B) 上傳前後比較 rag_storage_id 目錄，必須出現 *新的* id 檔（依 mtime 與檔名）
- 若任一條件不成立，一律不執行刪除，以避免「空向量庫」誤刪或錯刪。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from dotenv import load_dotenv

load_dotenv()  # 讓 GPT_API / OPENAI_API_KEY 等可從 .env 載入

# ── 腳本路徑 ────────────────────────────────────────────────────────────────
MERGE_SCRIPT = Path("src/knowledge_base_operation/news_merge/merge.py")
DEDUP_SCRIPT = Path("src/knowledge_base_operation/news_merge/dedup.py")
UPLOAD_SCRIPT = Path(
    "src/knowledge_base_operation/news_merge/upload_storage.py")
DELETE_SCRIPT = Path(
    "src/knowledge_base_operation/news_merge/delete_storage.py")

# ── I/O 檔案 ────────────────────────────────────────────────────────────────
INPUT_FILES: List[Path] = [
    Path("data/raw/news/cna/cna_news.json"),
    Path("data/raw/news/ey/ey_news.json"),
    Path("data/raw/news/moi/moi_news.json"),
    Path("data/raw/news/pts/pts_news.json"),
]
OUTPUT_JSON = Path("data/processed/news_merge/cna_ey_moi_pts.json")
OUTPUT_DEDUP_JSON = Path("data/processed/news_merge/cna_ey_moi_pts_DEDUP.json")

# RAG storage 上傳紀錄（由 upload_storage.py 產生）
RAG_ID_DIR = Path("data/processed/news_merge/rag_storage_id")


# ── 公用函式 ────────────────────────────────────────────────────────────────
def _latest_file(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    files = [p for p in directory.iterdir() if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _latest_file_snapshot(directory: Path) -> Tuple[Optional[Path], float]:
    p = _latest_file(directory)
    return (p, p.stat().st_mtime if p else 0.0)


def _newer_file_appeared(before: Tuple[Optional[Path], float],
                         after: Tuple[Optional[Path], float]) -> bool:
    """判斷上傳前後是否出現了 *新的* id 檔（檔案不同或 mtime 變新）。"""
    p_before, t_before = before
    p_after,  t_after = after
    if p_after is None:
        return False
    if p_before is None:
        return True
    if p_after != p_before:
        return True
    return t_after > t_before


# ── 任務是否需要重跑 ───────────────────────────────────────────────────────
def need_merge(force: bool) -> bool:
    if force or not OUTPUT_JSON.exists():
        return True
    out_m = OUTPUT_JSON.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > out_m for s in INPUT_FILES)


def need_dedup(force: bool) -> bool:
    if force or not OUTPUT_DEDUP_JSON.exists():
        return True
    return OUTPUT_JSON.exists() and OUTPUT_JSON.stat().st_mtime > OUTPUT_DEDUP_JSON.stat().st_mtime


def _latest_file_mtime(directory: Path) -> Optional[float]:
    p = _latest_file(directory)
    return p.stat().st_mtime if p else None


def need_upload(force: bool) -> bool:
    """
    上傳條件：
    - 強制上傳，或
    - 沒有任何 id 檔，或
    - dedup 輸出比目前最新 id 檔還新（代表內容更新）
    """
    if force:
        return True
    if not OUTPUT_DEDUP_JSON.exists():
        return False
    latest_id_mtime = _latest_file_mtime(RAG_ID_DIR)
    if latest_id_mtime is None:
        return True
    return OUTPUT_DEDUP_JSON.stat().st_mtime > latest_id_mtime


# ── 主流程 ────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="News merge + dedup + upload-to-RAG (+conditional delete) pipeline"
    )
    ap.add_argument("--force-merge",  action="store_true", help="強制重跑合併")
    ap.add_argument("--force-dedup",  action="store_true", help="強制重跑去重")
    ap.add_argument("--force-upload", action="store_true",
                    help="強制重跑上傳 RAG storage")
    # 傳遞給 upload/delete 的額外參數（可選）
    ap.add_argument("--upload-args", default="",
                    help="傳給 upload_storage.py 的其他參數字串")
    ap.add_argument("--delete-args", default="",
                    help="傳給 delete_storage.py 的其他參數字串")
    args = ap.parse_args()

    # 1) 合併
    if need_merge(args.force_merge):
        print("▶ 合併：執行 merge.py")
        r = subprocess.run([sys.executable, str(MERGE_SCRIPT)], check=False)
        if r.returncode != 0:
            sys.exit(r.returncode)
    else:
        print("✓ 合併：已是最新，跳過")

    # 2) 去重
    if need_dedup(args.force_dedup):
        print("▶ 去重：執行 dedup.py")
        r = subprocess.run([sys.executable, str(DEDUP_SCRIPT)], check=False)
        if r.returncode != 0:
            sys.exit(r.returncode)
    else:
        print("✓ 去重：已是最新，跳過")

    # 3) 上傳 RAG storage（雙保險判定：returncode + 新 id 檔）
    uploaded_this_run = False
    if need_upload(args.force_upload):
        before_snap = _latest_file_snapshot(RAG_ID_DIR)
        print("▶ 上傳：執行 upload_storage.py")
        cmd = [sys.executable, str(UPLOAD_SCRIPT)]
        if args.upload_args.strip():
            # 允許把一整段參數字串傳給子程式（例如 --name 自訂）
            cmd.extend(args.upload_args.split())
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            # upload_storage.py 設計：索引失敗會回傳非 0，直接中止
            sys.exit(r.returncode)
        after_snap = _latest_file_snapshot(RAG_ID_DIR)
        uploaded_this_run = _newer_file_appeared(before_snap, after_snap)
        if not uploaded_this_run:
            # 極少數情況（例如重複內容），保守處理：視為未成功，不做刪除
            print("⚠ 上傳腳本成功但未偵測到新 id 檔，將跳過刪除舊向量庫。")
    else:
        print("✓ 上傳：已是最新，跳過")

    # 4) 僅當「本次確實新增向量庫」才清理舊向量庫
    if uploaded_this_run:
        print("▶ 清理：執行 delete_storage.py（僅保留最新 RAG 向量庫）")
        cmd = [sys.executable, str(DELETE_SCRIPT)]
        if args.delete_args.strip():
            cmd.extend(args.delete_args.split())
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            sys.exit(r.returncode)
    else:
        print("✓ 清理：本次未新增向量庫，跳過 delete_storage.py")

    print("✅ Pipeline 完成 →", OUTPUT_DEDUP_JSON)
    if RAG_ID_DIR.exists():
        print("🆔 最新 RAG id 檔目錄：", RAG_ID_DIR)


if __name__ == "__main__":
    main()
