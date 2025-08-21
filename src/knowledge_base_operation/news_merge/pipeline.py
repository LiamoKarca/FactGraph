"""
Pipeline：
1) 執行 merge.py → 輸出 cna_ey_moi_pts.json
2) 執行 dedup.py → 輸出 cna_ey_moi_pts_DEDUP.json
3) 執行 upload_storage.py → 讀取 cna_ey_moi_pts_DEDUP.json 上傳至 RAG storage，
   並於 data/processed/news_merge/rag_storage_id 底下寫入「yyyy-mm-dd-hhmm」命名的 id 檔
4) 若第 3 步有「實際上傳成功」，才執行 delete_storage.py 以刪除舊向量庫
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MERGE_SCRIPT = Path("src/knowledge_base_operation/news_merge/merge.py")
DEDUP_SCRIPT = Path("src/knowledge_base_operation/news_merge/dedup.py")
UPLOAD_SCRIPT = Path(
    "src/knowledge_base_operation/news_merge/upload_storage.py")
DELETE_SCRIPT = Path(
    "src/knowledge_base_operation/news_merge/delete_storage.py")

INPUT_FILES = [
    Path("data/raw/news/cna/cna_news.json"),
    Path("data/raw/news/ey/ey_news.json"),
    Path("data/raw/news/moi/moi_news.json"),
    Path("data/raw/news/pts/pts_news.json"),
]
OUTPUT_JSON = Path("data/processed/news_merge/cna_ey_moi_pts.json")
OUTPUT_DEDUP_JSON = Path("data/processed/news_merge/cna_ey_moi_pts_DEDUP.json")

# RAG storage 上傳紀錄（由 upload_storage.py 產出的 id 檔會寫在這裡）
RAG_ID_DIR = Path("data/processed/news_merge/rag_storage_id")


def need_merge(force: bool) -> bool:
    if force:
        return True
    if not OUTPUT_JSON.exists():
        return True
    out_m = OUTPUT_JSON.stat().st_mtime
    for s in INPUT_FILES:
        if s.exists() and s.stat().st_mtime > out_m:
            return True
    return False


def need_dedup(force: bool) -> bool:
    if force:
        return True
    if not OUTPUT_DEDUP_JSON.exists():
        return True
    return OUTPUT_JSON.exists() and (OUTPUT_JSON.stat().st_mtime > OUTPUT_DEDUP_JSON.stat().st_mtime)


def _latest_file_mtime(directory: Path) -> float | None:
    if not directory.exists():
        return None
    mtimes = []
    for p in directory.iterdir():
        if p.is_file():
            try:
                mtimes.append(p.stat().st_mtime)
            except FileNotFoundError:
                continue
    return max(mtimes) if mtimes else None


def need_upload(force: bool) -> bool:
    """
    上傳條件：
    - 強制上傳，或
    - 沒有任何 id 檔，或
    - dedup 的輸出比目前最新 id 檔還新（代表內容更新）
    """
    if force:
        return True
    if not OUTPUT_DEDUP_JSON.exists():
        return False
    latest_id_mtime = _latest_file_mtime(RAG_ID_DIR)
    if latest_id_mtime is None:
        return True
    return OUTPUT_DEDUP_JSON.stat().st_mtime > latest_id_mtime


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="News merge + dedup + upload-to-RAG (+conditional delete) pipeline")
    ap.add_argument("--force-merge", action="store_true", help="強制重跑合併")
    ap.add_argument("--force-dedup", action="store_true", help="強制重跑去重")
    ap.add_argument("--force-upload", action="store_true",
                    help="強制重跑上傳 RAG storage")
    # 若 upload_storage.py / delete_storage.py 需要額外參數，可在此加入並往下傳遞
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

    # 3) 上傳 RAG storage
    uploaded_this_run = False
    if need_upload(args.force_upload):
        print("▶ 上傳：執行 upload_storage.py")
        r = subprocess.run([sys.executable, str(UPLOAD_SCRIPT)], check=False)
        if r.returncode != 0:
            sys.exit(r.returncode)
        uploaded_this_run = True
    else:
        print("✓ 上傳：已是最新，跳過")

    # 4) 僅當「本次有上傳成功」才清理舊向量庫
    if uploaded_this_run:
        print("▶ 清理：執行 delete_storage.py（僅保留最新 RAG 向量庫）")
        r = subprocess.run([sys.executable, str(DELETE_SCRIPT)], check=False)
        if r.returncode != 0:
            sys.exit(r.returncode)
    else:
        print("✓ 清理：本次未上傳新知識檔，跳過 delete_storage.py")

    print("✅ Pipeline 完成 →", OUTPUT_DEDUP_JSON)
    if RAG_ID_DIR.exists():
        print("🆔 最新 RAG id 檔目錄：", RAG_ID_DIR)


if __name__ == "__main__":
    main()
