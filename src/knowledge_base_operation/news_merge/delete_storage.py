"""
刪除 OpenAI 上較舊的 RAG 向量庫與其檔案，僅保留最新一份。

流程：
  - 解析 rag_storage_id 目錄下的所有檔名（yyyy-mm-dd-hhmm）
  - 找出最新的一個檔案，其內容為 vector_store_id（保留）
  - 其他較舊檔案所對應的 vector_store 全部刪除（含底下 Files），並刪除本機舊 id 檔
"""

import argparse
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Set

from openai import OpenAI
from openai.pagination import SyncPage

RAG_ID_DIR_DEFAULT = "data/processed/news_merge/rag_storage_id"
STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{4}$")  # yyyy-mm-dd-hhmm

# ──────────────────────────────────────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────────────────────────────────────


def get_api_key() -> str:
    key = os.getenv("GPT_API") or os.getenv(
        "OPENAI_API_KEY") or os.getenv("OPENAI_API")
    if not key:
        raise RuntimeError("找不到 GPT_API / OPENAI_API_KEY / OPENAI_API")
    return key


def parse_stamp(name: str) -> Optional[datetime]:
    if not STAMP_RE.match(name):
        return None
    return datetime.strptime(name, "%Y-%m-%d-%H%M")


def load_local_ids(root: Path) -> List[Tuple[datetime, Path, str]]:
    results: List[Tuple[datetime, Path, str]] = []
    if not root.exists():
        return results
    for p in root.iterdir():
        if not p.is_file():
            continue
        dt = parse_stamp(p.name)
        if not dt:
            continue
        try:
            text = p.read_text(encoding="utf-8-sig").strip()
        except Exception:
            continue
        if not text:
            continue
        results.append((dt, p, text))
    return sorted(results, key=lambda x: x[0])  # 由舊到新


def list_vector_store_files(client: OpenAI, vector_store_id: str) -> List[str]:
    file_ids: List[str] = []
    after: Optional[str] = None
    while True:
        page: SyncPage = client.vector_stores.files.list(
            vector_store_id=vector_store_id, limit=100, after=after
        )
        for item in page.data:
            if getattr(item, "id", None):
                file_ids.append(item.id)
        if not page.has_more:
            break
        after = page.last_id
    return file_ids


def delete_vector_store_and_files(
    client: OpenAI,
    vector_store_id: str,
    dry_run: bool = False,
    keep_files: bool = False,
) -> None:
    """刪除向量庫與其底下檔案（除非 keep_files=True）。"""
    try:
        file_ids = list_vector_store_files(client, vector_store_id)
    except Exception as e:
        print(f"[WARN] 無法列出 VS 檔案 vs={vector_store_id}: {e}")
        file_ids = []

    # 先解除綁定
    for fid in file_ids:
        try:
            if dry_run:
                print(
                    f"[DRY] detach file from vector_store: vs={vector_store_id} file={fid}")
            else:
                client.vector_stores.files.delete(
                    vector_store_id=vector_store_id, file_id=fid)
        except Exception as e:
            print(f"[WARN] detach 失敗 vs={vector_store_id} file={fid}: {e}")

    # 刪檔案（若 keep_files=False）
    if not keep_files:
        for fid in file_ids:
            try:
                if dry_run:
                    print(f"[DRY] delete file: {fid}")
                else:
                    client.files.delete(fid)
            except Exception as e:
                print(f"[WARN] 刪除 file 失敗 file={fid}: {e}")

    # 最後刪除向量庫本身
    try:
        if dry_run:
            print(f"[DRY] delete vector_store: {vector_store_id}")
        else:
            client.vector_stores.delete(vector_store_id)
    except Exception as e:
        print(f"[WARN] 刪除 vector_store 失敗 vs={vector_store_id}: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="刪除舊的 OpenAI RAG 向量庫，只保留最新一份。")
    ap.add_argument("--root-dir", default=RAG_ID_DIR_DEFAULT,
                    help="本機向量庫 ID 檔所在目錄")
    ap.add_argument("--dry-run", action="store_true", help="僅列印不刪除")
    ap.add_argument("--keep-local", action="store_true", help="保留本機舊 id 檔")
    ap.add_argument("--keep-files", action="store_true",
                    help="只刪向量庫，不刪底下 Files")
    args = ap.parse_args()

    root = Path(args.root_dir)
    entries = load_local_ids(root)
    if len(entries) <= 1:
        print("[INFO] 沒有或僅有一個向量庫 ID，無需清理。")
        return

    newest_dt, newest_path, newest_vs_id = entries[-1]
    print(f"[INFO] 保留最新向量庫：{newest_path.name} -> {newest_vs_id}")

    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    old_entries = entries[:-1]
    for dt, path, vs_id in old_entries:
        print(f"[CLEAN] 刪除較舊向量庫：{path.name} -> {vs_id}")
        delete_vector_store_and_files(
            client=client,
            vector_store_id=vs_id,
            dry_run=args.dry_run,
            keep_files=args.keep_files,
        )
        # 刪掉本機 id 檔
        if not args.keep_local:
            if args.dry_run:
                print(f"[DRY] remove local id file: {path}")
            else:
                try:
                    path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"[WARN] 移除本機 id 檔失敗：{path} - {e}")

    print("[DONE] 清理完成。")


if __name__ == "__main__":
    main()
