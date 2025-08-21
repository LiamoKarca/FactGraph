"""
刪除 OpenAI 上較舊的 RAG 向量庫與其檔案，僅保留最新一份。
依據本機資料夾 data/processed/news_merge/rag_storage_id/ 中的時間戳檔決定「最新」。

預設行為：
  - 解析 rag_storage_id 目錄下的所有檔名（yyyy-mm-dd-hhmm）
  - 找出最新的一個檔案，其內容為 vector_store_id（保留）
  - 其他較舊檔案所對應的 vector_store 全部刪除（含底下 Files），並刪除本機舊 id 檔
  - 會保護最新向量庫所用的 file_id，不會誤刪

環境變數：
  - GPT_API 或 OPENAI_API_KEY 或 OPENAI_API

選項：
  --root-dir         目錄（預設：data/processed/news_merge/rag_storage_id）
  --dry-run          只顯示將刪除的內容，不實際刪除
  --keep-local       刪除雲端對應項後，保留本機舊 id 檔
  --keep-files       只刪向量庫，不刪底下 Files
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


def get_api_key() -> str:
    key = os.getenv("GPT_API") or os.getenv(
        "OPENAI_API_KEY") or os.getenv("OPENAI_API")
    if not key:
        raise RuntimeError(
            "找不到 GPT_API / OPENAI_API_KEY / OPENAI_API 任何一個環境變數。")
    return key


def parse_stamp(name: str) -> Optional[datetime]:
    if not STAMP_RE.match(name):
        return None
    # yyyy-mm-dd-hhmm
    return datetime.strptime(name, "%Y-%m-%d-%H%M")


def load_local_ids(root: Path) -> List[Tuple[datetime, Path, str]]:
    """
    回傳 [(stamp_dt, path, vector_store_id), ...]，只讀取「內容不為空」的檔案。
    """
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
            text = p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not text:
            continue
        results.append((dt, p, text))
    return sorted(results, key=lambda x: x[0])  # 由舊到新


def list_vector_store_files(client: OpenAI, vector_store_id: str) -> List[str]:
    """
    列出向量庫底下的 file_id（分頁安全）。
    """
    file_ids: List[str] = []
    after: Optional[str] = None
    while True:
        page: SyncPage = client.vector_stores.files.list(
            vector_store_id=vector_store_id, limit=100, after=after
        )
        for item in page.data:
            # item 有 id 屬性
            if getattr(item, "id", None):
                file_ids.append(item.id)
        if not page.has_more:
            break
        after = page.last_id
    return file_ids


def delete_vector_store_and_files(
    client: OpenAI,
    vector_store_id: str,
    protected_file_ids: Set[str],
    dry_run: bool = False,
    keep_files: bool = False,
) -> None:
    """
    刪除向量庫，以及（預設）其底下所有 Files。
    - protected_file_ids 會被保護，不會刪除。
    - keep_files=True 時，不刪除 Files，只刪向量庫關聯與向量庫本身。
    """
    try:
        file_ids = list_vector_store_files(client, vector_store_id)
    except Exception as e:
        print(f"[WARN] 讀取向量庫檔案失敗（{vector_store_id}）：{e}")
        file_ids = []

    # 先解除綁定（必要時）
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

    # 刪除 Files（除非受保護或選擇保留）
    if not keep_files:
        for fid in file_ids:
            if fid in protected_file_ids:
                print(f"[SKIP] 保護中的 file_id，不刪除: {fid}")
                continue
            try:
                if dry_run:
                    print(f"[DRY] delete file: {fid}")
                else:
                    client.files.delete(fid)
            except Exception as e:
                print(f"[WARN] 刪除 file 失敗 file={fid}: {e}")

    # 最後刪除向量庫
    try:
        if dry_run:
            print(f"[DRY] delete vector_store: {vector_store_id}")
        else:
            client.vector_stores.delete(vector_store_id)
    except Exception as e:
        print(f"[WARN] 刪除 vector_store 失敗 vs={vector_store_id}: {e}")


def main():
    ap = argparse.ArgumentParser(description="刪除舊的 OpenAI RAG 向量庫，只保留最新一份。")
    ap.add_argument("--root-dir", default=RAG_ID_DIR_DEFAULT,
                    help="本機向量庫 ID 檔所在目錄")
    ap.add_argument("--dry-run", action="store_true", help="僅列印將會刪除的項目，不實際刪除")
    ap.add_argument("--keep-local", action="store_true",
                    help="刪除雲端後保留本機舊 id 檔")
    ap.add_argument("--keep-files", action="store_true",
                    help="只刪向量庫，不刪底下 Files")
    args = ap.parse_args()

    root = Path(args.root_dir)
    entries = load_local_ids(root)
    if len(entries) <= 1:
        print("[INFO] 沒有或僅有一個向量庫 ID，無需清理。")
        return

    # 最新一筆
    newest_dt, newest_path, newest_vs_id = entries[-1]
    print(f"[INFO] 保留最新向量庫：{newest_path.name} -> {newest_vs_id}")

    # 取得最新向量庫的 file_id（避免被刪）
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    try:
        protected_files = set(list_vector_store_files(client, newest_vs_id))
    except Exception as e:
        print(f"[WARN] 無法取得最新向量庫的檔案清單，將不保護任何 file（{e}）")
        protected_files = set()

    # 逐一刪除舊向量庫
    old_entries = entries[:-1]
    for dt, path, vs_id in old_entries:
        print(f"[CLEAN] 刪除較舊向量庫：{path.name} -> {vs_id}")
        delete_vector_store_and_files(
            client=client,
            vector_store_id=vs_id,
            protected_file_ids=protected_files,
            dry_run=args.dry_run,
            keep_files=args.keep_files,
        )

        # 刪除本機舊 id 檔
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
