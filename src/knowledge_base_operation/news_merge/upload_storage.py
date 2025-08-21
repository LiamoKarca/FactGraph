"""
將本地 JSON 上傳至 OpenAI 向量庫（RAG storage），並輸出向量庫 ID 檔。
預設：
  - 輸入檔: data/processed/news_merge/cna_ey_moi_pts_DEDUP.json
  - 輸出目錄: data/processed/news_merge/rag_storage_id
  - 檔名格式: yyyy-mm-dd-hhmm（無副檔名）
環境變數：
  - GPT_API 或 OPENAI_API_KEY 或 OPENAI_API
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from openai import OpenAI


DEFAULT_INPUT = "data/processed/news_merge/cna_ey_moi_pts_DEDUP.json"
DEFAULT_OUTDIR = "data/processed/news_merge/rag_storage_id"


def get_api_key() -> str:
    # 支援三種環境變數：GPT_API、OPENAI_API_KEY、OPENAI_API
    key = os.getenv("GPT_API") or os.getenv(
        "OPENAI_API_KEY") or os.getenv("OPENAI_API")
    if not key:
        raise RuntimeError(
            "找不到 GPT_API / OPENAI_API_KEY / OPENAI_API 任何一個環境變數。")
    return key


def now_stamp() -> str:
    # yyyy-mm-dd-hhmm
    return datetime.now().strftime("%Y-%m-%d-%H%M")


def create_vector_store(client: OpenAI, name: Optional[str] = None) -> str:
    vs = client.vector_stores.create(
        name=name or f"factgraph-news-{now_stamp()}")
    if not vs or not getattr(vs, "id", None):
        raise RuntimeError("建立向量庫失敗：未取得有效 ID。")
    return vs.id


def upload_file_and_attach(client: OpenAI, vector_store_id: str, file_path: str) -> str:
    # 1) 上傳檔案
    with open(file_path, "rb") as f:
        fobj = client.files.create(file=f, purpose="assistants")
    if not fobj or not getattr(fobj, "id", None):
        raise RuntimeError("檔案上傳失敗：未取得 file_id。")

    # 2) 綁定至向量庫
    client.vector_stores.files.create(
        vector_store_id=vector_store_id, file_id=fobj.id)
    return fobj.id


def write_id_file(outdir: str, vector_store_id: str, stamp: Optional[str] = None) -> Path:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    stamp = stamp or now_stamp()
    outpath = Path(outdir) / stamp  # 檔名即時間戳，無副檔名
    outpath.write_text(vector_store_id, encoding="utf-8")
    return outpath


def main():
    parser = argparse.ArgumentParser(
        description="Upload JSON to OpenAI Vector Store (RAG storage) and save its ID.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="要上傳的 JSON 路徑")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="向量庫 ID 輸出目錄")
    parser.add_argument("--name", default=None, help="向量庫名稱（可選）")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"找不到輸入檔：{in_path}")

    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    # 建立向量庫
    vs_id = create_vector_store(client, name=args.name)

    # 上傳並綁定
    file_id = upload_file_and_attach(client, vs_id, str(in_path))

    # 寫出向量庫 ID 檔
    stamp = now_stamp()
    outpath = write_id_file(args.outdir, vs_id, stamp=stamp)

    print("[OK] 已建立向量庫並綁定檔案")
    print(f"  - vector_store_id: {vs_id}")
    print(f"  - attached_file_id: {file_id}")
    print(f"  - id 檔案路徑: {outpath}")


if __name__ == "__main__":
    main()
