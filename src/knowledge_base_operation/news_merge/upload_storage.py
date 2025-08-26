"""
將本地 JSON 上傳至 OpenAI 向量庫（RAG storage），自動「切片」並等待索引完成。
- 目的：避免單一巨大 JSON 造成「Files attached → Failed（invalid_file / too large）」。
- 成功條件：所有分片皆索引成功，才寫入最新 Vector Store ID 檔；任何一片失敗則清除本次 VS 並退出非 0。

用法：
$ python src/knowledge_base_operation/news_merge/upload_storage.py
$ python src/knowledge_base_operation/news_merge/upload_storage.py --input path/to/file.json --name my-vs --shard-bytes 5242880
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from openai import OpenAI

DEFAULT_INPUT = "data/processed/news_merge/cna_ey_moi_pts_DEDUP.json"
DEFAULT_OUTDIR = "data/processed/news_merge/rag_storage_id"

# 目標每片大小（粗略，實際會略有偏差；預設 5MB）
DEFAULT_SHARD_BYTES = 5 * 1024 * 1024  # 5 MB

# 產生分片的暫存資料夾（位於輸入檔相同目錄）
SHARD_DIR_NAME = "_shards"


# ──────────────────────────────────────────────────────────────────────────────
# 基本工具
# ──────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.getenv("GPT_API") or os.getenv(
        "OPENAI_API_KEY") or os.getenv("OPENAI_API")
    if not key:
        raise RuntimeError(
            "找不到 GPT_API / OPENAI_API_KEY / OPENAI_API 任何一個環境變數。")
    return key


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H%M")


def _write_id_file(outdir: str, vector_store_id: str, stamp: str) -> Path:
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    outpath = outdir_p / stamp
    # 依需求以 UTF-8-SIG 寫入
    outpath.write_text(vector_store_id, encoding="utf-8-sig")
    return outpath


# ──────────────────────────────────────────────────────────────────────────────
# 分片（JSON → 多個 json）
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_json_shards(input_path: Path, target_bytes: int) -> List[Path]:
    """
    將輸入的 JSON 轉為多個 json 分片檔案：
    - 若原本就是 .json：依大小切片
    - 若是 .json（常見為巨大 list 或 {items:[...]}）：讀出每則，逐行寫入 json，並切片
    每行的結構：{"text": "<可檢索的正文>", "meta": {...可選}}
    回傳：所有分片檔案路徑（依序）
    """
    tmp_dir = input_path.parent / SHARD_DIR_NAME
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shards: List[Path] = []

    # 直接處理 json：逐行累積至 target_bytes，再滾動新檔
    if input_path.suffix.lower() == ".json":
        shard_idx = 0
        cur_bytes = 0
        cur_path = tmp_dir / f"{input_path.stem}.part{shard_idx:04d}.json"
        cur_f = cur_path.open("w", encoding="utf-8-sig")
        with input_path.open("r", encoding="utf-8-sig") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                # 確保每行為合法 JSON；失敗則包成 {"text": "..."}
                try:
                    json.loads(line)
                    out = line
                except Exception:
                    out = json.dumps({"text": line}, ensure_ascii=False)
                b = len(out.encode("utf-8-sig")) + 1  # + '\n'
                if cur_bytes + b > target_bytes and cur_bytes > 0:
                    cur_f.close()
                    shards.append(cur_path)
                    shard_idx += 1
                    cur_bytes = 0
                    cur_path = tmp_dir / \
                        f"{input_path.stem}.part{shard_idx:04d}.json"
                    cur_f = cur_path.open("w", encoding="utf-8-sig")
                cur_f.write(out + "\n")
                cur_bytes += b
        cur_f.close()
        if cur_path.exists() and cur_path.stat().st_size > 0:
            shards.append(cur_path)
        return shards

    # 處理 JSON：讀為 list（或 dict.items），每項轉為一行 json
    with input_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    # 支援兩種常見結構：直接 list 或 {items: [...]}
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        # 退一步：整份當作單一條目
        items = [data]

    def _line_of(item) -> str:
        # 嘗試組裝可檢索的「正文」文字；否則就序列化整個 item
        if isinstance(item, dict):
            parts = []
            for key in ("title", "headline", "subject"):
                if key in item and isinstance(item[key], str):
                    parts.append(item[key].strip())
            for key in ("content", "body", "text", "article", "summary"):
                if key in item and isinstance(item[key], str):
                    parts.append(item[key].strip())
            text = "\n".join([p for p in parts if p]).strip()
            if not text:
                text = json.dumps(item, ensure_ascii=False)
            # 將其他欄位塞進 meta 方便檢索回顧
            meta = {k: v for k, v in item.items() if k not in (
                "title", "headline", "subject", "content", "body", "text", "article", "summary")}
            obj = {"text": text, "meta": meta} if meta else {"text": text}
        else:
            obj = {"text": str(item)}
        return json.dumps(obj, ensure_ascii=False)

    shard_idx = 0
    cur_bytes = 0
    cur_path = tmp_dir / f"{input_path.stem}.part{shard_idx:04d}.json"
    cur_f = cur_path.open("w", encoding="utf-8-sig")

    for it in items:
        line = _line_of(it)
        b = len(line.encode("utf-8-sig")) + 1
        if cur_bytes + b > target_bytes and cur_bytes > 0:
            cur_f.close()
            shards.append(cur_path)
            shard_idx += 1
            cur_bytes = 0
            cur_path = tmp_dir / f"{input_path.stem}.part{shard_idx:04d}.json"
            cur_f = cur_path.open("w", encoding="utf-8-sig")
        cur_f.write(line + "\n")
        cur_bytes += b

    cur_f.close()
    if cur_path.exists() and cur_path.stat().st_size > 0:
        shards.append(cur_path)
    return shards


# ──────────────────────────────────────────────────────────────────────────────
# 上傳 + 等待索引
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Upload JSON/json to Vector Store with sharding & polling.")
    ap.add_argument("--input",       default=DEFAULT_INPUT,
                    help="輸入 JSON/json 路徑")
    ap.add_argument("--outdir",      default=DEFAULT_OUTDIR,
                    help="向量庫 ID 輸出目錄")
    ap.add_argument("--name",        default=None,           help="向量庫名稱（可選）")
    ap.add_argument("--shard-bytes", type=int,
                    default=DEFAULT_SHARD_BYTES, help="每片目標大小（bytes），預設 5MB")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"找不到輸入檔：{in_path}")

    client = OpenAI(api_key=_get_api_key())

    # 建立向量庫
    vs = client.vector_stores.create(
        name=args.name or f"factgraph-news-{_now_stamp()}")
    vs_id = vs.id
    print(f"[VS] created: {vs_id}")

    # 產生分片
    shards = _ensure_json_shards(in_path, target_bytes=args.shard_bytes)
    print(f"[SHARD] 產生 {len(shards)} 片：")
    for s in shards:
        print(f"  - {s.name} ({s.stat().st_size/1024:.1f} KB)")

    # 逐片上傳並等待索引（全部須成功，否則撤銷本次 VS）
    failed: List[Tuple[Path, str]] = []
    succeeded = 0

    for s in shards:
        with s.open("rb") as fh:
            batch = client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vs_id,
                files=[fh],
            )
        print(
            f"[BATCH] {s.name}: status={batch.status} counts={getattr(batch, 'file_counts', None)}")

        # 逐一檢查該 VS 內檔案狀態（找出這一片是否成功）
        this_ok = False
        last_error_msg = ""
        page = client.vector_stores.files.list(
            vector_store_id=vs_id, limit=100)
        # 只要 VS 內出現一個 completed/succeeded 的新檔，就視為本片成功
        for item in page.data:
            f = client.vector_stores.files.retrieve(
                vector_store_id=vs_id, file_id=item.id)
            status = getattr(f, "status", "")
            if status in ("completed", "ready", "succeeded"):
                this_ok = True
            elif status == "failed":
                last_error_msg = str(getattr(f, "last_error", ""))
        if this_ok:
            succeeded += 1
        else:
            failed.append((s, last_error_msg))

    if failed:
        # 有任一片失敗 → 刪 VS 並列印錯誤原因（避免留下半殘 VS）
        try:
            client.vector_stores.delete(vs_id)
            print(f"[CLEANUP] deleted vector_store: {vs_id}")
        except Exception as e:
            print(f"[WARN] failed to delete vector_store {vs_id}: {e}")
        print("[ERROR] 以下分片索引失敗：")
        for s, err in failed:
            print(f"  - {s.name} | last_error={err}")
        raise SystemExit(2)

    # 全部成功 → 寫出最新 VS id 檔（UTF-8-SIG）
    outpath = _write_id_file(args.outdir, vs_id, stamp=_now_stamp())
    print("[OK] 向量庫已就緒，所有分片完成索引")
    print(f"  - vector_store_id: {vs_id}")
    print(f"  - id 檔案路徑: {outpath}")


if __name__ == "__main__":
    main()
