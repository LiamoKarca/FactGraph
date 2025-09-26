"""
• 從「合併原檔」匯入到 MongoDB: News.Real_News
• _id = {publisher}_{hash16(title|date|url|publisher)}_{YYYY-MM-DD}
  - 雜湊來源含：publisher, title, date, url（降低撞庫）
  - 雜湊長度：16 hex（blake2b digest_size=8）
• 必填欄位不可為空或空白（url/date/publisher/category/title/content）
• 分開統計：
  - dup_in_batch：同一批來源內的重覆（先擋）
  - dup_in_db   ：資料庫已存在的重覆（DuplicateKey 或預查命中）
  - invalid     ：欄位空白/缺漏
• 支援 --dry-run：只做統計不寫入
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient, errors

load_dotenv()

# === 必填欄位 ===
REQUIRED_FIELDS: tuple[str, ...] = (
    "url", "date", "publisher", "category", "title", "content")
INCLUDE_LABEL_IN_REQUIRED = False  # 如需把 label 也列為必填，改 True

# === 來源檔（合併原檔） ===
RAW_PATH = r"data\processed\news_merge\cna_ey_moi_pts.json"


# ──────────────── 工具函式 ────────────────
def _normalize_windows_like_path(raw: str) -> Path:
    return Path(*raw.split("\\")).resolve()


def _iter_source_paths(src: Path) -> Iterable[Path]:
    if src.is_dir():
        yield from sorted(p for p in src.glob("*.json") if p.is_file())
    elif src.is_file():
        yield src
    else:
        raise FileNotFoundError(f"讀取不到 {src}，請檢查路徑！")


def _load_news_items(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"JSON 內容格式非 dict/list：{path}")


def _require_mongo_client() -> MongoClient:
    mongo_uri: str | None = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("環境變數 MONGODB_URI 未設置，無法連線 MongoDB。")
    return MongoClient(mongo_uri)


def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _hash16(s: str) -> str:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8)
    return h.hexdigest()  # 16 hex


def generate_news_id(publisher: str, title: str, date: str, url: str) -> str:
    pub = _collapse_spaces(publisher)
    tit = _collapse_spaces(title)
    dat = _collapse_spaces(date)
    ur = _collapse_spaces(url)
    base = f"{tit}|{dat}|{ur}|{pub}".lower()
    h16 = _hash16(base)
    return f"{pub}_{h16}_{dat}"


def _validate_and_normalize_item(item: Dict[str, Any]) -> Tuple[bool, str]:
    required = list(REQUIRED_FIELDS)
    if INCLUDE_LABEL_IN_REQUIRED:
        required.append("label")
    for key in required:
        if key not in item:
            return False, f"缺少必填欄位：{key}"
        val = item[key]
        if isinstance(val, str):
            val_norm = _collapse_spaces(val)
            if val_norm == "":
                return False, f"欄位 {key} 為空白字串"
            item[key] = val_norm
    # label 非必填，若存在試做溫和正規化
    if "label" in item and item["label"] is not None and isinstance(item["label"], str):
        low = item["label"].strip().lower()
        if low in ("true", "1", "yes"):
            item["label"] = True
        elif low in ("false", "0", "no"):
            item["label"] = False
        else:
            item["label"] = _collapse_spaces(item["label"])
    return True, ""

# ──────────────── 參數 ────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Import merged news JSON into MongoDB News.Real_News with detailed counters.")
    p.add_argument("--src", type=str, default=RAW_PATH,
                   help="JSON 檔路徑或資料夾（預設使用程式內 RAW_PATH）")
    p.add_argument("--dry-run", action="store_true", help="只統計不寫入")
    return p.parse_args()

# ──────────────── 主流程 ────────────────


def main() -> None:
    args = parse_args()

    # 連線
    client = _require_mongo_client()
    db = client["News"]
    coll = db["Real_News"]

    # 載入來源
    json_target = _normalize_windows_like_path(args.src)
    src_paths = list(_iter_source_paths(json_target))
    if not src_paths:
        raise FileNotFoundError(f"來源中沒有 JSON：{json_target}")

    # 先抓 DB 現有 _id（用於區分 dup_in_db）
    # 若集合未達數百萬級，這樣做很安全且快速
    existing_ids = set(doc["_id"] for doc in coll.find({}, {"_id": 1}))
    print(f"🔎 DB 既有唯一 _id 數：{len(existing_ids)}")

    inserted_total = 0
    dup_in_db_total = 0
    dup_in_batch_total = 0
    invalid_total = 0
    seen_ids_global: set[str] = set()  # 也可跨檔去重（保守：避免同一批多檔重覆）

    try:
        for file_path in src_paths:
            print(f"⏳ 處理 {file_path.name} ...")
            news_items = _load_news_items(file_path)

            inserted = 0
            dup_in_db = 0
            dup_in_batch = 0
            invalid = 0

            seen_ids: set[str] = set()  # 檔內去重

            for item in news_items:
                ok, reason = _validate_and_normalize_item(item)
                if not ok:
                    invalid += 1
                    continue

                news_id = generate_news_id(
                    publisher=item["publisher"],
                    title=item["title"],
                    date=item["date"],
                    url=item["url"],
                )
                item["_id"] = news_id

                # 先做批內去重（檔內 + 跨檔）
                if news_id in seen_ids or news_id in seen_ids_global:
                    dup_in_batch += 1
                    continue
                seen_ids.add(news_id)
                seen_ids_global.add(news_id)

                # 再判斷 DB 是否已有
                if news_id in existing_ids:
                    dup_in_db += 1
                    continue

                if args.dry_run:
                    # 不寫入，只統計，視為「可新增」
                    inserted += 1
                else:
                    try:
                        coll.insert_one(item)
                        inserted += 1
                        existing_ids.add(news_id)  # 寫入成功後同步更新快取集合
                    except errors.DuplicateKeyError:
                        # 如果競態或其他程序剛好插入，這裡仍可能撞到
                        dup_in_db += 1

            print(
                f"  ✅ 可新增/已新增 {inserted}，dup_in_db {dup_in_db}，dup_in_batch {dup_in_batch}，invalid {invalid}")
            inserted_total += inserted
            dup_in_db_total += dup_in_db
            dup_in_batch_total += dup_in_batch
            invalid_total += invalid

        print("\n🎉 全部完成")
        mode = "（乾跑，不寫入）" if args.dry_run else ""
        print(f"    模式：{mode}")
        print(f"    可新增/已新增：{inserted_total}")
        print(f"    dup_in_db    ：{dup_in_db_total}  ← 與『DB 已有』直接對應")
        print(f"    dup_in_batch ：{dup_in_batch_total}  ← 同一批來源檔內重覆")
        print(f"    invalid      ：{invalid_total}  ← 欄位空白/缺漏")
        if not args.dry_run:
            # 最終 DB 數量（精準）
            final_count = coll.count_documents({})
            print(f"    最終 DB 總筆數（countDocuments）：{final_count}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
