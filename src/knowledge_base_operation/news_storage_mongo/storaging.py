"""
• 將 去重後的四家媒體 json 檔，匯入 MongoDB 的 News.Real_News
• 新聞識別 ID ，利用 hash.py 產生哈希值：_id = publisher_titleHash8_YYYYMMDD
• 若 _id 已存在自動跳過（DuplicateKeyError）
"""

from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Iterable, List, Dict, Any
from pymongo import MongoClient, errors

from hash import generate_news_id

from dotenv import load_dotenv
load_dotenv()


def _normalize_windows_like_path(raw: str) -> Path:
    """
    將含有反斜線（Windows 風格）的字串，安全轉為跨平台 Path。
    不改變原字串常量，只在程式內部做正規化。
    """
    return Path(*raw.split("\\")).resolve()


def _iter_source_paths(src: Path) -> Iterable[Path]:
    """
    若傳入為資料夾：回傳底下所有 *.json
    若傳入為檔案：回傳單一該檔案
    其他狀況：raise
    """
    if src.is_dir():
        yield from sorted(p for p in src.glob("*.json") if p.is_file())
    elif src.is_file():
        yield src
    else:
        raise FileNotFoundError(f"讀取不到 {src}，請檢查路徑！")


def _load_news_items(path: Path) -> List[Dict[str, Any]]:
    """
    載入單一 JSON 檔；若內容是 dict 則轉成單一元素 list，若是 list 則原樣回傳。
    """
    with path.open(encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data

    raise ValueError(f"JSON 內容格式非 dict/list：{path}")


def _require_mongo_client() -> MongoClient:
    """
    依環境變數 MONGODB_URI 建立 MongoClient；若未設置則明確拋錯。
    """
    mongo_uri: str | None = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("環境變數 MONGODB_URI 未設置，無法連線 MongoDB。")
    return MongoClient(mongo_uri)


def main() -> None:
    # ──────────────── MongoDB 連線 ────────────────
    client = _require_mongo_client()
    db = client["News"]
    coll = db["Real_News"]

    # ──────────────── 資料夾設定 ────────────────
    # 指向"去重"後的新聞媒原檔
    raw_path = r"data\processed\news_merge\cna_ey_moi_pts_DEDUP.json"  # 原字串保留
    json_files = _normalize_windows_like_path(raw_path)

    inserted_total = 0
    skipped_total = 0

    try:
        for file_path in _iter_source_paths(json_files):
            print(f"⏳ 處理 {file_path.name} ...")
            news_items = _load_news_items(file_path)

            inserted = 0
            skipped = 0

            for item in news_items:
                try:
                    news_id = generate_news_id(
                        publisher=item["publisher"],
                        title=item["title"],
                        date=item["date"],
                    )
                    item["_id"] = news_id
                    coll.insert_one(item)
                    inserted += 1

                except errors.DuplicateKeyError:
                    # _id 重複，自動跳過
                    skipped += 1

                except KeyError as exc:
                    print(f"  ⚠️  缺少欄位 {exc}，已跳過：{item}")
                    skipped += 1

            print(f"  ✅ 新增 {inserted}，跳過 {skipped}")
            inserted_total += inserted
            skipped_total += skipped

        print(f"\n🎉 全部完成：共新增 {inserted_total} 筆，跳過 {skipped_total} 筆")
        print("    請檢查 MongoDB 資料庫是否正確！")

    finally:
        client.close()


if __name__ == "__main__":
    main()
