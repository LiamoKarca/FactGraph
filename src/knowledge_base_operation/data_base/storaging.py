"""
storaging.py ─ 資料夾批次匯入版
───────────────────────────────
• 將 data/conver_ok 資料夾下的所有 *.json 一次匯入 Real_News.news
• _id = publisher_titleHash8_YYYYMMDD（利用 HASH.py 產生）
• 若 _id 已存在自動跳過（DuplicateKeyError）
"""

import json
from pathlib import Path
from pymongo import MongoClient, errors
from hash import generate_news_id            # 確保 HASH.py 與此檔同資料夾

# ──────────────── MongoDB 連線 ────────────────
MONGO_URI = "mongodb://karca5103:lerryjoe5103@localhost:27017/?authSource=admin"
client    = MongoClient(MONGO_URI)
db        = client["News"]
#coll      = db["Real_News"]
coll      = db["Fake_News"]

# ──────────────── 資料夾設定 ────────────────
#DATA_DIR  = Path(r"data\convert_ok\real")          # ★ 指向你的資料夾
DATA_DIR  = Path(r"data\convert_ok\fake")          # ★ 指向你的資料夾
json_files = list(DATA_DIR.glob("*.json"))   # 找出所有 .json

if not json_files:
    raise FileNotFoundError(f"在 {DATA_DIR} 找不到任何 .json 檔")

inserted_total, skipped_total = 0, 0

for file in json_files:
    print(f"⏳ 處理 {file.name} ...")
    with file.open(encoding="utf-8") as f:
        news_items = json.load(f)

    # 若單篇 → 轉成 list
    if isinstance(news_items, dict):
        news_items = [news_items]

    inserted, skipped = 0, 0
    for item in news_items:
        try:
            news_id = generate_news_id(
                publisher=item["publisher"],
                title=item["title"],
                date=item["date"]
            )
            item["_id"] = news_id
            coll.insert_one(item)
            inserted += 1

        except errors.DuplicateKeyError:
            skipped += 1
        except KeyError as e:
            print(f"  ⚠️  缺少欄位 {e}，已跳過：{item}")
            skipped += 1

    print(f"  ✅ 新增 {inserted}，跳過 {skipped}")
    inserted_total += inserted
    skipped_total  += skipped

print(f"\n🎉 全部完成：共新增 {inserted_total} 筆，跳過 {skipped_total} 筆")
print("    請檢查 MongoDB 資料庫是否正確！")
client.close()