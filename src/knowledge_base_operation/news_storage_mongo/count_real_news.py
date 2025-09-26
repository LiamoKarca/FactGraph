"""
# 1) 精準計數（整個 News.Real_News）
python count_real_news.py

# 2) 估計計數（只在無過濾條件時可用）
python count_real_news.py --estimated

# 3) 篩選特定發佈者並精準計數
python count_real_news.py --where '{"publisher":"CNA"}'

# 4) 指定連線字串並偷看 5 筆 _id
python count_real_news.py --uri "mongodb+srv://<user>:<pwd>@<cluster>/" --peek 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from dotenv import load_dotenv

DB_NAME = "News"
COLL_NAME = "Real_News"

load_dotenv()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Count documents in MongoDB {DB_NAME}.{COLL_NAME} (exact or estimated)."
    )
    p.add_argument("--uri", type=str, default=os.getenv("MONGODB_URI"),
                   help="MongoDB connection URI. Defaults to env MONGODB_URI.")
    p.add_argument("--where", type=str, default=None,
                   help='Optional JSON filter, e.g. \'{"publisher":"CNA"}\'.')
    p.add_argument("--estimated", action="store_true",
                   help="Use estimatedDocumentCount when no filter is provided.")
    p.add_argument("--peek", type=int, default=0,
                   help="Optionally print first N docs (_id only). Default 0.")
    p.add_argument("--timeoutMS", type=int, default=30000,
                   help="Server selection / operation timeout in ms. Default 30000.")
    return p.parse_args()

def load_filter(where: str | None) -> Dict[str, Any]:
    if not where:
        return {}
    try:
        q = json.loads(where)
        if not isinstance(q, dict):
            raise ValueError("Filter JSON must be an object (dict).")
        return q
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --where: {e}")

def main() -> int:
    args = parse_args()

    if not args.uri:
        print("❌ 缺少 MongoDB 連線字串。請用 --uri 指定或在 .env 設定 MONGODB_URI。", file=sys.stderr)
        return 2

    try:
        client = MongoClient(args.uri, serverSelectionTimeoutMS=args.timeoutMS)
        client.admin.command("ping")  # test connection
    except (ConnectionFailure, ConfigurationError) as e:
        print(f"❌ 無法連線 MongoDB：{e}", file=sys.stderr)
        return 2

    coll = client[DB_NAME][COLL_NAME]

    try:
        filt = load_filter(args.where)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 2

    try:
        if args.estimated and not filt:
            count = coll.estimated_document_count()
            mode = "estimatedDocumentCount (估計)"
        else:
            count = coll.count_documents(filt, maxTimeMS=args.timeoutMS)
            mode = "countDocuments (精準)"
    except Exception as e:
        print(f"❌ 計數時發生錯誤：{e}", file=sys.stderr)
        return 2

    print("━━━━━━━━ Real_News Collection Counter ━━━━━━━━")
    shown_uri = args.uri.split('@')[-1] if '@' in args.uri else args.uri
    print(f"URI        : {shown_uri}")
    print(f"Database   : {DB_NAME}")
    print(f"Collection : {COLL_NAME}")
    print(f"Mode       : {mode}")
    print(f"Filter     : {filt if filt else '{}'}")
    print(f"Count      : {count}")

    if args.peek > 0:
        try:
            cur = coll.find(filt, {"_id": 1}).limit(args.peek)
            print(f"\nTop {args.peek} docs (_id):")
            for d in cur:
                print(f"  - {d.get('_id')}")
        except Exception as e:
            print(f"⚠️ peek 失敗：{e}", file=sys.stderr)

    client.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
