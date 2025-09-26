"""
從 Neo4j 匯出知識圖譜三元組到 CSV（不依賴伺服器端檔案寫入）。
輸出：experiment/data/raw/neo4j-kg-raw-graph.csv

欄位：
- head, relation, tail
- head_props, rel_props, tail_props  （以 JSON 字串儲存）

用法（預設即可）：
    python experiment/preliminary_work/neo4j-data-extraction.py

可選參數：
    --out experiment/data/raw/neo4j-kg-raw-graph.csv
    --database neo4j
    --limit 0            # >0 時僅匯出前 N 筆，除錯用
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from pathlib import Path
from typing import Dict, Any, Iterable
from neo4j import GraphDatabase, Driver

from dotenv import load_dotenv
load_dotenv()

DEFAULT_OUT = "experiment/data/raw/neo4j-kg-raw-graph.csv"
DEFAULT_DB = "neo4j"

CYPHER = """
MATCH (h)-[r]->(t)
RETURN
  h.name  AS head,
  type(r) AS relation,
  t.name  AS tail,
  properties(h) AS head_props,
  properties(r) AS rel_props,
  properties(t) AS tail_props
"""


def iter_records(driver: Driver, database: str, limit: int = 0) -> Iterable[Dict[str, Any]]:
    q = CYPHER + ("" if limit <= 0 else f"\nLIMIT {int(limit)}")
    with driver.session(database=database) as s:
        res = s.run(q)
        for rec in res:
            yield {
                "head": rec["head"],
                "relation": rec["relation"],
                "tail": rec["tail"],
                # 以 JSON 字串落地，避免 CSV 內嵌 dict 型態不一致
                "head_props": json.dumps(rec["head_props"] or {}, ensure_ascii=False),
                "rel_props":  json.dumps(rec["rel_props"] or {}, ensure_ascii=False),
                "tail_props": json.dumps(rec["tail_props"] or {}, ensure_ascii=False),
            }


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--database", default=os.getenv("NEO4J_DATABASE", DEFAULT_DB))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not (uri and user and pwd):
        raise RuntimeError(
            "請於環境變數或 .env 設定 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    total = 0
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["head", "relation", "tail",
                        "head_props", "rel_props", "tail_props"]
        )
        writer.writeheader()
        for row in iter_records(driver, args.database, args.limit):
            writer.writerow(row)
            total += 1

    driver.close()
    print(f"✅ 匯出完成：{out_path}（共 {total:,} 行）")


if __name__ == "__main__":
    main()
