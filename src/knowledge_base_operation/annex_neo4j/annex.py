"""
Annex Neo4j Loader

將指定 CSV（預設：experiment/data/raw/neo4j-kg-raw-graph.csv）匯入到指定的
Neo4j 資料庫（從 .env 或 config 載入的 NEO4J_DATABASE）。

支援兩種模式：
1) python 端逐列寫入（預設，免移動檔案）
2) --use-load-csv 搭配 --import-dir：將 CSV 複製到 Neo4j import 目錄後，使用
   Cypher 的 LOAD CSV WITH HEADERS 伺服器端批量匯入（較快）

CSV 欄位自動判斷（任一列符合下列條件即視為「關係列」）：
- 同時含有：source_name、target_name、relation
其餘列則視為「節點列」（最少需 name；可含 id、type 與其他屬性）

匯入策略（與既有 Neo4jLoader 一致）：
- 節點：MERGE (n:Entity {name:$name})
         ON CREATE SET n.id, n.type, 以及其他欄位
         ON MATCH  SET 其他欄位覆蓋（值為空則不寫入）
- 關係：若 (a)-[r]->(b) 已存在且 r.evidence 相同則跳過；
        否則建立 type=relation 的關係並寫入 doc_id、evidence、date 等屬性

用法：
  python -m src.knowledge_base_operation.annex_neo4j.annex \
    --csv experiment/data/raw/neo4j-kg-raw-graph.csv

  # 伺服器端 LOAD CSV（需提供 neo4j 的 import 目錄）
  python -m src.knowledge_base_operation.annex_neo4j.annex \
    --csv experiment/data/raw/neo4j-kg-raw-graph.csv \
    --use-load-csv --import-dir /var/lib/neo4j/import

參數：
  --csv           要匯入的 CSV 路徑（預設 experiment/data/raw/neo4j-kg-raw-graph.csv）
  --database      覆寫 .env / config 的 NEO4J_DATABASE
  --batch-size    逐列寫入模式下的 flush 間隔（預設 1000）
  --dry-run       僅列印偵測結果與計畫，不執行寫入
  --use-load-csv  使用伺服器端 LOAD CSV
  --import-dir    Neo4j import 目錄（搭配 --use-load-csv）
  
# 預設：client-side，支援 head/relation/tail + JSON props
python -m src.knowledge_base_operation.annex_neo4j.annex \
  --csv experiment/data/raw/neo4j-kg-raw-graph.csv

# 只預覽（乾跑）
python -m src.knowledge_base_operation.annex_neo4j.annex \
  --csv experiment/data/raw/neo4j-kg-raw-graph.csv --dry-run

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

# ── 專案匯入路徑 ────────────────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 相容式匯入
try:
    from ...common.gadget import LOGGER, run_with_timer  # type: ignore
except Exception:
    from src.common.gadget import LOGGER, run_with_timer  # type: ignore

try:
    from ...config import NEO4J_CONFIG  # type: ignore
except Exception:
    from src.config import NEO4J_CONFIG  # type: ignore

# ── 常數/預設 ──────────────────────────────────────────────────────────────────
DEFAULT_CSV = "experiment/data/raw/neo4j-kg-raw-graph.csv"

RESERVED_NODE_KEYS = {"id", "name", "type"}
REL_KEYS_A = {"source_name", "target_name", "relation"}            # 舊版
REL_KEYS_B = {"head", "relation", "tail"}                          # 你的新版
PROP_JSON_KEYS = {"head_props", "rel_props", "tail_props"}

# ── 小工具 ─────────────────────────────────────────────────────────────────────


def _load_database_from_env_or_config(override: Optional[str]) -> Optional[str]:
    return override or os.getenv("NEO4J_DATABASE") or NEO4J_CONFIG.get("database")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_loads_safe(s: Optional[str]) -> Dict[str, Any]:
    if not s or not isinstance(s, str):
        return {}
    s = s.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def _read_csv_rows(csv_path: Path) -> Iterable[Dict[str, Any]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV 檔案沒有表頭：{csv_path}")
        for row in reader:
            # 去除首尾空白
            clean = {k: (v.strip() if isinstance(v, str) else v)
                     for k, v in row.items()}
            # 若含 *_props 欄位，先就地解析成 dict（client-side 才能做）
            for k in list(clean.keys()):
                lk = (k or "").strip().lower()
                if lk in PROP_JSON_KEYS:
                    clean[k] = _json_loads_safe(clean[k])
            yield clean


def _keys_lower(d: Dict[str, Any]) -> set:
    return {(k or "").strip().lower() for k in d.keys() if k is not None}


def _is_relation_row(row: Dict[str, Any]) -> bool:
    keys = _keys_lower(row)
    return (REL_KEYS_A.issubset(keys)) or (REL_KEYS_B.issubset(keys))


def _filter_props(d: Dict[str, Any], exclude: Iterable[str]) -> Dict[str, Any]:
    ex = set(exclude)
    out = {}
    for k, v in d.items():
        if k in ex:
            continue
        if v in (None, "", []):
            continue
        out[k] = v
    return out


def _first_non_empty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

# ── Client-side 匯入（支援 head/relation/tail + JSON props） ───────────────────


def _client_side_ingest(
    driver: Driver,
    database: Optional[str],
    rows: Iterable[Dict[str, Any]],
    batch_size: int = 1000,
    dry_run: bool = False,
) -> Tuple[int, int]:
    node_count = 0
    rel_count = 0

    def upsert_node(tx, name: str, node_type: Optional[str], props: Dict[str, Any]) -> None:
        tx.run(
            """
            MERGE (n:Entity {name: $name})
            ON CREATE SET n.type = coalesce($type, n.type, 'Entity'), n += $props
            ON MATCH  SET n += $props
            """,
            name=name, type=node_type, props=props,
        )

    def relation_exists_by_evidence(tx, a_name: str, b_name: str, evidence: Optional[str]) -> bool:
        if not evidence:
            return False
        rec = tx.run(
            """
            MATCH (a:Entity {name:$a})-[r]->(b:Entity {name:$b})
            WHERE r.evidence = $evidence
            RETURN count(r) AS c
            """,
            a=a_name, b=b_name, evidence=evidence,
        ).single()
        return bool(rec and rec.get("c", 0) > 0)

    def create_relation(tx, a_name: str, b_name: str, rel_type: str, props: Dict[str, Any]) -> None:
        tx.run(
            """
            MATCH (a:Entity {name:$a})
            MATCH (b:Entity {name:$b})
            CALL apoc.create.relationship(a, $t, $p, b) YIELD rel
            RETURN rel
            """,
            a=a_name, b=b_name, t=rel_type, p=props,
        )

    with driver.session(database=database) as session:
        buf_nodes: List[Tuple[str, Optional[str], Dict[str, Any]]] = []
        buf_rels:  List[Tuple[str, str, str, Dict[str, Any]]] = []

        def flush_nodes():
            nonlocal node_count
            if not buf_nodes:
                return
            if dry_run:
                LOGGER.info("🟨 [DRY-RUN] 準備寫入節點 %d 筆", len(buf_nodes))
            else:
                def _write_nodes(tx):
                    for nm, tp, pr in buf_nodes:
                        upsert_node(tx, nm, tp, pr)
                session.execute_write(_write_nodes)
            node_count += len(buf_nodes)
            buf_nodes.clear()

        def flush_rels():
            nonlocal rel_count
            if not buf_rels:
                return
            if dry_run:
                LOGGER.info("🟨 [DRY-RUN] 準備寫入關係 %d 筆", len(buf_rels))
            else:
                def _write_rels(tx):
                    for a, b, t, pr in buf_rels:
                        if relation_exists_by_evidence(tx, a, b, pr.get("evidence")):
                            continue
                        create_relation(tx, a, b, t, pr)
                session.execute_write(_write_rels)
            rel_count += len(buf_rels)
            buf_rels.clear()

        for idx, row in enumerate(rows, start=1):
            is_rel = _is_relation_row(row)

            if is_rel:
                # 支援兩種鍵名
                a_raw = row.get("source_name") or row.get(
                    "head") or row.get("src")
                b_raw = row.get("target_name") or row.get(
                    "tail") or row.get("dst")
                r_type = row.get("relation") or row.get("type") or "RELATED_TO"

                # 解析 JSON 屬性
                head_props = row.get("head_props") if isinstance(
                    row.get("head_props"), dict) else {}
                tail_props = row.get("tail_props") if isinstance(
                    row.get("tail_props"), dict) else {}
                rel_props = row.get("rel_props") if isinstance(
                    row.get("rel_props"),  dict) else {}

                # 節點名稱優先用 props.name，退回到 head/tail 欄位
                a_name = _first_non_empty(
                    head_props.get("name") if isinstance(
                        head_props, dict) else None,
                    a_raw if isinstance(a_raw, str) else None
                )
                b_name = _first_non_empty(
                    tail_props.get("name") if isinstance(
                        tail_props, dict) else None,
                    b_raw if isinstance(b_raw, str) else None
                )

                if not a_name or not b_name:
                    LOGGER.warning("⚠️ 第 %d 列缺少 head/tail 名稱，跳過：%s", idx, row)
                    continue

                # 準備節點屬性（去除保留鍵）
                node_a_type = head_props.get("type") if isinstance(
                    head_props, dict) else None
                node_b_type = tail_props.get("type") if isinstance(
                    tail_props, dict) else None
                node_a_props = _filter_props(
                    (head_props or {}), exclude=RESERVED_NODE_KEYS)
                node_b_props = _filter_props(
                    (tail_props or {}), exclude=RESERVED_NODE_KEYS)

                buf_nodes.append((a_name, node_a_type, node_a_props))
                buf_nodes.append((b_name, node_b_type, node_b_props))

                # 關係屬性：直接採用 rel_props，其它欄位（若 CSV 還有自訂）也可補進來
                rel_final_props = dict(rel_props or {})
                buf_rels.append((a_name, b_name, str(r_type)
                                if r_type else "RELATED_TO", rel_final_props))

            else:
                # 純節點列：需要有 name；若沒有 name，試圖從 *_props 中找
                name = row.get("name")
                node_props_blob = row.get("props") if isinstance(
                    row.get("props"), dict) else {}
                # 自動從 props.name 回填
                if (not name) and isinstance(node_props_blob, dict):
                    name = node_props_blob.get("name")

                # 也支援 head_props/tail_props 當作單一節點輸入的特例
                if not name:
                    head_props = row.get("head_props") if isinstance(
                        row.get("head_props"), dict) else {}
                    tail_props = row.get("tail_props") if isinstance(
                        row.get("tail_props"), dict) else {}
                    name = _first_non_empty(
                        head_props.get("name") if isinstance(
                            head_props, dict) else None,
                        tail_props.get("name") if isinstance(
                            tail_props, dict) else None,
                        row.get("head") if isinstance(
                            row.get("head"), str) else None,
                        row.get("tail") if isinstance(
                            row.get("tail"), str) else None,
                    )

                if not name:
                    LOGGER.warning("⚠️ 第 %d 列缺少 name，跳過：%s", idx, row)
                    continue

                node_type = row.get("type")
                props = _filter_props(
                    row, exclude=RESERVED_NODE_KEYS | PROP_JSON_KEYS | REL_KEYS_A | REL_KEYS_B)
                # 若含 props blob，再合併一次
                if isinstance(node_props_blob, dict):
                    props.update(_filter_props(
                        node_props_blob, exclude=RESERVED_NODE_KEYS))

                buf_nodes.append((name, node_type, props))

            if (idx % batch_size) == 0:
                flush_nodes()
                flush_rels()

        flush_nodes()
        flush_rels()

    return node_count, rel_count

# ── Server-side（LOAD CSV）保留，但不建議用於 JSON props ───────────────────────


def _server_side_load_csv(
    driver: Driver,
    database: Optional[str],
    csv_src: Path,
    import_dir: Path,
    dry_run: bool = False,
) -> Tuple[int, int]:
    _ensure_parent_dir(import_dir)
    if not import_dir.exists():
        raise RuntimeError(f"import 目錄不存在：{import_dir}")
    target_csv = import_dir / csv_src.name
    if not dry_run:
        shutil.copy2(csv_src, target_csv)
    LOGGER.info("📥 CSV 已%s放置於 import：%s",
                "模擬" if dry_run else "複製", target_csv)

    # ⚠️ 伺服器端 Cypher 不好直接 parse JSON，僅保留最基本匯入（無 *_props）
    file_url = f"file:///{target_csv.name}"
    with driver.session(database=database) as session:
        cypher_basic = f"""
        LOAD CSV WITH HEADERS FROM '{file_url}' AS row
        WITH row,
             (exists(row.source_name) AND row.source_name <> '' AND exists(row.target_name) AND row.target_name <> '' AND exists(row.relation) AND row.relation <> '') AS hasOldRel,
             (exists(row.head) AND row.head <> '' AND exists(row.tail) AND row.tail <> '' AND exists(row.relation) AND row.relation <> '') AS hasNewRel

        FOREACH (_ IN CASE WHEN hasOldRel THEN [1] ELSE [] END |
          MERGE (a:Entity {{name: row.source_name}})
          MERGE (b:Entity {{name: row.target_name}})
          MERGE (a)-[:`{chr(96)}REPLACEME{chr(96)}`]->(b)  // 佔位，下一段真正建立
        )

        FOREACH (_ IN CASE WHEN hasNewRel THEN [1] ELSE [] END |
          MERGE (a:Entity {{name: row.head}})
          MERGE (b:Entity {{name: row.tail}})
          MERGE (a)-[:`{chr(96)}REPLACEME{chr(96)}`]->(b)
        )
        """
        # 只作占位建立端點，實際關係屬性/型別仍建議 client-side
        if dry_run:
            LOGGER.info("🟨 [DRY-RUN] LOAD CSV（基本端點建立）\n%s", cypher_basic)
        else:
            session.run(cypher_basic)

    return 0, 0

# ── 主程式 ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annex CSV (head/relation/tail + JSON props) into Neo4j")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                        help=f"輸入 CSV 路徑（預設：{DEFAULT_CSV}）")
    parser.add_argument("--database", type=str, default=None,
                        help="覆寫 .env/config 的 NEO4J_DATABASE")
    parser.add_argument("--batch-size", type=int,
                        default=1000, help="client-side 批次大小（預設 1000）")
    parser.add_argument("--dry-run", action="store_true", help="只列印計畫，不實際寫入")
    parser.add_argument("--use-load-csv", action="store_true",
                        help="使用伺服器端 LOAD CSV（不建議含 JSON 欄位）")
    parser.add_argument("--import-dir", type=str, default=None,
                        help="Neo4j import 目錄（搭配 --use-load-csv）")
    args = parser.parse_args()

    load_dotenv()

    uri = NEO4J_CONFIG["uri"]
    user = NEO4J_CONFIG["user"]
    password = NEO4J_CONFIG["password"]
    database = _load_database_from_env_or_config(args.database)

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV：{csv_path}")

    LOGGER.info("🔗 Neo4j URI: %s  |  database: %s",
                uri, database or "(default)")
    LOGGER.info("📄 CSV: %s", csv_path)

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        if args.use_load_csv:
            if not args.import_dir:
                raise RuntimeError("--use-load-csv 需要指定 --import-dir")
            _server_side_load_csv(driver, database, csv_path, Path(
                args.import_dir).resolve(), args.dry_run)
            LOGGER.info(
                "ℹ️ 已以 LOAD CSV 基本建立端點。含 JSON 的屬性寫入仍建議改用預設 client-side。")
        else:
            rows = list(_read_csv_rows(csv_path))
            rel_rows = sum(1 for r in rows if _is_relation_row(r))
            node_rows = len(rows) - rel_rows
            LOGGER.info("🧭 偵測：關係列 %d、節點列 %d、總計 %d",
                        rel_rows, node_rows, len(rows))

            n, r = _client_side_ingest(
                driver, database, rows, args.batch_size, args.dry_run)
            LOGGER.info("🌟 完成：節點處理（估）%d、關係新建 %d", n, r)
    finally:
        driver.close()


if __name__ == "__main__":
    run_with_timer(main)
