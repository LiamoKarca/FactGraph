"""
LlamaIndex x Neo4j 圖檢索器（PG / Cypher-first，支援 multi-hop）

功能
----
- 連線 Neo4j，依 (head, relation, tail) 查詢，支援 1..hops 跳的 variable-length path。
- 以 rel_props.date 由新至舊排序，並以 rel_props.evidence 做句級過濾。
- 介面與離線 CsvPropertyGraphRetriever 一致：search_triple(tp, top_k, hops)。

環境變數
--------
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
"""

from __future__ import annotations

import os
import re
import typing
from typing import Dict, Iterable, List, Tuple

from dotenv import load_dotenv

# 型別檢查友善匯入（未安裝套件時不讓編輯器滿螢幕紅線）
if typing.TYPE_CHECKING:  # pragma: no cover
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore as _Neo4jPGS  # type: ignore[import]
else:
    _Neo4jPGS = object  # noqa: N816

try:
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore  # type: ignore[import]
except Exception:
    Neo4jPropertyGraphStore = None  # type: ignore[misc]

load_dotenv()


class LIRetrievalError(RuntimeError):
    """自訂例外，便於上層回退處理。"""


class LlamaIndexNeo4jRetriever:
    """LlamaIndex x Neo4j 檢索器（Cypher-first）。"""

    def __init__(self, date_field: str = "date", evidence_field: str = "evidence", min_evidence_chars: int = 6) -> None:
        self.date_field = date_field
        self.evidence_field = evidence_field
        self.min_evidence_chars = min_evidence_chars

        if Neo4jPropertyGraphStore is None:
            raise LIRetrievalError("找不到 LlamaIndex Neo4j 模組，請先安裝：pip install -U llama-index-graph-stores-neo4j")

        url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "neo4j")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        self.store = Neo4jPropertyGraphStore(url=url, username=username, password=password, database=database)
        self.driver = self.store._driver  # type: ignore[attr-defined]

    # ── 公開介面 ───────────────────────────────────────────────

    def search_triple(self, tp: Dict[str, str], top_k: int = 50, hops: int = 2) -> List[Tuple[dict, Dict[str, dict]]]:
        h = (tp.get("head") or "").strip()
        r = (tp.get("relation") or "").strip()
        t = (tp.get("tail") or "").strip()

        if hops <= 1:
            cypher, params = self._build_cypher(h, r, t, top_k=top_k)
        else:
            cypher, params = self._build_cypher_multihop(h, r, t, hops=hops, top_k=top_k)

        rows = self._run_cypher(cypher, params)
        out: List[Tuple[dict, Dict[str, dict]]] = []
        for row in rows:
            tri, det = self._row_to_tri_det(row)
            if self._evidence_ok(det.get("rel", {}).get(self.evidence_field, "")):
                out.append((tri, det))
        return out

    # ── 私有工具 ───────────────────────────────────────────────

    @staticmethod
    def _normalize_token(x: str) -> str:
        s = re.sub(r"\s+", " ", (x or "")).strip()
        return s.casefold()

    def _build_cypher(self, h: str, r: str, t: str, top_k: int) -> tuple[str, dict]:
        where_clauses: List[str] = []
        params: dict = {}

        if h:
            where_clauses.append("(toLower(h.name) = $h_eq OR toLower(h.name) CONTAINS $h_like)")
            params["h_eq"] = self._normalize_token(h)
            params["h_like"] = self._normalize_token(h)
        if r:
            where_clauses.append("(toLower(type(rel)) = $r_eq OR toLower(type(rel)) CONTAINS $r_like)")
            params["r_eq"] = self._normalize_token(r)
            params["r_like"] = self._normalize_token(r)
        if t:
            where_clauses.append("(toLower(t.name) = $t_eq OR toLower(t.name) CONTAINS $t_like)")
            params["t_eq"] = self._normalize_token(t)
            params["t_like"] = self._normalize_token(t)

        where_sql = " AND ".join(where_clauses) if where_clauses else "true"

        cypher = f"""
        MATCH (h:Entity)-[rel]->(t:Entity)
        WHERE {where_sql}
        WITH h, rel, t,
             toString(rel.{self.date_field}) AS _date,
             coalesce(rel.{self.evidence_field}, "") AS _evi,
             coalesce(rel.source, "") AS _src
        RETURN h AS head, type(rel) AS relation, t AS tail,
               properties(h) AS head_props,
               properties(rel) AS rel_props,
               properties(t) AS tail_props,
               _date AS rel_date, _evi AS rel_evidence, _src AS rel_source
        ORDER BY _date DESC
        LIMIT $top_k
        """
        params["top_k"] = int(top_k)
        return cypher, params

    def _build_cypher_multihop(self, h: str, r: str, t: str, hops: int, top_k: int) -> tuple[str, dict]:
        where_head, where_tail, where_rel = [], [], []
        params: dict = {}

        if h:
            where_head.append("(toLower(h.name) = $h_eq OR toLower(h.name) CONTAINS $h_like)")
            params["h_eq"] = self._normalize_token(h)
            params["h_like"] = self._normalize_token(h)
        if t:
            where_tail.append("(toLower(t.name) = $t_eq OR toLower(t.name) CONTAINS $t_like)")
            params["t_eq"] = self._normalize_token(t)
            params["t_like"] = self._normalize_token(t)
        if r:
            where_rel.append("(any(rel IN relationships(p) WHERE toLower(type(rel)) = $r_eq OR toLower(type(rel)) CONTAINS $r_like))")
            params["r_eq"] = self._normalize_token(r)
            params["r_like"] = self._normalize_token(r)

        parts = []
        if where_head:
            parts.append(" AND ".join(where_head))
        if where_tail:
            parts.append(" AND ".join(where_tail))
        if where_rel:
            parts.append(" AND ".join(where_rel))
        where_sql = " AND ".join(parts) if parts else "true"

        cypher = f"""
        MATCH p=(h:Entity)-[rels*1..{int(hops)}]->(t:Entity)
        WHERE {where_sql}
        UNWIND rels AS rel
        WITH h, rel, endNode(rel) AS v,
             toString(rel.{self.date_field}) AS _date,
             coalesce(rel.{self.evidence_field}, "") AS _evi,
             coalesce(rel.source, "") AS _src
        RETURN h AS head, type(rel) AS relation, v AS tail,
               properties(h) AS head_props,
               properties(rel) AS rel_props,
               properties(v) AS tail_props,
               _date AS rel_date, _evi AS rel_evidence, _src AS rel_source
        ORDER BY _date DESC
        LIMIT $top_k
        """
        params["top_k"] = int(top_k)
        return cypher, params

    def _run_cypher(self, cypher: str, params: dict) -> Iterable[dict]:
        if not self.driver:
            raise LIRetrievalError("Neo4j driver 尚未初始化。")
        with self.driver.session(database=self.store._database) as sess:  # type: ignore[attr-defined]
            result = sess.run(cypher, **params)
            for rec in result:
                yield rec.data()

    def _row_to_tri_det(self, row: dict):
        head = row.get("head", {}) or {}
        tail = row.get("tail", {}) or {}
        tri = {"head": head.get("name") or "", "relation": row.get("relation") or "", "tail": tail.get("name") or ""}
        rel_props = dict(row.get("rel_props") or {})
        # 補齊常用欄位（若 rel_props 內未含）
        rel_props.setdefault("date", row.get("rel_date") or "")
        rel_props.setdefault("evidence", row.get("rel_evidence") or "")
        rel_props.setdefault("source", row.get("rel_source") or "")
        det = {"head": dict(head), "rel": rel_props, "tail": dict(tail)}
        return tri, det

    def _evidence_ok(self, evidence: str) -> bool:
        s = re.sub(r"\s+", " ", str(evidence or "")).strip(" ，、；,;:…")
        return len(s) >= self.min_evidence_chars
