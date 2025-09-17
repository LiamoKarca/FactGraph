"""
Neo4j Loader Module

本模組定義 Neo4jLoader 類別，用以將節點與關係資料寫入 Neo4j。
寫入前會檢查是否已有相同 evidence 的關係，以避免重複建立。
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver

from ...common.gadget import LOGGER
from ...config import NEO4J_CONFIG


class Neo4jLoader:
    """
    負責將節點與關係寫入 Neo4j 資料庫。
    """

    def __init__(self) -> None:
        """初始化 Neo4j 連線。"""
        uri = NEO4J_CONFIG["uri"]
        user = NEO4J_CONFIG["user"]
        password = NEO4J_CONFIG["password"]
        self.database: Optional[str] = NEO4J_CONFIG.get("database")
        try:
            self.driver: Driver = GraphDatabase.driver(
                uri, auth=(user, password))
        except Exception as exc:
            LOGGER.critical("無法建立 Neo4j 連線: %s", exc)
            sys.exit(1)

    def close(self) -> None:
        """關閉 Neo4j 連線。"""
        if hasattr(self, 'driver'):
            self.driver.close()

    def insert_data(
            self,
            nodes: List[Dict[str, Any]],
            relationships: List[Dict[str, Any]]
    ) -> None:
        """
        插入節點和關係到 Neo4j 資料庫。

        Args:
            nodes: 節點列表，每個節點為字典格式，至少包含 id、name、type。
            relationships: 關係列表，每個關係為字典格式，至少包含 source_name、target_name、relation、evidence。
        """
        with self.driver.session(database=self.database) as session:
            # 插入節點
            for node in nodes:
                try:
                    props = {k: v for k, v in node.items() if k not in [
                        'id', 'name', 'type']}
                    session.run(
                        """
                        MERGE (n:Entity {name: $name})
                        ON CREATE SET n.id = $id, n.type = $type, n += $props
                        ON MATCH SET n += $props
                        """,
                        id=node['id'],
                        name=node['name'],
                        type=node['type'],
                        props=props
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "插入節點失敗: %s, 錯誤: %s", node, exc
                    )

            # 插入關係
            for rel in relationships:
                source = rel.get('source_name')
                target = rel.get('target_name')
                evidence = rel.get('evidence')
                if not source or not target:
                    LOGGER.info(
                        "跳過關係，來源或目標缺失: %s", rel
                    )
                    continue
                if not evidence:
                    LOGGER.info(
                        "跳過關係，缺少 evidence: %s", rel
                    )
                    continue

                doc_id = rel.get('doc_id', '')
                rel_type = rel.get('relation', 'RELATED_TO')
                date = rel.get('date')

                try:
                    # 檢查是否已有相同 evidence 的關係
                    result = session.run(
                        """
                        MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                        OPTIONAL MATCH (a)-[r]->(b)
                        WHERE r.evidence = $evidence
                        RETURN count(r) AS rel_count
                        """,
                        source=source,
                        target=target,
                        evidence=evidence
                    )
                    record = result.single()
                    count = record['rel_count'] if record else 0

                    # 如果已存在，則略過
                    if count and count > 0:
                        LOGGER.info(
                            "跳過已存在相同 evidence 的關係: %.30s...", evidence
                        )
                        continue

                    # 創建新關係
                    session.run(
                        """
                        MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                        CALL apoc.create.relationship(
                            a,
                            $rel_type,
                            {doc_id: $doc_id, evidence: $evidence, date: $date},
                            b
                        ) YIELD rel
                        RETURN rel
                        """,
                        source=source,
                        target=target,
                        rel_type=rel_type,
                        doc_id=doc_id,
                        evidence=evidence,
                        date=date
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "插入關係失敗: %s, 錯誤: %s", rel, exc
                    )
