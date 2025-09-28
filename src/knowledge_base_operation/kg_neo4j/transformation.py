"""
Transformation Module

本模組將 NLP 模型抽取出的實體與關係結果轉換為 Neo4j 所需的節點與關係格式，
並解決 `source` / `target` 為 list 時造成的 TypeError，
同時展開一對多、多對多關係為多條 edge。
"""
from typing import Any, Dict, List, Tuple


def _as_id(x: Any) -> str:
    """
    將 source/target 正規化成可作為 dict key 的字串 ID（僅供查表使用）。
    規則：
      - str: 原樣回傳（完全不改舊行為）
      - dict: 取 x['id']，若無則取 x['name']，仍無則 str(x)
      - 其他: 直接 str(x)
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return (x.get("id") or x.get("name") or str(x))
    return str(x)


def _pick_one(x: Any) -> Any:
    """
    若 x 為 list，回傳第一個元素；否則原樣回傳。
    用於「僅名稱查表」階段，避免改變原始欄位資料型別。
    """
    if isinstance(x, list) and x:
        return x[0]
    return x


def transform_to_neo4j_format(extraction_result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    將抽取結果轉為 Neo4j 可用的 nodes 與 relationships。
    —— 須注意正名稱查表時的 dict 當 key 問題。
    """
    if not extraction_result:
        return [], []

    nodes: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []

    # 1) 照舊建立 id → name 對照與節點列表
    id_to_name: Dict[str, str] = {}
    for entity in extraction_result.get("entities", []):
        eid = entity.get("id", "")
        name = entity.get("name", "")
        etype = entity.get("type", "Entity") or "Entity"

        node: Dict[str, Any] = {
            "id": eid or name or "UNKNOWN_ENTITY",
            "name": name or eid or "UNKNOWN_ENTITY",
            "type": etype,
        }

        # 展開邏輯（若 attributes 存在就攤平）
        attrs = entity.get("attributes", {})
        if isinstance(attrs, dict):
            node.update(attrs)

        nodes.append(node)
        # 以 node["id"] 作為 key
        id_to_name[node["id"]] = node["name"]

    # 2) 建立關係；名稱查表時先正規化 key
    for rel in extraction_result.get("relations", []):
        s = rel.get("source")
        t = rel.get("target")

        # 只在查表時取第一個元素（若為 list）並正規化為可 hash 的 ID
        s_key = _as_id(_pick_one(s))
        t_key = _as_id(_pick_one(t))

        relationship: Dict[str, Any] = {
            "source": s,
            "target": t,
            "source_name": id_to_name.get(s_key, ""),
            "target_name": id_to_name.get(t_key, ""),
            "relation": rel.get("relation") or rel.get("type") or "RELATED_TO",
        }

        # evidence 與 attributes 的合併方式
        if "evidence" in rel:
            relationship["evidence"] = rel.get("evidence")
        attrs = rel.get("attributes", {})
        if isinstance(attrs, dict):
            relationship.update(attrs)

        relationships.append(relationship)

    return nodes, relationships
