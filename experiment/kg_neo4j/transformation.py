"""
Transformation Module

本模組將 NLP 模型抽取出的實體與關係結果轉換為 Neo4j 所需的節點與關係格式，
並解決 `source` / `target` 為 list 時造成的 TypeError，
同時展開一對多、多對多關係為多條 edge。
"""
from typing import Any, Dict, List, Tuple


def _ensure_list(value: Any) -> List[Any]:
    """確保 value 為 list；None 轉空 list，非 list 則包成 list。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def transform_to_neo4j_format(extraction_result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """將抽取結果轉為 Neo4j 可匯入的 nodes 與 relationships。"""
    if not extraction_result:
        return ([], [])
    nodes = []
    relationships = []
    id_to_name = {}
    for entity in extraction_result.get('entities', []):
        node = {'id': entity.get('id'), 'name': entity.get(
            'name'), 'type': entity.get('type', 'Entity')}
        for k, v in entity.get('attributes', {}).items():
            node[k] = v
        nodes.append(node)
        id_to_name[entity.get('id')] = entity.get('name')
    for rel in extraction_result.get('relations', []):
        src_ids = _ensure_list(rel.get('source'))
        tgt_ids = _ensure_list(rel.get('target'))
        if not src_ids or not tgt_ids:
            continue
        for s in src_ids:
            for t in tgt_ids:
                relationships.append({'source': s, 'target': t, 'source_name': id_to_name.get(s, ''),
                                      'target_name': id_to_name.get(t, ''),
                                      'relation': rel.get('relation', rel.get('type', 'RELATED_TO')),
                                      'evidence': rel.get('evidence', ''), **rel.get('attributes', {})})
    return (nodes, relationships)
