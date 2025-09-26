# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/karca5103/dev/FactGraph/src/qa/tools/data_utils.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2025-06-27 07:16:32 UTC (1751008592)

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple, Any

Triple = Dict[str, str]


def key(tp: Triple) -> Tuple[str, str, str]:
    return (tp['head'], tp['relation'], tp['tail'])


def json_to_triples(maybe_json: Any) -> List[Triple] | None:
    """LLM JSON→triples（一層扁平）。"""
    if not isinstance(maybe_json, dict):
        return
    ent_idx = {e['id']: e.get('name') for e in maybe_json.get('entities', [])}
    triples = []
    for rel in maybe_json.get('relations', []):
        h, t, r = (ent_idx.get(rel.get('source')), ent_idx.get(rel.get('target')), rel.get('relation'))
        if h and t and r:
            triples.append({'head': h, 'relation': r, 'tail': t})
    return triples


def merge_triples(*triple_lists: List[Triple]) -> List[Triple]:
    """多輪結果 → 去重後清單。"""
    seen = set()
    merged = []
    for raw in itertools.chain(*triple_lists):
        if not raw or isinstance(raw, str):
            continue
        if isinstance(raw, dict):
            rec = {'head': raw.get('head') or raw.get('source') or raw.get('source_name'),
                   'relation': raw.get('relation'),
                   'tail': raw.get('tail') or raw.get('target') or raw.get('target_name')}
        else:
            if len(raw) < 3:
                continue
            rec = {'head': raw[0], 'relation': raw[1], 'tail': raw[2]}
        if None in rec.values():
            continue
        k = key(rec)
        if k not in seen:
            seen.add(k)
            merged.append(rec)
    return merged


def parse_label(doc: Dict, collection: str) -> int:
    """將多型標籤欄位 → int(0/1)。"""
    if isinstance(doc.get('label'), (int, bool)):
        return int(doc['label'])
    iso = doc.get('is_fake')
    if iso is not None:
        return 1 if str(iso).strip().lower() in {'1', 'yes', 'true'} else 0
    return 1 if collection == 'Fake_News' else 0
