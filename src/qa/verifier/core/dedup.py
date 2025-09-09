"""
語意去重
"""
from typing import List

import numpy as np
from sentence_transformers import util

from .config import DUP_TH, ENTITY_RE
from .embeddings import embed_text


def deduplicate(lines: List[str]) -> List[str]:
    groups = {}
    kept = []
    for line in lines:
        ent = _first_entity(line)
        vec = embed_text(line)
        vecs = groups.get(ent, [])
        if vecs and util.cos_sim(vec, np.vstack(vecs))[0, :].max() >= DUP_TH:
            continue
        groups.setdefault(ent, []).append(vec)
        kept.append(line)
    return kept


def _first_entity(line: str) -> str:
    m = ENTITY_RE.match(line)
    if m:
        return m.group(1)
    return line.split(' ')[0]
