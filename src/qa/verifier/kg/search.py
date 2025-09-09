"""
cosine_search 與 row → detail
"""
from __future__ import annotations

import json
from typing import List, Dict, Tuple, Any

import numpy as np

from .loader import KG_DF, KG_VECS_NORM, HP_COL, RP_COL, TP_COL
from ..core.config import SIM_TH, TOP_K


def cosine_search(tp: dict, q_vec: np.ndarray) -> List[int]:
    """
    回傳相似度過門檻之行索引，並以 head/（可用時的）tail 做輕過濾。
    - 若 tail 為空或 '未知'，僅以 head 做過濾（維持舊行為但更穩健）
    """
    sims = KG_VECS_NORM @ q_vec
    idx = sims.argsort()[-TOP_K:][::-1]
    idx = idx[sims[idx] >= SIM_TH]

    q_head = (tp.get('head') or '').strip()
    q_tail = (tp.get('tail') or '').strip()

    filtered: List[int] = []
    for i in idx:
        h = KG_DF.at[i, 'head']
        t = KG_DF.at[i, 'tail']
        if h == q_head:
            filtered.append(i)
            continue
        if q_tail and q_tail != '未知' and t == q_tail:
            filtered.append(i)
    return filtered


def _safe_json_load(x: Any) -> Dict[str, Any]:
    if not x or (isinstance(x, float) and np.isnan(x)):  # NaN/None/空字串
        return {}
    try:
        return json.loads(x)
    except Exception:
        return {}


def kg_row_to_detail(idx: int) -> Tuple[dict, Dict[str, dict]]:
    row = KG_DF.iloc[idx]
    tri = {'head': row['head'],
           'relation': row['relation'], 'tail': row['tail']}
    det = {
        'head': _safe_json_load(row[HP_COL]) if HP_COL else {},
        'rel':  _safe_json_load(row[RP_COL]) if RP_COL else {},
        'tail': _safe_json_load(row[TP_COL]) if TP_COL else {},
    }
    return (tri, det)
