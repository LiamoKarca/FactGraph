"""本地向量檢索實作。"""

from __future__ import annotations

import threading
from typing import Dict, List

from ...core.embeddings import embed_triple
from ...kg.search import cosine_search, kg_row_to_detail
from ..common.config import MAX_EVID_CHARS, _dlog
from ..common.formatting import collect_hits_to_lines, norm_triple_dict

_VEC_TOOL_LOCK = threading.RLock()


def vector_search_impl(tp: Dict[str, str], top_k: int = 100) -> List[str]:
    """執行三元組向量檢索並渲染為 `[比對]` 行。"""
    with _VEC_TOOL_LOCK:
        try:
            tp = norm_triple_dict(tp)
            q_vec = embed_triple(tp)
            idxs = cosine_search(tp, q_vec)
            lines: List[str] = []
            for i in idxs[:top_k]:
                tri, det = kg_row_to_detail(i)
                lines.extend(collect_hits_to_lines([(tri, det)]))
            return lines
        except Exception as exc:
            _dlog("vector_search_impl: failed\n" + str(exc))
            return []
