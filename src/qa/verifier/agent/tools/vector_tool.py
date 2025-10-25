"""
本地向量檢索工具
"""
from __future__ import annotations

import json
import os
import traceback
from typing import List

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv(override=True)

from ..common.config import _dlog
from ..common.formatting import norm_triple_dict
from ..common.json_utils import parse_json_safely
from ..common.types import RankItem, make_rank_id
from ..extract.er_utils import take_first_triple

# 核心實作以 try import，保持自檢可用
try:
    from ..retrieval.vector_impl import vector_search_impl as _vector_search_impl
    _VECTOR_IMPL_AVAILABLE = True
except Exception:
    _VECTOR_IMPL_AVAILABLE = False
    _vector_search_impl = None  # type: ignore


def _env_vector_enabled() -> bool:
    """是否允許向量備援（由 .env 控制）。"""
    return os.getenv("ENABLE_VECTOR_FALLBACK", "1").lower() in ("1", "true", "yes", "y")


@tool(
    "vector_search",
    return_direct=False,
    description="本地向量檢索。輸入三元組 JSON 與 top_k；輸出 {'lines': [...], 'items': [...]}。",
)
def tool_vector_search(triple_json: str, top_k: int = 100) -> str:
    """工具：本地向量檢索。"""
    # 1) 先看環境是否允許
    if not _env_vector_enabled():
        _dlog("vector_search: disabled by env ENABLE_VECTOR_FALLBACK")
        return json.dumps(
            {
                "lines": [],
                "items": [],
                "error": "vector_disabled",
                "note": "Vector search disabled by ENABLE_VECTOR_FALLBACK=0",
            },
            ensure_ascii=False,
        )

    # 2) 實作可用性
    if not _VECTOR_IMPL_AVAILABLE or _vector_search_impl is None:
        _dlog("vector_search: implementation not available (import failed).")
        return json.dumps(
            {
                "lines": [],
                "items": [],
                "error": "vector_impl_unavailable",
                "note": "Vector search implementation failed to load.",
            },
            ensure_ascii=False,
        )

    # 3) 解析輸入 + 檢索
    try:
        try:
            tp_raw = parse_json_safely(triple_json)
        except Exception as e:
            preview = (triple_json or "").strip().replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + " ...<truncated>"
            _dlog(f"vector_search: invalid triple_json: {e}; preview={preview}")
            return json.dumps(
                {
                    "lines": [],
                    "items": [],
                    "error": "vector_invalid_json",
                    "note": str(e),
                    "preview": preview,
                },
                ensure_ascii=False,
            )

        # 標準化三元組
        tp = norm_triple_dict(take_first_triple(tp_raw))

        # 與（你以前穩定版）一致：不以 .env 介入 top_k；直接用入參
        lines = _vector_search_impl(tp, top_k=top_k)

        # 包成 RankItem（給 merge 工具合併排名/去重）
        items: List[RankItem] = []
        for i, ln in enumerate(lines, start=1):
            txt = ln if ln.startswith("[比對]") else f"[比對] {ln}"
            items.append(
                {
                    "id": make_rank_id(txt, "vector"),
                    "text": txt,
                    "source": "vector",
                    "rank": i,
                    "score": 0.0,
                    "payload": {"line": txt},
                }
            )

        _dlog(f"vector_search: returned_lines={len(lines)}")
        return json.dumps({"lines": lines, "items": items}, ensure_ascii=False)

    except Exception:
        _dlog("vector_search: failed\n" + traceback.format_exc())
        return json.dumps(
            {
                "lines": [],
                "items": [],
                "error": "vector_exception",
                "note": "vector_search failed",
            },
            ensure_ascii=False,
        )
