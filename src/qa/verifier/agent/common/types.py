"""型別定義與 ID 工具。"""

from __future__ import annotations

import hashlib
from typing import Dict, List, TypedDict


class RankItem(TypedDict):
    """排序/融合後的標準項目。"""

    id: str
    text: str
    source: str
    rank: int
    score: float
    payload: Dict


def make_rank_id(text: str, src: str) -> str:
    """以來源+內容派生短 ID。"""
    h = hashlib.sha1((src + "||" + (text or "")).encode("utf-8-sig")).hexdigest()[:12]
    return f"{src}:{h}"
