"""ER → Triples 與入參擷取工具。"""

from __future__ import annotations

from typing import Any, Dict, List

from ..common.formatting import norm_triple_dict


def er_to_triples(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """由抽取器輸出物件轉換為標準三元組清單。"""
    if not isinstance(obj, dict):
        return []
    if "entities" in obj or "relations" in obj:
        ents = {e.get("id"): e for e in obj.get("entities", []) if isinstance(e, dict)}
        triples: List[Dict[str, str]] = []
        for r in obj.get("relations", []) or []:
            s = ents.get(r.get("source") or "", {}) or {}
            t = ents.get(r.get("target") or "", {}) or {}
            head = (s.get("name") or "").strip() or "未知"
            rel = (r.get("relation") or "").strip() or "未知"
            tail = (t.get("name") or "").strip()
            if not tail:
                attrs = t.get("attributes") or {}
                val = attrs.get("value")
                unit = attrs.get("unit") or ""
                tail = f"{val}{unit}".strip() if val not in (None, "") else "未知"
            triples.append({"head": head, "relation": rel, "tail": tail})
        return triples
    if isinstance(obj.get("triples"), list):
        return [norm_triple_dict(x) for x in obj["triples"] if isinstance(x, dict)]
    return []


def take_first_triple(obj: Any) -> dict:
    """從各式入參格式中擷取第一個三元組。"""
    if isinstance(obj, dict):
        if any(k in obj for k in ("head", "relation", "tail")):
            return obj
        if "triples" in obj and isinstance(obj["triples"], list) and obj["triples"]:
            cand = obj["triples"][0]
            return cand if isinstance(cand, dict) else {}
        return {}
    if isinstance(obj, list) and obj:
        return obj[0] if isinstance(obj[0], dict) else {}
    return {}
