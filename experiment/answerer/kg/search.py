"""
KG 檢索模組（原文 + 三元組）
"""

from __future__ import annotations

import json
from typing import (
    List, Dict, Any, Tuple,
    Callable, Optional, Iterable
)

import numpy as np
import pandas as pd


# =========================
# 工具：安全取值 / 解析 JSON 欄位
# =========================

def _safe_get(series_or_dict, key: str):
    """對 pandas.Series 或 dict 取值並處理 NaN/None/空字串"""
    if series_or_dict is None:
        return None
    val = None
    if isinstance(series_or_dict, dict):
        val = series_or_dict.get(key)
    else:
        # pandas.Series 支援 .get
        val = series_or_dict.get(key)
    if val is None:
        return None
    # pandas 的 NaN 需要另外判斷
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    s = str(val).strip()
    return s if s else None


def _parse_json_field(row: pd.Series, col: Optional[str]) -> Dict[str, Any]:
    """將 JSON 欄位文字 -> dict；空值回 {}"""
    if not col:
        return {}
    raw = _safe_get(row, col)
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


# =========================
# 共同內核：相似度計算
# =========================

def _cosine_topk(
    query_vec: np.ndarray,
    kg_vecs_norm: np.ndarray,
    top_k: int,
    sim_th: float,
    min_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    對單一 query_vec 計算與 KG 向量的相似度，回傳 [(idx, score)] 由高到低。
    若門檻過嚴導致為空，保底取 top min_k。
    """
    sims = kg_vecs_norm @ query_vec  # kg 已正規化、query_vec 應為單位向量
    order = np.argsort(sims)[-top_k:][::-1]
    pairs = [(int(i), float(sims[i]))
             for i in order if float(sims[i]) >= sim_th]
    if not pairs:
        # 門檻過嚴，保底取前 min_k
        pairs = [(int(i), float(sims[i])) for i in order[:min_k]]
    return pairs


# =========================
# 原文檢索
# =========================

def search_by_texts(
    texts: Iterable[str],
    text_encoder: Callable[[str], np.ndarray],
    kg_vecs_norm: np.ndarray,
    kg_df: pd.DataFrame,
    build_block_fn: Callable[..., str],
    *,
    top_k: int = 100,
    sim_th: float = 0.6,
    min_k: int = 5,
    hp_col: Optional[str] = None,
    rp_col: Optional[str] = None,
    tp_col: Optional[str] = None,
) -> List[str]:
    """
    直接以「使用者原文」集合進行語意檢索，輸出為 build_block_fn 的逐行文字。
    """
    results: List[str] = []
    for text in texts:
        qv = text_encoder(text) 
        norm = float(np.linalg.norm(qv))
        if norm > 0:
            qv = qv / norm

        for idx, score in _cosine_topk(qv, kg_vecs_norm, top_k, sim_th, min_k):
            row = kg_df.iloc[idx]
            tri = {'head': _safe_get(row, 'head') or '',
                   'relation': _safe_get(row, 'relation') or '',
                   'tail': _safe_get(row, 'tail') or ''}

            det = {
                'head': _parse_json_field(row, hp_col),
                'rel':  _parse_json_field(row, rp_col),
                'tail': _parse_json_field(row, tp_col),
            }
            block = build_block_fn([tri], {tuple(tri.values()): det})
            results.extend(block.splitlines())
    return results


# =========================
# 三元組檢索
# =========================

def search_by_triples(
    triples: List[Dict[str, str]],
    embed_fn: Callable[[Dict[str, str]], np.ndarray],
    kg_vecs_norm: np.ndarray,
    kg_df: pd.DataFrame,
    build_block_fn: Callable[..., str],
    *,
    top_k: int = 100,
    sim_th: float = 0.6,
    min_k: int = 5,
    hp_col: Optional[str] = None,
    rp_col: Optional[str] = None,
    tp_col: Optional[str] = None,
) -> List[str]:
    """
    依據三元組列表進行向量檢索，回傳組合後的敘述區塊清單（逐行文字）。
    """
    results: List[str] = []

    for tp in triples:
        vec = embed_fn(tp)
        # embed_fn 建議已單位化；這裡再保險一次
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec = vec / n

        for idx, score in _cosine_topk(vec, kg_vecs_norm, top_k, sim_th, min_k):
            row = kg_df.iloc[idx]
            tri = {
                'head': _safe_get(row, 'head') or '',
                'relation': _safe_get(row, 'relation') or '',
                'tail': _safe_get(row, 'tail') or '',
            }
            det = {
                'head': _parse_json_field(row, hp_col),
                'rel':  _parse_json_field(row, rp_col),
                'tail': _parse_json_field(row, tp_col),
            }
            block = build_block_fn([tri], {tuple(tri.values()): det})
            results.extend(block.splitlines())
    return results


# ==========================================
# 新版問句抽取 → 舊三元組介面 Adapter
# ==========================================

DEFAULT_TAIL_PLACEHOLDER: str = "未知"


def adapt_query_triples_v2_to_v1(
    extracted: Dict[str, Any],
    tail_placeholder: str = DEFAULT_TAIL_PLACEHOLDER
) -> Tuple[List[Dict[str, str]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    """
    將新版問句輸出（含 subject/info_need 結構）轉為舊版 {'head','relation','tail'} 清單，
    並回傳 meta 供 embed_fn 使用。
    """
    triples_v1: List[Dict[str, str]] = []
    meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for item in extracted.get("triples", []):
        subj = item.get("subject", {}) or {}
        rel = item.get("relation", "") or "未知"
        need = (item.get("info_need", {}) or {}).get("target", {}) or {}

        head_text = (subj.get("text") or "未知").strip()
        relation = rel.strip() or "未知"
        tail = tail_placeholder

        triples_v1.append({
            "head": head_text,
            "relation": relation,
            "tail": tail,
        })
        meta[(head_text, relation, tail)] = {
            "subject": {
                "text": head_text,
                "type": subj.get("type") or "未知",
                "attributes": subj.get("attributes") or {},
            },
            "target": {
                "type": need.get("type") or "未知",
                "attributes": need.get("attributes") or {},
            }
        }

    return triples_v1, meta


# ==========================================
# 把 target/attributes 注入語意的 embed_fn
# ==========================================

def make_embed_fn_with_attrs(
    text_encoder: Callable[[str], np.ndarray],
    meta: Dict[Tuple[str, str, str], Dict[str, Any]],
    normalize: bool = True,
) -> Callable[[Dict[str, str]], np.ndarray]:

    def _attrs_to_text(attrs: Dict[str, Any]) -> str:
        if not attrs:
            return ""
        keys_priority = ["unit", "time", "scope", "location",
                         "version", "article", "party", "office"]
        parts: List[str] = []
        for k in keys_priority:
            v = attrs.get(k)
            if v not in (None, "", "未知"):
                parts.append(f"{k}={v}")
        for k, v in attrs.items():
            if k not in keys_priority and v not in (None, "", "未知"):
                parts.append(f"{k}={v}")
        return " ".join(parts)

    def embed_fn(tp: Dict[str, str]) -> np.ndarray:
        key = (tp["head"], tp["relation"], tp["tail"])
        m = meta.get(key, {"subject": {}, "target": {}})

        subj = m.get("subject", {})
        targ = m.get("target", {})

        subj_tag = subj.get("type", "未知")
        subj_attr = _attrs_to_text(subj.get("attributes", {}))

        target_type = targ.get("type", "未知")
        target_attr = _attrs_to_text(targ.get("attributes", {}))

        # 不把 tail=未知 當語意來源
        query_text = (
            f"HEAD:{tp['head']} ({subj_tag}) "
            f"REL:{tp['relation']} "
            f"NEEDS:{target_type} "
            f"{('[SUBJ] ' + subj_attr + ' ') if subj_attr else ''}"
            f"{('[NEED] ' + target_attr) if target_attr else ''}"
        ).strip()

        vec = text_encoder(query_text)
        if normalize:
            n = float(np.linalg.norm(vec))
            if n > 0:
                vec = vec / n
        return vec

    return embed_fn


# ===========================================================
# 屬性硬過濾封裝
# ===========================================================

def search_by_triples_with_filter(
    triples: List[Dict[str, str]],
    embed_fn: Callable[[Dict[str, str]], np.ndarray],
    kg_vecs_norm: np.ndarray,
    kg_df: pd.DataFrame,
    build_block_fn: Callable[..., str],
    *,
    top_k: int = 100,
    sim_th: float = 0.6,
    min_k: int = 5,
    hp_col: Optional[str] = None,
    rp_col: Optional[str] = None,
    tp_col: Optional[str] = None,
    row_filter: Optional[Callable[[
        Dict[str, Any], Dict[str, Any]], bool]] = None,
    meta: Optional[Dict[Tuple[str, str, str], Dict[str, Any]]] = None,
) -> List[str]:
    """
    在既有檢索流程外包一層可選的屬性硬過濾；不影響原函式與輸出模式。
    """
    results: List[str] = []
    for tp in triples:
        vec = embed_fn(tp)
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec = vec / n

        for idx, score in _cosine_topk(vec, kg_vecs_norm, top_k, sim_th, min_k):
            row = kg_df.iloc[idx]

            tri = {
                'head': _safe_get(row, 'head') or '',
                'relation': _safe_get(row, 'relation') or '',
                'tail': _safe_get(row, 'tail') or '',
            }
            det = {
                'head': _parse_json_field(row, hp_col),
                'rel':  _parse_json_field(row, rp_col),
                'tail': _parse_json_field(row, tp_col),
            }

            if row_filter and meta is not None:
                q_need = meta.get(
                    (tp['head'], tp['relation'], tp['tail']), {}).get('target', {})
                row_attrs = {'head': det.get('head', {}),
                             'rel': det.get('rel', {}),
                             'tail': det.get('tail', {})}
                if not row_filter(row_attrs, q_need):
                    continue

            block = build_block_fn([tri], {tuple(tri.values()): det})
            results.extend(block.splitlines())
    return results


# =========================
# 範例 row_filter
# =========================

def require_time_if_given(row_attrs: Dict[str, Any], q_need: Dict[str, Any]) -> bool:
    """若問句指定 time，則要求 KG 行也必須包含該 time（最簡字串包含）"""
    t_attr = (q_need or {}).get("attributes", {})
    q_time = (t_attr.get("time") or "").strip()
    if not q_time or q_time == "未知":
        return True
    row_time = (
        (row_attrs.get('tail', {}) or {}).get('time')
        or (row_attrs.get('rel', {}) or {}).get('time')
        or (row_attrs.get('head', {}) or {}).get('time')
        or ""
    )
    return q_time in str(row_time)
