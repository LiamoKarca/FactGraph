"""
KG 檢索模組
"""

from __future__ import annotations

import json
from typing import (
    List, Dict, Any, Tuple,
    Callable, Optional
)

import numpy as np
import pandas as pd

# =========================
# 主函式
# =========================


def search_by_triples(
        triples: List[Dict[str, str]],
        embed_fn: Callable[[Dict[str, str]], np.ndarray],
        kg_vecs_norm: np.ndarray,
        kg_df: pd.DataFrame,
        build_block_fn: Callable[..., str],
        top_k: int = 100,
        sim_th: float = 0.8,
        hp_col: Optional[str] = None,
        rp_col: Optional[str] = None,
        tp_col: Optional[str] = None
) -> List[str]:
    """
    依據輸入的三元組列表進行向量相似度檢索，
    回傳組合後的敘述區塊清單（逐行文字）。

    Args:
        triples: GPT 抽取出的三元組 dict 列表，格式包含 'head','relation','tail'.
        embed_fn: 對單一三元組執行嵌入並回傳向量的函式。
        kg_vecs_norm: 已正規化的 KG 向量矩陣，每列對應 kg_df 同一索引。
        kg_df: 包含至少 'head','relation','tail' 欄位，以及可選屬性 json 欄位。
        build_block_fn: 將三元組與屬性字典轉換為人類可讀文字區塊的函式。
        top_k: 每個三元組檢索的前 top_k 條結果。
        sim_th: 餘弦相似度門檻，低於此值將被捨棄。
        hp_col: head 屬性 json 欄位名稱，若無則設 None。
        rp_col: relation 屬性 json 欄位名稱，若無則設 None.
        tp_col: tail 屬性 json 欄位名稱，若無則設 None.

    Returns:
        符合條件的敘述區塊列表，每個元素為一行文字，保留原始編號。
    """
    results: List[str] = []

    for tp in triples:
        vec = embed_fn(tp)
        sims = kg_vecs_norm @ vec
        # 取出符合 sim_th 且排名前 top_k 的索引
        top_indices = np.argsort(sims)[-top_k:][::-1]

        for idx in top_indices:
            score = sims[idx]
            if score < sim_th:
                break
            row = kg_df.iloc[idx]
            # 基本三元組
            tri = {
                'head': row['head'],
                'relation': row['relation'],
                'tail': row['tail']
            }
            # 屬性詳情
            det: Dict[str, Dict[str, Any]] = {
                'head': json.loads(row[hp_col]) if hp_col and row.get(hp_col) else {},
                'rel': json.loads(row[rp_col]) if rp_col and row.get(rp_col) else {},
                'tail': json.loads(row[tp_col]) if tp_col and row.get(tp_col) else {}
            }
            # 使用外部函式組合文字區塊
            block = build_block_fn([tri], {tuple(tri.values()): det})
            # 拆分為多行並加入結果
            for line in block.splitlines():
                results.append(line)

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
    將新版問句抽取輸出（含 subject/info_need 結構）
    轉為舊版 `{'head','relation','tail'}` 清單，並回傳一份 meta 供 embed_fn 使用。

    Args:
        extracted: 新版問句抽取 JSON，例如：
            {
              "triples": [
                {
                  "id": "Q1_T1",
                  "subject": {"text":"柯文哲","type":"人物","attributes":{}},
                  "relation": "現況",
                  "info_need": {"target":{"type":"描述","attributes":{}}}
                }
              ]
            }
        tail_placeholder: tail 的佔位符（避免把抽象類別當作 tail 值去配 KG）

    Returns:
        triples_v1: List[{'head','relation','tail'}]
        meta: dict，key 為 (head, relation, tail)，value 為：
              {
                'subject': {'text','type','attributes':{...}},
                'target':  {'type','attributes':{...}}
              }
    """
    triples_v1: List[Dict[str, str]] = []
    meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for item in extracted.get("triples", []):
        subj = item.get("subject", {}) or {}
        rel = item.get("relation", "") or "未知"
        need = (item.get("info_need", {}) or {}).get("target", {}) or {}

        head_text = (subj.get("text") or "未知").strip()
        relation = rel.strip() or "未知"
        # tail 用佔位符，避免把「數值/描述/身份」這種抽象詞拿去比對 KG tail
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
    """
    產生符合原介面的 embed_fn(tp)。
    會將 target.type 與 attributes（unit/time/scope 等）串成檢索提示，
    提高在 KG 中匹配到正確敘述的機率。

    Args:
        text_encoder: 將字串轉為向量的函式（例如 Sentence-BERT encoder.encode）
        meta: 由 adapt_query_triples_v2_to_v1 產生的輔助資訊
        normalize: 是否單位化輸出向量（建議 True，搭配 kg_vecs_norm）

    Returns:
        embed_fn: Callable[[Dict[str,str]], np.ndarray]
    """

    def _attrs_to_text(attrs: Dict[str, Any]) -> str:
        if not attrs:
            return ""
        # 僅保留最小唯一識別的欄位，按優先序
        keys_priority = ["unit", "time", "scope", "location",
                         "version", "article", "party", "office"]
        parts: List[str] = []
        for k in keys_priority:
            v = attrs.get(k)
            if v not in (None, "", "未知"):
                parts.append(f"{k}={v}")
        # 收攏其餘鍵
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

        # 組合檢索文字：不把 tail=未知 當語意來源，改以 NEEDS 段承載需求語意
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
# 屬性硬過濾的封裝呼叫
# ===========================================================

def search_by_triples_with_filter(
    triples: List[Dict[str, str]],
    embed_fn: Callable[[Dict[str, str]], np.ndarray],
    kg_vecs_norm: np.ndarray,
    kg_df: pd.DataFrame,
    build_block_fn: Callable[..., str],
    top_k: int = 100,
    sim_th: float = 0.8,
    hp_col: Optional[str] = None,
    rp_col: Optional[str] = None,
    tp_col: Optional[str] = None,
    row_filter: Optional[Callable[[
        Dict[str, Any], Dict[str, Any]], bool]] = None,
    meta: Optional[Dict[Tuple[str, str, str], Dict[str, Any]]] = None,
) -> List[str]:
    """
    在既有檢索流程外包一層可選的屬性硬過濾；不影響原函式與輸出模式。

    Args:
        row_filter(row_attrs, q_need) -> bool
          - row_attrs: {'head':{}, 'rel':{}, 'tail':{}}  # 由 hp_col/rp_col/tp_col JSON 解析
          - q_need:    meta[(head,relation,tail)]['target']  # {'type','attributes':{}}

    Returns:
        List[str]：與原函式一致的逐行文字輸出
    """
    results: List[str] = []
    for tp in triples:
        vec = embed_fn(tp)
        sims = kg_vecs_norm @ vec
        top_indices = np.argsort(sims)[-top_k:][::-1]

        for idx in top_indices:
            score = sims[idx]
            if score < sim_th:
                break
            row = kg_df.iloc[idx]

            tri = {'head': row['head'],
                   'relation': row['relation'], 'tail': row['tail']}
            det = {
                'head': json.loads(row[hp_col]) if hp_col and row.get(hp_col) else {},
                'rel':  json.loads(row[rp_col]) if rp_col and row.get(rp_col) else {},
                'tail': json.loads(row[tp_col]) if tp_col and row.get(tp_col) else {},
            }

            if row_filter and meta is not None:
                q_need = meta.get(
                    (tp['head'], tp['relation'], tp['tail']), {}).get('target', {})
                row_attrs = {'head': det.get('head', {}), 'rel': det.get(
                    'rel', {}), 'tail': det.get('tail', {})}
                if not row_filter(row_attrs, q_need):
                    continue

            block = build_block_fn([tri], {tuple(tri.values()): det})
            results.extend(block.splitlines())
    return results


# =========================
# 範例 row_filter
# =========================

def require_time_if_given(row_attrs: Dict[str, Any], q_need: Dict[str, Any]) -> bool:
    """
    若問句 target.attributes 指定 time，則要求 KG 行也必須符合（最簡單字串包含）。
    否則不過濾。
    """
    t_attr = (q_need or {}).get("attributes", {})
    q_time = (t_attr.get("time") or "").strip()
    if not q_time or q_time == "未知":
        return True
    row_time = (
        row_attrs.get('tail', {}).get('time')
        or row_attrs.get('rel', {}).get('time')
        or row_attrs.get('head', {}).get('time')
        or ""
    )
    return q_time in str(row_time)
