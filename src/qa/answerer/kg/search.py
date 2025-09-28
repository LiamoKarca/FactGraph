"""
KG 檢索模組（原文 + 三元組）

本版本為「可直接覆蓋」版本，重點改動：
1) 純數學去噪排序器（不依賴關鍵字或時間窗）：
   - 自適應門檻（分位數 + Z-score）
   - kNN 局部密度過濾
   - 互為近鄰（Reciprocal NN）過濾
   - MMR 去冗與多樣化
2) 可用 `ranker="topk"` 回退傳統 Top-K。
3) 預設 `ranker="math"`，大幅降低長尾雜訊。

注意：本模組僅使用向量運算（numpy），無任何外部依賴。
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
# 共同內核：相似度計算 (舊)
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


def _cosine_topk_adaptive(
    query_vec: np.ndarray,
    kg_vecs_norm: np.ndarray,
    top_k: int,
    sim_th: float,
    min_k: int = 5,
    pct: float = 0.9,
) -> List[Tuple[int, float]]:
    """
    Top-K + 自適應分位數底線（較舊的輕量自適應）
    """
    sims = kg_vecs_norm @ query_vec
    if sims.size == 0:
        return []
    base = float(np.quantile(sims, pct))
    th = max(sim_th, base)
    order = np.argsort(sims)[-top_k:][::-1]
    pairs = [(int(i), float(sims[i])) for i in order if float(sims[i]) >= th]
    if not pairs:
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
    ranker: str = "math",  # "math" | "topk" | "adpt"
) -> List[str]:
    """
    直接以「使用者原文」集合進行語意檢索，輸出為 build_block_fn 的逐行文字。
    ranker:
      - "math": 純數學去噪排序器（推薦，預設）
      - "topk": 傳統 Top-K 門檻
      - "adpt": Top-K + 分位數自適應底線
    """
    results: List[str] = []
    for text in texts:
        qv = text_encoder(text)
        norm = float(np.linalg.norm(qv))
        if norm > 0:
            qv = qv / norm

        if ranker == "topk":
            idx_scores = _cosine_topk(qv, kg_vecs_norm, top_k, sim_th, min_k)
            kept_global_idx = [i for i, _ in idx_scores]
        elif ranker == "adpt":
            idx_scores = _cosine_topk_adaptive(
                qv, kg_vecs_norm, top_k, sim_th, min_k)
            kept_global_idx = [i for i, _ in idx_scores]
        else:
            # math 模式：不靠關鍵字/時間窗，純向量幾何去噪
            sims_all = kg_vecs_norm @ qv
            kept_global_idx = _math_only_filtering(
                sims_all,
                kg_vecs_norm,
                base_th=sim_th,
                q=0.90,
                z_th=-0.25,
                knn_k=5,
                dens_q=0.30,
                rnn_k=5,
                mmr_topk=min(top_k, 30),
                mmr_lambda=0.7,
            ).tolist()

        for idx in kept_global_idx:
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
# 三元組檢索（舊）
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
    ranker: str = "math",  # 預設使用數學去噪
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

        if ranker == "topk":
            idx_scores = _cosine_topk(vec, kg_vecs_norm, top_k, sim_th, min_k)
            kept_global_idx = [i for i, _ in idx_scores]
        elif ranker == "adpt":
            idx_scores = _cosine_topk_adaptive(
                vec, kg_vecs_norm, top_k, sim_th, min_k)
            kept_global_idx = [i for i, _ in idx_scores]
        else:
            sims_all = kg_vecs_norm @ vec
            kept_global_idx = _math_only_filtering(
                sims_all,
                kg_vecs_norm,
                base_th=sim_th,
                q=0.90,
                z_th=-0.25,
                knn_k=5,
                dens_q=0.30,
                rnn_k=5,
                mmr_topk=min(top_k, 30),
                mmr_lambda=0.7,
            ).tolist()

        for idx in kept_global_idx:
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
# 純數學去噪排序器（不依賴關鍵字或時間窗）
# ===========================================================

def _adaptive_gate(sims: np.ndarray,
                   base_th: float = 0.6,
                   q: float = 0.90,
                   z_th: float = -0.25) -> float:
    """
    針對當前 query 的自適應門檻：
      - 至少不低於 base_th
      - 至少不低於分位數 q
      - 同時以 Z-score 過濾太低分（均值以下太多）的尾端
    """
    if sims.size == 0:
        return base_th
    qv = float(np.quantile(sims, q))
    mu = float(np.mean(sims))
    sd = float(np.std(sims)) if sims.size > 1 else 0.0
    if sd > 1e-12:
        z_cut = mu + z_th * sd
        return max(base_th, qv, z_cut)
    return max(base_th, qv)


def _knn_density(sims_matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """
    sims_matrix: shape (N, N)，候選之間的兩兩相似度（對稱矩陣，對角可視為1）
    回傳每個點的「與其 k 個最近鄰」的平均相似度（不含自己）。
    """
    N = sims_matrix.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=float)
    order = np.argsort(sims_matrix, axis=1)[:, ::-1]
    topk_idx = order[:, 1:k+1] if k < N else order[:, 1:]
    rows = np.arange(N)[:, None]
    dens = sims_matrix[rows, topk_idx].mean(
        axis=1) if topk_idx.size else np.zeros(N)
    return dens


def _reciprocal_nn_mask(sims_matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """
    若 i 的 top-k 近鄰清單中包含 j，且 j 的 top-k 近鄰清單也包含 i，則 i 與至少一人互為近鄰。
    回傳布林遮罩，True=保留。
    """
    N = sims_matrix.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=bool)
    order = np.argsort(sims_matrix, axis=1)[:, ::-1]
    topk = order[:, 1:k+1] if k < N else order[:, 1:]
    mask = np.zeros(N, dtype=bool)
    for i in range(N):
        nbrs = set(topk[i].tolist())
        for j in nbrs:
            if j < 0 or j >= N:
                continue
            if i in set(topk[j].tolist()):
                mask[i] = True
                break
    if not mask.any():  # 全掛時保底全留
        mask[:] = True
    return mask


def _mmr_select(query_sims: np.ndarray,
                inter_sims: np.ndarray,
                top_k: int = 30,
                lam: float = 0.7) -> list:
    """
    query_sims: shape (N,)   -> 與 query 的相似度
    inter_sims: shape (N,N)  -> 候選間相似度
    回傳挑選的索引列表（局部座標）。
    """
    N = query_sims.shape[0]
    if N == 0:
        return []
    selected = []
    candidates = set(range(N))
    while len(selected) < min(top_k, N) and candidates:
        best_idx, best_score = None, -1e9
        for i in candidates:
            redundancy = max(inter_sims[i, selected]) if selected else 0.0
            score = lam * query_sims[i] - (1.0 - lam) * redundancy
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected


def _math_only_filtering(all_sims: np.ndarray,
                         kg_vecs_norm: np.ndarray,
                         base_th: float = 0.6,
                         q: float = 0.90,
                         z_th: float = -0.25,
                         knn_k: int = 5,
                         dens_q: float = 0.30,
                         rnn_k: int = 5,
                         mmr_topk: int = 30,
                         mmr_lambda: float = 0.7) -> np.ndarray:
    """
    all_sims: shape (M,) = 每個 KG 向量與本次 query 的相似度
    回傳「保留的索引」（相對於 kg_vecs_norm 的全域 index）

    階段：
      1) 自適應門檻 (分位數 + Z-score)
      2) 在保留集上建立候選間相似度矩陣
      3) kNN 密度過濾（丟掉密度分位數以下者）
      4) 互為近鄰過濾（保留具互惠近鄰者）
      5) MMR 多樣化選擇
    """
    # 1) Adaptive gate
    th = _adaptive_gate(all_sims, base_th=base_th, q=q, z_th=z_th)
    idx1 = np.where(all_sims >= th)[0]
    if idx1.size == 0:
        # 全部太低 -> 取前 20 作為候選
        idx1 = np.argsort(all_sims)[-20:][::-1]

    cand_vecs = kg_vecs_norm[idx1]
    # 2) 候選間相似度
    inter = cand_vecs @ cand_vecs.T
    # 3) kNN density
    dens = _knn_density(inter, k=knn_k)
    dens_cut = float(np.quantile(dens, dens_q)) if dens.size else -1
    keep2 = np.where(dens >= dens_cut)[0]
    if keep2.size == 0:
        keep2 = np.arange(len(idx1))
    idx2 = idx1[keep2]
    inter2 = inter[np.ix_(keep2, keep2)]
    qsim2 = all_sims[idx2]
    # 4) Reciprocal NN
    rmask = _reciprocal_nn_mask(inter2, k=rnn_k)
    idx3 = idx2[rmask]
    inter3 = inter2[np.ix_(rmask, rmask)]
    qsim3 = all_sims[idx3]
    # 5) MMR
    sel_local = _mmr_select(qsim3, inter3, top_k=mmr_topk, lam=mmr_lambda)
    return idx3[sel_local]


# ===========================================================
# 屬性硬過濾封裝（保留原函式簽名；內部可選數學排序器）
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
    # 兼容參數（仍然保留，但預設不用）：僅為向後相容
    prefilter_keywords: Optional[Iterable[str]] = None,
    prefilter_min_recall: float = 0.2,
    # 選擇排序器
    ranker: str = "math",  # "math" | "topk" | "adpt"
) -> List[str]:
    """
    在既有檢索流程外包一層可選的屬性硬過濾；不影響原函式與輸出模式。
    - 預設採用「純數學去噪」ranker（不依賴關鍵字/時間窗）
    - 如需回退傳統 Top-K，請傳 ranker="topk"（或 "adpt" 使用分位數自適應）
    """
    # 預先縮表（為相容存在，但預設不啟用；即使傳進來也只做極輕量處理）
    use_df = kg_df
    use_vecs = kg_vecs_norm
    if prefilter_keywords:
        kws = [k for k in prefilter_keywords if k]
        if kws:
            mask = np.zeros(len(kg_df), dtype=bool)
            head_col = kg_df.get('head', pd.Series([""] * len(kg_df)))
            tail_col = kg_df.get('tail', pd.Series([""] * len(kg_df)))
            head_norm = head_col.astype(str).str.replace(" ", "").str.lower()
            tail_norm = tail_col.astype(str).str.replace(" ", "").str.lower()
            for k in kws:
                kk = k.replace(" ", "").lower()
                mask |= head_norm.str.contains(
                    kk, na=False) | tail_norm.str.contains(kk, na=False)
            hit_ratio = mask.mean() if len(mask) else 0.0
            if hit_ratio >= prefilter_min_recall:
                use_df = kg_df[mask].reset_index(drop=True)
                use_vecs = kg_vecs_norm[mask]

    results: List[str] = []
    for tp in triples:
        vec = embed_fn(tp)
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec = vec / n

        if ranker == "topk":
            idx_scores = _cosine_topk(vec, use_vecs, top_k, sim_th, min_k)
            kept_local_idx = [i for i, _ in idx_scores]
            kept_global_idx = [int(use_df.index[i]) if hasattr(
                use_df.index, "__iter__") else i for i in kept_local_idx]
        elif ranker == "adpt":
            idx_scores = _cosine_topk_adaptive(
                vec, use_vecs, top_k, sim_th, min_k)
            kept_local_idx = [i for i, _ in idx_scores]
            kept_global_idx = [int(use_df.index[i]) if hasattr(
                use_df.index, "__iter__") else i for i in kept_local_idx]
        else:
            sims_all = use_vecs @ vec
            kept_local_idx = _math_only_filtering(
                sims_all,
                use_vecs,
                base_th=sim_th,
                q=0.90,
                z_th=-0.25,
                knn_k=5,
                dens_q=0.30,
                rnn_k=5,
                mmr_topk=min(top_k, 30),
                mmr_lambda=0.7,
            ).tolist()
            kept_global_idx = [int(use_df.index[i]) if hasattr(
                use_df.index, "__iter__") else i for i in kept_local_idx]

        for gi in kept_global_idx:
            row = kg_df.iloc[gi]

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
# 範例 row_filter（保留）
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
