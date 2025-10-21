# -*- coding: utf-8-sig -*-
"""
強化版知識檢索（對齊 answerer 模式，已移除來源白名單）
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from .loader import KG_DF, KG_VECS_NORM, HP_COL, RP_COL, TP_COL

load_dotenv(override=True)

SIM_TH: float = float(os.getenv("SIM_TH", "0.6"))
TOP_K: int = int(os.getenv("TOP_K", "200"))  # 僅作為上限
DEBUG: bool = os.getenv("VERIFIER_DEBUG", "0") == "1"
DEBUG_PATH: str = os.getenv(
    "VERIFIER_DEBUG_PATH",
    "data/processed/verifier/debug/verifier_search_debug.log",
)

def _dbg(msg: str) -> None:
    """追加式 debug 記錄（僅在 DEBUG=1 時有效）。"""
    if not DEBUG:
        return
    os.makedirs(os.path.dirname(DEBUG_PATH), exist_ok=True)
    with open(DEBUG_PATH, "a", encoding="utf-8-sig") as f:
        f.write(msg.rstrip() + "\n")

CAND_FACTOR: int = int(os.getenv("RAG_CAND_FACTOR", "8"))
EXPAND_GPT_MODEL: str = os.getenv("EXPAND_GPT_MODEL", "gpt-4o-mini")
USE_GPT_EXPAND: bool = True

try:
    from config import EVIDENCE_MIN_HITS, REQUIRE_ANCHOR_IN_EVIDENCE  # type: ignore
except Exception:
    EVIDENCE_MIN_HITS: int = int(os.getenv("EVIDENCE_MIN_HITS", "2"))
    REQUIRE_ANCHOR_IN_EVIDENCE: bool = os.getenv(
        "REQUIRE_ANCHOR_IN_EVIDENCE", "1"
    ) == "1"

_TOKEN_RE = re.compile(r"([\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,})")
_RE_SENT_SPLIT = re.compile(r"(?<=[。！？!?])\s+")
# 僅在「成對包裹」時修剪，例如避免把 "《中央社》" 剃成 "《中央社 "
_RE_FULLWIDTH_TRIM = re.compile(r"^(?:[（『「《(【]+)(.*?)(?:[）』」》)】]+)$")
_RE_REL_PAREN_FULL = re.compile(r"（.*?）")
_RE_REL_PAREN_HALF = re.compile(r"\(.*?\)")

def _get_gpt() -> Optional[Any]:
    """延遲載入 GPTClient；失敗時自動停用擴充功能。"""
    if not USE_GPT_EXPAND:
        return None
    try:
        from ..llm.gpt import GPTClient
        return GPTClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_id=EXPAND_GPT_MODEL,
        )
    except Exception:
        return None

def _safe_json_load(x: Any) -> Dict[str, Any]:
    """安全解析 JSON 格式欄位；失敗回傳空 dict。"""
    if x is None or x == "" or (isinstance(x, float) and np.isnan(x)):
        return {}
    try:
        return json.loads(x)
    except Exception:
        return {}
    
def _tri_get(tp: Dict[str, Any], key: str) -> str:
    """同時支援 h/r/t 與 head/relation/tail 的鍵名讀取。

    Args:
        tp: 三元組查詢物件。
        key: 'head' | 'relation' | 'tail'
    """
    alt = {"head": "h", "relation": "r", "tail": "t"}[key]
    return (str(tp.get(key) or tp.get(alt) or "")).strip()


def _split_sentences(text: str) -> List[str]:
    """以中式標點切句並做基本清理與過短句刪除。"""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    s = s.strip(" ，、；,;:…")
    m = _RE_FULLWIDTH_TRIM.match(s)
    if m:
        s = m.group(1)    
    if not s:
        return []
    parts = _RE_SENT_SPLIT.split(s) if _RE_SENT_SPLIT.search(s) else [s]
    out: List[str] = []
    for p in parts:
        t = p.strip(" 、，；,;:…")
        if not t:
            continue
        if not re.search(r"[。！？!?]$", t):
            t = t + "。"
        if len(re.sub(r"\s+", "", t)) < 6:
            continue
        out.append(t)
    return out

def _extract_terms(text: str) -> List[str]:
    """依規則從輸入字串抽取關鍵詞（含中文詞、英數詞）。"""
    return [m.group(0) for m in _TOKEN_RE.finditer(text or "")]

def _is_latin_token(s: str) -> bool:
    """檢查是否含英數字元（適用 word-boundary 規則）。"""
    return bool(re.search(r"[A-Za-z0-9]", s or ""))

def _normalize_ascii(s: str) -> str:
    """規範化字串至大小寫不敏感的形式。"""
    return (s or "").casefold()

def _match_terms(hay: str, needles: List[str]) -> set:
    """回傳在 hay 中被命中的「不同關鍵詞」集合。"""
    found = set()
    if not hay or not needles:
        return found
    text = str(hay)
    text_ascii = _normalize_ascii(text)
    for raw in needles:
        if not raw:
            continue
        term = str(raw).strip()
        if len(term) < 2:
            continue
        if _is_latin_token(term):
            pat = rf"(?<![0-9A-Za-z_]){re.escape(_normalize_ascii(term))}(?![0-9A-Za-z_])"
            if re.search(pat, text_ascii):
                found.add(term)
        else:
            if term in text:
                found.add(term)
    return found

def _parse_date_yyyy_mm_dd(s: str) -> int:
    """將日期字串 YYYY-MM-DD 轉整數 YYYYMMDD，供排序用；失敗回 0。"""
    if not s or not isinstance(s, str):
        return 0
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s.strip())
    if not m:
        return 0
    y, mm, dd = m.groups()
    try:
        return int(f"{int(y):04d}{int(mm):02d}{int(dd):02d}")
    except Exception:
        return 0

@lru_cache(maxsize=4096)
def _expand_terms_with_gpt_cached(base_query: str) -> Tuple[str, ...]:
    """用 GPT 對基底查詢做別名/同義詞擴充（LRU 快取）。"""
    gpt = _get_gpt()
    if gpt is None:
        return tuple()
    sys_prompt = """
    你是專門用於檢索的查詢擴展器 (Query Expander)。
    你的任務是根據使用者輸入的「查詢詞」，產生一個包含原始查詢詞以及其緊密同義詞、別名或相關術語的列表，以提高檢索系統的召回率。

    請嚴格遵守以下規則：
    1. 列表中第一個元素「必須」是未經修改的原始「查詢詞」。
    2. 請「只」輸出 JSON 格式，內容為 {"augmented_terms":["..."]}。
    3. 產生的同義詞應語意貼近。
    ```
    - 範例輸入1："街頭抗議"
    - 範例輸出1：{"augmented_terms":["街頭抗議", "群眾示威", "社會運動", "遊行集會"]}
    ```
    ```
    - 範例輸入2："國際制裁"
    - 範例輸出2：{"augmented_terms":["國際制裁", "經濟制裁", "外交制裁", "國際懲罰措施"]}
    ```
    輸出前必須再次確認「完全符合」上述規則。
    """

    user_prompt = f"[查詢]\n{base_query}\n"
    text = gpt.chat(system_prompt=sys_prompt, user_prompt=user_prompt)
    try:
        obj = json.loads(text)
        terms = obj.get("augmented_terms", []) or []
        return tuple(str(t).strip() for t in terms if t and len(str(t)) <= 24)
    except Exception:
        return tuple()

def cosine_search(tp: dict, q_vec: np.ndarray) -> List[int]:
    """
    以向量相似度檢索 KG，回傳通過篩選的行索引列表（已移除來源白名單）。
    """
    sims = KG_VECS_NORM @ q_vec
    per_query_k = max(TOP_K * CAND_FACTOR, TOP_K)
    if per_query_k >= len(sims):
        cand_idx = np.arange(len(sims))
    else:
        cand_idx = np.argpartition(sims, -per_query_k)[-per_query_k:]
    cand_idx = cand_idx[np.argsort(sims[cand_idx])[::-1]]

    cand_idx = cand_idx[sims[cand_idx] >= SIM_TH]
    if cand_idx.size == 0:
        return []

    # 允許外層以 h/r/t 傳入；內層統一取標準鍵名
    q_head = _tri_get(tp, "head")
    q_rel_raw = _tri_get(tp, "relation")
    q_rel = _RE_REL_PAREN_HALF.sub("", _RE_REL_PAREN_FULL.sub("", q_rel_raw))
    q_rel = re.sub(r"[，、。；;:：\s]+", "", q_rel)
    q_tail = _tri_get(tp, "tail")
    base_query = " ".join([q_head, q_rel, q_tail]).strip()

    terms: List[str] = []
    terms.extend(_extract_terms(q_head))
    terms.extend(_extract_terms(q_rel))
    terms.extend(_extract_terms(q_tail))
    if base_query:
        terms.extend(list(_expand_terms_with_gpt_cached(base_query)))
    if terms:
        seen: set[str] = set()
        terms = [t for t in terms if (t not in seen and not seen.add(t))]

    kept: List[Tuple[int, float, int]] = []
    # 一元查詢偵測/降級策略
    is_unary = (not q_rel) and (bool(q_head) ^ bool(q_tail))
    # 就地調整：一元查詢不要開太大
    if is_unary:
        # 只看前 2*K 候選，且稍後要求 hits 更嚴
        cand_idx = cand_idx[: max(TOP_K * 2, 100)]

    for i in cand_idx:
        h = KG_DF.at[i, "head"]
        t = KG_DF.at[i, "tail"]
        rp = _safe_json_load(KG_DF.at[i, RP_COL]) if RP_COL else {}

        if (q_head and h == q_head) or (q_tail and q_tail != "未知" and t == q_tail):
            date_int = _parse_date_yyyy_mm_dd(rp.get("date", ""))
            kept.append((i, float(sims[i]), date_int))
            _dbg(f"[keep-by-exact] {i} sim={sims[i]:.4f} head={h} tail={t} date={date_int}")
            continue

        ev = rp.get("evidence", "") or ""
        if not ev or not terms:
            continue

        # anchor 規則：若 head 空但 tail 有值，用 tail 當錨
        anchor_term = q_head or (q_tail if not q_head else "")
        anchor = anchor_term if REQUIRE_ANCHOR_IN_EVIDENCE and anchor_term else None        
        hits_all: set = set()
        anchor_ok = (anchor is None)

        for sent in _split_sentences(ev):
            hits_all |= _match_terms(sent, terms)
            if not anchor_ok and anchor and (anchor in sent):
                anchor_ok = True
            required_hits = (EVIDENCE_MIN_HITS + 1) if is_unary else EVIDENCE_MIN_HITS
            if anchor_ok and len(hits_all) >= required_hits:                
                break

        if not anchor_ok or len(hits_all) < required_hits:
            continue

        date_int = _parse_date_yyyy_mm_dd(rp.get("date", ""))
        kept.append((i, float(sims[i]), date_int))
        _dbg(
            f"[keep-by-evidence] {i} sim={sims[i]:.4f} "
            f"hits={len(hits_all)} terms={sorted(list(hits_all))} "
            f"anchor={'Y' if anchor else 'N'} date={date_int}"
        )

    if not kept:
        return []

    kept.sort(key=lambda x: (x[1], x[2]), reverse=True)
    out_idx = [i for (i, _, __) in kept]
    if len(out_idx) > TOP_K:
        out_idx = out_idx[:TOP_K]
    return out_idx

def kg_row_to_detail(idx: int) -> Tuple[dict, Dict[str, dict]]:
    """
    依現有 I/O 規格產生輸出。
    對外 tri 外觀統一為 h/r/t；det 使用 head/rel/tail_props。
    """
    row = KG_DF.iloc[idx]
    # 對外 façade：h/r/t
    tri = {"h": row["head"], "r": row["relation"], "t": row["tail"]}
    det = {
        "head": _safe_json_load(row[HP_COL]) if HP_COL else {},
        "rel": _safe_json_load(row[RP_COL]) if RP_COL else {},
        "tail": _safe_json_load(row[TP_COL]) if TP_COL else {},
    }
    return tri, det
