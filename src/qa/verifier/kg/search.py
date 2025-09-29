"""
強化版知識檢索（對齊 answerer 模式）

能力（預設啟用）：
1) evidence 逐句檢查 + must-terms（由 head/relation/tail 自動抽詞）
2) GPT 擴充詞（LRU 快取；模型預設 gpt-4o-mini）
3) 候選池放大（RAG_CAND_FACTOR）+ 相似度門檻（SIM_TH）
4) 來源白名單（預設：PTS,CNA,MOI,EY）
5) 同分時日期較新者優先（解析 rel_props.date 的 YYYY-MM-DD）

說明：
- 與短問句模式一致的召回策略：先以餘弦相似度擴大候選，再逐句檢查是否
  命中多個 must-term（更嚴格），並受來源白名單限制；最後以相似度與
  日期排序裁切至 TOP_K。
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .loader import KG_DF, KG_VECS_NORM, HP_COL, RP_COL, TP_COL
from ..core.config import SIM_TH, TOP_K

# ──────────────── 參數 ────────────────

# Debug（手動需要時可設 VERIFIER_DEBUG=1）
DEBUG = os.getenv("VERIFIER_DEBUG", "0") == "1"
DEBUG_PATH = os.getenv(
    "VERIFIER_DEBUG_PATH",
    "data/processed/verifier/debug/verifier_search_debug.log",
)


def _dbg(msg: str) -> None:
    """簡單的檔案追加式 debug 記錄（僅在 DEBUG=1 時有效）"""
    if not DEBUG:
        return
    os.makedirs(os.path.dirname(DEBUG_PATH), exist_ok=True)
    with open(DEBUG_PATH, "a", encoding="utf-8-sig") as f:
        f.write(msg.rstrip() + "\n")


# 候選放大倍數
CAND_FACTOR = int(os.getenv("RAG_CAND_FACTOR", "8"))

# GPT 擴充詞：預設啟用
EXPAND_GPT_MODEL = os.getenv("EXPAND_GPT_MODEL", "gpt-4o-mini")
USE_GPT_EXPAND = True

# 來源白名單（預設與短問句一致）
_DEFAULT_SOURCES = "PTS,CNA,MOI,EY"
ALLOWED_SOURCES = {
    s.strip().upper()
    for s in (os.getenv("RAG_ALLOWED_SOURCES", _DEFAULT_SOURCES)).split(",")
    if s.strip()
}

# evidence 最低關鍵詞命中數與 anchor 要求（可由 config 覆蓋）
try:
    # 若專案中有 config.py，可在那邊設定：
    # EVIDENCE_MIN_HITS = 2
    # REQUIRE_ANCHOR_IN_EVIDENCE = True
    from config import EVIDENCE_MIN_HITS, REQUIRE_ANCHOR_IN_EVIDENCE  # type: ignore
except Exception:
    EVIDENCE_MIN_HITS = int(os.getenv("EVIDENCE_MIN_HITS", "2"))
    REQUIRE_ANCHOR_IN_EVIDENCE = os.getenv(
        "REQUIRE_ANCHOR_IN_EVIDENCE", "1") == "1"

# token 規則：中文連續 >=2、或英數底線連續 >=2
_TOKEN_RE = re.compile(r"([\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,})")

# 以中文結句標點切句
_RE_SENT_SPLIT = re.compile(r"(?<=[。！？!?])\s+")
_RE_FULLWIDTH_TRIM = re.compile(r"^[（『「《(【]+|[）』」》)】]+$")


# ──────────────── GPT Client（延遲載入，失敗自動停用） ────────────────
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


# ──────────────── 小工具 ────────────────
def _safe_json_load(x: Any) -> Dict[str, Any]:
    """安全解析 JSON 格式欄位；失敗回傳空 dict。"""
    if x is None or x == "" or (isinstance(x, float) and np.isnan(x)):
        return {}
    try:
        return json.loads(x)
    except Exception:
        return {}


def _split_sentences(text: str) -> List[str]:
    """將 evidence 以中式標點切句，並做基本清理與過短句刪除。"""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    s = s.strip(" ，、；,;:…")
    s = _RE_FULLWIDTH_TRIM.sub("", s)
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


def _contains_any(hay: str, needles: List[str]) -> bool:
    """寬鬆判斷：hay 是否包含 needles 任一詞（保留舊邏輯用，避免連鎖影響）。"""
    s = str(hay or "")
    return any(n and n in s for n in needles)


def _is_latin_token(s: str) -> bool:
    """是否含英數（適用 word-boundary 的語料）。"""
    return bool(re.search(r"[A-Za-z0-9]", s or ""))


def _normalize_ascii(s: str) -> str:
    return (s or "").casefold()


def _match_terms(hay: str, needles: List[str]) -> set:
    """
    回傳在 hay 中被命中的「不同關鍵詞」集合。
    - 英數詞：採用 word boundary，避免 'net' 命中 'internet' 等誤擊
    - CJK 詞：維持子字串包含（中文沒有可靠的邊界符）
    - 自動過濾空字串與長度<2的 token
    """
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


def _src_from_doc_id(doc_id: str) -> str:
    """由 doc_id 取來源縮寫（如 CNA_... → CNA）。"""
    return (str(doc_id).split("_")[0] or "").upper()


def _parse_date_yyyy_mm_dd(s: str) -> int:
    """將日期字串 YYYY-MM-DD 轉整數 YYYYMMDD，排序用；失敗回 0。"""
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


# ──────────────── GPT 擴充詞（LRU 快取） ────────────────
@lru_cache(maxsize=4096)
def _expand_terms_with_gpt_cached(base_query: str) -> Tuple[str, ...]:
    """
    將「head relation tail」組成之 base_query 丟給 GPT 做別名/同義詞擴充。
    輸出以 tuple 回傳，利於 LRU 快取。
    """
    gpt = _get_gpt()
    if gpt is None:
        return tuple()
    sys_prompt = (
        "你是檢索查詢改寫器，請只輸出 JSON。"
        '格式：{"augmented_terms":["..."]}。'
        "請提供與查詢高度相關的別名/縮寫/同義詞（每個<=24字元）。"
    )
    user_prompt = f"[查詢]\n{base_query}\n"
    text = gpt.chat(system_prompt=sys_prompt, user_prompt=user_prompt)
    try:
        obj = json.loads(text)
        terms = obj.get("augmented_terms", []) or []
        return tuple(str(t).strip() for t in terms if t and len(str(t)) <= 24)
    except Exception:
        return tuple()


# ──────────────── 主檢索 ────────────────
def cosine_search(tp: dict, q_vec: np.ndarray) -> List[int]:
    """
    以向量相似度檢索 KG，回傳通過篩選的行索引列表。

    流程：
    1) 相似度取 Top-(TOP_K * CAND_FACTOR) 作為候選（提高召回）
    2) 以 SIM_TH 過濾低相似候選
    3) 建立 must-terms（head/relation/tail + GPT 擴充詞）
    4) 逐候選檢查：
       4.1 若 head/tail 精確相等 → 直接保留（仍需通過來源白名單）
       4.2 否則 evidence 逐句，比對多詞門檻（全文累計命中 >= K）
    5) 來源白名單過濾（PTS/CNA/MOI/EY）
    6) 依 (相似度, 日期) 由大到小排序，截至 TOP_K
    """
    # 1) 候選池放大
    sims = KG_VECS_NORM @ q_vec
    per_query_k = max(TOP_K * CAND_FACTOR, TOP_K)
    if per_query_k >= len(sims):
        cand_idx = np.arange(len(sims))
    else:
        cand_idx = np.argpartition(sims, -per_query_k)[-per_query_k:]
    cand_idx = cand_idx[np.argsort(sims[cand_idx])[::-1]]

    # 2) 相似度門檻
    cand_idx = cand_idx[sims[cand_idx] >= SIM_TH]
    if cand_idx.size == 0:
        return []

    # 3) must-terms（head/relation/tail）＋ GPT 擴充
    q_head = (tp.get("head") or "").strip()
    q_rel = (tp.get("relation") or "").strip()
    q_tail = (tp.get("tail") or "").strip()
    base_query = " ".join([q_head, q_rel, q_tail]).strip()

    terms: List[str] = []
    terms.extend(_extract_terms(q_head))
    terms.extend(_extract_terms(q_rel))
    terms.extend(_extract_terms(q_tail))
    if USE_GPT_EXPAND and base_query:
        terms.extend(list(_expand_terms_with_gpt_cached(base_query)))

    # 去重保序
    if terms:
        seen: set[str] = set()
        terms = [t for t in terms if (t not in seen and not seen.add(t))]

    # 4) 逐一檢查候選：先做舊有 head/tail 精確比對；否則用 evidence 逐句 + terms
    kept: List[Tuple[int, float, int]] = []  # (idx, sim, date_int)
    for i in cand_idx:
        h = KG_DF.at[i, "head"]
        t = KG_DF.at[i, "tail"]
        rp = _safe_json_load(KG_DF.at[i, RP_COL]) if RP_COL else {}

        # 4.1 精確 head/tail 命中（放寬 tail != "未知"）
        if (q_head and h == q_head) or (q_tail and q_tail != "未知" and t == q_tail):
            if ALLOWED_SOURCES:
                src = _src_from_doc_id(rp.get("doc_id", ""))
                if src and src not in ALLOWED_SOURCES:
                    _dbg(f"[drop-by-source] {i} src={src}")
                    continue
            date_int = _parse_date_yyyy_mm_dd(rp.get("date", ""))
            kept.append((i, float(sims[i]), date_int))
            _dbg(
                f"[keep-by-exact] {i} sim={sims[i]:.4f} "
                f"head={h} tail={t} date={date_int}"
            )
            continue

        # 4.2 否則：evidence 逐句，累計「不同關鍵詞」命中數（更嚴）
        ev = rp.get("evidence", "") or ""
        if not ev or not terms:
            # 沒 evidence 或沒查詢詞 → 不保留，避免全放行造成噪音
            continue

        # anchor：預設用 head 當主體；可由設定關閉
        anchor = q_head if REQUIRE_ANCHOR_IN_EVIDENCE else None

        hits_all: set = set()
        anchor_ok = (anchor is None)

        for sent in _split_sentences(ev):
            # 累計命中
            hits_all |= _match_terms(sent, terms)
            # anchor 必須命中（若有設）
            if not anchor_ok and anchor and (anchor in sent):
                anchor_ok = True
            # 快速提前通過（在極多句 evidence 時少跑一點）
            if anchor_ok and len(hits_all) >= EVIDENCE_MIN_HITS:
                break

        if not anchor_ok or len(hits_all) < EVIDENCE_MIN_HITS:
            continue

        if ALLOWED_SOURCES:
            src = _src_from_doc_id(rp.get("doc_id", ""))
            if src and src not in ALLOWED_SOURCES:
                _dbg(f"[drop-by-source] {i} src={src}")
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

    # 6) 排序與裁切：相似度優先；同分則日期新者優先
    kept.sort(key=lambda x: (x[1], x[2]), reverse=True)
    out_idx = [i for (i, _, __) in kept]
    if len(out_idx) > TOP_K:
        out_idx = out_idx[:TOP_K]
    return out_idx


def kg_row_to_detail(idx: int) -> Tuple[dict, Dict[str, dict]]:
    """
    依現有 I/O 規格回傳：
      tri = {'head', 'relation', 'tail'}
      det = {'head': {...}, 'rel': {...}, 'tail': {...}}
    """
    row = KG_DF.iloc[idx]
    tri = {
        "head": row["head"],
        "relation": row["relation"],
        "tail": row["tail"],
    }
    det = {
        "head": _safe_json_load(row[HP_COL]) if HP_COL else {},
        "rel": _safe_json_load(row[RP_COL]) if RP_COL else {},
        "tail": _safe_json_load(row[TP_COL]) if TP_COL else {},
    }
    return tri, det
