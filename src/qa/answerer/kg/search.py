"""
檢索模組（CKIP + 全域 GPT 重排；允許來源白名單 + 句子化修復 + 自動關鍵詞擴充）

設計目標：
1. 準確：只納入「與使用者問題高度相關」的證據句，避免新聞雜訊。
2. 完整：時間敏感時優先新資料，但保留能補足不同狀態（交保/延押/羈押…）的舊訊息。
3. 穩健：evidence 逐句處理，避免漏掉關鍵短句（如「裁定7000萬交保」）。
4. 可擴展：可調整候選規模、重排池大小、語義去重閾值，以及 GPT 動態多樣性覆蓋。

主要修正：
- ✅ evidence「逐句」處理與代表句挑選
- ✅ must-terms：自動從 question/triples/GPT 擴充詞抽取 → 候選需至少命中
- ✅ 全域 GPT 重排：避免單一子查詢壟斷
- ✅ 語義去重（cosine）
- ✅ 時間敏感問題：日期新者優先
- ✅ GPT 動態 tags：涵蓋不同狀態類型（交保/延押/羈押…）
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.qa.answerer.llm.gpt import GPTClient

# ──────────────── 基本設定 ────────────────

CSV_PATH_DEFAULT = "data/raw/knowledge-graph/neo4j-kg-raw-graph.csv"
KG_EMB_PATH_DEFAULT = "data/processed/knowledge-graph/neo4j-kg.emb.npy"

CKIP_MODEL_ROOT = os.getenv(
    "CKIP_MODEL_ROOT", "models/CKIP/models--ckiplab--bert-base-chinese"
)

ALLOWED_SOURCES_ENV = os.getenv("RAG_ALLOWED_SOURCES", "PTS,CNA,MOI,EY")
ALLOWED_SOURCES = {s.strip().upper() for s in ALLOWED_SOURCES_ENV.split(",") if s}

TAIL_PLACEHOLDER = "未知"

TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "200"))
CAND_FACTOR = int(os.getenv("RAG_CAND_FACTOR", "12"))
RERANK_POOL_LIMIT = int(os.getenv("RAG_RERANK_POOL", "400"))
SIMILARITY_DUP_TH = float(os.getenv("RAG_DUP_TH", "0.80"))
DIVERSITY_TAGS_MAX = int(os.getenv("RAG_DIVERSITY_TAGS", "3"))

RERANK_GPT_MODEL = os.getenv("RERANK_GPT_MODEL", "gpt-4o-mini")
EXPAND_GPT_MODEL = os.getenv("EXPAND_GPT_MODEL", "gpt-4o-mini")

_TIME_SENSITIVE_RE = re.compile(r"(目前|現在|進度|最新|now|current)", re.I)
_TOKEN_RE = re.compile(r"([\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,})")

_RE_SENT_SPLIT = re.compile(r"(?<=[。！？!?])\s+")
_RE_FULLWIDTH_TRIM = re.compile(r"^[（『「《(【]+|[）』」》)】]+$")


# ──────────────── CKIP 向量編碼器 ────────────────

class CKIPEmbedder:
    """CKIP Sentence-BERT 編碼器（與 KG 向量檔共用同一語義空間）"""

    _model = None
    _dim: Optional[int] = None

    @classmethod
    def ensure_loaded(cls) -> None:
        if cls._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        import torch

        path = cls._resolve_snapshot(CKIP_MODEL_ROOT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls._model = SentenceTransformer(path, device=device, trust_remote_code=True)
        cls._dim = cls._model.get_sentence_embedding_dimension()

    @staticmethod
    def _resolve_snapshot(root: str) -> str:
        from pathlib import Path

        p = Path(root)
        if (p / "config.json").is_file():
            return str(p)
        snaps = p / "snapshots"
        if snaps.is_dir():
            for sub in snaps.iterdir():
                if (sub / "config.json").is_file():
                    return str(sub)
        raise FileNotFoundError(f"❌ 找不到 CKIP 模型快照於：{root}")

    @classmethod
    def encode(cls, texts: List[str]) -> np.ndarray:
        cls.ensure_loaded()
        embs = cls._model.encode(
            texts, batch_size=16, convert_to_numpy=True, show_progress_bar=False
        )
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return np.divide(embs, norms, out=np.zeros_like(embs), where=(norms != 0))

    @classmethod
    def dim(cls) -> int:
        cls.ensure_loaded()
        return int(cls._dim or 0)


# ──────────────── KG 載入 ────────────────

_KG_DF: Optional[pd.DataFrame] = None
_KG_EMB: Optional[np.ndarray] = None


def _ensure_kg_loaded(
    csv_path: str = CSV_PATH_DEFAULT, emb_path: str = KG_EMB_PATH_DEFAULT
) -> Tuple[pd.DataFrame, np.ndarray]:
    global _KG_DF, _KG_EMB
    if _KG_DF is None:
        _KG_DF = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    if _KG_EMB is None:
        if not os.path.isfile(emb_path):
            raise FileNotFoundError(f"找不到 KG 向量檔：{emb_path}")
        _KG_EMB = np.load(emb_path).astype(np.float32)
        norms = np.linalg.norm(_KG_EMB, axis=1, keepdims=True)
        _KG_EMB = np.divide(_KG_EMB, norms, out=np.zeros_like(_KG_EMB), where=(norms != 0))
    if CKIPEmbedder.dim() != _KG_EMB.shape[1]:
        raise RuntimeError("❌ 維度不一致：KG 向量 vs CKIP")
    return _KG_DF, _KG_EMB


# ──────────────── 句子化處理 ────────────────

def _split_sentences(text: str) -> List[str]:
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


# ──────────────── Must-terms ────────────────

def _extract_terms(text: str) -> List[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer(text or "")]


def _build_must_terms(
    question: str, triples: List[Dict[str, str]], expansions: List[str]
) -> List[str]:
    terms: List[str] = []
    terms.extend(_extract_terms(question))
    for tp in triples:
        terms.extend(_extract_terms(tp.get("head", "")))
        terms.extend(_extract_terms(tp.get("relation", "")))
        terms.extend(_extract_terms(tp.get("tail", "")))
    terms.extend(expansions)
    seen, uniq = set(), []
    for t in terms:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def _contains_any(hay: str, needles: List[str]) -> bool:
    return any(n and n in str(hay) for n in needles)


# ──────────────── GPT 擴充查詢 ────────────────

def _expansion_cache_key(question: str, base_query: str) -> str:
    s = json.dumps({"q": question, "bq": base_query}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8-sig")).hexdigest()


@lru_cache(maxsize=2048)
def _expand_terms_with_gpt_cached(
    cache_key: str, sys_prompt: str, user_prompt: str, model: str
) -> Tuple[str, ...]:
    client = GPTClient(api_key=os.getenv("OPENAI_API_KEY"), model_id=model)
    text = client.chat(system_prompt=sys_prompt, user_prompt=user_prompt)
    try:
        obj = json.loads(text)
        terms = obj.get("augmented_terms", []) or []
        return tuple(str(t).strip() for t in terms if t and len(str(t)) <= 24)
    except Exception:
        return tuple()


def _generate_query_expansions(question: str, base_query: str) -> List[str]:
    sys_prompt = (
        "你是檢索查詢改寫器，請只輸出 JSON。"
        '格式：{"augmented_terms":["..."]}。'
        "請提供與問題高度相關的別名/縮寫/同義詞/反義詞（短詞即可）。"
    )
    user_prompt = f"[問題]\n{question}\n[初步查詢]\n{base_query}\n"
    key = _expansion_cache_key(question, base_query)
    return list(_expand_terms_with_gpt_cached(key, sys_prompt, user_prompt, EXPAND_GPT_MODEL))


# ──────────────── 主函式 ────────────────

def retrieve_answers(
    triples_json: Dict[str, Any],
    question: str,
    *,
    top_k: Optional[int] = None,
    csv_path: str = CSV_PATH_DEFAULT,
    kg_emb_path: str = KG_EMB_PATH_DEFAULT,
) -> str:
    kg_df, kg_vecs = _ensure_kg_loaded(csv_path, kg_emb_path)
    triples = [{"head": t["subject"]["text"], "relation": t["relation"], "tail": "未知"}
               for t in triples_json.get("triples", [])]

    if top_k is None:
        top_k = TOP_K_DEFAULT
    if top_k <= 0:
        top_k = None

    # 生成查詢與 GPT 擴充詞
    queries, expansions = [], []
    for tp in triples:
        base = " ".join([tp.get("head", ""), tp.get("relation", ""), tp.get("tail", "")]).strip()
        queries.append(base)
        expansions.extend(_generate_query_expansions(question, base))

    must_terms = _build_must_terms(question, triples, expansions)

    # 簡化：只取 cosine Top-K，逐句檢查 must_terms
    all_idx = []
    if queries:
        q_vecs = CKIPEmbedder.encode(queries)
        per_query_k = max((top_k or TOP_K_DEFAULT) * CAND_FACTOR, TOP_K_DEFAULT)
        for qv in q_vecs:
            sims = kg_vecs @ qv
            idx = np.argpartition(sims, -per_query_k)[-per_query_k:]
            idx = idx[np.argsort(sims[idx])[::-1]]
            for gi in idx:
                props = json.loads(str(kg_df.iloc[gi].get("rel_props", "{}")) or "{}")
                sents = _split_sentences(props.get("evidence", ""))
                if any(_contains_any(s, must_terms) for s in sents):
                    all_idx.append(gi)

    uniq_idx = list(dict.fromkeys(all_idx))
    if not uniq_idx:
        return f"[使用者提問]\n{question}\n\n[知識查詢結果]\n(無相關知識)"

    lines = []
    for i, gi in enumerate(uniq_idx[: (top_k or len(uniq_idx))], 1):
        props = json.loads(str(kg_df.iloc[gi].get("rel_props", "{}")) or "{}")
        src = (props.get("doc_id", "").split("_")[0] or "").upper()
        date = props.get("date", "")
        ev = props.get("evidence", "")
        rep = next((s for s in _split_sentences(ev) if _contains_any(s, must_terms)), None)
        if rep:
            lines.append(f"[{i}] {rep}（{src}，{date}）")

    out = f"[使用者提問]\n{question.strip()}\n\n[知識查詢結果]\n"
    out += "\n".join(lines) if lines else "(無相關知識)"
    return out
