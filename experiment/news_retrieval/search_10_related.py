# -*- coding: utf-8 -*-
"""
$ python experiment/news_retrieval/search_10_related.py

一體化流程：
1) 匯出 MongoDB: News.Real_News -> experiment/data/raw/entire_news.json
2) 以 CKIP BERT 向量化 -> experiment/data/interim/entire_news_vector.npy
   並輸出 idmap -> experiment/data/interim/entire_news_idmap.jsonl
3) 讀取 experiment/data/raw/10_news.txt 做向量檢索
   （增強）同時讀取 experiment/data/interim/10_sentence_extraction_result.json，
           將 Qn_T* 的 subject / relation / scope 生成「獨立子查詢」檢索，
           以 RRF / max / mean / sum 進行分數融合
4) 寫出排名/分數 -> experiment/data/processed/related_10_news.json
5) 依 idmap 對照，輸出原始新聞 -> experiment/data/processed/related_news.json

環境變數：MONGODB_URI（必要，用於連線 MongoDB）

預設模型路徑：
models/CKIP/models--ckiplab--bert-base-chinese（會自動往 snapshots/{亂數}/ 下找有效快照）
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

from pymongo import MongoClient
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import re
from collections import defaultdict

# ---- dotenv（可選）----
try:
    from dotenv import load_dotenv, find_dotenv
    from pathlib import Path as _P
    _found = find_dotenv(usecwd=True) or str(
        _P(__file__).resolve().parents[2] / ".env")
    load_dotenv(_found, override=False)
except Exception:
    pass  # 回退用系統環境變數

# ---- tqdm 進度條 ----
try:
    from tqdm import tqdm
    def _tqdm(x, **kw): return tqdm(x, **kw)
except Exception:
    def _tqdm(x, **kw): return x  # 無 tqdm 時降級


# =========================
# 第 0 部分：Mongo 連線工具
# =========================
def _require_mongo_client() -> MongoClient:
    mongo_uri: str | None = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("環境變數 MONGODB_URI 未設置，無法連線 MongoDB。")
    return MongoClient(mongo_uri)


# =========================
# 第 1 部分：檔案與寫入工具
# =========================
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json_array_streaming(docs: Iterable[Dict[str, Any]], out_path: Path) -> None:
    """
    以串流方式寫入「單一 JSON 陣列」檔案，避免一次吃進記憶體。
    """
    _ensure_parent_dir(out_path)
    with out_path.open("w", encoding="utf-8-sig") as f:
        f.write("[")
        first = True
        for doc in docs:
            if not first:
                f.write(",")
            first = False
            json.dump(doc, f, ensure_ascii=False)
        f.write("]")


# =========================
# 第 2 部分：模型載入與向量化
# =========================
def _resolve_snapshot(root: Path) -> Path:
    """找到包含 config.json 與模型權重的快照目錄"""
    root = root.expanduser()

    def _has_weights(p: Path) -> bool:
        candidates = [
            "pytorch_model.bin", "model.safetensors", "tf_model.h5",
            "model.ckpt.index", "flax_model.msgpack"
        ]
        return any((p / name).is_file() for name in candidates)

    # 直接 root
    if (root / "config.json").is_file() and _has_weights(root):
        return root

    # snapshots 下找（優先最新快照）
    snap = root / "snapshots"
    if snap.is_dir():
        cands = sorted(
            snap.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in cands:
            if (cand / "config.json").is_file() and _has_weights(cand):
                return cand

    raise FileNotFoundError(f"找不到有效模型快照於 {root}")


def load_embedder(model_root: Path, device: Optional[str] = None):
    """載入 CKIP-SBERT 模型並回傳 embedder（SentenceTransformer）"""
    from sentence_transformers import SentenceTransformer
    import torch
    resolved = _resolve_snapshot(model_root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 載入 CKIP-SBERT: {resolved} (device={device})", flush=True)
    return SentenceTransformer(str(resolved), device=device, trust_remote_code=True)


def embed_texts(emb_model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    批次嵌入，回傳單位向量（normalize_embeddings=True）
    """
    vecs = emb_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    # 轉為 float32 節省空間
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32, copy=False)
    return vecs


# =========================
# 第 3 部分：業務流程
# =========================
def export_mongo_entire_news(out_path: Path, force: bool = False) -> Tuple[int, Path]:
    """
    從 MongoDB 匯出所有新聞（News.Real_News）到單一 JSON 陣列檔案。
    ⚠️ 已加入 url 欄位，供後續輸出原始新聞用。
    """
    if out_path.exists() and not force:
        print(f"✅ 匯出略過（已存在）：{out_path}")
        return -1, out_path

    print("📦 從 MongoDB 匯出 News.Real_News ...")
    client = _require_mongo_client()
    coll = client["News"]["Real_News"]

    projection = {
        "_id": 1,
        "url": 1,
        "date": 1,
        "publisher": 1,
        "category": 1,
        "title": 1,
        "content": 1,
        "label": 1
    }
    cursor = coll.find({}, projection=projection, no_cursor_timeout=True)

    def _docs():
        count = 0
        for doc in cursor:
            if not isinstance(doc.get("_id"), str):
                doc["_id"] = str(doc["_id"])
            yield doc
            count += 1
        print(f"📄 匯出筆數：{count}")

    _write_json_array_streaming(_docs(), out_path)
    return 0, out_path


def load_entire_news(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_corpus_embeddings(
    news_json: Path,
    vector_out: Path,
    idmap_out: Path,
    model_root: Path,
    force: bool = False,
    title_weight: float = 1.0,
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    將 entire_news.json 向量化：
    - 向量 -> .npy
    - idmap -> jsonl（每行含 row/_id/title/date/publisher/category/label）
    回傳 (vectors, metas)
    """
    if vector_out.exists() and idmap_out.exists() and not force:
        print(f"✅ 向量化略過（已存在）：{vector_out} / {idmap_out}")
        vecs = np.load(vector_out, mmap_mode=None)
        metas = []
        with idmap_out.open("r", encoding="utf-8-sig") as f:
            for line in f:
                metas.append(json.loads(line))
        return vecs, metas

    print("🧮 構建新聞語料向量 ...")
    data = load_entire_news(news_json)
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for i, d in enumerate(_tqdm(data, desc="準備文本")):
        _id = d.get("_id")
        title = (d.get("title") or "").strip()
        content = (d.get("content") or "").strip()
        # 重要：用「標題 + 內文」組合做索引
        t = (title + "。" + content).strip() if content else title
        if title_weight > 1.0 and title:
            t = (title + "。") * int(title_weight) + content
        texts.append(t)
        metas.append({
            "row": i,
            "_id": _id,
            "title": title,
            "date": d.get("date"),
            "publisher": d.get("publisher"),
            "category": d.get("category"),
            "label": d.get("label"),
        })

    emb = load_embedder(model_root)
    vecs = embed_texts(emb, texts, batch_size=batch_size)

    _ensure_parent_dir(vector_out)
    np.save(vector_out, vecs)
    _ensure_parent_dir(idmap_out)
    with idmap_out.open("w", encoding="utf-8-sig") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    return vecs, metas


def _parse_10_queries(txt_path: Path) -> List[Tuple[int, str]]:
    """
    解析 10_news.txt：支援「1. 內容」「1) 內容」「1、內容」等常見編號格式。
    回傳 [(idx, text)]
    """
    lines = []
    with txt_path.open("r", encoding="utf-8-sig") as f:
        raw = f.read().splitlines()
    for line in raw:
        s = line.strip()
        if not s:
            continue
        # 去掉常見前綴編號
        s_clean = re.sub(r"^\s*\d+\s*[.)、．]\s*", "", s)
        s_clean = s_clean.strip()
        if s_clean:
            lines.append(s_clean)
    return list(enumerate(lines[:10], start=1))


# =========================
# ★ 關鍵字輔助檢索（獨立子查詢 + 分數融合）
# =========================
def _load_sentence_triples(path: Path) -> Optional[List[Dict[str, Any]]]:
    """
    嘗試讀取 10_sentence_extraction_result.json，失敗則回傳 None。
    """
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        triples = obj.get("triples")
        if isinstance(triples, list):
            return triples
        print("⚠️ triple 檔存在但 'triples' 欄位不是 list，忽略輔助檢索。")
        return None
    except FileNotFoundError:
        print(f"ℹ️ 找不到輔助關鍵字檔：{path}，將以原始查詢檢索。")
        return None
    except Exception as e:
        print(f"⚠️ 輔助關鍵字檔解析失敗：{e}，將以原始查詢檢索。")
        return None


def _group_subqueries_by_qindex(triples: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    依 triples 的 id（例如 Q3_T1）將**子查詢**歸到各自的查詢序號：
    產生短而聚焦的查詢：
      - subject
      - relation
      - subject + ' ' + relation（簡短合併）
      - scope（若 info_need.target.attributes.scope 存在）
      - relation + ' ' + scope（簡短合併）
    並去重、移除空字串。
    """
    q_pat = re.compile(r"^Q(?P<q>\d+)_", re.I)
    buckets: Dict[int, set] = defaultdict(set)

    for t in triples:
        tid = str(t.get("id") or "")
        m = q_pat.match(tid)
        if not m:
            continue
        q_idx = int(m.group("q"))

        subj = (((t.get("subject") or {}).get("text")) or "").strip()
        rel = (t.get("relation") or "").strip()

        scope = ""
        info_need = t.get("info_need") or {}
        target = info_need.get("target") or {}
        attrs = target.get("attributes") or {}
        if isinstance(attrs, dict):
            scope = str(attrs.get("scope") or "").strip()

        # 子查詢候選
        cand = []
        if subj:
            cand.append(subj)
        if rel:
            cand.append(rel)
        if subj and rel:
            cand.append(f"{subj} {rel}")
        if scope:
            cand.append(scope)
        if rel and scope:
            cand.append(f"{rel} {scope}")

        for q in cand:
            qn = q.strip()
            if qn:
                buckets[q_idx].add(qn)

    # 轉 list，穩定排序
    return {k: sorted(list(v)) for k, v in buckets.items()}


def _rrf_fusion(rank_lists: List[List[int]], N: int, k: int = 60) -> np.ndarray:
    """
    Reciprocal Rank Fusion（RRF）
    rank_lists: 每個子查詢對 corpus 的排名（索引序列）
    N: 語料大小
    k: 平滑常數（常用 60）
    產生每個 doc 的融合分數（越大越好）
    """
    scores = np.zeros(N, dtype=np.float32)
    for ranks in rank_lists:
        for r, doc_idx in enumerate(ranks, start=1):
            scores[doc_idx] += 1.0 / (k + r)
    return scores


def _aggregate_scores(
    per_query_scores: List[np.ndarray],
    method: str,
    per_query_indices: Optional[List[np.ndarray]] = None,
    rrf_k: int = 60,
) -> np.ndarray:
    """
    將多個子查詢的 score 向量做融合。
    method: rrf / max / mean / sum
    - rrf 需要 per_query_indices（每個子查詢的排序索引）
    """
    if not per_query_scores:
        raise ValueError("沒有子查詢分數可供融合。")

    method = method.lower()
    if method == "max":
        return np.max(np.stack(per_query_scores, axis=0), axis=0)
    elif method == "mean":
        return np.mean(np.stack(per_query_scores, axis=0), axis=0)
    elif method == "sum":
        return np.sum(np.stack(per_query_scores, axis=0), axis=0)
    elif method == "rrf":
        if per_query_indices is None:
            # 以每個子查詢自身分數決定排序
            per_query_indices = [np.argsort(-s) for s in per_query_scores]
        N = per_query_scores[0].shape[0]
        rank_lists = [idx.tolist() for idx in per_query_indices]
        return _rrf_fusion(rank_lists, N=N, k=rrf_k)
    else:
        raise ValueError(f"未知融合方法：{method}")


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    從 1D scores 取 top-k 索引（k 超過長度會自動裁切）
    """
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty((0,), dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def retrieve_for_queries(
    queries_path: Path,
    emb_model,
    corpus_vecs: np.ndarray,
    metas: List[Dict[str, Any]],
    topk: int = 10,
    min_score: float = 0.0,
    assist_keywords: bool = True,
    assist_path: Optional[Path] = None,
    fusion: str = "rrf",
    rrf_k: int = 60,
    per_sub_topk: Optional[int] = None,
    include_original_query: bool = True,
) -> List[Dict[str, Any]]:
    """
    主檢索：
    - 先把 Q1..Q10 句子解析出來
    - 若 assist_keywords=True 且有有效 triples，生成「每個 Qn 的子查詢集合」
    - 對「原句（可選）」與「子查詢們」各自做檢索，再做分數融合（預設 RRF）
    - 回傳每個 Qn 的融合排名結果
    """
    pairs = _parse_10_queries(queries_path)  # [(q_idx, q_text), ...]
    qidx2subqueries: Dict[int, List[str]] = {}

    if assist_keywords and assist_path is not None:
        triples = _load_sentence_triples(assist_path)
        if triples:
            qidx2subqueries = _group_subqueries_by_qindex(triples)
            covered = len(qidx2subqueries)
            total_kw = sum(len(v) for v in qidx2subqueries.values())
            print(f"🔎 已載入子查詢：{total_kw} 條（覆蓋 {covered}/{len(pairs)} 個查詢）")
        else:
            print("ℹ️ 關鍵字輔助檢索停用（檔案缺失或格式不符）。")
    else:
        print("ℹ️ 關鍵字輔助檢索停用。")

    results: List[Dict[str, Any]] = []
    N = corpus_vecs.shape[0]
    per_sub_topk = per_sub_topk or max(topk, 50)  # 每個子查詢取較深的候選以利融合

    for q_idx, q_text in _tqdm(pairs, desc="檢索中"):
        # === 構建本次要跑的查詢集合 ===
        subqueries = []
        if include_original_query:
            subqueries.append(q_text)  # 原句作為一個子查詢
        subqueries.extend(qidx2subqueries.get(q_idx, []))

        # 若沒有任何子查詢，就退回原本流程（只用原句）
        if not subqueries:
            subqueries = [q_text]

        # === 批量編碼所有子查詢 ===
        q_vecs = embed_texts(emb_model, subqueries, batch_size=64)  # (M, D)
        # === 與語料計算相似度 ===
        # corpus_vecs: (N, D)，q_vecs: (M, D) -> sim: (M, N)
        sim = q_vecs @ corpus_vecs.T

        # 針對每個子查詢，先各自取 topK 候選，避免為融合而全量排序
        per_scores: List[np.ndarray] = []
        per_indices_for_rrf: List[np.ndarray] = []
        candidate_mask = np.zeros(N, dtype=bool)

        for i in range(sim.shape[0]):
            s = sim[i]
            idx = _topk_indices(s, per_sub_topk)  # (K,)
            per_indices_for_rrf.append(idx)
            # 建一個稀疏全長 score：只在 topK 位置保留分數，其餘 0
            tmp = np.zeros(N, dtype=np.float32)
            tmp[idx] = s[idx]
            per_scores.append(tmp)
            candidate_mask[idx] = True

        # === 分數融合 ===
        fused = _aggregate_scores(
            per_query_scores=[s for s in per_scores],
            method=fusion,
            per_query_indices=per_indices_for_rrf,
            rrf_k=rrf_k,
        )

        # 僅在候選集合內取最終 topk，可大幅減少排序量
        cand_indices = np.where(candidate_mask)[0]
        cand_scores = fused[cand_indices]
        if cand_indices.size == 0:
            # 極端情況：沒有候選（理論上不會），fallback 全量
            cand_indices = np.arange(N)
            cand_scores = fused

        # 在候選中挑 topk
        k = min(topk, cand_indices.size)
        top_local = np.argpartition(-cand_scores, k - 1)[:k]
        top_order = top_local[np.argsort(-cand_scores[top_local])]
        final_indices = cand_indices[top_order]
        final_scores = fused[final_indices]

        # 過濾 min_score
        items = []
        for rank, (idx_doc, sc) in enumerate(zip(final_indices, final_scores), start=1):
            if sc < min_score:
                continue
            m = metas[idx_doc]
            items.append({
                "rank": rank,
                "score": float(round(sc, 6)),
                "_id": m["_id"],
                "title": m["title"],
                "date": m["date"],
                "publisher": m["publisher"],
                "category": m["category"],
                "label": m["label"],
            })

        results.append({
            "query_index": q_idx,
            "query_text": q_text,
            "subqueries": subqueries,
            "fusion": fusion,
            "matches": items,
        })

    return results


def save_retrieval_output(results: List[Dict[str, Any]], out_path: Path) -> None:
    _ensure_parent_dir(out_path)
    with out_path.open("w", encoding="utf-8-sig") as f:
        json.dump({
            "spec": {
                "note": "與 experiment/data/raw/10_news.txt 的逐條相似檢索結果；每條 query 使用 subqueries 與分數融合（fusion）產生。",
                "similarity": "cosine (內積 on unit vectors)",
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)


# =========================
# 第 4 部分：原文對照輸出
# =========================
def _build_id_to_doc_map(entire_news_json: Path) -> Dict[str, Dict[str, Any]]:
    """
    將 entire_news.json 轉為 { _id: doc } 映射，供快速取用原始 MongoDB 文章。
    """
    data = load_entire_news(entire_news_json)
    id2doc: Dict[str, Dict[str, Any]] = {}
    for d in data:
        _id = str(d.get("_id"))
        id2doc[_id] = d
    return id2doc


def enrich_with_full_docs(
    results: List[Dict[str, Any]],
    id2doc: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    將檢索結果 matches 以 _id 對照回原始 MongoDB 文章，輸出 matches_full。
    """
    enriched: List[Dict[str, Any]] = []
    for block in results:
        full_list: List[Dict[str, Any]] = []
        for m in block.get("matches", []):
            _id = str(m["_id"])
            d = id2doc.get(_id)
            if not d:
                continue
            full_list.append({
                "_id": d.get("_id"),
                "url": d.get("url"),
                "date": d.get("date"),
                "publisher": d.get("publisher"),
                "category": d.get("category"),
                "title": d.get("title"),
                "content": d.get("content"),
                "label": d.get("label"),
            })
        enriched.append({
            "query_index": block["query_index"],
            "query_text": block["query_text"],
            "subqueries": block.get("subqueries", []),
            "fusion": block.get("fusion"),
            "matches_full": full_list
        })
    return enriched


def save_full_docs_output(enriched: List[Dict[str, Any]], out_path: Path) -> None:
    """
    以每個查詢一個區塊，包含 matches_full（每項為完整新聞原文）。
    """
    _ensure_parent_dir(out_path)
    with out_path.open("w", encoding="utf-8-sig") as f:
        json.dump({
            "spec": {
                "note": "與 experiment/data/raw/10_news.txt 的逐條相似檢索結果（原始新聞全文）；以子查詢 + 融合生成。",
                "source": "experiment/data/raw/entire_news.json",
            },
            "results": enriched
        }, f, ensure_ascii=False, indent=2)


# =========================
# CLI
# =========================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="News 10-Query Retrieval Pipeline (CKIP BERT)")
    p.add_argument("--model-root", type=str,
                   default="models/CKIP/models--ckiplab--bert-base-chinese",
                   help="ckiplab/bert-base-chinese 的本機根目錄（會自動往 snapshots/ 下找快照）")
    p.add_argument("--topk", type=int, default=10, help="每條查詢返回的新聞數量")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="相似度下限（0~1 間，可考慮 0.4~0.6）")
    p.add_argument("--force", action="store_true", help="重建所有輸出（若已存在則覆蓋）")
    p.add_argument("--skip-export", action="store_true",
                   help="略過 Mongo 匯出（直接用已存在的 entire_news.json）")
    p.add_argument("--batch-size", type=int, default=64, help="向量化 batch size")
    p.add_argument("--title-weight", type=float, default=1.0,
                   help="標題加權（>1 會把標題重複一次簡單加權）")

    # 關鍵字輔助檢索（子查詢）
    p.add_argument("--assist-keywords", action="store_true",
                   help="啟用由 10_sentence_extraction_result.json 生成的子查詢輔助檢索（預設啟用）")
    p.add_argument("--no-assist-keywords", dest="assist_keywords", action="store_false",
                   help="停用子查詢輔助檢索")
    p.set_defaults(assist_keywords=True)
    p.add_argument("--assist-path", type=str,
                   default="experiment/data/interim/10_sentence_extraction_result.json",
                   help="關鍵字輔助檢索 JSON 路徑（含 triples 陣列）")

    # 融合策略
    p.add_argument("--fusion", type=str, default="rrf",
                   choices=["rrf", "max", "mean", "sum"],
                   help="多子查詢分數融合策略（預設 rrf）")
    p.add_argument("--rrf-k", type=int, default=60,
                   help="RRF 的 k 平滑常數（越大越保守，預設 60）")
    p.add_argument("--per-sub-topk", type=int, default=None,
                   help="每個子查詢各自先取的候選深度（預設 max(topk, 50)）")
    p.add_argument("--no-original", dest="include_original", action="store_false",
                   help="不把原始句子納入子查詢集合")
    p.set_defaults(include_original=True)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    # --- 路徑佈局 ---
    RAW_JSON = Path("experiment/data/raw/entire_news.json")
    QUERIES_TXT = Path("experiment/data/raw/10_news.txt")
    VEC_NPY = Path("experiment/data/interim/entire_news_vector.npy")
    IDMAP = Path("experiment/data/interim/entire_news_idmap.jsonl")

    # ✅ 調整：最終輸出到 processed/
    OUTPUT_SCORED = Path("experiment/data/processed/related_10_news.json")
    OUTPUT_FULL = Path("experiment/data/processed/related_news.json")

    ASSIST_PATH = Path(args.assist_path) if args.assist_path else None

    # 1) 匯出 Mongo -> entire_news.json
    if not args.skip_export:
        export_mongo_entire_news(RAW_JSON, force=args.force)
    else:
        print("⏭️  略過匯出：使用現有 entire_news.json")

    # 2) 向量化（如已存在且未 --force 則略過）
    vecs, metas = build_corpus_embeddings(
        news_json=RAW_JSON,
        vector_out=VEC_NPY,
        idmap_out=IDMAP,
        model_root=Path(args.model_root),
        force=args.force,
        title_weight=args.title_weight,
        batch_size=args.batch_size,
    )

    # 3) 讀取 10 條查詢並檢索（子查詢 + 融合）
    if not QUERIES_TXT.exists():
        raise FileNotFoundError(f"找不到查詢檔：{QUERIES_TXT}")
    emb = load_embedder(Path(args.model_root))  # 重用同一模型
    results = retrieve_for_queries(
        queries_path=QUERIES_TXT,
        emb_model=emb,
        corpus_vecs=vecs,
        metas=metas,
        topk=args.topk,
        min_score=args.min_score,
        assist_keywords=args.assist_keywords,
        assist_path=ASSIST_PATH,
        fusion=args.fusion,
        rrf_k=args.rrf_k,
        per_sub_topk=args.per_sub_topk,
        include_original_query=args.include_original,
    )

    # 4) 輸出「排名/分數」結果 JSON（processed）
    save_retrieval_output(results, OUTPUT_SCORED)
    print(f"✅ 完成排名/分數：{OUTPUT_SCORED}")

    # 5) 對照原始新聞並輸出（processed）
    id2doc = _build_id_to_doc_map(RAW_JSON)
    enriched = enrich_with_full_docs(results, id2doc)
    save_full_docs_output(enriched, OUTPUT_FULL)
    print(f"✅ 完成原文輸出：{OUTPUT_FULL}")


if __name__ == "__main__":
    main()
