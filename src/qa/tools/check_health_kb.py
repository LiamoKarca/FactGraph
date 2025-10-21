"""
KB/向量/PG 一鍵健康檢查
- 檢查 ST 嵌入器、KG 載入情形與維度
- 單條三元組跑 cosine_search 乾測
- 檢查 PG 快取檔與 query 乾測
執行：
  (venv) python -m src.qa.tools.check_health_kb
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

# 1) 嵌入器自檢
from src.qa.verifier.core.embeddings import get_embedder, embed_triple  # 路徑依專案實際位置
# 2) KG 載入與檢索
from src.qa.verifier.kg.search import cosine_search, kg_row_to_detail  # 需可正常 import
# 3) PG 檢索器
from src.qa.tools.property_graph.li_csv_pg_retriever import CsvPropertyGraphRetriever


def _ok(flag: bool) -> str:
    return "OK" if flag else "FAIL"


def main() -> None:
    print("=== [1] Sentence-Transformers 自檢 ===")
    model_name = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "")
    print(f"MODEL={model_name or '(CKIP動態轉檔)'}")
    emb = get_embedder()
    print(f" ├─ embedder: {type(emb)}")
    v = emb.encode("測試向量", convert_to_numpy=True)
    print(f" └─ dim={v.shape} dtype={v.dtype}")

    print("\n=== [2] KG 載入與 cosine_search 乾測 ===")
    tp = {"head": "黃國昌", "relation": "涉及", "tail": "偷拍"}
    try:
        q = embed_triple(tp)
        idxs = cosine_search(tp, q)[:3]
        print(f" ├─ top3 idx: {idxs}")
        for i in idxs:
            tri, det = kg_row_to_detail(i)
            print(" └─", tri, det.get("rel", {}) or det)
    except Exception as e:
        print(" └─ cosine_search ERROR:", repr(e))

    print("\n=== [3] PG 快取與檢索 乾測 ===")
    pkl = os.getenv("LI_PG_INDEX_PKL", "data/processed/knowledge-graph/pg_index.pkl")
    jsn = os.getenv("LI_PG_INDEX_JSON", "data/processed/knowledge-graph/pg_index.json")
    print(f" 檔案存在: pkl={Path(pkl).exists()} json={Path(jsn).exists()}")
    try:
        retr = CsvPropertyGraphRetriever.load_from_cache(idx_pkl=pkl, idx_json=jsn)
        hits = retr.search_triple(tp, top_k=5, hops=int(os.getenv("LI_PG_HOPS", "2")))
        print(f" └─ PG hits={len(hits)}; 範例={hits[:1]}")
    except Exception as e:
        print(" └─ PG ERROR:", repr(e))


if __name__ == "__main__":
    main()
