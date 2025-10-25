"""
Reciprocal Rank Fusion（RRF）穩健融合（支援來源權重與簡單歸一化）
公式：RRF(item) = Σ_src w_src * 1/(k + rank_src(item))
- k：典型 60
- rank：來源內（從 1 起）
- w_src：來源權重（預設 1.0）

回傳：按 rrf_score 由高到低排序的列表，每一項：
{ "item": RankItem, "rrf_score": float, "per_source": {"pg":score,...} }

參考：RRF 原始論文（SIGIR 2009），與後續教科書式介紹。:contentReference[oaicite:2]{index=2}
"""
from __future__ import annotations
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

def rrf_fuse(
    runs: Dict[str, List[dict]],
    k: int = 60,
    weights: Dict[str, float] | None = None
) -> List[dict]:
    weights = weights or {}
    pool: Dict[str, dict] = {}
    for src, items in (runs or {}).items():
        if not items:
            continue
        w = float(weights.get(src, 1.0))
        for it in items:
            rid = it["id"]
            rank = max(1, int(it.get("rank", 999999)))
            inc = w * (1.0 / (k + rank))
            obj = pool.setdefault(rid, {"item": it, "rrf_score": 0.0, "per_source": {}})
            obj["rrf_score"] += inc
            obj["per_source"][src] = obj["per_source"].get(src, 0.0) + inc

    out = list(pool.values())
    # 小範圍歸一化到 [0,1]，便於與 CE 機率做線性融合
    if out:
        mx = max(x["rrf_score"] for x in out)
        if mx > 0:
            for x in out:
                x["rrf_score"] = x["rrf_score"] / mx
    out.sort(key=lambda x: x["rrf_score"], reverse=True)
    return out
