"""
BGE-RERANKER-V2-M3 模型下載與快速測試腳本

此腳本會：
1) 從 Hugging Face 下載 BAAI/bge-reranker-v2-m3（交叉編碼器／重排器）
2) 將權重快取於 models/BGE_RERANKER_V2_M3
3) 進行極小測試：對「查詢 × 文件」配對產生相關性分數

執行方式：
    (venv) python models/download_BGE_RERANKER_V2_M3.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


@dataclass(frozen=True)
class ModelSpec:
    """模型規格與路徑設定。"""

    hf_repo: str = "BAAI/bge-reranker-v2-m3"
    cache_dir: str = "models/BGE_RERANKER_V2_M3"


def _ensure_dirs(path: str) -> None:
    """確保路徑存在。

    Args:
        path: 目標資料夾路徑。
    """
    os.makedirs(path, exist_ok=True)


def _load_model(spec: ModelSpec) -> tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    """載入 tokenizer 與 model，並回傳推論裝置。

    Args:
        spec: 模型規格。

    Returns:
        (tokenizer, model, device) 三元組。
    """
    _ensure_dirs(spec.cache_dir)

    # 自動選擇裝置（有 GPU 則用 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_repo,
        cache_dir=spec.cache_dir,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_repo,
        cache_dir=spec.cache_dir,
        trust_remote_code=True,
    ).to(device)
    elapsed = time.time() - t0
    print(f"✅ 模型與 tokenizer 載入完成（耗時 {elapsed:.2f} 秒, device={device}）")
    return tokenizer, model, device


def _score_pairs(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    pairs: List[Tuple[str, str]],
) -> List[float]:
    """對 Query-Document 配對進行打分。

    Args:
        tokenizer: 分詞器。
        model: 交叉編碼器模型本體。
        device: 推論裝置。
        pairs: (query, document) 配對列表。

    Returns:
        對應每個配對的分數（越高代表越相關）。
    """
    texts_a = [q for q, _ in pairs]
    texts_b = [d for _, d in pairs]

    inputs = tokenizer(
        texts_a,
        texts_b,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1)  # 形狀：[batch]
    scores = logits.detach().float().cpu().tolist()
    return scores


def _quick_demo(tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification, device: torch.device) -> None:
    """以繁中語境進行快速測試。"""
    query = "黃國昌是否指揮『政治狗仔』團隊進行蒐證？"
    docs = [
        "鏡報於 2025 年 9 月 26 日報導，黃國昌被指涉嫌指揮成員偷拍與蒐集政治人物行蹤。",
        "台北股市今日小漲，電子權值股表現穩健。",
        "中央社啟動內部調查，以釐清是否有記者涉入相關事件。",
    ]
    pairs = [(query, d) for d in docs]
    scores = _score_pairs(tokenizer, model, device, pairs)

    print("🔎 測試分數（越高越相關）：")
    for i, (doc, sc) in enumerate(zip(docs, scores), start=1):
        print(f"  {i:02d}. score={sc:.4f} | {doc[:48]}{'...' if len(doc) > 48 else ''}")


def main() -> None:
    """主程式入口。"""
    print("🚀 下載／載入 BGE-RERANKER-V2-M3 到 models/BGE_RERANKER_V2_M3")
    spec = ModelSpec()
    tokenizer, model, device = _load_model(spec)
    _quick_demo(tokenizer, model, device)
    print("🎉 完成。模型快取位於：models/BGE_RERANKER_V2_M3")


if __name__ == "__main__":
    main()
