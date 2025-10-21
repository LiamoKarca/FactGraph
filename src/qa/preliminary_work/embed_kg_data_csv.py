# -*- coding: utf-8-sig -*-
"""
將提取出的原始知識 CSV（三元組 head/relation/tail）轉成向量檔 .npy
（MiniLM L12 v2 / Sentence-Transformers，本機快取載入）

執行：
    (venv) python -m src.qa.preliminary_work.embed_kg_data_csv

輸入：
    data/raw/knowledge-graph/neo4j-kg-raw-graph.csv

輸出：
    data/processed/knowledge-graph/neo4j-kg.emb.npy

說明：
    - 模型固定走本機快取：
      models/MiniLM/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots
    - 若 MODEL_ROOT 指向 snapshots 根目錄，會自動挑選其下一層含 config.json 的快照。
    - SentenceTransformer 支援以本機資料夾載入模型；快取與 snapshots 結構由
      huggingface_hub 管理。參見官方快取與 CLI 文件。  # docs: hub cache + snapshots + scan-cache
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

# 參考：HF Hub 快取與 snapshots 結構／scan-cache 用法。  # :contentReference[oaicite:1]{index=1}

TQDM_CFG = dict(
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ncols=80,
)

CSV_PATH = "data/raw/knowledge-graph/neo4j-kg-raw-graph.csv"
MODEL_ROOT = Path(
    "models/MiniLM/models--sentence-transformers--"
    "paraphrase-multilingual-MiniLM-L12-v2/snapshots"
)
OUT_NPY = "data/processed/knowledge-graph/neo4j-kg.emb.npy"

WINDOW, STRIDE = 510, 256
BATCH_SRC, BATCH_ENC = 32, 16
INCLUDE_PROPS = False  # 改 True 可把 *_props 也併入嵌入文字


def resolve_snapshot(root: Path) -> str:
    """解析 Hugging Face 本機快取路徑，回傳可直接載入的 snapshot 目錄。

    支援三種輸入：
        1) 直接是某個 `<commit>` 目錄（含 config.json）
        2) 指向 snapshots 根目錄：.../models--ORG--REPO/snapshots
        3) 指向 repo 根：.../models--ORG--REPO  → 轉去其 snapshots/*

    Args:
        root: 模型根或 snapshots 目錄。

    Returns:
        str: SentenceTransformer 可直接載入的目錄字串。

    Raises:
        FileNotFoundError: 找不到任何含 config.json 的 snapshot 目錄。
    """
    # case 1: 已經是具體快照
    if (root / "config.json").is_file():
        return str(root)

    # case 2: 傳入 snapshots 根目錄；直接在其子目錄中尋找
    if root.name == "snapshots" and root.is_dir():
        for sub in sorted(root.iterdir()):
            if (sub / "config.json").is_file():
                return str(sub)

    # case 3: 傳入的是 repo 根目錄 → 轉去 repo/snapshots/*
    snaps = root / "snapshots"
    if snaps.is_dir():
        for sub in sorted(snaps.iterdir()):
            if (sub / "config.json").is_file():
                return str(sub)

    raise FileNotFoundError(f"❌ 找不到模型權重於 {root}")


def split_windows(tokenizer, text: str, window: int, stride: int) -> List[str]:
    """以 tokenizer 進行 sliding-window 切片。

    Args:
        tokenizer: HF tokenizer。
        text: 原始文本。
        window: 視窗 token 數。
        stride: 位移 token 數。

    Returns:
        List[str]: 切片文字列表。
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= window:
        return [text]
    chunks, pos = [], 0
    while pos < len(ids):
        chunk_ids = ids[pos : pos + window]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if pos + window >= len(ids):
            break
        pos += stride
    return chunks


def main() -> None:
    """主程式入口。"""
    # 解析 snapshot 目錄
    model_path = resolve_snapshot(MODEL_ROOT)
    print(f"[Model] resolved → {model_path}")

    # 載入模型與 tokenizer（本機路徑）
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dim = int(model.get_sentence_embedding_dimension())
    print(f"[Model] dim={dim} on {device}  ({time.time() - t0:.1f}s)")

    # 讀 CSV
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
    assert {"head", "relation", "tail"}.issubset(df.columns), "CSV schema error"

    if INCLUDE_PROPS:
        sentences = (
            df["head"].fillna("")
            + " "
            + df["relation"].fillna("")
            + " "
            + df["tail"].fillna("")
            + " "
            + df.get("head_props", "").fillna("")
            + " "
            + df.get("rel_props", "").fillna("")
            + " "
            + df.get("tail_props", "").fillna("")
        ).tolist()
    else:
        sentences = (
            df["head"].fillna("") + " " + df["relation"].fillna("") + " " + df["tail"].fillna("")
        ).tolist()

    print(f"[Data] 三元組 {len(sentences):,} 條")

    # 逐批 encode（sliding-window → mean pooling）
    embs = np.zeros((len(sentences), dim), dtype=np.float32)
    pbar = tqdm(range(0, len(sentences), BATCH_SRC), desc="Embedding", **TQDM_CFG)

    for i in pbar:
        batch_texts = sentences[i : i + BATCH_SRC]
        chunk_lists = [split_windows(tokenizer, t, WINDOW, STRIDE) for t in batch_texts]

        flat_texts = [c for lst in chunk_lists for c in lst]
        flat_embs = model.encode(
            flat_texts,
            batch_size=BATCH_ENC,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        idx = 0
        for j, chunks in enumerate(chunk_lists):
            n = len(chunks)
            embs[i + j] = flat_embs[idx : idx + n].mean(0)
            idx += n

    Path(OUT_NPY).parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, embs)
    print(f"[Save] {OUT_NPY}  shape={embs.shape}")


if __name__ == "__main__":
    main()
