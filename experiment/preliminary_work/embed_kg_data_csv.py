"""
Neo4j KG CSV → 向量化（CKIP-BERT, sliding-window + mean pooling）
輸入：experiment/data/raw/neo4j-kg-raw-graph.csv
輸出：experiment/data/processed/neo4j-kg.emb.npy

用法（預設即可）：
    python experiment/preliminary_work/embed_kg_data_csv.py

可選參數：
    --csv experiment/data/raw/neo4j-kg-raw-graph.csv
    --model-root models/CKIP/models--ckiplab--bert-base-chinese
    --out-npy experiment/data/processed/neo4j-kg.emb.npy
    --include-props            # 將三端屬性文本併入編碼
    --window 510 --stride 256  # 長文切窗
    --batch-src 32 --batch-enc 16
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# 進度條樣式
tqdm_cfg: Dict[str, Union[str, int]] = dict(
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ncols=80
)


def resolve_snapshot(root: Path) -> str:
    """解析 HuggingFace snapshot 目錄（相容單層或 snapshots/* 結構）"""
    if (root / "config.json").is_file():
        return str(root)
    snaps = root / "snapshots"
    if snaps.is_dir():
        for sub in snaps.iterdir():
            if (sub / "config.json").is_file():
                return str(sub)
    raise FileNotFoundError(f"❌ 找不到模型權重於 {root}")


def parse_prop(obj: Any) -> str:
    """將 props 欄位統一轉為純文字（JSON 序列化），避免 'nan' 或不規則型別"""
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return ""
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return ""
        # 優先嘗試把像 dict 的字串還原，再序列化，降低雜訊
        try:
            val = ast.literal_eval(s)
            if isinstance(val, dict):
                return json.dumps(val, ensure_ascii=False)
        except Exception:
            pass
        return s
    # 其餘型別（pandas 可能直接給 dict）
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def split_windows(text: str, tokenizer: PreTrainedTokenizerBase, window: int, stride: int) -> List[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= window:
        return [text]
    chunks: List[str] = []
    pos = 0
    while pos < len(ids):
        chunk_ids = ids[pos: pos + window]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if pos + window >= len(ids):
            break
        pos += stride
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", default="experiment/data/raw/neo4j-kg-raw-graph.csv")
    ap.add_argument(
        "--model-root", default="models/CKIP/models--ckiplab--bert-base-chinese")
    ap.add_argument(
        "--out-npy", default="experiment/data/processed/neo4j-kg.emb.npy")
    ap.add_argument("--include-props", action="store_true")
    ap.add_argument("--window", type=int, default=510)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--batch-src", type=int, default=32)
    ap.add_argument("--batch-enc", type=int, default=16)
    args = ap.parse_args()

    # 1) 模型
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = resolve_snapshot(Path(args.model_root))
    print(f"[Model] resolved → {model_path}")
    model = SentenceTransformer(
        model_path, device=device, trust_remote_code=True)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_path)
    dim = model.get_sentence_embedding_dimension()
    print(f"[Model] {dim}-d on {device}  ({time.time() - t0:.1f}s)")

    # 2) 讀 CSV + 組合語料
    df = pd.read_csv(args.csv, low_memory=False)
    required = {"head", "relation", "tail"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"CSV schema error：缺 {missing}")
    print(f"[Data] 三元組 {len(df):,} 條  file={args.csv}")

    if args.include_props:
        # 這裡故意使用 hp/rp/tp 三個變數，避免未定義的 props 變數。
        hp = df.get("head_props", "").map(parse_prop)
        rp = df.get("rel_props", "").map(parse_prop)
        tp = df.get("tail_props", "").map(parse_prop)
        sentences = (
            df["head"].astype(str) + " " +
            df["relation"].astype(str) + " " +
            df["tail"].astype(str) + " " +
            hp.astype(str) + " " +
            rp.astype(str) + " " +
            tp.astype(str)
        ).tolist()
    else:
        sentences = (
            df["head"].astype(str) + " " +
            df["relation"].astype(str) + " " +
            df["tail"].astype(str)
        ).tolist()

    # 3) 向量化（sliding-window + mean pooling）
    embs = np.zeros((len(sentences), dim), dtype=np.float32)
    pbar = tqdm(range(0, len(sentences), args.batch_src),
                desc="Embedding", **tqdm_cfg)
    for i in pbar:
        batch_texts = sentences[i:i + args.batch_src]
        chunk_lists = [split_windows(
            t, tokenizer, args.window, args.stride) for t in batch_texts]

        flat_texts = [c for lst in chunk_lists for c in lst]
        # 確保輸出是 float32 numpy
        flat_embs = model.encode(
            flat_texts,
            batch_size=args.batch_enc,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,   # 我們這裡先不歸一，保留原始句向量
        ).astype(np.float32, copy=False)

        idx = 0
        for j, chunks in enumerate(chunk_lists):
            n = len(chunks)
            # mean pooling
            embs[i + j] = flat_embs[idx:idx + n].mean(0)
            idx += n

        pbar.set_postfix(
            done=f"{min(i + args.batch_src, len(sentences))}/{len(sentences)}")

    # 4) 儲存
    out_path = Path(args.out_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embs)
    print(f"[Save] {out_path}  shape={embs.shape}")


if __name__ == "__main__":
    main()
