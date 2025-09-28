"""
$ python -m src.qa.preliminary_work.embed_kg_data_csv
將提取出的原始知識 data/raw/kg-raw-graph.csv（三元組格式）
轉成向量檔 data/processed/knowledge-graph/neo4j-kg.emb.npy
"""
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

# ─── 0. tqdm 全域格式 ─────────────────────────────────────────
tqdm_cfg = dict(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]",
                ncols=80)

# ─── 1. 路徑與參數 ───────────────────────────────────────────
CSV_PATH = "data/raw/knowledge-graph/neo4j-kg-raw-graph.csv"
MODEL_ROOT = Path("models/CKIP/models--ckiplab--bert-base-chinese")
OUT_NPY = "data/processed/knowledge-graph/neo4j-kg.emb.npy"

WINDOW, STRIDE = 510, 256  # sliding window
BATCH_SRC, BATCH_ENC = 32, 16  # 來源句數 / encode 批次


# ─── 2. 解析 HuggingFace snapshot 路徑 ───────────────────────
def resolve_snapshot(root: Path) -> str:
    if (root / "config.json").is_file():
        return str(root)
    snaps = root / "snapshots"
    for sub in snaps.iterdir():
        if (sub / "config.json").is_file():
            return str(sub)
    raise FileNotFoundError(f"❌ 找不到模型權重於 {root}")


MODEL_PATH = resolve_snapshot(MODEL_ROOT)
print(f"[Model] resolved → {MODEL_PATH}")

# ─── 3. 載入模型 ─────────────────────────────────────────────
t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_PATH, device=device, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
DIM = model.get_sentence_embedding_dimension()
print(f"[Model] {DIM}-d on {device}  ({time.time() - t0:.1f}s)")

# ─── 4. 讀 CSV（確保有 head / relation / tail）──────────────
df = pd.read_csv(CSV_PATH, low_memory=False)
assert {"head", "relation", "tail"}.issubset(df.columns), "CSV schema error"

# 若要把屬性一起編碼，把下方 False 改 True
INCLUDE_PROPS = False

if INCLUDE_PROPS:
    sentences = (df["head"] + " " + df["relation"] + " " + df["tail"] +
                 " " + df.get("head_props", "").fillna("") +
                 " " + df.get("rel_props", "").fillna("") +
                 " " + df.get("tail_props", "").fillna("")).tolist()
else:
    sentences = (df["head"] + " " + df["relation"] + " " + df["tail"]).tolist()

print(f"[Data] 三元組 {len(sentences):,} 條")


# ─── 5. sliding-window 分片 ─────────────────────────────────
def split_windows(text: str) -> List[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= WINDOW:
        return [text]
    chunks, pos = [], 0
    while pos < len(ids):
        chunk_ids = ids[pos: pos + WINDOW]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if pos + WINDOW >= len(ids):
            break
        pos += STRIDE
    return chunks


# ─── 6. 產生向量 ────────────────────────────────────────────
embs = np.zeros((len(sentences), DIM), dtype=np.float32)

pbar = tqdm(range(0, len(sentences), BATCH_SRC),
            desc="Embedding", **tqdm_cfg)
for i in pbar:
    batch_texts = sentences[i:i + BATCH_SRC]
    chunk_lists = [split_windows(t) for t in batch_texts]

    flat_texts = [c for lst in chunk_lists for c in lst]
    flat_embs = model.encode(flat_texts,
                             batch_size=BATCH_ENC,
                             convert_to_numpy=True,
                             show_progress_bar=False)

    idx = 0
    for j, chunks in enumerate(chunk_lists):
        n = len(chunks)
        embs[i + j] = flat_embs[idx:idx + n].mean(0)
        idx += n
    pbar.set_postfix(done=f"{min(i + BATCH_SRC, len(sentences))}/{len(sentences)}")

# ─── 7. 儲存 ───────────────────────────────────────────────
Path(OUT_NPY).parent.mkdir(parents=True, exist_ok=True)
np.save(OUT_NPY, embs)
print(f"[Save] {OUT_NPY}  shape={embs.shape}")
