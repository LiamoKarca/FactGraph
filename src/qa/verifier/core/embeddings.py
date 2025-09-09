"""
CKIP SBERT 載入與文字向量化
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .paths import CKIP_ROOT


def _resolve_snapshot(root: Path) -> Path:
    if (root / 'config.json').is_file():
        return root
    for sub in (root / 'snapshots').iterdir():
        if (sub / 'config.json').is_file():
            return sub
    else:
        raise FileNotFoundError('CKIP Sentence-BERT snapshot not found')


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    model_path = _resolve_snapshot(CKIP_ROOT)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SentenceTransformer(str(model_path), device=device, trust_remote_code=True)


def embed_text(text: str) -> np.ndarray:
    emb = get_embedder().encode(text, convert_to_numpy=True, show_progress_bar=False)
    return emb / np.linalg.norm(emb)


def embed_triple(tp: dict[str, str]) -> np.ndarray:
    return embed_text(f"{tp['head']} {tp['relation']} {tp['tail']}")
