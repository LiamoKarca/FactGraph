"""
Embedding helpers (CKIP-SBERT) with dedupe functionality.

提供：
 - load_embedder: 載入 SentenceTransformer 模型
 - embed_text: 將文字轉為單位向量
 - embed_triple: 將三元組轉為文字後嵌入
 - dedupe: 以實體前綴分組，保留語義最長，並重編號
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Callable

import numpy as np
from sentence_transformers import SentenceTransformer, util


# Regex patterns
ENTITY_PATTERN = re.compile(r"^\d+\.\s*([^\s（]+)")
NUMBERING_PATTERN = re.compile(r"^(?:\[\d+\]\.|\d+\.)\s*")


def _resolve_snapshot(root: Path) -> Path:
    """找到包含 config.json 與模型權重的快照目錄"""
    root = root.expanduser()
    if (root / 'config.json').is_file() and any(
        (root / ext).is_file() for ext in [
            'pytorch_model.bin', 'model.safetensors', 'tf_model.h5',
            'model.ckpt.index', 'flax_model.msgpack'
        ]
    ):
        return root

    snapshots = root / 'snapshots'
    if snapshots.is_dir():
        for cand in snapshots.iterdir():
            if (cand / 'config.json').is_file() and any(
                (cand / ext).is_file() for ext in [
                    'pytorch_model.bin', 'model.safetensors', 'tf_model.h5',
                    'model.ckpt.index', 'flax_model.msgpack'
                ]
            ):
                return cand

    raise FileNotFoundError(f'找不到有效模型快照於 {root}')


def load_embedder(model_root: Path, device: str | None = None) -> SentenceTransformer:
    """載入 CKIP-SBERT 模型並回傳 embedder"""
    resolved = _resolve_snapshot(model_root)
    import torch
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🔧 載入 CKIP-SBERT: {resolved} (device={device})', flush=True)
    return SentenceTransformer(str(resolved), device=device, trust_remote_code=True)


def embed_text(emb: SentenceTransformer, text: str) -> np.ndarray:
    """將文字嵌入並回傳單位向量（供原文檢索 search_by_texts 使用）"""
    vec = emb.encode(text, convert_to_numpy=True, show_progress_bar=False)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def embed_triple(emb: SentenceTransformer, tp: dict[str, str]) -> np.ndarray:
    """
    將三元組字典拼接後嵌入（舊版用法）
    ⚠️ 注意：若 tail 常為「未知」，會污染語意。新版建議改用 make_embed_fn_with_attrs。
    """
    head = tp.get('head', '') or ''
    rel = tp.get('relation', '') or ''
    tail = tp.get('tail', '') or ''
    return embed_text(emb, f"{head} {rel} {tail}".strip())


def dedupe(
    lines: List[str],
    embed_fn: Callable[[str], np.ndarray],
    threshold: float
) -> List[str]:
    """
    依第一實體分桶，同一桶內若相似度 >= threshold 視為重複，
    只保留最長敘述，最後重編號。
    """
    groups: dict[str, list[tuple[str, np.ndarray]]] = {}
    order: list[str] = []

    for line in lines:
        m = ENTITY_PATTERN.match(line)
        key = m.group(1) if m else line.split()[0]
        vec = embed_fn(line)

        bucket = groups.setdefault(key, [])
        replaced = False
        for idx, (existing, existing_vec) in enumerate(bucket):
            if util.cos_sim(vec, existing_vec).item() >= threshold:
                if len(line) > len(existing):
                    bucket[idx] = (line, vec)
                    pos = order.index(existing)
                    order[pos] = line
                replaced = True
                break

        if not replaced:
            bucket.append((line, vec))
            order.append(line)

    # 重編號並移除原有編號
    result: list[str] = []
    for i, original in enumerate(order, start=1):
        without_num = NUMBERING_PATTERN.sub('', original)
        result.append(f'[{i}] {without_num}')

    return result
