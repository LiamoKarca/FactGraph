# -*- coding: utf-8-sig -*-
"""
CKIP / HF 模型自動轉換為 Sentence-Transformers（含 Pooling 層）
================================================================

設計
----
- 若目錄含 modules.json：視為原生 Sentence-Transformers，直接載入。
- 若不含：以 Transformer+Pooling 動態組裝，保存到 st-converted/，後續皆載入該目錄。
- 關閉 Accelerate 的自動裝置對映，避免 meta tensor 問題。
- 以 RLock 確保首次初始化與首次 encode 的並行安全。

函式
----
- get_embedder() -> SentenceTransformer
- embed_text(text: str) -> np.ndarray
- embed_triple(triple: Dict[str, str]) -> np.ndarray
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict

# 在任何 torch/transformers/sentence_transformers 匯入前，關閉 Accelerate
os.environ.setdefault("ACCELERATE_DISABLE_DEVICE_MAP", "1")
os.environ.setdefault("TRANSFORMERS_NO_ACCELERATE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from sentence_transformers.models import Transformer, Pooling  # noqa: E402

try:
    from .paths import CKIP_ROOT  # type: ignore
except Exception:
    CKIP_ROOT = Path("~/.cache/ckip-sbert-base-zh").expanduser()

_INIT_LOCK = threading.RLock()
_ENCODE_LOCK = threading.RLock()


def _has_modules_json(path: Path) -> bool:
    """
    檢查是否為原生 Sentence-Transformers 模型目錄。

    Args:
        path: 可能的模型路徑。

    Returns:
        是否含 modules.json。
    """
    return (path / "modules.json").is_file()


def _resolve_snapshot(root: Path) -> Path:
    """
    解析 CKIP / HF 模型快取根目錄，回傳實際快照目錄或根目錄。

    優先順序：
      1) 根目錄含 config.json → 視為可用。
      2) snapshots/<hash>/config.json → 使用該目錄。
    """
    if (root / "config.json").is_file() or _has_modules_json(root):
        return root
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        for sub in snapshots.iterdir():
            if (sub / "config.json").is_file() or _has_modules_json(sub):
                return sub
    raise FileNotFoundError("CKIP / HF 模型快取未找到可用快照")


def _ensure_st_converted(path: Path) -> Path:
    """
    確保路徑為「可直接由 SentenceTransformer 載入」的 ST 模型目錄。
    若非 ST 目錄（無 modules.json），則以 Transformer+Pooling 組裝，保存到 st-converted/。

    Args:
        path: HF 模型快照或 ST 模型目錄。

    Returns:
        可被 SentenceTransformer 載入的 ST 目錄。
    """
    if _has_modules_json(path):
        return path

    out_dir = path / "st-converted"
    if _has_modules_json(out_dir):
        return out_dir

    # 建構為標準 ST：Transformer + Pooling（mean）
    word = Transformer(str(path))
    dim = word.get_word_embedding_dimension()
    pool = Pooling(
        word_embedding_dimension=dim,
        pooling_mode="mean",
    )
    model = SentenceTransformer(modules=[word, pool])
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    return out_dir


@lru_cache(maxsize=1)
def _get_embedder_unsafe() -> SentenceTransformer:
    """
    不加鎖的建模函式（供外層鎖保護後呼叫）。

    Returns:
        已在目標裝置初始化的嵌入器。
    """
    model_name = os.getenv("SENTENCE_TRANSFORMERS_MODEL", "").strip()
    target_device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name:
        # 顯式指定 ST 模型（建議用 sentence-transformers/* 或自行訓練後的 ST 目錄）
        return SentenceTransformer(model_name, device=target_device)

    # 走既有 CKIP_ROOT 路徑：若非 ST，動態轉換一次並快取
    snapshot = _resolve_snapshot(Path(CKIP_ROOT))
    st_dir = _ensure_st_converted(snapshot)
    return SentenceTransformer(str(st_dir), device=target_device)


def get_embedder() -> SentenceTransformer:
    """
    取得單例 SentenceTransformer（並行安全）。

    Returns:
        嵌入器單例。
    """
    with _INIT_LOCK:
        return _get_embedder_unsafe()


def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2 正規化。

    Args:
        vec: 向量或矩陣。
        eps: 避免除以零的小常數。

    Returns:
        float32 單位向量或矩陣。
    """
    if not isinstance(vec, np.ndarray):
        vec = np.asarray(vec)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    out = vec / norm
    if out.dtype != np.float32:
        out = out.astype(np.float32, copy=False)
    return out


def embed_text(text: str) -> np.ndarray:
    """
    產生單句向量（L2 正規化）。

    Args:
        text: 文字。

    Returns:
        形狀 (dim,) 的 float32 向量。
    """
    embedder = get_embedder()
    with _ENCODE_LOCK:
        emb = embedder.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
    return _l2_normalize(emb)


def embed_triple(triple: Dict[str, str]) -> np.ndarray:
    """
    將三元組串接為一句文本嵌入。

    Args:
        triple: 含 'head'、'relation'、'tail'。

    Returns:
        形狀 (dim,) 的 float32 向量。
    """
    text = f"{triple['head']} {triple['relation']} {triple['tail']}"
    return embed_text(text)
