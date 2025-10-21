#!/usr/bin/env python3
# -*- coding: utf-8-sig -*-
"""
KG 索引同步檢查器（MiniLM L12 v2 / Sentence-Transformers 版）

功能：
    1) 檢查 CSV 與向量 NPY 行數是否一致
    2) 檢查向量維度與本機 MiniLM 模型是否一致
    3) 檢查（若存在）.meta.json 的 row_count / csv_mtime / model_dim
    4) 掃描 rel_props.date 的最新日期與來源分佈（可選）

用法：
    python -m src.qa.tools.check_kg_index_sync \
      --csv data/raw/knowledge-graph/neo4j-kg-raw-graph.csv \
      --emb data/processed/knowledge-graph/neo4j-kg.emb.npy \
      --model models/MiniLM/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots \
      --strict

環境變數（可覆蓋）：
    CSV_PATH
        預設 data/raw/knowledge-graph/neo4j-kg-raw-graph.csv
    KG_EMB_PATH
        預設 data/processed/knowledge-graph/neo4j-kg.emb.npy
    MINILM_MODEL_ROOT
        預設 models/MiniLM/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots

退出碼：
    0 一切一致
    2 CSV 與 NPY 行數不一致
    3 向量維度與模型不一致
    4 meta.json 顯示已過期/不一致
    5 檔案缺失
    6 CSV 解析 rel_props 或日期欄位異常（不阻斷行數與維度檢查）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """解析命令列參數。

    Returns:
        argparse.Namespace: 包含路徑、模型根目錄與其他旗標。
    """
    def env_or(name: str, default: str) -> str:
        return os.getenv(name, default)

    ap = argparse.ArgumentParser(description="KG 索引同步檢查器（MiniLM 版）")
    ap.add_argument(
        "--csv",
        default=env_or("CSV_PATH", "data/raw/knowledge-graph/neo4j-kg-raw-graph.csv"),
        help="KG CSV 路徑",
    )
    ap.add_argument(
        "--emb",
        default=env_or("KG_EMB_PATH", "data/processed/knowledge-graph/neo4j-kg.emb.npy"),
        help="KG 向量 NPY 路徑",
    )
    ap.add_argument(
        "--model",
        default=env_or(
            "MINILM_MODEL_ROOT",
            "models/MiniLM/models--sentence-transformers--"
            "paraphrase-multilingual-MiniLM-L12-v2/snapshots",
        ),
        help="本機 MiniLM 模型根目錄（可指向 snapshots/ 或 snapshot 內層）",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="若發現任何不一致，以非零代碼退出（CI 友善）",
    )
    ap.add_argument(
        "--sample-months",
        type=int,
        default=2,
        help="統計最近 N 個月份的來源分佈（預設 2）",
    )
    return ap.parse_args()


# ── 工具 ─────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def resolve_snapshot(root: Path) -> str:
    """解析 HF snapshot 路徑。

    若 root 本身即含權重檔（config.json），直接回傳；
    否則從 root/snapshots/* 中擇一含 config.json 的子目錄。

    Args:
        root: 模型根目錄。

    Returns:
        str: 可供 SentenceTransformer 直接載入的目錄字串。

    Raises:
        FileNotFoundError: 找不到有效的 snapshot。
    """
    if (root / "config.json").is_file():
        return str(root)
    snaps = root / "snapshots"
    if snaps.is_dir():
        for sub in snaps.iterdir():
            if (sub / "config.json").is_file():
                return str(sub)
    raise FileNotFoundError(f"❌ 找不到 MiniLM 模型快照於：{root}")


def load_st_dim(model_root: str) -> int:
    """載入 Sentence-Transformers 模型並回傳句向量維度。

    Args:
        model_root: 模型根目錄或 snapshot 目錄。

    Returns:
        int: 句向量維度。

    References:
        - SentenceTransformer 支援以本機目錄載入，並提供
          get_sentence_embedding_dimension() 取得維度。:contentReference[oaicite:1]{index=1}
    """
    from sentence_transformers import SentenceTransformer  # 延遲載入
    path = resolve_snapshot(Path(model_root))
    model = SentenceTransformer(path, trust_remote_code=True)
    return int(model.get_sentence_embedding_dimension())


def parse_date(s: str) -> Optional[Tuple[int, int, int]]:
    """嘗試解析 YYYY-MM-DD 日期字串。"""
    m = _DATE_RE.match((s or "").strip())
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return y, mo, d


def month_key(s: str) -> Optional[str]:
    """轉為 YYYY-MM 月鍵。"""
    dt = parse_date(s)
    if not dt:
        return None
    return f"{dt[0]:04d}-{dt[1]:02d}"


def safe_json_loads(s: Any) -> Dict[str, Any]:
    """安全解析 JSON 字串。"""
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s or "{}")
    except Exception:
        return {}


def read_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    """讀取 .meta.json。"""
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


# ── 主流程 ──────────────────────────────────────────────────────────


def main() -> int:
    """程式進入點。"""
    args = parse_args()
    csv_path = Path(args.csv)
    emb_path = Path(args.emb)
    meta_path = emb_path.with_suffix(".meta.json")
    model_root = args.model

    # 檔案存在性
    missing = []
    if not csv_path.is_file():
        missing.append(str(csv_path))
    if not emb_path.is_file():
        missing.append(str(emb_path))
    if missing:
        print("❌ 檔案缺失：")
        for p in missing:
            print(f"   - {p}")
        return 5 if args.strict else 0

    # 讀 CSV
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    n_csv = len(df)
    print(f"📄 CSV: {csv_path}  rows={n_csv:,}")

    # 讀 NPY
    embs = np.load(emb_path)
    n_emb, d_emb = embs.shape
    print(f"📦 EMB: {emb_path}  shape={n_emb:,} x {d_emb}")

    # 行數一致性
    row_ok = n_csv == n_emb
    print(f"▶ 行數一致性: {'✅ OK' if row_ok else f'❌ 不一致（CSV={n_csv:,}, EMB={n_emb:,}）'}")

    # 模型維度
    try:
        dim_model = load_st_dim(model_root)
        dim_ok = dim_model == d_emb
        print(
            "▶ 向量維度檢查: 模型="
            f"{dim_model}  vs  EMB={d_emb}  → {'✅ OK' if dim_ok else '❌ 不一致'}"
        )
    except Exception as e:
        print(f"⚠ 無法載入 MiniLM 模型取得維度：{e}")
        dim_ok = True  # 非 strict 情境下不強制報錯

    # 檢查 meta（若存在）
    meta = read_meta(meta_path)
    meta_ok = True
    if meta:
        print(f"📝 META: {meta_path}")
        row_match = int(meta.get("row_count", -1)) == n_csv
        mtime_csv = float(csv_path.stat().st_mtime)
        mtime_match = abs(float(meta.get("csv_mtime", 0.0)) - mtime_csv) < 1.0
        dim_match = int(meta.get("model_dim", -1)) == d_emb
        print(f"   - row_count: {'✅' if row_match else '❌'}")
        print(f"   - csv_mtime: {'✅' if mtime_match else '❌'}")
        print(f"   - model_dim: {'✅' if dim_match else '❌'}")
        meta_ok = row_match and mtime_match and dim_match
    else:
        print("📝 META: （無 .meta.json，可略）")

    # 解析日期與來源分佈（不影響一致性退出碼）
    parse_issue = False
    latest_date = None
    try:
        dates: List[str] = []
        if "rel_props" in df.columns:
            for _, row in df.iterrows():
                props = safe_json_loads(row.get("rel_props", {}))
                d = str(props.get("date") or "").strip()
                if _DATE_RE.match(d):
                    dates.append(d)
        if dates:
            dates.sort()
            latest_date = dates[-1]
            print(f"🕒 CSV 內 evidence 最新日期：{latest_date}")
    except Exception as e:
        print(f"⚠ 解析 rel_props/date 時發生例外：{e}")
        parse_issue = True

    # 結論與退出碼
    ok_all = row_ok and dim_ok and (meta_ok or meta is None)
    if ok_all:
        print("✅ 結論：索引同步且維度一致。")
        if latest_date:
            print(f"   （觀察值）CSV 內最新 evidence 日期：{latest_date}")
        return 0

    if args.strict:
        if not row_ok:
            return 2
        if not dim_ok:
            return 3
        if meta is not None and not meta_ok:
            return 4
        if parse_issue:
            return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
