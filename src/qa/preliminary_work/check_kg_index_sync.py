#!/usr/bin/env python3
# -*- coding: utf-8-sig -*-
"""
KG 索引同步檢查器
- 檢查 CSV 與 NPY 行數是否一致
- 檢查 CKIP 向量維度與模型是否一致
- 檢查（若存在）.meta.json 的 row_count / csv_mtime / model_dim 是否一致
- 掃描 rel_props.date 的最新日期與來源分佈，用來確認是否包含近期資料

用法：
  python tools/check_kg_index_sync.py \
    --csv data/raw/knowledge-graph/neo4j-kg-raw-graph.csv \
    --emb data/processed/knowledge-graph/neo4j-kg.emb.npy \
    --model models/CKIP/models--ckiplab--bert-base-chinese

環境變數（可覆蓋）：
  CSV_PATH               預設 data/raw/knowledge-graph/neo4j-kg-raw-graph.csv
  KG_EMB_PATH            預設 data/processed/knowledge-graph/neo4j-kg.emb.npy
  CKIP_MODEL_ROOT        預設 models/CKIP/models--ckiplab--bert-base-chinese

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
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import re
import numpy as np
import pandas as pd

# ── CLI ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    def env_or(name: str, default: str) -> str:
        return os.getenv(name, default)

    ap = argparse.ArgumentParser(description="KG 索引同步檢查器")
    ap.add_argument("--csv", default=env_or("CSV_PATH",
                    "data/raw/knowledge-graph/neo4j-kg-raw-graph.csv"),
                    help="KG CSV 路徑")
    ap.add_argument("--emb", default=env_or("KG_EMB_PATH",
                    "data/processed/knowledge-graph/neo4j-kg.emb.npy"),
                    help="KG 向量 NPY 路徑")
    ap.add_argument("--model", default=env_or("CKIP_MODEL_ROOT",
                    "models/CKIP/models--ckiplab--bert-base-chinese"),
                    help="CKIP 模型根目錄（可為 snapshot 或含 snapshots/）")
    ap.add_argument("--strict", action="store_true",
                    help="若發現任何不一致，以非零代碼退出（CI 友善）")
    ap.add_argument("--sample-months", type=int, default=2,
                    help="統計最近 N 個月份的來源分佈（預設 2）")
    return ap.parse_args()

# ── 工具 ─────────────────────────────────────────────────────────────

def resolve_snapshot(root: Path) -> str:
    """解析 HF snapshot 路徑（相容你現有的寫法）"""
    if (root / "config.json").is_file():
        return str(root)
    snaps = root / "snapshots"
    if snaps.is_dir():
        for sub in snaps.iterdir():
            if (sub / "config.json").is_file():
                return str(sub)
    raise FileNotFoundError(f"❌ 找不到 CKIP 模型快照於：{root}")

def load_ckip_dim(model_root: str) -> int:
    """載入 SentenceTransformer 取得向量維度（不做 encode，僅載入）"""
    from sentence_transformers import SentenceTransformer
    import torch
    path = resolve_snapshot(Path(model_root))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(path, device=device, trust_remote_code=True)
    dim = int(model.get_sentence_embedding_dimension())
    return dim

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

def parse_date(s: str) -> Optional[Tuple[int,int,int]]:
    m = _DATE_RE.match(s.strip())
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return (y, mo, d)

def month_key(s: str) -> Optional[str]:
    dt = parse_date(s)
    if not dt:
        return None
    return f"{dt[0]:04d}-{dt[1]:02d}"

def safe_json_loads(s: Any) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s or "{}")
    except Exception:
        return {}

def read_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None

# ── 主流程 ──────────────────────────────────────────────────────────

def main() -> int:
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

    # 讀 CSV（與你 RAG 相同的 encoding）
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        # 兼容無 BOM 的 UTF-8
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    n_csv = len(df)
    print(f"📄 CSV: {csv_path}  rows={n_csv:,}")

    # 讀 NPY
    embs = np.load(emb_path)
    n_emb, d_emb = embs.shape
    print(f"📦 EMB: {emb_path}  shape={n_emb:,} x {d_emb}")

    # 行數一致性
    row_ok = (n_csv == n_emb)
    print(f"▶ 行數一致性: {'✅ OK' if row_ok else f'❌ 不一致（CSV={n_csv:,}, EMB={n_emb:,}）'}")

    # 模型維度
    try:
        dim_model = load_ckip_dim(model_root)
        dim_ok = (dim_model == d_emb)
        print(f"▶ 向量維度檢查: 模型={dim_model}  vs  EMB={d_emb}  → "
              f"{'✅ OK' if dim_ok else '❌ 不一致'}")
    except Exception as e:
        print(f"⚠ 無法載入 CKIP 模型取得維度：{e}")
        dim_ok = True  # 不阻斷（可選擇開 strict 才當錯）

    # 檢查 meta（若存在）
    meta = read_meta(meta_path)
    meta_ok = True
    if meta:
        print(f"📝 META: {meta_path}")
        row_match = (int(meta.get("row_count", -1)) == n_csv)
        mtime_csv = os.path.getmtime(csv_path)
        # 容忍小數點誤差（不同檔案系統）
        mtime_match = abs(float(meta.get("csv_mtime", 0)) - mtime_csv) < 1.0
        dim_match = (int(meta.get("model_dim", -1)) == d_emb)
        print(f"   - row_count: {'✅' if row_match else '❌'} (meta={meta.get('row_count')} vs csv={n_csv})")
        print(f"   - csv_mtime: {'✅' if mtime_match else '❌'} (meta={meta.get('csv_mtime')} vs real={mtime_csv})")
        print(f"   - model_dim: {'✅' if dim_match else '❌'} (meta={meta.get('model_dim')} vs emb={d_emb})")
        meta_ok = row_match and mtime_match and dim_match
    else:
        print("📝 META: （無 .meta.json，可略）")

    # 解析 rel_props 的日期與來源，協助判斷是否含近期資料
    parse_issue = False
    latest_date = None
    try:
        # 嘗試抓出 rel_props.date（若有）
        dates: List[str] = []
        srcs: List[str] = []
        if "rel_props" in df.columns:
            for _, row in df.head(n_csv).iterrows():
                props = safe_json_loads(row.get("rel_props", {}))
                d = str(props.get("date") or "").strip()
                if _DATE_RE.match(d):
                    dates.append(d)
                doc_id = str(props.get("doc_id") or "")
                if doc_id:
                    srcs.append(doc_id.split("_")[0].upper())
        if dates:
            dates.sort()
            latest_date = dates[-1]
            print(f"🕒 CSV 內 evidence 最新日期：{latest_date}")
            # 最近 N 個月份來源分佈
            months_needed = args.sample_months
            uniq_months = sorted({month_key(d) for d in dates if month_key(d)}, reverse=True)
            recent = set(uniq_months[:months_needed])
            if recent:
                # 計算最近月份的來源分佈
                month_src_cnt: Dict[str, Dict[str,int]] = {}
                for _, row in df.iterrows():
                    props = safe_json_loads(row.get("rel_props", {}))
                    d = str(props.get("date") or "").strip()
                    mk = month_key(d)
                    if not mk or mk not in recent:
                        continue
                    doc_id = str(props.get("doc_id") or "")
                    src = doc_id.split("_")[0].upper() if doc_id else "UNKNOWN"
                    month_src_cnt.setdefault(mk, {}).setdefault(src, 0)
                    month_src_cnt[mk][src] += 1
                print("📊 最近月份來源分佈：")
                for mk in sorted(month_src_cnt.keys(), reverse=True):
                    print(f"   - {mk}: " + ", ".join(f"{k}={v}" for k,v in sorted(month_src_cnt[mk].items())))
        else:
            print("🕒 未在 rel_props 找到有效的 YYYY-MM-DD 日期欄位（可能欄位缺失或格式不一）。")
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

    # 決定退出碼（優先級：行數→維度→meta→解析）
    code = 0
    if not row_ok:
        code = 2
    elif not dim_ok:
        code = 3
    elif meta is not None and not meta_ok:
        code = 4
    elif parse_issue:
        code = 6

    if args.strict:
        return code
    else:
        # 非 strict 模式不強制非零退出，僅報告
        return 0

if __name__ == "__main__":
    sys.exit(main())
