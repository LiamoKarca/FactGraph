"""
Pipeline：Neo4j → CSV → 向量檔
預設輸出：
  - CSV  : experiment/data/raw/neo4j-kg-raw-graph.csv
  - EMB  : experiment/data/processed/neo4j-kg.emb.npy

使用方式（最常見）：
  python experiment/preliminary_work/pipeline_embed_kg.py --include-props

可用參數：
  --extract-only / --embed-only      # 只跑其中一段
  --force                            # 即使目標檔已存在也重跑
  --dry-run                          # 只顯示將執行的指令，不實際執行
  --limit N                          # 抽樣匯出前 N 筆（三元組）除錯用
  --database DBNAME                  # 指定 Neo4j database
  --csv PATH                         # 預設 experiment/data/raw/neo4j-kg-raw-graph.csv
  --out-npy PATH                     # 預設 experiment/data/processed/neo4j-kg.emb.npy
  --model-root PATH                  # 預設 models/CKIP/models--ckiplab--bert-base-chinese
  --include-props                    # 向量化時將三端屬性併入語料
  --python-bin PATH                  # 指定 python 可執行檔；預設為目前解譯器
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time

from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

# 預設路徑（與前兩支腳本一致）
DEFAULT_CSV = "experiment/data/raw/neo4j-kg-raw-graph.csv"
DEFAULT_NPY = "experiment/data/processed/neo4j-kg.emb.npy"
DEFAULT_MODEL = "models/CKIP/models--ckiplab--bert-base-chinese"
EXTRACT_SCRIPT = "experiment/preliminary_work/neo4j-data-extraction.py"
EMBED_SCRIPT = "experiment/preliminary_work/embed_kg_data_csv.py"


def run(cmd: List[str], dry_run: bool) -> int:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"▶ {printable}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--database", type=str,
                    default=os.getenv("NEO4J_DATABASE", "neo4j"))
    ap.add_argument("--csv", type=str, default=DEFAULT_CSV)
    ap.add_argument("--out-npy", type=str, default=DEFAULT_NPY)
    ap.add_argument("--model-root", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--include-props", action="store_true")
    ap.add_argument("--python-bin", type=str, default=sys.executable)
    args = ap.parse_args()

    t0 = time.time()
    csv_path = Path(args.csv)
    npy_path = Path(args.out_npy)

    # 路徑存在性預處理
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== Step 1: Neo4j → CSV =====
    skip_extract = False
    if args.embed_only:
        skip_extract = True
    if csv_path.exists() and not args.force and not args.embed_only:
        print(f"ℹ️  CSV 已存在且未指定 --force，略過匯出：{csv_path}")
        skip_extract = True

    if not skip_extract:
        # 檢查基本環境變數
        for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
            if not os.getenv(k):
                print(f"❌ 缺少必要環境變數：{k}；請於 .env 或環境中設定。")
                sys.exit(2)

        cmd = [
            args.python_bin, EXTRACT_SCRIPT,
            "--out", str(csv_path),
            "--database", args.database,
        ]
        if args.limit and args.limit > 0:
            cmd += ["--limit", str(args.limit)]

        print("\n=== Step 1/2: Neo4j → CSV ===")
        rc = run(cmd, dry_run=args.dry_run)
        if rc != 0:
            print("❌ 匯出失敗（Neo4j → CSV）。")
            sys.exit(rc)

    if not csv_path.exists():
        print(f"❌ 找不到 CSV：{csv_path}。請先成功完成匯出。")
        sys.exit(3)

    # ===== Step 2: CSV → 向量 NPY =====
    skip_embed = False
    if args.extract_only:
        skip_embed = True
    if npy_path.exists() and not args.force and not args.extract_only:
        print(f"ℹ️  向量檔已存在且未指定 --force，略過向量化：{npy_path}")
        skip_embed = True

    if not skip_embed:
        cmd = [
            args.python_bin, EMBED_SCRIPT,
            "--csv", str(csv_path),
            "--model-root", args.model_root,
            "--out-npy", str(npy_path),
        ]
        if args.include_props:
            cmd.append("--include-props")

        print("\n=== Step 2/2: CSV → 向量 ===")
        rc = run(cmd, dry_run=args.dry_run)
        if rc != 0:
            print("❌ 向量化失敗（CSV → NPY）。")
            sys.exit(rc)

    # ===== Done =====
    t1 = time.time()
    print("\n✅ Pipeline 完成")
    if csv_path.exists():
        print(f"   CSV ：{csv_path}")
    if npy_path.exists():
        print(f"   NPY ：{npy_path}")
    print(f"   用時：{t1 - t0:.1f}s")


if __name__ == "__main__":
    main()
