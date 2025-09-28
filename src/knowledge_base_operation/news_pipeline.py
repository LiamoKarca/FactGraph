"""
$ python src/knowledge_base_operation/news_pipeline.py
一鍵串接：
1) 先跑新聞爬蟲總管線：src/knowledge_base_operation/news_crawler/pipeline.py
2) 成功後接續跑合併/去重/上傳總管線：src/knowledge_base_operation/news_merge/pipeline.py

附帶功能：
- --crawler-args 與 --merge-args 可把參數字串直通到子管線
- --skip-crawl / --skip-merge 可單獨跳過某段（除錯或重跑用）
- 保持 CWD 在專案根目錄，並注入 PROJECT_ROOT / PYTHONPATH 以避免相對路徑漂移
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# 專案根目錄偵測：自本檔往上尋找，同時包含 src/ 與 data/ 的目錄
THIS_FILE = Path(__file__).resolve()


def find_project_root(start: Path) -> Path:
    for anc in [start] + list(start.parents):
        if (anc / "src").is_dir() and (anc / "data").is_dir():
            return anc
    # 後備：若找不到同時包含 src/ 與 data/ 的目錄，就退回到含 src/ 的最近父層
    for anc in [start] + list(start.parents):
        if (anc / "src").is_dir():
            return anc
    # 最後保險
    return start.parents[2] if len(start.parents) >= 3 else Path.cwd()


PROJECT_ROOT = find_project_root(THIS_FILE)

PYTHON = sys.executable  # 使用當前虛擬環境的直譯器

CRAWLER_PIPELINE = PROJECT_ROOT / \
    "src/knowledge_base_operation/news_crawler/pipeline.py"
MERGE_PIPELINE = PROJECT_ROOT / "src/knowledge_base_operation/news_merge/pipeline.py"


# ──────────────────────────────────────────────────────────────────────────────
def run_subpipeline(script: Path, extra_args: Optional[List[str]] = None) -> int:
    """
    以專案根目錄為 CWD 執行子管線；把 PROJECT_ROOT / PYTHONPATH 打進環境。
    將子程序的 stdout/stderr 直接串流到目前的終端。
    傳回子程序的退出碼。
    """
    if not script.exists():
        print(f"[pipeline] 找不到子管線：{script}")
        return 127

    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

    cmd = [PYTHON, str(script)]
    if extra_args:
        cmd += extra_args

    print(f"[pipeline] 執行：{' '.join(shlex.quote(c) for c in cmd)}")
    try:
        # 不截取輸出，直接繼承父程序 stdout/stderr，以便即時觀察
        completed = subprocess.run(cmd, cwd=str(
            PROJECT_ROOT), env=env, check=False)
        return completed.returncode
    except KeyboardInterrupt:
        print("[pipeline] 收到中斷（Ctrl+C），停止。")
        return 130
    except Exception as e:
        print(f"[pipeline] 無法啟動子管線：{e}")
        return 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="一鍵串接：news_crawler → news_merge 的總管線"
    )
    ap.add_argument(
        "--crawler-args",
        type=str,
        default="",
        help="傳給 news_crawler/pipeline.py 的參數字串（例如：--only PTS,CNA --sequential）",
    )
    ap.add_argument(
        "--merge-args",
        type=str,
        default="",
        help="傳給 news_merge/pipeline.py 的參數字串（例如：--force-dedup --force-upload）",
    )
    ap.add_argument("--skip-crawl", action="store_true", help="跳過爬蟲階段")
    ap.add_argument("--skip-merge", action="store_true", help="跳過合併/去重/上傳階段")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # 1) 爬蟲總管線
    if not args.skip_crawl:
        crawler_extra = shlex.split(
            args.crawler_args) if args.crawler_args else []
        print("[pipeline] ==== 1/2：啟動爬蟲流水線 ====")
        rc = run_subpipeline(CRAWLER_PIPELINE, crawler_extra)
        if rc != 0:
            print(f"[pipeline] 爬蟲流水線非零退出碼：{rc}；停止後續步驟。")
            sys.exit(rc)
    else:
        print("[pipeline] 跳過爬蟲階段 (--skip-crawl)")

    # 2) 合併/去重/上傳總管線
    if not args.skip_merge:
        merge_extra = shlex.split(args.merge_args) if args.merge_args else []
        print("[pipeline] ==== 2/2：啟動合併/去重/上傳流水線 ====")
        rc = run_subpipeline(MERGE_PIPELINE, merge_extra)
        if rc != 0:
            print(f"[pipeline] 合併/去重/上傳流水線非零退出碼：{rc}")
            sys.exit(rc)
    else:
        print("[pipeline] 跳過合併/去重/上傳階段 (--skip-merge)")

    print("[pipeline] ✅ 全部完成。")


if __name__ == "__main__":
    main()
