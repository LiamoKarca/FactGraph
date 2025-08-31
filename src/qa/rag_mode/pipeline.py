"""
RAG Mode – 單一入口管線（one entrypoint）
- $ python src/qa/rag_mode/pipeline.py

流程：
1) keywords → 提取關鍵字
2) annex    → 合併原始文本與關鍵字
3) gpt      → RAG 判斷與生成（在執行前會先檢查「線上最新 VS」是否已有檔案，若仍在索引中則跳過並提示稍後重試）
"""

from __future__ import annotations
from openai import OpenAI  # type: ignore

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Optional

# ── Import path 設定：確保可以以 `from src.qa...` 匯入 ───────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qa.rag_mode.llm.config import PATHS, make_openai_client  # noqa: E402

# ── 步驟定義 ────────────────────────────────────────────────────────────────


@dataclass
class Step:
    name: str
    script: Path
    description: str = ""


STEPS: Dict[str, Step] = {
    "keywords": Step("keywords", Path("src/qa/rag_mode/llm/keywords.py"), "提取關鍵字"),
    "annex":   Step("annex",   Path("src/qa/rag_mode/llm/annex_kw.py"),   "合併原始文本與關鍵字"),
    "gpt":     Step("gpt",     Path("src/qa/rag_mode/llm/gpt_rag.py"),    "RAG 判斷與生成"),
}
ORDER = ["keywords", "annex", "gpt"]

# ── VS 預檢（線上最新 + 輪詢）───────────────────────────────────────────────


def _list_all_vector_stores(client: OpenAI, limit: int = 100):
    items = []
    after: Optional[str] = None
    while True:
        page = client.vector_stores.list(
            limit=limit, after=after) if after else client.vector_stores.list(limit=limit)
        items.extend(getattr(page, "data", []) or [])
        if not getattr(page, "has_more", False):
            break
        after = items[-1].id
    return items


def _pick_latest_online_vector_store_id(client: OpenAI) -> str:
    items = _list_all_vector_stores(client)
    if not items:
        raise RuntimeError("帳號下沒有任何 Vector Store。")
    items.sort(key=lambda x: getattr(x, "created_at", 0), reverse=True)
    return getattr(items[0], "id")


def _vector_store_file_count(client: OpenAI, vs_id: str) -> int:
    try:
        page = client.vector_stores.files.list(vector_store_id=vs_id, limit=1)
        return len(page.data)
    except Exception:
        vs = client.vector_stores.retrieve(vs_id)
        fc = getattr(vs, "file_counts", None) or getattr(
            vs, "stats", None) or {}
        return int(fc.get("total", 0)) if isinstance(fc, dict) else 0


def _wait_nonempty_vector_store(client: OpenAI, vs_id: str, retries: int = 8, delay_sec: float = 2.5) -> bool:
    for i in range(retries + 1):
        cnt = _vector_store_file_count(client, vs_id)
        if cnt > 0:
            return True
        if i < retries:
            print(f"[INFO] VS {vs_id} 暫無檔案（total=0），等待索引中… 重試 {i+1}/{retries}")
            time.sleep(delay_sec)
    return False


def _precheck_vs_before_gpt() -> bool:
    """
    回傳 True 表示可以進入 gpt；False 表示暫時不可（提示稍後重試）。
    """
    client = make_openai_client()
    vs_id = _pick_latest_online_vector_store_id(client)
    print(f"[VS] 將使用線上最新向量庫：{vs_id}")
    ready = _wait_nonempty_vector_store(
        client, vs_id, retries=8, delay_sec=2.5)
    if not ready:
        print(f"⚠ 剛更新完向量庫（{vs_id}），索引仍在進行中；請稍待幾分鐘後重試。")
        return False
    return True

# ── 執行器 ──────────────────────────────────────────────────────────────────


def _count_user_txts() -> int:
    return len(list(PATHS.USER_INPUT_DIR.glob("*.txt")))


def _validate_preconditions(plan: List[str]) -> None:
    if any(s in plan for s in ("annex", "gpt")):
    # 若由上游指定了 RAG_USER_FILE，代表採用 job-scoped 輸入，跳過全域目錄檢查
        if os.getenv("RAG_USER_FILE"):
            print("✓ 檢查略過：偵測到 RAG_USER_FILE（job-scoped 輸入），不掃全域 user-input。")
            return
        # 舊邏輯：僅允許 USER_INPUT_DIR 內存在「唯一一個 .txt」
        n = _count_user_txts()
        if n == 0:
            raise SystemExit(f"[錯誤] 找不到任何 .txt：{PATHS.USER_INPUT_DIR}/*.txt")
        if n > 1:
            raise SystemExit(f"[錯誤] {PATHS.USER_INPUT_DIR} 內有多個 .txt（{n} 個），僅支援單一檔案")


def _run(script: Path, args: List[str] | None = None) -> int:
    cmd = [sys.executable, str(script)]
    if args:
        cmd += args
    print(f"\n━━▶ 執行：{script}")
    print(f"    指令：{shlex.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
        env=os.environ.copy(),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
    proc.stdout.close()
    rc = proc.wait()
    print(f"◀━━ 結束：{script.name}（returncode={rc}）\n")
    return rc

# ── 參數與主流程 ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG Mode Pipeline (single entrypoint)")
    p.add_argument("--only",    default="keywords,annex,gpt",
                   help="只執行指定步驟（逗號分隔）")
    p.add_argument("--skip",    default="",
                   help="略過指定步驟（逗號分隔）")
    p.add_argument("--dry-run", action="store_true",
                   help="僅顯示規劃，不執行")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    only: Set[str] = set([s.strip()
                         for s in args.only.split(",") if s.strip()])
    skip: Set[str] = set([s.strip()
                         for s in args.skip.split(",") if s.strip()])
    unknown = (only | skip) - set(STEPS.keys())
    if unknown:
        raise SystemExit(f"[錯誤] 未知步驟：{', '.join(sorted(unknown))}")
    plan: List[str] = [s for s in ORDER if s in only and s not in skip]

    if not plan:
        print("[資訊] 無任何步驟需要執行")
        return

    print("RAG Mode Pipeline – 執行規劃：")
    for s in plan:
        st = STEPS[s]
        print(f"  - {st.name:8s} → {st.description}")

    if args.dry_run:
        print("[Dry-Run] 僅顯示規劃，不執行")
        return

    # 前置檢查
    _validate_preconditions(plan)

    # 跑 gpt 前先檢查線上最新 VS 是否已非空
    if "gpt" in plan and not _precheck_vs_before_gpt():
        print("✖ 已跳過 gpt 步驟（請稍待幾分鐘後重試）")
        plan = [s for s in plan if s != "gpt"]

    # 執行各步驟
    for s in plan:
        st = STEPS[s]
        rc = _run(st.script, [])
        if rc != 0:
            raise SystemExit(f"[失敗] {s} returncode={rc}，中止。")

    print("✓ Pipeline 全部完成")


if __name__ == "__main__":
    main()
