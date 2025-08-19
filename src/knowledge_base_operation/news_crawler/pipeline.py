"""
一次啟動 PTS / MOI / CNA / EY 四個爬蟲，並行執行、即時串流輸出、監控結束狀態。
- 預設並行；可用 --sequential 改為逐一執行
- 可用 --pts-args / --moi-args / --cna-args / --ey-args 傳遞各爬蟲參數（字串，shlex.split 解析）
- 可用 --only 篩選（例：--only PTS,CNA,EY）
- 可用 --max-idle-seconds 啟用無輸出 watchdog（0=停用）
- $ python src/knowledge_base_operation/news_crawler/pipeline.py
"""

from __future__ import annotations
import argparse
import asyncio
import contextlib
import os
import shlex
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# 專案根目錄偵測：自檔案往上找，直到同時包含 src/ 與 data/ 的目錄為止
PIPELINE_FILE = Path(__file__).resolve()


def find_project_root(start: Path) -> Path:
    for anc in [start] + list(start.parents):
        if (anc / "src").is_dir() and (anc / "data").is_dir():
            return anc
    # 後備：若找不到，就退到含 src/ 的最近父層
    for anc in [start] + list(start.parents):
        if (anc / "src").is_dir():
            return anc
    return start.parents[3] if len(start.parents) >= 4 else Path.cwd()


PROJECT_ROOT = find_project_root(PIPELINE_FILE)
PIPELINE_DIR = PIPELINE_FILE.parent

# 子腳本統一以「專案根」為基準建立路徑，避免相對定位漂移
CRAWLERS = {
    "PTS": PROJECT_ROOT / "src/knowledge_base_operation/news_crawler/pts/pts.py",
    "MOI": PROJECT_ROOT / "src/knowledge_base_operation/news_crawler/moi/moi.py",
    "CNA": PROJECT_ROOT / "src/knowledge_base_operation/news_crawler/cna/cna.py",
    "EY":  PROJECT_ROOT / "src/knowledge_base_operation/news_crawler/ey/ey.py",
}

PYTHON = sys.executable  # 使用當前虛擬環境的直譯器

# ──────────────────────────────────────────────────────────────────────────────


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def colorize(s: str, color: str, enable: bool) -> str:
    if not enable:
        return s
    C = {
        "blue": "\033[34m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "reset": "\033[0m",
        "dim": "\033[2m",
    }
    return f"{C.get(color, '')}{s}{C['reset']}"


@dataclass
class RunResult:
    name: str
    returncode: Optional[int]
    start_ts: float
    end_ts: float
    reason: str  # "normal", "terminated", "killed", "spawn_error", "not_found", "interrupted"

    @property
    def duration(self) -> float:
        end = self.end_ts if self.end_ts else time.time()
        return max(0.0, end - self.start_ts)


@dataclass
class Runner:
    name: str
    script: Path
    extra_args: List[str] = field(default_factory=list)
    cwd: Path = PROJECT_ROOT
    env: Optional[Dict[str, str]] = None
    max_idle_seconds: int = 0
    use_color: bool = True

    # 內部狀態
    proc: Optional[asyncio.subprocess.Process] = None
    last_output_ts: float = field(default_factory=lambda: time.time())
    start_ts: float = 0.0
    end_ts: float = 0.0
    result_reason: str = "normal"

    async def run(self) -> RunResult:
        if not self.script.exists():
            print(colorize(
                f"[{now()}] [{self.name}] 找不到檔案：{self.script}", "red", self.use_color))
            return RunResult(self.name, None, time.time(), time.time(), "not_found")

        cmd = [PYTHON, str(self.script), *self.extra_args]

        # 傳遞 PROJECT_ROOT 與 PYTHONPATH，確保子程序用「相對路徑」即可正確落在 data 下
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env["PROJECT_ROOT"] = str(PROJECT_ROOT)
        src_path = str(PROJECT_ROOT / "src")
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

        print(colorize(
            f"[{now()}] [{self.name}] 啟動：{' '.join(shlex.quote(c) for c in cmd)}", "cyan", self.use_color))

        self.start_ts = time.time()
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.cwd),   # ← 這裡鎖定在 PROJECT_ROOT
                env=env
            )
        except Exception as e:
            print(
                colorize(f"[{now()}] [{self.name}] 無法啟動：{e}", "red", self.use_color))
            return RunResult(self.name, None, self.start_ts, time.time(), "spawn_error")

        readers = [
            asyncio.create_task(self._pipe(self.proc.stdout, is_err=False)),
            asyncio.create_task(self._pipe(self.proc.stderr, is_err=True)),
        ]
        watchdog_task = asyncio.create_task(
            self._watchdog()) if self.max_idle_seconds > 0 else None

        try:
            rc = await self.proc.wait()
            self.end_ts = time.time()
            if watchdog_task:
                watchdog_task.cancel()
            for t in readers:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait_for(t, timeout=1.0)
        finally:
            if watchdog_task:
                watchdog_task.cancel()

        if rc == 0 and self.result_reason == "normal":
            print(colorize(
                f"[{now()}] ✅ [{self.name}] 已蒐集完成最新新聞（exit=0）", "green", self.use_color))
        else:
            print(colorize(
                f"[{now()}] ⚠️ [{self.name}] 非正常結束（exit={rc}, reason={self.result_reason}）", "yellow", self.use_color))

        return RunResult(self.name, rc, self.start_ts, self.end_ts, self.result_reason)

    async def _pipe(self, stream: asyncio.StreamReader, is_err: bool = False):
        while True:
            line = await stream.readline()
            if not line:
                break
            self.last_output_ts = time.time()
            text = line.decode("utf-8-sig", errors="replace").rstrip("\n")
            stamp = colorize(f"[{now()}]", "dim", self.use_color)
            tag = colorize(f"[{self.name}]", "blue", self.use_color)
            channel = "stderr" if is_err else "stdout"
            print(f"{stamp} {tag} {channel}> {text}")

    async def _watchdog(self):
        try:
            while self.proc and self.proc.returncode is None:
                await asyncio.sleep(1)
                if self.max_idle_seconds <= 0:
                    continue
                idle = time.time() - self.last_output_ts
                if idle >= self.max_idle_seconds:
                    print(colorize(
                        f"[{now()}] [{self.name}] Watchdog：{idle:.0f}s 無輸出，嘗試終止…", "magenta", self.use_color))
                    self.result_reason = "terminated"
                    await self._terminate(grace_period=10)
                    break
        except asyncio.CancelledError:
            pass

    async def _terminate(self, grace_period: int = 8):
        if not self.proc or self.proc.returncode is not None:
            return
        try:
            if os.name == "nt":
                self.proc.terminate()
            else:
                self.proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=grace_period)
        except asyncio.TimeoutError:
            print(colorize(
                f"[{now()}] [{self.name}] 逾時未結束，強制中止（SIGKILL/terminate）", "red", self.use_color))
            self.result_reason = "killed"
            with contextlib.suppress(Exception):
                if os.name == "nt":
                    self.proc.kill()
                else:
                    self.proc.send_signal("signal.SIGKILL")

# ──────────────────────────────────────────────────────────────────────────────


async def run_parallel(runners: List[Runner]) -> List[RunResult]:
    tasks = [asyncio.create_task(r.run()) for r in runners]
    results: List[RunResult] = []
    for t in asyncio.as_completed(tasks):
        with contextlib.suppress(Exception):
            results.append(await t)
    return results


async def run_sequential(runners: List[Runner]) -> List[RunResult]:
    results: List[RunResult] = []
    for r in runners:
        results.append(await r.run())
    return results


def summarize(results: List[RunResult], use_color: bool = True) -> int:
    print(colorize("\n──────────────── 結束彙整 ────────────────", "cyan", use_color))
    code = 0
    for r in sorted(results, key=lambda x: x.name):
        dur = f"{r.duration:.1f}s"
        if r.returncode == 0 and r.reason == "normal":
            print(
                colorize(f"✅ {r.name:<3} 完成  | 用時 {dur}", "green", use_color))
        else:
            code = 1
            rc = "None" if r.returncode is None else str(r.returncode)
            print(colorize(
                f"⚠️  {r.name:<3} 異常  | exit={rc:<3} reason={r.reason:<12} | 用時 {dur}", "yellow", use_color))
    return code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run PTS/MOI/CNA/EY crawlers in parallel.")
    p.add_argument("--only", type=str, default="",
                   help="只跑指定清單（逗號分隔）：PTS,MOI,CNA,EY")
    p.add_argument("--sequential", action="store_true", help="改為逐一執行（預設並行）")
    p.add_argument("--max-idle-seconds", type=int, default=0,
                   help="子程序最多可無輸出秒數；超過將被終止（0=停用）")
    p.add_argument("--no-color", action="store_true", help="停用 ANSI 色彩輸出")
    p.add_argument("--pts-args", type=str, default="", help="傳給 PTS 的參數字串")
    p.add_argument("--moi-args", type=str, default="", help="傳給 MOI 的參數字串")
    p.add_argument("--cna-args", type=str, default="", help="傳給 CNA 的參數字串")
    p.add_argument("--ey-args", type=str, default="", help="傳給 EY 的參數字串")
    return p.parse_args()


def build_runners(args: argparse.Namespace) -> List[Runner]:
    allow = set([k.strip().upper() for k in args.only.split(",")
                if k.strip()]) if args.only else set(CRAWLERS.keys())
    use_color = not args.no_color

    mapping_args = {
        "PTS": shlex.split(args.pts_args) if args.pts_args else [],
        "MOI": shlex.split(args.moi_args) if args.moi_args else [],
        "CNA": shlex.split(args.cna_args) if args.cna_args else [],
        "EY":  shlex.split(args.ey_args) if args.ey_args else [],
    }

    # 基礎環境（共用）：傳 PROJECT_ROOT + PYTHONPATH
    base_env = os.environ.copy()
    base_env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    src_path = str(PROJECT_ROOT / "src")
    base_env["PYTHONPATH"] = f"{src_path}{os.pathsep}{base_env.get('PYTHONPATH', '')}"

    runners: List[Runner] = []
    for name, script in CRAWLERS.items():
        if name not in allow:
            continue
        env = base_env.copy()
        # 如需個別定製（例如 webdriver-manager 快取），可在此追加
        if name == "CNA":
            env.setdefault("WDM_LOCAL", "1")
            env.setdefault("WDM_CACHE", str(PROJECT_ROOT / ".wdm-cache" / "cna"))
        if name == "EY":
            # 若 EY 也用到 selenium/webdriver-manager，可使用獨立快取
            env.setdefault("WDM_LOCAL", "1")
            env.setdefault("WDM_CACHE", str(PROJECT_ROOT / ".wdm-cache" / "ey"))
        runners.append(Runner(
            name=name,
            script=script,
            extra_args=mapping_args.get(name, []),
            cwd=PROJECT_ROOT,        # ← 關鍵：鎖 CWD 在專案根，確保相對路徑寫進 data/
            env=env,
            max_idle_seconds=args.max_idle_seconds,
            use_color=use_color,
        ))
    return runners


def main():
    args = parse_args()
    runners = build_runners(args)
    if not runners:
        print("未選擇任何爬蟲。請使用 --only PTS,MOI,CNA,EY 指定。")
        sys.exit(2)

    print(colorize(
        f"[{now()}] 啟動管線；模式：{'逐一' if args.sequential else '並行'}；工作目錄：{PROJECT_ROOT}", "cyan", not args.no_color))

    try:
        if args.sequential:
            results = asyncio.run(run_sequential(runners))
        else:
            results = asyncio.run(run_parallel(runners))
    except KeyboardInterrupt:
        print(colorize("\n收到中斷（Ctrl+C）。嘗試結束子程序…", "magenta", not args.no_color))
        results = [RunResult(r.name, None, time.time(),
                             time.time(), "interrupted") for r in runners]

    sys.exit(summarize(results, use_color=not args.no_color))


if __name__ == "__main__":
    main()