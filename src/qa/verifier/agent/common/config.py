"""共用設定與日誌工具。"""

from __future__ import annotations

import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
load_dotenv(override=True)

try:
    from ...core.paths import RES_DIR as _RES_DIR, USER_INPUT_DIR as _USER_INPUT_DIR
except ImportError:
    print(
        "Warning: Could not import core.paths. Using fallback paths. "
        "Run 'run_self_test()' for details."
    )
    _RES_DIR = Path("data/resources")
    _USER_INPUT_DIR = Path("data/input")


# 路徑/編碼
OUTPUT_ENCODING: Final[str] = "utf-8-sig"
RES_DIR: Final[Path] = _RES_DIR
USER_INPUT_DIR: Final[Path] = _USER_INPUT_DIR

# 除錯設定
VERIFIER_DEBUG: bool = os.getenv("VERIFIER_DEBUG", "0") == "1"
VERIFIER_DEBUG_PATH: Path = Path(
    os.getenv(
        "VERIFIER_DEBUG_PATH", "data/processed/verifier/debug/verifier_search_debug.log"
    )
)
VERIFIER_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)

# 代理行為參數
MAX_AGENT_CHARS: int = int(os.getenv("MAX_AGENT_CHARS", "15000"))
AGENT_MAX_STEPS: int = int(os.getenv("AGENT_MAX_STEPS", "24"))
AGENT_RECURSION_FACTOR: int = int(os.getenv("AGENT_RECURSION_FACTOR", "3"))
AGENT_RECURSION_EXTRA: int = int(os.getenv("AGENT_RECURSION_EXTRA", "3"))
RECURSION_LIMIT: int = max(
    AGENT_MAX_STEPS * AGENT_RECURSION_FACTOR + AGENT_RECURSION_EXTRA, 30
)

AGENT_TOTAL_TARGET: int = int(os.getenv("AGENT_TOTAL_TARGET", "0"))
AGENT_MIN_PER_TRIPLE: int = int(os.getenv("AGENT_MIN_PER_TRIPLE", "0"))
AGENT_NO_NEW_PATIENCE: int = int(os.getenv("AGENT_NO_NEW_PATIENCE", "5"))
AGENT_TOP_K_MAX: int = int(os.getenv("AGENT_TOP_K_MAX", "200"))
MAX_EVID_CHARS: int = int(os.getenv("MAX_EVID_CHARS", "300"))

ENABLE_VECTOR_FALLBACK: bool = os.getenv("ENABLE_VECTOR_FALLBACK", "1").lower() in ("1","true","yes","y")

# 當 PG 命中條數 < 這個數時，且 ENABLE_VECTOR_FALLBACK=1，才觸發向量檢索
VECTOR_TRIGGER_MIN: int = int(os.getenv("VECTOR_TRIGGER_MIN", "10"))

# 向量檢索 top_k；沒有就沿用工具內預設
VECTOR_TOPK: int | None = int(os.getenv("VECTOR_TOPK")) if os.getenv("VECTOR_TOPK") else None

# ORG 種子
ENABLE_ORG_SEEDS: bool = os.getenv("ENABLE_ORG_SEEDS", "1").lower() in (
    "1",
    "true",
    "yes",
    "y",
)
MAX_ORG_SEEDS: int = int(os.getenv("MAX_ORG_SEEDS", "8") or "8")
ORG_SEED_ONEPASS: bool = os.getenv("ORG_SEED_ONEPASS", "1").lower() in (
    "1",
    "true",
    "yes",
    "y",
)

# 模型環境
os.environ.setdefault("ACCELERATE_DISABLE_DEVICE_MAP", "1")
os.environ.setdefault("TRANSFORMERS_NO_ACCELERATE", "1")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Run-ID
CURRENT_RUN_ID: str = (
    os.getenv("NEWS_RUN_ID", "").strip()
    or datetime.now().strftime("run_%Y%m%d_%H%M%S")
)


def set_current_run_id_from_input(input_path_or_id: str) -> None:
    """依檔名/識別碼推導 Run ID 並寫回環境變數。

    Args:
        input_path_or_id: 檔名或識別碼（例如 t_1022.txt 或 t_1022）。
    """
    global CURRENT_RUN_ID
    base = Path(input_path_or_id).stem
    m = re.search(r"(t_\d+)", base)
    rid = m.group(1) if m else base
    os.environ["NEWS_RUN_ID"] = rid
    CURRENT_RUN_ID = rid


def _dlog(msg: str) -> None:
    """寫入除錯日誌（失敗時靜默）。

    Args:
        msg: 訊息內容。
    """
    if not VERIFIER_DEBUG:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(VERIFIER_DEBUG_PATH, "a", encoding="utf-8-sig") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def get_tool_availability() -> str:
    """回報工具可用性摘要字串。

    Returns:
        類似 "pg_search=available; neo4j_search=UNAVAILABLE; vector_search=available"。
    """
    try:
        from ..tools.pg_tool import _try_load_pg  # 避免頂層循環匯入
        from ..tools.neo4j_tool import _try_load_neo4j

        pg_ok = _try_load_pg() is not None
        neo_ok = _try_load_neo4j() is not None
    except ImportError:
        print(
            "Warning: Tool imports failed. Reporting tools as UNAVAILABLE. "
            "Run 'run_self_test()' for details."
        )
        pg_ok = False
        neo_ok = False

    vec_ok = os.getenv("ENABLE_VECTOR_FALLBACK", "1").lower() in ("1", "true", "yes")

    return (
        f"pg_search={'available' if pg_ok else 'UNAVAILABLE'}; "
        f"neo4j_search={'available' if neo_ok else 'UNAVAILABLE'}; "
        f"vector_search={'available' if vec_ok else 'UNAVAILABLE'}"
    )


def run_self_test() -> bool:
    """
    執行環境變數自我檢測。

    檢查所有在此模組中使用的環境變數是否已在環境中設定（例如透過 .env 檔案）。
    如果變數未設定，將會印出警告，告知目前使用的是程式碼中的預設值。

    Returns:
        bool: 如果所有變數都已在環境中明確設定，則回傳 True，否則回傳 False。
    """
    print("=" * 30)
    print("Running Configuration Self-Test...")
    print("=" * 30)

    # 這些變數是在 getenv 時提供了預設值
    # 檢查 os.getenv(KEY) 是否為 None 來判斷 .env 是否有設定
    vars_with_defaults = {
        "VERIFIER_DEBUG": "0",
        "VERIFIER_DEBUG_PATH": (
            "data/processed/verifier/debug/verifier_search_debug.log"
        ),
        "MAX_AGENT_CHARS": "15000",
        "AGENT_MAX_STEPS": "24",
        "AGENT_RECURSION_FACTOR": "3",
        "AGENT_RECURSION_EXTRA": "3",
        "AGENT_TOTAL_TARGET": "0",
        "AGENT_MIN_PER_TRIPLE": "0",
        "AGENT_NO_NEW_PATIENCE": "5",
        "AGENT_TOP_K_MAX": "200",
        "MAX_EVID_CHARS": "300",
        "ENABLE_ORG_SEEDS": "1",
        "MAX_ORG_SEEDS": "8",
        "ORG_SEED_ONEPASS": "1",
        "OPENAI_CHAT_MODEL": "gpt-4o-mini",
        "ENABLE_VECTOR_FALLBACK": "1",  # 用於 get_tool_availability
    }

    # 這些變數是動態設定或有特殊邏輯
    special_vars = {
        "NEWS_RUN_ID": "Dynamic (e.g., run_20251025_133000)",
        "ACCELERATE_DISABLE_DEVICE_MAP": "1 (setdefault)",
        "TRANSFORMERS_NO_ACCELERATE": "1 (setdefault)",
    }

    all_vars_set = True
    print("[Checking getenv() variables with defaults]")
    for var, default_val in vars_with_defaults.items():
        value = os.getenv(var)
        if value is None:
            all_vars_set = False
            print(f"  [WARNING] '{var}' not set. Using code default: '{default_val}'")
        else:
            print(f"  [OK] '{var}' is set to: '{value}'")

    print("\n[Checking special/setdefault variables]")
    for var, default_info in special_vars.items():
        value = os.getenv(var)
        if value is None:
            # 雖然 setdefault 會補上，但 .env 沒設就是沒設
            if "setdefault" in default_info:
                all_vars_set = False 
                print(
                    f"  [INFO] '{var}' not set in .env. Using default: {default_info}"
                )
            # NEWS_RUN_ID 是動態的，未設定是正常的
            elif var == "NEWS_RUN_ID":
                print(
                    f"  [INFO] '{var}' not set. Using dynamic default: {default_info}"
                )
        else:
            print(f"  [OK] '{var}' is set to: '{value}'")
    
    print("\n[Checking Tool Dependencies]")
    print(f"  Tools Status: {get_tool_availability()}")
    if "UNAVAILABLE" in get_tool_availability():
        print(
            "  [INFO] One or more tools are unavailable. "
            "Check connections (DB, Neo4j) or '.env' settings (e.g., ENABLE_VECTOR_FALLBACK)."
        )


    print("-" * 30)
    if all_vars_set:
        print("Self-Test Result: ALL CLEAR. All variables are explicitly set.")
    else:
        print("Self-Test Result: WARNINGS found. Some variables use code defaults.")
    print("=" * 30)

    return all_vars_set


if __name__ == "__main__":
    print("Module loaded directly. Running self-test...")
    print(f"Loading .env from: {Path('.env').resolve()}")
    run_self_test()