"""
Neo4j 檢索工具。

自我測試：
python -m src.qa.verifier.agent.tools.neo4j_tool
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path  # 為了自檢
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv(override=True)

from ..common.config import _dlog
from ..common.formatting import (
    collect_hits_to_lines,
    items_from_hits,
    norm_hits,
    norm_triple_dict,
)
from ..common.json_utils import parse_json_safely
from ..extract.er_utils import take_first_triple

_RET_NEO4J = None


def _try_load_neo4j() -> Any:
    """嘗試載入 Neo4j 檢索器。"""
    global _RET_NEO4J
    if _RET_NEO4J is not None:
        return _RET_NEO4J
    if os.getenv("ENABLE_LI_ONLINE", "0").lower() not in ("1", "true", "yes"):
        _RET_NEO4J = None
        _dlog("neo4j_search: disabled by env ENABLE_LI_ONLINE")
        return _RET_NEO4J
    try:
        from ...kg.llamaIndex.neo4j_li_retriever import (
            LlamaIndexNeo4jRetriever,  # type: ignore
        )

        _RET_NEO4J = LlamaIndexNeo4jRetriever(
            date_field="date", evidence_field="evidence"
        )
        return _RET_NEO4J
    except Exception:
        _dlog("neo4j_search: failed to init\n" + traceback.format_exc())
        _RET_NEO4J = None
        return _RET_NEO4J


@tool(
    "neo4j_search",
    return_direct=False,
    description=(
        "Neo4j 檢索（支援多跳）。輸入三元組 JSON、top_k、hops；輸出 {'lines': [...]。"
    ),
)
def tool_neo4j_search(triple_json: str, top_k: int = 50, hops: int = 2) -> str:
    """工具：Neo4j 檢索。"""
    retriever = _try_load_neo4j()
    if retriever is None:
        note = "Neo4j retriever not available（已關閉或連線失敗）"
        _dlog("neo4j_search: unavailable")
        return json.dumps(
            {"lines": [], "error": "neo4j_unavailable", "note": note},
            ensure_ascii=False,
        )

    try:
        try:
            tp_raw = parse_json_safely(triple_json)
        except Exception as e:
            preview = (triple_json or "").strip().replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + " ...<truncated>"
            _dlog(f"neo4j_search: invalid triple_json: {e}; preview={preview}")
            return json.dumps(
                {
                    "lines": [],
                    "error": "neo4j_invalid_json",
                    "note": str(e),
                    "preview": preview,
                },
                ensure_ascii=False,
            )

        tp = norm_triple_dict(take_first_triple(tp_raw))
        hits = retriever.search_triple(tp, top_k=top_k, hops=hops)
        hits = norm_hits(hits)
        _dlog(f"neo4j_search: hits={len(hits)}, hops={hops}, top_k={top_k}")
        return json.dumps(
            {
                "lines": collect_hits_to_lines(hits),
                "items": items_from_hits(hits, src="neo4j"),
            },
            ensure_ascii=False,
        )
    except Exception:
        _dlog("neo4j_search: search failed\n" + traceback.format_exc())
        return json.dumps(
            {"lines": [], "error": "neo4j_exception", "note": "neo4j_search failed"},
            ensure_ascii=False,
        )


def run_self_test() -> bool:
    """
    執行 Neo4j 工具自我檢測。

    1. 檢查 'ENABLE_LI_ONLINE' 環境變數是否被正確讀取。
    2. 如果啟用，檢查 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD' 是否存在。
    3. 測試 _try_load_neo4j() 是否能根據環境變數正確啟用或禁用。
    4. 如果啟用，測試延遲匯入和 LlamaIndexNeo4jRetriever 實例化是否成功。

    Returns:
        bool: 如果所有檢查都通過，則回傳 True，否則回傳 False。
    """
    print("=" * 30)
    print("Running Neo4j Tool Self-Test...")
    print("=" * 30)

    all_ok = True

    # 1. 檢查環境變數
    print("[Checking Environment Variables]")
    env_var = "ENABLE_LI_ONLINE"
    env_val = os.getenv(env_var, "0")
    is_enabled = env_val.lower() in ("1", "true", "yes")

    print(
        f"  [INFO] '{env_var}' is set to: '{env_val}' "
        f"(Status: {'ENABLED' if is_enabled else 'DISABLED'})"
    )

    # 2. 如果啟用，檢查 Neo4j 連線變數
    if is_enabled:
        print("  [INFO] Checking Neo4j connection variables (since enabled)...")
        neo4j_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        for var in neo4j_vars:
            if not os.getenv(var):
                print(
                    f"    [ERROR] '{var}' is not set. Retriever instantiation will fail."
                )
                all_ok = False
            else:
                print(f"    [OK] '{var}' is set.")

    # 3. 測試 _try_load_neo4j() 實例化
    print("\n[Checking Dependencies & Instantiation]")
    global _RET_NEO4J
    _RET_NEO4J = None  # 重設快取以進行乾淨測試
    print("  [INFO] Global cache _RET_NEO4J has been reset for test.")

    try:
        retriever_instance = _try_load_neo4j()
        
        if is_enabled:
            if retriever_instance is not None:
                print(
                    "  [OK] _try_load_neo4j() succeeded (Enabled and "
                    "instantiated successfully)."
                )
            else:
                print(
                    "  [ERROR] _try_load_neo4j() returned None even though it is ENABLED."
                )
                print(
                    "          Check import paths (LlamaIndexNeo4jRetriever) "
                    "or Neo4j connection details (see errors above)."
                )
                all_ok = False
        else:  # (if not is_enabled)
            if retriever_instance is None:
                print("  [OK] _try_load_neo4j() correctly returned None (Disabled).")
            else:
                print(
                    "  [ERROR] _try_load_neo4j() returned an instance "
                    "even though it is DISABLED."
                )
                all_ok = False
    except ImportError as e:
        print(f"  [ERROR] _try_load_neo4j() failed on import: {e}")
        print(
            "          (Is 'llama-index-graph-stores-neo4j' installed? "
            "Check relative path '..kg.llamaIndex...')"
        )
        all_ok = False
    except Exception as e:
        print(f"  [ERROR] _try_load_neo4j() raised an unexpected error: {e}")
        all_ok = False

    print("-" * 30)
    if all_ok:
        print("Self-Test Result: ALL CLEAR.")
    else:
        print("Self-Test Result: WARNINGS or ERRORS found.")
    print("=" * 30)

    return all_ok


if __name__ == "__main__":
    print("Module loaded directly. Running self-test...")
    env_path = Path(".env").resolve()
    if env_path.exists():
        print(f"Loading .env from: {env_path}")
    else:
        print(f"Warning: .env file not found at {env_path}.")
    run_self_test()