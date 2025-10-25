"""
PG 檢索工具（CSV → Property Graph）。

自我測試：
python -m src.qa.verifier.agent.tools.pg_tool
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List

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

# 快取 JSON 預設位置（僅供 retriever 使用）
CACHE_DIR = "data/processed/knowledge-graph"
JSON_CACHE_PATH = os.path.join(CACHE_DIR, "pg_index.json")

_RET_PG = None


def _try_load_pg() -> Any:
    """嘗試載入 CSV Property Graph 檢索器（JSON-only）。"""
    global _RET_PG
    if _RET_PG is not None:
        return _RET_PG

    if os.getenv("ENABLE_LI_PG", "1").lower() not in ("1", "true", "yes"):
        _dlog("pg_search: disabled by env ENABLE_LI_PG")
        _RET_PG = None
        return _RET_PG

    try:
        from ....tools.property_graph.li_csv_pg_retriever import (
            CsvPropertyGraphRetriever,
        )

        idx_json = os.getenv("LI_PG_INDEX_JSON", JSON_CACHE_PATH)

        retr = None
        if Path(idx_json).is_file():
            try:
                retr = CsvPropertyGraphRetriever.load_from_json(idx_json=idx_json)
                _dlog("pg_search: loaded JSON index json=True")
            except Exception:
                _dlog("pg_search: load_from_json failed\n" + traceback.format_exc())

        if retr is None:
            retr = CsvPropertyGraphRetriever.ensure_built_and_loaded_json()
            _dlog("pg_search: ensure_built_and_loaded_json ok")

        _RET_PG = retr
        return _RET_PG
    except Exception:
        _dlog("pg_search: failed to load retriever\n" + traceback.format_exc())
        _RET_PG = None
        return _RET_PG


@tool(
    "pg_search",
    return_direct=False,
    description=(
        "CSV→Property Graph 檢索（支援多跳）。"
        "輸入三元組 JSON、top_k、hops；輸出 {'lines': [...]}。"
    ),
)
def tool_pg_search(triple_json: str, top_k: int = 50, hops: int = 3) -> str:
    """工具：PG 檢索（正反雙向），自動處理方向顛倒情境。"""
    retriever = _try_load_pg()
    if retriever is None:
        hint = (
            "PG retriever not available. 請確認 .env 的 ENABLE_LI_PG=1、LI_PG_INDEX_JSON 路徑，"
            "或先執行：python -m src.qa.preliminary_work.build_csv_property_graph build <csv>"
        )
        _dlog("pg_search: unavailable\n" + hint)
        return json.dumps(
            {"lines": [], "hits": 0, "error": "pg_unavailable", "note": hint},
            ensure_ascii=False,
        )

    try:
        tp = norm_triple_dict(take_first_triple(tp_raw))
        # --- 讀取上限（允許收斂）---
        try:
            env_cap = int(os.getenv("LI_PG_TOPK", "0"))       # 全域上限（0=不限制）
        except Exception:
            env_cap = 0
        try:
            seed_cap = int(os.getenv("ORG_PG_TOPK", "10"))    # 機構種子上限（預設 10）
        except Exception:
            seed_cap = 10

        is_org_seed = bool(tp.get("head")) and not (tp.get("relation") or tp.get("tail"))
        eff_topk = seed_cap if is_org_seed else top_k

        if env_cap > 0:
            eff_topk = min(eff_topk, env_cap)

        _dlog(f"pg_search(effective): top_k={eff_topk}, hops={hops}")
    except Exception:
        pass

    try:
        env_hops = int(os.getenv("LI_PG_HOPS", str(hops)))
        hops = max(hops, env_hops)
    except Exception:
        pass

    # 除錯（和舊版行為一致）
    _dlog(f"pg_search(effective): top_k={top_k}, hops={hops}")

    try:
        try:
            tp_raw = parse_json_safely(triple_json)
        except Exception as e:
            preview = (triple_json or "").strip().replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + " ...<truncated>"
            _dlog(f"pg_search: invalid triple_json: {e}; preview={preview}")
            return json.dumps(
                {
                    "lines": [],
                    "hits": 0,
                    "error": "pg_invalid_json",
                    "note": str(e),
                    "preview": preview,
                },
                ensure_ascii=False,
            )

        tp = norm_triple_dict(take_first_triple(tp_raw))
        # ② 構造反向（head/tail 互換）
        tp_inv = {
            "head": tp.get("tail", ""),
            "relation": tp.get("relation", ""),
            "tail": tp.get("head", ""),
        }
        
        # --- ORG_SEED（無向 OR）偵測與日誌 ---
        def _is_org_seed(tri: dict) -> bool:
            try:
                h = (tri.get("head") or "").strip()
                r = (tri.get("relation") or "").strip()
                t = (tri.get("tail") or "").strip()
                return (r == "") and bool(h) ^ bool(t)
            except Exception:
                return False

        if _is_org_seed(tp):
            org_mode = os.getenv("ORG_SEED_MODE", "UNDIRECTED")
            org_min_should = os.getenv("ORG_SEED_MIN_SHOULD", "1")
            org_use_rel = os.getenv("ORG_SEED_USE_REL", "0")
            org_allow_inv = os.getenv("ORG_SEED_ALLOW_INVERT", "0")
            org_val = (tp.get("head") or tp.get("tail") or "").strip()
            effective = {
                "head": org_val,
                "relation": (org_val if str(org_use_rel).lower() in {"1","true","yes","y"} else ""),
                "tail": org_val,
            }
            _dlog(
                "pg_search: ORG_SEED detected "
                f"(MODE={org_mode}, MIN_SHOULD={org_min_should}, USE_REL={org_use_rel}, ALLOW_INVERT={org_allow_inv}) | "
                f"original={json.dumps(tp, ensure_ascii=False)} | effective={json.dumps(effective, ensure_ascii=False)}"
            )
            
        # 原向
        hits_norm = retriever.search_triple(tp, top_k=top_k, hops=hops)
        hits_inv: list = []
        
        # 反向（避免被動語態方向顛倒）
        hits_inv = []
        if tp.get("head") != tp.get("tail"):
            tp_inv = {"head": tp.get("tail", ""), "relation": tp.get("relation", ""), "tail": tp.get("head", "")}
            hits_inv = retriever.search_triple(tp_inv, top_k=top_k, hops=hops)

        hits_all = norm_hits(hits_norm + hits_inv)
        lines = collect_hits_to_lines(hits_all)
        items = items_from_hits(hits_all, src="pg")
        unique_lines = list(dict.fromkeys(lines))

        _dlog(
            "pg_search: summary | "
            f"hits_norm={len(hits_norm)}, hits_inv={len(hits_inv)}, "
            f"unique={len(unique_lines)}, hops={hops}, top_k={top_k}"
        )
        return json.dumps(
            {"lines": unique_lines, "items": items, "hits": len(unique_lines)},
            ensure_ascii=False,
        )
    except Exception:
        _dlog("pg_search: exception\n" + traceback.format_exc())
        return json.dumps(
            {
                "lines": [],
                "hits": 0,
                "error": "pg_exception",
                "note": "pg_search failed",
            },
            ensure_ascii=False,
        )


def run_self_test() -> bool:
    """
    執行 Property Graph 工具自我檢測。

    1. 檢查 'ENABLE_LI_PG' 環境變數是否被正確讀取。
    2. 檢查 'LI_PG_INDEX_JSON' 環境變數，並確認索引檔案是否存在。
    3. 測試 _try_load_pg() 是否能根據環境變數正確啟用或禁用。
    4. 如果啟用，測試延遲匯入和 CsvPropertyGraphRetriever 實例化是否成功。

    Returns:
        bool: 如果所有檢查都通過，則回傳 True，否則回傳 False。
    """
    print("=" * 30)
    print("Running Property Graph Tool Self-Test...")
    print("=" * 30)

    all_ok = True

    # 1. 檢查環境變數
    print("[Checking Environment Variables]")
    env_enable = "ENABLE_LI_PG"
    val_enable = os.getenv(env_enable, "1")  # 預設為 "1"
    is_enabled = val_enable.lower() in ("1", "true", "yes")
    print(
        f"  [INFO] '{env_enable}' is '{val_enable}' "
        f"(Status: {'ENABLED' if is_enabled else 'DISABLED'})"
    )

    env_path_var = "LI_PG_INDEX_JSON"
    val_path = os.getenv(env_path_var)
    # 函數內部邏輯：如果 env 未設定，則使用 JSON_CACHE_PATH
    actual_path = val_path or JSON_CACHE_PATH

    if val_path:
        print(f"  [OK] '{env_path_var}' is set. Using: {val_path}")
    else:
        print(f"  [INFO] '{env_path_var}' not set. Using default: {JSON_CACHE_PATH}")

    # 檢查其他可選參數
    for var in ["LI_PG_TOPK", "LI_PG_HOPS"]:
        if os.getenv(var):
            print(f"  [INFO] Optional param '{var}' is set to: {os.getenv(var)}")

    # 2. 檢查檔案系統依賴
    print("\n[Checking File Dependencies]")
    resolved_path = Path(actual_path).resolve()
    if resolved_path.is_file():
        print(f"  [OK] Index file found at: {resolved_path}")
    else:
        print(f"  [WARN] Index file NOT FOUND at: {resolved_path}")
        if is_enabled:
            print(
                "         (Retriever will try to build it via "
                "'ensure_built_and_loaded_json()')"
            )
        # 注意：這是一個警告，因為 'ensure_built' 可能會自動建立它

    # 3. 測試延遲匯入與實例化
    print("\n[Checking Dependencies & Instantiation]")
    global _RET_PG
    _RET_PG = None  # 重設快取以進行乾淨測試
    print("  [INFO] Global cache _RET_PG has been reset for test.")

    try:
        retriever_instance = _try_load_pg()

        if is_enabled:
            if retriever_instance is not None:
                print(
                    "  [OK] _try_load_pg() succeeded (Enabled and "
                    "instantiated/loaded successfully)."
                )
            else:
                print(
                    "  [ERROR] _try_load_pg() returned None even though it is ENABLED."
                )
                print(
                    "          This implies an error during import or "
                    "(if file was missing) build failure."
                )
                print(
                    "          Check import paths "
                    "(....tools.property_graph.li_csv_pg_retriever) "
                    "or run the build script manually."
                )
                all_ok = False
        else:  # (if not is_enabled)
            if retriever_instance is None:
                print("  [OK] _try_load_pg() correctly returned None (Disabled).")
            else:
                print(
                    "  [ERROR] _try_load_pg() returned an instance "
                    "even though it is DISABLED."
                )
                all_ok = False
    except ImportError as e:
        print(f"  [ERROR] _try_load_pg() failed on import: {e}")
        print(
            "          (Check relative path "
            "'....tools.property_graph.li_csv_pg_retriever')"
        )
        all_ok = False
    except Exception as e:
        print(f"  [ERROR] _try_load_pg() raised an unexpected error: {e}")
        print(
            "          (This might be a build failure from "
            "'ensure_built_and_loaded_json()')"
        )
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