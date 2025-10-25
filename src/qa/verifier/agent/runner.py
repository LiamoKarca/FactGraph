"""
Agent runner（入口點）

目的：
    - 對外提供 run_retrieval / run_factcheck_middle
    - 串接 LangGraph（graph.build.build_graph）並執行 ReAct 迴圈
    - 將最終輸出整理為「[原始文本] + [比對知識]」
    - ★ 補強：在執行前印出「環境＆工具自我檢測（env/tool snapshot）」

本版遵循舊 agent_langchain.py 的關鍵行為：
    1) 訊息萃取的去重採「字串保序去重」（不做語義併合）
    2) fallback 路徑亦採用相同的保序去重
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from tqdm import tqdm

# 以 override=True 載入 .env（若系統已有同名變數，以 .env 覆蓋）
# 參考：python-dotenv 官方說明 load_dotenv(override=True)
load_dotenv(override=True)

# 延遲載入，避免頂層循環依賴
_CONFIG = None
_GRAPH = None
_TOOLS = None
_JSON = None
_FMT = None
_EXTRACT = None


def _lazy_load():
    """Lazy import project modules to avoid circular deps at import-time."""
    global _CONFIG, _GRAPH, _TOOLS, _JSON, _FMT, _EXTRACT
    if _CONFIG is None:
        from .common import config as _CONFIG  # type: ignore
        globals()["_CONFIG"] = _CONFIG
    if _GRAPH is None:
        from .graph import build as _GRAPH  # type: ignore
        globals()["_GRAPH"] = _GRAPH
    if _TOOLS is None:
        from . import tools as _TOOLS  # type: ignore
        globals()["_TOOLS"] = _TOOLS
    if _JSON is None:
        from .common import json_utils as _JSON  # type: ignore
        globals()["_JSON"] = _JSON
    if _FMT is None:
        from .common import formatting as _FMT  # type: ignore
        globals()["_FMT"] = _FMT
    if _EXTRACT is None:
        from ..llm import extract as _EXTRACT  # type: ignore
        globals()["_EXTRACT"] = _EXTRACT


def _keep_order_dedup(lines: List[str]) -> List[str]:
    """Deduplicate while keeping first-appearance order (string equality)."""
    seen = set()
    out: List[str] = []
    for s in lines:
        s2 = str(s or "").strip()
        if not s2 or s2 in seen:
            continue
        seen.add(s2)
        out.append(s2)
    return out


def _extract_kb_lines_from_messages(messages: List[BaseMessage]) -> List[str]:
    """由 Agent/Tool 訊息聚合出 `[比對]` 行（回復舊版保序去重）。"""
    _lazy_load()
    text = ""

    # 聚合可見文字（LangChain message 可能是 content-block list；需逐塊取 text）
    for m in messages[::-1]:
        if hasattr(m, "content"):
            if isinstance(m.content, str):
                text += "\n" + m.content
            elif isinstance(m.content, list):
                for c in m.content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += "\n" + (c.get("text") or "")

    lines: List[str] = []

    # ① AI 直接說出的 [比對] 行（或前面被加了 [n] 前綴的）
    for raw in text.splitlines():
        s = (raw or "").strip()
        if s.startswith("[比對]"):
            lines.append(s)
        elif re.match(r"^\[\d+\]\s*\[比對\]", s):
            lines.append(re.sub(r"^\[\d+\]\s*", "", s))

    # ② tools 以 JSON 回的 {"lines":[...]}（ToolMessage.content 可能是字串 JSON）
    for m in messages[::-1]:
        raw = ""
        if hasattr(m, "content"):
            raw = m.content if isinstance(m.content, str) else ""
        raw = (raw or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and isinstance(obj.get("lines"), list):
                for s in obj["lines"]:
                    s = (s or "").strip()
                    if not s:
                        continue
                    lines.append(s if s.startswith("[比對]") else f"[比對] {s}")
        except Exception:
            pass

    return _keep_order_dedup(lines)  # 舊版保序去重


# ------------------- 自我檢測（新增） ------------------- #
def _print_env_and_tool_snapshot():
    """列印環境變數與實際生效值、工具可用性、以及執行參數。"""
    _lazy_load()
    print("\n========== 環境＆工具自我檢測（env/tool snapshot）==========")

    # 1) .env 路徑與是否存在
    env_path = Path(".env").resolve()
    print(f"• .env 路徑：{env_path} | 存在：{env_path.exists()}")

    # 2) 關鍵 .env → 直接讀 os.getenv（原樣顯示）
    env_keys = [
        # 工具可用與策略
        "ENABLE_VECTOR_FALLBACK",
        "ENABLE_ORG_SEEDS",
        "ENABLE_LI_ONLINE",
        # PG / Neo4j / Vector 重要參數
        "LI_PG_TOPK", "LI_PG_HOPS", "ORG_PG_TOPK",
        "VECTOR_TOPK", "ORG_VECTOR_TOPK",
        # Agent 控制
        "AGENT_TOP_K_MAX", "AGENT_MAX_STEPS",
        "AGENT_NO_NEW_PATIENCE", "AGENT_MIN_PER_TRIPLE", "AGENT_TOTAL_TARGET",
        "MAX_EVID_CHARS", "MAX_AGENT_CHARS",
        # OpenAI
        "OPENAI_CHAT_MODEL", "OPENAI_TEMPERATURE", "OPENAI_MAX_TOKENS",
    ]
    for k in env_keys:
        print(f"  - {k}={os.getenv(k)}")

    # 3) 實際生效（import 後的 config 內數值）
    print("\n• 實際生效（config.py 導入後的值）")
    print(f"  - AGENT_TOP_K_MAX={_CONFIG.AGENT_TOP_K_MAX}")
    print(f"  - AGENT_MAX_STEPS={_CONFIG.AGENT_MAX_STEPS} | AGENT_NO_NEW_PATIENCE={_CONFIG.AGENT_NO_NEW_PATIENCE}")
    print(f"  - AGENT_MIN_PER_TRIPLE={_CONFIG.AGENT_MIN_PER_TRIPLE} | AGENT_TOTAL_TARGET={_CONFIG.AGENT_TOTAL_TARGET}")
    print(f"  - MAX_EVID_CHARS={_CONFIG.MAX_EVID_CHARS} | MAX_AGENT_CHARS={_CONFIG.MAX_AGENT_CHARS}")
    print(f"  - RECURSION_LIMIT={getattr(_CONFIG, 'RECURSION_LIMIT', None)}")
    print(f"  - OUTPUT_ENCODING={_CONFIG.OUTPUT_ENCODING}")

    # 4) 工具可用性（get_tool_availability）
    try:
        availability = _CONFIG.get_tool_availability()
    except Exception as e:
        availability = f"<error: {e}>"
    print("\n• 工具可用性（get_tool_availability）")
    print(availability)

    # 5) PG/Vector 工具的「即時有效 top_k/hops」檢測（若模組提供）
    try:
        from .tools.pg_tool import DEFAULT_PG_TOPK  # type: ignore
        print(f"\n• PG 預設 top_k（模組內定）：{DEFAULT_PG_TOPK}")
    except Exception:
        pass
    try:
        from .tools.vector_tool import DEFAULT_VECTOR_TOPK  # type: ignore
        print(f"• Vector 預設 top_k（模組內定）：{DEFAULT_VECTOR_TOPK}")
    except Exception:
        pass

    # 6) LLM 參數
    print("\n• LLM 模型參數（將由 state.build_llm 讀取）")
    print(f"  - OPENAI_CHAT_MODEL={os.getenv('OPENAI_CHAT_MODEL')} | OPENAI_TEMPERATURE={os.getenv('OPENAI_TEMPERATURE')} | OPENAI_MAX_TOKENS={os.getenv('OPENAI_MAX_TOKENS')}")

    print("========================================================\n")


def _stream_graph(graph, payload: dict) -> List[str]:
    """執行 LangGraph 並以串流顯示進度，回傳「已編號」的 [比對] 行。"""
    _lazy_load()
    bar = tqdm(total=_CONFIG.AGENT_MAX_STEPS, desc="④ Agent 檢索中（ReAct 迴圈）", leave=True)
    messages_last: List[BaseMessage] = []
    accum_last = 0

    print("──────── Agent Streaming 開始 ────────")
    for update in graph.stream(
        payload,
        config={
            "recursion_limit": payload.get("recursion_limit", 60),
            "configurable": {"thread_id": payload.get("session_id", "default")},
        },
        stream_mode="updates",  # 只回傳每個節點的 state 更新（LangGraph 官方建議）
    ):
        node = list(update.keys())[0] if update else "unknown"
        delta = update.get(node, {}) or {}

        if "messages" in delta:
            messages_last = delta["messages"]
            last = messages_last[-1] if messages_last else None
            if last is not None and hasattr(last, "tool_calls") and last.tool_calls:
                print(f"• {node}: 產生工具呼叫 → {len(last.tool_calls)} 個")
                _CONFIG._dlog(f"stream: {node} tool_calls={len(last.tool_calls)}")
            elif last is not None and last.__class__.__name__ == "ToolMessage":
                preview = str(last.content)[:120].replace("\n", " ")
                print(f"• {node}: 工具回覆片段 ← {preview}...")
            elif isinstance(last, AIMessage):
                preview = str(last.content)[:120].replace("\n", " ")
                print(f"• {node}: AI 回覆片段 ← {preview}...")

        if "accum_lines" in delta:
            cur = len(delta["accum_lines"] or [])
            if cur != accum_last:
                print(f"  ↳ 累積比對行：{cur}（+{cur - accum_last}）")
                accum_last = cur

        if "step" in delta:
            step_now = int(delta["step"] or 0)
            if step_now > bar.n:
                bar.n = min(step_now, _CONFIG.AGENT_MAX_STEPS)
                bar.refresh()
    print("──────── Agent Streaming 結束 ────────")

    # 嘗試從 finalize 的最後訊息抓 [比對知識]
    final_msg = messages_last[-1] if messages_last else AIMessage(content="")
    content = final_msg.content if isinstance(final_msg.content, str) else str(final_msg.content)

    kb_lines: List[str] = []
    take = False
    for raw in (content or "").splitlines():
        s = (raw or "").strip()
        if s == "[比對知識]":
            take = True
            continue
        if take and re.match(r"^\[\d+\]\s*", s):
            kb_lines.append(s)

    # 若 finalize 沒帶全，改用訊息聚合（回復舊版能拿到 70+ 行的關鍵）
    if not kb_lines:
        merged = _extract_kb_lines_from_messages(messages_last)
        if merged:
            # 這裡只做編號，**不**做截斷；上限裁切交由 finalize 處理
            kb_lines = [f"[{i}] {re.sub(r'^\\[\\d+\\]\\s*', '', ln)}" for i, ln in enumerate(merged, 1)]

    return kb_lines


def run_retrieval(news_text: str, *, session_id: str = "default", model: str | None = None) -> List[str]:
    """執行 ReAct 代理檢索，回傳最終 `[比對]` 行（已編號）。"""
    _lazy_load()

    # ★ 在每次 run 前印出 snapshot
    _print_env_and_tool_snapshot()

    original_len = len(news_text or "")
    if original_len > _CONFIG.MAX_AGENT_CHARS:
        news_text = (news_text or "")[:_CONFIG.MAX_AGENT_CHARS]
        _CONFIG._dlog(f"guard: truncate news_text {original_len}->{len(news_text)}")
    else:
        _CONFIG._dlog(f"guard: news_text chars={original_len}")

    user_msg = HumanMessage(content=f"輸入新聞：\n{news_text.strip()}\n\n請開始依規則執行。")
    graph = _GRAPH.build_graph()

    payload = {
        "messages": [user_msg],
        "news_text": news_text.strip(),
        "triples": [],
        "accum_lines": [],
        "step": 0,
        "no_new_steps": 0,
        "tool_availability": _CONFIG.get_tool_availability(),
        "session_id": session_id,
        "recursion_limit": getattr(_CONFIG, "RECURSION_LIMIT", 75),
    }

    try:
        kb_lines = _stream_graph(graph, payload)
    except Exception as exc:
        _CONFIG._dlog(f"Graph execution failed: {exc}")
        kb_lines = []

    if kb_lines:
        _CONFIG._dlog(f"graph_done: kb_lines={len(kb_lines)}")
        return kb_lines

    # —— 保底：抽取 → 本地向量一次性補救（與舊版一致）——
    _CONFIG._dlog("fallback: no kb lines; try one-shot vector_search")
    try:
        raw = _EXTRACT.extract_entities_relations(news_text) or ""
        obj = _JSON.parse_json_safely(raw) if raw else {}
        triples = obj.get("triples") or []
        lines: List[str] = []
        if triples:
            from itertools import islice
            from .tools.vector_tool import _vector_search_impl  # type: ignore
            for tp in islice(triples, 0, 5):
                lines.extend(_vector_search_impl(tp, top_k=50))
        # 回復舊版：字串保序去重（不做語義併合）
        lines = _keep_order_dedup(lines)
        lines = [f"[比對] {t}" if not t.startswith("[比對]") else t for t in lines]
        kept = [f"[{i}] {re.sub(r'^\\[\\d+\\]\\s*', '', ln)}" for i, ln in enumerate(lines[:min(20, _CONFIG.AGENT_TOP_K_MAX)], 1)]
        _CONFIG._dlog(f"fallback: produced {len(kept)} lines")
        return kept
    except Exception as exc:
        _CONFIG._dlog(f"fallback error: {exc}")
        return []


def run_factcheck_middle(news_text: str, *, session_id: str = "default", model: str | None = None) -> str:
    """產出中繼文本：「[原始文本] + [比對知識]」"""
    kb_lines = run_retrieval(news_text, session_id=session_id, model=model)
    return "[原始文本]\n" + news_text.strip() + "\n\n[比對知識]\n" + "\n".join(kb_lines)


# ------------------- CLI ------------------- #
def _cmd_alias(_: list[str]) -> int:
    """代理 retriever 生成/更新 alias skeleton。"""
    try:
        from ...tools.property_graph import li_csv_pg_retriever as retr_mod
        if hasattr(retr_mod, "_cmd_alias"):
            return int(retr_mod._cmd_alias([]))  # type: ignore
        print("提示：亦可直接執行 retriever 的 alias：")
        print("  python -m src.qa.preliminary_work.build_csv_property_graph alias")
        return 0
    except SystemExit as e:
        return int(e.code)
    except Exception as e:
        print("❌ alias 生成失敗：", e)
        return 1


def _cmd_pg(args: list[str]) -> int:
    """偵錯用：直接對 PG 發查詢（輸出 lines/hits）。"""
    _lazy_load()
    if not args:
        print("用法：pg '<triple_json>' [top_k]")
        return 2
    tri_raw = args[0]
    top_k = int(args[1]) if len(args) >= 2 else 50
    res = _TOOLS.tool_pg_search.func(triple_json=tri_raw, top_k=top_k, hops=int(os.getenv("LI_PG_HOPS", "3")))  # type: ignore
    print(res)
    return 0


def main() -> None:
    """命令列進入點。"""
    _lazy_load()
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "alias":
            raise SystemExit(_cmd_alias(sys.argv[2:]))
        if cmd == "pg":
            raise SystemExit(_cmd_pg(sys.argv[2:]))

        arg = sys.argv[1]
        filename = arg if arg.endswith(".txt") else f"{arg}.txt"
        path = Path(filename)
        if not path.is_file():
            path = _CONFIG.USER_INPUT_DIR / filename
        if not path.is_file():
            raise SystemExit(f"❌ 找不到檔案：{path.resolve()}")

        news_text = path.read_text(encoding=_CONFIG.OUTPUT_ENCODING).strip()
        news_id = path.stem
        _CONFIG.set_current_run_id_from_input(news_id)

        print(f"📰 開始處理檔案：{path.resolve()} (Run ID: {_CONFIG.CURRENT_RUN_ID})")
        print(f"➡️  輸出目錄：{_CONFIG.RES_DIR.resolve()}")

        middle = run_factcheck_middle(news_text, session_id=news_id)
        out_path = _CONFIG.RES_DIR / f"news_kg_{news_id}_agent.txt"
        out_path.write_text(middle, encoding=_CONFIG.OUTPUT_ENCODING)
        print(f"✅ 完成 ▶ 中間檔：{out_path.resolve()}")
        return

    print("用法：python -m src.qa.verifier.agent.runner <news_id|檔名>")
    print("      python -m src.qa.verifier.agent.runner alias")
    print("      python -m src.qa.verifier.agent.runner pg '<triple_json>' [top_k]")


if __name__ == "__main__":
    main()
