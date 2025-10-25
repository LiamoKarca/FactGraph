"""
LangGraph 各節點實作 —— 回復舊版語義去重與向量嚴格備援策略。
"""

from __future__ import annotations

import json
import os
import re
import traceback
from typing import Dict, List, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dotenv import load_dotenv
load_dotenv(override=True)

from ..common.config import (
    AGENT_MAX_STEPS,
    AGENT_MIN_PER_TRIPLE,
    AGENT_NO_NEW_PATIENCE,
    AGENT_TOP_K_MAX,
    AGENT_TOTAL_TARGET,
    _dlog,
    get_tool_availability,
    ORG_SEED_ONEPASS,
    VECTOR_TRIGGER_MIN,
    VECTOR_TOPK,
    ENABLE_VECTOR_FALLBACK
)

from ..common.formatting import deduplicate as _dedup_keep_order
from ..common.json_utils import parse_json_safely
from ..extract.er_utils import er_to_triples
from ..extract.org_seeds import extract_org_seeds
from .state import AGENT_SYS_PROMPT, AgentState, build_llm
from ..tools import TOOLS
from ..tools.pg_tool import tool_pg_search
from ..tools.vector_tool import tool_vector_search
from ..tools.merge_tool import tool_merge_and_dedup

def extract_triples_node(state: AgentState) -> Dict:
    """抽取三元組 + 機構保底種子（與舊版一致）"""
    _dlog("node: extract_triples_node executing...")
    news_text = state.get("news_text", "") or ""
    triples: List[Dict[str, str]] = []
    try:
        from ...llm.extract import extract_entities_relations
        raw = extract_entities_relations(news_text) or ""
        try:
            obj = parse_json_safely(raw) if raw else {}
        except Exception:
            obj = {}
        triples = er_to_triples(obj)
    except Exception:
        _dlog("node: extract_triples_node failed\n" + traceback.format_exc())

    # 機構保底種子（保持舊版邏輯）
    org_triples: List[Dict[str, str]] = []
    try:
        orgs = extract_org_seeds(news_text)
        for org in orgs:
            s = (org or "").strip()
            if not s:
                continue
            if ORG_SEED_ONEPASS:
                # 舊版一條（交給 PG 檢索器做無向 OR）
                org_triples.append({"head": s, "relation": "", "tail": ""})
            else:
                # 舊版雙條：head 與 tail 各一條
                org_triples.append({"head": s, "relation": "", "tail": ""})
                org_triples.append({"head": "", "relation": "", "tail": s})
        if org_triples:
            _dlog(f"org_seeds: names={len(orgs)} -> triples={len(org_triples)} | {orgs}")
    except Exception:
        _dlog("org_seeds: failed\n" + traceback.format_exc())

    merged_triples = (org_triples + triples) if org_triples else triples
    _dlog(
        f"node: extract_triples_node found {len(triples)} triples; "
        f"merged={len(merged_triples)} (with org seeds)."
    )
    return {
        "triples": merged_triples,
        "triples_processed_index": 0,
        # 追蹤上一輪各工具命中數，供提示詞嚴格規訓向量備援
        "last_pg_lines": 0,
        "last_neo_lines": 0,
        "last_vec_lines": 0,
    }


def agent_node(state: AgentState) -> Dict:
    """代理產生工具呼叫：先 PG/Neo4j；若上一輪被標記 force_vector_next，就直接用 vector_search。"""
    triples = state.get("triples", []) or []
    idx = int(state.get("triples_processed_index") or 0)
    if not triples or idx >= len(triples):
        return {"messages": state.get("messages", []) + [AIMessage(content="No more triples.")]}

    current_triple = triples[idx]
    current_triple_json = json.dumps(current_triple, ensure_ascii=False)

    llm = build_llm().bind_tools(TOOLS, tool_choice="any")
    availability = state.get("tool_availability") or get_tool_availability()

    notes = state.get("notes") or ""
    sys_prompt = (
        AGENT_SYS_PROMPT
        + "\n\n【工具可用性】" + availability
        + "\n【目前統計】\n" + notes
    )

    if state.get("force_vector_next", False):
        # 上一輪 PG/Neo4j 0 命中 → 同一個 triple 改跑向量
        user_msg_content = (
            "上一輪 PG/Neo4j 皆 0 命中。請針對『同一個三元組』直接呼叫 vector_search，"
            "參數可用 top_k=100。\n"
            + current_triple_json
        )
    else:
        # 正常情況：指示先用 PG（或 Neo4j），不要同輪同時跑 vector
        user_msg_content = (
            "請為這個三元組檢索（先 pg_search 或 neo4j_search；若 0 命中我會要求你下一輪改用 vector）：\n"
            + current_triple_json
            + "\n請依規則選擇工具並執行。"
        )

    response = llm.invoke(
        [
            {"role": "system", "content": sys_prompt},
            HumanMessage(content=user_msg_content),
        ]
    )

    # ★ 關鍵：這裡不再前進索引；是否前進由 accumulate_node 根據命中結果與工具種類決定
    return {
        "messages": state.get("messages", []) + [response],
        "tool_availability": availability,
        "triples_processed_index": idx,
    }


def _parse_tool_payload_from_messages(messages: List) -> Dict[str, List[str]]:
    """從最近一則 ToolMessage 擷取工具名稱與 lines 陣列。"""
    out = {"tool_name": None, "lines": []}
    for m in messages[::-1]:
        if isinstance(m, ToolMessage):
            # name: LangChain 會帶入 tool 名稱
            name = getattr(m, "name", None)
            payload = m.content
            try:
                if isinstance(payload, str):
                    obj = json.loads(payload)
                elif isinstance(payload, dict):
                    obj = payload
                else:
                    obj = {}
            except Exception:
                obj = {}
            lines = obj.get("lines") if isinstance(obj, dict) else None
            if isinstance(lines, list):
                out["lines"] = [str(x).strip() for x in lines if isinstance(x, str) and str(x).strip()]
            out["tool_name"] = name
            break
    return out


def accumulate_node(state: AgentState) -> Dict:
    """累積比對行；並依工具與命中數，決定是否前進索引，或強制下一輪跑向量。"""
    prev = state.get("accum_lines", []) or []
    now: List[str] = []

    # 1) 掃描 AI 純文字訊息中的 [比對] 行（允許 [n] 前綴）
    text = ""
    for m in (state.get("messages") or [])[::-1]:
        if hasattr(m, "content"):
            if isinstance(m.content, str):
                text += "\n" + m.content
            elif isinstance(m.content, list):
                for c in m.content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += "\n" + (c.get("text") or "")
    for raw in text.splitlines():
        s = (raw or "").strip()
        if s.startswith("[比對]") or re.match(r"^\[\d+\]\s*\[比對\]", s):
            s2 = re.sub(r"^\[\d+\]\s*", "", s)
            if s2.startswith("[比對]"):
                now.append(s2)

    # 2) 最近一則 ToolMessage → 解析工具名與 lines
    parsed = _parse_tool_payload_from_messages(state.get("messages") or [])
    tool_name = parsed["tool_name"]
    tool_lines = parsed["lines"]

    # 標準化為 '[比對] ' 前綴
    for ln in tool_lines:
        s = ln if ln.startswith("[比對]") else f"[比對] {ln}"
        now.append(s)

    merged = _dedup_keep_order(prev + now, triples=None)

    no_new = state.get("no_new_steps", 0) or 0
    no_new = 0 if len(merged) > len(prev) else no_new + 1
    step = int(state.get("step") or 0) + 1

    # === 新增：是否要前進 triple 索引，與是否要強制向量下一輪 ===
    idx = int(state.get("triples_processed_index") or 0)
    force_vector_next = bool(state.get("force_vector_next", False))

    # 預設：不變
    advance_idx = False

    # 規則決策
    if tool_name in ("pg_search", "neo4j_search"):
        if len(tool_lines) == 0:
            # PG/Neo4j 0 命中 → 若開啟向量備援，下一輪強制 vector（同一個 triple 不前進）
            enable_vec = os.getenv("ENABLE_VECTOR_FALLBACK", "1").lower() in ("1", "true", "yes", "y")
            if enable_vec:
                force_vector_next = True
                advance_idx = False
            else:
                # 沒開向量 → 就當作這個 triple 完成，前進
                force_vector_next = False
                advance_idx = True
        else:
            # 有命中 → 這個 triple 完成，前進
            force_vector_next = False
            advance_idx = True
    elif tool_name == "vector_search":
        # 跑完向量（不論命中數） → 視為該 triple 流程結束，前進
        force_vector_next = False
        advance_idx = True
    else:
        # 其他工具（如 merge_and_dedup）不改變 index；保守處理
        pass

    if advance_idx:
        idx += 1

    tgt_note = "無下限（僅以耐心值/步數控制）" if AGENT_TOTAL_TARGET <= 0 else f"總行數≥{AGENT_TOTAL_TARGET}"
    per_note = "不檢查" if AGENT_MIN_PER_TRIPLE <= 0 else f"每 triple≥{AGENT_MIN_PER_TRIPLE}"
    notes = (
        f"累積比對行數：{len(merged)}；步數：{step}；連續無新增：{no_new}\n"
        f"條件：{tgt_note}、{per_note}；上限：{AGENT_TOP_K_MAX}\n"
        f"(this_round tool={tool_name or '-'} got={len(tool_lines)}; force_vector_next={force_vector_next})"
    )

    return {
        "accum_lines": merged[:AGENT_TOP_K_MAX] if AGENT_TOP_K_MAX > 0 else merged,
        "no_new_steps": no_new,
        "step": step,
        "notes": notes,
        "triples_processed_index": idx,           # <- 由這裡決定是否前進
        "force_vector_next": force_vector_next,   # <- 下輪 agent_node 會依此改成 vector_search
    }


def _min_per_triple_ok(lines: List[str], triples: List[Dict[str, str]]) -> bool:
    if AGENT_MIN_PER_TRIPLE <= 0:
        return True
    if not triples:
        return False
    counter: Dict[str, int] = {}
    for ln in lines:
        m = re.search(r"\[比對\]\s*(.+?)\s*透過關係【.+?】與\s*(.+?)\s*建立連結", ln)
        if m:
            key = m.group(1).strip() + "｜" + m.group(2).strip()
            counter[key] = counter.get(key, 0) + 1
    for tp in triples:
        h = (tp.get("head") or "").strip()
        t = (tp.get("tail") or "").strip()
        ok = any(
            ((h and h in k) or (t and t in k)) and v >= AGENT_MIN_PER_TRIPLE
            for k, v in counter.items()
        )
        if not ok:
            return False
    return True


def should_continue(state: AgentState) -> str:
    """維持舊版收斂條件；若超過上限也提前 finalize。"""
    lines = state.get("accum_lines", []) or []
    triples = state.get("triples", []) or []
    idx = int(state.get("triples_processed_index") or 0)
    step = int(state.get("step") or 0)
    no_new = int(state.get("no_new_steps") or 0)

    total_cond = AGENT_TOTAL_TARGET > 0 and len(lines) >= AGENT_TOTAL_TARGET
    per_cond = _min_per_triple_ok(lines, triples)

    if triples and idx >= len(triples) and step > 0:
        _dlog("assess: all triples processed, finalizing.")
        return "finalize"

    cap = step >= AGENT_MAX_STEPS
    patience = no_new >= AGENT_NO_NEW_PATIENCE

    _dlog(
        f"assess: lines={len(lines)} total_cond={total_cond} per_ok={per_cond} "
        f"step={step} cap={cap} no_new={no_new} patience={patience}"
    )

    # 舊版是「> 上限」才提前 finalize；保留一致性
    if AGENT_TOP_K_MAX > 0 and len(lines) > AGENT_TOP_K_MAX:
        _dlog("early stop: accumulated lines exceed top_k_max")
        return "finalize"

    if total_cond or (AGENT_MIN_PER_TRIPLE > 0 and per_cond) or cap or patience:
        return "finalize"
    return "agent"


def finalize_node(state: AgentState) -> Dict:
    """與舊版相同的輸出格式。"""
    news = (state.get("news_text") or "").strip()
    lines = state.get("accum_lines", []) or []

    cleaned: List[str] = []
    for ln in lines:
        s = re.sub(r"^\s*\[比對\]\s*", "", ln)
        s = re.sub(r"^\s*\[\d+\]\s*", "", s)
        cleaned.append(s.strip())

    numbered = [f"[{i}] {t}" for i, t in enumerate(cleaned, 1)]
    body = "[原始文本]\n" + news + "\n\n[比對知識]\n" + "\n".join(numbered)
    return {"messages": state.get("messages", []) + [AIMessage(content=body)]}


def route_from_agent(state: AgentState) -> str:
    last = state.get("messages", [])[-1] if state.get("messages") else None
    if last is not None and hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "accumulate"

def _safe_load(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}
    
def search_once_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    針對「當前 triple」做一次檢索：
    1) 先 PG
    2) 若 .env 開啟向量、且 PG 命中 < VECTOR_TRIGGER_MIN → 再跑向量
    3) merge_and_dedup 合併
    4) 把合併結果以 ToolMessage/AIMessage 風格寫回 messages（保持你原本的 streaming/統計行為）
    """
    messages: List[Any] = state.get("messages", []) or []
    triples = state.get("triples", []) or []
    idx = int(state.get("triples_processed_index") or 0)
    if not triples or idx == 0:
        # 在 extract_triples_node 之後，agent_node 已把 index+1；這裡按你現有流程取「上一個」處理的
        idx = max(0, idx - 1)
    if idx >= len(triples):
        return {"messages": messages}  # 沒有 triple 可處理

    current_triple = triples[idx]
    tri_json = json.dumps(current_triple, ensure_ascii=False)

    # 1) 先 PG
    pg_topk = int(os.getenv("LI_PG_TOPK", "50"))
    pg_hops = int(os.getenv("LI_PG_HOPS", "3"))
    pg_raw = tool_pg_search.func(triple_json=tri_json, top_k=pg_topk, hops=pg_hops)
    pg_obj = _safe_load(pg_raw)
    pg_hits = int(pg_obj.get("hits") or len(pg_obj.get("lines") or []))

    # 2) 視條件觸發 Vector
    need_vector = ENABLE_VECTOR_FALLBACK and (pg_hits < VECTOR_TRIGGER_MIN)
    vec_obj = {"lines": [], "items": []}
    if need_vector:
        vec_topk = VECTOR_TOPK if VECTOR_TOPK is not None else 100
        vec_raw = tool_vector_search.func(triple_json=tri_json, top_k=vec_topk)
        vec_obj = _safe_load(vec_raw)

    # 3) 合併
    merge_payload = json.dumps({"pg": pg_obj, "vector": vec_obj}, ensure_ascii=False)
    merged_raw = tool_merge_and_dedup.func(payload=merge_payload)
    merged_obj = _safe_load(merged_raw)
    merged_lines: List[str] = list(merged_obj.get("lines") or [])

    # 4) 把工具回覆片段寫回 messages（提供給 accumulate_node 統計）
    # 與你原本習慣一致：工具訊息 → {"lines":[...]} 字串
    from langchain_core.messages import ToolMessage
    tool_preview_pg = json.dumps({"lines": pg_obj.get("lines") or []}, ensure_ascii=False)
    messages.append(ToolMessage(content=tool_preview_pg, name="pg_search", tool_call_id="pg_once"))

    if need_vector:
        tool_preview_vec = json.dumps({"lines": vec_obj.get("lines") or []}, ensure_ascii=False)
        messages.append(ToolMessage(content=tool_preview_vec, name="vector_search", tool_call_id="vec_once"))

    # 再補一個「merge」的工具輸出，保持行為一致
    messages.append(ToolMessage(content=json.dumps({"lines": merged_lines}, ensure_ascii=False),
                                name="merge_and_dedup", tool_call_id="merge_once"))

    # 將此次合併結果交給後續 accumulate_node
    # （不用直接塞進 accum_lines，沿用你既有的 _extract_kb_lines_from_messages → accumulate 更安全）
    return {
        "messages": messages
    }