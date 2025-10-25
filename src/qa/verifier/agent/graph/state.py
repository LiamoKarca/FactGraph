"""
Graph 狀態/LLM 建構 —— 明確規則化『向量=嚴格備援』。
"""

from __future__ import annotations
import os
from typing import TypedDict, List, Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

from ..common.config import OPENAI_CHAT_MODEL, RECURSION_LIMIT, _dlog


# System Prompt（回復舊版工具策略；嚴格限制向量只能在 0 命中時使用）
AGENT_SYS_PROMPT: str = """
你是一個新聞事實對齊代理。你的工作：
1) 針對單一三元組，選擇正確工具檢索比對知識。
2) 工具策略：先用 pg_search 與 neo4j_search；**只有當兩者都 0 命中時**，才可使用 vector_search 作為備援。
3) 每次工具呼叫只做一件事，並輸出盡可能精煉的證據行（[比對] 開頭）。
4) 嚴格遵守最大證據長度限制；避免一次吐過多重複/相似內容（有語義去重）。
5) 依需求分多輪；若無新命中或達到步數上限即停止。

【硬性規則】
- 若上一輪 pg_lines>0 或 neo4j_lines>0 → 禁止呼叫 vector_search。
- 僅當 pg_lines==0 且 neo4j_lines==0 → 允許呼叫 vector_search。
- 不要在同一輪同時呼叫 pg 與 vector；遵守「嚴格備援」。
""".strip()


class AgentState(TypedDict, total=False):
    news_text: str
    triples: List[dict]
    triples_processed_index: int
    messages: List[Any]
    accum_lines: List[str]
    step: int
    no_new_steps: int
    notes: str
    tool_availability: str
    last_pg_lines: int
    last_neo_lines: int
    last_vec_lines: int
    # 當上一輪 PG/Neo4j 命中數為 0，就把這個旗標設 True，下一輪同一個 triple 強制改跑向量
    force_vector_next: bool


def build_llm() -> BaseChatModel:
    """建構 LLM（與舊版相同走 OpenAI Chat 路徑，吃 .env）"""
    model = os.getenv("OPENAI_CHAT_MODEL", OPENAI_CHAT_MODEL)
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
    _dlog(f"build_llm: model={model} temp={temperature} max_tokens={max_tokens}")
    return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

