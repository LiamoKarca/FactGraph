"""
LangGraph 建置。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..tools import tools_node
from .nodes import (
    AgentState,
    agent_node,
    accumulate_node,
    extract_triples_node,
    finalize_node,
    route_from_agent,
    should_continue,
)


def build_graph():
    """建置並編譯 LangGraph。"""
    graph = StateGraph(AgentState)

    # 節點
    graph.add_node("extract_triples", extract_triples_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)        # 工具節點
    graph.add_node("accumulate", accumulate_node)
    graph.add_node("finalize", finalize_node)

    # 邊
    graph.add_edge(START, "extract_triples")
    graph.add_edge("extract_triples", "agent")
    graph.add_conditional_edges(
        "agent",
        route_from_agent,
        {"tools": "tools", "accumulate": "accumulate"},
    )
    graph.add_edge("tools", "accumulate")
    graph.add_conditional_edges(
        "accumulate",
        should_continue,
        {"agent": "agent", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    # 重要：不要傳 config 給 compile()；recursion_limit 應在 stream/invoke 時提供
    return graph.compile(checkpointer=None)
