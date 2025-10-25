"""工具集合與 ToolNode 生成。"""

from __future__ import annotations

import os
from typing import List

from langgraph.prebuilt import ToolNode

from .pg_tool import tool_pg_search
from .merge_tool import tool_merge_and_dedup

TOOLS: List = [tool_pg_search, tool_merge_and_dedup]

# 依環境動態掛載
if os.getenv("ENABLE_LI_ONLINE", "0").lower() in ("1", "true", "yes"):
    from .neo4j_tool import tool_neo4j_search

    TOOLS.append(tool_neo4j_search)

if os.getenv("ENABLE_VECTOR_FALLBACK", "1").lower() in ("1", "true", "yes"):
    from .vector_tool import tool_vector_search

    TOOLS.append(tool_vector_search)

tools_node = ToolNode(TOOLS)

__all__ = ["TOOLS", "tools_node", "tool_pg_search"]
