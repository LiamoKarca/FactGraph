"""
集中管理超參數與正則表達式
"""
import re

SIM_TH: float = 0.8
TOP_K: int = 100
LLM_ROUNDS: int = 3
DUP_TH: float = 0.8
ENTITY_RE = re.compile('^\\d+\\.\\s*(.+?)\\s*透過關係')
