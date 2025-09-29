"""
集中管理超參數與正則表達式
"""
import re

# —— 召回優先（長文新聞）建議 —— 

SIM_TH: float = 0.6 # cosine 初篩門檻
TOP_K: int = 100     # 後段還有 dedup 與排序，先多撈
LLM_ROUNDS: int = 2  # 複雜稿可暫時調 3
DUP_TH: float = 0.8  # 語義去重臨界值（相似≥此值視為重覆）

# 擷取行首主語：支援「[1]」或「1. 」起頭，抓到第一個實體 token
ENTITY_RE = re.compile(r'^(?:\[\d+\]|\d+\.)\s*([^，、:：\s]+)')

# 針對 evidence 的最低關鍵詞命中數（降低噪音的關鍵開關）
# 建議先設 2；若仍覺得鬆，可以調到 3。
MIN_TERM_HITS_IN_EVIDENCE = 3