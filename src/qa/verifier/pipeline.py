"""
新聞事實驗證主流程 Pipeline
================================================
- ① 簡潔化與抽取
- ④ KG 比對：USE_AGENT_RETRIEVAL=1 → 走 ReAct 代理；否則舊三分流
- ⑤ Dirt Removal
- ⑥ Judge（此步切換 gpt-5）

Usage:
$ python -m src.qa.verifier.pipeline [news_id.txt]
$ python -m src.qa.verifier.pipeline [news_id]

Self-Test:
$ python -m src.qa.verifier.pipeline --self-test
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback  # 為了自檢
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from tqdm import tqdm

# 先載入 .env，並允許覆蓋現有環境
load_dotenv(override=True)

# --- 延後匯入本地模組 ---
# from ..tools import data_utils as du
# from ..tools import kg_nl as knl
# from .core.dedup import deduplicate
# from .core.embeddings import embed_text, embed_triple
# from .core.paths import RES_DIR, USER_INPUT_DIR, VEC_DIR
# from .kg.search import cosine_search, kg_row_to_detail
# from .llm.extract import extract_entities_relations, set_debug_basename as set_extract_debug_basename
# from .llm.judge import judge_news_kb, set_debug_basename as set_judge_debug_basename
# from .llm.dirt_removal import run_dirt_removal
# from .llm.gpt import GPTClient
# from .agent.runner import run_factcheck_middle as agent_run_middle
# ---


# 全域變數，延遲載入
_DU_MODULE = None
_KNL_MODULE = None
_DEDUP_MODULE = None
_EMBEDDINGS_MODULE = None
_PATHS_MODULE = None
_KG_SEARCH_MODULE = None
_LLM_EXTRACT_MODULE = None
_LLM_JUDGE_MODULE = None
_LLM_DIRT_MODULE = None
_LLM_GPT_MODULE = None
_AGENT_RUNNER_MODULE = None
_CONFIG_MODULE = None  # Agent Runner 也需要 config
_TOOLS_PG_MODULE = None # For old path _init_li_pg
_KG_NEO4J_MODULE = None # For old path _init_li_online

# 外部關鍵詞庫變數
_STRONG_REL_KEYWORDS = set()
_REL_PATH: Path | None = None


def _lazy_load_dependencies():
    """延遲載入模組，避免頂層載入問題，並方便自檢。"""
    global _DU_MODULE, _KNL_MODULE, _DEDUP_MODULE, _EMBEDDINGS_MODULE, _PATHS_MODULE
    global _KG_SEARCH_MODULE, _LLM_EXTRACT_MODULE, _LLM_JUDGE_MODULE, _LLM_DIRT_MODULE
    global _LLM_GPT_MODULE, _AGENT_RUNNER_MODULE, _CONFIG_MODULE, _STRONG_REL_KEYWORDS
    global _REL_PATH, _TOOLS_PG_MODULE, _KG_NEO4J_MODULE

    # 基礎工具和路徑
    if _DU_MODULE is None:
        from ..tools import data_utils as du
        _DU_MODULE = du
    if _KNL_MODULE is None:
        from ..tools import kg_nl as knl
        _KNL_MODULE = knl
    if _DEDUP_MODULE is None:
        from .core import dedup
        _DEDUP_MODULE = dedup
    if _EMBEDDINGS_MODULE is None:
        from .core import embeddings
        _EMBEDDINGS_MODULE = embeddings
    if _PATHS_MODULE is None:
        from .core import paths
        _PATHS_MODULE = paths
        # 在載入 paths 後設定 RES_DIR 等全域變數
        global RES_DIR, USER_INPUT_DIR, VEC_DIR, CONCISENESS_DIR, DIRT_DIR, DIRT_DEBUG_DIR
        RES_DIR = _PATHS_MODULE.RES_DIR
        USER_INPUT_DIR = _PATHS_MODULE.USER_INPUT_DIR
        VEC_DIR = _PATHS_MODULE.VEC_DIR
        CONCISENESS_DIR = Path("data/interim/verifier/conciseness") # 這個路徑是固定的
        DIRT_DIR = RES_DIR / "dirt_removal"
        DIRT_DEBUG_DIR = DIRT_DIR / "debug"
    if _KG_SEARCH_MODULE is None:
        from .kg import search
        _KG_SEARCH_MODULE = search

    # LLM 相關模組
    if _LLM_EXTRACT_MODULE is None:
        from .llm import extract
        _LLM_EXTRACT_MODULE = extract
    if _LLM_JUDGE_MODULE is None:
        from .llm import judge
        _LLM_JUDGE_MODULE = judge
    if _LLM_DIRT_MODULE is None:
        from .llm import dirt_removal
        _LLM_DIRT_MODULE = dirt_removal
    # GPTClient 是可選的
    if _LLM_GPT_MODULE is None:
        try:
            from .llm.gpt import GPTClient
            _LLM_GPT_MODULE = GPTClient # 存儲類本身
        except Exception:
            _LLM_GPT_MODULE = None # type: ignore

    # Agent Runner (新流程核心)
    if _AGENT_RUNNER_MODULE is None:
        try:
            from .agent import runner
            _AGENT_RUNNER_MODULE = runner
            # agent runner 也依賴 config，一併載入
            if _CONFIG_MODULE is None:
                 from .agent.common import config
                 _CONFIG_MODULE = config
        except Exception:
            _AGENT_RUNNER_MODULE = None # type: ignore

    # 舊流程相關模組
    if _TOOLS_PG_MODULE is None:
        try:
            # 相對路徑
            from ..tools.property_graph import li_csv_pg_retriever
            _TOOLS_PG_MODULE = li_csv_pg_retriever
        except ImportError:
             _TOOLS_PG_MODULE = None # type: ignore
    if _KG_NEO4J_MODULE is None:
        try:
            from .kg.llamaIndex import neo4j_li_retriever
            _KG_NEO4J_MODULE = neo4j_li_retriever
        except ImportError:
            _KG_NEO4J_MODULE = None # type: ignore


    # 載入外部關鍵詞庫
    if not _STRONG_REL_KEYWORDS and _REL_PATH is None:
        _REL_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "knowledge-graph" / "relation_dict_all.py"
        try:
            import importlib.util as _impspec
            spec = _impspec.spec_from_file_location("relation_dict_all", _REL_PATH)
            if spec and spec.loader:
                rel_module = _impspec.module_from_spec(spec)
                spec.loader.exec_module(rel_module) # type: ignore
                _STRONG_REL_KEYWORDS = set(getattr(rel_module, "RELATIONS_ALL", set()))
                print(f"⚙️ 已載入外部關鍵詞庫，共 {len(_STRONG_REL_KEYWORDS)} 個關鍵詞。")
            else:
                 print(f"⚠️ [WARN] 無法為 relation_dict_all.py 創建 spec: {_REL_PATH}")
        except Exception as exc:
            print(f"⚠️ [WARN] 無法載入 relation_dict_all.py：{exc}")
            _STRONG_REL_KEYWORDS = set() # 確保至少是空集合


# ─────────── 常數 (部分移到 lazy load 後) ───────────
LLM_ROUNDS = int(os.getenv("LLM_ROUNDS", "1"))
OUTPUT_ENCODING = "utf-8-sig" # 這個可以先定義

CONCISENESS_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CONCISENESS_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "conciseness.txt"
CONCISENESS_HARD_TIMEOUT_SEC = 30

# 路徑變數在 _lazy_load_dependencies 中賦值
RES_DIR: Path
USER_INPUT_DIR: Path
VEC_DIR: Path
CONCISENESS_DIR: Path
DIRT_DIR: Path
DIRT_DEBUG_DIR: Path

# ─────────── 除錯日誌設定 ───────────
VERIFIER_DEBUG = os.getenv("VERIFIER_DEBUG", "0") == "1"
VERIFIER_DEBUG_PATH = Path(
    os.getenv("VERIFIER_DEBUG_PATH", "data/processed/verifier/debug/verifier_search_debug.log")
)
# 確保目錄存在移到 _dlog 內部

def _dlog(msg: str) -> None:
    if not VERIFIER_DEBUG:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        VERIFIER_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True) # 每次寫入前確保
        with open(VERIFIER_DEBUG_PATH, "a", encoding="utf-8-sig") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass # 日誌失敗不應中斷主流程


# ─────────── 簡潔輸入 ───────────
def _ensure_conciseness_file(news_id: str, original_text: str) -> Path:
    _lazy_load_dependencies() # 需要 _LLM_GPT_MODULE 和路徑
    CONCISENESS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONCISENESS_DIR / f"{news_id}.txt"

    if out_path.is_file():
        try:
            if out_path.stat().st_size > 0:
                print(f"② 簡潔輸入 ▶ 使用現有簡潔檔：{out_path.resolve()}")
                return out_path
        except Exception:
            pass

    concise = original_text.strip()

    try:
        tmpl = CONCISENESS_PROMPT_PATH.read_text(encoding=OUTPUT_ENCODING).strip()
    except FileNotFoundError:
        out_path.write_text(concise, encoding=OUTPUT_ENCODING)
        print("② 簡潔輸入 ▶ [WARN] 找不到 conciseness 提示模板，已以原文回寫。")
        return out_path

    GPTClient = _LLM_GPT_MODULE # 從延遲載入獲取類
    if GPTClient is None or not os.getenv("OPENAI_API_KEY", "").strip():
        out_path.write_text(concise, encoding=OUTPUT_ENCODING)
        print("② 簡潔輸入 ▶ [WARN] 無 GPTClient 或無金鑰，已以原文回寫。")
        return out_path

    system_prompt = "擔任中文新聞綴字簡潔助手。僅做刪繁就簡，不得加入新資訊。"
    user_prompt = f"{tmpl}\n\n---\n【原始文字】\n{original_text.strip()}\n---\n請直接輸出簡潔後文本，不要夾帶說明或標籤。"

    try:
        try:
            # 嘗試 Langchain v0.1+ 的初始化方式
            gpt = GPTClient(api_key=os.getenv("OPENAI_API_KEY", ""), model_id=CONCISENESS_MODEL, temperature=0, timeout=CONCISENESS_HARD_TIMEOUT_SEC)
        except TypeError: # 捕獲可能的參數不匹配錯誤
            # 回退到舊版或不同的初始化方式
             gpt = GPTClient(os.getenv("OPENAI_API_KEY", ""), CONCISENESS_MODEL, temperature=0, timeout=CONCISENESS_HARD_TIMEOUT_SEC) # type: ignore
        except Exception as init_exc:
             print(f"② 簡潔輸入 ▶ [ERROR] GPTClient 初始化失敗: {init_exc}")
             raise # 重新拋出，讓外層捕獲

        concise = gpt.chat(system_prompt, user_prompt, max_retries=0).strip() or original_text.strip()
    except Exception as exc:
        print(f"② 簡潔輸入 ▶ [WARN] 生成簡潔檔失敗：{exc} → 以原文回寫。")
        concise = original_text.strip()

    out_path.write_text(concise, encoding=OUTPUT_ENCODING)
    print(f"② 簡潔輸入 ▶ 已寫入簡潔檔：{out_path.resolve()}")
    return out_path


# ─────────── ER → Triples（供舊三分流；Agent 版由工具處理） ───────────
def _er_to_triples(data: Dict[str, Any]) -> List[Dict[str, str]]: # 返回標準 Dict
    _lazy_load_dependencies() # 需要 _DU_MODULE
    triples: List[Dict[str, str]] = []
    if not isinstance(data, dict):
        return triples
    ents = {e.get("id"): e for e in data.get("entities", []) if isinstance(e, dict) and e.get("id")}
    for rel in data.get("relations", []) or []:
        if not isinstance(rel, dict):
            continue
        src_id = rel.get("source")
        tgt_id = rel.get("target")
        rel_name = (rel.get("relation") or "未知").strip()
        src_ent = ents.get(src_id, {}) or {}
        tgt_ent = ents.get(tgt_id, {}) or {}
        head = (src_ent.get("name") or "未知").strip()
        tail = (tgt_ent.get("name") or "").strip()
        if not tail:
            t_attrs = tgt_ent.get("attributes") or {}
            val = t_attrs.get("value")
            unit = t_attrs.get("unit") or ""
            tail = f"{val}{unit}".strip() if val not in (None, "") else "未知"
        triples.append({"head": head, "relation": rel_name, "tail": tail})
    return triples


def _pull_triples(text_for_extraction: str) -> List[Dict[str, str]]: # 返回標準 Dict
    _lazy_load_dependencies() # 需要 _LLM_EXTRACT_MODULE, _DU_MODULE
    all_rounds: List[List[Dict[str, str]]] = []
    last_error: Exception | None = None
    rounds = int(os.getenv("LLM_ROUNDS", "1"))
    _dlog(f"extract: rounds={rounds}, text_chars={len(text_for_extraction)}")
    for i in range(rounds):
        print(f"③ 抽取 ▶ GPT 抽取 round {i + 1}")
        raw = _LLM_EXTRACT_MODULE.extract_entities_relations(text_for_extraction) # type: ignore
        if not raw:
            print(f"③ 抽取 ▶ [WARN] 抽取回傳為空，跳過 round {i + 1}")
            continue
        try:
            # 嘗試使用 agent 的 json 解析器
            payload = _AGENT_RUNNER_MODULE._JSON_UTILS_MODULE.parse_json_safely(raw) # type: ignore
            triples: List[Dict[str, str]] = []
            # 檢查返回結構
            if isinstance(payload, dict) and ("entities" in payload or "relations" in payload):
                 triples = _er_to_triples(payload)
            elif isinstance(payload, (list, dict)): # 假設 du.json_to_triples 能處理 list 或 dict
                 # 使用 du 模塊的函數，假設它存在且功能兼容
                 if hasattr(_DU_MODULE, 'json_to_triples'):
                     triples = _DU_MODULE.json_to_triples(payload) or [] # type: ignore
                 else:
                     print(f"③ 抽取 ▶ [WARN] data_utils missing 'json_to_triples'. Cannot process legacy format.")

            if triples:
                all_rounds.append(triples)
        except Exception as exc:
            last_error = exc
            print(f"③ 抽取 ▶ [WARN] JSON 解析或轉換失敗於 round {i + 1}: {exc}")

    if not all_rounds:
        if last_error:
            print(f"③ 抽取 ▶ [ERROR] 所有輪次皆失敗: {last_error}", file=sys.stderr)
        return []

    # 合併三元組，假設 du 模塊有 merge_triples
    triples_merged: List[Dict[str, str]] = []
    if hasattr(_DU_MODULE, 'merge_triples'):
         triples_merged = _DU_MODULE.merge_triples(*all_rounds) # type: ignore
    else:
        print(f"③ 抽取 ▶ [WARN] data_utils missing 'merge_triples'. Returning unmerged.")
        # 簡單合併列表作為回退
        for round_list in all_rounds:
            triples_merged.extend(round_list)
        # 這裡可以加一個去重邏輯

    print(f"③ 抽取 ▶ 合併後三元組數：{len(triples_merged)}")
    _dlog(f"extract_done: triples={len(triples_merged)}")
    return triples_merged

# ─────────── 舊三分流（當 USE_AGENT_RETRIEVAL=0 或 Agent 失敗時回退） ───────────
def _init_li_pg():
    _lazy_load_dependencies() # 需要 _TOOLS_PG_MODULE
    if _TOOLS_PG_MODULE is None:
        raise ImportError("CsvPropertyGraphRetriever module not loaded.")
    # 假設 CsvPropertyGraphRetriever 在 _TOOLS_PG_MODULE 下
    CsvPropertyGraphRetriever = getattr(_TOOLS_PG_MODULE, 'CsvPropertyGraphRetriever', None)
    if CsvPropertyGraphRetriever is None:
         raise AttributeError("CsvPropertyGraphRetriever not found in module.")
    return CsvPropertyGraphRetriever.ensure_built_and_loaded()

def _init_li_online():
    _lazy_load_dependencies() # 需要 _KG_NEO4J_MODULE
    if _KG_NEO4J_MODULE is None:
         raise ImportError("LlamaIndexNeo4jRetriever module not loaded.")
    LlamaIndexNeo4jRetriever = getattr(_KG_NEO4J_MODULE, 'LlamaIndexNeo4jRetriever', None)
    if LlamaIndexNeo4jRetriever is None:
         raise AttributeError("LlamaIndexNeo4jRetriever not found in module.")
    return LlamaIndexNeo4jRetriever(date_field="date", evidence_field="evidence")

def _safe_build_block(tris: List[Dict], details: Dict[Tuple[str, str, str], Dict]) -> str:
    # 這個函數內部不依賴延遲載入模組
    lines: List[str] = []
    for tri in tris:
        # 確保 tri 是字典
        if not isinstance(tri, dict): continue
        key = (tri.get("head",""), tri.get("relation",""), tri.get("tail",""))
        det = details.get(key, {})
        rel_info = det.get("rel", {}) or {}
        expl = rel_info.get("evidence", "") or rel_info.get("desc", "") or ""
        date_str = rel_info.get("date", "")
        line = f"[比對] {key[0]} 透過關係與 {key[2]} 建立連結；說明：{expl}" + (f"；日期：{date_str}" if date_str else "")
        lines.append(line)
    return "\n".join(lines)

# _STRONG_REL_KEYWORDS 的載入移到 _lazy_load_dependencies

def _specificity(tri: Dict[str, str]) -> int:
    _lazy_load_dependencies() # 需要 _STRONG_REL_KEYWORDS
    score = 0
    if (tri.get("head") or "").strip():
        score += 1
    rel = (tri.get("relation") or "").strip()
    if rel:
        score += 1
        # 使用延遲載入的 _STRONG_REL_KEYWORDS
        if _STRONG_REL_KEYWORDS and any(k in rel for k in _STRONG_REL_KEYWORDS) and len(rel) <= 6:
            score += 1
    if (tri.get("tail") or "").strip():
        score += 1
    return min(score, 4)

def _decide_hops_strategy(tp: Dict[str, str], news_len: int, path_label: str) -> Tuple[int, int, int]:
    # 這個函數依賴 _specificity，它會觸發 lazy load
    auto_enable = os.getenv("AUTO_HOPS_ENABLE", "1") == "1"
    hard_max = int(os.getenv("AUTO_MAX_HOPS_HARD", "3"))
    base_min_hits = int(os.getenv("AUTO_BASE_MIN_HITS", "6"))
    long_news_chars = int(os.getenv("AUTO_LONG_NEWS_CHARS", "400"))
    default_start_env_key = "LI_PG_HOPS" if path_label == "PG" else "LI_ONLINE_HOPS"
    default_start = int(os.getenv(default_start_env_key, "2"))

    if not auto_enable:
        return max(1, default_start), max(1, default_start), base_min_hits

    spec = _specificity(tp)
    is_long = news_len >= long_news_chars
    start_hops = 1 if spec >= 3 else max(2, default_start)
    max_hops = 3 if (is_long or spec <= 1) else max(2, default_start)
    max_hops = min(max_hops, hard_max)

    if spec >= 3:
        min_hits = max(3, base_min_hits - 2)
    elif spec == 2:
        min_hits = base_min_hits
    else:
        min_hits = base_min_hits + 2
    return start_hops, max_hops, min_hits

def _query_with_autohops(retriever, tp: Dict[str, str], news_len: int, path_label: str, top_k: int) -> List[Tuple[Dict, Dict]]:
    # 這個函數依賴 _decide_hops_strategy
    start_hops, max_hops, min_hits = _decide_hops_strategy(tp, news_len, path_label)
    all_hits: List[Tuple[Dict, Dict]] = []
    for hops in range(max(1, start_hops), max_hops + 1):
        try:
             hits = retriever.search_triple(tp, top_k=top_k, hops=hops)
             # LlamaIndex 返回的可能是 NodeWithScore 或其他類型，需要規範化
             # 假設 search_triple 返回的是 [(tri_dict, det_dict), ...] 或類似結構
             # 如果不是，這裡需要添加轉換邏輯
             if isinstance(hits, list):
                  print(f"④ KG 比對 ▶ {path_label} auto-hops：hops={hops} → 命中 {len(hits)}（門檻 {min_hits}）")
                  all_hits = hits # 假設格式正確
                  if len(hits) >= min_hits:
                       return all_hits # 達到門檻，返回
             else:
                  print(f"④ KG 比對 ▶ [WARN] {path_label} search_triple 返回非預期類型: {type(hits)}")
                  return [] # 返回空列表表示失敗

        except Exception as e:
            print(f"④ KG 比對 ▶ [WARN] {path_label} search_triple (hops={hops}) 失敗: {e}")
            # 可以選擇在這裡返回空列表或繼續嘗試下一個 hop
            # 這裡選擇繼續嘗試
            continue

    # 如果所有 hops 都嘗試完畢或中途出錯，返回最後一次成功獲取或空的 hits
    return all_hits


# ─────────── 主處理 ───────────
def _collect_hits_to_lines(hits: List[Tuple[Dict, Dict]]) -> List[str]:
    # 這個函數內部不依賴延遲載入模組
    tris: List[Dict] = []
    dets: Dict[Tuple[str, str, str], Dict] = {}
    for hit in hits:
        # 確保 hit 是 (tri, det) 的元組
        if isinstance(hit, (tuple, list)) and len(hit) == 2:
            tri, det = hit
            # 確保 tri 和 det 是字典
            if isinstance(tri, dict) and isinstance(det, dict):
                 tris.append(tri)
                 key = (tri.get("head",""), tri.get("relation",""), tri.get("tail",""))
                 dets[key] = det
            else:
                 print(f"⚠️ [WARN] _collect_hits_to_lines: Invalid tri/det type in hit: {type(tri)}, {type(det)}")
        else:
            print(f"⚠️ [WARN] _collect_hits_to_lines: Invalid hit format: {type(hit)}")

    # 使用 _safe_build_block 作為後備
    block = _safe_build_block(tris, dets)
    return block.splitlines()


def _process_single(news_id: str, text: str) -> None:
    _lazy_load_dependencies() # 確保所有依賴已載入
    RES_DIR.mkdir(parents=True, exist_ok=True)
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    DIRT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # ① 使用者輸入 → 文本清理與向量保存
    text = text.replace("`", "")
    input_vec_path = VEC_DIR / f"{news_id}.npy"
    try:
        # 使用延遲載入的 embedding 函數
        vector = _EMBEDDINGS_MODULE.embed_text(text) # type: ignore
        np.save(input_vec_path, vector)
        print("① 使用者輸入 ▶ 檔案："
              f"{(USER_INPUT_DIR / (news_id + '.txt')).resolve()}  → 向量儲存："
              f"{input_vec_path.resolve()}")
        _dlog(f"step1_input: news_id={news_id}, chars={len(text)}")
    except Exception as e:
        print(f"① 使用者輸入 ▶ [ERROR] 文本向量化失敗: {e}")
        _dlog(f"step1_input_error: {e}")
        # 可以選擇在這裡退出或繼續（後續向量搜索會失敗）

    # ② 簡潔輸入
    concise_path = _ensure_conciseness_file(news_id, text)
    try:
        concise_text = concise_path.read_text(encoding=OUTPUT_ENCODING).strip()
    except Exception as exc:
        print(f"② 簡潔輸入 ▶ [WARN] 讀取簡潔檔失敗：{exc} → 改用原文。")
        concise_text = text
    news_len = len(concise_text)
    _dlog(f"step2_concise: chars={news_len}, file={concise_path}")

    # Debug 檔名前綴 (使用延遲載入的模組)
    _LLM_EXTRACT_MODULE.set_debug_basename(news_id) # type: ignore
    _LLM_JUDGE_MODULE.set_debug_basename(news_id) # type: ignore

    # ③ 抽取（僅供舊三分流；Agent 會自行抽取）
    triples: List[Dict[str, str]] = [] # 確保 triples 有預設值
    use_agent = os.getenv("USE_AGENT_RETRIEVAL", "1") == "1"
    if not use_agent:
        triples = _pull_triples(concise_text)

    # ④ KG 比對：ReAct 代理或舊三分流
    middle_text = ""
    agent_run_middle = getattr(_AGENT_RUNNER_MODULE, 'run_factcheck_middle', None) if _AGENT_RUNNER_MODULE else None

    if use_agent:
        if agent_run_middle is None:
            sys.exit("❌ ④ KG 比對 ▶ USE_AGENT_RETRIEVAL=1，但 agent 模組不可用或 'run_factcheck_middle' 函數缺失。")
        print("④ KG 比對 ▶ 由 ReAct 代理（LangGraph StateGraph）產生『中間產物』")
        middle_text = agent_run_middle(concise_text, session_id=news_id)
    else:
        # 舊三分流（PG → Neo4j → 向量）
        print("④ KG 比對 ▶ 舊三分流（PG → Neo4j → 向量）")
        raw_lines: List[str] = []
        use_li_pg = os.getenv("ENABLE_LI_PG", "1") == "1"
        use_li_online = os.getenv("ENABLE_LI_ONLINE", "0") == "1"
        use_vec_fallback = os.getenv("ENABLE_VECTOR_FALLBACK", "1") == "1"

        li_pg_hops = int(os.getenv("LI_PG_HOPS", "2"))
        li_online_hops = int(os.getenv("LI_ONLINE_HOPS", "2"))
        auto_enable = os.getenv("AUTO_HOPS_ENABLE", "1") == "1"
        auto_topk = int(os.getenv("AUTO_HOPS_TOPK", "50"))

        li_pg = None
        if use_li_pg:
            try:
                li_pg = _init_li_pg()
                print(f"④ KG 比對 ▶ 離線 PG 啟用（hops={li_pg_hops}{' / auto' if auto_enable else ''}）")
            except Exception as exc:
                print(f"④ KG 比對 ▶ [WARN] 離線 PG 初始化失敗：{exc}")
                use_li_pg = False

        li_online = None
        if use_li_online:
            try:
                li_online = _init_li_online()
                print(f"④ KG 比對 ▶ Neo4j 啟用（hops={li_online_hops}{' / auto' if auto_enable else ''}）")
            except Exception as exc:
                print(f"④ KG 比對 ▶ [WARN] Neo4j 初始化失敗：{exc}")
                use_li_online = False

        for tp in tqdm(triples, desc="④ KG 比對 ▶ 進度"):
            handled = False
            # PG 搜索
            if use_li_pg and li_pg is not None:
                try:
                    hits = (_query_with_autohops(li_pg, tp, news_len, "PG", auto_topk)
                            if auto_enable else li_pg.search_triple(tp, top_k=50, hops=li_pg_hops))
                    if hits:
                        raw_lines.extend(_collect_hits_to_lines(hits))
                        handled = True
                except Exception as exc:
                    print(f"④ KG 比對 ▶ [WARN] 離線 PG 檢索失敗：{exc}")

            # Neo4j 搜索 (如果 PG 未處理)
            if not handled and use_li_online and li_online is not None:
                try:
                    hits = (_query_with_autohops(li_online, tp, news_len, "Neo4j", auto_topk)
                            if auto_enable else li_online.search_triple(tp, top_k=50, hops=li_online_hops))
                    if hits:
                        raw_lines.extend(_collect_hits_to_lines(hits))
                        handled = True
                except Exception as exc:
                    print(f"④ KG 比對 ▶ [WARN] Neo4j 檢索失敗：{exc}")

            # 向量搜索 (如果前兩者都未處理)
            if not handled and use_vec_fallback:
                try:
                    q_vec = _EMBEDDINGS_MODULE.embed_triple(tp) # type: ignore
                    for idx in _KG_SEARCH_MODULE.cosine_search(tp, q_vec): # type: ignore
                        tri, det = _KG_SEARCH_MODULE.kg_row_to_detail(idx) # type: ignore
                        raw_lines.extend(_collect_hits_to_lines([(tri, det)])) # 傳入列表
                    handled = True
                except Exception as exc:
                    print(f"④ KG 比對 ▶ [WARN] 向量檢索失敗：{exc}")

            if not handled:
                print(f"④ KG 比對 ▶ [WARN] 此三元組未命中任何路徑: {tp}")

        if not raw_lines:
            print("⚠️ ④ KG 比對 ▶ 無命中，流程終止") # 改為警告，避免批次中斷
            middle_text = f"[原始文本]\n{text.strip()}\n\n[比對知識]\n" # 產生空的中介文本
        else:
            # 使用延遲載入的 dedup 函數
            kept = _DEDUP_MODULE.deduplicate(raw_lines, triples=triples) # type: ignore
            final = [re.sub(r"^\d+\.", f"[{i}]", ln, count=1) for i, ln in enumerate(kept, 1)]
            news_block = "[原始文本]\n" + text.strip() # 使用原始文本
            kb_block = "[比對知識]\n" + "\n".join(final)
            middle_text = f"{news_block}\n\n{kb_block}"

    # —— 寫入中間產物 —— #
    kg_file = RES_DIR / f"news_kg_{news_id}.txt"
    kg_file.write_text(middle_text, encoding=OUTPUT_ENCODING)
    print(f"④ KG 比對 ▶ 已寫入：{kg_file.resolve()}")

    # ⑤ Dirt Removal
    combined_text_raw = middle_text # 直接使用内存中的 middle_text
    if "[比對知識]" in combined_text_raw and "[原始文本]" in combined_text_raw:
        combined_text = combined_text_raw.replace("[原始文本]", "[原始新聞]", 1)
    elif "[比對知識]" in combined_text_raw and "[原始新聞]" in combined_text_raw:
        combined_text = combined_text_raw
    else:
        combined_text = f"[原始新聞]\n{text.strip()}\n\n[比對知識]\n{combined_text_raw.strip()}"

    filtered_text = combined_text # 預設值
    dirt_txt_path = str(kg_file)  # 預設路徑

    try:
        # 使用延遲載入的 dirt removal 函數
        filtered_text, dirt_txt_path_str = _LLM_DIRT_MODULE.run_dirt_removal( # type: ignore
            combined_text=combined_text, news_id=news_id, out_dir=str(RES_DIR)
        )
        dirt_txt_path = dirt_txt_path_str # 更新路徑
        print(f"⑤ Dirt Removal ▶ debug JSON：{(DIRT_DEBUG_DIR / f'news_kg_{news_id}.json').resolve()}")
        print(f"⑤ Dirt Removal ▶ 過濾後文本：{Path(dirt_txt_path).resolve()}")
    except Exception as exc:
        print(f"⑤ Dirt Removal ▶ [WARN] 流程失敗：{exc} → 以未過濾版本進入判斷。")
        # filtered_text 和 dirt_txt_path 已有預設值

    # ⑥ Judge（此步切換 gpt-5）
    prev_model = os.getenv("OPENAI_CHAT_MODEL", "")
    judge_file = RES_DIR / f"judge_result_{news_id}.txt" # 先定義路徑
    try:
        os.environ["OPENAI_CHAT_MODEL"] = "gpt-5"
        # 使用延遲載入的模組
        _LLM_JUDGE_MODULE.set_debug_basename(news_id) # type: ignore
        judge_input = filtered_text # 直接使用内存中的文本
        judge_input_src = Path(dirt_txt_path).resolve()

        print(f"⑥ 判斷 ▶ 使用模型：gpt-5，輸入來源檔：{judge_input_src}")
        judged_raw = _LLM_JUDGE_MODULE.judge_news_kb(judge_input, debug_name=news_id) # type: ignore
        judged_clean = judged_raw.replace("`", "")
        judge_file.write_text(judged_clean, encoding=OUTPUT_ENCODING)
        print(f"⑥ 判斷 ▶ 已寫入：{judge_file.resolve()}")
    except Exception as judge_exc:
         print(f"⑥ 判斷 ▶ [ERROR] Judge 流程失敗: {judge_exc}")
         # 即使失敗也寫入空文件或錯誤信息？看需求
         try:
             judge_file.write_text(f"Judge failed: {judge_exc}", encoding=OUTPUT_ENCODING)
         except Exception: pass # 忽略寫入失敗
    finally:
        if prev_model:
            os.environ["OPENAI_CHAT_MODEL"] = prev_model
        else:
             # 如果原始沒有設定，確保清空
             if "OPENAI_CHAT_MODEL" in os.environ:
                 del os.environ["OPENAI_CHAT_MODEL"]

    print("✅ 完成 ▶ 主要輸出：\n"
          f"   - KG：{kg_file.resolve()}\n"
          f"   - Dirt Removal：{Path(dirt_txt_path).resolve()}\n"
          f"   - Judge：{judge_file.resolve()}") # 使用定義好的路徑


# ─────────── 入口與批次 ───────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("FactGraph Verifier Pipeline")
    p.add_argument("news_id", nargs="?", help="新聞檔名（含或不含 .txt），留空則批次所有")
    p.add_argument("--self-test", action="store_true", help="執行自我檢測並退出")
    return p.parse_args()

def main() -> None:
    _lazy_load_dependencies() # 確保主函數依賴已載入
    args = _parse_args()

    # 如果指定了 --self-test，則不執行主要邏輯
    if args.self_test:
        print("Self-test argument detected. Skipping main execution.")
        return

    if args.news_id:
        name = args.news_id if args.news_id.endswith(".txt") else (f"{args.news_id}.txt")
        input_path = USER_INPUT_DIR / name
        if not input_path.is_file():
            sys.exit(f"❌ 找不到檔案：{input_path}")
        try:
             text = input_path.read_text(encoding=OUTPUT_ENCODING).strip()
             print("① 使用者輸入 ▶ 來源：" f"{input_path.resolve()}  (bytes={input_path.stat().st_size})")
             _process_single(Path(name).stem, text)
        except Exception as e:
             print(f"❌ 處理 '{name}' 時發生嚴重錯誤: {e}")
             _dlog(f"FATAL ERROR processing {name}: {traceback.format_exc()}")
             sys.exit(f"處理 '{name}' 失敗。") # 單檔案模式下失敗則退出
    else:
        # 批次模式
        processed_stems = {p.stem.split('_')[-1] for p in RES_DIR.glob("judge_result_*.txt")} # 以 judge 結果判斷是否完成
        all_files = sorted(USER_INPUT_DIR.glob("*.txt"))
        files_to_process = [p for p in all_files if p.stem not in processed_stems]

        print("🔍 批次模式 ▶ 總檔案 "
              f"{len(all_files)} 件, 已完成 {len(processed_stems)} 件, 本次待處理 {len(files_to_process)} 件")
        if not files_to_process:
             print("✅ 所有檔案皆已處理完成。")
             return

        print(f"Files to process: {[p.name for p in files_to_process]}")

        errors_occurred = False
        for path in files_to_process:
            nid = path.stem
            try:
                text = path.read_text(encoding=OUTPUT_ENCODING).strip()
                print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"▶▶ 開始處理：{nid}")
                print("① 使用者輸入 ▶ 來源：" f"{path.resolve()}  (bytes={path.stat().st_size})")
                _process_single(nid, text)
                print(f"◀◀ 完成：{nid}")
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            except Exception as e:
                print(f"❌ 處理 '{path.name}' 時發生錯誤: {e}")
                _dlog(f"ERROR processing batch file {path.name}: {traceback.format_exc()}")
                errors_occurred = True # 記錄錯誤，但繼續處理下一個文件
            finally:
                gc.collect() # 每次處理完畢後回收內存

        if errors_occurred:
             print("\n⚠️ 批次處理中有部分檔案發生錯誤，請檢查日誌。")
        else:
             print("\n✅ 批次處理完成，所有檔案皆成功。")


def run_self_test() -> bool:
    """
    執行 Pipeline 入口點自我檢測。
    檢查環境變數、路徑、外部檔案、工具可用性、模組匯入及下游 agent runner 自檢。
    """
    print("=" * 30)
    print("Running Pipeline Self-Test...")
    print("=" * 30)
    all_ok = True
    config_loaded = False
    paths_loaded = False
    agent_runner_checked = False
    agent_runner_ok = False
    env_path = Path(".env").resolve()

    # 1. 檢查 .env 檔案
    print("[Checking .env File]")
    if env_path.exists():
        print(f"  [OK] .env file found at: {env_path}")
        print(f"       (Already loaded by dotenv at module level)")
    else:
        print(f"  [WARN] .env file NOT FOUND at: {env_path}. Relying on system env vars.")

    # --- 關鍵環境變數檢查 (使用 os.getenv 直接讀取) ---
    print("\n[Checking Critical Environment Variables (Directly)]")
    critical_vars = {
        "OPENAI_API_KEY": None, # 檢查是否存在
        "USE_AGENT_RETRIEVAL": "1", # 預期預設值
        # 添加其他你認為絕對必要的變數和它們的預期預設值
        "ENABLE_LI_PG": "1",
        "ENABLE_VECTOR_FALLBACK": "1",
    }
    for var, default in critical_vars.items():
        val = os.getenv(var)
        if val is None:
            if default is None: # 表示必須存在
                 print(f"  [ERROR] Critical env var '{var}' is NOT SET.")
                 all_ok = False
            else:
                 print(f"  [INFO] Optional critical var '{var}' not set, using assumed default '{default}'.")
        else:
            masked_val = f"sk-...{val[-4:]}" if "API_KEY" in var else val
            print(f"  [OK] Critical env var '{var}' is set to: '{masked_val}'.")

    # --- 延遲載入依賴 ---
    print("\n[Lazy Loading Dependencies]")
    try:
        _lazy_load_dependencies()
        print("  [OK] Core dependencies seem loaded.")
        config_loaded = _CONFIG_MODULE is not None
        paths_loaded = _PATHS_MODULE is not None
    except Exception as e:
        print(f"  [ERROR] Failed during lazy loading: {e}\n{traceback.format_exc(limit=2)}")
        all_ok = False

    # --- 配置模組和變數檢查 (如果 config 載入成功) ---
    print("\n[Checking Config Module Variables (Imported vs Env)]")
    if config_loaded:
        config = _CONFIG_MODULE
        core_vars_check = {
            # key: (Expected Default, Imported Value)
            "LLM_ROUNDS": ("1", LLM_ROUNDS), # 這個在頂層定義了
            "OUTPUT_ENCODING": ("utf-8-sig", OUTPUT_ENCODING), # 這個也在頂層
            "CONCISENESS_MODEL": ("gpt-4o-mini", CONCISENESS_MODEL), # 這個也在頂層
            "VERIFIER_DEBUG": ("0", VERIFIER_DEBUG), # 這個也在頂層
            "MAX_AGENT_CHARS": ("15000", getattr(config, "MAX_AGENT_CHARS", "N/A")),
            "AGENT_MAX_STEPS": ("24", getattr(config, "AGENT_MAX_STEPS", "N/A")),
            "AGENT_TOP_K_MAX": ("200", getattr(config, "AGENT_TOP_K_MAX", "N/A")),
            "OPENAI_CHAT_MODEL": ("gpt-4o-mini", getattr(config, "OPENAI_CHAT_MODEL", "N/A")),
            "USE_AGENT_RETRIEVAL": ("1", os.getenv("USE_AGENT_RETRIEVAL", "1")), # 直接讀 env
        }
        for var, (default, imported) in core_vars_check.items():
            env_val = os.getenv(var)
            imported_str = str(imported)
            env_match = False # 預設不匹配

            # --- 處理 CONCISENESS_MODEL ---
            if var == "CONCISENESS_MODEL":
                openai_chat_model_env = os.getenv("OPENAI_CHAT_MODEL")
                # 預期值：如果 OPENAI_CHAT_MODEL 有設，用它的值；否則用 CONCISENESS_MODEL 的預設值
                expected_conciseness = openai_chat_model_env or default # 使用 core_vars_check 中定義的 default

                if env_val is None: # CONCISENESS_MODEL 環境變數未設定
                    print(f"  [INFO] '{var}' env var not set. Value derived from OPENAI_CHAT_MODEL.")
                    if imported_str == expected_conciseness:
                        print(f"    [OK] Imported value '{imported_str}' matches derived expectation '{expected_conciseness}'.")
                        env_match = True
                    else:
                        print(f"    [FAIL] Mismatch! Imported: '{imported_str}', Expected derived from OPENAI_CHAT_MODEL: '{expected_conciseness}'")
                        all_ok = False
                else: # CONCISENESS_MODEL 環境變數已設定 (優先級最高)
                     print(f"  [OK] '{var}' env var explicitly set to: '{env_val}'.")
                     if imported_str == env_val:
                          env_match = True
                     else:
                         print(f"    [FAIL] Mismatch! Imported: '{imported_str}', Explicit Env: '{env_val}'")
                         all_ok = False
                # 特殊處理結束後跳過通用檢查
                continue
            # --- 特殊處理結束 ---

            # --- 通用檢查 (適用於其他變數) ---
            if env_val is None:
                print(f"  [INFO] '{var}' not set. Using code default: '{default}'")
                if imported_str == default:
                    env_match = True
                else:
                    print(f"    [FAIL] Mismatch! Imported: '{imported_str}', Expected default: '{default}'")
                    all_ok = False
            else:
                 print(f"  [OK] '{var}' set to: '{env_val}'")
                 env_match = (imported_str == env_val) # 先進行字串比較

                 # --- 特別處理布林值 vs. 字串 "1"/"0" ---
                 is_bool_var = var in ["VERIFIER_DEBUG", "USE_AGENT_RETRIEVAL", "AUTO_HOPS_ENABLE",
                                     "ENABLE_LI_PG", "ENABLE_LI_ONLINE", "ENABLE_VECTOR_FALLBACK"]
                 if is_bool_var and not env_match:
                     bool_imported = None
                     if isinstance(imported, bool): bool_imported = imported
                     elif imported_str.lower() in ['true', '1']: bool_imported = True
                     elif imported_str.lower() in ['false', '0']: bool_imported = False

                     if bool_imported is not None:
                         if (bool_imported is True and env_val.lower() in ['true', '1']) or \
                            (bool_imported is False and env_val.lower() in ['false', '0']):
                             print(f"    [INFO] Allowing boolean ({imported}) == env string ('{env_val}') for {var}.")
                             env_match = True # 如果布林值和字串 "1"/"0" 對應，視為匹配
                 # --- 布林處理結束 ---

                 if not env_match:
                     print(f"    [FAIL] Mismatch! Imported: '{imported_str}', Env: '{env_val}'")
                     all_ok = False
            # --- 通用檢查結束 ---
    else:
        print("  [SKIP] Skipping config variable checks due to load failure.")

    # --- 路徑檢查 (如果 paths 載入成功) ---
    print("\n[Checking Path Configurations]")
    if paths_loaded:
        paths = _PATHS_MODULE
        paths_to_check = {
            "RES_DIR": RES_DIR, "USER_INPUT_DIR": USER_INPUT_DIR, "VEC_DIR": VEC_DIR,
            "CONCISENESS_DIR": CONCISENESS_DIR, "DIRT_DIR": DIRT_DIR, "DIRT_DEBUG_DIR": DIRT_DEBUG_DIR,
            "VERIFIER_DEBUG_PATH_Parent": VERIFIER_DEBUG_PATH.parent
        }
        for name, p in paths_to_check.items():
             if isinstance(p, Path):
                 print(f"  [OK] '{name}' points to: {p.resolve()}")
                 try:
                     p.mkdir(parents=True, exist_ok=True)
                     print(f"       Directory exists or was created.")
                 except OSError as e:
                     print(f"  [ERROR] Failed to create directory for '{name}' at {p.resolve()}: {e}")
                     all_ok = False
             else:
                 print(f"  [FAIL] '{name}' is not a Path object (Type: {type(p)})")
                 all_ok = False
    else:
        print("  [SKIP] Skipping path checks because paths module failed to load.")

    # --- 外部檔案依賴 ---
    print("\n[Checking External File Dependencies]")
    ext_files = {"Conciseness Prompt": CONCISENESS_PROMPT_PATH, "Relation Dict": _REL_PATH}
    for name, p in ext_files.items():
         if p is None:
             print(f"  [WARN] Path for '{name}' was not determined (likely lazy load issue).")
             continue # 不能檢查 None
         if p.is_file():
             print(f"  [OK] '{name}' found at: {p.resolve()}")
         else:
             print(f"  [ERROR] '{name}' NOT FOUND at: {p.resolve()}")
             all_ok = False

    # --- 工具可用性檢查 (需要 config 載入成功) ---
    print("\n[Checking Tool Availability (via Config)]")
    if config_loaded:
        config = _CONFIG_MODULE
        try:
            availability = config.get_tool_availability()
            print(f"  [INFO] Tool Status: {availability}")
            if "UNAVAILABLE" in availability:
                 print("    [WARN] One or more tools are unavailable. Check tool-specific env vars/dependencies.")
                 # (可以保留之前的具體提示)
        except Exception as e:
            print(f"  [FAIL] get_tool_availability() raised an error: {e}")
            all_ok = False
    else:
        print("  [SKIP] Skipping tool check due to config load failure.")

    # --- 核心模組匯入與功能檢查 ---
    print("\n[Checking Core Module Imports & Functionality]")
    modules_to_check = {
        "LLM Extract": (_LLM_EXTRACT_MODULE, "extract_entities_relations"),
        "LLM Judge": (_LLM_JUDGE_MODULE, "judge_news_kb"),
        "LLM Dirt Removal": (_LLM_DIRT_MODULE, "run_dirt_removal"),
        "Embeddings": (_EMBEDDINGS_MODULE, "embed_text"),
        "KG Search": (_KG_SEARCH_MODULE, "cosine_search"),
        "Deduplication": (_DEDUP_MODULE, "deduplicate"),
        "Agent Runner": (_AGENT_RUNNER_MODULE, "run_factcheck_middle"),
        "GPTClient (Optional)": (_LLM_GPT_MODULE, None), # 檢查模組本身是否載入
    }
    for name, (module, func_name) in modules_to_check.items():
         if module is None:
             if "Optional" not in name:
                 print(f"  [ERROR] Module '{name}' failed to load during lazy loading.")
                 all_ok = False
             else:
                  print(f"  [INFO] Optional module '{name}' is not available.")
         else:
             print(f"  [OK] Module '{name}' loaded.")
             if func_name:
                 if hasattr(module, func_name):
                     print(f"       Function '{func_name}' found.")
                 else:
                     print(f"  [ERROR] Module '{name}' is missing function '{func_name}'.")
                     all_ok = False

    # --- 下游 Agent Runner 自檢 ---
    print("\n[Checking Downstream Agent Runner Self-Test]")
    agent_runner_checked = True
    if _AGENT_RUNNER_MODULE is not None:
         if hasattr(_AGENT_RUNNER_MODULE, 'run_self_test'):
             print("  [INFO] Found 'run_self_test' in agent runner. Executing...")
             try:
                 # 執行 agent runner 的自檢
                 agent_runner_ok = _AGENT_RUNNER_MODULE.run_self_test() # type: ignore
                 if agent_runner_ok:
                     print("  [OK] Agent runner self-test PASSED.")
                 else:
                     print("  [ERROR] Agent runner self-test FAILED. See details above.")
                     all_ok = False # 如果下游失敗，整體也失敗
             except Exception as e:
                 print(f"  [ERROR] Agent runner self-test raised an exception: {e}\n{traceback.format_exc(limit=2)}")
                 all_ok = False
         else:
             print("  [WARN] Agent runner module loaded but has no 'run_self_test' function.")
             # 可以選擇是否將此視為錯誤
    else:
         print("  [SKIP] Skipping agent runner check because module failed to load.")
         # 如果 Agent Runner 是必須的 (USE_AGENT_RETRIEVAL=1)，這本身就是個問題
         if os.getenv("USE_AGENT_RETRIEVAL", "1") == "1":
             print("    [ERROR] Agent runner module failed to load, but USE_AGENT_RETRIEVAL=1 requires it.")
             all_ok = False


    # --- 最終結果 ---
    print("-" * 30)
    if all_ok:
        print("Self-Test Result: ALL CLEAR. Pipeline dependencies seem OK.")
    else:
        print("Self-Test Result: ERRORS or FAILURES found. Pipeline may not function correctly.")
    print("=" * 30)
    return all_ok


if __name__ == "__main__":
    _lazy_load_dependencies() # 先載入依賴，這樣 argparse 才能工作
    args = _parse_args()

    # 如果指定了 --self-test，執行自檢並退出
    if args.self_test or os.getenv("PIPELINE_SELF_TEST") == "1":
        print("Running self-test...")
        test_passed = run_self_test()
        sys.exit(0 if test_passed else 1) # 根據測試結果退出

    # --- 正常執行前的路徑和檔案檢查 ---
    print(f"⚙️ USER_INPUT_DIR: {USER_INPUT_DIR.resolve()} (exists: {USER_INPUT_DIR.is_dir()})")
    print(f"⚙️ RES_DIR:        {RES_DIR.resolve()} (exists: {RES_DIR.is_dir()})")
    try:
        files = list(USER_INPUT_DIR.glob("*.txt"))
        print(f"🔍 找到 {len(files)} 個輸入檔案。") # 簡化輸出
    except Exception as e:
        print(f"⚠️ [WARN] 無法列出輸入目錄中的檔案: {e}")

    # 正常執行 main
    main()