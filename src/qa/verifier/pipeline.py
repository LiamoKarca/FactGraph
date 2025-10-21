"""
新聞事實驗證主流程 Pipeline（StateGraph Agent 版）
================================================
- ① 簡潔化與抽取
- ④ KG 比對：USE_AGENT_RETRIEVAL=1 → 走 ReAct 代理；否則舊三分流
- ⑤ Dirt Removal
- ⑥ Judge（此步切換 gpt-5）

Usage:
$ python -m src.qa.verifier.pipeline [news_id.txt]
$ python -m src.qa.verifier.pipeline [news_id]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# 專案工具與模組
from ..tools import data_utils as du
from ..tools import kg_nl as knl
from .core.dedup import deduplicate
from .core.embeddings import embed_text, embed_triple
from .core.paths import RES_DIR, USER_INPUT_DIR, VEC_DIR
from .kg.search import cosine_search, kg_row_to_detail
from .llm.extract import (
    extract_entities_relations,
    set_debug_basename as set_extract_debug_basename,
)
from .llm.judge import judge_news_kb, set_debug_basename as set_judge_debug_basename
from .llm.dirt_removal import run_dirt_removal

# 簡潔（可選）
try:
    from .llm.gpt import GPTClient
except Exception:
    GPTClient = None  # type: ignore

# StateGraph Agent
try:
    from .agent_langchain import run_factcheck_middle as agent_run_middle
except Exception:
    agent_run_middle = None  # type: ignore

load_dotenv()

# ─────────── 常數 ───────────
LLM_ROUNDS = int(os.getenv("LLM_ROUNDS", "1"))
OUTPUT_ENCODING = "utf-8-sig"

CONCISENESS_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CONCISENESS_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "conciseness.txt"
CONCISENESS_DIR = Path("data/interim/verifier/conciseness")
CONCISENESS_HARD_TIMEOUT_SEC = 30

DIRT_DIR = RES_DIR / "dirt_removal"
DIRT_DEBUG_DIR = DIRT_DIR / "debug"

# ─────────── 除錯日誌設定 ───────────
VERIFIER_DEBUG = os.getenv("VERIFIER_DEBUG", "0") == "1"
VERIFIER_DEBUG_PATH = Path(os.getenv(
    "VERIFIER_DEBUG_PATH",
    "data/processed/verifier/debug/verifier_search_debug.log"
))
VERIFIER_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)

def _dlog(msg: str) -> None:
    if not VERIFIER_DEBUG:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        VERIFIER_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VERIFIER_DEBUG_PATH, "a", encoding="utf-8-sig") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass

# ─────────── 簡潔輸入 ───────────
def _ensure_conciseness_file(news_id: str, original_text: str) -> Path:
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

    if GPTClient is None or not os.getenv("OPENAI_API_KEY", "").strip():
        out_path.write_text(concise, encoding=OUTPUT_ENCODING)
        print("② 簡潔輸入 ▶ [WARN] 無 GPTClient 或無金鑰，已以原文回寫。")
        return out_path

    system_prompt = "擔任中文新聞綴字簡潔助手。僅做刪繁就簡，不得加入新資訊。"
    user_prompt = f"{tmpl}\n\n---\n【原始文字】\n{original_text.strip()}\n---\n請直接輸出簡潔後文本，不要夾帶說明或標籤。"

    try:
        try:
            gpt = GPTClient(api_key=os.getenv("OPENAI_API_KEY", ""), model_id=CONCISENESS_MODEL, temperature=0, timeout=CONCISENESS_HARD_TIMEOUT_SEC)
        except Exception:
            gpt = GPTClient(os.getenv("OPENAI_API_KEY", ""), CONCISENESS_MODEL, temperature=0, timeout=CONCISENESS_HARD_TIMEOUT_SEC)

        t0 = time.time()
        concise = gpt.chat(system_prompt, user_prompt, max_retries=0).strip()
        if (time.time() - t0) > CONCISENESS_HARD_TIMEOUT_SEC or not concise:
            concise = original_text.strip()
    except Exception as exc:
        print(f"② 簡潔輸入 ▶ [WARN] 生成簡潔檔失敗：{exc} → 以原文回寫。")
        concise = original_text.strip()

    out_path.write_text(concise, encoding=OUTPUT_ENCODING)
    print(f"② 簡潔輸入 ▶ 已寫入簡潔檔：{out_path.resolve()}")
    return out_path

# ─────────── ER → Triples（供舊三分流用；Agent 版由工具處理） ───────────
def _er_to_triples(data: Dict[str, Any]) -> List[du.Triple]:
    triples: List[du.Triple] = []
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

def _pull_triples(text_for_extraction: str) -> List[du.Triple]:
    all_rounds: List[List[du.Triple]] = []
    last_error: Exception | None = None
    rounds = int(os.getenv("LLM_ROUNDS", "1"))
    _dlog(f"extract: rounds={rounds}, text_chars={len(text_for_extraction)}")
    for i in range(rounds):
        print(f"③ 抽取 ▶ GPT 抽取 round {i + 1}")
        start = time.time()
        raw = extract_entities_relations(text_for_extraction)
        elapsed = time.time() - start
        print(f"   ↳ 完成，用時 {elapsed:.1f}s")

        if not raw:
            print(f"③ 抽取 ▶ [WARN] 抽取回傳為空，跳過 round {i + 1}")
            continue
        try:
            payload = json.loads(raw.replace("`", ""))
            if isinstance(payload, dict) and ("entities" in payload or "relations" in payload):
                triples = _er_to_triples(payload)
            else:
                triples = du.json_to_triples(payload) or []
            if triples:
                all_rounds.append(triples)
        except Exception as exc:
            last_error = exc
            print(f"③ 抽取 ▶ [WARN] JSON 解析失敗於 round {i + 1}: {exc}")
    if not all_rounds:
        if last_error:
            print(f"③ 抽取 ▶ [ERROR] 所有輪次皆失敗: {last_error}", file=sys.stderr)
        return []
    triples_merged = du.merge_triples(*all_rounds)
    print(f"③ 抽取 ▶ 合併後三元組數：{len(triples_merged)}")
    _dlog(f"extract_done: triples={len(triples_merged)}")
    return triples_merged

# ─────────── 舊三分流（當 USE_AGENT_RETRIEVAL=0 或 Agent 失敗時回退） ───────────
def _init_li_pg():
    from ..tools.property_graph.li_csv_pg_retriever import CsvPropertyGraphRetriever  # type: ignore
    return CsvPropertyGraphRetriever.ensure_built_and_loaded()

def _init_li_online():
    from .kg.llamaIndex.neo4j_li_retriever import LlamaIndexNeo4jRetriever  # type: ignore
    return LlamaIndexNeo4jRetriever(date_field="date", evidence_field="evidence")

def _safe_build_block(tris: List[Dict], details: Dict[Tuple[str, str, str], Dict]) -> str:
    lines: List[str] = []
    for tri in tris:
        key = (tri["head"], tri["relation"], tri["tail"])
        det = details.get(key, {})
        expl = det.get("rel", {}).get("evidence", "") or det.get("rel", {}).get("desc", "") or ""
        date_str = det.get("rel", {}).get("date", "")
        line = f"[比對] {tri['head']} 透過關係與 {tri['tail']} 建立連結；說明：{expl}" + (f"；日期：{date_str}" if date_str else "")
        lines.append(line)
    return "\n".join(lines)

import importlib.util as _impspec
_REL_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "knowledge-graph" / "relation_dict_all.py"
try:
    spec = _impspec.spec_from_file_location("relation_dict_all", _REL_PATH)
    rel_module = _impspec.module_from_spec(spec)
    spec.loader.exec_module(rel_module)  # type: ignore
    _STRONG_REL_KEYWORDS = set(rel_module.RELATIONS_ALL)
    print(f"⚙️ 已載入外部關鍵詞庫，共 {len(_STRONG_REL_KEYWORDS)} 個關鍵詞。")
except Exception as exc:
    print(f"⚠️ [WARN] 無法載入 relation_dict_all.py：{exc}")
    _STRONG_REL_KEYWORDS = set()

def _specificity(tri: Dict[str, str]) -> int:
    score = 0
    if (tri.get("head") or "").strip():
        score += 1
    rel = (tri.get("relation") or "").strip()
    if rel:
        score += 1
        if any(k in rel for k in _STRONG_REL_KEYWORDS) and len(rel) <= 6:
            score += 1
    if (tri.get("tail") or "").strip():
        score += 1
    return min(score, 4)

def _decide_hops_strategy(tp: Dict[str, str], news_len: int, path_label: str) -> Tuple[int, int, int]:
    auto_enable = os.getenv("AUTO_HOPS_ENABLE", "1") == "1"
    hard_max = int(os.getenv("AUTO_MAX_HOPS_HARD", "3"))
    base_min_hits = int(os.getenv("AUTO_BASE_MIN_HITS", "6"))
    long_news_chars = int(os.getenv("AUTO_LONG_NEWS_CHARS", "400"))
    default_start = int(os.getenv("LI_PG_HOPS" if path_label == "PG" else "LI_ONLINE_HOPS", "2"))
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
    start_hops, max_hops, min_hits = _decide_hops_strategy(tp, news_len, path_label)
    for hops in range(max(1, start_hops), max_hops + 1):
        hits = retriever.search_triple(tp, top_k=top_k, hops=hops)
        print(f"④ KG 比對 ▶ {path_label} auto-hops：hops={hops} → 命中 {len(hits)}（門檻 {min_hits}）")
        if len(hits) >= min_hits or hops >= max_hops:
            return hits
    return []

# ─────────── 主處理 ───────────
def _collect_hits_to_lines(hits: List[Tuple[Dict, Dict]]) -> List[str]:
    tris: List[Dict] = []
    dets: Dict[Tuple[str, str, str], Dict] = {}
    for tri, det in hits:
        tris.append(tri)
        dets[(tri["head"], tri["relation"], tri["tail"])] = det
    block = knl.build_block(tris, dets) if hasattr(knl, "build_block") else _safe_build_block(tris, dets)
    return block.splitlines()

def _process_single(news_id: str, text: str) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    DIRT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # ① 使用者輸入 → 文本清理與向量保存
    text = text.replace("`", "")
    input_vec_path = VEC_DIR / f"{news_id}.npy"
    np.save(input_vec_path, embed_text(text))
    print("① 使用者輸入 ▶ 檔案："
          f"{(USER_INPUT_DIR / (news_id + '.txt')).resolve()}  → 向量儲存："
          f"{input_vec_path.resolve()}")
    _dlog(f"step1_input: news_id={news_id}, chars={len(text)}")

    # ② 簡潔輸入
    concise_path = _ensure_conciseness_file(news_id, text)
    try:
        concise_text = concise_path.read_text(encoding=OUTPUT_ENCODING).strip()
    except Exception as exc:
        print(f"② 簡潔輸入 ▶ [WARN] 讀取簡潔檔失敗：{exc} → 改用原文。")
        concise_text = text
    news_len = len(concise_text)
    _dlog(f"step2_concise: chars={news_len}, file={concise_path}")

    # Debug 檔名前綴
    set_extract_debug_basename(news_id)
    set_judge_debug_basename(news_id)

    # ③ 抽取（僅供舊三分流；Agent 會自行抽取）
    triples = _pull_triples(concise_text)

    # ④ KG 比對：ReAct 代理或舊三分流
    use_agent = os.getenv("USE_AGENT_RETRIEVAL", "1") == "1"
    if use_agent:
        if agent_run_middle is None:
            sys.exit("❌ ④ KG 比對 ▶ USE_AGENT_RETRIEVAL=1，但 agent 模組不可用")
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
            if use_li_pg and li_pg is not None:
                try:
                    hits = (_query_with_autohops(li_pg, tp, news_len, "PG", auto_topk)
                            if auto_enable else li_pg.search_triple(tp, top_k=50, hops=li_pg_hops))
                    if hits:
                        raw_lines.extend(_collect_hits_to_lines(hits))
                        handled = True
                except Exception as exc:
                    print(f"④ KG 比對 ▶ [WARN] 離線 PG 檢索失敗：{exc}")

            if not handled and use_li_online and li_online is not None:
                try:
                    hits = (_query_with_autohops(li_online, tp, news_len, "Neo4j", auto_topk)
                            if auto_enable else li_online.search_triple(tp, top_k=50, hops=li_online_hops))
                    if hits:
                        raw_lines.extend(_collect_hits_to_lines(hits))
                        handled = True
                except Exception as exc:
                    print(f"④ KG 比對 ▶ [WARN] Neo4j 檢索失敗：{exc}")

            if not handled and use_vec_fallback:
                try:
                    q_vec = embed_triple(tp)
                    for idx in cosine_search(tp, q_vec):
                        tri, det = kg_row_to_detail(idx)
                        raw_lines.extend(_collect_hits_to_lines([(tri, det)]))
                    handled = True
                except Exception as exc:
                    print(f"④ KG 比對 ▶ [WARN] 向量檢索失敗：{exc}")

            if not handled:
                print("④ KG 比對 ▶ [WARN] 此三元組未命中任何路徑。")

        if not raw_lines:
            sys.exit("⚠️ ④ KG 比對 ▶ 無命中，流程終止")
        kept = deduplicate(raw_lines, triples=triples)
        final = [re.sub(r"^\d+\.", f"[{i}]", ln, count=1) for i, ln in enumerate(kept, 1)]
        news_block = "[原始文本]\n" + text
        kb_block = "[比對知識]\n" + "\n".join(final)
        middle_text = f"{news_block}\n\n{kb_block}"

    # —— 寫入中間產物 —— #
    kg_file = RES_DIR / f"news_kg_{news_id}.txt"
    kg_file.write_text(middle_text, encoding=OUTPUT_ENCODING)
    print(f"④ KG 比對 ▶ 已寫入：{kg_file.resolve()}")

    # ⑤ Dirt Removal
    # === 保證 Dirt Removal 時仍含原始新聞 ===
    combined_text_raw = kg_file.read_text(encoding=OUTPUT_ENCODING)
    if "[原始新聞]" not in combined_text_raw:
        # 若 Agent 版僅有比對知識，手動補上原文
        combined_text = (
            f"[原始新聞]\n{text.strip()}\n\n"
            f"[比對知識]\n{combined_text_raw.strip()}"
        )
    else:
        combined_text = combined_text_raw

    try:
        filtered_text, dirt_txt_path = run_dirt_removal(
            combined_text=combined_text, news_id=news_id, out_dir=str(RES_DIR)
        )
        print(f"⑤ Dirt Removal ▶ debug JSON：{(DIRT_DEBUG_DIR / f'news_kg_{news_id}.json').resolve()}")
        print(f"⑤ Dirt Removal ▶ 過濾後文本：{Path(dirt_txt_path).resolve()}")
    except Exception as exc:
        print(f"⑤ Dirt Removal ▶ [WARN] 流程失敗：{exc} → 以未過濾版本進入判斷。")
        filtered_text = combined_text
        dirt_txt_path = str(kg_file)

    # ⑥ Judge（此步切換 gpt-5）
    prev_model = os.getenv("OPENAI_CHAT_MODEL", "")
    try:
        os.environ["OPENAI_CHAT_MODEL"] = "gpt-5"
        set_judge_debug_basename(news_id)
        try:
            judge_input = Path(dirt_txt_path).read_text(encoding=OUTPUT_ENCODING)
            judge_input_src = Path(dirt_txt_path).resolve()
        except Exception:
            judge_input = filtered_text
            judge_input_src = Path(dirt_txt_path).resolve() if dirt_txt_path else kg_file.resolve()
        judge_file = RES_DIR / f"judge_result_{news_id}.txt"
        print(f"⑥ 判斷 ▶ 使用模型：gpt-5，輸入來源檔：{judge_input_src}")
        judged_raw = judge_news_kb(judge_input, debug_name=news_id)
        judged_clean = judged_raw.replace("`", "")
        judge_file.write_text(judged_clean, encoding=OUTPUT_ENCODING)
        print(f"⑥ 判斷 ▶ 已寫入：{judge_file.resolve()}")
    finally:
        if prev_model:
            os.environ["OPENAI_CHAT_MODEL"] = prev_model

    print("✅ 完成 ▶ 主要輸出：\n"
          f"   - KG：{kg_file.resolve()}\n"
          f"   - Dirt Removal：{Path(dirt_txt_path).resolve()}\n"
          f"   - Judge：{(RES_DIR / f'judge_result_{news_id}.txt').resolve()}")

# ─────────── 入口與批次 ───────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("FactGraph Verifier Pipeline")
    p.add_argument("news_id", nargs="?", help="新聞檔名（含或不含 .txt），留空則批次所有")
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    if args.news_id:
        name = args.news_id if args.news_id.endswith(".txt") else (f"{args.news_id}.txt")
        input_path = USER_INPUT_DIR / name
        if not input_path.is_file():
            sys.exit(f"❌ 找不到檔案：{input_path}")
        text = input_path.read_text(encoding=OUTPUT_ENCODING).strip()
        print("① 使用者輸入 ▶ 來源：" f"{input_path.resolve()}  (bytes={input_path.stat().st_size})")
        _process_single(Path(name).stem, text)
    else:
        processed = {p.stem.removeprefix("news_kg_") for p in RES_DIR.glob("news_kg_*")}
        files = sorted(USER_INPUT_DIR.glob("*.txt"))
        print("🔍 批次模式 ▶ 待處理 "
              f"{len(files)} 件（已存在 KG 的將略過）：{[p.name for p in files]}")
        for path in files:
            nid = path.stem
            if nid in processed:
                print(f"⏭️  略過 ▶ 已有 KG 檔：news_kg_{nid}.txt")
                continue
            text = path.read_text(encoding=OUTPUT_ENCODING).strip()
            print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"▶▶ 開始處理：{nid}")
            print("① 使用者輸入 ▶ 來源：" f"{path.resolve()}  (bytes={path.stat().st_size})")
            _process_single(nid, text)
            print(f"◀◀ 完成：{nid}")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    gc.collect()

if __name__ == "__main__":
    print(f"⚙️ USER_INPUT_DIR: {USER_INPUT_DIR.resolve()} (exists: {USER_INPUT_DIR.is_dir()})")
    print(f"⚙️ RES_DIR:        {RES_DIR.resolve()} (exists: {RES_DIR.is_dir()})")
    files = list(USER_INPUT_DIR.glob("*.txt"))
    print(f"🔍 找到 {len(files)} 個輸入檔案：{[p.name for p in files]}")
    main()
