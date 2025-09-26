"""
answerer ─ 問答主流程（Orchestrator）

0. python -m experiment.answerer.pipeline <id.txt>
1. 讀取問題 → slug
2. 抽取三元組（gpt-4o-mini）
3. 向量檢索 KG
4. 去重
5. 最終評估（優先 gpt-5；若截斷/推理吃光 → 自動續寫；仍不行才 fallback 到 gpt-4o-mini）
6. 依輸入檔名輸出 user_kg_*.txt / user_qa_judge_*.txt
"""

from __future__ import annotations
from .core.embedding import load_embedder, embed_triple, embed_text, dedupe
from .core.paths import (
    CKIP_ROOT,
    KG_EMB_PATH,
    KG_CSV_PATH,
    OUT_DIR,
    USER_INPUT_DIR,
    EXTRACT_PROMPT_PATH,
    JUDGE_PROMPT_PATH,
)
from .core.utils import safe_json_loads, clean_json_block
from .kg.loader import load_kg_vectors, load_kg_df
from .kg.search import search_by_triples
from .llm.gpt import GPTClient
from .llm.prompt_loader import load_prompt
from .tools import data_utils as du
from .tools import kg_nl as knl

import argparse
import os
import re
import sys
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable

from dotenv import load_dotenv
load_dotenv()

# ───────────────────────────── 參數設定 ─────────────────────────
SIM_TH: float = 0.80
TOP_K: int = 100
TAIL_PLACEHOLDER: str = "未知"
USE_ATTR_EMBED: bool = True

EXTRACT_MODEL_ID = os.getenv("ANSWERER_EXTRACT_MODEL", "gpt-4o-mini")
JUDGE_MODEL_ID = os.getenv("ANSWERER_JUDGE_MODEL", "gpt-5")
JUDGE_FALLBACK_MODEL_ID = os.getenv(
    "ANSWERER_JUDGE_FALLBACK_MODEL", "gpt-4o-mini")

# ──────────────────────────── 輔助函式 ───────────────────────────


def _adapt_extracted_to_v1(
    data: Any,
    tail_placeholder: str = TAIL_PLACEHOLDER
) -> Tuple[List[Dict[str, str]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    triples_v1: List[Dict[str, str]] = []
    meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    if not isinstance(data, dict) or "triples" not in data:
        legacy = du.json_to_triples(data) or []
        return legacy, meta

    for t in data.get("triples", []):
        subj_raw = t.get("subject")
        if isinstance(subj_raw, dict):
            head_text = (subj_raw.get("text") or "未知").strip()
            subj_type = subj_raw.get("type") or "未知"
            subj_attr = subj_raw.get("attributes") or {}
        else:
            head_text = (subj_raw or "未知").strip()
            subj_type = "未知"
            subj_attr = {}

        relation = (t.get("relation") or "未知").strip()

        if "object" in t:
            tail_text = (t.get("object") or "未知").strip()
            target_type = "未知"
            target_attr = {}
        else:
            tail_text = tail_placeholder
            info_need = (t.get("info_need") or {})
            target = (info_need.get("target") or {}) if isinstance(
                info_need, dict) else {}
            target_type = target.get("type") or "未知"
            target_attr = target.get("attributes") or {}

        triple_v1 = {"head": head_text,
                     "relation": relation, "tail": tail_text}
        triples_v1.append(triple_v1)

        meta[(head_text, relation, tail_text)] = {
            "subject": {"text": head_text, "type": subj_type, "attributes": subj_attr},
            "target": {"type": target_type, "attributes": target_attr},
        }

    return triples_v1, meta


def _make_attr_embed_fn(
    emb,
    meta: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Callable[[Dict[str, str]], "np.ndarray"]:
    def _attrs_to_text(attrs: Dict[str, Any]) -> str:
        if not attrs:
            return ""
        keys_priority = ["unit", "time", "scope", "location",
                         "version", "article", "party", "office"]
        parts: List[str] = []
        for k in keys_priority:
            v = attrs.get(k)
            if v not in (None, "", "未知"):
                parts.append(f"{k}={v}")
        for k, v in attrs.items():
            if k not in keys_priority and v not in (None, "", "未知"):
                parts.append(f"{k}={v}")
        return " ".join(parts)

    def embed_fn(tp: Dict[str, str]) -> "np.ndarray":
        key = (tp["head"], tp["relation"], tp["tail"])
        mi = meta.get(key, {"subject": {}, "target": {}})

        subj = mi.get("subject", {})
        targ = mi.get("target", {})

        subj_tag = subj.get("type", "未知")
        subj_attr = _attrs_to_text(subj.get("attributes", {}))
        target_type = targ.get("type", "未知")
        target_attr = _attrs_to_text(targ.get("attributes", {}))

        query_text = (
            f"HEAD:{tp['head']} ({subj_tag}) "
            f"REL:{tp['relation']} "
            f"NEEDS:{target_type} "
            f"{('[SUBJ] ' + subj_attr + ' ') if subj_attr else ''}"
            f"{('[NEED] ' + target_attr) if target_attr else ''}"
        ).strip()

        vec = embed_text(emb, query_text)
        n = float((vec ** 2).sum()) ** 0.5
        if n > 0:
            vec = vec / n
        return vec

    return embed_fn


# ──────────────────────────────── 主流程 ───────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answerer pipeline: 指定問題檔案 <id>.txt")
    parser.add_argument(
        "input_file", help="Path or filename of question file, e.g. '2024-...txt'")
    args = parser.parse_args()

    # 讀取問題檔案（utf-8-sig）
    input_path = Path(args.input_file)
    if not input_path.is_file():
        candidate = Path(USER_INPUT_DIR) / args.input_file
        if candidate.is_file():
            input_path = candidate
    if not input_path.is_file():
        for p in Path(USER_INPUT_DIR).rglob(Path(args.input_file).name):
            input_path = p
            break
    if not input_path.is_file():
        sys.exit(f"❌ 無效的輸入檔案: {args.input_file}")

    question = input_path.read_text(encoding="utf-8-sig").strip()
    slug = input_path.stem
    print(f"🔸 Question: {question}")

    # 資源初始化
    emb = load_embedder(CKIP_ROOT)
    kg_vecs, kg_vecs_norm = load_kg_vectors(KG_EMB_PATH)
    kg_df, hp_col, rp_col, tp_col = load_kg_df(KG_CSV_PATH)
    extract_prompt = load_prompt(EXTRACT_PROMPT_PATH)
    judge_prompt = load_prompt(JUDGE_PROMPT_PATH)

    if not judge_prompt or not judge_prompt.strip():
        print("[WARN] 判定用 system prompt (character.txt) 為空或讀取失敗。")

    # 建立輸出資料夾（成果與 debug 分開）
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR = OUT_DIR / "debug"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # A) 抽取三元組：gpt-4o-mini（預設不設上限）
    gpt_extract = GPTClient(
        api_key=os.getenv("GPT_API"),
        model_id=EXTRACT_MODEL_ID,
        temperature=0.2,
        top_p=0.9,
        # 不傳 max_completion_tokens → 交由模型最大輸出
    )

    # B) 判定：gpt-5（不送抽樣參數、也不送上限）
    gpt_judge = GPTClient(
        api_key=os.getenv("GPT_API"),
        model_id=JUDGE_MODEL_ID,
        # 不傳 max_completion_tokens → 交由模型最大輸出
    )

    # 2. 抽取
    raw_resp = gpt_extract.chat(extract_prompt, question)
    print("🪵 GPT raw response:\n", raw_resp)

    # 擷取 JSON block
    block = clean_json_block(raw_resp)
    cleaned = re.sub(r'^\s*json\s*', '', block, flags=re.IGNORECASE)
    cleaned = cleaned.replace("`", "").strip()
    print("🪵 Cleaned JSON block:\n", cleaned)

    try:
        data = safe_json_loads(cleaned)
    except Exception:
        print("[ERROR] 無法解析 JSON，cleaned 內容如下：", cleaned)
        sys.exit("❌ GPT 回傳的內容不是合法 JSON，請檢查模型輸出與 prompt 設定")

    # 轉舊介面 triple v1 + meta
    triples_v1, meta = _adapt_extracted_to_v1(data)
    print(f"🪲 Parsed triples count: {len(triples_v1)}")
    if not triples_v1:
        sys.exit("❌ GPT 未抽取到三元組")

    # 3. KG 檢索
    embed_fn: Callable[[Dict[str, str]], "np.ndarray"]
    if USE_ATTR_EMBED:
        embed_fn = _make_attr_embed_fn(emb, meta)
    else:
        def embed_fn(tp): return embed_triple(emb, tp)

    raw_lines = search_by_triples(
        triples=triples_v1,
        embed_fn=embed_fn,
        kg_vecs_norm=kg_vecs_norm,
        top_k=TOP_K,
        sim_th=SIM_TH,
        kg_df=kg_df,
        hp_col=hp_col,
        rp_col=rp_col,
        tp_col=tp_col,
        build_block_fn=knl.build_block,
    )
    if not raw_lines:
        sys.exit("⚠️ KG 無任何匹配")

    # 4. 去重
    final_lines = dedupe(
        raw_lines,
        embed_fn=lambda ln: embed_text(emb, ln),
        threshold=0.80,
    )

    # 5. 成果輸出（utf-8-sig）
    kg_out = OUT_DIR / f"user_kg_{slug}.txt"
    judge_out = OUT_DIR / f"user_qa_judge_{slug}.txt"
    debug_in = DEBUG_DIR / f"debug_judge_input_{slug}.txt"
    debug_out = DEBUG_DIR / f"debug_judge_output_{slug}.txt"

    kg_text = (
        "[使用者提問]\n"
        f"{question}\n\n[知識查詢結果]\n"
        + "\n".join(final_lines)
        + "\n"
    )
    kg_out.write_text(kg_text, encoding="utf-8-sig")

    # 6. 判定（gpt-5 → 若截斷/推理吃光 → 自動續寫最多 2 次 → 再不行才 fallback）
    def _run_and_log(model_client: GPTClient, tag: str, user_text: str) -> tuple[str, dict]:
        txt, finish_reason, usage, model_id = model_client.chat_with_meta(
            judge_prompt, user_text)
        debug_in.write_text(
            f"[MODEL] {tag}\n[MODEL_ID] {model_id}\n"
            f"[SYSTEM PROMPT]\n{judge_prompt}\n\n[USER INPUT]\n{user_text}\n",
            encoding="utf-8-sig",
        )
        debug_out.write_text(
            f"[MODEL] {tag}\n[FINISH] {finish_reason}\n[USAGE] {usage}\n"
            f"[LEN]={len(txt)}\n[RAW OUTPUT]\n{txt}\n",
            encoding="utf-8-sig",
        )
        return txt, {"finish_reason": finish_reason, "usage": usage, "model_id": model_id}

    def _reasoning_starved(meta: dict, text: str) -> bool:
        usage = meta.get("usage") or {}
        comp = usage.get("completion_tokens")
        # 有產生 completion tokens 但沒有可見文字
        return (comp is not None and comp > 0) and (not text.strip())

    # 首次判定
    judge_result, meta_info = _run_and_log(gpt_judge, JUDGE_MODEL_ID, kg_text)

    # 若被長度截斷或推理吃光，啟動「自動續寫」最多 2 次
    def _continue_prompt(prev_output: str) -> str:
        return (
            kg_text
            + "\n\n[上次輸出續寫]\n"
            + prev_output
            + "\n\n請從上次停下的地方繼續完整輸出，不要重複前文。"
        )

    continue_round = 0
    while continue_round < 2:
        need_more = (
            (meta_info.get("finish_reason") == "length")
            or _reasoning_starved(meta_info, judge_result)
        )
        if not need_more:
            break
        continue_round += 1
        print(f"[INFO] gpt-5 續寫 round {continue_round} ...")
        cont_input = _continue_prompt(judge_result)
        cont_text, meta_info = _run_and_log(
            gpt_judge, f"{JUDGE_MODEL_ID}-cont{continue_round}", cont_input)
        judge_result = (judge_result + "\n" + cont_text).strip()

    # 若仍空才 fallback
    if not judge_result.strip():
        print(f"[WARN] 判定輸出為空，啟用 fallback → {JUDGE_FALLBACK_MODEL_ID}")
        gpt_judge_fb = GPTClient(
            api_key=os.getenv("GPT_API"),
            model_id=JUDGE_FALLBACK_MODEL_ID,
            temperature=0.3,
            top_p=0.9,
            # 一樣不設上限
        )
        judge_result, _ = _run_and_log(
            gpt_judge_fb, f"{JUDGE_FALLBACK_MODEL_ID}", kg_text)

    if not judge_result.strip():
        print("[ERROR] 判定仍為空，請檢查 debug 檔案內容。")
        (OUT_DIR / f"user_qa_judge_{slug}.txt").write_text(
            "[ERROR] 模型回傳空白，請檢查 debug 檔。", encoding="utf-8-sig")
        sys.exit(1)

    judge_out.write_text((judge_result), encoding="utf-8-sig")

    print("✅ finished; outputs saved under", OUT_DIR)
    print("   KG    →", kg_out.name)
    print("   JUDGE →", judge_out.name)


if __name__ == "__main__":
    main()
