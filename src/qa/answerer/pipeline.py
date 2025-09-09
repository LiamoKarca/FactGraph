"""
answerer ─ 問答主流程（Orchestrator）

職責只做「流程協調」，不處理繁雜業務邏輯：
0. python -m src.qa.answerer.pipeline <id.txt>
1. 讀取使用者問題（檔案或 stdin）並取得 slug
2. 呼叫 GPT 抽取三元組
3. 以向量搜尋 KG 相關敘述
4. 去重（相似僅保留最長條目）
5. 呼叫 GPT 評估最終結果
6. 依輸入檔名動態輸出 user_kg_*.txt 與 user_qa_judge_*.txt
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
from ..tools import data_utils as du
from ..tools import kg_nl as knl

import argparse
import json
import os
import re
import sys
import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

from dotenv import load_dotenv
load_dotenv()

# ───────────────────────────── 參數設定 ─────────────────────────
SIM_TH: float = 0.80  # KG 相似度門檻
TOP_K: int = 100      # 每個三元組取前 TOP_K 條
TAIL_PLACEHOLDER: str = "未知"  # 新版抽取格式無 object 時的尾詞佔位
USE_ATTR_EMBED: bool = True    # 保持 False → 完全沿用舊 embed_triple 行為（輸出模式不變）


# ──────────────────────────── 輔助函式 ───────────────────────────
def _adapt_extracted_to_v1(
    data: Any,
    tail_placeholder: str = TAIL_PLACEHOLDER
) -> Tuple[List[Dict[str, str]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    """
    同時支援兩種抽取格式：
      (A) 舊：{"triples":[{"subject":"A","relation":"R","object":"B"}, ...]}
      (B) 新：{"triples":[{"subject":{"text":"A","type":"人物","attributes":{}}},
                         {"relation":"R"},
                         {"info_need":{"target":{"type":"數值","attributes":{...}}}} ]}
    轉成舊介面可食用的 triple v1（head, relation, tail），並返回 meta（供進階 embed 使用）。

    Returns:
        triples_v1: List[{'head','relation','tail'}]
        meta: dict; key=(head,relation,tail) → {
                'subject': {'text','type','attributes':{}},
                'target':  {'type','attributes':{}}
              }
    """
    triples_v1: List[Dict[str, str]] = []
    meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    if not isinstance(data, dict) or "triples" not in data:
        # 交給舊的 util 嘗試（兼容你原本的 fallback）
        legacy = du.json_to_triples(data) or []
        return legacy, meta

    for t in data.get("triples", []):
        # 兼容 subject 是字串或物件
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

        # 舊格式：若存在 object 就用；否則用佔位
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
    emb,  # 原本 load_embedder(...) 的回傳物件
    meta: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> Callable[[Dict[str, str]], "np.ndarray"]:
    """
    將新版 target/attributes 注入檢索語意的 embed_fn。
    預設 pipeline 不啟用，以確保語義/行為與舊版一致。
    """

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

        # 將需求語意顯式放入 query 文本
        query_text = (
            f"HEAD:{tp['head']} ({subj_tag}) "
            f"REL:{tp['relation']} "
            f"NEEDS:{target_type} "
            f"{('[SUBJ] ' + subj_attr + ' ') if subj_attr else ''}"
            f"{('[NEED] ' + target_attr) if target_attr else ''}"
        ).strip()

        vec = embed_text(emb, query_text)  # 與你現有的 embed_text 一致
        # 正規化（維持與 kg_vecs_norm 的 cosine 對齊）
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

    # 讀取問題檔案
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

    question = input_path.read_text(encoding="utf-8").strip()
    slug = input_path.stem
    print(f"🔸 Question: {question}")

    # 資源初始化
    emb = load_embedder(CKIP_ROOT)
    kg_vecs, kg_vecs_norm = load_kg_vectors(KG_EMB_PATH)
    kg_df, hp_col, rp_col, tp_col = load_kg_df(KG_CSV_PATH)
    extract_prompt = load_prompt(EXTRACT_PROMPT_PATH)
    judge_prompt = load_prompt(JUDGE_PROMPT_PATH)
    gpt = GPTClient(
        api_key=os.getenv("GPT_API"),
        model_id=os.getenv("GPT_MODEL", "gpt-4o"),
        temperature=0.4,
        top_p=0.9,
        max_tokens=2048,
    )

    # 2. 呼叫 GPT 抽取三元組
    raw_resp = gpt.chat(extract_prompt, question)
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

    # 兼容新舊抽取格式 → 舊介面 triple v1 + meta
    triples_v1, meta = _adapt_extracted_to_v1(data)
    print(f"🪲 Parsed triples count: {len(triples_v1)}")
    if not triples_v1:
        sys.exit("❌ GPT 未抽取到三元組")

    # 3. KG 向量檢索
    # 預設：完全沿用舊 embed_triple 行為（輸出/語義最保守）
    embed_fn: Callable[[Dict[str, str]], "np.ndarray"]
    if USE_ATTR_EMBED:
        embed_fn = _make_attr_embed_fn(emb, meta)  # 啟用屬性強化檢索
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

    # 4. 語意去重
    final_lines = dedupe(
        raw_lines,
        embed_fn=lambda ln: embed_text(emb, ln),
        threshold=0.80,
    )

    # 5. 輸出至檔案（維持原格式）
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kg_out = OUT_DIR / f"user_kg_{slug}.txt"
    judge_out = OUT_DIR / f"user_qa_judge_{slug}.txt"

    kg_out.write_text(
        "[使用者提問]\n"
        f"{question}\n\n[知識查詢結果]\n"
        + "\n".join(final_lines)
        + "\n",
        encoding="utf-8",
    )

    # 6. GPT 最終判斷（維持原處理）
    judge_result = gpt.chat(
        judge_prompt, kg_out.read_text(encoding="utf-8-sig"))
    judge_result = (
        judge_result
        .replace("`", "")
        .replace("#", "")
        .replace("*", "")
    )
    judge_out.write_text(judge_result, encoding="utf-8-sig")

    print("✅ finished; outputs saved under", OUT_DIR)
    print("   KG    →", kg_out.name)
    print("   JUDGE →", judge_out.name)


if __name__ == "__main__":
    main()
