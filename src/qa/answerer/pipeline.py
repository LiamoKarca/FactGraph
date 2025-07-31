#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from ..tools import kg_nl as knl
from ..tools import data_utils as du
from .llm.prompt_loader import load_prompt
from .llm.gpt import GPTClient
from .kg.search import search_by_triples
from .kg.loader import load_kg_vectors, load_kg_df
from .core.utils import safe_json_loads, clean_json_block
from .core.paths import (
    CKIP_ROOT,
    KG_EMB_PATH,
    KG_CSV_PATH,
    OUT_DIR,
    USER_INPUT_DIR,
    EXTRACT_PROMPT_PATH,
    JUDGE_PROMPT_PATH,
)
from .core.embedding import load_embedder, embed_triple, embed_text, dedupe

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ──────────────────────── 本專案自製模組 ────────────────────────
# 需用到 qa.tools 生成敘述區塊

# ───────────────────────────── 參數設定 ─────────────────────────
SIM_TH: float = 0.80  # KG 相似度門檻
TOP_K: int = 100  # 每個三元組取前 TOP_K 條


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
    # 移除所有反引號，並去掉可能的 "json" 前綴
    cleaned = re.sub(r'^\s*json\s*', '', block, flags=re.IGNORECASE)
    cleaned = cleaned.replace("`", "").strip()
    print("🪵 Cleaned JSON block:\n", cleaned)

    try:
        data = safe_json_loads(cleaned)
    except Exception as e:
        print("[ERROR] 無法解析 JSON，cleaned 內容如下：", cleaned)
        sys.exit("❌ GPT 回傳的內容不是合法 JSON，請檢查模型輸出與 prompt 設定")

    if isinstance(data, dict) and "triples" in data:
        triples = [
            {"head": t["subject"], "relation": t["relation"],
                "tail": t["object"]}
            for t in data["triples"]
            if t.get("subject") and t.get("relation")
        ]
    else:
        triples = du.json_to_triples(data) or []
    print(f"🪲 Parsed triples count: {len(triples)}")
    if not triples:
        sys.exit("❌ GPT 未抽取到三元組")

    # 3. KG 向量檢索
    raw_lines = search_by_triples(
        triples,
        embed_fn=lambda tp: embed_triple(emb, tp),
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

    # 5. 輸出至檔案
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

    # 6. GPT 最終判斷
    judge_result = gpt.chat(
        judge_prompt, kg_out.read_text(encoding="utf-8-sig"))
    # 移除所有反引號、井號與星號
    judge_result = (judge_result
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
