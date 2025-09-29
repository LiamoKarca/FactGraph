"""
新聞事實驗證主流程 Pipeline

執行方式：
  - 全量：python -m src.qa.verifier.pipeline
  - 單篇：python -m src.qa.verifier.pipeline <news_id.txt>

說明：
- 本版已對齊短問句檢索策略：放大候選、逐句匹配任一關鍵詞、白名單過濾，
  並在 dedup 中啟用放寬三元組防線與 GPT 過濾，以降低長新聞「0 命中」。
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from ..tools import data_utils as du
from ..tools import kg_nl as knl
from .core.config import LLM_ROUNDS
from .core.dedup import deduplicate
from .core.embeddings import embed_text, embed_triple
from .core.paths import RES_DIR, USER_INPUT_DIR, VEC_DIR
from .kg.search import cosine_search, kg_row_to_detail
from .llm.extract import extract_entities_relations
from .llm.judge import judge_news_kb

load_dotenv()

# 統一輸出編碼（依專案規範）
OUTPUT_ENCODING = "utf-8-sig"


# ────────────────────────── 抽取相容轉換 ──────────────────────────
def _er_to_triples(data: Dict[str, Any]) -> List[du.Triple]:
    """
    將新的抽取輸出（entities + relations）轉為舊介面的三元組：
      [{'head': <name>, 'relation': <str>, 'tail': <name>}]

    注意：
    - relation.source/target 為 entity id，需映射至 entities 的 name
    - 若 target 為數值且 name 空，嘗試以 attributes.value + unit 組合；
      仍不可得則以「未知」回退（與舊版行為相容）
    """
    triples: List[du.Triple] = []
    if not isinstance(data, dict):
        return triples

    ents = {
        e.get("id"): e
        for e in data.get("entities", [])
        if isinstance(e, dict) and e.get("id")
    }

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
            # 數值類：value + unit；否則回「未知」
            t_attrs = tgt_ent.get("attributes") or {}
            val = t_attrs.get("value")
            unit = t_attrs.get("unit") or ""
            if val not in (None, ""):
                tail = f"{val}{unit}".strip()
            else:
                tail = "未知"

        triples.append({"head": head, "relation": rel_name, "tail": tail})

    return triples


def _pull_triples(text: str) -> List[du.Triple]:
    """
    多輪 LLM 抽取並合併。
    相容兩種格式：
      (A) 舊：{"triples":[{"subject":"A","relation":"R","object":"B"}, ...]}
      (B) 新：{"entities":[...], "relations":[...]}
    """
    all_rounds: List[List[du.Triple]] = []
    last_error: Exception | None = None

    for i in range(LLM_ROUNDS):
        print(f"🔸 GPT 抽取 round {i + 1}")
        start = time.time()
        raw = extract_entities_relations(text)  # 僅回傳 JSON 字串
        elapsed = time.time() - start
        print(f"  ↳ 完成，用時 {elapsed:.1f}s")

        if not raw:
            print(f"[WARN] 抽取回傳為空，跳過 round {i + 1}")
            continue

        try:
            # 清除可能殘留的 Markdown 反引號
            payload = json.loads(raw.replace("`", ""))
            if isinstance(payload, dict) and (
                "entities" in payload or "relations" in payload
            ):
                triples = _er_to_triples(payload)
            else:
                # 舊格式回退
                triples = du.json_to_triples(payload) or []
            if triples:
                all_rounds.append(triples)
        except Exception as e:
            last_error = e
            print(f"[WARN] JSON 解析失敗於 round {i + 1}: {e}")

    if not all_rounds:
        if last_error:
            print(f"[ERROR] 所有輪次皆失敗: {last_error}", file=sys.stderr)
        return []

    return du.merge_triples(*all_rounds)


# ────────────────────────── 主處理 ──────────────────────────
def _process_single(news_id: str, text: str) -> None:
    """處理單一新聞文本：嵌入 → 抽取 → 檢索 → 去重 → 判定 → 輸出"""
    RES_DIR.mkdir(parents=True, exist_ok=True)
    VEC_DIR.mkdir(parents=True, exist_ok=True)

    # 清除可能殘留的 Markdown 反引號，避免影響抽取
    text = text.replace("`", "")

    # 文章向量（可供後續分析或快取）
    vec_path = VEC_DIR / f"{news_id}.npy"
    np.save(vec_path, embed_text(text))

    # 1) 三元組抽取
    triples = _pull_triples(text)
    if not triples:
        sys.exit("❌ LLM 未抽取到任何三元組，流程終止")

    # 2) KG 檢索（強化：候選放大＋逐句 must-term）
    raw_lines: List[str] = []
    for tp in tqdm(triples, desc="🔍 KG 比對"):
        q_vec = embed_triple(tp)
        for idx in cosine_search(tp, q_vec):
            tri, det = kg_row_to_detail(idx)
            block = knl.build_block([tri], {tuple(tri.values()): det})
            raw_lines.extend(block.splitlines())

    if not raw_lines:
        sys.exit("⚠️ 無 KG 命中")

    # 3) 證據過濾與去重（放寬防線 + GPT YES/NO，並含保底回退）
    kept = deduplicate(raw_lines, triples=triples)

    # 4) 整理輸出區塊
    final = [re.sub(r"^\d+\.", f"[{i}]", ln, count=1)
             for i, ln in enumerate(kept, 1)]
    news_block = "[原始新聞]\n" + text
    kb_block = "[比對知識]\n" + "\n".join(final)

    # 5) 寫入檔案
    kg_file = RES_DIR / f"news_kg_{news_id}.txt"
    judge_file = RES_DIR / f"judge_result_{news_id}.txt"
    kg_file.write_text(f"{news_block}\n\n{kb_block}", encoding=OUTPUT_ENCODING)

    # 6) 判定（gpt-5 等），並輸出
    judged_raw = judge_news_kb(f"{news_block}\n\n{kb_block}")
    judged_clean = judged_raw.replace("`", "")
    judge_file.write_text(judged_clean, encoding=OUTPUT_ENCODING)

    print(f"✅ 輸出：{kg_file.name}, {judge_file.name}")


# ────────────────────────── 入口 ──────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("FactGraph Verifier Pipeline")
    p.add_argument(
        "news_id",
        nargs="?",
        help="新聞檔名（含或不含 .txt），留空則批次所有",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.news_id:
        # 允許傳入「不含 .txt」或「含 .txt」
        name = args.news_id if args.news_id.endswith(
            ".txt") else f"{args.news_id}.txt"
        input_path = USER_INPUT_DIR / name
        if not input_path.is_file():
            sys.exit(f"❌ 找不到檔案：{input_path}")
        text = input_path.read_text(encoding="utf-8-sig").strip()
        _process_single(Path(name).stem, text)
    else:
        processed = {p.stem.removeprefix("news_kg_")
                     for p in RES_DIR.glob("news_kg_*")}
        files = sorted(USER_INPUT_DIR.glob("*.txt"))
        for path in files:
            nid = path.stem
            if nid in processed:
                continue
            text = path.read_text(encoding="utf-8-sig").strip()
            _process_single(nid, text)

    gc.collect()


if __name__ == "__main__":
    print(
        f"⚙️ USER_INPUT_DIR: {USER_INPUT_DIR.resolve()} "
        f"(exists: {USER_INPUT_DIR.is_dir()})"
    )
    print(
        f"⚙️ RES_DIR:        {RES_DIR.resolve()} (exists: {RES_DIR.is_dir()})")
    files = list(USER_INPUT_DIR.glob("*.txt"))
    print(f"🔍 找到 {len(files)} 個輸入檔案：{[p.name for p in files]}")
    main()
