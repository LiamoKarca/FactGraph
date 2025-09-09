"""
新聞事實驗證主流程 Pipeline

執行方式：
  - 全量：python -m src.qa.verifier.pipeline
  - 單篇：python -m src.qa.verifier.pipeline <news_id.txt>
"""

from __future__ import annotations
from ..tools import kg_nl as knl
from ..tools import data_utils as du
from .llm.judge import judge_news_kb
from .llm.extract import extract_entities_relations
from .kg.search import cosine_search, kg_row_to_detail
from .core.paths import USER_INPUT_DIR, VEC_DIR, RES_DIR
from .core.embeddings import embed_text, embed_triple
from .core.dedup import deduplicate
from .core.config import LLM_ROUNDS

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


# ────────────────────────── 抽取相容轉換 ──────────────────────────
def _er_to_triples(data: Dict[str, Any]) -> List[du.Triple]:
    """
    將新的抽取輸出（entities + relations）轉為舊介面的三元組：
      [{'head': <name>, 'relation': <str>, 'tail': <name>}]

    - relation.source/target 是實體 id，需映射至 entities 中的 name
    - 若目標為數值實體且 name 空白，嘗試以 attributes.value + unit 組合字串
    """
    triples: List[du.Triple] = []
    if not isinstance(data, dict):
        return triples

    ents = {e.get("id"): e for e in data.get("entities", [])
            if isinstance(e, dict) and e.get("id")}
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
            # 若為數值型且未提供 name，以 value+unit 組合；否則以「未知」回退
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
        print(f'🔸 GPT 抽取 round {i + 1}')
        start = time.time()
        raw = extract_entities_relations(text)  # 僅回傳 JSON 字串（依抽取 prompt）
        elapsed = time.time() - start
        print(f'  ↳ 完成，用時 {elapsed:.1f}s')

        if not raw:
            print(f'[WARN] 抽取回傳為空，跳過 round {i + 1}')
            continue

        try:
            payload = json.loads(raw.replace("`", ""))  # 移除 API 回傳中的所有反引號
            if isinstance(payload, dict) and ("entities" in payload or "relations" in payload):
                triples = _er_to_triples(payload)
            else:
                # 舊格式回退
                triples = du.json_to_triples(payload) or []
            if triples:
                all_rounds.append(triples)
        except Exception as e:
            last_error = e
            print(f'[WARN] JSON 解析失敗於 round {i + 1}: {e}')

    if not all_rounds:
        if last_error:
            print(f'[ERROR] 所有輪次皆失敗: {last_error}', file=sys.stderr)
        return []

    return du.merge_triples(*all_rounds)


def _process_single(news_id: str, text: str) -> None:
    """
    處理單篇新聞：
      1. 嵌入全文
      2. LLM 抽取（相容新/舊格式）→ 三元組
      3. 向量檢索 KG
      4. 去重與編號
      5. 輸出結果與事實判斷
    """
    RES_DIR.mkdir(parents=True, exist_ok=True)
    VEC_DIR.mkdir(parents=True, exist_ok=True)

    # 依現有慣例，先移除輸入新聞中的所有反引號
    text = text.replace("`", "")

    # 全文嵌入
    vec_path = VEC_DIR / f'{news_id}.npy'
    np.save(vec_path, embed_text(text))

    # 三元組抽取
    triples = _pull_triples(text)
    if not triples:
        sys.exit('❌ LLM 未抽取到任何三元組，流程終止')

    # KG 比對
    raw_lines: List[str] = []
    for tp in tqdm(triples, desc='🔍 KG 比對'):
        q_vec = embed_triple(tp)
        for idx in cosine_search(tp, q_vec):
            tri, det = kg_row_to_detail(idx)
            block = knl.build_block([tri], {tuple(tri.values()): det})
            raw_lines.extend(block.splitlines())

    if not raw_lines:
        sys.exit('⚠️ 無 KG 命中')

    # 去重與重編號（保持既有輸出模式）
    kept = deduplicate(raw_lines)
    final = [re.sub(r'^\d+\.', f'[{i}]', ln, count=1)
             for i, ln in enumerate(kept, 1)]

    # 組合輸出（不加任何反引號圍欄）
    news_block = "[原始新聞]\n" + text
    kb_block = "[比對知識]\n" + "\n".join(final)

    # 寫入結果檔案
    kg_file = RES_DIR / f'news_kg_{news_id}'
    judge_file = RES_DIR / f'judge_result_{news_id}'
    kg_file.write_text(f"{news_block}\n\n{kb_block}", encoding='utf-8')

    # 事實判斷
    judged_raw = judge_news_kb(f"{news_block}\n\n{kb_block}")
    judged_clean = judged_raw.replace("`", "")
    judge_file.write_text(judged_clean, encoding='utf-8')

    print(f'✅ 輸出：{kg_file.name}, {judge_file.name}')


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('FactGraph Verifier Pipeline')
    p.add_argument(
        'news_id',
        nargs='?',
        help='新聞檔名（不含 .txt），留空則批次所有'
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.news_id:
        input_path = USER_INPUT_DIR / f'{args.news_id}'
        if not input_path.is_file():
            sys.exit(f'❌ 找不到檔案：{input_path}')
        text = input_path.read_text(encoding='utf-8-sig').strip()
        _process_single(args.news_id, text)
    else:
        processed = {
            p.stem.removeprefix('news_kg_')
            for p in RES_DIR.glob('news_kg_*.txt')
        }
        for path in sorted(USER_INPUT_DIR.glob('*.txt')):
            nid = path.stem
            if nid in processed:
                continue
            text = path.read_text(encoding='utf-8-sig').strip()
            _process_single(nid, text)

    gc.collect()


if __name__ == '__main__':
    print(
        f'⚙️ USER_INPUT_DIR: {USER_INPUT_DIR.resolve()} (exists: {USER_INPUT_DIR.is_dir()})')
    print(
        f'⚙️ RES_DIR:        {RES_DIR.resolve()} (exists: {RES_DIR.is_dir()})')
    files = list(USER_INPUT_DIR.glob("*.txt"))
    print(f'🔍 找到 {len(files)} 個輸入檔案：{[p.name for p in files]}')

    main()
