"""
新聞事實驗證主流程 Pipeline

執行方式：
  - 全量：python -m src.qa.verifier.pipeline
  - 單篇：python -m src.qa.verifier.pipeline <news_id.txt>

說明：
- 「抽取」之前有前置「綴字簡潔」，但現在**抽取一律從硬碟讀檔**：
    data/interim/verifier/conciseness/<news_id>.txt
  若檔不存在才現場生成一次；之後都讀檔，不保存在記憶體。
- 向量與輸出內容仍用「原始文本」，不變。
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import os
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
from .llm.extract import extract_entities_relations, set_debug_basename as set_extract_debug_basename
from .llm.judge import judge_news_kb, set_debug_basename as set_judge_debug_basename

# 僅在本檔使用（可選）gpt client；若不存在，不影響主流程
try:
    from .llm.gpt import GPTClient  # 你專案內的 wrapper（非串流）
except Exception:
    GPTClient = None  # type: ignore

load_dotenv()

# ───────────────────── 常數 ─────────────────────
OUTPUT_ENCODING = "utf-8-sig"
CONCISENESS_MODEL = "gpt-4o-mini"  # 寫死；僅用於生成簡潔檔
CONCISENESS_PROMPT_PATH = Path(__file__).resolve(
).parent / "prompts" / "conciseness.txt"
CONCISENESS_DIR = Path("data/interim/verifier/conciseness")
CONCISENESS_HARD_TIMEOUT_SEC = 30  # 生成簡潔檔時的最大等待秒數


# ───────────────────── 綴字簡潔（生成到檔案；失敗不致命） ─────────────────────
def _ensure_conciseness_file(news_id: str, original_text: str) -> Path:
    """
    確保 conciseness 檔案存在。
    - 若已存在且非空，直接回傳路徑。
    - 若不存在或為空：嘗試生成一次；失敗則寫入原文當作簡潔檔（避免抽取找不到檔）。
    """
    CONCISENESS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONCISENESS_DIR / f"{news_id}.txt"

    # 已有檔案且非空
    if out_path.is_file():
        try:
            if out_path.stat().st_size > 0:
                print(f"[INFO] 使用現有簡潔檔：{out_path}")
                return out_path
        except Exception:
            pass  # 讀不到 size 就當作無效，改為重寫

    concise = original_text.strip()

    # 讀提示模板
    try:
        prompt_tmpl = CONCISENESS_PROMPT_PATH.read_text(
            encoding=OUTPUT_ENCODING).strip()
    except FileNotFoundError:
        # 找不到模板 → 直接寫原文，保證檔案存在
        out_path.write_text(concise, encoding=OUTPUT_ENCODING)
        print("[WARN] 找不到 conciseness 提示模板，已以原文回寫簡潔檔。")
        return out_path

    # 無可用 GPT 客戶端或沒金鑰 → 直接寫原文
    if GPTClient is None or not os.getenv("OPENAI_API_KEY", "").strip():
        out_path.write_text(concise, encoding=OUTPUT_ENCODING)
        print("[WARN] 無 GPTClient 或無金鑰，已以原文回寫簡潔檔。")
        return out_path

    system_prompt = "你是精通中文新聞寫作的簡潔化助手，只做『綴字簡潔』，不可加入新資訊。"
    user_prompt = (
        f"{prompt_tmpl}\n\n---\n【原始文字】\n{original_text.strip()}\n---\n"
        "請直接輸出簡潔後文本，不要夾帶說明或標籤。"
    )

    # 單次呼叫 + 硬超時；失敗一律回寫原文確保有檔
    try:
        try:
            gpt = GPTClient(api_key=os.getenv("OPENAI_API_KEY", ""), model_id=CONCISENESS_MODEL,
                            temperature=0, timeout=CONCISENESS_HARD_TIMEOUT_SEC)
        except Exception:
            gpt = GPTClient(os.getenv("OPENAI_API_KEY", ""), CONCISENESS_MODEL,
                            temperature=0, timeout=CONCISENESS_HARD_TIMEOUT_SEC)

        t0 = time.time()
        concise = gpt.chat(system_prompt, user_prompt, max_retries=0).strip()
        if (time.time() - t0) > CONCISENESS_HARD_TIMEOUT_SEC or not concise:
            concise = original_text.strip()
    except Exception as e:
        print(f"[WARN] 生成簡潔檔失敗：{e} → 以原文回寫。")
        concise = original_text.strip()

    out_path.write_text(concise, encoding=OUTPUT_ENCODING)
    print(f"[INFO] 已寫入簡潔檔：{out_path}")
    return out_path


# ────────────────────────── 抽取相容轉換 ──────────────────────────
def _er_to_triples(data: Dict[str, Any]) -> List[du.Triple]:
    """將新的抽取輸出（entities + relations）轉為舊介面的三元組。"""
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
            t_attrs = tgt_ent.get("attributes") or {}
            val = t_attrs.get("value")
            unit = t_attrs.get("unit") or ""
            tail = f"{val}{unit}".strip() if val not in (None, "") else "未知"

        triples.append({"head": head, "relation": rel_name, "tail": tail})

    return triples


def _pull_triples(text_for_extraction: str) -> List[du.Triple]:
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
        raw = extract_entities_relations(text_for_extraction)  # 非串流，回 JSON 字串
        elapsed = time.time() - start
        print(f"  ↳ 完成，用時 {elapsed:.1f}s")

        if not raw:
            print(f"[WARN] 抽取回傳為空，跳過 round {i + 1}")
            continue

        try:
            payload = json.loads(raw.replace("`", ""))  # 清除可能殘留反引號
            if isinstance(payload, dict) and ("entities" in payload or "relations" in payload):
                triples = _er_to_triples(payload)
            else:
                triples = du.json_to_triples(payload) or []  # 舊格式回退
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
    RES_DIR.mkdir(parents=True, exist_ok=True)
    VEC_DIR.mkdir(parents=True, exist_ok=True)

    text = text.replace("`", "")
    vec_path = VEC_DIR / f"{news_id}.npy"
    np.save(vec_path, embed_text(text))

    # 一律從硬碟 conciseness 讀取
    concise_path = _ensure_conciseness_file(news_id, text)
    try:
        concise_text = concise_path.read_text(encoding=OUTPUT_ENCODING).strip()
    except Exception as e:
        print(f"[WARN] 讀取簡潔檔失敗：{e} → 改用原文。")
        concise_text = text

    # 設定 debug 檔名前綴（抽取與判斷各自一份）
    set_extract_debug_basename(news_id)
    set_judge_debug_basename(news_id)

    # 三元組抽取（用簡潔文本）
    triples = _pull_triples(concise_text)
    if not triples:
        sys.exit("❌ LLM 未抽取到任何三元組，流程終止")

    # 後續流程同原本...
    raw_lines: List[str] = []
    for tp in tqdm(triples, desc="🔍 KG 比對"):
        q_vec = embed_triple(tp)
        for idx in cosine_search(tp, q_vec):
            tri, det = kg_row_to_detail(idx)
            block = knl.build_block([tri], {tuple(tri.values()): det})
            raw_lines.extend(block.splitlines())

    if not raw_lines:
        sys.exit("⚠️ 無 KG 命中")

    kept = deduplicate(raw_lines, triples=triples)
    final = [re.sub(r"^\d+\.", f"[{i}]", ln, count=1)
             for i, ln in enumerate(kept, 1)]
    news_block = "[原始新聞]\n" + text
    kb_block = "[比對知識]\n" + "\n".join(final)

    kg_file = RES_DIR / f"news_kg_{news_id}.txt"
    judge_file = RES_DIR / f"judge_result_{news_id}.txt"
    kg_file.write_text(f"{news_block}\n\n{kb_block}", encoding=OUTPUT_ENCODING)

    # 🧑‍⚖️ 送交 GPT 判斷（同時會自動把請求/回應/錯誤落到 data/processed/verifier/debug/）
    judged_raw = judge_news_kb(
        f"{news_block}\n\n{kb_block}", debug_name=news_id)
    judged_clean = judged_raw.replace("`", "")
    judge_file.write_text(judged_clean, encoding=OUTPUT_ENCODING)

    print(f"✅ 輸出：{kg_file.name}, {judge_file.name}")


# ────────────────────────── 入口 ──────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("FactGraph Verifier Pipeline")
    p.add_argument("news_id", nargs="?", help="新聞檔名（含或不含 .txt），留空則批次所有")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.news_id:
        name = args.news_id if args.news_id.endswith(
            ".txt") else f"{args.news_id}.txt"
        input_path = USER_INPUT_DIR / name
        if not input_path.is_file():
            sys.exit(f"❌ 找不到檔案：{input_path}")
        text = input_path.read_text(encoding=OUTPUT_ENCODING).strip()
        _process_single(Path(name).stem, text)
    else:
        processed = {p.stem.removeprefix("news_kg_")
                     for p in RES_DIR.glob("news_kg_*")}
        files = sorted(USER_INPUT_DIR.glob("*.txt"))
        for path in files:
            nid = path.stem
            if nid in processed:
                continue
            text = path.read_text(encoding=OUTPUT_ENCODING).strip()
            _process_single(nid, text)

    gc.collect()


if __name__ == "__main__":
    print(
        f"⚙️ USER_INPUT_DIR: {USER_INPUT_DIR.resolve()} (exists: {USER_INPUT_DIR.is_dir()})")
    print(
        f"⚙️ RES_DIR:        {RES_DIR.resolve()} (exists: {RES_DIR.is_dir()})")
    files = list(USER_INPUT_DIR.glob("*.txt"))
    print(f"🔍 找到 {len(files)} 個輸入檔案：{[p.name for p in files]}")
    main()
