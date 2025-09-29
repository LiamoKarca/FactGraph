"""
Answerer Pipeline（主流程）

流程：
0. python -m src.qa.answerer.pipeline <id.txt>
1. 讀取問題檔案
2. 抽取三元組（gpt-4o-mini）
3. 知識檢索（改為 rag_search.retrieve_answers）
4. 判定（gpt-5 → 續寫 → fallback gpt-4o-mini）
5. 輸出 user_kg_*.txt / user_qa_judge_*.txt 以及 debug 檔案
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv

from .llm.gpt import GPTClient
from .llm.prompt_loader import load_prompt
from ..tools import data_utils as du
from ..tools import kg_nl as knl
from .core.paths import (
    OUT_DIR,
    USER_INPUT_DIR,
    EXTRACT_PROMPT_PATH,
    JUDGE_PROMPT_PATH,
)
from .core.utils import safe_json_loads, clean_json_block
from .kg.search import retrieve_answers

# 載入 .env
load_dotenv()

# ──────────────────────────────── 環境參數 ───────────────────────────────
EXTRACT_MODEL_ID = os.getenv("ANSWERER_EXTRACT_MODEL", "gpt-4o-mini")
JUDGE_MODEL_ID = os.getenv("ANSWERER_JUDGE_MODEL", "gpt-5")
JUDGE_FALLBACK_MODEL_ID = os.getenv("ANSWERER_JUDGE_FALLBACK_MODEL",
                                    "gpt-4o-mini")


# ──────────────────────────────── 輔助函式 ───────────────────────────────
def _adapt_extracted_to_v1(
    data: Any, tail_placeholder: str = "未知"
) -> Tuple[list[Dict[str, str]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    """
    把抽取結果（新版 JSON）轉為舊版三元組格式。
    """
    triples_v1: list[Dict[str, str]] = []
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

        triple_v1 = {
            "head": head_text,
            "relation": relation,
            "tail": tail_text,
        }
        triples_v1.append(triple_v1)

        meta[(head_text, relation, tail_text)] = {
            "subject": {"text": head_text, "type": subj_type,
                        "attributes": subj_attr},
            "target": {"type": target_type, "attributes": target_attr},
        }

    return triples_v1, meta


# ──────────────────────────────── 主流程 ───────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Answerer pipeline: 指定問題檔案 <id>.txt"
    )
    parser.add_argument(
        "input_file", help="Path or filename of question file, e.g. '2024-...txt'"
    )
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

    # 載入 prompt
    extract_prompt = load_prompt(EXTRACT_PROMPT_PATH)
    judge_prompt = load_prompt(JUDGE_PROMPT_PATH)

    if not judge_prompt or not judge_prompt.strip():
        print("[WARN] 判定用 system prompt (character.txt) 為空或讀取失敗。")

    # 建立輸出資料夾
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR = OUT_DIR / "debug"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # A) 抽取三元組：gpt-4o-mini
    gpt_extract = GPTClient(
        api_key=os.getenv("GPT_API"),
        model_id=EXTRACT_MODEL_ID,
        temperature=0.2,
        top_p=0.9,
    )

    # B) 判定：gpt-5（fallback → gpt-4o-mini）
    gpt_judge = GPTClient(
        api_key=os.getenv("GPT_API"),
        model_id=JUDGE_MODEL_ID,
    )

    # 2. 抽取
    raw_resp = gpt_extract.chat(extract_prompt, question)
    print("🪵 GPT raw response:\n", raw_resp)

    # 擷取 JSON 區塊
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

    # 3. 知識檢索（改用 rag_search）
    kg_text = retrieve_answers(triples_json=data, question=question)

    # 5. 成果輸出（utf-8-sig）
    kg_out = OUT_DIR / f"user_kg_{slug}.txt"
    judge_out = OUT_DIR / f"user_qa_judge_{slug}.txt"
    debug_in = DEBUG_DIR / f"debug_judge_input_{slug}.txt"
    debug_out = DEBUG_DIR / f"debug_judge_output_{slug}.txt"

    kg_out.write_text(kg_text, encoding="utf-8-sig")

    # 6. 判定（gpt-5 → 續寫 → fallback）
    def _run_and_log(model_client: GPTClient, tag: str, user_text: str) -> tuple[str, dict]:
        txt, finish_reason, usage, model_id = model_client.chat_with_meta(
            judge_prompt, user_text
        )
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
        return txt, {"finish_reason": finish_reason,
                     "usage": usage, "model_id": model_id}

    def _reasoning_starved(meta: dict, text: str) -> bool:
        usage = meta.get("usage") or {}
        comp = usage.get("completion_tokens")
        return (comp is not None and comp > 0) and (not text.strip())

    # 首次判定
    judge_result, meta_info = _run_and_log(gpt_judge, JUDGE_MODEL_ID, kg_text)

    # 自動續寫（最多 2 次）
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
            gpt_judge, f"{JUDGE_MODEL_ID}-cont{continue_round}", cont_input
        )
        judge_result = (judge_result + "\n" + cont_text).strip()

    # fallback
    if not judge_result.strip():
        print(f"[WARN] 判定輸出為空，啟用 fallback → {JUDGE_FALLBACK_MODEL_ID}")
        gpt_judge_fb = GPTClient(
            api_key=os.getenv("GPT_API"),
            model_id=JUDGE_FALLBACK_MODEL_ID,
            temperature=0.3,
            top_p=0.9,
        )
        judge_result, _ = _run_and_log(
            gpt_judge_fb, f"{JUDGE_FALLBACK_MODEL_ID}", kg_text
        )

    if not judge_result.strip():
        print("[ERROR] 判定仍為空，請檢查 debug 檔案內容。")
        judge_out.write_text(
            "[ERROR] 模型回傳空白，請檢查 debug 檔。", encoding="utf-8-sig"
        )
        sys.exit(1)

    judge_out.write_text(judge_result, encoding="utf-8-sig")

    print("✅ finished; outputs saved under", OUT_DIR)
    print("   KG    →", kg_out.name)
    print("   JUDGE →", judge_out.name)


if __name__ == "__main__":
    main()
