"""
語意去重 + 三元組防線（放寬版）+ GPT 輔助過濾（預設關閉）

層級：
A. 雜訊剃除：過短片段不保留
B. 三元組防線（放寬版）：
   - 命中 head 或 tail 任一（排除 '未知'）
   - 句首主語（由 ENTITY_RE 擷取）若屬任一 head
   - 命中 all_terms 任一（含 relation 與 GPT 擴充詞）
C. 語義去重：以第一個實體分組，餘弦相似度 > DUP_TH 視為重複
D. GPT 輔助：針對每行 evidence，讓模型判定是否支撐三元組（YES/NO）
E. 保底回退：若 B/C/D 導致全被剃除，回退「關閉三元組防線 + 關閉 GPT 濾器」
   （只做語義去重），保證不至於 0 命中

注意：
- 本檔預設關閉 GPT 過濾（USE_GPT_FILTER=False），避免實務上過嚴。
- 若要開啟，可將 USE_GPT_FILTER 設為 True（或用環境變數啟動）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os
import re

import numpy as np
from sentence_transformers import util

from .config import DUP_TH, ENTITY_RE
from .embeddings import embed_text

# ──────────────── 參數與 GPT 客戶端 ────────────────

# 預設開啟
USE_GPT_FILTER = os.getenv("DEDUP_USE_GPT", "0") == "1"
GPT_MODEL = os.getenv("DEDUP_GPT_MODEL", "gpt-4o-mini")

_gpt_client = None
if USE_GPT_FILTER:
    try:
        from src.qa.answerer.llm.gpt import GPTClient

        _gpt_client = GPTClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_id=GPT_MODEL,
        )
    except Exception:
        # 金鑰缺失或導入失敗 → 自動關閉
        USE_GPT_FILTER = False
        _gpt_client = None


# ──────────────── 小工具 ────────────────
def _normalize(s: str) -> str:
    """去空白、大小寫無關比較用。"""
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)
    return s.casefold()


def _build_terms(triples: List[Dict[str, Any]] | None) -> Tuple[List[str], List[str]]:
    """
    由三元組列表建構兩個關鍵詞桶：
    - head_tail_terms：僅收集 head 與 tail（排除 '未知'）
    - all_terms：收集 head、relation、tail（含 '未知' 以外者）
    回傳：(head_tail_terms, all_terms)
    """
    ht: List[str] = []
    all_terms: List[str] = []
    if not triples:
        return ht, all_terms

    seen_ht: set[str] = set()
    seen_all: set[str] = set()
    for tp in triples:
        h = str(tp.get("head") or "").strip()
        r = str(tp.get("relation") or "").strip()
        t = str(tp.get("tail") or "").strip()
        for term, bucket, seen in (
            (h, ht, seen_ht),                     # head → head_tail_terms
            (t if t != "未知" else "", ht, seen_ht),  # tail（排除未知）
            (h, all_terms, seen_all),            # head → all_terms
            (r, all_terms, seen_all),            # relation → all_terms
            (t, all_terms, seen_all),            # tail → all_terms（含未知）
        ):
            if term and term not in seen:
                bucket.append(term)
                seen.add(term)
    return ht, all_terms


def _passes_tuple_guard(
    line: str,
    head_tail_terms: List[str],
    all_terms: List[str],
) -> bool:
    """
    放寬版三元組防線：
      1) 行文字包含任一 head 或 tail 名稱（排除 '未知'）→ 通過
      2) 句首主語（ENTITY_RE 擷取）匹配任一 head → 通過
      3) 行文字包含任一 all_terms（含 relation 或擴充詞）→ 通過
    三者皆未命中才會返回 False（即被剃除）。
    """
    if not (head_tail_terms or all_terms):
        return True

    line_n = _normalize(line)

    # 1) 命中 head/tail 任一
    for term in head_tail_terms:
        if _normalize(term) and _normalize(term) in line_n:
            return True

    # 2) 句首主語命中 head
    m = ENTITY_RE.match(line)
    if m:
        subj = _normalize(m.group(1))
        for term in head_tail_terms:
            if subj and subj == _normalize(term):
                return True

    # 3) 命中任一 all_terms
    for term in all_terms:
        if _normalize(term) and _normalize(term) in line_n:
            return True

    return False


def _first_entity(line: str) -> str:
    """使用 ENTITY_RE 擷取第一個實體；失敗時退回行首詞。"""
    m = ENTITY_RE.match(line)
    if m:
        return m.group(1)
    return line.split(" ")[0]


# ──────────────── 主流程 ────────────────
def deduplicate(
    lines: List[str],
    triples: List[Dict[str, Any]] | None = None,
) -> List[str]:
    """
    綜合過濾流程（A ~ E，見檔頭說明）。

    參數：
    - lines：原始證據行列表（通常來自 build_block 拆行）
    - triples：抽取到的三元組（啟用三元組防線與 GPT 輔助時所需）
    """
    head_tail_terms, all_terms = _build_terms(triples)

    def _run(with_tuple_guard: bool, with_gpt: bool) -> List[str]:
        """執行一次完整過濾；可選擇是否啟用三元組防線與 GPT 濾器。"""
        groups: Dict[str, List[np.ndarray]] = {}
        kept: List[str] = []

        for line in lines:
            text = (line or "").strip()

            # A. 雜訊剃除：過短片段不要
            pure = re.sub(r"\s+", "", text)
            if len(pure) < 6:
                continue

            # B. 三元組防線（可關閉）
            if with_tuple_guard and not _passes_tuple_guard(
                text, head_tail_terms, all_terms
            ):
                continue

            # C. 語義去重（以第一個實體分組；相似度 >= DUP_TH 視為重複）
            ent = _first_entity(text)
            vec = embed_text(text)
            vecs = groups.get(ent, [])
            if vecs and util.cos_sim(vec, np.vstack(vecs))[0, :].max() >= DUP_TH:
                continue
            groups.setdefault(ent, []).append(vec)

            # D. GPT 輔助（YES/NO；失敗不阻塞；可關閉）
            if with_gpt and USE_GPT_FILTER and _gpt_client is not None and triples:
                prompt = (
                    "請判斷此句是否支撐下述三元組（任一即可）。"
                    "只回 YES 或 NO。\n"
                    f"句子: {text}\n"
                    f"三元組關鍵詞(head/tail優先): {head_tail_terms}\n"
                    f"其它關鍵詞(含relation): {all_terms}\n"
                )
                try:
                    res = str(
                        _gpt_client.chat(
                            system_prompt="你是嚴格的資訊過濾器",
                            user_prompt=prompt,
                        )
                    ).strip()
                    if not res.upper().startswith("YES"):
                        continue
                except Exception:
                    # GPT 調用失敗則略過此步，不影響整體流程
                    pass

            kept.append(text)

        return kept

    # 先跑「含三元組防線」+（依使用者設定）是否含 GPT 濾器
    kept = _run(with_tuple_guard=True, with_gpt=USE_GPT_FILTER)
    if kept:
        return kept

    # E. 保底回退：關閉三元組防線 + 關閉 GPT 濾器，只做語義去重
    return _run(with_tuple_guard=False, with_gpt=False)
