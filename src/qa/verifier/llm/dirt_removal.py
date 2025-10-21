"""
知識雜訊剔除（Dirt Removal）

功能概述
--------
- 輸入：一份「合併文本」字串，包含兩個區塊：
    [原始新聞]
    <原文多段文字...>

    [比對知識]
    [1] ...
    [2] ...
    ...

- 流程：僅對「[比對知識]」內的條目呼叫 LLM 逐條標註
       （related / partially_related / irrelevant），
        保留 related 與 partially_related，並重新連號。
- 輸出（三種檔案，皆以 utf-8-sig 寫入）：
  1) {out_dir}/dirt_removal/debug/news_kg_{news_id}.json
     - LLM 回傳的原始 JSON（可能是 {"items":[...]} 或直接 list）。
  2) {out_dir}/dirt_removal/debug/news_kg_{news_id}_dirt_removal_kb_only.txt
     - 僅含過濾後的知識條目清單（相容舊腳本）。
  3) {out_dir}/dirt_removal/news_kg_{news_id}_dirt_removal.txt
     - 合併輸出：保留 [原始新聞] 區塊 + 以「已過濾且重排」覆蓋 [比對知識]。
     - 供下一步「查核結果判斷」作為輸入。

設計要點
--------
- OpenAI 的 `json_schema` 規範：root 必須是 "object"。
  因此本模組使用 {"type": "object", "properties": {"items": [...]}}。
  拿到回覆後，會統一轉為「陣列 JSON 字串」送往下游。
- 回傳值為 (combined_text, combined_path)，其中 `combined_text` 為「合併輸出」。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

# ==== 可調參數 ====
DEFAULT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# 最低保留等級：只保留 related 與 partially_related
RETAIN_TAGS = {"related", "partially_related"}


# =========================
# 對外主要流程
# =========================
def run_dirt_removal(
    combined_text: str,
    news_id: str,
    out_dir: str,
) -> Tuple[str, str]:
    """
    執行雜訊剔除與合併回寫。

    參數
    ----
    combined_text : str
        由上游整合出的文本，必含：
        - "[原始新聞]\\n<原文...>\\n\\n[比對知識]\\n<編號條目...>"
    news_id : str
        輸入新聞 ID（用於輸出檔名）。
    out_dir : str
        輸出根目錄。例如 "data/processed/verifier"。

    目錄規格
    --------
    - 除錯 JSON 與 only 比對知識：
        {out_dir}/dirt_removal/debug/
    - 合併輸出（供下一步判斷）：
        {out_dir}/dirt_removal/

    回傳
    ----
    combined_output : str
        合併輸出：保留 [原始新聞]，覆蓋 [比對知識]（過濾後且重排）。
    combined_path : str
        合併輸出的檔案路徑（.../dirt_removal/news_kg_{news_id}_dirt_removal.txt）。
    """
    main_dir = os.path.join(out_dir, "dirt_removal")
    debug_dir = os.path.join(main_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # 解析兩段
    news_block, kb_lines = parse_combined_text_to_sections(combined_text)
    if not kb_lines:
        raise ValueError("解析失敗：未在 [比對知識] 區段找到任何以 [n] 開頭的條目。")

    # 呼叫 LLM 標註關聯性（回傳「陣列 JSON」字串）
    raw_json_text = call_llm_to_tag_relevance(kb_lines)

    # 除錯：存 LLM 原始 JSON（保持物件或陣列原樣）
    debug_json_path = os.path.join(debug_dir, f"news_kg_{news_id}.json")
    _write_debug_json(debug_json_path, raw_json_text)

    # 將 LLM 回傳轉為 Python list[dict]
    relevance_list = _ensure_list_from_text(raw_json_text)

    # 過濾條目並重新連號
    kept_lines = filter_kb_by_relevance(kb_lines, relevance_list)
    kb_filtered = _renumber_and_join(kept_lines)

    # === 寫檔：1) 合併輸出（主要成品；供下一步判斷） ===
    combined_output = build_combined_output(news_block, kb_filtered)
    combined_path = os.path.join(
        main_dir, f"news_kg_{news_id}_dirt_removal.txt"
    )
    _write_text(combined_path, combined_output)

    # === 寫檔：2) 純知識條目（相容舊腳本；放在 debug） ===
    kb_only_path = os.path.join(
        debug_dir, f"news_kg_{news_id}_dirt_removal_kb_only.txt"
    )
    _write_text(kb_only_path, kb_filtered)

    return combined_output, combined_path


# =========================
# 區塊解析與重組
# =========================
def parse_combined_text_to_sections(text: str) -> Tuple[str, List[str]]:
    """
    解析「[原始新聞]」與「[比對知識]」兩段；回傳 (news_block, kb_lines)。

    規則
    ----
    - 使用錨點標題：^\\[原始新聞\\] 與 ^\\[比對知識\\]（行首比對）。
    - news_block：回傳「不含標題行」的原文（原始換行保留）。
    - kb_lines：只擷取 [比對知識] 區段裡「以 [n] 開頭」的條目行。

    若未找到錨點，仍嘗試以舊規則抓 [n] 清單，但 news_block 會為空字串。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    news_idx = _find_anchor_line_index(text, r"^\[原始新聞\]\s*$")
    kb_idx = _find_anchor_line_index(text, r"^\[比對知識\]\s*$")

    news_block = ""
    kb_lines: List[str] = []

    if news_idx is not None and kb_idx is not None and news_idx < kb_idx:
        lines = text.split("\n")
        news_block = "\n".join(lines[news_idx + 1: kb_idx]).strip("\n")
        kb_section = "\n".join(lines[kb_idx + 1:])
        kb_lines = parse_kb_lines_only(kb_section)
        return news_block, kb_lines

    kb_lines = parse_kb_lines_only(text)
    return news_block, kb_lines


def build_combined_output(news_block: str, kb_filtered: str) -> str:
    """
    依專案格式重建合併輸出文本：
        [原始新聞]
        <news_block>

        [比對知識]
        <kb_filtered>
    """
    parts: List[str] = ["[原始新聞]"]
    parts.append(news_block.strip("\n"))
    parts.append("")
    parts.append("[比對知識]")
    parts.append(kb_filtered.strip("\n"))
    return "\n".join(parts).rstrip() + "\n"


# =========================
# LLM 交互
# =========================
def call_llm_to_tag_relevance(kb_lines: List[str]) -> str:
    """
    呼叫 OpenAI，請模型對每條知識判斷關聯性，回傳「陣列 JSON 字串」。

    策略
    ----
    1. 以 Responses API 呼叫，指定 json_schema（root=object, items=[...]）。
    2. 若 SDK/參數不相容則回退 Chat Completions，同樣帶入 json_schema。
    3. 不論回傳是 {"items":[...]} 或直接 list[...]
       → 最終皆轉為「陣列 JSON 字串」。
    """
    client = OpenAI(api_key=OPENAI_API_KEY or None)
    model = DEFAULT_MODEL

    schema = _json_schema()
    system_prompt, user_prompt = _build_prompts(kb_lines)

    # 1) 嘗試 Responses API
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "output_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            response_format={"type": "json_schema", "json_schema": schema},
            max_output_tokens=4096,
        )
        text = getattr(resp, "output_text", None) or ""
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "items" in obj:
                return json.dumps(obj["items"], ensure_ascii=False)
            if isinstance(obj, list):
                return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return text
        return text
    except TypeError:
        pass

    # 2) 回退 Chat Completions
    cc = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0,
    )
    json_text = cc.choices[0].message.content or ""
    try:
        obj = json.loads(json_text)
        if isinstance(obj, dict) and "items" in obj:
            return json.dumps(obj["items"], ensure_ascii=False)
        if isinstance(obj, list):
            return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json_text
    return json_text


def _json_schema() -> Dict[str, Any]:
    """
    OpenAI 的 `json_schema`：root 必須為 type=object。
    實際清單放在 "items" 欄位，元素包含：
      - id: integer >= 1
      - relevance: "related" | "partially_related" | "irrelevant"
      - reason: 非空字串
    """
    return {
        "name": "relevance_list",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer", "minimum": 1},
                            "relevance": {
                                "type": "string",
                                "enum": ["related", "partially_related", "irrelevant"],
                            },
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": ["id", "relevance", "reason"],
                    },
                }
            },
            "required": ["items"],
        },
    }


def _build_prompts(kb_lines: List[str]) -> Tuple[str, str]:
    """
    建立 system 與 user 提示詞。
    - system：角色與決策標準、輸出格式要求。
    - user：提供具編號的知識條目清單，請模型依 schema 回傳。
    """
    kb_only = "\n".join(kb_lines)

    system_prompt = (
        "擔任新聞查核輔助標註員。請逐條判斷知識條目與輸入新聞主張是否相關，"
        "relevance 僅能是 related / partially_related / irrelevant。"
        "reason 必須具體且精煉。"
        "輸出需符合 schema，且根物件為 { items: [...] }。"
    )
    user_prompt = (
        "以下為待判斷的知識條目，每條開頭都有 [n] 編號：\n\n"
        f"{kb_only}\n\n"
        "請逐條判斷並依 schema 輸出 JSON 物件：{ items: [...] }。"
    )
    return system_prompt, user_prompt


# =========================
# 文本解析與過濾
# =========================
def parse_kb_lines_only(text: str) -> List[str]:
    """
    從任意文本中解析出以 [n] 開頭的知識條目行。
    """
    lines: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if re.match(r"^\[\d+\]\s*", s):
            lines.append(s)
    return lines


def filter_kb_by_relevance(
    kb_lines: List[str],
    relevance_list: List[Dict[str, Any]],
) -> List[str]:
    """
    依 LLM 回傳的 relevance_list 過濾知識條目，保留 related/partially_related。

    參數
    ----
    kb_lines : List[str]
        原始知識條目行。
    relevance_list : List[Dict[str, Any]]
        形如 [{"id":1,"relevance":"related","reason":"..."}, ...]。

    回傳
    ----
    kept_lines : List[str]
        保留下來的條目行（仍維持原 [n] 編號，後續會重新連號）。
    """
    tag_by_id: Dict[int, str] = {}
    for item in relevance_list:
        try:
            idx = int(item.get("id"))
            tag = str(item.get("relevance"))
        except Exception:
            continue
        tag_by_id[idx] = tag

    kept: List[str] = []
    for line in kb_lines:
        m = re.match(r"^\[(\d+)\]\s*", line)
        if not m:
            continue
        idx = int(m.group(1))
        tag = tag_by_id.get(idx, "irrelevant")
        if tag in RETAIN_TAGS:
            kept.append(line)
    return kept


def _renumber_and_join(lines: Iterable[str]) -> str:
    """
    將保留的條目重新以 [1]..[k] 連號，並以換行組合成文本。
    僅替換前綴編號，不動行內其他文字。
    """
    out: List[str] = []
    new_id = 1
    for line in lines:
        body = re.sub(r"^\[\d+\]\s*", "", line)
        out.append(f"[{new_id}] {body}")
        new_id += 1
    return "\n".join(out)


# =========================
# 小工具（I/O 與 JSON）
# =========================
def _find_anchor_line_index(text: str, pattern: str) -> Optional[int]:
    """
    在多行文本中尋找首個滿足 pattern 的行索引；找不到回傳 None。
    """
    for i, line in enumerate(text.split("\n")):
        if re.search(pattern, line):
            return i
    return None


def _ensure_list_from_text(text: str) -> List[Dict[str, Any]]:
    """
    將 JSON 字串轉為 list[dict]：
    - 若為 {"items":[...]} 結構則取 items。
    """
    try:
        obj = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"LLM 回傳不是合法 JSON：{exc}") from exc

    if isinstance(obj, dict) and "items" in obj:
        if isinstance(obj["items"], list):
            return obj["items"]
        raise ValueError("schema 錯誤：'items' 不是陣列。")
    if isinstance(obj, list):
        return obj
    raise ValueError("解析失敗：JSON 既不是陣列，也不是含 items 的物件。")


def _write_text(path: str, content: str) -> None:
    """以 utf-8-sig 寫入文字檔（LF 換行）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write(content)


def _write_json(path: str, obj: Any) -> None:
    """以 utf-8-sig 寫入 JSON（確保可被一般編輯器正常辨識）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_debug_json(path: str, raw_json_text: str) -> None:
    """
    將回傳 JSON（可能是字串形式的 list 或 {"items":[...] }）以原樣物件寫入。
    """
    try:
        obj = json.loads(raw_json_text)
    except Exception:
        obj = {"items": []}
    _write_json(path, obj)
