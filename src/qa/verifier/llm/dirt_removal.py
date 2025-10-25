"""
知識雜訊剔除（Dirt Removal）模組

概述:
    - 以 OpenAI Responses API（優先）呼叫 GPT-5 / GPT-4o 系列，嚴格套用 JSON Schema。
    - 回退路徑保留 Chat Completions，相容舊有部署。
    - 維持原先三種輸出檔，編碼與格式不變。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


# =========================
# 可調參數與環境變數
# =========================
# 模型選擇: 優先讀取 OPENAI_RESP_MODEL，其次回退 OPENAI_CHAT_MODEL，最後預設 gpt-5-mini。
DEFAULT_MODEL = os.getenv(
    "OPENAI_RESP_MODEL",
    os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"),
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# GPT-5 進階可選參數（Responses API 專屬；非 GPT-5 時自動忽略）
# 參考: reasoning_effort / verbosity（Azure/OpenAI GPT-5 說明）
GPT5_REASONING_EFFORT = os.getenv("GPT5_REASONING_EFFORT")  # minimal/low/medium/high
GPT5_VERBOSITY = os.getenv("GPT5_VERBOSITY")  # low/medium/high

# 最低保留等級：只保留 related 與 partially_related
RETAIN_TAGS = {"related", "partially_related"}

# 外部提示詞路徑
DIRT_FILTER_PROMPT_PATH = os.getenv(
    "DIRT_FILTER_PROMPT_PATH",
    "src/qa/verifier/prompts/dirt_filter.md",
)


# =========================
# 對外主要流程
# =========================
def run_dirt_removal(
    combined_text: str,
    news_id: str,
    out_dir: str,
) -> Tuple[str, str]:
    """執行雜訊剔除與合併回寫。

    Args:
        combined_text: 由上游整合出的文本，必含：
            - "[原始新聞]\\n<原文...>\\n\\n[比對知識]\\n<編號條目...>"
        news_id: 輸入新聞 ID（用於輸出檔名）。
        out_dir: 輸出根目錄。例如 "data/processed/verifier"。

    Directory Layout:
        - 除錯 JSON 與 only 比對知識：{out_dir}/dirt_removal/debug/
        - 合併輸出（供下一步判斷）：{out_dir}/dirt_removal/

    Returns:
        Tuple[str, str]: (combined_output, combined_path)
            - combined_output: 合併輸出（保留 [原始新聞]，覆蓋 [比對知識]，已過濾且重排）。
            - combined_path: 合併輸出檔案完整路徑。
    """
    main_dir = os.path.join(out_dir, "dirt_removal")
    debug_dir = os.path.join(main_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # 解析兩段（支援 [原始新聞] 與 [原始文本]）
    news_block, kb_lines = parse_combined_text_to_sections(combined_text)
    if not kb_lines:
        raise ValueError("解析失敗：未在 [比對知識] 區段找到任何以 [n] 開頭的條目。")

    # 呼叫 LLM 標註關聯性（回傳「陣列 JSON」字串），同時送入原始新聞與外部 PROMPT
    raw_json_text = call_llm_to_tag_relevance(news_block, kb_lines)

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
    combined_path = os.path.join(main_dir, f"news_kg_{news_id}_dirt_removal.txt")
    _write_text(combined_path, combined_output)

    # === 寫檔：2) 純知識條目（相容舊腳本；放在 debug） ===
    kb_only_path = os.path.join(debug_dir, f"news_kg_{news_id}_dirt_removal_kb_only.txt")
    _write_text(kb_only_path, kb_filtered)

    return combined_output, combined_path


# =========================
# 區塊解析與重組
# =========================
def parse_combined_text_to_sections(text: str) -> Tuple[str, List[str]]:
    """解析「[原始新聞]」與「[比對知識]」兩段；回傳 (news_block, kb_lines)。

    規則:
        - 使用錨點標題：^\\[原始新聞\\] 與 ^\\[比對知識\\]（行首比對）。
        - news_block：回傳「不含標題行」的原文（原始換行保留）。
        - kb_lines：只擷取 [比對知識] 區段裡「以 [n] 開頭」的條目行。

    若未找到錨點，仍嘗試以舊規則抓 [n] 清單，但 news_block 會為空字串。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 支援兩種錨點名稱
    news_idx = _find_anchor_line_index(text, r"^\[(原始新聞|原始文本)\]\s*$")    
    kb_idx = _find_anchor_line_index(text, r"^\[比對知識\]\s*$")

    news_block = ""
    kb_lines: List[str] = []

    if news_idx is not None and kb_idx is not None and news_idx < kb_idx:
        lines = text.split("\n")
        news_block = "\n".join(lines[news_idx + 1 : kb_idx]).strip("\n")
        kb_section = "\n".join(lines[kb_idx + 1 :])
        kb_lines = parse_kb_lines_only(kb_section)
        return news_block, kb_lines

    kb_lines = parse_kb_lines_only(text)
    return news_block, kb_lines


def build_combined_output(news_block: str, kb_filtered: str) -> str:
    """依專案格式重建合併輸出文本。

    Returns:
        str: 形如
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
# LLM 交互（Responses 為主，Chat 回退）
# =========================
def call_llm_to_tag_relevance(news_block: str, kb_lines: List[str]) -> str:
    """呼叫 OpenAI 模型逐條判斷關聯性，回傳「陣列 JSON 字串」。

    策略:
        1) 以 Responses API 呼叫，指定 json_schema（root=object, items=[...]）。
        2) 若 SDK/參數不相容則回退 Chat Completions，同樣帶入 json_schema。
        3) 不論回傳是 {"items":[...]} 或直接 list[...] → 一律轉為陣列 JSON 字串。
    """
    client = OpenAI(api_key=OPENAI_API_KEY or None)
    model = DEFAULT_MODEL

    schema = _json_schema()
    system_prompt, user_prompt = _build_prompts(news_block, kb_lines)

    # ===== 1) Responses API（GPT-5 相容首選） =====
    try:
        resp_kwargs: Dict[str, Any] = {
            "model": model,
            # 外部 PROMPT + 內建規範 → instructions（Responses 慣例）
            "instructions": system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                }
            ],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "max_output_tokens": 4096,
        }

        # 若是 GPT-5 系列，可選擇性加入 reasoning/verbosity
        if str(model).startswith("gpt-5"):
            if GPT5_REASONING_EFFORT:
                resp_kwargs["reasoning"] = {"effort": GPT5_REASONING_EFFORT}
            if GPT5_VERBOSITY:
                resp_kwargs["verbosity"] = GPT5_VERBOSITY

        resp = client.responses.create(**resp_kwargs)

        # 官方 SDK 建議以 output_text 直接取最終字串
        # 參見 openai-python README
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
    except Exception:
        # 落入回退
        pass

    # ===== 2) Chat Completions 回退（保險機制） =====
    # GPT-5 mini/nano 與若干推理型模型不接受非預設 temperature，
    # 嚴格 JSON Schema。 
    cc = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
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
    """回傳 JSON Schema（root=object; items 為 list[object]）。

    欄位:
        - id: integer >= 1
        - relevance: "related" | "partially_related" | "irrelevant"
        - reason: 非空字串

    注意:
        - `strict: True` 以啟用嚴格模式，保證結構符合 Schema。
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


def _build_prompts(news_block: str, kb_lines: List[str]) -> Tuple[str, str]:
    """組合 instructions 與 user 提示詞，包含外部 md 與原始新聞全文。"""
    kb_only = "\n".join(kb_lines)
    # 讀取外部 md；若失敗則採用內建後備
    ext = _read_file_if_exists(DIRT_FILTER_PROMPT_PATH)
    if not ext:
        ext = (
            "【任務】對 [比對知識] 列表逐條判斷其與 [原始新聞] 的關聯性。\n"
            "標記為 related / partially_related / irrelevant，並給出精煉 reason。\n"
            "輸出需符合 JSON Schema（root=object, items=[...]）。"
        )
    # instructions：外部規則 + 嚴格輸出要求
    system_prompt = (
        f"{ext}\n\n"
        "【輸出要求】必須符合嚴格 JSON Schema，根物件為 { items: [...] }；"
        "不得輸出與 schema 無關的自然語言。"
    )
    # user：同時提供原始新聞與知識清單
    user_prompt = (
        "[原始新聞]\n"
        f"{news_block.strip()}\n\n"
        "[比對知識]\n"
        f"{kb_only}\n\n"
        "請逐條判斷並依 Schema 輸出 JSON 物件：{ items: [...] }。"
    )
    return system_prompt, user_prompt

def _read_file_if_exists(path: str) -> str:
    """嘗試以 utf-8-sig/utf-8 讀取外部檔案；不存在則回傳空字串。"""
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except FileNotFoundError:
            return ""
        except UnicodeDecodeError:
            continue
    return ""


# =========================
# 文本解析與過濾
# =========================
def parse_kb_lines_only(text: str) -> List[str]:
    """從任意文本中解析出以 [n] 開頭的知識條目行。"""
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
    """依 LLM 回傳的 relevance_list 過濾知識條目。

    Args:
        kb_lines: 原始知識條目行。
        relevance_list: 形如 [{"id":1,"relevance":"related","reason":"..."}, ...]。

    Returns:
        List[str]: 保留下來的條目行（仍維持原 [n] 編號，後續會重新連號）。
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
    """將保留的條目重新以 [1]..[k] 連號，並以換行組合成文本。"""
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
    """在多行文本中尋找首個滿足 pattern 的行索引；找不到回傳 None。"""
    for i, line in enumerate(text.split("\n")):
        if re.search(pattern, line):
            return i
    return None


def _ensure_list_from_text(text: str) -> List[Dict[str, Any]]:
    """將 JSON 字串轉為 list[dict]；若為 {"items":[...]} 結構則取 items。"""
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
    """以 utf-8-sig 寫入 JSON。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_debug_json(path: str, raw_json_text: str) -> None:
    """將回傳 JSON（可能是字串形式的 list 或 {"items":[...]}）以原樣物件寫入。"""
    try:
        obj = json.loads(raw_json_text)
    except Exception:
        obj = {"items": []}
    _write_json(path, obj)
