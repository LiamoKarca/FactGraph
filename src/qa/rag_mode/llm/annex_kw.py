"""
功能：
- 從 data/interim/rag_mode/user-input/ 取得「唯一」的 .txt 檔內容（若多於一個檔案會報錯）
- 讀取 data/interim/rag_mode/keywords/ 內最新的 *_keywords.json
- 產生 <stem>_kw_annex.json，內容為：
  [
    {"user_question": "<txt 檔內容>"},
    <keywords.json 的原始內容或其第一層資料>
  ]
- 輸出固定寫入 data/interim/rag_mode/keywords/

相依：
- src/qa/rag_mode/config.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import (
    PATHS, ENCODING,
    read_text, read_json, write_json,
    find_single_txt_or_error, latest_keywords_json,
    ensure_annex_list, build_kw_annex_output_name
)


def _normalize_keywords_payload(payload: Any) -> Any:
    """
    關鍵詞檔可能是：
    - 合法 JSON（list/dict）
    - 或單純是一段可被解析的 JSON 文字
    - 或單純是字串（模型未嚴格輸出 JSON）
    這裡盡量轉為 JSON；若失敗則包在字串中回傳。
    """
    if isinstance(payload, (dict, list)):
        return payload

    if isinstance(payload, str):
        txt = payload.strip()
        # 嘗試抓到首尾 JSON 區塊
        js, je = txt.find("{"), txt.rfind("}")
        ls, le = txt.find("["), txt.rfind("]")
        try_spans = []
        if 0 <= js < je:
            try_spans.append((js, je + 1))
        if 0 <= ls < le:
            try_spans.append((ls, le + 1))

        for s, e in try_spans:
            try:
                return json.loads(txt[s:e])
            except Exception:
                pass
        # 最後仍解析不了就原樣回傳字串
        return txt

    return payload


def main() -> None:
    # 1) 嚴格只允許一個 user-input txt
    txt_path = find_single_txt_or_error(PATHS.USER_INPUT_DIR)
    txt_content = read_text(txt_path).strip()

    # 2) 找到最新的 *_keywords.json
    kw_path = latest_keywords_json()
    raw_kw = read_text(kw_path)  # 保留原文以便 fallback
    try:
        kw_obj = read_json(kw_path)
    except Exception:
        kw_obj = raw_kw  # 非 JSON：保留字串，後續 _normalize_keywords_payload 會處理

    kw_obj = _normalize_keywords_payload(kw_obj)

    # 3) annex head 與整併
    annex_head = {"user_question": txt_content}
    annex_obj = ensure_annex_list(kw_obj, annex_head)

    # 4) 產生輸出檔名（<stem>_kw_annex.json），並固定輸出到 KEYWORDS_DIR
    out_path = build_kw_annex_output_name(kw_path)
    if out_path.parent != PATHS.KEYWORDS_DIR:
        out_path = PATHS.KEYWORDS_DIR / out_path.name

    write_json(out_path, annex_obj, ensure_ascii=False, indent=2)
    print(f"[完成] {txt_path.name} + {kw_path.name} → {out_path}")


if __name__ == "__main__":
    main()
