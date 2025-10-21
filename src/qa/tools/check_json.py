"""
JSON 結構驗證工具

用途
----
- 驗證 LLM 回傳的 relevance_list 是否符合預期結構：
  [{"id": int>=1, "relevance": "...", "reason": "非空字串"}, ...]

- 若上游採用 OpenAI json_schema 的 root=object（{"items":[...]})，可先在呼叫端
  轉成陣列再丟進本驗證器；或使用 `ensure_list()` 幫忙抽出 items。

說明
----
- 僅進行輕量驗證，避免過度擋住可自動修復的情況。
- 如需更嚴格規則，可擴充本模組。
"""

from __future__ import annotations

from typing import Any, Dict, List


ALLOWED_TAGS = {"related", "partially_related", "irrelevant"}


def ensure_list(obj: Any) -> List[Dict[str, Any]]:
    """
    將輸入轉為 list[dict]。若是 {"items":[...]} 結構會取出 items。

    這個函式供其他模組使用；本專案主要呼叫端已統一轉成 list，
    但保留此函式以利重複使用或單元測試。
    """
    if isinstance(obj, dict) and "items" in obj:
        obj = obj["items"]
    if isinstance(obj, list):
        return obj
    raise ValueError("期望為陣列或含 'items' 的物件。")


def validate_relevance_list(items: List[Dict[str, Any]]) -> None:
    """
    輕量驗證 relevance_list 的每一筆元素。

    規則
    ----
    - 必須包含 id (int>=1), relevance (限定枚舉), reason (非空字串)。
    - 忽略額外欄位（上游已在 json_schema 設定 additionalProperties=False）。
    """
    if not isinstance(items, list):
        raise TypeError("relevance_list 應為 list。")

    for i, obj in enumerate(items, start=1):
        if not isinstance(obj, dict):
            raise TypeError(f"第 {i} 筆不是物件。")

        if "id" not in obj or not isinstance(obj["id"], int) or obj["id"] < 1:
            raise ValueError(f"第 {i} 筆 id 不合法：{obj.get('id')}")

        tag = obj.get("relevance")
        if tag not in ALLOWED_TAGS:
            raise ValueError(f"第 {i} 筆 relevance 非法：{tag}")

        reason = obj.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"第 {i} 筆 reason 為空或型別錯誤。")
