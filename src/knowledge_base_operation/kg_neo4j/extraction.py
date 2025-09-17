"""
Extraction Module

本模組透過 OpenAI 官方 SDK 抽取新聞文本中的實體與關係，
包含：
  - 預設 prompt 檔案讀取
  - 呼叫 GPT API
  - 清理並擷取回應中的 JSON 區塊
  - 批次處理 CSV 新聞文件

設定方式：
  - 環境變數 GPT_API 儲存 API Key
  - 環境變數 GPT_MODEL 儲存模型名稱，預設 gpt-4o
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, Optional

import openai

from ...common.gadget import LOGGER

# 讀取環境變數
GPT_API_KEY: str | None = os.getenv('GPT_API')
GPT_MODEL: str = os.getenv('GPT_MODEL', 'gpt-4o')

if not GPT_API_KEY:
    LOGGER.critical('找不到 GPT_API 環境變數，請確認 .env 設定！')
    sys.exit(1)
openai.api_key = GPT_API_KEY

# 參數設定
DEFAULT_TEMPERATURE: float = 0.2
MAX_TOKENS: int = 4096
DEFAULT_PROMPT_FILE: str = 'src/knowledge_base_operation/kg_neo4j/prompts/extraction-prompt.txt'


def get_default_prompt(prompt_file: str = DEFAULT_PROMPT_FILE) -> str:
    """讀取並回傳預設 prompt 內容。

    Args:
        prompt_file: prompt 檔案路徑。

    Returns:
        str: prompt 文字。

    若檔案不存在則記錄錯誤並終止程式。
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8-sig') as f:
            return f.read()
    except Exception as exc:
        LOGGER.critical('無法讀取 prompt 檔案 %s: %s', prompt_file, exc)
        sys.exit(1)


def call_gpt_api(text: str) -> Optional[str]:
    """呼叫 GPT API 並回傳原始回應文字。

    Args:
        text: 使用者輸入文本。

    Returns:
        str | None: GPT 回傳內容，失敗時回傳 None。
    """
    system_prompt = get_default_prompt()
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': text}
    ]

    try:
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={'type': 'json_object'}
        )
    except openai.OpenAIError as exc:
        LOGGER.error('API 呼叫失敗: %s', exc)
        return None

    try:
        return response.choices[0].message.content  # type: ignore
    except (AttributeError, IndexError) as exc:
        LOGGER.error('解析 API 回應失敗: %s', exc)
        return None


def clean_json_string(json_string: str, *, debug: bool = False) -> str:
    """預留 JSON 清理邏輯，目前為 no-op。

    Args:
        json_string: 原始 JSON 字串。
        debug: 是否輸出除錯訊息。

    Returns:
        str: 清理後的 JSON 字串。
    """
    if debug:
        print('[clean_json_string] no-op mode — returning original input')
    return json_string


def extract_first_json_object(text: str) -> Optional[str]:
    """尋找並回傳文字中第一個平衡的大括號 JSON 物件字串。"""
    start = text.find('{')
    if start == -1:
        return None

    brace_depth = 0
    for idx, ch in enumerate(text[start:], start=start):
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0:
                return text[start:idx + 1]

    return None


def extract_json_block(response_text: str) -> Optional[Dict[str, Any]]:
    """從 GPT 回應文字擷取 JSON 區塊並解析為字典。

    優先偵測 ```json ... ``` 標記，若未命中則以第一個平衡大括號回退。
    """
    # 嘗試匹配 ```json``` 區段
    pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
    json_block = match.group(
        1) if match else extract_first_json_object(response_text)

    if not json_block:
        LOGGER.warning('未找到 JSON 區塊，回傳原始內容：%s', response_text)
        return None

    cleaned = clean_json_string(json_block)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        LOGGER.warning('JSON 解析失敗: %s', exc)
        LOGGER.debug('原始 JSON：%s', cleaned)
        return None


def extract_entities_relations(text: str) -> Optional[Dict[str, Any]]:
    """抽取文本中的實體與關係三元組。

    Args:
        text: 新聞文本內容。

    Returns:
        dict | None: 標準化後 JSON 結構，包含 entities 和 relations。
    """
    raw = call_gpt_api(text)
    if not raw:
        LOGGER.error('GPT 回應為空或錯誤。')
        return None

    parsed = extract_json_block(raw)
    if not parsed:
        LOGGER.error('未能從 GPT 回應中解析 JSON 結果。')
    return parsed
