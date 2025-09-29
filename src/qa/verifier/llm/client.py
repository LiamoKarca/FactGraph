"""
OpenAI 初始化與共用 kwargs（最小化；分階段超時由呼叫端帶入）
"""
from typing import Dict, Any
from openai import OpenAI

from ..core.paths import OPENAI_API_KEY, MODEL_ID

if not OPENAI_API_KEY:
    raise RuntimeError("環境變數 GPT_API 尚未設定")

client = OpenAI(api_key=OPENAI_API_KEY)

# 僅保留跨模型通用且安全的參數；超時與取樣參數改由各步驟自行設定
GPT_KWARGS: Dict[str, Any] = {
    "model": MODEL_ID,
    "max_tokens": 4096,   # 需要時會在呼叫端互轉為 max_completion_tokens
}
