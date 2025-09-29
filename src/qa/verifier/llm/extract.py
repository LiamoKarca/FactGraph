"""
_gpt_extract 包裝

設計目標：
1) I/O 完全相容：extract_entities_relations(text) -> str（JSON 字串）
2) 先嘗試「串流」以保留即時列印；若中途逾時/中斷 → 立即回退一次「非串流」
3) 指數退避重試；避免因臨時網路抖動或服務端延遲而整段崩潰
4) 明確日誌：只在例外發生時輸出警告，避免未定義變數造成二次錯誤
"""

from __future__ import annotations

import time
from itertools import count
from typing import List

from openai import OpenAIError, APITimeoutError

# 與既有客戶端設定保持一致；不要動
from .client import client, GPT_KWARGS
from ..core.paths import EXTRACT_PROMPT_PATH

EXTRACTION_PROMPT = EXTRACT_PROMPT_PATH.read_text(encoding="utf-8-sig")


def _non_stream_fallback(messages: List[dict]) -> str:
    """
    回退：以非串流單次呼叫拿完整 JSON。
    - 保持與主路徑相同的 response_format / 溫度等參數（由 GPT_KWARGS 控制）
    """
    resp = client.chat.completions.create(
        stream=False,
        response_format={"type": "json_object"},
        messages=messages,
        **GPT_KWARGS,
    )
    content = (resp.choices[0].message.content or "").strip()
    return content


def extract_entities_relations(text: str) -> str:
    """
    抽取實體與關係，回傳 JSON 字串。
    - 優先使用串流（沿用你原來逐塊 print 的行為）
    - 若串流中途中斷/逾時/錯誤，立即回退一次非串流
    - 若仍失敗則進入指數退避重試
    """
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": text},
    ]

    backoff = 5  # 秒；最小退避
    for attempt in count(1):
        try:
            # --- 首選：串流 ---
            stream = client.chat.completions.create(
                stream=True,
                response_format={"type": "json_object"},
                messages=messages,
                **GPT_KWARGS,
            )
            chunks: List[str] = []
            for ch in stream:
                delta = ch.choices[0].delta.content
                if delta:
                    # 保留你原有的即時列印行為（便於除錯）
                    print(delta, end="", flush=True)
                    chunks.append(delta)

            result = "".join(chunks).strip()
            if result:
                return result

            # 正常結束但結果是空的，視為異常處理 → 走回退
            raise RuntimeError("empty streamed result")

        except (APITimeoutError, OpenAIError, Exception) as exc:
            # --- 串流失敗：嘗試一次非串流回退 ---
            print(
                f"[WARN] GPT 串流抽取失敗（第 {attempt} 次）："
                f"{exc.__class__.__name__}: {exc} → 嘗試非串流回退",
                flush=True,
            )
            try:
                content = _non_stream_fallback(messages)
                if content:
                    print("[INFO] 非串流回退成功。", flush=True)
                    return content
                else:
                    print(
                        "[WARN] 非串流回退回傳為空，將進入重試。",
                        flush=True,
                    )
            except (APITimeoutError, OpenAIError, Exception) as exc2:
                print(
                    f"[WARN] 非串流回退也失敗：{exc2.__class__.__name__}: {exc2}",
                    flush=True,
                )

            # --- 指數退避重試 ---
            print(f"[INFO] {backoff}s 後重試。", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
