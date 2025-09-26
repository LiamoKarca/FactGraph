from __future__ import annotations

import time
from typing import Any, Dict, Tuple, Optional

from openai import OpenAI, OpenAIError, APITimeoutError

__all__ = ['GPTClient']


class GPTClient:
    """
    - 不預設 max_completion_tokens（若呼叫端不傳，就不送，交由模型最大上限）
    - 舊參數 `max_tokens` → 自動轉為 `max_completion_tokens`
    - 遇到不支援的抽樣參數（temperature/top_p/...）會自動移除並重試
    - 提供 chat() 與 chat_with_meta()（回傳 finish_reason / usage）
    """

    _MAYBE_UNSUPPORTED_KEYS = (
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    )

    def __init__(self, api_key: str, model_id: str, **kwargs):
        self._client = OpenAI(api_key=api_key)

        base: Dict[str, Any] = {"model": model_id, **kwargs}

        # 1) 參數名稱相容：max_tokens → max_completion_tokens
        if "max_tokens" in base and "max_completion_tokens" not in base:
            base["max_completion_tokens"] = base.pop("max_tokens")

        # 2) 移除值為 None 的 key，避免送出 None
        for k in list(base.keys()):
            if base[k] is None:
                base.pop(k)

        self._base_kwargs = base

    def _strip_unsupported_from_error(self, err: Exception) -> bool:
        """
        從錯誤訊息中偵測不支援參數；若有就從 _base_kwargs 移除並回傳 True（可立即重試）。
        """
        msg = str(err)
        did_change = False

        # 雙保險處理 max_tokens
        if ("'param': 'max_tokens'" in msg or '"param": "max_tokens"' in msg
                or "Unsupported parameter: 'max_tokens'" in msg):
            if "max_tokens" in self._base_kwargs:
                val = self._base_kwargs.pop("max_tokens")
                self._base_kwargs["max_completion_tokens"] = val
                print("[INFO] Converted 'max_tokens' → 'max_completion_tokens'.")
                did_change = True

        # 常見抽樣參數
        for key in self._MAYBE_UNSUPPORTED_KEYS:
            if (f"'param': '{key}'" in msg or f'"param": "{key}"' in msg
                    or f"Unsupported value: '{key}'" in msg
                    or f"Unsupported parameter: '{key}'" in msg):
                if key in self._base_kwargs:
                    self._base_kwargs.pop(key, None)
                    print(
                        f"[INFO] Removed unsupported parameter '{key}' for model {self._base_kwargs.get('model')}.")
                    did_change = True

        return did_change

    def _create(self):
        return self._client.chat.completions.create

    def chat_with_meta(
        self, system_prompt: str, user_prompt: str, *, max_retries: int = 5
    ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        """
        回傳: (text, finish_reason, usage(dict), model_id)
        """
        backoff = 5
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._create()(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **self._base_kwargs,
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                finish_reason = getattr(choice, "finish_reason", None)
                usage = getattr(resp, "usage", None)
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                } if usage else {}
                model_id = getattr(resp, "model", None)
                return text, finish_reason, usage_dict, model_id

            except KeyboardInterrupt:
                print("\n[INFO] 使用者中止（KeyboardInterrupt）。")
                raise

            except (OpenAIError, APITimeoutError) as err:
                if self._strip_unsupported_from_error(err):
                    continue
                if attempt >= max_retries:
                    raise
                print(
                    f"[WARN] GPT retry in {backoff}s (attempt {attempt}/{max_retries}) → {err}")
                try:
                    time.sleep(backoff)
                except KeyboardInterrupt:
                    print("\n[INFO] 使用者中止（KeyboardInterrupt）。")
                    raise
                backoff = min(backoff * 2, 60)

    def chat(self, system_prompt: str, user_prompt: str, *, max_retries: int = 5) -> str:
        text, _, _, _ = self.chat_with_meta(
            system_prompt, user_prompt, max_retries=max_retries)
        return text
