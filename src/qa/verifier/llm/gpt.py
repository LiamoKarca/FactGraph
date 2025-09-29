from __future__ import annotations

import time
from typing import Any, Dict, Tuple, Optional

from openai import OpenAI, OpenAIError, APITimeoutError

__all__ = ["GPTClient"]

class GPTClient:
    """
    OpenAI Chat Completions 輕量封裝（非串流）：
    - HTTP 層 timeout 用 client/with_options 設定，不放進 kwargs
    - max_tokens → 自動轉為 max_completion_tokens
    - 碰到 400（Unsupported/Unrecognized）會剔除該 key 後重試
    """

    _MAYBE_UNSUPPORTED_KEYS = (
        "temperature","top_p","presence_penalty","frequency_penalty",
        "seed","response_format","stream",
    )

    def __init__(self, api_key: str, model_id: str, **kwargs):
        self._timeout = kwargs.pop("timeout", None)
        # 移除可能殘留的 request_timeout/timeout，避免被當成 body 參數
        kwargs.pop("request_timeout", None)

        # 建立 client；若舊版 SDK 不吃 timeout 參數，改用預設再在呼叫時 with_options
        try:
            self._client = OpenAI(api_key=api_key, timeout=self._timeout) if self._timeout else OpenAI(api_key=api_key)
        except TypeError:
            self._client = OpenAI(api_key=api_key)

        base: Dict[str, Any] = {"model": model_id, **kwargs}
        if "max_tokens" in base and "max_completion_tokens" not in base:
            base["max_completion_tokens"] = base.pop("max_tokens")
        for k in list(base.keys()):
            if base[k] is None:
                base.pop(k)
        self._base_kwargs = base

    def _strip_unsupported_from_error(self, err: Exception) -> bool:
        s = str(err); changed = False
        if ("'param': 'max_tokens'" in s or '"param": "max_tokens"' in s or "Unsupported parameter: 'max_tokens'" in s):
            if "max_tokens" in self._base_kwargs:
                val = self._base_kwargs.pop("max_tokens")
                self._base_kwargs["max_completion_tokens"] = val
                print("[INFO] Converted 'max_tokens' → 'max_completion_tokens'."); changed = True
        if ("Unsupported parameter" in s) or ("Unsupported value" in s) or ("Unrecognized request argument" in s):
            for key in list(self._base_kwargs.keys()):
                for cand in self._MAYBE_UNSUPPORTED_KEYS:
                    if key == cand and cand in s:
                        self._base_kwargs.pop(key, None)
                        print(f"[INFO] Removed unsupported parameter '{key}'."); changed = True
        return changed

    def _create(self):
        return (self._client.with_options(timeout=self._timeout) if self._timeout else self._client).chat.completions.create

    def chat_with_meta(self, system_prompt: str, user_prompt: str, *, max_retries: int = 3
                       ) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        backoff = 5; attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._create()(
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}],
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
                print("\n[INFO] 使用者中止（KeyboardInterrupt）。"); raise
            except (OpenAIError, APITimeoutError) as err:
                if self._strip_unsupported_from_error(err):
                    continue
                if attempt >= max_retries: raise
                print(f"[WARN] GPT retry in {backoff}s (attempt {attempt}/{max_retries}) → {err}")
                try: time.sleep(backoff)
                except KeyboardInterrupt: print("\n[INFO] 使用者中止（KeyboardInterrupt）。"); raise
                backoff = min(backoff * 2, 60)

    def chat(self, system_prompt: str, user_prompt: str, *, max_retries: int = 3) -> str:
        text, _, _, _ = self.chat_with_meta(system_prompt, user_prompt, max_retries=max_retries)
        return text
