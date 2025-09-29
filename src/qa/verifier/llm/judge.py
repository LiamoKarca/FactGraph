"""
gpt_judge 包裝
"""
import time
from itertools import count

from openai import OpenAIError, APITimeoutError

from .client import client, GPT_KWARGS
from ..core.paths import JUDGE_PROMPT_PATH

JUDGE_PROMPT = JUDGE_PROMPT_PATH.read_text(encoding='utf-8-sig')


def judge_news_kb(text: str) -> str:
    backoff = 5
    for _ in count():
        try:
            stream = client.chat.completions.create(
                stream=True,
                messages=[
                    {'role': 'system', 'content': JUDGE_PROMPT},
                    {'role': 'user', 'content': text}
                ],
                **GPT_KWARGS
            )
            chunks = []
            for ch in stream:
                delta = ch.choices[0].delta.content
                if delta:
                    print(delta, end='', flush=True)
                    chunks.append(delta)
            result = ''.join(chunks).strip()
            return result
        except (OpenAIError, APITimeoutError) as exc:
            # 只在 except 作用域中使用 exc，避免 UnboundLocalError
            print(f'[WARN] GPT 判斷失敗: {exc} -> {backoff}s retry')
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        except Exception as exc:
            # 收斂不可預期例外，仍然以同樣退避重試
            print(f'[WARN] 非預期例外: {exc} -> {backoff}s retry')
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
