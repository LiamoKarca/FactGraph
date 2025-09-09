"""
_gpt_extract 包裝
"""
import time
from itertools import count

from openai import OpenAIError, APITimeoutError

from .client import client, GPT_KWARGS
from ..core.paths import EXTRACT_PROMPT_PATH

EXTRACTION_PROMPT = EXTRACT_PROMPT_PATH.read_text(encoding='utf-8-sig')


def extract_entities_relations(text: str) -> str:
    backoff = 5
    for _ in count():
        try:
            stream = client.chat.completions.create(stream=True, response_format={'type': 'json_object'},
                                                    messages=[{'role': 'system', 'content': EXTRACTION_PROMPT},
                                                              {'role': 'user', 'content': text}], **GPT_KWARGS)
            chunks = []
            for ch in stream:
                delta = ch.choices[0].delta.content
                if delta:
                    print(delta, end='', flush=True)
                    chunks.append(delta)
            result = ''.join(chunks).strip()
            return result
        except (OpenAIError, APITimeoutError) as exc:
            pass
        print(f'[WARN] GPT 抽取失敗: {exc} -> {backoff}s retry')
        time.sleep(backoff)
        backoff = min(backoff * 2, 60)
