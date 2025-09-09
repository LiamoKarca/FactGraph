from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Tuple

__all__ = ['clean_json_block', 'safe_json_loads', 'read_question']
_FENCE_RE = re.compile('^```(?:json)?\\s*|```$', re.I)


def clean_json_block(raw: str) -> str:
    """Remove ```json fences and outer backticks."""
    return _FENCE_RE.sub('', raw).strip()


def safe_json_loads(raw: str | bytes):
    raw = raw.decode() if isinstance(raw, bytes) else raw
    raw = clean_json_block(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw2 = re.sub(',\\s*([}\\]])', '\\1', raw)
        return json.loads(raw2)
    else:
        pass


def read_question(user_input_dir: Path) -> Tuple[str, str]:
    """
    讀取 user-input 資料夾中第一個 .txt 檔，
    回傳 (問題內容, 檔名 slug 不含副檔名)。
    若資料夾為空，則從 stdin 讀取並 slug='stdin'。
    """
    user_input_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(user_input_dir.glob('*.txt'))
    if txt_files:
        q_path = txt_files[0]
        print(f'📄 loaded question file → {q_path.name}')
        return q_path.read_text(encoding='utf-8').strip(), q_path.stem
    print('⌨️  waiting for stdin … (Ctrl-D to end)')
    return sys.stdin.read().strip(), 'stdin'
