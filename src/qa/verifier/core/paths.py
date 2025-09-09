"""
Verifier 子模組的路徑與環境變數設定。

定義：
  - 自動尋找專案根目錄
  - 各種資源路徑常數 (模型、資料、prompt)
  - 載入 .env，並讀取 OPENAI_API_KEY 與 MODEL_ID
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 載入環境變數
load_dotenv()


def _find_repo_root(marker: str = 'FactGraph',
                    env_var: str = 'PROJECT_ROOT') -> Path:
    """
    從當前檔案位置向上遍歷，直到資料夾名稱符合 marker。

    Args:
        marker: 標記專案根目錄的資料夾名稱
    Returns:
        專案根目錄 Path
    Raises:
        RuntimeError: 無法找到標記的根目錄
    """
    # 1️⃣ 若有環境變數就直接用
    if env_val := os.getenv(env_var):
        p = Path(env_val).resolve()
        if p.is_dir():
            return p

    # 2️⃣ 否則沿父層尋找指定資料夾
    current = Path(__file__).resolve()
    for parent in (current, *current.parents):
        if parent.name == marker:
            return parent
    raise RuntimeError(f'找不到專案根目錄: {marker}')


# 根目錄
PROJECT_ROOT: Path = _find_repo_root()

# 模型與資料目錄
CKIP_ROOT: Path = PROJECT_ROOT / 'models' / 'CKIP' / 'models--ckiplab--bert-base-chinese'
KG_EMB_PATH: Path = PROJECT_ROOT / 'data' / 'processed' / 'knowledge-graph' / 'kg-triplet.emb.npy'
KG_CSV_PATH: Path = PROJECT_ROOT / 'data' / 'raw' / 'knowledge-graph' / 'neo4j-kg-raw-graph.csv'

# 中介資料與結果目錄
USER_INPUT_DIR: Path = PROJECT_ROOT / 'data' / 'interim' / 'verifier' / 'user-input'
VEC_DIR: Path = PROJECT_ROOT / 'data' / 'interim' / 'verifier'
RES_DIR: Path = PROJECT_ROOT / 'data' / 'processed' / 'verifier'

# Prompt 檔案路徑
PROMPTS_DIR: Path = PROJECT_ROOT / 'src' / 'qa' / 'verifier' / 'prompts'
EXTRACT_PROMPT_PATH: Path = PROMPTS_DIR / 'news-triple-extraction.txt'
JUDGE_PROMPT_PATH: Path = PROMPTS_DIR / 'judger-character.txt'

# 環境變數
OPENAI_API_KEY: str | None = os.getenv('GPT_API')
MODEL_ID: str | None = os.getenv('GPT_MODEL')

if not OPENAI_API_KEY:
    raise RuntimeError('未設定環境變數 GPT_API (OPENAI_API_KEY)')

if not MODEL_ID:
    MODEL_ID = 'gpt-4o'  # 預設模型
