"""
集中管理 answerer 子套件所需的路徑常數。

- 自動尋找 FactGraph 專案根目錄（需包含 models/ 與 data/）
- 定義各子目錄及檔案路徑
- 提供 print_paths() 方便除錯
$ python -m experiment.answerer.core.paths
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

__all__ = [
    'PROJECT_ROOT', 'FACTGRAPH_SRC', 'ANSWERER_ROOT', 'DATA_DIR',
    'RAW_KG_DIR', 'PROCESSED_KG_DIR', 'INTERIM_ANSWERER_DIR', 'USER_INPUT_DIR',
    'KG_EMB_PATH', 'KG_CSV_PATH', 'OUT_DIR', 'USER_KG_PATH', 'USER_JUDGE_PATH',
    'CKIP_ROOT', 'PROMPTS_DIR', 'EXTRACT_PROMPT_PATH', 'JUDGE_PROMPT_PATH',
    'print_paths'
]


def _find_project_root(start: Path | None = None) -> Path:
    """從 start 位置向上尋找包含 models/ 與 data/ 的目錄作為專案根目錄。"""
    current = Path(start or __file__).resolve()
    for parent in (current, *current.parents):
        if (parent / 'models').is_dir() and (parent / 'data').is_dir():
            return parent
    raise RuntimeError(
        "無法定位 FactGraph 專案根目錄，需包含 'models' 與 'data' 資料夾"
    )


# 專案結構根目錄
PROJECT_ROOT: Path = _find_project_root()
FACTGRAPH_SRC: Path = PROJECT_ROOT / 'experiment'
ANSWERER_ROOT: Path = FACTGRAPH_SRC / 'answerer'

# 資料目錄
DATA_DIR: Path = PROJECT_ROOT / 'experiment' / 'data'
RAW_KG_DIR: Path = DATA_DIR / 'raw'
PROCESSED_KG_DIR: Path = DATA_DIR / 'processed'
INTERIM_ANSWERER_DIR: Path = DATA_DIR / 'interim' / 'answerer'
USER_INPUT_DIR: Path = INTERIM_ANSWERER_DIR / 'user-input'

# 知識圖譜檔案
KG_EMB_PATH: Path = PROCESSED_KG_DIR / 'neo4j-kg.emb.npy'
KG_CSV_PATH: Path = RAW_KG_DIR / 'neo4j-kg-raw-graph.csv'

# Answerer 輸出目錄
OUT_DIR: Path = DATA_DIR / 'processed' / 'answerer'
USER_KG_PATH: Path = OUT_DIR / 'user-kg.txt'
USER_JUDGE_PATH: Path = OUT_DIR / 'user-qa-judge.txt'

# 模型與 prompt
CKIP_ROOT: Path = PROJECT_ROOT / 'models' / 'CKIP' / 'models--ckiplab--bert-base-chinese'
PROMPTS_DIR: Path = ANSWERER_ROOT / 'prompts'
EXTRACT_PROMPT_PATH: Path = PROMPTS_DIR / 'sentence-triple-extraction.txt'
JUDGE_PROMPT_PATH: Path = PROMPTS_DIR / 'character.txt'


def print_paths() -> None:
    """列印所有預設路徑以供除錯用。"""
    from pprint import pprint
    paths: Dict[str, Any] = {name: globals()[name] for name in __all__}
    pprint(paths)
