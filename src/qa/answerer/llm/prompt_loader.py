from pathlib import Path

__all__ = ['load_prompt']


def load_prompt(path: Path) -> str:
    return path.read_text(encoding='utf-8-sig')
