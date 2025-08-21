"""
合併 CNA / EY / MOI / PTS 的新聞 JSON 檔案（純合併，不做向量化）
輸出：data/processed/news_merge/cna_ey_moi_pts.json
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Any

DEFAULT_INPUTS = [
    Path("data/raw/news/cna/cna_news.json"),
    Path("data/raw/news/ey/ey_news.json"),
    Path("data/raw/news/moi/moi_news.json"),
    Path("data/raw/news/pts/pts_news.json"),
]
DEFAULT_OUTPUT = Path("data/processed/news_merge/cna_ey_moi_pts.json")


def load_json(file_path: Path) -> Any:
    if not file_path.exists():
        print(f"⚠️ 檔案不存在: {file_path}")
        return []
    try:
        with file_path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失敗: {file_path} ({e})")
        return []


def save_json(data, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8-sig") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge(inputs: List[Path], output: Path) -> int:
    merged: List[Any] = []
    for f in inputs:
        payload = load_json(f)
        if isinstance(payload, list):
            merged.extend(payload)
        else:
            print(f"⚠️ {f} 內容不是 list，已忽略。")
    save_json(merged, output)
    print(f"✅ 合併完成，共 {len(merged)} 筆 -> {output}")
    return len(merged)


def main():
    # 若需自訂，可改成 argparse；預設使用常用路徑
    merge(DEFAULT_INPUTS, DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
