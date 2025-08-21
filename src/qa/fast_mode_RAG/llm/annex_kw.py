"""
annex_kw.py
- 讀取 data/interim/fast_mode/user-input 下唯一的 .txt 檔之「內容」
- 載入 *_keywords.json
- 最終輸出：<原始檔名>_kw_annex.json，其中 <原始檔名> 會把 keywords 縮寫成 kw
  例如：test_keywords.json -> test_kw_annex.json
"""

import json
from pathlib import Path
import sys
from typing import Any, List

# 路徑設定（必要時可改成 argparse 參數）
USER_INPUT_DIR = Path("data/interim/fast_mode/user-input")
KEYWORDS_FILE = Path("data/interim/fast_mode/keywords/test_keywords.json")


def _find_single_txt(base_dir: Path) -> Path:
    txt_files = sorted(base_dir.glob("*.txt"))
    if not txt_files:
        print("[錯誤] user-input 資料夾內沒有 txt 檔案")
        sys.exit(1)
    if len(txt_files) > 1:
        print("[錯誤] user-input 資料夾內存在多個 txt 檔案，請確認")
        sys.exit(1)
    return txt_files[0]


def _load_json(p: Path) -> Any:
    if not p.exists():
        print(f"[錯誤] 找不到 {p}")
        sys.exit(1)
    try:
        with p.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[錯誤] JSON 解析失敗: {e}")
        sys.exit(1)


def _ensure_annex(data: Any, annex_str: str) -> List[Any]:
    """
    保證回傳 list 供後續一致處理。
    - 若 data 是 list：把 annex_str 放到 index 0；若已在其他位置則移到最前；若已在第一格則不動。
    - 若 data 非 list：輸出 [annex_str, data]
    """
    if isinstance(data, list):
        if data and data[0] == annex_str:
            return data
        try:
            idx = data.index(annex_str)
            data.pop(idx)
        except ValueError:
            pass
        data.insert(0, annex_str)
        return data
    return [annex_str, data]


def _build_output_name(src_json: Path) -> Path:
    """
    將 <name>_keywords.json -> <name>_kw_annex.json
    若檔名不含 `_keywords`，則做一般性的 'keywords' -> 'kw' 取代。
    """
    stem = src_json.stem  # 不含 .json
    if stem.endswith("_keywords"):
        stem = stem[:-len("_keywords")] + "_kw"
    else:
        stem = stem.replace("keywords", "kw")
    return src_json.with_name(stem + "_annex.json")


def main():
    # 讀取唯一 txt 檔之內容
    txt_file = _find_single_txt(USER_INPUT_DIR)
    txt_content = txt_file.read_text(encoding="utf-8-sig").strip()
    annex_str = {"user_question":txt_content}

    # 載入原始 keywords JSON
    data = _load_json(KEYWORDS_FILE)

    # 整併
    out_list = _ensure_annex(data, annex_str)

    # 產生輸出檔名（縮寫 keywords -> kw）
    output_file = _build_output_name(KEYWORDS_FILE)

    # 輸出
    with output_file.open("w", encoding="utf-8-sig") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已輸出至 {output_file}")


if __name__ == "__main__":
    main()
