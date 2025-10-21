"""
CLI：雜訊剔除單檔測試

用法
----
(venv) python -m src.qa.verifier.llm.dirt_removal_test_cli \
  data/processed/verifier/TEST_news_kg_1017_mirro_TEST.txt

說明
----
- 讀入合併文本檔案（包含以 [n] 開頭的知識條目）。
- 以檔名（或其主檔名）推導 news_id。
- 呼叫 run_dirt_removal，輸出剔除後的知識庫與除錯 JSON。
- 所有輸出皆採用 utf-8-sig 編碼。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

from .dirt_removal import run_dirt_removal


def _read_file_utf8_sig(path: str) -> str:
    """以 utf-8 或 utf-8-sig 嘗試讀取文字檔。"""
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "rb") as f:  # 最後保底
        return f.read().decode("utf-8", errors="ignore")


def _guess_news_id(input_path: str) -> str:
    """
    以檔名推估 news_id：
    - 取去除副檔名後的檔名
    - 移除非英數與底線、連字號以外的字元
    """
    name = os.path.splitext(os.path.basename(input_path))[0]
    safe = "".join(ch for ch in name if ch.isalnum() or ch in {"_", "-"})
    return safe or "unknown"


def main(argv: list[str] | None = None) -> Tuple[str, str]:
    parser = argparse.ArgumentParser(prog="dirt_removal_test_cli")
    parser.add_argument(
        "input_path",
        type=str,
        help="合併文本輸入檔（包含 [n] 開頭的知識條目）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/verifier",
        help="輸出資料夾（預設：data/processed/verifier）",
    )
    parser.add_argument(
        "--news-id",
        type=str,
        default=None,
        help="自訂新聞 ID（預設以輸入檔名推估）",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"找不到輸入檔：{args.input_path}")

    combined_text = _read_file_utf8_sig(args.input_path)
    news_id = args.news_id or _guess_news_id(args.input_path)

    print(f"🧪 INPUT: {args.input_path}")
    print(f"🆔 NEWS_ID: {news_id}")
    print(f"📂 OUT_DIR: {args.out_dir}")

    filtered, saved_path = run_dirt_removal(
        combined_text=combined_text,
        news_id=news_id,
        out_dir=args.out_dir,
    )

    print("✅ 完成雜訊剔除。")
    print(f"📝 Saved: {saved_path}")
    return filtered, saved_path


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 執行失敗：{exc}", file=sys.stderr)
        sys.exit(1)
