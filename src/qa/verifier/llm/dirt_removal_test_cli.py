"""
CLI：雜訊剔除單檔測試（GPT-5 / Responses API 相容）

用途:
    - 讀入合併文本檔（包含以 [n] 開頭的知識條目）。
    - 以檔名（或指定）推導 news_id。
    - 呼叫 run_dirt_removal（其內部已採用 Responses API 優先與 JSON Schema 嚴格模式）。
    - 所有輸出皆採用 utf-8-sig 編碼。

使用範例:
    python -m src.qa.verifier.llm.dirt_removal_test_cli \
        data/processed/verifier/dirt_removal/test/TEST_news_kg_test_1024.txt

備註:
    - 模型與回應行為由 run_dirt_removal 所屬模組決定（預設 gpt-5 系列）。
    - 可透過環境變數 OPENAI_RESP_MODEL / OPENAI_CHAT_MODEL 調整，或由 --model 覆寫。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

# 只負責 CLI 讀寫與參數解析；LLM 呼叫與檔案生成都委派給下層模組（SRP）
from .dirt_removal import run_dirt_removal  # noqa: E402


def _read_file_utf8_sig(path: str) -> str:
    """以 utf-8 或 utf-8-sig 嘗試讀取文字檔。

    Args:
        path: 檔案路徑。

    Returns:
        str: 檔案內容。
    """
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "rb") as f:  # 最後保底
        return f.read().decode("utf-8", errors="ignore")


def _guess_news_id(input_path: str) -> str:
    """以檔名推估新聞 ID。

    規則:
        - 取去除副檔名後的檔名。
        - 移除非英數、底線與連字號以外的字元。

    Args:
        input_path: 輸入檔路徑。

    Returns:
        str: 推估後的 news_id；若為空則回傳 "unknown"。
    """
    name = os.path.splitext(os.path.basename(input_path))[0]
    safe = "".join(ch for ch in name if ch.isalnum() or ch in {"_", "-"})
    return safe or "unknown"


def _apply_model_override(model: str | None) -> None:
    """以環境變數覆寫下層模組的模型選擇（非必要，僅提供 CLI 便利）。

    Args:
        model: 指定模型名稱；None 表示不覆寫。
    """
    if model:
        # run_dirt_removal 會優先讀取 OPENAI_RESP_MODEL
        os.environ["OPENAI_RESP_MODEL"] = model


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    """解析命令列參數。

    Args:
        argv: 參數清單；None 則讀取 sys.argv[1:].

    Returns:
        argparse.Namespace: 解析結果。
    """
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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="指定模型（例如 gpt-5-mini）；僅覆寫此行程的 OPENAI_RESP_MODEL。",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> Tuple[str, str]:
    """執行 CLI 主流程。

    Args:
        argv: 命令列參數清單。

    Returns:
        Tuple[str, str]: (filtered_text, saved_path)
            filtered_text: 合併輸出文本。
            saved_path: 合併輸出檔案路徑。
    """
    args = _parse_args(argv)

    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"找不到輸入檔：{args.input_path}")

    _apply_model_override(args.model)

    combined_text = _read_file_utf8_sig(args.input_path)
    news_id = args.news_id or _guess_news_id(args.input_path)

    print(f"🧪 INPUT: {args.input_path}")
    print(f"🆔 NEWS_ID: {news_id}")
    print(f"📂 OUT_DIR: {args.out_dir}")
    if args.model:
        print(f"🧭 MODEL (override): {args.model}")

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
