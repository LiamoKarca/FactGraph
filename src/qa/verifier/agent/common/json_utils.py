"""健壯 JSON 解析工具。"""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fence(text: str) -> str:
    """移除 Markdown 風格 code fence。

    Args:
        text: 原始文字。

    Returns:
        移除圍欄後的文字。
    """
    text = (text or "").strip()
    if text.startswith("```"):
        i = text.find("\n")
        if i != -1:
            text = text[i + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _find_first_json_blob(text: str) -> str | None:
    """尋找第一段配對的大括號/中括號 JSON 片段。"""
    start = None
    open_ch = None
    depth = 0
    for i, ch in enumerate(text):
        if start is None and ch in "{[":
            start = i
            open_ch = ch
            depth = 1
            continue
        if start is not None:
            if ch == "{":
                if open_ch == "{":
                    depth += 1
            elif ch == "}":
                if open_ch == "{":
                    depth -= 1
            elif ch == "[":
                if open_ch == "[":
                    depth += 1
            elif ch == "]":
                if open_ch == "[":
                    depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_fix_minor_json_issues(s: str) -> str:
    """嘗試修正尾逗號等輕微錯誤。"""
    return re.sub(r",\s*([}\]])", r"\1", s or "").strip()


def parse_json_safely(raw: str) -> Any:
    """健壯 JSON 解析。

    Args:
        raw: 可能含雜訊/圍欄/多餘逗號的字串。

    Returns:
        解析後的 Python 物件。

    Raises:
        ValueError: 無法解析時。
    """
    text = _strip_code_fence(raw or "")
    try:
        return json.loads(text)
    except Exception:
        pass

    blob = _find_first_json_blob(text)
    if blob:
        for cand in (blob, _try_fix_minor_json_issues(blob)):
            try:
                return json.loads(cand)
            except Exception:
                pass

    for line in text.splitlines():
        s = (line or "").strip()
        if not s:
            continue
        try:
            return json.loads(s)
        except Exception:
            per = _find_first_json_blob(s)
            if per:
                try:
                    return json.loads(_try_fix_minor_json_issues(per))
                except Exception:
                    continue

    raise ValueError("Unable to parse JSON from input.")


def run_self_test() -> bool:
    """
    執行 JSON 解析器自我檢測。

    測試各種髒資料（如代碼圍欄、尾隨逗號、周圍文字）是否能被成功解析。

    Returns:
        bool: 如果所有測試都通過，則回傳 True，否則回傳 False。
    """
    print("=" * 30)
    print("Running JSON Parser Self-Test...")
    print("=" * 30)

    test_cases = [
        # (描述, 輸入字串, 預期輸出)
        ("Simple Dict", '{"key": "value"}', {"key": "value"}),
        ("Simple List", '[1, 2, 3]', [1, 2, 3]),
        (
            "Code Fence (json)",
            '```json\n{"a": 1}\n```',
            {"a": 1}
        ),
        (
            "Code Fence (no lang)",
            '```\n{"b": 2}\n```',
            {"b": 2}
        ),
        (
            "Trailing Comma (Dict)",
            '{"a": 1, "b": 2,}',
            {"a": 1, "b": 2}
        ),
        (
            "Trailing Comma (List)",
            '[1, 2, 3,]',
            [1, 2, 3]
        ),
        (
            "Text Surrounding Blob",
            'Here is the data: {"id": 123}. Please review.',
            {"id": 123}
        ),
        (
            "Blob on a new line",
            'Blah blah\nDATA: [1, 2, 3]\nMore blah',
            [1, 2, 3]
        ),
        (
            "Fence + Trailing Comma",
            '```json\n{"data": [1, 2,], "status": "ok",}\n```',
            {"data": [1, 2], "status": "ok"}
        ),
        (
            "Nested Trailing Comma",
            '{"outer": {"inner": [1,],}}',
            {"outer": {"inner": [1]}}
        ),
        # 預期會失敗並引發 ValueError 的測試
        (
            "Invalid (Just Text)",
            "This is just text.",
            "RAISE_ERROR"
        ),
        (
            "Invalid (Mismatched)",
            '{"a": 1, "b": [2, 3}',
            "RAISE_ERROR"
        ),
    ]

    success_count = 0
    failed_tests = []

    for desc, raw_input, expected_output in test_cases:
        try:
            result = parse_json_safely(raw_input)
            if expected_output == "RAISE_ERROR":
                # 預期失敗但成功了，這是一個錯誤
                failed_tests.append(
                    f"  [FAIL] {desc}: Expected ValueError, but got: {result}"
                )
            elif result == expected_output:
                # 成功且結果匹配
                success_count += 1
                print(f"  [OK] {desc}")
            else:
                # 成功但結果不匹配
                failed_tests.append(
                    f"  [FAIL] {desc}: Output mismatch.\n"
                    f"         Expected: {expected_output}\n"
                    f"         Got: {result}"
                )
        except ValueError:
            if expected_output == "RAISE_ERROR":
                # 失敗了，且符合預期
                success_count += 1
                print(f"  [OK] {desc} (Correctly raised ValueError)")
            else:
                # 預期成功但意外失敗了
                failed_tests.append(
                    f"  [FAIL] {desc}: Raised ValueError unexpectedly."
                )
        except Exception as e:
            # 其他未預期的嚴重錯誤
            failed_tests.append(
                f"  [CRITICAL] {desc}: Raised unexpected error: {e}"
            )

    print("-" * 30)
    if not failed_tests:
        print(f"Self-Test Result: ALL CLEAR ({success_count}/{len(test_cases)} passed).")
        print("=" * 30)
        return True
    else:
        print(f"Self-Test Result: {len(failed_tests)} FAILED TESTS.")
        for failure in failed_tests:
            print(failure)
        print("=" * 30)
        return False


if __name__ == "__main__":
    print("Module loaded directly. Running self-test...")
    run_self_test()