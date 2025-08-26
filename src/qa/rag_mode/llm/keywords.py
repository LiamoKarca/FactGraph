"""
功能：
- 讀取 data/interim/rag_mode/user-input/*.txt
- 使用 PROMPT_KEYWORDS 生成關鍵詞/線索
- 逐檔輸出至 data/interim/rag_mode/keywords/<stem>_keywords.json（純 JSON，無 ``` 標記）

相依：
- src/qa/rag_mode/llm/config.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Any

from config import (
    PATHS, MODELS,
    make_openai_client,
    read_text, write_text,
)

# ──────────────────────────────────────────────────────────────────────────────
# 工具：清除 code fence、抽出純 JSON
# ──────────────────────────────────────────────────────────────────────────────

_CODEFENCE_RE = re.compile(
    r"^```(?:json|JSON)?\s*([\s\S]*?)\s*```$", re.MULTILINE)
_BRACED_RE = re.compile(r"\{[\s\S]*\}")
_BRACKET_RE = re.compile(r"\[[\s\S]*\]")


def _try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _strip_all_codefences(text: str) -> str:
    """
    移除所有三引號 code fence；保留其中內容。
    支援 ```json / ```JSON / ``` 任意語言。
    """
    t = text.strip()
    # 先嘗試匹配單一大區塊
    m = _CODEFENCE_RE.search(t)
    if m:
        return m.group(1).strip()
    # 全域刪除所有 ```...``` 標記
    t = re.sub(r"```(?:\w+)?", "", t)
    t = t.replace("```", "")
    return t.strip()


def _extract_first_json_segment(text: str) -> Optional[str]:
    """
    在雜訊中擷取第一個可能的 JSON 片段：
    1) 先找 {...}
    2) 再找 [...]
    找到後回傳原始片段字串；否則 None。
    """
    m = _BRACED_RE.search(text)
    if m:
        return m.group(0)
    m = _BRACKET_RE.search(text)
    if m:
        return m.group(0)
    return None


def _sanitize_to_pure_json(raw: str) -> str:
    """
    將 LLM 原始輸出轉為「純 JSON 字串」：
    - 去除 ```json / ``` 等 code fence
    - 移除多餘描述性文字（若存在）
    - 必要時從雜訊中擷取第一個 {...} 或 [...] 片段
    - 驗證可被 json.loads 解析後，再以 json.dumps 美化輸出
    """
    if not raw or not raw.strip():
        raise ValueError("LLM 輸出為空")

    # 0) 先移除 code fence
    cleaned = _strip_all_codefences(raw)

    # 1) 直接解析
    obj = _try_parse_json(cleaned)
    if obj is None:
        # 2) 從雜訊擷取 JSON 片段
        seg = _extract_first_json_segment(cleaned)
        if not seg:
            # 3) 有些模型會加上前綴 'JSON'、'Here is JSON:' 等，嘗試清掉後再抓
            tmp = re.sub(r"(?i)\bjson\b\s*[:：]?", "", cleaned).strip()
            seg = _extract_first_json_segment(tmp)
            if not seg:
                raise ValueError("無法在 LLM 輸出中找到合法的 JSON 片段")
        obj = _try_parse_json(seg)
        if obj is None:
            raise ValueError("擷取到的 JSON 片段無法解析")

    # 重新 dump 成乾淨 JSON（不帶任何 ``` 或多餘文字）
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# LLM 與流程
# ──────────────────────────────────────────────────────────────────────────────

def _list_txt_files() -> List[Path]:
    return sorted(PATHS.USER_INPUT_DIR.glob("*.txt"))


def _load_prompt_template() -> str:
    if not PATHS.PROMPT_KEYWORDS.exists():
        raise FileNotFoundError(f"找不到關鍵詞抽取 Prompt：{PATHS.PROMPT_KEYWORDS}")
    return read_text(PATHS.PROMPT_KEYWORDS)


def _call_llm(client, text: str, prompt_tmpl: str) -> str:
    # 將使用者文本放入提示
    prompt = prompt_tmpl.replace("{user_text}", text)

    resp = client.responses.create(
        model=MODELS.keywords_model,
        input=[{"role": "user", "content": prompt}],
    )

    out = (getattr(resp, "output_text", "") or "").strip()
    if not out:
        # 嘗試其它欄位（不同 SDK 版本差異）
        try:
            out = (
                resp.output[0].content[0].text  # type: ignore[attr-defined]
                if getattr(resp, "output", None) else ""
            )
        except Exception:
            out = ""
    if not out:
        raise RuntimeError("LLM 回傳為空，請檢查模型或 API 設定")

    return out


def main() -> None:
    client = make_openai_client()
    tmpl = _load_prompt_template()

    txt_files = _list_txt_files()
    if not txt_files:
        print(f"[資訊] 未找到任何 .txt：{PATHS.USER_INPUT_DIR}")
        return

    PATHS.KEYWORDS_DIR.mkdir(parents=True, exist_ok=True)

    for p in txt_files:
        stem = p.stem
        user_text = read_text(p)
        raw_result = _call_llm(client, user_text, tmpl)

        # 轉為純 JSON
        pure_json_text = _sanitize_to_pure_json(raw_result)

        out_path = PATHS.KEYWORDS_DIR / f"{stem}_keywords.json"
        write_text(out_path, pure_json_text)
        print(f"[完成] {p.name} → {out_path}")


if __name__ == "__main__":
    main()
