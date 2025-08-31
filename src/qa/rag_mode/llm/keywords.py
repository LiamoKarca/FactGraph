from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import List, Optional, Any

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
    t = text.strip()
    m = _CODEFENCE_RE.search(t)
    if m:
        return m.group(1).strip()
    t = re.sub(r"```(?:\w+)?", "", t)
    t = t.replace("```", "")
    return t.strip()


def _extract_first_json_segment(text: str) -> Optional[str]:
    m = _BRACED_RE.search(text)
    if m:
        return m.group(0)
    m = _BRACKET_RE.search(text)
    if m:
        return m.group(0)
    return None


def _sanitize_to_pure_json(raw: str) -> str:
    if not raw or not raw.strip():
        raise ValueError("LLM 輸出為空")

    cleaned = _strip_all_codefences(raw)

    obj = _try_parse_json(cleaned)
    if obj is None:
        seg = _extract_first_json_segment(cleaned)
        if not seg:
            tmp = re.sub(r"(?i)\bjson\b\s*[:：]?", "", cleaned).strip()
            seg = _extract_first_json_segment(tmp)
            if not seg:
                raise ValueError("無法在 LLM 輸出中找到合法的 JSON 片段")
        obj = _try_parse_json(seg)
        if obj is None:
            raise ValueError("擷取到的 JSON 片段無法解析")

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
    prompt = prompt_tmpl.replace("{user_text}", text)
    resp = client.responses.create(
        model=MODELS.keywords_model,
        input=[{"role": "user", "content": prompt}],
    )
    out = (getattr(resp, "output_text", "") or "").strip()
    if not out:
        try:
            out = (resp.output[0].content[0].text  # type: ignore[attr-defined]
                   if getattr(resp, "output", None) else "")
        except Exception:
            out = ""
    if not out:
        raise RuntimeError("LLM 回傳為空，請檢查模型或 API 設定")
    return out


def _process_one_file(client, tmpl: str, txt_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = txt_path.stem
    user_text = read_text(txt_path)
    raw_result = _call_llm(client, user_text, tmpl)
    pure_json_text = _sanitize_to_pure_json(raw_result)
    out_path = out_dir / f"{stem}_keywords.json"
    write_text(out_path, pure_json_text)
    print(f"[keywords] {txt_path} → {out_path}")
    return out_path


def main() -> None:
    client = make_openai_client()
    tmpl = _load_prompt_template()

    # Job-scoped（優先）：若提供 RAG_USER_FILE，僅處理該檔並寫入 ${RAG_JOB_DIR}/keywords
    env_user = os.getenv("RAG_USER_FILE")
    job_dir = os.getenv("RAG_JOB_DIR")
    if env_user:
        txt_path = Path(env_user)
        if not txt_path.exists():
            raise SystemExit(f"[keywords] RAG_USER_FILE 不存在：{txt_path}")
        out_dir = Path(job_dir) / "keywords" if job_dir else PATHS.KEYWORDS_DIR
        _process_one_file(client, tmpl, txt_path, out_dir)
        return

    # 全域相容模式：逐一處理 data/interim/rag_mode/user-input/*.txt → PATHS.KEYWORDS_DIR
    txt_files = _list_txt_files()
    if not txt_files:
        print(f"[資訊] 未找到任何 .txt：{PATHS.USER_INPUT_DIR}")
        return
    for p in txt_files:
        _process_one_file(client, tmpl, p, PATHS.KEYWORDS_DIR)


if __name__ == "__main__":
    main()
