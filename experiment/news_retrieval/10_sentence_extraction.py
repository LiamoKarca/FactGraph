"""
10_sentence_extraction.py
從 experiment/data/raw/10_news.txt 讀取 10 條短句，
使用 prompts/sentence-triple-extraction.txt 作為 system prompt，
呼叫 OpenAI (gpt-4o-mini) 進行 (subject, relation, info_need) 三元組抽取，
最後將 {"triples": [...]} 寫入 experiment/data/interim/10_sentence_extraction_result.json。

用法：
  python experiment/news_retrieval/10_sentence_extraction.py
可選參數：
  --input  自訂輸入檔 (預設 experiment/data/raw/10_news.txt)
  --prompt 自訂提示檔 (預設 experiment/news_retrieval/prompts/sentence-triple-extraction.txt)
  --output 自訂輸出檔 (預設 experiment/data/interim/10_sentence_extraction_result.json)
  --model  自訂模型 (預設 gpt-4o-mini)
  
python experiment/news_retrieval/10_sentence_extraction.py \
  --input experiment/data/raw/10_news.txt \
  --prompt experiment/news_retrieval/prompts/sentence-triple-extraction.txt \
  --output experiment/data/interim/10_sentence_extraction_result.json \
  --model gpt-4o-mini

"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

# 允許從 .env 載入 GPT_API
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

# OpenAI SDK（新版）
try:
    from openai import OpenAI  # type: ignore
    _USE_NEW_SDK = True
except Exception:
    _USE_NEW_SDK = False
    import openai  # type: ignore


DEFAULT_INPUT = "experiment/data/raw/10_news.txt"
DEFAULT_PROMPT = "experiment/news_retrieval/prompts/sentence-triple-extraction.txt"
# 後備：使用者上傳的同名檔（避免路徑尚未建立時可直接測試）
FALLBACK_PROMPT = "/mnt/data/sentence-triple-extraction.txt"
DEFAULT_OUTPUT = "experiment/data/interim/10_sentence_extraction_result.json"
DEFAULT_MODEL = "gpt-4o-mini"


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_lines_strip_numbering(fp: str | Path) -> List[str]:
    """
    讀取檔案每行，移除像是「1. 」或「1、」等編號前綴，回傳乾淨句子（保留原標點）。
    """
    out: List[str] = []
    pat = re.compile(r"^\s*\d+[\.\、\)]?\s*")
    with open(fp, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = pat.sub("", s)  # 去除行首編號
            out.append(s)
    return out


def load_prompt(prompt_path: str | Path) -> str:
    """
    載入 system prompt。若預設路徑不存在，回退到 /mnt/data/sentence-triple-extraction.txt。
    """
    p = Path(prompt_path)
    if p.is_file():
        return p.read_text(encoding="utf-8-sig")
    fb = Path(FALLBACK_PROMPT)
    if fb.is_file():
        return fb.read_text(encoding="utf-8-sig")
    raise FileNotFoundError(
        f"找不到提示檔：{prompt_path}，且後備檔案 {FALLBACK_PROMPT} 也不存在。"
    )


def build_user_message(sentences: List[str]) -> str:
    """
    將 10 條短句編號為 Q1..Q10，組成給 LLM 的 user content。
    """
    lines = ["請依據下列 10 條短句進行三元組抽取；每行視為獨立問句，問句序為 Q1..Q10：", ""]
    for i, s in enumerate(sentences, start=1):
        lines.append(f"Q{i}: {s}")
    lines.append("")
    lines.append("請嚴格依照你收到的系統規則輸出唯一一個 JSON 物件（只含 {\"triples\": [...] }）。")
    return "\n".join(lines)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    從 LLM 回應中擷取第一個 JSON 物件，容忍 ```json ... ``` 包裹與前後雜訊。
    """
    # 去除 Markdown code fence
    fence = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S | re.I)
    m = fence.search(text)
    if m:
        text = m.group(1)

    # 從第一個 { 到最後一個 } 之間擷取
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start: end + 1]
    else:
        candidate = text.strip()

    # 嘗試直接 parse
    try:
        return json.loads(candidate)
    except Exception:
        # 移除尾端多餘逗號等常見噪音後再試
        candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(candidate2)


def validate_triples_schema(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    確認最外層包含 "triples" 且為 list；僅做輕度驗證，不深入檢查每一鍵。
    """
    if not isinstance(obj, dict) or "triples" not in obj:
        raise ValueError("LLM 回傳的 JSON 缺少 'triples' 欄位。")
    triples = obj["triples"]
    if not isinstance(triples, list):
        raise ValueError("'triples' 應為陣列。")
    return triples


def call_llm(
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
) -> str:
    """
    呼叫 OpenAI Chat Completions，回傳文字。
    """
    if _USE_NEW_SDK:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    else:
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(  # type: ignore
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        return resp["choices"][0]["message"]["content"]  # type: ignore


def load_api_key() -> str:
    """
    依序嘗試：
    1) .env -> GPT_API
    2) 環境變數 GPT_API
    3) 環境變數 OPENAI_API_KEY
    """
    if load_dotenv is not None:
        # 嘗試在專案根目錄與當前目錄載入 .env
        for root_candidate in [Path("."), Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) >= 3 else Path(".")]:
            env_path = root_candidate / ".env"
            if env_path.is_file():
                load_dotenv(dotenv_path=str(env_path))
                break
        else:
            load_dotenv()

    api_key = os.getenv("GPT_API") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "找不到 API Key。請在 .env 設定 GPT_API=... 或設定環境變數 OPENAI_API_KEY。"
        )
    return api_key


def main():
    parser = argparse.ArgumentParser(
        description="Extract triples from 10 short sentences.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="輸入檔（每行一條短句）")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="系統提示檔案路徑")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="輸出 JSON 檔案路徑")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="模型名稱（預設 gpt-4o-mini）")
    args = parser.parse_args()

    input_path = Path(args.input)
    prompt_path = Path(args.prompt)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"找不到輸入檔：{input_path}")

    sentences = read_lines_strip_numbering(input_path)
    if not sentences:
        raise ValueError(f"輸入檔無內容：{input_path}")
    if len(sentences) != 10:
        # 不強制，但提示
        print(f"[警告] 預期 10 條，實得 {len(sentences)} 條。仍將全部送交抽取。")

    system_prompt = load_prompt(prompt_path)
    user_message = build_user_message(sentences)
    api_key = load_api_key()

    print("📤 呼叫 LLM 進行三元組抽取（模型：%s）..." % args.model)
    raw_text = call_llm(api_key=api_key, model=args.model,
                        system_prompt=system_prompt, user_message=user_message)

    print("🧩 嘗試解析 JSON ...")
    obj = extract_json_object(raw_text)
    triples = validate_triples_schema(obj)


    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump({"triples": triples}, f, ensure_ascii=False, indent=2)

    print(f"✅ 已輸出：{output_path.resolve()}")
    print(f"📝 句子數：{len(sentences)}，三元組數：{len(triples)}")


if __name__ == "__main__":
    main()
