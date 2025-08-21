"""
功能簡介
--------
本程式結合 OpenAI Assistants API 與既有的向量檢索資料庫 (Vector Store)，
用於針對使用者輸入的問題進行 RAG (Retrieval-Augmented Generation) 查詢與回答。

主要流程
--------
1. 從 `data/interim/fast_mode/user-input/` 讀取最新的 `.txt` 檔，
   使用 GPT API 與 `src/qa/fast_mode/prompts/classify_content.txt` 判斷問題類型：
   - 「長篇問題」 → 使用 `prompts/long.txt` 指令
   - 「短問句」   → 使用 `prompts/short.txt` 指令

2. 從 `data/interim/fast_mode/keywords/` 取得最新的 `*_annex.json`，
   擷取 `user_question` 與輔助線索 (entities/events/dates/numbers/focus)，
   作為 Assistant 的 user message。

3. 從 `data/processed/news_merge/rag_storage_id/` 讀取最新 `yyyy-mm-dd-hhmm.txt`，
   取得既有的 Vector Store ID (不再上傳語料)。

4. 呼叫 OpenAI Assistants，綁定 `file_search` 工具與指定 Vector Store，
   根據指令檔 (long/short) 產生回答。

5. 輸出結果：
   - JSON：`data/processed/fast_mode/<annex_basename>_answer.json`
   - Markdown：`data/processed/fast_mode/<annex_basename>_answer.md`

設計特點
--------
- 全程使用 UTF-8-SIG 讀寫，確保跨平台兼容性。
- 自動挑選最新檔案，減少人工指定。
- 長篇問題回答會依照模板生成「查核明細 / 總結 / 可信度評分」。
- 短問句回答則專注於直接回答問題，仍附來源引用。

使用方式
--------
請先設定環境變數：
    export GPT_API=sk-xxxx 或 OPENAI_API=sk-xxxx
    export GPT_MODEL=gpt-4.1

執行：
    python src/qa/fast_mode/llm/gpt_rag.py

前置需求：
- `data/interim/fast_mode/user-input/*.txt` (至少一個問題輸入)
- `data/interim/fast_mode/keywords/*_annex.json` (問題與線索)
- `data/processed/news_merge/rag_storage_id/*.txt` (儲存的 vector_store_id)
- `src/qa/fast_mode/prompts/{long.txt, short.txt, classify_content.txt}`
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─────────── Paths ───────────
KEYWORDS_DIR = Path("data/interim/fast_mode/keywords")
USER_INPUT_DIR = Path("data/interim/fast_mode/user-input")
RAG_ID_DIR = Path("data/processed/news_merge/rag_storage_id")
PROMPT_LONG = Path("src/qa/fast_mode/prompts/long.txt")
PROMPT_SHORT = Path("src/qa/fast_mode/prompts/short.txt")
PROMPT_CLASSIFY = Path("src/qa/fast_mode/prompts/classify_content.txt")
OUTDIR = Path("data/processed/fast_mode")

# ─────────── IO helpers (utf-8-sig ONLY) ───────────


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig", errors="ignore")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8-sig")


def read_json(p: Path) -> Any:
    return json.loads(read_text(p))


def pick_latest(paths: List[Path]) -> Path:
    paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return paths[0]


def list_annex_files() -> List[Path]:
    return list(KEYWORDS_DIR.rglob("*_annex.json"))


def auto_find_annex() -> Path:
    files = list_annex_files()
    if not files:
        raise FileNotFoundError(f"找不到 annex 檔：{KEYWORDS_DIR}/**/*_annex.json")
    return pick_latest(files)


def auto_find_latest_user_txt() -> Path:
    files = list(USER_INPUT_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"找不到使用者輸入檔：{USER_INPUT_DIR}/*.txt")
    return pick_latest(files)


def get_latest_vector_store_id() -> str:
    files = list(RAG_ID_DIR.glob("*"))
    if not files:
        raise FileNotFoundError(f"找不到任何 RAG id 檔於 {RAG_ID_DIR}")
    latest = pick_latest(files)
    return read_text(latest).strip()

# ─────────── Classifier using external prompt ───────────


def classify_kind(client: OpenAI, model: str, user_text: str) -> Tuple[str, str]:
    """
    使用 classify_content.txt 內容作為提示，將 {user_text} 置換後送入 Responses API。
    回傳 (type, reason)。若提示未要求 reason，則回傳空字串。
    """
    if not PROMPT_CLASSIFY.exists():
        raise FileNotFoundError(f"找不到分類提示檔：{PROMPT_CLASSIFY}")
    prompt = read_text(PROMPT_CLASSIFY).replace("{user_text}", user_text)

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )

    # 雙路徑保底取得文字
    out = (getattr(resp, "output_text", "") or "").strip()
    if not out:
        try:
            out = (
                resp.output[0].content[0].text  # type: ignore[attr-defined]
                if getattr(resp, "output", None) else ""
            )
        except Exception:
            out = ""

    # 嘗試抓 JSON 區段
    js, je = out.find("{"), out.rfind("}")
    if js != -1 and je != -1 and je > js:
        try:
            obj = json.loads(out[js:je + 1])
            t = str(obj.get("type", "")).strip()
            r = str(obj.get("reason", "")).strip() if "reason" in obj else ""
            if t in ("長篇問題", "短問句"):
                return t, r
        except Exception:
            pass

    # 後備啟發式（長度）
    return ("長篇問題" if len(user_text) >= 120 else "短問句", "fallback by length heuristic")


def render_user_prompt_from_annex(annex_obj: Any) -> str:
    """
    將 annex 內容轉成「問題 + 輔助線索」給 GPT。
    支援格式：list[0] 為含 user_question 的 dict，list[1] 為 clues（entities/events/dates/numbers/focus）。
    若缺欄位會自動略過。
    """
    # 1) 取得 user_question
    user_q = ""
    if isinstance(annex_obj, list) and annex_obj:
        head = annex_obj[0]
        if isinstance(head, dict) and "user_question" in head:
            user_q = str(head.get("user_question", "")).strip()
    if not user_q and isinstance(annex_obj, dict):
        user_q = str(annex_obj.get("user_question", "")).strip()
    if not user_q and not isinstance(annex_obj, (list, dict)):
        user_q = str(annex_obj).strip()

    # 2) 嘗試讀取輔助線索（通常在 list[1]）
    clues = {}
    if isinstance(annex_obj, list) and len(annex_obj) >= 2 and isinstance(annex_obj[1], dict):
        clues = annex_obj[1]
    elif isinstance(annex_obj, dict):
        clues = annex_obj  # 平鋪的情況

    def _fmt_list(name: str, arr: Any, max_items: int = 30) -> str:
        if not isinstance(arr, list) or not arr:
            return ""
        items = []
        for x in arr[:max_items]:
            if isinstance(x, dict) and "claim" in x:  # for focus list
                items.append(str(x.get("claim", "")).strip())
            else:
                items.append(str(x).strip())
        return f"{name}: " + ", ".join([s for s in items if s]) + "\n"

    # 3) 組合「輔助線索」段落（僅輸出存在者）
    aux_lines = []
    aux_lines.append(_fmt_list("entities", clues.get("entities")))
    aux_lines.append(_fmt_list("events", clues.get("events")))
    aux_lines.append(_fmt_list("dates", clues.get("dates")))
    aux_lines.append(_fmt_list("numbers", clues.get("numbers")))

    # focus 以條列顯示，便於模型理解「關注主張」
    focus_list = clues.get("focus")
    if isinstance(focus_list, list) and focus_list:
        claims = [str(x.get("claim", "")).strip()
                  for x in focus_list if isinstance(x, dict) and x.get("claim")]
        if claims:
            aux_lines.append("focus:\n- " + "\n- ".join(claims) + "\n")

    aux_text = "".join([x for x in aux_lines if x]).strip()
    if not aux_text:
        aux_text = "(無)"

    # 4) 回傳最終 user message
    return f"[使用者提問]\n{user_q}\n\n[輔助線索]\n{aux_text}\n"

# ─────────── Assistants（既有 Vector Store，不上傳） ───────────


def assistants_rag(
    client: OpenAI,
    model: str,
    vector_store_id: str,
    assistant_instructions: str,
    user_prompt: str,
) -> Dict[str, Any]:
    assistant = client.beta.assistants.create(
        name="FastMode RAG Assistant",
        model=model,
        instructions=assistant_instructions,   # 僅使用 long/short 檔案內容
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {
            "vector_store_ids": [vector_store_id]}},
    )

    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_prompt)
    run = client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=assistant.id)

    # 輪詢加上超時，避免意外卡死
    t0 = time.time()
    max_wait = 180  # 秒
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id, run_id=run.id)
        if run.status in ("completed", "failed", "cancelled", "expired"):
            break
        if time.time() - t0 > max_wait:
            break
        time.sleep(0.8)

    msgs = client.beta.threads.messages.list(thread_id=thread.id).data
    msgs.sort(key=lambda m: getattr(m, "created_at", 0))
    last = next((m for m in reversed(msgs) if getattr(
        m, "role", "") == "assistant"), None)

    def _extract_text(msg) -> str:
        if not msg:
            return ""
        out = []
        for c in msg.content:
            if getattr(c, "type", "") == "text":
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    out.append(t)
                else:
                    # 支援新版 text 結構（含 annotations）
                    val = getattr(t, "value", "") if t else ""
                    if not val and hasattr(t, "annotations"):
                        val = getattr(t, "value", "")
                    out.append(val)
        return "\n".join(x for x in out if x)

    meta: Dict[str, Any] = {
        "status": getattr(run, "status", ""),
        "assistant_id": getattr(assistant, "id", None),
        "thread_id": getattr(thread, "id", None),
        "answer_text": _extract_text(last),
    }
    if getattr(run, "status", "") != "completed":
        meta["error"] = getattr(run, "last_error", None)
    return meta

# ─────────── MAIN ───────────


def main():
    api_key = os.getenv("GPT_API") or os.getenv("OPENAI_API")
    model = os.getenv("GPT_MODEL", "gpt-4.1")
    if not api_key:
        raise SystemExit("請設定 GPT_API 或 OPENAI_API")
    client = OpenAI(api_key=api_key)

    # 1) 讀最新 user-input 的 .txt → 分類
    latest_user_txt = auto_find_latest_user_txt()
    user_raw = read_text(latest_user_txt)
    qtype, qreason = classify_kind(client, model, user_raw)

    # 2) 載入對應的 assistant 指令檔
    if qtype == "長篇問題":
        if not PROMPT_LONG.exists():
            raise FileNotFoundError(f"找不到長篇提示檔：{PROMPT_LONG}")
        assistant_instructions = read_text(PROMPT_LONG)
    else:
        if not PROMPT_SHORT.exists():
            raise FileNotFoundError(f"找不到短問句提示檔：{PROMPT_SHORT}")
        assistant_instructions = read_text(PROMPT_SHORT)

    # 3) 讀 annex 作為「問題本體」（同時附上輔助線索）
    annex_file = auto_find_annex()
    annex_obj = read_json(annex_file)
    user_prompt = render_user_prompt_from_annex(annex_obj)

    # 4) 讀最新 vector_store_id
    vector_store_id = get_latest_vector_store_id()

    # 5) 呼叫 Assistants（不上傳語料）
    print(f"[Classify] type={qtype} reason={qreason}")
    print(
        f"[RAG] vector_store_id={vector_store_id} | annex={annex_file.name} | user_txt={latest_user_txt.name}")
    rag = assistants_rag(client, model, vector_store_id,
                         assistant_instructions, user_prompt)

    # 6) 輸出
    out_json = OUTDIR / f"{annex_file.stem}_answer.json"
    out_md = OUTDIR / f"{annex_file.stem}_answer.md"
    result = {
        "type": qtype,
        "type_reason": qreason,
        "question_annex": str(annex_file),
        "user_input_txt": str(latest_user_txt),
        "model": model,
        "vector_store_id": vector_store_id,
        "status": rag["status"],
        "error": rag.get("error"),
        "answer": rag["answer_text"],
    }
    write_text(out_json, json.dumps(result, ensure_ascii=False, indent=2))
    write_text(out_md, result["answer"])

    print(f"[Save] {out_json}")
    print(f"[Save] {out_md}")


if __name__ == "__main__":
    main()
