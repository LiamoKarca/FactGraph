"""
RAG Mode – 嚴格吃 annex、完整展開 clues，並加入向量庫「輪詢檢查」避免時間差。
策略：優先抓取「線上最新」向量庫（explicit > online）。若輪詢仍為空，回報「稍待幾分鐘後重試」並停止本次 RAG。

【本版重點改動】
1) 支援以環境變數覆寫「輸入/輸出」與命名：
   - RAG_USER_FILE：指定本次 job 的使用者原始輸入檔（優先於 auto_find_latest_user_txt）
   - RAG_OUT_DIR   ：指定輸出目錄；未提供時仍落在 PATHS.RAG_OUTDIR
   - RAG_JOB_ID    ：若存在，輸出檔名改為「{job_id}_{YYYYmmdd-HHMMSS}_rag.*」並另寫 {job_id}.md 以利除錯
2) 檔名時間戳提升到「秒」等級，避免覆蓋。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    PATHS, MODELS,
    make_openai_client,
    read_text, write_text,
    auto_find_latest_user_txt,
    auto_find_annex,
    classify_kind
)
from openai import OpenAI  # 型別提示

# ──────────────────────────────────────────────────────────────────────────────
# Vector Store utilities（線上最新優先）
# ──────────────────────────────────────────────────────────────────────────────


def _list_all_vector_stores(client: OpenAI, limit: int = 100) -> List[Any]:
    items: List[Any] = []
    after: Optional[str] = None
    while True:
        page = client.vector_stores.list(
            limit=limit, after=after) if after else client.vector_stores.list(limit=limit)
        data = getattr(page, "data", []) or []
        items.extend(data)
        if not getattr(page, "has_more", False):
            break
        after = data[-1].id if data else None
        if not after:
            break
    return items


def _pick_latest_online_vector_store_id(client: OpenAI) -> str:
    items = _list_all_vector_stores(client)
    if not items:
        raise RuntimeError("你的帳號下沒有任何 Vector Store（RAG Storage）。")
    items.sort(key=lambda x: getattr(x, "created_at", 0), reverse=True)
    latest = items[0]
    vs_id = getattr(latest, "id", "") or ""
    if not vs_id:
        raise RuntimeError("最新 Vector Store 沒有有效的 id。")
    return vs_id


def _vector_store_file_count(client: OpenAI, vs_id: str) -> int:
    """以 list API 取檔案數；必要時退回 retrieve().file_counts"""
    try:
        page = client.vector_stores.files.list(vector_store_id=vs_id, limit=1)
        return len(page.data)
    except Exception:
        vs = client.vector_stores.retrieve(vs_id)
        fc = getattr(vs, "file_counts", None) or getattr(
            vs, "stats", None) or {}
        return int(fc.get("total", 0)) if isinstance(fc, dict) else 0


def _wait_nonempty_vector_store(client: OpenAI, vs_id: str, retries: int = 8, delay_sec: float = 2.5) -> bool:
    """
    輪詢向量庫是否已有檔案（避免剛上傳的時間差）。
    回傳 True 表示已偵測到 >=1 檔；False 表示仍為 0。
    """
    for i in range(retries + 1):
        cnt = _vector_store_file_count(client, vs_id)
        if cnt > 0:
            return True
        if i < retries:
            print(f"[INFO] VS {vs_id} 暫無檔案（total=0），等待索引中… 重試 {i+1}/{retries}")
            time.sleep(delay_sec)
    return False

# ──────────────────────────────────────────────────────────────────────────────
# Annex 讀取與 user prompt 組裝（嚴格）
# ──────────────────────────────────────────────────────────────────────────────
# --- 優先找本 job 的 annex（環境變數 → job 目錄 → 全域） ----------------------
from pathlib import Path

def _auto_find_annex_job_first():
    env_annex = os.getenv("RAG_ANNEX_FILE")
    if env_annex:
        p = Path(env_annex)
        if p.exists():
            return p

    job_dir = os.getenv("RAG_JOB_DIR")
    if job_dir:
        annex_dir = Path(job_dir) / "annex"
        if annex_dir.exists():
            cands = sorted(annex_dir.glob("*_annex.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if cands:
                return cands[0]

    # 後備：沿用既有的全域尋檔策略
    return auto_find_annex()

def _load_annex_strict(p: Path) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    raw = read_text(p)
    try:
        obj = json.loads(raw)
    except Exception as e:
        raise ValueError(f"annex 不是合法 JSON：{p} ({e})")

    user_q = ""
    clues: Dict[str, Any] = {}
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and "user_question" in obj[0]:
            user_q = str(obj[0].get("user_question", "")).strip()
        if len(obj) >= 2 and isinstance(obj[1], dict):
            clues = obj[1]
    elif isinstance(obj, dict):
        user_q = str(obj.get("user_question", "")).strip()
        for k in ("entities", "events", "dates", "numbers", "focus"):
            if k in obj:
                clues[k] = obj[k]
    else:
        user_q = str(obj)

    if not user_q:
        raise ValueError("annex 缺少 user_question 內容。")

    def as_list(x):
        return x if isinstance(x, list) else ([] if x in (None, "") else [x])

    entities = as_list(clues.get("entities"))
    events = as_list(clues.get("events"))
    dates = as_list(clues.get("dates"))
    numbers = as_list(clues.get("numbers"))
    focus = as_list(clues.get("focus"))

    norm_clues = {
        "entities": entities,
        "events": events,
        "dates": dates,
        "numbers": numbers,
        "focus": focus,
    }

    stats = {
        "len_user_question": len(user_q),
        "n_entities": len(entities),
        "n_events": len(events),
        "n_dates": len(dates),
        "n_numbers": len(numbers),
        "n_focus": len(focus),
    }
    return user_q, norm_clues, stats


def _render_user_prompt(user_q: str, clues: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("[使用者提問]")
    lines.append(user_q.strip())
    lines.append("")

    def _join_list(label: str, arr: List[Any]) -> None:
        if not arr:
            return
        items: List[str] = []
        for x in arr:
            if isinstance(x, dict):
                if "claim" in x:
                    items.append(str(x.get("claim", "")).strip())
                elif "name" in x:
                    name = str(x.get("name", "")).strip()
                    aliases = x.get("aliases") or []
                    alias_txt = ""
                    if isinstance(aliases, list) and aliases:
                        alias_txt = "（別名：" + \
                            "、".join([str(a).strip()
                                     for a in aliases if a]) + "）"
                    items.append(name + alias_txt if name else "")
                else:
                    items.append(json.dumps(x, ensure_ascii=False))
            else:
                items.append(str(x).strip())
        items = [s for s in items if s]
        if not items:
            return
        lines.append(f"[{label}]")
        for s in items:
            lines.append(f"- {s}")
        lines.append("")

    _join_list("entities", clues.get("entities", []))
    _join_list("events",   clues.get("events", []))
    _join_list("dates",    clues.get("dates", []))
    _join_list("numbers",  clues.get("numbers", []))
    _join_list("focus",    clues.get("focus", []))

    return "\n".join(lines).strip()


def _build_seed_queries(clues: Dict[str, Any], max_queries: int = 15) -> List[str]:
    seeds: List[str] = []
    for x in clues.get("focus", []) or []:
        if isinstance(x, dict) and x.get("claim"):
            seeds.append(str(x["claim"]).strip())
    for x in clues.get("entities", []) or []:
        if isinstance(x, dict) and x.get("name"):
            seeds.append(str(x["name"]).strip())
            for a in (x.get("aliases") or []):
                seeds.append(str(a).strip())
    for k in ("events", "dates", "numbers"):
        for v in clues.get(k, []) or []:
            seeds.append(str(v).strip())
    dedup = []
    seen = set()
    for s in seeds:
        s = s.replace("「", "").replace("」", "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        dedup.append(s)
    return dedup[:max_queries]

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG Mode – 使用 Responses + File Search（嚴格 annex 版本）")
    p.add_argument("--vector-store-id", default=None,
                   help="指定 Vector Store ID（可選）")
    p.add_argument("--vs-retries", type=int,
                   default=8, help="VS 檢查重試次數（預設 8 次）")
    p.add_argument("--vs-delay", type=float, default=2.5,
                   help="VS 檢查每次延遲秒數（預設 2.5s）")
    return p.parse_args()


def _resolve_vector_store_id(client: OpenAI, explicit_id: Optional[str]) -> Tuple[str, str]:
    if explicit_id:
        return explicit_id.strip(), "explicit"
    # 只用「線上最新」
    return _pick_latest_online_vector_store_id(client), "online"

# ──────────────────────────────────────────────────────────────────────────────
# metadata 字串化
# ──────────────────────────────────────────────────────────────────────────────


def _meta_str(val: Any, *, max_len: int = 1800) -> str:
    s = json.dumps(val, ensure_ascii=False) if isinstance(
        val, (dict, list)) else str(val)
    if len(s) > max_len:
        s = s[:max_len] + f"...(truncated {len(s)-max_len} chars)"
    return s

# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────


def _select_prompt_path(kind: str) -> Path:
    return PATHS.PROMPT_LONG if "長" in kind else PATHS.PROMPT_SHORT


def _extract_output_text(resp) -> str:
    if not hasattr(resp, "output_text"):
        raise AttributeError("此 SDK 版本不支援 resp.output_text，請升級 openai。")
    return (resp.output_text or "").strip()


def _save_outputs(answer_text: str, meta: Dict[str, Any]) -> None:
    """
    儲存輸出檔案：
    1) 優先使用環境變數 RAG_OUT_DIR 作為輸出目錄（job-scoped）
    2) 檔名改到「秒」，並加上 job_id 前綴：{job_id}_{YYYYmmdd-HHMMSS}_rag.*
    3) 另寫一份穩定檔名 {job_id}.md 以利除錯（若 job_id 存在）
    4) 若無環境變數，退回舊邏輯（寫到 PATHS.RAG_OUTDIR；檔名仍為「秒」以避免覆蓋）
    """
    out_dir = Path(os.getenv("RAG_OUT_DIR") or PATHS.RAG_OUTDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    job = os.getenv("RAG_JOB_ID", "").strip()

    if job:
        out_json = out_dir / f"{job}_{ts}_rag.json"
        out_md = out_dir / f"{job}_{ts}_rag.md"
        latest_md = out_dir / f"{job}.md"
    else:
        out_json = out_dir / f"{ts}_rag.json"
        out_md = out_dir / f"{ts}_rag.md"
        latest_md = None

    payload = {"answer": answer_text, "meta": meta}
    write_text(out_json, json.dumps(payload, ensure_ascii=False, indent=2))
    write_text(out_md, answer_text)
    if latest_md:
        try:
            write_text(latest_md, answer_text)
        except Exception:
            # latest 寫入失敗不阻斷主流程
            pass

    print(f"[Save] {out_json}")
    print(f"[Save] {out_md}")
    if latest_md:
        print(f"[Save] {latest_md} (latest pointer)")


def main() -> None:
    args = parse_args()
    client = make_openai_client()

    # 1) 讀 user 文本（優先 job-scoped：RAG_USER_FILE）
    env_user_file = os.getenv("RAG_USER_FILE")
    if env_user_file:
        user_txt_path = Path(env_user_file)
        if not user_txt_path.exists():
            raise FileNotFoundError(f"RAG_USER_FILE 指定的檔案不存在：{user_txt_path}")
        print(f"▶ 使用 job-scoped user input：{user_txt_path}")
    else:
        user_txt_path = auto_find_latest_user_txt()
        print(f"▶ 使用全域最新 user input：{user_txt_path}")

    user_txt = read_text(user_txt_path)

    # 2) 分類決定 Prompt
    kind, reason = classify_kind(client, MODELS.rag_model, user_txt)
    prompt_path = _select_prompt_path(kind)
    if not prompt_path.exists():
        raise FileNotFoundError(f"找不到對應 Prompt：{prompt_path}")
    sys_prompt = read_text(prompt_path)

    # 3) annex → 完整 user prompt + 檢索提示（嚴格）
    annex_path = _auto_find_annex_job_first()
    user_question, clues, stats = _load_annex_strict(annex_path)
    user_prompt = _render_user_prompt(user_question, clues)
    seed_queries = _build_seed_queries(clues, max_queries=15)
    if seed_queries:
        user_prompt += "\n\n[檢索提示]\n" + \
            "\n".join(f"- {q}" for q in seed_queries)

    # 4) 取得 vector_store_id（線上最新）並輪詢是否有檔案
    vector_store_id, vs_source = _resolve_vector_store_id(
        client, args.vector_store_id)
    if not _wait_nonempty_vector_store(client, vector_store_id, retries=args.vs_retries, delay_sec=args.vs_delay):
        raise SystemExit(f"剛更新完向量庫（{vector_store_id}），索引仍在進行中；請稍待幾分鐘後重試。")

    # 5) Responses + file_search
    resp = client.responses.create(
        model=MODELS.rag_model,
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
        metadata={
            "mode": "rag_mode",
            "question_kind": _meta_str(kind),
            "vector_store_source": _meta_str(vs_source),
            "annex_stats": _meta_str(stats),
            "user_preview": _meta_str(user_question[:200]),
            "seed_queries": _meta_str(seed_queries),
            "job_id": os.getenv("RAG_JOB_ID", "").strip(),
        },
    )

    answer_text = _extract_output_text(resp)
    if not answer_text:
        raise RuntimeError("Responses 成功但未取得 output_text")

    # 6) 輸出
    meta: Dict[str, Any] = {
        "kind": kind,
        "reason": reason,
        "annex_file": str(annex_path),
        "user_input_file": str(user_txt_path),
        "vector_store_id": str(vector_store_id),
        "vector_store_source": str(vs_source),
        "sdk_path": "tools[0].vector_store_ids",
        "annex_stats": stats,
        "user_question_preview": user_question[:200],
        "user_prompt_len": len(user_prompt),
        "seed_queries": seed_queries,
        "job_id": os.getenv("RAG_JOB_ID", "").strip(),
        "out_dir": str(Path(os.getenv("RAG_OUT_DIR") or PATHS.RAG_OUTDIR)),
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_outputs(answer_text, meta)


if __name__ == "__main__":
    main()
