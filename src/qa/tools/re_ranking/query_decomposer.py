"""
Query Decomposer: 使用 gpt-4o-mini 將自然語言查詢拆分為多個子問題（Sub-Queries）。

設計目標:
    - 對每條查詢進行 3~8 條的語義子問題拆分（保留人名/機構/時間/地點/關係等約束）。
    - 產出穩定 JSON 結構，便於後續 CE/RRF 管線重用。
    - 符合 PEP8 與 Google Style docstring，維持 SRP/SoC。

相依:
    - openai>=1.0（官方 Python SDK）
    - 以環境變數 OPENAI_API_KEY 取得金鑰
    - 預設模型 gpt-4o-mini（可用 QD_MODEL 覆寫）

輸出:
    - queries.split.json: List[{"q": <原查詢>, "splits": [<子問題>...]}]
    - queries.split.txt:  單檔列出所有子問題（逐行）

參考:
    - OpenAI 模型與 API（gpt-4o-mini、Responses API）:contentReference[oaicite:1]{index=1}
    - 問題分解 / Self-Ask 與多步檢索思想 :contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple


try:
    # 官方 SDK（1.x）
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    OpenAI = None  # type: ignore


_JSON_FALLBACK_RE = re.compile(r"\[.*\]", flags=re.S | re.M)


@dataclass(frozen=True)
class QDConfig:
    """Query Decomposition 的參數集合。"""

    model: str = os.getenv("QD_MODEL", "gpt-4o-mini")
    max_splits: int = int(os.getenv("QD_MAX_SPLITS", "6"))
    min_splits: int = int(os.getenv("QD_MIN_SPLITS", "3"))
    style: str = os.getenv("QD_STYLE", "factual-tight")
    temperature: float = float(os.getenv("QD_TEMPERATURE", "0.2"))
    timeout: int = int(os.getenv("QD_TIMEOUT_SEC", "60"))


_SYSTEM_PROMPT = (
    "角色: 專業查詢分解器。\n"
    "目標: 將輸入查詢分解為多個可直接檢索的子問題，"
    "每條子問題需保留原查詢的重要約束（人名、機構、時間、地點、關係/事件）。\n"
    "輸出: 僅輸出 JSON 陣列，每個元素為一條子問題的短句（不超過 30 字）。\n"
    "風格: 客觀、可檢索、避免同義重複。\n"
)

_USER_PROMPT_TMPL = """\
請分解下列查詢為 {min_splits} 到 {max_splits} 條子問題，中文輸出。
僅輸出 JSON 陣列（例如: ["子問1","子問2"]），不要加註解或多餘文字。

查詢:
{query}
"""

def _build_client() -> Any:
    """建立 OpenAI 客戶端。"""
    if OpenAI is None:
        raise RuntimeError("openai 套件未安裝或版本不符。")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未設定 OPENAI_API_KEY。")
    return OpenAI(api_key=api_key)


def _call_llm(client: Any, cfg: QDConfig, query: str) -> List[str]:
    """呼叫 LLM 輸出 JSON 陣列的子問題列表。

    Args:
        client: OpenAI 客戶端。
        cfg: QDConfig。
        query: 原查詢文字。

    Returns:
        子問題列表（若解析失敗則回傳空陣列）。
    """
    # 使用 Responses API 的 messages 風格（向前相容）
    prompt = _USER_PROMPT_TMPL.format(
        min_splits=cfg.min_splits,
        max_splits=cfg.max_splits,
        query=query.strip(),
    )

    try:
        rsp = client.chat.completions.create(  # noqa: E501
            model=cfg.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=cfg.temperature,
            timeout=cfg.timeout,
        )
        text = rsp.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover
        text = ""

    text = (text or "").strip()
    # 期望是純 JSON；若前後有雜訊，嘗試用正則萃取第一段 JSON 陣列
    candidate = text
    if not (candidate.startswith("[") and candidate.endswith("]")):
        m = _JSON_FALLBACK_RE.search(text)
        if m:
            candidate = m.group(0)

    try:
        arr = json.loads(candidate)
        splits = [str(s).strip() for s in arr if str(s).strip()]
    except Exception:
        splits = []

    # 去重、裁切到上限
    seen = set()
    uniq: List[str] = []
    for s in splits:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
        if len(uniq) >= cfg.max_splits:
            break
    return uniq


def decompose_queries(
    queries: Iterable[str],
    out_dir: Path,
    cfg: QDConfig | None = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """對多條查詢進行拆分，並輸出兩個檔案。

    輸出:
        - queries.split.json: List[{"q": <原查詢>, "splits": [<子問題>...]}]
        - queries.split.txt:  逐行列出所有子問題（以 utf-8-sig 寫出）

    Args:
        queries: 多條原始查詢。
        out_dir: 輸出目錄（會自動建立）。
        cfg: 可選設定。

    Returns:
        pair: (records_json, all_splits)
    """
    cfg = cfg or QDConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    client = _build_client()

    records: List[Dict[str, Any]] = []
    all_splits: List[str] = []

    for q in queries:
        q = str(q).strip()
        if not q:
            continue
        subs = _call_llm(client, cfg, q)
        records.append({"q": q, "splits": subs})
        all_splits.extend(subs)

    with (out_dir / "queries.split.json").open("w", encoding="utf-8-sig") as f:
        f.write(json.dumps(records, ensure_ascii=False, indent=2))

    with (out_dir / "queries.split.txt").open("w", encoding="utf-8-sig") as f:
        for s in all_splits:
            f.write(s + "\n")

    return records, all_splits
