"""
合併多路檢索結果並去重

用法（工具）：
  merge_and_dedup(payload: str) -> str
  - payload: JSON 字串，格式為：
      {
        "pg": {"lines": [...], "items": [...]},
        "vector": {"lines": [...], "items": [...]},
        "neo4j": {"lines": [...], "items": [...]}
      }
  - 回傳：JSON 字串 {"lines": [...]}

說明：
  1) 若三路皆無 items：退回到把各路 lines 合併後保序去重。
  2) 若任一路有 items：將所有 items 攤平渲染為 [比對] 行，再保序去重。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv(override=True)

from ..common.config import MAX_EVID_CHARS, _dlog
from ..common.formatting import norm_triple_dict  # 與 agent_langchain 同名函式語義一致


# ---- 與 agent_langchain.py 相同的關係名正規化 ----
import os
import importlib.util
from pathlib import Path

_RELATION_DICT_PATH = (
    Path(__file__).resolve().parents[5]
    / "data"
    / "processed"
    / "knowledge-graph"
    / "relation_dict_all.py"
)

def _load_relation_dict() -> set:
    try:
        spec = importlib.util.spec_from_file_location("relation_dict_all", _RELATION_DICT_PATH)
        module = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore
        rels = getattr(module, "RELATIONS_ALL", set())
        return set(rels) if rels else set()
    except Exception:
        return set()

RELATIONS_ALL = _load_relation_dict()

import re as _re
_RE_PAREN_FULL = _re.compile(r"（.*?）")
_RE_PAREN_HALF = _re.compile(r"\(.*?\)")
_RE_SPLIT_CAND = _re.compile(r"[／/、,，;；\s]+")

def _normalize_relation(rel: str, whitelist: set | None = None) -> str:
    if not isinstance(rel, str):
        return "未知"
    s = rel.strip()
    if not s:
        return "未知"
    paren_chunks: List[str] = []
    for m in _RE_PAREN_FULL.findall(s):
        paren_chunks.append(m.strip("（）"))
    for m in _RE_PAREN_HALF.findall(s):
        paren_chunks.append(m.strip("()"))
    base = _RE_PAREN_FULL.sub("", s)
    base = _RE_PAREN_HALF.sub("", base)
    base = re.sub(r"[，、。；;:：\s]+", "", base)
    candidates: List[str] = []
    for chunk in paren_chunks:
        for tok in _RE_SPLIT_CAND.split(chunk):
            tok = tok.strip()
            if tok:
                candidates.append(tok)
    if base:
        candidates.append(base)
    wl = whitelist or set()
    for cand in candidates:
        if cand in wl:
            return cand
    if base:
        return base
    return s


def _unwrap_props(d: dict) -> dict:
    """若為雙層 props，回傳最內層；否則原樣（與 agent_langchain 同步）。"""
    if isinstance(d, dict) and "props" in d and isinstance(d["props"], dict):
        return d["props"]
    return d if isinstance(d, dict) else {}


def _format_kb_line(tri: dict, det: dict, max_evid: int) -> str:
    """輸出單條比對行（含 type/屬性/關係名/說明/事件時間）——與 agent_langchain.py 完全一致。"""
    h = _unwrap_props(det.get("head", {}) or {})
    t = _unwrap_props(det.get("tail", {}) or {})
    r = _unwrap_props(det.get("rel", {}) or {})

    h_name = tri.get("head", "") or h.get("name", "") or ""
    t_name = tri.get("tail", "") or t.get("name", "") or ""
    rel_name = (
        tri.get("relation")
        or r.get("relation")
        or r.get("name")
        or r.get("label")
        or r.get("關係")
        or r.get("type")
        or r.get("relation_name")
        or r.get("rel")
        or r.get("關係名")
        or ""
    )
    rel_name = str(rel_name).strip()
    try:
        rel_name = _normalize_relation(rel_name, RELATIONS_ALL)
    except Exception:
        rel_name = _normalize_relation(rel_name, None)
    if not rel_name:
        rel_name = "提及"

    def _fmt_attrs(d: dict) -> str:
        if not isinstance(d, dict) or not d:
            return ""
        items = [f"{k}:{v}" for k, v in d.items() if v not in (None, "", [])]
        return "；".join(items)

    h_type = h.get("type")
    t_type = t.get("type")
    h_attrs = _fmt_attrs({k: v for k, v in h.items() if k not in ("name", "type")})
    t_attrs = _fmt_attrs({k: v for k, v in t.items() if k not in ("name", "type")})

    date = r.get("date") or r.get("事件時間") or ""
    evi = r.get("evidence") or r.get("desc") or ""
    if isinstance(evi, str) and max_evid > 0 and len(evi) > max_evid:
        evi = evi[:max_evid] + "…"

    h_block = h_name
    det_parts: List[str] = []
    if h_type:
        det_parts.append(f"type:{h_type}")
    if h_attrs:
        det_parts.append(h_attrs)
    if det_parts:
        h_block += f"（{'；'.join(det_parts)}）"

    t_block = t_name
    det_parts = []
    if t_type:
        det_parts.append(f"type:{t_type}")
    if t_attrs:
        det_parts.append(t_attrs)
    if det_parts:
        t_block += f"（{'；'.join(det_parts)}）"

    line = f"[比對] {h_block} 透過關係與 {t_block} 建立連結，說明：{evi}"
    if date:
        line += f"；事件時間：{date}"
    line += "。"
    return line


@tool(
    "merge_and_dedup",
    return_direct=False,
    description=("合併多路檢索結果並去重。輸入 JSON 含 'pg'、'neo4j'、'vector'（各自含 'lines' / 'items'）；輸出 {'lines': [...]}。"),
)
def tool_merge_and_dedup(payload: str) -> str:
    """工具：合併 → 輸出 lines（行為與 agent_langchain.py 一致）。"""
    try:
        obj = json.loads(payload)
    except Exception:
        obj = {}

    def _get_items(key: str) -> List[dict]:
        node = obj.get(key) or {}
        arr = node.get("items") or []
        return list(arr) if isinstance(arr, list) else []

    runs = {
        "pg": _get_items("pg"),
        "vector": _get_items("vector"),
        "neo4j": _get_items("neo4j"),
    }

    # 情況 A：完全沒有 items -> 合併三路 lines 做保序去重
    if not any(runs.values()):
        lines_all: List[str] = []
        for k in ("pg", "neo4j", "vector"):
            node = obj.get(k) or {}
            arr = node.get("lines") or []
            if isinstance(arr, list):
                lines_all.extend([str(x) for x in arr if isinstance(x, (str, int, float))])
        # 保序去重
        kept: List[str] = []
        seen = set()
        for s in lines_all:
            s2 = str(s or "").strip()
            if not s2:
                continue
            if not s2.startswith("[比對]"):
                s2 = f"[比對] {s2}"
            if s2 in seen:
                continue
            seen.add(s2)
            kept.append(s2)
        _dlog(f"merge_and_dedup(fallback): in={len(lines_all)}, out={len(kept)}")
        return json.dumps({"lines": kept}, ensure_ascii=False)

    # 情況 B：有 items -> 逐一渲染
    all_items: List[dict] = []
    all_items.extend(runs["pg"])
    all_items.extend(runs["vector"])
    all_items.extend(runs["neo4j"])

    lines: List[str] = []
    for it in all_items:
        txt = str(it.get("text") or "")
        # 1) 試著解析 JSON（pg/neo4j）
        try:
            objj = json.loads(txt)
            tri = objj.get("tri") or objj.get("triple") or {}
            det = objj.get("det") or objj.get("detail") or {}
            s = _format_kb_line(
                norm_triple_dict(tri if isinstance(tri, dict) else {}),
                det if isinstance(det, dict) else {},
                max_evid=MAX_EVID_CHARS,
            )
        except Exception:
            # 2) 不是 JSON：視為向量或已成型的 [比對] 行
            s = txt if txt.startswith("[比對]") else f"[比對] {txt}"
        lines.append(s)

    # 保序去重
    kept: List[str] = []
    seen = set()
    for s in lines:
        s2 = str(s or "").strip()
        if not s2:
            continue
        if s2 in seen:
            continue
        seen.add(s2)
        kept.append(s2)

    return json.dumps({"lines": kept}, ensure_ascii=False)
