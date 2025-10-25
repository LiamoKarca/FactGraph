"""
三元組正規化、關係詞規範、比對行渲染與去重。

自我測試：
python -m src.qa.verifier.agent.common.formatting
"""

from __future__ import annotations

import importlib.util
import json
import re
import os  # 為了自我測試中的 os.getenv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv
load_dotenv(override=True)
from ...core.dedup import deduplicate
from ..common.types import RankItem, make_rank_id
from ..common.config import MAX_EVID_CHARS

# 允許的鍵名映射
_TRIPLE_KEY_CANDIDATES: Dict[str, List[str]] = {
    "head": ["head", "h", "source", "src", "s", "from", "subject"],
    "relation": ["relation", "rel", "r", "edge", "predicate", "type"],
    "tail": ["tail", "t", "target", "dst", "d", "to", "object", "value", "name"],
}

# 關係詞白名單（外部可選）
_RELATION_DICT_PATH = (
    Path(__file__).resolve().parents[5]
    / "data"
    / "processed"
    / "knowledge-graph"
    / "relation_dict_all.py"
)


def _load_relation_dict() -> set:
    try:
        spec = importlib.util.spec_from_file_location(
            "relation_dict_all", _RELATION_DICT_PATH
        )
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore
        rels = getattr(module, "RELATIONS_ALL", set())
        return set(rels) if rels else set()
    except Exception:
        return set()


RELATIONS_ALL = _load_relation_dict()

_RE_PAREN_FULL = re.compile(r"（.*?）")
_RE_PAREN_HALF = re.compile(r"\(.*?\)")
_RE_SPLIT_CAND = re.compile(r"[／/、,，;；\s]+")


def normalize_relation(rel: str, whitelist: Optional[Set[str]] = None) -> str:
    """正規化關係詞，優先命中白名單。

    Args:
        rel: 原始關係詞。
        whitelist: 白名單集合。

    Returns:
        正規化後的關係詞。
    """
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
    return base or s


def norm_triple_dict(tri: dict) -> dict:
    """將各種鍵名正規化為 {head, relation, tail}。"""
    out = {"head": "未知", "relation": "未知", "tail": "未知"}
    if not isinstance(tri, dict):
        return out
    for std_key, cands in _TRIPLE_KEY_CANDIDATES.items():
        for k in cands:
            if k in tri and isinstance(tri[k], str) and tri[k].strip():
                out[std_key] = tri[k].strip()
                break
    try:
        out["relation"] = normalize_relation(out.get("relation", "未知"), RELATIONS_ALL)
    except Exception:
        out["relation"] = normalize_relation(out.get("relation", "未知"), None)
    return out


def norm_hits(hits: Any) -> List[Tuple[Dict, Dict]]:
    """將命中結果轉為 (tri, det) 清單。"""
    normed: List[Tuple[Dict, Dict]] = []
    for h in hits or []:
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            tri, det = h[0], h[1]
        elif isinstance(h, dict):
            tri = h.get("triple") or h.get("tri") or h
            det = h.get("detail") or h.get("det") or h
        else:
            continue
        normed.append(
            (
                norm_triple_dict(tri if isinstance(tri, dict) else {}),
                det if isinstance(det, dict) else {},
            )
        )
    return normed


def fmt_attrs(d: dict) -> str:
    """將屬性 dict 轉為「k1:v1；k2:v2」字串。"""
    if not isinstance(d, dict) or not d:
        return ""
    items = [f"{k}:{v}" for k, v in d.items() if v not in (None, "", [])]
    return "；".join(items)


def unwrap_props(d: dict) -> dict:
    """若為雙層 props，回傳最內層；否則原樣。"""
    if isinstance(d, dict) and "props" in d and isinstance(d["props"], dict):
        return d["props"]
    return d if isinstance(d, dict) else {}


def format_kb_line(tri: dict, det: dict, max_evid: int = MAX_EVID_CHARS) -> str:
    """輸出單條比對行（含 type/屬性/關係名/說明/事件時間）。"""
    h = unwrap_props(det.get("head", {}) or {})
    t = unwrap_props(det.get("tail", {}) or {})
    r = unwrap_props(det.get("rel", {}) or {})

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
    rel_name = normalize_relation(str(rel_name).strip(), RELATIONS_ALL or None) or "提及"

    h_type = h.get("type")
    t_type = t.get("type")
    h_attrs = fmt_attrs({k: v for k, v in h.items() if k not in ("name", "type")})
    t_attrs = fmt_attrs({k: v for k, v in t.items() if k not in ("name", "type")})

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

    line = f"[比對] {h_block} 透過關係【{rel_name}】與 {t_block} 建立連結，說明：{evi}"
    if date:
        line += f"；事件時間：{date}"
    line += "。"
    return line


def collect_hits_to_lines(hits: List[Tuple[Dict, Dict]]) -> List[str]:
    """將檢索命中渲染為多行 `[比對]`。"""
    out: List[str] = []
    for tri, det in hits:
        tri = norm_triple_dict(tri)
        try:
            out.append(format_kb_line(tri, det, max_evid=MAX_EVID_CHARS))
        except Exception:
            # 後備格式
            h = (det.get("head", {}) or {}).get("name") or tri.get("head") or "未知"
            t = (det.get("tail", {}) or {}).get("name") or tri.get("tail") or "未知"
            r = (det.get("rel", {}) or {})
            ev = r.get("evidence", "") or r.get("desc", "") or ""
            if MAX_EVID_CHARS > 0 and isinstance(ev, str) and len(ev) > MAX_EVID_CHARS:
                ev = ev[:MAX_EVID_CHARS] + "…"
            rel_name = (
                (r.get("relation") or r.get("name") or r.get("label") or r.get("type") or "")
                .strip()
                or "提及"
            )
            line = f"[比對] {h} 透過關係【{rel_name}】與 {t} 建立連結，說明：{ev}。"
            out.append(line)
    return out


def items_from_hits(hits: List[Tuple[Dict, Dict]], src: str) -> List[RankItem]:
    """將 (tri, det) 命中轉成標準 RankItem 列表。"""
    items: List[RankItem] = []
    for i, (tri, det) in enumerate(hits, start=1):
        txt = json.dumps({"tri": tri, "det": det}, ensure_ascii=False)
        items.append(
            {
                "id": make_rank_id(txt, src),
                "text": txt,
                "source": src,
                "rank": i,
                "score": (
                    float(
                        (det.get("meta", {}) or {}).get("_debug", {}).get("cond_sum", 0.0)
                    )
                    if isinstance(det, dict)
                    else 0.0
                ),
                "payload": {"tri": tri, "det": det},
            }
        )
    return items


def deduplicate_kb_lines(lines: List[str]) -> List[str]:
    """比對行去重（保序）。"""
    return deduplicate(lines, triples=None)


def run_self_test() -> bool:
    """
    執行環境變數與依賴項自我檢測。

    檢查此模組依賴的環境變數是否已設定，以及外部檔案依賴是否存在。

    Returns:
        bool: 如果所有檢查都通過，則回傳 True，否則回傳 False。
    """
    print("=" * 30)
    print("Running Triples Module Self-Test...")
    print("=" * 30)

    all_ok = True

    # 1. 檢查從 config 匯入的環境變數
    # (假設 config.py 中的預設值是 "300")
    print("[Checking Environment Variable (via config.py dependency)]")
    var_name = "MAX_EVID_CHARS"
    value = os.getenv(var_name)

    if value is None:
        all_ok = False
        print(f"  [WARNING] '{var_name}' not set in .env. Using code default (via config).")
        print(f"            Current value (from config import): {MAX_EVID_CHARS}")
    else:
        print(f"  [OK] '{var_name}' is set to: '{value}'")
        print(f"       Current value (from config import): {MAX_EVID_CHARS}")
        if str(MAX_EVID_CHARS) != value:
            print(
                f"       [INFO] Value mismatch (config={MAX_EVID_CHARS}, env={value}). "
                "This might indicate an issue in 'config.py' or load order."
            )
            all_ok = False

    # 2. 檢查外部檔案依賴
    print("\n[Checking File Dependencies]")
    try:
        resolved_path = _RELATION_DICT_PATH.resolve()
        if resolved_path.exists():
            print(f"  [OK] Relation dictionary found at: {resolved_path}")
            if not RELATIONS_ALL:
                print(f"      [WARNING] File exists, but 'RELATIONS_ALL' list is empty.")
                all_ok = False
            else:
                print(f"      [INFO] Loaded {len(RELATIONS_ALL)} relations from whitelist.")
        else:
            print(f"  [ERROR] Relation dictionary NOT FOUND at: {resolved_path}")
            print(
                f"          'RELATIONS_ALL' is empty. Relation normalization will be degraded."
            )
            all_ok = False
    except Exception as e:
        print(f"  [ERROR] Failed to check path {_RELATION_DICT_PATH}: {e}")
        all_ok = False

    print("-" * 30)
    if all_ok:
        print("Self-Test Result: ALL CLEAR.")
    else:
        print("Self-Test Result: WARNINGS or ERRORS found.")
    print("=" * 30)

    return all_ok


if __name__ == "__main__":
    print("Module loaded directly. Running self-test...")
    print(f"Loading .env from: {Path('.env').resolve()}")
    run_self_test()