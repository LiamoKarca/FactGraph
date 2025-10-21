"""
CSV → Property Graph 建置模組（建置/別名/健康檢查）
提供：
    - CsvPropertyGraph（資料結構/序列化）
    - build_from_csv() / build_from_csv_and_save()
    - load_json() / save_json() / ensure_built_json()
    - CLI：build | ensure | doctor | alias
執行：
python -m src.qa.preliminary_work.build_csv_property_graph build data/raw/knowledge-graph/neo4j-kg-raw-graph.csv
python -m src.qa.preliminary_work.build_csv_property_graph ensure
python -m src.qa.preliminary_work.build_csv_property_graph doctor
python -m src.qa.preliminary_work.build_csv_property_graph alias
"""
from __future__ import annotations
import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import sys
import time
import random
import tempfile
import traceback

# 索引預設路徑
CACHE_DIR = Path("data/processed/knowledge-graph")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
JSON_CACHE_PATH = Path(
    os.getenv("LI_PG_INDEX_JSON", str(CACHE_DIR / "pg_index.json")))
RAW_CSV_DEFAULT = Path(
    os.getenv("LI_PG_RAW_CSV", "data/raw/knowledge-graph/neo4j-kg-raw-graph.csv"))


@dataclass
class Node:
    id: str
    name: str = ""
    type: str = ""
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    head: str
    relation: str
    tail: str
    rel_props: Dict[str, Any] = field(default_factory=dict)
    head_props: Dict[str, Any] = field(default_factory=dict)
    tail_props: Dict[str, Any] = field(default_factory=dict)


class CsvPropertyGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []

    def is_valid(self) -> bool:
        if not isinstance(self.nodes, dict) or not isinstance(self.edges, list):
            return False
        if not self.edges:
            return False
        e = self.edges[0]
        return isinstance(e, Edge) and isinstance(e.head, str) and isinstance(e.relation, str) and isinstance(e.tail, str)

    def to_json_obj(self) -> Dict[str, Any]:
        return {
            "nodes": {
                nid: {"id": n.id, "name": n.name,
                      "type": n.type, "props": n.props}
                for nid, n in self.nodes.items()
            },
            "edges": [
                {
                    "head": e.head,
                    "relation": e.relation,
                    "tail": e.tail,
                    "rel_props": e.rel_props,
                    "head_props": e.head_props,
                    "tail_props": e.tail_props,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_json_obj(cls, obj: Dict[str, Any]) -> "CsvPropertyGraph":
        g = cls()
        for nid, nd in (obj.get("nodes") or {}).items():
            g.nodes[str(nid)] = Node(
                id=str(nd.get("id", nid)),
                name=str(nd.get("name") or ""),
                type=str(nd.get("type") or ""),
                props=dict(nd.get("props") or {}),
            )
        for ed in obj.get("edges", []) or []:
            g.edges.append(
                Edge(
                    head=str(ed.get("head", "")),
                    relation=str(ed.get("relation", "")),
                    tail=str(ed.get("tail", "")),
                    rel_props=dict(ed.get("rel_props") or {}),
                    head_props=dict(ed.get("head_props") or {}),
                    tail_props=dict(ed.get("tail_props") or {}),
                )
            )
        return g

    def save_json(self, path: Path = JSON_CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig") as f:
            json.dump(self.to_json_obj(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: Path = JSON_CACHE_PATH) -> "CsvPropertyGraph":
        with path.open("r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        g = cls.from_json_obj(obj)
        if not g.is_valid():
            raise ValueError("❌ JSON 索引存在但內容異常（無有效邊或結構錯誤）。")
        return g

    @staticmethod
    def _decode_cell(v: Any) -> Any:
        s = (str(v) if v is not None else "").strip()
        if not s:
            return s
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return s
        return s

    @classmethod
    def build_from_csv(
        cls,
        csv_path: Path,
        *,
        head_col: str = "head",
        rel_col: str = "relation",
        tail_col: str = "tail",
        head_alias: Iterable[str] = ("h", "source", "src", "from"),
        rel_alias: Iterable[str] = ("rel", "r", "type", "edge"),
        tail_alias: Iterable[str] = ("t", "target", "dst", "to", "object"),
    ) -> "CsvPropertyGraph":
        def _pick(row: Dict[str, Any], key: str, alias: Iterable[str]) -> str:
            if key in row and row[key]:
                return str(row[key]).strip()
            for k in alias:
                if k in row and row[k]:
                    return str(row[k]).strip()
            return ""

        def _route_prop(k: str, v: Any) -> Tuple[str, str, Any]:
            k0 = k.strip()
            k1 = k0.lower()
            m = re.match(r"^(head|h)[._:-](.+)$", k1)
            if m:
                return ("head", m.group(2), cls._decode_cell(v))
            m = re.match(r"^(tail|t)[._:-](.+)$", k1)
            if m:
                return ("tail", m.group(2), cls._decode_cell(v))
            m = re.match(r"^(rel|relation|r)[._:-](.+)$", k1)
            if m:
                return ("rel", m.group(2), cls._decode_cell(v))
            return ("rel", k0, cls._decode_cell(v))

        g = cls()
        csv_path = csv_path.resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"❌ 找不到 CSV：{csv_path}")

        core_keys = {
            head_col.lower(), rel_col.lower(), tail_col.lower(),
            *[a.lower() for a in head_alias], *[a.lower()
                                                for a in rel_alias], *[a.lower() for a in tail_alias],
        }

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = _pick(row, head_col, head_alias)
                r = _pick(row, rel_col, rel_alias)
                t = _pick(row, tail_col, tail_alias)
                if not (h and r):  # tail 可為空
                    continue

                h_props: Dict[str, Any] = {}
                t_props: Dict[str, Any] = {}
                r_props: Dict[str, Any] = {}
                for k, v in row.items():
                    if k is None or k.lower() in core_keys:
                        continue
                    if v is None or str(v).strip() == "":
                        continue
                    side, kk, vv = _route_prop(k, v)
                    if side == "head":
                        h_props[kk] = vv
                    elif side == "tail":
                        t_props[kk] = vv
                    else:
                        r_props[kk] = vv

                if h not in g.nodes:
                    g.nodes[h] = Node(id=h, name=h, type=str(
                        h_props.get("type", "") or ""), props=h_props)
                else:
                    g.nodes[h].props.update(
                        {k: v for k, v in h_props.items() if k not in g.nodes[h].props})

                if t:
                    if t not in g.nodes:
                        g.nodes[t] = Node(id=t, name=t, type=str(
                            t_props.get("type", "") or ""), props=t_props)
                    else:
                        g.nodes[t].props.update(
                            {k: v for k, v in t_props.items() if k not in g.nodes[t].props})

                g.edges.append(Edge(head=h, relation=r, tail=t,
                               rel_props=r_props, head_props=h_props, tail_props=t_props))

        if not g.is_valid():
            raise ValueError("❌ 由 CSV 建表後內容異常（可能無有效邊）。")
        return g


def build_from_csv_and_save(csv_path: Path, *, idx_json: Path = JSON_CACHE_PATH) -> None:
    g = CsvPropertyGraph.build_from_csv(csv_path)
    g.save_json(idx_json)


def ensure_built_json(*, csv_path: Path | None = None, idx_json: Path = JSON_CACHE_PATH) -> CsvPropertyGraph:
    jsn = Path(idx_json)
    if jsn.is_file():
        return CsvPropertyGraph.load_json(jsn)
    src = Path(csv_path) if csv_path else RAW_CSV_DEFAULT
    g = CsvPropertyGraph.build_from_csv(src)
    g.save_json(jsn)
    return g

# ─────────────────────────
# 共用輔助：寫檔 / 分批 / 指數退避
# ─────────────────────────
def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8-sig") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _chunks(lst: List[Any], size: int):
    if not lst:
        return []
    size = max(1, int(size))
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

def _retry(call, *, max_tries: int = 5, base: float = 0.8, jitter: float = 0.3):
    for i in range(max_tries):
        try:
            return call()
        except Exception:
            if i == max_tries - 1:
                raise
            sleep_s = (base * (2**i)) + random.uniform(0, jitter)
            time.sleep(min(10.0, sleep_s))
# ─────────────────────────
# LLM 別名（alias）生成與合併
# ─────────────────────────
def _safe_int(v: Any, dv: int) -> int:
    try:
        return int(str(v))
    except Exception:
        return dv

def _llm_suggest_aliases(payload: Dict[str, Any]) -> Dict[str, Any]:
    """以 Chat Completions（JSON mode）產生 alias 規則增量。"""
    import json as _json
    enable = os.getenv("ALIAS_LLM_ENABLE", "1") == "1"
    if not enable:
        return {}
    model = os.getenv("ALIAS_LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
    max_tokens = _safe_int(os.getenv("ALIAS_LLM_MAX_TOKENS", "8000"), 8000)
    temperature = float(os.getenv("ALIAS_LLM_TEMPERATURE", "0.2"))
    system = (
        "你是台灣新聞/政治領域的資料工程助手。"
        "請產出查詢端同義規則的【增量 JSON】（僅 JSON）。"
    )
    schema_hint = {
        "relation": {"map": {}, "contains": {}, "regex": []},
        "head": {"aliases": {}},
        "tail": {"contains": {}},
    }
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("GPT_API") or os.getenv("OPENAI_API_KEY"),
                        timeout=float(os.getenv("OPENAI_TIMEOUT_SEC", "120")))
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _json.dumps({"schema": schema_hint, "payload": payload}, ensure_ascii=False)},
            ],
        )
        txt = resp.choices[0].message.content or "{}"
        return json.loads(txt)
    except Exception:
        # 後備：HTTP 直呼
        try:
            import requests
            url = os.getenv("MODEL_CONFIG_ENDPOINT") or "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('GPT_API') or os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
            }
            data = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": _json.dumps({"schema": schema_hint, "payload": payload}, ensure_ascii=False)},
                ],
            }
            r = requests.post(url, headers=headers, json=data,
                              timeout=float(os.getenv("OPENAI_TIMEOUT_SEC", "120")))
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            return json.loads(txt or "{}")
        except Exception:
            return {}

def _merge_alias_rules(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """合併 LLM 增量規則"""
    out = dict(base or {})
    def _merge_map(dst: Dict[str, List[str]], src: Dict[str, List[str]]) -> None:
        for k, vs in (src or {}).items():
            seed = k
            base_set = set(dst.get(k, []))
            add_set = set(v for v in (vs or []) if v and v != seed)
            merged = base_set | add_set
            if merged and not (len(merged) == 1 and (seed in merged)):
                dst[k] = list(merged)
    rel = out.setdefault("relation", {"map": {}, "contains": {}, "regex": []})
    _merge_map(rel.setdefault("map", {}), (delta.get("relation") or {}).get("map") or {})
    _merge_map(rel.setdefault("contains", {}), (delta.get("relation") or {}).get("contains") or {})
    seen = {tuple(x) for x in rel.setdefault("regex", []) if isinstance(x, list)}
    for pair in (delta.get("relation") or {}).get("regex") or []:
        t = tuple(pair) if isinstance(pair, list) else None
        if t and t not in seen:
            rel["regex"].append(list(t))
            seen.add(t)
    head = out.setdefault("head", {"aliases": {}})
    for k, vs in ((delta.get("head") or {}).get("aliases") or {}).items():
        head.setdefault("aliases", {}).setdefault(k, [])
        _merge_map(head["aliases"], {k: vs})
    tail = out.setdefault("tail", {"contains": {}})
    for k, vs in ((delta.get("tail") or {}).get("contains") or {}).items():
        tail.setdefault("contains", {}).setdefault(k, [])
        _merge_map(tail["contains"], {k: vs})
    return out
# ─────────────────────────
# CLI：build / ensure / doctor / alias
# ─────────────────────────
def _cmd_build(args: List[str]) -> int:
    if not args:
        print("用法：build <csv_path>")
        return 2
    csv_path = Path(args[0])
    build_from_csv_and_save(csv_path, idx_json=JSON_CACHE_PATH)
    print("✅ 已建表並輸出：")
    print(f"  - {JSON_CACHE_PATH}")
    return 0

def _cmd_ensure(args: List[str]) -> int:
    del args
    g = ensure_built_json(idx_json=JSON_CACHE_PATH)
    ok = g.is_valid()
    print(f"✅ ensure 完成：有效={ok}，節點={len(g.nodes)}，邊={len(g.edges)}")
    print(f"  JSON：{JSON_CACHE_PATH}")
    return 0

def _cmd_doctor(args: List[str]) -> int:
    del args
    try:
        g = CsvPropertyGraph.load_json(JSON_CACHE_PATH)
    except FileNotFoundError:
        print("❌ 找不到 JSON 索引，請先執行：build <csv_path> 或 ensure")
        return 2
    except Exception:
        print("❌ JSON 解析失敗。")
        return 1
    if not g.is_valid():
        print("❌ JSON 載入後內容異常（無有效邊）。")
        return 1
    print(f"✅ 健診成功，節點={len(g.nodes)}，邊={len(g.edges)}")
    return 0

def _mk_hints(names: List[str]) -> Dict[str, Dict[str, bool]]:
    org_suffix = ("黨","部","院","署","局","會","法院","地方法院","檢署","公司","媒體","新聞","通訊社","電視","廣播")
    out: Dict[str, Dict[str, bool]] = {}
    for s in names:
        h: Dict[str, bool] = {}
        if s.endswith(org_suffix):
            h["need_en_abbrev"] = True
        if any(k in s for k in ("通訊社","新聞","日報","時報","電視","廣播")):
            h["media_like"] = True
        if len(s) >= 2 and not s.endswith(org_suffix):
            h["person_like"] = True
        if h:
            out[s] = h
    return out

def _cmd_alias(args: List[str]) -> int:
    del args
    # 讀圖；若不存在就先 ensure
    if not JSON_CACHE_PATH.is_file():
        ensure_built_json(idx_json=JSON_CACHE_PATH)
    g = CsvPropertyGraph.load_json(JSON_CACHE_PATH)

    # 蒐集高頻詞與參數
    from collections import Counter
    rel_topn = _safe_int(os.getenv("ALIAS_LLM_REL_TOPN", "120"), 120)
    head_topn = _safe_int(os.getenv("ALIAS_LLM_HEAD_TOPN", "80"), 80)
    tail_topn = _safe_int(os.getenv("ALIAS_LLM_TAIL_TOPN", "80"), 80)
    rel_chunk = _safe_int(os.getenv("ALIAS_LLM_REL_CHUNK", "10"), 10)
    head_chunk = _safe_int(os.getenv("ALIAS_LLM_HEAD_CHUNK", "10"), 10)
    tail_chunk = _safe_int(os.getenv("ALIAS_LLM_TAIL_CHUNK", "10"), 10)
    gap_s = float(os.getenv("ALIAS_LLM_GAP_SEC", "0.4") or 0.4)
    write_every = _safe_int(os.getenv("ALIAS_WRITE_EVERY", "1"), 1)
    verbose = os.getenv("ALIAS_PROGRESS_VERBOSE", "1") == "1"

    rel_counter = Counter(e.relation for e in g.edges if isinstance(e.relation, str) and e.relation.strip())
    head_counter = Counter(e.head for e in g.edges if isinstance(e.head, str) and e.head.strip())
    tail_counter = Counter(e.tail for e in g.edges if isinstance(e.tail, str) and e.tail.strip())

    rels = [r for r, _ in rel_counter.most_common(rel_topn)]
    heads, tails = [], []
    for name, _ in head_counter.most_common(head_topn * 2):
        s = str(name).strip()
        if len(s) >= 2 and not re.fullmatch(r"[0-9\-./]+", s):
            heads.append(s)
        if len(heads) >= head_topn:
            break
    for name, _ in tail_counter.most_common(tail_topn * 2):
        s = str(name).strip()
        if len(s) >= 2 and not re.fullmatch(r"[0-9\-./]+", s):
            tails.append(s)
        if len(tails) >= tail_topn:
            break

    out_path = Path(os.getenv("ALIAS_RULES_JSON", "data/processed/knowledge-graph/alias_rules.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = json.loads(out_path.read_text(encoding="utf-8-sig"))
        if not isinstance(old, dict):
            old = {}
    except Exception:
        old = {}

    # 骨架
    new_map = {r: [r] for r in rels}
    old_rel = old.get("relation", {}) if isinstance(old.get("relation"), dict) else {}
    old_map = old_rel.get("map", {}) if isinstance(old_rel.get("map"), dict) else {}
    new_map.update(old_map)
    result = {
        "relation": {"map": new_map, "regex": old_rel.get("regex", []), "contains": old_rel.get("contains", {})},
        "head": {"aliases": (old.get("head") or {}).get("aliases", {})},
        "tail": {"contains": (old.get("tail") or {}).get("contains", {})},
    }

    batch_done = 0
    # a) relations
    for rel_batch in _chunks(rels, rel_chunk):
        payload = {
            "relations": rel_batch,
            "sample_heads": [],
            "sample_tails": [],
            "hints": {"heads": {}, "tails": {}},
            "existing_rules": result,
        }
        delta = _retry(lambda: _llm_suggest_aliases(payload))
        result = _merge_alias_rules(result, delta)
        batch_done += 1
        if verbose:
            print(f"[alias] merged batch #{batch_done} (relations) → rel_map={len(result['relation']['map'])}")
        if write_every <= 1 or (batch_done % write_every == 0):
            _atomic_write_json(out_path.with_suffix(".ckpt.json"), result)
            _atomic_write_json(out_path, result)
        time.sleep(gap_s)

    # b) heads
    for h_batch in _chunks(heads, head_chunk):
        payload = {
            "relations": [],
            "sample_heads": h_batch,
            "sample_tails": [],
            "hints": {"heads": _mk_hints(h_batch), "tails": {}},
            "existing_rules": result,
        }
        delta = _retry(lambda: _llm_suggest_aliases(payload))
        result = _merge_alias_rules(result, delta)
        batch_done += 1
        if verbose:
            print(f"[alias] merged batch #{batch_done} (heads) → head_aliases={len(result['head']['aliases'])}")
        if write_every <= 1 or (batch_done % write_every == 0):
            _atomic_write_json(out_path.with_suffix(".ckpt.json"), result)
            _atomic_write_json(out_path, result)
        time.sleep(gap_s)

    # c) tails
    for t_batch in _chunks(tails, tail_chunk):
        payload = {
            "relations": [],
            "sample_heads": [],
            "sample_tails": t_batch,
            "hints": {"heads": {}, "tails": _mk_hints(t_batch)},
            "existing_rules": result,
        }
        delta = _retry(lambda: _llm_suggest_aliases(payload))
        result = _merge_alias_rules(result, delta)
        batch_done += 1
        if verbose:
            print(f"[alias] merged batch #{batch_done} (tails) → tail_keys={len(result['tail']['contains'])}")
        if write_every <= 1 or (batch_done % write_every == 0):
            _atomic_write_json(out_path.with_suffix(".ckpt.json"), result)
            _atomic_write_json(out_path, result)
        time.sleep(gap_s)

    _atomic_write_json(out_path.with_suffix(".ckpt.json"), result)
    _atomic_write_json(out_path, result)
    print("✅ 已同步產生/更新同義規則：")
    print(f"  - {out_path}")
    return 0
def main() -> None:
    if len(sys.argv) <= 1:
        print(
            "用法：python -m src.qa.preliminary_work.build_csv_property_graph "
            "[build <csv_path> | ensure | doctor | alias]\n"
            f"預設路徑：CSV={RAW_CSV_DEFAULT}  JSON={JSON_CACHE_PATH}"
        )
        return
    cmd = sys.argv[1].lower()
    args = sys.argv[2:]
    try:
        if cmd == "build":
            raise SystemExit(_cmd_build(args))
        if cmd == "ensure":
            raise SystemExit(_cmd_ensure(args))
        if cmd == "doctor":
            raise SystemExit(_cmd_doctor(args))
        if cmd == "alias":
            raise SystemExit(_cmd_alias(args))
        print(f"未知子命令：{cmd}")
        raise SystemExit(2)
    except SystemExit:
        raise
    except Exception:
        print("❌ 執行失敗：")
        print(traceback.format_exc())
        raise SystemExit(1)

if __name__ == "__main__":
    main()
