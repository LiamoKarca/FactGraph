"""
$ python -m src.qa.preliminary_work.strong_rel_keywords dumpall

從 CSV 自動抽取「關係關鍵詞」工具（強化版 + 全量輸出）
- 輸入欄位：head, relation, tail, head_props, rel_props, tail_props
- 模式：
    1) build   → 產出「強關鍵詞」(強化指標、去泛用)
    2) dumpall → 產出「全關鍵詞字典」(幾乎不過濾，僅基本去噪)，並輸出統計

輸出路徑（預設）
---------------
data/processed/knowledge-graph/strong_rel_keywords.json
data/processed/knowledge-graph/strong_rel_keywords.py         (STRONG_REL_KEYWORDS = {...})
data/processed/knowledge-graph/strong_rel_keywords_debug.csv  (強詞排行與指標)

data/processed/knowledge-graph/relation_dict_all.json
data/processed/knowledge-graph/relation_dict_all.py           (RELATIONS_ALL = {...})
data/processed/knowledge-graph/relation_dict_all.csv          (全詞統計)

常用指令
--------
# 建立強關鍵詞（篩掉泛用，強化「涉及/指控/偵辦/搜索/到案…」）
python -m src.qa.preliminary_work.strong_rel_keywords build

# 匯出全關鍵詞（一次把 CSV 裡所有 relation 字串彙總）
python -m src.qa.preliminary_work.strong_rel_keywords dumpall
# → 可加參數（見下方 CLI 說明）

環境參數（可調）
----------------
LI_PG_RAW_CSV            預設 data/raw/knowledge-graph/neo4j-kg-raw-graph.csv
SRK_TOP_N                預設 256     # 強關鍵詞輸出上限
SRK_MIN_COUNT            預設 20      # 強關鍵詞最低出現次數
SRK_MIN_EVI_RATIO        預設 0.25    # 強關鍵詞最低 evidence 覆蓋率
SRK_MIN_LEN              預設 1
SRK_MAX_LEN              預設 6
SRK_ALLOW_ASCII          預設 0       # 僅中文為主
SRK_ALPHA_SELECTIVITY    預設 1.2     # 泛用度懲罰係數
SRK_W_FREQ               預設 1.0
SRK_W_EVI                預設 10.0
SRK_W_LEN                預設 0.2
SRK_W_STRONGPAT          預設 1.0
SRK_OUT_DIR              預設 data/processed/knowledge-graph

dumpall 專用（亦可用 CLI 覆寫）
------------------------------
SRK_DA_MIN_LEN           預設 1
SRK_DA_MAX_LEN           預設 12
SRK_DA_ALLOW_ASCII       預設 1      # 全量模式預設允許 ASCII
SRK_DA_KEEP_WEAK         預設 1      # 全量模式預設保留弱動詞（不過濾）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# ==== 路徑 ====
RAW_CSV_PATH = os.getenv("LI_PG_RAW_CSV", "data/raw/knowledge-graph/neo4j-kg-raw-graph.csv")
OUT_DIR = Path(os.getenv("SRK_OUT_DIR", "data/processed/knowledge-graph")).resolve()

# 強關鍵詞輸出
JSON_OUT = OUT_DIR / "strong_rel_keywords.json"
PY_OUT = OUT_DIR / "strong_rel_keywords.py"
DBG_OUT = OUT_DIR / "strong_rel_keywords_debug.csv"

# 全量輸出
ALL_JSON_OUT = OUT_DIR / "relation_dict_all.json"
ALL_PY_OUT = OUT_DIR / "relation_dict_all.py"
ALL_CSV_OUT = OUT_DIR / "relation_dict_all.csv"

# ==== 參數（強關鍵詞）====
TOP_N = int(os.getenv("SRK_TOP_N", "256"))
MIN_COUNT = int(os.getenv("SRK_MIN_COUNT", "20"))
MIN_EVI_RATIO = float(os.getenv("SRK_MIN_EVI_RATIO", "0.25"))
MIN_LEN = int(os.getenv("SRK_MIN_LEN", "1"))
MAX_LEN = int(os.getenv("SRK_MAX_LEN", "6"))
ALLOW_ASCII = os.getenv("SRK_ALLOW_ASCII", "0") == "1"

ALPHA_SELECTIVITY = float(os.getenv("SRK_ALPHA_SELECTIVITY", "1.2"))
W_FREQ = float(os.getenv("SRK_W_FREQ", "1.0"))
W_EVI = float(os.getenv("SRK_W_EVI", "10.0"))
W_LEN = float(os.getenv("SRK_W_LEN", "0.2"))
W_STRONGPAT = float(os.getenv("SRK_W_STRONGPAT", "1.0"))

# ==== 參數（dumpall）====
DA_MIN_LEN = int(os.getenv("SRK_DA_MIN_LEN", "1"))
DA_MAX_LEN = int(os.getenv("SRK_DA_MAX_LEN", "12"))
DA_ALLOW_ASCII = os.getenv("SRK_DA_ALLOW_ASCII", "1") == "1"
DA_KEEP_WEAK = os.getenv("SRK_DA_KEEP_WEAK", "1") == "1"

# 停用弱動詞（在「強關鍵詞」模式會過濾；dumpall 預設不過濾）
_STOP_WEAK = {
    "表示", "指出", "強調", "說明", "提到", "提及", "認為", "希望", "呼籲", "感謝",
    "公布", "公佈", "宣布", "說", "談", "提問", "回應", "提出", "提供", "包含",
    "編列", "核定", "推動", "主持", "召開", "舉辦", "舉行", "參加", "出席", "參與",
    "合作", "視察", "接見", "頒發", "影響", "重申", "提醒", "承諾", "邀請", "支持",
    "協助", "報告", "發表", "發布", "發佈",
}

# 強動詞樣式（指控、偵辦、違法、衝突等）
_STRONG_PATTERNS = [
    r"涉", r"涉及", r"涉嫌",
    r"指控", r"質疑", r"爆料", r"揭露", r"踢爆",
    r"偵辦", r"搜索", r"搜查", r"查扣", r"羈押", r"到案", r"約談", r"傳喚", r"起訴",
    r"違", r"違法", r"違規", r"違憲",
    r"抗議", r"衝突", r"攻擊", r"辱罵", r"滋擾",
    r"調查", r"追查", r"追訴", r"檢舉",
    r"批評", r"抨擊", r"譴責",
    r"到場", r"現身",
]

# 後備預設（尚未建檔時使用）
_FALLBACK_SET: Set[str] = {
    "涉及", "涉嫌", "指控", "質疑", "批評", "偵辦", "搜索", "到案", "爆料", "抨擊", "譴責", "調查", "追查", "到場"
}

# ==== 工具 regex ====
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]")  # 非字母數字與 CJK
_MULTI_SPACE_RE = re.compile(r"\s+")
_NUMERIC_LIKE_RE = re.compile(r"^\s*[\d\.\-_/]+\s*$")


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = _MULTI_SPACE_RE.sub(" ", s)
    return s


def _is_cjk_heavy(s: str) -> bool:
    if not s:
        return False
    cjk_count = len(_CJK_RE.findall(s))
    return cjk_count >= max(1, len(s) // 2)


def _looks_noise(s: str, *, min_len: int, max_len: int, allow_ascii: bool) -> bool:
    """過濾噪音 relation：純數字/版本號、含過多標點、太長或太短。"""
    if not s:
        return True
    if _NUMERIC_LIKE_RE.match(s):
        return True
    # 太多非字母數字與 CJK 的字元
    punct = _PUNCT_RE.findall(s)
    if len(punct) >= max(1, len(s) // 3):
        return True
    n = len(s)
    if n < min_len or n > max_len:
        return True
    if not allow_ascii and not _is_cjk_heavy(s):
        return True
    return False


def _parse_rel_props(raw: str) -> Dict:
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        s2 = (s.replace("'", '"')
                .replace("None", "null")
                .replace("True", "true")
                .replace("False", "false"))
        try:
            return json.loads(s2)
        except Exception:
            return {}


def _iter_rows(csv_path: str) -> Iterable[Tuple[str, str, str, Dict]]:
    """yield (head, relation, tail, rel_props)"""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"head", "relation", "tail", "head_props", "rel_props", "tail_props"}
        missing = [x for x in required if x not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV 欄位缺少：{missing}")

        for row in reader:
            h = _norm(row.get("head", ""))
            r = _norm(row.get("relation", ""))
            t = _norm(row.get("tail", ""))
            if not r:
                continue
            rp = _parse_rel_props(row.get("rel_props"))
            yield h, r, t, rp


# ========= 強關鍵詞（精選） =========

def build_keywords(csv_path: str = RAW_CSV_PATH) -> Tuple[List[str], List[Tuple]]:
    """
    回傳 (keywords, debug_rows)
    debug_rows 欄位：
        [rank, relation, score, count, evi_ratio, avg_evi_len, uniq_heads, uniq_tails, selectivity_penalty, strongpat]
    """
    freq = Counter()
    evi_hit = defaultdict(int)
    evi_len_sum = defaultdict(int)
    uniq_heads = defaultdict(set)
    uniq_tails = defaultdict(set)

    for h, r, t, rp in _iter_rows(csv_path):
        if _looks_noise(r, min_len=MIN_LEN, max_len=MAX_LEN, allow_ascii=ALLOW_ASCII):
            continue
        if r in _STOP_WEAK:  # 精選模式：移除弱動詞
            continue

        freq[r] += 1
        evi = str(rp.get("evidence", "") or "")
        if len(evi.strip()) >= 6:
            evi_hit[r] += 1
            evi_len_sum[r] += len(evi.strip())
        if h:
            uniq_heads[r].add(h)
        if t:
            uniq_tails[r].add(t)

    if not freq:
        return sorted(_FALLBACK_SET), []

    # 候選
    cand = {r for r, c in freq.items() if c >= MIN_COUNT}
    if not cand:
        cand = {r for r, _ in freq.most_common(TOP_N * 5)}

    scored = []
    for r in cand:
        c = freq[r]
        eh = evi_hit[r]
        e_ratio = (eh / c) if c > 0 else 0.0
        if e_ratio < MIN_EVI_RATIO:
            continue
        avg_e_len = (evi_len_sum[r] / max(1, eh)) if eh > 0 else 0.0

        uh = len(uniq_heads[r]) or 1
        ut = len(uniq_tails[r]) or 1
        sel_pen = 1.0 / (1.0 + ALPHA_SELECTIVITY * math.log1p(math.sqrt(uh * ut)))

        strongpat = 0.0
        for p in _STRONG_PATTERNS:
            if re.fullmatch(p, r) or (len(r) <= 6 and re.search(p, r)):
                strongpat = 1.0
                break

        base_score = (W_FREQ * c) + (W_EVI * e_ratio) + (0.02 * avg_e_len) - (W_LEN * len(r))
        score = base_score * sel_pen + (W_STRONGPAT * strongpat)
        scored.append((score, r, c, e_ratio, avg_e_len, uh, ut, sel_pen, strongpat))

    if not scored:
        for r, c in freq.most_common(TOP_N * 5):
            if r in _STOP_WEAK or _looks_noise(r, min_len=MIN_LEN, max_len=MAX_LEN, allow_ascii=ALLOW_ASCII):
                continue
            eh = evi_hit[r]; e_ratio = (eh / c) if c > 0 else 0.0
            avg_e_len = (evi_len_sum[r] / max(1, eh)) if eh > 0 else 0.0
            uh = len(uniq_heads[r]) or 1; ut = len(uniq_tails[r]) or 1
            sel_pen = 1.0 / (1.0 + ALPHA_SELECTIVITY * math.log1p(math.sqrt(uh * ut)))
            strongpat = 1.0 if any((re.fullmatch(p, r) or re.search(p, r)) for p in _STRONG_PATTERNS) else 0.0
            base_score = (W_FREQ * c) + (W_EVI * e_ratio) + (0.02 * avg_e_len) - (W_LEN * len(r))
            score = base_score * sel_pen + (W_STRONGPAT * strongpat)
            scored.append((score, r, c, e_ratio, avg_e_len, uh, ut, sel_pen, strongpat))

    scored.sort(key=lambda x: (-x[0], -x[2], len(x[1]), x[1]))
    top = scored[:TOP_N]
    keywords = [r for _, r, *_rest in top]

    # 去重保序
    seen = set(); uniq = []
    for k in keywords:
        if k not in seen:
            seen.add(k); uniq.append(k)

    dbg_rows = []
    for rank, row in enumerate(top, 1):
        score, r, c, e_ratio, avg_e_len, uh, ut, sel_pen, strongpat = row
        dbg_rows.append((rank, r, f"{score:.3f}", c, f"{e_ratio:.3f}", f"{avg_e_len:.1f}", uh, ut, f"{sel_pen:.3f}", int(strongpat)))

    return uniq, dbg_rows


def save_strong_outputs(keywords: List[str], dbg_rows: List[Tuple]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "size": len(keywords),
        "min_count": MIN_COUNT,
        "min_evidence_ratio": MIN_EVI_RATIO,
        "min_len": MIN_LEN,
        "max_len": MAX_LEN,
        "allow_ascii": ALLOW_ASCII,
        "alpha_selectivity": ALPHA_SELECTIVITY,
        "weights": {"freq": W_FREQ, "evi": W_EVI, "len_pen": W_LEN, "strongpat": W_STRONGPAT},
        "source_csv": str(Path(RAW_CSV_PATH).resolve()),
    }
    JSON_OUT.write_text(json.dumps({"keywords": keywords, "meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    py_lines = [
        "# -*- coding: utf-8-sig -*-",
        "\"\"\"Auto-generated by strong_rel_keywords.py (strong set). Do not edit manually.\"\"\"",
        "STRONG_REL_KEYWORDS = {",
    ]
    for k in keywords:
        py_lines.append(f"    {json.dumps(k, ensure_ascii=False)},")
    py_lines.append("}\n")
    PY_OUT.write_text("\n".join(py_lines), encoding="utf-8-sig")

    with open(DBG_OUT, "w", encoding="utf-8-sig", newline="") as f:
        f.write("rank,relation,score,count,evi_ratio,avg_evi_len,uniq_heads,uniq_tails,selectivity_penalty,strongpat\n")
        for row in dbg_rows:
            f.write(",".join(map(str, row)) + "\n")

    print(f"✅ 強關鍵詞已輸出：\n  - {JSON_OUT}\n  - {PY_OUT}\n  - {DBG_OUT}")


# ========= 全關鍵詞（幾乎不過濾） =========

def dump_all_relations(
    csv_path: str = RAW_CSV_PATH,
    *,
    min_len: int = DA_MIN_LEN,
    max_len: int = DA_MAX_LEN,
    allow_ascii: bool = DA_ALLOW_ASCII,
    keep_weak: bool = DA_KEEP_WEAK,
) -> Tuple[List[str], List[Tuple]]:
    """
    回傳 (all_relations, debug_rows)
    debug_rows 欄位：
        [rank, relation, count, evi_ratio, avg_evi_len, uniq_heads, uniq_tails]
    - 僅做基本去噪（長度、標點、純數字/版本號、CJK 佔比），預設保留弱動詞
    """
    freq = Counter()
    evi_hit = defaultdict(int)
    evi_len_sum = defaultdict(int)
    uniq_heads = defaultdict(set)
    uniq_tails = defaultdict(set)

    for h, r, t, rp in _iter_rows(csv_path):
        if _looks_noise(r, min_len=min_len, max_len=max_len, allow_ascii=allow_ascii):
            continue
        if (not keep_weak) and (r in _STOP_WEAK):
            continue

        freq[r] += 1
        evi = str(rp.get("evidence", "") or "")
        if len(evi.strip()) >= 6:
            evi_hit[r] += 1
            evi_len_sum[r] += len(evi.strip())
        if h:
            uniq_heads[r].add(h)
        if t:
            uniq_tails[r].add(t)

    if not freq:
        return [], []

    # 排序：頻率 desc → evidence 比例 desc → 短字優先 → 字典序
    scored = []
    for r, c in freq.items():
        eh = evi_hit[r]; e_ratio = (eh / c) if c > 0 else 0.0
        avg_e_len = (evi_len_sum[r] / max(1, eh)) if eh > 0 else 0.0
        scored.append((c, e_ratio, -len(r), r, avg_e_len, len(uniq_heads[r]), len(uniq_tails[r])))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    all_rel = [row[3] for row in scored]

    dbg_rows = []
    for rank, row in enumerate(scored, 1):
        c, e_ratio, _neglen, r, avg_e_len, uh, ut = row
        dbg_rows.append((rank, r, c, f"{e_ratio:.3f}", f"{avg_e_len:.1f}", uh, ut))

    return all_rel, dbg_rows


def save_all_outputs(all_rel: List[str], dbg_rows: List[Tuple]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALL_JSON_OUT.write_text(json.dumps({"relations": all_rel}, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    py_lines = [
        "# -*- coding: utf-8-sig -*-",
        "\"\"\"Auto-generated by strong_rel_keywords.py (ALL relations). Do not edit manually.\"\"\"",
        "RELATIONS_ALL = {",
    ]
    for k in all_rel:
        py_lines.append(f"    {json.dumps(k, ensure_ascii=False)},")
    py_lines.append("}\n")
    ALL_PY_OUT.write_text("\n".join(py_lines), encoding="utf-8-sig")

    with open(ALL_CSV_OUT, "w", encoding="utf-8-sig", newline="") as f:
        f.write("rank,relation,count,evi_ratio,avg_evi_len,uniq_heads,uniq_tails\n")
        for row in dbg_rows:
            f.write(",".join(map(str, row)) + "\n")

    print(f"✅ 全關鍵詞已輸出：\n  - {ALL_JSON_OUT}\n  - {ALL_PY_OUT}\n  - {ALL_CSV_OUT}")


# ========= 讀取器（供 pipeline 使用） =========

def load_strong_rel_keywords() -> Set[str]:
    # 優先讀 .py
    try:
        ns: Dict[str, object] = {}
        code = PY_OUT.read_text(encoding="utf-8-sig")
        exec(compile(code, str(PY_OUT), "exec"), ns, ns)
        ks = ns.get("STRONG_REL_KEYWORDS")
        if isinstance(ks, (set, list, tuple)):
            return set(ks)
    except Exception:
        pass

    # 次讀 .json
    try:
        data = json.loads(JSON_OUT.read_text(encoding="utf-8-sig"))
        ks = data.get("keywords", [])
        if isinstance(ks, list) and ks:
            return set(ks)
    except Exception:
        pass

    return set(_FALLBACK_SET)


def load_all_relations() -> Set[str]:
    """載入全關係集合（若尚未產生，回傳空集合）"""
    try:
        ns: Dict[str, object] = {}
        code = ALL_PY_OUT.read_text(encoding="utf-8-sig")
        exec(compile(code, str(ALL_PY_OUT), "exec"), ns, ns)
        ks = ns.get("RELATIONS_ALL")
        if isinstance(ks, (set, list, tuple)):
            return set(ks)
    except Exception:
        pass

    try:
        data = json.loads(ALL_JSON_OUT.read_text(encoding="utf-8-sig"))
        ks = data.get("relations", [])
        if isinstance(ks, list) and ks:
            return set(ks)
    except Exception:
        pass

    return set()


# ========= CLI =========

def _cli_build(_: argparse.Namespace) -> None:
    print(f"📄 CSV: {RAW_CSV_PATH}")
    kws, dbg = build_keywords(csv_path=RAW_CSV_PATH)
    print(f"📊 強關鍵詞（前 20）：{kws[:20]}")
    save_strong_outputs(kws, dbg)


def _cli_dumpall(args: argparse.Namespace) -> None:
    min_len = int(args.min_len if args.min_len is not None else DA_MIN_LEN)
    max_len = int(args.max_len if args.max_len is not None else DA_MAX_LEN)
    allow_ascii = bool(int(args.allow_ascii)) if args.allow_ascii is not None else DA_ALLOW_ASCII
    keep_weak = bool(int(args.keep_weak)) if args.keep_weak is not None else DA_KEEP_WEAK

    print(f"📄 CSV: {RAW_CSV_PATH}")
    print(f"🛠 參數：min_len={min_len}  max_len={max_len}  allow_ascii={int(allow_ascii)}  keep_weak={int(keep_weak)}")
    all_rel, dbg = dump_all_relations(
        csv_path=RAW_CSV_PATH,
        min_len=min_len,
        max_len=max_len,
        allow_ascii=allow_ascii,
        keep_weak=keep_weak,
    )
    print(f"📊 全關鍵詞數：{len(all_rel)}（前 20 示意：{all_rel[:20]}）")
    save_all_outputs(all_rel, dbg)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Strong Relation Keywords Builder / Loader (strong & all)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="建立強關鍵詞清單（去泛用、強化辦案/指控類）")
    p_build.set_defaults(func=_cli_build)

    p_all = sub.add_parser("dumpall", help="匯出全關鍵詞（僅基本去噪，可選保留/過濾弱動詞與 ASCII）")
    p_all.add_argument("--min_len", type=int, default=None, help=f"最小長度（預設 {DA_MIN_LEN}）")
    p_all.add_argument("--max_len", type=int, default=None, help=f"最大長度（預設 {DA_MAX_LEN}）")
    p_all.add_argument("--allow_ascii", type=int, choices=[0, 1], default=None, help=f"允許 ASCII（預設 {int(DA_ALLOW_ASCII)}）")
    p_all.add_argument("--keep_weak", type=int, choices=[0, 1], default=None, help=f"保留弱動詞（預設 {int(DA_KEEP_WEAK)}）")
    p_all.set_defaults(func=_cli_dumpall)

    return p.parse_args()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
