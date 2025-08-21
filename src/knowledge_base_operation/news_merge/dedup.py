"""
News DEDUP（保留時間線；過短內容直接清理）
讀取  data/processed/news_merge/cna_ey_moi_pts.json
輸出  data/processed/news_merge/cna_ey_moi_pts_DEDUP.json

流程摘要：
0) 先行清理：移除「內文字數 < 40」的新聞（含 variants 過濾）
1) URL 完全一致 -> 合併（保留 variants）
2) 標題正規化一致 -> 合併（保留 variants）
3) SimHash-64（標題 + 內文前綴）近重複 (Hamming ≤ 3) -> 分群（保留 variants）

代表選擇（pick_canonical）：
- 內容較長者優先
- 其次以標題正規化字典序穩定化
- 不依據發布時間早晚，以完整保留時間線價值

variants 補充：
- 依 published_at 升冪排序
- variant 相對 canonical 的新增日期/數字會放入 delta_facts
"""

from __future__ import annotations
import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- 固定 I/O ----------
IN_PATH = Path("data/processed/news_merge/cna_ey_moi_pts.json")
OUT_PATH = Path("data/processed/news_merge/cna_ey_moi_pts_DEDUP.json")

# ---------- 欄位別名 ----------
URL_FIELDS = ["url", "link", "source_url"]
TITLE_FIELDS = ["title", "headline"]
CONTENT_FIELDS = ["content", "body", "text", "article"]
DATE_FIELDS = ["published_at", "pubDate", "date", "time",
               "publish_time", "publishedAt", "created_at", "updated_at"]
SOURCE_FIELDS = ["source", "publisher", "site"]

# ---------- 參數 ----------
DEFAULT_SIMHASH_BITS = 64
CONTENT_PREFIX = 320         # 參與 SimHash 的內文前綴字元數
SIMHASH_HAMMING_THRESH = 3   # Hamming ≤ 3 視為近重複
MIN_CONTENT_LEN = 40         # 低於此閾值的新聞將被直接剔除

# ---------- I/O ----------


def safe_read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------- 小工具 ----------
_PUNCT = re.compile(
    r"[\s\.\,\!\?\:\;\-\_\(\)\[\]\{\}\"\'\/\\\|\+\=\*\&\^\%\$#@~`，。！？：；、（）【】《》〈〉…]+", re.UNICODE)
_DATE_PAT = re.compile(
    r"(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2})")
_NUM_PAT = re.compile(r"(?<!\w)(\d{1,3}(?:,\d{3})*|\d+)(?!\w)")


def normalize_title(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def get_first(obj: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in obj and obj[k]:
            return obj[k]
    return None


def extract_items(payload: Any) -> List[Dict[str, Any]]:
    """將輸入 payload 解析為文章列表。"""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("items", "articles", "data"):
            if key in payload and isinstance(payload[key], list):
                return [x for x in payload[key] if isinstance(x, dict)]
        # 若每個 value 都是 list，合併
        vals = []
        ok = True
        for v in payload.values():
            if isinstance(v, list):
                vals.extend([x for x in v if isinstance(x, dict)])
            else:
                ok = False
        if ok and vals:
            return vals
    return [payload] if isinstance(payload, dict) else []

# ---------- SimHash ----------


def md5_64bit(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def _tokenize_for_simhash(title: str, content: str, content_prefix: int) -> List[str]:
    toks: List[str] = []
    t = (title or "").strip()
    c = (content or "").strip()[:max(0, content_prefix)]
    text = f"{t} || {c}"
    # 拉丁/數字詞
    for w in re.findall(r"[A-Za-z0-9]+", text):
        toks.append(w.lower())
    # CJK bigram
    cjks = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(cjks) - 1):
        toks.append(cjks[i] + cjks[i + 1])
    if len(cjks) >= 3:
        toks.append("".join(cjks[:3]))
    return toks if toks else [text.lower()]


def simhash64(tokens: List[str]) -> int:
    bits = DEFAULT_SIMHASH_BITS
    v = [0] * bits
    for tok in tokens:
        hv = md5_64bit(tok)
        for i in range(bits):
            v[i] += 1 if ((hv >> i) & 1) else -1
    fp = 0
    for i in range(bits):
        if v[i] >= 0:
            fp |= (1 << i)
    return fp


def hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

# ---------- 差異摘要 ----------


def extract_delta_facts(canonical_text: str, variant_text: str) -> Dict[str, List[str]]:
    """回傳 variant 相對 canonical 的新增日期/數字（僅做輕量提示用）。"""
    can_dates = set(m.group(1)
                    for m in _DATE_PAT.finditer(canonical_text or ""))
    var_dates = set(m.group(1) for m in _DATE_PAT.finditer(variant_text or ""))
    can_nums = set(m.group(1) for m in _NUM_PAT.finditer(canonical_text or ""))
    var_nums = set(m.group(1) for m in _NUM_PAT.finditer(variant_text or ""))

    add_dates = sorted(list(var_dates - can_dates))
    add_nums = sorted(list(var_nums - can_nums))
    out: Dict[str, List[str]] = {}
    if add_dates:
        out["added_dates"] = add_dates
    if add_nums:
        out["added_numbers"] = add_nums
    return out

# ---------- 代表選擇（不依據時間早晚） ----------


def pick_canonical(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    代表選擇規則：
    1) 內容較長者優先
    2) 標題正規化字典序較小者（穩定化）
    """
    lena = len(str(get_first(a, CONTENT_FIELDS) or ""))
    lenb = len(str(get_first(b, CONTENT_FIELDS) or ""))
    if lena != lenb:
        return a if lena > lenb else b

    ta = normalize_title(get_first(a, TITLE_FIELDS))
    tb = normalize_title(get_first(b, TITLE_FIELDS))
    if ta != tb:
        return a if ta < tb else b
    return a

# ---------- 長度過濾 ----------


def is_short(it: Dict[str, Any], min_len: int = MIN_CONTENT_LEN) -> bool:
    txt = get_first(it, CONTENT_FIELDS)
    if not isinstance(txt, str):
        return True
    return len(txt.strip()) < min_len


def drop_shorts(items: List[Dict[str, Any]], min_len: int = MIN_CONTENT_LEN) -> Tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    removed = 0
    for it in items:
        if is_short(it, min_len=min_len):
            removed += 1
        else:
            kept.append(it)
    return kept, removed

# ---------- 去重主流程（保留 variants） ----------


def dedup_with_variants(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    before = len(items)

    # (0) 先行剔除過短內容
    items, removed_short_0 = drop_shorts(items, MIN_CONTENT_LEN)

    # (1) URL 完全一致合併（保留 variants）
    by_url: Dict[str, List[Dict[str, Any]]] = {}
    no_url: List[Dict[str, Any]] = []
    for it in items:
        url = get_first(it, URL_FIELDS)
        if isinstance(url, str) and url.strip():
            by_url.setdefault(url.strip(), []).append(it)
        else:
            no_url.append(it)

    merged1: List[Dict[str, Any]] = []
    for lst in by_url.values():
        if len(lst) == 1:
            merged1.extend(lst)
        else:
            rep = lst[0]
            for it in lst[1:]:
                rep = pick_canonical(rep, it)
            variants = [it for it in lst if it is not rep]
            rep = attach_variants(rep, variants, min_len=MIN_CONTENT_LEN)
            merged1.append(rep)
    merged1.extend(no_url)

    # (2) 標題正規化一致合併（保留 variants）
    by_title: Dict[str, List[Dict[str, Any]]] = {}
    no_title: List[Dict[str, Any]] = []
    for it in merged1:
        norm = normalize_title(get_first(it, TITLE_FIELDS))
        if norm:
            by_title.setdefault(norm, []).append(it)
        else:
            no_title.append(it)

    merged2: List[Dict[str, Any]] = []
    for lst in by_title.values():
        if len(lst) == 1:
            merged2.extend(lst)
        else:
            rep = lst[0]
            all_variants: List[Dict[str, Any]] = []
            for it in lst[1:]:
                rep = pick_canonical(rep, it)
            for it in lst:
                if it is not rep:
                    all_variants.extend(detach_variants(it))  # 收攏既有 variants
            rep = attach_variants(rep, all_variants, min_len=MIN_CONTENT_LEN)
            merged2.append(rep)
    merged2.extend(no_title)

    # (3) SimHash 近重複分桶 + 合併（保留 variants）
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for it in merged2:
        title = str(get_first(it, TITLE_FIELDS) or "")
        content = str(get_first(it, CONTENT_FIELDS) or "")
        fp = simhash64(_tokenize_for_simhash(title, content, CONTENT_PREFIX))
        bucket = fp >> (DEFAULT_SIMHASH_BITS - 16)
        lst = buckets.setdefault(bucket, [])

        merged = False
        for j, rep in enumerate(lst):
            rtitle = str(get_first(rep, TITLE_FIELDS) or "")
            rcontent = str(get_first(rep, CONTENT_FIELDS) or "")
            rfp = simhash64(_tokenize_for_simhash(
                rtitle, rcontent, CONTENT_PREFIX))
            if hamming_distance64(fp, rfp) <= SIMHASH_HAMMING_THRESH:
                new_rep = pick_canonical(rep, it)
                variants = []
                variants.extend(detach_variants(
                    rep if new_rep is not rep else it))
                variants.extend(detach_variants(it if new_rep is rep else rep))
                new_rep = attach_variants(
                    new_rep, variants, min_len=MIN_CONTENT_LEN)
                lst[j] = new_rep
                merged = True
                break
        if not merged:
            lst.append(it)

    # 收攏與標注 cluster_id
    out: List[Dict[str, Any]] = []
    cluster_idx = 0
    for lst in buckets.values():
        for rep in lst:
            rep["dedup_cluster_id"] = f"cluster_{cluster_idx:06d}"
            rep["canonical"] = True
            # 排序 variants 以保留時間線（舊→新）
            if "variants" in rep and isinstance(rep["variants"], list):
                # variants 也二次過短過濾（防止任何殘留）
                rep["variants"] = [v for v in rep["variants"]
                                   if not is_short(v, MIN_CONTENT_LEN)]
                rep["variants"].sort(key=lambda x: str(
                    get_first(x, DATE_FIELDS) or ""))
            out.append(rep)
            cluster_idx += 1

    after = len(out)
    stats = {
        "before": before,
        "after": after,
        "removed": before - after,
        "removed_short": removed_short_0
    }
    return out, stats

# ---------- variants 附加/拆解 ----------


def attach_variants(rep: Dict[str, Any], variants: List[Dict[str, Any]], min_len: int) -> Dict[str, Any]:
    """將 variants 掛到代表，並生成 delta_facts；同時濾除過短內容。"""
    if not variants:
        return rep
    rep = dict(rep)  # 淺拷貝
    cur_vars = rep.get("variants", [])
    if not isinstance(cur_vars, list):
        cur_vars = []
    canon_text = str(get_first(rep, CONTENT_FIELDS) or "")
    new_vars = []
    for v in variants:
        if v is rep or is_short(v, min_len=min_len):
            continue
        v = dict(v)
        v["canonical"] = False
        delta = extract_delta_facts(canon_text, str(
            get_first(v, CONTENT_FIELDS) or ""))
        if delta:
            v["delta_facts"] = delta
        new_vars.append(v)
    if new_vars:
        rep["variants"] = cur_vars + new_vars
    return rep


def detach_variants(it: Dict[str, Any]) -> List[Dict[str, Any]]:
    """取出條目上的 variants 並清空，回傳 variants 列表。"""
    vars_ = it.get("variants", [])
    if isinstance(vars_, list) and vars_:
        it["variants"] = []
        return vars_
    return []

# ---------- 主程序 ----------


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input not found: {IN_PATH}")

    payload = safe_read_json(IN_PATH)
    items = extract_items(payload)
    if not items:
        print(f"⚠️ 無可處理的文章：{IN_PATH}")
        safe_write_json(OUT_PATH, [])
        return

    deduped, stats = dedup_with_variants(items)

    # 輸出為【代表化列表】，每個代表包含 variants（完整時間線），且已剔除過短內容
    safe_write_json(OUT_PATH, deduped)

    print("[DEDUP] 完成")
    print(f"  - Input : {IN_PATH}")
    print(f"  - Output: {OUT_PATH}")
    print(
        f"  - Stats : {stats['before']} -> {stats['after']} (removed {stats['removed']})")
    print(
        f"  - Purge : removed_short(<{MIN_CONTENT_LEN} chars) = {stats['removed_short']}")
    print("  - 保留策略：內容長度優先，其次標題字典序（不使用發布時間早晚做裁決）。")
    print("  - 時間線：近重複版本保存在 canonical 的 variants 中；variants 亦已過短過濾。")


if __name__ == "__main__":
    main()
