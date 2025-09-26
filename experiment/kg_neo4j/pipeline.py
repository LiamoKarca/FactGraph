"""
ETL：experiment/data/processed/related_news.json → 實體/關係抽取 → 轉 Neo4j 寫入。
以 CSV（data/processed/knowledge-graph/kg_times.csv）追蹤每篇新聞已抽取次數；
同一篇達上限（預設 3 次）即跳過以節省 API。

注意：
- 實驗用知識圖庫 NEO4J_DATABASE=experiment
- 注意 .env 設定 

處理順序（預設）：
- 以「今天」為起點，若當日無新聞，會自動往前一天繼續，直到最舊有資料的日期。
- 支援 date 欄位為字串日期（YYYY-MM-DD / YYYY/M/D / 含時間前綴）。

可選項：
- --json-path        JSON 路徑（預設 experiment/data/processed/related_news.json）
- --id-csv           僅處理 CSV 中列出的 _id（忽略日期回退邏輯）
- --start-date       起始日期（含），預設今天（YYYY-MM-DD）
- --until-date       最舊處理日期（含）；不提供則回退到資料中最舊日期
- --kg-times-csv     次數記錄檔路徑（預設 experiment/data/interim/kg_times.csv）
- --kg-max-times     同一篇的最大處理次數（預設 3）


$ python -m experiment.kg_neo4j.pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import date as ddate, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

# ── 路徑自動修正：以「直接執行檔案」啟動時，將專案根加入 sys.path ─────────────
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]  # <PROJECT_ROOT>
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .gadget import LOGGER, run_with_timer  # type: ignore
from .extraction import extract_entities_relations  # type: ignore
from .neo4j_loader import Neo4jLoader  # type: ignore
from .transformation import transform_to_neo4j_format  # type: ignore

# ── 常數 ───────────────────────────────────────────────────────────────────────────
KG_TIMES_CSV_DEFAULT = "experiment/data/interim/kg_times.csv"
KG_MAX_TIMES_DEFAULT = 3
MAX_RETRY_LLM = 5


# ── 小工具 ─────────────────────────────────────────────────────────────────────────
def ensure_parent_dir(path: Path) -> None:
    """確保路徑的父資料夾存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def to_str_id(mongo_id: Any) -> str:
    """安全轉字串 id。"""
    return "" if mongo_id is None else str(mongo_id)


def try_parse_date(s: str) -> Optional[ddate]:
    """嘗試解析字串為 date（支援 YYYY-MM-DD、YYYY/M/D、或含時間前綴）。"""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # 嘗試剝掉尾端時間（像 '2025-04-09T12:34:00Z' 或 'YYYY-MM-DD HH:MM:SS'）
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        y, mo, da = map(int, m.groups())
        try:
            return ddate(y, mo, da)
        except Exception:
            return None
    return None


def date_to_str(d: ddate) -> str:
    """標準化為 'YYYY-MM-DD'。"""
    return d.strftime("%Y-%m-%d")


def read_ids_from_csv(csv_path: str) -> List[str]:
    """
    自 CSV 讀取一批 _id：
    - 含表頭：欄名 'id' 或 '_id'
    - 無表頭：取第 1 欄
    皆以字串回傳。
    """
    ids: List[str] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            lc = [x.strip().lower() for x in reader.fieldnames]
            key = None
            if "id" in lc:
                key = reader.fieldnames[lc.index("id")]
            elif "_id" in lc:
                key = reader.fieldnames[lc.index("_id")]
            if key:
                for row in reader:
                    raw = str(row.get(key, "")).strip()
                    if raw:
                        ids.append(raw)
                return ids

        f.seek(0)
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            raw = str(row[0]).strip()
            if raw:
                ids.append(raw)
    return ids


def load_kg_times(csv_path: Path) -> Dict[str, int]:
    """讀取 kg_times.csv 為 dict[id]=times；若檔案不存在則回傳空 dict。"""
    if not csv_path.exists():
        return {}
    times_map: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        try:
            reader = csv.DictReader(f)
            if reader.fieldnames and "id" in reader.fieldnames and "times" in reader.fieldnames:
                for row in reader:
                    k = str(row.get("id", "")).strip()
                    v = str(row.get("times", "0")).strip()
                    if not k:
                        continue
                    try:
                        times_map[k] = int(v)
                    except ValueError:
                        times_map[k] = 0
                return times_map
        except Exception:
            pass

        f.seek(0)
        rdr = csv.reader(f)
        for row in rdr:
            if len(row) < 2:
                continue
            k = str(row[0]).strip()
            v = str(row[1]).strip()
            if not k:
                continue
            try:
                times_map[k] = int(v)
            except ValueError:
                times_map[k] = 0
    return times_map


def dump_kg_times(csv_path: Path, times_map: Dict[str, int]) -> None:
    """將 dict[id]=times 寫回 CSV（含表頭）。"""
    ensure_parent_dir(csv_path)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "times"])
        for k in sorted(times_map.keys()):
            writer.writerow([k, times_map[k]])


# ── JSON 載入與日期索引 ──────────────────────────────────────────────────────────
def load_related_news(json_path: Path) -> List[Dict[str, Any]]:
    """
    讀取 experiment/data/processed/related_news.json，整平成文件清單：
    來源結構：root.results[*].matches_full[*]
    欄位篩選：_id, date, title, content（必要）
    去重：以 _id 去重；若重複，保留較新的 date（無法比較則保留第一筆）。
    """
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    results = data.get("results", [])
    docs_map: Dict[str, Dict[str, Any]] = {}

    for grp in results:
        matches = grp.get("matches_full", []) or []
        for it in matches:
            _id = to_str_id(it.get("_id"))
            date_str = str(it.get("date", "")).strip()
            title = str(it.get("title", "")).strip()
            content = str(it.get("content", "")).strip()
            if not _id or not date_str or not (title or content):
                continue
            # 若重複，以較新日期取代
            if _id in docs_map:
                old = docs_map[_id]
                d_new = try_parse_date(date_str)
                d_old = try_parse_date(str(old.get("date", "")))
                if d_new and d_old and d_new > d_old:
                    docs_map[_id] = {"_id": _id, "date": date_str,
                                     "title": title, "content": content}
            else:
                docs_map[_id] = {"_id": _id, "date": date_str,
                                 "title": title, "content": content}

    docs = list(docs_map.values())
    return docs


def compute_bounds_and_index_by_date(
    docs: List[Dict[str, Any]]
) -> Tuple[Optional[ddate], Optional[ddate], Dict[str, List[Dict[str, Any]]]]:
    """
    回傳 (最舊日, 最新日, 以 YYYY-MM-DD 為鍵的日期索引)。
    解析失敗的日期將被忽略（不放入索引）。
    """
    idx: Dict[str, List[Dict[str, Any]]] = {}
    all_dates: List[ddate] = []

    for doc in docs:
        d = try_parse_date(str(doc.get("date", "")))
        if not d:
            continue
        ds = date_to_str(d)
        idx.setdefault(ds, []).append(doc)
        all_dates.append(d)

    if not all_dates:
        return None, None, idx
    return min(all_dates), max(all_dates), idx


def iter_docs_by_date_index(
    index_by_date: Dict[str, List[Dict[str, Any]]],
    start_date: ddate,
    until_date: Optional[ddate],
    oldest: ddate,
    newest: ddate,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """
    由 start_date 起，逐日往舊遞減產出 (date_str, doc)。
    - 若 until_date 提供，當日 < until_date 即停止；
      否則會退到資料中實際最舊日期。
    - 無論某天是否有新聞，都不會中斷；會繼續往前日。
    """
    start = min(start_date, newest)
    lower_bound = until_date if until_date is not None else oldest
    lower_bound = max(lower_bound, oldest)

    day = start
    while day >= lower_bound:
        ds = date_to_str(day)
        for doc in index_by_date.get(ds, []):
            yield ds, doc
        day -= timedelta(days=1)


# ── 主流程 ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ETL for related_news.json -> Neo4j with times control "
            "and newest-to-oldest date ordering"
        )
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="experiment/data/processed/related_news.json",
        help="來源 JSON 路徑（預設 experiment/data/processed/related_news.json）。",
    )
    parser.add_argument(
        "--id-csv",
        type=str,
        default=None,
        help="只處理 CSV 中列出的 _id（含 'id' / '_id' 表頭或純第一欄）。",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="起始日期（含），預設為今天。格式 YYYY-MM-DD。",
    )
    parser.add_argument(
        "--until-date",
        type=str,
        default=None,
        help="最舊處理日期（含）。未提供則回退到資料中最舊日期為止。",
    )
    parser.add_argument(
        "--kg-times-csv",
        type=str,
        default=KG_TIMES_CSV_DEFAULT,
        help="計數檔路徑（CSV：id,times）。",
    )
    parser.add_argument(
        "--kg-max-times",
        type=int,
        default=KG_MAX_TIMES_DEFAULT,
        help="同一篇新聞的最大處理次數（達上限即跳過）。",
    )
    args = parser.parse_args()

    # 載入 JSON 並建立日期索引
    src_path = Path(args.json_path).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"來源 JSON 不存在：{src_path}")
    docs = load_related_news(src_path)
    oldest, newest, index_by_date = compute_bounds_and_index_by_date(docs)

    LOGGER.info("📄 來源：%s", src_path)
    LOGGER.info("🗞️ 讀入文件數（去重後）：%d", len(docs))
    if oldest is None or newest is None:
        LOGGER.info("📭 來源內沒有可解析的日期資料，無事可做。")
        return
    LOGGER.info("🗓️ 日期範圍：%s ～ %s", date_to_str(oldest), date_to_str(newest))

    # Neo4j 連線
    neo4j_loader = Neo4jLoader()

    # 計數檔
    kg_times_csv = Path(args.kg_times_csv).resolve()
    times_map = load_kg_times(kg_times_csv)
    max_times = int(args.kg_max_times)
    LOGGER.info("🧮 計數檔：%s；上限：%d", kg_times_csv, max_times)

    processed_ok = 0
    skipped_by_times = 0

    try:
        # 準備 iterable
        if args.id_csv:
            ids = set(read_ids_from_csv(args.id_csv))
            LOGGER.info("🗂 以 CSV 中的 %d 筆 _id 進行處理（忽略日期回退）", len(ids))
            # 只挑出符合 id 的文件，並依 date desc, _id asc 排序
            picked: List[Tuple[ddate, Dict[str, Any]]] = []
            for doc in docs:
                _id = to_str_id(doc.get("_id"))
                if _id in ids:
                    d = try_parse_date(str(doc.get("date", "")))
                    if d:
                        picked.append((d, doc))
            picked.sort(key=lambda x: (
                x[0], to_str_id(x[1].get("_id"))), reverse=True)
            iterable: Iterable[Tuple[str, Dict[str, Any]]] = (
                (date_to_str(d), doc) for d, doc in picked
            )
        else:
            today = ddate.today()
            start = try_parse_date(
                args.start_date) if args.start_date else today
            if start is None:
                start = today
            until = try_parse_date(
                args.until_date) if args.until_date else None

            real_start = min(start, newest)
            LOGGER.info(
                "📅 由新至舊：start=%s（實際啟動=%s），until=%s（未提供則回退至最舊：%s）",
                date_to_str(start),
                date_to_str(real_start),
                (date_to_str(until) if until else "auto"),
                date_to_str(oldest),
            )
            iterable = iter_docs_by_date_index(
                index_by_date=index_by_date,
                start_date=real_start,
                until_date=until,
                oldest=oldest,
                newest=newest,
            )

        for idx, (ds, doc) in enumerate(iterable, start=1):
            doc_key = to_str_id(doc.get("_id"))
            if not doc_key:
                LOGGER.warning("⚠️ 第 %d 筆缺少 _id，已跳過。", idx)
                continue

            # 初遇該 _id 即在 CSV 建立 times=0（便於追蹤）
            if doc_key not in times_map:
                times_map[doc_key] = 0
                dump_kg_times(kg_times_csv, times_map)

            current_times = times_map.get(doc_key, 0)
            if current_times >= max_times:
                LOGGER.info(
                    "⏭️ [跳過] 第 %d 筆 _id=%s 已達 %d 次 (>= %d)",
                    idx,
                    doc_key,
                    current_times,
                    max_times,
                )
                skipped_by_times += 1
                continue

            title = str(doc.get("title", "")).strip()
            content = str(doc.get("content", "")).strip()
            LOGGER.info("🔍 [處理第 %d 筆] _id=%s，date=%s", idx, doc_key, ds)

            # 實體／關係抽取（含重試）
            text = f"日期: {ds}\n標題: {title}\n內容: {content}"
            extraction = None
            attempt = 0
            while extraction is None and attempt < MAX_RETRY_LLM:
                attempt += 1
                try:
                    extraction = extract_entities_relations(text)
                    if extraction is None:
                        LOGGER.warning("⚠️ 抽取回傳 None（第 %d 次），重試中…", attempt)
                except Exception as exc:
                    LOGGER.warning("⚠️ 抽取異常（第 %d 次）：%s", attempt, exc)

            if extraction is None:
                LOGGER.error("❌ 超過重試上限，放棄該筆：_id=%s", doc_key)
                continue

            # 轉換並寫入 Neo4j
            nodes, rels = transform_to_neo4j_format(extraction)
            for r in rels:
                r["doc_id"] = doc_key
                r["date"] = ds
            neo4j_loader.insert_data(nodes, rels)
            LOGGER.info("✅ 第 %d 筆成功寫入 Neo4j", idx)

            # 成功才累加，並立即落盤
            times_map[doc_key] = current_times + 1
            dump_kg_times(kg_times_csv, times_map)
            processed_ok += 1

    finally:
        neo4j_loader.close()
        LOGGER.info(
            "🌟 作業完成：成功處理 %d 筆，達上限而跳過 %d 筆",
            processed_ok,
            skipped_by_times,
        )


if __name__ == "__main__":
    run_with_timer(main)
