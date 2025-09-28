"""
ETL：MongoDB (News.Real_News) → 實體/關係抽取 → 轉 Neo4j 寫入。
以 CSV（data/processed/knowledge-graph/kg_times.csv）追蹤每篇新聞已抽取次數；
同一篇達上限（預設 3 次）即跳過以節省 API。

處理順序（預設）：
- 以「今天」為起點，若當日無新聞，會自動往前一天繼續，直到最舊有資料的日期。
- 支援 date 欄位為「BSON datetime」或「字串日期（YYYY-MM-DD / YYYY/M/D）」。

可選項：
- --id-csv           僅處理 CSV 中列出的 _id（忽略日期回退邏輯）
- --start-date       起始日期（含），預設今天（YYYY-MM-DD）
- --until-date       最舊處理日期（含）；不提供則回退到集合最舊日期
- --kg-times-csv     次數記錄檔路徑（預設 data/processed/knowledge-graph/kg_times.csv）
- --kg-max-times     同一篇的最大處理次數（預設 3）

使用範例，起訖日期：
python -m src.knowledge_base_operation.kg_neo4j.pipeline \
  --start-date 2025-09-16 --until-date 2025-08-01
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import date as ddate, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

# ── 路徑自動修正：以「直接執行檔案」啟動時，將專案根加入 sys.path ─────────────
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]  # <PROJECT_ROOT>
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── 兼容式匯入（相對→失敗則絕對） ───────────────────────────────────────────────
try:
    from ...common.gadget import LOGGER, run_with_timer  # type: ignore
except Exception:  # pragma: no cover
    from src.common.gadget import LOGGER, run_with_timer  # type: ignore

try:
    from .extraction import extract_entities_relations  # type: ignore
    from .neo4j_loader import Neo4jLoader  # type: ignore
    from .transformation import transform_to_neo4j_format  # type: ignore
except Exception:  # pragma: no cover
    from src.knowledge_base_operation.kg_neo4j.extraction import (  # type: ignore
        extract_entities_relations,
    )
    from src.knowledge_base_operation.kg_neo4j.neo4j_loader import (  # type: ignore
        Neo4jLoader,
    )
    from src.knowledge_base_operation.kg_neo4j.transformation import (  # type: ignore
        transform_to_neo4j_format,
    )

# ── 常數 ───────────────────────────────────────────────────────────────────────────
KG_TIMES_CSV_DEFAULT = "data/processed/knowledge-graph/kg_times.csv" # 預設計數檔路徑
KG_MAX_TIMES_DEFAULT = 1 # 同一篇新聞的最大處理次數
MAX_RETRY_LLM = 5 # LLM 呼叫失敗重試次數上限


# ── 小工具 ─────────────────────────────────────────────────────────────────────────
def ensure_parent_dir(path: Path) -> None:
    """確保路徑的父資料夾存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def mongo_client_from_env() -> MongoClient:
    """從環境變數 MONGODB_URI 建立 MongoClient。"""
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("環境變數 MONGODB_URI 未設置，無法連線 MongoDB。")
    return MongoClient(mongo_uri)


def to_str_id(mongo_id: Any) -> str:
    """將 Mongo 的 _id 安全轉字串（含 ObjectId 或自定字串）。"""
    return "" if mongo_id is None else str(mongo_id)


def try_parse_date(s: str) -> Optional[ddate]:
    """嘗試解析字串為 date（支援 YYYY-MM-DD、YYYY/M/D）。"""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # 嘗試剝掉尾端時間（像 '2025-04-09T12:34:00Z'）
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


def read_ids_from_csv(csv_path: str) -> List[Any]:
    """
    自 CSV 讀取一批 _id：
    - 含表頭：欄名 'id' 或 '_id'
    - 無表頭：取第 1 欄
    若值像 ObjectId 會轉為 ObjectId；否則保留字串。
    """
    ids: List[Any] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
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
                    if not raw:
                        continue
                    try:
                        ids.append(ObjectId(raw))
                    except Exception:
                        ids.append(raw)
                return ids

        f.seek(0)
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            raw = str(row[0]).strip()
            if not raw:
                continue
            try:
                ids.append(ObjectId(raw))
            except Exception:
                ids.append(raw)
    return ids


def load_kg_times(csv_path: Path) -> Dict[str, int]:
    """讀取 kg_times.csv 為 dict[id]=times；若檔案不存在則回傳空 dict。"""
    if not csv_path.exists():
        return {}
    times_map: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
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
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "times"])
        for k in sorted(times_map.keys()):
            writer.writerow([k, times_map[k]])


# ── 日期邊界與查詢 ────────────────────────────────────────────────────────────────
def normalize_to_date(obj: Any) -> Optional[ddate]:
    """將可能為 datetime/date/str 的值轉為 date。"""
    if obj is None:
        return None
    if isinstance(obj, datetime):
        return obj.date()
    if isinstance(obj, ddate):
        return obj
    if isinstance(obj, str):
        return try_parse_date(obj)
    return None


def get_available_date_bounds(coll) -> Tuple[Optional[ddate], Optional[ddate]]:
    """
    用 distinct('date') 取得所有可解析的日期值，計算實際的最舊/最新日期。
    這比單純 sort('date') 更穩健（避免字串格式不一致）。
    """
    values = coll.distinct("date")
    parsed: List[ddate] = []
    for v in values:
        d = normalize_to_date(v)
        if d:
            parsed.append(d)
    if not parsed:
        return None, None
    return min(parsed), max(parsed)


def build_day_regex(day: ddate) -> str:
    """
    產生能匹配 'YYYY-MM-DD' 與 'YYYY/M/D' 的正則：
    e.g., 2025-09-07 → ^2025[-/]0?9[-/]0?7$
    """
    y = day.year
    m = day.month
    d = day.day
    return rf"^{y}[-/]{m:02d}|{y}[-/]{m}[-/]{d:02d}$"  # 避免誤配再做保守重寫


def query_docs_for_day(coll, day: ddate, projection: Optional[Dict[str, int]] = None):
    """
    回傳整天的文件 Cursor。兼顧兩種 date 型別：
    - datetime：用 $gte/$lt 區間
    - 字串：用 ^YYYY[-/]M{1,2}[-/]D{1,2}$ 的 regex 匹配
    """
    start_dt = datetime.combine(day, time.min)
    end_dt = datetime.combine(day + timedelta(days=1), time.min)

    # 正則需嚴謹些，避免誤配（使用完整錨點）
    y, m, d = day.year, day.month, day.day
    pattern = rf"^{y}[-/](0?{m})[-/](0?{d})$"

    query = {
        "$or": [
            {"date": {"$gte": start_dt, "$lt": end_dt}},
            {"date": {"$regex": pattern}},
        ]
    }
    return coll.find(query, projection=projection).sort("_id", ASCENDING)


def iter_docs_by_date(
    coll,
    start_date: ddate,
    until_date: Optional[ddate] = None,
    projection: Optional[Dict[str, int]] = None,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """
    由 start_date 起，逐日往舊遞減產出 (date_str, doc)。
    - 若 until_date 提供，當日 < until_date 即停止；
      否則會退到集合中實際最舊日期。
    - 無論某天是否有新聞，都不會中斷；會繼續往前日。
    """
    oldest, newest = get_available_date_bounds(coll)
    if oldest is None or newest is None:
        return

    start = min(start_date, newest)
    lower_bound = until_date if until_date is not None else oldest
    lower_bound = max(lower_bound, oldest)

    day = start
    while day >= lower_bound:
        ds = date_to_str(day)
        cursor = query_docs_for_day(coll, day, projection=projection)
        any_yield = False
        for doc in cursor:
            any_yield = True
            yield ds, doc
        # 即便沒有文件，也會往前一天，直到 lower_bound
        day -= timedelta(days=1)


# ── 主流程 ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ETL for News.Real_News -> Neo4j with times control "
            "and newest-to-oldest date ordering"
        )
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
        help="最舊處理日期（含）。未提供則回退到集合最舊日期為止。",
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

    # 連線 MongoDB
    load_dotenv()
    client = mongo_client_from_env()
    db = client["News"]
    coll = db["Real_News"]
    LOGGER.info("🔗 已連線 MongoDB：News.Real_News")

    # Neo4j 連線
    neo4j_loader = Neo4jLoader()

    # 計數檔
    kg_times_csv = Path(args.kg_times_csv).resolve()
    times_map = load_kg_times(kg_times_csv)
    max_times = int(args.kg_max_times)
    LOGGER.info("🧮 計數檔：%s；上限：%d", kg_times_csv, max_times)

    processed_ok = 0
    skipped_by_times = 0
    projection = {"_id": 1, "date": 1, "title": 1, "content": 1}

    try:
        if args.id_csv:
            ids = read_ids_from_csv(args.id_csv)
            LOGGER.info("🗂 以 CSV 中的 %d 筆 _id 進行查詢（忽略日期回退）", len(ids))
            cursor = coll.find({"_id": {"$in": ids}}, projection=projection).sort(
                [("date", DESCENDING), ("_id", ASCENDING)]
            )
            iterable: Iterable[Tuple[str, Dict[str, Any]]] = (
                (str(doc.get("date", "")), doc) for doc in cursor
            )
        else:
            today = ddate.today()
            start = try_parse_date(
                args.start_date) if args.start_date else today
            if start is None:
                start = today
            until = try_parse_date(
                args.until_date) if args.until_date else None

            # 真實可用邊界（避免資料格式差異）
            oldest, newest = get_available_date_bounds(coll)
            if oldest is None or newest is None:
                LOGGER.info("📭 集合內沒有可解析的日期資料，無事可做。")
                return

            # 起點以實際最新日期為上限
            real_start = min(start, newest)
            LOGGER.info(
                "📅 由新至舊：start=%s（實際啟動=%s），until=%s（未提供則回退至集合最舊：%s）",
                date_to_str(start),
                date_to_str(real_start),
                (date_to_str(until) if until else "auto"),
                date_to_str(oldest),
            )
            iterable = iter_docs_by_date(
                coll=coll,
                start_date=real_start,
                until_date=until,
                projection=projection,
            )

        for idx, (ds, doc) in enumerate(iterable, start=1):
            mongo_id = doc.get("_id")
            doc_key = to_str_id(mongo_id)
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
                r["doc_id"] = mongo_id
                r["date"] = ds
            neo4j_loader.insert_data(nodes, rels)
            LOGGER.info("✅ 第 %d 筆成功寫入 Neo4j", idx)

            # 成功才累加，並立即落盤
            times_map[doc_key] = current_times + 1
            dump_kg_times(kg_times_csv, times_map)
            processed_ok += 1

    finally:
        neo4j_loader.close()
        client.close()
        LOGGER.info(
            "🌟 作業完成：成功處理 %d 筆，達上限而跳過 %d 筆",
            processed_ok,
            skipped_by_times,
        )


if __name__ == "__main__":
    run_with_timer(main)
