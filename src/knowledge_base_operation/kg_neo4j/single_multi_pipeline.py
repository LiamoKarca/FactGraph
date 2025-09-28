"""
$ python -m src.knowledge_base_operation.kg_neo4j.single_multi_pipeline \
  --workers 10 \
  --kg-max-times 1
  
測試用再補上：
  --kg-times-csv data/processed/knowledge-graph/my_times.csv\

📌 檔案用途
ETL：MongoDB(News.Real_News) → LLM 實體/關係抽取 → 轉換 → 寫入 Neo4j。
以 CSV（data/processed/knowledge-graph/kg_times.csv）追蹤每篇新聞的「已抽取次數」，
確保單篇不會被超過指定上限處理（避免浪費 API）。

⚙️ 並行與一致性保證（單程序多執行緒）
- 主執行緒「單點分派」新聞 → 天然避免同一篇被多執行緒搶到。
- 「在途預約」reservations：僅當 (已完成次數 + 在途預約) < 上限 時才會分派。
- 任務完成後「立即」序列化寫入 Neo4j 與 CSV（以鎖保護、CSV 原子覆寫）；
  多條執行緒同時完成時會依序排隊寫入，避免打架。
- 抽取失敗會自動撤銷預約，不增加次數。

🧪 預設
- 單工（--workers=1）以維持舊行為；設定 --workers>1 才會並行。
- 每篇最大處理次數（--kg-max-times）預設 3。
- 次數檔：data/processed/knowledge-graph/kg_times.csv（UTF-8-SIG 編碼）。

──────────────────────────────────────────────────────────────────────────────
# 使用方法（Examples）

## A. 依日期範圍處理
python -m src.knowledge_base_operation.kg_neo4j.single_multi_pipeline \
  --start-date 2025-09-16 \
  --until-date 2025-08-01 \
  --workers 8 \
  --kg-max-times 3
→ 從 2025-09-16 往舊處理到 2025-08-01；同時 8 執行緒；每篇最多抽取 3 次。

## B. 依 ID 清單處理
python -m src.knowledge_base_operation.kg_neo4j.single_multi_pipeline \
  --id-csv experiment/data/interim/to_process_ids.csv \
  --workers 6 \
  --kg-max-times 2
→ 僅處理 CSV 清單中的 _id；6 執行緒；每篇最多 2 次。

## C. 指定自訂次數檔
python -m src.knowledge_base_operation.kg_neo4j.single_multi_pipeline \
  --start-date 2025-09-20 \
  --kg-times-csv experiment/data/processed/my_times.csv \
  --workers 4
→ 從 2025-09-20 開始；改用 my_times.csv 記錄次數；4 執行緒。

──────────────────────────────────────────────────────────────────────────────
# 參數說明（Arguments）

--start-date YYYY-MM-DD
  起始日期（含），未指定則為今天。處理順序為由新到舊。

--until-date YYYY-MM-DD
  最舊處理日期（含）；未指定則回退到集合最舊日期。

--id-csv PATH
  只處理 CSV 清單中的 _id（可有表頭 id/_id 或無表頭單欄）。指定此參數時，
  會忽略日期回退邏輯，直接依清單處理。

--kg-times-csv PATH
  已處理次數記錄檔（CSV，UTF-8-SIG）。預設 data/processed/knowledge-graph/kg_times.csv。

--kg-max-times N
  每篇新聞最大抽取次數上限（達上限即跳過），預設 3。

--workers N
  抽取並行執行緒數；預設 1（單工）。I/O 型（LLM API 等）可設 6~12，
  請視機器資源與上游 API 限速調整。

──────────────────────────────────────────────────────────────────────────────
# 日常操作與觀察

- 觀察 log：
  🔍 [抽取] …   → 某篇開始進行抽取
  ✅ … times=K → 寫入 Neo4j 成功，且該篇計數更新為 K
  ⏭️ [跳過] …  → 已達/已預約至上限，直接跳過

- 進度可重跑：kg_times.csv 會持續更新；中斷重啟會依照已完成次數自動跳過。

- 效能控制：
  程式內以 workers×4 控制最多 in-flight 任務量，避免一次塞太多造成記憶體或 API 壓力。

- 安全性：
  CSV 採「原子覆寫」避免半寫入；寫入 Neo4j 與 CSV 由單一鎖序列化，避免競態。

──────────────────────────────────────────────────────────────────────────────
# 需求快速檢核
☑ 多執行緒並行抽取（--workers N）
☑ 同一篇新聞不會被多執行緒重複提取（主執行緒分派 + reservations 仲裁）
☑ 每篇完成後「立即」寫入計數，並序列化避免打架（write_lock + 原子覆寫）
☑ 每個執行緒完成後，自動接續尚未被他線程處理的下一篇
☑ 向下相容：不指定 --workers 即維持單工老行為
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
from datetime import date as ddate, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import threading

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
KG_TIMES_CSV_DEFAULT = "data/processed/knowledge-graph/kg_times.csv"  # 預設計數檔路徑
KG_MAX_TIMES_DEFAULT = 3  # 同一篇新聞的最大處理次數（依你的需求，可改參數）
MAX_RETRY_LLM = 5  # LLM 呼叫失敗重試次數上限
DEFAULT_WORKERS = 1  # 預設單工，向下相容
MAX_INFLIGHT_FACTOR = 4  # 控制最多同時排隊的任務量：workers * 4

# ── 執行緒級鎖，保障單程序內寫入序列化 ───────────────────────────────────────────
write_lock = threading.Lock()       # 控 Neo4j/CSV 寫入互斥
reservation_lock = threading.Lock()  # 控 reservations 與分派判斷互斥


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
    """嘗試解析字串為 date（支援 YYYY-MM-DD、YYYY/M/D 與前綴時間字串）。"""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
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


def _atomic_write_csv(csv_path: Path, rows: List[Tuple[str, int]]) -> None:
    """以臨時檔 + 原子覆寫，避免半寫入狀態。"""
    ensure_parent_dir(csv_path)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=csv_path.name, dir=str(csv_path.parent))
    with os.fdopen(tmp_fd, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "times"])
        for k, v in rows:
            w.writerow([k, v])
    os.replace(tmp_path, csv_path)  # 原子替換（同檔名 swap，不會有中間態）


def dump_kg_times(csv_path: Path, times_map: Dict[str, int]) -> None:
    """
    將 dict[id]=times 寫回 CSV（含表頭）。
    以 write_lock 確保單程序內互斥；原子覆寫確保不會破檔。
    """
    with write_lock:
        rows = [(k, times_map[k]) for k in sorted(times_map.keys())]
        _atomic_write_csv(csv_path, rows)


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


def query_docs_for_day(coll, day: ddate, projection: Optional[Dict[str, int]] = None):
    """
    回傳整天的文件 Cursor。兼顧兩種 date 型別：
    - datetime：用 $gte/$lt 區間
    - 字串：用 ^YYYY[-/]M{1,2}[-/]D{1,2}$ 的 regex 匹配
    """
    start_dt = datetime.combine(day, time.min)
    end_dt = datetime.combine(day + timedelta(days=1), time.min)
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
        for doc in cursor:
            yield ds, doc
        day -= timedelta(days=1)


# ── 主流程 ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ETL for News.Real_News -> Neo4j with times control "
            "and newest-to-oldest date ordering (threaded)"
        )
    )
    parser.add_argument("--id-csv", type=str, default=None,
                        help="只處理 CSV 中列出的 _id（含 'id' / '_id' 表頭或純第一欄）。")
    parser.add_argument("--start-date", type=str, default=None,
                        help="起始日期（含），預設為今天。格式 YYYY-MM-DD。")
    parser.add_argument("--until-date", type=str, default=None,
                        help="最舊處理日期（含）。未提供則回退到集合最舊日期為止。")
    parser.add_argument("--kg-times-csv", type=str, default=KG_TIMES_CSV_DEFAULT,
                        help="計數檔路徑（CSV：id,times）。")
    parser.add_argument("--kg-max-times", type=int, default=KG_MAX_TIMES_DEFAULT,
                        help="同一篇新聞的最大處理次數（達上限即跳過）。")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="LLM 抽取的並行 worker 數；1 表示單工（維持舊行為）。")
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
    times_map = load_kg_times(kg_times_csv)  # 已完成次數
    reservations: Dict[str, int] = {}        # 在途預約次數（尚未完成的分派）
    max_times = int(args.kg_max_times)
    workers = max(1, int(args.workers))
    max_inflight = workers * MAX_INFLIGHT_FACTOR

    LOGGER.info("🧮 計數檔：%s；上限：%d；workers=%d", kg_times_csv, max_times, workers)

    processed_ok = 0
    skipped_by_times = 0
    projection = {"_id": 1, "date": 1, "title": 1, "content": 1}

    # ── 在途與分派控制 ────────────────────────────────────────────────────────────
    def can_reserve(doc_key: str) -> bool:
        """是否允許分派：times + reservations < max_times。"""
        with reservation_lock:
            t = times_map.get(doc_key, 0)
            r = reservations.get(doc_key, 0)
            return (t + r) < max_times

    def reserve(doc_key: str) -> None:
        """標記在途預約 +1（只改記憶體，不落盤）。"""
        with reservation_lock:
            reservations[doc_key] = reservations.get(doc_key, 0) + 1

    def unreserve(doc_key: str) -> None:
        """在途預約 -1。"""
        with reservation_lock:
            if reservations.get(doc_key, 0) > 0:
                reservations[doc_key] -= 1
                if reservations[doc_key] == 0:
                    reservations.pop(doc_key, None)

    def commit_success(doc_key: str) -> int:
        """
        成功完成一篇後，序列化增加已完成次數 +1，立即落盤 CSV。
        此步在 write_lock 保護下進行。回傳最新 times。
        """
        with write_lock:
            times_map[doc_key] = times_map.get(doc_key, 0) + 1
            rows = [(k, times_map[k]) for k in sorted(times_map.keys())]
            _atomic_write_csv(kg_times_csv, rows)  # 完成即落盤（需求指定）
            return times_map[doc_key]

    # ── 工作執行緒：抽取→轉換（不寫 Neo4j / 不改 CSV） ───────────────────────────
    def worker_extract(idx: int, ds: str, doc: Dict[str, Any]):
        """
        成功回傳 (idx, doc_key, mongo_id, ds, nodes, rels)；失敗回傳 None。
        """
        mongo_id = doc.get("_id")
        doc_key = to_str_id(mongo_id)
        title = str(doc.get("title", "")).strip()
        content = str(doc.get("content", "")).strip()
        LOGGER.info("🔍 [抽取] 第 %d 筆 _id=%s，date=%s", idx, doc_key, ds)

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
            return None

        nodes, rels = transform_to_neo4j_format(extraction)
        for r in rels:
            r["doc_id"] = mongo_id
            r["date"] = ds
        return (idx, doc_key, mongo_id, ds, nodes, rels)

    try:
        # 準備資料來源 iterable
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

            oldest, newest = get_available_date_bounds(coll)
            if oldest is None or newest is None:
                LOGGER.info("📭 集合內沒有可解析的日期資料，無事可做。")
                return

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

        # 執行緒池 + 有界 in-flight 管控（用 wait(FIRST_COMPLETED) 取代易丟錯的 as_completed+timeout）
        with ThreadPoolExecutor(max_workers=workers) as ex:
            running: set = set()  # 目前 in-flight 的 futures
            submit_idx = 0

            def drain_done(done_set):
                """
                主執行緒：消化完成的 future，依序做
                Neo4j 寫入 → times +1 → 撤銷預約 → 記錄成功筆數。
                """
                nonlocal processed_ok
                for fut in done_set:
                    running.discard(fut)
                    result = fut.result()
                    if result is None:
                        # 失敗：撤銷預約
                        unreserve_key = getattr(fut, "_doc_key", None)
                        if unreserve_key:
                            unreserve(unreserve_key)
                        continue
                    idx, done_key, mongo_id_done, ds_done, nodes, rels = result
                    # 寫入 Neo4j + 計數 + 落盤（序列化）
                    neo4j_loader.insert_data(nodes, rels)
                    new_times = commit_success(done_key)
                    # 完成即撤銷預約 1
                    unreserve(done_key)
                    processed_ok += 1
                    LOGGER.info("✅ 第 %d 筆成功寫入 Neo4j；_id=%s；times=%d",
                                idx, done_key, new_times)

            # 逐篇迭代分派
            for ds, doc in iterable:
                mongo_id = doc.get("_id")
                doc_key = to_str_id(mongo_id)
                submit_idx += 1

                if not doc_key:
                    LOGGER.warning("⚠️ 第 %d 筆缺少 _id，已跳過。", submit_idx)
                    continue

                # 初遇即建立 times_map 條目（不落盤）
                if doc_key not in times_map:
                    times_map[doc_key] = 0

                # 達上限/已滿預約 → 直接跳過
                if not can_reserve(doc_key):
                    skipped_by_times += 1
                    LOGGER.info(
                        "⏭️ [跳過] 第 %d 筆 _id=%s，已達/已預約至上限（times=%d, reservations=%d, max=%d）",
                        submit_idx, doc_key, times_map.get(
                            doc_key, 0), reservations.get(doc_key, 0), max_times
                    )
                    continue

                # 有界 in-flight：若已達上限，等待至少一個完成（不丟例外）
                while len(running) >= max_inflight:
                    done, _not_done = wait(
                        running, timeout=1.0, return_when=FIRST_COMPLETED)
                    if not done:
                        # 沒有完成者，代表大家都還在跑；繼續等
                        continue
                    drain_done(done)

                # 正式預約並提交
                reserve(doc_key)
                future = ex.submit(worker_extract, submit_idx, ds, doc)
                setattr(future, "_doc_key", doc_key)  # 失敗時回收預約用
                running.add(future)

            # 迭代結束後，收尾等待所有 in-flight 完成
            if running:
                done, _ = wait(running, return_when=ALL_COMPLETED)
                drain_done(done)

    finally:
        neo4j_loader.close()
        client.close()
        LOGGER.info(
            "🌟 作業完成：成功處理 %d 筆，達上限/預約上限而跳過 %d 筆",
            processed_ok,
            skipped_by_times,
        )
        # 🔔 額外提示：讓操作者一眼知道已完成，並且提醒次數檔位置與 workers 設定
        LOGGER.info("🔔 提示：本次使用 workers=%d；次數檔已更新 → %s", workers, kg_times_csv)
        LOGGER.info(
            "📄 你可用任何表格工具開啟（UTF-8-SIG）：id,times 兩欄；確認指定 _id 的 times 已正確遞增。",)


if __name__ == "__main__":
    run_with_timer(main)
