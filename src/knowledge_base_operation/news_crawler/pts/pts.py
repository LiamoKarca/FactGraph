"""
PTS（公視）文章爬蟲：增量更新 + 頁末即時存檔 + 連續重複終止 + CLI 起始頁
======================================================================

功能總覽
--------
1) 根列表頁： https://news.pts.org.tw/category/1
   實際抓取頁： https://news.pts.org.tw/category/1?page={n}
2) 預設翻頁範圍 1..100，可用 CLI 指定起始頁（--start）與上限（--max）。
3) 增量更新：
   - 讀取 data/raw/news/pts/pts_news.json 作為舊資料（若存在）
   - 新抓資料與舊資料以 id 去重合併
   - 每「頁」處理結束立即寫回 pts_news.json（避免當機全失）
4) 早停條件：
   - 連續 3 則「重複新聞」（以 URL 衍生之 id 判定）→ 立即終止整體爬取
   - 解析失敗不算重複，會中斷「連續」計數
   
5) 指令列參數：
   - 由第 1 頁開始，最多到第 100 頁 $python src/knowledge_base_operation/news_crawler/pts/pts.py
   - 從第 37 頁開始，最多到第 100 頁 $python src/knowledge_base_operation/news_crawler/pts/pts.py --start 37
   - 從第 20 頁開始到第 60 頁 $python src/knowledge_base_operation/news_crawler/pts/pts.py --start 20 --max 60

資料欄位
--------
date, publisher, category, title, content, label, url, id
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

# ──────────────────────────────
# 常數與路徑設定
# ──────────────────────────────

BASE_CATEGORY_URL: str = "https://news.pts.org.tw/category/1"
PAGED_CATEGORY_URL: str = "https://news.pts.org.tw/category/1?page={}"

# 預設最大翻頁上限（需求指定到 100）
DEFAULT_MAX_PAGES: int = 100

# 固定輸出（增量合併後覆寫寫入，但資料為累積結果）
OUTPUT_DIR: Path = Path("data/raw/news/pts")
OUTPUT_FILE: Path = OUTPUT_DIR / "pts_news.json"

# 請求標頭：模擬常見瀏覽器，降低被擋風險
HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
    "Connection": "keep-alive",
}

# 若需啟用內容過濾可設定關鍵字（空字串代表不過濾）
KEYWORD: str = ""

# 台北時區（顯示用）
TPE_TZ: ZoneInfo = ZoneInfo("Asia/Taipei")

# 文字清理用正則：移除多餘符號與連續空白
_CLEAN_PAT = re.compile(r'[\n"「」：；:;,{}\[\]]|\s{2,}', flags=re.UNICODE)


# ──────────────────────────────
# 基礎工具
# ──────────────────────────────
def clean_text(text: str) -> str:
    """
    將段落中的多餘符號、連續空白移除並修剪首尾空白。

    設計理由：
    - 原始 HTML 文字常有換行與中英文標點參雜，先行正規化可提升一致性。
    """
    return _CLEAN_PAT.sub(" ", text).strip()


def parse_date(date_text: str) -> str:
    """
    嘗試將多種格式的日期字串轉為 YYYY-MM-DD；失敗時回傳空字串。

    支援的常見格式（可按需擴充）：
    - 2025/08/11, 2025-08-11, 2025年08月11日, 2025.08.11
    - 含時間者會僅取日期部分（例如 2025/08/11 12:30）
    """
    date_text = clean_text(date_text)
    date_formats: Tuple[str, ...] = (
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y年%m月%d日",
        "%Y.%m.%d",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
    )
    for fmt in date_formats:
        try:
            cleaned = date_text.split(" ")[0]
            dt_obj = datetime.strptime(cleaned, fmt)
            return dt_obj.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def make_article_id(url: str) -> str:
    """
    由 URL 衍生穩定短 ID：'PTS_' + md5(url) 前 8 碼。

    理念：
    - 同一篇文章的 URL 應該唯一，因此以其作為主鍵可天然去重。
    """
    return f"PTS_{md5(url.encode('utf-8-sig')).hexdigest()[:8]}"


# ──────────────────────────────
# HTTP 與 HTML 解析
# ──────────────────────────────
def fetch_html(url: str, session: requests.Session) -> Optional[str]:
    """
    以 GET 取得 HTML；成功回傳字串，失敗回傳 None。

    風險控管：
    - 設定 timeout 避免長時間掛住。
    - 非 200 狀態碼直接視為失敗，不嘗試解析錯誤頁。
    """
    try:
        res = session.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            res.encoding = "utf-8-sig"
            return res.text
        print(f"❌ HTTP {res.status_code} → {url}")
    except requests.RequestException as exc:
        print(f"❌ 請求錯誤：{exc} → {url}")
    return None


def extract_links(list_html: str) -> List[str]:
    """
    從分類列表頁 HTML 擷取所有文章連結。

    選擇器策略：
    - 使用 'h2 a[href*="/article/"]'
    - 相對路徑補上 https://news.pts.org.tw
    """
    soup = BeautifulSoup(list_html, "html.parser")
    anchors = soup.select('h2 a[href*="/article/"]')
    links: List[str] = []
    for a in anchors:
        href = a.get("href") or ""
        if not href:
            continue
        if href.startswith("/article/"):
            links.append("https://news.pts.org.tw" + href)
        elif href.startswith("https://news.pts.org.tw/article/"):
            links.append(href)
    return links


def parse_article(url: str, session: requests.Session) -> Optional[Dict]:
    """
    解析單篇新聞頁面，回傳結構化資料；失敗回傳 None。

    萃取欄位說明：
    - title：優先 'h1.article-title'，退而求其次 'h1'
    - date：嘗試 'time' 或 'span.date'，多格式解析
    - category：嘗試自 'div.news-info' 文字推斷（常見格式含發布者｜分類）
    - content：自 'div.post-article.text-align-left' 取全文，依 '。' 切句後重組
    """
    html = fetch_html(url, session)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # 標題
    title_tag = soup.select_one("h1.article-title") or soup.select_one("h1")
    title = clean_text(title_tag.text) if title_tag else ""
    if not title:
        print("⚠️ 無標題，跳過")
        return None

    # 日期
    date_tag = soup.select_one("time") or soup.select_one("span.date")
    date_raw = date_tag.text if date_tag else ""
    date_str = parse_date(date_raw)

    # 分類（從資訊列推斷）
    category = "未知分類"
    info_div = soup.select_one("div.news-info")
    if info_div:
        all_txt = clean_text(info_div.get_text())
        if "|" in all_txt:
            parts = [p.strip() for p in all_txt.split("|") if p.strip()]
            if len(parts) >= 2:
                category = parts[1]
            elif parts:
                category = parts[0]
        elif all_txt:
            category = all_txt

    # 內文
    content_div = soup.select_one("div.post-article.text-align-left")
    content_raw = clean_text(content_div.get_text()) if content_div else ""
    sentences = [s.strip() for s in content_raw.split("。") if s.strip()]
    content = "\n".join(f"{s}。" for s in sentences)

    item = {
        "date": date_str,
        "publisher": "pts",
        "category": category,
        "title": title,
        "content": content,
        "label": True,
        "url": url,
        "id": make_article_id(url),
    }

    # 可選的關鍵字過濾（KEYWORD 非空時啟用）
    if KEYWORD and (KEYWORD not in (item["title"] + item["content"])):
        return None

    return item


# ──────────────────────────────
# 增量更新：讀舊檔、合併去重、寫回
# ──────────────────────────────
def load_existing_records(file_path: Path) -> List[Dict]:
    """
    讀取既有的 pts_news.json；若不存在或格式錯誤，回傳空清單。
    """
    if not file_path.exists():
        return []
    try:
        with file_path.open("r", encoding="utf-8-sig") as fp:
            data = json.load(fp)
            return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠️ 舊檔讀取失敗：{exc} → {file_path}")
        return []


def merge_deduplicate(
    old_items: Iterable[Dict],
    new_items: Iterable[Dict],
) -> List[Dict]:
    """
    將舊資料與新資料合併並去重；同 ID 時「保留舊資料版本」。

    理念：
    - 舊資料可能已經過人工修訂或清洗，保留舊版本較為穩健。
    """
    merged: List[Dict] = []
    seen: Set[str] = set()

    # 先放舊資料
    for it in old_items:
        _id = it.get("id")
        if not _id or _id in seen:
            continue
        merged.append(it)
        seen.add(_id)

    # 再補入新資料中尚未出現的
    for it in new_items:
        _id = it.get("id")
        if not _id or _id in seen:
            continue
        merged.append(it)
        seen.add(_id)

    return merged


def atomic_save_json(file_path: Path, data: List[Dict]) -> None:
    """
    採用「先寫臨時檔 → 原子轉名」的方式寫入 JSON，降低當機時檔案毀損風險。
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8-sig") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    tmp_path.replace(file_path)


# ──────────────────────────────
# 主流程
# ──────────────────────────────
def scrape_pts(start_page: int, max_pages: int) -> None:
    """
    PTS 爬蟲進入點。

    具體流程：
    1) 讀取既有檔案（若存在）→ 取得舊資料與其 ID 集合
    2) 自 page=start_page 起依序抓取至 page=max_pages
    3) 逐連結：
       - 先以 URL 計算 id，若 id 已存在（舊或本輪已新增）→ 記一次「重複」
       - 若「連續重複」達 3 次 → 視為已無新料，立即終止整體流程
       - 否則解析文章並加入結果集；計數重置
    4) **每頁結束**即與舊資料合併去重並寫回固定檔案（避免當機全失）
    """
    if start_page < 1:
        print(f"⚠️ 起始頁 {start_page} 無效，改用 1")
        start_page = 1
    if max_pages < start_page:
        print(f"⚠️ max_pages({max_pages}) < start_page({start_page})，不執行")
        return

    start = datetime.now(TPE_TZ)
    print("▶ 開始：", start.strftime("%Y-%m-%d %H:%M:%S"))
    print("◆ 根列表頁：", BASE_CATEGORY_URL)
    print(f"◆ 翻頁範圍：{start_page}..{max_pages}（或連續 3 重複即提前終止）")

    # 1) 讀取舊資料 → 轉成以 id 為鍵的字典以利快速判重與累積
    existing_list = load_existing_records(OUTPUT_FILE)
    records_by_id: Dict[str, Dict] = {}
    for it in existing_list:
        _id = it.get("id")
        if _id and _id not in records_by_id:
            records_by_id[_id] = it

    print(f"📦 舊資料筆數：{len(records_by_id)}")

    session = requests.Session()
    consecutive_dups = 0
    stop = False

    # 2) 逐頁抓取
    for page in range(start_page, max_pages + 1):
        if stop:
            break

        page_url = PAGED_CATEGORY_URL.format(page)
        print(f"\n🔍 解析第 {page} 頁 → {page_url}")

        list_html = fetch_html(page_url, session)
        if not list_html:
            print("⚠️ 列表頁抓取失敗，跳過此頁")
            # 即便此頁失敗，也在頁末保存目前累積（維持「每頁結束就存」精神）
            atomic_save_json(OUTPUT_FILE, list(records_by_id.values()))
            continue

        links = extract_links(list_html)
        print(f"✅ 取得連結數：{len(links)}")

        # 逐連結處理
        for idx, article_url in enumerate(links, start=1):
            art_id = make_article_id(article_url)

            # 先以 id 判重（避免無謂請求）
            if art_id in records_by_id:
                consecutive_dups += 1
                print(
                    f"  ↪︎ ({idx}/{len(links)}) 重複（{consecutive_dups}/3）："
                    f"{article_url}"
                )
                if consecutive_dups >= 3:
                    print("🛑 連續 3 則重複，終止爬取")
                    stop = True
                    break
                continue

            # 確認為新文章 → 解析內容
            item = parse_article(article_url, session)
            if not item:
                # 解析失敗不算重複，重置連續重複計數
                consecutive_dups = 0
                continue

            # 關鍵字過濾若啟用，parse_article 已處理返回 None
            records_by_id[item["id"]] = item

            # 新文章出現 → 連續重複歸零
            consecutive_dups = 0

            print(f"  ➜ ({idx}/{len(links)}) 新增：{item['title']}")
            time.sleep(2)  # 禮貌性延遲，降低對方伺服負載

        # 3) 頁末立即存檔（即便本頁完全沒新增也會覆寫一次，確保狀態一致）
        atomic_save_json(OUTPUT_FILE, list(records_by_id.values()))
        print(f"💾 已存檔：{OUTPUT_FILE.resolve()}（累計 {len(records_by_id)} 筆）")

    end = datetime.now(TPE_TZ)
    print("\n🎉 完成")
    print("📝 輸出檔案：", OUTPUT_FILE.resolve())
    print("📊 最終總筆數：", len(records_by_id))
    print("⏱️ 耗時（秒）：", f"{(end - start).total_seconds():.1f}")


# ──────────────────────────────
# CLI 入口
# ──────────────────────────────
def parse_args() -> argparse.Namespace:
    """
    解析命令列參數。
    - --start / -s：指定起始頁（預設 1）
    - --max / -m：指定最大頁（預設 100）
    """
    parser = argparse.ArgumentParser(
        description="PTS（公視）分類新聞爬蟲（增量更新／頁末即時存檔／連續重複早停）"
    )
    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=1,
        help="起始頁（預設：1）",
    )
    parser.add_argument(
        "--max",
        "-m",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f"最大頁（預設：{DEFAULT_MAX_PAGES}）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scrape_pts(start_page=args.start, max_pages=args.max)
