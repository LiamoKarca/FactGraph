"""
PTS (公視) 文章爬蟲
===================

爬取分類頁 (/category/1) 前 `MAX_PAGES` 頁的所有文章。
輸出一支 JSON 檔：
    FactGraph/data/raw/news/pts_<timestamp>.json
    timestamp = yyyymmddHHMM (Asia/Taipei)

每則文章字段：
    date, publisher, category, title, content, label
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

# ──────────────────────────────
# 參數／常數
# ──────────────────────────────
KEYWORD: str = "京華城"  # 如需關鍵字過濾，可自行使用
BASE_URL: str = "https://news.pts.org.tw"
SEARCH_URL: str = f"{BASE_URL}/category/1?page={{}}"

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
    "Connection": "keep-alive",
}

MAX_PAGES: int = 1
TPE_TZ = ZoneInfo("Asia/Taipei")
OUTPUT_DIR = Path("FactGraph/data/raw/news/pts/")

# ──────────────────────────────
# 工具函式
# ──────────────────────────────
_CLEAN_PAT = re.compile(
    r'[\n"「」：；:;,{}\[\]]|\s{2,}',
    flags=re.UNICODE,
)


def clean_text(text: str) -> str:
    """移除多餘符號與空白。"""
    return _CLEAN_PAT.sub(" ", text).strip()


def parse_date(date_text: str) -> str:
    """將各式日期字串轉成 YYYY-MM-DD。失敗則回傳空字串。"""
    date_text = clean_text(date_text)
    date_formats = (
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y年%m月%d日",
        "%Y.%m.%d",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
    )
    for fmt in date_formats:
        try:
            # 去掉時間，僅保留日期部分
            cleaned = date_text.split(" ")[0]
            dt_obj = datetime.strptime(cleaned, fmt)
            return dt_obj.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def make_article_id(url: str) -> str:
    """生成簡短且唯一的文章 ID（8 位 md5）。"""
    return f"PTS_{md5(url.encode()).hexdigest()[:8]}"


# ──────────────────────────────
# 文章解析
# ──────────────────────────────
def fetch_html(url: str, session: requests.Session) -> Optional[str]:
    """GET 請求，成功回傳 HTML；失敗回傳 None。"""
    try:
        res = session.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            res.encoding = "utf-8"
            return res.text
        print(f"❌ 失敗 {res.status_code} → {url}")
    except requests.RequestException as exc:
        print(f"❌ 請求錯誤：{exc} → {url}")
    return None


def parse_article(url: str, session: requests.Session) -> Optional[dict]:
    """擷取單篇新聞資料。失敗回傳 None。"""
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

    # 分類
    category = "未知分類"
    info_div = soup.select_one("div.news-info")
    if info_div:
        all_txt = clean_text(info_div.get_text())
        if "|" in all_txt:
            category = all_txt.split("|")[1].strip()
        else:
            category = all_txt or category

    # 內容
    content_div = soup.select_one("div.post-article.text-align-left")
    content_raw = clean_text(content_div.get_text()) if content_div else ""
    sentences = [s for s in content_raw.split("。") if s.strip()]
    content = "\n".join(f"{s}。" for s in sentences)

    return {
        "date": date_str,
        "publisher": "pts",
        "category": category,
        "title": title,
        "content": content,
        "label": True,
        "url": url,  # 若不需存 URL，可刪除此行
        "id": make_article_id(url),
    }


# ──────────────────────────────
# 搜尋頁解析
# ──────────────────────────────
def extract_links(html: str) -> List[str]:
    """從列表頁拿出所有文章連結。"""
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.select('h2 a[href*="/article/"]')
    links: list[str] = []
    for a in anchors:
        href = a.get("href", "")
        if href.startswith("/article/"):
            links.append(BASE_URL + href)
        elif href.startswith("https://news.pts.org.tw/article/"):
            links.append(href)
    return links


# ──────────────────────────────
# 主流程
# ──────────────────────────────
def scrape_pts(max_pages: int = MAX_PAGES) -> None:
    """PTS 爬蟲入口。"""
    start = datetime.now(TPE_TZ)
    print("開始時間:", start.strftime("%Y-%m-%d %H:%M:%S"))

    session = requests.Session()
    results: list[dict] = []
    seen_ids: set[str] = set()

    for page in range(1, max_pages + 1):
        print(f"\n🔍 解析第 {page} 頁")
        list_html = fetch_html(SEARCH_URL.format(page), session)
        if not list_html:
            break

        links = extract_links(list_html)
        print(f"✅ 找到 {len(links)} 則連結")

        for idx, article_url in enumerate(links, 1):
            print(f"  ➡️ ({idx}/{len(links)}) {article_url}")
            item = parse_article(article_url, session)
            if not item:
                continue
            if item["id"] in seen_ids:
                print("   ↪︎ 重複，跳過")
                continue
            # 若需關鍵字過濾，可在此判斷
            # if KEYWORD not in item["content"]:
            #     continue
            results.append(item)
            seen_ids.add(item["id"])
            time.sleep(2)  # 禮貌延遲

    # ── 輸出 JSON ─────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = start.strftime("%Y%m%d%H%M")
    out_path = OUTPUT_DIR / f"pts_{ts}.json"

    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    end = datetime.now(TPE_TZ)
    print("\n🎉 完成！")
    print("輸出檔案：", out_path.resolve())
    print(f"共 {len(results)} 筆文章")
    print("結束時間:", end.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"耗時 {(end - start).total_seconds():.1f} 秒")


if __name__ == "__main__":
    scrape_pts()
