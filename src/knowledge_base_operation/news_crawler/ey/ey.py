"""
EY（行政院）新聞爬蟲
- 列表頁： https://www.ey.gov.tw/Page/6485009ABEC1CB9C?page={n}&PS=200&
- 內頁：擷取標題、日期（民國→西元）、分類、發布者、內容、URL
- 清理：移除換行與雜訊符號，保留「。」作為分句，最終內容不含換行
- 去重：依 url 與 title
- 終止條件：起始頁之後，連續 5 則重複即終止
- 即時保存：每成功新增一篇立刻原子寫入 data/raw/news/ey/ey_news.json
- 參數：--start-page（起始頁，預設 1；起始頁不套用連續重複規則）
        --max-page（最大頁，預設程式常數）
- 輸出：data/raw/news/ey/ey_news.json（相對路徑，增量更新）
- python src/knowledge_base_operation/news_crawler/ey/ey.py --start-page 1 --max-page 106
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# -------------------------- 常數設定 --------------------------

BASE_URL = "https://www.ey.gov.tw/Page/6485009ABEC1CB9C"
PAGE_SIZE = 200
MAX_PAGE = 106  # 站內觀測上限；可依需求調整
PUBLISHER = "EY"
CATEGORY = "政府"
DUP_STOP_THRESHOLD = 5  # 連續 5 則重複即終止（起始頁不計）

# 清理內容時要移除的符號（保留中文句號）
SYMBOLS_TO_REMOVE = ['"', "「", "」", "：",
                     "；", ":", ";", ",", "{", "}", "[", "]"]

# -------------------------- 資料結構 --------------------------


@dataclass
class Article:
    date: str
    publisher: str
    category: str
    title: str
    content: str
    label: bool
    url: str

# -------------------------- 路徑/IO --------------------------


def find_project_root(marker: str = "data") -> Path:
    """自此檔一路往上找，直到找到含 marker 子目錄；找不到則回傳 CWD。"""
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / marker).exists():
            return parent
    return Path.cwd()


def output_path() -> Path:
    """相對路徑：data/raw/news/ey/ey_news.json"""
    root = find_project_root("data")
    out = root / "data" / "raw" / "news" / "ey" / "ey_news.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def load_existing(filepath: Path) -> List[dict]:
    """載入既有 JSON 陣列；若無或格式錯誤則回空陣列。"""
    if not filepath.exists():
        return []
    try:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def atomic_save_json(filepath: Path, data: List[dict]) -> None:
    """
    原子寫入 JSON：先寫入臨時檔再以 replace() 覆蓋正式檔，
    確保任何時刻磁碟上的檔案都是完整可解析的。
    """
    tmp = filepath.with_suffix(filepath.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
    tmp.replace(filepath)

# -------------------------- 爬蟲核心 --------------------------


def setup_driver() -> webdriver.Chrome:
    """建立 Headless Chrome WebDriver。"""
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


def get_list_page_url(page_index: int) -> str:
    return f"{BASE_URL}?page={page_index}&PS={PAGE_SIZE}&"


def extract_links_from_list_html(html: str) -> List[str]:
    """從列表頁 HTML 擷取新聞相對連結。"""
    soup = BeautifulSoup(html, "html.parser")
    boxes = soup.find_all("div", class_="news_box pdf_box")
    links: List[str] = []
    for box in boxes:
        tag_a = box.find("a")
        if not tag_a:
            continue
        href = tag_a.get("href", "").strip()
        if href:
            links.append(href)
    return links


def parse_roc_date(roc_text: str) -> str:
    """
    解析民國日期字串，例如「日期：114-07-29」→ 西元「2025-07-29」。
    """
    text = roc_text.strip().replace("日期：", "").replace("日期:", "").strip()
    parts = text.split("-")
    if len(parts) != 3:
        return text
    year = int(parts[0]) + 1911
    month = int(parts[1])
    day = int(parts[2])
    return f"{year:04d}-{month:02d}-{day:02d}"


def clean_text(raw: str) -> str:
    """
    內容清理：
    - 去除 \n/\r 與雜訊符號（保留「。」）
    - 以「。」分句、過濾空白片段並接回
    - 最終不含換行字元；若非空字串，確保以「。」結尾
    """
    text = raw.strip().replace("\n", "").replace("\r", "")
    for sym in SYMBOLS_TO_REMOVE:
        text = text.replace(sym, "")
    parts = [p.strip() for p in text.split("。") if p.strip()]
    if not parts:
        return ""
    joined = "。".join(parts)
    if not joined.endswith("。"):
        joined += "。"
    return joined


def extract_article_from_detail_html(html: str, url: str) -> Optional[Article]:
    """從內頁 HTML 擷取一篇 Article；若結構缺失則回傳 None。"""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("span", class_="h2")
    if not title_tag:
        return None
    title = title_tag.get_text(strip=True).replace(" ", "").replace("　", "")

    time_tag = soup.find("span", class_="date_style2")
    if not time_tag:
        return None
    span = time_tag.find("span")
    if not span:
        return None
    iso_date = parse_roc_date(span.get_text())

    paragraphs = soup.select("div.words_content p")
    texts: List[str] = []
    for p in paragraphs:
        t = p.get_text(separator=" ", strip=True)
        if ("報導" in t) or ("新聞來源" in t):
            continue
        texts.append(t)
    content = clean_text("".join(texts))

    return Article(
        date=iso_date,
        publisher=PUBLISHER,
        category=CATEGORY,
        title=title,
        content=content,
        label=True,
        url=url,
    )

# -------------------------- 主流程 --------------------------


def crawl(start_page: int, max_page: int) -> None:
    start_time = datetime.now()
    print(f"開始時間: {start_time:%Y-%m-%d %H:%M:%S}")

    out_file = output_path()
    all_articles: List[dict] = load_existing(out_file)

    # 去重集合（從既有資料初始化）
    seen_urls = {item.get("url", "") for item in all_articles if "url" in item}
    seen_titles = {item.get("title", "")
                   for item in all_articles if "title" in item}

    # 連續重複計數（起始頁不計）
    dup_streak = 0

    driver = setup_driver()

    try:
        for page_idx in range(start_page, max_page + 1):
            list_url = get_list_page_url(page_idx)
            print(f"📄 列表頁：{list_url}")

            driver.get(list_url)
            time.sleep(random.uniform(2.0, 4.0))

            links = extract_links_from_list_html(driver.page_source)
            print(f"  ↳ 本頁共 {len(links)} 則")

            for idx, rel in enumerate(links, start=1):
                full_url = f"https://www.ey.gov.tw{rel}"

                # URL 去重
                if full_url in seen_urls:
                    print(f"    ⏩ 已存在（URL）：{full_url}")
                    if page_idx != start_page:
                        dup_streak += 1
                        print(f"    ↳ 連續重複 {dup_streak}/{DUP_STOP_THRESHOLD}")
                        if dup_streak >= DUP_STOP_THRESHOLD:
                            print("    ⛔ 觸發：連續 5 則重複，準備落盤並終止。")
                            atomic_save_json(out_file, all_articles)
                            print(f"  💾 已寫入：{out_file}")
                            return
                    continue

                # 進入內頁
                driver.get(full_url)
                time.sleep(random.uniform(1.5, 3.5))

                article = extract_article_from_detail_html(
                    driver.page_source, full_url)
                if not article:
                    print(f"    ⚠️ 結構缺失，略過：{full_url}")
                    # 結構缺失不是重複，不累計 dup_streak
                    continue

                # Title 去重（保守）
                if article.title in seen_titles:
                    print(f"    ⏩ 已存在（Title）：{article.title}")
                    if page_idx != start_page:
                        dup_streak += 1
                        print(f"    ↳ 連續重複 {dup_streak}/{DUP_STOP_THRESHOLD}")
                        if dup_streak >= DUP_STOP_THRESHOLD:
                            print("    ⛔ 觸發：連續 5 則重複，準備落盤並終止。")
                            atomic_save_json(out_file, all_articles)
                            print(f"  💾 已寫入：{out_file}")
                            return
                    continue

                # 寫入新資料
                all_articles.append(asdict(article))
                seen_urls.add(full_url)
                seen_titles.add(article.title)
                dup_streak = 0  # 新增成功，重置連續重複計數

                # ✅ 即時保存（原子寫入）
                atomic_save_json(out_file, all_articles)

                print(
                    f"    ✅ {article.date} | {article.title[:30]}..."
                    f" （第 {idx}/{len(links)} 則）"
                )

    finally:
        driver.quit()

    end_time = datetime.now()
    print(f"結束時間: {end_time:%Y-%m-%d %H:%M:%S}")
    elapsed = (end_time - start_time).total_seconds()
    print(f"程式耗時: {elapsed:.1f} 秒")
    print(f"總計累積：{len(all_articles)} 則")

# -------------------------- 入口點 --------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EY 新聞爬蟲（增量去重、即時保存、連續重複終止）"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="起始頁（起始頁不套用連續重複終止規則）",
    )
    parser.add_argument(
        "--max-page",
        type=int,
        default=MAX_PAGE,
        help=f"最大頁，預設 {MAX_PAGE}",
    )
    args = parser.parse_args()
    if args.start_page < 1:
        raise ValueError("--start-page 需為 >= 1 的整數")
    if args.max_page < args.start_page:
        raise ValueError("--max-page 不可小於 --start-page")
    return args


if __name__ == "__main__":
    cli = parse_args()
    crawl(start_page=cli.start_page, max_page=cli.max_page)
