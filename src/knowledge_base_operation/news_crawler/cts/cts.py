"""
CTS (華視) Topic Crawler
=======================

抓取分類（政治、國際、社會）最新新聞，輸出 JSON：

[
    {
        "date": "YYYY-MM-DD",
        "publisher": "CTS",
        "category": "政治",
        "title": "標題",
        "content": "內文…",
        "label": true
    },
    ...
]

輸出路徑：
    FactGraph/data/raw/news/cts_<timestamp>.json
    timestamp = yyyymmddHHMM (Asia/Taipei)
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ──────────────────────────────
# 常數／設定
# ──────────────────────────────
TPE_TZ = ZoneInfo("Asia/Taipei")

BASE_URL = "https://news.cts.com.tw/"
CATEGORIES = {
    "politics": "政治",
    "international": "國際",
    "society": "社會",
}
SCROLL_COUNT: int = 3  # 每一分類往下滾動次數
WAIT_BETWEEN_SCROLL: float = 3  # 每次滾動後等待秒數
RANDOM_WAIT_MIN: float = 1.5  # 進內文隨機等待下限
RANDOM_WAIT_MAX: float = 5.0  # 進內文隨機等待上限
USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

OUTPUT_DIR = Path("FactGraph/data/raw/news/cts/")


# ──────────────────────────────
# Selenium Driver
# ──────────────────────────────
def build_driver(headless: bool = True) -> webdriver.Chrome:
    """Return a configured Chrome driver."""
    chrome_opts = Options()
    if headless:
        chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--start-maximized")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument(f"user-agent={USER_AGENT}")
    chrome_opts.add_experimental_option(
        "excludeSwitches", ["enable-automation"]
    )
    chrome_opts.add_experimental_option("useAutomationExtension", False)

    drv = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_opts,
    )
    # 隱藏 webdriver 特徵
    drv.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": (
                "Object.defineProperty("
                "navigator,'webdriver',{get:()=>undefined});"
            )
        },
    )
    return drv


# ──────────────────────────────
# 工具函式
# ──────────────────────────────
def scroll_page(driver: webdriver.Chrome, times: int = 1) -> None:
    """Scroll to the bottom `times` times to load dynamic content."""
    for i in range(times):
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )
        print(f"  ↓ 第 {i + 1}/{times} 次滾動")
        time.sleep(WAIT_BETWEEN_SCROLL)


def parse_article(driver: webdriver.Chrome, url: str, category_tw: str) -> dict:
    """Open `url` and return structured article data."""
    driver.get(url)
    time.sleep(random.uniform(RANDOM_WAIT_MIN, RANDOM_WAIT_MAX))

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # 標題
    title_tag = soup.select_one("div.artical-titlebar h1")
    if not title_tag:
        raise ValueError("無法擷取標題")
    title = title_tag.text.replace(" ", "").replace("　", "")  # 去除全半形空格

    # 時間 → YYYY-MM-DD
    time_tag = soup.select_one("div.titlebar-top time")
    raw_dt = datetime.strptime(time_tag.text.strip(), "%Y/%m/%d %H:%M")
    date_str = raw_dt.strftime("%Y-%m-%d")

    # 內文
    paragraphs = soup.select("div.artical-content p")
    content_parts = [
        p.get_text(strip=True)
        for p in paragraphs
        if "報導" not in p.text and "新聞來源" not in p.text
    ]
    for sym in ['\n', '"', '「', '」', '：', '；', ':', ';', ',', '{', '}', '[', ']']:
        content_parts = [txt.replace(sym, "") for txt in content_parts]

    raw_content = "".join(content_parts)
    sentences = [s for s in raw_content.split("。") if s.strip()]
    content = "\n".join(f"{s}。" for s in sentences)

    return {
        "date": date_str,
        "publisher": "CTS",
        "category": category_tw,
        "title": title,
        "content": content,
        "label": True,
    }


# ──────────────────────────────
# 主流程
# ──────────────────────────────
def main() -> None:
    """Crawl all categories and export to a timestamped JSON."""
    start_time = datetime.now(TPE_TZ)
    print("開始時間:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    driver = build_driver(headless=True)
    articles: list[dict] = []
    seen_titles: set[str] = set()  # 當次執行內去重

    try:
        for slug, category_tw in CATEGORIES.items():
            url = f"{BASE_URL}{slug}/index.html"
            print(f"\n📄 處理分類頁：{url}")
            driver.get(url)
            time.sleep(5)  # 等主要內容載入

            print("🔄 滾動頁面載入更多內容…")
            scroll_page(driver, SCROLL_COUNT)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            link_container = soup.select_one(
                "div.newslist-container.flexbox.one_row_style"
            )
            if not link_container:
                print("⚠️ 找不到新聞清單容器，跳過此分類")
                continue

            news_links = [a["href"] for a in link_container.select("a")]
            print(f"📑 共 {len(news_links)} 筆連結")

            for idx, article_url in enumerate(news_links, 1):
                print(f"  ➡️ ({idx}/{len(news_links)}) {article_url}")
                try:
                    article = parse_article(driver, article_url, category_tw)
                    if article["title"] in seen_titles:
                        print("   ↪︎ 重複標題，跳過")
                        continue
                    articles.append(article)
                    seen_titles.add(article["title"])
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"   ⚠️ 解析失敗：{exc}")
                finally:
                    driver.delete_all_cookies()
                    driver.get("about:blank")
                    time.sleep(random.uniform(RANDOM_WAIT_MIN, RANDOM_WAIT_MAX))

    finally:
        driver.quit()

    # 輸出
    ts = start_time.strftime("%Y%m%d%H%M")
    output_path = OUTPUT_DIR / f"cts_{ts}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(articles, fp, ensure_ascii=False, indent=2)

    end_time = datetime.now(TPE_TZ)
    elapsed = (end_time - start_time).total_seconds()
    print("\n✅ 完成！")
    print("輸出檔案：", output_path.resolve())
    print("結束時間:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"程式耗時: {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
