"""
MOI 新聞爬蟲（全量 + 動態偵測上限頁數 + 逐篇存檔 + 連三重複即中斷 + 起始頁防坎 + 連5頁0篇視為維護 + 可指定起始頁）
----------------------------------------------------------------
重點：
1) 指定起始頁：python src/knowledge_base_operation/news_crawler/moi/moi.py --start-page <PAGE>
2) 動態偵測上限頁數；失敗回退為 1500。
3) 逐篇即時存檔（原子寫入）。
4) 連續三篇皆為重複 → 中斷（僅非起始頁生效）。
5) 起始頁防坎：起始頁不套用中斷門檻，且起始頁結束時重置連號。
6) 若連續 5 頁列表皆顯示 0 篇 → 研判站方維護中 → 立即終止（並保險存檔一次）。
輸出檔：data/raw/news/moi/moi_news.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# =========================
# 全域常數設定
# =========================
BASE_URL = "https://www.moi.gov.tw/News.aspx?n=4&sms=9009&page="
PAGE_SIZE = 15  # 列表每頁筆數
OUTPUT_PATH = Path("data/raw/news/moi/moi_news.json")  # 單一輸出檔案
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

# XPath（Selenium 不能使用 /text()）
XPATH_CANDIDATES = [
    '//*[@id="CCMS_Content"]/div/div/div/div[3]/div/div/div/ul[2]/li[2]/span',
    (
        "/html/body/form/div[3]/div/div/div/div[3]/div/div/div/div[2]/div/div/div/"
        "div/div/div/div/div[2]/div/div/div/div[3]/div/div/div/div[3]/div/div/div/"
        "ul[2]/li[2]/span"
    ),
]

FALLBACK_MAX_PAGE = 1500                 # 動態頁數解析失敗時回退
MAX_CONSEC_DUP = 3                       # 連續重複門檻
SUPPRESS_DUP_STOP_ON_START_PAGE = True   # 起始頁是否關閉門檻
ZERO_PAGE_LIMIT = 5                      # ★ 連續 0 篇頁數達此上限 → 視為維護中

# =========================
# 工具函式
# =========================


def init_driver() -> webdriver.Chrome:
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={USER_AGENT}")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options,
    )
    return driver


def load_existing_data(path: Path = OUTPUT_PATH) -> Tuple[List[Dict], set]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            old_data = json.load(f)
    else:
        old_data = []
    old_titles = {item.get("title", "")
                  for item in old_data if "title" in item}
    return old_data, old_titles


def atomic_write_json(data: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def page_url(page_index: int) -> str:
    return f"{BASE_URL}{page_index}&PageSize={PAGE_SIZE}"


def parse_news_links(html: str) -> List[str]:
    """
    回傳內頁相對連結清單。維護頁面通常不含預期的表格結構，將返回空列表。
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    if not rows:
        return []
    rows = rows[1:]  # 移除表頭

    hrefs: List[str] = []
    for row in rows:
        a_tag = row.find("a", class_="aspx")
        href = a_tag.get("href") if a_tag else None
        if href and not href.lower().startswith("javascript:"):
            hrefs.append(href)
    return hrefs


def detect_max_page(driver: webdriver.Chrome, timeout_sec: int = 10) -> tuple[int, str]:
    driver.get(page_url(1))
    wait = WebDriverWait(driver, timeout_sec)

    # 1) 主要 XPath
    for xp in XPATH_CANDIDATES:
        try:
            elem = wait.until(EC.presence_of_element_located((By.XPATH, xp)))
            text = (elem.text or "").strip()
            nums = re.findall(r"\d+", text)
            if nums:
                max_page = int(nums[-1])
                return max_page, f'XPath 命中：{xp} | span.text="{text}"'
        except Exception:
            continue

    # 2) 備援：掃描 ul[2] li
    try:
        ul_xpath = '//*[@id="CCMS_Content"]/div/div/div/div[3]/div/div/div/ul[2]'
        ul_elem = driver.find_element(By.XPATH, ul_xpath)
        li_elems = ul_elem.find_elements(By.TAG_NAME, "li")
        bucket: List[int] = []
        preview_texts: List[str] = []
        for li in li_elems:
            t = (li.text or "").strip()
            if len(preview_texts) < 5:
                preview_texts.append(t)
            bucket.extend(int(n) for n in re.findall(r"\d+", t))
        if bucket:
            mp = max(bucket)
            return mp, f"UL[2] 掃描 | 取最大數字={mp} | 範例文字={preview_texts}"
    except Exception:
        pass

    # 3) 備援：尋找「最後」連結，從 href 解析 page=
    try:
        last_link = driver.find_element(By.PARTIAL_LINK_TEXT, "最後")
        href = last_link.get_attribute("href") or ""
        m = re.search(r"page=(\d+)", href)
        if m:
            mp = int(m.group(1))
            return mp, f'連結「最後」命中 | href="{href}" → page={mp}'
    except Exception:
        pass

    # 4) 全部失敗，回退
    return FALLBACK_MAX_PAGE, f"⚠️ 解析失敗，回退為 {FALLBACK_MAX_PAGE}（請檢查 DOM 或更新 XPath）"


def parse_news_page(driver: webdriver.Chrome, relative_url: str) -> Optional[Dict]:
    full_url = f"https://www.moi.gov.tw/{relative_url}"
    driver.get(full_url)
    time.sleep(random.uniform(1.5, 3.0))

    soup = BeautifulSoup(driver.page_source, "html.parser")

    title_text: Optional[str] = None
    title_block = soup.find("div", class_="simple-text title")
    if title_block:
        inner = title_block.find("div", class_="in")
        if inner:
            title_text = inner.get_text("h3")
    if not title_text:
        h3 = soup.find("h3")
        if h3 and h3.text:
            title_text = h3.get_text()
    if not title_text:
        return None

    title = title_text.replace(" ", "").replace("　", "")

    date_str = ""
    info_block = soup.find("div", class_="list-text detail")
    if info_block:
        li = info_block.find("li")
        if li and li.text:
            date_str = li.text.strip().replace("發布日期：", "").split(" ")[0]

    paragraphs = soup.select("div.p p")
    fragments: List[str] = []
    for p in paragraphs:
        txt = p.get_text(strip=True)
        if "報導" in txt or "新聞來源" in txt:
            continue
        fragments.append(txt)

    for sym in ["\n", '"', "「", "」", "：", "；", ":", ";", ",", "{", "}", "[", "]"]:
        fragments = [t.replace(sym, "") for t in fragments]

    raw = "".join(fragments)
    parts = [seg for seg in raw.split("。") if seg.strip()]
    content = "\n".join(seg + "。" for seg in parts) if parts else ""

    return {
        "date": date_str,
        "publisher": "MOI",
        "category": "政府",
        "title": title,
        "content": content,
        "label": True,
        "url": full_url,
    }


# =========================
# 主流程
# =========================
def crawl_moi_news(start_page: int = 1) -> None:
    start_time = datetime.now()
    print(f"開始時間: {start_time:%Y-%m-%d %H:%M:%S}")

    driver = init_driver()
    old_data, old_titles = load_existing_data(OUTPUT_PATH)

    consecutive_dup = 0
    consecutive_zero_pages = 0   # ★ 新增：連續 0 篇頁面計數
    stop_requested = False

    try:
        max_page, proof = detect_max_page(driver, timeout_sec=10)
        print(f"🔎 已實際讀取到的頁數上限：{max_page}")
        print(f"   └─ 來源證據：{proof}")
        if "回退" in proof or "解析失敗" in proof:
            print("⚠️ 注意：目前使用保守回退值。建議檢查分頁 DOM 與 XPath。")

        if start_page < 1:
            print(f"⚠️ 起始頁 {start_page} 非法，已更正為 1。")
            start_page = 1
        if start_page > max_page:
            print(f"⚠️ 起始頁 {start_page} 超過上限 {max_page}，不進行爬取。")
            return

        print(f"▶️  將從第 {start_page} 頁開始爬取，直到第 {max_page} 頁。")

        for page_index in range(start_page, max_page + 1):
            if stop_requested:
                break

            # 起始頁的門檻關閉開關（僅該頁）
            dup_gate_enabled = not (
                SUPPRESS_DUP_STOP_ON_START_PAGE and page_index == start_page
            )

            url = page_url(page_index)
            print(
                f"📄 正在處理列表頁：{url}（起始頁門檻 {'開啟' if dup_gate_enabled else '關閉'}）"
            )
            driver.get(url)
            time.sleep(3)

            hrefs = parse_news_links(driver.page_source)
            print(f"📑 本頁共 {len(hrefs)} 篇")

            # ★ 維護偵測：若本頁 0 篇 → 累計；否則重置
            if len(hrefs) == 0:
                consecutive_zero_pages += 1
                print(
                    f"🧰 維護偵測：連續 0 篇頁數 {consecutive_zero_pages}/{ZERO_PAGE_LIMIT}"
                )
                if consecutive_zero_pages >= ZERO_PAGE_LIMIT:
                    # 保險存檔一次（逐篇已寫，但再確保）
                    atomic_write_json(old_data, OUTPUT_PATH)
                    print("🛑 偵測到連續 5 頁 0 篇，研判『內政部全球資訊網 系統維護中』，提前終止。")
                    stop_requested = True
                    break
                # 0 篇頁面沒有文章可處理，直接進入下一頁
                print(f"➡️ 頁面進度：{page_index}/{max_page}")
                continue
            else:
                # 只要遇到有文章的頁面，即重置維護計數
                if consecutive_zero_pages > 0:
                    print("🔄 偵測到非 0 篇頁面，維護計數已重置為 0。")
                consecutive_zero_pages = 0

            for i, rel in enumerate(hrefs, start=1):
                if stop_requested:
                    break

                try:
                    article = parse_news_page(driver, rel)

                    if not article:
                        print(f"⚠️ 內頁解析失敗，視為非重複事件，重置連號；略過：{rel}")
                        consecutive_dup = 0
                        continue

                    title = article["title"]

                    if title in old_titles:
                        consecutive_dup += 1
                        print(
                            f"⏭ 跳過重複新聞（連號 {consecutive_dup}/{MAX_CONSEC_DUP}；"
                            f"起始頁門檻{'開啟' if dup_gate_enabled else '關閉'}）：{title}"
                        )
                        if consecutive_dup >= MAX_CONSEC_DUP:
                            if dup_gate_enabled:
                                atomic_write_json(old_data, OUTPUT_PATH)
                                print("🛑 連續三篇重複，觸發中斷並已保險存檔。")
                                stop_requested = True
                            else:
                                print("⛳ 起始頁門檻關閉：即使連三重複亦不中斷。")
                        continue
                    else:
                        consecutive_dup = 0
                        old_data.append(article)
                        old_titles.add(title)
                        atomic_write_json(old_data, OUTPUT_PATH)
                        print(f"✅ 新增並存檔：{title}  ({i}/{len(hrefs)})")

                except Exception as exc:
                    print(f"⚠️ 單篇處理例外（視為非重複並重置連號）：{rel}：{exc!r}")
                    consecutive_dup = 0
                finally:
                    try:
                        driver.delete_all_cookies()
                        driver.get("about:blank")
                    except Exception:
                        pass
                    time.sleep(random.uniform(1.5, 3.0))

            # ★ 起始頁處理完畢 → 重置連號，避免外溢到下一頁
            if page_index == start_page:
                if not dup_gate_enabled:
                    consecutive_dup = 0
                    print("🔁 起始頁結束：已重置連續重複計數（門檻將於下一頁起生效）。")

            print(f"➡️ 頁面進度：{page_index}/{max_page}")

    finally:
        try:
            driver.quit()
        except Exception:
            pass

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        print(f"結束時間: {end_time:%Y-%m-%d %H:%M:%S}")
        print(f"程式耗時: {elapsed:.2f} 秒")


# =========================
# 參數解析與進入點
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOI 新聞爬蟲（全量 + 動態頁數偵測 + 逐篇原子寫入 + 連三重複即中斷 + 起始頁防坎 + 連5頁0篇視為維護 + 可指定起始頁）"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="指定從哪一頁開始爬取（預設：1）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    crawl_moi_news(start_page=args.start_page)
