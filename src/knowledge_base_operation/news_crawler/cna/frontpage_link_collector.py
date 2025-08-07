import json
import random
import re
import time
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC

from selenium_config import driver, wait

# ====== 參數設置 ======
URL_LIST_PAGE = "https://www.cna.com.tw/list/aipl.aspx"
VIEW_MORE_BTN = "#SiteContent_uiViewMoreBtn_Style3"
NEWS_LIST_ITEM = "#jsMainList > li"
TARGET_NUM = 9999  # 可視專案上限調整
OUT_PATH = Path("data/raw/news/cna/link/list.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 新聞日期格式驗證（2025/08/01 13:11）
DATE_PATTERN = re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}$")


def is_strict_valid(item: dict) -> bool:
    """檢查標題、日期、連結皆合法。"""
    if not item:
        return False
    title = item.get("title", "").strip()
    date = item.get("date", "").strip()
    link = item.get("link", "").strip()
    if not (title and date and link):
        return False
    if not link.startswith("https://www.cna.com.tw/news/aipl/"):
        return False
    if not DATE_PATTERN.match(date):
        return False
    return True


def _extract_items():
    """僅收錄符合標準的 li。"""
    items, results = driver.find_elements(By.CSS_SELECTOR, NEWS_LIST_ITEM), []
    for li in items:
        try:
            a_tag = li.find_element(By.CSS_SELECTOR, "a")
            link = a_tag.get_attribute("href").strip() if a_tag else ""
            title = ""
            date = ""
            try:
                title_span = li.find_element(
                    By.CSS_SELECTOR, "div.listInfo h2 span")
                title = title_span.text.strip()
            except Exception:
                pass
            try:
                date_div = li.find_element(By.CSS_SELECTOR, "div.listInfo div")
                date = date_div.text.strip()
            except Exception:
                pass

            item = {"title": title, "date": date, "link": link}
            if is_strict_valid(item):
                results.append(item)
        except Exception:
            pass
    return results


def _click_view_more():
    """點擊查看更多（若有），若無則直接略過。"""
    try:
        btn = wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, VIEW_MORE_BTN))
        )
        driver.execute_script("arguments[0].click();", btn)
        return True
    except Exception:
        return False


def _scroll_down(body):
    """模擬 PAGE_DOWN，略加隨機等待。"""
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(random.uniform(0.25, 0.45))


def collect_cna_ai_links():
    driver.get(URL_LIST_PAGE)
    body = driver.find_element(By.TAG_NAME, "body")
    collected = {}
    last_update = time.time()

    while len(collected) < TARGET_NUM:

        _click_view_more()
        _scroll_down(body)

        items = _extract_items()

        for item in items:
            if item["link"] not in collected:
                collected[item["link"]] = item
                last_update = time.time()
                print(f"⇢ 收錄：{item['title']} ({item['date']})")

        if time.time() - last_update >= 12:
            print("⚠️  12 秒內無新資料，強制結束。")
            break

    return list(collected.values())


if __name__ == "__main__":
    try:
        # ========== 增量儲存區塊 ==========
        # 1. 讀取舊 list.json
        if OUT_PATH.exists():
            with OUT_PATH.open("r", encoding="utf-8-sig") as fp:
                old_data = json.load(fp)
            print(f"🗂️  檢測到舊有新聞 {len(old_data)} 筆")
        else:
            old_data = []
            print("🗂️  未偵測到舊檔，將建立新檔案。")

        # 2. 建立 link → item 快取（僅有效資料）
        old_link_map = {
            item["link"]: item for item in old_data if is_strict_valid(item)}
        print(f"🔍 已整理出 {len(old_link_map)} 筆有效舊新聞")

        # 3. 執行新一輪新聞爬取
        strict_data = collect_cna_ai_links()
        print(f"✅ 本次共收錄 {len(strict_data)} 則新聞。")

        # 4. 比對新增
        new_links = [item for item in strict_data if item["link"]
                     not in old_link_map]
        num_new = len(new_links)

        # 5. 合併所有新聞
        for item in new_links:
            old_link_map[item["link"]] = item

        total = len(old_link_map)
        print(f"⚠️  與舊有新聞比對後，新增 {num_new} 筆新的新聞。")
        print(
            f"🔸 累積總數 (舊有 {len(old_link_map)-num_new} 筆) + (新增 {num_new} 筆) 共 {total} 則，已進行更新並寫入 {OUT_PATH.resolve()}")

        # 6. 寫回（新到舊排序）
        all_news = sorted(
            old_link_map.values(),
            key=lambda x: x["date"],
            reverse=True
        )
        with OUT_PATH.open("w", encoding="utf-8-sig") as fp:
            json.dump(all_news, fp, ensure_ascii=False, indent=2)

    finally:
        driver.quit()
