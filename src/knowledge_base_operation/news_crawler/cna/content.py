import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from selenium_config import driver, wait

# ===== 專用例外：起始即連續 N 則皆已存在 → 提前終止 =====


class EarlyStopAtStart(Exception):
    """起始階段連續多次皆為已存在 → 提前終止整體流程"""
    pass


def close_ad():
    """嘗試關閉可能出現的擋版叉叉彈窗"""
    try:
        btn = driver.find_element(
            By.CSS_SELECTOR,
            "body > div > div.TopSection > div.fixedBottom > div.line-ad.need > svg > path"
        )
        btn.click()
        print("✖️ 關閉擋人的彈窗")
    except Exception:
        pass


def load_old_records(out_path: Path):
    """讀取舊有新聞紀錄與已存在的 URL 集合"""
    if out_path.exists():
        with out_path.open("r", encoding="utf-8-sig") as f:
            old_records = json.load(f)
        old_url_set = {rec.get("url") for rec in old_records if "url" in rec}
    else:
        old_records = []
        old_url_set = set()
    print(f"🗂️  檢測到舊有新聞 {len(old_records)} 筆")
    return old_records, old_url_set


def collect_urls(topic_url: str, target_num: int, old_url_set: set):
    """
    自動滾動並蒐集新聞頁網址。
    新增兩道門檻：
      A. 起始連續 3 次皆已存在 → 觸發 EarlyStopAtStart 例外，交由上層結束整體流程（exit 100）
      B. 非起始連續 5 次皆已存在 → 提前結束本主題頁的蒐集（保留已蒐集的新網址，正常返回）
    """
    driver.get(topic_url)
    time.sleep(1 + random.uniform(0.5, 1.0))
    close_ad()
    body = driver.find_element(By.TAG_NAME, "body")

    urls = []
    seen = set()
    current_url = driver.execute_script("return window.location.href")
    seen.add(current_url)

    # ===== 門檻常數 =====
    START_EXIST_STREAK_THRESHOLD = 3   # A：起始連續「已存在」N 次 → 全局停止
    EXIST_STREAK_BREAK = 5             # B：非起始連續「已存在」N 次 → 提前結束當前主題頁

    # ===== 控制旗標/計數 =====
    start_gate_active = True           # True 直到遇到第一個『新網址』
    exist_streak_at_start = 0          # 起始階段「已存在」連續次數
    exist_streak_after_start = 0       # 非起始階段「已存在」連續次數
    soft_stop_topic = False            # 觸發 B 後為 True，結束本主題頁的蒐集

    def record_if_new(url, at_beginning: bool):
        """
        判斷 url 是否為新網址；同時維護起始/非起始兩種連續已存在門檻。
        """
        nonlocal start_gate_active, exist_streak_at_start, exist_streak_after_start, soft_stop_topic

        if url not in old_url_set:
            urls.append(url)
            print("⇢ 新網址", url)

            # 任一新網址出現後，起始門檻永久失效；並重置「非起始」連續計數
            if start_gate_active:
                start_gate_active = False
            exist_streak_after_start = 0

        else:
            print(f"⏩ 已存在，跳過：{url}")

            # A：僅在「尚未出現任何新網址」之前，才累計起始連續已存在
            if at_beginning and start_gate_active:
                exist_streak_at_start += 1
                if exist_streak_at_start >= START_EXIST_STREAK_THRESHOLD:
                    msg = (f"[STOP] 起始即連續 {exist_streak_at_start} 則皆為已存在 → "
                           f"判定本輪無新內容，停止爬蟲。")
                    print(msg)
                    raise EarlyStopAtStart(msg)
            else:
                # B：非起始（已出現過至少一個新網址）時，連續已存在計數
                if not start_gate_active:
                    exist_streak_after_start += 1
                    if exist_streak_after_start >= EXIST_STREAK_BREAK:
                        print(f"➡️ 非起始階段連續 {exist_streak_after_start} 次皆為已存在 → "
                              f"提前結束本主題頁的蒐集（已抓到新網址 {len(urls)} 筆）。")
                        soft_stop_topic = True

    # 起始第一筆（當前網址）要計入起始門檻
    record_if_new(current_url, at_beginning=True)
    last_time = time.time()

    while len(urls) < target_num and not soft_stop_topic:
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(random.uniform(0.3, 0.5))
        new_url = driver.execute_script("return window.location.href")
        if new_url != current_url and new_url not in seen:
            seen.add(new_url)
            current_url = new_url
            last_time = time.time()

            # 若仍在起始階段，這筆要繼續算「起始門檻」；否則算「非起始」
            record_if_new(new_url, at_beginning=start_gate_active)

        if soft_stop_topic:
            break

        if time.time() - last_time >= 10:
            print("⚠️ 10 秒內無新網址，停止滾動。")
            break

    print(f"✅ 本次共收錄 {len(urls)} 則新聞。")
    return urls


def fetch_article_record(url: str):
    """爬取單篇新聞資料，回傳 record dict"""
    driver.get(url)
    time.sleep(1 + random.uniform(0.5, 1.0))
    close_ad()

    try:
        title = wait.until(
            lambda d: d.find_element(By.CSS_SELECTOR, "h1 span")
        ).text.strip()
    except Exception:
        title = "標題未取得"

    re_date = re.compile(r"/news/[^/]+/(\d{12})\.aspx")
    m = re_date.search(url)
    date = (
        datetime.strptime(m.group(1)[:8], "%Y%m%d").strftime("%Y-%m-%d")
        if m else "日期未取得"
    )

    cat_elems = driver.find_elements(By.CSS_SELECTOR, ".breadcrumb a")
    category = cat_elems[1].text.strip() if len(cat_elems) > 1 else ""

    paras = driver.find_elements(
        By.CSS_SELECTOR, "#article-body p, div.paragraph p")
    content = "\n".join(
        p.text.strip()
        for p in paras
        if p.text.strip() and "本網站之文字" not in p.text
    )

    return {
        "url": url,
        "date": date,
        "publisher": "CNA",
        "category": category,
        "title": title,
        "content": content,
        "label": True,
    }


def save_records(old_records: list, new_records: list, out_path: Path):
    """合併舊有與新增記錄並寫入 JSON 檔案"""
    final = old_records + new_records
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(
        f"🔸 累積總數 (舊有 {len(old_records)} 筆) + (新增 {len(new_records)} 筆) 共 {len(final)} 則，"
        f"已進行更新並寫入 {out_path.resolve()}"
    )
    return final


def fetch_cna_content(topic_url: str, target_num: int = 300, out_path: str = "data/raw/news/cna/content.json"):
    """主流程：呼叫各副程式完成新聞爬取及存檔"""
    OUT_PATH = Path(out_path)
    old_records, old_url_set = load_old_records(OUT_PATH)
    try:
        urls = collect_urls(topic_url, target_num, old_url_set)
    except EarlyStopAtStart:
        # A：全局停止
        try:
            driver.quit()
        finally:
            print("[EXIT 100] content.py 因『起始即連續 3 則皆已存在』提前結束。")
            import sys
            sys.exit(100)

    # B 情況下（或一般情況），此處正常處理已蒐集到的新網址
    new_records = []
    for url in urls:
        record = fetch_article_record(url)
        new_records.append(record)

    print(f"⚠️  與舊有新聞比對後，新增 {len(new_records)} 筆新的新聞。")
    final = save_records(old_records, new_records, OUT_PATH)
    driver.quit()
    return final


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        # CLI 直接給主題頁網址（支援 pipeline 批次執行）
        topic_url = sys.argv[1]
        try:
            fetch_cna_content(topic_url)
        except SystemExit:
            # 保留 exit code（例如 100：起始即連續 3 則已存在）
            raise
        except Exception as e:
            print(f"[ERR] content.py 發生異常：{e}")
        finally:
            try:
                driver.quit()
            except Exception as e:
                print(f"[WARN] driver.quit() 異常：{e}")
            print("【content.py 結束】")
    else:
        # 預設測試網址
        test_url = "https://www.cna.com.tw/news/aipl/202507300286.aspx?topic=4782"
        fetch_cna_content(test_url)
