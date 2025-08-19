import json
import sys
import subprocess
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
PROJECT_ROOT = PIPELINE_DIR.parents[3]

LINK_LIST_PATH = PROJECT_ROOT / "data/raw/news/cna/link/list.json"
CONTENT_PATH = PROJECT_ROOT / "data/raw/news/cna/cna_news.json"
FRONTPAGE_SCRIPT = PIPELINE_DIR / "frontpage_link_collector.py"
CONTENT_SCRIPT = PIPELINE_DIR / "content.py"


def run_link_collector():
    print("[INFO] 強制執行首頁新聞連結蒐集...")
    subprocess.run(
        [sys.executable, str(FRONTPAGE_SCRIPT)],
        check=True
    )
    print("[INFO] 首頁連結蒐集完成")


def load_all_links():
    with LINK_LIST_PATH.open("r", encoding="utf-8-sig") as f:
        links = json.load(f)
    return [item["link"] for item in links if "link" in item]


def run_content_crawler(topic_url):
    try:
        # 設定 timeout，避免卡死一頁卡住全流程；check=False 以便檢查 returncode
        cp = subprocess.run(
            [
                sys.executable,
                str(CONTENT_SCRIPT),
                topic_url
            ],
            check=False,
            timeout=600  # 單個主題頁最多爬10分鐘，可視規模自行調整
        )
        if cp.returncode == 0:
            print(f"[OK] 已爬取主題頁內容：{topic_url}")
            return "OK"
        elif cp.returncode == 100:
            print(f"[STOP] {topic_url} 回報『起始即連續 3 則皆已存在』→ 結束整體流程。")
            return "STOP"
        else:
            print(f"[ERR] 主題頁爬取失敗：{topic_url} (returncode={cp.returncode})")
            return "ERR"
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {topic_url} 超過 10 分鐘，自動跳過。")
        return "TIMEOUT"


def main():
    run_link_collector()  # 每次 pipeline 啟動都重抓主題頁列表

    all_topic_urls = load_all_links()
    print(f"共有 {len(all_topic_urls)} 個主題頁，將依序逐一刷新內容。")

    for idx, topic_url in enumerate(all_topic_urls, 1):
        print(f"[{idx}/{len(all_topic_urls)}] 開始爬取主題頁：{topic_url}")
        status = run_content_crawler(topic_url)
        if status == "STOP":
            break

    print("【流程結束】")


if __name__ == "__main__":
    main()
