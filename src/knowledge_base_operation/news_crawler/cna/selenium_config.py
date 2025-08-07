from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# ---------- 1. Selenium 基本設定 ----------
opt = webdriver.ChromeOptions()
opt.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
opt.add_experimental_option("excludeSwitches", ["enable-automation"])
opt.add_experimental_option("useAutomationExtension", False)

# ── 改用官方 .deb 版 Chrome 的 headless 啟動 ──
opt.add_argument("--headless=new")
opt.add_argument("--no-sandbox")
opt.add_argument("--disable-dev-shm-usage")
opt.add_argument("--disable-gpu")
# Chrome binary 位置改成官方安裝路徑
opt.binary_location = "/usr/bin/google-chrome-stable"

# ---------- 1-1. chromedriver 來源 ----------
service = Service(ChromeDriverManager().install())

# 建立 driver 與 WebDriverWait
driver = webdriver.Chrome(service=service, options=opt)
driver.execute_cdp_cmd(
    "Page.addScriptToEvaluateOnNewDocument",
    {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"},
)
wait = WebDriverWait(driver, 20)
