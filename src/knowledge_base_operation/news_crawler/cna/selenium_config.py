"""
CNA selenium 設定（跨 webdriver-manager 版本）
- 檔案鎖避免多進程同時下載/解壓導致壞 ZIP
- 遇 EOFError/BadZipFile/IndexError/PermissionError → 清快取並重試
- 優先使用 CHROMEDRIVER / CHROME_BIN（若已手動安裝）
- 盡量將快取隔離到專案 .wdm-cache/cna/（若目前版本支援 path 參數）
"""

from __future__ import annotations
import os
import sys
import time
import shutil
import zipfile
from pathlib import Path
from inspect import signature

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# ──────────────────────────────────────────────────────────────────────────────
# 專案根目錄（.../src/knowledge_base_operation/news_crawler/cna/selenium_config.py → 專案根）
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[4] if len(_THIS.parents) >= 5 else Path.cwd()

# 嘗試把快取放在專案（若當前版本支持 ChromeDriverManager(path=...)）
LOCAL_WDM_CACHE = PROJECT_ROOT / ".wdm-cache" / "cna"
LOCAL_WDM_CACHE.mkdir(parents=True, exist_ok=True)

# 降低噪音；WDM_LOCAL 不同版本行為不一，先設著不依賴
os.environ.setdefault("WDM_LOG_LEVEL", "0")

LOCK_FILE = LOCAL_WDM_CACHE / ".chromedriver.lock"


def _with_file_lock(func):
    """Linux/WSL 檔案鎖，避免多進程同時安裝。"""
    try:
        import fcntl
    except ImportError:
        # Windows 無 fcntl，直接執行
        return func()
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            return func()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _purge_cache():
    """清理本地快取（盡量只清 chromedriver 相關目錄）"""
    targets = [
        LOCAL_WDM_CACHE / "drivers" / "chromedriver",
        LOCAL_WDM_CACHE / "chromedriver",
        LOCAL_WDM_CACHE,
    ]
    for t in targets:
        try:
            if t.exists():
                shutil.rmtree(t, ignore_errors=True)
        except Exception:
            pass
    LOCAL_WDM_CACHE.mkdir(parents=True, exist_ok=True)


def _detect_chrome_binary() -> str | None:
    env_bin = os.environ.get("CHROME_BIN")
    candidates = [env_bin] if env_bin else []
    candidates += [
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    return None


def _install_chromedriver_with_retry(retries: int = 2) -> str:
    """
    取得 chromedriver：
    1) 若 CHROMEDRIVER 指定且存在，直接用
    2) 否則使用 webdriver-manager（跨版本）：優先帶 path（若支援），否則不帶參數
       → 以檔案鎖包住，並對壞包/半包自動清快取重試
    """
    sys_cd = os.environ.get("CHROMEDRIVER")
    if sys_cd and Path(sys_cd).exists():
        return sys_cd

    last_err: Exception | None = None

    def _install_once():
        # 反射檢查是否支援 path 參數（有些版本支援，有些沒有）
        kwargs = {}
        try:
            if "path" in signature(ChromeDriverManager).parameters:
                kwargs["path"] = str(LOCAL_WDM_CACHE)
        except Exception:
            # 某些極舊版本無法用 signature；忽略，走無參數
            pass
        mgr = ChromeDriverManager(**kwargs)
        return mgr.install()

    for _ in range(retries + 1):
        try:
            return _with_file_lock(_install_once)
        except (EOFError, zipfile.BadZipFile, IndexError, PermissionError) as e:
            # 壞 ZIP / 半包 / 權限 → 清快取後重試
            last_err = e
            _purge_cache()
            time.sleep(1.0)
            continue
        except TypeError as e:
            # 某些版本不支援我們傳的參數 → 改用無參數重試
            last_err = e

            def _install_once_compat():
                mgr = ChromeDriverManager()  # 不帶任何參數
                return mgr.install()
            try:
                return _with_file_lock(_install_once_compat)
            except Exception as e2:
                last_err = e2
                _purge_cache()
                time.sleep(1.0)
                continue
        except Exception as e:
            last_err = e
            time.sleep(0.5)
            continue

    if last_err:
        raise last_err
    raise RuntimeError("Unknown error while installing chromedriver")


# ──────────────────────────────────────────────────────────────────────────────
# Selenium 基本設定
opt = webdriver.ChromeOptions()
opt.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
opt.add_experimental_option("excludeSwitches", ["enable-automation"])
opt.add_experimental_option("useAutomationExtension", False)

opt.add_argument("--headless=new")
opt.add_argument("--no-sandbox")
opt.add_argument("--disable-dev-shm-usage")
opt.add_argument("--disable-gpu")
opt.add_argument("--window-size=1920,1080")

bin_path = _detect_chrome_binary()
if bin_path:
    opt.binary_location = bin_path  # 若找不到就讓 Selenium 自行尋徑

# ──────────────────────────────────────────────────────────────────────────────
# 初始化 driver / wait
try:
    driver_path = _install_chromedriver_with_retry(retries=2)
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=opt)

    # 反自動化偵測
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"},
    )
    wait = WebDriverWait(driver, 20)

except Exception as e:
    sys.stderr.write(
        f"[CNA selenium_config] 無法初始化 ChromeDriver：{e}\n"
        f"建議處置：\n"
        f"  - 清快取：rm -rf {LOCAL_WDM_CACHE}\n"
        f"  - 更新套件：pip install -U webdriver-manager selenium\n"
        f"  - 指定瀏覽器：export CHROME_BIN=/usr/bin/google-chrome-stable\n"
        f"  - 或指定系統驅動：export CHROMEDRIVER=/usr/bin/chromedriver\n"
    )
    raise
