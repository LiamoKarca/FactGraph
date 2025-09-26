#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL 工具模組

此模組提供兩大核心功能：

1. LOGGER — 透過 _init_logger() 產生的全域單例 Logger。
   - 同時輸出到 logs/YYYY-MM-DD/HHMMSS.log 與 stdout。
   - 過濾 Neo4j 非致命提示 (AggregationSkippedNull)。
   - 保證只初始化一次，避免重複印出相同訊息。

2. run_with_timer(func, *args, **kwargs) — 包裝任何可呼叫物件並統計執行時間，
   自動處理 Ctrl-C、requests 連線錯誤與其他例外，並依情況回傳退出碼。
"""

from __future__ import annotations

import logging
import pathlib
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Callable, TypeVar

import requests

T = TypeVar("T")


class _Neo4jLogFilter(logging.Filter):
    """過濾非致命的 Neo4j 通知訊息。"""

    _SKIP_TOKEN = "Neo.ClientNotification.Statement.AggregationSkippedNull"

    def filter(self, record: logging.LogRecord) -> bool:
        """若訊息包含 SKIP_TOKEN 則忽略。"""
        return self._SKIP_TOKEN not in record.getMessage()


def _build_handlers(log_file: pathlib.Path) -> list[logging.Handler]:
    """建立檔案與 stdout 處理器，並共用同一個 Formatter。"""
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    return [file_handler, stream_handler]


def _init_logger() -> logging.Logger:
    """初始化並取得全域 Logger (單例)。"""
    logger = logging.getLogger("ETL")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        now = datetime.now()
        log_dir = pathlib.Path("logs") / now.strftime("%Y-%m-%d")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{now.strftime('%H%M%S')}.log"

        for handler in _build_handlers(log_file):
            logger.addHandler(handler)

        logger.addFilter(_Neo4jLogFilter())

    return logger


LOGGER: logging.Logger = _init_logger()


def run_with_timer(func: Callable[..., T], *args: Any, **kwargs: Any) -> None:
    """包裝並執行 func，最終列印總耗時並退出程式。

    退出碼:
      0: 執行成功
      130: 使用者中斷 (Ctrl-C)
      1: 其他例外 (網路錯誤或未預期例外)
    """
    start_time = time.perf_counter()
    LOGGER.info("=== Pipeline 開始 ===")

    exit_code = 0
    status_msg = "正常結束"

    try:
        result = func(*args, **kwargs)
    except KeyboardInterrupt:
        status_msg = "使用者中斷 (Ctrl-C)"
        exit_code = 130
    except requests.exceptions.RequestException as e:
        LOGGER.error("LLM 連線中斷: %s", e)
        status_msg = "LLM 連線中斷"
        exit_code = 1
    except Exception as e:
        LOGGER.exception("非預期錯誤: %s", e)
        status_msg = f"非預期錯誤: {type(e).__name__} – {e}"
        exit_code = 1
    else:
        # func 執行成功，保留 result（若有需要可處理）
        _ = result  # noqa: F841
    finally:
        elapsed = timedelta(seconds=round(time.perf_counter() - start_time))
        LOGGER.info("=== %s，耗時：%s ===", status_msg, elapsed)
        sys.exit(exit_code)
