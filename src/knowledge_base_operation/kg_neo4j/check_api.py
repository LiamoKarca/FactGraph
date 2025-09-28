#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM API 連線健檢（強化版）
1) 網路層檢查：DNS 解析、TCP:443 可達、/v1/models 探活（5s 連線、10s 讀取）
2) API 層檢查：Responses API；失敗再回退 Chat Completions
3) 提供 base_url（OpenAI 相容端點）與代理環境變數的診斷提示
"""

import argparse
import os
import sys
import socket
import time
import traceback
from typing import Any, Dict, Optional

# httpx 做網路層探活（比 requests 更好控制 timeout）
import httpx

# OpenAI SDK（2024+）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def resolve_dns(host: str, family=socket.AF_UNSPEC):
    addrs = []
    try:
        for res in socket.getaddrinfo(host, 443, family, socket.SOCK_STREAM):
            af, socktype, proto, canonname, sa = res
            addrs.append(sa[0])
    except Exception as e:
        return False, [], f"DNS 解析失敗：{e}"
    return True, sorted(set(addrs)), None


def tcp_connect(host: str, port: int, timeout: float = 5.0):
    try:
        t0 = time.perf_counter()
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return True, time.perf_counter() - t0, None
    except Exception as e:
        return False, None, f"TCP 連線失敗：{type(e).__name__}: {e}"


def http_probe_models(base_url: str, api_key: str, connect_timeout=5.0, read_timeout=10.0):
    """
    用 GET /v1/models 快速探活；不進 LLM 回答，只測授權/網路。
    """
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(http2=True, timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=10.0)) as c:
            r = c.get(url, headers=headers)
            return True, r.status_code, r.text[:200]
    except httpx.ConnectTimeout:
        return False, None, "ConnectTimeout（無法在時限內連上 /models）"
    except httpx.ReadTimeout:
        return False, None, "ReadTimeout（已連上但遲遲無回應）"
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def build_client(api_key: Optional[str], base_url: Optional[str], connect_timeout=5.0, read_timeout=30.0):
    if not OPENAI_AVAILABLE:
        print("❌ 未安裝 openai 套件或版本不符。請先：pip install --upgrade openai")
        sys.exit(2)

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        print("❌ 找不到 API Key。請用 --api-key 或環境變數 OPENAI_API_KEY。", file=sys.stderr)
        sys.exit(2)

    client = OpenAI(
        api_key=key,
        base_url=base_url or None,
        timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=30.0),
    )
    return client, key


def call_once(client, model: str, message: str, per_call_timeout: float) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        # 優先 Responses API
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": message}],
            timeout=per_call_timeout,
        )
        t1 = time.perf_counter()
        text = None
        try:
            text = resp.output[0].content[0].text if getattr(resp, "output", None) else None
        except Exception:
            text = None
        usage = getattr(resp, "usage", None)
        return {
            "ok": True,
            "latency_s": t1 - t0,
            "api": "responses",
            "output_preview": (text or "").strip()[:200],
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            },
        }
    except Exception as e_res:
        # 回退 Chat Completions
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                timeout=per_call_timeout,
            )
            t1 = time.perf_counter()
            text = chat.choices[0].message.content if chat.choices else ""
            usage = getattr(chat, "usage", None)
            return {
                "ok": True,
                "latency_s": t1 - t0,
                "api": "chat.completions",
                "output_preview": (text or "").strip()[:200],
                "usage": {
                    "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
                },
            }
        except Exception as e_chat:
            return {
                "ok": False,
                "error": f"{type(e_chat).__name__}: {e_chat}",
                "fallback_error": f"(responses error was {type(e_res).__name__}: {e_res})",
                "trace": traceback.format_exc(limit=2),
            }


def with_retries(fn, retries: int, backoff=(0.8, 1.8)):
    base, factor = backoff
    for i in range(retries + 1):
        res = fn()
        if res.get("ok"):
            return res
        if i < retries:
            time.sleep(base * (factor ** i))
    return res


def main():
    p = argparse.ArgumentParser(description="LLM API 連線健檢（DNS/TCP/API 多層檢查）")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"要測的模型（預設：{DEFAULT_MODEL}）")
    p.add_argument("--message", default="ping", help="測試訊息（越短越便宜）")
    p.add_argument("--api-key", default=None, help="API Key（預設讀 OPENAI_API_KEY）")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"API base_url（預設：{DEFAULT_BASE_URL}）")
    p.add_argument("--connect-timeout", type=float, default=5.0, help="TCP 連線逾時（秒）")
    p.add_argument("--read-timeout", type=float, default=30.0, help="讀取逾時（秒）")
    p.add_argument("--retries", type=int, default=1, help="API 呼叫失敗重試次數")
    args = p.parse_args()

    base_url = args.base_url.rstrip("/")
    host = base_url.replace("https://", "").replace("http://", "").split("/")[0]

    print("=== 1) 網路層健檢 ===")
    ok_dns, addrs, dns_err = resolve_dns(host)
    if ok_dns:
        print(f"✅ DNS 解析成功：{host} → {', '.join(addrs) or '(無位址?)'}")
    else:
        print(f"❌ DNS 解析失敗：{dns_err}")
        print("建議：檢查 /etc/resolv.conf、公司 DNS、改用 8.8.8.8/1.1.1.1；或改用 --base-url 測其他相容端點")
        sys.exit(1)

    ok_tcp, rtt, tcp_err = tcp_connect(host, 443, timeout=args.connect_timeout)
    if ok_tcp:
        print(f"✅ TCP 連線成功：443 / RTT ~ {rtt:.2f}s")
    else:
        print(f"❌ TCP 連線失敗：{tcp_err}")
        print("建議：檢查公司/防火牆是否封鎖到該域名:443；若需代理，設定 HTTPS_PROXY/HTTP_PROXY 環境變數")
        sys.exit(1)

    client, api_key = build_client(args.api_key, base_url, args.connect_timeout, args.read_timeout)

    ok_http, status, info = http_probe_models(base_url, api_key, connect_timeout=args.connect_timeout, read_timeout=10.0)
    if ok_http:
        print(f"✅ /models 探活成功：HTTP {status}")
    else:
        print(f"❌ /models 探活失敗：{info}")
        print("建議：\n- 若是 ConnectTimeout → 代理/防火牆/地區路由問題\n- 若是 ReadTimeout → 連得上但服務端或路由延遲\n- 可改用 --base-url 指向相容端點（如 OpenRouter），或設定企業代理")
        sys.exit(1)

    print("\n=== 2) API 層健檢 ===")
    one = with_retries(lambda: call_once(client, args.model, args.message, args.read_timeout), retries=args.retries)
    if one.get("ok"):
        print(f"✅ 呼叫成功（{one['api']}）")
        print(f"   延遲：{one['latency_s']:.2f}s")
        usage = one.get("usage") or {}
        print(f"   tokens: in={usage.get('input_tokens')} out={usage.get('output_tokens')} total={usage.get('total_tokens')}")
        if one.get("output_preview"):
            print(f"   節錄：{one['output_preview']!r}")
        print("\n🎉 健檢完成：連線 & API 正常。")
    else:
        print("❌ API 呼叫失敗")
        print(f"   錯誤：{one.get('error')}")
        if one.get("fallback_error"):
            print(f"   備註：{one['fallback_error']}")
        print("建議：檢查 API Key 權限、模型名稱是否可用；或嘗試不同 base_url/代理。")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 使用者中斷。")
