"""独立测试 Commenter API 是否可用。

用法：
    python scripts/test_commenter_api.py
    # 或覆盖环境变量：
    COMMENTER_API_KEY=sk-xxx COMMENTER_MODEL=qwen-plus python scripts/test_commenter_api.py

默认读取 train_paper_writing_2gpu.sh 里导出的环境变量：
    COMMENTER_API_KEY / COMMENTER_BASE_URL / COMMENTER_MODEL

分四个子测试：
    1) DNS / 基础 HTTP 可达性
    2) OpenAI 客户端单条调用
    3) 并发批量调用（模拟训练时 1024 样本 × max_workers=64 的压力）
    4) 统计失败率与平均延迟
"""
from __future__ import annotations

import os
import socket
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# httpx+openai 读到 ALL_PROXY=socks5://... 会强制创建 SOCKS transport，
# 若缺少 `socksio` 则直接 ImportError。这里统一剥掉 SOCKS，只保留 http/https 代理。
for _var in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)

API_KEY = os.environ.get("COMMENTER_API_KEY")
if not API_KEY:
    print("ERROR: COMMENTER_API_KEY environment variable is not set.", file=sys.stderr)
    print("Please source ~/.bashrc or export it before running this script.", file=sys.stderr)
    sys.exit(1)
BASE_URL = os.environ.get(
    "COMMENTER_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = os.environ.get("COMMENTER_MODEL", "qwen-plus")
BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", "16"))
MAX_WORKERS = int(os.environ.get("TEST_MAX_WORKERS", "8"))


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def test_dns_reachability() -> bool:
    print("\n=== [1/4] DNS / TCP reachability ===")
    host = urlparse(BASE_URL).hostname
    try:
        ip = socket.gethostbyname(host)
        _ok(f"DNS resolve: {host} -> {ip}")
    except Exception as e:
        _fail(f"DNS resolve failed for {host}: {e}")
        return False
    try:
        with socket.create_connection((host, 443), timeout=5):
            _ok(f"TCP 443 connect OK to {host}")
        return True
    except Exception as e:
        _fail(f"TCP connect failed: {e}")
        return False


def _make_client():
    from openai import OpenAI

    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def test_single_call() -> bool:
    print("\n=== [2/4] Single chat.completions call ===")
    try:
        client = _make_client()
    except Exception as e:
        _fail(f"Client instantiation failed: {e}")
        traceback.print_exc()
        return False

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with the single word: OK"},
            ],
            max_tokens=16,
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )
        latency = time.time() - t0
        content = resp.choices[0].message.content
        _ok(f"model={MODEL} latency={latency:.2f}s content={content!r}")
        return True
    except Exception as e:
        _fail(f"Single call failed: {e}")
        traceback.print_exc()
        return False


def _one_commenter_call(client, idx: int) -> tuple[int, bool, float, str]:
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an experienced academic reviewer. "
                    "Provide 1-sentence revision feedback.",
                },
                {
                    "role": "user",
                    "content": f"Draft #{idx}: Our paper proposes a method for X.",
                },
            ],
            max_tokens=64,
            temperature=0.8,
            extra_body={"enable_thinking": False},
        )
        content = resp.choices[0].message.content
        return idx, True, time.time() - t0, content[:80] if content else ""
    except Exception as e:
        return idx, False, time.time() - t0, f"{type(e).__name__}: {e}"


def test_concurrent_batch() -> bool:
    print(
        f"\n=== [3/4] Concurrent batch: {BATCH_SIZE} requests × {MAX_WORKERS} workers ==="
    )
    try:
        client = _make_client()
    except Exception as e:
        _fail(f"Client instantiation failed: {e}")
        return False

    results: list[tuple[int, bool, float, str]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_one_commenter_call, client, i) for i in range(BATCH_SIZE)]
        for fut in as_completed(futures):
            results.append(fut.result())
    wall = time.time() - t0

    ok_count = sum(1 for _, ok, *_ in results if ok)
    fail_count = BATCH_SIZE - ok_count
    latencies = [lat for _, ok, lat, _ in results if ok]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    max_lat = max(latencies) if latencies else 0.0

    print(
        f"  total={BATCH_SIZE} ok={ok_count} fail={fail_count} "
        f"wall={wall:.2f}s avg_latency={avg_lat:.2f}s max_latency={max_lat:.2f}s"
    )
    # print first few failures for diagnosis
    shown = 0
    for idx, ok, lat, msg in results:
        if not ok and shown < 5:
            print(f"  [FAIL #{idx}] ({lat:.2f}s) {msg}")
            shown += 1
    if ok_count == BATCH_SIZE:
        _ok("All concurrent calls succeeded")
        return True
    _fail(f"{fail_count}/{BATCH_SIZE} calls failed")
    return False


def test_env_dump() -> None:
    print("\n=== [4/4] Environment dump ===")
    print(f"  COMMENTER_API_KEY : {'set (' + API_KEY[:8] + '...)' if API_KEY else 'MISSING'}")
    print(f"  COMMENTER_BASE_URL: {BASE_URL}")
    print(f"  COMMENTER_MODEL   : {MODEL}")
    print(f"  BATCH_SIZE        : {BATCH_SIZE}")
    print(f"  MAX_WORKERS       : {MAX_WORKERS}")
    try:
        import openai

        print(f"  openai version    : {openai.__version__}")
    except Exception as e:
        print(f"  openai import failed: {e}")
    for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "NO_PROXY"):
        val = os.environ.get(proxy_var)
        if val:
            print(f"  {proxy_var}={val}")


def main() -> int:
    if not API_KEY:
        print("[FATAL] COMMENTER_API_KEY not set")
        return 2

    test_env_dump()
    steps = [
        ("DNS/TCP", test_dns_reachability),
        ("single", test_single_call),
        ("concurrent", test_concurrent_batch),
    ]
    results = {}
    for name, fn in steps:
        try:
            results[name] = fn()
        except Exception as e:  # pragma: no cover - defensive
            print(f"[ERROR] test {name} raised: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {name:12s}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
