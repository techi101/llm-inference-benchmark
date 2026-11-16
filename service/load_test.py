"""Async load test for the /generate endpoint: N concurrent clients, reports p50/p95 latency."""
import argparse
import asyncio
import json
import statistics
import time

import httpx

PROMPTS = [
    "Explain what a KV cache is in two sentences.",
    "Give three reasons a bank might run an LLM on-premise.",
    "What is quantization in machine learning?",
    "Write a short status update about a failed deployment.",
]


async def one(client, url, i, max_tokens):
    r = await client.post(url, json={"prompt": PROMPTS[i % len(PROMPTS)], "max_tokens": max_tokens}, timeout=600)
    r.raise_for_status()
    return r.json()


async def main(url, concurrency, requests, max_tokens):
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def guarded(i):
            async with sem:
                return await one(client, url, i, max_tokens)

        t0 = time.perf_counter()
        res = await asyncio.gather(*[guarded(i) for i in range(requests)])
        wall = time.perf_counter() - t0
    lat = sorted(r["latency_ms"] for r in res)
    toks = sum(r["completion_tokens"] for r in res)
    out = {
        "concurrency": concurrency,
        "requests": requests,
        "p50_ms": statistics.median(lat),
        "p95_ms": lat[max(0, int(round(0.95 * len(lat))) - 1)],
        "mean_queue_ms": statistics.mean(r["queue_ms"] for r in res),
        "throughput_tok_s": toks / wall,
        "requests_per_s": requests / wall,
    }
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/generate")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    asyncio.run(main(a.url, a.concurrency, a.requests, a.max_tokens))
