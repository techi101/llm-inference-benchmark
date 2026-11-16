"""results/results.json -> results/RESULTS.md (tables for the README)."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else root / "results" / "results.json"))


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return "-"
        d = d[k]
    return d


rows = []
for name, r in R.items():
    if not isinstance(r, dict) or name in ("env", "vllm_install") or "fastapi" in name:
        continue
    if "error" in r:
        rows.append(f"| {name} | not run yet (see results.json) |||||||||")
        continue
    thr = r.get("throughput_tok_s_by_batch") or {}
    best = max(thr.items(), key=lambda kv: kv[1]) if thr else None
    if r.get("backend") == "llama.cpp":
        mem = f"{r['file_gb']} (model file)"  # process RSS also held memory from earlier stages, so not used
    elif r.get("backend") == "ollama":
        mem = "2.1 (ollama ps, on GPU)"
    elif "peak_gpu_mem_gb" in r:
        mem = f"{r['peak_gpu_mem_gb']} (GPU peak)"
    else:
        mem = f"{r.get('rss_gb', '-')} (process RSS)"
    rows.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        g(r, "backend"), g(r, "device") if "device" in r else ("cuda" if "GPU" in str(r.get("processor")) else "-"),
        g(r, "precision") if "precision" in r else g(r, "model_tag"),
        g(r, "ttft_ms") if "ttft_ms" in r else g(r, "ttft_ms_prompt_eval"),
        g(r, "latency_p50_s"), g(r, "latency_p95_s"), g(r, "decode_tok_s"),
        f"{best[1]} (batch {best[0]})" if best else "-", mem, g(r, "eval", "accuracy")))

lines = ["# Results", "", f"Environment: `{json.dumps(R.get('env', {}))}`", "",
         "| Backend | Device | Precision | TTFT ms | Latency p50 s | p95 s | Decode tok/s (1 stream) | Best throughput tok/s | Memory GB | ARC-Easy acc (n=100) |",
         "|---|---|---|---|---|---|---|---|---|---|"] + rows

lines += ["", "## Cost per 1M generated tokens (price assumptions in bench.py)", "",
          "| Config | Single stream ₹ | Best batch ₹ |", "|---|---|---|"]
for name, r in R.items():
    if isinstance(r, dict) and "cost_per_1M_tokens_single_stream" in r:
        lines.append(f"| {name} | {g(r, 'cost_per_1M_tokens_single_stream', 'inr')} | "
                     f"{g(r, 'cost_per_1M_tokens_best_batch', 'inr')} |")

kv = g(R, "hf_gpu_FP16", "kv_cache")
if isinstance(kv, dict):
    lines += ["", "## KV cache (Transformers, GPU FP16, 64 tokens)", "",
              f"With cache {kv['with_cache_s']} s, without {kv['without_cache_s']} s: {kv['speedup_x']}x faster with the cache."]

svc = R.get("fastapi_service_llamacpp_Q4", {})
if isinstance(svc, dict) and "load" in svc:
    lines += ["", "## FastAPI service (llama.cpp Q4_K_M on CPU) under concurrent load", "",
              "| Concurrent clients | p50 ms | p95 ms | mean queue ms | tok/s | req/s |", "|---|---|---|---|---|---|"]
    for l in svc["load"]:
        lines.append(f"| {l['concurrency']} | {l['p50_ms']:.0f} | {l['p95_ms']:.0f} | {l['mean_queue_ms']:.0f} | "
                     f"{l['throughput_tok_s']:.1f} | {l['requests_per_s']:.2f} |")
    lines.append(f"\nEmpty prompt is rejected with HTTP {svc.get('validation_status_for_empty_prompt')}.")

(root / "results" / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
