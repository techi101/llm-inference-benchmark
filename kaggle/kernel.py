"""LLM inference benchmark: one open model, several backends, precisions and devices.

Model: Qwen2.5-1.5B-Instruct.
Backends: Hugging Face Transformers (GPU FP16 / INT8 / INT4, CPU FP32), llama.cpp (CPU F16 / Q8_0 / Q4_K_M),
Ollama, vLLM (GPU FP16 and AWQ INT4), and a FastAPI service around llama.cpp under concurrent load.
Measures: time to first token, latency p50/p95, decode tokens/s, batched throughput, peak memory,
cost per 1M tokens (stated price assumptions) and accuracy on 100 ARC-Easy questions.

Every stage is isolated: a failure is recorded in results.json and the run continues.
"""
import gc
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback

os.environ.setdefault("ONLY", "env,vllm_install,vllm_gpu_FP16,vllm_gpu_AWQ_INT4")
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)
RES_PATH = os.path.join(OUT, "results.json")
RES = json.load(open(RES_PATH)) if os.path.exists(RES_PATH) else {}
ONLY = set(filter(None, os.environ.get("ONLY", "").split(",")))  # run only these stages (re-runs)
VLLM_PY = "/tmp/vllm_env/bin/python"

HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILES = {"F16": "qwen2.5-1.5b-instruct-fp16.gguf", "Q8_0": "qwen2.5-1.5b-instruct-q8_0.gguf",
              "Q4_K_M": "qwen2.5-1.5b-instruct-q4_k_m.gguf"}
AWQ_MODEL = "Qwen/Qwen2.5-1.5B-Instruct-AWQ"
N_EVAL = 100
GEN_TOKENS = 128
SPEED_PROMPTS = [
    "Write a detailed 400-word explanation of how a hospital could use AI to schedule appointments.",
    "Write a long story about a data centre engineer who fixes a failing GPU server at night.",
    "Explain in detail, with many examples, how banks check loan applications for fraud.",
    "Write a long essay about the history of the internet, from ARPANET to today.",
    "Describe in great detail how to plan a week-long trip across India by train.",
    "Write a long, step by step guide to learning Python for a complete beginner.",
    "Explain in detail how speech recognition turns audio into text, for a school student.",
    "Write a long report on why cloud costs grow and how a company can reduce them.",
]
# Price assumptions (on-demand list prices, rounded; edit to your provider): used only for cost per token.
PRICE_USD_HR = {"gpu_t4": 0.35, "cpu_4vcpu": 0.17}
USD_INR = 88.0


def save():
    json.dump(RES, open(RES_PATH, "w"), indent=1)


def sh(cmd, check=False):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-2500:], flush=True)
    if check and r.returncode:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd} | STDOUT: {r.stdout[-800:]} | STDERR: {r.stderr[-1500:]}")
    return r


def stage(name):
    def deco(fn):
        def run(*a, **k):
            if (ONLY and name not in ONLY) or (name in RES and "error" not in RES[name]):
                print(f"== skip {name} (done)", flush=True)
                return
            print(f"\n======== {name}", flush=True)
            t0 = time.time()
            try:
                RES[name] = fn(*a, **k) or {}
            except Exception as e:
                RES[name] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
                print(RES[name]["trace"], flush=True)
            RES[name]["stage_seconds"] = round(time.time() - t0, 1)
            save()
        return run
    return deco


def pct(xs, q):
    xs = sorted(xs)
    return xs[max(0, int(round(q * len(xs))) - 1)]


def cost_per_m(tok_s, price_key):
    if not tok_s:
        return None
    usd = PRICE_USD_HR[price_key] / (tok_s * 3600) * 1e6
    return {"usd": round(usd, 4), "inr": round(usd * USD_INR, 2), "price_assumption_usd_hr": PRICE_USD_HR[price_key]}


# ---------------------------------------------------------------- eval set
def load_eval():
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(N_EVAL))
    items = []
    for r in ds:
        labels = r["choices"]["label"]
        opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, r["choices"]["text"]))
        q = f"Question: {r['question']}\n{opts}\nAnswer with only the letter of the correct option."
        items.append({"prompt": q, "labels": labels, "answer": r["answerKey"]})
    return items


def parse_choice(text, labels):
    t = text.strip()
    for l in labels:
        if t.upper().startswith(l.upper()):
            return l
    m = re.search(r"\b(" + "|".join(map(re.escape, labels)) + r")\b", t)
    return m.group(1) if m else None


def score(outputs, items):
    ok = sum(parse_choice(o, it["labels"]) == it["answer"] for o, it in zip(outputs, items))
    return {"correct": ok, "n": len(items), "accuracy": round(ok / len(items), 3)}


# ---------------------------------------------------------------- environment
@stage("env")
def env():
    info = {"python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    r = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    info["gpu"] = r.stdout.strip()
    r = sh("lscpu | grep 'Model name'")
    info["cpu"] = r.stdout.strip()
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except Exception as e:
        info["torch"] = str(e)
    return info


# ---------------------------------------------------------------- Hugging Face Transformers
def hf_run(device, precision, items, batch_sizes=(1, 8, 32), cache_test=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    tok.padding_side = "left"
    kw = {}
    if precision == "FP16":
        kw["torch_dtype"] = torch.float16
    elif precision == "FP32":
        kw["torch_dtype"] = torch.float32
    elif precision == "INT8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "INT4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.float16)
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        kw["device_map"] = "cuda"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, **kw)
    if device == "cpu":
        model.to("cpu")
    model.eval()
    load_s = time.time() - t0

    def enc(prompts):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        return tok(texts, return_tensors="pt", padding=True).to(device)

    def gen(prompts, n, use_cache=True, fixed=True):
        x = enc(prompts)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            y = model.generate(**x, max_new_tokens=n, min_new_tokens=n if fixed else 0, do_sample=False,
                               use_cache=use_cache, pad_token_id=tok.eos_token_id)
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t, y[:, x["input_ids"].shape[1]:]

    gen(SPEED_PROMPTS[:1], 8)  # warm-up
    ttft = statistics.median(gen(SPEED_PROMPTS[i:i + 1], 1)[0] for i in range(5))
    n_tok = GEN_TOKENS if device == "cuda" else 64
    lat = [gen([p], n_tok)[0] for p in SPEED_PROMPTS[: (8 if device == "cuda" else 3)]]
    res = {"backend": "transformers", "device": device, "precision": precision, "load_s": round(load_s, 1),
           "gen_tokens": n_tok, "ttft_ms": round(ttft * 1000, 1),
           "latency_p50_s": round(statistics.median(lat), 3), "latency_p95_s": round(pct(lat, 0.95), 3),
           "decode_tok_s": round(n_tok / max(statistics.median(lat) - ttft, 1e-6), 1)}
    thr = {}
    for b in (batch_sizes if device == "cuda" else (1,)):
        prompts = [SPEED_PROMPTS[i % len(SPEED_PROMPTS)] for i in range(b)]
        dt, _ = gen(prompts, n_tok)
        thr[str(b)] = round(b * n_tok / dt, 1)
    res["throughput_tok_s_by_batch"] = thr
    if cache_test:
        on = statistics.median(gen([SPEED_PROMPTS[0]], 64, use_cache=True)[0] for _ in range(2))
        off = statistics.median(gen([SPEED_PROMPTS[0]], 64, use_cache=False)[0] for _ in range(2))
        res["kv_cache"] = {"tokens": 64, "with_cache_s": round(on, 3), "without_cache_s": round(off, 3),
                           "speedup_x": round(off / on, 2)}
    outs = []
    bs = 16 if device == "cuda" else 8
    t = time.perf_counter()
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        _, y = gen([c["prompt"] for c in chunk], 5, fixed=False)
        outs += tok.batch_decode(y, skip_special_tokens=True)
    res["eval_seconds"] = round(time.perf_counter() - t, 1)
    res["eval"] = score(outs, items)
    if device == "cuda":
        res["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    else:
        import psutil
        res["rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    best = max(thr.values())
    pk = "gpu_t4" if device == "cuda" else "cpu_4vcpu"
    res["cost_per_1M_tokens_single_stream"] = cost_per_m(res["decode_tok_s"], pk)
    res["cost_per_1M_tokens_best_batch"] = cost_per_m(best, pk)
    del model; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- llama.cpp (CPU)
def gguf_path(q):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(GGUF_REPO, GGUF_FILES[q])


def llamacpp_run(q, items):
    import psutil
    from llama_cpp import Llama
    path = gguf_path(q)
    t0 = time.time()
    llm = Llama(model_path=path, n_ctx=2048, n_threads=os.cpu_count(), n_gpu_layers=0, verbose=False)
    load_s = time.time() - t0
    ban = {}
    try:
        for s in (b"<|im_end|>", b"<|endoftext|>"):
            for tid in llm.tokenize(s, add_bos=False, special=True):
                ban[tid] = -1e9
    except Exception:
        pass

    def _chunks(prompt, n):
        kw = dict(messages=[{"role": "user", "content": prompt}], max_tokens=n, temperature=0.0, stream=True)
        if ban:
            try:
                import itertools
                g = iter(llm.create_chat_completion(logit_bias=ban, **kw))
                return itertools.chain([next(g)], g)  # first chunk pulled here, so TTFT timing is kept
            except Exception:
                ban.clear()  # this llama-cpp-python build rejects logit_bias: count natural-length outputs
        return llm.create_chat_completion(**kw)

    def stream(prompt, n):
        t = time.perf_counter(); first = None; count = 0
        for ch in _chunks(prompt, n):
            if ch["choices"][0]["delta"].get("content"):
                count += 1
                if first is None:
                    first = time.perf_counter() - t
        return first or 0.0, time.perf_counter() - t, count

    stream("Hello", 4)  # warm-up
    runs = [stream(p, GEN_TOKENS) for p in SPEED_PROMPTS[:4]]
    ttft = statistics.median(r[0] for r in runs)
    lat = [r[1] for r in runs]
    toks = statistics.median(r[2] for r in runs)
    dec = statistics.median((r[2] - 1) / max(r[1] - r[0], 1e-6) for r in runs)
    outs = []
    t = time.perf_counter()
    for it in items:
        o = llm.create_chat_completion(messages=[{"role": "user", "content": it["prompt"]}], max_tokens=5,
                                       temperature=0.0)
        outs.append(o["choices"][0]["message"]["content"])
    res = {"backend": "llama.cpp", "device": "cpu", "precision": q, "threads": os.cpu_count(),
           "file_gb": round(os.path.getsize(path) / 1e9, 2), "load_s": round(load_s, 1),
           "gen_tokens_median": toks, "ttft_ms": round(ttft * 1000, 1),
           "latency_p50_s": round(statistics.median(lat), 3), "latency_p95_s": round(pct(lat, 0.95), 3),
           "decode_tok_s": round(dec, 1), "eval_seconds": round(time.perf_counter() - t, 1),
           "eval": score(outs, items), "rss_gb": round(psutil.Process().memory_info().rss / 1e9, 2)}
    res["cost_per_1M_tokens_single_stream"] = cost_per_m(res["decode_tok_s"], "cpu_4vcpu")
    del llm; gc.collect()
    return res


# ---------------------------------------------------------------- Ollama
def ollama_run(items):
    import requests
    sh("apt-get update -qq && apt-get install -y -qq zstd pciutils lshw")
    sh("curl -fsSL https://ollama.com/install.sh | sh", check=True)
    subprocess.Popen("nohup ollama serve > /tmp/ollama.log 2>&1 &", shell=True)
    for _ in range(60):
        try:
            requests.get("http://127.0.0.1:11434/api/tags", timeout=2); break
        except Exception:
            time.sleep(2)
    sh("ollama pull qwen2.5:1.5b", check=True)
    url = "http://127.0.0.1:11434/api/chat"

    def chat(p, n):
        r = requests.post(url, json={"model": "qwen2.5:1.5b", "stream": False,
                                     "messages": [{"role": "user", "content": p}],
                                     "options": {"temperature": 0, "num_predict": n}}, timeout=600).json()
        return r

    chat("Hello", 4)
    runs = [chat(p, GEN_TOKENS) for p in SPEED_PROMPTS[:6]]
    dec = [r["eval_count"] / (r["eval_duration"] / 1e9) for r in runs]
    ttft = [(r.get("prompt_eval_duration", 0)) / 1e6 for r in runs]
    lat = [r["total_duration"] / 1e9 for r in runs]
    t = time.perf_counter()
    outs = [chat(it["prompt"], 5)["message"]["content"] for it in items]
    ps = sh("ollama ps").stdout
    res = {"backend": "ollama", "model_tag": "qwen2.5:1.5b (Q4_K_M)", "processor": ps.strip(),
           "ttft_ms_prompt_eval": round(statistics.median(ttft), 1),
           "latency_p50_s": round(statistics.median(lat), 3), "latency_p95_s": round(pct(lat, 0.95), 3),
           "decode_tok_s": round(statistics.median(dec), 1), "eval_seconds": round(time.perf_counter() - t, 1),
           "eval": score(outs, items)}
    gpu = "GPU" in ps.upper()
    res["cost_per_1M_tokens_single_stream"] = cost_per_m(res["decode_tok_s"], "gpu_t4" if gpu else "cpu_4vcpu")
    sh("pkill -f 'ollama serve'")
    return res


# ---------------------------------------------------------------- FastAPI service under load
SERVER_SRC = r'''"""FastAPI endpoint around a quantized GGUF model served by llama.cpp on CPU.

Run:  MODEL_PATH=/path/model.gguf uvicorn server:app --port 8000
"""
import asyncio
import os
import time

from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "model.gguf")
N_THREADS = int(os.environ.get("N_THREADS", os.cpu_count() or 4))
N_CTX = int(os.environ.get("N_CTX", 2048))

app = FastAPI(title="llm-inference-benchmark service")
llm = None
lock = asyncio.Lock()  # llama.cpp context is not thread safe: one request at a time


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    queue_ms: float
    latency_ms: float
    tokens_per_second: float


@app.on_event("startup")
def load_model():
    global llm
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model file not found: {MODEL_PATH}")
    llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_threads=N_THREADS, verbose=False)


@app.get("/health")
def health():
    return {"status": "ok" if llm is not None else "loading", "model": os.path.basename(MODEL_PATH)}


def _run(req: GenerateRequest):
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": req.prompt}],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    dt = time.perf_counter() - t0
    u = out["usage"]
    return out["choices"][0]["message"]["content"], u["prompt_tokens"], u["completion_tokens"], dt


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if llm is None:
        raise HTTPException(503, "model not loaded")
    t_arrive = time.perf_counter()
    async with lock:
        t_start = time.perf_counter()
        try:
            text, pt, ct, dt = await asyncio.to_thread(_run, req)
        except Exception as e:  # surface model errors as 500 with the reason
            raise HTTPException(500, f"generation failed: {e}")
    return GenerateResponse(
        text=text,
        prompt_tokens=pt,
        completion_tokens=ct,
        queue_ms=(t_start - t_arrive) * 1000,
        latency_ms=(time.perf_counter() - t_arrive) * 1000,
        tokens_per_second=ct / dt if dt > 0 else 0.0,
    )
'''
LOAD_SRC = r'''"""Async load test for the /generate endpoint: N concurrent clients, reports p50/p95 latency."""
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
'''


def service_run():
    import requests
    d = os.path.join(OUT, "service"); os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "server.py"), "w").write(SERVER_SRC)
    open(os.path.join(d, "load_test.py"), "w").write(LOAD_SRC)
    sh(f"{sys.executable} -m pip install -q fastapi uvicorn httpx")
    env_ = dict(os.environ, MODEL_PATH=gguf_path("Q4_K_M"), N_THREADS=str(os.cpu_count()))
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--port", "8000"], cwd=d, env=env_,
                         stdout=open("/tmp/uvicorn.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                if requests.get("http://127.0.0.1:8000/health", timeout=2).json().get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        bad = requests.post("http://127.0.0.1:8000/generate", json={"prompt": ""}).status_code
        out = {"validation_status_for_empty_prompt": bad, "load": []}
        for c in (1, 4):
            r = sh(f"{sys.executable} load_test.py --concurrency {c} --requests 12 --max-tokens 64".replace(
                "load_test.py", os.path.join(d, "load_test.py")))
            out["load"].append(json.loads(r.stdout.strip().splitlines()[-1]))
        return out
    finally:
        p.terminate()


# ---------------------------------------------------------------- vLLM (separate process after install)
VLLM_SRC = r'''
import json, os, statistics, sys, time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
model, quant, items_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
PROMPTS = json.loads(os.environ["SPEED_PROMPTS"]); N = int(os.environ["GEN_TOKENS"])
items = json.load(open(items_path))
tok = AutoTokenizer.from_pretrained(model)
fmt = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
kw = dict(model=model, dtype="half", gpu_memory_utilization=0.85, max_model_len=2048, seed=0)
if quant != "none": kw["quantization"] = quant
t0 = time.time(); llm = LLM(**kw); load_s = time.time() - t0
sp = lambda n: SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
llm.generate([fmt("Hello")], sp(8))
ttft = statistics.median((lambda t: (llm.generate([fmt(PROMPTS[i])], sp(1), use_tqdm=False), time.perf_counter() - t)[1])(time.perf_counter()) for i in range(5))
lat = []
for p in PROMPTS:
    t = time.perf_counter(); llm.generate([fmt(p)], sp(N), use_tqdm=False); lat.append(time.perf_counter() - t)
thr = {}
for b in (1, 8, 32, 64):
    ps = [fmt(PROMPTS[i % len(PROMPTS)]) for i in range(b)]
    t = time.perf_counter(); llm.generate(ps, sp(N), use_tqdm=False); thr[str(b)] = round(b * N / (time.perf_counter() - t), 1)
t = time.perf_counter()
outs = llm.generate([fmt(it["prompt"]) for it in items], SamplingParams(temperature=0, max_tokens=5), use_tqdm=False)
ev = time.perf_counter() - t
lat.sort()
json.dump({"load_s": round(load_s, 1), "ttft_ms": round(ttft * 1000, 1), "gen_tokens": N,
           "latency_p50_s": round(statistics.median(lat), 3), "latency_p95_s": round(lat[max(0, round(0.95 * len(lat)) - 1)], 3),
           "decode_tok_s": round(N / max(statistics.median(lat) - ttft, 1e-6), 1), "throughput_tok_s_by_batch": thr,
           "eval_seconds": round(ev, 1), "outputs": [o.outputs[0].text for o in outs]}, open(out_path, "w"))
'''


def vllm_install():
    """vLLM 0.6.6 (last line with T4 / xformers support) in its own Python 3.11 env, since Kaggle runs 3.13."""
    sh(f"{sys.executable} -m pip install -q uv", check=True)
    clean = "env -u UV_SYSTEM_PYTHON -u UV_PYTHON -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT"  # Kaggle points uv at /usr
    sh(f"{clean} uv python install 3.11", check=True)
    sh(f"rm -rf /tmp/vllm_env && {clean} uv venv --python 3.11 --python-preference only-managed /tmp/vllm_env", check=True)
    sh(f"{VLLM_PY} --version", check=True)
    r = sh(f"{clean} uv pip install --python {VLLM_PY} vllm==0.6.6.post1", check=True)
    v = sh(f"{VLLM_PY} -c 'import vllm, torch; print(vllm.__version__, torch.__version__, torch.cuda.is_available())'",
           check=True)
    return {"python": "3.11 venv", "versions": v.stdout.strip()}


def vllm_run(model, quant, items, label):
    ip = os.path.join(OUT, "eval_items.json"); json.dump(items, open(ip, "w"))
    sp = os.path.join(OUT, "vllm_worker.py"); open(sp, "w").write(VLLM_SRC)
    op = os.path.join(OUT, f"vllm_{label}.json")
    env_ = dict(os.environ, SPEED_PROMPTS=json.dumps(SPEED_PROMPTS), GEN_TOKENS=str(GEN_TOKENS))
    r = subprocess.run([VLLM_PY, sp, model, quant, ip, op], env=env_, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-4000:], flush=True)
    if r.returncode or not os.path.exists(op):
        raise RuntimeError("vLLM worker failed: " + r.stderr[-1500:])
    res = json.load(open(op))
    outs = res.pop("outputs")
    res.update({"backend": "vllm", "device": "cuda", "precision": label, "eval": score(outs, items)})
    res["cost_per_1M_tokens_single_stream"] = cost_per_m(res["decode_tok_s"], "gpu_t4")
    res["cost_per_1M_tokens_best_batch"] = cost_per_m(max(res["throughput_tok_s_by_batch"].values()), "gpu_t4")
    return res


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    t_all = time.time()
    env()
    sh(f"{sys.executable} -m pip install -q -U bitsandbytes accelerate datasets psutil")
    items = load_eval()
    for prec in ("FP16", "INT8", "INT4"):
        stage(f"hf_gpu_{prec}")(hf_run)("cuda", prec, items, cache_test=(prec == "FP16"))
    stage("hf_cpu_FP32")(hf_run)("cpu", "FP32", items)
    sh(f"{sys.executable} -m pip install -q llama-cpp-python==0.3.5 "
       "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu")
    for q in ("F16", "Q8_0", "Q4_K_M"):
        stage(f"llamacpp_cpu_{q}")(llamacpp_run)(q, items)
    stage("fastapi_service_llamacpp_Q4")(service_run)()
    stage("ollama")(ollama_run)(items)
    stage("vllm_install")(vllm_install)()
    stage("vllm_gpu_FP16")(vllm_run)(HF_MODEL, "none", items, "FP16")
    stage("vllm_gpu_AWQ_INT4")(vllm_run)(AWQ_MODEL, "awq", items, "AWQ_INT4")
    RES["total_minutes"] = round((time.time() - t_all) / 60, 1)
    save()
    print(json.dumps({k: (v.get("error") or "ok") if isinstance(v, dict) else v for k, v in RES.items()}, indent=1))
