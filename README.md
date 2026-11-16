# LLM Inference Benchmark

How much does it cost, and how fast is it, to serve the same small LLM with different engines, precisions and hardware?

One model (**Qwen2.5-1.5B-Instruct**), run on a Kaggle **NVIDIA T4** and a **4-vCPU Intel Xeon**, through:

- **Hugging Face Transformers**: FP16 / INT8 / INT4 (bitsandbytes) on GPU, FP32 on CPU
- **llama.cpp** (llama-cpp-python): GGUF F16 / Q8_0 / Q4_K_M on CPU
- **Ollama**: qwen2.5:1.5b (Q4_K_M) on GPU
- **vLLM**: FP16 and AWQ INT4 on GPU (run in progress, see below)

Every setup is measured the same way: time to first token, p50/p95 latency, decode speed, batched throughput, memory, accuracy on 100 ARC-Easy questions, and cost per 1M generated tokens in ₹.
A small **FastAPI** service wraps llama.cpp and gets load-tested under concurrent clients.

## Results

All numbers are real outputs of `bench/bench.py` on Kaggle (raw data: `results/results.json`, full tables: `results/RESULTS.md`).

| Backend | Device | Precision | TTFT ms | Latency p50 s | Decode tok/s (1 stream) | Best throughput tok/s | Memory GB | ARC-Easy acc (n=100) |
|---|---|---|---|---|---|---|---|---|
| Transformers | T4 GPU | FP16 | 46.7 | 4.97 | 26.0 | 786.8 (batch 32) | 3.37 (GPU peak) | 0.90 |
| Transformers | T4 GPU | INT8 | 254.3 | 27.15 | 4.8 | 122.7 (batch 32) | 2.13 (GPU peak) | 0.90 |
| Transformers | T4 GPU | INT4 (NF4) | 63.4 | 7.58 | 17.0 | 420.2 (batch 32) | 1.56 (GPU peak) | 0.90 |
| Transformers | CPU | FP32 | 1293.6 | 19.63 | 3.5 | 3.3 (batch 1) | 8.78 (process RSS) | 0.90 |
| llama.cpp | CPU | F16 | 549.2 | 18.72 | 7.0 | - | 3.56 (model file) | 0.89 |
| llama.cpp | CPU | Q8_0 | 715.4 | 13.01 | 10.4 | - | 1.89 (model file) | 0.89 |
| llama.cpp | CPU | Q4_K_M | 940.3 | 10.72 | 13.0 | - | 1.12 (model file) | 0.89 |
| Ollama | T4 GPU | Q4_K_M | 14.7 * | 0.93 | 140.5 | - | 2.1 (`ollama ps`) | 0.90 |
| vLLM | T4 GPU | FP16 / AWQ INT4 | pending | | | | | |

\* Ollama reports prompt-eval time rather than a streamed first token.

Latency = generating 128 tokens for one request.

### Cost per 1M generated tokens

| Setup | Single stream ₹ | Best batch ₹ |
|---|---|---|
| Transformers GPU FP16 | 329.06 | 10.87 |
| Transformers GPU INT8 | 1782.41 | 69.73 |
| Transformers GPU INT4 | 503.27 | 20.36 |
| Transformers CPU FP32 | 1187.30 | - |
| llama.cpp CPU F16 | 593.65 | - |
| llama.cpp CPU Q8_0 | 399.57 | - |
| llama.cpp CPU Q4_K_M | 319.66 | - |
| Ollama GPU Q4_K_M | 60.89 | - |

Cost = hourly price ÷ tokens per hour. The hourly prices are **assumptions**, not quotes: T4 GPU = $0.35/hr, 4 vCPU = $0.17/hr, $1 = ₹88. They are set in `bench/bench.py` (`PRICE_USD_HR`, `USD_INR`).

### KV cache

Transformers GPU FP16, 64 new tokens: 2.51 s with the KV cache vs 2.74 s without (1.09x). The gap is small at this short length because recomputation cost grows with sequence length.

### FastAPI service under load (llama.cpp Q4_K_M, CPU)

| Concurrent clients | p50 ms | p95 ms | Mean queue ms | Total tok/s |
|---|---|---|---|---|
| 1 | 5127 | 5383 | 0 | 12.1 |
| 4 | 17405 | 19122 | 11526 | 12.8 |

The model is guarded by one lock, so four clients barely raise total throughput: most of the extra latency is time spent waiting in the queue. That is the problem continuous batching (vLLM) is built to solve. Input is validated; an empty prompt returns HTTP 422.

## What the numbers say

1. **Batching is the biggest lever on GPU.** Transformers FP16 goes from 26 tok/s at batch 1 to 787 tok/s at batch 32, which takes cost from ₹329 to ₹11 per 1M tokens.
2. **The serving engine matters more than precision.** The same Q4_K_M model is 140 tok/s on GPU through Ollama vs 13 tok/s on CPU through llama.cpp, and plain Transformers FP16 manages 26 tok/s.
3. **Quantization did not cost accuracy here.** FP16, INT8 and INT4 all scored 0.90; the GGUF variants all scored 0.89.
4. **Quantization is not automatically faster.** bitsandbytes INT8 was the slowest GPU setup (4.8 tok/s) because of its dequantization overhead. On CPU, Q4_K_M was the fastest llama.cpp setup since decode is memory-bound and the file is about 3x smaller than F16.
5. **For one user, a CPU can be cost-competitive.** llama.cpp Q4 on CPU (₹320) costs about the same per token as unbatched Transformers FP16 on GPU (₹329).

## Caveats

- Accuracy uses only the first 100 ARC-Easy test questions, so a 0.01 difference is one question and not significant.
- Transformers `generate` is a baseline and not an optimized server; it is included to show what the serving engine adds.
- llama.cpp memory is the model file size, because process RSS also held memory from earlier stages in the same run.
- One run per setup, on shared Kaggle machines; expect some run-to-run noise.
- Prices are assumptions (see above). Change them in `bench.py` and re-run `summarize.py`.

## Run it

**Benchmark (Kaggle, T4 GPU, internet on):**

```bash
python kaggle/make_kernel.py            # inline service/ into kaggle/kernel.py
kaggle kernels push -p kaggle           # set your own id in kaggle/kernel-metadata.json
# re-run only some stages:
python kaggle/make_kernel.py env,ollama,vllm_install,vllm_gpu_FP16
```

Then download `results.json` into `results/runN/` and run:

```bash
python bench/merge_runs.py              # later runs fill stages that failed earlier
python bench/summarize.py               # -> results/RESULTS.md
```

**Service (Docker):**

```bash
cd service
docker build -t llm-service .
docker run -p 8000:8000 llm-service
curl -X POST localhost:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "What is KV caching?", "max_tokens": 64}'
python load_test.py --url http://localhost:8000/generate --concurrency 4
```

## Layout

```
bench/bench.py        all benchmark stages (each stage is resumable and skippable)
bench/summarize.py    results.json -> markdown tables
bench/merge_runs.py   merge several Kaggle runs
service/server.py     FastAPI /health and /generate around llama.cpp
service/load_test.py  async concurrent load test (httpx)
service/Dockerfile    container for the service
kaggle/               kernel builder + metadata
results/              raw and summarized results
```
