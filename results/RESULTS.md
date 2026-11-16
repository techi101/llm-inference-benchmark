# Results

Environment: `{"python": "3.13.16", "cpu_count": 4, "gpu": "Tesla T4, 15360 MiB, 580.178.04\nTesla T4, 15360 MiB, 580.178.04", "cpu": "Model name:                              Intel(R) Xeon(R) CPU @ 2.00GHz", "torch": "2.11.0+cu130", "cuda": true, "stage_seconds": 9.0, "source_run": "run1"}`

| Backend | Device | Precision | TTFT ms | Latency p50 s | p95 s | Decode tok/s (1 stream) | Best throughput tok/s | Memory GB | ARC-Easy acc (n=100) |
|---|---|---|---|---|---|---|---|---|---|
| transformers | cuda | FP16 | 46.7 | 4.968 | 5.073 | 26.0 | 786.8 (batch 32) | 3.37 (GPU peak) | 0.9 |
| transformers | cuda | INT8 | 254.3 | 27.153 | 27.509 | 4.8 | 122.7 (batch 32) | 2.13 (GPU peak) | 0.9 |
| transformers | cuda | INT4 | 63.4 | 7.58 | 7.652 | 17.0 | 420.2 (batch 32) | 1.56 (GPU peak) | 0.9 |
| transformers | cpu | FP32 | 1293.6 | 19.629 | 19.633 | 3.5 | 3.3 (batch 1) | 8.78 (process RSS) | 0.9 |
| llama.cpp | cpu | F16 | 549.2 | 18.715 | 18.776 | 7.0 | - | 3.56 (model file) | 0.89 |
| llama.cpp | cpu | Q8_0 | 715.4 | 13.007 | 13.067 | 10.4 | - | 1.89 (model file) | 0.89 |
| llama.cpp | cpu | Q4_K_M | 940.3 | 10.716 | 10.951 | 13.0 | - | 1.12 (model file) | 0.89 |
| ollama | cuda | qwen2.5:1.5b (Q4_K_M) | 14.7 | 0.932 | 1.014 | 140.5 | - | 2.1 (ollama ps, on GPU) | 0.9 |
| vllm_gpu_FP16 | not run yet (see results.json) |||||||||
| vllm_gpu_AWQ_INT4 | not run yet (see results.json) |||||||||

## Cost per 1M generated tokens (price assumptions in bench.py)

| Config | Single stream ₹ | Best batch ₹ |
|---|---|---|
| hf_gpu_FP16 | 329.06 | 10.87 |
| hf_gpu_INT8 | 1782.41 | 69.73 |
| hf_gpu_INT4 | 503.27 | 20.36 |
| hf_cpu_FP32 | 1187.3 | 1259.26 |
| llamacpp_cpu_F16 | 593.65 | - |
| llamacpp_cpu_Q8_0 | 399.57 | - |
| llamacpp_cpu_Q4_K_M | 319.66 | - |
| ollama | 60.89 | - |

## KV cache (Transformers, GPU FP16, 64 tokens)

With cache 2.51 s, without 2.736 s: 1.09x faster with the cache.

## FastAPI service (llama.cpp Q4_K_M on CPU) under concurrent load

| Concurrent clients | p50 ms | p95 ms | mean queue ms | tok/s | req/s |
|---|---|---|---|---|---|
| 1 | 5127 | 5383 | 0 | 12.1 | 0.20 |
| 4 | 17405 | 19122 | 11526 | 12.8 | 0.21 |

Empty prompt is rejected with HTTP 422.
