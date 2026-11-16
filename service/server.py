"""FastAPI endpoint around a quantized GGUF model served by llama.cpp on CPU.

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
