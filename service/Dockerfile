FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Q4_K_M quantized Qwen2.5-1.5B-Instruct (about 1 GB)
RUN curl -L -o /app/model.gguf \
    https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

COPY server.py .
ENV MODEL_PATH=/app/model.gguf
EXPOSE 8000
HEALTHCHECK CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
