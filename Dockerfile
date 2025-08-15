# Use a CUDA‑ready base that already has PyTorch and FastAPI
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=/models/gpt-oss-20b \
    REDIS_URL=redis://redis:6379 \
    LLM_STUDIO_URL=http://127.0.0.1:1234 \
    MODEL_BACKENDS='{"gpt-oss-20b":"huggingface","studio-lora-7b":"studio"}'

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
