---
inclusion: always
---

# Technology Stack

## Core Framework
- **FastAPI** - Main web framework for REST API
- **Pydantic** - Data validation and settings management
- **Uvicorn** - ASGI server for FastAPI

## AI/ML Stack
- **PyTorch** - Deep learning framework
- **Transformers** (HuggingFace) - LLM model loading and inference
- **Sentence-Transformers** - Text embeddings for memory retrieval
- **FAISS** - Vector similarity search for memory management
- **BLIP2** - Vision-language model for image captioning

## Infrastructure
- **Redis** - Caching and message queuing
- **Docker** - Containerization with CUDA support
- **NVIDIA PyTorch base image** - GPU-accelerated container

## LLM Backends
The system supports multiple LLM backends configured via `MODEL_BACKENDS`:
- **HuggingFace Transformers** - Default backend for local models
- **vLLM** - High-performance inference server (GPU required)
- **LLM Studio** - External model serving via HTTP API

## Common Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Health check
python health_check.py

# Run tests
python tests.py
```

### Docker Deployment
```bash
# Start full stack (Redis + LLM Studio + Orchestrator)
docker-compose up -d

# Build orchestrator only
docker build -t orchestrator .

# Run orchestrator container
docker run -p 8001:8000 orchestrator
```

### Environment Variables
- `MODEL_PATH` - Path to local model files
- `REDIS_URL` - Redis connection string
- `LLM_STUDIO_URL` - External LLM Studio endpoint
- `MODEL_BACKENDS` - JSON mapping of model names to backend types
- `USE_REAL_LLM` - Enable actual LLM calls (vs mocked responses)