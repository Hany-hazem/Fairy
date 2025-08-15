---
inclusion: always
---

# Project Structure

## Root Level
- `.env` - Environment variables for local development
- `agent.py` - Simple test script for LLM API calls
- `health_check.py` - Health check utility for LLM Studio connectivity
- `tests.py` - Test suite
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container build configuration
- `docker-compose.yml` - Multi-service deployment setup

## Core Application (`app/`)
Main FastAPI application with modular architecture:

- `main.py` - FastAPI app entry point and main query handler
- `config.py` - Pydantic settings and environment configuration
- `llm_adapter.py` - Abstraction layer for multiple LLM backends
- `llm_studio_client.py` - Client for external LLM Studio API
- `agent_registry.py` - Intent-based agent routing system
- `memory_manager.py` - Vector-based context retrieval
- `safety_filter.py` - Content moderation and safety checks
- `mcp.py` - Message Control Protocol for Redis communication

## Agents (`agents/`)
Specialized agent implementations:

- `text_agent.py` - Text processing agent (placeholder)
- `vision_agent.py` - Image captioning using BLIP2 model
- `__init__.py` - Package initialization

## Workers (`workers/`)
Background processing components:

- `studio_worker.py` - Worker for LLM Studio integration

## Scripts (`scripts/`)
Deployment and maintenance utilities:

- `deploy.sh` - Deployment automation script
- `train_self_evolve.py` - Self-evolution training pipeline

## Architecture Patterns

### Agent Registration
Agents are registered via JSON configuration mapping intents to agent types and topics for MCP communication.

### Memory Management
Uses vector embeddings with FAISS for similarity-based context retrieval. Stores query-response pairs for future reference.

### Safety Layer
All responses pass through safety filtering before being returned to users.

### Backend Abstraction
LLM backends are abstracted through the `LLMAdapter` class, supporting HuggingFace, vLLM, and LLM Studio seamlessly.

### Environment-Based Configuration
All configuration is environment-driven using Pydantic settings, supporting both local development and containerized deployment.