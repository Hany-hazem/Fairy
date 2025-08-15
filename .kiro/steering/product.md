---
inclusion: always
---

# Product Overview

This is a **Self-Evolving AI Agent** orchestrator that provides a unified API for multiple AI agents with different capabilities. The system acts as an intelligent routing layer that:

- Routes queries to appropriate specialized agents (text, vision, etc.)
- Manages memory and context retrieval using vector embeddings
- Applies safety filtering to all responses
- Supports multiple LLM backends (HuggingFace Transformers, vLLM, LLM Studio)
- Uses Redis for caching and message queuing via MCP (Message Control Protocol)

The architecture is designed for scalability and modularity, allowing easy addition of new agent types and LLM backends. The system can handle both synchronous and asynchronous agent interactions.

## Key Features
- Multi-agent orchestration with intent-based routing
- Vector-based memory management for context retrieval
- Safety filtering and content moderation
- Flexible LLM backend support
- Docker-based deployment with Redis integration
- RESTful API interface via FastAPI