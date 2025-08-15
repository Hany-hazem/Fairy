# app/config.py
import os
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # General
    MODEL_PATH: str = Field("./models", env="MODEL_PATH")          # local path to GPT‑OSS‑20B
    REDIS_URL: str = "redis://localhost:6379"
    USE_VLLM: bool = False                                 # set True if you have GPU + vllm

    # Safety
    OPENAI_MODERATION_KEY: str | None = Field(None, env="OPENAI_MODERATION_KEY")
    SAFETY_THRESHOLD: float = 0.8   # reject probability > threshold

    # Memory
    VECTOR_DB_PATH: str = "./vector_db"

    # Agent routing (json file)
    AGENT_REGISTRY: str = "./agents/agent_registry.json"

    # New: map model_name → backend type (huggingface, vllm, studio)
    MODEL_BACKENDS: dict[str, str] = Field(
        default={"gpt-oss-20b": "huggingface", "studio-lora-7b": "studio"},
        env="MODEL_BACKENDS"
    )

    # URL of the LLM Studio server (default to localhost if you run it locally)
    LLMS_STUDIO_URL: str = Field("http://localhost:1234", env="LLM_STUDIO_URL")
    
    # LM Studio specific configuration for GPT-OSS-20B
    LM_STUDIO_MODEL: str = Field("gpt-oss-20b", env="LM_STUDIO_MODEL")
    LM_STUDIO_TEMPERATURE: float = Field(0.7, env="LM_STUDIO_TEMPERATURE")
    LM_STUDIO_MAX_TOKENS: int = Field(2048, env="LM_STUDIO_MAX_TOKENS")
    LM_STUDIO_TIMEOUT: int = Field(30, env="LM_STUDIO_TIMEOUT")
    LM_STUDIO_RETRY_ATTEMPTS: int = Field(3, env="LM_STUDIO_RETRY_ATTEMPTS")

settings = Settings()
