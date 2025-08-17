# app/config.py
import os
from typing import Dict, List, Optional, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum


class MCPServerMode(str, Enum):
    """MCP Server operation modes"""
    STANDALONE = "standalone"
    DISTRIBUTED = "distributed"
    CLUSTER = "cluster"


class GitWorkflowMode(str, Enum):
    """Git workflow operation modes"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    HYBRID = "hybrid"


class RedisSecurityMode(str, Enum):
    """Redis security modes"""
    NONE = "none"
    AUTH = "auth"
    TLS = "tls"
    TLS_AUTH = "tls_auth"


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

    # ===== MCP Server Configuration =====
    # Server settings
    MCP_SERVER_HOST: str = Field("0.0.0.0", env="MCP_SERVER_HOST")
    MCP_SERVER_PORT: int = Field(8765, env="MCP_SERVER_PORT")
    MCP_SERVER_MODE: MCPServerMode = Field(MCPServerMode.STANDALONE, env="MCP_SERVER_MODE")
    MCP_SERVER_MAX_CONNECTIONS: int = Field(100, env="MCP_SERVER_MAX_CONNECTIONS")
    MCP_SERVER_HEARTBEAT_INTERVAL: int = Field(30, env="MCP_SERVER_HEARTBEAT_INTERVAL")
    MCP_SERVER_MESSAGE_TIMEOUT: int = Field(60, env="MCP_SERVER_MESSAGE_TIMEOUT")
    
    # Message handling
    MCP_MESSAGE_MAX_SIZE: int = Field(1048576, env="MCP_MESSAGE_MAX_SIZE")  # 1MB
    MCP_MESSAGE_COMPRESSION: bool = Field(True, env="MCP_MESSAGE_COMPRESSION")
    MCP_MESSAGE_BATCH_SIZE: int = Field(10, env="MCP_MESSAGE_BATCH_SIZE")
    MCP_MESSAGE_BATCH_TIMEOUT: float = Field(0.1, env="MCP_MESSAGE_BATCH_TIMEOUT")
    
    # Connection management
    MCP_CONNECTION_RETRY_ATTEMPTS: int = Field(3, env="MCP_CONNECTION_RETRY_ATTEMPTS")
    MCP_CONNECTION_RETRY_DELAY: float = Field(1.0, env="MCP_CONNECTION_RETRY_DELAY")
    MCP_CONNECTION_MAX_RETRY_DELAY: float = Field(60.0, env="MCP_CONNECTION_MAX_RETRY_DELAY")
    MCP_CONNECTION_EXPONENTIAL_BACKOFF: bool = Field(True, env="MCP_CONNECTION_EXPONENTIAL_BACKOFF")
    
    # Context synchronization
    MCP_CONTEXT_SYNC_ENABLED: bool = Field(True, env="MCP_CONTEXT_SYNC_ENABLED")
    MCP_CONTEXT_SYNC_INTERVAL: int = Field(10, env="MCP_CONTEXT_SYNC_INTERVAL")
    MCP_CONTEXT_MAX_SIZE: int = Field(10485760, env="MCP_CONTEXT_MAX_SIZE")  # 10MB
    MCP_CONTEXT_COMPRESSION: bool = Field(True, env="MCP_CONTEXT_COMPRESSION")
    
    # ===== Redis Configuration =====
    # Connection settings
    REDIS_MAX_CONNECTIONS: int = Field(20, env="REDIS_MAX_CONNECTIONS")
    REDIS_CONNECTION_TIMEOUT: int = Field(10, env="REDIS_CONNECTION_TIMEOUT")
    REDIS_SOCKET_TIMEOUT: int = Field(5, env="REDIS_SOCKET_TIMEOUT")
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    REDIS_SOCKET_KEEPALIVE: bool = Field(True, env="REDIS_SOCKET_KEEPALIVE")
    REDIS_RETRY_ON_TIMEOUT: bool = Field(True, env="REDIS_RETRY_ON_TIMEOUT")
    
    # Security settings
    REDIS_SECURITY_MODE: RedisSecurityMode = Field(RedisSecurityMode.NONE, env="REDIS_SECURITY_MODE")
    REDIS_PASSWORD: Optional[str] = Field(None, env="REDIS_PASSWORD")
    REDIS_USERNAME: Optional[str] = Field(None, env="REDIS_USERNAME")
    REDIS_TLS_CERT_FILE: Optional[str] = Field(None, env="REDIS_TLS_CERT_FILE")
    REDIS_TLS_KEY_FILE: Optional[str] = Field(None, env="REDIS_TLS_KEY_FILE")
    REDIS_TLS_CA_CERT_FILE: Optional[str] = Field(None, env="REDIS_TLS_CA_CERT_FILE")
    REDIS_TLS_VERIFY_MODE: str = Field("required", env="REDIS_TLS_VERIFY_MODE")
    
    # Message queue settings
    REDIS_MESSAGE_TTL: int = Field(3600, env="REDIS_MESSAGE_TTL")  # 1 hour
    REDIS_MESSAGE_MAX_RETRIES: int = Field(3, env="REDIS_MESSAGE_MAX_RETRIES")
    REDIS_QUEUE_MAX_SIZE: int = Field(10000, env="REDIS_QUEUE_MAX_SIZE")
    REDIS_CLEANUP_INTERVAL: int = Field(300, env="REDIS_CLEANUP_INTERVAL")  # 5 minutes
    REDIS_BATCH_SIZE: int = Field(100, env="REDIS_BATCH_SIZE")
    
    # Performance settings
    REDIS_CONNECTION_POOL_SIZE: int = Field(10, env="REDIS_CONNECTION_POOL_SIZE")
    REDIS_PIPELINE_SIZE: int = Field(100, env="REDIS_PIPELINE_SIZE")
    REDIS_COMPRESSION_ENABLED: bool = Field(True, env="REDIS_COMPRESSION_ENABLED")
    REDIS_COMPRESSION_THRESHOLD: int = Field(1024, env="REDIS_COMPRESSION_THRESHOLD")  # 1KB
    
    # ===== Git Workflow Configuration =====
    # Repository settings
    GIT_REPO_PATH: str = Field(".", env="GIT_REPO_PATH")
    GIT_DEFAULT_BRANCH: str = Field("main", env="GIT_DEFAULT_BRANCH")
    GIT_WORKFLOW_MODE: GitWorkflowMode = Field(GitWorkflowMode.AUTOMATIC, env="GIT_WORKFLOW_MODE")
    GIT_AUTO_COMMIT: bool = Field(True, env="GIT_AUTO_COMMIT")
    GIT_AUTO_PUSH: bool = Field(False, env="GIT_AUTO_PUSH")
    
    # Branch management
    GIT_BRANCH_PREFIX_FEATURE: str = Field("feature", env="GIT_BRANCH_PREFIX_FEATURE")
    GIT_BRANCH_PREFIX_TASK: str = Field("task", env="GIT_BRANCH_PREFIX_TASK")
    GIT_BRANCH_PREFIX_HOTFIX: str = Field("hotfix", env="GIT_BRANCH_PREFIX_HOTFIX")
    GIT_BRANCH_PREFIX_BUGFIX: str = Field("bugfix", env="GIT_BRANCH_PREFIX_BUGFIX")
    GIT_BRANCH_CLEANUP_ENABLED: bool = Field(True, env="GIT_BRANCH_CLEANUP_ENABLED")
    GIT_BRANCH_CLEANUP_DAYS: int = Field(30, env="GIT_BRANCH_CLEANUP_DAYS")
    
    # Commit settings
    GIT_COMMIT_MESSAGE_TEMPLATE: str = Field(
        "{type}({scope}): {description}\n\n{body}\n\nTask-ID: {task_id}\nRequirements: {requirements}",
        env="GIT_COMMIT_MESSAGE_TEMPLATE"
    )
    GIT_COMMIT_MAX_SUBJECT_LENGTH: int = Field(72, env="GIT_COMMIT_MAX_SUBJECT_LENGTH")
    GIT_COMMIT_MAX_BODY_LENGTH: int = Field(500, env="GIT_COMMIT_MAX_BODY_LENGTH")
    GIT_COMMIT_SIGN: bool = Field(False, env="GIT_COMMIT_SIGN")
    GIT_COMMIT_GPG_KEY: Optional[str] = Field(None, env="GIT_COMMIT_GPG_KEY")
    
    # User configuration
    GIT_USER_NAME: Optional[str] = Field(None, env="GIT_USER_NAME")
    GIT_USER_EMAIL: Optional[str] = Field(None, env="GIT_USER_EMAIL")
    
    # Merge and conflict resolution
    GIT_MERGE_STRATEGY: str = Field("recursive", env="GIT_MERGE_STRATEGY")
    GIT_CONFLICT_RESOLUTION_TIMEOUT: int = Field(300, env="GIT_CONFLICT_RESOLUTION_TIMEOUT")  # 5 minutes
    GIT_AUTO_RESOLVE_CONFLICTS: bool = Field(False, env="GIT_AUTO_RESOLVE_CONFLICTS")
    GIT_MERGE_COMMIT_MESSAGE_TEMPLATE: str = Field(
        "Merge {source_branch} into {target_branch}\n\nCompleted tasks: {task_ids}",
        env="GIT_MERGE_COMMIT_MESSAGE_TEMPLATE"
    )
    
    # Task integration
    GIT_TASK_TRACKING_ENABLED: bool = Field(True, env="GIT_TASK_TRACKING_ENABLED")
    GIT_TASK_BRANCH_MAPPING_FILE: str = Field(".git/task_branch_mapping.json", env="GIT_TASK_BRANCH_MAPPING_FILE")
    GIT_TASK_COMPLETION_AUTO_MERGE: bool = Field(False, env="GIT_TASK_COMPLETION_AUTO_MERGE")
    GIT_TASK_DEPENDENCY_CHECK: bool = Field(True, env="GIT_TASK_DEPENDENCY_CHECK")
    
    # Performance and monitoring
    GIT_OPERATION_TIMEOUT: int = Field(60, env="GIT_OPERATION_TIMEOUT")
    GIT_LARGE_FILE_THRESHOLD: int = Field(104857600, env="GIT_LARGE_FILE_THRESHOLD")  # 100MB
    GIT_HISTORY_LIMIT: int = Field(1000, env="GIT_HISTORY_LIMIT")
    GIT_STATUS_CHECK_INTERVAL: int = Field(30, env="GIT_STATUS_CHECK_INTERVAL")
    
    # Remote repository settings
    GIT_REMOTE_NAME: str = Field("origin", env="GIT_REMOTE_NAME")
    GIT_REMOTE_URL: Optional[str] = Field(None, env="GIT_REMOTE_URL")
    GIT_REMOTE_PUSH_ENABLED: bool = Field(False, env="GIT_REMOTE_PUSH_ENABLED")
    GIT_REMOTE_FETCH_INTERVAL: int = Field(300, env="GIT_REMOTE_FETCH_INTERVAL")  # 5 minutes
    
    @validator('MCP_SERVER_PORT')
    def validate_mcp_port(cls, v):
        if not 1024 <= v <= 65535:
            raise ValueError('MCP server port must be between 1024 and 65535')
        return v
    
    @validator('REDIS_MESSAGE_TTL')
    def validate_redis_ttl(cls, v):
        if v <= 0:
            raise ValueError('Redis message TTL must be positive')
        return v
    
    @validator('GIT_COMMIT_MAX_SUBJECT_LENGTH')
    def validate_commit_subject_length(cls, v):
        if not 10 <= v <= 100:
            raise ValueError('Git commit subject length must be between 10 and 100 characters')
        return v
    
    @validator('GIT_REPO_PATH')
    def validate_git_repo_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f'Git repository path does not exist: {v}')
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
