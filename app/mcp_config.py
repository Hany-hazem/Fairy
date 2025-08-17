# app/mcp_config.py
"""
MCP Configuration Management

This module provides comprehensive configuration management for MCP server,
client, and integration components with validation and environment support.
"""

import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, validator
from pathlib import Path


class MCPServerMode(str, Enum):
    """MCP Server operation modes"""
    STANDALONE = "standalone"
    DISTRIBUTED = "distributed"
    CLUSTER = "cluster"


class CompressionAlgorithm(str, Enum):
    """Message compression algorithms"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"


class SecurityLevel(str, Enum):
    """Security levels for MCP communication"""
    NONE = "none"
    BASIC = "basic"
    ENHANCED = "enhanced"
    STRICT = "strict"


@dataclass
class MCPServerConfig:
    """MCP Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8765
    mode: MCPServerMode = MCPServerMode.STANDALONE
    max_connections: int = 100
    heartbeat_interval: int = 30
    message_timeout: int = 60
    enable_compression: bool = True
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    security_level: SecurityLevel = SecurityLevel.BASIC
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not 1024 <= self.port <= 65535:
            raise ValueError(f"Invalid port number: {self.port}")
        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")
        if self.heartbeat_interval <= 0:
            raise ValueError("Heartbeat interval must be positive")


@dataclass
class MCPClientConfig:
    """MCP Client configuration"""
    agent_id: str
    capabilities: List[Dict[str, Any]] = field(default_factory=list)
    server_host: str = "localhost"
    server_port: int = 8765
    heartbeat_interval: int = 30
    message_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    auto_reconnect: bool = True
    enable_compression: bool = True
    compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.agent_id:
            raise ValueError("Agent ID cannot be empty")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.retry_delay <= 0:
            raise ValueError("Retry delay must be positive")


@dataclass
class MCPMessageConfig:
    """MCP Message handling configuration"""
    max_size: int = 1048576  # 1MB
    batch_size: int = 10
    batch_timeout: float = 0.1
    compression_threshold: int = 1024  # 1KB
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    message_ttl: int = 3600  # 1 hour
    priority_levels: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.max_size <= 0:
            raise ValueError("Max message size must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.batch_timeout <= 0:
            raise ValueError("Batch timeout must be positive")


@dataclass
class MCPContextConfig:
    """MCP Context synchronization configuration"""
    sync_enabled: bool = True
    sync_interval: int = 10
    max_context_size: int = 10485760  # 10MB
    compression_enabled: bool = True
    conflict_resolution_strategy: str = "merge"
    version_tracking: bool = True
    access_control_enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.sync_interval <= 0:
            raise ValueError("Sync interval must be positive")
        if self.max_context_size <= 0:
            raise ValueError("Max context size must be positive")


@dataclass
class MCPPerformanceConfig:
    """MCP Performance optimization configuration"""
    connection_pool_size: int = 10
    message_queue_size: int = 1000
    worker_threads: int = 4
    enable_metrics: bool = True
    metrics_interval: int = 60
    enable_profiling: bool = False
    memory_limit: int = 536870912  # 512MB
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.connection_pool_size <= 0:
            raise ValueError("Connection pool size must be positive")
        if self.worker_threads <= 0:
            raise ValueError("Worker threads must be positive")


@dataclass
class MCPSecurityConfig:
    """MCP Security configuration"""
    enable_authentication: bool = True
    authentication_method: str = "token"
    token_expiry: int = 3600  # 1 hour
    enable_authorization: bool = True
    role_based_access: bool = True
    audit_logging: bool = True
    rate_limiting: bool = True
    max_requests_per_minute: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.token_expiry <= 0:
            raise ValueError("Token expiry must be positive")
        if self.max_requests_per_minute <= 0:
            raise ValueError("Rate limit must be positive")


class MCPConfig(BaseModel):
    """Comprehensive MCP configuration with validation"""
    
    server: MCPServerConfig = Field(default_factory=MCPServerConfig)
    client: MCPClientConfig = Field(default_factory=lambda: MCPClientConfig(agent_id="default"))
    message: MCPMessageConfig = Field(default_factory=MCPMessageConfig)
    context: MCPContextConfig = Field(default_factory=MCPContextConfig)
    performance: MCPPerformanceConfig = Field(default_factory=MCPPerformanceConfig)
    security: MCPSecurityConfig = Field(default_factory=MCPSecurityConfig)
    
    @classmethod
    def from_env(cls, prefix: str = "MCP_") -> "MCPConfig":
        """Create configuration from environment variables"""
        config_data = {}
        
        # Server configuration
        server_config = {}
        if os.getenv(f"{prefix}SERVER_HOST"):
            server_config["host"] = os.getenv(f"{prefix}SERVER_HOST")
        if os.getenv(f"{prefix}SERVER_PORT"):
            server_config["port"] = int(os.getenv(f"{prefix}SERVER_PORT"))
        if os.getenv(f"{prefix}SERVER_MODE"):
            server_config["mode"] = MCPServerMode(os.getenv(f"{prefix}SERVER_MODE"))
        if os.getenv(f"{prefix}SERVER_MAX_CONNECTIONS"):
            server_config["max_connections"] = int(os.getenv(f"{prefix}SERVER_MAX_CONNECTIONS"))
        
        if server_config:
            config_data["server"] = MCPServerConfig(**server_config)
        
        # Client configuration
        client_config = {}
        if os.getenv(f"{prefix}CLIENT_AGENT_ID"):
            client_config["agent_id"] = os.getenv(f"{prefix}CLIENT_AGENT_ID")
        if os.getenv(f"{prefix}CLIENT_SERVER_HOST"):
            client_config["server_host"] = os.getenv(f"{prefix}CLIENT_SERVER_HOST")
        if os.getenv(f"{prefix}CLIENT_SERVER_PORT"):
            client_config["server_port"] = int(os.getenv(f"{prefix}CLIENT_SERVER_PORT"))
        
        if client_config and "agent_id" in client_config:
            config_data["client"] = MCPClientConfig(**client_config)
        
        # Message configuration
        message_config = {}
        if os.getenv(f"{prefix}MESSAGE_MAX_SIZE"):
            message_config["max_size"] = int(os.getenv(f"{prefix}MESSAGE_MAX_SIZE"))
        if os.getenv(f"{prefix}MESSAGE_BATCH_SIZE"):
            message_config["batch_size"] = int(os.getenv(f"{prefix}MESSAGE_BATCH_SIZE"))
        if os.getenv(f"{prefix}MESSAGE_COMPRESSION_THRESHOLD"):
            message_config["compression_threshold"] = int(os.getenv(f"{prefix}MESSAGE_COMPRESSION_THRESHOLD"))
        
        if message_config:
            config_data["message"] = MCPMessageConfig(**message_config)
        
        # Context configuration
        context_config = {}
        if os.getenv(f"{prefix}CONTEXT_SYNC_ENABLED"):
            context_config["sync_enabled"] = os.getenv(f"{prefix}CONTEXT_SYNC_ENABLED").lower() == "true"
        if os.getenv(f"{prefix}CONTEXT_SYNC_INTERVAL"):
            context_config["sync_interval"] = int(os.getenv(f"{prefix}CONTEXT_SYNC_INTERVAL"))
        if os.getenv(f"{prefix}CONTEXT_MAX_SIZE"):
            context_config["max_context_size"] = int(os.getenv(f"{prefix}CONTEXT_MAX_SIZE"))
        
        if context_config:
            config_data["context"] = MCPContextConfig(**context_config)
        
        # Performance configuration
        performance_config = {}
        if os.getenv(f"{prefix}PERFORMANCE_POOL_SIZE"):
            performance_config["connection_pool_size"] = int(os.getenv(f"{prefix}PERFORMANCE_POOL_SIZE"))
        if os.getenv(f"{prefix}PERFORMANCE_WORKER_THREADS"):
            performance_config["worker_threads"] = int(os.getenv(f"{prefix}PERFORMANCE_WORKER_THREADS"))
        
        if performance_config:
            config_data["performance"] = MCPPerformanceConfig(**performance_config)
        
        # Security configuration
        security_config = {}
        if os.getenv(f"{prefix}SECURITY_AUTHENTICATION"):
            security_config["enable_authentication"] = os.getenv(f"{prefix}SECURITY_AUTHENTICATION").lower() == "true"
        if os.getenv(f"{prefix}SECURITY_RATE_LIMIT"):
            security_config["max_requests_per_minute"] = int(os.getenv(f"{prefix}SECURITY_RATE_LIMIT"))
        
        if security_config:
            config_data["security"] = MCPSecurityConfig(**security_config)
        
        return cls(**config_data)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "MCPConfig":
        """Load configuration from JSON or YAML file"""
        import json
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    config_data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to load YAML configuration files")
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls(**config_data)
    
    def to_file(self, config_path: Union[str, Path], format: str = "json") -> None:
        """Save configuration to file"""
        import json
        
        config_path = Path(config_path)
        config_data = self.dict()
        
        with open(config_path, 'w') as f:
            if format.lower() == 'json':
                json.dump(config_data, f, indent=2, default=str)
            elif format.lower() in ['yml', 'yaml']:
                try:
                    import yaml
                    yaml.dump(config_data, f, default_flow_style=False)
                except ImportError:
                    raise ImportError("PyYAML is required to save YAML configuration files")
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def validate_configuration(self) -> List[str]:
        """Validate the entire configuration and return any issues"""
        issues = []
        
        # Validate server configuration
        try:
            if self.server.port == self.client.server_port and self.server.host == self.client.server_host:
                # This is expected for local setup
                pass
        except Exception as e:
            issues.append(f"Server configuration issue: {e}")
        
        # Validate message size limits
        if self.message.max_size < self.context.max_context_size:
            issues.append("Message max size should be at least as large as context max size")
        
        # Validate performance settings
        if self.performance.connection_pool_size > self.server.max_connections:
            issues.append("Connection pool size should not exceed server max connections")
        
        return issues


# Default MCP configuration instance
default_mcp_config = MCPConfig()


def get_mcp_config() -> MCPConfig:
    """Get MCP configuration from environment or default"""
    try:
        return MCPConfig.from_env()
    except Exception:
        return default_mcp_config


def validate_mcp_config(config: MCPConfig) -> bool:
    """Validate MCP configuration and log any issues"""
    import logging
    
    logger = logging.getLogger(__name__)
    issues = config.validate_configuration()
    
    if issues:
        logger.warning("MCP configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("MCP configuration validation passed")
    return True