# app/redis_config.py
"""
Redis Configuration Management

This module provides comprehensive configuration management for Redis
connections, security, and performance optimization.
"""

import os
import ssl
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, validator
from pathlib import Path
from urllib.parse import urlparse


class RedisSecurityMode(str, Enum):
    """Redis security modes"""
    NONE = "none"
    AUTH = "auth"
    TLS = "tls"
    TLS_AUTH = "tls_auth"


class RedisConnectionMode(str, Enum):
    """Redis connection modes"""
    SINGLE = "single"
    SENTINEL = "sentinel"
    CLUSTER = "cluster"


class RedisCompressionAlgorithm(str, Enum):
    """Redis compression algorithms"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"


@dataclass
class RedisConnectionConfig:
    """Redis connection configuration"""
    url: str = "redis://localhost:6379"
    host: Optional[str] = None
    port: Optional[int] = None
    db: int = 0
    max_connections: int = 20
    connection_timeout: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    socket_keepalive: bool = True
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    def __post_init__(self):
        """Parse URL and extract connection details"""
        if self.url and not self.host:
            parsed = urlparse(self.url)
            self.host = parsed.hostname or "localhost"
            self.port = parsed.port or 6379
            if parsed.path and parsed.path != "/":
                try:
                    self.db = int(parsed.path.lstrip("/"))
                except ValueError:
                    pass
        
        # Validate configuration
        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")
        if self.connection_timeout <= 0:
            raise ValueError("Connection timeout must be positive")


@dataclass
class RedisSecurityConfig:
    """Redis security configuration"""
    mode: RedisSecurityMode = RedisSecurityMode.NONE
    password: Optional[str] = None
    username: Optional[str] = None
    
    # TLS configuration
    tls_cert_file: Optional[str] = None
    tls_key_file: Optional[str] = None
    tls_ca_cert_file: Optional[str] = None
    tls_verify_mode: str = "required"
    tls_check_hostname: bool = True
    
    def __post_init__(self):
        """Validate security configuration"""
        if self.mode in [RedisSecurityMode.AUTH, RedisSecurityMode.TLS_AUTH]:
            if not self.password:
                raise ValueError("Password required for AUTH mode")
        
        if self.mode in [RedisSecurityMode.TLS, RedisSecurityMode.TLS_AUTH]:
            if self.tls_cert_file and not Path(self.tls_cert_file).exists():
                raise ValueError(f"TLS cert file not found: {self.tls_cert_file}")
            if self.tls_key_file and not Path(self.tls_key_file).exists():
                raise ValueError(f"TLS key file not found: {self.tls_key_file}")
            if self.tls_ca_cert_file and not Path(self.tls_ca_cert_file).exists():
                raise ValueError(f"TLS CA cert file not found: {self.tls_ca_cert_file}")
    
    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for TLS connections"""
        if self.mode not in [RedisSecurityMode.TLS, RedisSecurityMode.TLS_AUTH]:
            return None
        
        context = ssl.create_default_context()
        
        if self.tls_verify_mode == "none":
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        elif self.tls_verify_mode == "optional":
            context.verify_mode = ssl.CERT_OPTIONAL
        else:  # required
            context.verify_mode = ssl.CERT_REQUIRED
        
        if self.tls_ca_cert_file:
            context.load_verify_locations(self.tls_ca_cert_file)
        
        if self.tls_cert_file and self.tls_key_file:
            context.load_cert_chain(self.tls_cert_file, self.tls_key_file)
        
        context.check_hostname = self.tls_check_hostname
        
        return context


@dataclass
class RedisMessageQueueConfig:
    """Redis message queue configuration"""
    message_ttl: int = 3600  # 1 hour
    max_retries: int = 3
    queue_max_size: int = 10000
    cleanup_interval: int = 300  # 5 minutes
    batch_size: int = 100
    ack_timeout: int = 30
    dead_letter_queue: bool = True
    
    def __post_init__(self):
        """Validate queue configuration"""
        if self.message_ttl <= 0:
            raise ValueError("Message TTL must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.queue_max_size <= 0:
            raise ValueError("Queue max size must be positive")


@dataclass
class RedisPerformanceConfig:
    """Redis performance configuration"""
    connection_pool_size: int = 10
    pipeline_size: int = 100
    compression_enabled: bool = True
    compression_algorithm: RedisCompressionAlgorithm = RedisCompressionAlgorithm.GZIP
    compression_threshold: int = 1024  # 1KB
    memory_limit: int = 536870912  # 512MB
    eviction_policy: str = "allkeys-lru"
    
    def __post_init__(self):
        """Validate performance configuration"""
        if self.connection_pool_size <= 0:
            raise ValueError("Connection pool size must be positive")
        if self.pipeline_size <= 0:
            raise ValueError("Pipeline size must be positive")
        if self.compression_threshold < 0:
            raise ValueError("Compression threshold cannot be negative")


@dataclass
class RedisSentinelConfig:
    """Redis Sentinel configuration"""
    sentinels: List[Dict[str, Any]] = field(default_factory=list)
    service_name: str = "mymaster"
    socket_timeout: float = 0.1
    sentinel_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate Sentinel configuration"""
        if not self.sentinels:
            raise ValueError("At least one sentinel must be configured")
        if not self.service_name:
            raise ValueError("Service name is required for Sentinel")


@dataclass
class RedisClusterConfig:
    """Redis Cluster configuration"""
    startup_nodes: List[Dict[str, Any]] = field(default_factory=list)
    max_connections_per_node: int = 50
    skip_full_coverage_check: bool = False
    decode_responses: bool = True
    
    def __post_init__(self):
        """Validate Cluster configuration"""
        if not self.startup_nodes:
            raise ValueError("At least one startup node must be configured")


class RedisConfig(BaseModel):
    """Comprehensive Redis configuration with validation"""
    
    connection: RedisConnectionConfig = Field(default_factory=RedisConnectionConfig)
    security: RedisSecurityConfig = Field(default_factory=RedisSecurityConfig)
    queue: RedisMessageQueueConfig = Field(default_factory=RedisMessageQueueConfig)
    performance: RedisPerformanceConfig = Field(default_factory=RedisPerformanceConfig)
    
    # Connection mode and related configs
    connection_mode: RedisConnectionMode = RedisConnectionMode.SINGLE
    sentinel: Optional[RedisSentinelConfig] = None
    cluster: Optional[RedisClusterConfig] = None
    
    @classmethod
    def from_env(cls, prefix: str = "REDIS_") -> "RedisConfig":
        """Create configuration from environment variables"""
        config_data = {}
        
        # Connection configuration
        connection_config = {}
        if os.getenv(f"{prefix}URL"):
            connection_config["url"] = os.getenv(f"{prefix}URL")
        if os.getenv(f"{prefix}HOST"):
            connection_config["host"] = os.getenv(f"{prefix}HOST")
        if os.getenv(f"{prefix}PORT"):
            connection_config["port"] = int(os.getenv(f"{prefix}PORT"))
        if os.getenv(f"{prefix}DB"):
            connection_config["db"] = int(os.getenv(f"{prefix}DB"))
        if os.getenv(f"{prefix}MAX_CONNECTIONS"):
            connection_config["max_connections"] = int(os.getenv(f"{prefix}MAX_CONNECTIONS"))
        if os.getenv(f"{prefix}CONNECTION_TIMEOUT"):
            connection_config["connection_timeout"] = int(os.getenv(f"{prefix}CONNECTION_TIMEOUT"))
        
        if connection_config:
            config_data["connection"] = RedisConnectionConfig(**connection_config)
        
        # Security configuration
        security_config = {}
        if os.getenv(f"{prefix}SECURITY_MODE"):
            security_config["mode"] = RedisSecurityMode(os.getenv(f"{prefix}SECURITY_MODE"))
        if os.getenv(f"{prefix}PASSWORD"):
            security_config["password"] = os.getenv(f"{prefix}PASSWORD")
        if os.getenv(f"{prefix}USERNAME"):
            security_config["username"] = os.getenv(f"{prefix}USERNAME")
        if os.getenv(f"{prefix}TLS_CERT_FILE"):
            security_config["tls_cert_file"] = os.getenv(f"{prefix}TLS_CERT_FILE")
        if os.getenv(f"{prefix}TLS_KEY_FILE"):
            security_config["tls_key_file"] = os.getenv(f"{prefix}TLS_KEY_FILE")
        if os.getenv(f"{prefix}TLS_CA_CERT_FILE"):
            security_config["tls_ca_cert_file"] = os.getenv(f"{prefix}TLS_CA_CERT_FILE")
        
        if security_config:
            config_data["security"] = RedisSecurityConfig(**security_config)
        
        # Queue configuration
        queue_config = {}
        if os.getenv(f"{prefix}MESSAGE_TTL"):
            queue_config["message_ttl"] = int(os.getenv(f"{prefix}MESSAGE_TTL"))
        if os.getenv(f"{prefix}MESSAGE_MAX_RETRIES"):
            queue_config["max_retries"] = int(os.getenv(f"{prefix}MESSAGE_MAX_RETRIES"))
        if os.getenv(f"{prefix}QUEUE_MAX_SIZE"):
            queue_config["queue_max_size"] = int(os.getenv(f"{prefix}QUEUE_MAX_SIZE"))
        if os.getenv(f"{prefix}BATCH_SIZE"):
            queue_config["batch_size"] = int(os.getenv(f"{prefix}BATCH_SIZE"))
        
        if queue_config:
            config_data["queue"] = RedisMessageQueueConfig(**queue_config)
        
        # Performance configuration
        performance_config = {}
        if os.getenv(f"{prefix}CONNECTION_POOL_SIZE"):
            performance_config["connection_pool_size"] = int(os.getenv(f"{prefix}CONNECTION_POOL_SIZE"))
        if os.getenv(f"{prefix}PIPELINE_SIZE"):
            performance_config["pipeline_size"] = int(os.getenv(f"{prefix}PIPELINE_SIZE"))
        if os.getenv(f"{prefix}COMPRESSION_ENABLED"):
            performance_config["compression_enabled"] = os.getenv(f"{prefix}COMPRESSION_ENABLED").lower() == "true"
        if os.getenv(f"{prefix}COMPRESSION_THRESHOLD"):
            performance_config["compression_threshold"] = int(os.getenv(f"{prefix}COMPRESSION_THRESHOLD"))
        
        if performance_config:
            config_data["performance"] = RedisPerformanceConfig(**performance_config)
        
        # Connection mode
        if os.getenv(f"{prefix}CONNECTION_MODE"):
            config_data["connection_mode"] = RedisConnectionMode(os.getenv(f"{prefix}CONNECTION_MODE"))
        
        return cls(**config_data)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "RedisConfig":
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
    
    def get_redis_kwargs(self) -> Dict[str, Any]:
        """Get Redis connection kwargs from configuration"""
        kwargs = {
            "host": self.connection.host,
            "port": self.connection.port,
            "db": self.connection.db,
            "socket_timeout": self.connection.socket_timeout,
            "socket_connect_timeout": self.connection.socket_connect_timeout,
            "socket_keepalive": self.connection.socket_keepalive,
            "retry_on_timeout": self.connection.retry_on_timeout,
            "health_check_interval": self.connection.health_check_interval,
        }
        
        # Add authentication if configured
        if self.security.mode in [RedisSecurityMode.AUTH, RedisSecurityMode.TLS_AUTH]:
            if self.security.password:
                kwargs["password"] = self.security.password
            if self.security.username:
                kwargs["username"] = self.security.username
        
        # Add TLS if configured
        if self.security.mode in [RedisSecurityMode.TLS, RedisSecurityMode.TLS_AUTH]:
            kwargs["ssl"] = True
            ssl_context = self.security.get_ssl_context()
            if ssl_context:
                kwargs["ssl_context"] = ssl_context
        
        return kwargs
    
    def validate_redis_connection(self) -> List[str]:
        """Validate Redis connection configuration"""
        issues = []
        
        # Validate connection parameters
        if not self.connection.host:
            issues.append("Redis host is required")
        
        if not 1 <= self.connection.port <= 65535:
            issues.append(f"Invalid Redis port: {self.connection.port}")
        
        if not 0 <= self.connection.db <= 15:
            issues.append(f"Invalid Redis database number: {self.connection.db}")
        
        # Validate security configuration
        if self.security.mode in [RedisSecurityMode.TLS, RedisSecurityMode.TLS_AUTH]:
            if self.security.tls_cert_file and not Path(self.security.tls_cert_file).exists():
                issues.append(f"TLS certificate file not found: {self.security.tls_cert_file}")
        
        # Validate Sentinel configuration
        if self.connection_mode == RedisConnectionMode.SENTINEL:
            if not self.sentinel or not self.sentinel.sentinels:
                issues.append("Sentinel configuration required for Sentinel mode")
        
        # Validate Cluster configuration
        if self.connection_mode == RedisConnectionMode.CLUSTER:
            if not self.cluster or not self.cluster.startup_nodes:
                issues.append("Cluster configuration required for Cluster mode")
        
        return issues


# Default Redis configuration instance
default_redis_config = RedisConfig()


def get_redis_config() -> RedisConfig:
    """Get Redis configuration from environment or default"""
    try:
        return RedisConfig.from_env()
    except Exception:
        return default_redis_config


def validate_redis_config(config: RedisConfig) -> bool:
    """Validate Redis configuration and log any issues"""
    import logging
    
    logger = logging.getLogger(__name__)
    issues = config.validate_redis_connection()
    
    if issues:
        logger.warning("Redis configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("Redis configuration validation passed")
    return True