# app/health_endpoints.py
"""
Health Check Endpoints for MCP and Git Integration

This module provides comprehensive health check endpoints for monitoring
the status of MCP server, Redis backend, Git workflow, and integration components.
"""

import asyncio
import json
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from .config_manager import get_config_manager
from .redis_config import get_redis_config
from .git_config import get_git_config
from .mcp_config import get_mcp_config

logger = logging.getLogger(__name__)

# Create router for health endpoints
health_router = APIRouter(prefix="/health", tags=["health"])


class HealthStatus(BaseModel):
    """Health status model"""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    details: Dict[str, Any]
    checks: Dict[str, Dict[str, Any]]


class ComponentHealth(BaseModel):
    """Individual component health"""
    name: str
    status: str
    message: str
    last_check: datetime
    metrics: Optional[Dict[str, Any]] = None


class HealthChecker:
    """Comprehensive health checker for all integration components"""
    
    def __init__(self):
        """Initialize health checker"""
        self.config_manager = get_config_manager()
        self.redis_config = get_redis_config()
        self.git_config = get_git_config()
        self.mcp_config = get_mcp_config()
    
    async def check_overall_health(self) -> HealthStatus:
        """Check overall system health"""
        checks = {}
        overall_status = "healthy"
        
        # Check all components
        components = [
            ("redis", self.check_redis_health),
            ("git", self.check_git_health),
            ("mcp_server", self.check_mcp_server_health),
            ("configuration", self.check_configuration_health),
            ("filesystem", self.check_filesystem_health),
        ]
        
        for component_name, check_func in components:
            try:
                component_health = await check_func()
                checks[component_name] = {
                    "status": component_health.status,
                    "message": component_health.message,
                    "last_check": component_health.last_check.isoformat(),
                    "metrics": component_health.metrics or {}
                }
                
                # Update overall status
                if component_health.status == "unhealthy":
                    overall_status = "unhealthy"
                elif component_health.status == "degraded" and overall_status == "healthy":
                    overall_status = "degraded"
                    
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                checks[component_name] = {
                    "status": "unhealthy",
                    "message": f"Health check failed: {e}",
                    "last_check": datetime.now().isoformat(),
                    "metrics": {}
                }
                overall_status = "unhealthy"
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.now(),
            details={
                "total_components": len(components),
                "healthy_components": sum(1 for c in checks.values() if c["status"] == "healthy"),
                "degraded_components": sum(1 for c in checks.values() if c["status"] == "degraded"),
                "unhealthy_components": sum(1 for c in checks.values() if c["status"] == "unhealthy"),
            },
            checks=checks
        )
    
    async def check_redis_health(self) -> ComponentHealth:
        """Check Redis health"""
        try:
            import redis.asyncio as redis
            
            # Create Redis connection
            redis_client = redis.Redis(
                host=self.redis_config.connection.host,
                port=self.redis_config.connection.port,
                db=self.redis_config.connection.db,
                password=self.redis_config.security.password,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            start_time = datetime.now()
            await redis_client.ping()
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Get Redis info
            info = await redis_client.info()
            
            # Get memory usage
            memory_used = info.get('used_memory', 0)
            memory_max = info.get('maxmemory', 0)
            memory_usage_percent = (memory_used / memory_max * 100) if memory_max > 0 else 0
            
            # Determine status
            status = "healthy"
            message = "Redis is healthy"
            
            if response_time > 1000:  # 1 second
                status = "degraded"
                message = f"Redis response time is high: {response_time:.2f}ms"
            
            if memory_usage_percent > 90:
                status = "degraded" if status == "healthy" else "unhealthy"
                message = f"Redis memory usage is high: {memory_usage_percent:.1f}%"
            
            await redis_client.close()
            
            return ComponentHealth(
                name="redis",
                status=status,
                message=message,
                last_check=datetime.now(),
                metrics={
                    "response_time_ms": response_time,
                    "memory_used_bytes": memory_used,
                    "memory_usage_percent": memory_usage_percent,
                    "connected_clients": info.get('connected_clients', 0),
                    "total_commands_processed": info.get('total_commands_processed', 0),
                    "uptime_seconds": info.get('uptime_in_seconds', 0)
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="redis",
                status="unhealthy",
                message=f"Redis connection failed: {e}",
                last_check=datetime.now()
            )
    
    async def check_git_health(self) -> ComponentHealth:
        """Check Git repository health"""
        try:
            repo_path = Path(self.git_config.repository.repo_path)
            
            # Check if repository exists
            if not repo_path.exists():
                return ComponentHealth(
                    name="git",
                    status="unhealthy",
                    message=f"Repository path does not exist: {repo_path}",
                    last_check=datetime.now()
                )
            
            if not (repo_path / ".git").exists():
                return ComponentHealth(
                    name="git",
                    status="unhealthy",
                    message=f"Not a Git repository: {repo_path}",
                    last_check=datetime.now()
                )
            
            # Check Git status
            status_cmd = ["git", "-C", str(repo_path), "status", "--porcelain"]
            status_result = subprocess.run(status_cmd, capture_output=True, text=True, timeout=10)
            
            if status_result.returncode != 0:
                return ComponentHealth(
                    name="git",
                    status="unhealthy",
                    message=f"Git status check failed: {status_result.stderr}",
                    last_check=datetime.now()
                )
            
            # Get current branch
            branch_cmd = ["git", "-C", str(repo_path), "branch", "--show-current"]
            branch_result = subprocess.run(branch_cmd, capture_output=True, text=True, timeout=5)
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
            
            # Get repository size
            repo_size = sum(f.stat().st_size for f in repo_path.rglob('*') if f.is_file())
            
            # Count uncommitted changes
            uncommitted_files = len(status_result.stdout.strip().split('\n')) if status_result.stdout.strip() else 0
            
            # Determine status
            status = "healthy"
            message = "Git repository is healthy"
            
            if uncommitted_files > 10:
                status = "degraded"
                message = f"Many uncommitted changes: {uncommitted_files} files"
            
            # Check if repository is too large (> 1GB)
            if repo_size > 1024 * 1024 * 1024:
                status = "degraded" if status == "healthy" else status
                message = f"Repository is large: {repo_size / (1024*1024*1024):.1f}GB"
            
            return ComponentHealth(
                name="git",
                status=status,
                message=message,
                last_check=datetime.now(),
                metrics={
                    "current_branch": current_branch,
                    "uncommitted_files": uncommitted_files,
                    "repository_size_bytes": repo_size,
                    "repository_path": str(repo_path)
                }
            )
            
        except subprocess.TimeoutExpired:
            return ComponentHealth(
                name="git",
                status="degraded",
                message="Git command timed out",
                last_check=datetime.now()
            )
        except Exception as e:
            return ComponentHealth(
                name="git",
                status="unhealthy",
                message=f"Git health check failed: {e}",
                last_check=datetime.now()
            )
    
    async def check_mcp_server_health(self) -> ComponentHealth:
        """Check MCP server health"""
        try:
            import socket
            
            # Test MCP server connection
            host = self.mcp_config.server.host
            port = self.mcp_config.server.port
            
            start_time = datetime.now()
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Determine status based on response time
            status = "healthy"
            message = "MCP server is healthy"
            
            if response_time > 1000:  # 1 second
                status = "degraded"
                message = f"MCP server response time is high: {response_time:.2f}ms"
            
            return ComponentHealth(
                name="mcp_server",
                status=status,
                message=message,
                last_check=datetime.now(),
                metrics={
                    "response_time_ms": response_time,
                    "server_host": host,
                    "server_port": port,
                    "server_mode": self.mcp_config.server.mode.value
                }
            )
            
        except socket.timeout:
            return ComponentHealth(
                name="mcp_server",
                status="unhealthy",
                message="MCP server connection timed out",
                last_check=datetime.now()
            )
        except ConnectionRefusedError:
            return ComponentHealth(
                name="mcp_server",
                status="unhealthy",
                message="MCP server connection refused",
                last_check=datetime.now()
            )
        except Exception as e:
            return ComponentHealth(
                name="mcp_server",
                status="unhealthy",
                message=f"MCP server health check failed: {e}",
                last_check=datetime.now()
            )
    
    async def check_configuration_health(self) -> ComponentHealth:
        """Check configuration health"""
        try:
            # Validate all configurations
            config_status = self.config_manager.validate_all_configurations()
            
            status = "healthy" if config_status.is_valid else "degraded"
            message = "Configuration is valid" if config_status.is_valid else "Configuration has issues"
            
            if config_status.issues:
                status = "unhealthy"
                message = f"Configuration has {len(config_status.issues)} critical issues"
            
            return ComponentHealth(
                name="configuration",
                status=status,
                message=message,
                last_check=datetime.now(),
                metrics={
                    "is_valid": config_status.is_valid,
                    "issues_count": len(config_status.issues),
                    "warnings_count": len(config_status.warnings),
                    "last_validated": config_status.last_validated.isoformat(),
                    "issues": config_status.issues[:5],  # First 5 issues
                    "warnings": config_status.warnings[:5]  # First 5 warnings
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="configuration",
                status="unhealthy",
                message=f"Configuration health check failed: {e}",
                last_check=datetime.now()
            )
    
    async def check_filesystem_health(self) -> ComponentHealth:
        """Check filesystem health"""
        try:
            import shutil
            
            # Check disk space
            total, used, free = shutil.disk_usage("/")
            usage_percent = (used / total) * 100
            
            # Check important directories
            important_dirs = [
                Path("/app"),
                Path("/app/logs"),
                Path(self.git_config.repository.repo_path),
            ]
            
            missing_dirs = [d for d in important_dirs if not d.exists()]
            
            # Determine status
            status = "healthy"
            message = "Filesystem is healthy"
            
            if usage_percent > 90:
                status = "unhealthy"
                message = f"Disk usage is critical: {usage_percent:.1f}%"
            elif usage_percent > 80:
                status = "degraded"
                message = f"Disk usage is high: {usage_percent:.1f}%"
            
            if missing_dirs:
                status = "unhealthy"
                message = f"Missing directories: {[str(d) for d in missing_dirs]}"
            
            return ComponentHealth(
                name="filesystem",
                status=status,
                message=message,
                last_check=datetime.now(),
                metrics={
                    "disk_total_bytes": total,
                    "disk_used_bytes": used,
                    "disk_free_bytes": free,
                    "disk_usage_percent": usage_percent,
                    "missing_directories": [str(d) for d in missing_dirs]
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                name="filesystem",
                status="unhealthy",
                message=f"Filesystem health check failed: {e}",
                last_check=datetime.now()
            )


# Global health checker instance
health_checker = HealthChecker()


@health_router.get("/", response_model=HealthStatus)
async def get_overall_health():
    """Get overall system health status"""
    return await health_checker.check_overall_health()


@health_router.get("/redis", response_model=ComponentHealth)
async def get_redis_health():
    """Get Redis health status"""
    return await health_checker.check_redis_health()


@health_router.get("/git", response_model=ComponentHealth)
async def get_git_health():
    """Get Git repository health status"""
    return await health_checker.check_git_health()


@health_router.get("/mcp", response_model=ComponentHealth)
async def get_mcp_health():
    """Get MCP server health status"""
    return await health_checker.check_mcp_server_health()


@health_router.get("/config", response_model=ComponentHealth)
async def get_config_health():
    """Get configuration health status"""
    return await health_checker.check_configuration_health()


@health_router.get("/filesystem", response_model=ComponentHealth)
async def get_filesystem_health():
    """Get filesystem health status"""
    return await health_checker.check_filesystem_health()


@health_router.get("/ready")
async def readiness_probe():
    """Kubernetes readiness probe endpoint"""
    health_status = await health_checker.check_overall_health()
    
    if health_status.status == "unhealthy":
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": datetime.now().isoformat()}


@health_router.get("/live")
async def liveness_probe():
    """Kubernetes liveness probe endpoint"""
    # Simple liveness check - just return OK if the service is running
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@health_router.get("/metrics")
async def get_metrics():
    """Get Prometheus-compatible metrics"""
    health_status = await health_checker.check_overall_health()
    
    metrics = []
    
    # Overall health metric
    status_value = {"healthy": 1, "degraded": 0.5, "unhealthy": 0}.get(health_status.status, 0)
    metrics.append(f"mcp_integration_health_status {status_value}")
    
    # Component health metrics
    for component, check in health_status.checks.items():
        component_value = {"healthy": 1, "degraded": 0.5, "unhealthy": 0}.get(check["status"], 0)
        metrics.append(f'mcp_integration_component_health{{component="{component}"}} {component_value}')
        
        # Add component-specific metrics
        for metric_name, metric_value in check.get("metrics", {}).items():
            if isinstance(metric_value, (int, float)):
                metrics.append(f'mcp_integration_{component}_{metric_name} {metric_value}')
    
    return "\n".join(metrics)