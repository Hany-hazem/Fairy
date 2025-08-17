# MCP and Git Integration - Deployment Guide

This guide provides comprehensive instructions for deploying and maintaining the MCP (Model Context Protocol) and Git integration system.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Backup and Recovery](#backup-and-recovery)
- [Maintenance](#maintenance)
- [Troubleshooting](#troubleshooting)

## Overview

The MCP and Git integration system provides:

- **MCP Server**: Centralized communication hub for AI agents
- **Redis Backend**: Message queuing and persistence
- **Git Workflow**: Automated version control integration
- **Health Monitoring**: Comprehensive system health checks
- **Backup System**: Automated backup and recovery
- **Configuration Management**: Centralized configuration with validation

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Git**: Version 2.30+
- **Python**: Version 3.11+ (for configuration validation)
- **Memory**: Minimum 4GB RAM (8GB+ recommended for production)
- **Storage**: Minimum 20GB free space

### Network Requirements

- **Ports**: 
  - 8765 (MCP Server)
  - 6379 (Redis)
  - 26379 (Redis Sentinel, if HA enabled)
  - 9090 (Prometheus, if monitoring enabled)
  - 3000 (Grafana, if monitoring enabled)

## Configuration

### 1. Environment Configuration

Copy the example environment file and customize it:

```bash
cp .env.mcp.example .env.mcp
```

Edit `.env.mcp` with your specific settings:

```bash
# Basic configuration
DEPLOYMENT_MODE=production
LOG_LEVEL=INFO

# MCP Server
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8765
MCP_SERVER_MAX_CONNECTIONS=100

# Redis
REDIS_URL=redis://redis-mcp:6379
REDIS_MAX_MEMORY=512mb

# Git
GIT_REPO_PATH=/workspace
GIT_USER_NAME="Your Name"
GIT_USER_EMAIL="your.email@example.com"
```

### 2. Configuration Validation

Validate your configuration before deployment:

```bash
python3 scripts/validate_config.py --config-dir ./config
```

### 3. Advanced Configuration

For advanced configuration options, see the configuration modules:

- `app/mcp_config.py` - MCP server configuration
- `app/git_config.py` - Git workflow configuration  
- `app/redis_config.py` - Redis configuration
- `app/config_manager.py` - Configuration management

## Deployment

### Quick Start

For a basic development deployment:

```bash
./scripts/deploy.sh deploy
```

### Production Deployment

For production with monitoring and backup:

```bash
./scripts/deploy.sh -m production -p monitoring,backup deploy
```

### High Availability Deployment

For high availability with Redis Sentinel:

```bash
./scripts/deploy.sh -m production -p ha,monitoring,backup deploy
```

### Deployment Options

| Option | Description | Example |
|--------|-------------|---------|
| `-m, --mode` | Deployment mode | `development`, `staging`, `production` |
| `-p, --profiles` | Docker Compose profiles | `backup`, `monitoring`, `ha` |
| `-f, --file` | Compose file | `docker-compose.mcp.yml` |
| `-c, --config` | Config directory | `./config` |

### Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Server    │    │  Git Workflow   │    │   Monitoring    │
│   (Port 8765)   │    │   Service       │    │ (Prometheus)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Redis Backend  │
                    │   (Port 6379)   │
                    └─────────────────┘
```

## Monitoring

### Health Checks

The system provides comprehensive health monitoring:

```bash
# Check overall system health
curl http://localhost:8765/health/

# Check specific components
curl http://localhost:8765/health/redis
curl http://localhost:8765/health/git
curl http://localhost:8765/health/mcp
```

### Prometheus Metrics

If monitoring is enabled, metrics are available at:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Service Status

Check service status:

```bash
./scripts/deploy.sh status
```

View service logs:

```bash
./scripts/deploy.sh logs
./scripts/deploy.sh logs mcp-server
```

## Backup and Recovery

### Automated Backups

Enable automated backups by setting:

```bash
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=7
```

### Manual Backup

Create a manual backup:

```bash
./scripts/deploy.sh backup
```

### Backup Contents

Backups include:
- Redis data and configuration
- Git repository state and history
- System configuration
- Application logs
- Metadata and manifests

### Recovery

To restore from a backup:

```bash
./scripts/deploy.sh restore backup_name
```

## Maintenance

### Automated Maintenance

Run maintenance operations:

```bash
# Run all maintenance operations
python3 scripts/maintenance.py --operation all

# Run specific operations
python3 scripts/maintenance.py --operation health
python3 scripts/maintenance.py --operation cleanup
python3 scripts/maintenance.py --operation optimize
```

### Log Rotation

Logs are automatically rotated when they exceed 100MB. Configure rotation:

```python
# In maintenance.py
maintenance.rotate_logs(max_size_mb=100, max_files=5)
```

### Resource Cleanup

Clean up unused Docker resources:

```bash
./scripts/deploy.sh cleanup
```

### System Reports

Generate comprehensive system reports:

```bash
python3 scripts/maintenance.py --operation report --output system_report.json
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

**Symptoms**: Services fail to start or immediately exit

**Solutions**:
- Check configuration: `python3 scripts/validate_config.py`
- Verify ports are available: `netstat -tulpn | grep :8765`
- Check Docker logs: `docker-compose -f docker-compose.mcp.yml logs`

#### 2. Redis Connection Issues

**Symptoms**: MCP server can't connect to Redis

**Solutions**:
- Verify Redis is running: `docker-compose -f docker-compose.mcp.yml ps redis-mcp`
- Check Redis logs: `docker-compose -f docker-compose.mcp.yml logs redis-mcp`
- Test Redis connection: `redis-cli -h localhost -p 6379 ping`

#### 3. Git Repository Issues

**Symptoms**: Git operations fail or repository is corrupted

**Solutions**:
- Check repository status: `git status`
- Verify Git configuration: `git config --list`
- Run Git fsck: `git fsck --full`
- Optimize repository: `python3 scripts/maintenance.py --operation optimize`

#### 4. High Memory Usage

**Symptoms**: System runs out of memory

**Solutions**:
- Check Redis memory usage: `redis-cli info memory`
- Adjust Redis max memory: Set `REDIS_MAX_MEMORY` in `.env.mcp`
- Clean up Docker resources: `./scripts/deploy.sh cleanup`
- Monitor with: `docker stats`

#### 5. Configuration Issues

**Symptoms**: Services behave unexpectedly

**Solutions**:
- Validate configuration: `python3 scripts/validate_config.py`
- Check environment variables: `env | grep -E "(MCP|GIT|REDIS)_"`
- Review configuration files in `./config/`

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set in .env.mcp
LOG_LEVEL=DEBUG
DEBUG=true
```

### Health Check Endpoints

Use health endpoints for diagnostics:

```bash
# Overall health
curl -s http://localhost:8765/health/ | jq .

# Component health
curl -s http://localhost:8765/health/redis | jq .
curl -s http://localhost:8765/health/git | jq .
curl -s http://localhost:8765/health/config | jq .

# Metrics
curl -s http://localhost:8765/health/metrics
```

### Log Analysis

Important log locations:
- Application logs: `./logs/`
- Docker logs: `docker-compose logs`
- System logs: `/var/log/`

Common log patterns to look for:
- `ERROR`: Critical errors requiring attention
- `WARNING`: Issues that may need investigation
- `Connection refused`: Network connectivity issues
- `Permission denied`: File system permission issues

### Performance Monitoring

Monitor system performance:

```bash
# System resources
htop
df -h
free -h

# Docker resources
docker stats
docker system df

# Network connections
netstat -tulpn
ss -tulpn
```

## Security Considerations

### Network Security

- Use firewalls to restrict access to service ports
- Consider using TLS for Redis connections in production
- Implement proper authentication for monitoring interfaces

### Data Security

- Enable Redis authentication if needed
- Use Git commit signing for integrity
- Secure backup storage with encryption
- Implement proper access controls

### Container Security

- Run containers as non-root users
- Keep base images updated
- Scan images for vulnerabilities
- Use Docker secrets for sensitive data

## Performance Tuning

### Redis Optimization

```bash
# In .env.mcp
REDIS_MAX_MEMORY=1gb
REDIS_CONNECTION_POOL_SIZE=20
REDIS_PIPELINE_SIZE=100
```

### MCP Server Optimization

```bash
# In .env.mcp
MCP_SERVER_MAX_CONNECTIONS=200
MCP_MESSAGE_BATCH_SIZE=20
MCP_MESSAGE_COMPRESSION=true
```

### Git Optimization

```bash
# In .env.mcp
GIT_OPERATION_TIMEOUT=120
GIT_LARGE_FILE_THRESHOLD=52428800  # 50MB
```

## Support and Documentation

- **Configuration Reference**: See `app/*_config.py` files
- **API Documentation**: Available at service endpoints
- **Health Monitoring**: Use `/health/` endpoints
- **Logs**: Check `./logs/` directory
- **Issues**: Review troubleshooting section above

For additional support, check the system logs and health endpoints for detailed diagnostic information.