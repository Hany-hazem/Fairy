# app/config_manager.py
"""
Configuration Manager for MCP and Git Integration

This module provides centralized configuration management for all MCP and Git
integration components with validation, environment loading, and file persistence.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from .mcp_config import MCPConfig, get_mcp_config, validate_mcp_config
from .git_config import GitWorkflowConfig, get_git_config, validate_git_config
from .redis_config import RedisConfig, get_redis_config, validate_redis_config

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationStatus:
    """Configuration validation status"""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    last_validated: datetime
    
    def add_issue(self, issue: str) -> None:
        """Add a configuration issue"""
        self.issues.append(issue)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a configuration warning"""
        self.warnings.append(warning)


class ConfigurationManager:
    """
    Centralized configuration manager for MCP and Git integration
    
    Provides:
    - Unified configuration loading from environment and files
    - Configuration validation and issue reporting
    - Configuration persistence and backup
    - Runtime configuration updates
    - Configuration monitoring and change detection
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory for configuration files (default: ./.config)
        """
        self.config_dir = Path(config_dir or "./.config")
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration instances
        self._mcp_config: Optional[MCPConfig] = None
        self._git_config: Optional[GitWorkflowConfig] = None
        self._redis_config: Optional[RedisConfig] = None
        
        # Configuration status
        self._status = ConfigurationStatus(
            is_valid=False,
            issues=[],
            warnings=[],
            last_validated=datetime.now()
        )
        
        # Configuration file paths
        self.mcp_config_file = self.config_dir / "mcp_config.json"
        self.git_config_file = self.config_dir / "git_config.json"
        self.redis_config_file = self.config_dir / "redis_config.json"
        self.combined_config_file = self.config_dir / "integration_config.json"
    
    @property
    def mcp_config(self) -> MCPConfig:
        """Get MCP configuration"""
        if self._mcp_config is None:
            self._mcp_config = self._load_mcp_config()
        return self._mcp_config
    
    @property
    def git_config(self) -> GitWorkflowConfig:
        """Get Git workflow configuration"""
        if self._git_config is None:
            self._git_config = self._load_git_config()
        return self._git_config
    
    @property
    def redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        if self._redis_config is None:
            self._redis_config = self._load_redis_config()
        return self._redis_config
    
    @property
    def status(self) -> ConfigurationStatus:
        """Get configuration status"""
        return self._status
    
    def _load_mcp_config(self) -> MCPConfig:
        """Load MCP configuration from environment and files"""
        try:
            # Try to load from file first
            if self.mcp_config_file.exists():
                logger.info(f"Loading MCP config from {self.mcp_config_file}")
                return MCPConfig.from_file(self.mcp_config_file)
            else:
                # Load from environment
                logger.info("Loading MCP config from environment")
                return get_mcp_config()
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            self._status.add_issue(f"MCP config loading failed: {e}")
            return MCPConfig()  # Return default config
    
    def _load_git_config(self) -> GitWorkflowConfig:
        """Load Git workflow configuration from environment and files"""
        try:
            # Try to load from file first
            if self.git_config_file.exists():
                logger.info(f"Loading Git config from {self.git_config_file}")
                return GitWorkflowConfig.from_file(self.git_config_file)
            else:
                # Load from environment
                logger.info("Loading Git config from environment")
                return get_git_config()
        except Exception as e:
            logger.error(f"Failed to load Git config: {e}")
            self._status.add_issue(f"Git config loading failed: {e}")
            return GitWorkflowConfig()  # Return default config
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration from environment and files"""
        try:
            # Try to load from file first
            if self.redis_config_file.exists():
                logger.info(f"Loading Redis config from {self.redis_config_file}")
                return RedisConfig.from_file(self.redis_config_file)
            else:
                # Load from environment
                logger.info("Loading Redis config from environment")
                return get_redis_config()
        except Exception as e:
            logger.error(f"Failed to load Redis config: {e}")
            self._status.add_issue(f"Redis config loading failed: {e}")
            return RedisConfig()  # Return default config
    
    def validate_all_configurations(self) -> ConfigurationStatus:
        """Validate all configurations and return status"""
        self._status = ConfigurationStatus(
            is_valid=True,
            issues=[],
            warnings=[],
            last_validated=datetime.now()
        )
        
        # Validate MCP configuration
        try:
            if not validate_mcp_config(self.mcp_config):
                self._status.add_issue("MCP configuration validation failed")
        except Exception as e:
            self._status.add_issue(f"MCP configuration validation error: {e}")
        
        # Validate Git configuration
        try:
            if not validate_git_config(self.git_config):
                self._status.add_issue("Git configuration validation failed")
        except Exception as e:
            self._status.add_issue(f"Git configuration validation error: {e}")
        
        # Validate Redis configuration
        try:
            if not validate_redis_config(self.redis_config):
                self._status.add_issue("Redis configuration validation failed")
        except Exception as e:
            self._status.add_issue(f"Redis configuration validation error: {e}")
        
        # Cross-configuration validation
        self._validate_cross_configuration()
        
        logger.info(f"Configuration validation completed. Valid: {self._status.is_valid}")
        if self._status.issues:
            logger.warning(f"Configuration issues: {self._status.issues}")
        if self._status.warnings:
            logger.info(f"Configuration warnings: {self._status.warnings}")
        
        return self._status
    
    def _validate_cross_configuration(self) -> None:
        """Validate cross-configuration dependencies"""
        # Check Redis URL consistency
        if hasattr(self.mcp_config, 'redis_url') and self.redis_config.connection.url:
            # This would require extending MCP config to include Redis URL
            pass
        
        # Check port conflicts
        mcp_port = self.mcp_config.server.port
        redis_port = self.redis_config.connection.port
        
        if mcp_port == redis_port:
            self._status.add_warning(f"MCP server and Redis using same port: {mcp_port}")
        
        # Check Git repository path
        git_repo_path = Path(self.git_config.repository.repo_path)
        if not git_repo_path.exists():
            self._status.add_issue(f"Git repository path does not exist: {git_repo_path}")
        elif not (git_repo_path / ".git").exists():
            self._status.add_issue(f"Not a Git repository: {git_repo_path}")
    
    def save_configurations(self) -> bool:
        """Save all configurations to files"""
        try:
            # Save individual configurations
            self.mcp_config.to_file(self.mcp_config_file)
            self.git_config.to_file(self.git_config_file)
            self.redis_config.to_file(self.redis_config_file)
            
            # Save combined configuration
            combined_config = {
                "mcp": self.mcp_config.dict(),
                "git": self.git_config.dict(),
                "redis": self.redis_config.dict(),
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "version": "1.0.0"
                }
            }
            
            with open(self.combined_config_file, 'w') as f:
                json.dump(combined_config, f, indent=2, default=str)
            
            logger.info(f"Configurations saved to {self.config_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
            return False
    
    def load_configurations(self) -> bool:
        """Reload all configurations from files"""
        try:
            self._mcp_config = None
            self._git_config = None
            self._redis_config = None
            
            # Force reload
            _ = self.mcp_config
            _ = self.git_config
            _ = self.redis_config
            
            logger.info("Configurations reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configurations: {e}")
            return False
    
    def backup_configurations(self, backup_dir: Optional[Union[str, Path]] = None) -> bool:
        """Create backup of current configurations"""
        try:
            backup_dir = Path(backup_dir or (self.config_dir / "backups"))
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"config_backup_{timestamp}.json"
            
            backup_data = {
                "mcp": self.mcp_config.dict(),
                "git": self.git_config.dict(),
                "redis": self.redis_config.dict(),
                "metadata": {
                    "backup_created": datetime.now().isoformat(),
                    "version": "1.0.0"
                }
            }
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Configuration backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create configuration backup: {e}")
            return False
    
    def restore_configurations(self, backup_file: Union[str, Path]) -> bool:
        """Restore configurations from backup"""
        try:
            backup_file = Path(backup_file)
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_file}")
            
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            # Restore configurations
            if "mcp" in backup_data:
                self._mcp_config = MCPConfig(**backup_data["mcp"])
            
            if "git" in backup_data:
                self._git_config = GitWorkflowConfig(**backup_data["git"])
            
            if "redis" in backup_data:
                self._redis_config = RedisConfig(**backup_data["redis"])
            
            logger.info(f"Configurations restored from: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore configurations: {e}")
            return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            "mcp": {
                "server_host": self.mcp_config.server.host,
                "server_port": self.mcp_config.server.port,
                "mode": self.mcp_config.server.mode.value,
                "max_connections": self.mcp_config.server.max_connections,
            },
            "git": {
                "repo_path": self.git_config.repository.repo_path,
                "default_branch": self.git_config.repository.default_branch,
                "workflow_mode": self.git_config.workflow_mode.value,
                "auto_commit": self.git_config.commit.auto_commit,
            },
            "redis": {
                "host": self.redis_config.connection.host,
                "port": self.redis_config.connection.port,
                "db": self.redis_config.connection.db,
                "security_mode": self.redis_config.security.mode.value,
            },
            "status": {
                "is_valid": self._status.is_valid,
                "issues_count": len(self._status.issues),
                "warnings_count": len(self._status.warnings),
                "last_validated": self._status.last_validated.isoformat(),
            }
        }
    
    def update_mcp_config(self, **kwargs) -> bool:
        """Update MCP configuration with new values"""
        try:
            # Create new config with updates
            current_dict = self.mcp_config.dict()
            
            # Apply updates (nested updates supported)
            for key, value in kwargs.items():
                if '.' in key:
                    # Handle nested keys like 'server.port'
                    keys = key.split('.')
                    target = current_dict
                    for k in keys[:-1]:
                        target = target[k]
                    target[keys[-1]] = value
                else:
                    current_dict[key] = value
            
            self._mcp_config = MCPConfig(**current_dict)
            logger.info("MCP configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update MCP configuration: {e}")
            return False
    
    def update_git_config(self, **kwargs) -> bool:
        """Update Git configuration with new values"""
        try:
            # Create new config with updates
            current_dict = self.git_config.dict()
            
            # Apply updates (nested updates supported)
            for key, value in kwargs.items():
                if '.' in key:
                    # Handle nested keys like 'repository.repo_path'
                    keys = key.split('.')
                    target = current_dict
                    for k in keys[:-1]:
                        target = target[k]
                    target[keys[-1]] = value
                else:
                    current_dict[key] = value
            
            self._git_config = GitWorkflowConfig(**current_dict)
            logger.info("Git configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update Git configuration: {e}")
            return False
    
    def update_redis_config(self, **kwargs) -> bool:
        """Update Redis configuration with new values"""
        try:
            # Create new config with updates
            current_dict = self.redis_config.dict()
            
            # Apply updates (nested updates supported)
            for key, value in kwargs.items():
                if '.' in key:
                    # Handle nested keys like 'connection.host'
                    keys = key.split('.')
                    target = current_dict
                    for k in keys[:-1]:
                        target = target[k]
                    target[keys[-1]] = value
                else:
                    current_dict[key] = value
            
            self._redis_config = RedisConfig(**current_dict)
            logger.info("Redis configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update Redis configuration: {e}")
            return False


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(config_dir: Optional[Union[str, Path]] = None) -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_dir)
    
    return _config_manager


def initialize_configuration(config_dir: Optional[Union[str, Path]] = None) -> ConfigurationStatus:
    """Initialize and validate all configurations"""
    config_manager = get_config_manager(config_dir)
    return config_manager.validate_all_configurations()


def get_integration_config() -> Dict[str, Any]:
    """Get complete integration configuration"""
    config_manager = get_config_manager()
    return {
        "mcp": config_manager.mcp_config,
        "git": config_manager.git_config,
        "redis": config_manager.redis_config,
        "status": config_manager.status
    }