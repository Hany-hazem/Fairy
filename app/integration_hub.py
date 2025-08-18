"""
Integration Hub for External Tool Connections

This module provides a centralized hub for integrating with external tools and services,
including productivity tools, cloud services, development tools, and communication platforms.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import aiohttp
import os
from pathlib import Path

from .personal_assistant_models import UserPreferences
from .privacy_security_manager import PrivacySecurityManager


class IntegrationType(Enum):
    """Types of external integrations"""
    PRODUCTIVITY = "productivity"
    CLOUD_STORAGE = "cloud_storage"
    DEVELOPMENT = "development"
    COMMUNICATION = "communication"
    CALENDAR = "calendar"
    EMAIL = "email"
    NOTE_TAKING = "note_taking"
    TASK_MANAGEMENT = "task_management"


class IntegrationStatus(Enum):
    """Status of an integration"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AUTHENTICATING = "authenticating"
    RATE_LIMITED = "rate_limited"


@dataclass
class IntegrationConfig:
    """Configuration for an external integration"""
    name: str
    integration_type: IntegrationType
    enabled: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    oauth_token: Optional[str] = None
    refresh_token: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: int = 100  # requests per hour
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationResult:
    """Result of an integration operation"""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseIntegration(ABC):
    """Base class for all external integrations"""
    
    def __init__(self, config: IntegrationConfig, privacy_manager: PrivacySecurityManager):
        self.config = config
        self.privacy_manager = privacy_manager
        self.status = IntegrationStatus.DISCONNECTED
        self.last_error: Optional[str] = None
        self.request_count = 0
        self.last_request_time: Optional[datetime] = None
        self.logger = logging.getLogger(f"integration.{config.name}")
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the external service"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test the connection to the external service"""
        pass
    
    @abstractmethod
    async def get_data(self, endpoint: str, params: Dict[str, Any] = None) -> IntegrationResult:
        """Get data from the external service"""
        pass
    
    @abstractmethod
    async def send_data(self, endpoint: str, data: Dict[str, Any]) -> IntegrationResult:
        """Send data to the external service"""
        pass
    
    async def _make_request(self, method: str, url: str, **kwargs) -> IntegrationResult:
        """Make an HTTP request with rate limiting and error handling"""
        try:
            # Check rate limiting
            if not self._check_rate_limit():
                return IntegrationResult(
                    success=False,
                    error_message="Rate limit exceeded"
                )
            
            # Add authentication headers
            headers = kwargs.get('headers', {})
            headers.update(self._get_auth_headers())
            kwargs['headers'] = headers
            
            # Set timeout
            kwargs['timeout'] = aiohttp.ClientTimeout(total=self.config.timeout)
            
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, **kwargs) as response:
                    self.request_count += 1
                    self.last_request_time = datetime.now()
                    
                    if response.status == 200:
                        data = await response.json()
                        return IntegrationResult(
                            success=True,
                            data=data,
                            status_code=response.status
                        )
                    else:
                        error_text = await response.text()
                        return IntegrationResult(
                            success=False,
                            error_message=f"HTTP {response.status}: {error_text}",
                            status_code=response.status
                        )
        
        except asyncio.TimeoutError:
            return IntegrationResult(
                success=False,
                error_message="Request timeout"
            )
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}")
            return IntegrationResult(
                success=False,
                error_message=str(e)
            )
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        if self.last_request_time is None:
            return True
        
        time_diff = (datetime.now() - self.last_request_time).total_seconds()
        if time_diff < 3600:  # Within the last hour
            return self.request_count < self.config.rate_limit
        else:
            self.request_count = 0  # Reset counter
            return True
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests"""
        headers = {}
        if self.config.api_key:
            headers['Authorization'] = f"Bearer {self.config.api_key}"
        elif self.config.oauth_token:
            headers['Authorization'] = f"Bearer {self.config.oauth_token}"
        return headers


class GoogleDriveIntegration(BaseIntegration):
    """Integration with Google Drive"""
    
    async def authenticate(self) -> bool:
        """Authenticate with Google Drive API"""
        try:
            # In a real implementation, this would handle OAuth2 flow
            if self.config.oauth_token:
                self.status = IntegrationStatus.CONNECTED
                return True
            else:
                self.status = IntegrationStatus.DISCONNECTED
                return False
        except Exception as e:
            self.last_error = str(e)
            self.status = IntegrationStatus.ERROR
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to Google Drive"""
        result = await self.get_data("files", {"pageSize": 1})
        return result.success
    
    async def get_data(self, endpoint: str, params: Dict[str, Any] = None) -> IntegrationResult:
        """Get data from Google Drive API"""
        url = f"https://www.googleapis.com/drive/v3/{endpoint}"
        return await self._make_request("GET", url, params=params)
    
    async def send_data(self, endpoint: str, data: Dict[str, Any]) -> IntegrationResult:
        """Send data to Google Drive API"""
        url = f"https://www.googleapis.com/drive/v3/{endpoint}"
        return await self._make_request("POST", url, json=data)
    
    async def list_files(self, query: str = None) -> IntegrationResult:
        """List files in Google Drive"""
        params = {"pageSize": 100}
        if query:
            params["q"] = query
        return await self.get_data("files", params)
    
    async def download_file(self, file_id: str) -> IntegrationResult:
        """Download a file from Google Drive"""
        return await self.get_data(f"files/{file_id}", {"alt": "media"})


class OneDriveIntegration(BaseIntegration):
    """Integration with Microsoft OneDrive"""
    
    async def authenticate(self) -> bool:
        """Authenticate with OneDrive API"""
        try:
            if self.config.oauth_token:
                self.status = IntegrationStatus.CONNECTED
                return True
            else:
                self.status = IntegrationStatus.DISCONNECTED
                return False
        except Exception as e:
            self.last_error = str(e)
            self.status = IntegrationStatus.ERROR
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to OneDrive"""
        result = await self.get_data("me/drive/root/children")
        return result.success
    
    async def get_data(self, endpoint: str, params: Dict[str, Any] = None) -> IntegrationResult:
        """Get data from OneDrive API"""
        url = f"https://graph.microsoft.com/v1.0/{endpoint}"
        return await self._make_request("GET", url, params=params)
    
    async def send_data(self, endpoint: str, data: Dict[str, Any]) -> IntegrationResult:
        """Send data to OneDrive API"""
        url = f"https://graph.microsoft.com/v1.0/{endpoint}"
        return await self._make_request("POST", url, json=data)


class GitHubIntegration(BaseIntegration):
    """Integration with GitHub"""
    
    async def authenticate(self) -> bool:
        """Authenticate with GitHub API"""
        try:
            if self.config.api_key:  # GitHub personal access token
                self.status = IntegrationStatus.CONNECTED
                return True
            else:
                self.status = IntegrationStatus.DISCONNECTED
                return False
        except Exception as e:
            self.last_error = str(e)
            self.status = IntegrationStatus.ERROR
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to GitHub"""
        result = await self.get_data("user")
        return result.success
    
    async def get_data(self, endpoint: str, params: Dict[str, Any] = None) -> IntegrationResult:
        """Get data from GitHub API"""
        url = f"https://api.github.com/{endpoint}"
        return await self._make_request("GET", url, params=params)
    
    async def send_data(self, endpoint: str, data: Dict[str, Any]) -> IntegrationResult:
        """Send data to GitHub API"""
        url = f"https://api.github.com/{endpoint}"
        return await self._make_request("POST", url, json=data)
    
    async def get_repositories(self) -> IntegrationResult:
        """Get user repositories"""
        return await self.get_data("user/repos")
    
    async def get_issues(self, repo: str) -> IntegrationResult:
        """Get issues for a repository"""
        return await self.get_data(f"repos/{repo}/issues")


class SlackIntegration(BaseIntegration):
    """Integration with Slack"""
    
    async def authenticate(self) -> bool:
        """Authenticate with Slack API"""
        try:
            if self.config.oauth_token:
                self.status = IntegrationStatus.CONNECTED
                return True
            else:
                self.status = IntegrationStatus.DISCONNECTED
                return False
        except Exception as e:
            self.last_error = str(e)
            self.status = IntegrationStatus.ERROR
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to Slack"""
        result = await self.get_data("auth.test")
        return result.success
    
    async def get_data(self, endpoint: str, params: Dict[str, Any] = None) -> IntegrationResult:
        """Get data from Slack API"""
        url = f"https://slack.com/api/{endpoint}"
        return await self._make_request("GET", url, params=params)
    
    async def send_data(self, endpoint: str, data: Dict[str, Any]) -> IntegrationResult:
        """Send data to Slack API"""
        url = f"https://slack.com/api/{endpoint}"
        return await self._make_request("POST", url, json=data)
    
    async def send_message(self, channel: str, text: str) -> IntegrationResult:
        """Send a message to a Slack channel"""
        return await self.send_data("chat.postMessage", {
            "channel": channel,
            "text": text
        })


class IntegrationHub:
    """Central hub for managing external tool integrations"""
    
    def __init__(self, privacy_manager: PrivacySecurityManager):
        self.privacy_manager = privacy_manager
        self.integrations: Dict[str, BaseIntegration] = {}
        self.configs: Dict[str, IntegrationConfig] = {}
        self.logger = logging.getLogger("integration_hub")
        
        # Integration class mapping
        self.integration_classes = {
            "google_drive": GoogleDriveIntegration,
            "onedrive": OneDriveIntegration,
            "github": GitHubIntegration,
            "slack": SlackIntegration,
        }
    
    async def initialize(self, config_path: Optional[str] = None) -> bool:
        """Initialize the integration hub with configurations"""
        try:
            if config_path and os.path.exists(config_path):
                await self._load_configs(config_path)
            else:
                await self._create_default_configs()
            
            # Initialize enabled integrations
            for name, config in self.configs.items():
                if config.enabled:
                    await self.add_integration(name, config)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize integration hub: {str(e)}")
            return False
    
    async def add_integration(self, name: str, config: IntegrationConfig) -> bool:
        """Add a new integration"""
        try:
            if name in self.integration_classes:
                integration_class = self.integration_classes[name]
                integration = integration_class(config, self.privacy_manager)
                
                # Authenticate if credentials are available
                if config.api_key or config.oauth_token:
                    await integration.authenticate()
                
                self.integrations[name] = integration
                self.configs[name] = config
                
                self.logger.info(f"Added integration: {name}")
                return True
            else:
                self.logger.error(f"Unknown integration type: {name}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to add integration {name}: {str(e)}")
            return False
    
    async def remove_integration(self, name: str) -> bool:
        """Remove an integration"""
        try:
            if name in self.integrations:
                del self.integrations[name]
                if name in self.configs:
                    del self.configs[name]
                self.logger.info(f"Removed integration: {name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove integration {name}: {str(e)}")
            return False
    
    async def get_integration(self, name: str) -> Optional[BaseIntegration]:
        """Get an integration by name"""
        return self.integrations.get(name)
    
    async def list_integrations(self) -> Dict[str, Dict[str, Any]]:
        """List all integrations with their status"""
        result = {}
        for name, integration in self.integrations.items():
            result[name] = {
                "type": integration.config.integration_type.value,
                "status": integration.status.value,
                "enabled": integration.config.enabled,
                "last_error": integration.last_error,
                "request_count": integration.request_count
            }
        return result
    
    async def test_all_connections(self) -> Dict[str, bool]:
        """Test connections for all integrations"""
        results = {}
        for name, integration in self.integrations.items():
            if integration.status == IntegrationStatus.CONNECTED:
                results[name] = await integration.test_connection()
            else:
                results[name] = False
        return results
    
    async def sync_files_from_cloud(self, user_id: str, service: str = None) -> List[Dict[str, Any]]:
        """Sync files from cloud storage services"""
        files = []
        
        # Check user permissions
        if not await self.privacy_manager.check_permission(user_id, "cloud_access"):
            self.logger.warning(f"User {user_id} denied cloud access permission")
            return files
        
        services_to_sync = [service] if service else ["google_drive", "onedrive"]
        
        for service_name in services_to_sync:
            integration = self.integrations.get(service_name)
            if integration and integration.status == IntegrationStatus.CONNECTED:
                try:
                    if service_name == "google_drive":
                        result = await integration.list_files()
                        if result.success and result.data:
                            for file_data in result.data.get("files", []):
                                files.append({
                                    "service": service_name,
                                    "id": file_data.get("id"),
                                    "name": file_data.get("name"),
                                    "type": file_data.get("mimeType"),
                                    "modified": file_data.get("modifiedTime"),
                                    "size": file_data.get("size")
                                })
                    
                    elif service_name == "onedrive":
                        result = await integration.get_data("me/drive/root/children")
                        if result.success and result.data:
                            for file_data in result.data.get("value", []):
                                files.append({
                                    "service": service_name,
                                    "id": file_data.get("id"),
                                    "name": file_data.get("name"),
                                    "type": file_data.get("file", {}).get("mimeType", "folder"),
                                    "modified": file_data.get("lastModifiedDateTime"),
                                    "size": file_data.get("size")
                                })
                
                except Exception as e:
                    self.logger.error(f"Failed to sync files from {service_name}: {str(e)}")
        
        return files
    
    async def get_development_context(self, user_id: str) -> Dict[str, Any]:
        """Get development context from integrated tools"""
        context = {}
        
        # Check user permissions
        if not await self.privacy_manager.check_permission(user_id, "dev_tools_access"):
            return context
        
        # Get GitHub repositories and issues
        github = self.integrations.get("github")
        if github and github.status == IntegrationStatus.CONNECTED:
            try:
                repos_result = await github.get_repositories()
                if repos_result.success:
                    context["repositories"] = repos_result.data
                
                # Get issues for the first few repositories
                if repos_result.data:
                    context["issues"] = {}
                    for repo in repos_result.data[:3]:  # Limit to first 3 repos
                        repo_name = repo.get("full_name")
                        if repo_name:
                            issues_result = await github.get_issues(repo_name)
                            if issues_result.success:
                                context["issues"][repo_name] = issues_result.data
            
            except Exception as e:
                self.logger.error(f"Failed to get GitHub context: {str(e)}")
        
        return context
    
    async def send_notification(self, user_id: str, message: str, channel: str = None) -> bool:
        """Send notification through communication tools"""
        # Check user permissions
        if not await self.privacy_manager.check_permission(user_id, "communication_access"):
            return False
        
        # Try Slack first
        slack = self.integrations.get("slack")
        if slack and slack.status == IntegrationStatus.CONNECTED:
            try:
                result = await slack.send_message(channel or "#general", message)
                return result.success
            except Exception as e:
                self.logger.error(f"Failed to send Slack notification: {str(e)}")
        
        return False
    
    async def _load_configs(self, config_path: str) -> None:
        """Load integration configurations from file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for name, config_dict in config_data.items():
                config = IntegrationConfig(
                    name=name,
                    integration_type=IntegrationType(config_dict.get("type", "productivity")),
                    enabled=config_dict.get("enabled", True),
                    api_key=config_dict.get("api_key"),
                    api_secret=config_dict.get("api_secret"),
                    oauth_token=config_dict.get("oauth_token"),
                    refresh_token=config_dict.get("refresh_token"),
                    base_url=config_dict.get("base_url"),
                    rate_limit=config_dict.get("rate_limit", 100),
                    timeout=config_dict.get("timeout", 30),
                    retry_attempts=config_dict.get("retry_attempts", 3),
                    custom_config=config_dict.get("custom_config", {})
                )
                self.configs[name] = config
        
        except Exception as e:
            self.logger.error(f"Failed to load configs: {str(e)}")
            raise
    
    async def _create_default_configs(self) -> None:
        """Create default integration configurations"""
        default_configs = {
            "google_drive": IntegrationConfig(
                name="google_drive",
                integration_type=IntegrationType.CLOUD_STORAGE,
                enabled=False  # Disabled by default until user provides credentials
            ),
            "onedrive": IntegrationConfig(
                name="onedrive",
                integration_type=IntegrationType.CLOUD_STORAGE,
                enabled=False
            ),
            "github": IntegrationConfig(
                name="github",
                integration_type=IntegrationType.DEVELOPMENT,
                enabled=False
            ),
            "slack": IntegrationConfig(
                name="slack",
                integration_type=IntegrationType.COMMUNICATION,
                enabled=False
            )
        }
        
        self.configs.update(default_configs)
    
    async def save_configs(self, config_path: str) -> bool:
        """Save integration configurations to file"""
        try:
            config_data = {}
            for name, config in self.configs.items():
                config_data[name] = {
                    "type": config.integration_type.value,
                    "enabled": config.enabled,
                    "api_key": config.api_key,
                    "api_secret": config.api_secret,
                    "oauth_token": config.oauth_token,
                    "refresh_token": config.refresh_token,
                    "base_url": config.base_url,
                    "rate_limit": config.rate_limit,
                    "timeout": config.timeout,
                    "retry_attempts": config.retry_attempts,
                    "custom_config": config.custom_config
                }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configs: {str(e)}")
            return False