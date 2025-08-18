"""
Tests for Integration Hub

This module contains comprehensive tests for the IntegrationHub class and its
external tool integration capabilities.
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from app.integration_hub import (
    IntegrationHub, BaseIntegration, GoogleDriveIntegration, OneDriveIntegration,
    GitHubIntegration, SlackIntegration, IntegrationConfig, IntegrationResult,
    IntegrationType, IntegrationStatus
)
from app.privacy_security_manager import PrivacySecurityManager


class TestIntegrationConfig:
    """Test IntegrationConfig data class"""
    
    def test_integration_config_creation(self):
        """Test creating an integration config"""
        config = IntegrationConfig(
            name="test_integration",
            integration_type=IntegrationType.PRODUCTIVITY,
            api_key="test_key"
        )
        
        assert config.name == "test_integration"
        assert config.integration_type == IntegrationType.PRODUCTIVITY
        assert config.api_key == "test_key"
        assert config.enabled is True
        assert config.rate_limit == 100
        assert config.timeout == 30


class TestIntegrationResult:
    """Test IntegrationResult data class"""
    
    def test_integration_result_success(self):
        """Test successful integration result"""
        result = IntegrationResult(
            success=True,
            data={"test": "data"},
            status_code=200
        )
        
        assert result.success is True
        assert result.data == {"test": "data"}
        assert result.status_code == 200
        assert result.error_message is None
    
    def test_integration_result_failure(self):
        """Test failed integration result"""
        result = IntegrationResult(
            success=False,
            error_message="Test error",
            status_code=400
        )
        
        assert result.success is False
        assert result.error_message == "Test error"
        assert result.status_code == 400
        assert result.data is None


class MockIntegration(BaseIntegration):
    """Mock integration for testing"""
    
    async def authenticate(self) -> bool:
        if self.config.api_key:
            self.status = IntegrationStatus.CONNECTED
            return True
        return False
    
    async def test_connection(self) -> bool:
        return self.status == IntegrationStatus.CONNECTED
    
    async def get_data(self, endpoint: str, params: dict = None) -> IntegrationResult:
        if self.status == IntegrationStatus.CONNECTED:
            return IntegrationResult(success=True, data={"endpoint": endpoint, "params": params})
        return IntegrationResult(success=False, error_message="Not connected")
    
    async def send_data(self, endpoint: str, data: dict) -> IntegrationResult:
        if self.status == IntegrationStatus.CONNECTED:
            return IntegrationResult(success=True, data={"sent": data})
        return IntegrationResult(success=False, error_message="Not connected")


class TestBaseIntegration:
    """Test BaseIntegration abstract class"""
    
    @pytest.fixture
    def privacy_manager(self):
        """Create a mock privacy manager"""
        manager = Mock(spec=PrivacySecurityManager)
        manager.check_permission = AsyncMock(return_value=True)
        return manager
    
    @pytest.fixture
    def integration_config(self):
        """Create a test integration config"""
        return IntegrationConfig(
            name="test_integration",
            integration_type=IntegrationType.PRODUCTIVITY,
            api_key="test_key"
        )
    
    @pytest.fixture
    def mock_integration(self, integration_config, privacy_manager):
        """Create a mock integration"""
        return MockIntegration(integration_config, privacy_manager)
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, mock_integration):
        """Test successful authentication"""
        result = await mock_integration.authenticate()
        assert result is True
        assert mock_integration.status == IntegrationStatus.CONNECTED
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, privacy_manager):
        """Test failed authentication"""
        config = IntegrationConfig(
            name="test_integration",
            integration_type=IntegrationType.PRODUCTIVITY
            # No API key
        )
        integration = MockIntegration(config, privacy_manager)
        
        result = await integration.authenticate()
        assert result is False
        assert integration.status == IntegrationStatus.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_test_connection(self, mock_integration):
        """Test connection testing"""
        # Not connected initially
        result = await mock_integration.test_connection()
        assert result is False
        
        # Connect and test again
        await mock_integration.authenticate()
        result = await mock_integration.test_connection()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_data_success(self, mock_integration):
        """Test successful data retrieval"""
        await mock_integration.authenticate()
        
        result = await mock_integration.get_data("test_endpoint", {"param": "value"})
        assert result.success is True
        assert result.data["endpoint"] == "test_endpoint"
        assert result.data["params"]["param"] == "value"
    
    @pytest.mark.asyncio
    async def test_get_data_not_connected(self, mock_integration):
        """Test data retrieval when not connected"""
        result = await mock_integration.get_data("test_endpoint")
        assert result.success is False
        assert result.error_message == "Not connected"
    
    @pytest.mark.asyncio
    async def test_send_data_success(self, mock_integration):
        """Test successful data sending"""
        await mock_integration.authenticate()
        
        test_data = {"key": "value"}
        result = await mock_integration.send_data("test_endpoint", test_data)
        assert result.success is True
        assert result.data["sent"] == test_data
    
    def test_rate_limiting(self, mock_integration):
        """Test rate limiting functionality"""
        # Should allow requests initially
        assert mock_integration._check_rate_limit() is True
        
        # Simulate many requests
        mock_integration.request_count = mock_integration.config.rate_limit + 1
        mock_integration.last_request_time = datetime.now()
        
        # Should be rate limited
        assert mock_integration._check_rate_limit() is False
    
    def test_auth_headers(self, mock_integration):
        """Test authentication header generation"""
        headers = mock_integration._get_auth_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_key"


class TestGoogleDriveIntegration:
    """Test Google Drive integration"""
    
    @pytest.fixture
    def privacy_manager(self):
        """Create a mock privacy manager"""
        manager = Mock(spec=PrivacySecurityManager)
        manager.check_permission = AsyncMock(return_value=True)
        return manager
    
    @pytest.fixture
    def google_drive_config(self):
        """Create Google Drive config"""
        return IntegrationConfig(
            name="google_drive",
            integration_type=IntegrationType.CLOUD_STORAGE,
            oauth_token="test_oauth_token"
        )
    
    @pytest.fixture
    def google_drive_integration(self, google_drive_config, privacy_manager):
        """Create Google Drive integration"""
        return GoogleDriveIntegration(google_drive_config, privacy_manager)
    
    @pytest.mark.asyncio
    async def test_authentication(self, google_drive_integration):
        """Test Google Drive authentication"""
        result = await google_drive_integration.authenticate()
        assert result is True
        assert google_drive_integration.status == IntegrationStatus.CONNECTED
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.request')
    async def test_list_files(self, mock_request, google_drive_integration):
        """Test listing files from Google Drive"""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "files": [
                {"id": "1", "name": "test.txt", "mimeType": "text/plain"},
                {"id": "2", "name": "doc.pdf", "mimeType": "application/pdf"}
            ]
        })
        mock_request.return_value.__aenter__.return_value = mock_response
        
        await google_drive_integration.authenticate()
        result = await google_drive_integration.list_files()
        
        assert result.success is True
        assert len(result.data["files"]) == 2
        assert result.data["files"][0]["name"] == "test.txt"


class TestGitHubIntegration:
    """Test GitHub integration"""
    
    @pytest.fixture
    def privacy_manager(self):
        """Create a mock privacy manager"""
        manager = Mock(spec=PrivacySecurityManager)
        manager.check_permission = AsyncMock(return_value=True)
        return manager
    
    @pytest.fixture
    def github_config(self):
        """Create GitHub config"""
        return IntegrationConfig(
            name="github",
            integration_type=IntegrationType.DEVELOPMENT,
            api_key="test_github_token"
        )
    
    @pytest.fixture
    def github_integration(self, github_config, privacy_manager):
        """Create GitHub integration"""
        return GitHubIntegration(github_config, privacy_manager)
    
    @pytest.mark.asyncio
    async def test_authentication(self, github_integration):
        """Test GitHub authentication"""
        result = await github_integration.authenticate()
        assert result is True
        assert github_integration.status == IntegrationStatus.CONNECTED
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.request')
    async def test_get_repositories(self, mock_request, github_integration):
        """Test getting repositories from GitHub"""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[
            {"id": 1, "name": "repo1", "full_name": "user/repo1"},
            {"id": 2, "name": "repo2", "full_name": "user/repo2"}
        ])
        mock_request.return_value.__aenter__.return_value = mock_response
        
        await github_integration.authenticate()
        result = await github_integration.get_repositories()
        
        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0]["name"] == "repo1"


class TestIntegrationHub:
    """Test IntegrationHub main class"""
    
    @pytest.fixture
    def privacy_manager(self):
        """Create a mock privacy manager"""
        manager = Mock(spec=PrivacySecurityManager)
        manager.check_permission = AsyncMock(return_value=True)
        return manager
    
    @pytest.fixture
    def integration_hub(self, privacy_manager):
        """Create an integration hub"""
        return IntegrationHub(privacy_manager)
    
    @pytest.mark.asyncio
    async def test_initialization(self, integration_hub):
        """Test hub initialization"""
        result = await integration_hub.initialize()
        assert result is True
        assert len(integration_hub.configs) > 0
    
    @pytest.mark.asyncio
    async def test_initialization_with_config_file(self, integration_hub):
        """Test hub initialization with config file"""
        # Create temporary config file
        config_data = {
            "test_integration": {
                "type": "productivity",
                "enabled": True,
                "api_key": "test_key"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            result = await integration_hub.initialize(config_path)
            assert result is True
            assert "test_integration" in integration_hub.configs
        finally:
            os.unlink(config_path)
    
    @pytest.mark.asyncio
    async def test_add_integration_success(self, integration_hub):
        """Test adding an integration successfully"""
        await integration_hub.initialize()
        
        config = IntegrationConfig(
            name="google_drive",
            integration_type=IntegrationType.CLOUD_STORAGE,
            oauth_token="test_token"
        )
        
        result = await integration_hub.add_integration("google_drive", config)
        assert result is True
        assert "google_drive" in integration_hub.integrations
    
    @pytest.mark.asyncio
    async def test_add_integration_unknown_type(self, integration_hub):
        """Test adding an unknown integration type"""
        await integration_hub.initialize()
        
        config = IntegrationConfig(
            name="unknown_integration",
            integration_type=IntegrationType.PRODUCTIVITY
        )
        
        result = await integration_hub.add_integration("unknown_integration", config)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_remove_integration(self, integration_hub):
        """Test removing an integration"""
        await integration_hub.initialize()
        
        config = IntegrationConfig(
            name="google_drive",
            integration_type=IntegrationType.CLOUD_STORAGE
        )
        
        await integration_hub.add_integration("google_drive", config)
        assert "google_drive" in integration_hub.integrations
        
        result = await integration_hub.remove_integration("google_drive")
        assert result is True
        assert "google_drive" not in integration_hub.integrations
    
    @pytest.mark.asyncio
    async def test_get_integration(self, integration_hub):
        """Test getting an integration"""
        await integration_hub.initialize()
        
        config = IntegrationConfig(
            name="google_drive",
            integration_type=IntegrationType.CLOUD_STORAGE
        )
        
        await integration_hub.add_integration("google_drive", config)
        
        integration = await integration_hub.get_integration("google_drive")
        assert integration is not None
        assert integration.config.name == "google_drive"
        
        # Test non-existent integration
        integration = await integration_hub.get_integration("non_existent")
        assert integration is None
    
    @pytest.mark.asyncio
    async def test_list_integrations(self, integration_hub):
        """Test listing all integrations"""
        await integration_hub.initialize()
        
        config = IntegrationConfig(
            name="google_drive",
            integration_type=IntegrationType.CLOUD_STORAGE,
            oauth_token="test_token"
        )
        
        await integration_hub.add_integration("google_drive", config)
        
        integrations = await integration_hub.list_integrations()
        assert "google_drive" in integrations
        assert integrations["google_drive"]["type"] == "cloud_storage"
        assert integrations["google_drive"]["enabled"] is True
    
    @pytest.mark.asyncio
    async def test_test_all_connections(self, integration_hub):
        """Test testing all connections"""
        await integration_hub.initialize()
        
        # Add a mock integration
        config = IntegrationConfig(
            name="google_drive",
            integration_type=IntegrationType.CLOUD_STORAGE,
            oauth_token="test_token"
        )
        
        await integration_hub.add_integration("google_drive", config)
        
        # Mock the integration's test_connection method
        integration = integration_hub.integrations["google_drive"]
        integration.test_connection = AsyncMock(return_value=True)
        integration.status = IntegrationStatus.CONNECTED
        
        results = await integration_hub.test_all_connections()
        assert "google_drive" in results
        assert results["google_drive"] is True
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.request')
    async def test_sync_files_from_cloud(self, mock_request, integration_hub):
        """Test syncing files from cloud storage"""
        await integration_hub.initialize()
        
        # Mock successful Google Drive response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "files": [
                {
                    "id": "1",
                    "name": "test.txt",
                    "mimeType": "text/plain",
                    "modifiedTime": "2023-01-01T00:00:00Z",
                    "size": "1024"
                }
            ]
        })
        mock_request.return_value.__aenter__.return_value = mock_response
        
        # Add Google Drive integration
        config = IntegrationConfig(
            name="google_drive",
            integration_type=IntegrationType.CLOUD_STORAGE,
            oauth_token="test_token"
        )
        
        await integration_hub.add_integration("google_drive", config)
        integration = integration_hub.integrations["google_drive"]
        integration.status = IntegrationStatus.CONNECTED
        
        files = await integration_hub.sync_files_from_cloud("test_user")
        
        assert len(files) == 1
        assert files[0]["service"] == "google_drive"
        assert files[0]["name"] == "test.txt"
        assert files[0]["id"] == "1"
    
    @pytest.mark.asyncio
    async def test_sync_files_permission_denied(self, integration_hub):
        """Test syncing files when permission is denied"""
        # Mock privacy manager to deny permission
        integration_hub.privacy_manager.check_permission = AsyncMock(return_value=False)
        
        await integration_hub.initialize()
        
        files = await integration_hub.sync_files_from_cloud("test_user")
        assert len(files) == 0
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.request')
    async def test_get_development_context(self, mock_request, integration_hub):
        """Test getting development context"""
        await integration_hub.initialize()
        
        # Mock successful GitHub response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[
            {"id": 1, "name": "repo1", "full_name": "user/repo1"}
        ])
        mock_request.return_value.__aenter__.return_value = mock_response
        
        # Add GitHub integration
        config = IntegrationConfig(
            name="github",
            integration_type=IntegrationType.DEVELOPMENT,
            api_key="test_token"
        )
        
        await integration_hub.add_integration("github", config)
        integration = integration_hub.integrations["github"]
        integration.status = IntegrationStatus.CONNECTED
        
        context = await integration_hub.get_development_context("test_user")
        
        assert "repositories" in context
        assert len(context["repositories"]) == 1
        assert context["repositories"][0]["name"] == "repo1"
    
    @pytest.mark.asyncio
    async def test_get_development_context_permission_denied(self, integration_hub):
        """Test getting development context when permission is denied"""
        # Mock privacy manager to deny permission
        integration_hub.privacy_manager.check_permission = AsyncMock(return_value=False)
        
        await integration_hub.initialize()
        
        context = await integration_hub.get_development_context("test_user")
        assert len(context) == 0
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.request')
    async def test_send_notification(self, mock_request, integration_hub):
        """Test sending notifications"""
        await integration_hub.initialize()
        
        # Mock successful Slack response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ok": True})
        mock_request.return_value.__aenter__.return_value = mock_response
        
        # Add Slack integration
        config = IntegrationConfig(
            name="slack",
            integration_type=IntegrationType.COMMUNICATION,
            oauth_token="test_token"
        )
        
        await integration_hub.add_integration("slack", config)
        integration = integration_hub.integrations["slack"]
        integration.status = IntegrationStatus.CONNECTED
        
        result = await integration_hub.send_notification("test_user", "Test message")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_send_notification_permission_denied(self, integration_hub):
        """Test sending notification when permission is denied"""
        # Mock privacy manager to deny permission
        integration_hub.privacy_manager.check_permission = AsyncMock(return_value=False)
        
        await integration_hub.initialize()
        
        result = await integration_hub.send_notification("test_user", "Test message")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_save_configs(self, integration_hub):
        """Test saving configurations to file"""
        await integration_hub.initialize()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            result = await integration_hub.save_configs(config_path)
            assert result is True
            
            # Verify file was created and contains data
            with open(config_path, 'r') as f:
                saved_data = json.load(f)
            
            assert len(saved_data) > 0
            assert "google_drive" in saved_data
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])