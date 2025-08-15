# tests/test_lm_studio_client.py
import pytest
import httpx
from unittest.mock import Mock, patch, MagicMock
import time

from app.llm_studio_client import (
    LMStudioClient, 
    LMStudioConfig, 
    LMStudioConnectionError,
    get_studio_client
)

class TestLMStudioConfig:
    def test_default_config(self):
        """Test default configuration values"""
        config = LMStudioConfig()
        assert config.endpoint_url == "http://localhost:1234"
        assert config.model_name == "gpt-oss-20b"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.timeout == 30
        assert config.retry_attempts == 3

    def test_custom_config(self):
        """Test custom configuration values"""
        config = LMStudioConfig(
            endpoint_url="http://localhost:8000",
            model_name="custom-model",
            temperature=0.5,
            max_tokens=1024
        )
        assert config.endpoint_url == "http://localhost:8000"
        assert config.model_name == "custom-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024

class TestLMStudioClient:
    @pytest.fixture
    def client(self):
        """Create test client with default config"""
        config = LMStudioConfig()
        return LMStudioClient(config)

    @pytest.fixture
    def mock_httpx_client(self):
        """Mock httpx client"""
        with patch('app.llm_studio_client.httpx.Client') as mock_client:
            yield mock_client.return_value

    def test_client_initialization(self, client):
        """Test client initialization"""
        assert client.base_url == "http://localhost:1234"
        assert client.config.model_name == "gpt-oss-20b"
        assert client._health_check_interval == 60

    def test_health_check_success(self, client, mock_httpx_client):
        """Test successful health check"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_httpx_client.get.return_value = mock_response
        
        result = client.health_check()
        
        assert result is True
        assert client._is_healthy is True
        mock_httpx_client.get.assert_called_once_with(
            "http://localhost:1234/v1/models", 
            timeout=5
        )

    def test_health_check_failure(self, client, mock_httpx_client):
        """Test failed health check"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_httpx_client.get.return_value = mock_response
        
        result = client.health_check()
        
        assert result is False
        assert client._is_healthy is False

    def test_health_check_exception(self, client, mock_httpx_client):
        """Test health check with connection exception"""
        mock_httpx_client.get.side_effect = httpx.ConnectError("Connection failed")
        
        result = client.health_check()
        
        assert result is False
        assert client._is_healthy is False

    def test_health_check_caching(self, client, mock_httpx_client):
        """Test health check result caching"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_httpx_client.get.return_value = mock_response
        
        # First call
        result1 = client.health_check()
        # Second call within cache interval
        result2 = client.health_check()
        
        assert result1 is True
        assert result2 is True
        # Should only call once due to caching
        assert mock_httpx_client.get.call_count == 1

    def test_chat_success(self, client, mock_httpx_client):
        """Test successful chat request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help you?"}}]
        }
        mock_httpx_client.post.return_value = mock_response
        
        result = client.chat("Hello")
        
        assert result == "Hello! How can I help you?"
        mock_httpx_client.post.assert_called_once()
        
        # Check the payload
        call_args = mock_httpx_client.post.call_args
        payload = call_args[1]['json']
        assert payload['model'] == 'gpt-oss-20b'
        assert payload['messages'] == [{"role": "user", "content": "Hello"}]
        assert payload['max_tokens'] == 2048
        assert payload['temperature'] == 0.7

    def test_chat_with_messages(self, client, mock_httpx_client):
        """Test chat with conversation history"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "I understand."}}]
        }
        mock_httpx_client.post.return_value = mock_response
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = client.chat("", messages=messages)
        
        assert result == "I understand."
        
        # Check the payload contains the full conversation
        call_args = mock_httpx_client.post.call_args
        payload = call_args[1]['json']
        assert payload['messages'] == messages

    def test_chat_with_overrides(self, client, mock_httpx_client):
        """Test chat with parameter overrides"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_httpx_client.post.return_value = mock_response
        
        result = client.chat("Hello", max_new_tokens=512, temperature=0.5)
        
        # Check overridden parameters
        call_args = mock_httpx_client.post.call_args
        payload = call_args[1]['json']
        assert payload['max_tokens'] == 512
        assert payload['temperature'] == 0.5

    def test_chat_retry_on_timeout(self, client, mock_httpx_client):
        """Test retry logic on timeout"""
        # First two calls timeout, third succeeds
        mock_httpx_client.post.side_effect = [
            httpx.TimeoutException("Timeout"),
            httpx.TimeoutException("Timeout"),
            Mock(status_code=200, json=lambda: {"choices": [{"message": {"content": "Success"}}]})
        ]
        
        with patch('time.sleep'):  # Speed up test
            result = client.chat("Hello")
        
        assert result == "Success"
        assert mock_httpx_client.post.call_count == 3

    def test_chat_retry_exhausted(self, client, mock_httpx_client):
        """Test retry exhaustion raises exception"""
        mock_httpx_client.post.side_effect = httpx.TimeoutException("Timeout")
        
        with patch('time.sleep'):  # Speed up test
            with pytest.raises(LMStudioConnectionError):
                client.chat("Hello")
        
        assert mock_httpx_client.post.call_count == 3

    def test_validate_connection_success(self, client, mock_httpx_client):
        """Test successful connection validation"""
        # Mock health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        
        # Mock chat response
        mock_chat_response = Mock()
        mock_chat_response.status_code = 200
        mock_chat_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        
        mock_httpx_client.get.return_value = mock_health_response
        mock_httpx_client.post.return_value = mock_chat_response
        
        result = client.validate_connection()
        
        assert result["status"] == "connected"
        assert result["healthy"] is True
        assert result["endpoint"] == "http://localhost:1234"
        assert result["model"] == "gpt-oss-20b"
        assert result["test_response_length"] == 13  # "Test response"
        assert "config" in result

    def test_validate_connection_failure(self, client, mock_httpx_client):
        """Test connection validation failure"""
        mock_httpx_client.get.side_effect = httpx.ConnectError("Connection failed")
        
        result = client.validate_connection()
        
        assert result["status"] == "error"
        assert result["healthy"] is False
        assert "error" in result

def test_get_studio_client_singleton():
    """Test singleton pattern for studio client"""
    # Reset singleton for test
    import app.llm_studio_client
    app.llm_studio_client._studio_client = None
    
    with patch('app.llm_studio_client.create_studio_client') as mock_create:
        mock_client = Mock()
        mock_create.return_value = mock_client
        
        client1 = get_studio_client()
        client2 = get_studio_client()
        
        assert client1 is client2  # Same instance
        assert mock_create.call_count == 1  # Only created once