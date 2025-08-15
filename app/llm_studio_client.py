# app/llm_studio_client.py
import json
import time
import logging
from typing import Dict, Any, Optional, List
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class LMStudioConfig(BaseModel):
    """Configuration for LM Studio connection and model parameters"""
    endpoint_url: str = "http://localhost:1234"
    model_name: str = "gpt-oss-20b"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    retry_attempts: int = 3

class LMStudioConnectionError(Exception):
    """Raised when LM Studio connection fails"""
    pass

class LMStudioClient:
    def __init__(self, config: Optional[LMStudioConfig] = None):
        self.config = config or LMStudioConfig()
        self.base_url = self.config.endpoint_url.rstrip("/")
        self._client = httpx.Client(timeout=self.config.timeout)
        self._last_health_check = 0
        self._health_check_interval = 60  # seconds
        self._is_healthy = None

    def __del__(self):
        """Clean up HTTP client"""
        if hasattr(self, '_client'):
            self._client.close()

    def health_check(self) -> bool:
        """Check if LM Studio is available and responding"""
        current_time = time.time()
        
        # Use cached result if recent
        if (self._is_healthy is not None and 
            current_time - self._last_health_check < self._health_check_interval):
            return self._is_healthy
        
        try:
            # Try to get models endpoint first
            resp = self._client.get(f"{self.base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                self._is_healthy = True
                logger.info("LM Studio health check passed")
            else:
                self._is_healthy = False
                logger.warning(f"LM Studio health check failed: {resp.status_code}")
        except Exception as e:
            self._is_healthy = False
            logger.error(f"LM Studio health check error: {e}")
        
        self._last_health_check = current_time
        return self._is_healthy

    def _make_request_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make request with exponential backoff retry logic"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                resp = self._client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.config.timeout
                )
                resp.raise_for_status()
                return resp.json()
                
            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(f"LM Studio request timeout (attempt {attempt + 1})")
                
            except httpx.HTTPStatusError as e:
                last_exception = e
                logger.warning(f"LM Studio HTTP error {e.response.status_code} (attempt {attempt + 1})")
                
            except Exception as e:
                last_exception = e
                logger.warning(f"LM Studio request error: {e} (attempt {attempt + 1})")
            
            # Exponential backoff: 1s, 2s, 4s
            if attempt < self.config.retry_attempts - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        # All retries failed
        raise LMStudioConnectionError(f"Failed to connect to LM Studio after {self.config.retry_attempts} attempts: {last_exception}")

    def chat(self, prompt: str, max_new_tokens: Optional[int] = None, 
             temperature: Optional[float] = None, messages: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Enhanced chat method with conversation support and parameter overrides
        
        Args:
            prompt: User prompt (used if messages not provided)
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            messages: Full conversation history in OpenAI format
        """
        # Use provided messages or create from prompt
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": 0.9,
        }
        
        logger.debug(f"Sending request to LM Studio: {len(messages)} messages")
        
        try:
            data = self._make_request_with_retry(payload)
            response = data["choices"][0]["message"]["content"].strip()
            logger.debug(f"Received response from LM Studio: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"LM Studio chat error: {e}")
            raise LMStudioConnectionError(f"Chat request failed: {e}")

    def validate_connection(self) -> Dict[str, Any]:
        """Validate connection and return status information"""
        try:
            # Test health check
            is_healthy = self.health_check()
            
            # Test simple chat
            test_response = self.chat("Hello", max_new_tokens=10)
            
            return {
                "status": "connected",
                "healthy": is_healthy,
                "endpoint": self.base_url,
                "model": self.config.model_name,
                "test_response_length": len(test_response),
                "config": {
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "timeout": self.config.timeout,
                    "retry_attempts": self.config.retry_attempts
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "healthy": False,
                "endpoint": self.base_url,
                "error": str(e),
                "config": {
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "timeout": self.config.timeout,
                    "retry_attempts": self.config.retry_attempts
                }
            }

# Enhanced singleton with configuration
def create_studio_client():
    """Create LM Studio client with settings from config"""
    from .config import settings
    
    config = LMStudioConfig(
        endpoint_url=settings.LLMS_STUDIO_URL,
        model_name=settings.LM_STUDIO_MODEL,
        temperature=settings.LM_STUDIO_TEMPERATURE,
        max_tokens=settings.LM_STUDIO_MAX_TOKENS,
        timeout=settings.LM_STUDIO_TIMEOUT,
        retry_attempts=settings.LM_STUDIO_RETRY_ATTEMPTS
    )
    return LMStudioClient(config)

# Lazy initialization to avoid circular imports
_studio_client = None

def get_studio_client() -> LMStudioClient:
    """Get the singleton LM Studio client"""
    global _studio_client
    if _studio_client is None:
        _studio_client = create_studio_client()
    return _studio_client

# For backward compatibility - lazy initialization
studio_client = None
