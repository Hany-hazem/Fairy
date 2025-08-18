# Development Guide - Getting Started

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ installed
- Git configured with your credentials
- Code editor (VS Code recommended)
- Basic understanding of async/await in Python

### Initial Setup
```bash
# Clone the repository
git clone <repository-url>
cd personal-assistant-enhancement

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest tests/test_multi_modal_simple.py -v

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Project Structure Navigation
```
📁 personal-assistant-enhancement/
├── 📁 app/                          # Main application code
│   ├── 📄 personal_assistant_core.py    # Central orchestrator
│   ├── 📄 interaction_mode_manager.py   # Multi-modal coordination
│   ├── 📄 voice_processor.py            # Voice I/O
│   ├── 📄 screen_overlay.py             # Visual feedback
│   ├── 📄 text_completion.py            # Text suggestions
│   ├── 📄 accessibility_manager.py      # Accessibility features
│   └── 📄 [other modules...]
├── 📁 tests/                        # Test suite
│   ├── 📄 test_multi_modal_simple.py    # Component tests
│   └── 📄 [other test files...]
├── 📁 obsidian-notes/              # Documentation
│   ├── 📁 team-onboarding/             # Team guides
│   ├── 📁 component-maps/              # Technical docs
│   └── 📄 [other docs...]
├── 📁 .kiro/specs/                 # Project specifications
└── 📄 requirements.txt             # Python dependencies
```

## 🛠️ Development Workflow

### 1. Understanding the Request Flow
Every user interaction follows this pattern:

```python
# 1. User makes request (voice, text, or visual)
request = AssistantRequest(
    user_id="user123",
    request_type=RequestType.VOICE_COMMAND,
    content="open file document.pdf",
    metadata={"intent": "file_operation"}
)

# 2. PersonalAssistantCore processes request
response = await assistant_core.process_request(request)

# 3. Response sent back to user
# Response includes content, success status, and suggestions
```

### 2. Adding a New Feature - Step by Step

#### Example: Adding a "Weather" capability

**Step 1: Create the capability module**
```python
# app/weather_manager.py
import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WeatherManager:
    """Manages weather information and forecasts"""
    
    def __init__(self):
        self.api_key = None  # Would be configured
        self.cache = {}
        
    async def get_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather for location"""
        try:
            # Check cache first
            if location in self.cache:
                return self.cache[location]
            
            # Fetch weather data (mock for example)
            weather_data = {
                "location": location,
                "temperature": "22°C",
                "condition": "Sunny",
                "humidity": "45%"
            }
            
            # Cache result
            self.cache[location] = weather_data
            
            logger.info(f"Weather retrieved for {location}")
            return weather_data
            
        except Exception as e:
            logger.error(f"Error getting weather for {location}: {e}")
            raise
    
    async def get_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        """Get weather forecast"""
        # Implementation here
        pass
```

**Step 2: Add request type**
```python
# In app/personal_assistant_core.py
class RequestType(Enum):
    # ... existing types ...
    WEATHER_REQUEST = "weather_request"
```

**Step 3: Add request handler**
```python
# In PersonalAssistantCore class
async def _handle_weather_request(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
    """Handle weather requests"""
    try:
        weather_manager = self._capability_modules.get('weather')
        if not weather_manager:
            return AssistantResponse(
                content="Weather service is not available.",
                success=False,
                metadata={},
                suggestions=["Try again later."]
            )
        
        location = request.metadata.get('location', 'current location')
        action = request.metadata.get('action', 'current')
        
        if action == 'current':
            weather_data = await weather_manager.get_weather(location)
            return AssistantResponse(
                content=f"Weather in {location}: {weather_data['temperature']}, {weather_data['condition']}",
                success=True,
                metadata={"weather_data": weather_data},
                suggestions=["Get forecast", "Check other locations"]
            )
        
        elif action == 'forecast':
            days = request.metadata.get('days', 5)
            forecast_data = await weather_manager.get_forecast(location, days)
            return AssistantResponse(
                content=f"{days}-day forecast for {location}",
                success=True,
                metadata={"forecast_data": forecast_data},
                suggestions=[]
            )
        
    except Exception as e:
        logger.error(f"Error handling weather request: {e}")
        return AssistantResponse(
            content=f"Weather request failed: {str(e)}",
            success=False,
            metadata={"error": str(e)},
            suggestions=["Try again later."]
        )
```

**Step 4: Initialize the module**
```python
# In PersonalAssistantCore._initialize_capability_modules()
self._capability_modules['weather'] = WeatherManager()
logger.info("WeatherManager initialized")
```

**Step 5: Add routing**
```python
# In PersonalAssistantCore._route_request()
elif request.request_type == RequestType.WEATHER_REQUEST:
    return await self._handle_weather_request(request, context)
```

**Step 6: Write tests**
```python
# tests/test_weather_manager.py
import pytest
from app.weather_manager import WeatherManager
from app.personal_assistant_core import PersonalAssistantCore, RequestType, AssistantRequest

class TestWeatherManager:
    @pytest.fixture
    def weather_manager(self):
        return WeatherManager()
    
    @pytest.mark.asyncio
    async def test_get_weather(self, weather_manager):
        weather_data = await weather_manager.get_weather("London")
        
        assert weather_data["location"] == "London"
        assert "temperature" in weather_data
        assert "condition" in weather_data
    
    @pytest.mark.asyncio
    async def test_weather_request_integration(self):
        assistant = PersonalAssistantCore(":memory:")
        
        request = AssistantRequest(
            user_id="test_user",
            request_type=RequestType.WEATHER_REQUEST,
            content="What's the weather like?",
            metadata={"location": "London", "action": "current"}
        )
        
        response = await assistant.process_request(request)
        
        assert response.success
        assert "London" in response.content
        assert "weather_data" in response.metadata
```

### 3. Working with Multi-Modal Features

#### Adding Voice Command Support
```python
# In app/voice_processor.py - add to command patterns
self.command_patterns["weather"] = [
    r"(?:what's|what is) (?:the )?weather (?:like )?(?:in )?(.+)?",
    r"(?:weather|forecast) (?:for )?(.+)",
    r"(?:how's|how is) (?:the )?weather"
]
```

#### Adding Visual Feedback
```python
# In your request handler
if weather_data["condition"] == "Sunny":
    await self.show_visual_feedback(
        f"☀️ Sunny weather in {location}",
        feedback_type="success"
    )
elif weather_data["condition"] == "Rainy":
    await self.show_visual_feedback(
        f"🌧️ Rainy weather in {location}",
        feedback_type="info"
    )
```

#### Adding Text Completion
```python
# In app/text_completion.py - add to context completions
def _get_weather_completions(self, current_input: str) -> List[Completion]:
    weather_phrases = [
        "What's the weather like in",
        "Weather forecast for",
        "How's the weather today",
        "Will it rain tomorrow"
    ]
    
    completions = []
    for phrase in weather_phrases:
        if phrase.lower().startswith(current_input.lower()):
            completions.append(Completion(
                text=phrase,
                completion_type=CompletionType.PHRASE,
                confidence=0.7,
                source="weather_suggestions"
            ))
    
    return completions
```

## 🧪 Testing Best Practices

### 1. Test Structure
```python
class TestNewFeature:
    @pytest.fixture
    def setup_data(self):
        """Setup test data"""
        return {"test": "data"}
    
    def test_basic_functionality(self, setup_data):
        """Test basic feature works"""
        # Arrange
        input_data = setup_data
        
        # Act
        result = process_data(input_data)
        
        # Assert
        assert result is not None
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async features"""
        result = await async_function()
        assert result is not None
    
    def test_error_handling(self):
        """Test error scenarios"""
        with pytest.raises(ValueError):
            invalid_function(None)
```

### 2. Mocking External Dependencies
```python
from unittest.mock import Mock, patch, AsyncMock

class TestWithMocks:
    @patch('app.weather_manager.requests.get')
    async def test_weather_api_call(self, mock_get):
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {"temp": "20°C"}
        mock_get.return_value = mock_response
        
        # Test
        weather_manager = WeatherManager()
        result = await weather_manager.get_weather("London")
        
        # Verify
        assert result["temperature"] == "20°C"
        mock_get.assert_called_once()
```

### 3. Integration Testing
```python
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_request_flow(self):
        """Test complete request processing"""
        # Setup
        assistant = PersonalAssistantCore(":memory:")
        
        # Create request
        request = AssistantRequest(
            user_id="test_user",
            request_type=RequestType.WEATHER_REQUEST,
            content="Weather in London",
            metadata={"location": "London"}
        )
        
        # Process
        response = await assistant.process_request(request)
        
        # Verify
        assert response.success
        assert "London" in response.content
        
        # Check context was updated
        context = await assistant.get_context("test_user")
        assert len(context.recent_interactions) > 0
```

## 🔧 Debugging Tips

### 1. Logging
```python
import logging

# Setup logging in your module
logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Something unexpected happened")
logger.error("An error occurred")
logger.critical("Critical system error")

# Log with context
logger.info(f"Processing request for user {user_id}: {request.content}")
```

### 2. Using the Debugger
```python
# Add breakpoints in your code
import pdb; pdb.set_trace()

# Or use the more modern debugger
import ipdb; ipdb.set_trace()

# In VS Code, just click in the gutter to set breakpoints
```

### 3. Testing Individual Components
```python
# Test a single component in isolation
async def test_component():
    voice_processor = VoiceProcessor()
    command = voice_processor._parse_command("open file test.txt")
    print(f"Intent: {command.intent}")
    print(f"Entities: {command.entities}")

# Run with: python -c "import asyncio; asyncio.run(test_component())"
```

## 📊 Performance Monitoring

### 1. Timing Operations
```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timing_decorator
async def slow_operation():
    # Your code here
    pass
```

### 2. Memory Usage
```python
import psutil
import os

def log_memory_usage(operation_name):
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage after {operation_name}: {memory_mb:.2f} MB")
```

### 3. Profiling
```python
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

## 🔄 Git Workflow

### 1. Branch Naming
```bash
# Feature branches
git checkout -b feature/weather-integration
git checkout -b feature/voice-commands-enhancement

# Bug fixes
git checkout -b bugfix/text-completion-crash
git checkout -b bugfix/accessibility-focus-issue

# Documentation
git checkout -b docs/api-documentation-update
```

### 2. Commit Messages
```bash
# Good commit messages
git commit -m "Add weather integration with API support

- Implement WeatherManager class with caching
- Add weather request type and handler
- Include voice command patterns for weather queries
- Add comprehensive tests with mocking
- Update documentation with weather examples"

# Bad commit messages (avoid these)
git commit -m "fix stuff"
git commit -m "update code"
git commit -m "changes"
```

### 3. Pull Request Process
1. Create feature branch from main
2. Implement feature with tests
3. Update documentation
4. Run full test suite
5. Create pull request with description
6. Address review feedback
7. Merge after approval

## 📚 Common Patterns & Examples

### 1. Adding Configuration
```python
# In app/config.py
class Settings(BaseSettings):
    # Existing settings...
    
    # New weather settings
    WEATHER_API_KEY: str = ""
    WEATHER_CACHE_TTL: int = 300  # 5 minutes
    WEATHER_DEFAULT_LOCATION: str = "London"
    
    class Config:
        env_file = ".env"
```

### 2. Error Handling
```python
async def safe_operation(self, data):
    try:
        result = await self.risky_operation(data)
        return result
    except SpecificException as e:
        logger.warning(f"Expected error in operation: {e}")
        return self.fallback_result()
    except Exception as e:
        logger.error(f"Unexpected error in operation: {e}")
        raise  # Re-raise unexpected errors
```

### 3. Async Context Managers
```python
class DatabaseConnection:
    async def __aenter__(self):
        self.connection = await self.connect()
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.connection.close()

# Usage
async with DatabaseConnection() as conn:
    result = await conn.execute("SELECT * FROM users")
```

## 🎯 Next Steps for New Developers

### Week 1: Getting Familiar
- [ ] Set up development environment
- [ ] Run all tests and understand the test structure
- [ ] Read through the main components (PersonalAssistantCore, InteractionModeManager)
- [ ] Try making a simple change and running tests

### Week 2: First Contribution
- [ ] Pick a small feature or bug fix
- [ ] Implement following the patterns shown above
- [ ] Write comprehensive tests
- [ ] Create pull request and go through review process

### Week 3: Understanding Architecture
- [ ] Study the multi-modal interaction system
- [ ] Understand the privacy and security patterns
- [ ] Learn the learning and adaptation mechanisms
- [ ] Contribute to documentation improvements

### Ongoing: Becoming a Core Contributor
- [ ] Take on larger features
- [ ] Help with code reviews
- [ ] Mentor new team members
- [ ] Contribute to architectural decisions

Remember: Don't hesitate to ask questions! The codebase is complex, and it's normal to need clarification. Use the team communication channels and refer back to this documentation as needed.

## Related Documentation & Resources

### Architecture & Patterns
- [[technical-concepts]] - Core patterns and implementation concepts used in examples
- [[architecture-deep-dive]] - Detailed architectural analysis and design decisions
- [[../architecture-overview]] - System architecture overview and component relationships

### Project Context
- [[project-overview-complete]] - Complete project overview and system understanding
- [[../project-overview]] - Current project status and achievements
- [[../task-progress]] - Implementation progress and upcoming milestones
- [[../requirements-map]] - Requirements tracking and feature completion

### Component Implementation
- [[../component-maps/personal-assistant-core]] - Core system implementation details
- [[../component-maps/multi-modal-interaction]] - Multi-modal system example
- [[../component-maps/README]] - All component documentation and examples

### Educational Resources
- [[../educational-assessment]] - Educational value for software engineering learning
- [[README]] - Complete onboarding guide and learning paths
- [[../daily-logs/2025-08-17]] - Real development process and decision-making

### Testing & Quality
- `tests/test_multi_modal_simple.py` - Comprehensive testing examples
- [[technical-concepts#Testing Patterns]] - Testing pattern implementations
- [[architecture-deep-dive#Performance Architecture]] - Performance considerations

### Navigation
- [[../README]] - Main knowledge base hub
- [[README#Phase 2]] - Technical deep dive phase
- [[README#Phase 3]] - Hands-on development phase