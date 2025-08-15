# Quick Reference - Self-Improving AI Assistant

## 🚀 Getting Started (30 seconds)

```bash
# 1. Setup
git clone <repo> && cd self-improving-ai-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Start (choose one)
python start_assistant.py          # Interactive menu
uvicorn app.main:app --reload      # Web server
python examples/basic_usage.py     # See examples
```

## 🎯 Quick Commands

### Web API (Start server first: `uvicorn app.main:app --reload`)
```bash
# Chat with AI
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Help me optimize my Python code", "user_id": "dev"}'

# Trigger self-improvement
curl -X POST "http://localhost:8000/improve" \
  -H "Content-Type: application/json" \
  -d '{"trigger": "manual"}'

# Check status
curl "http://localhost:8000/health"
curl "http://localhost:8000/improve/status"
```

### Python API
```python
import asyncio
from app.ai_assistant_service import AIAssistantService

async def quick_chat():
    service = AIAssistantService()
    result = await service.process_query(
        query="Analyze my code for improvements",
        user_id="developer"
    )
    print(result['response'])

asyncio.run(quick_chat())
```

### Code Analysis
```python
from app.code_analyzer import CodeAnalyzer

analyzer = CodeAnalyzer(".")
reports = analyzer.analyze_project()
summary = analyzer.get_project_summary(reports)
print(f"Quality: {summary['average_quality_score']:.1f}/100")
```

### Self-Improvement
```python
import asyncio
from app.self_improvement_engine import SelfImprovementEngine

async def improve():
    engine = SelfImprovementEngine(".")
    cycle_id = await engine.trigger_improvement_cycle("manual")
    # Monitor with: engine.get_current_status()

asyncio.run(improve())
```

## 📋 Key Features

| Feature | Command | Description |
|---------|---------|-------------|
| **Web Interface** | `uvicorn app.main:app --reload` | Full web UI at http://localhost:8000 |
| **Interactive Chat** | `python start_assistant.py` → Option 2 | Command-line conversation |
| **Code Analysis** | `python start_assistant.py` → Option 3 | Analyze code quality |
| **Self-Improvement** | `python start_assistant.py` → Option 4 | Auto-improve code safely |
| **Run Tests** | `python run_integration_tests.py` | Comprehensive test suite |

## 🔧 Configuration

### Environment Variables (.env file)
```bash
REDIS_URL=redis://localhost:6379
LLM_STUDIO_URL=http://localhost:1234
SAFETY_LEVEL=conservative
AUTO_APPLY_THRESHOLD=8.0
LOG_LEVEL=INFO
```

### Safety Levels
- `conservative`: Only low-risk improvements (recommended)
- `moderate`: Low and medium-risk improvements  
- `aggressive`: All improvements (use with caution)

## 🧪 Testing

```bash
# All tests
python run_integration_tests.py

# Functionality check
python test_project_functionality.py

# Specific tests
python -m pytest tests/test_ai_assistant_service.py -v
```

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Redis connection failed | Install Redis or use in-memory fallback (automatic) |
| LM Studio not available | Start LM Studio on localhost:1234 or use mocked responses |
| Import errors | Activate virtual environment: `source venv/bin/activate` |
| Permission denied | Ensure write permissions for `.kiro/` directory |

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome page |
| `/health` | GET | Health check |
| `/chat` | POST | Chat with AI assistant |
| `/improve` | POST | Trigger self-improvement |
| `/improve/status` | GET | Get improvement status |
| `/analyze` | POST | Analyze code |
| `/docs` | GET | API documentation |

## 🔒 Safety Features

✅ **Risk Assessment** - All improvements are risk-rated  
✅ **Code Validation** - Syntax and import checking  
✅ **Test Validation** - Comprehensive testing before changes  
✅ **Git Integration** - Automatic rollback points  
✅ **Emergency Stop** - Immediate halt capability  

## 📚 File Structure

```
self-improving-ai-assistant/
├── app/                    # Core application
│   ├── main.py            # FastAPI app
│   ├── ai_assistant_service.py
│   ├── self_improvement_engine.py
│   └── ...
├── agents/                # AI agents
├── tests/                 # Test suite
├── examples/              # Usage examples
├── start_assistant.py     # Easy startup
├── USAGE_GUIDE.md        # Detailed guide
└── requirements.txt      # Dependencies
```

## 🎯 Common Use Cases

### 1. Code Review
```python
from app.code_analyzer import CodeAnalyzer
analyzer = CodeAnalyzer("./my_project")
reports = analyzer.analyze_project()
# Review reports for issues and recommendations
```

### 2. Automated Improvement
```python
import asyncio
from app.self_improvement_engine import SelfImprovementEngine

async def improve_project():
    engine = SelfImprovementEngine("./my_project")
    await engine.trigger_improvement_cycle("manual")

asyncio.run(improve_project())
```

### 3. AI Assistant Chat
```python
import asyncio
from app.ai_assistant_service import AIAssistantService

async def get_help():
    service = AIAssistantService()
    result = await service.process_query(
        query="How can I optimize this Python function?",
        user_id="developer"
    )
    print(result['response'])

asyncio.run(get_help())
```

## 🔗 Quick Links

- **Start Interactive**: `python start_assistant.py`
- **Web Interface**: http://localhost:8000 (after starting server)
- **API Docs**: http://localhost:8000/docs
- **Examples**: `python examples/basic_usage.py`
- **Full Guide**: [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **Test Results**: [TEST_RESULTS_SUMMARY.md](TEST_RESULTS_SUMMARY.md)

---

**Need help?** Run `python start_assistant.py` and choose option 6 for help, or check the full [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed instructions.