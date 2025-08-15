# Self-Improving AI Assistant - Usage Guide

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Git** (for version control features)
- **Redis** (optional - will use in-memory fallback)
- **LM Studio** (optional - for real LLM responses)

### Installation

1. **Clone and setup the project:**
```bash
git clone <repository-url>
cd self-improving-ai-assistant
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Initialize the project:**
```bash
# Initialize git repository (if not already done)
git init
git add .
git commit -m "Initial commit"

# Create necessary directories
mkdir -p .kiro/audit .kiro/backups test_results
```

## 🎯 Usage Options

### Option 1: FastAPI Web Application (Recommended)

**Start the web server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Access the application:**
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

**Example API Usage:**
```bash
# Start a conversation
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, how can you help me improve my code?", "user_id": "user123"}'

# Continue conversation
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze my Python code for improvements", "session_id": "session_id_from_previous_response", "user_id": "user123"}'

# Trigger self-improvement
curl -X POST "http://localhost:8000/improve" \
  -H "Content-Type: application/json" \
  -d '{"trigger": "manual"}'

# Check improvement status
curl "http://localhost:8000/improve/status"
```

### Option 2: Python API (Programmatic)

**Conversation Management:**
```python
import asyncio
from app.ai_assistant_service import AIAssistantService

async def main():
    # Initialize the service
    service = AIAssistantService()
    
    # Start a conversation
    result = await service.process_query(
        query="Hello, I need help with Python optimization",
        user_id="developer123"
    )
    
    print(f"Response: {result['response']}")
    session_id = result['session_id']
    
    # Continue the conversation
    result2 = await service.process_query(
        query="Can you analyze this code for performance issues?",
        session_id=session_id,
        user_id="developer123"
    )
    
    print(f"Follow-up: {result2['response']}")
    
    # Get conversation history
    history = await service.get_conversation_history(session_id)
    print(f"Messages: {len(history['messages'])}")

# Run the example
asyncio.run(main())
```

**Self-Improvement Engine:**
```python
import asyncio
from app.self_improvement_engine import SelfImprovementEngine

async def main():
    # Initialize the engine
    engine = SelfImprovementEngine(
        project_root=".",
        config={
            "safety_level": "conservative",
            "auto_apply_threshold": 8.0
        }
    )
    
    # Trigger an improvement cycle
    cycle_id = await engine.trigger_improvement_cycle("manual")
    print(f"Started improvement cycle: {cycle_id}")
    
    # Monitor status
    while True:
        status = engine.get_current_status()
        if not status["current_cycle"]:
            break
        
        print(f"Status: {status['current_cycle']['status']}")
        await asyncio.sleep(2)
    
    # Get results
    history = engine.get_cycle_history(limit=1)
    if history:
        cycle = history[0]
        print(f"Cycle completed: {cycle['status']}")
        print(f"Applied improvements: {len(cycle['applied_improvements'])}")

# Run the example
asyncio.run(main())
```

**Code Analysis:**
```python
from app.code_analyzer import CodeAnalyzer

# Initialize analyzer
analyzer = CodeAnalyzer(".")

# Analyze a single file
report = analyzer.analyze_file("app/main.py")
print(f"Quality Score: {report.quality_score}")
print(f"Issues Found: {len(report.issues)}")

# Analyze entire project
reports = analyzer.analyze_project()
summary = analyzer.get_project_summary(reports)
print(f"Total Files: {summary['total_files_analyzed']}")
print(f"Average Quality: {summary['average_quality_score']}")
```

### Option 3: Docker Deployment

**Build and run with Docker:**
```bash
# Build the image
docker build -t self-improving-ai .

# Run with Docker Compose (includes Redis)
docker-compose up -d

# Or run standalone
docker run -p 8000:8000 self-improving-ai
```

**Docker Compose Configuration:**
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - LLM_STUDIO_URL=http://host.docker.internal:1234
    depends_on:
      - redis
    volumes:
      - .:/app
  
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379

# LM Studio Configuration
LLM_STUDIO_URL=http://localhost:1234
LM_STUDIO_MODEL=gpt-oss-20b
LM_STUDIO_TEMPERATURE=0.7
LM_STUDIO_MAX_TOKENS=2048
LM_STUDIO_TIMEOUT=30

# Application Settings
LOG_LEVEL=INFO
USE_REAL_LLM=true
TESTING=false

# Self-Improvement Settings
SAFETY_LEVEL=conservative
AUTO_APPLY_THRESHOLD=8.0
MAX_CONCURRENT_IMPROVEMENTS=3
```

### Configuration Options

**Safety Levels:**
- `conservative`: Only low-risk improvements (recommended)
- `moderate`: Low and medium-risk improvements
- `aggressive`: All improvements (use with caution)

**Auto-Apply Threshold:**
- Score 0-10: Higher scores = more likely to auto-apply
- Recommended: 8.0 for conservative use

## 📋 Common Use Cases

### 1. Code Review and Analysis

```python
from app.code_analyzer import CodeAnalyzer

# Analyze your codebase
analyzer = CodeAnalyzer("./my_project")
reports = analyzer.analyze_project()

# Get recommendations
for file_path, report in reports.items():
    if report.quality_score < 70:
        print(f"\n{file_path} needs attention:")
        for issue in report.issues:
            print(f"  - {issue.message} (Line {issue.line_number})")
        
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  • {rec}")
```

### 2. Automated Code Improvement

```python
import asyncio
from app.self_improvement_engine import SelfImprovementEngine

async def improve_codebase():
    engine = SelfImprovementEngine("./my_project")
    
    # Start improvement cycle
    cycle_id = await engine.trigger_improvement_cycle("manual")
    
    # Wait for completion
    while engine.current_cycle:
        await asyncio.sleep(5)
    
    # Check results
    history = engine.get_cycle_history(limit=1)
    cycle = history[0]
    
    if cycle['status'] == 'completed':
        print(f"✅ Applied {len(cycle['applied_improvements'])} improvements")
    else:
        print(f"❌ Cycle failed: {cycle['status']}")

asyncio.run(improve_codebase())
```

### 3. Interactive AI Assistant

```python
import asyncio
from app.ai_assistant_service import AIAssistantService

async def chat_session():
    service = AIAssistantService()
    session_id = None
    user_id = "developer"
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ['quit', 'exit']:
            break
        
        result = await service.process_query(
            query=query,
            session_id=session_id,
            user_id=user_id
        )
        
        print(f"AI: {result['response']}")
        session_id = result['session_id']

asyncio.run(chat_session())
```

### 4. Performance Monitoring

```python
from app.performance_monitor import PerformanceMonitor
import time

# Initialize monitor
monitor = PerformanceMonitor()

# Track operations
start_time = time.time()
# ... your code here ...
duration = time.time() - start_time

monitor.record_response_time("my_operation", duration)

# Get performance report
report = monitor.get_performance_report()
print(f"Performance Summary: {report.summary}")
```

## 🧪 Testing

### Run All Tests
```bash
# Integration tests
python run_integration_tests.py

# Functionality tests
python test_project_functionality.py

# Specific test suites
python -m pytest tests/test_ai_assistant_service.py -v
python -m pytest tests/test_conversation_manager.py -v
python -m pytest tests/test_code_analyzer.py -v
```

### Test with Real LM Studio
```bash
# Start LM Studio on localhost:1234 first
python -m pytest tests/test_e2e_conversation_integration.py -m integration -v
```

## 🔒 Safety Features

### Built-in Safety Mechanisms

1. **Risk Assessment**: All improvements are risk-rated
2. **Safety Filtering**: Dangerous changes are blocked
3. **Code Validation**: Syntax and import checking
4. **Test Validation**: Comprehensive testing before applying changes
5. **Git Integration**: Automatic rollback points
6. **Emergency Stop**: Immediate halt capability

### Safety Best Practices

```python
# Always use conservative mode for production
engine = SelfImprovementEngine(
    project_root=".",
    config={"safety_level": "conservative"}
)

# Monitor improvement cycles
status = engine.get_current_status()
if status["current_cycle"]:
    print(f"Active cycle: {status['current_cycle']['status']}")

# Emergency stop if needed
await engine.emergency_stop()
```

## 🚨 Troubleshooting

### Common Issues

**1. Redis Connection Failed**
```
Error: Redis connection refused
Solution: Install and start Redis, or use in-memory fallback (automatic)
```

**2. LM Studio Not Available**
```
Error: LM Studio connection failed
Solution: Start LM Studio on localhost:1234, or use mocked responses for testing
```

**3. Import Errors**
```
Error: ModuleNotFoundError
Solution: Ensure virtual environment is activated and dependencies installed
```

**4. Permission Errors**
```
Error: Permission denied
Solution: Ensure write permissions for .kiro/, test_results/, and project directories
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m pytest tests/ -v -s --tb=long

# Check application logs
tail -f logs/app.log  # if logging to file
```

## 📊 Monitoring and Metrics

### Performance Metrics
- Response times for all operations
- Memory usage and optimization
- Error rates and types
- Improvement success rates
- Code quality trends

### Health Checks
```bash
# Application health
curl http://localhost:8000/health

# Component status
curl http://localhost:8000/status

# Performance metrics
curl http://localhost:8000/metrics
```

## 🔄 Continuous Improvement

### Automated Improvement Cycles

```python
# Schedule regular improvements
import asyncio
from app.self_improvement_engine import SelfImprovementEngine

async def scheduled_improvements():
    engine = SelfImprovementEngine(".")
    
    # Start scheduler for automatic improvements
    await engine.start_scheduler()
    
    # Let it run (will check every hour by default)
    await asyncio.sleep(3600)  # Run for 1 hour
    
    # Stop scheduler
    await engine.stop_scheduler()

asyncio.run(scheduled_improvements())
```

### Custom Improvement Rules

```python
# Add custom improvement patterns
from app.improvement_engine import ImprovementEngine

engine = ImprovementEngine(".")

# Analyze with custom focus
improvements = engine.analyze_and_suggest_improvements()
performance_improvements = [
    imp for imp in improvements 
    if imp.type.value == "performance"
]

print(f"Found {len(performance_improvements)} performance improvements")
```

## 📚 Advanced Usage

### Custom Agents

```python
from app.agent_registry import registry, AgentType

# Register custom agent
registry.register_agent(
    agent_type=AgentType.CUSTOM,
    handler=my_custom_handler,
    topics=["custom_tasks"]
)
```

### Custom Code Patterns

```python
from app.improvement_engine import PatternDetector

# Add custom improvement patterns
detector = PatternDetector()
detector.add_pattern(
    name="custom_optimization",
    pattern=r"my_pattern_regex",
    improvement_type="performance",
    suggestion="Use optimized version"
)
```

## 🎯 Production Deployment

### Production Checklist

- [ ] Redis server configured and running
- [ ] LM Studio or alternative LLM service available
- [ ] Environment variables configured
- [ ] Logging configured for production
- [ ] Monitoring and alerting set up
- [ ] Backup strategy for conversation data
- [ ] Security review completed
- [ ] Load testing performed

### Scaling Considerations

- Use Redis Cluster for high availability
- Deploy multiple app instances behind load balancer
- Monitor memory usage and optimize as needed
- Implement rate limiting for API endpoints
- Set up proper logging and monitoring

## 📞 Support

### Getting Help

1. **Check the logs** for error messages
2. **Run the test suite** to identify issues
3. **Review the documentation** for configuration options
4. **Check GitHub issues** for known problems
5. **Create a new issue** with detailed information

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## 🎉 You're Ready!

The self-improving AI assistant is now ready to help you:

- **Analyze your code** for quality and performance issues
- **Suggest improvements** with safety validation
- **Automate code improvements** with rollback protection
- **Provide conversational AI assistance** for development tasks
- **Monitor performance** and track improvements over time

Start with the FastAPI web interface for the easiest experience, or dive into the Python API for programmatic control. Happy coding! 🚀