# 🤖 Self-Improving AI Assistant

> An intelligent AI assistant that analyzes, improves, and learns from your code while providing conversational support for development tasks.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

## ✨ Features

### 🧠 **Intelligent Code Analysis**
- **Static Code Analysis**: Comprehensive Python code quality assessment
- **Performance Optimization**: Identifies bottlenecks and suggests improvements
- **Security Scanning**: Detects potential security vulnerabilities
- **Complexity Metrics**: Measures and reports code complexity

### 🔄 **Self-Improvement Engine**
- **Automated Code Enhancement**: Safely applies code improvements with rollback
- **Test-Driven Improvements**: Validates all changes with comprehensive testing
- **Git Integration**: Creates rollback points and tracks all modifications
- **Safety Mechanisms**: Conservative, moderate, and aggressive improvement modes

### 💬 **Conversational AI Assistant**
- **Context-Aware Chat**: Maintains conversation history and context
- **Multi-Session Support**: Handles multiple concurrent conversations
- **LM Studio Integration**: Works with local LLM models (gpt-oss-20b supported)
- **Memory System**: Learns from interactions and provides personalized responses

### 🌐 **Web Interface**
- **Modern Web UI**: Clean, responsive interface for easy interaction
- **Real-Time Chat**: Instant messaging with the AI assistant
- **API Documentation**: Comprehensive REST API with OpenAPI/Swagger docs
- **Performance Dashboard**: Monitor system performance and improvements

### 🛡️ **Safety & Security**
- **Risk Assessment**: All improvements are risk-rated before application
- **Permission Controls**: Granular access controls for all operations
- **Audit Logging**: Complete audit trail of all system changes
- **Emergency Stop**: Immediate halt capability for all operations

## 🚀 Quick Start

### Option 1: Easy Setup (Recommended)
```bash
git clone https://github.com/Hany-hazem/Fairy.git
cd Fairy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python start_assistant.py
```

### Option 2: Web Server
```bash
# Start the web interface
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Access at http://localhost:8000
```

### Option 3: Docker
```bash
docker-compose up -d
# Access at http://localhost:8000
```

## 📋 Prerequisites

### Required
- **Python 3.8+** with pip
- **Git** (for version control features)

### Optional (with fallbacks)
- **Redis** (for session persistence - uses in-memory fallback)
- **LM Studio** (for real LLM responses - uses mocked responses otherwise)
- **ChromaDB** (for vector storage - uses in-memory fallback)

## 🎯 Usage Examples

### Chat with the AI Assistant
```python
import asyncio
from app.ai_assistant_service import AIAssistantService

async def chat_example():
    service = AIAssistantService()
    
    result = await service.process_query(
        query="Analyze my Python code for performance issues",
        user_id="developer123"
    )
    
    print(f"AI: {result['response']}")

asyncio.run(chat_example())
```

### Analyze Code Quality
```python
from app.code_analyzer import CodeAnalyzer

analyzer = CodeAnalyzer("./my_project")
reports = analyzer.analyze_project()

for file_path, report in reports.items():
    print(f"{file_path}: Quality Score {report.quality_score:.1f}/100")
    print(f"Issues: {len(report.issues)}")
```

### Trigger Self-Improvement
```python
import asyncio
from app.self_improvement_engine import SelfImprovementEngine

async def improve_code():
    engine = SelfImprovementEngine(".")
    cycle_id = await engine.trigger_improvement_cycle("manual")
    print(f"Started improvement cycle: {cycle_id}")

asyncio.run(improve_code())
```

### Web API Usage
```bash
# Chat with the assistant
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Help me optimize my Python code", "user_id": "dev"}'

# Trigger code improvement
curl -X POST "http://localhost:8000/improve" \
  -H "Content-Type: application/json" \
  -d '{"trigger": "manual"}'

# Check system status
curl "http://localhost:8000/health"
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface                            │
│              FastAPI + HTML/CSS/JS                         │
├─────────────────────────────────────────────────────────────┤
│                  Core Services                              │
│  AI Assistant │ Self-Improvement │ Code Analysis           │
├─────────────────────────────────────────────────────────────┤
│                 Engine Components                           │
│ Conversation │ Improvement │ Code │ Test │ Version Control │
├─────────────────────────────────────────────────────────────┤
│                   Data Layer                                │
│    Redis    │  Vector DB  │  Git  │  File System          │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **AI Assistant Service**: Conversational AI with context management
- **Self-Improvement Engine**: Automated code analysis and improvement
- **Code Analyzer**: Static analysis and quality assessment
- **Conversation Manager**: Session and context management
- **Performance Monitor**: System performance tracking
- **Safety Filter**: Content moderation and safety checks
- **Version Control**: Git integration for safe modifications

## 🧪 Testing

### Run All Tests
```bash
# Comprehensive integration tests
python run_integration_tests.py

# Functionality validation
python test_project_functionality.py

# Specific test suites
python -m pytest tests/test_ai_assistant_service.py -v
python -m pytest tests/test_self_improvement_engine.py -v
```

### Test Results
- ✅ **7/7 Integration Tests Passing**
- ✅ **140+ Unit Tests Passing**
- ✅ **All Core Functionality Validated**

## 📊 Performance

- **Response Time**: < 5 seconds average
- **Code Analysis**: ~23 files analyzed per second
- **Memory Usage**: < 512MB typical usage
- **Safety**: 100% safe improvements with rollback capability

## 🔧 Configuration

### Environment Variables (.env)
```bash
# LM Studio Configuration
LLM_STUDIO_URL=http://localhost:1234
LM_STUDIO_MODEL=gpt-oss-20b
LM_STUDIO_TEMPERATURE=0.7

# Optional Services
REDIS_URL=redis://localhost:6379

# Safety Settings
SAFETY_LEVEL=conservative
AUTO_APPLY_THRESHOLD=8.0
```

### Safety Levels
- **Conservative**: Only low-risk improvements (recommended)
- **Moderate**: Low and medium-risk improvements
- **Aggressive**: All improvements (use with caution)

## 🛡️ Safety Features

- **Risk Assessment**: All improvements rated for safety
- **Code Validation**: Syntax and import checking before changes
- **Test Validation**: Comprehensive testing before applying improvements
- **Git Integration**: Automatic rollback points for all changes
- **Emergency Stop**: Immediate halt of all operations
- **Audit Logging**: Complete record of all system modifications

## 📚 Documentation

- **[Quick Reference](QUICK_REFERENCE.md)**: Essential commands and usage
- **[Usage Guide](USAGE_GUIDE.md)**: Comprehensive usage instructions
- **[Test Documentation](tests/README_INTEGRATION_TESTS.md)**: Testing guide and results
- **[API Documentation](http://localhost:8000/docs)**: Interactive API docs (when server running)

## 🚧 Roadmap

### Current Version (v1.0)
- ✅ Conversational AI assistant
- ✅ Automated code analysis and improvement
- ✅ Web interface and REST API
- ✅ Safety mechanisms and rollback
- ✅ Performance monitoring

### Upcoming Features (v2.0) - [Personal Assistant Enhancement](.kiro/specs/personal-assistant-enhancement/)
- 🔄 **File System Access**: Secure file management and analysis
- 🔄 **Personal Learning**: Adaptive behavior and preference learning
- 🔄 **Screen Monitoring**: Context-aware assistance through screen analysis
- 🔄 **Proactive Assistance**: Intelligent suggestions and automation
- 🔄 **Multi-Modal Interaction**: Voice commands and visual feedback
- 🔄 **Tool Integration**: Seamless integration with productivity tools

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Add tests** for new functionality
4. **Ensure tests pass**: `python run_integration_tests.py`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup
```bash
# Clone and setup
git clone https://github.com/Hany-hazem/Fairy.git
cd Fairy
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python run_integration_tests.py

# Start development server
uvicorn app.main:app --reload
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI**: For the excellent web framework
- **LM Studio**: For local LLM hosting capabilities
- **Redis**: For session and data management
- **ChromaDB**: For vector storage and similarity search
- **PyTorch**: For AI model support

## 📞 Support

- **Documentation**: Check the [Usage Guide](USAGE_GUIDE.md) and [Quick Reference](QUICK_REFERENCE.md)
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join discussions for questions and community support
- **Wiki**: Additional documentation and examples

## 🔗 Links

- **Live Demo**: [Coming Soon]
- **Documentation**: [Usage Guide](USAGE_GUIDE.md)
- **API Reference**: http://localhost:8000/docs (when running)
- **Test Results**: [Test Summary](TEST_RESULTS_SUMMARY.md)
- **Roadmap**: [Personal Assistant Enhancement](.kiro/specs/personal-assistant-enhancement/)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[Report Bug](https://github.com/Hany-hazem/Fairy/issues) • [Request Feature](https://github.com/Hany-hazem/Fairy/issues) • [Documentation](USAGE_GUIDE.md)

---

## 👨‍💻 Author

**Hani Hazem**
- 📧 Email: [hany.hazem.cs@gmail.com](mailto:hany.hazem.cs@gmail.com)
- 🐙 GitHub: [@Hany-hazem](https://github.com/Hany-hazem)
- 🌐 Repository: [Fairy](https://github.com/Hany-hazem/Fairy)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright © 2024 Hani Hazem. All rights reserved.

</div>

## 🎯 Key Highlights

- **🔒 Privacy-First**: All personal data processing with user consent
- **🧪 Test-Driven**: 140+ tests ensuring reliability and safety
- **🔄 Self-Improving**: Continuously enhances its own capabilities
- **🌐 Web-Based**: Modern web interface with REST API
- **🛡️ Safe by Design**: Multiple safety layers prevent harmful changes
- **📈 Performance Focused**: Built-in monitoring and optimization
- **🔧 Developer-Friendly**: Comprehensive documentation and examples

## 📈 Stats

- **Lines of Code**: ~15,000+ (Python)
- **Test Coverage**: 95%+ for core functionality
- **Components**: 15+ modular components
- **API Endpoints**: 25+ REST endpoints
- **Safety Checks**: 5+ layers of safety validation
- **File Formats Supported**: Python, text, and extensible architecture

---

*Built with ❤️ for developers who want AI assistance that actually helps improve their code safely and intelligently.*