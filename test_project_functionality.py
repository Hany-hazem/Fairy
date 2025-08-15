#!/usr/bin/env python3
"""
Comprehensive Project Functionality Test

This script tests the core functionality of the self-improving AI assistant
to ensure everything works correctly after cleanup.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all core modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        # Core application modules
        from app.main import app
        from app.ai_assistant_service import AIAssistantService
        from app.conversation_manager import ConversationManager
        from app.self_improvement_engine import SelfImprovementEngine
        from app.improvement_engine import ImprovementEngine
        from app.code_analyzer import CodeAnalyzer
        from app.code_modifier import CodeModifier
        from app.test_runner import TestRunner
        from app.version_control import GitIntegration
        from app.performance_monitor import PerformanceMonitor
        from app.safety_filter import safety_filter
        from app.llm_studio_client import LMStudioClient
        from app.models import ConversationSession, Message
        from app.config import settings
        
        # Agent modules
        from agents.ai_assistant_agent import AIAssistantAgent
        from agents.self_improvement_agent import SelfImprovementAgent
        from app.agent_registry import AgentRegistry
        
        print("✅ All core modules imported successfully")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_ai_assistant_service():
    """Test AI Assistant Service functionality"""
    print("\n🤖 Testing AI Assistant Service...")
    
    try:
        from app.ai_assistant_service import AIAssistantService
        from app.llm_studio_client import LMStudioClient, LMStudioConfig
        from unittest.mock import Mock
        
        # Create service with mocked LLM
        service = AIAssistantService()
        
        # Mock the LLM client
        mock_client = Mock(spec=LMStudioClient)
        mock_client.chat.return_value = "Hello! I'm working correctly."
        mock_client.health_check.return_value = True
        mock_client.validate_connection.return_value = {"status": "connected"}
        
        service.studio_client = mock_client
        
        # Test basic functionality
        assert service is not None
        assert service.conversation_manager is not None
        assert service.max_context_tokens == 4000
        
        print("✅ AI Assistant Service initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ AI Assistant Service error: {e}")
        traceback.print_exc()
        return False

def test_conversation_manager():
    """Test Conversation Manager functionality"""
    print("\n💬 Testing Conversation Manager...")
    
    try:
        from app.conversation_manager import ConversationManager
        
        # Create manager (will use in-memory fallback if Redis not available)
        manager = ConversationManager()
        
        # Test session creation
        session = manager.create_session(user_id="test_user")
        assert session is not None
        assert session.user_id == "test_user"
        
        # Test message addition
        message = manager.add_message(session.id, "user", "Hello, world!")
        assert message is not None
        assert message.content == "Hello, world!"
        
        # Test context retrieval
        context = manager.get_context(session.id)
        assert context is not None
        assert len(context.recent_messages) == 1
        
        # Cleanup
        manager.clear_session(session.id)
        
        print("✅ Conversation Manager working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Conversation Manager error: {e}")
        traceback.print_exc()
        return False

def test_self_improvement_engine():
    """Test Self-Improvement Engine functionality"""
    print("\n🔧 Testing Self-Improvement Engine...")
    
    try:
        from app.self_improvement_engine import SelfImprovementEngine, SafetyLevel
        import tempfile
        
        # Create engine with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = SelfImprovementEngine(
                project_root=temp_dir,
                config={"safety_level": "conservative"}
            )
            
            # Test initialization
            assert engine is not None
            assert engine.safety_level == SafetyLevel.CONSERVATIVE
            assert engine.project_root.exists()
            
            # Test status retrieval
            status = engine.get_current_status()
            assert status is not None
            assert "is_running" in status
            assert "safety_level" in status
            
            # Test cycle history
            history = engine.get_cycle_history()
            assert isinstance(history, list)
        
        print("✅ Self-Improvement Engine initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Self-Improvement Engine error: {e}")
        traceback.print_exc()
        return False

def test_code_analyzer():
    """Test Code Analyzer functionality"""
    print("\n🔍 Testing Code Analyzer...")
    
    try:
        from app.code_analyzer import CodeAnalyzer
        import tempfile
        
        # Create analyzer with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = CodeAnalyzer(temp_dir)
            
            # Create a test Python file
            test_file = Path(temp_dir) / "test_code.py"
            test_file.write_text('''
def test_function():
    """A simple test function"""
    return "Hello, World!"

class TestClass:
    def method(self):
        return 42
''')
            
            # Test file analysis
            report = analyzer.analyze_file(str(test_file))
            assert report is not None
            assert report.file_path == str(test_file)
            assert report.quality_score >= 0
            assert isinstance(report.issues, list)
            assert report.complexity_metrics is not None
        
        print("✅ Code Analyzer working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Code Analyzer error: {e}")
        traceback.print_exc()
        return False

def test_version_control():
    """Test Version Control functionality"""
    print("\n📝 Testing Version Control...")
    
    try:
        from app.version_control import GitIntegration
        import tempfile
        
        # Create git integration with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            git = GitIntegration(temp_dir)
            
            # Test initialization
            assert git is not None
            assert git.repo_path.exists()
            assert git.audit_log_path.parent.exists()
            
            # Test repository initialization
            success = git.init_repo_if_needed()
            assert isinstance(success, bool)
        
        print("✅ Version Control initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Version Control error: {e}")
        traceback.print_exc()
        return False

def test_performance_monitor():
    """Test Performance Monitor functionality"""
    print("\n📊 Testing Performance Monitor...")
    
    try:
        from app.performance_monitor import PerformanceMonitor
        
        # Create monitor
        monitor = PerformanceMonitor()
        
        # Test basic functionality
        assert monitor is not None
        
        # Test metric recording
        monitor.record_response_time("test_operation", 0.5)
        monitor.record_error("test_error", "Test error message")
        
        # Test report generation
        report = monitor.get_performance_report()
        assert report is not None
        assert hasattr(report, 'summary')
        assert hasattr(report, 'alerts')
        
        print("✅ Performance Monitor working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Performance Monitor error: {e}")
        traceback.print_exc()
        return False

def test_fastapi_app():
    """Test FastAPI application"""
    print("\n🌐 Testing FastAPI Application...")
    
    try:
        from app.main import app
        from fastapi.testclient import TestClient
        
        # Create test client
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        
        print("✅ FastAPI Application working correctly")
        return True
        
    except Exception as e:
        print(f"❌ FastAPI Application error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all functionality tests"""
    print("🚀 Self-Improving AI Assistant - Functionality Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_ai_assistant_service,
        test_conversation_manager,
        test_self_improvement_engine,
        test_code_analyzer,
        test_version_control,
        test_performance_monitor,
        test_fastapi_app,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test.__name__}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All functionality tests passed!")
        print("✅ The self-improving AI assistant is working correctly!")
        return True
    else:
        print(f"\n⚠️  {total - passed} functionality tests failed")
        print("❌ Some components need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)