#!/usr/bin/env python3
"""
Basic Usage Examples for Self-Improving AI Assistant

This script demonstrates the core functionality of the AI assistant.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import the app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

async def example_conversation():
    """Example: Basic conversation with the AI assistant"""
    print("🤖 Example 1: Basic Conversation")
    print("=" * 40)
    
    try:
        from app.ai_assistant_service import AIAssistantService
        
        # Initialize the service
        service = AIAssistantService()
        
        # Start a conversation
        print("Starting conversation...")
        result = await service.process_query(
            query="Hello! I'm a Python developer. Can you help me improve my code?",
            user_id="example_user"
        )
        
        print(f"AI Response: {result['response']}")
        session_id = result['session_id']
        
        # Continue the conversation
        result2 = await service.process_query(
            query="What are some common Python performance issues I should look out for?",
            session_id=session_id,
            user_id="example_user"
        )
        
        print(f"AI Response: {result2['response']}")
        
        # Get conversation history
        history = await service.get_conversation_history(session_id)
        print(f"\nConversation Summary:")
        print(f"- Session ID: {session_id}")
        print(f"- Total messages: {len(history['messages'])}")
        print(f"- User ID: {history['user_id']}")
        
        # Clean up
        await service.clear_session(session_id)
        print("✅ Conversation example completed")
        
    except Exception as e:
        print(f"❌ Error in conversation example: {e}")

def example_code_analysis():
    """Example: Analyze code quality"""
    print("\n🔍 Example 2: Code Analysis")
    print("=" * 40)
    
    try:
        from app.code_analyzer import CodeAnalyzer
        import tempfile
        
        # Create a sample Python file with some issues
        sample_code = '''
def inefficient_function(data):
    result = ""
    for item in data:
        result += str(item)  # Inefficient string concatenation
    return result

def complex_function(a, b, c, d, e, f, g, h):  # Too many parameters
    if a > 0:
        if b > 0:
            if c > 0:  # Deep nesting
                return a + b + c + d + e + f + g + h
    return 0

class SampleClass:
    def method_without_docstring(self):
        magic_number = 42  # Magic number
        return magic_number * 2
'''
        
        # Write sample code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(sample_code)
            temp_file = f.name
        
        try:
            # Analyze the code
            analyzer = CodeAnalyzer()
            report = analyzer.analyze_file(temp_file)
            
            print(f"Analysis Results for sample code:")
            print(f"- Quality Score: {report.quality_score:.1f}/100")
            print(f"- Issues Found: {len(report.issues)}")
            print(f"- Lines of Code: {report.complexity_metrics.lines_of_code}")
            print(f"- Functions: {report.complexity_metrics.function_count}")
            print(f"- Classes: {report.complexity_metrics.class_count}")
            
            if report.issues:
                print(f"\nTop Issues:")
                for issue in report.issues[:3]:
                    print(f"  • Line {issue.line_number}: {issue.message}")
                    if issue.suggestion:
                        print(f"    Suggestion: {issue.suggestion}")
            
            if report.recommendations:
                print(f"\nRecommendations:")
                for rec in report.recommendations[:3]:
                    print(f"  • {rec}")
            
            print("✅ Code analysis example completed")
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"❌ Error in code analysis example: {e}")

async def example_self_improvement():
    """Example: Self-improvement cycle (simulation)"""
    print("\n🔧 Example 3: Self-Improvement Simulation")
    print("=" * 40)
    
    try:
        from app.self_improvement_engine import SelfImprovementEngine, SafetyLevel
        import tempfile
        
        # Create a temporary project directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a sample project structure
            (temp_path / "app").mkdir()
            (temp_path / "tests").mkdir()
            
            # Create sample code with improvement opportunities
            sample_code = '''
def slow_function(data):
    """Function that could be optimized"""
    result = ""
    for item in data:
        result += str(item)  # This could be improved
    return result

def another_function():
    """Another function"""
    return "Hello, World!"
'''
            
            (temp_path / "app" / "sample.py").write_text(sample_code)
            
            # Create sample test
            test_code = '''
import unittest
import sys
sys.path.append('..')
from app.sample import slow_function, another_function

class TestSample(unittest.TestCase):
    def test_slow_function(self):
        result = slow_function([1, 2, 3])
        self.assertEqual(result, "123")
    
    def test_another_function(self):
        result = another_function()
        self.assertEqual(result, "Hello, World!")

if __name__ == "__main__":
    unittest.main()
'''
            
            (temp_path / "tests" / "test_sample.py").write_text(test_code)
            
            # Initialize self-improvement engine
            engine = SelfImprovementEngine(
                project_root=str(temp_path),
                config={
                    "safety_level": "conservative",
                    "auto_apply_threshold": 8.0
                }
            )
            
            print(f"Initialized engine for project: {temp_path}")
            print(f"Safety level: {engine.safety_level.value}")
            
            # Get current status
            status = engine.get_current_status()
            print(f"Engine status: {status['is_running']}")
            print(f"Total cycles run: {status['total_cycles']}")
            print(f"Successful cycles: {status['successful_cycles']}")
            
            # Simulate improvement analysis (without actually running)
            print(f"\n🔍 Simulating improvement analysis...")
            print(f"- Would analyze code quality")
            print(f"- Would identify improvement opportunities")
            print(f"- Would create safe modification plans")
            print(f"- Would run tests to validate changes")
            print(f"- Would apply improvements with rollback points")
            
            print("✅ Self-improvement simulation completed")
            
    except Exception as e:
        print(f"❌ Error in self-improvement example: {e}")

def example_performance_monitoring():
    """Example: Performance monitoring"""
    print("\n📊 Example 4: Performance Monitoring")
    print("=" * 40)
    
    try:
        from app.performance_monitor import PerformanceMonitor
        import time
        import random
        
        # Initialize monitor
        monitor = PerformanceMonitor()
        
        # Simulate some operations with performance tracking
        operations = ["database_query", "api_call", "file_processing", "computation"]
        
        print("Simulating operations and tracking performance...")
        
        for i in range(10):
            operation = random.choice(operations)
            
            # Simulate operation duration
            duration = random.uniform(0.1, 2.0)
            time.sleep(0.1)  # Brief pause for realism
            
            # Record the performance
            monitor.record_response_time(operation, duration)
            
            # Occasionally record an error
            if random.random() < 0.1:
                monitor.record_error(operation, f"Simulated error {i}")
            
            print(f"  • {operation}: {duration:.3f}s")
        
        # Get performance report
        report = monitor.get_performance_report()
        
        print(f"\nPerformance Report:")
        print(f"- Metrics collected: {len(report.metrics)}")
        print(f"- Time period: {report.start_time.strftime('%H:%M:%S')} - {report.end_time.strftime('%H:%M:%S')}")
        
        if report.summary:
            print(f"- Summary statistics available for {len(report.summary)} operations")
            
        if report.alerts:
            print(f"- Alerts: {len(report.alerts)}")
            for alert in report.alerts[:3]:
                print(f"  ⚠️  {alert}")
        
        print("✅ Performance monitoring example completed")
        
    except Exception as e:
        print(f"❌ Error in performance monitoring example: {e}")

def example_version_control():
    """Example: Version control integration"""
    print("\n📝 Example 5: Version Control Integration")
    print("=" * 40)
    
    try:
        from app.version_control import GitIntegration
        import tempfile
        
        # Create temporary directory for git operations
        with tempfile.TemporaryDirectory() as temp_dir:
            git = GitIntegration(temp_dir)
            
            print(f"Working in temporary directory: {temp_dir}")
            
            # Initialize repository
            if git.init_repo_if_needed():
                print("✅ Git repository initialized")
            else:
                print("ℹ️  Git repository already exists or initialization failed")
            
            # Check if it's a git repo
            is_repo = git.is_git_repo()
            print(f"Is git repository: {is_repo}")
            
            if is_repo:
                # Get current branch
                branch = git.get_current_branch()
                print(f"Current branch: {branch}")
                
                # Create a test file
                test_file = Path(temp_dir) / "test.txt"
                test_file.write_text("Hello, World!")
                
                # Add and commit
                git.add_files(["test.txt"])
                commit_hash = git.commit_changes("Add test file")
                
                if commit_hash:
                    print(f"✅ Created commit: {commit_hash[:8]}")
                    
                    # Create rollback point
                    rollback_point = git.create_rollback_point("Before changes")
                    if rollback_point:
                        print(f"✅ Created rollback point: {rollback_point[:8]}")
                
                # Get change history
                history = git.get_change_history(limit=5)
                print(f"Change history entries: {len(history)}")
            
            print("✅ Version control example completed")
            
    except Exception as e:
        print(f"❌ Error in version control example: {e}")

async def main():
    """Run all examples"""
    print("🚀 Self-Improving AI Assistant - Basic Usage Examples")
    print("=" * 60)
    print()
    
    # Run examples
    await example_conversation()
    example_code_analysis()
    await example_self_improvement()
    example_performance_monitoring()
    example_version_control()
    
    print("\n🎉 All examples completed!")
    print("\nNext steps:")
    print("- Try the interactive startup script: python start_assistant.py")
    print("- Start the web server: uvicorn app.main:app --reload")
    print("- Read the full usage guide: USAGE_GUIDE.md")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted")
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()