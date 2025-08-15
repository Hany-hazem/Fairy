#!/usr/bin/env python3
"""
Fairy AI Assistant - Easy Startup Script

This script provides an easy way to start and interact with the AI assistant.

Copyright (c) 2024 Hani Hazem
Licensed under the MIT License. See LICENSE file in the project root for full license information.
Repository: https://github.com/Hany-hazem/Fairy
Contact: hany.hazem.cs@gmail.com
"""

import asyncio
import sys
import os
from pathlib import Path
import subprocess
import time

def print_banner():
    """Print startup banner"""
    print("🤖 Self-Improving AI Assistant")
    print("=" * 50)
    print("Choose how you'd like to use the assistant:")
    print()

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if in virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment recommended")
    
    # Check required packages
    try:
        import fastapi
        import uvicorn
        import redis
        print("✅ Core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def check_services():
    """Check optional services"""
    print("\n🔧 Checking optional services...")
    
    services = {
        "Redis": check_redis(),
        "LM Studio": check_lm_studio(),
        "Git": check_git()
    }
    
    for service, available in services.items():
        status = "✅" if available else "⚠️ "
        print(f"{status} {service}: {'Available' if available else 'Not available (will use fallback)'}")
    
    return services

def check_redis():
    """Check if Redis is available"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_timeout=1)
        r.ping()
        return True
    except:
        return False

def check_lm_studio():
    """Check if LM Studio is available"""
    try:
        import httpx
        response = httpx.get("http://localhost:1234/v1/models", timeout=2)
        return response.status_code == 200
    except:
        return False

def check_git():
    """Check if Git is available"""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except:
        return False

def start_web_server():
    """Start the FastAPI web server"""
    print("\n🌐 Starting web server...")
    print("Access the assistant at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

async def interactive_chat():
    """Start interactive chat session"""
    print("\n💬 Starting interactive chat...")
    print("Type 'quit' to exit")
    print()
    
    try:
        from app.ai_assistant_service import AIAssistantService
        
        service = AIAssistantService()
        session_id = None
        user_id = "interactive_user"
        
        while True:
            try:
                query = input("You: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                print("🤔 Thinking...")
                result = await service.process_query(
                    query=query,
                    session_id=session_id,
                    user_id=user_id
                )
                
                print(f"🤖 AI: {result['response']}")
                session_id = result['session_id']
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Try again or type 'quit' to exit")
        
        print("\n👋 Chat session ended")
        
    except ImportError as e:
        print(f"❌ Could not start chat: {e}")
        print("Make sure all dependencies are installed")

async def run_code_analysis():
    """Run code analysis on the current project"""
    print("\n🔍 Running code analysis...")
    
    try:
        from app.code_analyzer import CodeAnalyzer
        
        analyzer = CodeAnalyzer(".")
        print("Analyzing project files...")
        
        reports = analyzer.analyze_project()
        summary = analyzer.get_project_summary(reports)
        
        print(f"\n📊 Analysis Results:")
        print(f"Files analyzed: {summary['total_files_analyzed']}")
        print(f"Total issues: {summary['total_issues']}")
        print(f"Average quality score: {summary['average_quality_score']:.1f}/100")
        
        if summary['worst_files']:
            print(f"\n⚠️  Files needing attention:")
            for file_info in summary['worst_files'][:3]:
                print(f"  • {file_info['file']} (Score: {file_info['score']:.1f}, Issues: {file_info['issues']})")
        
        print(f"\n💡 Recommendations:")
        for rec in summary['recommendations'][:3]:
            print(f"  • {rec}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

async def trigger_self_improvement():
    """Trigger a self-improvement cycle"""
    print("\n🔧 Starting self-improvement cycle...")
    print("This will analyze and potentially improve the codebase")
    
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        return
    
    try:
        from app.self_improvement_engine import SelfImprovementEngine
        
        engine = SelfImprovementEngine(
            project_root=".",
            config={"safety_level": "conservative"}
        )
        
        print("🚀 Starting improvement cycle...")
        cycle_id = await engine.trigger_improvement_cycle("manual")
        print(f"Cycle ID: {cycle_id}")
        
        # Monitor progress
        last_status = None
        while engine.current_cycle:
            status = engine.get_current_status()
            current_status = status['current_cycle']['status'] if status['current_cycle'] else None
            
            if current_status != last_status:
                print(f"Status: {current_status}")
                last_status = current_status
            
            await asyncio.sleep(2)
        
        # Get results
        history = engine.get_cycle_history(limit=1)
        if history:
            cycle = history[0]
            print(f"\n✅ Cycle completed: {cycle['status']}")
            print(f"Applied improvements: {len(cycle['applied_improvements'])}")
            print(f"Failed improvements: {len(cycle['failed_improvements'])}")
            
            if cycle['applied_improvements']:
                print("🎉 Your code has been improved!")
            else:
                print("ℹ️  No improvements were applied (code is already good or changes were too risky)")
        
    except Exception as e:
        print(f"❌ Self-improvement failed: {e}")

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running tests...")
    
    try:
        # Run integration tests
        result = subprocess.run([sys.executable, "run_integration_tests.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed")
            print(result.stdout)
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Could not run tests: {e}")

async def main():
    """Main menu"""
    print_banner()
    
    if not check_dependencies():
        return
    
    services = check_services()
    
    while True:
        print("\n📋 Options:")
        print("1. 🌐 Start Web Server (Recommended)")
        print("2. 💬 Interactive Chat")
        print("3. 🔍 Analyze Code")
        print("4. 🔧 Self-Improve Code")
        print("5. 🧪 Run Tests")
        print("6. ❓ Help")
        print("0. 👋 Exit")
        
        choice = input("\nChoose an option (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            start_web_server()
        elif choice == "2":
            await interactive_chat()
        elif choice == "3":
            await run_code_analysis()
        elif choice == "4":
            await trigger_self_improvement()
        elif choice == "5":
            run_tests()
        elif choice == "6":
            show_help()
        else:
            print("❌ Invalid choice. Please try again.")

def show_help():
    """Show help information"""
    print("\n❓ Help Information")
    print("=" * 30)
    print()
    print("🌐 Web Server: Start the full web interface with API")
    print("   - Best for development and testing")
    print("   - Access at http://localhost:8000")
    print("   - API docs at http://localhost:8000/docs")
    print()
    print("💬 Interactive Chat: Command-line conversation")
    print("   - Direct chat with the AI assistant")
    print("   - Good for quick questions")
    print()
    print("🔍 Code Analysis: Analyze code quality")
    print("   - Scans project for issues")
    print("   - Provides quality scores and recommendations")
    print()
    print("🔧 Self-Improve: Automatically improve code")
    print("   - Analyzes and applies safe improvements")
    print("   - Uses conservative safety settings")
    print("   - Creates rollback points")
    print()
    print("🧪 Run Tests: Execute the test suite")
    print("   - Validates all functionality")
    print("   - Ensures system is working correctly")
    print()
    print("📚 For more information, see USAGE_GUIDE.md")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)