#!/usr/bin/env python3
"""
Start Self-Improving AI Assistant with LM Studio Integration

This script helps you start the AI assistant with proper LM Studio configuration.
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_lm_studio():
    """Check if LM Studio is running and accessible"""
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ LM Studio is running")
            
            if models.get('data'):
                print("📋 Available models:")
                for model in models['data']:
                    print(f"   • {model.get('id', 'Unknown')}")
                return True
            else:
                print("⚠️  LM Studio is running but no models are loaded")
                return False
        else:
            print(f"❌ LM Studio responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to LM Studio at http://localhost:1234")
        print("   Make sure LM Studio is running and the server is started")
        return False
    except Exception as e:
        print(f"❌ Error checking LM Studio: {e}")
        return False

def create_env_file():
    """Create .env file with LM Studio configuration"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("ℹ️  .env file already exists")
        return
    
    print("📝 Creating .env file for LM Studio...")
    
    env_content = """# LM Studio Configuration
LLM_STUDIO_URL=http://localhost:1234
LM_STUDIO_MODEL=gpt-oss-20b
LM_STUDIO_TEMPERATURE=0.7
LM_STUDIO_MAX_TOKENS=2048
LM_STUDIO_TIMEOUT=30

# Application Settings
LOG_LEVEL=INFO
USE_REAL_LLM=true
SAFETY_LEVEL=conservative

# Optional Services (will use fallbacks if not available)
REDIS_URL=redis://localhost:6379
"""
    
    env_file.write_text(env_content)
    print("✅ Created .env file")

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting AI Assistant server...")
    print("🌐 Web UI will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🔧 API info at: http://localhost:8000/api")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
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

def main():
    """Main function"""
    print("🤖 Self-Improving AI Assistant - LM Studio Setup")
    print("=" * 55)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    print("✅ Python version OK")
    
    # Create .env file
    create_env_file()
    
    # Check LM Studio
    print("\n🔍 Checking LM Studio connection...")
    lm_studio_ok = check_lm_studio()
    
    if not lm_studio_ok:
        print("\n⚠️  LM Studio Issues Detected")
        print("\n📋 To fix LM Studio connection:")
        print("1. Open LM Studio application")
        print("2. Go to the 'Local Server' tab")
        print("3. Load your gpt-oss-20b model")
        print("4. Click 'Start Server' (should show port 1234)")
        print("5. Make sure the server is running and accessible")
        
        choice = input("\nContinue anyway? The assistant will work with mocked responses (y/N): ")
        if choice.lower() != 'y':
            print("👋 Setup cancelled. Fix LM Studio and try again.")
            return
        
        print("\n⚠️  Starting with mocked LLM responses...")
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()