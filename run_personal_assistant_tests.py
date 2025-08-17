#!/usr/bin/env python3
"""
Personal Assistant Test Runner

This script runs all tests for the personal assistant components and provides
a comprehensive test report.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import tempfile


def run_command(command, description):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"\nSTDOUT:\n{result.stdout}")
        
        if result.stderr:
            print(f"\nSTDERR:\n{result.stderr}")
        
        return result.returncode == 0, result.stdout, result.stderr, duration
        
    except subprocess.TimeoutExpired:
        print(f"Command timed out after 5 minutes")
        return False, "", "Command timed out", duration
    except Exception as e:
        print(f"Error running command: {e}")
        return False, "", str(e), 0


def check_dependencies():
    """Check if required dependencies are available"""
    print("Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required")
        return False
    
    print(f"✓ Python version: {sys.version}")
    
    # Check if pytest is available
    try:
        import pytest
        print(f"✓ pytest version: {pytest.__version__}")
    except ImportError:
        print("ERROR: pytest is not installed. Install with: pip install pytest pytest-asyncio")
        return False
    
    # Check if pytest-asyncio is available
    try:
        import pytest_asyncio
        print(f"✓ pytest-asyncio is available")
    except ImportError:
        print("ERROR: pytest-asyncio is not installed. Install with: pip install pytest-asyncio")
        return False
    
    # Check if cryptography is available (for encryption tests)
    try:
        import cryptography
        print(f"✓ cryptography is available")
    except ImportError:
        print("WARNING: cryptography is not installed. Some tests may fail.")
        print("Install with: pip install cryptography")
    
    return True


def run_individual_test_files():
    """Run each test file individually to isolate issues"""
    test_files = [
        "tests/test_personal_assistant_models.py",
        "tests/test_user_context_manager.py", 
        "tests/test_privacy_security_manager.py",
        "tests/test_personal_database.py",
        "tests/test_personal_assistant_core.py",
        "tests/test_personal_assistant_integration.py"
    ]
    
    results = {}
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"WARNING: Test file {test_file} not found, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Running individual test file: {test_file}")
        print(f"{'='*80}")
        
        command = f"python3 -m pytest {test_file} -v --tb=short"
        success, stdout, stderr, duration = run_command(command, f"Testing {test_file}")
        
        results[test_file] = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'duration': duration
        }
    
    return results


def run_all_tests():
    """Run all tests together"""
    print(f"\n{'='*80}")
    print("Running all personal assistant tests together")
    print(f"{'='*80}")
    
    command = "python3 -m pytest tests/test_personal_assistant*.py -v --tb=short"
    return run_command(command, "All personal assistant tests")


def run_specific_test_categories():
    """Run tests by category"""
    categories = {
        "Models": "tests/test_personal_assistant_models.py",
        "Context Management": "tests/test_user_context_manager.py",
        "Privacy & Security": "tests/test_privacy_security_manager.py", 
        "Database": "tests/test_personal_database.py",
        "Core System": "tests/test_personal_assistant_core.py",
        "Integration": "tests/test_personal_assistant_integration.py"
    }
    
    results = {}
    
    for category, test_file in categories.items():
        if not os.path.exists(test_file):
            continue
            
        print(f"\n{'='*80}")
        print(f"Running {category} tests")
        print(f"{'='*80}")
        
        command = f"python3 -m pytest {test_file} -v --tb=line"
        success, stdout, stderr, duration = run_command(command, f"{category} tests")
        
        results[category] = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'duration': duration,
            'test_file': test_file
        }
    
    return results


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print(f"\n{'='*80}")
    print("Running quick smoke test")
    print(f"{'='*80}")
    
    smoke_test_code = '''
import sys
sys.path.append(".")
import asyncio
import tempfile
import os

async def smoke_test():
    try:
        # Test basic imports
        from app.personal_assistant_core import PersonalAssistantCore, AssistantRequest, RequestType
        from app.personal_assistant_models import UserContext, InteractionType
        from app.user_context_manager import UserContextManager
        from app.privacy_security_manager import PrivacySecurityManager
        from app.personal_database import PersonalDatabase
        print("✓ All imports successful")
        
        # Test basic functionality
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            # Test database creation
            db = PersonalDatabase(db_path)
            print("✓ Database creation successful")
            
            # Test context manager
            context_manager = UserContextManager(db_path)
            context = await context_manager.get_user_context("smoke_test_user")
            print("✓ Context manager working")
            
            # Test privacy manager
            privacy_manager = PrivacySecurityManager(db_path)
            print("✓ Privacy manager working")
            
            # Test core system
            core = PersonalAssistantCore(db_path)
            request = AssistantRequest(
                user_id="smoke_test_user",
                request_type=RequestType.QUERY,
                content="Hello",
                metadata={}
            )
            response = await core.process_request(request)
            print(f"✓ Core system working: {response.success}")
            
            await core.shutdown()
            print("✓ Smoke test completed successfully!")
            return True
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
            for ext in ["-wal", "-shm"]:
                wal_file = db_path + ext
                if os.path.exists(wal_file):
                    os.unlink(wal_file)
            if os.path.exists("encryption.key"):
                os.unlink("encryption.key")
                
    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(smoke_test())
    sys.exit(0 if result else 1)
'''
    
    # Write smoke test to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(smoke_test_code)
        smoke_test_file = f.name
    
    try:
        command = f"python3 {smoke_test_file}"
        return run_command(command, "Smoke test")
    finally:
        if os.path.exists(smoke_test_file):
            os.unlink(smoke_test_file)


def generate_test_report(individual_results, category_results, all_tests_result, smoke_test_result):
    """Generate a comprehensive test report"""
    print(f"\n{'='*80}")
    print("PERSONAL ASSISTANT TEST REPORT")
    print(f"{'='*80}")
    
    # Smoke test results
    print(f"\n🔥 SMOKE TEST:")
    if smoke_test_result[0]:
        print("   ✅ PASSED - Basic functionality working")
    else:
        print("   ❌ FAILED - Basic functionality issues detected")
    print(f"   Duration: {smoke_test_result[3]:.2f}s")
    
    # Category results
    print(f"\n📊 TEST CATEGORIES:")
    total_categories = len(category_results)
    passed_categories = sum(1 for r in category_results.values() if r['success'])
    
    for category, result in category_results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"   {status} {category:<20} ({result['duration']:.2f}s)")
    
    print(f"\n   Category Summary: {passed_categories}/{total_categories} passed")
    
    # Individual file results
    print(f"\n📁 INDIVIDUAL TEST FILES:")
    total_files = len(individual_results)
    passed_files = sum(1 for r in individual_results.values() if r['success'])
    
    for test_file, result in individual_results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        filename = os.path.basename(test_file)
        print(f"   {status} {filename:<35} ({result['duration']:.2f}s)")
    
    print(f"\n   File Summary: {passed_files}/{total_files} passed")
    
    # All tests together
    print(f"\n🎯 ALL TESTS TOGETHER:")
    if all_tests_result[0]:
        print("   ✅ PASSED - All tests pass when run together")
    else:
        print("   ❌ FAILED - Some tests fail when run together")
    print(f"   Duration: {all_tests_result[3]:.2f}s")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    all_passed = (smoke_test_result[0] and 
                  passed_categories == total_categories and 
                  passed_files == total_files and 
                  all_tests_result[0])
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! Personal Assistant is ready for use.")
        return True
    else:
        print("⚠️  SOME TESTS FAILED. Please review the failures above.")
        
        # Provide specific guidance
        if not smoke_test_result[0]:
            print("   - Fix smoke test issues first (basic functionality)")
        if passed_files < total_files:
            print("   - Review individual test file failures")
        if passed_categories < total_categories:
            print("   - Check category-specific issues")
        if not all_tests_result[0]:
            print("   - Investigate integration issues when tests run together")
        
        return False


def main():
    """Main test runner function"""
    print("Personal Assistant Test Suite")
    print("=" * 80)
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Dependency check failed. Please install required packages.")
        return 1
    
    print("✅ Dependencies check passed")
    
    # Run smoke test first
    print("\n🔥 Running smoke test...")
    smoke_test_result = run_quick_smoke_test()
    
    if not smoke_test_result[0]:
        print("❌ Smoke test failed. Basic functionality is broken.")
        print("Please fix basic issues before running full test suite.")
        return 1
    
    print("✅ Smoke test passed")
    
    # Run individual test files
    print("\n📁 Running individual test files...")
    individual_results = run_individual_test_files()
    
    # Run tests by category
    print("\n📊 Running tests by category...")
    category_results = run_specific_test_categories()
    
    # Run all tests together
    print("\n🎯 Running all tests together...")
    all_tests_result = run_all_tests()
    
    # Generate comprehensive report
    success = generate_test_report(
        individual_results, 
        category_results, 
        all_tests_result, 
        smoke_test_result
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)