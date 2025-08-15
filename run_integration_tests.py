#!/usr/bin/env python3
"""
Integration Test Runner for Self-Improving AI Assistant

This script runs the comprehensive end-to-end integration tests for both
conversation management and self-improvement cycles.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_test_suite(test_pattern: str, description: str) -> bool:
    """Run a test suite and return success status"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_pattern, 
            "-v", 
            "--tb=short",
            "--disable-warnings"
        ], capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ PASSED")
            # Show summary line
            lines = result.stdout.split('\n')
            for line in lines:
                if "passed" in line and ("warning" in line or "error" in line or line.strip().endswith("passed")):
                    print(f"Summary: {line.strip()}")
                    break
        else:
            print("❌ FAILED")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all integration tests"""
    print("Self-Improving AI Assistant - Integration Test Suite")
    print("=" * 60)
    
    # Check if test files exist
    conversation_tests = Path("tests/test_e2e_conversation_integration.py")
    improvement_tests = Path("tests/test_e2e_self_improvement_integration.py")
    
    if not conversation_tests.exists():
        print(f"❌ Conversation integration tests not found: {conversation_tests}")
        return False
    
    if not improvement_tests.exists():
        print(f"❌ Self-improvement integration tests not found: {improvement_tests}")
        return False
    
    # Test suites to run
    test_suites = [
        {
            "pattern": "tests/test_e2e_conversation_integration.py::TestE2EConversationIntegration::test_single_conversation_flow",
            "description": "Single Conversation Flow Test"
        },
        {
            "pattern": "tests/test_e2e_conversation_integration.py::TestE2EConversationIntegration::test_multi_session_conversation_management",
            "description": "Multi-Session Conversation Management Test"
        },
        {
            "pattern": "tests/test_e2e_conversation_integration.py::TestE2EConversationIntegration::test_context_persistence_and_retrieval",
            "description": "Context Persistence and Retrieval Test"
        },
        {
            "pattern": "tests/test_e2e_conversation_integration.py::TestE2EConversationIntegration::test_conversation_performance_benchmarking",
            "description": "Conversation Performance Benchmarking Test"
        },
        {
            "pattern": "tests/test_e2e_self_improvement_integration.py::TestE2ESelfImprovementIntegration::test_safety_mechanism_validation",
            "description": "Self-Improvement Safety Mechanism Test"
        },
        {
            "pattern": "tests/test_e2e_self_improvement_integration.py::TestSelfImprovementSafetyMechanisms::test_safety_filter_blocks_dangerous_changes",
            "description": "Safety Filter Validation Test"
        },
        {
            "pattern": "tests/test_e2e_self_improvement_integration.py::TestSelfImprovementSafetyMechanisms::test_code_validation_prevents_syntax_errors",
            "description": "Code Validation Test"
        }
    ]
    
    # Run tests
    results = []
    total_start_time = time.time()
    
    for suite in test_suites:
        success = run_test_suite(suite["pattern"], suite["description"])
        results.append((suite["description"], success))
    
    total_duration = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Duration: {total_duration:.2f} seconds")
    print()
    
    passed = 0
    failed = 0
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {description}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All integration tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)