#!/usr/bin/env python3
"""
Integration Test Runner for MCP and Git Workflow Systems

This script runs comprehensive integration tests for the MCP and Git workflow
integration systems, providing detailed reporting and validation.
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any


def run_test_suite(test_file: str, verbose: bool = False, capture_output: bool = True) -> Dict[str, Any]:
    """Run a specific test suite and return results"""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", test_file, "-v"]
    if verbose:
        cmd.extend(["--tb=short", "-s"])
    else:
        cmd.append("--tb=line")
    
    # Add coverage if available
    try:
        import pytest_cov
        cmd.extend(["--cov=app", "--cov-report=term-missing"])
    except ImportError:
        pass
    
    # Run tests
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            output = result.stdout
            error_output = result.stderr
            return_code = result.returncode
        else:
            result = subprocess.run(cmd, timeout=300)
            output = ""
            error_output = ""
            return_code = result.returncode
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print results
        if not capture_output or verbose:
            if output:
                print(output)
            if error_output:
                print("STDERR:", error_output)
        
        # Parse results
        if return_code == 0:
            print(f"✅ {test_file} PASSED ({duration:.2f}s)")
            status = "PASSED"
        else:
            print(f"❌ {test_file} FAILED ({duration:.2f}s)")
            status = "FAILED"
            if capture_output and not verbose:
                print("Output:", output[-500:] if len(output) > 500 else output)
                if error_output:
                    print("Errors:", error_output[-500:] if len(error_output) > 500 else error_output)
        
        return {
            "test_file": test_file,
            "status": status,
            "duration": duration,
            "return_code": return_code,
            "output": output,
            "error_output": error_output
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file} TIMEOUT (>300s)")
        return {
            "test_file": test_file,
            "status": "TIMEOUT",
            "duration": 300.0,
            "return_code": -1,
            "output": "",
            "error_output": "Test timed out after 300 seconds"
        }
    except Exception as e:
        print(f"💥 {test_file} ERROR: {e}")
        return {
            "test_file": test_file,
            "status": "ERROR",
            "duration": 0.0,
            "return_code": -1,
            "output": "",
            "error_output": str(e)
        }


def check_dependencies() -> List[str]:
    """Check for required dependencies"""
    missing = []
    
    required_packages = [
        "pytest",
        "asyncio",
        "redis",
        "subprocess",
        "pathlib"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run MCP and Git workflow integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-capture", action="store_true", help="Don't capture output")
    parser.add_argument("--mcp-only", action="store_true", help="Run only MCP tests")
    parser.add_argument("--git-only", action="store_true", help="Run only Git workflow tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--test-file", help="Run specific test file")
    
    args = parser.parse_args()
    
    print("🚀 MCP and Git Workflow Integration Test Runner")
    print("=" * 60)
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing packages and try again.")
        return 1
    
    # Define test suites
    test_suites = []
    
    if args.test_file:
        # Run specific test file
        test_suites = [args.test_file]
    else:
        # Define available test suites
        if not args.git_only:
            test_suites.append("tests/test_mcp_integration_comprehensive.py")
        
        if not args.mcp_only:
            test_suites.append("tests/test_git_workflow_integration_comprehensive.py")
        
        # Add other related tests if not quick mode
        if not args.quick:
            additional_tests = [
                "tests/test_mcp_server_infrastructure.py",
                "tests/test_agent_mcp_integration.py",
                "tests/test_context_synchronization.py",
                "tests/test_git_workflow_automation.py",
                "tests/test_task_git_integration.py"
            ]
            
            # Only add tests that exist
            for test_file in additional_tests:
                if Path(test_file).exists():
                    test_suites.append(test_file)
    
    if not test_suites:
        print("❌ No test suites found to run")
        return 1
    
    print(f"📋 Running {len(test_suites)} test suite(s):")
    for suite in test_suites:
        print(f"  - {suite}")
    print()
    
    # Run test suites
    results = []
    total_start_time = time.time()
    
    for test_suite in test_suites:
        if not Path(test_suite).exists():
            print(f"⚠️  Test file not found: {test_suite}")
            results.append({
                "test_file": test_suite,
                "status": "NOT_FOUND",
                "duration": 0.0,
                "return_code": -1,
                "output": "",
                "error_output": f"Test file {test_suite} not found"
            })
            continue
        
        result = run_test_suite(
            test_suite, 
            verbose=args.verbose, 
            capture_output=not args.no_capture
        )
        results.append(result)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    errors = sum(1 for r in results if r["status"] in ["ERROR", "TIMEOUT", "NOT_FOUND"])
    
    print(f"Total test suites: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"💥 Errors: {errors}")
    print(f"⏱️  Total time: {total_duration:.2f}s")
    print()
    
    # Detailed results
    for result in results:
        status_emoji = {
            "PASSED": "✅",
            "FAILED": "❌", 
            "ERROR": "💥",
            "TIMEOUT": "⏰",
            "NOT_FOUND": "⚠️"
        }.get(result["status"], "❓")
        
        print(f"{status_emoji} {result['test_file']}: {result['status']} ({result['duration']:.2f}s)")
        
        if result["status"] in ["FAILED", "ERROR", "TIMEOUT"] and result["error_output"]:
            print(f"   Error: {result['error_output'][:200]}...")
    
    print()
    
    # Return appropriate exit code
    if failed > 0 or errors > 0:
        print("❌ Some tests failed or had errors")
        return 1
    else:
        print("🎉 All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())