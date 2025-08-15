# app/test_runner.py
"""
Automated test execution and validation system for self-improvement framework
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import unittest
import pytest
import coverage


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class TestType(Enum):
    """Types of tests that can be executed"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    REGRESSION = "regression"
    SAFETY = "safety"


@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float
    message: str = ""
    error_details: str = ""
    output: str = ""
    coverage_data: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['status'] = self.status.value
        result['test_type'] = self.test_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class TestSuiteResult:
    """Results from running a test suite"""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration: float
    coverage_percentage: float
    test_results: List[TestResult]
    performance_comparison: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['test_results'] = [tr.to_dict() for tr in self.test_results]
        result['success_rate'] = self.success_rate
        return result


class TestEnvironment:
    """Manages isolated test environments"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.temp_dir: Optional[Path] = None
        self.original_cwd: Optional[str] = None
        self.env_vars: Dict[str, str] = {}
    
    def __enter__(self):
        """Enter test environment context"""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit test environment context"""
        self.cleanup()
    
    def setup(self):
        """Set up isolated test environment"""
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_env_"))
        
        # Copy source code to temp directory
        self._copy_source_code()
        
        # Store original working directory
        self.original_cwd = os.getcwd()
        
        # Change to test environment
        os.chdir(self.temp_dir)
        
        # Set environment variables for testing
        self.env_vars = {
            'TESTING': 'true',
            'USE_REAL_LLM': 'false',
            'REDIS_URL': 'redis://localhost:6379/15',  # Use test database
            'LOG_LEVEL': 'WARNING'
        }
        
        for key, value in self.env_vars.items():
            os.environ[key] = value
    
    def cleanup(self):
        """Clean up test environment"""
        # Restore original working directory
        if self.original_cwd:
            os.chdir(self.original_cwd)
        
        # Remove environment variables
        for key in self.env_vars:
            os.environ.pop(key, None)
        
        # Remove temporary directory
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _copy_source_code(self):
        """Copy source code to test environment"""
        # Copy main application code
        app_src = self.base_path / "app"
        app_dst = self.temp_dir / "app"
        if app_src.exists():
            shutil.copytree(app_src, app_dst)
        
        # Copy tests
        tests_src = self.base_path / "tests"
        tests_dst = self.temp_dir / "tests"
        if tests_src.exists():
            shutil.copytree(tests_src, tests_dst)
        
        # Copy requirements and other necessary files
        for file_name in ["requirements.txt", "tests.py"]:
            src_file = self.base_path / file_name
            if src_file.exists():
                shutil.copy2(src_file, self.temp_dir / file_name)


class TestRunner:
    """Automated test execution and validation system"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Test discovery patterns
        self.test_patterns = {
            TestType.UNIT: ["test_*.py", "*_test.py"],
            TestType.INTEGRATION: ["test_integration_*.py", "*_integration_test.py"],
            TestType.PERFORMANCE: ["test_performance_*.py", "*_performance_test.py"],
            TestType.REGRESSION: ["test_regression_*.py", "*_regression_test.py"],
            TestType.SAFETY: ["test_safety_*.py", "*_safety_test.py"]
        }
        
        # Performance baseline storage
        self.baseline_file = self.results_dir / "performance_baseline.json"
        self.baseline_data: Dict[str, Any] = self._load_baseline()
    
    def discover_tests(self, test_type: Optional[TestType] = None) -> List[str]:
        """Discover test files based on type"""
        test_files = []
        
        # Determine which patterns to use
        if test_type:
            patterns = self.test_patterns.get(test_type, [])
        else:
            patterns = []
            for type_patterns in self.test_patterns.values():
                patterns.extend(type_patterns)
        
        # Search for test files
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            for pattern in patterns:
                test_files.extend(str(f) for f in tests_dir.glob(pattern))
        
        # Also check for main tests.py file
        main_test_file = self.project_root / "tests.py"
        if main_test_file.exists() and (not test_type or test_type == TestType.UNIT):
            test_files.append(str(main_test_file))
        
        return sorted(list(set(test_files)))
    
    def run_test_suite(
        self,
        test_type: Optional[TestType] = None,
        test_files: Optional[List[str]] = None,
        isolated: bool = True,
        coverage_enabled: bool = True
    ) -> TestSuiteResult:
        """Run a complete test suite"""
        start_time = time.time()
        
        # Discover tests if not provided
        if test_files is None:
            test_files = self.discover_tests(test_type)
        
        if not test_files:
            return TestSuiteResult(
                suite_name=f"{test_type.value if test_type else 'all'}_tests",
                total_tests=0,
                passed=0,
                failed=0,
                errors=0,
                skipped=0,
                duration=0.0,
                coverage_percentage=0.0,
                test_results=[]
            )
        
        # Run tests in isolated environment if requested
        if isolated:
            with TestEnvironment(self.project_root) as env:
                return self._execute_test_suite(test_files, test_type, coverage_enabled)
        else:
            return self._execute_test_suite(test_files, test_type, coverage_enabled)
    
    def _execute_test_suite(
        self,
        test_files: List[str],
        test_type: Optional[TestType],
        coverage_enabled: bool
    ) -> TestSuiteResult:
        """Execute test suite in current environment"""
        test_results = []
        total_tests = 0
        passed = 0
        failed = 0
        errors = 0
        skipped = 0
        
        # Set up coverage if enabled
        cov = None
        if coverage_enabled:
            cov = coverage.Coverage(source=['app'])
            cov.start()
        
        start_time = time.time()
        
        try:
            for test_file in test_files:
                result = self._run_single_test_file(test_file, test_type or TestType.UNIT)
                test_results.append(result)
                
                # Update counters
                total_tests += 1
                if result.status == TestStatus.PASSED:
                    passed += 1
                elif result.status == TestStatus.FAILED:
                    failed += 1
                elif result.status == TestStatus.ERROR:
                    errors += 1
                elif result.status == TestStatus.SKIPPED:
                    skipped += 1
        
        finally:
            # Stop coverage collection
            coverage_percentage = 0.0
            if cov:
                cov.stop()
                coverage_percentage = cov.report()
        
        duration = time.time() - start_time
        
        # Create suite result
        suite_result = TestSuiteResult(
            suite_name=f"{test_type.value if test_type else 'all'}_tests",
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration=duration,
            coverage_percentage=coverage_percentage,
            test_results=test_results
        )
        
        # Add performance comparison if this is a performance test
        if test_type == TestType.PERFORMANCE:
            suite_result.performance_comparison = self._compare_with_baseline(suite_result)
        
        # Save results
        self._save_test_results(suite_result)
        
        return suite_result
    
    def _run_single_test_file(self, test_file: str, test_type: TestType) -> TestResult:
        """Run a single test file"""
        start_time = time.time()
        test_name = Path(test_file).stem
        
        try:
            # Determine test runner based on file
            if test_file.endswith('.py') and 'pytest' in test_file or any(
                'pytest' in line for line in Path(test_file).read_text().splitlines()[:10]
            ):
                return self._run_pytest_file(test_file, test_type)
            else:
                return self._run_unittest_file(test_file, test_type)
        
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=duration,
                message=f"Failed to execute test: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    def _run_unittest_file(self, test_file: str, test_type: TestType) -> TestResult:
        """Run unittest-based test file"""
        start_time = time.time()
        test_name = Path(test_file).stem
        
        try:
            # Run unittest via subprocess to capture output
            cmd = [sys.executable, "-m", "unittest", test_file, "-v"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                status = TestStatus.PASSED
                message = "All tests passed"
            else:
                status = TestStatus.FAILED
                message = "Some tests failed"
            
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=status,
                duration=duration,
                message=message,
                output=result.stdout,
                error_details=result.stderr if result.stderr else ""
            )
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=duration,
                message="Test execution timed out",
                error_details="Test execution exceeded 5 minute timeout"
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=duration,
                message=f"Test execution failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    def _run_pytest_file(self, test_file: str, test_type: TestType) -> TestResult:
        """Run pytest-based test file"""
        start_time = time.time()
        test_name = Path(test_file).stem
        
        try:
            # Run pytest via subprocess
            cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            # Parse pytest results
            if result.returncode == 0:
                status = TestStatus.PASSED
                message = "All tests passed"
            elif "FAILED" in result.stdout:
                status = TestStatus.FAILED
                message = "Some tests failed"
            else:
                status = TestStatus.ERROR
                message = "Test execution error"
            
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=status,
                duration=duration,
                message=message,
                output=result.stdout,
                error_details=result.stderr if result.stderr else ""
            )
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=duration,
                message="Test execution timed out",
                error_details="Test execution exceeded 5 minute timeout"
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=duration,
                message=f"Test execution failed: {str(e)}",
                error_details=traceback.format_exc()
            )
    
    def run_performance_comparison(
        self,
        baseline_version: str,
        current_version: str
    ) -> Dict[str, Any]:
        """Compare performance between two code versions"""
        comparison_results = {
            "baseline_version": baseline_version,
            "current_version": current_version,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
            "improvements": [],
            "regressions": [],
            "summary": {}
        }
        
        # Run performance tests for current version
        current_results = self.run_test_suite(TestType.PERFORMANCE, isolated=True)
        
        # Load baseline performance data
        baseline_metrics = self.baseline_data.get("metrics", {})
        
        # Compare metrics
        for test_result in current_results.test_results:
            if test_result.performance_metrics:
                test_name = test_result.test_name
                current_metrics = test_result.performance_metrics
                baseline_test_metrics = baseline_metrics.get(test_name, {})
                
                comparison_results["metrics"][test_name] = {
                    "current": current_metrics,
                    "baseline": baseline_test_metrics,
                    "changes": {}
                }
                
                # Calculate changes
                for metric_name, current_value in current_metrics.items():
                    baseline_value = baseline_test_metrics.get(metric_name)
                    if baseline_value is not None:
                        change_percent = ((current_value - baseline_value) / baseline_value) * 100
                        comparison_results["metrics"][test_name]["changes"][metric_name] = {
                            "absolute": current_value - baseline_value,
                            "percentage": change_percent
                        }
                        
                        # Categorize as improvement or regression
                        if abs(change_percent) > 5:  # Significant change threshold
                            change_info = {
                                "test": test_name,
                                "metric": metric_name,
                                "change_percent": change_percent,
                                "current_value": current_value,
                                "baseline_value": baseline_value
                            }
                            
                            if change_percent < 0:  # Lower is better for most metrics
                                comparison_results["improvements"].append(change_info)
                            else:
                                comparison_results["regressions"].append(change_info)
        
        # Generate summary
        comparison_results["summary"] = {
            "total_improvements": len(comparison_results["improvements"]),
            "total_regressions": len(comparison_results["regressions"]),
            "net_performance_change": len(comparison_results["improvements"]) - len(comparison_results["regressions"])
        }
        
        return comparison_results
    
    def update_performance_baseline(self, test_results: TestSuiteResult):
        """Update performance baseline with new results"""
        if test_results.test_results:
            baseline_metrics = {}
            
            for test_result in test_results.test_results:
                if test_result.performance_metrics:
                    baseline_metrics[test_result.test_name] = test_result.performance_metrics
            
            self.baseline_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "current",
                "metrics": baseline_metrics
            }
            
            self._save_baseline()
    
    def _compare_with_baseline(self, suite_result: TestSuiteResult) -> Dict[str, Any]:
        """Compare current results with baseline"""
        if not self.baseline_data.get("metrics"):
            return {"message": "No baseline data available"}
        
        return self.run_performance_comparison(
            baseline_version=self.baseline_data.get("version", "unknown"),
            current_version="current"
        )
    
    def _load_baseline(self) -> Dict[str, Any]:
        """Load performance baseline data"""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_baseline(self):
        """Save performance baseline data"""
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baseline_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save baseline data: {e}")
    
    def _save_test_results(self, suite_result: TestSuiteResult):
        """Save test results to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"test_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(suite_result.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save test results: {e}")
    
    def get_test_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get test execution history"""
        history = []
        cutoff_time = datetime.utcnow().timestamp() - (days * 24 * 3600)
        
        for results_file in self.results_dir.glob("test_results_*.json"):
            try:
                # Extract timestamp from filename
                timestamp_str = results_file.stem.split("_", 2)[-1]
                file_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").timestamp()
                
                if file_time >= cutoff_time:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        history.append(data)
            except Exception:
                continue
        
        return sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def validate_code_changes(
        self,
        changed_files: List[str],
        test_types: Optional[List[TestType]] = None
    ) -> Dict[str, Any]:
        """Validate code changes by running relevant tests"""
        if test_types is None:
            test_types = [TestType.UNIT, TestType.INTEGRATION]
        
        validation_results = {
            "changed_files": changed_files,
            "test_types": [t.value for t in test_types],
            "timestamp": datetime.utcnow().isoformat(),
            "results": {},
            "overall_status": "unknown",
            "recommendations": []
        }
        
        all_passed = True
        
        for test_type in test_types:
            # Run tests for this type
            suite_result = self.run_test_suite(test_type, isolated=True)
            validation_results["results"][test_type.value] = suite_result.to_dict()
            
            if suite_result.failed > 0 or suite_result.errors > 0:
                all_passed = False
        
        # Determine overall status
        if all_passed:
            validation_results["overall_status"] = "passed"
            validation_results["recommendations"].append(
                "All tests passed. Code changes appear to be safe."
            )
        else:
            validation_results["overall_status"] = "failed"
            validation_results["recommendations"].append(
                "Some tests failed. Review failures before proceeding with changes."
            )
            
            # Add specific recommendations based on failures
            for test_type in test_types:
                suite_result_dict = validation_results["results"][test_type.value]
                if suite_result_dict["failed"] > 0:
                    validation_results["recommendations"].append(
                        f"Fix {suite_result_dict['failed']} failing {test_type.value} tests"
                    )
                if suite_result_dict["errors"] > 0:
                    validation_results["recommendations"].append(
                        f"Resolve {suite_result_dict['errors']} {test_type.value} test errors"
                    )
        
        return validation_results


# Convenience function for getting test runner instance
def get_test_runner(project_root: Optional[str] = None) -> TestRunner:
    """Get TestRunner instance"""
    if project_root is None:
        project_root = os.getcwd()
    return TestRunner(project_root)