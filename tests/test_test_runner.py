# tests/test_test_runner.py
"""
Unit tests for the automated test execution framework
"""

import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.test_runner import (
    TestRunner, TestEnvironment, TestResult, TestSuiteResult,
    TestStatus, TestType, get_test_runner
)


class TestTestResult(unittest.TestCase):
    """Test TestResult data class"""
    
    def test_test_result_creation(self):
        """Test creating a test result"""
        result = TestResult(
            test_name="test_example",
            test_type=TestType.UNIT,
            status=TestStatus.PASSED,
            duration=1.5,
            message="Test passed successfully"
        )
        
        self.assertEqual(result.test_name, "test_example")
        self.assertEqual(result.test_type, TestType.UNIT)
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertEqual(result.duration, 1.5)
        self.assertIsNotNone(result.timestamp)
    
    def test_test_result_to_dict(self):
        """Test converting test result to dictionary"""
        result = TestResult(
            test_name="test_example",
            test_type=TestType.UNIT,
            status=TestStatus.PASSED,
            duration=1.5
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["test_name"], "test_example")
        self.assertEqual(result_dict["test_type"], "unit")
        self.assertEqual(result_dict["status"], "passed")
        self.assertIn("timestamp", result_dict)


class TestTestSuiteResult(unittest.TestCase):
    """Test TestSuiteResult data class"""
    
    def test_test_suite_result_creation(self):
        """Test creating a test suite result"""
        test_results = [
            TestResult("test1", TestType.UNIT, TestStatus.PASSED, 1.0),
            TestResult("test2", TestType.UNIT, TestStatus.FAILED, 2.0)
        ]
        
        suite_result = TestSuiteResult(
            suite_name="unit_tests",
            total_tests=2,
            passed=1,
            failed=1,
            errors=0,
            skipped=0,
            duration=3.0,
            coverage_percentage=85.0,
            test_results=test_results
        )
        
        self.assertEqual(suite_result.suite_name, "unit_tests")
        self.assertEqual(suite_result.total_tests, 2)
        self.assertEqual(suite_result.success_rate, 50.0)
        self.assertEqual(len(suite_result.test_results), 2)
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        # All passed
        suite_result = TestSuiteResult(
            "test", 5, 5, 0, 0, 0, 10.0, 90.0, []
        )
        self.assertEqual(suite_result.success_rate, 100.0)
        
        # Half passed
        suite_result = TestSuiteResult(
            "test", 4, 2, 2, 0, 0, 10.0, 90.0, []
        )
        self.assertEqual(suite_result.success_rate, 50.0)
        
        # No tests
        suite_result = TestSuiteResult(
            "test", 0, 0, 0, 0, 0, 0.0, 0.0, []
        )
        self.assertEqual(suite_result.success_rate, 0.0)


class TestTestEnvironment(unittest.TestCase):
    """Test TestEnvironment class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_base = tempfile.mkdtemp()
        self.base_path = Path(self.temp_base)
        
        # Create mock project structure
        (self.base_path / "app").mkdir()
        (self.base_path / "app" / "main.py").write_text("# Main app")
        (self.base_path / "tests").mkdir()
        (self.base_path / "tests" / "test_main.py").write_text("# Test file")
        (self.base_path / "requirements.txt").write_text("fastapi==0.68.0")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_base, ignore_errors=True)
    
    def test_test_environment_context_manager(self):
        """Test test environment as context manager"""
        original_cwd = os.getcwd()
        
        with TestEnvironment(str(self.base_path)) as env:
            # Should be in temp directory
            self.assertNotEqual(os.getcwd(), original_cwd)
            self.assertTrue(env.temp_dir.exists())
            
            # Should have copied files
            self.assertTrue((env.temp_dir / "app" / "main.py").exists())
            self.assertTrue((env.temp_dir / "tests" / "test_main.py").exists())
            
            # Should have set environment variables
            self.assertEqual(os.environ.get("TESTING"), "true")
        
        # Should be restored after context
        self.assertEqual(os.getcwd(), original_cwd)
        self.assertIsNone(os.environ.get("TESTING"))
    
    def test_test_environment_setup_cleanup(self):
        """Test manual setup and cleanup"""
        env = TestEnvironment(str(self.base_path))
        original_cwd = os.getcwd()
        
        # Setup
        env.setup()
        self.assertNotEqual(os.getcwd(), original_cwd)
        self.assertTrue(env.temp_dir.exists())
        self.assertEqual(os.environ.get("TESTING"), "true")
        
        # Cleanup
        env.cleanup()
        self.assertEqual(os.getcwd(), original_cwd)
        self.assertIsNone(os.environ.get("TESTING"))


class TestTestRunner(unittest.TestCase):
    """Test TestRunner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create mock project structure
        (self.project_root / "app").mkdir()
        (self.project_root / "tests").mkdir()
        
        # Create sample test files
        (self.project_root / "tests" / "test_unit.py").write_text("""
import unittest

class TestExample(unittest.TestCase):
    def test_pass(self):
        self.assertTrue(True)
    
    def test_fail(self):
        self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
""")
        
        (self.project_root / "tests" / "test_integration_example.py").write_text("""
import unittest

class TestIntegration(unittest.TestCase):
    def test_integration(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
""")
        
        (self.project_root / "tests.py").write_text("""
import unittest

class TestMain(unittest.TestCase):
    def test_main(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
""")
        
        self.runner = TestRunner(str(self.project_root))
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_test_runner_initialization(self):
        """Test test runner initialization"""
        self.assertEqual(self.runner.project_root, self.project_root)
        self.assertTrue(self.runner.results_dir.exists())
        self.assertIsInstance(self.runner.test_patterns, dict)
        self.assertIn(TestType.UNIT, self.runner.test_patterns)
    
    def test_discover_tests_all(self):
        """Test discovering all tests"""
        test_files = self.runner.discover_tests()
        
        self.assertGreater(len(test_files), 0)
        # Should find both unit and integration tests
        test_names = [Path(f).name for f in test_files]
        self.assertIn("test_unit.py", test_names)
        self.assertIn("test_integration_example.py", test_names)
        self.assertIn("tests.py", test_names)
    
    def test_discover_tests_by_type(self):
        """Test discovering tests by type"""
        # Unit tests
        unit_tests = self.runner.discover_tests(TestType.UNIT)
        unit_names = [Path(f).name for f in unit_tests]
        self.assertIn("test_unit.py", unit_names)
        self.assertIn("tests.py", unit_names)  # Main test file is considered unit
        
        # Integration tests
        integration_tests = self.runner.discover_tests(TestType.INTEGRATION)
        integration_names = [Path(f).name for f in integration_tests]
        self.assertIn("test_integration_example.py", integration_names)
        self.assertNotIn("test_unit.py", integration_names)
    
    @patch('app.test_runner.subprocess.run')
    def test_run_unittest_file_success(self, mock_run):
        """Test running unittest file successfully"""
        # Mock successful test execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "test_pass ... ok\ntest_fail ... FAIL\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = self.runner._run_unittest_file("test_example.py", TestType.UNIT)
        
        self.assertEqual(result.test_name, "test_example")
        self.assertEqual(result.test_type, TestType.UNIT)
        self.assertEqual(result.status, TestStatus.PASSED)
        self.assertGreater(result.duration, 0)
    
    @patch('app.test_runner.subprocess.run')
    def test_run_unittest_file_failure(self, mock_run):
        """Test running unittest file with failures"""
        # Mock failed test execution
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "test_pass ... ok\ntest_fail ... FAIL\n"
        mock_result.stderr = "FAILED (failures=1)"
        mock_run.return_value = mock_result
        
        result = self.runner._run_unittest_file("test_example.py", TestType.UNIT)
        
        self.assertEqual(result.status, TestStatus.FAILED)
        self.assertIn("Some tests failed", result.message)
    
    @patch('app.test_runner.subprocess.run')
    def test_run_unittest_file_timeout(self, mock_run):
        """Test running unittest file with timeout"""
        # Mock timeout
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired("cmd", 300)
        
        result = self.runner._run_unittest_file("test_example.py", TestType.UNIT)
        
        self.assertEqual(result.status, TestStatus.ERROR)
        self.assertIn("timed out", result.message)
    
    @patch('app.test_runner.TestRunner._execute_test_suite')
    def test_run_test_suite_isolated(self, mock_execute):
        """Test running test suite in isolated environment"""
        # Mock test suite execution
        mock_result = TestSuiteResult(
            "test_suite", 2, 1, 1, 0, 0, 5.0, 80.0, []
        )
        mock_execute.return_value = mock_result
        
        result = self.runner.run_test_suite(TestType.UNIT, isolated=True)
        
        self.assertEqual(result.suite_name, "test_suite")
        mock_execute.assert_called_once()
    
    def test_run_test_suite_no_tests(self):
        """Test running test suite with no tests found"""
        # Create runner with empty directory
        empty_dir = tempfile.mkdtemp()
        empty_runner = TestRunner(empty_dir)
        
        try:
            result = empty_runner.run_test_suite(TestType.PERFORMANCE)
            
            self.assertEqual(result.total_tests, 0)
            self.assertEqual(result.suite_name, "performance_tests")
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)
    
    def test_performance_baseline_operations(self):
        """Test performance baseline save/load operations"""
        # Create test suite result with performance metrics
        test_results = [
            TestResult(
                "perf_test",
                TestType.PERFORMANCE,
                TestStatus.PASSED,
                2.0,
                performance_metrics={"response_time": 1.5, "memory_usage": 100.0}
            )
        ]
        
        suite_result = TestSuiteResult(
            "perf_tests", 1, 1, 0, 0, 0, 2.0, 90.0, test_results
        )
        
        # Update baseline
        self.runner.update_performance_baseline(suite_result)
        
        # Check baseline was saved
        self.assertTrue(self.runner.baseline_file.exists())
        
        # Load baseline and verify
        baseline = self.runner._load_baseline()
        self.assertIn("metrics", baseline)
        self.assertIn("perf_test", baseline["metrics"])
        self.assertEqual(baseline["metrics"]["perf_test"]["response_time"], 1.5)
    
    def test_performance_comparison(self):
        """Test performance comparison between versions"""
        # Set up baseline
        baseline_data = {
            "version": "v1.0",
            "metrics": {
                "test_performance": {
                    "response_time": 2.0,
                    "memory_usage": 100.0
                }
            }
        }
        self.runner.baseline_data = baseline_data
        
        # Mock current performance test results
        with patch.object(self.runner, 'run_test_suite') as mock_run:
            current_results = TestSuiteResult(
                "perf_tests", 1, 1, 0, 0, 0, 2.0, 90.0,
                [TestResult(
                    "test_performance",
                    TestType.PERFORMANCE,
                    TestStatus.PASSED,
                    1.8,
                    performance_metrics={"response_time": 1.5, "memory_usage": 120.0}
                )]
            )
            mock_run.return_value = current_results
            
            comparison = self.runner.run_performance_comparison("v1.0", "v2.0")
        
        self.assertEqual(comparison["baseline_version"], "v1.0")
        self.assertEqual(comparison["current_version"], "v2.0")
        self.assertIn("metrics", comparison)
        self.assertIn("improvements", comparison)
        self.assertIn("regressions", comparison)
        
        # Should detect response time improvement and memory regression
        self.assertGreater(len(comparison["improvements"]), 0)
        self.assertGreater(len(comparison["regressions"]), 0)
    
    def test_validate_code_changes(self):
        """Test code change validation"""
        changed_files = ["app/main.py", "app/service.py"]
        
        with patch.object(self.runner, 'run_test_suite') as mock_run:
            # Mock successful test results
            mock_run.return_value = TestSuiteResult(
                "unit_tests", 5, 5, 0, 0, 0, 10.0, 85.0, []
            )
            
            validation = self.runner.validate_code_changes(
                changed_files, [TestType.UNIT]
            )
        
        self.assertEqual(validation["changed_files"], changed_files)
        self.assertEqual(validation["overall_status"], "passed")
        self.assertIn("results", validation)
        self.assertIn("recommendations", validation)
        self.assertTrue(any("safe" in rec for rec in validation["recommendations"]))
    
    def test_validate_code_changes_with_failures(self):
        """Test code change validation with test failures"""
        changed_files = ["app/main.py"]
        
        with patch.object(self.runner, 'run_test_suite') as mock_run:
            # Mock failed test results
            mock_run.return_value = TestSuiteResult(
                "unit_tests", 5, 3, 2, 0, 0, 10.0, 85.0, []
            )
            
            validation = self.runner.validate_code_changes(
                changed_files, [TestType.UNIT]
            )
        
        self.assertEqual(validation["overall_status"], "failed")
        self.assertTrue(any("Fix" in rec for rec in validation["recommendations"]))
    
    def test_get_test_history(self):
        """Test getting test execution history"""
        # Create mock result files
        results_dir = self.runner.results_dir
        
        # Create test result files with different timestamps
        timestamps = [
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow() - timedelta(days=3),
            datetime.utcnow() - timedelta(days=10)  # This should be excluded
        ]
        
        for i, timestamp in enumerate(timestamps):
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            result_file = results_dir / f"test_results_{timestamp_str}.json"
            
            result_data = {
                "suite_name": f"test_suite_{i}",
                "timestamp": timestamp.isoformat(),
                "total_tests": 5,
                "passed": 4,
                "failed": 1
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f)
        
        # Get history for last 7 days
        history = self.runner.get_test_history(days=7)
        
        # Should only get 2 results (excluding the 10-day old one)
        self.assertEqual(len(history), 2)
        
        # Should be sorted by timestamp (newest first)
        self.assertGreater(history[0]["timestamp"], history[1]["timestamp"])
    
    def test_save_and_load_test_results(self):
        """Test saving and loading test results"""
        # Create test suite result
        test_results = [
            TestResult("test1", TestType.UNIT, TestStatus.PASSED, 1.0),
            TestResult("test2", TestType.UNIT, TestStatus.FAILED, 2.0)
        ]
        
        suite_result = TestSuiteResult(
            "unit_tests", 2, 1, 1, 0, 0, 3.0, 85.0, test_results
        )
        
        # Save results
        self.runner._save_test_results(suite_result)
        
        # Check that file was created
        result_files = list(self.runner.results_dir.glob("test_results_*.json"))
        self.assertGreater(len(result_files), 0)
        
        # Load and verify content
        with open(result_files[0], 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data["suite_name"], "unit_tests")
        self.assertEqual(loaded_data["total_tests"], 2)
        self.assertEqual(len(loaded_data["test_results"]), 2)


class TestGetTestRunner(unittest.TestCase):
    """Test get_test_runner convenience function"""
    
    def test_get_test_runner_default(self):
        """Test getting test runner with default project root"""
        runner = get_test_runner()
        
        self.assertIsInstance(runner, TestRunner)
        self.assertEqual(str(runner.project_root), os.getcwd())
    
    def test_get_test_runner_custom_root(self):
        """Test getting test runner with custom project root"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            runner = get_test_runner(temp_dir)
            
            self.assertIsInstance(runner, TestRunner)
            self.assertEqual(str(runner.project_root), temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()