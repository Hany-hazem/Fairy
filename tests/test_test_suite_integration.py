# tests/test_test_suite_integration.py
"""
Unit tests for test suite integration
"""

import unittest
import tempfile
import shutil
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.test_suite_integration import (
    TestSuiteIntegration, TestHistoryManager, CoverageAnalyzer,
    CoverageReport, RegressionTestResult, TestCoverageAnalysis,
    get_test_suite_integration
)
from app.test_runner import TestResult, TestSuiteResult, TestType, TestStatus


class TestCoverageReport(unittest.TestCase):
    """Test CoverageReport data class"""
    
    def test_coverage_report_creation(self):
        """Test creating a coverage report"""
        report = CoverageReport(
            total_lines=100,
            covered_lines=85,
            missing_lines=[10, 15, 20],
            coverage_percentage=85.0,
            file_path="app/main.py"
        )
        
        self.assertEqual(report.total_lines, 100)
        self.assertEqual(report.covered_lines, 85)
        self.assertEqual(report.coverage_percentage, 85.0)
        self.assertEqual(report.file_path, "app/main.py")
        self.assertIsNotNone(report.timestamp)
    
    def test_coverage_report_to_dict(self):
        """Test converting coverage report to dictionary"""
        report = CoverageReport(
            total_lines=100,
            covered_lines=85,
            missing_lines=[10, 15, 20],
            coverage_percentage=85.0,
            file_path="app/main.py"
        )
        
        result_dict = report.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["total_lines"], 100)
        self.assertEqual(result_dict["file_path"], "app/main.py")
        self.assertIn("timestamp", result_dict)


class TestRegressionTestResult(unittest.TestCase):
    """Test RegressionTestResult data class"""
    
    def test_regression_test_result_creation(self):
        """Test creating a regression test result"""
        result = RegressionTestResult(
            test_name="test_example",
            baseline_status=TestStatus.PASSED,
            current_status=TestStatus.FAILED,
            is_regression=True,
            baseline_duration=1.0,
            current_duration=1.5,
            performance_change_percent=50.0
        )
        
        self.assertEqual(result.test_name, "test_example")
        self.assertEqual(result.baseline_status, TestStatus.PASSED)
        self.assertEqual(result.current_status, TestStatus.FAILED)
        self.assertTrue(result.is_regression)
        self.assertEqual(result.performance_change_percent, 50.0)
    
    def test_regression_test_result_to_dict(self):
        """Test converting regression test result to dictionary"""
        result = RegressionTestResult(
            test_name="test_example",
            baseline_status=TestStatus.PASSED,
            current_status=TestStatus.FAILED,
            is_regression=True,
            baseline_duration=1.0,
            current_duration=1.5,
            performance_change_percent=50.0
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["test_name"], "test_example")
        self.assertEqual(result_dict["baseline_status"], "passed")
        self.assertEqual(result_dict["current_status"], "failed")
        self.assertTrue(result_dict["is_regression"])


class TestTestHistoryManager(unittest.TestCase):
    """Test TestHistoryManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_history.db"
        self.history_manager = TestHistoryManager(str(self.db_path))
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test database initialization"""
        self.assertTrue(self.db_path.exists())
        
        # Check tables exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            self.assertIn("test_runs", tables)
            self.assertIn("test_results", tables)
            self.assertIn("coverage_history", tables)
    
    def test_store_test_run(self):
        """Test storing test run results"""
        # Create test suite result
        test_results = [
            TestResult("test1", TestType.UNIT, TestStatus.PASSED, 1.0),
            TestResult("test2", TestType.UNIT, TestStatus.FAILED, 2.0)
        ]
        
        suite_result = TestSuiteResult(
            "unit_tests", 2, 1, 1, 0, 0, 3.0, 85.0, test_results
        )
        
        git_info = {"commit": "abc123", "branch": "main"}
        
        # Store test run
        run_id = self.history_manager.store_test_run(suite_result, git_info)
        
        self.assertIsInstance(run_id, int)
        self.assertGreater(run_id, 0)
        
        # Verify data was stored
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check test run
            cursor.execute("SELECT * FROM test_runs WHERE id = ?", (run_id,))
            run_data = cursor.fetchone()
            self.assertIsNotNone(run_data)
            self.assertEqual(run_data[1], "unit_tests")  # suite_name
            self.assertEqual(run_data[10], "abc123")  # git_commit
            
            # Check test results
            cursor.execute("SELECT * FROM test_results WHERE run_id = ?", (run_id,))
            results_data = cursor.fetchall()
            self.assertEqual(len(results_data), 2)
    
    def test_get_test_history(self):
        """Test getting test history"""
        # Store some test runs
        for i in range(3):
            test_results = [
                TestResult(f"test{i}", TestType.UNIT, TestStatus.PASSED, 1.0)
            ]
            
            suite_result = TestSuiteResult(
                "unit_tests", 1, 1, 0, 0, 0, 1.0, 90.0, test_results
            )
            suite_result.timestamp = datetime.utcnow() - timedelta(days=i)
            
            self.history_manager.store_test_run(suite_result)
        
        # Get history
        history = self.history_manager.get_test_history(days=7)
        
        self.assertEqual(len(history), 3)
        # Should be ordered by timestamp (newest first)
        self.assertGreater(history[0]["timestamp"], history[1]["timestamp"])
    
    def test_get_test_history_with_suite_filter(self):
        """Test getting test history filtered by suite name"""
        # Store runs for different suites
        for suite_name in ["unit_tests", "integration_tests"]:
            test_results = [TestResult("test1", TestType.UNIT, TestStatus.PASSED, 1.0)]
            suite_result = TestSuiteResult(
                suite_name, 1, 1, 0, 0, 0, 1.0, 90.0, test_results
            )
            self.history_manager.store_test_run(suite_result)
        
        # Get history for specific suite
        unit_history = self.history_manager.get_test_history(days=7, suite_name="unit_tests")
        
        self.assertEqual(len(unit_history), 1)
        self.assertEqual(unit_history[0]["suite_name"], "unit_tests")
    
    def test_get_regression_baseline(self):
        """Test getting regression baseline"""
        # Store a successful test run
        test_results = [
            TestResult("test1", TestType.UNIT, TestStatus.PASSED, 1.0),
            TestResult("test2", TestType.UNIT, TestStatus.PASSED, 1.5)
        ]
        
        suite_result = TestSuiteResult(
            "unit_tests", 2, 2, 0, 0, 0, 2.5, 90.0, test_results
        )
        
        run_id = self.history_manager.store_test_run(suite_result)
        
        # Get baseline
        baseline = self.history_manager.get_regression_baseline("unit_tests")
        
        self.assertIsNotNone(baseline)
        self.assertIn("run_info", baseline)
        self.assertIn("test_results", baseline)
        self.assertEqual(baseline["run_info"]["id"], run_id)
        self.assertEqual(len(baseline["test_results"]), 2)
    
    def test_get_regression_baseline_no_successful_runs(self):
        """Test getting regression baseline when no successful runs exist"""
        # Store a failed test run
        test_results = [TestResult("test1", TestType.UNIT, TestStatus.FAILED, 1.0)]
        suite_result = TestSuiteResult(
            "unit_tests", 1, 0, 1, 0, 0, 1.0, 90.0, test_results
        )
        
        self.history_manager.store_test_run(suite_result)
        
        # Get baseline
        baseline = self.history_manager.get_regression_baseline("unit_tests")
        
        self.assertIsNone(baseline)
    
    def test_analyze_test_trends(self):
        """Test analyzing test trends"""
        # Store test runs with varying success rates
        success_rates = [100, 80, 90, 70, 85]  # Declining trend
        
        for i, success_rate in enumerate(success_rates):
            passed = int(10 * success_rate / 100)
            failed = 10 - passed
            
            test_results = []
            for j in range(10):
                status = TestStatus.PASSED if j < passed else TestStatus.FAILED
                test_results.append(TestResult(f"test{j}", TestType.UNIT, status, 1.0))
            
            suite_result = TestSuiteResult(
                "unit_tests", 10, passed, failed, 0, 0, 10.0, 90.0, test_results
            )
            suite_result.timestamp = datetime.utcnow() - timedelta(days=i)
            
            self.history_manager.store_test_run(suite_result)
        
        # Analyze trends
        trends = self.history_manager.analyze_test_trends("unit_tests", days=30)
        
        self.assertIn("trends", trends)
        self.assertIn("success_rate", trends["trends"])
        self.assertEqual(trends["suite_name"], "unit_tests")
        self.assertEqual(trends["total_runs"], 5)
    
    def test_analyze_test_trends_insufficient_data(self):
        """Test analyzing trends with insufficient data"""
        trends = self.history_manager.analyze_test_trends("unit_tests", days=30)
        
        self.assertIn("message", trends)
        self.assertIn("Insufficient data", trends["message"])


class TestCoverageAnalyzer(unittest.TestCase):
    """Test CoverageAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create mock project structure
        (self.project_root / "app").mkdir()
        (self.project_root / "app" / "main.py").write_text("""
def hello_world():
    return "Hello, World!"

def unused_function():
    return "This is not covered"
""")
        
        (self.project_root / "tests").mkdir()
        (self.project_root / "tests" / "test_main.py").write_text("""
import unittest
import sys
sys.path.append('..')
from app.main import hello_world

class TestMain(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual(hello_world(), "Hello, World!")

if __name__ == '__main__':
    unittest.main()
""")
        
        self.analyzer = CoverageAnalyzer(str(self.project_root))
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('app.test_suite_integration.coverage.Coverage')
    def test_analyze_coverage(self, mock_coverage_class):
        """Test coverage analysis"""
        # Mock coverage object
        mock_cov = Mock()
        mock_coverage_class.return_value = mock_cov
        
        # Mock coverage data
        mock_data = Mock()
        mock_data.measured_files.return_value = [str(self.project_root / "app" / "main.py")]
        mock_cov.get_data.return_value = mock_data
        
        # Mock analysis results
        mock_cov.analysis2.return_value = (
            "app/main.py",  # filename
            [1, 2, 3],      # executed lines
            [],             # excluded lines
            [5, 6],         # missing lines
            "missing lines text"
        )
        
        test_files = [str(self.project_root / "tests" / "test_main.py")]
        
        analysis = self.analyzer.analyze_coverage(test_files)
        
        self.assertIsInstance(analysis, TestCoverageAnalysis)
        self.assertGreater(analysis.overall_coverage, 0)
        self.assertIn("app/main.py", analysis.coverage_by_file)
    
    def test_generate_coverage_recommendations(self):
        """Test coverage recommendation generation"""
        # Create coverage reports
        coverage_by_file = {
            "app/main.py": CoverageReport(100, 60, [40, 41, 42], 60.0, "app/main.py"),
            "app/service.py": CoverageReport(50, 45, [5], 90.0, "app/service.py")
        }
        
        recommendations = self.analyzer._generate_coverage_recommendations(
            coverage_by_file, 75.0, ["app/main.py"]
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should recommend improving low coverage files
        rec_text = " ".join(recommendations)
        self.assertIn("main.py", rec_text)


class TestTestSuiteIntegration(unittest.TestCase):
    """Test TestSuiteIntegration class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create mock project structure
        (self.project_root / "app").mkdir()
        (self.project_root / "tests").mkdir()
        (self.project_root / "test_results").mkdir()
        
        self.integration = TestSuiteIntegration(str(self.project_root))
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test test suite integration initialization"""
        self.assertEqual(self.integration.project_root, self.project_root)
        self.assertIsNotNone(self.integration.test_runner)
        self.assertIsNotNone(self.integration.coverage_analyzer)
        self.assertIsNotNone(self.integration.history_manager)
    
    @patch('app.test_suite_integration.TestSuiteIntegration._get_git_info')
    @patch('app.test_runner.TestRunner.run_test_suite')
    def test_run_comprehensive_test_suite(self, mock_run_suite, mock_git_info):
        """Test running comprehensive test suite"""
        # Mock git info
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        
        # Mock test results
        mock_suite_result = TestSuiteResult(
            "unit_tests", 5, 5, 0, 0, 0, 10.0, 85.0, []
        )
        mock_run_suite.return_value = mock_suite_result
        
        changed_files = ["app/main.py"]
        
        results = self.integration.run_comprehensive_test_suite(
            changed_files=changed_files,
            include_regression=False,  # Skip regression for simplicity
            include_coverage=False     # Skip coverage for simplicity
        )
        
        self.assertIn("timestamp", results)
        self.assertEqual(results["changed_files"], changed_files)
        self.assertIn("test_results", results)
        self.assertEqual(results["overall_status"], "passed")
        self.assertIn("recommendations", results)
    
    @patch('app.test_suite_integration.TestSuiteIntegration._get_git_info')
    @patch('app.test_runner.TestRunner.run_test_suite')
    def test_run_comprehensive_test_suite_with_failures(self, mock_run_suite, mock_git_info):
        """Test running comprehensive test suite with failures"""
        # Mock git info
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        
        # Mock test results with failures
        mock_suite_result = TestSuiteResult(
            "unit_tests", 5, 3, 2, 0, 0, 10.0, 85.0, []
        )
        mock_run_suite.return_value = mock_suite_result
        
        results = self.integration.run_comprehensive_test_suite(
            include_regression=False,
            include_coverage=False
        )
        
        self.assertEqual(results["overall_status"], "failed")
        self.assertTrue(any("failed" in rec for rec in results["recommendations"]))
    
    @patch('app.test_suite_integration.TestHistoryManager.get_regression_baseline')
    @patch('app.test_runner.TestRunner.run_test_suite')
    def test_run_regression_analysis(self, mock_run_suite, mock_get_baseline):
        """Test regression analysis"""
        # Mock baseline data
        baseline_test_results = [
            {"test_name": "test1", "status": "passed", "duration": 1.0},
            {"test_name": "test2", "status": "passed", "duration": 2.0}
        ]
        
        mock_get_baseline.return_value = {
            "run_info": {"id": 1, "timestamp": "2023-01-01T00:00:00"},
            "test_results": baseline_test_results
        }
        
        # Mock current test results
        current_test_results = [
            TestResult("test1", TestType.UNIT, TestStatus.FAILED, 1.5),  # Regression
            TestResult("test2", TestType.UNIT, TestStatus.PASSED, 2.1),  # OK
            TestResult("test3", TestType.UNIT, TestStatus.FAILED, 1.0)   # New failure
        ]
        
        mock_suite_result = TestSuiteResult(
            "unit_tests", 3, 1, 2, 0, 0, 4.6, 85.0, current_test_results
        )
        mock_run_suite.return_value = mock_suite_result
        
        regression_results = self.integration._run_regression_analysis()
        
        self.assertIn("regressions", regression_results)
        self.assertIn("improvements", regression_results)
        self.assertIn("new_failures", regression_results)
        self.assertIn("summary", regression_results)
        
        # Should detect regression for test1
        self.assertGreater(len(regression_results["regressions"]), 0)
        # Should detect new failure for test3
        self.assertGreater(len(regression_results["new_failures"]), 0)
    
    @patch('subprocess.run')
    def test_get_git_info(self, mock_run):
        """Test getting git information"""
        # Mock git commands
        def mock_git_command(cmd, **kwargs):
            if "rev-parse HEAD" in " ".join(cmd):
                result = Mock()
                result.returncode = 0
                result.stdout = "abc123def456\n"
                return result
            elif "rev-parse --abbrev-ref HEAD" in " ".join(cmd):
                result = Mock()
                result.returncode = 0
                result.stdout = "main\n"
                return result
            else:
                result = Mock()
                result.returncode = 1
                return result
        
        mock_run.side_effect = mock_git_command
        
        git_info = self.integration._get_git_info()
        
        self.assertEqual(git_info["commit"], "abc123def456")
        self.assertEqual(git_info["branch"], "main")
    
    @patch('subprocess.run')
    def test_get_git_info_failure(self, mock_run):
        """Test getting git information when git fails"""
        # Mock git command failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        git_info = self.integration._get_git_info()
        
        self.assertEqual(git_info["commit"], "")
        self.assertEqual(git_info["branch"], "")
    
    @patch('app.test_suite_integration.TestHistoryManager.analyze_test_trends')
    def test_get_test_trends(self, mock_analyze_trends):
        """Test getting test trends"""
        # Mock trend data
        mock_trend_data = {
            "suite_name": "unit_tests",
            "trends": {"success_rate": {"current": 85.0, "trend": "stable"}}
        }
        mock_analyze_trends.return_value = mock_trend_data
        
        trends = self.integration.get_test_trends(days=30)
        
        self.assertIn("trends_by_type", trends)
        self.assertIn("unit", trends["trends_by_type"])
        self.assertIn("integration", trends["trends_by_type"])
        self.assertEqual(trends["analysis_period_days"], 30)
    
    @patch('app.test_suite_integration.TestSuiteIntegration.get_test_trends')
    @patch('app.test_suite_integration.TestHistoryManager.get_test_history')
    def test_generate_test_report(self, mock_get_history, mock_get_trends):
        """Test generating comprehensive test report"""
        # Mock trends
        mock_get_trends.return_value = {
            "trends_by_type": {
                "unit": {
                    "trends": {
                        "success_rate": {"current": 75.0, "trend": "declining"}
                    }
                }
            }
        }
        
        # Mock recent runs
        mock_get_history.return_value = [
            {"id": 1, "suite_name": "unit_tests", "passed": 8, "failed": 2}
        ]
        
        report = self.integration.generate_test_report(days=7)
        
        self.assertIn("summary", report)
        self.assertIn("trends", report)
        self.assertIn("recent_runs", report)
        self.assertIn("recommendations", report)
        self.assertEqual(report["report_period_days"], 7)
        
        # Should have recommendations for declining success rate
        rec_text = " ".join(report["recommendations"])
        self.assertIn("declining", rec_text)


class TestGetTestSuiteIntegration(unittest.TestCase):
    """Test get_test_suite_integration convenience function"""
    
    def test_get_test_suite_integration_default(self):
        """Test getting test suite integration with default project root"""
        integration = get_test_suite_integration()
        
        self.assertIsInstance(integration, TestSuiteIntegration)
    
    def test_get_test_suite_integration_custom_root(self):
        """Test getting test suite integration with custom project root"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            integration = get_test_suite_integration(temp_dir)
            
            self.assertIsInstance(integration, TestSuiteIntegration)
            self.assertEqual(str(integration.project_root), temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()