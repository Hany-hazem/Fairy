# app/test_suite_integration.py
"""
Test suite integration for automated testing framework
Provides integration with existing test suites, coverage analysis, and regression testing
"""

import os
import sys
import subprocess
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import coverage
import sqlite3
from collections import defaultdict

from app.test_runner import TestRunner, TestResult, TestSuiteResult, TestType, TestStatus


@dataclass
class CoverageReport:
    """Code coverage analysis report"""
    total_lines: int
    covered_lines: int
    missing_lines: List[int]
    coverage_percentage: float
    file_path: str
    branch_coverage: Optional[float] = None
    function_coverage: Optional[Dict[str, bool]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class RegressionTestResult:
    """Result of regression testing"""
    test_name: str
    baseline_status: TestStatus
    current_status: TestStatus
    is_regression: bool
    baseline_duration: float
    current_duration: float
    performance_change_percent: float
    error_details: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['baseline_status'] = self.baseline_status.value
        result['current_status'] = self.current_status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class TestCoverageAnalysis:
    """Analysis of test coverage for code changes"""
    changed_files: List[str]
    covered_files: List[str]
    uncovered_files: List[str]
    coverage_by_file: Dict[str, CoverageReport]
    overall_coverage: float
    coverage_change: float
    new_lines_covered: int
    new_lines_total: int
    recommendations: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['coverage_by_file'] = {
            k: v.to_dict() for k, v in self.coverage_by_file.items()
        }
        result['timestamp'] = self.timestamp.isoformat()
        return result


class TestHistoryManager:
    """Manages historical test data and trends"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for test history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_tests INTEGER,
                    passed INTEGER,
                    failed INTEGER,
                    errors INTEGER,
                    skipped INTEGER,
                    duration REAL,
                    coverage_percentage REAL,
                    git_commit TEXT,
                    branch TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    test_name TEXT NOT NULL,
                    test_type TEXT,
                    status TEXT,
                    duration REAL,
                    message TEXT,
                    error_details TEXT,
                    FOREIGN KEY (run_id) REFERENCES test_runs (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS coverage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    file_path TEXT NOT NULL,
                    total_lines INTEGER,
                    covered_lines INTEGER,
                    coverage_percentage REAL,
                    missing_lines TEXT,
                    FOREIGN KEY (run_id) REFERENCES test_runs (id)
                )
            """)
            
            conn.commit()
    
    def store_test_run(self, suite_result: TestSuiteResult, git_info: Optional[Dict[str, str]] = None) -> int:
        """Store test run results in database"""
        git_commit = git_info.get('commit', '') if git_info else ''
        git_branch = git_info.get('branch', '') if git_info else ''
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert test run
            cursor.execute("""
                INSERT INTO test_runs 
                (suite_name, timestamp, total_tests, passed, failed, errors, skipped, 
                 duration, coverage_percentage, git_commit, branch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                suite_result.suite_name,
                suite_result.timestamp.isoformat(),
                suite_result.total_tests,
                suite_result.passed,
                suite_result.failed,
                suite_result.errors,
                suite_result.skipped,
                suite_result.duration,
                suite_result.coverage_percentage,
                git_commit,
                git_branch
            ))
            
            run_id = cursor.lastrowid
            
            # Insert individual test results
            for test_result in suite_result.test_results:
                cursor.execute("""
                    INSERT INTO test_results 
                    (run_id, test_name, test_type, status, duration, message, error_details)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    test_result.test_name,
                    test_result.test_type.value,
                    test_result.status.value,
                    test_result.duration,
                    test_result.message,
                    test_result.error_details
                ))
            
            conn.commit()
            return run_id
    
    def get_test_history(self, days: int = 30, suite_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get test execution history"""
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        query = """
            SELECT * FROM test_runs 
            WHERE timestamp >= ?
        """
        params = [cutoff_date]
        
        if suite_name:
            query += " AND suite_name = ?"
            params.append(suite_name)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_regression_baseline(self, suite_name: str, days_back: int = 7) -> Optional[Dict[str, Any]]:
        """Get baseline test results for regression testing"""
        cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get the most recent successful run within the time window
            cursor.execute("""
                SELECT * FROM test_runs 
                WHERE suite_name = ? AND timestamp >= ? AND failed = 0 AND errors = 0
                ORDER BY timestamp DESC LIMIT 1
            """, (suite_name, cutoff_date))
            
            baseline_run = cursor.fetchone()
            if not baseline_run:
                return None
            
            # Get individual test results for this run
            cursor.execute("""
                SELECT * FROM test_results WHERE run_id = ?
            """, (baseline_run['id'],))
            
            test_results = [dict(row) for row in cursor.fetchall()]
            
            return {
                'run_info': dict(baseline_run),
                'test_results': test_results
            }
    
    def analyze_test_trends(self, suite_name: str, days: int = 30) -> Dict[str, Any]:
        """Analyze test trends over time"""
        history = self.get_test_history(days, suite_name)
        
        if len(history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        # Calculate trends
        success_rates = []
        durations = []
        coverage_percentages = []
        
        for run in history:
            if run['total_tests'] > 0:
                success_rate = (run['passed'] / run['total_tests']) * 100
                success_rates.append(success_rate)
            
            if run['duration']:
                durations.append(run['duration'])
            
            if run['coverage_percentage']:
                coverage_percentages.append(run['coverage_percentage'])
        
        trends = {}
        
        if success_rates:
            trends['success_rate'] = {
                'current': success_rates[0],
                'average': sum(success_rates) / len(success_rates),
                'trend': 'improving' if len(success_rates) > 1 and success_rates[0] > success_rates[-1] else 'declining'
            }
        
        if durations:
            trends['duration'] = {
                'current': durations[0],
                'average': sum(durations) / len(durations),
                'trend': 'improving' if len(durations) > 1 and durations[0] < durations[-1] else 'declining'
            }
        
        if coverage_percentages:
            trends['coverage'] = {
                'current': coverage_percentages[0],
                'average': sum(coverage_percentages) / len(coverage_percentages),
                'trend': 'improving' if len(coverage_percentages) > 1 and coverage_percentages[0] > coverage_percentages[-1] else 'declining'
            }
        
        return {
            'suite_name': suite_name,
            'analysis_period_days': days,
            'total_runs': len(history),
            'trends': trends,
            'recent_runs': history[:5]  # Last 5 runs
        }


class CoverageAnalyzer:
    """Analyzes code coverage for test suites"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.source_dirs = ['app']  # Directories to analyze for coverage
    
    def analyze_coverage(
        self,
        test_files: List[str],
        changed_files: Optional[List[str]] = None
    ) -> TestCoverageAnalysis:
        """Analyze test coverage, optionally focusing on changed files"""
        # Set up coverage
        cov = coverage.Coverage(
            source=self.source_dirs,
            omit=['*/tests/*', '*/test_*', '*/__pycache__/*']
        )
        
        coverage_by_file = {}
        
        try:
            cov.start()
            
            # Run tests to collect coverage
            self._run_tests_for_coverage(test_files)
            
            cov.stop()
            cov.save()
            
            # Analyze coverage data
            coverage_data = cov.get_data()
            
            for filename in coverage_data.measured_files():
                rel_path = os.path.relpath(filename, self.project_root)
                
                # Get coverage info for this file
                analysis = cov.analysis2(filename)
                executed_lines = analysis[1]
                missing_lines = analysis[3]
                total_lines = len(analysis[1]) + len(missing_lines)
                
                if total_lines > 0:
                    coverage_percentage = (len(executed_lines) / total_lines) * 100
                else:
                    coverage_percentage = 0.0
                
                coverage_by_file[rel_path] = CoverageReport(
                    total_lines=total_lines,
                    covered_lines=len(executed_lines),
                    missing_lines=list(missing_lines),
                    coverage_percentage=coverage_percentage,
                    file_path=rel_path
                )
        
        except Exception as e:
            print(f"Coverage analysis failed: {e}")
            # Return empty analysis on failure
            return TestCoverageAnalysis(
                changed_files=changed_files or [],
                covered_files=[],
                uncovered_files=[],
                coverage_by_file={},
                overall_coverage=0.0,
                coverage_change=0.0,
                new_lines_covered=0,
                new_lines_total=0,
                recommendations=["Coverage analysis failed"]
            )
        
        # Calculate overall coverage
        total_lines = sum(report.total_lines for report in coverage_by_file.values())
        covered_lines = sum(report.covered_lines for report in coverage_by_file.values())
        overall_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0.0
        
        # Analyze changed files if provided
        covered_files = []
        uncovered_files = []
        new_lines_covered = 0
        new_lines_total = 0
        
        if changed_files:
            for file_path in changed_files:
                if file_path in coverage_by_file:
                    report = coverage_by_file[file_path]
                    if report.coverage_percentage > 0:
                        covered_files.append(file_path)
                    else:
                        uncovered_files.append(file_path)
                    
                    # For changed files, consider all lines as "new"
                    new_lines_total += report.total_lines
                    new_lines_covered += report.covered_lines
                else:
                    uncovered_files.append(file_path)
        
        # Generate recommendations
        recommendations = self._generate_coverage_recommendations(
            coverage_by_file, overall_coverage, changed_files
        )
        
        return TestCoverageAnalysis(
            changed_files=changed_files or [],
            covered_files=covered_files,
            uncovered_files=uncovered_files,
            coverage_by_file=coverage_by_file,
            overall_coverage=overall_coverage,
            coverage_change=0.0,  # Would need baseline to calculate
            new_lines_covered=new_lines_covered,
            new_lines_total=new_lines_total,
            recommendations=recommendations
        )
    
    def _run_tests_for_coverage(self, test_files: List[str]):
        """Run tests to collect coverage data"""
        for test_file in test_files:
            try:
                # Import and run the test module
                spec = importlib.util.spec_from_file_location("test_module", test_file)
                test_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(test_module)
                
                # Run tests if it's a unittest module
                if hasattr(test_module, 'unittest'):
                    loader = unittest.TestLoader()
                    suite = loader.loadTestsFromModule(test_module)
                    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
                    runner.run(suite)
            
            except Exception as e:
                print(f"Failed to run test file {test_file} for coverage: {e}")
                continue
    
    def _generate_coverage_recommendations(
        self,
        coverage_by_file: Dict[str, CoverageReport],
        overall_coverage: float,
        changed_files: Optional[List[str]]
    ) -> List[str]:
        """Generate coverage improvement recommendations"""
        recommendations = []
        
        # Overall coverage recommendations
        if overall_coverage < 70:
            recommendations.append(
                f"Overall coverage is {overall_coverage:.1f}%. Consider adding more tests to reach 70%+ coverage."
            )
        elif overall_coverage < 90:
            recommendations.append(
                f"Good coverage at {overall_coverage:.1f}%. Consider targeting 90%+ for critical components."
            )
        
        # File-specific recommendations
        low_coverage_files = [
            (path, report) for path, report in coverage_by_file.items()
            if report.coverage_percentage < 70
        ]
        
        if low_coverage_files:
            recommendations.append(
                f"Files with low coverage (<70%): {', '.join(path for path, _ in low_coverage_files[:3])}"
            )
        
        # Changed files recommendations
        if changed_files:
            uncovered_changed = [
                file for file in changed_files
                if file in coverage_by_file and coverage_by_file[file].coverage_percentage < 50
            ]
            
            if uncovered_changed:
                recommendations.append(
                    f"Changed files with low coverage: {', '.join(uncovered_changed)}. "
                    "Add tests for modified code."
                )
        
        return recommendations


class TestSuiteIntegration:
    """Integrates automated testing framework with existing test suites"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_runner = TestRunner(str(project_root))
        self.coverage_analyzer = CoverageAnalyzer(str(project_root))
        self.history_manager = TestHistoryManager(
            str(self.project_root / "test_results" / "test_history.db")
        )
    
    def run_comprehensive_test_suite(
        self,
        changed_files: Optional[List[str]] = None,
        include_regression: bool = True,
        include_coverage: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive test suite with coverage and regression analysis"""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "changed_files": changed_files or [],
            "test_results": {},
            "coverage_analysis": None,
            "regression_analysis": None,
            "recommendations": [],
            "overall_status": "unknown"
        }
        
        # Run different types of tests
        test_types = [TestType.UNIT, TestType.INTEGRATION]
        all_passed = True
        
        for test_type in test_types:
            suite_result = self.test_runner.run_test_suite(
                test_type=test_type,
                isolated=True,
                coverage_enabled=include_coverage
            )
            
            results["test_results"][test_type.value] = suite_result.to_dict()
            
            if suite_result.failed > 0 or suite_result.errors > 0:
                all_passed = False
            
            # Store in history
            git_info = self._get_git_info()
            self.history_manager.store_test_run(suite_result, git_info)
        
        # Coverage analysis
        if include_coverage and changed_files:
            try:
                all_test_files = self.test_runner.discover_tests()
                coverage_analysis = self.coverage_analyzer.analyze_coverage(
                    all_test_files, changed_files
                )
                results["coverage_analysis"] = coverage_analysis.to_dict()
                results["recommendations"].extend(coverage_analysis.recommendations)
            except Exception as e:
                results["coverage_analysis"] = {"error": str(e)}
        
        # Regression analysis
        if include_regression:
            try:
                regression_results = self._run_regression_analysis()
                results["regression_analysis"] = regression_results
                
                if regression_results.get("regressions"):
                    all_passed = False
                    results["recommendations"].append(
                        f"Found {len(regression_results['regressions'])} test regressions"
                    )
            except Exception as e:
                results["regression_analysis"] = {"error": str(e)}
        
        # Overall status and recommendations
        if all_passed:
            results["overall_status"] = "passed"
            results["recommendations"].insert(0, "All tests passed successfully")
        else:
            results["overall_status"] = "failed"
            results["recommendations"].insert(0, "Some tests failed - review failures before proceeding")
        
        return results
    
    def _run_regression_analysis(self) -> Dict[str, Any]:
        """Run regression analysis against recent baseline"""
        regression_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "baseline_info": None,
            "regressions": [],
            "improvements": [],
            "new_failures": [],
            "summary": {}
        }
        
        # Get baseline for unit tests
        baseline = self.history_manager.get_regression_baseline("unit_tests")
        if not baseline:
            return {"message": "No suitable baseline found for regression testing"}
        
        regression_results["baseline_info"] = baseline["run_info"]
        
        # Run current tests
        current_results = self.test_runner.run_test_suite(TestType.UNIT, isolated=True)
        
        # Compare results
        baseline_tests = {
            result["test_name"]: result for result in baseline["test_results"]
        }
        
        for current_test in current_results.test_results:
            test_name = current_test.test_name
            baseline_test = baseline_tests.get(test_name)
            
            if baseline_test:
                # Compare with baseline
                baseline_status = TestStatus(baseline_test["status"])
                current_status = current_test.status
                
                regression_result = RegressionTestResult(
                    test_name=test_name,
                    baseline_status=baseline_status,
                    current_status=current_status,
                    is_regression=False,
                    baseline_duration=baseline_test["duration"],
                    current_duration=current_test.duration,
                    performance_change_percent=0.0
                )
                
                # Check for regression
                if baseline_status == TestStatus.PASSED and current_status != TestStatus.PASSED:
                    regression_result.is_regression = True
                    regression_results["regressions"].append(regression_result.to_dict())
                
                # Check for improvement
                elif baseline_status != TestStatus.PASSED and current_status == TestStatus.PASSED:
                    regression_results["improvements"].append(regression_result.to_dict())
                
                # Check performance regression
                if baseline_test["duration"] > 0:
                    perf_change = ((current_test.duration - baseline_test["duration"]) / 
                                 baseline_test["duration"]) * 100
                    regression_result.performance_change_percent = perf_change
                    
                    if perf_change > 50:  # 50% slower is significant
                        regression_result.is_regression = True
                        if regression_result.to_dict() not in regression_results["regressions"]:
                            regression_results["regressions"].append(regression_result.to_dict())
            
            else:
                # New test (not in baseline)
                if current_test.status != TestStatus.PASSED:
                    regression_results["new_failures"].append({
                        "test_name": test_name,
                        "status": current_test.status.value,
                        "message": current_test.message
                    })
        
        # Generate summary
        regression_results["summary"] = {
            "total_regressions": len(regression_results["regressions"]),
            "total_improvements": len(regression_results["improvements"]),
            "new_failures": len(regression_results["new_failures"]),
            "baseline_date": baseline["run_info"]["timestamp"]
        }
        
        return regression_results
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get current git commit and branch information"""
        git_info = {"commit": "", "branch": ""}
        
        try:
            # Get current commit
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                git_info["commit"] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()
        
        except Exception:
            pass  # Git info is optional
        
        return git_info
    
    def get_test_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get test trends and historical analysis"""
        trends = {}
        
        for test_type in [TestType.UNIT, TestType.INTEGRATION]:
            suite_name = f"{test_type.value}_tests"
            trend_data = self.history_manager.analyze_test_trends(suite_name, days)
            trends[test_type.value] = trend_data
        
        return {
            "analysis_period_days": days,
            "trends_by_type": trends,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def generate_test_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            "report_period_days": days,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {},
            "trends": {},
            "recent_runs": {},
            "recommendations": []
        }
        
        # Get trends
        trends = self.get_test_trends(days)
        report["trends"] = trends
        
        # Get recent test runs
        for test_type in [TestType.UNIT, TestType.INTEGRATION]:
            suite_name = f"{test_type.value}_tests"
            recent_runs = self.history_manager.get_test_history(days, suite_name)
            report["recent_runs"][test_type.value] = recent_runs[:5]  # Last 5 runs
        
        # Generate summary and recommendations
        total_runs = sum(len(runs) for runs in report["recent_runs"].values())
        report["summary"]["total_runs"] = total_runs
        
        if total_runs == 0:
            report["recommendations"].append("No test runs found in the specified period")
        else:
            # Analyze trends for recommendations
            for test_type, trend_data in trends["trends_by_type"].items():
                if "trends" in trend_data:
                    type_trends = trend_data["trends"]
                    
                    if "success_rate" in type_trends:
                        success_trend = type_trends["success_rate"]
                        if success_trend["trend"] == "declining":
                            report["recommendations"].append(
                                f"{test_type.title()} test success rate is declining - "
                                f"current: {success_trend['current']:.1f}%"
                            )
                    
                    if "duration" in type_trends:
                        duration_trend = type_trends["duration"]
                        if duration_trend["trend"] == "declining":  # Getting slower
                            report["recommendations"].append(
                                f"{test_type.title()} test duration is increasing - "
                                f"current: {duration_trend['current']:.1f}s"
                            )
        
        return report


# Convenience function for getting test suite integration instance
def get_test_suite_integration(project_root: Optional[str] = None) -> TestSuiteIntegration:
    """Get TestSuiteIntegration instance"""
    if project_root is None:
        project_root = os.getcwd()
    return TestSuiteIntegration(project_root)


# Add missing import
import importlib.util
import unittest