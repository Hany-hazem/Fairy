# tests/test_e2e_self_improvement_integration.py
"""
End-to-end self-improvement cycle integration tests in isolated environment
Tests complete self-improvement workflows, safety mechanisms, rollback scenarios,
and performance improvement validation.

Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5
"""

import pytest
import asyncio
import tempfile
import shutil
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.self_improvement_engine import (
    SelfImprovementEngine, ImprovementCycle, ImprovementCycleStatus, SafetyLevel
)
from app.improvement_engine import Improvement, ImprovementType, RiskLevel, ImprovementPriority
from app.code_modifier import ModificationPlan, FileModification
from app.test_runner import TestRunner, TestType, TestSuiteResult, TestStatus, TestResult
from app.performance_monitor import PerformanceMonitor, PerformanceReport
from app.code_analyzer import CodeAnalyzer, QualityReport, ComplexityMetrics, CodeIssue, IssueSeverity
from app.version_control import GitIntegration, ChangeRecord

logger = logging.getLogger(__name__)

class TestE2ESelfImprovementIntegration:
    """End-to-end self-improvement integration tests"""
    
    @pytest.fixture
    def isolated_project_env(self):
        """Create isolated project environment for testing"""
        temp_dir = Path(tempfile.mkdtemp(prefix="self_improvement_test_"))
        
        # Create basic project structure
        (temp_dir / "app").mkdir()
        (temp_dir / "tests").mkdir()
        (temp_dir / ".kiro").mkdir()
        
        # Create sample Python files for testing
        sample_code = '''
def inefficient_function(data):
    """Sample function with performance issues"""
    result = ""
    for item in data:
        result += str(item)  # Inefficient string concatenation
    return result

def complex_function(a, b, c, d, e, f, g, h):
    """Function with too many parameters"""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:  # Deep nesting
                        return a + b + c + d + e + f + g + h
    return 0

class SampleClass:
    def method_without_docstring(self):
        magic_number = 42  # Magic number
        return magic_number * 2
'''
        
        (temp_dir / "app" / "sample_module.py").write_text(sample_code)
        
        # Create sample test file
        test_code = '''
import unittest
from app.sample_module import inefficient_function, complex_function

class TestSampleModule(unittest.TestCase):
    def test_inefficient_function(self):
        result = inefficient_function([1, 2, 3])
        self.assertEqual(result, "123")
    
    def test_complex_function(self):
        result = complex_function(1, 1, 1, 1, 1, 1, 1, 1)
        self.assertEqual(result, 8)

if __name__ == "__main__":
    unittest.main()
'''
        
        (temp_dir / "tests" / "test_sample_module.py").write_text(test_code)
        
        # Create requirements.txt
        (temp_dir / "requirements.txt").write_text("pytest\ncoverage\n")
        
        # Initialize git repository
        git_integration = GitIntegration(str(temp_dir))
        git_integration.init_repo_if_needed()
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_performance_monitor(self):
        """Create mock performance monitor with realistic data"""
        monitor = Mock(spec=PerformanceMonitor)
        
        # Mock performance report
        mock_report = Mock(spec=PerformanceReport)
        mock_report.response_times = {"api_endpoint": 0.5, "database_query": 0.2}
        mock_report.memory_usage = {"peak_mb": 256, "average_mb": 128}
        mock_report.error_rates = {"total_errors": 5, "error_rate": 0.01}
        mock_report.alerts = ["Response time threshold exceeded for api_endpoint"]
        mock_report.recommendations = ["Optimize database queries", "Implement caching"]
        
        monitor.get_performance_report.return_value = mock_report
        monitor.collect_system_metrics.return_value = True
        
        return monitor
    
    @pytest.fixture
    def self_improvement_engine(self, isolated_project_env, mock_performance_monitor):
        """Create self-improvement engine with mocked dependencies"""
        engine = SelfImprovementEngine(
            project_root=str(isolated_project_env),
            config={
                "safety_level": "conservative",
                "auto_apply_threshold": 8.0,
                "max_concurrent_improvements": 2
            }
        )
        
        # Replace performance monitor with mock
        engine.performance_monitor = mock_performance_monitor
        engine.performance_analyzer.performance_monitor = mock_performance_monitor
        
        return engine
    
    @pytest.mark.asyncio
    async def test_complete_self_improvement_cycle(self, self_improvement_engine):
        """Test complete self-improvement cycle from analysis to application"""
        engine = self_improvement_engine
        
        # Mock improvement suggestions
        mock_improvements = [
            Improvement(
                id="imp_001",
                title="Fix inefficient string concatenation",
                description="Replace string concatenation in loop with list.append() and join()",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.LOW,
                impact_score=8.5,
                affected_files=["app/sample_module.py"],
                proposed_changes={
                    "app/sample_module.py": '''
def inefficient_function(data):
    """Sample function with performance issues - IMPROVED"""
    result_parts = []
    for item in data:
        result_parts.append(str(item))  # Efficient list append
    return "".join(result_parts)

def complex_function(a, b, c, d, e, f, g, h):
    """Function with too many parameters"""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:  # Deep nesting
                        return a + b + c + d + e + f + g + h
    return 0

class SampleClass:
    def method_without_docstring(self):
        magic_number = 42  # Magic number
        return magic_number * 2
'''
                },
                implementation_effort="1.0 hours"
            )
        ]
        
        # Mock the improvement engine to return our test improvements
        engine.improvement_engine.analyze_and_suggest_improvements = Mock(return_value=mock_improvements)
        
        # Start improvement cycle
        cycle_id = await engine.trigger_improvement_cycle("test_cycle")
        
        # Wait for cycle to complete (with timeout)
        max_wait = 30  # seconds
        start_time = time.time()
        
        while engine.current_cycle and engine.current_cycle.status not in [
            ImprovementCycleStatus.COMPLETED,
            ImprovementCycleStatus.FAILED,
            ImprovementCycleStatus.ROLLED_BACK
        ]:
            if time.time() - start_time > max_wait:
                pytest.fail("Improvement cycle timed out")
            
            await asyncio.sleep(0.5)
        
        # Verify cycle completed successfully
        assert engine.current_cycle is None  # Should be moved to history
        
        # Check cycle history
        history = engine.get_cycle_history(limit=1)
        assert len(history) == 1
        
        completed_cycle = history[0]
        assert completed_cycle["id"] == cycle_id
        assert completed_cycle["status"] == ImprovementCycleStatus.COMPLETED.value
        assert len(completed_cycle["applied_improvements"]) > 0
        
        # Verify file was actually modified
        modified_file = Path(engine.project_root) / "app" / "sample_module.py"
        modified_content = modified_file.read_text()
        assert "IMPROVED" in modified_content
        assert "result_parts = []" in modified_content
        assert "result_parts.append" in modified_content
    
    @pytest.mark.asyncio
    async def test_safety_mechanism_validation(self, self_improvement_engine):
        """Test safety mechanisms prevent dangerous changes"""
        engine = self_improvement_engine
        
        # Create high-risk improvement that should be filtered out
        dangerous_improvement = Improvement(
            id="imp_danger",
            title="Dangerous system modification",
            description="This change could break the system",
            type=ImprovementType.CODE_QUALITY,
            priority=ImprovementPriority.HIGH,
            risk_level=RiskLevel.VERY_HIGH,  # Should be filtered in conservative mode
            impact_score=9.0,
            affected_files=["app/sample_module.py"],
            proposed_changes={
                "app/sample_module.py": "# This would delete all code"
            },
            implementation_effort="0.5 hours"
        )
        
        # Mock improvement engine to return dangerous improvement
        engine.improvement_engine.analyze_and_suggest_improvements = Mock(
            return_value=[dangerous_improvement]
        )
        
        # Trigger cycle with conservative safety level
        engine.safety_level = SafetyLevel.CONSERVATIVE
        cycle_id = await engine.trigger_improvement_cycle("safety_test")
        
        # Wait for cycle to complete
        max_wait = 20
        start_time = time.time()
        
        while engine.current_cycle and engine.current_cycle.status not in [
            ImprovementCycleStatus.COMPLETED,
            ImprovementCycleStatus.FAILED,
            ImprovementCycleStatus.ROLLED_BACK
        ]:
            if time.time() - start_time > max_wait:
                break
            await asyncio.sleep(0.5)
        
        # Verify dangerous improvement was filtered out
        history = engine.get_cycle_history(limit=1)
        assert len(history) == 1
        
        completed_cycle = history[0]
        # Should complete but with no applied improvements due to safety filtering
        assert len(completed_cycle["applied_improvements"]) == 0
        
        # Verify original file is unchanged
        original_file = Path(engine.project_root) / "app" / "sample_module.py"
        original_content = original_file.read_text()
        assert "This would delete all code" not in original_content
    
    @pytest.mark.asyncio
    async def test_rollback_on_test_failure(self, self_improvement_engine):
        """Test automatic rollback when tests fail after applying changes"""
        engine = self_improvement_engine
        
        # Create improvement that will break tests
        breaking_improvement = Improvement(
            id="imp_breaking",
            title="Change that breaks tests",
            description="This change will cause test failures",
            type=ImprovementType.CODE_QUALITY,
            priority=ImprovementPriority.MEDIUM,
            risk_level=RiskLevel.LOW,  # Low risk but will break tests
            impact_score=6.0,
            affected_files=["app/sample_module.py"],
            proposed_changes={
                "app/sample_module.py": '''
def inefficient_function(data):
    """This change breaks the expected return value"""
    return "BROKEN"  # This will fail the test

def complex_function(a, b, c, d, e, f, g, h):
    """Function with too many parameters"""
    return "ALSO_BROKEN"  # This will also fail

class SampleClass:
    def method_without_docstring(self):
        return "BROKEN_TOO"
'''
            },
            implementation_effort="1.0 hours"
        )
        
        # Mock improvement engine
        engine.improvement_engine.analyze_and_suggest_improvements = Mock(
            return_value=[breaking_improvement]
        )
        
        # Store original file content for comparison
        original_file = Path(engine.project_root) / "app" / "sample_module.py"
        original_content = original_file.read_text()
        
        # Trigger cycle
        cycle_id = await engine.trigger_improvement_cycle("rollback_test")
        
        # Wait for cycle to complete
        max_wait = 30
        start_time = time.time()
        
        while engine.current_cycle and engine.current_cycle.status not in [
            ImprovementCycleStatus.COMPLETED,
            ImprovementCycleStatus.FAILED,
            ImprovementCycleStatus.ROLLED_BACK
        ]:
            if time.time() - start_time > max_wait:
                break
            await asyncio.sleep(0.5)
        
        # Verify cycle failed or was rolled back
        history = engine.get_cycle_history(limit=1)
        assert len(history) == 1
        
        completed_cycle = history[0]
        assert completed_cycle["status"] in [
            ImprovementCycleStatus.FAILED.value,
            ImprovementCycleStatus.ROLLED_BACK.value
        ]
        
        # Verify file was rolled back to original content
        current_content = original_file.read_text()
        assert current_content == original_content
        assert "BROKEN" not in current_content
    
    @pytest.mark.asyncio
    async def test_performance_improvement_validation(self, self_improvement_engine):
        """Test validation of performance improvements"""
        engine = self_improvement_engine
        
        # Create performance improvement
        perf_improvement = Improvement(
            id="imp_perf",
            title="Optimize string operations",
            description="Replace inefficient string concatenation",
            type=ImprovementType.PERFORMANCE,
            priority=ImprovementPriority.HIGH,
            risk_level=RiskLevel.LOW,
            impact_score=7.5,
            affected_files=["app/sample_module.py"],
            proposed_changes={
                "app/sample_module.py": '''
def inefficient_function(data):
    """Optimized function for better performance"""
    return "".join(str(item) for item in data)  # More efficient

def complex_function(a, b, c, d, e, f, g, h):
    """Function with too many parameters"""
    if all(x > 0 for x in [a, b, c, d, e]):  # Simplified logic
        return sum([a, b, c, d, e, f, g, h])
    return 0

class SampleClass:
    def method_without_docstring(self):
        """Added docstring for better maintainability"""
        MAGIC_CONSTANT = 42  # Named constant
        return MAGIC_CONSTANT * 2
'''
            },
            implementation_effort="2.0 hours"
        )
        
        # Mock improvement engine
        engine.improvement_engine.analyze_and_suggest_improvements = Mock(
            return_value=[perf_improvement]
        )
        
        # Mock performance comparison to show improvement
        original_performance = {"execution_time": 0.1, "memory_usage": 100}
        improved_performance = {"execution_time": 0.05, "memory_usage": 80}
        
        with patch.object(engine.test_runner, 'run_performance_comparison') as mock_perf_compare:
            mock_perf_compare.return_value = {
                "improvements": [
                    {
                        "test": "test_inefficient_function",
                        "metric": "execution_time",
                        "change_percent": -50.0,  # 50% improvement
                        "current_value": 0.05,
                        "baseline_value": 0.1
                    }
                ],
                "regressions": [],
                "summary": {
                    "total_improvements": 1,
                    "total_regressions": 0,
                    "net_performance_change": 1
                }
            }
            
            # Trigger cycle
            cycle_id = await engine.trigger_improvement_cycle("performance_test")
            
            # Wait for completion
            max_wait = 30
            start_time = time.time()
            
            while engine.current_cycle and engine.current_cycle.status not in [
                ImprovementCycleStatus.COMPLETED,
                ImprovementCycleStatus.FAILED,
                ImprovementCycleStatus.ROLLED_BACK
            ]:
                if time.time() - start_time > max_wait:
                    break
                await asyncio.sleep(0.5)
            
            # Verify successful completion
            history = engine.get_cycle_history(limit=1)
            assert len(history) == 1
            
            completed_cycle = history[0]
            assert completed_cycle["status"] == ImprovementCycleStatus.COMPLETED.value
            assert len(completed_cycle["applied_improvements"]) > 0
            
            # Verify performance improvement was validated
            # (In a real scenario, this would involve actual performance measurements)
    
    @pytest.mark.asyncio
    async def test_concurrent_improvement_cycles_prevention(self, self_improvement_engine):
        """Test that concurrent improvement cycles are prevented"""
        engine = self_improvement_engine
        
        # Mock improvement engine
        engine.improvement_engine.analyze_and_suggest_improvements = Mock(
            return_value=[
                Improvement(
                    id="imp_concurrent",
                    title="Test concurrent improvement",
                    description="Test improvement for concurrency",
                    type=ImprovementType.PERFORMANCE,
                    priority=ImprovementPriority.MEDIUM,
                    risk_level=RiskLevel.LOW,
                    impact_score=5.0,
                    affected_files=["app/sample_module.py"],
                    proposed_changes={"app/sample_module.py": "# Test change"},
                    implementation_effort="1.0 hours"
                )
            ]
        )
        
        # Start first cycle
        cycle_id_1 = await engine.trigger_improvement_cycle("concurrent_test_1")
        
        # Try to start second cycle while first is running
        with pytest.raises(ValueError, match="Another improvement cycle is already running"):
            await engine.trigger_improvement_cycle("concurrent_test_2")
        
        # Wait for first cycle to complete
        max_wait = 20
        start_time = time.time()
        
        while engine.current_cycle and engine.current_cycle.status not in [
            ImprovementCycleStatus.COMPLETED,
            ImprovementCycleStatus.FAILED,
            ImprovementCycleStatus.ROLLED_BACK
        ]:
            if time.time() - start_time > max_wait:
                break
            await asyncio.sleep(0.5)
        
        # Now second cycle should be allowed
        cycle_id_2 = await engine.trigger_improvement_cycle("concurrent_test_2")
        assert cycle_id_2 != cycle_id_1
    
    @pytest.mark.asyncio
    async def test_emergency_stop_functionality(self, self_improvement_engine):
        """Test emergency stop functionality"""
        engine = self_improvement_engine
        
        # Mock improvement engine with slow operation
        async def slow_improvement_analysis():
            await asyncio.sleep(10)  # Simulate long-running analysis
            return []
        
        engine.improvement_engine.analyze_and_suggest_improvements = slow_improvement_analysis
        
        # Start improvement cycle
        cycle_id = await engine.trigger_improvement_cycle("emergency_test")
        
        # Wait a bit for cycle to start
        await asyncio.sleep(1)
        
        # Trigger emergency stop
        stop_result = await engine.emergency_stop()
        assert stop_result is True
        
        # Verify cycle was stopped
        assert engine.current_cycle is None or engine.current_cycle.status == ImprovementCycleStatus.FAILED
        assert engine.scheduler_task is None or engine.scheduler_task.cancelled()
        assert engine.is_running is False
    
    @pytest.mark.asyncio
    async def test_improvement_cycle_with_multiple_files(self, self_improvement_engine):
        """Test improvement cycle affecting multiple files"""
        engine = self_improvement_engine
        
        # Create additional test file
        additional_file = Path(engine.project_root) / "app" / "another_module.py"
        additional_file.write_text('''
def another_inefficient_function():
    result = ""
    for i in range(100):
        result += str(i)
    return result
''')
        
        # Create multi-file improvement
        multi_file_improvement = Improvement(
            id="imp_multi",
            title="Multi-file optimization",
            description="Optimize string operations across multiple files",
            type=ImprovementType.PERFORMANCE,
            priority=ImprovementPriority.HIGH,
            risk_level=RiskLevel.LOW,
            impact_score=8.0,
            affected_files=["app/sample_module.py", "app/another_module.py"],
            proposed_changes={
                "app/sample_module.py": '''
def inefficient_function(data):
    """Optimized function"""
    return "".join(str(item) for item in data)

def complex_function(a, b, c, d, e, f, g, h):
    """Simplified function"""
    return sum([a, b, c, d, e, f, g, h]) if all(x > 0 for x in [a, b, c, d, e]) else 0

class SampleClass:
    def method_without_docstring(self):
        """Added docstring"""
        MAGIC_CONSTANT = 42
        return MAGIC_CONSTANT * 2
''',
                "app/another_module.py": '''
def another_inefficient_function():
    """Optimized function"""
    return "".join(str(i) for i in range(100))
'''
            },
            implementation_effort="3.0 hours"
        )
        
        # Mock improvement engine
        engine.improvement_engine.analyze_and_suggest_improvements = Mock(
            return_value=[multi_file_improvement]
        )
        
        # Trigger cycle
        cycle_id = await engine.trigger_improvement_cycle("multi_file_test")
        
        # Wait for completion
        max_wait = 30
        start_time = time.time()
        
        while engine.current_cycle and engine.current_cycle.status not in [
            ImprovementCycleStatus.COMPLETED,
            ImprovementCycleStatus.FAILED,
            ImprovementCycleStatus.ROLLED_BACK
        ]:
            if time.time() - start_time > max_wait:
                break
            await asyncio.sleep(0.5)
        
        # Verify both files were modified
        modified_sample = Path(engine.project_root) / "app" / "sample_module.py"
        modified_another = Path(engine.project_root) / "app" / "another_module.py"
        
        sample_content = modified_sample.read_text()
        another_content = modified_another.read_text()
        
        assert "Optimized function" in sample_content
        assert "Optimized function" in another_content
        assert "join" in sample_content
        assert "join" in another_content
        
        # Verify cycle completed successfully
        history = engine.get_cycle_history(limit=1)
        assert len(history) == 1
        assert history[0]["status"] == ImprovementCycleStatus.COMPLETED.value
    
    @pytest.mark.asyncio
    async def test_improvement_cycle_status_tracking(self, self_improvement_engine):
        """Test detailed status tracking throughout improvement cycle"""
        engine = self_improvement_engine
        
        # Create simple improvement
        test_improvement = Improvement(
            id="imp_status",
            title="Status tracking test",
            description="Test improvement for status tracking",
            type=ImprovementType.MAINTAINABILITY,
            priority=ImprovementPriority.MEDIUM,
            risk_level=RiskLevel.LOW,
            impact_score=6.0,
            affected_files=["app/sample_module.py"],
            proposed_changes={
                "app/sample_module.py": '''
def inefficient_function(data):
    """Function with improved documentation"""
    result_parts = []
    for item in data:
        result_parts.append(str(item))
    return "".join(result_parts)

def complex_function(a, b, c, d, e, f, g, h):
    """Function with better parameter handling"""
    params = [a, b, c, d, e, f, g, h]
    return sum(params) if all(x > 0 for x in params[:5]) else 0

class SampleClass:
    def method_without_docstring(self):
        """Method with proper documentation"""
        MAGIC_CONSTANT = 42
        return MAGIC_CONSTANT * 2
'''
            },
            implementation_effort="1.5 hours"
        )
        
        # Mock improvement engine
        engine.improvement_engine.analyze_and_suggest_improvements = Mock(
            return_value=[test_improvement]
        )
        
        # Track status changes
        status_history = []
        
        # Start cycle and monitor status
        cycle_id = await engine.trigger_improvement_cycle("status_test")
        
        # Monitor status changes
        max_wait = 30
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < max_wait:
            current_status = engine.get_current_status()
            
            if current_status["current_cycle"]:
                cycle_status = current_status["current_cycle"]["status"]
                if cycle_status != last_status:
                    status_history.append({
                        "status": cycle_status,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    last_status = cycle_status
            
            # Check if cycle completed
            if not engine.current_cycle or engine.current_cycle.status in [
                ImprovementCycleStatus.COMPLETED,
                ImprovementCycleStatus.FAILED,
                ImprovementCycleStatus.ROLLED_BACK
            ]:
                break
            
            await asyncio.sleep(0.5)
        
        # Verify we tracked the expected status progression
        expected_statuses = [
            ImprovementCycleStatus.ANALYZING.value,
            ImprovementCycleStatus.PLANNING.value,
            ImprovementCycleStatus.TESTING.value,
            ImprovementCycleStatus.APPLYING.value,
            ImprovementCycleStatus.VALIDATING.value
        ]
        
        tracked_statuses = [entry["status"] for entry in status_history]
        
        # Should have tracked at least some of the expected statuses
        assert len(tracked_statuses) > 0
        assert ImprovementCycleStatus.ANALYZING.value in tracked_statuses
        
        # Verify final completion
        history = engine.get_cycle_history(limit=1)
        assert len(history) == 1
        final_cycle = history[0]
        assert final_cycle["status"] in [
            ImprovementCycleStatus.COMPLETED.value,
            ImprovementCycleStatus.FAILED.value
        ]
        
        logger.info(f"Status progression: {tracked_statuses}")
        logger.info(f"Final status: {final_cycle['status']}")

class TestSelfImprovementSafetyMechanisms:
    """Test safety mechanisms in isolation"""
    
    @pytest.fixture
    def safety_test_env(self):
        """Create environment for safety testing"""
        temp_dir = Path(tempfile.mkdtemp(prefix="safety_test_"))
        
        # Create test files
        (temp_dir / "app").mkdir()
        
        # Create file with potential security issues
        unsafe_code = '''
import os
import subprocess

def dangerous_function(user_input):
    # This is unsafe - direct command execution
    os.system(user_input)
    
def another_dangerous_function(filename):
    # This is unsafe - arbitrary file access
    with open(filename, 'w') as f:
        f.write("dangerous content")

def sql_injection_risk(query):
    # This simulates SQL injection risk
    return f"SELECT * FROM users WHERE name = '{query}'"
'''
        
        (temp_dir / "app" / "unsafe_module.py").write_text(unsafe_code)
        
        yield temp_dir
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_safety_filter_blocks_dangerous_changes(self, safety_test_env):
        """Test that safety mechanisms block dangerous code changes"""
        engine = SelfImprovementEngine(
            project_root=str(safety_test_env),
            config={"safety_level": "conservative"}
        )
        
        # Define unsafe code
        unsafe_code = '''
import os
import subprocess

def dangerous_function(user_input):
    # This is unsafe - direct command execution
    os.system(user_input)
    
def another_dangerous_function(filename):
    # This is unsafe - arbitrary file access
    with open(filename, 'w') as f:
        f.write("dangerous content")

def sql_injection_risk(query):
    # This simulates SQL injection risk
    return f"SELECT * FROM users WHERE name = '{query}'"
'''
        
        # Create dangerous improvement
        dangerous_improvement = Improvement(
            id="imp_dangerous",
            title="Add more dangerous code",
            description="This would add more security vulnerabilities",
            type=ImprovementType.SECURITY,
            priority=ImprovementPriority.HIGH,
            risk_level=RiskLevel.VERY_HIGH,
            impact_score=9.0,
            affected_files=["app/unsafe_module.py"],
            proposed_changes={
                "app/unsafe_module.py": unsafe_code + '''

def even_more_dangerous():
    # Delete all files
    subprocess.run(["rm", "-rf", "/"])
'''
            },
            implementation_effort="1.0 hours"
        )
        
        # Test safety filtering
        filtered_improvements = engine._filter_improvements_by_safety([dangerous_improvement])
        
        # Should be filtered out in conservative mode
        assert len(filtered_improvements) == 0
        
        # Test with aggressive mode
        engine.safety_level = SafetyLevel.AGGRESSIVE
        filtered_aggressive = engine._filter_improvements_by_safety([dangerous_improvement])
        
        # Should be allowed in aggressive mode
        assert len(filtered_aggressive) == 1
    
    def test_code_validation_prevents_syntax_errors(self, safety_test_env):
        """Test that code validation prevents syntax errors"""
        engine = SelfImprovementEngine(project_root=str(safety_test_env))
        
        # Create modification with syntax error
        bad_modification = FileModification(
            file_path="app/test_file.py",
            original_content="def good_function():\n    return True",
            modified_content="def bad_function(\n    return True  # Missing closing parenthesis",
            modification_type="update"
        )
        
        # Test validation
        validation_result = engine.code_modifier.validate_modification(bad_modification)
        
        assert validation_result.is_valid is False
        assert validation_result.syntax_valid is False
        assert len(validation_result.errors) > 0
        assert any("syntax" in error.lower() for error in validation_result.errors)
    
    def test_rollback_point_creation(self, safety_test_env):
        """Test that rollback points are created before changes"""
        engine = SelfImprovementEngine(project_root=str(safety_test_env))
        
        # Initialize git repository
        git_integration = GitIntegration(str(safety_test_env))
        git_integration.init_repo_if_needed()
        
        # Create initial commit
        (safety_test_env / "test_file.txt").write_text("initial content")
        git_integration.add_files(["test_file.txt"])
        initial_commit = git_integration.commit_changes("Initial commit")
        
        # Create rollback point
        rollback_commit = git_integration.create_rollback_point("Test rollback point")
        
        assert rollback_commit is not None
        assert rollback_commit != initial_commit
        
        # Verify we can rollback
        (safety_test_env / "test_file.txt").write_text("modified content")
        git_integration.add_files(["test_file.txt"])
        git_integration.commit_changes("Modify file")
        
        # Rollback
        rollback_success = git_integration.rollback_to_commit(rollback_commit)
        assert rollback_success is True
        
        # Verify content was restored
        restored_content = (safety_test_env / "test_file.txt").read_text()
        assert restored_content == "initial content"

class TestSelfImprovementPerformanceValidation:
    """Test performance improvement validation"""
    
    @pytest.fixture
    def performance_test_env(self):
        """Create environment for performance testing"""
        temp_dir = Path(tempfile.mkdtemp(prefix="perf_test_"))
        
        # Create performance test files
        (temp_dir / "app").mkdir()
        (temp_dir / "tests").mkdir()
        
        # Create module with performance issues
        perf_code = '''
import time

def slow_function(n):
    """Intentionally slow function for testing"""
    result = 0
    for i in range(n):
        result += i * i
        time.sleep(0.001)  # Simulate slow operation
    return result

def inefficient_sort(data):
    """Bubble sort - intentionally inefficient"""
    n = len(data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data
'''
        
        (temp_dir / "app" / "performance_module.py").write_text(perf_code)
        
        # Create performance test
        perf_test = '''
import unittest
import time
from app.performance_module import slow_function, inefficient_sort

class TestPerformance(unittest.TestCase):
    def test_slow_function_performance(self):
        start_time = time.time()
        result = slow_function(10)
        duration = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(duration, 1.0)
        self.assertGreater(result, 0)
    
    def test_sort_performance(self):
        data = list(range(100, 0, -1))  # Reverse sorted
        start_time = time.time()
        sorted_data = inefficient_sort(data.copy())
        duration = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(duration, 0.1)
        self.assertEqual(sorted_data, list(range(1, 101)))

if __name__ == "__main__":
    unittest.main()
'''
        
        (temp_dir / "tests" / "test_performance.py").write_text(perf_test)
        
        yield temp_dir
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_performance_improvement_measurement(self, performance_test_env):
        """Test measurement of performance improvements"""
        engine = SelfImprovementEngine(project_root=str(performance_test_env))
        
        # Create performance improvement
        perf_improvement = Improvement(
            id="imp_perf_measure",
            title="Optimize slow functions",
            description="Replace inefficient implementations with optimized versions",
            type=ImprovementType.PERFORMANCE,
            priority=ImprovementPriority.HIGH,
            risk_level=RiskLevel.LOW,
            impact_score=8.5,
            affected_files=["app/performance_module.py"],
            proposed_changes={
                "app/performance_module.py": '''
import time

def slow_function(n):
    """Optimized function"""
    # Remove unnecessary sleep and optimize calculation
    return sum(i * i for i in range(n))

def inefficient_sort(data):
    """Use built-in efficient sort"""
    return sorted(data)
'''
            },
            implementation_effort="2.0 hours"
        )
        
        # Mock test runner to simulate performance measurements
        original_run_test_suite = engine.test_runner.run_test_suite
        
        def mock_run_test_suite(test_type=None, **kwargs):
            if test_type == TestType.PERFORMANCE:
                # Simulate performance test results
                return TestSuiteResult(
                    suite_name="performance_tests",
                    total_tests=2,
                    passed=2,
                    failed=0,
                    errors=0,
                    skipped=0,
                    duration=0.5,  # Faster than before
                    coverage_percentage=90.0,
                    test_results=[
                        TestResult(
                            test_name="test_slow_function_performance",
                            test_type=TestType.PERFORMANCE,
                            status=TestStatus.PASSED,
                            duration=0.1,  # Much faster
                            performance_metrics={
                                "execution_time": 0.1,
                                "memory_usage": 50
                            }
                        ),
                        TestResult(
                            test_name="test_sort_performance",
                            test_type=TestType.PERFORMANCE,
                            status=TestStatus.PASSED,
                            duration=0.01,  # Much faster
                            performance_metrics={
                                "execution_time": 0.01,
                                "memory_usage": 30
                            }
                        )
                    ]
                )
            else:
                return original_run_test_suite(test_type, **kwargs)
        
        engine.test_runner.run_test_suite = mock_run_test_suite
        
        # Mock baseline performance data
        engine.test_runner.baseline_data = {
            "metrics": {
                "test_slow_function_performance": {
                    "execution_time": 0.5,  # Slower baseline
                    "memory_usage": 100
                },
                "test_sort_performance": {
                    "execution_time": 0.05,  # Slower baseline
                    "memory_usage": 60
                }
            }
        }
        
        # Mock improvement engine
        engine.improvement_engine.analyze_and_suggest_improvements = Mock(
            return_value=[perf_improvement]
        )
        
        # Trigger cycle
        cycle_id = await engine.trigger_improvement_cycle("performance_validation")
        
        # Wait for completion
        max_wait = 30
        start_time = time.time()
        
        while engine.current_cycle and engine.current_cycle.status not in [
            ImprovementCycleStatus.COMPLETED,
            ImprovementCycleStatus.FAILED,
            ImprovementCycleStatus.ROLLED_BACK
        ]:
            if time.time() - start_time > max_wait:
                break
            await asyncio.sleep(0.5)
        
        # Verify performance improvement was validated
        history = engine.get_cycle_history(limit=1)
        assert len(history) == 1
        
        completed_cycle = history[0]
        assert completed_cycle["status"] == ImprovementCycleStatus.COMPLETED.value
        assert len(completed_cycle["applied_improvements"]) > 0
        
        # Verify optimized code was applied
        optimized_file = Path(performance_test_env) / "app" / "performance_module.py"
        optimized_content = optimized_file.read_text()
        assert "Optimized function" in optimized_content
        assert "sorted(data)" in optimized_content
        assert "time.sleep" not in optimized_content

if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main([__file__ + "::TestE2ESelfImprovementIntegration::test_complete_self_improvement_cycle", "-v"])