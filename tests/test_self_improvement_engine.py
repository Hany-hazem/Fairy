# tests/test_self_improvement_engine.py
"""
Unit tests for SelfImprovementEngine orchestrator
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from app.self_improvement_engine import (
    SelfImprovementEngine,
    ImprovementCycle,
    ImprovementCycleStatus,
    SafetyLevel,
    get_self_improvement_engine
)
from app.improvement_engine import Improvement, ImprovementType, ImprovementPriority, RiskLevel
from app.test_runner import TestSuiteResult, TestResult, TestType, TestStatus
from app.code_modifier import ModificationPlan, FileModification
from app.performance_monitor import PerformanceReport


class TestSelfImprovementEngine:
    """Test cases for SelfImprovementEngine"""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_components(self):
        """Mock all component dependencies"""
        with patch.multiple(
            'app.self_improvement_engine',
            PerformanceMonitor=Mock,
            PerformanceAnalyzer=Mock,
            CodeAnalyzer=Mock,
            ImprovementEngine=Mock,
            TestRunner=Mock,
            CodeModifier=Mock,
            GitIntegration=Mock
        ) as mocks:
            # Configure the mocks to return Mock instances
            for mock_class in mocks.values():
                mock_class.return_value = Mock()
            yield mocks
    
    @pytest.fixture
    def engine(self, temp_project_dir, mock_components):
        """Create SelfImprovementEngine instance with mocked components"""
        config = {
            "safety_level": "conservative",
            "auto_apply_threshold": 8.0,
            "max_concurrent_improvements": 2,
            "performance_check_interval": 60
        }
        
        engine = SelfImprovementEngine(temp_project_dir, config)
        
        # Configure mocks
        engine.performance_monitor.get_performance_report.return_value = PerformanceReport(
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow(),
            metrics=[],
            summary={},
            alerts=[]
        )
        
        engine.improvement_engine.analyze_and_suggest_improvements.return_value = [
            Improvement(
                id="test_improvement_1",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.LOW,
                title="Test Improvement 1",
                description="Test improvement description",
                affected_files=["test_file.py"],
                proposed_changes={"test_file.py": "improved content"},
                expected_benefit="Performance improvement",
                impact_score=8.5,
                confidence_score=9.0
            )
        ]
        
        engine.test_runner.run_test_suite.return_value = TestSuiteResult(
            suite_name="test_suite",
            total_tests=10,
            passed=10,
            failed=0,
            errors=0,
            skipped=0,
            duration=5.0,
            coverage_percentage=85.0,
            test_results=[]
        )
        
        engine.code_modifier.create_modification_plan.return_value = ModificationPlan(
            id="test_plan_1",
            description="Test modification plan",
            modifications=[
                FileModification(
                    file_path="test_file.py",
                    original_content="original content",
                    modified_content="improved content",
                    modification_type="update"
                )
            ],
            created_at=datetime.utcnow()
        )
        
        engine.code_modifier.apply_modification_plan.return_value = True
        engine.code_modifier.verify_integrity.return_value = True
        
        return engine
    
    def test_initialization(self, temp_project_dir):
        """Test engine initialization"""
        config = {"safety_level": "moderate"}
        
        with patch.multiple(
            'app.self_improvement_engine',
            PerformanceMonitor=Mock,
            PerformanceAnalyzer=Mock,
            CodeAnalyzer=Mock,
            ImprovementEngine=Mock,
            TestRunner=Mock,
            CodeModifier=Mock,
            GitIntegration=Mock
        ):
            engine = SelfImprovementEngine(temp_project_dir, config)
            
            assert engine.project_root == Path(temp_project_dir).resolve()
            assert engine.safety_level == SafetyLevel.MODERATE
            assert engine.current_cycle is None
            assert len(engine.cycle_history) == 0
            assert not engine.is_running
    
    @pytest.mark.asyncio
    async def test_trigger_improvement_cycle(self, engine):
        """Test triggering an improvement cycle"""
        cycle_id = await engine.trigger_improvement_cycle("manual")
        
        assert cycle_id.startswith("cycle_")
        assert engine.current_cycle is not None
        assert engine.current_cycle.id == cycle_id
        assert engine.current_cycle.trigger == "manual"
        assert engine.current_cycle.safety_level == SafetyLevel.CONSERVATIVE
    
    @pytest.mark.asyncio
    async def test_trigger_cycle_when_running(self, engine):
        """Test that triggering a cycle when one is running raises error"""
        # Start first cycle
        await engine.trigger_improvement_cycle("manual")
        
        # Try to start second cycle
        with pytest.raises(ValueError, match="Another improvement cycle is already running"):
            await engine.trigger_improvement_cycle("manual")
    
    @pytest.mark.asyncio
    async def test_analyze_system_performance(self, engine):
        """Test system performance analysis phase"""
        cycle = ImprovementCycle(
            id="test_cycle",
            status=ImprovementCycleStatus.IDLE,
            started_at=datetime.utcnow()
        )
        
        await engine._analyze_system_performance(cycle)
        
        assert cycle.status == ImprovementCycleStatus.ANALYZING
        assert cycle.performance_report is not None
        assert len(cycle.improvements) > 0
        assert cycle.total_impact_score > 0
        
        # Verify component calls
        engine.performance_monitor.collect_system_metrics.assert_called_once()
        engine.performance_monitor.get_performance_report.assert_called_once()
        engine.improvement_engine.analyze_and_suggest_improvements.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_plan_improvements(self, engine):
        """Test improvement planning phase"""
        cycle = ImprovementCycle(
            id="test_cycle",
            status=ImprovementCycleStatus.ANALYZING,
            started_at=datetime.utcnow()
        )
        
        # Add test improvements
        cycle.improvements = [
            Improvement(
                id="low_risk",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.LOW,
                title="Low Risk Improvement",
                description="Safe improvement",
                affected_files=["test.py"],
                proposed_changes={"test.py": "new content"},
                impact_score=8.0,
                confidence_score=9.0
            ),
            Improvement(
                id="high_risk",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.HIGH,
                title="High Risk Improvement",
                description="Risky improvement",
                affected_files=["test.py"],
                proposed_changes={"test.py": "risky content"},
                impact_score=9.0,
                confidence_score=7.0
            )
        ]
        
        await engine._plan_improvements(cycle)
        
        assert cycle.status == ImprovementCycleStatus.PLANNING
        # Only low-risk improvement should be selected (conservative safety level)
        assert len(cycle.selected_improvements) == 1
        assert cycle.selected_improvements[0].id == "low_risk"
        assert len(cycle.modification_plans) > 0
    
    @pytest.mark.asyncio
    async def test_test_improvements(self, engine):
        """Test improvement testing phase"""
        cycle = ImprovementCycle(
            id="test_cycle",
            status=ImprovementCycleStatus.PLANNING,
            started_at=datetime.utcnow()
        )
        
        # Add test modification plan
        plan = ModificationPlan(
            id="test_plan",
            description="Test plan",
            modifications=[],
            created_at=datetime.utcnow()
        )
        cycle.modification_plans = [plan]
        
        await engine._test_improvements(cycle)
        
        assert cycle.status == ImprovementCycleStatus.TESTING
        assert "baseline" in cycle.test_results
        assert plan.id in cycle.test_results
        
        # Verify test runner calls
        assert engine.test_runner.run_test_suite.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_apply_improvements(self, engine):
        """Test improvement application phase"""
        cycle = ImprovementCycle(
            id="test_cycle",
            status=ImprovementCycleStatus.TESTING,
            started_at=datetime.utcnow()
        )
        
        # Add test modification plan
        plan = ModificationPlan(
            id="test_plan",
            description="Test plan",
            modifications=[],
            improvement_id="test_improvement",
            created_at=datetime.utcnow()
        )
        plan.rollback_point = "commit_hash"
        cycle.modification_plans = [plan]
        
        await engine._apply_improvements(cycle)
        
        assert cycle.status == ImprovementCycleStatus.APPLYING
        assert "test_improvement" in cycle.applied_improvements
        assert "commit_hash" in cycle.rollback_points
        
        # Verify code modifier call
        engine.code_modifier.apply_modification_plan.assert_called_once_with(plan)
    
    @pytest.mark.asyncio
    async def test_validate_improvements(self, engine):
        """Test improvement validation phase"""
        cycle = ImprovementCycle(
            id="test_cycle",
            status=ImprovementCycleStatus.APPLYING,
            started_at=datetime.utcnow()
        )
        
        # Add baseline test results
        cycle.test_results["baseline"] = TestSuiteResult(
            suite_name="baseline",
            total_tests=10,
            passed=10,
            failed=0,
            errors=0,
            skipped=0,
            duration=5.0,
            coverage_percentage=85.0,
            test_results=[]
        )
        
        # Add applied improvement and plan
        cycle.applied_improvements = ["test_improvement"]
        plan = ModificationPlan(
            id="test_plan",
            description="Test plan",
            modifications=[],
            improvement_id="test_improvement",
            created_at=datetime.utcnow()
        )
        cycle.modification_plans = [plan]
        
        await engine._validate_improvements(cycle)
        
        assert cycle.status == ImprovementCycleStatus.VALIDATING
        assert "validation" in cycle.test_results
        
        # Verify validation calls
        engine.code_modifier.verify_integrity.assert_called_once_with(plan)
    
    @pytest.mark.asyncio
    async def test_rollback_cycle(self, engine):
        """Test cycle rollback functionality"""
        cycle = ImprovementCycle(
            id="test_cycle",
            status=ImprovementCycleStatus.VALIDATING,
            started_at=datetime.utcnow()
        )
        
        # Add applied improvement and plan
        cycle.applied_improvements = ["test_improvement"]
        plan = ModificationPlan(
            id="test_plan",
            description="Test plan",
            modifications=[],
            improvement_id="test_improvement",
            created_at=datetime.utcnow()
        )
        cycle.modification_plans = [plan]
        
        # Mock successful rollback
        engine.code_modifier.rollback_plan.return_value = True
        
        await engine._rollback_cycle(cycle)
        
        assert cycle.status == ImprovementCycleStatus.ROLLED_BACK
        assert "test_improvement" not in cycle.applied_improvements
        
        # Verify rollback call
        engine.code_modifier.rollback_plan.assert_called_once_with(plan.id)
    
    def test_filter_improvements_by_safety(self, engine):
        """Test improvement filtering based on safety level"""
        improvements = [
            Improvement(
                id="low_risk",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.LOW,
                title="Low Risk",
                description="Safe",
                affected_files=[],
                impact_score=8.0,
                confidence_score=9.0
            ),
            Improvement(
                id="medium_risk",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.MEDIUM,
                title="Medium Risk",
                description="Moderate",
                affected_files=[],
                impact_score=8.5,
                confidence_score=8.0
            ),
            Improvement(
                id="high_risk",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.HIGH,
                title="High Risk",
                description="Risky",
                affected_files=[],
                impact_score=9.0,
                confidence_score=7.0
            )
        ]
        
        # Test conservative filtering
        engine.safety_level = SafetyLevel.CONSERVATIVE
        filtered = engine._filter_improvements_by_safety(improvements)
        assert len(filtered) == 1
        assert filtered[0].id == "low_risk"
        
        # Test moderate filtering
        engine.safety_level = SafetyLevel.MODERATE
        filtered = engine._filter_improvements_by_safety(improvements)
        assert len(filtered) == 2
        assert {imp.id for imp in filtered} == {"low_risk", "medium_risk"}
        
        # Test aggressive filtering
        engine.safety_level = SafetyLevel.AGGRESSIVE
        filtered = engine._filter_improvements_by_safety(improvements)
        assert len(filtered) == 3
    
    def test_select_improvements(self, engine):
        """Test improvement selection logic"""
        improvements = [
            Improvement(
                id="high_impact_low_risk",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.LOW,
                title="High Impact Low Risk",
                description="Best option",
                affected_files=[],
                impact_score=9.0,
                confidence_score=9.0
            ),
            Improvement(
                id="medium_impact_low_risk",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.MEDIUM,
                risk_level=RiskLevel.LOW,
                title="Medium Impact Low Risk",
                description="Good option",
                affected_files=[],
                impact_score=7.0,
                confidence_score=8.0
            ),
            Improvement(
                id="high_impact_high_risk",
                type=ImprovementType.PERFORMANCE,
                priority=ImprovementPriority.HIGH,
                risk_level=RiskLevel.HIGH,
                title="High Impact High Risk",
                description="Risky option",
                affected_files=[],
                impact_score=9.5,
                confidence_score=7.0
            )
        ]
        
        selected = engine._select_improvements(improvements, max_count=2)
        
        assert len(selected) == 2
        # Should select high impact low risk first, then medium impact low risk
        assert selected[0].id == "high_impact_low_risk"
        assert selected[1].id == "medium_impact_low_risk"
    
    def test_create_modifications_from_improvement(self, engine):
        """Test creating file modifications from improvement"""
        improvement = Improvement(
            id="test_improvement",
            type=ImprovementType.PERFORMANCE,
            priority=ImprovementPriority.HIGH,
            risk_level=RiskLevel.LOW,
            title="Test Improvement",
            description="Test",
            affected_files=["new_file.py", "existing_file.py"],
            proposed_changes={
                "new_file.py": "new file content",
                "existing_file.py": "updated content"
            },
            impact_score=8.0,
            confidence_score=9.0
        )
        
        # Create existing file
        existing_file = Path(engine.project_root) / "existing_file.py"
        existing_file.parent.mkdir(parents=True, exist_ok=True)
        existing_file.write_text("original content")
        
        modifications = engine._create_modifications_from_improvement(improvement)
        
        assert len(modifications) == 2
        
        # Check new file modification
        new_file_mod = next(mod for mod in modifications if mod.file_path == "new_file.py")
        assert new_file_mod.modification_type == "create"
        assert new_file_mod.original_content == ""
        assert new_file_mod.modified_content == "new file content"
        
        # Check existing file modification
        existing_file_mod = next(mod for mod in modifications if mod.file_path == "existing_file.py")
        assert existing_file_mod.modification_type == "update"
        assert existing_file_mod.original_content == "original content"
        assert existing_file_mod.modified_content == "updated content"
    
    def test_get_current_status(self, engine):
        """Test getting current engine status"""
        status = engine.get_current_status()
        
        assert "is_running" in status
        assert "current_cycle" in status
        assert "safety_level" in status
        assert "scheduler_running" in status
        assert "last_performance_check" in status
        assert "total_cycles" in status
        assert "successful_cycles" in status
        
        assert status["safety_level"] == "conservative"
        assert status["current_cycle"] is None
        assert status["total_cycles"] == 0
    
    def test_get_cycle_history(self, engine):
        """Test getting cycle history"""
        # Add some test cycles to history
        for i in range(5):
            cycle = ImprovementCycle(
                id=f"cycle_{i}",
                status=ImprovementCycleStatus.COMPLETED,
                started_at=datetime.utcnow() - timedelta(hours=i)
            )
            engine.cycle_history.append(cycle)
        
        history = engine.get_cycle_history(limit=3)
        
        assert len(history) == 3
        # Should be sorted by start time (newest first)
        assert history[0]["id"] == "cycle_0"
        assert history[1]["id"] == "cycle_1"
        assert history[2]["id"] == "cycle_2"
    
    def test_get_cycle_details(self, engine):
        """Test getting detailed cycle information"""
        cycle = ImprovementCycle(
            id="test_cycle",
            status=ImprovementCycleStatus.COMPLETED,
            started_at=datetime.utcnow()
        )
        engine.cycle_history.append(cycle)
        
        details = engine.get_cycle_details("test_cycle")
        
        assert details is not None
        assert details["id"] == "test_cycle"
        assert "improvements" in details
        assert "selected_improvements" in details
        assert "test_results" in details
        assert "modification_plans" in details
        
        # Test non-existent cycle
        assert engine.get_cycle_details("non_existent") is None
    
    @pytest.mark.asyncio
    async def test_emergency_stop(self, engine):
        """Test emergency stop functionality"""
        # Start a cycle
        await engine.trigger_improvement_cycle("manual")
        assert engine.current_cycle is not None
        
        # Trigger emergency stop
        result = await engine.emergency_stop()
        
        assert result is True
        assert not engine.is_running
        assert engine.current_cycle.status == ImprovementCycleStatus.FAILED
        assert "Emergency stop triggered" in engine.current_cycle.error_messages
    
    def test_update_safety_level(self, engine):
        """Test updating safety level"""
        assert engine.safety_level == SafetyLevel.CONSERVATIVE
        
        engine.update_safety_level(SafetyLevel.AGGRESSIVE)
        
        assert engine.safety_level == SafetyLevel.AGGRESSIVE
    
    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, engine):
        """Test scheduler start and stop"""
        # Start scheduler
        await engine.start_scheduler()
        assert engine.scheduler_task is not None
        assert not engine.scheduler_task.done()
        
        # Stop scheduler
        await engine.stop_scheduler()
        assert engine.scheduler_task is None or engine.scheduler_task.done()
    
    def test_should_run_performance_check(self, engine):
        """Test performance check timing logic"""
        # Recent check - should not run
        engine.last_performance_check = datetime.utcnow()
        assert not engine._should_run_performance_check()
        
        # Old check - should run
        engine.last_performance_check = datetime.utcnow() - timedelta(hours=2)
        assert engine._should_run_performance_check()
    
    @pytest.mark.asyncio
    async def test_check_performance_thresholds(self, engine):
        """Test performance threshold checking"""
        # No alerts - should not trigger
        engine.performance_monitor.get_performance_report.return_value.alerts = []
        result = await engine._check_performance_thresholds()
        assert result is False
        
        # With threshold alert - should trigger
        engine.performance_monitor.get_performance_report.return_value.alerts = [
            "response_time: 5 threshold violations (>2.0s)"
        ]
        result = await engine._check_performance_thresholds()
        assert result is True


class TestGlobalInstance:
    """Test global instance management"""
    
    def test_get_self_improvement_engine(self):
        """Test getting global engine instance"""
        # Clear global instance
        import app.self_improvement_engine
        app.self_improvement_engine.self_improvement_engine = None
        
        with patch.multiple(
            'app.self_improvement_engine',
            PerformanceMonitor=Mock,
            PerformanceAnalyzer=Mock,
            CodeAnalyzer=Mock,
            ImprovementEngine=Mock,
            TestRunner=Mock,
            CodeModifier=Mock,
            GitIntegration=Mock
        ):
            # First call should create instance
            engine1 = get_self_improvement_engine()
            assert engine1 is not None
            
            # Second call should return same instance
            engine2 = get_self_improvement_engine()
            assert engine1 is engine2
    
    def teardown_method(self):
        """Clean up global instance after each test"""
        import app.self_improvement_engine
        app.self_improvement_engine.self_improvement_engine = None


if __name__ == "__main__":
    pytest.main([__file__])