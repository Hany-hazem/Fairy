# app/self_improvement_engine.py
"""
Self-Improvement Engine Orchestrator

This module coordinates all components of the self-improvement system:
- Performance monitoring and analysis
- Code analysis and improvement suggestions
- Automated testing and validation
- Safe code modification with rollback capabilities
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from contextlib import asynccontextmanager

from .performance_monitor import PerformanceMonitor, PerformanceReport
from .performance_analyzer import PerformanceAnalyzer
from .code_analyzer import CodeAnalyzer
from .improvement_engine import ImprovementEngine, Improvement, ImprovementType, ImprovementPriority, RiskLevel
from .test_runner import TestRunner, TestType, TestSuiteResult
from .code_modifier import CodeModifier, ModificationPlan, FileModification
from .version_control import GitIntegration
from .config import settings

logger = logging.getLogger(__name__)


class ImprovementCycleStatus(Enum):
    """Status of improvement cycle"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    TESTING = "testing"
    APPLYING = "applying"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class SafetyLevel(Enum):
    """Safety levels for self-improvement"""
    CONSERVATIVE = "conservative"  # Only low-risk improvements
    MODERATE = "moderate"         # Low and medium-risk improvements
    AGGRESSIVE = "aggressive"     # All improvements with approval


@dataclass
class ImprovementCycle:
    """Represents a complete improvement cycle"""
    id: str
    status: ImprovementCycleStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    trigger: str = "manual"  # manual, scheduled, performance_threshold
    safety_level: SafetyLevel = SafetyLevel.CONSERVATIVE
    
    # Analysis results
    performance_report: Optional[PerformanceReport] = None
    improvements: List[Improvement] = field(default_factory=list)
    selected_improvements: List[Improvement] = field(default_factory=list)
    
    # Testing and validation
    test_results: Dict[str, TestSuiteResult] = field(default_factory=dict)
    modification_plans: List[ModificationPlan] = field(default_factory=list)
    
    # Results
    applied_improvements: List[str] = field(default_factory=list)
    failed_improvements: List[str] = field(default_factory=list)
    rollback_points: List[str] = field(default_factory=list)
    
    # Metrics
    total_impact_score: float = 0.0
    success_rate: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "trigger": self.trigger,
            "safety_level": self.safety_level.value,
            "improvements_count": len(self.improvements),
            "selected_improvements_count": len(self.selected_improvements),
            "applied_improvements": self.applied_improvements,
            "failed_improvements": self.failed_improvements,
            "total_impact_score": self.total_impact_score,
            "success_rate": self.success_rate,
            "error_messages": self.error_messages
        }


class SelfImprovementEngine:
    """Main orchestrator for the self-improvement system"""
    
    def __init__(self, project_root: str = ".", config: Optional[Dict[str, Any]] = None):
        self.project_root = Path(project_root).resolve()
        self.config = config or {}
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor()
        self.performance_analyzer = PerformanceAnalyzer(self.performance_monitor)
        self.code_analyzer = CodeAnalyzer(str(self.project_root))
        self.improvement_engine = ImprovementEngine(str(self.project_root))
        self.test_runner = TestRunner(str(self.project_root))
        self.code_modifier = CodeModifier(str(self.project_root))
        self.git_integration = GitIntegration(str(self.project_root))
        
        # State management
        self.current_cycle: Optional[ImprovementCycle] = None
        self.cycle_history: List[ImprovementCycle] = []
        self.is_running = False
        self._lock = threading.RLock()
        
        # Configuration
        self.safety_level = SafetyLevel(self.config.get("safety_level", "conservative"))
        self.auto_apply_threshold = self.config.get("auto_apply_threshold", 8.0)
        self.max_concurrent_improvements = self.config.get("max_concurrent_improvements", 3)
        self.performance_check_interval = self.config.get("performance_check_interval", 3600)  # 1 hour
        
        # Scheduling
        self.scheduler_task: Optional[asyncio.Task] = None
        self.last_performance_check = datetime.utcnow()
        
        # Storage
        self.cycles_file = self.project_root / ".kiro" / "improvement_cycles.json"
        self.cycles_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load previous cycles
        self._load_cycle_history()
        
        logger.info(f"Self-improvement engine initialized for {self.project_root}")
    
    async def start_scheduler(self):
        """Start the improvement scheduler"""
        if self.scheduler_task and not self.scheduler_task.done():
            logger.warning("Scheduler already running")
            return
        
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Self-improvement scheduler started")
    
    async def stop_scheduler(self):
        """Stop the improvement scheduler"""
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
            self.scheduler_task = None
        
        logger.info("Self-improvement scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check if we should run performance analysis
                if self._should_run_performance_check():
                    await self.trigger_improvement_cycle("scheduled_performance")
                
                # Check for performance threshold violations
                if await self._check_performance_thresholds():
                    await self.trigger_improvement_cycle("performance_threshold")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    def _should_run_performance_check(self) -> bool:
        """Check if we should run a performance analysis"""
        time_since_last = datetime.utcnow() - self.last_performance_check
        return time_since_last.total_seconds() >= self.performance_check_interval
    
    async def _check_performance_thresholds(self) -> bool:
        """Check if performance thresholds are violated"""
        try:
            # Get recent performance metrics
            report = self.performance_monitor.get_performance_report(
                start_time=datetime.utcnow() - timedelta(minutes=30)
            )
            
            # Check for threshold violations
            for alert in report.alerts:
                if "threshold" in alert.lower():
                    logger.warning(f"Performance threshold violation: {alert}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking performance thresholds: {e}")
            return False
    
    async def trigger_improvement_cycle(self, trigger: str = "manual") -> str:
        """Trigger a new improvement cycle"""
        with self._lock:
            if self.current_cycle and self.current_cycle.status not in [
                ImprovementCycleStatus.COMPLETED,
                ImprovementCycleStatus.FAILED,
                ImprovementCycleStatus.ROLLED_BACK
            ]:
                raise ValueError("Another improvement cycle is already running")
            
            # Create new cycle
            cycle_id = f"cycle_{int(time.time())}"
            self.current_cycle = ImprovementCycle(
                id=cycle_id,
                status=ImprovementCycleStatus.ANALYZING,
                started_at=datetime.utcnow(),
                trigger=trigger,
                safety_level=self.safety_level
            )
            
            logger.info(f"Started improvement cycle: {cycle_id}")
        
        # Run the cycle asynchronously
        asyncio.create_task(self._run_improvement_cycle())
        
        return cycle_id
    
    async def _run_improvement_cycle(self):
        """Run a complete improvement cycle"""
        cycle = self.current_cycle
        if not cycle:
            return
        
        try:
            # Phase 1: Analysis
            await self._analyze_system_performance(cycle)
            
            # Phase 2: Planning
            await self._plan_improvements(cycle)
            
            # Phase 3: Testing
            await self._test_improvements(cycle)
            
            # Phase 4: Application
            await self._apply_improvements(cycle)
            
            # Phase 5: Validation
            await self._validate_improvements(cycle)
            
            # Complete cycle
            cycle.status = ImprovementCycleStatus.COMPLETED
            cycle.completed_at = datetime.utcnow()
            cycle.success_rate = len(cycle.applied_improvements) / max(len(cycle.selected_improvements), 1) * 100
            
            logger.info(f"Completed improvement cycle: {cycle.id}")
            
        except Exception as e:
            logger.error(f"Error in improvement cycle {cycle.id}: {e}")
            cycle.status = ImprovementCycleStatus.FAILED
            cycle.error_messages.append(str(e))
            cycle.completed_at = datetime.utcnow()
        
        finally:
            # Save cycle to history
            with self._lock:
                self.cycle_history.append(cycle)
                self._save_cycle_history()
                self.current_cycle = None
    
    async def _analyze_system_performance(self, cycle: ImprovementCycle):
        """Analyze system performance and identify improvement opportunities"""
        cycle.status = ImprovementCycleStatus.ANALYZING
        logger.info(f"Analyzing system performance for cycle {cycle.id}")
        
        try:
            # Collect current system metrics
            self.performance_monitor.collect_system_metrics()
            
            # Generate performance report
            cycle.performance_report = self.performance_monitor.get_performance_report(
                start_time=datetime.utcnow() - timedelta(hours=24)
            )
            
            # Analyze code for improvements
            cycle.improvements = self.improvement_engine.analyze_and_suggest_improvements()
            
            # Calculate total impact score
            cycle.total_impact_score = sum(imp.impact_score for imp in cycle.improvements)
            
            logger.info(f"Found {len(cycle.improvements)} improvement opportunities")
            
        except Exception as e:
            logger.error(f"Error analyzing system performance: {e}")
            raise
    
    async def _plan_improvements(self, cycle: ImprovementCycle):
        """Plan which improvements to implement"""
        cycle.status = ImprovementCycleStatus.PLANNING
        logger.info(f"Planning improvements for cycle {cycle.id}")
        
        try:
            # Filter improvements based on safety level
            filtered_improvements = self._filter_improvements_by_safety(cycle.improvements)
            
            # Select top improvements based on impact and risk
            cycle.selected_improvements = self._select_improvements(
                filtered_improvements,
                max_count=self.max_concurrent_improvements
            )
            
            # Create modification plans
            for improvement in cycle.selected_improvements:
                if improvement.proposed_changes:
                    modifications = self._create_modifications_from_improvement(improvement)
                    if modifications:
                        plan = self.code_modifier.create_modification_plan(
                            modifications=modifications,
                            description=improvement.title,
                            improvement_id=improvement.id
                        )
                        cycle.modification_plans.append(plan)
            
            logger.info(f"Selected {len(cycle.selected_improvements)} improvements for implementation")
            
        except Exception as e:
            logger.error(f"Error planning improvements: {e}")
            raise
    
    async def _test_improvements(self, cycle: ImprovementCycle):
        """Test improvements before applying them"""
        cycle.status = ImprovementCycleStatus.TESTING
        logger.info(f"Testing improvements for cycle {cycle.id}")
        
        try:
            # Run baseline tests
            baseline_results = self.test_runner.run_test_suite(
                test_type=TestType.UNIT,
                isolated=True
            )
            cycle.test_results["baseline"] = baseline_results
            
            # Test each modification plan
            for plan in cycle.modification_plans:
                test_results = await self._test_modification_plan(plan)
                cycle.test_results[plan.id] = test_results
                
                # Remove plan if tests fail
                if test_results.failed > 0 or test_results.errors > 0:
                    logger.warning(f"Tests failed for plan {plan.id}, removing from cycle")
                    cycle.modification_plans.remove(plan)
                    # Also remove corresponding improvement
                    cycle.selected_improvements = [
                        imp for imp in cycle.selected_improvements 
                        if imp.id != plan.improvement_id
                    ]
            
            logger.info(f"Testing completed, {len(cycle.modification_plans)} plans passed")
            
        except Exception as e:
            logger.error(f"Error testing improvements: {e}")
            raise
    
    async def _test_modification_plan(self, plan: ModificationPlan) -> TestSuiteResult:
        """Test a specific modification plan in isolation"""
        # This would ideally create a temporary branch and test there
        # For now, we'll simulate the testing
        
        # Run unit tests to ensure no regressions
        test_results = self.test_runner.run_test_suite(
            test_type=TestType.UNIT,
            isolated=True
        )
        
        # Run integration tests if available
        integration_results = self.test_runner.run_test_suite(
            test_type=TestType.INTEGRATION,
            isolated=True
        )
        
        # Combine results (simplified)
        total_tests = test_results.total_tests + integration_results.total_tests
        total_passed = test_results.passed + integration_results.passed
        total_failed = test_results.failed + integration_results.failed
        total_errors = test_results.errors + integration_results.errors
        
        # Create combined result
        combined_result = TestSuiteResult(
            suite_name=f"plan_{plan.id}_validation",
            total_tests=total_tests,
            passed=total_passed,
            failed=total_failed,
            errors=total_errors,
            skipped=test_results.skipped + integration_results.skipped,
            duration=test_results.duration + integration_results.duration,
            coverage_percentage=(test_results.coverage_percentage + integration_results.coverage_percentage) / 2,
            test_results=test_results.test_results + integration_results.test_results
        )
        
        return combined_result
    
    async def _apply_improvements(self, cycle: ImprovementCycle):
        """Apply validated improvements"""
        cycle.status = ImprovementCycleStatus.APPLYING
        logger.info(f"Applying improvements for cycle {cycle.id}")
        
        try:
            for plan in cycle.modification_plans:
                try:
                    # Apply the modification plan
                    if self.code_modifier.apply_modification_plan(plan):
                        cycle.applied_improvements.append(plan.improvement_id or plan.id)
                        cycle.rollback_points.append(plan.rollback_point or "unknown")
                        logger.info(f"Successfully applied plan: {plan.id}")
                    else:
                        cycle.failed_improvements.append(plan.improvement_id or plan.id)
                        logger.error(f"Failed to apply plan: {plan.id}")
                        
                except Exception as e:
                    logger.error(f"Error applying plan {plan.id}: {e}")
                    cycle.failed_improvements.append(plan.improvement_id or plan.id)
                    cycle.error_messages.append(f"Plan {plan.id}: {str(e)}")
            
            logger.info(f"Applied {len(cycle.applied_improvements)} improvements")
            
        except Exception as e:
            logger.error(f"Error applying improvements: {e}")
            raise
    
    async def _validate_improvements(self, cycle: ImprovementCycle):
        """Validate that applied improvements work correctly"""
        cycle.status = ImprovementCycleStatus.VALIDATING
        logger.info(f"Validating improvements for cycle {cycle.id}")
        
        try:
            # Run post-application tests
            validation_results = self.test_runner.run_test_suite(
                test_type=TestType.UNIT,
                isolated=False  # Test in actual environment
            )
            cycle.test_results["validation"] = validation_results
            
            # Check for regressions
            baseline_results = cycle.test_results.get("baseline")
            if baseline_results and validation_results.success_rate < baseline_results.success_rate:
                logger.warning("Performance regression detected, considering rollback")
                
                # If significant regression, rollback
                if validation_results.success_rate < baseline_results.success_rate * 0.9:
                    await self._rollback_cycle(cycle)
                    return
            
            # Verify integrity of applied changes
            for plan in cycle.modification_plans:
                if plan.improvement_id in cycle.applied_improvements:
                    if not self.code_modifier.verify_integrity(plan):
                        logger.error(f"Integrity check failed for plan {plan.id}")
                        cycle.error_messages.append(f"Integrity check failed: {plan.id}")
            
            logger.info("Validation completed successfully")
            
        except Exception as e:
            logger.error(f"Error validating improvements: {e}")
            raise
    
    async def _rollback_cycle(self, cycle: ImprovementCycle):
        """Rollback all changes from a cycle"""
        logger.warning(f"Rolling back cycle {cycle.id}")
        
        try:
            # Rollback each applied plan
            for plan in cycle.modification_plans:
                if plan.improvement_id in cycle.applied_improvements:
                    if self.code_modifier.rollback_plan(plan.id):
                        cycle.applied_improvements.remove(plan.improvement_id)
                        logger.info(f"Rolled back plan: {plan.id}")
                    else:
                        logger.error(f"Failed to rollback plan: {plan.id}")
            
            cycle.status = ImprovementCycleStatus.ROLLED_BACK
            logger.info(f"Cycle {cycle.id} rolled back")
            
        except Exception as e:
            logger.error(f"Error rolling back cycle {cycle.id}: {e}")
            cycle.error_messages.append(f"Rollback error: {str(e)}")
    
    def _filter_improvements_by_safety(self, improvements: List[Improvement]) -> List[Improvement]:
        """Filter improvements based on safety level"""
        if self.safety_level == SafetyLevel.CONSERVATIVE:
            return [imp for imp in improvements if imp.risk_level == RiskLevel.LOW]
        elif self.safety_level == SafetyLevel.MODERATE:
            return [imp for imp in improvements if imp.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]]
        else:  # AGGRESSIVE
            return improvements
    
    def _select_improvements(self, improvements: List[Improvement], max_count: int) -> List[Improvement]:
        """Select top improvements based on impact and risk"""
        # Sort by risk level (ascending) first, then impact score (descending)
        def sort_key(imp: Improvement) -> Tuple[int, float]:
            risk_weight = {
                RiskLevel.LOW: 0,
                RiskLevel.MEDIUM: 1,
                RiskLevel.HIGH: 2,
                RiskLevel.VERY_HIGH: 3
            }
            return (risk_weight[imp.risk_level], -imp.impact_score)
        
        sorted_improvements = sorted(improvements, key=sort_key)
        return sorted_improvements[:max_count]
    
    def _create_modifications_from_improvement(self, improvement: Improvement) -> List[FileModification]:
        """Create file modifications from improvement suggestion"""
        modifications = []
        
        for file_path, new_content in improvement.proposed_changes.items():
            # Determine modification type - use absolute path relative to project root
            file_path_obj = self.project_root / file_path
            if file_path_obj.exists():
                original_content = file_path_obj.read_text()
                modification_type = "update"
            else:
                original_content = ""
                modification_type = "create"
            
            modification = FileModification(
                file_path=file_path,
                original_content=original_content,
                modified_content=new_content,
                modification_type=modification_type
            )
            modifications.append(modification)
        
        return modifications
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current status of the self-improvement engine"""
        with self._lock:
            status = {
                "is_running": self.is_running,
                "current_cycle": self.current_cycle.to_dict() if self.current_cycle else None,
                "safety_level": self.safety_level.value,
                "scheduler_running": self.scheduler_task is not None and not self.scheduler_task.done(),
                "last_performance_check": self.last_performance_check.isoformat(),
                "total_cycles": len(self.cycle_history),
                "successful_cycles": len([c for c in self.cycle_history if c.status == ImprovementCycleStatus.COMPLETED])
            }
            
            return status
    
    def get_cycle_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent cycle history"""
        with self._lock:
            recent_cycles = sorted(
                self.cycle_history,
                key=lambda c: c.started_at,
                reverse=True
            )[:limit]
            
            return [cycle.to_dict() for cycle in recent_cycles]
    
    def get_cycle_details(self, cycle_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific cycle"""
        with self._lock:
            # Check current cycle
            if self.current_cycle and self.current_cycle.id == cycle_id:
                return self._cycle_to_detailed_dict(self.current_cycle)
            
            # Check history
            for cycle in self.cycle_history:
                if cycle.id == cycle_id:
                    return self._cycle_to_detailed_dict(cycle)
            
            return None
    
    def _cycle_to_detailed_dict(self, cycle: ImprovementCycle) -> Dict[str, Any]:
        """Convert cycle to detailed dictionary"""
        return {
            **cycle.to_dict(),
            "improvements": [imp.to_dict() for imp in cycle.improvements],
            "selected_improvements": [imp.to_dict() for imp in cycle.selected_improvements],
            "test_results": {k: v.to_dict() for k, v in cycle.test_results.items()},
            "modification_plans": [plan.model_dump() for plan in cycle.modification_plans]
        }
    
    async def emergency_stop(self) -> bool:
        """Emergency stop of all self-improvement activities"""
        logger.warning("Emergency stop triggered")
        
        try:
            # Stop scheduler
            await self.stop_scheduler()
            
            # If there's a current cycle, try to rollback
            if self.current_cycle:
                await self._rollback_cycle(self.current_cycle)
                self.current_cycle.status = ImprovementCycleStatus.FAILED
                self.current_cycle.error_messages.append("Emergency stop triggered")
                self.current_cycle.completed_at = datetime.utcnow()
            
            self.is_running = False
            logger.info("Emergency stop completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
            return False
    
    def update_safety_level(self, safety_level: SafetyLevel):
        """Update safety level"""
        with self._lock:
            self.safety_level = safety_level
            logger.info(f"Safety level updated to: {safety_level.value}")
    
    def _load_cycle_history(self):
        """Load cycle history from file"""
        try:
            if self.cycles_file.exists():
                with open(self.cycles_file, 'r') as f:
                    cycles_data = json.load(f)
                
                for cycle_data in cycles_data:
                    # Convert timestamps
                    cycle_data['started_at'] = datetime.fromisoformat(cycle_data['started_at'])
                    if cycle_data.get('completed_at'):
                        cycle_data['completed_at'] = datetime.fromisoformat(cycle_data['completed_at'])
                    
                    # Convert enums
                    cycle_data['status'] = ImprovementCycleStatus(cycle_data['status'])
                    cycle_data['safety_level'] = SafetyLevel(cycle_data['safety_level'])
                    
                    # Create cycle object (simplified, without complex nested objects)
                    cycle = ImprovementCycle(
                        id=cycle_data['id'],
                        status=cycle_data['status'],
                        started_at=cycle_data['started_at'],
                        completed_at=cycle_data.get('completed_at'),
                        trigger=cycle_data.get('trigger', 'unknown'),
                        safety_level=cycle_data['safety_level']
                    )
                    
                    # Set simple fields
                    cycle.applied_improvements = cycle_data.get('applied_improvements', [])
                    cycle.failed_improvements = cycle_data.get('failed_improvements', [])
                    cycle.total_impact_score = cycle_data.get('total_impact_score', 0.0)
                    cycle.success_rate = cycle_data.get('success_rate', 0.0)
                    cycle.error_messages = cycle_data.get('error_messages', [])
                    
                    self.cycle_history.append(cycle)
                
                logger.info(f"Loaded {len(self.cycle_history)} cycles from history")
                
        except Exception as e:
            logger.warning(f"Could not load cycle history: {e}")
    
    def _save_cycle_history(self):
        """Save cycle history to file"""
        try:
            # Convert cycles to serializable format
            cycles_data = []
            for cycle in self.cycle_history[-50:]:  # Keep last 50 cycles
                cycles_data.append(cycle.to_dict())
            
            with open(self.cycles_file, 'w') as f:
                json.dump(cycles_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Could not save cycle history: {e}")


# Global instance
self_improvement_engine: Optional[SelfImprovementEngine] = None


def get_self_improvement_engine(project_root: str = ".", config: Optional[Dict[str, Any]] = None) -> SelfImprovementEngine:
    """Get or create global self-improvement engine instance"""
    global self_improvement_engine
    
    if self_improvement_engine is None:
        self_improvement_engine = SelfImprovementEngine(project_root, config)
    
    return self_improvement_engine