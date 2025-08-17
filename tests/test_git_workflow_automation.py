"""
Tests for Git Workflow Automation Service

This module contains comprehensive tests for the Git workflow automation
and monitoring systems.
"""

import pytest
import asyncio
import tempfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.git_workflow_integration import GitWorkflowIntegration
from app.git_workflow_automation_service import (
    GitWorkflowAutomationService, WorkflowEvent, WorkflowEventType, WorkflowTrigger
)
from app.git_workflow_monitor import GitWorkflowMonitor, HealthStatus, RecoveryAction
from app.git_workflow_manager import GitWorkflowManager
from app.task_git_bridge import TaskGitBridge
from app.task_dependency_manager import TaskDependencyManager


class TestGitWorkflowAutomationService:
    """Test cases for Git Workflow Automation Service"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary Git repository for testing"""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)
        
        # Initialize Git repository
        subprocess.run(['git', 'init'], cwd=repo_path, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo_path, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo_path, check=True)
        
        # Create initial commit
        (repo_path / 'README.md').write_text('# Test Repository')
        subprocess.run(['git', 'add', 'README.md'], cwd=repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=repo_path, check=True)
        
        yield str(repo_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def automation_service(self, temp_repo):
        """Create automation service for testing"""
        git_manager = GitWorkflowManager(temp_repo)
        task_git_bridge = TaskGitBridge(git_manager)
        dependency_manager = TaskDependencyManager(task_git_bridge, git_manager)
        
        service = GitWorkflowAutomationService(
            git_manager, task_git_bridge, dependency_manager
        )
        
        await service.start()
        yield service
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, temp_repo):
        """Test service initialization"""
        git_manager = GitWorkflowManager(temp_repo)
        task_git_bridge = TaskGitBridge(git_manager)
        dependency_manager = TaskDependencyManager(task_git_bridge, git_manager)
        
        service = GitWorkflowAutomationService(
            git_manager, task_git_bridge, dependency_manager
        )
        
        assert service.git_manager is not None
        assert service.task_git_bridge is not None
        assert service.dependency_manager is not None
        assert len(service.triggers) > 0  # Should have default triggers
        assert service.config is not None
    
    @pytest.mark.asyncio
    async def test_event_emission_and_processing(self, automation_service):
        """Test event emission and processing"""
        # Create test event
        event = WorkflowEvent(
            event_type=WorkflowEventType.TASK_STARTED,
            task_id="test-task-1",
            metadata={'task_name': 'Test Task', 'task_description': 'Test Description'}
        )
        
        # Emit event
        await automation_service.emit_event(event)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check event was added to history
        history = automation_service.get_event_history()
        assert len(history) > 0
        assert history[-1]['task_id'] == 'test-task-1'
        assert history[-1]['event_type'] == 'task_started'
    
    @pytest.mark.asyncio
    async def test_task_started_trigger(self, automation_service):
        """Test task started trigger creates branch"""
        # Trigger task started event
        await automation_service.trigger_task_started(
            task_id="test-task-2",
            task_name="Test Task 2",
            task_description="Test task for automation",
            requirements=["req-1", "req-2"]
        )
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Check that branch was created and task was linked
        mapping = automation_service.task_git_bridge.get_task_mapping("test-task-2")
        assert mapping is not None
        assert mapping.task_id == "test-task-2"
        assert mapping.branch_name is not None
        assert "test-task-2" in mapping.branch_name.lower()
    
    @pytest.mark.asyncio
    async def test_task_progress_trigger(self, automation_service):
        """Test task progress trigger creates commits"""
        # First start a task
        await automation_service.trigger_task_started(
            task_id="test-task-3",
            task_name="Test Task 3"
        )
        await asyncio.sleep(2)
        
        # Create a test file to commit
        test_file = Path(automation_service.git_manager.repo_path) / "test_file.txt"
        test_file.write_text("Test content")
        
        # Trigger progress event
        await automation_service.trigger_task_progress(
            task_id="test-task-3",
            files_changed=["test_file.txt"],
            progress_notes="Added test file"
        )
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Check that commit was created
        mapping = automation_service.task_git_bridge.get_task_mapping("test-task-3")
        assert mapping is not None
        assert len(mapping.commits) > 0
    
    @pytest.mark.asyncio
    async def test_task_completed_trigger(self, automation_service):
        """Test task completed trigger"""
        # Start and progress a task
        await automation_service.trigger_task_started("test-task-4", "Test Task 4")
        await asyncio.sleep(2)
        
        # Complete the task
        await automation_service.trigger_task_completed(
            task_id="test-task-4",
            completion_notes="Task completed successfully",
            requirements_addressed=["req-1", "req-2"]
        )
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Check task status
        mapping = automation_service.task_git_bridge.get_task_mapping("test-task-4")
        assert mapping is not None
        # Note: Status might not be COMPLETED yet as it depends on Git operations
    
    @pytest.mark.asyncio
    async def test_trigger_management(self, automation_service):
        """Test adding and removing triggers"""
        initial_count = len(automation_service.triggers)
        
        # Add custom trigger
        custom_trigger = WorkflowTrigger(
            event_type=WorkflowEventType.TASK_STARTED,
            handler=AsyncMock(),
            priority=5
        )
        
        automation_service.add_trigger(custom_trigger)
        assert len(automation_service.triggers) == initial_count + 1
        
        # Remove triggers
        removed = automation_service.remove_trigger(WorkflowEventType.TASK_STARTED)
        assert removed
        assert len(automation_service.triggers) < initial_count + 1
    
    @pytest.mark.asyncio
    async def test_service_status(self, automation_service):
        """Test service status reporting"""
        status = automation_service.get_service_status()
        
        assert 'running' in status
        assert 'triggers_count' in status
        assert 'events_processed' in status
        assert 'last_health_check' in status
        assert 'config' in status
        
        assert status['running'] is True
        assert status['triggers_count'] > 0
    
    @pytest.mark.asyncio
    async def test_manual_trigger(self, automation_service):
        """Test manual event triggering"""
        success = await automation_service.manual_trigger(
            event_type="task_started",
            task_id="manual-task",
            metadata={'task_name': 'Manual Task'}
        )
        
        assert success
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check event was processed
        history = automation_service.get_event_history()
        manual_events = [e for e in history if e['source'] == 'manual']
        assert len(manual_events) > 0


class TestGitWorkflowMonitor:
    """Test cases for Git Workflow Monitor"""
    
    @pytest.fixture
    async def monitor(self, temp_repo):
        """Create monitor for testing"""
        git_manager = GitWorkflowManager(temp_repo)
        task_git_bridge = TaskGitBridge(git_manager)
        
        monitor = GitWorkflowMonitor(git_manager, task_git_bridge)
        await monitor.start_monitoring()
        yield monitor
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_monitor_initialization(self, temp_repo):
        """Test monitor initialization"""
        git_manager = GitWorkflowManager(temp_repo)
        task_git_bridge = TaskGitBridge(git_manager)
        
        monitor = GitWorkflowMonitor(git_manager, task_git_bridge)
        
        assert monitor.git_manager is not None
        assert monitor.task_git_bridge is not None
        assert monitor.config is not None
        assert monitor.monitoring_enabled is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, monitor):
        """Test comprehensive health check"""
        health = await monitor.perform_health_check()
        
        assert health is not None
        assert health.overall_status in [status for status in HealthStatus]
        assert health.last_check is not None
        assert isinstance(health.issues_count, int)
        assert isinstance(health.warnings_count, int)
        assert isinstance(health.components, dict)
        
        # Check that all expected components were checked
        expected_components = [
            'git_status', 'task_mappings', 'stale_branches',
            'merge_conflicts', 'repository_integrity', 'disk_space'
        ]
        
        for component in expected_components:
            assert component in health.components
    
    @pytest.mark.asyncio
    async def test_git_status_check(self, monitor):
        """Test Git status health check"""
        result = await monitor._check_git_status()
        
        assert result.component == 'git_status'
        assert result.status in [status for status in HealthStatus]
        assert result.message is not None
        assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_stale_branches_check(self, monitor):
        """Test stale branches check"""
        # Create a fake stale mapping
        from app.task_git_models import TaskGitMapping, TaskStatus
        
        old_date = datetime.now() - timedelta(days=10)
        stale_mapping = TaskGitMapping(
            task_id="stale-task",
            branch_name="task/stale-task",
            status=TaskStatus.IN_PROGRESS,
            created_at=old_date
        )
        
        monitor.task_git_bridge.mappings["stale-task"] = stale_mapping
        
        result = await monitor._check_stale_branches()
        
        assert result.component == 'stale_branches'
        # Should detect the stale branch
        assert result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_recovery_scheduling(self, monitor):
        """Test recovery operation scheduling"""
        operation_id = await monitor._schedule_recovery(
            RecoveryAction.CLEANUP_STALE_BRANCHES,
            "test-target",
            {}
        )
        
        assert operation_id is not None
        assert operation_id in monitor.active_recoveries
        
        operation = monitor.active_recoveries[operation_id]
        assert operation.action == RecoveryAction.CLEANUP_STALE_BRANCHES
        assert operation.target == "test-target"
        assert operation.status == "pending"
    
    @pytest.mark.asyncio
    async def test_monitoring_status(self, monitor):
        """Test monitoring status reporting"""
        status = monitor.get_monitoring_status()
        
        assert 'monitoring_enabled' in status
        assert 'background_tasks_count' in status
        assert 'active_recoveries_count' in status
        assert 'health_checks_performed' in status
        assert 'config' in status
        
        assert status['monitoring_enabled'] is True
    
    @pytest.mark.asyncio
    async def test_manual_recovery(self, monitor):
        """Test manual recovery triggering"""
        operation_id = await monitor.manual_recovery(
            "cleanup_stale_branches",
            "manual-target"
        )
        
        assert operation_id is not None
        assert operation_id in monitor.active_recoveries
        
        # Test invalid recovery action
        with pytest.raises(ValueError):
            await monitor.manual_recovery("invalid_action", "target")


class TestGitWorkflowIntegration:
    """Test cases for Git Workflow Integration"""
    
    @pytest.fixture
    async def integration(self, temp_repo):
        """Create integration for testing"""
        integration = GitWorkflowIntegration(temp_repo)
        await integration.start()
        yield integration
        await integration.stop()
    
    @pytest.mark.asyncio
    async def test_integration_initialization(self, temp_repo):
        """Test integration initialization"""
        integration = GitWorkflowIntegration(temp_repo)
        
        assert integration.git_manager is not None
        assert integration.task_git_bridge is not None
        assert integration.dependency_manager is not None
        assert integration.automation_service is not None
        assert integration.monitor is not None
        assert integration.running is False
    
    @pytest.mark.asyncio
    async def test_start_stop_integration(self, temp_repo):
        """Test starting and stopping integration"""
        integration = GitWorkflowIntegration(temp_repo)
        
        # Start
        await integration.start()
        assert integration.running is True
        
        # Stop
        await integration.stop()
        assert integration.running is False
    
    @pytest.mark.asyncio
    async def test_task_lifecycle(self, integration):
        """Test complete task lifecycle"""
        task_id = "integration-test-1"
        task_name = "Integration Test Task"
        
        # Start task
        success = await integration.start_task(
            task_id=task_id,
            task_name=task_name,
            task_description="Test task for integration testing"
        )
        assert success
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Check task status
        status = await integration.get_task_status(task_id)
        assert status is not None
        assert status['task_id'] == task_id
        
        # Create test file for progress update
        test_file = Path(integration.git_manager.repo_path) / "integration_test.txt"
        test_file.write_text("Integration test content")
        
        # Update progress
        success = await integration.update_task_progress(
            task_id=task_id,
            files_changed=["integration_test.txt"],
            progress_notes="Added integration test file"
        )
        assert success
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Complete task
        success = await integration.complete_task(
            task_id=task_id,
            completion_notes="Integration test completed"
        )
        assert success
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Check final status
        final_status = await integration.get_task_status(task_id)
        assert final_status is not None
        assert len(final_status['mapping']['commits']) > 0
    
    @pytest.mark.asyncio
    async def test_system_health(self, integration):
        """Test system health reporting"""
        health = await integration.get_system_health()
        
        assert 'overall_status' in health
        assert 'repository_health' in health
        assert 'automation_service' in health
        assert 'monitoring_system' in health
        assert 'system_running' in health
        assert 'timestamp' in health
        
        assert health['system_running'] is True
    
    @pytest.mark.asyncio
    async def test_dependency_management(self, integration):
        """Test dependency management through integration"""
        # Create tasks with dependencies
        await integration.start_task("dep-task-1", "Dependency Task 1")
        await integration.start_task(
            "dep-task-2", 
            "Dependency Task 2",
            dependencies=["dep-task-1"]
        )
        
        await asyncio.sleep(2)
        
        # Check dependencies
        deps = integration.dependency_manager.get_task_dependencies("dep-task-2")
        assert "dep-task-1" in deps
        
        dependents = integration.dependency_manager.get_task_dependents("dep-task-1")
        assert "dep-task-2" in dependents
        
        # Check ready tasks
        ready_tasks = integration.get_ready_tasks()
        assert "dep-task-1" in ready_tasks  # No dependencies
        assert "dep-task-2" not in ready_tasks  # Has unresolved dependency
    
    @pytest.mark.asyncio
    async def test_all_tasks_listing(self, integration):
        """Test listing all tasks"""
        # Create some tasks
        await integration.start_task("list-task-1", "List Task 1")
        await integration.start_task("list-task-2", "List Task 2")
        await asyncio.sleep(2)
        
        all_tasks = integration.get_all_tasks()
        assert len(all_tasks) >= 2
        
        task_ids = [task['task_id'] for task in all_tasks]
        assert "list-task-1" in task_ids
        assert "list-task-2" in task_ids
    
    @pytest.mark.asyncio
    async def test_event_and_health_history(self, integration):
        """Test event and health history"""
        # Generate some events
        await integration.start_task("history-task", "History Task")
        await asyncio.sleep(2)
        
        # Get event history
        events = integration.get_event_history(limit=10)
        assert len(events) > 0
        
        # Get health history
        health_history = integration.get_health_history(limit=5)
        # Might be empty if no health checks have run yet
        assert isinstance(health_history, list)
    
    @pytest.mark.asyncio
    async def test_manual_operations(self, integration):
        """Test manual workflow operations"""
        # Manual branch creation
        branch_name = await integration.manual_branch_creation(
            "manual-task", "Manual Task"
        )
        assert branch_name is not None
        assert "manual-task" in branch_name
        
        # Create test file for manual commit
        test_file = Path(integration.git_manager.repo_path) / "manual_test.txt"
        test_file.write_text("Manual test content")
        
        # Manual commit
        commit_hash = await integration.manual_commit(
            "manual-task",
            "Manual commit test",
            ["manual_test.txt"]
        )
        assert commit_hash is not None
        assert len(commit_hash) > 0
    
    @pytest.mark.asyncio
    async def test_system_validation(self, integration):
        """Test system validation"""
        validation = await integration.validate_system()
        
        assert 'git_repository' in validation
        assert 'task_mappings' in validation
        assert 'dependencies' in validation
        assert 'automation_service' in validation
        assert 'monitoring_system' in validation
        assert 'overall_valid' in validation
        
        # Most components should be valid in a fresh system
        assert validation['git_repository'] is True
        assert validation['task_mappings'] is True
        assert validation['dependencies'] is True


# Integration test that runs multiple components together
@pytest.mark.asyncio
async def test_full_workflow_integration():
    """Test full workflow with multiple tasks and dependencies"""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        # Initialize Git repository
        subprocess.run(['git', 'init'], cwd=repo_path, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo_path, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo_path, check=True)
        
        # Create initial commit
        (repo_path / 'README.md').write_text('# Full Integration Test')
        subprocess.run(['git', 'add', 'README.md'], cwd=repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=repo_path, check=True)
        
        # Create integration
        integration = GitWorkflowIntegration(str(repo_path))
        
        try:
            await integration.start()
            
            # Create a complex workflow
            tasks = [
                ("foundation", "Foundation Task", []),
                ("feature-a", "Feature A", ["foundation"]),
                ("feature-b", "Feature B", ["foundation"]),
                ("integration", "Integration Task", ["feature-a", "feature-b"])
            ]
            
            # Start all tasks
            for task_id, task_name, deps in tasks:
                await integration.start_task(
                    task_id=task_id,
                    task_name=task_name,
                    dependencies=deps
                )
            
            await asyncio.sleep(5)  # Wait for processing
            
            # Check that dependency relationships are correct
            ready_tasks = integration.get_ready_tasks()
            assert "foundation" in ready_tasks
            assert "feature-a" not in ready_tasks  # Depends on foundation
            assert "feature-b" not in ready_tasks  # Depends on foundation
            assert "integration" not in ready_tasks  # Depends on features
            
            # Complete foundation task
            await integration.complete_task("foundation", "Foundation completed")
            await asyncio.sleep(3)
            
            # Now feature tasks should be ready
            ready_tasks = integration.get_ready_tasks()
            assert "feature-a" in ready_tasks
            assert "feature-b" in ready_tasks
            assert "integration" not in ready_tasks
            
            # Complete feature tasks
            await integration.complete_task("feature-a", "Feature A completed")
            await integration.complete_task("feature-b", "Feature B completed")
            await asyncio.sleep(3)
            
            # Now integration task should be ready
            ready_tasks = integration.get_ready_tasks()
            assert "integration" in ready_tasks
            
            # Check system health
            health = await integration.get_system_health()
            assert health['overall_status'] in ['healthy', 'warning']  # Should be healthy or have minor warnings
            
            # Get merge strategy
            strategy = await integration.get_merge_strategy(
                ["foundation", "feature-a", "feature-b", "integration"]
            )
            assert len(strategy['merge_order']) == 4
            assert strategy['merge_order'][0] == "foundation"  # Should be first
            
        finally:
            await integration.stop()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])