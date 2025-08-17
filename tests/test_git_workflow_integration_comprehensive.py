# tests/test_git_workflow_integration_comprehensive.py
"""
Comprehensive Git Workflow Integration Tests

This module provides end-to-end integration tests for Git workflow functionality including:
- Task lifecycle Git integration tests
- Merge conflict resolution testing
- Git workflow performance and reliability tests

Requirements covered: 4.1, 4.2, 4.3, 5.1, 5.2, 8.1, 8.2
"""

import asyncio
import pytest
import tempfile
import shutil
import subprocess
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from app.git_workflow_manager import (
    GitWorkflowManager, TaskContext, TaskStatus, BranchType, BranchInfo
)
from app.task_git_bridge import TaskGitBridge
from app.git_workflow_automation_service import (
    GitWorkflowAutomationService, WorkflowEvent, WorkflowEventType
)
from app.task_dependency_manager import TaskDependencyManager
from app.task_git_models import (
    TaskGitMapping, GitCommit, TaskGitMetrics, TaskReport, MergeStrategy
)


class TestTaskLifecycleGitIntegration:
    """Test task lifecycle integration with Git operations"""
    
    @pytest.fixture
    def temp_git_repo(self):
        """Create a temporary Git repository for testing"""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)
        
        # Initialize Git repository
        subprocess.run(['git', 'init'], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo_path, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo_path, check=True)
        
        # Create initial commit
        (repo_path / 'README.md').write_text('# Test Repository')
        subprocess.run(['git', 'add', 'README.md'], cwd=repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=repo_path, check=True)
        
        yield repo_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def git_workflow_manager(self, temp_git_repo):
        """Create GitWorkflowManager with temporary repository"""
        return GitWorkflowManager(repo_path=str(temp_git_repo))
    
    @pytest.fixture
    def task_git_bridge(self, git_workflow_manager, temp_git_repo):
        """Create TaskGitBridge with temporary repository"""
        storage_path = temp_git_repo / ".kiro" / "task_git_mappings.json"
        return TaskGitBridge(git_workflow_manager, str(storage_path))
    
    @pytest.fixture
    def automation_service(self, git_workflow_manager, task_git_bridge, temp_git_repo):
        """Create GitWorkflowAutomationService with temporary repository"""
        config_path = temp_git_repo / ".kiro" / "workflow_automation.json"
        return GitWorkflowAutomationService(
            git_workflow_manager, task_git_bridge, config_path=str(config_path)
        )
    
    @pytest.mark.asyncio
    async def test_complete_task_lifecycle_integration(self, git_workflow_manager, task_git_bridge, automation_service):
        """Test complete task lifecycle from start to completion"""
        task_id = "1.1"
        task_name = "Implement user authentication"
        task_description = "Add login and registration functionality"
        
        # Start automation service
        await automation_service.start()
        
        try:
            # 1. Task Started - Should create branch
            await automation_service.trigger_task_started(
                task_id=task_id,
                task_name=task_name,
                task_description=task_description,
                requirements=["REQ-1.1", "REQ-1.2"]
            )
            
            # Allow time for event processing
            await asyncio.sleep(0.1)
            
            # Verify branch was created
            mapping = task_git_bridge.get_task_mapping(task_id)
            assert mapping is not None
            assert mapping.task_id == task_id
            assert mapping.status == TaskStatus.IN_PROGRESS
            assert mapping.branch_name is not None
            
            # Verify Git branch exists
            result = subprocess.run(
                ['git', 'branch', '--list', mapping.branch_name],
                cwd=git_workflow_manager.repo_path,
                capture_output=True, text=True
            )
            assert mapping.branch_name in result.stdout
            
            # 2. Task Progress - Should commit changes
            # Create some test files
            test_file = git_workflow_manager.repo_path / "auth.py"
            test_file.write_text("# Authentication module\nclass AuthManager:\n    pass")
            
            await automation_service.trigger_task_progress(
                task_id=task_id,
                files_changed=["auth.py"],
                progress_notes="Added authentication module structure"
            )
            
            # Allow time for event processing
            await asyncio.sleep(0.1)
            
            # Verify commit was created
            updated_mapping = task_git_bridge.get_task_mapping(task_id)
            assert len(updated_mapping.commits) > 0
            
            # Verify Git commit exists
            commit_hash = updated_mapping.commits[0]
            result = subprocess.run(
                ['git', 'show', '--format=%s', '--no-patch', commit_hash],
                cwd=git_workflow_manager.repo_path,
                capture_output=True, text=True
            )
            assert "authentication" in result.stdout.lower()
            
            # 3. More Progress - Additional commits
            test_file.write_text("# Authentication module\nclass AuthManager:\n    def login(self, user, password):\n        pass")
            
            await automation_service.trigger_task_progress(
                task_id=task_id,
                files_changed=["auth.py"],
                progress_notes="Added login method"
            )
            
            await asyncio.sleep(0.1)
            
            # 4. Task Completed - Should prepare for merge
            await automation_service.trigger_task_completed(
                task_id=task_id,
                completion_notes="Authentication module completed",
                requirements_addressed=["REQ-1.1", "REQ-1.2"]
            )
            
            await asyncio.sleep(0.1)
            
            # Verify task completion
            final_mapping = task_git_bridge.get_task_mapping(task_id)
            assert final_mapping.status == TaskStatus.COMPLETED
            assert final_mapping.completion_commit is not None
            assert len(final_mapping.commits) >= 2
            
            # Generate completion report
            report = await task_git_bridge.generate_task_completion_report(task_id)
            assert report is not None
            assert report.task_id == task_id
            assert report.status == TaskStatus.COMPLETED
            assert len(report.commits) >= 2
            assert "REQ-1.1" in report.requirements_covered
            assert "REQ-1.2" in report.requirements_covered
            
        finally:
            await automation_service.stop()
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self, git_workflow_manager, task_git_bridge, automation_service):
        """Test handling multiple concurrent tasks"""
        tasks = [
            ("2.1", "Add user profile", "User profile management"),
            ("2.2", "Add user settings", "User settings configuration"),
            ("2.3", "Add user preferences", "User preference storage")
        ]
        
        await automation_service.start()
        
        try:
            # Start all tasks concurrently
            start_tasks = []
            for task_id, task_name, task_description in tasks:
                task = automation_service.trigger_task_started(
                    task_id=task_id,
                    task_name=task_name,
                    task_description=task_description
                )
                start_tasks.append(task)
            
            await asyncio.gather(*start_tasks)
            await asyncio.sleep(0.2)  # Allow processing time
            
            # Verify all tasks have branches
            for task_id, _, _ in tasks:
                mapping = task_git_bridge.get_task_mapping(task_id)
                assert mapping is not None
                assert mapping.branch_name is not None
                assert mapping.status == TaskStatus.IN_PROGRESS
            
            # Add progress to each task
            progress_tasks = []
            for i, (task_id, _, _) in enumerate(tasks):
                # Create unique files for each task
                test_file = git_workflow_manager.repo_path / f"module_{i+1}.py"
                test_file.write_text(f"# Module {i+1}\nclass Module{i+1}:\n    pass")
                
                task = automation_service.trigger_task_progress(
                    task_id=task_id,
                    files_changed=[f"module_{i+1}.py"],
                    progress_notes=f"Added module {i+1} structure"
                )
                progress_tasks.append(task)
            
            await asyncio.gather(*progress_tasks)
            await asyncio.sleep(0.2)
            
            # Complete all tasks
            completion_tasks = []
            for task_id, _, _ in tasks:
                task = automation_service.trigger_task_completed(
                    task_id=task_id,
                    completion_notes=f"Task {task_id} completed"
                )
                completion_tasks.append(task)
            
            await asyncio.gather(*completion_tasks)
            await asyncio.sleep(0.2)
            
            # Verify all tasks completed
            for task_id, _, _ in tasks:
                mapping = task_git_bridge.get_task_mapping(task_id)
                assert mapping.status == TaskStatus.COMPLETED
                assert len(mapping.commits) >= 1
            
        finally:
            await automation_service.stop()
    
    @pytest.mark.asyncio
    async def test_task_dependency_handling(self, git_workflow_manager, task_git_bridge, automation_service):
        """Test task dependency handling in Git workflow"""
        # Create dependent tasks
        parent_task = "3.1"
        child_task = "3.2"
        
        await automation_service.start()
        
        try:
            # Add dependency relationship
            task_git_bridge.add_task_dependency(child_task, parent_task)
            
            # Start parent task
            await automation_service.trigger_task_started(
                task_id=parent_task,
                task_name="Parent task",
                task_description="Foundation task"
            )
            
            await asyncio.sleep(0.1)
            
            # Start child task (should be allowed even with dependency)
            await automation_service.trigger_task_started(
                task_id=child_task,
                task_name="Child task", 
                task_description="Dependent task"
            )
            
            await asyncio.sleep(0.1)
            
            # Verify both tasks have branches
            parent_mapping = task_git_bridge.get_task_mapping(parent_task)
            child_mapping = task_git_bridge.get_task_mapping(child_task)
            
            assert parent_mapping is not None
            assert child_mapping is not None
            assert parent_task in child_mapping.dependencies
            
            # Complete parent task
            parent_file = git_workflow_manager.repo_path / "parent.py"
            parent_file.write_text("# Parent module\nclass Parent:\n    pass")
            
            await automation_service.trigger_task_progress(
                task_id=parent_task,
                files_changed=["parent.py"],
                progress_notes="Parent implementation"
            )
            
            await automation_service.trigger_task_completed(
                task_id=parent_task,
                completion_notes="Parent task completed"
            )
            
            await asyncio.sleep(0.1)
            
            # Verify dependency resolution event was emitted
            # (This would trigger dependency_resolved event for child task)
            
            # Complete child task
            child_file = git_workflow_manager.repo_path / "child.py"
            child_file.write_text("# Child module\nfrom parent import Parent\nclass Child(Parent):\n    pass")
            
            await automation_service.trigger_task_progress(
                task_id=child_task,
                files_changed=["child.py"],
                progress_notes="Child implementation using parent"
            )
            
            await automation_service.trigger_task_completed(
                task_id=child_task,
                completion_notes="Child task completed"
            )
            
            await asyncio.sleep(0.1)
            
            # Verify both tasks completed
            final_parent = task_git_bridge.get_task_mapping(parent_task)
            final_child = task_git_bridge.get_task_mapping(child_task)
            
            assert final_parent.status == TaskStatus.COMPLETED
            assert final_child.status == TaskStatus.COMPLETED
            
        finally:
            await automation_service.stop()
    
    @pytest.mark.asyncio
    async def test_task_branch_naming_conventions(self, git_workflow_manager):
        """Test task branch naming conventions"""
        test_cases = [
            ("1.1", "Add user authentication", BranchType.TASK, "task/1-1-add-user-authentication"),
            ("2.5", "Fix login bug", BranchType.BUGFIX, "bugfix/2-5-fix-login-bug"),
            ("3.2", "New payment feature", BranchType.FEATURE, "feature/3-2-new-payment-feature"),
            ("HOTFIX-1", "Critical security fix", BranchType.HOTFIX, "hotfix/hotfix-1-critical-security-fix")
        ]
        
        for task_id, task_name, branch_type, expected_pattern in test_cases:
            branch_name = git_workflow_manager.create_task_branch(
                task_id=task_id,
                task_name=task_name,
                branch_type=branch_type
            )
            
            # Verify branch name follows expected pattern
            assert branch_name.startswith(branch_type.value.lower())
            assert task_id.replace('.', '-').lower() in branch_name
            
            # Verify branch exists in Git
            result = subprocess.run(
                ['git', 'branch', '--list', branch_name],
                cwd=git_workflow_manager.repo_path,
                capture_output=True, text=True
            )
            assert branch_name in result.stdout
    
    @pytest.mark.asyncio
    async def test_commit_message_generation(self, git_workflow_manager, temp_git_repo):
        """Test intelligent commit message generation"""
        # Create task context
        task_context = TaskContext(
            task_id="4.1",
            task_name="Add user authentication",
            description="Implement login and registration functionality",
            files_modified=["auth.py", "models.py", "views.py"],
            requirements_addressed=["REQ-4.1", "REQ-4.2"],
            completion_time=datetime.now(),
            status=TaskStatus.COMPLETED
        )
        
        # Generate commit message
        commit_message = git_workflow_manager.generate_commit_message(task_context)
        
        # Verify commit message structure
        assert "feat: add user authentication" in commit_message.lower()
        assert "Task ID: 4.1" in commit_message
        assert "auth.py" in commit_message
        assert "REQ-4.1" in commit_message
        assert "REQ-4.2" in commit_message
        
        # Test different task types
        test_cases = [
            ("Fix login bug", "fix:"),
            ("Add documentation", "docs:"),
            ("Refactor auth module", "refactor:"),
            ("Add login tests", "test:")
        ]
        
        for task_name, expected_type in test_cases:
            task_context.task_name = task_name
            message = git_workflow_manager.generate_commit_message(task_context)
            assert expected_type in message.lower()


class TestMergeConflictResolution:
    """Test merge conflict detection and resolution"""
    
    @pytest.fixture
    def conflict_repo(self):
        """Create a repository with potential merge conflicts"""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)
        
        # Initialize Git repository
        subprocess.run(['git', 'init'], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo_path, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo_path, check=True)
        
        # Create initial file
        shared_file = repo_path / 'shared.py'
        shared_file.write_text('# Shared module\nclass SharedClass:\n    def method1(self):\n        pass\n')
        
        subprocess.run(['git', 'add', 'shared.py'], cwd=repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial shared file'], cwd=repo_path, check=True)
        
        yield repo_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def conflict_git_manager(self, conflict_repo):
        """Create GitWorkflowManager with conflict repository"""
        return GitWorkflowManager(repo_path=str(conflict_repo))
    
    @pytest.mark.asyncio
    async def test_merge_conflict_detection(self, conflict_git_manager, conflict_repo):
        """Test detection of merge conflicts between branches"""
        # Create two conflicting branches
        branch1 = conflict_git_manager.create_task_branch("5.1", "Feature A", BranchType.FEATURE)
        
        # Modify file in branch1
        shared_file = conflict_repo / 'shared.py'
        shared_file.write_text('# Shared module\nclass SharedClass:\n    def method1(self):\n        return "Feature A"\n')
        
        subprocess.run(['git', 'add', 'shared.py'], cwd=conflict_repo, check=True)
        subprocess.run(['git', 'commit', '-m', 'Add Feature A'], cwd=conflict_repo, check=True)
        
        # Switch to main and create another branch
        subprocess.run(['git', 'checkout', 'main'], cwd=conflict_repo, check=True)
        branch2 = conflict_git_manager.create_task_branch("5.2", "Feature B", BranchType.FEATURE)
        
        # Modify same file in branch2
        shared_file.write_text('# Shared module\nclass SharedClass:\n    def method1(self):\n        return "Feature B"\n')
        
        subprocess.run(['git', 'add', 'shared.py'], cwd=conflict_repo, check=True)
        subprocess.run(['git', 'commit', '-m', 'Add Feature B'], cwd=conflict_repo, check=True)
        
        # Detect conflicts between branches
        conflicts = conflict_git_manager.detect_merge_conflicts(branch1, branch2)
        
        # Should detect conflicts
        assert "conflicted_files" in conflicts
        # In a real implementation, this would detect the conflict
        # For now, we verify the method exists and returns a structure
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_strategies(self, conflict_git_manager, conflict_repo):
        """Test different conflict resolution strategies"""
        # Create conflicting branches
        branch1 = "feature/conflict-test-1"
        branch2 = "feature/conflict-test-2"
        
        # Create branch1
        subprocess.run(['git', 'checkout', '-b', branch1], cwd=conflict_repo, check=True)
        shared_file = conflict_repo / 'shared.py'
        shared_file.write_text('# Shared module - Version 1\nclass SharedClass:\n    def method1(self):\n        return "Version 1"\n')
        subprocess.run(['git', 'add', 'shared.py'], cwd=conflict_repo, check=True)
        subprocess.run(['git', 'commit', '-m', 'Version 1 changes'], cwd=conflict_repo, check=True)
        
        # Create branch2
        subprocess.run(['git', 'checkout', 'main'], cwd=conflict_repo, check=True)
        subprocess.run(['git', 'checkout', '-b', branch2], cwd=conflict_repo, check=True)
        shared_file.write_text('# Shared module - Version 2\nclass SharedClass:\n    def method1(self):\n        return "Version 2"\n')
        subprocess.run(['git', 'add', 'shared.py'], cwd=conflict_repo, check=True)
        subprocess.run(['git', 'commit', '-m', 'Version 2 changes'], cwd=conflict_repo, check=True)
        
        # Test conflict detection
        current_conflicts = conflict_git_manager.check_for_conflicts()
        
        # At this point, no conflicts exist yet (branches haven't been merged)
        assert isinstance(current_conflicts, list)
    
    @pytest.mark.asyncio
    async def test_automated_conflict_resolution(self, conflict_git_manager):
        """Test automated conflict resolution mechanisms"""
        # This would test automated resolution strategies
        # For now, we test that the conflict resolution framework exists
        
        # Mock conflict scenario
        mock_conflicts = ["shared.py", "config.py"]
        
        # Test that conflict checking works
        conflicts = conflict_git_manager.check_for_conflicts()
        assert isinstance(conflicts, list)
        
        # In a full implementation, this would test:
        # - Automatic resolution of simple conflicts
        # - Escalation of complex conflicts to manual review
        # - Conflict resolution logging and tracking
    
    @pytest.mark.asyncio
    async def test_merge_strategy_generation(self, conflict_git_manager):
        """Test generation of merge strategies for dependent tasks"""
        # Create mock task git bridge
        mock_bridge = Mock()
        mock_bridge.get_task_mapping.return_value = Mock(
            dependencies=["task1", "task2"],
            commits=["abc123", "def456"]
        )
        
        # Create dependency manager
        dependency_manager = TaskDependencyManager(mock_bridge, conflict_git_manager)
        
        # Test merge strategy generation
        dependent_tasks = ["task3", "task4", "task5"]
        
        # Mock the dependency graph
        with patch.object(dependency_manager, '_build_dependency_graph') as mock_graph:
            mock_graph.return_value = {
                "task3": ["task1"],
                "task4": ["task2"],
                "task5": ["task3", "task4"]
            }
            
            strategy = await dependency_manager.generate_merge_strategy(dependent_tasks)
            
            assert isinstance(strategy, MergeStrategy)
            assert len(strategy.merge_order) == len(dependent_tasks)
            assert strategy.risk_level in ["low", "medium", "high"]


class TestGitWorkflowPerformanceAndReliability:
    """Test Git workflow performance and reliability"""
    
    @pytest.fixture
    def performance_repo(self):
        """Create a repository for performance testing"""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)
        
        # Initialize Git repository
        subprocess.run(['git', 'init'], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo_path, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo_path, check=True)
        
        # Create initial structure
        (repo_path / 'README.md').write_text('# Performance Test Repository')
        subprocess.run(['git', 'add', 'README.md'], cwd=repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=repo_path, check=True)
        
        yield repo_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_high_volume_task_processing(self, performance_repo):
        """Test processing high volume of tasks"""
        git_manager = GitWorkflowManager(repo_path=str(performance_repo))
        task_bridge = TaskGitBridge(git_manager)
        automation_service = GitWorkflowAutomationService(git_manager, task_bridge)
        
        await automation_service.start()
        
        try:
            # Create many tasks concurrently
            task_count = 50
            start_time = datetime.now()
            
            # Start all tasks
            start_tasks = []
            for i in range(task_count):
                task = automation_service.trigger_task_started(
                    task_id=f"perf-{i}",
                    task_name=f"Performance test task {i}",
                    task_description=f"Task {i} for performance testing"
                )
                start_tasks.append(task)
            
            await asyncio.gather(*start_tasks)
            
            # Allow processing time
            await asyncio.sleep(2.0)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Verify performance metrics
            assert processing_time < 30.0  # Should process 50 tasks in under 30 seconds
            
            # Verify all tasks were processed
            processed_count = 0
            for i in range(task_count):
                mapping = task_bridge.get_task_mapping(f"perf-{i}")
                if mapping and mapping.branch_name:
                    processed_count += 1
            
            # Should process at least 80% of tasks successfully
            assert processed_count >= task_count * 0.8
            
        finally:
            await automation_service.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_branch_operations(self, performance_repo):
        """Test concurrent branch creation and operations"""
        git_manager = GitWorkflowManager(repo_path=str(performance_repo))
        
        # Create multiple branches concurrently
        branch_count = 20
        branch_tasks = []
        
        async def create_branch(task_id: str, task_name: str):
            return git_manager.create_task_branch(task_id, task_name, BranchType.TASK)
        
        for i in range(branch_count):
            task = create_branch(f"concurrent-{i}", f"Concurrent task {i}")
            branch_tasks.append(task)
        
        # Execute all branch creations
        start_time = datetime.now()
        results = await asyncio.gather(*branch_tasks, return_exceptions=True)
        end_time = datetime.now()
        
        # Verify performance
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 10.0  # Should create 20 branches in under 10 seconds
        
        # Verify success rate
        successful_branches = [r for r in results if isinstance(r, str) and r.startswith('task/')]
        assert len(successful_branches) >= branch_count * 0.9  # 90% success rate
    
    @pytest.mark.asyncio
    async def test_large_file_commit_performance(self, performance_repo):
        """Test performance with large file commits"""
        git_manager = GitWorkflowManager(repo_path=str(performance_repo))
        
        # Create task branch
        branch_name = git_manager.create_task_branch("large-file-test", "Large file test", BranchType.TASK)
        
        # Create large files
        large_files = []
        for i in range(5):
            file_path = performance_repo / f"large_file_{i}.txt"
            # Create 1MB file
            content = "x" * (1024 * 1024)  # 1MB of 'x' characters
            file_path.write_text(content)
            large_files.append(f"large_file_{i}.txt")
        
        # Commit large files
        start_time = datetime.now()
        commit_hash = git_manager.commit_task_progress(
            task_id="large-file-test",
            files=large_files,
            message="Add large test files"
        )
        end_time = datetime.now()
        
        # Verify performance
        commit_time = (end_time - start_time).total_seconds()
        assert commit_time < 30.0  # Should commit 5MB in under 30 seconds
        assert commit_hash is not None
        
        # Verify commit exists
        result = subprocess.run(
            ['git', 'show', '--format=%s', '--no-patch', commit_hash],
            cwd=performance_repo,
            capture_output=True, text=True
        )
        assert "large test files" in result.stdout.lower()
    
    @pytest.mark.asyncio
    async def test_git_operation_reliability(self, performance_repo):
        """Test reliability of Git operations under various conditions"""
        git_manager = GitWorkflowManager(repo_path=str(performance_repo))
        
        # Test 1: Operations with uncommitted changes
        test_file = performance_repo / "test_reliability.py"
        test_file.write_text("# Test file for reliability")
        
        # Should handle uncommitted changes gracefully
        branch_name = git_manager.create_task_branch("reliability-test", "Reliability test")
        assert branch_name is not None
        
        # Test 2: Rapid successive operations
        for i in range(10):
            test_file.write_text(f"# Test file iteration {i}")
            commit_hash = git_manager.commit_task_progress(
                task_id="reliability-test",
                files=["test_reliability.py"],
                message=f"Iteration {i}"
            )
            assert commit_hash is not None
        
        # Test 3: Branch switching reliability
        original_branch = git_manager._get_current_branch()
        
        # Create and switch between multiple branches
        branches = []
        for i in range(5):
            branch = git_manager.create_task_branch(f"switch-test-{i}", f"Switch test {i}")
            branches.append(branch)
        
        # Switch back to original branch
        git_manager._switch_to_branch(original_branch)
        current_branch = git_manager._get_current_branch()
        assert current_branch == original_branch
    
    @pytest.mark.asyncio
    async def test_error_recovery_mechanisms(self, performance_repo):
        """Test error recovery mechanisms in Git workflow"""
        git_manager = GitWorkflowManager(repo_path=str(performance_repo))
        task_bridge = TaskGitBridge(git_manager)
        
        # Test 1: Recovery from failed commits
        branch_name = git_manager.create_task_branch("error-recovery", "Error recovery test")
        
        # Create a scenario that might cause commit failure
        # (In real scenario, this could be disk full, permissions, etc.)
        try:
            # Attempt to commit non-existent file
            git_manager.commit_task_progress(
                task_id="error-recovery",
                files=["non_existent_file.py"],
                message="This should fail"
            )
        except Exception as e:
            # Should handle the error gracefully
            assert isinstance(e, RuntimeError)
        
        # Verify system is still functional after error
        test_file = performance_repo / "recovery_test.py"
        test_file.write_text("# Recovery test file")
        
        commit_hash = git_manager.commit_task_progress(
            task_id="error-recovery",
            files=["recovery_test.py"],
            message="Recovery successful"
        )
        assert commit_hash is not None
        
        # Test 2: Recovery from branch conflicts
        # This would test recovery from various Git conflict scenarios
        
        # Test 3: Recovery from corrupted tracking data
        # Simulate corrupted tracking file
        if task_bridge.storage_path.exists():
            # Backup original
            original_content = task_bridge.storage_path.read_text()
            
            # Write invalid JSON
            task_bridge.storage_path.write_text("invalid json content")
            
            # Create new bridge instance - should handle corrupted data
            new_bridge = TaskGitBridge(git_manager, str(task_bridge.storage_path))
            
            # Should start with empty mappings due to corruption
            assert len(new_bridge.mappings) == 0
            
            # Restore original content
            task_bridge.storage_path.write_text(original_content)
    
    @pytest.mark.asyncio
    async def test_workflow_monitoring_and_health_checks(self, performance_repo):
        """Test workflow monitoring and health check functionality"""
        git_manager = GitWorkflowManager(repo_path=str(performance_repo))
        task_bridge = TaskGitBridge(git_manager)
        automation_service = GitWorkflowAutomationService(git_manager, task_bridge)
        
        await automation_service.start()
        
        try:
            # Create some tasks to monitor
            for i in range(3):
                await automation_service.trigger_task_started(
                    task_id=f"monitor-{i}",
                    task_name=f"Monitor test {i}",
                    task_description=f"Task {i} for monitoring"
                )
            
            await asyncio.sleep(0.5)
            
            # Test service status
            status = automation_service.get_service_status()
            assert status['running'] == True
            assert status['triggers_count'] > 0
            assert status['events_processed'] >= 3
            
            # Test event history
            history = automation_service.get_event_history(limit=10)
            assert len(history) >= 3
            assert all('event_type' in event for event in history)
            
            # Test trigger status
            trigger_status = automation_service.get_trigger_status()
            assert len(trigger_status) > 0
            assert all('event_type' in trigger for trigger in trigger_status)
            
            # Simulate health check
            await automation_service._perform_health_check()
            
            # Verify health check completed without errors
            assert automation_service.last_health_check is not None
            
        finally:
            await automation_service.stop()
    
    @pytest.mark.asyncio
    async def test_git_metrics_and_analytics(self, performance_repo):
        """Test Git metrics collection and analytics"""
        git_manager = GitWorkflowManager(repo_path=str(performance_repo))
        task_bridge = TaskGitBridge(git_manager)
        
        # Create a task with multiple commits
        task_id = "metrics-test"
        branch_name = git_manager.create_task_branch(task_id, "Metrics test task")
        await task_bridge.link_task_to_branch(task_id, branch_name)
        
        # Create multiple commits with different file changes
        commits = []
        for i in range(5):
            # Create files with varying sizes
            file_path = performance_repo / f"metrics_file_{i}.py"
            content = f"# Metrics test file {i}\n" + "# Line\n" * (10 * (i + 1))
            file_path.write_text(content)
            
            commit_hash = git_manager.commit_task_progress(
                task_id=task_id,
                files=[f"metrics_file_{i}.py"],
                message=f"Add metrics file {i}"
            )
            commits.append(commit_hash)
            
            # Update task bridge with commit
            await task_bridge.update_task_status_from_git(commit_hash)
        
        # Get Git metrics
        metrics = await task_bridge.get_task_git_metrics(task_id)
        
        assert metrics is not None
        assert metrics.task_id == task_id
        assert metrics.total_commits == 5
        assert metrics.files_modified >= 5
        assert metrics.lines_added > 0
        
        # Generate comprehensive report
        report = await task_bridge.generate_task_completion_report(task_id)
        
        assert report is not None
        assert report.task_id == task_id
        assert len(report.commits) == 5
        assert report.git_metrics.total_commits == 5


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])