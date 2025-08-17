"""
Tests for Task-Git Integration Bridge

This module tests the task tracking and Git mapping system,
including dependency management and merge strategies.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from app.task_git_bridge import TaskGitBridge
from app.task_git_models import (
    TaskGitMapping, TaskStatus, MergeStatus, GitCommit,
    TaskGitMetrics, TaskReport, MergeStrategy
)
from app.task_dependency_manager import TaskDependencyManager, DependencyNode
from app.git_history_analytics import GitHistoryAnalytics, ProjectMetrics
from app.git_workflow_manager import GitWorkflowManager


class TestTaskGitBridge:
    """Test cases for TaskGitBridge"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_git_manager(self):
        """Mock GitWorkflowManager"""
        return Mock(spec=GitWorkflowManager)
    
    @pytest.fixture
    def task_git_bridge(self, mock_git_manager, temp_dir):
        """Create TaskGitBridge instance for testing"""
        storage_path = Path(temp_dir) / "test_mappings.json"
        return TaskGitBridge(mock_git_manager, str(storage_path))
    
    @pytest.mark.asyncio
    async def test_link_task_to_branch(self, task_git_bridge):
        """Test linking a task to a Git branch"""
        with patch.object(task_git_bridge, '_get_current_commit', return_value='abc123'):
            result = await task_git_bridge.link_task_to_branch('task-1', 'feature/task-1')
            
            assert result is True
            assert 'task-1' in task_git_bridge.mappings
            
            mapping = task_git_bridge.mappings['task-1']
            assert mapping.task_id == 'task-1'
            assert mapping.branch_name == 'feature/task-1'
            assert mapping.start_commit == 'abc123'
            assert mapping.status == TaskStatus.IN_PROGRESS
    
    @pytest.mark.asyncio
    async def test_update_task_status_from_git(self, task_git_bridge):
        """Test updating task status from Git commit"""
        # Setup existing mapping
        mapping = TaskGitMapping(
            task_id='1.1',
            branch_name='feature/task-1-1',
            status=TaskStatus.IN_PROGRESS
        )
        task_git_bridge.mappings['1.1'] = mapping
        
        # Mock commit info
        commit_info = {
            'message': 'Complete task 1.1: Implement feature',
            'author': 'Test User',
            'timestamp': '2024-01-01T10:00:00',
            'files_changed': ['app/feature.py']
        }
        
        with patch.object(task_git_bridge, '_get_commit_info', return_value=commit_info):
            result = await task_git_bridge.update_task_status_from_git('def456')
            
            assert result is True
            assert 'def456' in mapping.commits
            assert mapping.status == TaskStatus.COMPLETED
            assert mapping.completion_commit == 'def456'
    
    @pytest.mark.asyncio
    async def test_generate_task_completion_report(self, task_git_bridge):
        """Test generating task completion report"""
        # Setup mapping with commits
        mapping = TaskGitMapping(
            task_id='2.1',
            branch_name='feature/task-2-1',
            status=TaskStatus.COMPLETED,
            commits=['abc123', 'def456'],
            requirement_refs=['2.1', '2.2']
        )
        task_git_bridge.mappings['2.1'] = mapping
        
        # Mock methods
        with patch.object(task_git_bridge, '_calculate_git_metrics') as mock_metrics, \
             patch.object(task_git_bridge, '_get_commit_info') as mock_commit_info:
            
            mock_metrics.return_value = TaskGitMetrics(
                task_id='2.1',
                total_commits=2,
                files_modified=3,
                lines_added=100,
                lines_deleted=20,
                duration_hours=8.5,
                branch_age_days=2,
                merge_conflicts_count=0,
                dependency_count=1
            )
            
            mock_commit_info.side_effect = [
                {
                    'message': 'Start task 2.1',
                    'author': 'Test User',
                    'timestamp': '2024-01-01T09:00:00',
                    'files_changed': ['app/module.py']
                },
                {
                    'message': 'Complete task 2.1',
                    'author': 'Test User',
                    'timestamp': '2024-01-01T17:30:00',
                    'files_changed': ['app/module.py', 'tests/test_module.py']
                }
            ]
            
            report = await task_git_bridge.generate_task_completion_report('2.1')
            
            assert report is not None
            assert report.task_id == '2.1'
            assert report.status == TaskStatus.COMPLETED
            assert len(report.commits) == 2
            assert report.git_metrics.total_commits == 2
            assert report.requirements_covered == ['2.1', '2.2']
    
    @pytest.mark.asyncio
    async def test_handle_task_dependency_merge(self, task_git_bridge):
        """Test handling task dependency merge"""
        # Setup mappings with dependencies
        task_git_bridge.mappings['1.1'] = TaskGitMapping(
            task_id='1.1',
            branch_name='feature/task-1-1',
            dependencies=[],
            commits=['abc123']
        )
        task_git_bridge.mappings['1.2'] = TaskGitMapping(
            task_id='1.2',
            branch_name='feature/task-1-2',
            dependencies=['1.1'],
            commits=['def456']
        )
        
        with patch.object(task_git_bridge, '_assess_merge_conflicts') as mock_conflicts:
            mock_conflicts.return_value = {}
            
            strategy = await task_git_bridge.handle_task_dependency_merge(['1.1', '1.2'])
            
            assert isinstance(strategy, MergeStrategy)
            assert '1.1' in strategy.merge_order
            assert '1.2' in strategy.merge_order
            # 1.1 should come before 1.2 due to dependency
            assert strategy.merge_order.index('1.1') < strategy.merge_order.index('1.2')
    
    def test_add_task_dependency(self, task_git_bridge):
        """Test adding task dependency"""
        # Create initial mapping
        task_git_bridge.mappings['task-1'] = TaskGitMapping(
            task_id='task-1',
            branch_name='feature/task-1'
        )
        
        result = task_git_bridge.add_task_dependency('task-1', 'task-0')
        
        assert result is True
        assert 'task-0' in task_git_bridge.mappings['task-1'].dependencies
    
    def test_add_requirement_reference(self, task_git_bridge):
        """Test adding requirement reference"""
        # Create initial mapping
        task_git_bridge.mappings['task-1'] = TaskGitMapping(
            task_id='task-1',
            branch_name='feature/task-1'
        )
        
        result = task_git_bridge.add_requirement_reference('task-1', '1.1')
        
        assert result is True
        assert '1.1' in task_git_bridge.mappings['task-1'].requirement_refs
    
    def test_get_task_mapping(self, task_git_bridge):
        """Test getting task mapping"""
        mapping = TaskGitMapping(
            task_id='test-task',
            branch_name='feature/test'
        )
        task_git_bridge.mappings['test-task'] = mapping
        
        result = task_git_bridge.get_task_mapping('test-task')
        
        assert result == mapping
        assert result.task_id == 'test-task'
    
    def test_get_all_mappings(self, task_git_bridge):
        """Test getting all mappings"""
        mapping1 = TaskGitMapping(task_id='task-1', branch_name='feature/task-1')
        mapping2 = TaskGitMapping(task_id='task-2', branch_name='feature/task-2')
        
        task_git_bridge.mappings['task-1'] = mapping1
        task_git_bridge.mappings['task-2'] = mapping2
        
        result = task_git_bridge.get_all_mappings()
        
        assert len(result) == 2
        assert 'task-1' in result
        assert 'task-2' in result


class TestTaskDependencyManager:
    """Test cases for TaskDependencyManager"""
    
    @pytest.fixture
    def mock_task_git_bridge(self):
        """Mock TaskGitBridge"""
        bridge = Mock(spec=TaskGitBridge)
        bridge.get_all_mappings.return_value = {
            'task-1': TaskGitMapping(task_id='task-1', branch_name='feature/task-1'),
            'task-2': TaskGitMapping(task_id='task-2', branch_name='feature/task-2', dependencies=['task-1'])
        }
        return bridge
    
    @pytest.fixture
    def mock_git_manager(self):
        """Mock GitWorkflowManager"""
        return Mock(spec=GitWorkflowManager)
    
    @pytest.fixture
    def dependency_manager(self, mock_task_git_bridge, mock_git_manager):
        """Create TaskDependencyManager instance"""
        return TaskDependencyManager(mock_task_git_bridge, mock_git_manager)
    
    def test_add_dependency(self, dependency_manager):
        """Test adding dependency"""
        result = dependency_manager.add_dependency('task-3', 'task-1')
        
        assert result is True
        assert 'task-3' in dependency_manager.dependency_graph
        assert 'task-1' in dependency_manager.dependency_graph['task-3'].dependencies
        assert 'task-3' in dependency_manager.dependency_graph['task-1'].dependents
    
    def test_add_circular_dependency(self, dependency_manager):
        """Test preventing circular dependencies"""
        # task-2 already depends on task-1
        result = dependency_manager.add_dependency('task-1', 'task-2')
        
        assert result is False  # Should prevent circular dependency
    
    def test_get_ready_tasks(self, dependency_manager):
        """Test getting ready tasks"""
        # Set task-1 as completed
        dependency_manager.dependency_graph['task-1'].status = TaskStatus.COMPLETED
        
        ready_tasks = dependency_manager.get_ready_tasks()
        
        assert 'task-2' in ready_tasks  # Should be ready since task-1 is completed
    
    def test_calculate_critical_path(self, dependency_manager):
        """Test calculating critical path"""
        critical_path = dependency_manager.calculate_critical_path()
        
        assert isinstance(critical_path, list)
        assert len(critical_path) > 0
        # task-1 should come before task-2 in critical path
        if 'task-1' in critical_path and 'task-2' in critical_path:
            assert critical_path.index('task-1') < critical_path.index('task-2')
    
    @pytest.mark.asyncio
    async def test_generate_merge_strategy(self, dependency_manager):
        """Test generating merge strategy"""
        with patch.object(dependency_manager, '_analyze_merge_conflicts') as mock_conflicts:
            mock_conflicts.return_value = Mock(
                conflicting_files={},
                conflict_severity='low',
                estimated_resolution_time=0
            )
            
            strategy = await dependency_manager.generate_merge_strategy(['task-1', 'task-2'])
            
            assert isinstance(strategy, MergeStrategy)
            assert len(strategy.merge_order) == 2
            assert strategy.risk_level in ['low', 'medium', 'high']
    
    def test_get_dependency_graph_visualization(self, dependency_manager):
        """Test getting dependency graph visualization data"""
        viz_data = dependency_manager.get_dependency_graph_visualization()
        
        assert 'nodes' in viz_data
        assert 'edges' in viz_data
        assert 'critical_path' in viz_data
        assert len(viz_data['nodes']) >= 2
        assert len(viz_data['edges']) >= 1


class TestGitHistoryAnalytics:
    """Test cases for GitHistoryAnalytics"""
    
    @pytest.fixture
    def mock_task_git_bridge(self):
        """Mock TaskGitBridge"""
        bridge = Mock(spec=TaskGitBridge)
        bridge.get_all_mappings.return_value = {
            'task-1': TaskGitMapping(
                task_id='task-1',
                branch_name='feature/task-1',
                status=TaskStatus.COMPLETED,
                created_at=datetime.now() - timedelta(days=2),
                completed_at=datetime.now() - timedelta(days=1)
            )
        }
        return bridge
    
    @pytest.fixture
    def analytics(self, mock_task_git_bridge):
        """Create GitHistoryAnalytics instance"""
        return GitHistoryAnalytics(mock_task_git_bridge)
    
    @pytest.mark.asyncio
    async def test_get_project_metrics(self, analytics):
        """Test getting project metrics"""
        with patch.object(analytics, '_get_commits_since') as mock_commits:
            mock_commits.return_value = [
                GitCommit(
                    hash='abc123',
                    author='Test User',
                    timestamp=datetime.now(),
                    message='Test commit',
                    files_changed=['app/test.py']
                )
            ]
            
            with patch.object(analytics, '_get_commit_lines_added', return_value=10), \
                 patch.object(analytics, '_get_commit_lines_deleted', return_value=5), \
                 patch.object(analytics, '_calculate_completion_trend', return_value=[]):
                
                metrics = await analytics.get_project_metrics(30)
                
                assert isinstance(metrics, ProjectMetrics)
                assert metrics.total_commits == 1
                assert metrics.total_tasks == 1
                assert metrics.completed_tasks == 1
                assert metrics.total_lines_added == 10
                assert metrics.total_lines_deleted == 5
    
    @pytest.mark.asyncio
    async def test_analyze_task_complexity(self, analytics):
        """Test analyzing task complexity"""
        with patch.object(analytics, '_get_commit_details') as mock_commit_details:
            mock_commit_details.return_value = GitCommit(
                hash='abc123',
                author='Test User',
                timestamp=datetime.now(),
                message='Implement complex feature',
                files_changed=['app/feature.py', 'tests/test_feature.py']
            )
            
            with patch.object(analytics, '_get_commit_lines_added', return_value=100), \
                 patch.object(analytics, '_get_commit_lines_deleted', return_value=20):
                
                # Mock task mapping
                analytics.task_git_bridge.get_task_mapping.return_value = TaskGitMapping(
                    task_id='task-1',
                    branch_name='feature/task-1',
                    commits=['abc123'],
                    created_at=datetime.now() - timedelta(hours=8),
                    completed_at=datetime.now()
                )
                
                task_analytics = await analytics.analyze_task_complexity('task-1')
                
                assert task_analytics.task_id == 'task-1'
                assert task_analytics.complexity_score >= 0.0
                assert task_analytics.code_churn_ratio == 0.2  # 20/100
                assert task_analytics.collaboration_score == 1  # One author
    
    @pytest.mark.asyncio
    async def test_get_task_timeline(self, analytics):
        """Test getting task timeline"""
        mapping = TaskGitMapping(
            task_id='task-1',
            branch_name='feature/task-1',
            commits=['abc123'],
            created_at=datetime.now() - timedelta(hours=2),
            completed_at=datetime.now()
        )
        
        analytics.task_git_bridge.get_task_mapping.return_value = mapping
        
        with patch.object(analytics, '_get_commit_details') as mock_commit_details:
            mock_commit_details.return_value = GitCommit(
                hash='abc123',
                author='Test User',
                timestamp=datetime.now() - timedelta(hours=1),
                message='Implement feature',
                files_changed=['app/feature.py']
            )
            
            timeline = await analytics.get_task_timeline('task-1')
            
            assert len(timeline) == 3  # start, commit, completion
            assert timeline[0]['event_type'] == 'task_started'
            assert timeline[1]['event_type'] == 'commit'
            assert timeline[2]['event_type'] == 'task_completed'
    
    @pytest.mark.asyncio
    async def test_generate_productivity_report(self, analytics):
        """Test generating productivity report"""
        with patch.object(analytics, 'get_project_metrics') as mock_project_metrics, \
             patch.object(analytics, 'get_developer_metrics') as mock_dev_metrics:
            
            mock_project_metrics.return_value = ProjectMetrics(
                total_tasks=10,
                completed_tasks=8,
                completion_rate=0.8,
                avg_task_duration=24.0
            )
            
            mock_dev_metrics.return_value = {
                'dev1': Mock(name='Developer 1', completed_tasks=5, total_commits=20, avg_commit_size=50.0)
            }
            
            report = await analytics.generate_productivity_report(30)
            
            assert 'productivity_score' in report
            assert 'project_metrics' in report
            assert 'top_developers' in report
            assert 'recommendations' in report
            assert report['period_days'] == 30


@pytest.mark.integration
class TestTaskGitIntegrationEnd2End:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def temp_git_repo(self):
        """Create temporary Git repository for testing"""
        import subprocess
        
        temp_dir = tempfile.mkdtemp()
        
        # Initialize Git repo
        subprocess.run(['git', 'init'], cwd=temp_dir, check=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=temp_dir, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=temp_dir, check=True)
        
        # Create initial commit
        test_file = Path(temp_dir) / 'README.md'
        test_file.write_text('# Test Repository')
        subprocess.run(['git', 'add', 'README.md'], cwd=temp_dir, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=temp_dir, check=True)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_full_task_lifecycle(self, temp_git_repo):
        """Test complete task lifecycle with Git integration"""
        # Setup components
        git_manager = GitWorkflowManager(temp_git_repo)
        task_git_bridge = TaskGitBridge(git_manager, str(Path(temp_git_repo) / "mappings.json"))
        
        # Start task
        task_id = '1.1'
        branch_name = f'feature/task-{task_id}'
        
        # Link task to branch
        result = await task_git_bridge.link_task_to_branch(task_id, branch_name)
        assert result is True
        
        # Verify mapping created
        mapping = task_git_bridge.get_task_mapping(task_id)
        assert mapping is not None
        assert mapping.task_id == task_id
        assert mapping.branch_name == branch_name
        
        # Add requirement reference
        task_git_bridge.add_requirement_reference(task_id, '1.1')
        
        # Verify requirement added
        updated_mapping = task_git_bridge.get_task_mapping(task_id)
        assert '1.1' in updated_mapping.requirement_refs
        
        print(f"✅ Task {task_id} lifecycle test completed successfully")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])