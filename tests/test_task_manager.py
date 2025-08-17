"""
Unit tests for TaskManager module

Tests the intelligent task and project management capabilities including
task tracking, deadline management, progress monitoring, and productivity suggestions.
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.task_manager import (
    TaskManager, Task, Project, TaskPriority, TaskStatus, ProjectStatus,
    ProductivitySuggestion, TaskAnalytics
)
from app.personal_assistant_models import UserContext, TaskContext


class TestTaskManager:
    """Test suite for TaskManager class"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)
    
    @pytest.fixture
    def task_manager(self, temp_db):
        """Create TaskManager instance with temporary database"""
        return TaskManager(db_path=temp_db)
    
    @pytest.fixture
    def sample_user_id(self):
        """Sample user ID for testing"""
        return "test_user_123"
    
    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing"""
        return {
            'title': 'Test Task',
            'description': 'This is a test task',
            'priority': TaskPriority.HIGH,
            'due_date': datetime.now() + timedelta(days=7),
            'estimated_duration': 120,  # 2 hours
            'tags': ['test', 'important']
        }
    
    @pytest.fixture
    def sample_project_data(self):
        """Sample project data for testing"""
        return {
            'name': 'Test Project',
            'description': 'This is a test project',
            'due_date': datetime.now() + timedelta(days=30),
            'tags': ['test', 'project']
        }
    
    # Task Management Tests
    
    @pytest.mark.asyncio
    async def test_create_task(self, task_manager, sample_user_id, sample_task_data):
        """Test task creation"""
        task = await task_manager.create_task(
            user_id=sample_user_id,
            **sample_task_data
        )
        
        assert task.id is not None
        assert task.title == sample_task_data['title']
        assert task.description == sample_task_data['description']
        assert task.priority == sample_task_data['priority']
        assert task.status == TaskStatus.NOT_STARTED
        assert task.user_id == sample_user_id
        assert task.tags == sample_task_data['tags']
        assert task.progress == 0.0
    
    @pytest.mark.asyncio
    async def test_get_task(self, task_manager, sample_user_id, sample_task_data):
        """Test task retrieval"""
        # Create a task first
        created_task = await task_manager.create_task(
            user_id=sample_user_id,
            **sample_task_data
        )
        
        # Retrieve the task
        retrieved_task = await task_manager.get_task(created_task.id, sample_user_id)
        
        assert retrieved_task is not None
        assert retrieved_task.id == created_task.id
        assert retrieved_task.title == created_task.title
        assert retrieved_task.user_id == sample_user_id
    
    @pytest.mark.asyncio
    async def test_get_task_nonexistent(self, task_manager, sample_user_id):
        """Test retrieval of non-existent task"""
        task = await task_manager.get_task("nonexistent_id", sample_user_id)
        assert task is None
    
    @pytest.mark.asyncio
    async def test_update_task(self, task_manager, sample_user_id, sample_task_data):
        """Test task update"""
        # Create a task
        task = await task_manager.create_task(
            user_id=sample_user_id,
            **sample_task_data
        )
        
        # Update the task
        task.title = "Updated Task Title"
        task.status = TaskStatus.IN_PROGRESS
        task.progress = 0.5
        
        success = await task_manager.update_task(task)
        assert success is True
        
        # Verify the update
        updated_task = await task_manager.get_task(task.id, sample_user_id)
        assert updated_task.title == "Updated Task Title"
        assert updated_task.status == TaskStatus.IN_PROGRESS
        assert updated_task.progress == 0.5
    
    @pytest.mark.asyncio
    async def test_get_user_tasks(self, task_manager, sample_user_id, sample_task_data):
        """Test getting all tasks for a user"""
        # Create multiple tasks
        task1 = await task_manager.create_task(
            user_id=sample_user_id,
            title="Task 1",
            **{k: v for k, v in sample_task_data.items() if k != 'title'}
        )
        
        task2 = await task_manager.create_task(
            user_id=sample_user_id,
            title="Task 2",
            **{k: v for k, v in sample_task_data.items() if k != 'title'}
        )
        
        # Get all tasks
        tasks = await task_manager.get_user_tasks(sample_user_id)
        
        assert len(tasks) == 2
        task_titles = [t.title for t in tasks]
        assert "Task 1" in task_titles
        assert "Task 2" in task_titles
    
    @pytest.mark.asyncio
    async def test_get_user_tasks_filtered_by_status(self, task_manager, sample_user_id, sample_task_data):
        """Test getting tasks filtered by status"""
        # Create tasks with different statuses
        task1 = await task_manager.create_task(
            user_id=sample_user_id,
            title="Task 1",
            **{k: v for k, v in sample_task_data.items() if k != 'title'}
        )
        
        task2 = await task_manager.create_task(
            user_id=sample_user_id,
            title="Task 2",
            **{k: v for k, v in sample_task_data.items() if k != 'title'}
        )
        
        # Update one task to in progress
        task1.status = TaskStatus.IN_PROGRESS
        await task_manager.update_task(task1)
        
        # Get only in-progress tasks
        in_progress_tasks = await task_manager.get_user_tasks(
            sample_user_id, 
            status=TaskStatus.IN_PROGRESS
        )
        
        assert len(in_progress_tasks) == 1
        assert in_progress_tasks[0].title == "Task 1"
    
    # Project Management Tests
    
    @pytest.mark.asyncio
    async def test_create_project(self, task_manager, sample_user_id, sample_project_data):
        """Test project creation"""
        project = await task_manager.create_project(
            user_id=sample_user_id,
            **sample_project_data
        )
        
        assert project.id is not None
        assert project.name == sample_project_data['name']
        assert project.description == sample_project_data['description']
        assert project.status == ProjectStatus.PLANNING
        assert project.user_id == sample_user_id
        assert project.tags == sample_project_data['tags']
        assert project.progress == 0.0
    
    @pytest.mark.asyncio
    async def test_get_project(self, task_manager, sample_user_id, sample_project_data):
        """Test project retrieval"""
        # Create a project first
        created_project = await task_manager.create_project(
            user_id=sample_user_id,
            **sample_project_data
        )
        
        # Retrieve the project
        retrieved_project = await task_manager.get_project(created_project.id, sample_user_id)
        
        assert retrieved_project is not None
        assert retrieved_project.id == created_project.id
        assert retrieved_project.name == created_project.name
        assert retrieved_project.user_id == sample_user_id
    
    @pytest.mark.asyncio
    async def test_get_user_projects(self, task_manager, sample_user_id, sample_project_data):
        """Test getting all projects for a user"""
        # Create multiple projects
        project1 = await task_manager.create_project(
            user_id=sample_user_id,
            name="Project 1",
            **{k: v for k, v in sample_project_data.items() if k != 'name'}
        )
        
        project2 = await task_manager.create_project(
            user_id=sample_user_id,
            name="Project 2",
            **{k: v for k, v in sample_project_data.items() if k != 'name'}
        )
        
        # Get all projects
        projects = await task_manager.get_user_projects(sample_user_id)
        
        assert len(projects) == 2
        project_names = [p.name for p in projects]
        assert "Project 1" in project_names
        assert "Project 2" in project_names
    
    # Deadline Management Tests
    
    @pytest.mark.asyncio
    async def test_get_upcoming_deadlines(self, task_manager, sample_user_id):
        """Test getting upcoming deadlines"""
        # Create tasks with different due dates
        tomorrow = datetime.now() + timedelta(days=1)
        next_week = datetime.now() + timedelta(days=8)
        
        task1 = await task_manager.create_task(
            user_id=sample_user_id,
            title="Due Tomorrow",
            due_date=tomorrow
        )
        
        task2 = await task_manager.create_task(
            user_id=sample_user_id,
            title="Due Next Week",
            due_date=next_week
        )
        
        # Get upcoming deadlines (within 7 days)
        upcoming = await task_manager.get_upcoming_deadlines(sample_user_id, days_ahead=7)
        
        assert len(upcoming) == 1
        assert upcoming[0].title == "Due Tomorrow"
    
    @pytest.mark.asyncio
    async def test_get_overdue_tasks(self, task_manager, sample_user_id):
        """Test getting overdue tasks"""
        # Create tasks with past due dates
        yesterday = datetime.now() - timedelta(days=1)
        tomorrow = datetime.now() + timedelta(days=1)
        
        overdue_task = await task_manager.create_task(
            user_id=sample_user_id,
            title="Overdue Task",
            due_date=yesterday
        )
        
        future_task = await task_manager.create_task(
            user_id=sample_user_id,
            title="Future Task",
            due_date=tomorrow
        )
        
        # Get overdue tasks
        overdue = await task_manager.get_overdue_tasks(sample_user_id)
        
        assert len(overdue) == 1
        assert overdue[0].title == "Overdue Task"
    
    @pytest.mark.asyncio
    async def test_update_task_progress(self, task_manager, sample_user_id, sample_task_data):
        """Test updating task progress"""
        # Create a task
        task = await task_manager.create_task(
            user_id=sample_user_id,
            **sample_task_data
        )
        
        # Update progress to 50%
        success = await task_manager.update_task_progress(task.id, sample_user_id, 0.5)
        assert success is True
        
        # Verify progress update
        updated_task = await task_manager.get_task(task.id, sample_user_id)
        assert updated_task.progress == 0.5
        assert updated_task.status == TaskStatus.NOT_STARTED  # Should not auto-complete
        
        # Update progress to 100%
        success = await task_manager.update_task_progress(task.id, sample_user_id, 1.0)
        assert success is True
        
        # Verify auto-completion
        completed_task = await task_manager.get_task(task.id, sample_user_id)
        assert completed_task.progress == 1.0
        assert completed_task.status == TaskStatus.COMPLETED
    
    # Productivity Suggestions Tests
    
    @pytest.mark.asyncio
    async def test_generate_productivity_suggestions_overdue(self, task_manager, sample_user_id):
        """Test productivity suggestions for overdue tasks"""
        # Create an overdue task
        yesterday = datetime.now() - timedelta(days=1)
        await task_manager.create_task(
            user_id=sample_user_id,
            title="Overdue Task",
            due_date=yesterday
        )
        
        # Generate suggestions
        suggestions = await task_manager.generate_productivity_suggestions(sample_user_id)
        
        # Should have suggestion for overdue tasks
        overdue_suggestions = [s for s in suggestions if s.type == "deadline_management"]
        assert len(overdue_suggestions) > 0
        assert "overdue" in overdue_suggestions[0].description.lower()
    
    @pytest.mark.asyncio
    async def test_generate_productivity_suggestions_upcoming_deadlines(self, task_manager, sample_user_id):
        """Test productivity suggestions for upcoming deadlines"""
        # Create tasks due soon
        tomorrow = datetime.now() + timedelta(days=1)
        await task_manager.create_task(
            user_id=sample_user_id,
            title="Due Soon",
            due_date=tomorrow
        )
        
        # Generate suggestions
        suggestions = await task_manager.generate_productivity_suggestions(sample_user_id)
        
        # Should have suggestion for upcoming deadlines
        deadline_suggestions = [s for s in suggestions if s.type == "deadline_preparation"]
        assert len(deadline_suggestions) > 0
        assert "next 3 days" in deadline_suggestions[0].description.lower()
    
    @pytest.mark.asyncio
    async def test_generate_productivity_suggestions_high_priority(self, task_manager, sample_user_id):
        """Test productivity suggestions for high priority tasks"""
        # Create multiple high priority tasks
        for i in range(6):  # More than 5 to trigger suggestion
            await task_manager.create_task(
                user_id=sample_user_id,
                title=f"High Priority Task {i}",
                priority=TaskPriority.HIGH
            )
        
        # Generate suggestions
        suggestions = await task_manager.generate_productivity_suggestions(sample_user_id)
        
        # Should have suggestion for prioritization
        priority_suggestions = [s for s in suggestions if s.type == "prioritization"]
        assert len(priority_suggestions) > 0
        assert "high-priority" in priority_suggestions[0].description.lower()
    
    # Analytics Tests
    
    @pytest.mark.asyncio
    async def test_get_task_analytics(self, task_manager, sample_user_id):
        """Test task analytics generation"""
        # Create some tasks with different statuses
        completed_task = await task_manager.create_task(
            user_id=sample_user_id,
            title="Completed Task"
        )
        completed_task.status = TaskStatus.COMPLETED
        completed_task.actual_duration = 60  # 1 hour
        await task_manager.update_task(completed_task)
        
        await task_manager.create_task(
            user_id=sample_user_id,
            title="Pending Task"
        )
        
        # Create overdue task
        yesterday = datetime.now() - timedelta(days=1)
        await task_manager.create_task(
            user_id=sample_user_id,
            title="Overdue Task",
            due_date=yesterday
        )
        
        # Get analytics
        analytics = await task_manager.get_task_analytics(sample_user_id)
        
        assert analytics.user_id == sample_user_id
        assert analytics.total_tasks == 3
        assert analytics.completed_tasks == 1
        assert analytics.overdue_tasks == 1
        assert analytics.productivity_score > 0.0
        assert len(analytics.most_productive_hours) > 0
    
    # UserContext Integration Tests
    
    @pytest.mark.asyncio
    async def test_update_user_context_with_tasks(self, task_manager, sample_user_id):
        """Test updating user context with task information"""
        # Create a project and tasks
        project = await task_manager.create_project(
            user_id=sample_user_id,
            name="Active Project"
        )
        project.status = ProjectStatus.ACTIVE
        await task_manager.update_project(project)
        
        # Create active task
        task = await task_manager.create_task(
            user_id=sample_user_id,
            title="Active Task",
            project_id=project.id,
            priority=TaskPriority.URGENT
        )
        task.status = TaskStatus.IN_PROGRESS
        await task_manager.update_task(task)
        
        # Create user context
        context = UserContext(user_id=sample_user_id)
        
        # Update context with tasks
        await task_manager.update_user_context_with_tasks(context)
        
        assert "Active Task" in context.task_context.current_tasks
        assert "Active Project" in context.task_context.active_projects
        assert context.task_context.productivity_mode is True  # Due to urgent task
        assert context.task_context.focus_area == "Active Task"
    
    # Task Model Tests
    
    def test_task_is_overdue(self):
        """Test task overdue detection"""
        yesterday = datetime.now() - timedelta(days=1)
        tomorrow = datetime.now() + timedelta(days=1)
        
        overdue_task = Task(title="Overdue", due_date=yesterday, status=TaskStatus.IN_PROGRESS)
        future_task = Task(title="Future", due_date=tomorrow, status=TaskStatus.IN_PROGRESS)
        completed_task = Task(title="Completed", due_date=yesterday, status=TaskStatus.COMPLETED)
        
        assert overdue_task.is_overdue() is True
        assert future_task.is_overdue() is False
        assert completed_task.is_overdue() is False
    
    def test_task_days_until_due(self):
        """Test task days until due calculation"""
        tomorrow = datetime.now() + timedelta(days=1)
        task = Task(title="Test", due_date=tomorrow)
        
        days_until = task.days_until_due()
        assert days_until in [0, 1]  # Could be 0 or 1 depending on exact timing
        
        # Test task without due date
        task_no_due = Task(title="No Due Date")
        assert task_no_due.days_until_due() is None
    
    def test_task_serialization(self):
        """Test task to_dict and from_dict methods"""
        original_task = Task(
            title="Test Task",
            description="Test Description",
            priority=TaskPriority.HIGH,
            status=TaskStatus.IN_PROGRESS,
            tags=["test", "important"],
            progress=0.5
        )
        
        # Convert to dict and back
        task_dict = original_task.to_dict()
        restored_task = Task.from_dict(task_dict)
        
        assert restored_task.title == original_task.title
        assert restored_task.description == original_task.description
        assert restored_task.priority == original_task.priority
        assert restored_task.status == original_task.status
        assert restored_task.tags == original_task.tags
        assert restored_task.progress == original_task.progress
    
    def test_project_serialization(self):
        """Test project to_dict and from_dict methods"""
        original_project = Project(
            name="Test Project",
            description="Test Description",
            status=ProjectStatus.ACTIVE,
            tags=["test", "project"],
            progress=0.3
        )
        
        # Convert to dict and back
        project_dict = original_project.to_dict()
        restored_project = Project.from_dict(project_dict)
        
        assert restored_project.name == original_project.name
        assert restored_project.description == original_project.description
        assert restored_project.status == original_project.status
        assert restored_project.tags == original_project.tags
        assert restored_project.progress == original_project.progress
    
    # Helper method tests
    
    def test_priority_weight(self, task_manager):
        """Test priority weight calculation"""
        assert task_manager._priority_weight(TaskPriority.LOW) == 1
        assert task_manager._priority_weight(TaskPriority.MEDIUM) == 2
        assert task_manager._priority_weight(TaskPriority.HIGH) == 3
        assert task_manager._priority_weight(TaskPriority.URGENT) == 4
    
    def test_calculate_estimation_accuracy(self, task_manager):
        """Test estimation accuracy calculation"""
        tasks = [
            Task(title="Task 1", estimated_duration=60, actual_duration=60),  # 100% accurate
            Task(title="Task 2", estimated_duration=60, actual_duration=120),  # 50% accurate
            Task(title="Task 3", estimated_duration=120, actual_duration=60),  # 50% accurate
        ]
        
        accuracy = task_manager._calculate_estimation_accuracy(tasks)
        expected_accuracy = (1.0 + 0.5 + 0.5) / 3  # Average of accuracies
        assert abs(accuracy - expected_accuracy) < 0.01
    
    # Error handling tests
    
    @pytest.mark.asyncio
    async def test_update_task_progress_nonexistent_task(self, task_manager, sample_user_id):
        """Test updating progress for non-existent task"""
        success = await task_manager.update_task_progress("nonexistent_id", sample_user_id, 0.5)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_progress_clamping(self, task_manager, sample_user_id, sample_task_data):
        """Test that progress is clamped between 0 and 1"""
        task = await task_manager.create_task(
            user_id=sample_user_id,
            **sample_task_data
        )
        
        # Test negative progress
        await task_manager.update_task_progress(task.id, sample_user_id, -0.5)
        updated_task = await task_manager.get_task(task.id, sample_user_id)
        assert updated_task.progress == 0.0
        
        # Test progress > 1
        await task_manager.update_task_progress(task.id, sample_user_id, 1.5)
        updated_task = await task_manager.get_task(task.id, sample_user_id)
        assert updated_task.progress == 1.0


# Additional helper method for project updates (needed for tests)
async def update_project(self, project: Project) -> bool:
    """Update an existing project"""
    import sqlite3
    import json
    
    project.updated_at = datetime.now()
    
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            UPDATE projects SET
                name = ?, description = ?, status = ?, updated_at = ?,
                start_date = ?, end_date = ?, due_date = ?, progress = ?, tags = ?
            WHERE id = ? AND user_id = ?
        """, (
            project.name, project.description, project.status.value,
            project.updated_at.isoformat(),
            project.start_date.isoformat() if project.start_date else None,
            project.end_date.isoformat() if project.end_date else None,
            project.due_date.isoformat() if project.due_date else None,
            project.progress, json.dumps(project.tags),
            project.id, project.user_id
        ))
        conn.commit()
    
    self._project_cache[project.id] = project
    return True

# Monkey patch the update_project method for testing
TaskManager.update_project = update_project


if __name__ == "__main__":
    pytest.main([__file__])