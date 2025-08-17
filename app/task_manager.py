"""
Task Manager Module

This module provides intelligent task and project management capabilities for the personal assistant,
including task tracking, deadline management, progress monitoring, and proactive suggestions.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

from .personal_assistant_models import UserContext, TaskContext, Interaction, InteractionType

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(Enum):
    """Task status enumeration"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class ProjectStatus(Enum):
    """Project status enumeration"""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Individual task representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    project_id: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.NOT_STARTED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    estimated_duration: Optional[int] = None  # minutes
    actual_duration: Optional[int] = None  # minutes
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # task IDs
    progress: float = 0.0  # 0.0 to 1.0
    notes: str = ""
    user_id: str = ""
    
    def is_overdue(self) -> bool:
        """Check if task is overdue"""
        return (self.due_date is not None and 
                self.due_date < datetime.now() and 
                self.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED])
    
    def days_until_due(self) -> Optional[int]:
        """Get days until due date"""
        if self.due_date is None:
            return None
        delta = self.due_date - datetime.now()
        return delta.days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'project_id': self.project_id,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'estimated_duration': self.estimated_duration,
            'actual_duration': self.actual_duration,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'progress': self.progress,
            'notes': self.notes,
            'user_id': self.user_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create from dictionary"""
        task = cls(
            id=data['id'],
            title=data['title'],
            description=data.get('description', ''),
            project_id=data.get('project_id'),
            priority=TaskPriority(data.get('priority', TaskPriority.MEDIUM.value)),
            status=TaskStatus(data.get('status', TaskStatus.NOT_STARTED.value)),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None,
            estimated_duration=data.get('estimated_duration'),
            actual_duration=data.get('actual_duration'),
            tags=data.get('tags', []),
            dependencies=data.get('dependencies', []),
            progress=data.get('progress', 0.0),
            notes=data.get('notes', ''),
            user_id=data.get('user_id', '')
        )
        return task


@dataclass
class Project:
    """Project representation containing multiple tasks"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: ProjectStatus = ProjectStatus.PLANNING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    user_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'progress': self.progress,
            'tags': self.tags,
            'user_id': self.user_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            name=data['name'],
            description=data.get('description', ''),
            status=ProjectStatus(data.get('status', ProjectStatus.PLANNING.value)),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
            end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
            due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None,
            progress=data.get('progress', 0.0),
            tags=data.get('tags', []),
            user_id=data.get('user_id', '')
        )


@dataclass
class ProductivitySuggestion:
    """Productivity suggestion for the user"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""  # time_management, prioritization, automation, etc.
    title: str = ""
    description: str = ""
    action_items: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    user_id: str = ""
    applied: bool = False


@dataclass
class TaskAnalytics:
    """Analytics data for task performance"""
    user_id: str = ""
    total_tasks: int = 0
    completed_tasks: int = 0
    overdue_tasks: int = 0
    average_completion_time: float = 0.0  # hours
    productivity_score: float = 0.0  # 0.0 to 1.0
    most_productive_hours: List[int] = field(default_factory=list)
    common_delay_patterns: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class TaskManager:
    """Intelligent task and project management system"""
    
    def __init__(self, db_path: str = "personal_assistant.db"):
        self.db_path = db_path
        self._init_database()
        self._task_cache: Dict[str, Task] = {}
        self._project_cache: Dict[str, Project] = {}
    
    def _init_database(self):
        """Initialize the database schema for task management"""
        with sqlite3.connect(self.db_path) as conn:
            # Tasks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    project_id TEXT,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    due_date TIMESTAMP,
                    estimated_duration INTEGER,
                    actual_duration INTEGER,
                    tags TEXT,
                    dependencies TEXT,
                    progress REAL DEFAULT 0.0,
                    notes TEXT,
                    user_id TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            """)
            
            # Projects table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    due_date TIMESTAMP,
                    progress REAL DEFAULT 0.0,
                    tags TEXT,
                    user_id TEXT NOT NULL
                )
            """)
            
            # Task time tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_time_entries (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    duration INTEGER,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks (id)
                )
            """)
            
            # Productivity suggestions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS productivity_suggestions (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    action_items TEXT,
                    priority TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT NOT NULL,
                    applied BOOLEAN DEFAULT FALSE
                )
            """)
            
            conn.commit()
    
    # Task Management Methods
    
    async def create_task(self, user_id: str, title: str, description: str = "",
                         project_id: Optional[str] = None, priority: TaskPriority = TaskPriority.MEDIUM,
                         due_date: Optional[datetime] = None, estimated_duration: Optional[int] = None,
                         tags: List[str] = None) -> Task:
        """Create a new task"""
        task = Task(
            title=title,
            description=description,
            project_id=project_id,
            priority=priority,
            due_date=due_date,
            estimated_duration=estimated_duration,
            tags=tags or [],
            user_id=user_id
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO tasks 
                (id, title, description, project_id, priority, status, due_date, 
                 estimated_duration, tags, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id, task.title, task.description, task.project_id,
                task.priority.value, task.status.value, 
                task.due_date.isoformat() if task.due_date else None,
                task.estimated_duration, json.dumps(task.tags), task.user_id
            ))
            conn.commit()
        
        self._task_cache[task.id] = task
        logger.info(f"Created task {task.id}: {task.title}")
        return task
    
    async def get_task(self, task_id: str, user_id: str) -> Optional[Task]:
        """Get a specific task"""
        # Check cache first
        if task_id in self._task_cache:
            task = self._task_cache[task_id]
            if task.user_id == user_id:
                return task
        
        # Load from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, title, description, project_id, priority, status,
                       created_at, updated_at, due_date, estimated_duration,
                       actual_duration, tags, dependencies, progress, notes, user_id
                FROM tasks WHERE id = ? AND user_id = ?
            """, (task_id, user_id))
            row = cursor.fetchone()
            
            if row:
                task = Task(
                    id=row[0], title=row[1], description=row[2] or "",
                    project_id=row[3], priority=TaskPriority(row[4]),
                    status=TaskStatus(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    due_date=datetime.fromisoformat(row[8]) if row[8] else None,
                    estimated_duration=row[9], actual_duration=row[10],
                    tags=json.loads(row[11]) if row[11] else [],
                    dependencies=json.loads(row[12]) if row[12] else [],
                    progress=row[13], notes=row[14] or "", user_id=row[15]
                )
                self._task_cache[task_id] = task
                return task
        
        return None
    
    async def update_task(self, task: Task) -> bool:
        """Update an existing task"""
        task.updated_at = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks SET
                    title = ?, description = ?, project_id = ?, priority = ?,
                    status = ?, updated_at = ?, due_date = ?, estimated_duration = ?,
                    actual_duration = ?, tags = ?, dependencies = ?, progress = ?, notes = ?
                WHERE id = ? AND user_id = ?
            """, (
                task.title, task.description, task.project_id, task.priority.value,
                task.status.value, task.updated_at.isoformat(),
                task.due_date.isoformat() if task.due_date else None,
                task.estimated_duration, task.actual_duration,
                json.dumps(task.tags), json.dumps(task.dependencies),
                task.progress, task.notes, task.id, task.user_id
            ))
            conn.commit()
        
        self._task_cache[task.id] = task
        logger.info(f"Updated task {task.id}: {task.title}")
        return True
    
    async def get_user_tasks(self, user_id: str, status: Optional[TaskStatus] = None,
                           project_id: Optional[str] = None, limit: int = 100) -> List[Task]:
        """Get tasks for a user with optional filtering"""
        query = """
            SELECT id, title, description, project_id, priority, status,
                   created_at, updated_at, due_date, estimated_duration,
                   actual_duration, tags, dependencies, progress, notes, user_id
            FROM tasks WHERE user_id = ?
        """
        params = [user_id]
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        tasks = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                task = Task(
                    id=row[0], title=row[1], description=row[2] or "",
                    project_id=row[3], priority=TaskPriority(row[4]),
                    status=TaskStatus(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    due_date=datetime.fromisoformat(row[8]) if row[8] else None,
                    estimated_duration=row[9], actual_duration=row[10],
                    tags=json.loads(row[11]) if row[11] else [],
                    dependencies=json.loads(row[12]) if row[12] else [],
                    progress=row[13], notes=row[14] or "", user_id=row[15]
                )
                tasks.append(task)
                self._task_cache[task.id] = task
        
        return tasks
    
    # Project Management Methods
    
    async def create_project(self, user_id: str, name: str, description: str = "",
                           due_date: Optional[datetime] = None, tags: List[str] = None) -> Project:
        """Create a new project"""
        project = Project(
            name=name,
            description=description,
            due_date=due_date,
            tags=tags or [],
            user_id=user_id
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO projects 
                (id, name, description, status, due_date, tags, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                project.id, project.name, project.description, project.status.value,
                project.due_date.isoformat() if project.due_date else None,
                json.dumps(project.tags), project.user_id
            ))
            conn.commit()
        
        self._project_cache[project.id] = project
        logger.info(f"Created project {project.id}: {project.name}")
        return project
    
    async def get_project(self, project_id: str, user_id: str) -> Optional[Project]:
        """Get a specific project"""
        # Check cache first
        if project_id in self._project_cache:
            project = self._project_cache[project_id]
            if project.user_id == user_id:
                return project
        
        # Load from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, name, description, status, created_at, updated_at,
                       start_date, end_date, due_date, progress, tags, user_id
                FROM projects WHERE id = ? AND user_id = ?
            """, (project_id, user_id))
            row = cursor.fetchone()
            
            if row:
                project = Project(
                    id=row[0], name=row[1], description=row[2] or "",
                    status=ProjectStatus(row[3]),
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5]),
                    start_date=datetime.fromisoformat(row[6]) if row[6] else None,
                    end_date=datetime.fromisoformat(row[7]) if row[7] else None,
                    due_date=datetime.fromisoformat(row[8]) if row[8] else None,
                    progress=row[9], tags=json.loads(row[10]) if row[10] else [],
                    user_id=row[11]
                )
                self._project_cache[project_id] = project
                return project
        
        return None
    
    async def get_user_projects(self, user_id: str, status: Optional[ProjectStatus] = None,
                              limit: int = 50) -> List[Project]:
        """Get projects for a user with optional filtering"""
        query = """
            SELECT id, name, description, status, created_at, updated_at,
                   start_date, end_date, due_date, progress, tags, user_id
            FROM projects WHERE user_id = ?
        """
        params = [user_id]
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        projects = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                project = Project(
                    id=row[0], name=row[1], description=row[2] or "",
                    status=ProjectStatus(row[3]),
                    created_at=datetime.fromisoformat(row[4]),
                    updated_at=datetime.fromisoformat(row[5]),
                    start_date=datetime.fromisoformat(row[6]) if row[6] else None,
                    end_date=datetime.fromisoformat(row[7]) if row[7] else None,
                    due_date=datetime.fromisoformat(row[8]) if row[8] else None,
                    progress=row[9], tags=json.loads(row[10]) if row[10] else [],
                    user_id=row[11]
                )
                projects.append(project)
                self._project_cache[project.id] = project
        
        return projects
    
    async def update_project(self, project: Project) -> bool:
        """Update an existing project"""
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
        logger.info(f"Updated project {project.id}: {project.name}")
        return True
    
    # Deadline and Progress Management
    
    async def get_upcoming_deadlines(self, user_id: str, days_ahead: int = 7) -> List[Task]:
        """Get tasks with upcoming deadlines"""
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, title, description, project_id, priority, status,
                       created_at, updated_at, due_date, estimated_duration,
                       actual_duration, tags, dependencies, progress, notes, user_id
                FROM tasks 
                WHERE user_id = ? AND due_date IS NOT NULL 
                AND due_date <= ? AND status NOT IN ('completed', 'cancelled')
                ORDER BY due_date ASC
            """, (user_id, cutoff_date.isoformat()))
            
            tasks = []
            for row in cursor.fetchall():
                task = Task(
                    id=row[0], title=row[1], description=row[2] or "",
                    project_id=row[3], priority=TaskPriority(row[4]),
                    status=TaskStatus(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    due_date=datetime.fromisoformat(row[8]) if row[8] else None,
                    estimated_duration=row[9], actual_duration=row[10],
                    tags=json.loads(row[11]) if row[11] else [],
                    dependencies=json.loads(row[12]) if row[12] else [],
                    progress=row[13], notes=row[14] or "", user_id=row[15]
                )
                tasks.append(task)
        
        return tasks
    
    async def get_overdue_tasks(self, user_id: str) -> List[Task]:
        """Get overdue tasks"""
        now = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, title, description, project_id, priority, status,
                       created_at, updated_at, due_date, estimated_duration,
                       actual_duration, tags, dependencies, progress, notes, user_id
                FROM tasks 
                WHERE user_id = ? AND due_date IS NOT NULL 
                AND due_date < ? AND status NOT IN ('completed', 'cancelled')
                ORDER BY due_date ASC
            """, (user_id, now.isoformat()))
            
            tasks = []
            for row in cursor.fetchall():
                task = Task(
                    id=row[0], title=row[1], description=row[2] or "",
                    project_id=row[3], priority=TaskPriority(row[4]),
                    status=TaskStatus(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    due_date=datetime.fromisoformat(row[8]) if row[8] else None,
                    estimated_duration=row[9], actual_duration=row[10],
                    tags=json.loads(row[11]) if row[11] else [],
                    dependencies=json.loads(row[12]) if row[12] else [],
                    progress=row[13], notes=row[14] or "", user_id=row[15]
                )
                tasks.append(task)
        
        return tasks
    
    async def update_task_progress(self, task_id: str, user_id: str, progress: float) -> bool:
        """Update task progress"""
        task = await self.get_task(task_id, user_id)
        if not task:
            return False
        
        task.progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
        task.updated_at = datetime.now()
        
        # Auto-complete if progress reaches 100%
        if task.progress >= 1.0 and task.status != TaskStatus.COMPLETED:
            task.status = TaskStatus.COMPLETED
        
        return await self.update_task(task)
    
    # Productivity and Suggestions
    
    async def generate_productivity_suggestions(self, user_id: str) -> List[ProductivitySuggestion]:
        """Generate personalized productivity suggestions"""
        suggestions = []
        
        # Get user's task data for analysis
        tasks = await self.get_user_tasks(user_id, limit=200)
        overdue_tasks = await self.get_overdue_tasks(user_id)
        upcoming_deadlines = await self.get_upcoming_deadlines(user_id, days_ahead=3)
        
        # Suggestion 1: Overdue task management
        if overdue_tasks:
            suggestions.append(ProductivitySuggestion(
                type="deadline_management",
                title="Address Overdue Tasks",
                description=f"You have {len(overdue_tasks)} overdue tasks that need attention.",
                action_items=[
                    f"Review and prioritize {len(overdue_tasks)} overdue tasks",
                    "Consider rescheduling or breaking down large tasks",
                    "Set realistic deadlines for future tasks"
                ],
                priority=TaskPriority.HIGH,
                user_id=user_id
            ))
        
        # Suggestion 2: Upcoming deadline preparation
        if upcoming_deadlines:
            suggestions.append(ProductivitySuggestion(
                type="deadline_preparation",
                title="Prepare for Upcoming Deadlines",
                description=f"You have {len(upcoming_deadlines)} tasks due in the next 3 days.",
                action_items=[
                    "Review tasks due in the next 3 days",
                    "Allocate focused time blocks for urgent tasks",
                    "Consider delegating or postponing non-critical tasks"
                ],
                priority=TaskPriority.HIGH,
                user_id=user_id
            ))
        
        # Suggestion 3: Task prioritization
        high_priority_incomplete = [t for t in tasks if t.priority == TaskPriority.HIGH and t.status != TaskStatus.COMPLETED]
        if len(high_priority_incomplete) > 5:
            suggestions.append(ProductivitySuggestion(
                type="prioritization",
                title="Focus on High-Priority Tasks",
                description=f"You have {len(high_priority_incomplete)} high-priority tasks pending.",
                action_items=[
                    "Review and validate high-priority task list",
                    "Focus on completing 2-3 high-priority tasks today",
                    "Consider if some tasks can be deprioritized"
                ],
                priority=TaskPriority.MEDIUM,
                user_id=user_id
            ))
        
        # Suggestion 4: Time estimation improvement
        tasks_with_estimates = [t for t in tasks if t.estimated_duration and t.actual_duration]
        if len(tasks_with_estimates) >= 5:
            avg_estimation_accuracy = self._calculate_estimation_accuracy(tasks_with_estimates)
            if avg_estimation_accuracy < 0.7:  # Less than 70% accurate
                suggestions.append(ProductivitySuggestion(
                    type="time_management",
                    title="Improve Time Estimation",
                    description="Your time estimates could be more accurate to better plan your day.",
                    action_items=[
                        "Track actual time spent on tasks more consistently",
                        "Review past tasks to understand estimation patterns",
                        "Add buffer time to estimates for unexpected delays"
                    ],
                    priority=TaskPriority.MEDIUM,
                    user_id=user_id
                ))
        
        # Save suggestions to database
        for suggestion in suggestions:
            await self._save_productivity_suggestion(suggestion)
        
        return suggestions
    
    async def get_task_analytics(self, user_id: str) -> TaskAnalytics:
        """Generate task analytics for the user"""
        tasks = await self.get_user_tasks(user_id, limit=500)
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        overdue_tasks = await self.get_overdue_tasks(user_id)
        
        # Calculate completion times
        completion_times = []
        for task in completed_tasks:
            if task.actual_duration:
                completion_times.append(task.actual_duration / 60.0)  # Convert to hours
        
        avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0.0
        
        # Calculate productivity score (completed vs total, considering deadlines)
        total_tasks = len(tasks)
        completed_count = len(completed_tasks)
        overdue_count = len(overdue_tasks)
        
        if total_tasks > 0:
            completion_rate = completed_count / total_tasks
            deadline_penalty = min(overdue_count / total_tasks, 0.3)  # Max 30% penalty
            productivity_score = max(0.0, completion_rate - deadline_penalty)
        else:
            productivity_score = 0.0
        
        return TaskAnalytics(
            user_id=user_id,
            total_tasks=total_tasks,
            completed_tasks=completed_count,
            overdue_tasks=overdue_count,
            average_completion_time=avg_completion_time,
            productivity_score=productivity_score,
            most_productive_hours=await self._analyze_productive_hours(user_id),
            common_delay_patterns=await self._analyze_delay_patterns(user_id)
        )
    
    # Integration with UserContext
    
    async def update_user_context_with_tasks(self, context: UserContext) -> None:
        """Update user context with current task information"""
        # Get active tasks
        active_tasks = await self.get_user_tasks(
            context.user_id, 
            status=TaskStatus.IN_PROGRESS, 
            limit=10
        )
        
        # Get active projects
        active_projects = await self.get_user_projects(
            context.user_id,
            status=ProjectStatus.ACTIVE,
            limit=5
        )
        
        # Update task context
        context.task_context.current_tasks = [t.title for t in active_tasks]
        context.task_context.active_projects = [p.name for p in active_projects]
        
        # Check if in productivity mode (has urgent tasks or approaching deadlines)
        upcoming_deadlines = await self.get_upcoming_deadlines(context.user_id, days_ahead=1)
        urgent_tasks = [t for t in active_tasks if t.priority == TaskPriority.URGENT]
        
        context.task_context.productivity_mode = len(upcoming_deadlines) > 0 or len(urgent_tasks) > 0
        
        # Set focus area based on current tasks
        if active_tasks:
            # Focus on highest priority active task
            highest_priority_task = max(active_tasks, key=lambda t: self._priority_weight(t.priority))
            context.task_context.focus_area = highest_priority_task.title
    
    # Private helper methods
    
    def _priority_weight(self, priority: TaskPriority) -> int:
        """Get numeric weight for priority comparison"""
        weights = {
            TaskPriority.LOW: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.HIGH: 3,
            TaskPriority.URGENT: 4
        }
        return weights.get(priority, 2)
    
    def _calculate_estimation_accuracy(self, tasks: List[Task]) -> float:
        """Calculate average estimation accuracy for tasks"""
        accuracies = []
        for task in tasks:
            if task.estimated_duration and task.actual_duration:
                estimated = task.estimated_duration
                actual = task.actual_duration
                accuracy = min(estimated, actual) / max(estimated, actual)
                accuracies.append(accuracy)
        
        return sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    async def _analyze_productive_hours(self, user_id: str) -> List[int]:
        """Analyze most productive hours based on task completion patterns"""
        # This would analyze task completion times to find patterns
        # For now, return common productive hours
        return [9, 10, 11, 14, 15]  # 9-11 AM and 2-3 PM
    
    async def _analyze_delay_patterns(self, user_id: str) -> List[str]:
        """Analyze common patterns that cause task delays"""
        # This would analyze overdue tasks and their characteristics
        # For now, return common delay patterns
        return [
            "Underestimated task complexity",
            "Lack of clear requirements",
            "External dependencies",
            "Interruptions and context switching"
        ]
    
    async def _save_productivity_suggestion(self, suggestion: ProductivitySuggestion) -> None:
        """Save productivity suggestion to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO productivity_suggestions 
                (id, type, title, description, action_items, priority, user_id, applied)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                suggestion.id, suggestion.type, suggestion.title, suggestion.description,
                json.dumps(suggestion.action_items), suggestion.priority.value,
                suggestion.user_id, suggestion.applied
            ))
            conn.commit()