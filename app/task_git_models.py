"""
Task-Git Integration Models

This module defines the data models for tracking tasks and their Git operations,
providing the foundation for task-based version control integration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class TaskStatus(Enum):
    """Task status enumeration for Git integration"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    MERGED = "merged"
    ABANDONED = "abandoned"
    CONFLICT = "conflict"


class MergeStatus(Enum):
    """Merge status for task branches"""
    PENDING = "pending"
    READY = "ready"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GitCommit:
    """Represents a Git commit with task context"""
    hash: str
    message: str
    author: str
    timestamp: datetime
    files_changed: List[str]
    task_id: Optional[str] = None
    requirement_refs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'hash': self.hash,
            'message': self.message,
            'author': self.author,
            'timestamp': self.timestamp.isoformat(),
            'files_changed': self.files_changed,
            'task_id': self.task_id,
            'requirement_refs': self.requirement_refs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GitCommit':
        """Create from dictionary"""
        return cls(
            hash=data['hash'],
            message=data['message'],
            author=data['author'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            files_changed=data['files_changed'],
            task_id=data.get('task_id'),
            requirement_refs=data.get('requirement_refs', [])
        )


@dataclass
class TaskGitMapping:
    """Maps tasks to Git branches and commits"""
    task_id: str
    branch_name: str
    commits: List[str] = field(default_factory=list)
    start_commit: Optional[str] = None
    completion_commit: Optional[str] = None
    status: TaskStatus = TaskStatus.NOT_STARTED
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    merge_conflicts: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    merge_status: MergeStatus = MergeStatus.PENDING
    requirement_refs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'branch_name': self.branch_name,
            'commits': self.commits,
            'start_commit': self.start_commit,
            'completion_commit': self.completion_commit,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'merge_conflicts': self.merge_conflicts,
            'dependencies': self.dependencies,
            'merge_status': self.merge_status.value,
            'requirement_refs': self.requirement_refs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskGitMapping':
        """Create from dictionary"""
        return cls(
            task_id=data['task_id'],
            branch_name=data['branch_name'],
            commits=data.get('commits', []),
            start_commit=data.get('start_commit'),
            completion_commit=data.get('completion_commit'),
            status=TaskStatus(data.get('status', TaskStatus.NOT_STARTED.value)),
            created_at=datetime.fromisoformat(data['created_at']),
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            merge_conflicts=data.get('merge_conflicts', []),
            dependencies=data.get('dependencies', []),
            merge_status=MergeStatus(data.get('merge_status', MergeStatus.PENDING.value)),
            requirement_refs=data.get('requirement_refs', [])
        )


@dataclass
class TaskGitMetrics:
    """Git metrics for a specific task"""
    task_id: str
    total_commits: int
    files_modified: int
    lines_added: int
    lines_deleted: int
    duration_hours: float
    branch_age_days: int
    merge_conflicts_count: int
    dependency_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'total_commits': self.total_commits,
            'files_modified': self.files_modified,
            'lines_added': self.lines_added,
            'lines_deleted': self.lines_deleted,
            'duration_hours': self.duration_hours,
            'branch_age_days': self.branch_age_days,
            'merge_conflicts_count': self.merge_conflicts_count,
            'dependency_count': self.dependency_count
        }


@dataclass
class TaskReport:
    """Comprehensive task completion report"""
    task_id: str
    task_description: str
    status: TaskStatus
    git_metrics: TaskGitMetrics
    commits: List[GitCommit]
    requirements_covered: List[str]
    dependencies_resolved: List[str]
    merge_conflicts_resolved: List[str]
    completion_notes: str
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'task_description': self.task_description,
            'status': self.status.value,
            'git_metrics': self.git_metrics.to_dict(),
            'commits': [commit.to_dict() for commit in self.commits],
            'requirements_covered': self.requirements_covered,
            'dependencies_resolved': self.dependencies_resolved,
            'merge_conflicts_resolved': self.merge_conflicts_resolved,
            'completion_notes': self.completion_notes,
            'generated_at': self.generated_at.isoformat()
        }


@dataclass
class MergeStrategy:
    """Strategy for merging dependent tasks"""
    merge_order: List[str]
    parallel_groups: List[List[str]]
    conflict_resolution: Dict[str, str]
    estimated_duration: int  # minutes
    risk_level: str  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'merge_order': self.merge_order,
            'parallel_groups': self.parallel_groups,
            'conflict_resolution': self.conflict_resolution,
            'estimated_duration': self.estimated_duration,
            'risk_level': self.risk_level
        }