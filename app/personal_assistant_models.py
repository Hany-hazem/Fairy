"""
Personal Assistant Data Models

This module contains the core data models for the personal assistant functionality,
including user context, session management, and related data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import uuid


class InteractionType(Enum):
    """Types of user interactions"""
    QUERY = "query"
    COMMAND = "command"
    FEEDBACK = "feedback"
    FILE_ACCESS = "file_access"
    SCREEN_CONTEXT = "screen_context"


class PermissionType(Enum):
    """Types of permissions for user data access"""
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SCREEN_MONITOR = "screen_monitor"
    PERSONAL_DATA = "personal_data"
    LEARNING = "learning"
    AUTOMATION = "automation"


@dataclass
class Interaction:
    """Represents a single user interaction"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    interaction_type: InteractionType = InteractionType.QUERY
    content: str = ""
    response: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    context_data: Dict[str, Any] = field(default_factory=dict)
    feedback_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreferences:
    """User preferences and settings"""
    user_id: str = ""
    language: str = "en"
    timezone: str = "UTC"
    notification_settings: Dict[str, bool] = field(default_factory=dict)
    privacy_settings: Dict[str, Any] = field(default_factory=dict)
    interface_preferences: Dict[str, Any] = field(default_factory=dict)
    learning_preferences: Dict[str, bool] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TaskContext:
    """Current task and project context"""
    current_tasks: List[str] = field(default_factory=list)
    active_projects: List[str] = field(default_factory=list)
    recent_files: List[str] = field(default_factory=list)
    work_session_start: Optional[datetime] = None
    focus_area: Optional[str] = None
    productivity_mode: bool = False


@dataclass
class KnowledgeState:
    """User's knowledge base state"""
    indexed_documents: List[str] = field(default_factory=list)
    knowledge_topics: List[str] = field(default_factory=list)
    expertise_areas: Dict[str, float] = field(default_factory=dict)
    learning_goals: List[str] = field(default_factory=list)
    last_knowledge_update: Optional[datetime] = None


@dataclass
class UserContext:
    """Complete user context including current state and history"""
    user_id: str = ""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_activity: str = ""
    active_applications: List[str] = field(default_factory=list)
    current_files: List[str] = field(default_factory=list)
    recent_interactions: List[Interaction] = field(default_factory=list)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    knowledge_state: KnowledgeState = field(default_factory=KnowledgeState)
    task_context: TaskContext = field(default_factory=TaskContext)
    session_start: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    context_version: int = 1


@dataclass
class UserPermission:
    """User permission for specific data access"""
    user_id: str = ""
    permission_type: PermissionType = PermissionType.PERSONAL_DATA
    granted: bool = False
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    scope: Dict[str, Any] = field(default_factory=dict)
    revoked: bool = False
    revoked_at: Optional[datetime] = None


@dataclass
class SessionData:
    """Session-specific data and state"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    session_data: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    expires_at: Optional[datetime] = None


@dataclass
class ContextHistory:
    """Historical context data for tracking changes over time"""
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    changes: List[str] = field(default_factory=list)
    trigger_event: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)