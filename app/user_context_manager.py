"""
User Context Manager

This module handles user context management, session state, and context history tracking
for the personal assistant functionality.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .personal_assistant_models import (
    UserContext, UserPreferences, TaskContext, KnowledgeState,
    SessionData, ContextHistory, Interaction, InteractionType
)

logger = logging.getLogger(__name__)


class UserContextManager:
    """Manages user context, sessions, and context history"""
    
    def __init__(self, db_path: str = "personal_assistant.db"):
        self.db_path = db_path
        self._init_database()
        self._active_contexts: Dict[str, UserContext] = {}
        self._active_sessions: Dict[str, SessionData] = {}
    
    def get_connection(self):
        """Get a database connection"""
        return sqlite3.connect(self.db_path)
    
    def _init_database(self):
        """Initialize the database schema for context management"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_contexts (
                    user_id TEXT PRIMARY KEY,
                    context_data TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_version INTEGER DEFAULT 1
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    expires_at TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_snapshot TEXT NOT NULL,
                    changes TEXT,
                    trigger_event TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_data TEXT,
                    feedback_score REAL,
                    metadata TEXT
                )
            """)
            
            conn.commit()
    
    async def get_user_context(self, user_id: str) -> UserContext:
        """Retrieve user context, creating a new one if it doesn't exist"""
        # Check if context is already loaded in memory
        if user_id in self._active_contexts:
            context = self._active_contexts[user_id]
            context.last_activity = datetime.now()
            return context
        
        # Try to load from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT context_data, context_version FROM user_contexts WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row:
                context_data = json.loads(row[0])
                context = self._deserialize_context(context_data)
                context.user_id = user_id
                context.context_version = row[1]
            else:
                # Create new context
                context = UserContext(user_id=user_id)
                context.preferences.user_id = user_id
                await self._save_context(context)
        
        # Load recent interactions
        await self._load_recent_interactions(context)
        
        # Cache in memory
        self._active_contexts[user_id] = context
        return context
    
    async def update_user_context(self, context: UserContext) -> None:
        """Update user context and save to database"""
        context.last_activity = datetime.now()
        context.context_version += 1
        
        # Save context history before updating
        await self._save_context_history(context, "context_update")
        
        # Save updated context
        await self._save_context(context)
        
        # Update in-memory cache
        self._active_contexts[context.user_id] = context
    
    async def create_session(self, user_id: str, expires_in_hours: int = 24) -> SessionData:
        """Create a new user session"""
        session = SessionData(
            user_id=user_id,
            expires_at=datetime.now() + timedelta(hours=expires_in_hours)
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_sessions 
                (session_id, user_id, session_data, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.user_id,
                json.dumps(session.session_data),
                session.expires_at,
                session.is_active
            ))
            conn.commit()
        
        self._active_sessions[session.session_id] = session
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve session data"""
        # Check memory cache first
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            if session.expires_at and session.expires_at < datetime.now():
                await self._expire_session(session_id)
                return None
            session.last_accessed = datetime.now()
            return session
        
        # Load from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT user_id, session_data, created_at, last_accessed, is_active, expires_at
                FROM user_sessions WHERE session_id = ? AND is_active = TRUE
            """, (session_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            session = SessionData(
                session_id=session_id,
                user_id=row[0],
                session_data=json.loads(row[1]),
                created_at=datetime.fromisoformat(row[2]),
                last_accessed=datetime.fromisoformat(row[3]),
                is_active=row[4],
                expires_at=datetime.fromisoformat(row[5]) if row[5] else None
            )
            
            # Check if expired
            if session.expires_at and session.expires_at < datetime.now():
                await self._expire_session(session_id)
                return None
            
            # Update last accessed
            session.last_accessed = datetime.now()
            await self._update_session_access(session)
            
            self._active_sessions[session_id] = session
            return session
    
    async def add_interaction(self, user_id: str, interaction: Interaction) -> None:
        """Add a new interaction to the user's context"""
        interaction.user_id = user_id
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_interactions 
                (id, user_id, interaction_type, content, response, context_data, 
                 feedback_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction.id,
                interaction.user_id,
                interaction.interaction_type.value,
                interaction.content,
                interaction.response,
                json.dumps(interaction.context_data),
                interaction.feedback_score,
                json.dumps(interaction.metadata)
            ))
            conn.commit()
        
        # Update context
        context = await self.get_user_context(user_id)
        context.recent_interactions.append(interaction)
        
        # Keep only recent interactions (last 50)
        if len(context.recent_interactions) > 50:
            context.recent_interactions = context.recent_interactions[-50:]
        
        await self.update_user_context(context)
    
    async def get_context_history(self, user_id: str, limit: int = 100) -> List[ContextHistory]:
        """Retrieve context history for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, context_snapshot, changes, trigger_event, metadata
                FROM context_history 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (user_id, limit))
            
            history = []
            for row in cursor.fetchall():
                history.append(ContextHistory(
                    user_id=user_id,
                    timestamp=datetime.fromisoformat(row[0]),
                    context_snapshot=json.loads(row[1]),
                    changes=json.loads(row[2]) if row[2] else [],
                    trigger_event=row[3],
                    metadata=json.loads(row[4]) if row[4] else {}
                ))
            
            return history
    
    async def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE user_sessions 
                SET is_active = FALSE 
                WHERE expires_at < ? AND is_active = TRUE
            """, (datetime.now(),))
            conn.commit()
        
        # Remove from memory cache
        expired_sessions = [
            session_id for session_id, session in self._active_sessions.items()
            if session.expires_at and session.expires_at < datetime.now()
        ]
        for session_id in expired_sessions:
            del self._active_sessions[session_id]
    
    async def _save_context(self, context: UserContext) -> None:
        """Save context to database"""
        context_data = self._serialize_context(context)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_contexts 
                (user_id, context_data, context_version)
                VALUES (?, ?, ?)
            """, (
                context.user_id,
                json.dumps(context_data),
                context.context_version
            ))
            conn.commit()
    
    async def _save_context_history(self, context: UserContext, trigger_event: str) -> None:
        """Save context snapshot to history"""
        context_snapshot = self._serialize_context(context)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO context_history 
                (user_id, context_snapshot, trigger_event)
                VALUES (?, ?, ?)
            """, (
                context.user_id,
                json.dumps(context_snapshot),
                trigger_event
            ))
            conn.commit()
    
    async def _load_recent_interactions(self, context: UserContext) -> None:
        """Load recent interactions for the context"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, interaction_type, content, response, timestamp, 
                       context_data, feedback_score, metadata
                FROM user_interactions 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            """, (context.user_id,))
            
            interactions = []
            for row in cursor.fetchall():
                interaction = Interaction(
                    id=row[0],
                    user_id=context.user_id,
                    interaction_type=InteractionType(row[1]),
                    content=row[2],
                    response=row[3] or "",
                    timestamp=datetime.fromisoformat(row[4]),
                    context_data=json.loads(row[5]) if row[5] else {},
                    feedback_score=row[6],
                    metadata=json.loads(row[7]) if row[7] else {}
                )
                interactions.append(interaction)
            
            context.recent_interactions = list(reversed(interactions))
    
    async def _expire_session(self, session_id: str) -> None:
        """Mark session as expired"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE user_sessions 
                SET is_active = FALSE 
                WHERE session_id = ?
            """, (session_id,))
            conn.commit()
        
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
    
    async def _update_session_access(self, session: SessionData) -> None:
        """Update session last accessed time"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE user_sessions 
                SET last_accessed = ?, session_data = ?
                WHERE session_id = ?
            """, (
                session.last_accessed,
                json.dumps(session.session_data),
                session.session_id
            ))
            conn.commit()
    
    def _serialize_context(self, context: UserContext) -> Dict[str, Any]:
        """Serialize context to dictionary for storage"""
        return {
            "session_id": context.session_id,
            "current_activity": context.current_activity,
            "active_applications": context.active_applications,
            "current_files": context.current_files,
            "preferences": {
                "language": context.preferences.language,
                "timezone": context.preferences.timezone,
                "notification_settings": context.preferences.notification_settings,
                "privacy_settings": context.preferences.privacy_settings,
                "interface_preferences": context.preferences.interface_preferences,
                "learning_preferences": context.preferences.learning_preferences,
                "updated_at": context.preferences.updated_at.isoformat()
            },
            "knowledge_state": {
                "indexed_documents": context.knowledge_state.indexed_documents,
                "knowledge_topics": context.knowledge_state.knowledge_topics,
                "expertise_areas": context.knowledge_state.expertise_areas,
                "learning_goals": context.knowledge_state.learning_goals,
                "last_knowledge_update": context.knowledge_state.last_knowledge_update.isoformat() if context.knowledge_state.last_knowledge_update else None
            },
            "task_context": {
                "current_tasks": context.task_context.current_tasks,
                "active_projects": context.task_context.active_projects,
                "recent_files": context.task_context.recent_files,
                "work_session_start": context.task_context.work_session_start.isoformat() if context.task_context.work_session_start else None,
                "focus_area": context.task_context.focus_area,
                "productivity_mode": context.task_context.productivity_mode
            },
            "session_start": context.session_start.isoformat(),
            "last_activity": context.last_activity.isoformat()
        }
    
    def _deserialize_context(self, data: Dict[str, Any]) -> UserContext:
        """Deserialize context from dictionary"""
        context = UserContext()
        context.session_id = data.get("session_id", context.session_id)
        context.current_activity = data.get("current_activity", "")
        context.active_applications = data.get("active_applications", [])
        context.current_files = data.get("current_files", [])
        
        # Deserialize preferences
        prefs_data = data.get("preferences", {})
        context.preferences = UserPreferences(
            language=prefs_data.get("language", "en"),
            timezone=prefs_data.get("timezone", "UTC"),
            notification_settings=prefs_data.get("notification_settings", {}),
            privacy_settings=prefs_data.get("privacy_settings", {}),
            interface_preferences=prefs_data.get("interface_preferences", {}),
            learning_preferences=prefs_data.get("learning_preferences", {}),
            updated_at=datetime.fromisoformat(prefs_data["updated_at"]) if prefs_data.get("updated_at") else datetime.now()
        )
        
        # Deserialize knowledge state
        knowledge_data = data.get("knowledge_state", {})
        context.knowledge_state = KnowledgeState(
            indexed_documents=knowledge_data.get("indexed_documents", []),
            knowledge_topics=knowledge_data.get("knowledge_topics", []),
            expertise_areas=knowledge_data.get("expertise_areas", {}),
            learning_goals=knowledge_data.get("learning_goals", []),
            last_knowledge_update=datetime.fromisoformat(knowledge_data["last_knowledge_update"]) if knowledge_data.get("last_knowledge_update") else None
        )
        
        # Deserialize task context
        task_data = data.get("task_context", {})
        context.task_context = TaskContext(
            current_tasks=task_data.get("current_tasks", []),
            active_projects=task_data.get("active_projects", []),
            recent_files=task_data.get("recent_files", []),
            work_session_start=datetime.fromisoformat(task_data["work_session_start"]) if task_data.get("work_session_start") else None,
            focus_area=task_data.get("focus_area"),
            productivity_mode=task_data.get("productivity_mode", False)
        )
        
        context.session_start = datetime.fromisoformat(data["session_start"]) if data.get("session_start") else datetime.now()
        context.last_activity = datetime.fromisoformat(data["last_activity"]) if data.get("last_activity") else datetime.now()
        
        return context