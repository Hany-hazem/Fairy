"""
Tests for User Context Manager
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
import sqlite3
from datetime import datetime, timedelta

from app.user_context_manager import UserContextManager
from app.personal_assistant_models import (
    UserContext, Interaction, InteractionType, UserPreferences
)


class TestUserContextManager:
    """Test cases for UserContextManager"""
    
    @pytest_asyncio.fixture
    async def context_manager(self):
        """Create a test context manager with temporary database"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            manager = UserContextManager(db_path)
            yield manager
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)
            # Also cleanup WAL files
            for ext in ["-wal", "-shm"]:
                wal_file = db_path + ext
                if os.path.exists(wal_file):
                    os.unlink(wal_file)
    
    @pytest.mark.asyncio
    async def test_get_new_user_context(self, context_manager):
        """Test getting context for a new user"""
        user_id = "new_user"
        
        context = await context_manager.get_user_context(user_id)
        
        assert context.user_id == user_id
        assert context.session_id is not None
        assert context.preferences.user_id == user_id
        assert isinstance(context.recent_interactions, list)
        assert len(context.recent_interactions) == 0
        assert context.context_version == 1
    
    @pytest.mark.asyncio
    async def test_get_existing_user_context(self, context_manager):
        """Test getting context for an existing user"""
        user_id = "existing_user"
        
        # Create initial context
        context1 = await context_manager.get_user_context(user_id)
        context1.current_activity = "testing"
        await context_manager.update_user_context(context1)
        
        # Get context again
        context2 = await context_manager.get_user_context(user_id)
        
        assert context2.user_id == user_id
        assert context2.current_activity == "testing"
        assert context2.context_version > 1
    
    @pytest.mark.asyncio
    async def test_update_user_context(self, context_manager):
        """Test updating user context"""
        user_id = "test_user"
        
        # Get initial context
        context = await context_manager.get_user_context(user_id)
        initial_version = context.context_version
        initial_activity = context.last_activity
        
        # Update context
        context.current_activity = "coding"
        context.active_applications = ["vscode", "terminal"]
        context.preferences.language = "es"
        
        await context_manager.update_user_context(context)
        
        # Verify update
        updated_context = await context_manager.get_user_context(user_id)
        assert updated_context.current_activity == "coding"
        assert "vscode" in updated_context.active_applications
        assert updated_context.preferences.language == "es"
        assert updated_context.context_version > initial_version
        assert updated_context.last_activity > initial_activity
    
    @pytest.mark.asyncio
    async def test_create_session(self, context_manager):
        """Test creating a user session"""
        user_id = "session_user"
        
        session = await context_manager.create_session(user_id, expires_in_hours=12)
        
        assert session.user_id == user_id
        assert session.session_id is not None
        assert session.is_active is True
        assert session.expires_at is not None
        assert session.expires_at > datetime.now()
        
        # Verify session is stored
        retrieved_session = await context_manager.get_session(session.session_id)
        assert retrieved_session is not None
        assert retrieved_session.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_get_session(self, context_manager):
        """Test retrieving a session"""
        user_id = "session_user"
        
        # Create session
        session = await context_manager.create_session(user_id)
        session_id = session.session_id
        
        # Retrieve session
        retrieved_session = await context_manager.get_session(session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == session_id
        assert retrieved_session.user_id == user_id
        assert retrieved_session.is_active is True
    
    @pytest.mark.asyncio
    async def test_expired_session(self, context_manager):
        """Test handling of expired sessions"""
        user_id = "expired_user"
        
        # Create session that expires immediately
        session = await context_manager.create_session(user_id, expires_in_hours=0)
        session_id = session.session_id
        
        # Manually set expiration to past
        with context_manager.get_connection() as conn:
            conn.execute("""
                UPDATE user_sessions 
                SET expires_at = ? 
                WHERE session_id = ?
            """, (datetime.now() - timedelta(hours=1), session_id))
            conn.commit()
        
        # Try to retrieve expired session
        retrieved_session = await context_manager.get_session(session_id)
        
        assert retrieved_session is None
    
    @pytest.mark.asyncio
    async def test_add_interaction(self, context_manager):
        """Test adding interactions to user context"""
        user_id = "interaction_user"
        
        # Create interaction
        interaction = Interaction(
            interaction_type=InteractionType.QUERY,
            content="Hello, how are you?",
            response="I'm doing well, thank you!",
            feedback_score=0.8
        )
        
        await context_manager.add_interaction(user_id, interaction)
        
        # Verify interaction was added
        context = await context_manager.get_user_context(user_id)
        assert len(context.recent_interactions) == 1
        
        stored_interaction = context.recent_interactions[0]
        assert stored_interaction.user_id == user_id
        assert stored_interaction.content == "Hello, how are you?"
        assert stored_interaction.response == "I'm doing well, thank you!"
        assert stored_interaction.feedback_score == 0.8
    
    @pytest.mark.asyncio
    async def test_interaction_limit(self, context_manager):
        """Test that interaction history is limited"""
        user_id = "heavy_user"
        
        # Add many interactions
        for i in range(60):  # More than the 50 limit
            interaction = Interaction(
                interaction_type=InteractionType.QUERY,
                content=f"Query {i}",
                response=f"Response {i}"
            )
            await context_manager.add_interaction(user_id, interaction)
        
        # Verify only recent interactions are kept
        context = await context_manager.get_user_context(user_id)
        assert len(context.recent_interactions) == 50
        
        # Verify the most recent interactions are kept
        last_interaction = context.recent_interactions[-1]
        assert "Query 59" in last_interaction.content
    
    @pytest.mark.asyncio
    async def test_get_context_history(self, context_manager):
        """Test retrieving context history"""
        user_id = "history_user"
        
        # Create and update context multiple times
        context = await context_manager.get_user_context(user_id)
        
        context.current_activity = "activity1"
        await context_manager.update_user_context(context)
        
        context.current_activity = "activity2"
        await context_manager.update_user_context(context)
        
        context.current_activity = "activity3"
        await context_manager.update_user_context(context)
        
        # Get history
        history = await context_manager.get_context_history(user_id, limit=10)
        
        assert len(history) >= 3  # At least 3 history entries
        assert all(h.user_id == user_id for h in history)
        assert all(h.trigger_event == "context_update" for h in history)
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, context_manager):
        """Test cleanup of expired sessions"""
        user_id = "cleanup_user"
        
        # Create sessions with different expiration times
        active_session = await context_manager.create_session(user_id, expires_in_hours=24)
        expired_session = await context_manager.create_session(user_id, expires_in_hours=1)
        
        # Manually expire one session
        with context_manager.get_connection() as conn:
            conn.execute("""
                UPDATE user_sessions 
                SET expires_at = ? 
                WHERE session_id = ?
            """, (datetime.now() - timedelta(hours=1), expired_session.session_id))
            conn.commit()
        
        # Run cleanup
        await context_manager.cleanup_expired_sessions()
        
        # Verify active session still exists
        active_retrieved = await context_manager.get_session(active_session.session_id)
        assert active_retrieved is not None
        assert active_retrieved.is_active is True
        
        # Verify expired session is marked inactive
        expired_retrieved = await context_manager.get_session(expired_session.session_id)
        assert expired_retrieved is None
    
    @pytest.mark.asyncio
    async def test_context_serialization(self, context_manager):
        """Test context serialization and deserialization"""
        user_id = "serialize_user"
        
        # Create complex context
        context = await context_manager.get_user_context(user_id)
        context.current_activity = "complex_task"
        context.active_applications = ["app1", "app2", "app3"]
        context.current_files = ["file1.py", "file2.js"]
        
        # Update preferences
        context.preferences.language = "fr"
        context.preferences.timezone = "Europe/Paris"
        context.preferences.notification_settings = {"email": True, "push": False}
        
        # Update knowledge state
        context.knowledge_state.indexed_documents = ["doc1.md", "doc2.pdf"]
        context.knowledge_state.expertise_areas = {"python": 0.9, "javascript": 0.7}
        context.knowledge_state.learning_goals = ["learn rust", "master docker"]
        
        # Update task context
        context.task_context.current_tasks = ["task1", "task2"]
        context.task_context.active_projects = ["project1"]
        context.task_context.productivity_mode = True
        context.task_context.work_session_start = datetime.now()
        
        # Save context
        await context_manager.update_user_context(context)
        
        # Clear cache and retrieve again
        if user_id in context_manager._active_contexts:
            del context_manager._active_contexts[user_id]
        
        retrieved_context = await context_manager.get_user_context(user_id)
        
        # Verify all data was preserved
        assert retrieved_context.current_activity == "complex_task"
        assert retrieved_context.active_applications == ["app1", "app2", "app3"]
        assert retrieved_context.current_files == ["file1.py", "file2.js"]
        assert retrieved_context.preferences.language == "fr"
        assert retrieved_context.preferences.timezone == "Europe/Paris"
        assert retrieved_context.preferences.notification_settings["email"] is True
        assert retrieved_context.knowledge_state.expertise_areas["python"] == 0.9
        assert "learn rust" in retrieved_context.knowledge_state.learning_goals
        assert retrieved_context.task_context.productivity_mode is True
        assert len(retrieved_context.task_context.current_tasks) == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_context_access(self, context_manager):
        """Test concurrent access to user context"""
        user_id = "concurrent_user"
        
        async def update_context(activity_name):
            context = await context_manager.get_user_context(user_id)
            context.current_activity = activity_name
            await context_manager.update_user_context(context)
            return context.context_version
        
        # Run concurrent updates
        tasks = [
            update_context("activity1"),
            update_context("activity2"),
            update_context("activity3")
        ]
        
        versions = await asyncio.gather(*tasks)
        
        # Verify all updates completed
        assert len(versions) == 3
        assert all(v > 0 for v in versions)
        
        # Get final context
        final_context = await context_manager.get_user_context(user_id)
        assert final_context.current_activity in ["activity1", "activity2", "activity3"]
    
    @pytest.mark.asyncio
    async def test_database_integrity(self, context_manager):
        """Test database integrity after operations"""
        user_id = "integrity_user"
        
        # Perform various operations
        context = await context_manager.get_user_context(user_id)
        
        # Add interactions
        for i in range(5):
            interaction = Interaction(
                interaction_type=InteractionType.QUERY,
                content=f"Query {i}",
                response=f"Response {i}"
            )
            await context_manager.add_interaction(user_id, interaction)
        
        # Update context multiple times
        for i in range(3):
            context.current_activity = f"activity_{i}"
            await context_manager.update_user_context(context)
        
        # Create sessions
        for i in range(2):
            await context_manager.create_session(user_id)
        
        # Verify database integrity by checking counts
        with context_manager.get_connection() as conn:
            # Check user_contexts table
            cursor = conn.execute("SELECT COUNT(*) FROM user_contexts WHERE user_id = ?", (user_id,))
            context_count = cursor.fetchone()[0]
            assert context_count == 1
            
            # Check user_interactions table
            cursor = conn.execute("SELECT COUNT(*) FROM user_interactions WHERE user_id = ?", (user_id,))
            interaction_count = cursor.fetchone()[0]
            assert interaction_count == 5
            
            # Check context_history table
            cursor = conn.execute("SELECT COUNT(*) FROM context_history WHERE user_id = ?", (user_id,))
            history_count = cursor.fetchone()[0]
            assert history_count >= 3  # At least 3 history entries
            
            # Check user_sessions table
            cursor = conn.execute("SELECT COUNT(*) FROM user_sessions WHERE user_id = ?", (user_id,))
            session_count = cursor.fetchone()[0]
            assert session_count == 2