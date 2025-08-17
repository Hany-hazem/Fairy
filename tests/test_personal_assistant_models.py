"""
Tests for Personal Assistant Models
"""

import pytest
from datetime import datetime, timedelta
from app.personal_assistant_models import (
    UserContext, UserPreferences, TaskContext, KnowledgeState,
    Interaction, InteractionType, UserPermission, PermissionType,
    SessionData, ContextHistory
)


class TestPersonalAssistantModels:
    """Test cases for personal assistant data models"""
    
    def test_user_context_creation(self):
        """Test UserContext creation with defaults"""
        context = UserContext(user_id="test_user")
        
        assert context.user_id == "test_user"
        assert context.session_id is not None
        assert context.current_activity == ""
        assert isinstance(context.active_applications, list)
        assert isinstance(context.current_files, list)
        assert isinstance(context.recent_interactions, list)
        assert isinstance(context.preferences, UserPreferences)
        assert isinstance(context.knowledge_state, KnowledgeState)
        assert isinstance(context.task_context, TaskContext)
        assert context.context_version == 1
    
    def test_user_preferences_defaults(self):
        """Test UserPreferences default values"""
        prefs = UserPreferences(user_id="test_user")
        
        assert prefs.user_id == "test_user"
        assert prefs.language == "en"
        assert prefs.timezone == "UTC"
        assert isinstance(prefs.notification_settings, dict)
        assert isinstance(prefs.privacy_settings, dict)
        assert isinstance(prefs.interface_preferences, dict)
        assert isinstance(prefs.learning_preferences, dict)
        assert isinstance(prefs.updated_at, datetime)
    
    def test_task_context_functionality(self):
        """Test TaskContext functionality"""
        task_context = TaskContext()
        
        # Test defaults
        assert isinstance(task_context.current_tasks, list)
        assert isinstance(task_context.active_projects, list)
        assert isinstance(task_context.recent_files, list)
        assert task_context.work_session_start is None
        assert task_context.focus_area is None
        assert task_context.productivity_mode is False
        
        # Test setting values
        task_context.current_tasks = ["task1", "task2"]
        task_context.productivity_mode = True
        task_context.work_session_start = datetime.now()
        
        assert len(task_context.current_tasks) == 2
        assert task_context.productivity_mode is True
        assert isinstance(task_context.work_session_start, datetime)
    
    def test_knowledge_state_functionality(self):
        """Test KnowledgeState functionality"""
        knowledge = KnowledgeState()
        
        # Test defaults
        assert isinstance(knowledge.indexed_documents, list)
        assert isinstance(knowledge.knowledge_topics, list)
        assert isinstance(knowledge.expertise_areas, dict)
        assert isinstance(knowledge.learning_goals, list)
        assert knowledge.last_knowledge_update is None
        
        # Test setting values
        knowledge.indexed_documents = ["doc1.txt", "doc2.pdf"]
        knowledge.expertise_areas = {"python": 0.8, "javascript": 0.6}
        knowledge.learning_goals = ["learn rust", "improve algorithms"]
        
        assert len(knowledge.indexed_documents) == 2
        assert knowledge.expertise_areas["python"] == 0.8
        assert "learn rust" in knowledge.learning_goals
    
    def test_interaction_creation(self):
        """Test Interaction model"""
        interaction = Interaction(
            user_id="test_user",
            interaction_type=InteractionType.QUERY,
            content="Hello",
            response="Hi there!"
        )
        
        assert interaction.user_id == "test_user"
        assert interaction.interaction_type == InteractionType.QUERY
        assert interaction.content == "Hello"
        assert interaction.response == "Hi there!"
        assert interaction.id is not None
        assert isinstance(interaction.timestamp, datetime)
        assert isinstance(interaction.context_data, dict)
        assert isinstance(interaction.metadata, dict)
        assert interaction.feedback_score is None
    
    def test_interaction_types(self):
        """Test InteractionType enum"""
        assert InteractionType.QUERY.value == "query"
        assert InteractionType.COMMAND.value == "command"
        assert InteractionType.FEEDBACK.value == "feedback"
        assert InteractionType.FILE_ACCESS.value == "file_access"
        assert InteractionType.SCREEN_CONTEXT.value == "screen_context"
    
    def test_user_permission_model(self):
        """Test UserPermission model"""
        permission = UserPermission(
            user_id="test_user",
            permission_type=PermissionType.FILE_READ,
            granted=True,
            granted_at=datetime.now()
        )
        
        assert permission.user_id == "test_user"
        assert permission.permission_type == PermissionType.FILE_READ
        assert permission.granted is True
        assert isinstance(permission.granted_at, datetime)
        assert permission.expires_at is None
        assert isinstance(permission.scope, dict)
        assert permission.revoked is False
        assert permission.revoked_at is None
    
    def test_permission_types(self):
        """Test PermissionType enum"""
        assert PermissionType.FILE_READ.value == "file_read"
        assert PermissionType.FILE_WRITE.value == "file_write"
        assert PermissionType.SCREEN_MONITOR.value == "screen_monitor"
        assert PermissionType.PERSONAL_DATA.value == "personal_data"
        assert PermissionType.LEARNING.value == "learning"
        assert PermissionType.AUTOMATION.value == "automation"
    
    def test_session_data_model(self):
        """Test SessionData model"""
        session = SessionData(
            user_id="test_user",
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        assert session.user_id == "test_user"
        assert session.session_id is not None
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_accessed, datetime)
        assert isinstance(session.session_data, dict)
        assert session.is_active is True
        assert isinstance(session.expires_at, datetime)
    
    def test_context_history_model(self):
        """Test ContextHistory model"""
        history = ContextHistory(
            user_id="test_user",
            context_snapshot={"test": "data"},
            changes=["field1", "field2"],
            trigger_event="user_action"
        )
        
        assert history.user_id == "test_user"
        assert isinstance(history.timestamp, datetime)
        assert history.context_snapshot == {"test": "data"}
        assert history.changes == ["field1", "field2"]
        assert history.trigger_event == "user_action"
        assert isinstance(history.metadata, dict)
    
    def test_model_field_validation(self):
        """Test that models handle field validation properly"""
        # Test that required fields are enforced by dataclass
        context = UserContext()  # Should work with defaults
        assert context.user_id == ""  # Default empty string
        
        # Test that we can set all fields
        context.user_id = "new_user"
        context.current_activity = "coding"
        context.active_applications = ["vscode"]
        
        assert context.user_id == "new_user"
        assert context.current_activity == "coding"
        assert "vscode" in context.active_applications
    
    def test_interaction_with_feedback(self):
        """Test Interaction with feedback score"""
        interaction = Interaction(
            user_id="test_user",
            interaction_type=InteractionType.FEEDBACK,
            content="This was helpful",
            response="Thank you for the feedback!",
            feedback_score=0.9
        )
        
        assert interaction.feedback_score == 0.9
        assert interaction.interaction_type == InteractionType.FEEDBACK
    
    def test_permission_expiration(self):
        """Test permission with expiration"""
        future_time = datetime.now() + timedelta(days=30)
        permission = UserPermission(
            user_id="test_user",
            permission_type=PermissionType.AUTOMATION,
            granted=True,
            granted_at=datetime.now(),
            expires_at=future_time
        )
        
        assert permission.expires_at == future_time
        # Test that permission is not expired yet
        assert permission.expires_at > datetime.now()
    
    def test_complex_context_scenario(self):
        """Test a complex context scenario with all components"""
        # Create a comprehensive user context
        context = UserContext(user_id="power_user")
        
        # Set up preferences
        context.preferences.language = "es"
        context.preferences.timezone = "America/New_York"
        context.preferences.learning_preferences = {"auto_learn": True, "feedback_frequency": "high"}
        
        # Set up knowledge state
        context.knowledge_state.indexed_documents = ["project1/README.md", "docs/api.md"]
        context.knowledge_state.expertise_areas = {"python": 0.9, "react": 0.7}
        context.knowledge_state.learning_goals = ["master kubernetes", "learn rust"]
        
        # Set up task context
        context.task_context.current_tasks = ["implement auth", "write tests", "deploy app"]
        context.task_context.active_projects = ["web-app", "mobile-app"]
        context.task_context.productivity_mode = True
        context.task_context.work_session_start = datetime.now() - timedelta(hours=2)
        
        # Add some interactions
        interaction1 = Interaction(
            user_id="power_user",
            interaction_type=InteractionType.QUERY,
            content="How do I implement JWT auth?",
            response="Here's how to implement JWT authentication...",
            feedback_score=0.8
        )
        
        interaction2 = Interaction(
            user_id="power_user",
            interaction_type=InteractionType.COMMAND,
            content="Run tests",
            response="Running test suite...",
            context_data={"project": "web-app", "test_type": "unit"}
        )
        
        context.recent_interactions = [interaction1, interaction2]
        context.current_files = ["auth.py", "test_auth.py", "config.py"]
        context.active_applications = ["vscode", "terminal", "browser"]
        
        # Verify the complex context
        assert context.user_id == "power_user"
        assert context.preferences.language == "es"
        assert len(context.knowledge_state.expertise_areas) == 2
        assert len(context.task_context.current_tasks) == 3
        assert len(context.recent_interactions) == 2
        assert context.task_context.productivity_mode is True
        assert len(context.current_files) == 3
        assert "vscode" in context.active_applications
        
        # Test that work session duration can be calculated
        session_duration = datetime.now() - context.task_context.work_session_start
        assert session_duration.total_seconds() > 7000  # More than 2 hours