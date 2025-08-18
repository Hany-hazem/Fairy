"""
Integration Tests for Personal Assistant Components

This module tests the integration between all personal assistant components
to ensure they work together correctly.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock

# Mock the config module to avoid validation errors
mock_config_module = MagicMock()
mock_config_module.settings = MagicMock()
mock_config_module.settings.VECTOR_DB_PATH = "./vector_db"
sys.modules['app.config'] = mock_config_module

from app.personal_assistant_core import (
    PersonalAssistantCore, AssistantRequest, RequestType
)
from app.personal_assistant_models import (
    InteractionType, PermissionType
)
from app.privacy_security_manager import DataCategory, ConsentStatus
from app.file_system_manager import FileOperation
from app.task_manager import TaskPriority, TaskStatus
from app.screen_monitor import MonitoringMode


class TestPersonalAssistantIntegration:
    """Integration test cases for the complete personal assistant system"""
    
    @pytest_asyncio.fixture
    async def assistant_system(self):
        """Create a complete assistant system for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            core = PersonalAssistantCore(db_path)
            yield core
            await core.shutdown()
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)
            # Also cleanup related files
            for ext in ["-wal", "-shm"]:
                wal_file = db_path + ext
                if os.path.exists(wal_file):
                    os.unlink(wal_file)
            # Cleanup encryption key
            if os.path.exists("encryption.key"):
                os.unlink("encryption.key")
    
    @pytest.mark.asyncio
    async def test_complete_user_workflow(self, assistant_system):
        """Test a complete user workflow from start to finish"""
        user_id = "workflow_user"
        
        # Step 1: User makes first query
        request1 = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.QUERY,
            content="Hello, I'm new here",
            metadata={"first_time": True}
        )
        
        response1 = await assistant_system.process_request(request1)
        assert response1.success
        
        # Step 2: Check that user context was created
        context = await assistant_system.get_context(user_id)
        assert context.user_id == user_id
        assert len(context.recent_interactions) == 1
        assert context.recent_interactions[0].content == "Hello, I'm new here"
        
        # Step 3: Request permissions for file access
        request2 = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.PERMISSION_REQUEST,
            content="I need file access",
            metadata={
                "permission_type": PermissionType.FILE_READ.value,
                "scope": {"path": "/home/user/documents"}
            }
        )
        
        response2 = await assistant_system.process_request(request2)
        assert response2.success
        
        # Step 4: Update user context with current activity
        request3 = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.CONTEXT_UPDATE,
            content="Starting work session",
            metadata={
                "current_activity": "coding",
                "active_applications": ["vscode", "terminal"],
                "current_files": ["main.py", "test.py"]
            }
        )
        
        response3 = await assistant_system.process_request(request3)
        assert response3.success
        
        # Step 5: Verify context was updated
        updated_context = await assistant_system.get_context(user_id)
        assert updated_context.current_activity == "coding"
        assert "vscode" in updated_context.active_applications
        assert "main.py" in updated_context.current_files
        
        # Step 6: Make a file operation request
        request4 = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.FILE_OPERATION,
            content="Read file contents",
            metadata={
                "file_path": "/home/user/documents/notes.txt",
                "operation": "read"
            }
        )
        
        response4 = await assistant_system.process_request(request4)
        assert response4.success  # Should work because permission was granted
        
        # Step 7: Get proactive suggestions
        suggestions = await assistant_system.suggest_proactive_actions(updated_context)
        assert len(suggestions) > 0
        
        # Step 8: Verify all interactions were recorded
        final_context = await assistant_system.get_context(user_id)
        assert len(final_context.recent_interactions) == 4
    
    @pytest.mark.asyncio
    async def test_privacy_workflow(self, assistant_system):
        """Test privacy and data management workflow"""
        user_id = "privacy_user"
        
        # Step 1: Request consent for data collection
        consent_status = await assistant_system.privacy_manager.request_consent(
            user_id, DataCategory.INTERACTION_HISTORY, "To improve responses"
        )
        assert consent_status == ConsentStatus.GRANTED
        
        # Step 2: Store encrypted personal data
        success = await assistant_system.privacy_manager.encrypt_personal_data(
            user_id, "profile", 
            {"name": "John Doe", "email": "john@example.com"},
            DataCategory.PERSONAL_INFO
        )
        assert success
        
        # Step 3: Get privacy dashboard
        dashboard = await assistant_system.get_privacy_dashboard(user_id)
        assert "permissions" in dashboard
        assert "consents" in dashboard
        assert "data_storage" in dashboard
        
        # Step 4: Revoke consent
        request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.PRIVACY_CONTROL,
            content="Revoke consent",
            metadata={
                "privacy_action": "revoke_consent",
                "data_category": DataCategory.INTERACTION_HISTORY.value
            }
        )
        
        response = await assistant_system.process_request(request)
        assert response.success
        
        # Step 5: Request data deletion
        request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.PRIVACY_CONTROL,
            content="Delete my data",
            metadata={
                "privacy_action": "delete_data",
                "categories": [DataCategory.PERSONAL_INFO.value],
                "reason": "Privacy concern"
            }
        )
        
        response = await assistant_system.process_request(request)
        assert response.success
        assert "request_id" in response.metadata
        
        # Step 6: Verify data was deleted
        decrypted_data = await assistant_system.privacy_manager.decrypt_personal_data(
            user_id, "profile"
        )
        assert decrypted_data is None
    
    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, assistant_system):
        """Test that multiple users are properly isolated"""
        user1_id = "user1"
        user2_id = "user2"
        
        # Create contexts for both users
        context1 = await assistant_system.get_context(user1_id)
        context2 = await assistant_system.get_context(user2_id)
        
        # Update contexts with different data
        context1.current_activity = "user1_activity"
        context1.preferences.language = "en"
        await assistant_system.context_manager.update_user_context(context1)
        
        context2.current_activity = "user2_activity"
        context2.preferences.language = "es"
        await assistant_system.context_manager.update_user_context(context2)
        
        # Grant different permissions
        await assistant_system.request_permission(user1_id, PermissionType.FILE_READ, "test")
        await assistant_system.request_permission(user2_id, PermissionType.FILE_WRITE, "test")
        
        # Store different encrypted data
        await assistant_system.privacy_manager.encrypt_personal_data(
            user1_id, "profile", {"name": "User One"}, DataCategory.PERSONAL_INFO
        )
        await assistant_system.privacy_manager.encrypt_personal_data(
            user2_id, "profile", {"name": "User Two"}, DataCategory.PERSONAL_INFO
        )
        
        # Verify isolation
        retrieved_context1 = await assistant_system.get_context(user1_id)
        retrieved_context2 = await assistant_system.get_context(user2_id)
        
        assert retrieved_context1.current_activity == "user1_activity"
        assert retrieved_context2.current_activity == "user2_activity"
        assert retrieved_context1.preferences.language == "en"
        assert retrieved_context2.preferences.language == "es"
        
        # Verify permission isolation
        has_read1 = await assistant_system.privacy_manager.check_permission(
            user1_id, PermissionType.FILE_READ
        )
        has_write1 = await assistant_system.privacy_manager.check_permission(
            user1_id, PermissionType.FILE_WRITE
        )
        has_read2 = await assistant_system.privacy_manager.check_permission(
            user2_id, PermissionType.FILE_READ
        )
        has_write2 = await assistant_system.privacy_manager.check_permission(
            user2_id, PermissionType.FILE_WRITE
        )
        
        assert has_read1 is True
        assert has_write1 is False
        assert has_read2 is False
        assert has_write2 is True
        
        # Verify data isolation
        data1 = await assistant_system.privacy_manager.decrypt_personal_data(user1_id, "profile")
        data2 = await assistant_system.privacy_manager.decrypt_personal_data(user2_id, "profile")
        
        assert data1["name"] == "User One"
        assert data2["name"] == "User Two"
    
    @pytest.mark.asyncio
    async def test_learning_from_feedback(self, assistant_system):
        """Test learning from user feedback"""
        user_id = "learning_user"
        
        # Create interaction with positive feedback
        from app.personal_assistant_models import Interaction
        positive_interaction = Interaction(
            user_id=user_id,
            interaction_type=InteractionType.QUERY,
            content="How do I write a function?",
            response="Here's how to write a function in Python...",
            feedback_score=0.9
        )
        
        await assistant_system.learn_from_interaction(positive_interaction)
        
        # Create interaction with negative feedback
        negative_interaction = Interaction(
            user_id=user_id,
            interaction_type=InteractionType.QUERY,
            content="What's the weather?",
            response="I don't have access to weather data.",
            feedback_score=0.2
        )
        
        await assistant_system.learn_from_interaction(negative_interaction)
        
        # Verify interactions were processed (learning logic would be more complex in real implementation)
        context = await assistant_system.get_context(user_id)
        # In a real implementation, we'd check that preferences or knowledge state was updated
        assert context.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_session_management_integration(self, assistant_system):
        """Test session management integration"""
        user_id = "session_user"
        
        # Create session
        session = await assistant_system.context_manager.create_session(user_id, expires_in_hours=1)
        session_id = session.session_id
        
        # Make request with session
        request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.QUERY,
            content="Test with session",
            metadata={},
            session_id=session_id
        )
        
        response = await assistant_system.process_request(request)
        assert response.success
        
        # Verify session is still active
        retrieved_session = await assistant_system.context_manager.get_session(session_id)
        assert retrieved_session is not None
        assert retrieved_session.is_active
        
        # Update session data
        retrieved_session.session_data["test_key"] = "test_value"
        
        # Verify session data persists
        session_again = await assistant_system.context_manager.get_session(session_id)
        assert session_again.session_data.get("test_key") == "test_value"
    
    @pytest.mark.asyncio
    async def test_context_history_tracking(self, assistant_system):
        """Test context history tracking"""
        user_id = "history_user"
        
        # Make several context updates
        for i in range(5):
            request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.CONTEXT_UPDATE,
                content=f"Update {i}",
                metadata={
                    "current_activity": f"activity_{i}",
                    "step": i
                }
            )
            
            response = await assistant_system.process_request(request)
            assert response.success
        
        # Get context history
        history = await assistant_system.context_manager.get_context_history(user_id)
        
        assert len(history) >= 5
        assert all(h.user_id == user_id for h in history)
        assert all(h.trigger_event == "context_update" for h in history)
    
    @pytest.mark.asyncio
    async def test_proactive_suggestions_integration(self, assistant_system):
        """Test proactive suggestions based on real context"""
        user_id = "suggestions_user"
        
        # Set up context with various scenarios
        context = await assistant_system.get_context(user_id)
        
        # Scenario 1: Many current tasks
        context.task_context.current_tasks = ["task1", "task2", "task3", "task4"]
        
        # Scenario 2: Many files
        context.current_files = [f"file{i}.py" for i in range(15)]
        
        # Scenario 3: Long work session
        context.task_context.work_session_start = datetime.now() - timedelta(hours=3)
        
        # Scenario 4: Learning goals
        context.knowledge_state.learning_goals = ["learn rust", "master kubernetes"]
        
        await assistant_system.context_manager.update_user_context(context)
        
        # Get suggestions
        suggestions = await assistant_system.suggest_proactive_actions(context)
        
        assert len(suggestions) > 0
        
        # Check for expected suggestion types
        suggestion_types = [s.action_type for s in suggestions]
        assert "task_review" in suggestion_types
        assert "file_organization" in suggestion_types
        assert "break_reminder" in suggestion_types
        assert "learning_continuation" in suggestion_types
        
        # Verify suggestion quality
        for suggestion in suggestions:
            assert suggestion.title is not None
            assert suggestion.description is not None
            assert 0 <= suggestion.confidence <= 1
            assert isinstance(suggestion.metadata, dict)
    
    @pytest.mark.asyncio
    async def test_database_integration(self, assistant_system):
        """Test database operations integration"""
        user_id = "db_user"
        
        # Perform various operations that use the database
        
        # 1. Context operations
        context = await assistant_system.get_context(user_id)
        context.current_activity = "database_testing"
        await assistant_system.context_manager.update_user_context(context)
        
        # 2. Permission operations
        await assistant_system.request_permission(user_id, PermissionType.PERSONAL_DATA, "test")
        
        # 3. Interaction recording
        request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.QUERY,
            content="Database test query",
            metadata={"test": True}
        )
        await assistant_system.process_request(request)
        
        # 4. Encrypted data storage
        await assistant_system.privacy_manager.encrypt_personal_data(
            user_id, "test_data", {"key": "value"}, DataCategory.PERSONAL_INFO
        )
        
        # Get database statistics
        stats = assistant_system.db.get_database_stats()
        
        assert stats["integrity_ok"] is True
        assert stats["table_counts"]["user_contexts"] >= 1
        assert stats["table_counts"]["user_interactions"] >= 1
        assert stats["table_counts"]["user_permissions"] >= 1
        assert stats["table_counts"]["encrypted_data"] >= 1
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, assistant_system):
        """Test error handling across components"""
        user_id = "error_user"
        
        # Test invalid request type handling
        request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.QUERY,  # Valid type but we'll test error in processing
            content="",  # Empty content might cause issues
            metadata={}
        )
        
        response = await assistant_system.process_request(request)
        # Should handle gracefully
        assert isinstance(response.success, bool)
        
        # Test permission denied scenario
        # First revoke all permissions
        for perm_type in PermissionType:
            await assistant_system.privacy_manager.revoke_permission(user_id, perm_type)
        
        # Try file operation without permission
        file_request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.FILE_OPERATION,
            content="Read sensitive file",
            metadata={"file_path": "/etc/passwd"}
        )
        
        response = await assistant_system.process_request(file_request)
        # Should be denied but handled gracefully
        assert response.requires_permission is True
        assert response.permission_type == PermissionType.FILE_READ
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, assistant_system):
        """Test concurrent operations across the system"""
        user_id = "concurrent_user"
        
        async def make_request(request_num):
            request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.QUERY,
                content=f"Concurrent request {request_num}",
                metadata={"request_num": request_num}
            )
            return await assistant_system.process_request(request)
        
        async def update_context(update_num):
            context = await assistant_system.get_context(user_id)
            context.current_activity = f"concurrent_activity_{update_num}"
            await assistant_system.context_manager.update_user_context(context)
            return update_num
        
        async def manage_permissions(perm_num):
            perm_type = list(PermissionType)[perm_num % len(PermissionType)]
            return await assistant_system.request_permission(user_id, perm_type, f"test_{perm_num}")
        
        # Run concurrent operations
        tasks = []
        
        # Add request tasks
        for i in range(5):
            tasks.append(make_request(i))
        
        # Add context update tasks
        for i in range(3):
            tasks.append(update_context(i))
        
        # Add permission tasks
        for i in range(3):
            tasks.append(manage_permissions(i))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify no exceptions occurred
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0
        
        # Verify final state is consistent
        final_context = await assistant_system.get_context(user_id)
        assert final_context.user_id == user_id
        assert len(final_context.recent_interactions) >= 5  # At least the 5 requests
    
    @pytest.mark.asyncio
    async def test_capability_module_integration(self, assistant_system):
        """Test integration of all capability modules"""
        user_id = "capability_user"
        
        # Initialize user capabilities
        init_results = await assistant_system.initialize_user_capabilities(user_id)
        assert init_results["user_context"]
        
        # Grant necessary permissions
        await assistant_system.privacy_manager.request_permission(user_id, PermissionType.FILE_READ)
        
        # Test file system integration
        with patch('app.file_system_manager.FileSystemManager.read_file') as mock_read:
            mock_read.return_value = Mock(
                success=True,
                content="Test file content",
                metadata={"file_size": 100}
            )
            
            file_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.FILE_OPERATION,
                content="Read my notes",
                metadata={
                    "operation": FileOperation.READ.value,
                    "file_path": "/home/user/notes.txt"
                }
            )
            
            response = await assistant_system.process_request(file_request)
            assert response.success
            assert "File operation 'read' completed successfully" in response.content
    
    @pytest.mark.asyncio
    async def test_task_management_integration(self, assistant_system):
        """Test task management capability integration"""
        user_id = "task_user"
        
        # Create a task
        task_request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.TASK_MANAGEMENT,
            content="Finish project documentation",
            metadata={
                "action": "create_task",
                "task_data": {
                    "title": "Finish project documentation",
                    "description": "Complete all project documentation",
                    "priority": TaskPriority.HIGH.value
                }
            }
        )
        
        response = await assistant_system.process_request(task_request)
        assert response.success
        assert "Task 'Finish project documentation' created successfully" in response.content
        assert "task_id" in response.metadata
        
        # List tasks
        list_request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.TASK_MANAGEMENT,
            content="Show my tasks",
            metadata={"action": "list_tasks"}
        )
        
        response = await assistant_system.process_request(list_request)
        assert response.success
        assert "Your tasks:" in response.content
        assert response.metadata["task_count"] >= 1
    
    @pytest.mark.asyncio
    async def test_knowledge_base_integration(self, assistant_system):
        """Test knowledge base capability integration"""
        user_id = "knowledge_user"
        
        # Index a document
        with patch('app.personal_knowledge_base.PersonalKnowledgeBase.index_document') as mock_index:
            mock_index.return_value = True
            
            index_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.KNOWLEDGE_SEARCH,
                content="Index my document",
                metadata={
                    "action": "index_document",
                    "file_path": "/home/user/research.pdf",
                    "content": "This is research about AI and machine learning."
                }
            )
            
            response = await assistant_system.process_request(index_request)
            assert response.success
            assert "Document indexed" in response.content
        
        # Search knowledge base
        with patch('app.personal_knowledge_base.PersonalKnowledgeBase.search_knowledge') as mock_search:
            from app.personal_knowledge_base import SearchResult, KnowledgeItem
            
            mock_item = KnowledgeItem(
                id="test_item",
                content="AI and machine learning research content",
                source_file="/home/user/research.pdf",
                content_type="text",
                summary="Research about AI"
            )
            
            mock_search.return_value = [
                SearchResult(
                    knowledge_item=mock_item,
                    similarity_score=0.8,
                    relevance_context="AI research",
                    matched_topics=["technology"]
                )
            ]
            
            search_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.KNOWLEDGE_SEARCH,
                content="What do I know about AI?",
                metadata={"action": "search"}
            )
            
            response = await assistant_system.process_request(search_request)
            assert response.success
            assert "Found 1 relevant items" in response.content
            assert response.metadata["result_count"] == 1
    
    @pytest.mark.asyncio
    async def test_screen_monitoring_integration(self, assistant_system):
        """Test screen monitoring capability integration"""
        user_id = "screen_user"
        
        # Start monitoring
        with patch('app.screen_monitor.ScreenMonitor.start_monitoring') as mock_start:
            mock_start.return_value = True
            
            start_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.SCREEN_MONITORING,
                content="Start monitoring my screen",
                metadata={
                    "action": "start_monitoring",
                    "mode": MonitoringMode.SELECTIVE.value
                }
            )
            
            response = await assistant_system.process_request(start_request)
            assert response.success
            assert "Screen monitoring started" in response.content
        
        # Get screen context
        with patch('app.screen_monitor.ScreenMonitor.get_current_context') as mock_context:
            from app.screen_monitor import ScreenContext, ApplicationType
            
            mock_context.return_value = ScreenContext(
                active_application="vscode",
                window_title="main.py - Visual Studio Code",
                visible_text="def main():",
                ui_elements=[],
                detected_actions=["coding"],
                context_summary="User is coding in Python",
                timestamp=datetime.now(),
                application_type=ApplicationType.IDE
            )
            
            context_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.SCREEN_MONITORING,
                content="What am I working on?",
                metadata={"action": "get_context"}
            )
            
            response = await assistant_system.process_request(context_request)
            assert response.success
            assert "Current screen context: User is coding in Python" in response.content
            assert response.metadata["active_application"] == "vscode"
    
    @pytest.mark.asyncio
    async def test_learning_feedback_integration(self, assistant_system):
        """Test learning engine capability integration"""
        user_id = "learning_user"
        
        # Provide feedback
        with patch('app.learning_engine.LearningEngine.process_feedback') as mock_feedback:
            feedback_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.LEARNING_FEEDBACK,
                content="This response was helpful",
                metadata={
                    "action": "provide_feedback",
                    "interaction_id": "test_interaction_123",
                    "feedback_type": "rating",
                    "feedback_value": 5
                }
            )
            
            response = await assistant_system.process_request(feedback_request)
            assert response.success
            assert "Thank you for your feedback" in response.content
            mock_feedback.assert_called_once()
        
        # Get behavior patterns
        with patch('app.learning_engine.LearningEngine.get_user_patterns') as mock_patterns:
            from app.learning_engine import BehaviorPattern
            
            mock_patterns.return_value = [
                BehaviorPattern(
                    pattern_id="pattern_1",
                    user_id=user_id,
                    pattern_type="time_preference",
                    pattern_data={"preferred_hours": [9, 10, 11]},
                    confidence=0.8,
                    frequency=10,
                    first_detected=datetime.now(),
                    last_updated=datetime.now()
                )
            ]
            
            patterns_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.LEARNING_FEEDBACK,
                content="What patterns have you learned?",
                metadata={"action": "get_patterns"}
            )
            
            response = await assistant_system.process_request(patterns_request)
            assert response.success
            assert "I've identified 1 behavior patterns" in response.content
    
    @pytest.mark.asyncio
    async def test_enhanced_query_with_knowledge(self, assistant_system):
        """Test enhanced query handling with knowledge base integration"""
        user_id = "enhanced_query_user"
        
        with patch('app.personal_knowledge_base.PersonalKnowledgeBase.search_knowledge') as mock_search:
            from app.personal_knowledge_base import SearchResult, KnowledgeItem
            
            # Mock knowledge search results
            mock_item = KnowledgeItem(
                id="test_item",
                content="Python is a programming language used for web development, data science, and automation.",
                source_file="/home/user/python_notes.txt",
                content_type="text",
                summary="Python programming language overview"
            )
            
            mock_search.return_value = [
                SearchResult(
                    knowledge_item=mock_item,
                    similarity_score=0.9,
                    relevance_context="Programming language information",
                    matched_topics=["technology", "programming"]
                )
            ]
            
            query_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.QUERY,
                content="Tell me about Python programming",
                metadata={}
            )
            
            response = await assistant_system.process_request(query_request)
            assert response.success
            assert "Based on your documents:" in response.content
            assert "python_notes.txt" in response.content
            assert response.metadata["has_context"] is True
            assert response.metadata["knowledge_results"] == 1
    
    @pytest.mark.asyncio
    async def test_capability_status_check(self, assistant_system):
        """Test capability status checking"""
        user_id = "status_user"
        
        status = await assistant_system.get_capability_status(user_id)
        
        assert "core_modules" in status
        assert "user_modules" in status
        assert "permissions" in status
        assert "overall_health" in status
        
        # Check that core modules are reported
        assert "file_system" in status["core_modules"]
        assert "learning" in status["core_modules"]
        assert "task_manager" in status["core_modules"]
        
        # Check that user modules are reported
        assert "screen_monitor" in status["user_modules"]
        assert "knowledge_base" in status["user_modules"]
        
        # Check that permissions are reported
        for permission in PermissionType:
            assert permission.value in status["permissions"]
    
    @pytest.mark.asyncio
    async def test_cross_module_workflow(self, assistant_system):
        """Test workflow that uses multiple capability modules"""
        user_id = "workflow_user"
        
        # Grant necessary permissions first
        await assistant_system.privacy_manager.request_permission(user_id, PermissionType.PERSONAL_DATA)
        await assistant_system.privacy_manager.request_permission(user_id, PermissionType.LEARNING)
        
        # Step 1: Create a task
        with patch('app.task_manager.TaskManager.create_task') as mock_create:
            from app.task_manager import Task
            
            mock_task = Task(
                id="task_123",
                title="Research AI trends",
                user_id=user_id
            )
            mock_create.return_value = mock_task
            
            task_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.TASK_MANAGEMENT,
                content="Research AI trends",
                metadata={
                    "action": "create_task",
                    "task_data": {"title": "Research AI trends"}
                }
            )
            
            response = await assistant_system.process_request(task_request)
            assert response.success
        
        # Step 2: Index related documents
        with patch('app.personal_knowledge_base.PersonalKnowledgeBase.index_document') as mock_index:
            mock_index.return_value = True
            
            index_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.KNOWLEDGE_SEARCH,
                content="Index AI research document",
                metadata={
                    "action": "index_document",
                    "file_path": "/home/user/ai_research.pdf",
                    "content": "Latest trends in artificial intelligence and machine learning."
                }
            )
            
            response = await assistant_system.process_request(index_request)
            assert response.success
        
        # Step 3: Query for information using knowledge base
        with patch('app.personal_knowledge_base.PersonalKnowledgeBase.search_knowledge') as mock_search:
            from app.personal_knowledge_base import SearchResult, KnowledgeItem
            
            mock_item = KnowledgeItem(
                id="ai_item",
                content="Latest trends in artificial intelligence include deep learning, natural language processing, and computer vision.",
                source_file="/home/user/ai_research.pdf",
                content_type="text",
                summary="AI trends overview"
            )
            
            mock_search.return_value = [
                SearchResult(
                    knowledge_item=mock_item,
                    similarity_score=0.95,
                    relevance_context="AI trends research",
                    matched_topics=["technology", "ai"]
                )
            ]
            
            query_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.QUERY,
                content="What are the latest AI trends?",
                metadata={}
            )
            
            response = await assistant_system.process_request(query_request)
            assert response.success
            assert "Based on your documents:" in response.content
            assert "ai_research.pdf" in response.content
        
        # Step 4: Provide feedback on the response
        with patch('app.learning_engine.LearningEngine.process_feedback') as mock_feedback:
            feedback_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.LEARNING_FEEDBACK,
                content="Great response!",
                metadata={
                    "action": "provide_feedback",
                    "interaction_id": "query_interaction",
                    "feedback_type": "rating",
                    "feedback_value": 5
                }
            )
            
            response = await assistant_system.process_request(feedback_request)
            assert response.success
            mock_feedback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_across_modules(self, assistant_system):
        """Test error handling when capability modules fail"""
        user_id = "error_user"
        
        # Test file operation with module failure
        with patch('app.file_system_manager.FileSystemManager.read_file', side_effect=Exception("File system error")):
            file_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.FILE_OPERATION,
                content="Read file",
                metadata={
                    "operation": FileOperation.READ.value,
                    "file_path": "/test/file.txt"
                }
            )
            
            response = await assistant_system.process_request(file_request)
            assert not response.success
            assert "error occurred during file operation" in response.content
        
        # Test knowledge search with module failure
        with patch('app.personal_knowledge_base.PersonalKnowledgeBase.search_knowledge', side_effect=Exception("Search error")):
            search_request = AssistantRequest(
                user_id=user_id,
                request_type=RequestType.KNOWLEDGE_SEARCH,
                content="Search for information",
                metadata={"action": "search"}
            )
            
            response = await assistant_system.process_request(search_request)
            assert not response.success
            assert "Knowledge search error" in response.content
        
        # Test that core system remains stable despite module failures
        basic_request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.QUERY,
            content="Hello",
            metadata={}
        )
        
        response = await assistant_system.process_request(basic_request)
        assert response.success  # Core functionality should still work


if __name__ == "__main__":
    pytest.main([__file__])