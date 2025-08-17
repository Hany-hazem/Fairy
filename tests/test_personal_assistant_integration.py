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
from datetime import datetime, timedelta

from app.personal_assistant_core import (
    PersonalAssistantCore, AssistantRequest, RequestType
)
from app.personal_assistant_models import (
    InteractionType, PermissionType
)
from app.privacy_security_manager import DataCategory, ConsentStatus


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