"""
Tests for Personal Assistant Core functionality
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from datetime import datetime
from pathlib import Path

from app.personal_assistant_core import (
    PersonalAssistantCore, AssistantRequest, RequestType, PermissionType
)
from app.personal_assistant_models import InteractionType


class TestPersonalAssistantCore:
    """Test cases for PersonalAssistantCore"""
    
    @pytest_asyncio.fixture
    async def assistant_core(self):
        """Create a test assistant core with temporary database"""
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
    
    @pytest.mark.asyncio
    async def test_process_query_request(self, assistant_core):
        """Test processing a basic query request"""
        request = AssistantRequest(
            user_id="test_user",
            request_type=RequestType.QUERY,
            content="Hello, how are you?",
            metadata={}
        )
        
        response = await assistant_core.process_request(request)
        
        assert response.success
        assert "Hello, how are you?" in response.content
        assert isinstance(response.suggestions, list)
    
    @pytest.mark.asyncio
    async def test_get_user_context(self, assistant_core):
        """Test getting user context"""
        user_id = "test_user"
        
        context = await assistant_core.get_context(user_id)
        
        assert context.user_id == user_id
        assert context.session_id is not None
        assert isinstance(context.recent_interactions, list)
        assert context.preferences.user_id == user_id
    
    @pytest.mark.asyncio
    async def test_context_update_request(self, assistant_core):
        """Test updating user context"""
        request = AssistantRequest(
            user_id="test_user",
            request_type=RequestType.CONTEXT_UPDATE,
            content="Update my current activity",
            metadata={
                "current_activity": "coding",
                "active_applications": ["vscode", "terminal"]
            }
        )
        
        response = await assistant_core.process_request(request)
        
        assert response.success
        assert "Context updated successfully" in response.content
        
        # Verify context was updated
        context = await assistant_core.get_context("test_user")
        assert context.current_activity == "coding"
        assert "vscode" in context.active_applications
        assert "terminal" in context.active_applications
    
    @pytest.mark.asyncio
    async def test_permission_request(self, assistant_core):
        """Test requesting permissions"""
        user_id = "test_user"
        
        # Request file read permission
        granted = await assistant_core.request_permission(
            user_id, PermissionType.FILE_READ, "To help with file management"
        )
        
        # Should be granted for non-sensitive permissions in test
        assert granted
        
        # Verify permission is recorded
        permission = await assistant_core.privacy_manager.get_permission(
            user_id, PermissionType.FILE_READ
        )
        assert permission is not None
        assert permission.granted
        assert not permission.revoked
    
    @pytest.mark.asyncio
    async def test_proactive_suggestions(self, assistant_core):
        """Test generating proactive suggestions"""
        user_id = "test_user"
        
        # Set up context with some data
        context = await assistant_core.get_context(user_id)
        context.task_context.current_tasks = ["task1", "task2", "task3"]
        context.current_files = [f"file{i}.txt" for i in range(15)]  # Many files
        await assistant_core.context_manager.update_user_context(context)
        
        # Get suggestions
        suggestions = await assistant_core.suggest_proactive_actions(context)
        
        assert len(suggestions) > 0
        
        # Should suggest task review
        task_suggestions = [s for s in suggestions if s.action_type == "task_review"]
        assert len(task_suggestions) > 0
        
        # Should suggest file organization
        file_suggestions = [s for s in suggestions if s.action_type == "file_organization"]
        assert len(file_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_privacy_dashboard(self, assistant_core):
        """Test getting privacy dashboard data"""
        user_id = "test_user"
        
        # Grant some permissions first
        await assistant_core.request_permission(user_id, PermissionType.FILE_READ, "test")
        await assistant_core.request_permission(user_id, PermissionType.PERSONAL_DATA, "test")
        
        dashboard_data = await assistant_core.get_privacy_dashboard(user_id)
        
        assert "permissions" in dashboard_data
        assert "consents" in dashboard_data
        assert "data_storage" in dashboard_data
        
        # Check that granted permissions are reflected
        permissions = dashboard_data["permissions"]
        assert permissions[PermissionType.FILE_READ.value]["granted"]
        assert permissions[PermissionType.PERSONAL_DATA.value]["granted"]
    
    @pytest.mark.asyncio
    async def test_interaction_recording(self, assistant_core):
        """Test that interactions are properly recorded"""
        user_id = "test_user"
        
        # Make a request
        request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.QUERY,
            content="Test query",
            metadata={"test": "data"}
        )
        
        response = await assistant_core.process_request(request)
        assert response.success
        
        # Check that interaction was recorded
        context = await assistant_core.get_context(user_id)
        assert len(context.recent_interactions) > 0
        
        latest_interaction = context.recent_interactions[-1]
        assert latest_interaction.user_id == user_id
        assert latest_interaction.content == "Test query"
        assert latest_interaction.interaction_type == InteractionType.QUERY
        assert latest_interaction.context_data == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_privacy_request_handling(self, assistant_core):
        """Test handling privacy-related requests"""
        user_id = "test_user"
        
        # Test data deletion request
        request = AssistantRequest(
            user_id=user_id,
            request_type=RequestType.PRIVACY_CONTROL,
            content="Delete my data",
            metadata={
                "privacy_action": "delete_data",
                "categories": ["interaction_history", "personal_info"],
                "reason": "Privacy concern"
            }
        )
        
        response = await assistant_core.process_request(request)
        
        assert response.success
        assert "deletion request submitted" in response.content.lower()
        assert "request_id" in response.metadata


if __name__ == "__main__":
    # Run a simple test
    async def run_simple_test():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            db_path = tmp_file.name
        
        try:
            core = PersonalAssistantCore(db_path)
            
            # Test basic functionality
            request = AssistantRequest(
                user_id="test_user",
                request_type=RequestType.QUERY,
                content="Hello!",
                metadata={}
            )
            
            response = await core.process_request(request)
            print(f"Response: {response.content}")
            print(f"Success: {response.success}")
            
            # Test context
            context = await core.get_context("test_user")
            print(f"User context created for: {context.user_id}")
            print(f"Session ID: {context.session_id}")
            
            await core.shutdown()
            print("Test completed successfully!")
            
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    asyncio.run(run_simple_test())