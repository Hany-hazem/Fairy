"""
Personal Assistant API Endpoints Integration Tests

This module contains comprehensive tests for all personal assistant API endpoints
including file operations, screen monitoring, task management, knowledge base,
and privacy controls.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json

from app.main import app
from app.personal_assistant_models import PermissionType, UserContext, Interaction, InteractionType
from app.personal_assistant_core import AssistantRequest, AssistantResponse, RequestType


class TestPersonalAssistantAPIEndpoints:
    """Test suite for personal assistant API endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
        self.test_user_id = "test_user_123"
        self.test_session_id = "test_session_456"
        
    # File System Operation Tests
    
    def test_file_operation_read_success(self):
        """Test successful file read operation"""
        with patch('app.personal_assistant_core.PersonalAssistantCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            # Mock successful response
            mock_response = AssistantResponse(
                content="File content retrieved successfully",
                success=True,
                metadata={"file_path": "/test/file.txt", "file_size": 1024},
                suggestions=["Consider organizing similar files"],
                requires_permission=False
            )
            mock_core.process_request = AsyncMock(return_value=mock_response)
            
            response = self.client.post("/assistant/files/operation", json={
                "user_id": self.test_user_id,
                "operation": "read",
                "file_path": "/test/file.txt"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "File content retrieved successfully" in data["content"]
            assert data["metadata"]["file_path"] == "/test/file.txt"
            assert len(data["suggestions"]) > 0
    
    def test_file_operation_permission_required(self):
        """Test file operation requiring permission"""
        with patch('app.personal_assistant_core.PersonalAssistantCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            # Mock response requiring permission
            mock_response = AssistantResponse(
                content="File read permission required",
                success=False,
                metadata={},
                suggestions=[],
                requires_permission=True,
                permission_type=PermissionType.FILE_READ
            )
            mock_core.process_request = AsyncMock(return_value=mock_response)
            
            response = self.client.post("/assistant/files/operation", json={
                "user_id": self.test_user_id,
                "operation": "read",
                "file_path": "/sensitive/file.txt"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert data["requires_permission"] is True
            assert data["permission_type"] == "file_read"
    
    def test_list_user_files_success(self):
        """Test successful file listing"""
        with patch('app.file_system_manager.FileSystemManager') as mock_fs_class, \
             patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            mock_privacy.check_permission = AsyncMock(return_value=True)
            
            mock_fs = Mock()
            mock_fs_class.return_value = mock_fs
            mock_fs.list_files = AsyncMock(return_value=[
                {"name": "file1.txt", "path": "/test/file1.txt", "size": 1024},
                {"name": "file2.pdf", "path": "/test/file2.pdf", "size": 2048}
            ])
            
            response = self.client.get(f"/assistant/files/list/{self.test_user_id}")
            
            if response.status_code != 200:
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text}")
            assert response.status_code == 200
            data = response.json()
            assert len(data["files"]) == 2
            assert data["count"] == 2
            assert data["files"][0]["name"] == "file1.txt"
    
    def test_list_user_files_permission_denied(self):
        """Test file listing with permission denied"""
        with patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            mock_privacy.check_permission = AsyncMock(return_value=False)
            
            response = self.client.get(f"/assistant/files/list/{self.test_user_id}")
            
            assert response.status_code == 403
            assert "File read permission required" in response.json()["detail"]
    
    def test_search_files_success(self):
        """Test successful file search"""
        with patch('app.file_system_manager.FileSystemManager') as mock_fs_class, \
             patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            mock_privacy.check_permission = AsyncMock(return_value=True)
            
            mock_fs = Mock()
            mock_fs_class.return_value = mock_fs
            mock_fs.search_files = AsyncMock(return_value=[
                {"name": "document.txt", "path": "/docs/document.txt", "relevance": 0.95},
                {"name": "notes.md", "path": "/notes/notes.md", "relevance": 0.87}
            ])
            
            response = self.client.get(
                f"/assistant/files/search/{self.test_user_id}?query=important document"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 2
            assert data["query"] == "important document"
            assert data["results"][0]["relevance"] == 0.95
    
    # Screen Monitoring Tests
    
    def test_screen_monitoring_start_success(self):
        """Test successful screen monitoring start"""
        with patch('app.personal_assistant_core.PersonalAssistantCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            mock_response = AssistantResponse(
                content="Screen monitoring started successfully",
                success=True,
                metadata={"monitoring_active": True, "config": {"mode": "selective"}},
                suggestions=["Configure privacy filters"],
                requires_permission=False
            )
            mock_core.process_request = AsyncMock(return_value=mock_response)
            
            response = self.client.post("/assistant/screen/control", json={
                "user_id": self.test_user_id,
                "action": "start",
                "config": {"mode": "selective", "exclude_passwords": True}
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "monitoring started" in data["content"].lower()
    
    def test_screen_monitoring_permission_required(self):
        """Test screen monitoring requiring permission"""
        with patch('app.personal_assistant_core.PersonalAssistantCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            mock_response = AssistantResponse(
                content="Screen monitoring permission required",
                success=False,
                metadata={},
                suggestions=[],
                requires_permission=True,
                permission_type=PermissionType.SCREEN_MONITOR
            )
            mock_core.process_request = AsyncMock(return_value=mock_response)
            
            response = self.client.post("/assistant/screen/control", json={
                "user_id": self.test_user_id,
                "action": "start"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert data["requires_permission"] is True
            assert data["permission_type"] == "screen_monitor"
    
    def test_get_screen_context_success(self):
        """Test successful screen context retrieval"""
        with patch('app.screen_monitor.ScreenMonitor') as mock_screen_class, \
             patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            mock_privacy.check_permission = AsyncMock(return_value=True)
            
            mock_screen = Mock()
            mock_screen_class.return_value = mock_screen
            
            # Mock screen context
            mock_context = Mock()
            mock_context.to_dict.return_value = {
                "active_application": "VS Code",
                "window_title": "main.py - Personal Assistant",
                "visible_text": "def test_function():",
                "timestamp": datetime.now().isoformat()
            }
            
            mock_screen.get_current_context = AsyncMock(return_value=mock_context)
            mock_screen.is_monitoring_active = AsyncMock(return_value=True)
            
            response = self.client.get(f"/assistant/screen/context/{self.test_user_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["context"]["active_application"] == "VS Code"
            assert data["monitoring_active"] is True
    
    def test_get_screen_context_with_history(self):
        """Test screen context retrieval with history"""
        with patch('app.screen_monitor.ScreenMonitor') as mock_screen_class, \
             patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            mock_privacy.check_permission = AsyncMock(return_value=True)
            
            mock_screen = Mock()
            mock_screen_class.return_value = mock_screen
            
            # Mock current context
            mock_context = Mock()
            mock_context.to_dict.return_value = {
                "active_application": "VS Code",
                "timestamp": datetime.now().isoformat()
            }
            
            # Mock history
            mock_history_item = Mock()
            mock_history_item.to_dict.return_value = {
                "active_application": "Browser",
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat()
            }
            
            mock_screen.get_current_context = AsyncMock(return_value=mock_context)
            mock_screen.is_monitoring_active = AsyncMock(return_value=True)
            mock_screen.get_context_history = AsyncMock(return_value=[mock_history_item])
            
            response = self.client.get(
                f"/assistant/screen/context/{self.test_user_id}?include_history=true"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "history" in data
            assert len(data["history"]) == 1
            assert data["history"][0]["active_application"] == "Browser"
    
    # Task Management Tests
    
    def test_create_task_success(self):
        """Test successful task creation"""
        with patch('app.personal_assistant_core.PersonalAssistantCore') as mock_core_class:
            mock_core = Mock()
            mock_core_class.return_value = mock_core
            
            mock_response = AssistantResponse(
                content="Task created successfully",
                success=True,
                metadata={"task_id": "task_123", "title": "Complete project"},
                suggestions=["Set a reminder for the due date"],
                requires_permission=False
            )
            mock_core.process_request = AsyncMock(return_value=mock_response)
            
            response = self.client.post("/assistant/tasks/manage", json={
                "user_id": self.test_user_id,
                "action": "create",
                "title": "Complete project",
                "description": "Finish the personal assistant project",
                "priority": "high",
                "due_date": "2024-12-31T23:59:59"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "created successfully" in data["content"].lower()
            assert data["metadata"]["task_id"] == "task_123"
    
    def test_get_user_tasks_success(self):
        """Test successful task retrieval"""
        with patch('app.task_manager.TaskManager') as mock_task_class, \
             patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            
            mock_task_manager = Mock()
            mock_task_class.return_value = mock_task_manager
            
            # Mock tasks
            mock_task1 = Mock()
            mock_task1.to_dict.return_value = {
                "id": "task_1",
                "title": "Task 1",
                "status": "pending",
                "priority": "high"
            }
            
            mock_task2 = Mock()
            mock_task2.to_dict.return_value = {
                "id": "task_2",
                "title": "Task 2",
                "status": "completed",
                "priority": "medium"
            }
            
            mock_task_manager.get_user_tasks = AsyncMock(return_value=[mock_task1, mock_task2])
            
            response = self.client.get(f"/assistant/tasks/{self.test_user_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["tasks"]) == 2
            assert data["tasks"][0]["title"] == "Task 1"
            assert data["count"] == 2
    
    def test_get_upcoming_deadlines_success(self):
        """Test successful upcoming deadlines retrieval"""
        with patch('app.task_manager.TaskManager') as mock_task_class, \
             patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            
            mock_task_manager = Mock()
            mock_task_class.return_value = mock_task_manager
            
            mock_deadlines = [
                {
                    "task_id": "task_1",
                    "title": "Urgent task",
                    "due_date": "2024-12-25T10:00:00",
                    "days_remaining": 3
                },
                {
                    "task_id": "task_2",
                    "title": "Important meeting",
                    "due_date": "2024-12-27T14:00:00",
                    "days_remaining": 5
                }
            ]
            
            mock_task_manager.get_upcoming_deadlines = AsyncMock(return_value=mock_deadlines)
            
            response = self.client.get(f"/assistant/tasks/{self.test_user_id}/upcoming?days=7")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["deadlines"]) == 2
            assert data["days_ahead"] == 7
            assert data["deadlines"][0]["days_remaining"] == 3
    
    # Knowledge Base Tests
    
    def test_search_knowledge_base_success(self):
        """Test successful knowledge base search"""
        with patch('app.personal_knowledge_base.PersonalKnowledgeBase') as mock_kb_class, \
             patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            
            mock_kb = Mock()
            mock_kb_class.return_value = mock_kb
            
            # Mock search results
            mock_result1 = Mock()
            mock_result1.to_dict.return_value = {
                "id": "doc_1",
                "title": "Python Programming Guide",
                "content": "Python is a programming language...",
                "relevance": 0.92
            }
            
            mock_result2 = Mock()
            mock_result2.to_dict.return_value = {
                "id": "doc_2",
                "title": "API Development",
                "content": "REST APIs are...",
                "relevance": 0.85
            }
            
            mock_kb.search = AsyncMock(return_value=[mock_result1, mock_result2])
            
            response = self.client.post("/assistant/knowledge/search", json={
                "user_id": self.test_user_id,
                "query": "Python programming",
                "search_type": "semantic",
                "limit": 10
            })
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 2
            assert data["query"] == "Python programming"
            assert data["results"][0]["relevance"] == 0.92
    
    def test_get_knowledge_topics_success(self):
        """Test successful knowledge topics retrieval"""
        with patch('app.personal_knowledge_base.PersonalKnowledgeBase') as mock_kb_class, \
             patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            
            mock_kb = Mock()
            mock_kb_class.return_value = mock_kb
            
            mock_topics = [
                {"topic": "Python Programming", "document_count": 15, "expertise_level": 0.8},
                {"topic": "Machine Learning", "document_count": 8, "expertise_level": 0.6},
                {"topic": "Web Development", "document_count": 12, "expertise_level": 0.7}
            ]
            
            mock_kb.get_user_topics = AsyncMock(return_value=mock_topics)
            
            response = self.client.get(f"/assistant/knowledge/{self.test_user_id}/topics")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["topics"]) == 3
            assert data["topics"][0]["topic"] == "Python Programming"
            assert data["topics"][0]["expertise_level"] == 0.8
    
    def test_index_document_success(self):
        """Test successful document indexing"""
        with patch('app.personal_knowledge_base.PersonalKnowledgeBase') as mock_kb_class, \
             patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            
            mock_kb = Mock()
            mock_kb_class.return_value = mock_kb
            mock_kb.index_document = AsyncMock(return_value=True)
            
            response = self.client.post("/assistant/knowledge/index", json={
                "user_id": self.test_user_id,
                "document_path": "/docs/python_guide.pdf",
                "document_type": "pdf",
                "extract_entities": True
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["document_path"] == "/docs/python_guide.pdf"
            assert data["extract_entities"] is True
    
    # Privacy Control Tests
    
    def test_grant_permission_success(self):
        """Test successful permission granting"""
        with patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            mock_privacy.grant_permission = AsyncMock(return_value=True)
            
            response = self.client.post("/assistant/privacy/permissions/grant", json={
                "user_id": self.test_user_id,
                "permission_type": "file_read",
                "scope": {"directories": ["/home/user/documents"]},
                "expires_in_days": 30
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["permission_type"] == "file_read"
            assert data["expires_in_days"] == 30
    
    def test_revoke_permission_success(self):
        """Test successful permission revocation"""
        with patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            mock_privacy.revoke_permission = AsyncMock(return_value=True)
            
            response = self.client.post("/assistant/privacy/permissions/revoke", json={
                "user_id": self.test_user_id,
                "permission_type": "screen_monitor"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["permission_type"] == "screen_monitor"
    
    def test_list_user_permissions_success(self):
        """Test successful user permissions listing"""
        with patch('app.privacy_security_manager.PrivacySecurityManager') as mock_privacy_class:
            mock_privacy = Mock()
            mock_privacy_class.return_value = mock_privacy
            
            # Mock permissions
            mock_perm1 = Mock()
            mock_perm1.permission_type = PermissionType.FILE_READ
            mock_perm1.granted = True
            mock_perm1.granted_at = datetime.now()
            mock_perm1.expires_at = None
            mock_perm1.scope = {"directories": ["/home/user"]}
            mock_perm1.revoked = False
            
            mock_perm2 = Mock()
            mock_perm2.permission_type = PermissionType.SCREEN_MONITOR
            mock_perm2.granted = False
            mock_perm2.granted_at = None
            mock_perm2.expires_at = None
            mock_perm2.scope = {}
            mock_perm2.revoked = True
            
            mock_privacy.get_all_permissions = AsyncMock(return_value=[mock_perm1, mock_perm2])
            
            response = self.client.get(f"/assistant/privacy/permissions/{self.test_user_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["permissions"]) == 2
            assert data["permissions"][0]["permission_type"] == "file_read"
            assert data["permissions"][0]["granted"] is True
            assert data["permissions"][1]["permission_type"] == "screen_monitor"
            assert data["permissions"][1]["revoked"] is True
    
    # Context Management Tests
    
    def test_get_user_context_success(self):
        """Test successful user context retrieval"""
        with patch('app.user_context_manager.UserContextManager') as mock_context_class:
            mock_context_manager = Mock()
            mock_context_class.return_value = mock_context_manager
            
            # Mock user context
            mock_context = Mock()
            mock_context.to_dict.return_value = {
                "user_id": self.test_user_id,
                "current_activity": "coding",
                "active_applications": ["VS Code", "Terminal"],
                "session_start": datetime.now().isoformat()
            }
            
            mock_context_manager.get_user_context = AsyncMock(return_value=mock_context)
            
            response = self.client.get(f"/assistant/context/{self.test_user_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["context"]["user_id"] == self.test_user_id
            assert data["context"]["current_activity"] == "coding"
            assert len(data["context"]["active_applications"]) == 2
    
    def test_get_user_context_with_history(self):
        """Test user context retrieval with interaction history"""
        with patch('app.user_context_manager.UserContextManager') as mock_context_class:
            mock_context_manager = Mock()
            mock_context_class.return_value = mock_context_manager
            
            # Mock interaction
            mock_interaction = Mock()
            mock_interaction.to_dict.return_value = {
                "id": "interaction_1",
                "content": "Help me with Python",
                "response": "Sure, I can help with Python",
                "timestamp": datetime.now().isoformat()
            }
            
            # Mock user context with interactions
            mock_context = Mock()
            mock_context.to_dict.return_value = {
                "user_id": self.test_user_id,
                "current_activity": "coding"
            }
            mock_context.recent_interactions = [mock_interaction]
            
            mock_context_manager.get_user_context = AsyncMock(return_value=mock_context)
            
            response = self.client.get(
                f"/assistant/context/{self.test_user_id}?include_history=true"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "interaction_history" in data
            assert len(data["interaction_history"]) == 1
            assert data["interaction_history"][0]["content"] == "Help me with Python"
    
    def test_update_user_context_success(self):
        """Test successful user context update"""
        with patch('app.user_context_manager.UserContextManager') as mock_context_class:
            mock_context_manager = Mock()
            mock_context_class.return_value = mock_context_manager
            mock_context_manager.update_context = AsyncMock(return_value=True)
            
            context_updates = {
                "current_activity": "debugging",
                "active_applications": ["VS Code", "Browser", "Terminal"]
            }
            
            response = self.client.post("/assistant/context/update", json={
                "user_id": self.test_user_id,
                "context_updates": context_updates,
                "merge_strategy": "merge"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["merge_strategy"] == "merge"
    
    # Error Handling Tests
    
    def test_file_operation_error_handling(self):
        """Test error handling in file operations"""
        with patch('app.personal_assistant_core.PersonalAssistantCore') as mock_core_class:
            mock_core_class.side_effect = Exception("File system error")
            
            response = self.client.post("/assistant/files/operation", json={
                "user_id": self.test_user_id,
                "operation": "read",
                "file_path": "/nonexistent/file.txt"
            })
            
            assert response.status_code == 500
            assert "File system error" in response.json()["detail"]
    
    def test_invalid_permission_type(self):
        """Test handling of invalid permission type"""
        response = self.client.post("/assistant/privacy/permissions/grant", json={
            "user_id": self.test_user_id,
            "permission_type": "invalid_permission",
            "scope": {}
        })
        
        assert response.status_code == 500  # Should handle enum validation error
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        response = self.client.post("/assistant/files/operation", json={
            "operation": "read"  # Missing user_id
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_user_id_format(self):
        """Test handling of invalid user ID format"""
        response = self.client.get("/assistant/context/")  # Empty user_id
        
        assert response.status_code == 404  # Not found due to invalid path


@pytest.mark.asyncio
class TestPersonalAssistantAPIIntegration:
    """Integration tests for personal assistant API endpoints"""
    
    async def test_full_workflow_integration(self):
        """Test a complete workflow integration"""
        client = TestClient(app)
        user_id = "integration_test_user"
        
        # This would be a more comprehensive integration test
        # that tests the full workflow from permission granting
        # to file operations to knowledge base updates
        
        # For now, we'll test that the endpoints are properly registered
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test that our new endpoints are documented
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_spec = response.json()
        
        # Check that our new endpoints are in the OpenAPI spec
        paths = openapi_spec.get("paths", {})
        assert "/assistant/files/operation" in paths
        assert "/assistant/screen/control" in paths
        assert "/assistant/tasks/manage" in paths
        assert "/assistant/knowledge/search" in paths
        assert "/assistant/privacy/permissions/grant" in paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])