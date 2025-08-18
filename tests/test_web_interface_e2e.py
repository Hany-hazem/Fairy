#!/usr/bin/env python3
"""
End-to-end tests for the enhanced web interface
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import json

# Import the FastAPI app
from app.main import app

class TestWebInterfaceE2E:
    """End-to-end tests for the enhanced web interface"""
    
    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)
    
    def test_serve_web_interface(self):
        """Test that the main web interface is served"""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "Personal Assistant" in response.text
    
    def test_personal_assistant_chat(self):
        """Test the enhanced chat endpoint"""
        chat_data = {
            "message": "Hello, can you help me organize my files?",
            "user_id": "test_user",
            "context": {"current_tab": "chat"}
        }
        
        with patch('app.main.PersonalAssistantCore') as mock_core:
            # Mock the assistant core response
            mock_response = MagicMock()
            mock_response.content = "I'd be happy to help you organize your files!"
            mock_response.suggestions = ["Check the file manager tab"]
            mock_response.metadata = {"processed": True}
            
            mock_instance = AsyncMock()
            mock_instance.process_request.return_value = mock_response
            mock_core.return_value = mock_instance
            
            response = self.client.post("/personal-assistant/chat", json=chat_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "response" in data
            assert "session_id" in data
            assert "timestamp" in data
    
    def test_file_list_endpoint(self):
        """Test the file listing endpoint"""
        response = self.client.get("/personal-assistant/files/list?path=/home/user")
        
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert "path" in data
        assert isinstance(data["files"], list)
        
        # Check that mock data is returned
        file_names = [f["name"] for f in data["files"]]
        assert ".." in file_names
        assert "Documents" in file_names
    
    def test_file_analysis_endpoint(self):
        """Test the file analysis endpoint"""
        analysis_data = {"path": "/home/user/Documents"}
        
        response = self.client.post("/personal-assistant/files/analyze", json=analysis_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "analysis" in data
        assert "details" in data
        assert "timestamp" in data
    
    def test_file_organization_endpoint(self):
        """Test the file organization endpoint"""
        organize_data = {"path": "/home/user/Downloads"}
        
        response = self.client.post("/personal-assistant/files/organize", json=organize_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "details" in data
        assert "timestamp" in data
    
    def test_task_list_endpoint(self):
        """Test the task listing endpoint"""
        response = self.client.get("/personal-assistant/tasks/list")
        
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "timestamp" in data
        assert isinstance(data["tasks"], list)
        
        # Check mock task data
        if data["tasks"]:
            task = data["tasks"][0]
            assert "id" in task
            assert "title" in task
            assert "completed" in task
    
    def test_task_toggle_endpoint(self):
        """Test the task toggle endpoint"""
        response = self.client.post("/personal-assistant/tasks/1/toggle")
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "task_id" in data
        assert data["task_id"] == "1"
    
    def test_task_creation_endpoint(self):
        """Test the task creation endpoint"""
        task_data = {
            "title": "Test task",
            "priority": "High",
            "due_date": "Tomorrow"
        }
        
        response = self.client.post("/personal-assistant/tasks/create", json=task_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "task_id" in data
        assert "title" in data
        assert data["title"] == "Test task"
    
    def test_knowledge_recent_endpoint(self):
        """Test the recent knowledge endpoint"""
        response = self.client.get("/personal-assistant/knowledge/recent")
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "timestamp" in data
        assert isinstance(data["items"], list)
        
        # Check mock knowledge data
        if data["items"]:
            item = data["items"][0]
            assert "title" in item
            assert "snippet" in item
            assert "source" in item
    
    def test_knowledge_search_endpoint(self):
        """Test the knowledge search endpoint"""
        search_data = {"query": "Python best practices"}
        
        response = self.client.post("/personal-assistant/knowledge/search", json=search_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert "timestamp" in data
        assert data["query"] == "Python best practices"
    
    def test_permission_update_endpoint(self):
        """Test the permission update endpoint"""
        permission_data = {
            "permission": "file_access",
            "enabled": True
        }
        
        response = self.client.post("/personal-assistant/privacy/permissions", json=permission_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "permission" in data
        assert "enabled" in data
        assert data["permission"] == "file_access"
        assert data["enabled"] is True
    
    def test_data_export_endpoint(self):
        """Test the data export endpoint"""
        response = self.client.post("/personal-assistant/privacy/export")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "attachment" in response.headers.get("content-disposition", "")
        
        # Verify it's valid JSON
        data = response.json()
        assert "user_id" in data
        assert "export_date" in data
        assert "data" in data
    
    def test_data_deletion_endpoint(self):
        """Test the data deletion endpoint"""
        response = self.client.delete("/personal-assistant/privacy/delete-all")
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "request_id" in data
        assert "message" in data
        assert data["success"] is True
    
    def test_invalid_permission_type(self):
        """Test handling of invalid permission types"""
        permission_data = {
            "permission": "invalid_permission",
            "enabled": True
        }
        
        response = self.client.post("/personal-assistant/privacy/permissions", json=permission_data)
        
        # Should still return success due to mock fallback
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
    
    def test_web_interface_accessibility(self):
        """Test that the web interface includes accessibility features"""
        response = self.client.get("/")
        assert response.status_code == 200
        
        # Check for basic accessibility features in HTML
        html_content = response.text
        assert 'alt=' in html_content or 'aria-' in html_content or 'role=' in html_content
        assert 'lang=' in html_content
        assert '<title>' in html_content
    
    def test_responsive_design_elements(self):
        """Test that the web interface includes responsive design elements"""
        response = self.client.get("/")
        assert response.status_code == 200
        
        html_content = response.text
        assert 'viewport' in html_content
        assert 'grid' in html_content or 'flex' in html_content
        assert '@media' in html_content or 'responsive' in html_content.lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])