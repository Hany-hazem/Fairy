"""
Tests for File System Manager

This module tests the secure file system manager with permission controls,
content analysis, and file organization capabilities.
"""

import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.file_system_manager import (
    FileSystemManager, FileSystemPermissionManager, FileAccessType, 
    FilePermissionScope, FileContent, FileSearchResult
)
from app.file_content_analyzer import FileContentAnalyzer, ContentAnalysis, ContentType
from app.file_organization_manager import (
    FileOrganizationManager, OrganizationStrategy, DuplicateDetectionMethod
)
from app.privacy_security_manager import PrivacySecurityManager
from app.personal_assistant_models import PermissionType


class TestFileSystemPermissionManager:
    """Test file system permission management"""
    
    @pytest.fixture
    def privacy_manager(self):
        """Mock privacy manager"""
        mock = Mock(spec=PrivacySecurityManager)
        mock.get_user_permissions = AsyncMock(return_value=[])
        return mock
    
    @pytest.fixture
    def permission_manager(self, privacy_manager):
        """File system permission manager instance"""
        return FileSystemPermissionManager(privacy_manager)
    
    @pytest.mark.asyncio
    async def test_request_file_access(self, permission_manager):
        """Test requesting file access permission"""
        request_id = await permission_manager.request_file_access(
            "user123", "/test/file.txt", FileAccessType.READ, "Test read access"
        )
        
        assert request_id is not None
        assert request_id in permission_manager.pending_requests
        
        request = permission_manager.pending_requests[request_id]
        assert request.user_id == "user123"
        assert request.path == "/test/file.txt"
        assert request.access_type == FileAccessType.READ
        assert request.reason == "Test read access"
    
    @pytest.mark.asyncio
    async def test_approve_file_access(self, permission_manager):
        """Test approving file access permission"""
        # Request access first
        request_id = await permission_manager.request_file_access(
            "user123", "/test/file.txt", FileAccessType.READ, "Test read access"
        )
        
        # Approve the request
        success = await permission_manager.approve_file_access(request_id, expires_in_hours=24)
        
        assert success is True
        assert "user123" in permission_manager.permissions
        assert len(permission_manager.permissions["user123"]) == 1
        
        permission = permission_manager.permissions["user123"][0]
        assert permission.path == "/test/file.txt"
        assert FileAccessType.READ in permission.access_types
    
    @pytest.mark.asyncio
    async def test_check_file_permission(self, permission_manager):
        """Test checking file permissions"""
        # Request and approve access
        request_id = await permission_manager.request_file_access(
            "user123", "/test/file.txt", FileAccessType.READ, "Test read access"
        )
        await permission_manager.approve_file_access(request_id)
        
        # Check permission
        has_permission = await permission_manager.check_file_permission(
            "user123", "/test/file.txt", FileAccessType.READ
        )
        
        assert has_permission is True
        
        # Check non-existent permission
        has_permission = await permission_manager.check_file_permission(
            "user123", "/test/file.txt", FileAccessType.WRITE
        )
        
        assert has_permission is False
    
    @pytest.mark.asyncio
    async def test_revoke_file_access(self, permission_manager):
        """Test revoking file access permission"""
        # Request and approve access
        request_id = await permission_manager.request_file_access(
            "user123", "/test/file.txt", FileAccessType.READ, "Test read access"
        )
        await permission_manager.approve_file_access(request_id)
        
        # Revoke access
        success = await permission_manager.revoke_file_access(
            "user123", "/test/file.txt", FileAccessType.READ
        )
        
        assert success is True
        
        # Check permission is revoked
        has_permission = await permission_manager.check_file_permission(
            "user123", "/test/file.txt", FileAccessType.READ
        )
        
        assert has_permission is False


class TestFileSystemManager:
    """Test file system manager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def privacy_manager(self):
        """Mock privacy manager"""
        mock = Mock(spec=PrivacySecurityManager)
        mock.get_user_permissions = AsyncMock(return_value=[])
        return mock
    
    @pytest.fixture
    def file_manager(self, privacy_manager):
        """File system manager instance"""
        return FileSystemManager(privacy_manager)
    
    @pytest.mark.asyncio
    async def test_read_file(self, file_manager, temp_dir):
        """Test reading file content"""
        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_content = "Hello, World!\nThis is a test file."
        test_file.write_text(test_content)
        
        # Read file
        file_content = await file_manager.read_file("user123", str(test_file))
        
        assert file_content is not None
        assert file_content.content == test_content
        assert file_content.file_type == ".txt"
        assert file_content.size == len(test_content.encode())
    
    @pytest.mark.asyncio
    async def test_write_file(self, file_manager, temp_dir):
        """Test writing file content"""
        test_file = Path(temp_dir) / "new_file.txt"
        test_content = "This is new content."
        
        # Mock permission approval for write operations
        with patch.object(file_manager.permission_manager, 'check_file_permission', return_value=True):
            success = await file_manager.write_file("user123", str(test_file), test_content)
        
        assert success is True
        assert test_file.exists()
        assert test_file.read_text() == test_content
    
    @pytest.mark.asyncio
    async def test_list_directory(self, file_manager, temp_dir):
        """Test listing directory contents"""
        # Create test files
        (Path(temp_dir) / "file1.txt").write_text("Content 1")
        (Path(temp_dir) / "file2.py").write_text("print('Hello')")
        (Path(temp_dir) / "subdir").mkdir()
        
        # List directory
        files = await file_manager.list_directory("user123", temp_dir)
        
        assert len(files) == 3
        file_names = [f['name'] for f in files]
        assert "file1.txt" in file_names
        assert "file2.py" in file_names
        assert "subdir" in file_names
    
    @pytest.mark.asyncio
    async def test_analyze_file_content(self, file_manager, temp_dir):
        """Test file content analysis"""
        # Create test file with content
        test_file = Path(temp_dir) / "test.txt"
        test_content = "This is a test document with email@example.com and phone 123-456-7890."
        test_file.write_text(test_content)
        
        # Analyze content
        analysis = await file_manager.analyze_file_content("user123", str(test_file))
        
        assert analysis is not None
        assert analysis.extracted_text == test_content
        assert analysis.content_type == ContentType.TEXT
        assert len(analysis.entities) > 0  # Should find email and phone entities
    
    @pytest.mark.asyncio
    async def test_search_file_content(self, file_manager, temp_dir):
        """Test searching file content"""
        # Create test files
        (Path(temp_dir) / "file1.txt").write_text("This contains the search term.")
        (Path(temp_dir) / "file2.txt").write_text("This does not contain it.")
        (Path(temp_dir) / "file3.py").write_text("# Python code with search term")
        
        # Search for content
        results = await file_manager.search_file_content("user123", temp_dir, "search term")
        
        assert len(results) == 2  # Should find file1.txt and file3.py
        paths = [r.path for r in results]
        assert str(Path(temp_dir) / "file1.txt") in paths
        assert str(Path(temp_dir) / "file3.py") in paths


class TestFileContentAnalyzer:
    """Test file content analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """File content analyzer instance"""
        return FileContentAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_text_content(self, analyzer):
        """Test analyzing text content"""
        content = "This is a test document with email@example.com and phone 123-456-7890."
        
        analysis = await analyzer.analyze_file_content("test.txt", content)
        
        assert analysis.content_type == ContentType.TEXT
        assert analysis.extracted_text == content
        assert analysis.word_count > 0
        assert analysis.char_count == len(content)
        assert len(analysis.entities) > 0  # Should find email and phone
    
    @pytest.mark.asyncio
    async def test_extract_entities(self, analyzer):
        """Test entity extraction"""
        text = "Contact John at john@example.com or call 555-123-4567. Visit https://example.com"
        
        entities = await analyzer._extract_entities(text)
        
        entity_types = [e.entity_type for e in entities]
        assert 'email' in entity_types
        assert 'phone' in entity_types
        assert 'url' in entity_types
    
    def test_determine_content_type(self, analyzer):
        """Test content type determination"""
        assert analyzer._determine_content_type(Path("test.txt")) == ContentType.TEXT
        assert analyzer._determine_content_type(Path("test.pdf")) == ContentType.PDF
        assert analyzer._determine_content_type(Path("test.jpg")) == ContentType.IMAGE
        assert analyzer._determine_content_type(Path("test.py")) == ContentType.CODE
        assert analyzer._determine_content_type(Path("test.unknown")) == ContentType.UNKNOWN


class TestFileOrganizationManager:
    """Test file organization manager"""
    
    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def content_analyzer(self):
        """Mock content analyzer"""
        return Mock(spec=FileContentAnalyzer)
    
    @pytest.fixture
    def org_manager(self, content_analyzer):
        """File organization manager instance"""
        return FileOrganizationManager(content_analyzer)
    
    @pytest.mark.asyncio
    async def test_index_directory(self, org_manager, temp_dir):
        """Test directory indexing"""
        # Create test files
        (Path(temp_dir) / "file1.txt").write_text("Content 1")
        (Path(temp_dir) / "file2.py").write_text("print('Hello')")
        (Path(temp_dir) / "image.jpg").write_bytes(b"fake image data")
        
        # Index directory
        count = await org_manager.index_directory(temp_dir)
        
        assert count == 3
        assert len(org_manager.file_index.files) == 3
    
    @pytest.mark.asyncio
    async def test_find_duplicates_by_hash(self, org_manager, temp_dir):
        """Test finding duplicates by hash"""
        # Create duplicate files
        content = "This is duplicate content"
        (Path(temp_dir) / "file1.txt").write_text(content)
        (Path(temp_dir) / "file2.txt").write_text(content)
        (Path(temp_dir) / "file3.txt").write_text("Different content")
        
        # Find duplicates
        duplicates = await org_manager.find_duplicates(temp_dir, DuplicateDetectionMethod.HASH)
        
        assert len(duplicates) == 1  # One group of duplicates
        assert len(duplicates[0].files) == 2  # Two duplicate files
    
    @pytest.mark.asyncio
    async def test_create_organization_plan_by_type(self, org_manager, temp_dir):
        """Test creating organization plan by type"""
        # Create files of different types
        (Path(temp_dir) / "document.txt").write_text("Text content")
        (Path(temp_dir) / "script.py").write_text("print('Hello')")
        (Path(temp_dir) / "image.jpg").write_bytes(b"fake image")
        
        # Create organization plan
        plan = await org_manager.create_organization_plan(temp_dir, OrganizationStrategy.BY_TYPE)
        
        assert plan.strategy == OrganizationStrategy.BY_TYPE
        assert len(plan.moves) == 3  # Should move all files
        assert len(plan.creates) > 0  # Should create category directories
    
    @pytest.mark.asyncio
    async def test_search_files(self, org_manager, temp_dir):
        """Test file searching"""
        # Create and index test files
        (Path(temp_dir) / "important.txt").write_text("Important document")
        (Path(temp_dir) / "script.py").write_text("Python script")
        
        await org_manager.index_directory(temp_dir)
        
        # Search files
        results = await org_manager.search_files("important")
        
        assert len(results) == 1
        assert "important.txt" in results[0].name


@pytest.mark.asyncio
async def test_integration_file_system_operations():
    """Integration test for file system operations"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup
        privacy_manager = Mock(spec=PrivacySecurityManager)
        privacy_manager.get_user_permissions = AsyncMock(return_value=[])
        
        file_manager = FileSystemManager(privacy_manager)
        
        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_content = "This is a test file with email@test.com"
        test_file.write_text(test_content)
        
        # Test read operation
        file_content = await file_manager.read_file("user123", str(test_file))
        assert file_content is not None
        assert file_content.content == test_content
        
        # Test content analysis
        analysis = await file_manager.analyze_file_content("user123", str(test_file))
        assert analysis is not None
        assert len(analysis.entities) > 0  # Should find email entity
        
        # Test indexing
        count = await file_manager.index_files("user123", temp_dir)
        assert count == 1
        
        # Test search
        results = await file_manager.search_indexed_files("user123", "test")
        assert len(results) == 1
        assert results[0]['name'] == "test.txt"


if __name__ == "__main__":
    pytest.main([__file__])