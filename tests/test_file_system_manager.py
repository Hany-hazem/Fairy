"""
Unit tests for FileSystemManager

Tests the secure file access functionality, permission checks, and integration
with existing components.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import the modules we're testing
from app.file_system_manager import (
    FileSystemManager, FileOperation, AccessLevel, FileAccessRequest, 
    FileAccessResult, FileSystemStats
)
from app.privacy_security_manager import PrivacySecurityManager, PermissionType
from app.user_context_manager import UserContextManager
from app.file_content_analyzer import FileContentAnalyzer
from app.file_organization_manager import FileOrganizationManager
from app.personal_assistant_models import UserContext, InteractionType


class TestFileSystemManager:
    """Test cases for FileSystemManager"""
    
    @pytest_asyncio.fixture
    async def setup_managers(self):
        """Set up test managers with mocks"""
        # Create mock managers
        privacy_manager = Mock(spec=PrivacySecurityManager)
        context_manager = Mock(spec=UserContextManager)
        content_analyzer = Mock(spec=FileContentAnalyzer)
        organization_manager = Mock(spec=FileOrganizationManager)
        
        # Configure mock behaviors
        privacy_manager.check_permission = AsyncMock(return_value=True)
        privacy_manager.request_permission = AsyncMock(return_value=True)
        
        context_manager.get_user_context = AsyncMock(return_value=UserContext(user_id="test_user"))
        context_manager.update_user_context = AsyncMock()
        context_manager.add_interaction = AsyncMock()
        
        content_analyzer.analyze_file_content = AsyncMock()
        
        organization_manager.search_files = AsyncMock(return_value=[])
        organization_manager.create_organization_plan = AsyncMock()
        organization_manager.execute_organization_plan = AsyncMock()
        
        # Create FileSystemManager
        fs_manager = FileSystemManager(
            privacy_manager=privacy_manager,
            context_manager=context_manager,
            content_analyzer=content_analyzer,
            organization_manager=organization_manager
        )
        
        return {
            'fs_manager': fs_manager,
            'privacy_manager': privacy_manager,
            'context_manager': context_manager,
            'content_analyzer': content_analyzer,
            'organization_manager': organization_manager
        }
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Test file content\nLine 2\nLine 3")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_files = [
                'test1.txt', 'test2.py', 'test3.json', 'subdir/test4.md'
            ]
            
            for file_path in test_files:
                full_path = Path(temp_dir) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(f"Content of {file_path}")
            
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_read_file_success(self, setup_managers, temp_file):
        """Test successful file reading"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        result = await fs_manager.read_file("test_user", temp_file, "Testing file read")
        
        assert result.success is True
        assert result.operation == FileOperation.READ
        assert result.file_path == temp_file
        assert "Test file content" in result.content
        assert result.error_message is None
        
        # Verify permission was checked
        managers['privacy_manager'].check_permission.assert_called_with("test_user", PermissionType.FILE_READ)
        
        # Verify interaction was logged
        managers['context_manager'].add_interaction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_read_file_permission_denied(self, setup_managers, temp_file):
        """Test file reading with permission denied"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Mock permission denied
        managers['privacy_manager'].check_permission.return_value = False
        managers['privacy_manager'].request_permission.return_value = False
        
        result = await fs_manager.read_file("test_user", temp_file)
        
        assert result.success is False
        assert result.permission_required is True
        assert "permission denied" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_read_file_not_found(self, setup_managers):
        """Test reading non-existent file"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        result = await fs_manager.read_file("test_user", "/nonexistent/file.txt")
        
        assert result.success is False
        assert "not found" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_read_file_restricted_path(self, setup_managers):
        """Test reading from restricted path"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Try to read from a restricted system directory
        result = await fs_manager.read_file("test_user", "/etc/passwd")
        
        assert result.success is False
        assert "restricted" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_write_file_success(self, setup_managers):
        """Test successful file writing"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_write.txt")
            test_content = "This is test content for writing"
            
            result = await fs_manager.write_file("test_user", test_file, test_content)
            
            assert result.success is True
            assert result.operation == FileOperation.WRITE
            assert result.file_path == test_file
            
            # Verify file was actually written
            assert os.path.exists(test_file)
            with open(test_file, 'r') as f:
                assert f.read() == test_content
            
            # Verify permission was checked
            managers['privacy_manager'].check_permission.assert_called_with("test_user", PermissionType.FILE_WRITE)
    
    @pytest.mark.asyncio
    async def test_write_file_with_backup(self, setup_managers, temp_file):
        """Test file writing with backup creation"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        original_content = Path(temp_file).read_text()
        new_content = "New content replacing old content"
        
        result = await fs_manager.write_file("test_user", temp_file, new_content, backup=True)
        
        assert result.success is True
        
        # Verify new content was written
        assert Path(temp_file).read_text() == new_content
        
        # Verify backup was created
        backup_file = temp_file + ".backup"
        assert os.path.exists(backup_file)
        assert Path(backup_file).read_text() == original_content
        
        # Cleanup backup
        os.unlink(backup_file)
    
    @pytest.mark.asyncio
    async def test_write_file_create_directories(self, setup_managers):
        """Test file writing with directory creation"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_file = os.path.join(temp_dir, "new", "nested", "dir", "test.txt")
            test_content = "Content in nested directory"
            
            result = await fs_manager.write_file("test_user", nested_file, test_content, create_dirs=True)
            
            assert result.success is True
            assert os.path.exists(nested_file)
            assert Path(nested_file).read_text() == test_content
    
    @pytest.mark.asyncio
    async def test_list_directory_success(self, setup_managers, temp_directory):
        """Test successful directory listing"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        result = await fs_manager.list_directory("test_user", temp_directory)
        
        assert result.success is True
        assert result.operation == FileOperation.LIST
        
        # Parse the metadata
        listing = result.metadata
        assert 'directories' in listing
        assert 'files' in listing
        assert listing['total_files'] > 0
        
        # Check that some expected files are present
        file_names = [f['name'] for f in listing['files']]
        assert 'test1.txt' in file_names
        assert 'test2.py' in file_names
    
    @pytest.mark.asyncio
    async def test_list_directory_recursive(self, setup_managers, temp_directory):
        """Test recursive directory listing"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        result = await fs_manager.list_directory("test_user", temp_directory, recursive=True)
        
        assert result.success is True
        
        # Should find files in subdirectories too
        listing = result.metadata
        file_paths = [f['path'] for f in listing['files']]
        
        # Should include the file in the subdirectory
        subdir_file_found = any('test4.md' in path for path in file_paths)
        assert subdir_file_found
    
    @pytest.mark.asyncio
    async def test_analyze_file_content(self, setup_managers, temp_file):
        """Test file content analysis"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Mock the content analyzer
        from app.file_content_analyzer import ContentAnalysis, ContentType
        mock_analysis = ContentAnalysis(
            file_path=temp_file,
            content_type=ContentType.TEXT,
            extracted_text="Test file content",
            original_content="Test file content",
            file_size=100,
            word_count=3
        )
        managers['content_analyzer'].analyze_file_content.return_value = mock_analysis
        
        result = await fs_manager.analyze_file_content("test_user", temp_file)
        
        assert result.success is True
        assert result.operation == FileOperation.ANALYZE
        assert result.analysis is not None
        assert result.analysis.word_count == 3
        
        # Verify content analyzer was called
        managers['content_analyzer'].analyze_file_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_files(self, setup_managers):
        """Test file searching functionality"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Mock search results
        from app.file_organization_manager import FileMetadata
        mock_results = [
            FileMetadata(
                path="/test/file1.txt",
                name="file1.txt",
                size=100,
                created_at=datetime.now(),
                modified_at=datetime.now(),
                accessed_at=datetime.now(),
                file_type=".txt",
                mime_type="text/plain",
                category="documents"
            )
        ]
        managers['organization_manager'].search_files.return_value = mock_results
        
        result = await fs_manager.search_files("test_user", "test query", "/test/dir")
        
        assert result.success is True
        assert result.operation == FileOperation.SEARCH
        assert 'results' in result.metadata
        assert len(result.metadata['results']) == 1
        
        # Verify organization manager was called
        managers['organization_manager'].search_files.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_organize_files_dry_run(self, setup_managers):
        """Test file organization in dry run mode"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Mock organization plan
        from app.file_organization_manager import OrganizationPlan, OrganizationStrategy
        mock_plan = OrganizationPlan(
            strategy=OrganizationStrategy.BY_TYPE,
            moves=[{'source': '/test/file1.txt', 'destination': '/test/documents/file1.txt'}],
            creates=['/test/documents'],
            estimated_time=1.0,
            benefits=['Files grouped by type'],
            warnings=[]
        )
        managers['organization_manager'].create_organization_plan.return_value = mock_plan
        
        result = await fs_manager.organize_files("test_user", "/test/dir", "by_type", dry_run=True)
        
        assert result.success is True
        assert result.operation == FileOperation.ORGANIZE
        assert 'plan' in result.metadata
        assert result.metadata['plan']['moves'] == 1
        
        # Verify plan was created but not executed
        managers['organization_manager'].create_organization_plan.assert_called_once()
        managers['organization_manager'].execute_organization_plan.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_file_stats(self, setup_managers, temp_file):
        """Test file statistics retrieval"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Perform some operations to generate stats
        await fs_manager.read_file("test_user", temp_file)
        await fs_manager.read_file("test_user", temp_file)
        
        stats = await fs_manager.get_file_stats("test_user")
        
        assert stats['total_files_accessed'] == 2
        assert stats['total_bytes_read'] > 0
        assert FileOperation.READ.value in stats['operations_by_type']
        assert stats['operations_by_type'][FileOperation.READ.value] == 2
    
    @pytest.mark.asyncio
    async def test_path_traversal_protection(self, setup_managers):
        """Test protection against path traversal attacks"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Try various path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/test/../../../etc/passwd",
            "test/../../sensitive/file.txt"
        ]
        
        for path in malicious_paths:
            result = await fs_manager.read_file("test_user", path)
            assert result.success is False
            assert "traversal" in result.error_message.lower() or "restricted" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_file_size_limit(self, setup_managers):
        """Test file size limits"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Create a large temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            # Write content that exceeds the limit (simulate large file)
            large_content = "x" * (fs_manager._max_file_size + 1000)
            f.write(large_content)
            large_file_path = f.name
        
        try:
            result = await fs_manager.read_file("test_user", large_file_path)
            assert result.success is False
            assert "too large" in result.error_message.lower()
        finally:
            os.unlink(large_file_path)
    
    @pytest.mark.asyncio
    async def test_unsafe_file_extension_blocked(self, setup_managers):
        """Test that unsafe file extensions are blocked"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Create a file with unsafe extension
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.exe') as f:
            f.write("fake executable content")
            unsafe_file = f.name
        
        try:
            result = await fs_manager.read_file("test_user", unsafe_file)
            assert result.success is False
            assert "not allowed" in result.error_message.lower()
        finally:
            os.unlink(unsafe_file)
    
    @pytest.mark.asyncio
    async def test_context_update_on_file_access(self, setup_managers, temp_file):
        """Test that user context is updated on file access"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Mock user context
        mock_context = UserContext(user_id="test_user")
        managers['context_manager'].get_user_context.return_value = mock_context
        
        await fs_manager.read_file("test_user", temp_file)
        
        # Verify context was retrieved and updated
        managers['context_manager'].get_user_context.assert_called_with("test_user")
        managers['context_manager'].update_user_context.assert_called_once()
        
        # Verify interaction was added
        managers['context_manager'].add_interaction.assert_called_once()
        call_args = managers['context_manager'].add_interaction.call_args
        interaction = call_args[0][1]  # Second argument is the interaction
        assert interaction.interaction_type == InteractionType.FILE_ACCESS
        assert temp_file in interaction.content
    
    @pytest.mark.asyncio
    async def test_permission_request_flow(self, setup_managers, temp_file):
        """Test the permission request flow"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Mock no initial permission, but grant when requested
        managers['privacy_manager'].check_permission.return_value = False
        managers['privacy_manager'].request_permission.return_value = True
        
        result = await fs_manager.read_file("test_user", temp_file)
        
        assert result.success is True
        
        # Verify permission was checked and requested
        managers['privacy_manager'].check_permission.assert_called_with("test_user", PermissionType.FILE_READ)
        managers['privacy_manager'].request_permission.assert_called_once()
        
        # Check the request_permission call arguments
        call_args = managers['privacy_manager'].request_permission.call_args
        assert call_args[0][0] == "test_user"  # user_id
        assert call_args[0][1] == PermissionType.FILE_READ  # permission_type
        assert 'scope' in call_args[1]  # keyword arguments
    
    @pytest.mark.asyncio
    async def test_error_handling_and_logging(self, setup_managers):
        """Test error handling and logging"""
        managers = setup_managers
        fs_manager = managers['fs_manager']
        
        # Mock an exception in the context manager
        managers['context_manager'].add_interaction.side_effect = Exception("Database error")
        
        # This should still succeed despite the logging error
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_file = f.name
        
        try:
            result = await fs_manager.read_file("test_user", temp_file)
            # The file read should still succeed even if logging fails
            assert result.success is True
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])