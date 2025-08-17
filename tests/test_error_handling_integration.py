# tests/test_error_handling_integration.py
"""
Integration tests for MCP and Git error handling systems
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.mcp_error_handler import (
    MCPErrorHandler, ErrorSeverity, ErrorCategory, RecoveryAction,
    BackoffConfig, CircuitBreakerConfig, CircuitBreaker
)
from app.git_error_handler import (
    GitErrorHandler, GitErrorType, ConflictResolutionStrategy,
    RecoveryAction as GitRecoveryAction
)
from app.git_workflow_manager import GitWorkflowManager
from app.version_control import GitIntegration


class TestMCPErrorHandler:
    """Test MCP error handling functionality"""
    
    @pytest.fixture
    async def error_handler(self):
        """Create MCP error handler for testing"""
        handler = MCPErrorHandler()
        await handler.start()
        yield handler
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_handle_connection_error(self, error_handler):
        """Test handling of connection errors with exponential backoff"""
        # Simulate connection error
        connection_error = ConnectionError("Connection refused")
        
        # Handle the error
        error_info = await error_handler.handle_error(
            connection_error,
            context={"operation": "redis_connection"},
            operation="redis_connection"
        )
        
        # Verify error was handled
        assert error_info.error_type == "ConnectionError"
        assert error_info.category == ErrorCategory.CONNECTION
        assert error_info.severity == ErrorSeverity.HIGH
        assert RecoveryAction.RETRY in error_info.recovery_actions
        assert RecoveryAction.RECONNECT in error_info.recovery_actions
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self, error_handler):
        """Test exponential backoff functionality"""
        config = BackoffConfig(
            initial_delay=1.0,
            max_delay=10.0,
            multiplier=2.0,
            max_retries=3
        )
        
        error_handler.configure_backoff("test_operation", config)
        
        # Test connection error handling with backoff
        connection_error = ConnectionError("Network timeout")
        
        start_time = datetime.now()
        success = await error_handler.handle_connection_error(connection_error, "test")
        end_time = datetime.now()
        
        # Should have taken some time due to backoff
        duration = (end_time - start_time).total_seconds()
        assert duration >= 1.0  # At least initial delay
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, error_handler):
        """Test circuit breaker functionality"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1,
            half_open_max_calls=1,
            success_threshold=1
        )
        
        circuit_breaker = CircuitBreaker(config)
        error_handler._circuit_breakers["test_service"] = circuit_breaker
        
        # Initially should be closed (allow operations)
        assert circuit_breaker.can_execute()
        
        # Record failures to trip the breaker
        circuit_breaker.record_failure()
        circuit_breaker.record_failure()
        
        # Should now be open (block operations)
        assert not circuit_breaker.can_execute()
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should now be half-open (allow limited operations)
        assert circuit_breaker.can_execute()
    
    @pytest.mark.asyncio
    async def test_redis_unavailable_fallback(self, error_handler):
        """Test graceful degradation when Redis is unavailable"""
        # Test message publishing fallback
        result = await error_handler.handle_redis_unavailable(
            "publish_message",
            {"message": "test"}
        )
        
        # Should return fallback result
        assert result is not None
        assert error_handler._degradation_mode
        assert "redis" in error_handler._degraded_services
    
    @pytest.mark.asyncio
    async def test_error_reporting(self, error_handler):
        """Test comprehensive error reporting"""
        # Generate some test errors
        error1 = Exception("Test error 1")
        error2 = ConnectionError("Test connection error")
        
        await error_handler.handle_error(error1, {"test": "context1"})
        await error_handler.handle_error(error2, {"test": "context2"})
        
        # Get error report
        report = await error_handler.get_error_report()
        
        assert "statistics" in report
        assert "errors" in report
        assert "circuit_breakers" in report
        assert len(report["errors"]) == 2
        assert report["statistics"]["total_errors"] == 2


class TestGitErrorHandler:
    """Test Git error handling functionality"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create temporary Git repository for testing"""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)
        
        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        
        # Create initial commit
        test_file = repo_path / "test.txt"
        test_file.write_text("Initial content")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)
        
        yield repo_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def git_error_handler(self, temp_repo):
        """Create Git error handler for testing"""
        git_manager = GitWorkflowManager(str(temp_repo))
        git_integration = GitIntegration(str(temp_repo))
        handler = GitErrorHandler(git_manager, git_integration)
        return handler
    
    @pytest.mark.asyncio
    async def test_classify_git_errors(self, git_error_handler):
        """Test Git error classification"""
        # Test merge conflict error
        merge_error = await git_error_handler.handle_git_error(
            "git merge feature-branch",
            1,
            "",
            "CONFLICT (content): Merge conflict in file.txt\nAutomatic merge failed"
        )
        assert merge_error.error_type == GitErrorType.MERGE_CONFLICT
        
        # Test authentication error
        auth_error = await git_error_handler.handle_git_error(
            "git push origin main",
            128,
            "",
            "fatal: Authentication failed for 'https://github.com/user/repo.git/'"
        )
        assert auth_error.error_type == GitErrorType.AUTHENTICATION
        
        # Test network error
        network_error = await git_error_handler.handle_git_error(
            "git clone https://github.com/user/repo.git",
            128,
            "",
            "fatal: unable to access 'https://github.com/user/repo.git/': Could not resolve host"
        )
        assert network_error.error_type == GitErrorType.NETWORK
    
    @pytest.mark.asyncio
    async def test_merge_conflict_resolution(self, git_error_handler, temp_repo):
        """Test merge conflict resolution"""
        # Create a file with conflict markers
        conflict_file = temp_repo / "conflict.txt"
        conflict_content = """Line 1
<<<<<<< HEAD
Our changes
=======
Their changes
>>>>>>> feature-branch
Line 3"""
        conflict_file.write_text(conflict_content)
        
        # Test conflict resolution
        success = await git_error_handler.handle_merge_conflict(
            ["conflict.txt"],
            ConflictResolutionStrategy.OURS
        )
        
        # Should resolve successfully
        assert success
        
        # Check that conflict markers are removed
        resolved_content = conflict_file.read_text()
        assert "<<<<<<< HEAD" not in resolved_content
        assert "=======" not in resolved_content
        assert ">>>>>>> feature-branch" not in resolved_content
    
    @pytest.mark.asyncio
    async def test_repository_corruption_detection(self, git_error_handler):
        """Test repository corruption detection"""
        corruption_info = await git_error_handler.detect_repository_corruption()
        
        assert "corrupted" in corruption_info
        assert "issues" in corruption_info
        assert "severity" in corruption_info
        assert "recommended_actions" in corruption_info
        
        # For a healthy repo, should not be corrupted
        assert not corruption_info["corrupted"]
        assert corruption_info["severity"] == "none"
    
    @pytest.mark.asyncio
    async def test_rollback_operation(self, git_error_handler, temp_repo):
        """Test Git operation rollback"""
        # Create a commit to rollback to
        test_file = temp_repo / "test.txt"
        original_content = test_file.read_text()
        
        # Make changes and commit
        test_file.write_text("Modified content")
        import subprocess
        subprocess.run(["git", "add", "test.txt"], cwd=temp_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Test commit"], cwd=temp_repo, check=True)
        
        # Get commit hash to rollback to
        result = subprocess.run(
            ["git", "rev-parse", "HEAD~1"],
            cwd=temp_repo,
            capture_output=True,
            text=True,
            check=True
        )
        rollback_commit = result.stdout.strip()
        
        # Test rollback
        success = await git_error_handler.rollback_operation("test_op", rollback_commit)
        assert success
        
        # Verify content was rolled back
        assert test_file.read_text() == original_content
    
    @pytest.mark.asyncio
    async def test_repository_backup_and_restore(self, git_error_handler):
        """Test repository backup and restore functionality"""
        # Create backup
        backup_id = await git_error_handler._create_repository_backup("test_backup")
        assert backup_id is not None
        assert backup_id in git_error_handler._backups
        
        # Verify backup info
        backup_info = git_error_handler.get_backup_info()
        assert len(backup_info) == 1
        assert backup_info[0]["backup_id"] == backup_id
        assert backup_info[0]["description"] == "test_backup"
    
    @pytest.mark.asyncio
    async def test_recovery_actions(self, git_error_handler):
        """Test various recovery actions"""
        # Test retry operation
        git_error = await git_error_handler.handle_git_error(
            "git status",
            0,
            "On branch main",
            ""
        )
        
        # Should have recovery actions
        assert len(git_error.recovery_actions) > 0
        
        # Test specific recovery action execution
        success = await git_error_handler._execute_single_recovery_action(
            git_error,
            GitRecoveryAction.CLEAN_WORKSPACE
        )
        # Should succeed for clean workspace operation
        assert success
    
    def test_error_statistics(self, git_error_handler):
        """Test error handler statistics"""
        stats = git_error_handler.get_handler_stats()
        
        assert "total_errors" in stats
        assert "resolved_errors" in stats
        assert "merge_conflicts_resolved" in stats
        assert "automatic_recoveries" in stats
        assert "manual_interventions" in stats
        assert "repository_corruptions" in stats
        assert "backups_created" in stats


class TestErrorHandlerIntegration:
    """Test integration between MCP and Git error handlers"""
    
    @pytest.fixture
    async def integrated_handlers(self, temp_repo):
        """Create integrated error handlers"""
        # Create MCP error handler
        mcp_handler = MCPErrorHandler()
        await mcp_handler.start()
        
        # Create Git error handler
        git_manager = GitWorkflowManager(str(temp_repo))
        git_handler = GitErrorHandler(git_manager)
        
        yield mcp_handler, git_handler
        
        await mcp_handler.stop()
    
    @pytest.mark.asyncio
    async def test_cascading_error_handling(self, integrated_handlers):
        """Test handling of cascading errors between MCP and Git"""
        mcp_handler, git_handler = integrated_handlers
        
        # Simulate a Git error that affects MCP operations
        git_error = await git_handler.handle_git_error(
            "git push origin main",
            128,
            "",
            "fatal: repository not found"
        )
        
        # This should trigger MCP degradation
        await mcp_handler.handle_redis_unavailable("git_sync_operation")
        
        # Verify both handlers are in appropriate states
        assert git_error.error_type == GitErrorType.AUTHENTICATION
        assert mcp_handler._degradation_mode
    
    @pytest.mark.asyncio
    async def test_coordinated_recovery(self, integrated_handlers):
        """Test coordinated recovery between error handlers"""
        mcp_handler, git_handler = integrated_handlers
        
        # Simulate coordinated recovery scenario
        # 1. Git operation fails
        git_error = await git_handler.handle_git_error(
            "git commit -m 'test'",
            1,
            "",
            "nothing to commit, working tree clean"
        )
        
        # 2. MCP should handle the failure gracefully
        mcp_error = await mcp_handler.handle_error(
            Exception("Git commit failed"),
            context={"git_error_id": git_error.error_id}
        )
        
        # Verify coordinated handling
        assert git_error.error_type == GitErrorType.COMMIT_FAILED
        assert mcp_error.category == ErrorCategory.SYSTEM
    
    @pytest.mark.asyncio
    async def test_error_correlation(self, integrated_handlers):
        """Test error correlation between systems"""
        mcp_handler, git_handler = integrated_handlers
        
        # Create correlated errors
        correlation_id = "test_correlation_123"
        
        git_error = await git_handler.handle_git_error(
            "git merge feature",
            1,
            "",
            "CONFLICT: merge conflict in file.txt",
            context={"correlation_id": correlation_id}
        )
        
        mcp_error = await mcp_handler.handle_error(
            Exception("Task completion failed due to Git conflict"),
            context={"correlation_id": correlation_id, "git_error_id": git_error.error_id}
        )
        
        # Verify correlation
        assert git_error.context.get("correlation_id") == correlation_id
        assert mcp_error.context.get("correlation_id") == correlation_id
        assert mcp_error.context.get("git_error_id") == git_error.error_id


@pytest.mark.asyncio
async def test_end_to_end_error_recovery():
    """Test end-to-end error recovery scenario"""
    # Create temporary repository
    temp_dir = tempfile.mkdtemp()
    repo_path = Path(temp_dir)
    
    try:
        # Initialize repository
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        
        # Create initial commit
        test_file = repo_path / "test.txt"
        test_file.write_text("Initial content")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)
        
        # Create error handlers
        mcp_handler = MCPErrorHandler()
        await mcp_handler.start()
        
        git_manager = GitWorkflowManager(str(repo_path))
        git_handler = GitErrorHandler(git_manager)
        
        # Simulate complex error scenario
        # 1. Create merge conflict
        subprocess.run(["git", "checkout", "-b", "feature"], cwd=repo_path, check=True)
        test_file.write_text("Feature content")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Feature commit"], cwd=repo_path, check=True)
        
        subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True)
        test_file.write_text("Main content")
        subprocess.run(["git", "add", "test.txt"], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Main commit"], cwd=repo_path, check=True)
        
        # 2. Attempt merge (will fail)
        result = subprocess.run(
            ["git", "merge", "feature"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        # 3. Handle the Git error
        git_error = await git_handler.handle_git_error(
            "git merge feature",
            result.returncode,
            result.stdout,
            result.stderr
        )
        
        # 4. Handle related MCP error
        mcp_error = await mcp_handler.handle_error(
            Exception("Task merge failed"),
            context={"git_error_id": git_error.error_id}
        )
        
        # 5. Verify recovery
        assert git_error.error_type == GitErrorType.MERGE_CONFLICT
        assert mcp_error.category == ErrorCategory.SYSTEM
        
        # 6. Test conflict resolution
        conflicted_files = ["test.txt"]
        resolution_success = await git_handler.handle_merge_conflict(
            conflicted_files,
            ConflictResolutionStrategy.OURS
        )
        
        # Should resolve successfully
        assert resolution_success
        
        await mcp_handler.stop()
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])