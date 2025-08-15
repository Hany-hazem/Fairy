"""
Unit tests for version control integration
"""

import pytest
import tempfile
import shutil
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.version_control import GitIntegration, ChangeRecord


class TestGitIntegration:
    """Test cases for GitIntegration class"""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def git_integration(self, temp_repo):
        """Create GitIntegration instance with temp repo"""
        return GitIntegration(temp_repo)
    
    def test_init_creates_audit_directory(self, temp_repo):
        """Test that initialization creates audit directory"""
        git_integration = GitIntegration(temp_repo)
        
        audit_dir = Path(temp_repo) / ".kiro" / "audit"
        assert audit_dir.exists()
        assert (audit_dir / "changes.json").exists()
    
    @patch('subprocess.run')
    def test_is_git_repo_true(self, mock_run, git_integration):
        """Test is_git_repo returns True for valid repo"""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = git_integration.is_git_repo()
        
        assert result is True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_is_git_repo_false(self, mock_run, git_integration):
        """Test is_git_repo returns False for invalid repo"""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')
        
        result = git_integration.is_git_repo()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_init_repo_if_needed_success(self, mock_run, git_integration):
        """Test successful repository initialization"""
        # First call (is_git_repo) fails, second call (git init) succeeds
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, 'git'),  # is_git_repo fails
            MagicMock(returncode=0)  # git init succeeds
        ]
        
        result = git_integration.init_repo_if_needed()
        
        assert result is True
        assert mock_run.call_count == 2
    
    @patch('subprocess.run')
    def test_init_repo_if_needed_already_exists(self, mock_run, git_integration):
        """Test when repository already exists"""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = git_integration.init_repo_if_needed()
        
        assert result is True
        mock_run.assert_called_once()  # Only is_git_repo called
    
    @patch('subprocess.run')
    def test_get_current_commit_hash(self, mock_run, git_integration):
        """Test getting current commit hash"""
        expected_hash = "abc123def456"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f"{expected_hash}\n"
        )
        
        result = git_integration.get_current_commit_hash()
        
        assert result == expected_hash
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "HEAD"],
            cwd=git_integration.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('subprocess.run')
    def test_get_current_branch(self, mock_run, git_integration):
        """Test getting current branch name"""
        expected_branch = "main"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f"{expected_branch}\n"
        )
        
        result = git_integration.get_current_branch()
        
        assert result == expected_branch
    
    @patch('subprocess.run')
    def test_create_rollback_point_success(self, mock_run, git_integration):
        """Test successful rollback point creation"""
        expected_hash = "rollback123"
        
        # Mock the sequence: is_git_repo, git add, git commit, get_current_commit_hash
        mock_run.side_effect = [
            MagicMock(returncode=0),  # is_git_repo
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=0),  # git commit
            MagicMock(returncode=0, stdout=f"{expected_hash}\n")  # get commit hash
        ]
        
        result = git_integration.create_rollback_point("Test rollback")
        
        assert result == expected_hash
        assert mock_run.call_count == 4
    
    @patch('subprocess.run')
    def test_create_improvement_branch(self, mock_run, git_integration):
        """Test creating improvement branch"""
        improvement_id = "test-improvement"
        
        mock_run.side_effect = [
            MagicMock(returncode=0),  # is_git_repo
            MagicMock(returncode=0)   # git checkout -b
        ]
        
        result = git_integration.create_improvement_branch(improvement_id)
        
        assert result is not None
        assert result.startswith(f"improvement-{improvement_id}-")
        assert mock_run.call_count == 2
    
    @patch('subprocess.run')
    def test_switch_to_branch(self, mock_run, git_integration):
        """Test switching to branch"""
        branch_name = "test-branch"
        mock_run.return_value = MagicMock(returncode=0)
        
        result = git_integration.switch_to_branch(branch_name)
        
        assert result is True
        mock_run.assert_called_once_with(
            ["git", "checkout", branch_name],
            cwd=git_integration.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('subprocess.run')
    def test_commit_changes(self, mock_run, git_integration):
        """Test committing changes"""
        files = ["file1.py", "file2.py"]
        message = "Test commit"
        expected_hash = "commit123"
        
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add file1.py
            MagicMock(returncode=0),  # git add file2.py
            MagicMock(returncode=0),  # git commit
            MagicMock(returncode=0, stdout=f"{expected_hash}\n")  # get commit hash
        ]
        
        result = git_integration.commit_changes(files, message)
        
        assert result == expected_hash
        assert mock_run.call_count == 4
    
    @patch('subprocess.run')
    def test_rollback_to_commit(self, mock_run, git_integration):
        """Test rolling back to commit"""
        commit_hash = "abc123"
        mock_run.return_value = MagicMock(returncode=0)
        
        result = git_integration.rollback_to_commit(commit_hash)
        
        assert result is True
        mock_run.assert_called_once_with(
            ["git", "reset", "--hard", commit_hash],
            cwd=git_integration.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('subprocess.run')
    def test_get_file_changes(self, mock_run, git_integration):
        """Test getting file changes between commits"""
        commit1 = "abc123"
        commit2 = "def456"
        expected_files = ["file1.py", "file2.py"]
        
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="\n".join(expected_files) + "\n"
        )
        
        result = git_integration.get_file_changes(commit1, commit2)
        
        assert result == expected_files
        mock_run.assert_called_once_with(
            ["git", "diff", "--name-only", commit1, commit2],
            cwd=git_integration.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('subprocess.run')
    def test_get_uncommitted_changes(self, mock_run, git_integration):
        """Test getting uncommitted changes"""
        modified_files = ["modified.py"]
        staged_files = ["staged.py"]
        
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="\n".join(modified_files) + "\n"),
            MagicMock(returncode=0, stdout="\n".join(staged_files) + "\n")
        ]
        
        result = git_integration.get_uncommitted_changes()
        
        assert set(result) == set(modified_files + staged_files)
        assert mock_run.call_count == 2
    
    def test_log_change(self, git_integration):
        """Test logging a change to audit log"""
        change_record = ChangeRecord(
            id="test-change-123",
            timestamp=datetime.now(),
            branch_name="test-branch",
            commit_hash="abc123",
            files_modified=["test.py"],
            description="Test change",
            rollback_point="rollback123",
            status="applied"
        )
        
        git_integration.log_change(change_record)
        
        # Verify the change was logged
        changes = git_integration.get_change_history()
        assert len(changes) == 1
        assert changes[0].id == change_record.id
        assert changes[0].description == change_record.description
    
    def test_get_change_history(self, git_integration):
        """Test getting change history"""
        # Create multiple change records
        changes = []
        for i in range(3):
            change = ChangeRecord(
                id=f"change-{i}",
                timestamp=datetime.now(),
                branch_name=f"branch-{i}",
                commit_hash=f"hash-{i}",
                files_modified=[f"file{i}.py"],
                description=f"Change {i}",
                rollback_point=f"rollback-{i}",
                status="applied"
            )
            changes.append(change)
            git_integration.log_change(change)
        
        # Get history
        history = git_integration.get_change_history()
        
        assert len(history) == 3
        # Should be sorted by timestamp, most recent first
        assert history[0].id == "change-2"
        assert history[1].id == "change-1"
        assert history[2].id == "change-0"
    
    def test_get_change_history_with_limit(self, git_integration):
        """Test getting change history with limit"""
        # Create multiple change records
        for i in range(5):
            change = ChangeRecord(
                id=f"change-{i}",
                timestamp=datetime.now(),
                branch_name=f"branch-{i}",
                commit_hash=f"hash-{i}",
                files_modified=[f"file{i}.py"],
                description=f"Change {i}",
                rollback_point=f"rollback-{i}",
                status="applied"
            )
            git_integration.log_change(change)
        
        # Get limited history
        history = git_integration.get_change_history(limit=2)
        
        assert len(history) == 2
    
    def test_generate_change_id(self, git_integration):
        """Test generating unique change ID"""
        improvement_id = "test-improvement"
        files = ["file1.py", "file2.py"]
        
        change_id1 = git_integration.generate_change_id(improvement_id, files)
        change_id2 = git_integration.generate_change_id(improvement_id, files)
        
        # Should be different due to timestamp
        assert change_id1 != change_id2
        assert len(change_id1) == 12
        assert len(change_id2) == 12
    
    @patch('subprocess.run')
    def test_cleanup_old_branches(self, mock_run, git_integration):
        """Test cleaning up old branches"""
        old_timestamp = int(datetime.now().timestamp()) - (10 * 24 * 3600)  # 10 days ago
        recent_timestamp = int(datetime.now().timestamp()) - (1 * 24 * 3600)  # 1 day ago
        
        branches_output = f"""  improvement-test-{old_timestamp}
  improvement-test-{recent_timestamp}
* main"""
        
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=branches_output),  # git branch --list
            MagicMock(returncode=0)  # git branch -D (for old branch)
        ]
        
        git_integration.cleanup_old_branches(keep_days=7)
        
        # Should call git branch --list and git branch -D for old branch
        assert mock_run.call_count == 2
        
        # Verify the old branch deletion call
        delete_call = mock_run.call_args_list[1]
        assert delete_call[0][0] == ["git", "branch", "-D", f"improvement-test-{old_timestamp}"]


class TestChangeRecord:
    """Test cases for ChangeRecord model"""
    
    def test_change_record_creation(self):
        """Test creating a ChangeRecord"""
        change = ChangeRecord(
            id="test-123",
            timestamp=datetime.now(),
            branch_name="test-branch",
            commit_hash="abc123",
            files_modified=["test.py"],
            description="Test change",
            rollback_point="rollback123",
            status="applied"
        )
        
        assert change.id == "test-123"
        assert change.branch_name == "test-branch"
        assert change.commit_hash == "abc123"
        assert change.files_modified == ["test.py"]
        assert change.description == "Test change"
        assert change.rollback_point == "rollback123"
        assert change.status == "applied"
    
    def test_change_record_optional_fields(self):
        """Test ChangeRecord with optional fields"""
        change = ChangeRecord(
            id="test-123",
            timestamp=datetime.now(),
            branch_name="test-branch",
            commit_hash="abc123",
            files_modified=["test.py"],
            description="Test change",
            rollback_point="rollback123",
            status="applied",
            improvement_id="improvement-456"
        )
        
        assert change.improvement_id == "improvement-456"
    
    def test_change_record_serialization(self):
        """Test ChangeRecord serialization to dict"""
        change = ChangeRecord(
            id="test-123",
            timestamp=datetime.now(),
            branch_name="test-branch",
            commit_hash="abc123",
            files_modified=["test.py"],
            description="Test change",
            rollback_point="rollback123",
            status="applied"
        )
        
        change_dict = change.model_dump()
        
        assert change_dict["id"] == "test-123"
        assert change_dict["branch_name"] == "test-branch"
        assert change_dict["commit_hash"] == "abc123"
        assert change_dict["files_modified"] == ["test.py"]
        assert change_dict["description"] == "Test change"
        assert change_dict["rollback_point"] == "rollback123"
        assert change_dict["status"] == "applied"