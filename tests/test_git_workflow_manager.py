"""
Tests for GitWorkflowManager - automated Git workflow integration.
"""

import unittest
import tempfile
import shutil
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from app.git_workflow_manager import GitWorkflowManager, TaskContext


class TestGitWorkflowManager(unittest.TestCase):
    """Test cases for GitWorkflowManager functionality."""
    
    def setUp(self):
        """Set up test environment with temporary Git repository."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Initialize a Git repository
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
        
        # Create initial commit
        Path("README.md").write_text("# Test Repository")
        subprocess.run(["git", "add", "README.md"], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
        
        self.manager = GitWorkflowManager(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test GitWorkflowManager initialization."""
        self.assertTrue(self.manager.repo_path.exists())
        self.assertTrue(self.manager.task_tracking_file.exists())
        
        # Check tracking file content
        tracking_data = json.loads(self.manager.task_tracking_file.read_text())
        self.assertIn("completed_tasks", tracking_data)
        self.assertIn("current_branch", tracking_data)
        self.assertIn("last_sync", tracking_data)
    
    def test_get_git_status(self):
        """Test Git status retrieval."""
        # Create a modified file
        Path("test_file.py").write_text("print('hello')")
        
        status = self.manager.get_git_status()
        
        self.assertIn("test_file.py", status["untracked"])
        self.assertIsInstance(status["modified"], list)
        self.assertIsInstance(status["added"], list)
        self.assertIsInstance(status["deleted"], list)
    
    def test_generate_commit_message(self):
        """Test commit message generation."""
        task_context = TaskContext(
            task_id="1.1",
            task_name="Implement user authentication",
            description="Add login and registration functionality",
            files_modified=["app/auth.py", "tests/test_auth.py"],
            requirements_addressed=["User login", "User registration"],
            completion_time=datetime.now()
        )
        
        message = self.manager.generate_commit_message(task_context)
        
        self.assertIn("feat: implement user authentication", message)
        self.assertIn("Task ID: 1.1", message)
        self.assertIn("app/auth.py", message)
        self.assertIn("User login", message)
    
    def test_create_feature_branch(self):
        """Test feature branch creation."""
        branch_name = self.manager.create_feature_branch("1.1")
        
        self.assertEqual(branch_name, "task/1-1")
        
        # Check that we're on the new branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True
        )
        self.assertEqual(result.stdout.strip(), "task/1-1")
    
    def test_get_modified_files(self):
        """Test getting list of modified files."""
        # Create some test files
        Path("file1.py").write_text("content1")
        Path("file2.py").write_text("content2")
        
        modified_files = self.manager.get_modified_files()
        
        self.assertIn("file1.py", modified_files)
        self.assertIn("file2.py", modified_files)
    
    def test_update_task_tracking(self):
        """Test task tracking updates."""
        task_context = TaskContext(
            task_id="1.0",
            task_name="Test task",
            description="Test description",
            files_modified=["test.py"],
            requirements_addressed=["Test requirement"],
            completion_time=datetime.now()
        )
        
        self.manager.update_task_tracking(task_context, "abc123")
        
        # Check tracking file was updated
        tracking_data = json.loads(self.manager.task_tracking_file.read_text())
        self.assertEqual(len(tracking_data["completed_tasks"]), 1)
        self.assertEqual(tracking_data["completed_tasks"][0]["task_id"], "1.0")
        self.assertEqual(tracking_data["completed_tasks"][0]["commit_hash"], "abc123")
    
    @patch('subprocess.run')
    def test_commit_task_completion(self, mock_run):
        """Test task completion commit process."""
        # Mock subprocess calls
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=0),  # git commit
            MagicMock(stdout="abc123def", returncode=0),  # git rev-parse HEAD
            MagicMock(returncode=0)   # git push
        ]
        
        task_context = TaskContext(
            task_id="1.0",
            task_name="Test task",
            description="Test description",
            files_modified=["test.py"],
            requirements_addressed=["Test requirement"],
            completion_time=datetime.now()
        )
        
        commit_hash = self.manager.commit_task_completion(task_context)
        
        self.assertEqual(commit_hash, "abc123def")
        self.assertEqual(mock_run.call_count, 4)
    
    @patch('subprocess.run')
    def test_push_to_remote(self, mock_run):
        """Test pushing to remote repository."""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = self.manager.push_to_remote()
        
        self.assertTrue(result)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_push_to_remote_failure(self, mock_run):
        """Test handling of push failures."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git push")
        
        result = self.manager.push_to_remote()
        
        self.assertFalse(result)
    
    def test_check_for_conflicts(self):
        """Test conflict detection."""
        conflicts = self.manager.check_for_conflicts()
        
        # Should be empty list when no conflicts
        self.assertEqual(conflicts, [])
        self.assertIsInstance(conflicts, list)
    
    def test_auto_complete_task(self):
        """Test automatic task completion workflow."""
        # Create a test file to be committed
        Path("completed_task.py").write_text("# Task completed")
        
        with patch.object(self.manager, 'commit_task_completion') as mock_commit:
            mock_commit.return_value = "abc123"
            
            commit_hash = self.manager.auto_complete_task(
                task_id="1.0",
                task_name="Test task completion",
                description="Automated task completion test",
                requirements=["Test requirement"]
            )
            
            self.assertEqual(commit_hash, "abc123")
            mock_commit.assert_called_once()


class TestTaskContext(unittest.TestCase):
    """Test cases for TaskContext data class."""
    
    def test_task_context_creation(self):
        """Test TaskContext object creation."""
        context = TaskContext(
            task_id="1.1",
            task_name="Test Task",
            description="Test Description",
            files_modified=["file1.py", "file2.py"],
            requirements_addressed=["Req1", "Req2"],
            completion_time=datetime.now(),
            branch_name="feature/test"
        )
        
        self.assertEqual(context.task_id, "1.1")
        self.assertEqual(context.task_name, "Test Task")
        self.assertEqual(len(context.files_modified), 2)
        self.assertEqual(len(context.requirements_addressed), 2)
        self.assertEqual(context.branch_name, "feature/test")


if __name__ == '__main__':
    unittest.main()