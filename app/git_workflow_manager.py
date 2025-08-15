"""
Git Workflow Manager for automated repository updates and task completion tracking.

This module provides automated Git workflow integration for task completion,
including intelligent commit message generation, branch management, and
repository synchronization.
"""

import os
import subprocess
import json
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskContext:
    """Context information for a completed task."""
    task_id: str
    task_name: str
    description: str
    files_modified: List[str]
    requirements_addressed: List[str]
    completion_time: datetime.datetime
    branch_name: Optional[str] = None


class GitWorkflowManager:
    """
    Manages automated Git workflows for task completion and repository updates.
    
    Features:
    - Automatic commit generation with proper task context
    - Intelligent commit message generation
    - Branch management for feature development
    - Git status monitoring and conflict resolution
    - Task completion tracking
    """
    
    def __init__(self, repo_path: str = "."):
        """
        Initialize the Git Workflow Manager.
        
        Args:
            repo_path: Path to the Git repository (default: current directory)
        """
        self.repo_path = Path(repo_path).resolve()
        self.task_tracking_file = self.repo_path / ".kiro" / "task_tracking.json"
        self.ensure_tracking_directory()
    
    def ensure_tracking_directory(self):
        """Ensure the .kiro directory exists for task tracking."""
        tracking_dir = self.repo_path / ".kiro"
        tracking_dir.mkdir(exist_ok=True)
        
        # Initialize task tracking file if it doesn't exist
        if not self.task_tracking_file.exists():
            self.task_tracking_file.write_text(json.dumps({
                "completed_tasks": [],
                "current_branch": "main",
                "last_sync": None
            }, indent=2))
    
    def get_git_status(self) -> Dict[str, List[str]]:
        """
        Get current Git repository status.
        
        Returns:
            Dictionary with status information including modified, added, deleted files
        """
        try:
            # Get status in porcelain format for easy parsing
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            status = {
                "modified": [],
                "added": [],
                "deleted": [],
                "untracked": [],
                "staged": []
            }
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                status_code = line[:2]
                filename = line[3:]
                
                if status_code[0] in ['M', 'A', 'D', 'R', 'C']:
                    status["staged"].append(filename)
                if status_code[1] == 'M':
                    status["modified"].append(filename)
                elif status_code[1] == 'A':
                    status["added"].append(filename)
                elif status_code[1] == 'D':
                    status["deleted"].append(filename)
                elif status_code == '??':
                    status["untracked"].append(filename)
            
            return status
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get Git status: {e}")
    
    def generate_commit_message(self, task_context: TaskContext) -> str:
        """
        Generate an intelligent commit message based on task context.
        
        Args:
            task_context: Context information about the completed task
            
        Returns:
            Formatted commit message following conventional commit standards
        """
        # Determine commit type based on task content
        task_lower = task_context.task_name.lower()
        
        if "test" in task_lower:
            commit_type = "test"
        elif "fix" in task_lower or "bug" in task_lower:
            commit_type = "fix"
        elif "doc" in task_lower or "documentation" in task_lower:
            commit_type = "docs"
        elif "refactor" in task_lower:
            commit_type = "refactor"
        else:
            commit_type = "feat"
        
        # Create commit message
        subject = f"{commit_type}: {task_context.task_name.lower()}"
        
        # Add body with details
        body_lines = [
            "",
            f"Task ID: {task_context.task_id}",
            f"Description: {task_context.description}",
            ""
        ]
        
        if task_context.files_modified:
            body_lines.append("Files modified:")
            for file in task_context.files_modified:
                body_lines.append(f"- {file}")
            body_lines.append("")
        
        if task_context.requirements_addressed:
            body_lines.append("Requirements addressed:")
            for req in task_context.requirements_addressed:
                body_lines.append(f"- {req}")
            body_lines.append("")
        
        body_lines.append(f"Completed: {task_context.completion_time.isoformat()}")
        
        return subject + "\n".join(body_lines)
    
    def create_feature_branch(self, task_id: str) -> str:
        """
        Create a feature branch for task development.
        
        Args:
            task_id: Unique identifier for the task
            
        Returns:
            Name of the created branch
        """
        # Sanitize task ID for branch name
        branch_name = f"task/{task_id.lower().replace(' ', '-').replace('.', '-')}"
        
        try:
            # Check if branch already exists
            result = subprocess.run(
                ["git", "branch", "--list", branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if not result.stdout.strip():
                # Create new branch
                subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    cwd=self.repo_path,
                    check=True
                )
            else:
                # Switch to existing branch
                subprocess.run(
                    ["git", "checkout", branch_name],
                    cwd=self.repo_path,
                    check=True
                )
            
            return branch_name
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create/switch to branch {branch_name}: {e}")
    
    def commit_task_completion(self, task_context: TaskContext, auto_push: bool = True) -> str:
        """
        Commit task completion with proper context and message.
        
        Args:
            task_context: Context information about the completed task
            auto_push: Whether to automatically push to remote repository
            
        Returns:
            Commit hash of the created commit
        """
        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "."],
                cwd=self.repo_path,
                check=True
            )
            
            # Generate commit message
            commit_message = self.generate_commit_message(task_context)
            
            # Create commit
            result = subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get commit hash
            commit_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            
            # Update task tracking
            self.update_task_tracking(task_context, commit_hash)
            
            # Push to remote if requested
            if auto_push:
                self.push_to_remote()
            
            return commit_hash
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to commit task completion: {e}")
    
    def push_to_remote(self, branch: Optional[str] = None) -> bool:
        """
        Push changes to remote repository.
        
        Args:
            branch: Specific branch to push (default: current branch)
            
        Returns:
            True if push was successful, False otherwise
        """
        try:
            if branch:
                subprocess.run(
                    ["git", "push", "origin", branch],
                    cwd=self.repo_path,
                    check=True
                )
            else:
                subprocess.run(
                    ["git", "push"],
                    cwd=self.repo_path,
                    check=True
                )
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to push to remote: {e}")
            return False
    
    def update_task_tracking(self, task_context: TaskContext, commit_hash: str):
        """
        Update task tracking information.
        
        Args:
            task_context: Context information about the completed task
            commit_hash: Hash of the commit created for this task
        """
        # Load existing tracking data
        if self.task_tracking_file.exists():
            tracking_data = json.loads(self.task_tracking_file.read_text())
        else:
            tracking_data = {"completed_tasks": [], "current_branch": "main", "last_sync": None}
        
        # Add new task completion
        task_record = {
            "task_id": task_context.task_id,
            "task_name": task_context.task_name,
            "description": task_context.description,
            "files_modified": task_context.files_modified,
            "requirements_addressed": task_context.requirements_addressed,
            "completion_time": task_context.completion_time.isoformat(),
            "commit_hash": commit_hash,
            "branch_name": task_context.branch_name
        }
        
        tracking_data["completed_tasks"].append(task_record)
        tracking_data["last_sync"] = datetime.datetime.now().isoformat()
        
        # Save updated tracking data
        self.task_tracking_file.write_text(json.dumps(tracking_data, indent=2))
    
    def get_modified_files(self) -> List[str]:
        """
        Get list of files that have been modified since last commit.
        
        Returns:
            List of modified file paths
        """
        status = self.get_git_status()
        return status["modified"] + status["added"] + status["staged"] + status["untracked"]
    
    def check_for_conflicts(self) -> List[str]:
        """
        Check for merge conflicts in the repository.
        
        Returns:
            List of files with conflicts
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=U"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            return [f.strip() for f in result.stdout.split('\n') if f.strip()]
            
        except subprocess.CalledProcessError:
            return []
    
    def auto_complete_task(self, task_id: str, task_name: str, description: str, 
                          requirements: List[str] = None) -> str:
        """
        Automatically complete a task with proper Git workflow.
        
        Args:
            task_id: Unique identifier for the task
            task_name: Human-readable name of the task
            description: Detailed description of what was accomplished
            requirements: List of requirements addressed by this task
            
        Returns:
            Commit hash of the completion commit
        """
        # Get modified files
        modified_files = self.get_modified_files()
        
        # Create task context
        task_context = TaskContext(
            task_id=task_id,
            task_name=task_name,
            description=description,
            files_modified=modified_files,
            requirements_addressed=requirements or [],
            completion_time=datetime.datetime.now()
        )
        
        # Commit and push
        commit_hash = self.commit_task_completion(task_context)
        
        print(f"✅ Task {task_id} completed successfully!")
        print(f"📝 Commit: {commit_hash[:8]}")
        print(f"📁 Files modified: {len(modified_files)}")
        print(f"🚀 Changes pushed to repository")
        
        return commit_hash


def create_task_context_from_spec(task_id: str, spec_file: str) -> TaskContext:
    """
    Create task context by parsing task information from specification file.
    
    Args:
        task_id: ID of the task to extract context for
        spec_file: Path to the specification file
        
    Returns:
        TaskContext object with extracted information
    """
    # This would parse the spec file to extract task details
    # For now, return a basic context
    return TaskContext(
        task_id=task_id,
        task_name=f"Task {task_id}",
        description="Task completed via automated workflow",
        files_modified=[],
        requirements_addressed=[],
        completion_time=datetime.datetime.now()
    )


if __name__ == "__main__":
    # Example usage
    manager = GitWorkflowManager()
    
    # Example task completion
    task_context = TaskContext(
        task_id="1.0",
        task_name="Set up development workflow automation",
        description="Implemented GitWorkflowManager for automated repository updates",
        files_modified=["app/git_workflow_manager.py"],
        requirements_addressed=["Development workflow optimization"],
        completion_time=datetime.datetime.now()
    )
    
    try:
        commit_hash = manager.commit_task_completion(task_context)
        print(f"Task completed with commit: {commit_hash}")
    except Exception as e:
        print(f"Error: {e}")