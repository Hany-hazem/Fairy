"""
Git Workflow Manager for automated repository updates and task completion tracking.

This module provides automated Git workflow integration for task completion,
including intelligent commit message generation, branch management, and
repository synchronization with enhanced task-based workflow support.

Copyright (c) 2024 Hani Hazem
Licensed under the MIT License. See LICENSE file in the project root for full license information.
Repository: https://github.com/Hany-hazem/Fairy
Contact: hany.hazem.cs@gmail.com
"""

import os
import subprocess
import json
import datetime
import re
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    MERGED = "merged"
    ABANDONED = "abandoned"


class BranchType(Enum):
    """Branch type enumeration."""
    FEATURE = "feature"
    TASK = "task"
    HOTFIX = "hotfix"
    BUGFIX = "bugfix"


@dataclass
class TaskContext:
    """Context information for a task with enhanced workflow support."""
    task_id: str
    task_name: str
    description: str
    files_modified: List[str]
    requirements_addressed: List[str]
    completion_time: datetime.datetime
    branch_name: Optional[str] = None
    status: TaskStatus = TaskStatus.NOT_STARTED
    branch_type: BranchType = BranchType.TASK
    parent_branch: str = "main"
    created_at: Optional[datetime.datetime] = None
    started_at: Optional[datetime.datetime] = None
    commits: List[str] = field(default_factory=list)
    merge_ready: bool = False
    
    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.datetime.now()


@dataclass
class BranchInfo:
    """Information about a Git branch."""
    name: str
    branch_type: BranchType
    task_id: Optional[str]
    created_at: datetime.datetime
    last_commit: Optional[str] = None
    is_merged: bool = False
    merge_target: str = "main"


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
    
    def create_task_branch(self, task_id: str, task_name: str, 
                          branch_type: BranchType = BranchType.TASK,
                          parent_branch: str = "main") -> str:
        """
        Create a task branch with descriptive naming conventions.
        
        Args:
            task_id: Unique identifier for the task
            task_name: Human-readable name of the task
            branch_type: Type of branch to create
            parent_branch: Parent branch to branch from
            
        Returns:
            Name of the created branch
        """
        # Generate descriptive branch name
        branch_name = self._generate_branch_name(task_id, task_name, branch_type)
        
        try:
            # Ensure we're on the parent branch and it's up to date
            self._ensure_branch_updated(parent_branch)
            
            # Check if branch already exists
            if self._branch_exists(branch_name):
                logger.info(f"Branch {branch_name} already exists, switching to it")
                self._switch_to_branch(branch_name)
            else:
                # Create new branch from parent
                logger.info(f"Creating new branch {branch_name} from {parent_branch}")
                subprocess.run(
                    ["git", "checkout", "-b", branch_name, parent_branch],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )
            
            # Update task tracking with branch info
            self._track_branch_creation(task_id, branch_name, branch_type, parent_branch)
            
            return branch_name
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create/switch to branch {branch_name}: {e}")
    
    def _generate_branch_name(self, task_id: str, task_name: str, 
                             branch_type: BranchType) -> str:
        """
        Generate descriptive branch name following naming conventions.
        
        Args:
            task_id: Task identifier
            task_name: Task name
            branch_type: Type of branch
            
        Returns:
            Generated branch name
        """
        # Sanitize task name for branch naming
        sanitized_name = re.sub(r'[^a-zA-Z0-9\s-]', '', task_name.lower())
        sanitized_name = re.sub(r'\s+', '-', sanitized_name.strip())
        sanitized_name = sanitized_name[:50]  # Limit length
        
        # Sanitize task ID
        sanitized_id = task_id.replace('.', '-').replace(' ', '-').lower()
        
        # Generate branch name based on type
        if branch_type == BranchType.FEATURE:
            return f"feature/{sanitized_id}-{sanitized_name}"
        elif branch_type == BranchType.HOTFIX:
            return f"hotfix/{sanitized_id}-{sanitized_name}"
        elif branch_type == BranchType.BUGFIX:
            return f"bugfix/{sanitized_id}-{sanitized_name}"
        else:  # TASK
            return f"task/{sanitized_id}-{sanitized_name}"
    
    def _ensure_branch_updated(self, branch_name: str) -> None:
        """
        Ensure branch exists and is up to date.
        
        Args:
            branch_name: Name of the branch to update
        """
        try:
            # Switch to branch
            subprocess.run(
                ["git", "checkout", branch_name],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            
            # Try to pull latest changes if remote exists
            try:
                subprocess.run(
                    ["git", "pull", "origin", branch_name],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError:
                # Remote branch might not exist, that's okay
                pass
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not update branch {branch_name}: {e}")
    
    def _branch_exists(self, branch_name: str) -> bool:
        """
        Check if a branch exists locally.
        
        Args:
            branch_name: Name of the branch to check
            
        Returns:
            True if branch exists, False otherwise
        """
        try:
            result = subprocess.run(
                ["git", "branch", "--list", branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False
    
    def _switch_to_branch(self, branch_name: str) -> None:
        """
        Switch to specified branch.
        
        Args:
            branch_name: Name of the branch to switch to
        """
        subprocess.run(
            ["git", "checkout", branch_name],
            cwd=self.repo_path,
            check=True,
            capture_output=True
        )
    
    def _track_branch_creation(self, task_id: str, branch_name: str, 
                              branch_type: BranchType, parent_branch: str) -> None:
        """
        Track branch creation in task tracking system.
        
        Args:
            task_id: Task identifier
            branch_name: Name of created branch
            branch_type: Type of branch
            parent_branch: Parent branch name
        """
        # Load existing tracking data
        tracking_data = self._load_tracking_data()
        
        # Add branch info
        if "branches" not in tracking_data:
            tracking_data["branches"] = {}
            
        tracking_data["branches"][branch_name] = {
            "task_id": task_id,
            "branch_type": branch_type.value,
            "parent_branch": parent_branch,
            "created_at": datetime.datetime.now().isoformat(),
            "is_merged": False,
            "commits": []
        }
        
        # Save tracking data
        self._save_tracking_data(tracking_data)
    
    def create_feature_branch(self, task_id: str) -> str:
        """
        Create a feature branch for task development (legacy method).
        
        Args:
            task_id: Unique identifier for the task
            
        Returns:
            Name of the created branch
        """
        # Sanitize task ID for branch name (legacy behavior)
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
    
    def commit_task_progress(self, task_id: str, files: List[str], 
                            message: str, auto_stage: bool = True) -> str:
        """
        Commit task progress with automatic staging and context.
        
        Args:
            task_id: Task identifier
            files: List of files to commit (empty list commits all changes)
            message: Commit message
            auto_stage: Whether to automatically stage files
            
        Returns:
            Commit hash of the created commit
        """
        try:
            # Stage files
            if auto_stage:
                if files:
                    # Stage specific files
                    for file_path in files:
                        subprocess.run(
                            ["git", "add", file_path],
                            cwd=self.repo_path,
                            check=True
                        )
                else:
                    # Stage all changes
                    subprocess.run(
                        ["git", "add", "."],
                        cwd=self.repo_path,
                        check=True
                    )
            
            # Generate enhanced commit message with task context
            enhanced_message = self._enhance_commit_message(task_id, message, files)
            
            # Create commit
            subprocess.run(
                ["git", "commit", "-m", enhanced_message],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            
            # Get commit hash
            commit_hash = self._get_current_commit_hash()
            
            # Update task tracking with commit
            self._track_task_commit(task_id, commit_hash, files, message)
            
            logger.info(f"Committed task {task_id} progress: {commit_hash[:8]}")
            return commit_hash
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to commit task progress: {e}")
    
    def _enhance_commit_message(self, task_id: str, message: str, 
                               files: List[str]) -> str:
        """
        Enhance commit message with task context and metadata.
        
        Args:
            task_id: Task identifier
            message: Original commit message
            files: List of modified files
            
        Returns:
            Enhanced commit message
        """
        # Load task context if available
        tracking_data = self._load_tracking_data()
        task_info = None
        
        # Find task in completed tasks or current branches
        for task in tracking_data.get("completed_tasks", []):
            if task["task_id"] == task_id:
                task_info = task
                break
        
        # Build enhanced message
        lines = [message, ""]
        lines.append(f"Task-ID: {task_id}")
        
        if task_info:
            if task_info.get("requirements_addressed"):
                lines.append(f"Requirements: {', '.join(task_info['requirements_addressed'])}")
        
        if files:
            lines.append(f"Files: {', '.join(files[:5])}")  # Limit to first 5 files
            if len(files) > 5:
                lines.append(f"... and {len(files) - 5} more files")
        
        lines.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
        
        return "\n".join(lines)
    
    def _get_current_commit_hash(self) -> str:
        """Get the current commit hash."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    
    def _track_task_commit(self, task_id: str, commit_hash: str, 
                          files: List[str], message: str) -> None:
        """
        Track task commit in the tracking system.
        
        Args:
            task_id: Task identifier
            commit_hash: Hash of the commit
            files: List of files in the commit
            message: Commit message
        """
        tracking_data = self._load_tracking_data()
        
        # Find and update branch info
        current_branch = self._get_current_branch()
        if current_branch and "branches" in tracking_data:
            branch_info = tracking_data["branches"].get(current_branch)
            if branch_info and branch_info.get("task_id") == task_id:
                branch_info["commits"].append({
                    "hash": commit_hash,
                    "message": message,
                    "files": files,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        self._save_tracking_data(tracking_data)
    
    def _get_current_branch(self) -> Optional[str]:
        """Get the current branch name."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def complete_task_branch(self, task_id: str, completion_notes: str = "") -> bool:
        """
        Complete a task branch and prepare it for merge.
        
        Args:
            task_id: Task identifier
            completion_notes: Additional notes about task completion
            
        Returns:
            True if branch was successfully prepared for merge
        """
        try:
            current_branch = self._get_current_branch()
            if not current_branch:
                raise RuntimeError("Could not determine current branch")
            
            # Ensure all changes are committed
            if self._has_uncommitted_changes():
                raise RuntimeError("Branch has uncommitted changes. Please commit or stash them first.")
            
            # Update task tracking
            tracking_data = self._load_tracking_data()
            
            if "branches" in tracking_data and current_branch in tracking_data["branches"]:
                branch_info = tracking_data["branches"][current_branch]
                if branch_info.get("task_id") == task_id:
                    branch_info["completed_at"] = datetime.datetime.now().isoformat()
                    branch_info["completion_notes"] = completion_notes
                    branch_info["merge_ready"] = True
                    
                    # Create completion commit if there are any final changes
                    self._create_completion_commit(task_id, completion_notes)
                    
                    self._save_tracking_data(tracking_data)
                    
                    logger.info(f"Task {task_id} branch {current_branch} marked as complete and ready for merge")
                    return True
            
            logger.warning(f"Could not find branch info for task {task_id}")
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to complete task branch: {e}")
            return False
    
    def _has_uncommitted_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return True  # Assume there are changes if we can't check
    
    def _create_completion_commit(self, task_id: str, completion_notes: str) -> Optional[str]:
        """
        Create a completion commit if needed.
        
        Args:
            task_id: Task identifier
            completion_notes: Completion notes
            
        Returns:
            Commit hash if commit was created, None otherwise
        """
        if not self._has_uncommitted_changes():
            return None
        
        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "."],
                cwd=self.repo_path,
                check=True
            )
            
            # Create completion commit
            message = f"Complete task {task_id}"
            if completion_notes:
                message += f"\n\n{completion_notes}"
            
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            
            return self._get_current_commit_hash()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create completion commit: {e}")
            return None
    
    def commit_task_completion(self, task_context: TaskContext, auto_push: bool = True) -> str:
        """
        Commit task completion with proper context and message (legacy method).
        
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
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get commit hash
            commit_hash = self._get_current_commit_hash()
            
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
    
    def _load_tracking_data(self) -> Dict:
        """Load task tracking data from file."""
        if self.task_tracking_file.exists():
            return json.loads(self.task_tracking_file.read_text())
        else:
            return {
                "completed_tasks": [],
                "current_branch": "main",
                "last_sync": None,
                "branches": {}
            }
    
    def _save_tracking_data(self, tracking_data: Dict) -> None:
        """Save task tracking data to file."""
        self.task_tracking_file.write_text(json.dumps(tracking_data, indent=2))
    
    def update_task_tracking(self, task_context: TaskContext, commit_hash: str):
        """
        Update task tracking information.
        
        Args:
            task_context: Context information about the completed task
            commit_hash: Hash of the commit created for this task
        """
        # Load existing tracking data
        tracking_data = self._load_tracking_data()
        
        # Add new task completion
        task_record = {
            "task_id": task_context.task_id,
            "task_name": task_context.task_name,
            "description": task_context.description,
            "files_modified": task_context.files_modified,
            "requirements_addressed": task_context.requirements_addressed,
            "completion_time": task_context.completion_time.isoformat(),
            "commit_hash": commit_hash,
            "branch_name": task_context.branch_name,
            "status": task_context.status.value if hasattr(task_context.status, 'value') else str(task_context.status)
        }
        
        tracking_data["completed_tasks"].append(task_record)
        tracking_data["last_sync"] = datetime.datetime.now().isoformat()
        
        # Save updated tracking data
        self._save_tracking_data(tracking_data)
    
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
    
    def detect_merge_conflicts(self, source_branch: str, target_branch: str) -> Dict[str, List[str]]:
        """
        Detect potential merge conflicts between two branches.
        
        Args:
            source_branch: Source branch to merge from
            target_branch: Target branch to merge into
            
        Returns:
            Dictionary with conflict information
        """
        try:
            # Save current branch
            current_branch = self._get_current_branch()
            
            # Switch to target branch
            self._switch_to_branch(target_branch)
            
            # Try a dry-run merge to detect conflicts
            result = subprocess.run(
                ["git", "merge", "--no-commit", "--no-ff", source_branch],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            conflicts = {
                "conflicted_files": [],
                "merge_possible": result.returncode == 0,
                "conflict_details": []
            }
            
            if result.returncode != 0:
                # Get conflicted files
                conflicts["conflicted_files"] = self.check_for_conflicts()
                
                # Get detailed conflict information
                for file_path in conflicts["conflicted_files"]:
                    conflict_info = self._analyze_file_conflicts(file_path)
                    conflicts["conflict_details"].append({
                        "file": file_path,
                        "conflicts": conflict_info
                    })
                
                # Abort the merge
                subprocess.run(
                    ["git", "merge", "--abort"],
                    cwd=self.repo_path,
                    capture_output=True
                )
            
            # Return to original branch
            if current_branch:
                self._switch_to_branch(current_branch)
            
            return conflicts
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to detect merge conflicts: {e}")
            return {"conflicted_files": [], "merge_possible": False, "conflict_details": []}
    
    def _analyze_file_conflicts(self, file_path: str) -> List[Dict[str, str]]:
        """
        Analyze conflicts in a specific file.
        
        Args:
            file_path: Path to the conflicted file
            
        Returns:
            List of conflict information
        """
        try:
            file_full_path = self.repo_path / file_path
            if not file_full_path.exists():
                return []
            
            content = file_full_path.read_text()
            conflicts = []
            
            # Parse conflict markers
            lines = content.split('\n')
            in_conflict = False
            current_conflict = {}
            
            for i, line in enumerate(lines):
                if line.startswith('<<<<<<<'):
                    in_conflict = True
                    current_conflict = {
                        "start_line": i + 1,
                        "head_content": [],
                        "incoming_content": [],
                        "current_section": "head"
                    }
                elif line.startswith('=======') and in_conflict:
                    current_conflict["current_section"] = "incoming"
                elif line.startswith('>>>>>>>') and in_conflict:
                    current_conflict["end_line"] = i + 1
                    conflicts.append(current_conflict)
                    in_conflict = False
                elif in_conflict:
                    if current_conflict["current_section"] == "head":
                        current_conflict["head_content"].append(line)
                    else:
                        current_conflict["incoming_content"].append(line)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Failed to analyze conflicts in {file_path}: {e}")
            return []
    
    def resolve_conflict_assistance(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Provide assistance for resolving merge conflicts in a file.
        
        Args:
            file_path: Path to the conflicted file
            
        Returns:
            Dictionary with resolution suggestions
        """
        conflicts = self._analyze_file_conflicts(file_path)
        
        suggestions = {
            "file": file_path,
            "total_conflicts": len(conflicts),
            "suggestions": [],
            "auto_resolvable": False
        }
        
        for i, conflict in enumerate(conflicts):
            head_content = conflict.get("head_content", [])
            incoming_content = conflict.get("incoming_content", [])
            
            suggestion = {
                "conflict_number": i + 1,
                "location": f"Lines {conflict.get('start_line', 0)}-{conflict.get('end_line', 0)}",
                "head_changes": head_content,
                "incoming_changes": incoming_content,
                "resolution_options": []
            }
            
            # Analyze conflict type and provide suggestions
            if not head_content:
                suggestion["resolution_options"].append("Accept incoming changes (file was added)")
                suggestion["recommended"] = "incoming"
            elif not incoming_content:
                suggestion["resolution_options"].append("Keep current changes (file was deleted)")
                suggestion["recommended"] = "head"
            elif head_content == incoming_content:
                suggestion["resolution_options"].append("No actual conflict (identical changes)")
                suggestion["recommended"] = "either"
                suggestions["auto_resolvable"] = True
            else:
                suggestion["resolution_options"].extend([
                    "Keep current changes",
                    "Accept incoming changes",
                    "Merge both changes",
                    "Manual resolution required"
                ])
                suggestion["recommended"] = "manual"
            
            suggestions["suggestions"].append(suggestion)
        
        return suggestions
    
    def auto_resolve_simple_conflicts(self, file_path: str) -> bool:
        """
        Automatically resolve simple conflicts that don't require manual intervention.
        
        Args:
            file_path: Path to the conflicted file
            
        Returns:
            True if conflicts were resolved automatically
        """
        try:
            assistance = self.resolve_conflict_assistance(file_path)
            
            if not assistance["auto_resolvable"]:
                return False
            
            file_full_path = self.repo_path / file_path
            content = file_full_path.read_text()
            
            # Remove conflict markers for identical changes
            lines = content.split('\n')
            resolved_lines = []
            skip_until_end = False
            
            for line in lines:
                if line.startswith('<<<<<<<'):
                    skip_until_end = True
                    continue
                elif line.startswith('=======') and skip_until_end:
                    continue
                elif line.startswith('>>>>>>>') and skip_until_end:
                    skip_until_end = False
                    continue
                elif not skip_until_end:
                    resolved_lines.append(line)
            
            # Write resolved content
            file_full_path.write_text('\n'.join(resolved_lines))
            
            # Stage the resolved file
            subprocess.run(
                ["git", "add", file_path],
                cwd=self.repo_path,
                check=True
            )
            
            logger.info(f"Auto-resolved conflicts in {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to auto-resolve conflicts in {file_path}: {e}")
            return False
    
    def start_task_workflow(self, task_id: str, task_name: str, 
                           description: str = "", branch_type: BranchType = BranchType.TASK,
                           parent_branch: str = "main") -> str:
        """
        Start a complete task workflow with branch creation and tracking.
        
        Args:
            task_id: Unique identifier for the task
            task_name: Human-readable name of the task
            description: Task description
            branch_type: Type of branch to create
            parent_branch: Parent branch to branch from
            
        Returns:
            Name of the created branch
        """
        try:
            # Create task branch
            branch_name = self.create_task_branch(task_id, task_name, branch_type, parent_branch)
            
            # Create initial task context
            task_context = TaskContext(
                task_id=task_id,
                task_name=task_name,
                description=description,
                files_modified=[],
                requirements_addressed=[],
                completion_time=datetime.datetime.now(),
                branch_name=branch_name,
                status=TaskStatus.IN_PROGRESS,
                branch_type=branch_type,
                parent_branch=parent_branch,
                started_at=datetime.datetime.now()
            )
            
            # Track task start
            self._track_task_start(task_context)
            
            logger.info(f"Started task workflow for {task_id} on branch {branch_name}")
            return branch_name
            
        except Exception as e:
            logger.error(f"Failed to start task workflow: {e}")
            raise
    
    def _track_task_start(self, task_context: TaskContext) -> None:
        """
        Track task start in the tracking system.
        
        Args:
            task_context: Task context information
        """
        tracking_data = self._load_tracking_data()
        
        # Add to active tasks
        if "active_tasks" not in tracking_data:
            tracking_data["active_tasks"] = {}
            
        tracking_data["active_tasks"][task_context.task_id] = {
            "task_id": task_context.task_id,
            "task_name": task_context.task_name,
            "description": task_context.description,
            "branch_name": task_context.branch_name,
            "status": task_context.status.value,
            "branch_type": task_context.branch_type.value,
            "parent_branch": task_context.parent_branch,
            "started_at": task_context.started_at.isoformat() if task_context.started_at else None,
            "created_at": task_context.created_at.isoformat() if task_context.created_at else None
        }
        
        self._save_tracking_data(tracking_data)
    
    def get_task_branch_info(self, task_id: str) -> Optional[Dict]:
        """
        Get branch information for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Branch information dictionary or None if not found
        """
        tracking_data = self._load_tracking_data()
        
        # Check active tasks
        if "active_tasks" in tracking_data and task_id in tracking_data["active_tasks"]:
            return tracking_data["active_tasks"][task_id]
        
        # Check completed tasks
        for task in tracking_data.get("completed_tasks", []):
            if task["task_id"] == task_id:
                return task
        
        return None
    
    def list_active_task_branches(self) -> List[Dict]:
        """
        List all active task branches.
        
        Returns:
            List of active task branch information
        """
        tracking_data = self._load_tracking_data()
        return list(tracking_data.get("active_tasks", {}).values())
    
    def prepare_branch_for_merge(self, task_id: str, target_branch: str = "main") -> bool:
        """
        Prepare a task branch for merge by ensuring it's up to date and clean.
        
        Args:
            task_id: Task identifier
            target_branch: Target branch for merge
            
        Returns:
            True if branch is ready for merge
        """
        try:
            task_info = self.get_task_branch_info(task_id)
            if not task_info:
                logger.error(f"Task {task_id} not found")
                return False
            
            branch_name = task_info.get("branch_name")
            if not branch_name:
                logger.error(f"No branch found for task {task_id}")
                return False
            
            # Switch to task branch
            self._switch_to_branch(branch_name)
            
            # Ensure branch is clean
            if self._has_uncommitted_changes():
                logger.error(f"Branch {branch_name} has uncommitted changes")
                return False
            
            # Try to rebase on target branch
            try:
                self._ensure_branch_updated(target_branch)
                subprocess.run(
                    ["git", "rebase", target_branch],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )
                logger.info(f"Successfully rebased {branch_name} on {target_branch}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Rebase failed, branch may have conflicts: {e}")
                return False
            
            # Mark as merge ready
            tracking_data = self._load_tracking_data()
            if "active_tasks" in tracking_data and task_id in tracking_data["active_tasks"]:
                tracking_data["active_tasks"][task_id]["merge_ready"] = True
                tracking_data["active_tasks"][task_id]["merge_target"] = target_branch
                self._save_tracking_data(tracking_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare branch for merge: {e}")
            return False
    
    def create_hotfix_branch(self, hotfix_id: str, description: str, 
                            base_branch: str = "main") -> str:
        """
        Create a hotfix branch for emergency changes.
        
        Args:
            hotfix_id: Unique identifier for the hotfix
            description: Description of the hotfix
            base_branch: Base branch to create hotfix from
            
        Returns:
            Name of the created hotfix branch
        """
        try:
            # Generate hotfix branch name
            sanitized_desc = re.sub(r'[^a-zA-Z0-9\s-]', '', description.lower())
            sanitized_desc = re.sub(r'\s+', '-', sanitized_desc.strip())[:30]
            branch_name = f"hotfix/{hotfix_id}-{sanitized_desc}"
            
            # Ensure base branch is up to date
            self._ensure_branch_updated(base_branch)
            
            # Create hotfix branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name, base_branch],
                cwd=self.repo_path,
                check=True,
                capture_output=True
            )
            
            # Track hotfix creation
            self._track_hotfix_creation(hotfix_id, branch_name, description, base_branch)
            
            logger.info(f"Created hotfix branch: {branch_name}")
            return branch_name
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create hotfix branch: {e}")
    
    def _track_hotfix_creation(self, hotfix_id: str, branch_name: str, 
                              description: str, base_branch: str) -> None:
        """
        Track hotfix creation in the tracking system.
        
        Args:
            hotfix_id: Hotfix identifier
            branch_name: Name of hotfix branch
            description: Hotfix description
            base_branch: Base branch for hotfix
        """
        tracking_data = self._load_tracking_data()
        
        if "hotfixes" not in tracking_data:
            tracking_data["hotfixes"] = {}
        
        tracking_data["hotfixes"][hotfix_id] = {
            "hotfix_id": hotfix_id,
            "branch_name": branch_name,
            "description": description,
            "base_branch": base_branch,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "in_progress",
            "commits": [],
            "deployed": False
        }
        
        self._save_tracking_data(tracking_data)
    
    def complete_hotfix(self, hotfix_id: str, commit_message: str = None) -> Dict[str, str]:
        """
        Complete a hotfix and prepare it for deployment.
        
        Args:
            hotfix_id: Hotfix identifier
            commit_message: Optional commit message
            
        Returns:
            Dictionary with completion information
        """
        try:
            tracking_data = self._load_tracking_data()
            
            if "hotfixes" not in tracking_data or hotfix_id not in tracking_data["hotfixes"]:
                raise RuntimeError(f"Hotfix {hotfix_id} not found")
            
            hotfix_info = tracking_data["hotfixes"][hotfix_id]
            branch_name = hotfix_info["branch_name"]
            base_branch = hotfix_info["base_branch"]
            
            # Switch to hotfix branch
            self._switch_to_branch(branch_name)
            
            # Commit any remaining changes
            if self._has_uncommitted_changes():
                if not commit_message:
                    commit_message = f"Complete hotfix {hotfix_id}: {hotfix_info['description']}"
                
                subprocess.run(
                    ["git", "add", "."],
                    cwd=self.repo_path,
                    check=True
                )
                
                subprocess.run(
                    ["git", "commit", "-m", commit_message],
                    cwd=self.repo_path,
                    check=True
                )
            
            # Get final commit hash
            final_commit = self._get_current_commit_hash()
            
            # Update tracking
            hotfix_info["status"] = "completed"
            hotfix_info["completed_at"] = datetime.datetime.now().isoformat()
            hotfix_info["final_commit"] = final_commit
            
            self._save_tracking_data(tracking_data)
            
            # Generate deployment information
            deployment_info = {
                "hotfix_id": hotfix_id,
                "branch_name": branch_name,
                "base_branch": base_branch,
                "final_commit": final_commit,
                "ready_for_merge": True,
                "merge_command": f"git checkout {base_branch} && git merge --no-ff {branch_name}",
                "tag_suggestion": f"hotfix-{hotfix_id}-{datetime.datetime.now().strftime('%Y%m%d')}"
            }
            
            logger.info(f"Hotfix {hotfix_id} completed and ready for deployment")
            return deployment_info
            
        except Exception as e:
            logger.error(f"Failed to complete hotfix: {e}")
            raise
    
    def deploy_hotfix(self, hotfix_id: str, create_tag: bool = True) -> bool:
        """
        Deploy a completed hotfix by merging to base branch.
        
        Args:
            hotfix_id: Hotfix identifier
            create_tag: Whether to create a tag for the deployment
            
        Returns:
            True if deployment was successful
        """
        try:
            tracking_data = self._load_tracking_data()
            
            if "hotfixes" not in tracking_data or hotfix_id not in tracking_data["hotfixes"]:
                raise RuntimeError(f"Hotfix {hotfix_id} not found")
            
            hotfix_info = tracking_data["hotfixes"][hotfix_id]
            
            if hotfix_info["status"] != "completed":
                raise RuntimeError(f"Hotfix {hotfix_id} is not completed")
            
            branch_name = hotfix_info["branch_name"]
            base_branch = hotfix_info["base_branch"]
            
            # Switch to base branch
            self._switch_to_branch(base_branch)
            
            # Pull latest changes
            try:
                subprocess.run(
                    ["git", "pull", "origin", base_branch],
                    cwd=self.repo_path,
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError:
                logger.warning("Could not pull latest changes from remote")
            
            # Merge hotfix branch
            merge_message = f"Merge hotfix {hotfix_id}: {hotfix_info['description']}"
            subprocess.run(
                ["git", "merge", "--no-ff", "-m", merge_message, branch_name],
                cwd=self.repo_path,
                check=True
            )
            
            # Create tag if requested
            if create_tag:
                tag_name = f"hotfix-{hotfix_id}-{datetime.datetime.now().strftime('%Y%m%d')}"
                subprocess.run(
                    ["git", "tag", "-a", tag_name, "-m", f"Hotfix {hotfix_id} deployment"],
                    cwd=self.repo_path,
                    check=True
                )
                logger.info(f"Created deployment tag: {tag_name}")
            
            # Update tracking
            hotfix_info["deployed"] = True
            hotfix_info["deployed_at"] = datetime.datetime.now().isoformat()
            hotfix_info["merge_commit"] = self._get_current_commit_hash()
            
            self._save_tracking_data(tracking_data)
            
            # Push changes
            try:
                subprocess.run(
                    ["git", "push", "origin", base_branch],
                    cwd=self.repo_path,
                    check=True
                )
                if create_tag:
                    subprocess.run(
                        ["git", "push", "origin", "--tags"],
                        cwd=self.repo_path,
                        check=True
                    )
            except subprocess.CalledProcessError:
                logger.warning("Could not push changes to remote")
            
            logger.info(f"Successfully deployed hotfix {hotfix_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy hotfix: {e}")
            return False
    
    def list_active_hotfixes(self) -> List[Dict]:
        """
        List all active hotfixes.
        
        Returns:
            List of active hotfix information
        """
        tracking_data = self._load_tracking_data()
        hotfixes = tracking_data.get("hotfixes", {})
        
        active_hotfixes = []
        for hotfix_id, hotfix_info in hotfixes.items():
            if hotfix_info["status"] in ["in_progress", "completed"] and not hotfix_info.get("deployed", False):
                active_hotfixes.append(hotfix_info)
        
        return active_hotfixes
    
    def get_hotfix_status(self, hotfix_id: str) -> Optional[Dict]:
        """
        Get status information for a specific hotfix.
        
        Args:
            hotfix_id: Hotfix identifier
            
        Returns:
            Hotfix status information or None if not found
        """
        tracking_data = self._load_tracking_data()
        return tracking_data.get("hotfixes", {}).get(hotfix_id)
    
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
            completion_time=datetime.datetime.now(),
            status=TaskStatus.COMPLETED
        )
        
        # Commit and push
        commit_hash = self.commit_task_completion(task_context)
        
        print(f"✅ Task {task_id} completed successfully!")
        print(f"📝 Commit: {commit_hash[:8]}")
        print(f"📁 Files modified: {len(modified_files)}")
        print(f"🚀 Changes pushed to repository")
        
        return commit_hash
    
    def generate_pull_request_description(self, task_id: str, source_branch: str, 
                                         target_branch: str = "main") -> Dict[str, str]:
        """
        Generate comprehensive pull request description.
        
        Args:
            task_id: Task identifier
            source_branch: Source branch for the PR
            target_branch: Target branch for the PR
            
        Returns:
            Dictionary with PR title and description
        """
        try:
            # Get task information
            task_info = self.get_task_branch_info(task_id)
            if not task_info:
                task_info = {"task_name": f"Task {task_id}", "description": ""}
            
            # Get commit history for the branch
            commits = self._get_branch_commits(source_branch, target_branch)
            
            # Get file changes
            changed_files = self._get_branch_file_changes(source_branch, target_branch)
            
            # Generate title
            title = f"{task_info.get('task_name', f'Task {task_id}')}"
            
            # Generate description
            description_parts = []
            
            # Task overview
            description_parts.append("## Overview")
            if task_info.get("description"):
                description_parts.append(task_info["description"])
            else:
                description_parts.append(f"Implementation of task {task_id}")
            
            # Changes summary
            description_parts.append("\n## Changes")
            if commits:
                description_parts.append("### Commits")
                for commit in commits[:10]:  # Limit to 10 commits
                    short_hash = commit["hash"][:8]
                    description_parts.append(f"- `{short_hash}` {commit['message'].split(chr(10))[0]}")
                
                if len(commits) > 10:
                    description_parts.append(f"... and {len(commits) - 10} more commits")
            
            # File changes
            if changed_files:
                description_parts.append("\n### Files Changed")
                for file_change in changed_files[:20]:  # Limit to 20 files
                    status = file_change.get("status", "M")
                    file_path = file_change.get("file", "")
                    if status == "A":
                        description_parts.append(f"- ➕ `{file_path}` (added)")
                    elif status == "D":
                        description_parts.append(f"- ➖ `{file_path}` (deleted)")
                    elif status == "M":
                        description_parts.append(f"- 📝 `{file_path}` (modified)")
                    else:
                        description_parts.append(f"- 🔄 `{file_path}` ({status})")
                
                if len(changed_files) > 20:
                    description_parts.append(f"... and {len(changed_files) - 20} more files")
            
            # Requirements addressed
            if task_info.get("requirements_addressed"):
                description_parts.append("\n## Requirements Addressed")
                for req in task_info["requirements_addressed"]:
                    description_parts.append(f"- {req}")
            
            # Testing notes
            description_parts.append("\n## Testing")
            description_parts.append("- [ ] Unit tests pass")
            description_parts.append("- [ ] Integration tests pass")
            description_parts.append("- [ ] Manual testing completed")
            
            # Checklist
            description_parts.append("\n## Checklist")
            description_parts.append("- [ ] Code follows project style guidelines")
            description_parts.append("- [ ] Self-review of code completed")
            description_parts.append("- [ ] Documentation updated if needed")
            description_parts.append("- [ ] No breaking changes introduced")
            
            return {
                "title": title,
                "description": "\n".join(description_parts)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate PR description: {e}")
            return {
                "title": f"Task {task_id}",
                "description": f"Pull request for task {task_id}"
            }
    
    def _get_branch_commits(self, source_branch: str, target_branch: str) -> List[Dict[str, str]]:
        """
        Get commits that are in source branch but not in target branch.
        
        Args:
            source_branch: Source branch
            target_branch: Target branch
            
        Returns:
            List of commit information
        """
        try:
            result = subprocess.run(
                ["git", "log", f"{target_branch}..{source_branch}", "--pretty=format:%H|%s|%an|%ad", "--date=short"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|', 3)
                    if len(parts) >= 4:
                        commits.append({
                            "hash": parts[0],
                            "message": parts[1],
                            "author": parts[2],
                            "date": parts[3]
                        })
            
            return commits
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get branch commits: {e}")
            return []
    
    def _get_branch_file_changes(self, source_branch: str, target_branch: str) -> List[Dict[str, str]]:
        """
        Get file changes between two branches.
        
        Args:
            source_branch: Source branch
            target_branch: Target branch
            
        Returns:
            List of file change information
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-status", f"{target_branch}..{source_branch}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            changes = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t', 1)
                    if len(parts) >= 2:
                        changes.append({
                            "status": parts[0],
                            "file": parts[1]
                        })
            
            return changes
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get file changes: {e}")
            return []
    
    def create_pull_request_template(self, task_id: str, output_file: str = None) -> str:
        """
        Create a pull request template file.
        
        Args:
            task_id: Task identifier
            output_file: Output file path (optional)
            
        Returns:
            Path to the created template file
        """
        current_branch = self._get_current_branch()
        if not current_branch:
            raise RuntimeError("Could not determine current branch")
        
        pr_info = self.generate_pull_request_description(task_id, current_branch)
        
        if not output_file:
            output_file = f"pr-template-{task_id}.md"
        
        template_path = self.repo_path / output_file
        
        template_content = f"""# {pr_info['title']}

{pr_info['description']}

---

**Branch:** `{current_branch}` → `main`
**Task ID:** {task_id}
**Generated:** {datetime.datetime.now().isoformat()}
"""
        
        template_path.write_text(template_content)
        logger.info(f"Created PR template: {template_path}")
        
        return str(template_path)


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
