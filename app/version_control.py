"""
Version Control Integration for Safe Code Modifications

This module provides Git integration for automatic backup creation,
rollback point management, and change tracking for the self-improvement system.
"""

import os
import subprocess
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import hashlib

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class ChangeRecord(BaseModel):
    """Record of a code change for audit logging"""
    id: str
    timestamp: datetime
    branch_name: str
    commit_hash: str
    files_modified: List[str]
    description: str
    improvement_id: Optional[str] = None
    rollback_point: str
    status: str  # 'applied', 'rolled_back', 'testing'


class GitIntegration:
    """Git integration for safe code modifications with rollback capabilities"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.audit_log_path = self.repo_path / ".kiro" / "audit" / "changes.json"
        self.ensure_audit_directory()
        
    def ensure_audit_directory(self) -> None:
        """Ensure audit directory exists"""
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.audit_log_path.exists():
            self.audit_log_path.write_text("[]")
    
    def is_git_repo(self) -> bool:
        """Check if current directory is a git repository"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def init_repo_if_needed(self) -> bool:
        """Initialize git repository if it doesn't exist"""
        if not self.is_git_repo():
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info("Initialized new git repository")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to initialize git repository: {e}")
                return False
        return True
    
    def get_current_commit_hash(self) -> Optional[str]:
        """Get current commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def get_current_branch(self) -> Optional[str]:
        """Get current branch name"""
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
    
    def create_rollback_point(self, description: str) -> Optional[str]:
        """Create a rollback point by committing current state"""
        if not self.init_repo_if_needed():
            return None
            
        try:
            # Add all files to staging
            subprocess.run(
                ["git", "add", "."],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Create commit
            commit_message = f"Rollback point: {description}"
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commit_hash = self.get_current_commit_hash()
            logger.info(f"Created rollback point: {commit_hash}")
            return commit_hash
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create rollback point: {e}")
            return None
    
    def create_improvement_branch(self, improvement_id: str) -> Optional[str]:
        """Create a new branch for testing improvements"""
        if not self.init_repo_if_needed():
            return None
            
        branch_name = f"improvement-{improvement_id}-{int(datetime.now().timestamp())}"
        
        try:
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Created improvement branch: {branch_name}")
            return branch_name
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create improvement branch: {e}")
            return None
    
    def switch_to_branch(self, branch_name: str) -> bool:
        """Switch to specified branch"""
        try:
            subprocess.run(
                ["git", "checkout", branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Switched to branch: {branch_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to switch to branch {branch_name}: {e}")
            return False
    
    def commit_changes(self, files: List[str], message: str) -> Optional[str]:
        """Commit specific files with a message"""
        try:
            # Add specific files
            for file_path in files:
                subprocess.run(
                    ["git", "add", file_path],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            # Commit changes
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commit_hash = self.get_current_commit_hash()
            logger.info(f"Committed changes: {commit_hash}")
            return commit_hash
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e}")
            return None
    
    def rollback_to_commit(self, commit_hash: str) -> bool:
        """Rollback to a specific commit"""
        try:
            subprocess.run(
                ["git", "reset", "--hard", commit_hash],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Rolled back to commit: {commit_hash}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to rollback to commit {commit_hash}: {e}")
            return False
    
    def get_file_changes(self, commit1: str, commit2: str) -> List[str]:
        """Get list of files changed between two commits"""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", commit1, commit2],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return [f.strip() for f in result.stdout.split('\n') if f.strip()]
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get file changes: {e}")
            return []
    
    def get_uncommitted_changes(self) -> List[str]:
        """Get list of files with uncommitted changes"""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            modified = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            
            # Also check staged files
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            staged = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            
            return list(set(modified + staged))
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get uncommitted changes: {e}")
            return []
    
    def log_change(self, change_record: ChangeRecord) -> None:
        """Log a change to the audit log"""
        try:
            # Read existing log
            if self.audit_log_path.exists():
                with open(self.audit_log_path, 'r') as f:
                    changes = json.load(f)
            else:
                changes = []
            
            # Add new change
            changes.append(change_record.model_dump())
            
            # Write back to file
            with open(self.audit_log_path, 'w') as f:
                json.dump(changes, f, indent=2, default=str)
                
            logger.info(f"Logged change: {change_record.id}")
            
        except Exception as e:
            logger.error(f"Failed to log change: {e}")
    
    def get_change_history(self, limit: Optional[int] = None) -> List[ChangeRecord]:
        """Get change history from audit log"""
        try:
            if not self.audit_log_path.exists():
                return []
                
            with open(self.audit_log_path, 'r') as f:
                changes_data = json.load(f)
            
            changes = []
            for change_data in changes_data:
                # Convert timestamp string back to datetime
                if isinstance(change_data['timestamp'], str):
                    change_data['timestamp'] = datetime.fromisoformat(change_data['timestamp'])
                changes.append(ChangeRecord(**change_data))
            
            # Sort by timestamp, most recent first
            changes.sort(key=lambda x: x.timestamp, reverse=True)
            
            if limit:
                changes = changes[:limit]
                
            return changes
            
        except Exception as e:
            logger.error(f"Failed to get change history: {e}")
            return []
    
    def generate_change_id(self, improvement_id: str, files: List[str]) -> str:
        """Generate unique change ID"""
        content = f"{improvement_id}:{':'.join(sorted(files))}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def cleanup_old_branches(self, keep_days: int = 7) -> None:
        """Clean up old improvement branches"""
        try:
            # Get all branches
            result = subprocess.run(
                ["git", "branch", "--list", "improvement-*"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            branches = [b.strip().replace('* ', '') for b in result.stdout.split('\n') if b.strip()]
            current_time = datetime.now().timestamp()
            
            for branch in branches:
                # Extract timestamp from branch name
                try:
                    parts = branch.split('-')
                    if len(parts) >= 3:
                        branch_timestamp = int(parts[-1])
                        age_days = (current_time - branch_timestamp) / (24 * 3600)
                        
                        if age_days > keep_days:
                            subprocess.run(
                                ["git", "branch", "-D", branch],
                                cwd=self.repo_path,
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            logger.info(f"Cleaned up old branch: {branch}")
                            
                except (ValueError, IndexError):
                    continue
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cleanup old branches: {e}")