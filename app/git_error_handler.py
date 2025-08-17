# app/git_error_handler.py
"""
Git Operation Error Handling and Recovery System

This module provides comprehensive error handling and recovery mechanisms
for Git operations including merge conflict resolution, operation rollback,
and repository corruption detection and recovery.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .git_workflow_manager import GitWorkflowManager, TaskContext, TaskStatus, BranchType
from .version_control import GitIntegration, ChangeRecord

logger = logging.getLogger(__name__)


class GitErrorType(Enum):
    """Types of Git errors"""
    MERGE_CONFLICT = "merge_conflict"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    REPOSITORY_CORRUPTION = "repository_corruption"
    PERMISSION = "permission"
    DISK_SPACE = "disk_space"
    INVALID_OPERATION = "invalid_operation"
    BRANCH_NOT_FOUND = "branch_not_found"
    COMMIT_FAILED = "commit_failed"
    PUSH_FAILED = "push_failed"
    PULL_FAILED = "pull_failed"
    CHECKOUT_FAILED = "checkout_failed"
    UNKNOWN = "unknown"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving merge conflicts"""
    MANUAL = "manual"
    OURS = "ours"
    THEIRS = "theirs"
    AUTO_MERGE = "auto_merge"
    ABORT = "abort"
    INTERACTIVE = "interactive"


class RecoveryAction(Enum):
    """Recovery actions for Git errors"""
    RETRY = "retry"
    ROLLBACK = "rollback"
    RESET_HARD = "reset_hard"
    CLEAN_WORKSPACE = "clean_workspace"
    RECREATE_BRANCH = "recreate_branch"
    CLONE_FRESH = "clone_fresh"
    MANUAL_INTERVENTION = "manual_intervention"
    IGNORE = "ignore"


@dataclass
class GitError:
    """Comprehensive Git error information"""
    error_id: str
    timestamp: datetime
    error_type: GitErrorType
    command: str
    exit_code: int
    stdout: str
    stderr: str
    working_directory: str
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    affected_files: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_notes: str = ""


@dataclass
class MergeConflict:
    """Information about a merge conflict"""
    file_path: str
    conflict_markers: List[str]
    our_content: str
    their_content: str
    base_content: Optional[str] = None
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolved: bool = False
    resolution_content: Optional[str] = None


@dataclass
class RepositoryBackup:
    """Information about a repository backup"""
    backup_id: str
    timestamp: datetime
    backup_path: str
    original_path: str
    branch: str
    commit_hash: str
    description: str
    size_bytes: int


class GitErrorHandler:
    """
    Comprehensive Git error handling and recovery system
    
    Features:
    - Merge conflict resolution assistance and automation
    - Git operation rollback and recovery mechanisms
    - Repository corruption detection and recovery
    - Automatic retry with exponential backoff
    - Repository backup and restore
    """
    
    def __init__(self, 
                 git_manager: GitWorkflowManager,
                 git_integration: GitIntegration = None,
                 backup_dir: str = ".kiro/backups"):
        self.git_manager = git_manager
        self.git_integration = git_integration or GitIntegration(git_manager.repo_path)
        self.repo_path = Path(git_manager.repo_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Error tracking
        self._errors: Dict[str, GitError] = {}
        self._conflict_history: List[MergeConflict] = []
        self._backups: Dict[str, RepositoryBackup] = {}
        
        # Configuration
        self.max_retry_attempts = 3
        self.retry_delay_base = 2.0  # seconds
        self.max_retry_delay = 60.0  # seconds
        self.backup_retention_days = 30
        self.auto_resolve_conflicts = True
        self.conflict_resolution_timeout = 300  # 5 minutes
        
        # Recovery handlers
        self._recovery_handlers: Dict[GitErrorType, List[RecoveryAction]] = {
            GitErrorType.MERGE_CONFLICT: [RecoveryAction.RETRY, RecoveryAction.ROLLBACK],
            GitErrorType.AUTHENTICATION: [RecoveryAction.RETRY, RecoveryAction.MANUAL_INTERVENTION],
            GitErrorType.NETWORK: [RecoveryAction.RETRY],
            GitErrorType.REPOSITORY_CORRUPTION: [RecoveryAction.CLONE_FRESH, RecoveryAction.ROLLBACK],
            GitErrorType.PERMISSION: [RecoveryAction.MANUAL_INTERVENTION],
            GitErrorType.DISK_SPACE: [RecoveryAction.CLEAN_WORKSPACE, RecoveryAction.MANUAL_INTERVENTION],
            GitErrorType.INVALID_OPERATION: [RecoveryAction.ROLLBACK, RecoveryAction.RESET_HARD],
            GitErrorType.BRANCH_NOT_FOUND: [RecoveryAction.RECREATE_BRANCH],
            GitErrorType.COMMIT_FAILED: [RecoveryAction.RETRY, RecoveryAction.RESET_HARD],
            GitErrorType.PUSH_FAILED: [RecoveryAction.RETRY, RecoveryAction.PULL_FAILED],
            GitErrorType.PULL_FAILED: [RecoveryAction.RETRY, RecoveryAction.RESET_HARD],
            GitErrorType.CHECKOUT_FAILED: [RecoveryAction.CLEAN_WORKSPACE, RecoveryAction.RESET_HARD]
        }
        
        # Statistics
        self.stats = {
            "total_errors": 0,
            "resolved_errors": 0,
            "merge_conflicts_resolved": 0,
            "automatic_recoveries": 0,
            "manual_interventions": 0,
            "repository_corruptions": 0,
            "backups_created": 0
        }
        
        logger.info("Git Error Handler initialized")
    
    async def handle_git_error(self, 
                              command: str, 
                              exit_code: int, 
                              stdout: str, 
                              stderr: str,
                              context: Dict[str, Any] = None) -> GitError:
        """
        Handle a Git error with comprehensive analysis and recovery
        
        Args:
            command: The Git command that failed
            exit_code: Exit code from the command
            stdout: Standard output from the command
            stderr: Standard error from the command
            context: Additional context information
            
        Returns:
            GitError object with error details and recovery actions
        """
        try:
            # Create error object
            git_error = self._create_git_error(command, exit_code, stdout, stderr, context)
            
            # Store error
            self._errors[git_error.error_id] = git_error
            self.stats["total_errors"] += 1
            
            # Log error
            logger.error(f"Git error [{git_error.error_id}]: {git_error.error_type.value} - {stderr[:200]}")
            
            # Determine recovery actions
            recovery_actions = self._determine_recovery_actions(git_error)
            git_error.recovery_actions = recovery_actions
            
            # Execute recovery actions
            success = await self._execute_recovery_actions(git_error)
            
            if success:
                git_error.resolved = True
                git_error.resolution_time = datetime.now()
                self.stats["resolved_errors"] += 1
                self.stats["automatic_recoveries"] += 1
                logger.info(f"Successfully recovered from Git error {git_error.error_id}")
            else:
                logger.warning(f"Could not automatically recover from Git error {git_error.error_id}")
                self.stats["manual_interventions"] += 1
            
            return git_error
            
        except Exception as e:
            logger.error(f"Error in Git error handler: {e}")
            # Return basic error info even if handling fails
            return GitError(
                error_id=f"git_error_handler_failure_{int(time.time())}",
                timestamp=datetime.now(),
                error_type=GitErrorType.UNKNOWN,
                command=command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                working_directory=str(self.repo_path)
            )
    
    async def handle_merge_conflict(self, 
                                  conflicted_files: List[str],
                                  strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.AUTO_MERGE) -> bool:
        """
        Handle merge conflicts with specified resolution strategy
        
        Args:
            conflicted_files: List of files with conflicts
            strategy: Resolution strategy to use
            
        Returns:
            True if conflicts were resolved successfully
        """
        try:
            logger.info(f"Handling merge conflicts in {len(conflicted_files)} files with strategy: {strategy.value}")
            
            # Create backup before attempting resolution
            backup_id = await self._create_repository_backup("pre_conflict_resolution")
            
            conflicts = []
            for file_path in conflicted_files:
                conflict = await self._analyze_conflict(file_path)
                if conflict:
                    conflicts.append(conflict)
            
            # Apply resolution strategy
            resolved_count = 0
            for conflict in conflicts:
                if await self._resolve_conflict(conflict, strategy):
                    resolved_count += 1
            
            if resolved_count == len(conflicts):
                # All conflicts resolved, commit the resolution
                success = await self._commit_conflict_resolution(conflicts)
                if success:
                    self.stats["merge_conflicts_resolved"] += len(conflicts)
                    logger.info(f"Successfully resolved {resolved_count} merge conflicts")
                    return True
            
            # Partial or failed resolution, rollback
            if backup_id:
                await self._restore_from_backup(backup_id)
                logger.warning(f"Rolled back due to incomplete conflict resolution ({resolved_count}/{len(conflicts)})")
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling merge conflicts: {e}")
            return False
    
    async def detect_repository_corruption(self) -> Dict[str, Any]:
        """
        Detect repository corruption and integrity issues
        
        Returns:
            Dictionary with corruption detection results
        """
        try:
            corruption_issues = {
                "corrupted": False,
                "issues": [],
                "severity": "none",
                "recommended_actions": []
            }
            
            # Check Git object integrity
            try:
                result = await self._run_git_command(["git", "fsck", "--full"])
                if result["exit_code"] != 0:
                    corruption_issues["corrupted"] = True
                    corruption_issues["issues"].append("Git object corruption detected")
                    corruption_issues["severity"] = "high"
            except Exception as e:
                corruption_issues["issues"].append(f"Could not run fsck: {e}")
            
            # Check for missing objects
            try:
                result = await self._run_git_command(["git", "rev-list", "--objects", "--all"])
                if result["exit_code"] != 0:
                    corruption_issues["corrupted"] = True
                    corruption_issues["issues"].append("Missing Git objects detected")
                    corruption_issues["severity"] = "high"
            except Exception as e:
                corruption_issues["issues"].append(f"Could not check objects: {e}")
            
            # Check index integrity
            try:
                result = await self._run_git_command(["git", "status", "--porcelain"])
                if "fatal" in result["stderr"].lower():
                    corruption_issues["corrupted"] = True
                    corruption_issues["issues"].append("Git index corruption detected")
                    corruption_issues["severity"] = "medium"
            except Exception as e:
                corruption_issues["issues"].append(f"Could not check index: {e}")
            
            # Check for unreachable objects
            try:
                result = await self._run_git_command(["git", "fsck", "--unreachable"])
                unreachable_count = result["stdout"].count("unreachable")
                if unreachable_count > 100:  # Threshold for concern
                    corruption_issues["issues"].append(f"High number of unreachable objects: {unreachable_count}")
                    corruption_issues["severity"] = "low"
            except Exception:
                pass
            
            # Determine recommended actions
            if corruption_issues["corrupted"]:
                if corruption_issues["severity"] == "high":
                    corruption_issues["recommended_actions"] = [
                        RecoveryAction.CLONE_FRESH.value,
                        RecoveryAction.ROLLBACK.value
                    ]
                elif corruption_issues["severity"] == "medium":
                    corruption_issues["recommended_actions"] = [
                        RecoveryAction.RESET_HARD.value,
                        RecoveryAction.CLEAN_WORKSPACE.value
                    ]
                else:
                    corruption_issues["recommended_actions"] = [
                        RecoveryAction.CLEAN_WORKSPACE.value
                    ]
                
                self.stats["repository_corruptions"] += 1
            
            logger.info(f"Repository corruption check completed: {corruption_issues['severity']} severity")
            return corruption_issues
            
        except Exception as e:
            logger.error(f"Error detecting repository corruption: {e}")
            return {
                "corrupted": False,
                "issues": [f"Corruption detection failed: {e}"],
                "severity": "unknown",
                "recommended_actions": [RecoveryAction.MANUAL_INTERVENTION.value]
            }
    
    async def rollback_operation(self, 
                               operation_id: str, 
                               rollback_point: str = None) -> bool:
        """
        Rollback a Git operation to a previous state
        
        Args:
            operation_id: Identifier for the operation to rollback
            rollback_point: Specific commit hash to rollback to
            
        Returns:
            True if rollback was successful
        """
        try:
            # Create backup before rollback
            backup_id = await self._create_repository_backup(f"pre_rollback_{operation_id}")
            
            if rollback_point:
                # Rollback to specific commit
                success = await self._rollback_to_commit(rollback_point)
            else:
                # Find appropriate rollback point
                rollback_point = await self._find_safe_rollback_point()
                if rollback_point:
                    success = await self._rollback_to_commit(rollback_point)
                else:
                    logger.error("Could not find safe rollback point")
                    return False
            
            if success:
                logger.info(f"Successfully rolled back operation {operation_id} to {rollback_point}")
                return True
            else:
                # Restore from backup if rollback failed
                if backup_id:
                    await self._restore_from_backup(backup_id)
                return False
            
        except Exception as e:
            logger.error(f"Error rolling back operation {operation_id}: {e}")
            return False
    
    async def recover_from_corruption(self, corruption_info: Dict[str, Any]) -> bool:
        """
        Recover from repository corruption
        
        Args:
            corruption_info: Corruption detection results
            
        Returns:
            True if recovery was successful
        """
        try:
            severity = corruption_info.get("severity", "unknown")
            
            if severity == "high":
                # Severe corruption - try to clone fresh
                return await self._recover_via_fresh_clone()
            
            elif severity == "medium":
                # Medium corruption - try reset and cleanup
                return await self._recover_via_reset_cleanup()
            
            elif severity == "low":
                # Low severity - try cleanup only
                return await self._recover_via_cleanup()
            
            else:
                logger.warning("Unknown corruption severity, attempting manual intervention")
                return False
            
        except Exception as e:
            logger.error(f"Error recovering from corruption: {e}")
            return False
    
    def _create_git_error(self, 
                         command: str, 
                         exit_code: int, 
                         stdout: str, 
                         stderr: str,
                         context: Dict[str, Any] = None) -> GitError:
        """Create a GitError object from command results"""
        error_id = f"git_error_{int(time.time() * 1000)}"
        
        # Determine error type from stderr
        error_type = self._classify_git_error(stderr, exit_code)
        
        # Get current Git state
        current_branch = self._get_current_branch()
        current_commit = self._get_current_commit()
        
        # Extract affected files if available
        affected_files = self._extract_affected_files(stdout, stderr)
        
        return GitError(
            error_id=error_id,
            timestamp=datetime.now(),
            error_type=error_type,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            working_directory=str(self.repo_path),
            branch=current_branch,
            commit_hash=current_commit,
            affected_files=affected_files,
            context=context or {}
        )
    
    def _classify_git_error(self, stderr: str, exit_code: int) -> GitErrorType:
        """Classify Git error based on stderr output"""
        stderr_lower = stderr.lower()
        
        # Merge conflict patterns
        if any(pattern in stderr_lower for pattern in [
            "merge conflict", "conflict", "automatic merge failed"
        ]):
            return GitErrorType.MERGE_CONFLICT
        
        # Authentication patterns
        if any(pattern in stderr_lower for pattern in [
            "authentication failed", "permission denied", "access denied",
            "could not read username", "could not read password"
        ]):
            return GitErrorType.AUTHENTICATION
        
        # Network patterns
        if any(pattern in stderr_lower for pattern in [
            "network", "connection", "timeout", "could not resolve host",
            "failed to connect", "operation timed out"
        ]):
            return GitErrorType.NETWORK
        
        # Repository corruption patterns
        if any(pattern in stderr_lower for pattern in [
            "corrupt", "bad object", "missing object", "invalid object",
            "fatal: not a git repository", "index file corrupt"
        ]):
            return GitErrorType.REPOSITORY_CORRUPTION
        
        # Permission patterns
        if any(pattern in stderr_lower for pattern in [
            "permission denied", "operation not permitted", "access denied"
        ]):
            return GitErrorType.PERMISSION
        
        # Disk space patterns
        if any(pattern in stderr_lower for pattern in [
            "no space left", "disk full", "not enough space"
        ]):
            return GitErrorType.DISK_SPACE
        
        # Branch not found patterns
        if any(pattern in stderr_lower for pattern in [
            "branch not found", "no such branch", "unknown revision"
        ]):
            return GitErrorType.BRANCH_NOT_FOUND
        
        # Command-specific patterns
        if "commit" in stderr_lower and any(pattern in stderr_lower for pattern in [
            "nothing to commit", "no changes added"
        ]):
            return GitErrorType.COMMIT_FAILED
        
        if "push" in stderr_lower:
            return GitErrorType.PUSH_FAILED
        
        if "pull" in stderr_lower:
            return GitErrorType.PULL_FAILED
        
        if "checkout" in stderr_lower:
            return GitErrorType.CHECKOUT_FAILED
        
        return GitErrorType.UNKNOWN
    
    def _determine_recovery_actions(self, git_error: GitError) -> List[RecoveryAction]:
        """Determine appropriate recovery actions for a Git error"""
        error_type = git_error.error_type
        
        # Get default recovery actions for error type
        actions = self._recovery_handlers.get(error_type, [RecoveryAction.MANUAL_INTERVENTION])
        
        # Customize based on specific error details
        if error_type == GitErrorType.MERGE_CONFLICT:
            if self.auto_resolve_conflicts:
                actions.insert(0, RecoveryAction.RETRY)  # Try auto-resolution first
        
        elif error_type == GitErrorType.NETWORK:
            # Add more retries for network issues
            actions = [RecoveryAction.RETRY] * 3 + actions
        
        elif error_type == GitErrorType.DISK_SPACE:
            # Clean workspace before manual intervention
            actions = [RecoveryAction.CLEAN_WORKSPACE, RecoveryAction.MANUAL_INTERVENTION]
        
        return actions
    
    async def _execute_recovery_actions(self, git_error: GitError) -> bool:
        """Execute recovery actions for a Git error"""
        try:
            for action in git_error.recovery_actions:
                logger.info(f"Executing recovery action: {action.value} for error {git_error.error_id}")
                
                success = await self._execute_single_recovery_action(git_error, action)
                
                if success:
                    git_error.resolution_notes += f"Recovered using {action.value}; "
                    return True
                else:
                    git_error.resolution_notes += f"Failed {action.value}; "
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing recovery actions: {e}")
            return False
    
    async def _execute_single_recovery_action(self, 
                                            git_error: GitError, 
                                            action: RecoveryAction) -> bool:
        """Execute a single recovery action"""
        try:
            if action == RecoveryAction.RETRY:
                return await self._retry_operation(git_error)
            
            elif action == RecoveryAction.ROLLBACK:
                return await self._rollback_operation(git_error)
            
            elif action == RecoveryAction.RESET_HARD:
                return await self._reset_hard_operation(git_error)
            
            elif action == RecoveryAction.CLEAN_WORKSPACE:
                return await self._clean_workspace_operation(git_error)
            
            elif action == RecoveryAction.RECREATE_BRANCH:
                return await self._recreate_branch_operation(git_error)
            
            elif action == RecoveryAction.CLONE_FRESH:
                return await self._clone_fresh_operation(git_error)
            
            elif action == RecoveryAction.MANUAL_INTERVENTION:
                logger.warning(f"Manual intervention required for error {git_error.error_id}")
                return False
            
            elif action == RecoveryAction.IGNORE:
                logger.info(f"Ignoring error {git_error.error_id}")
                return True
            
            else:
                logger.warning(f"Unknown recovery action: {action.value}")
                return False
            
        except Exception as e:
            logger.error(f"Error executing recovery action {action.value}: {e}")
            return False
    
    async def _retry_operation(self, git_error: GitError) -> bool:
        """Retry the failed Git operation with exponential backoff"""
        try:
            for attempt in range(self.max_retry_attempts):
                if attempt > 0:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.retry_delay_base * (2 ** (attempt - 1)),
                        self.max_retry_delay
                    )
                    logger.info(f"Retrying in {delay} seconds (attempt {attempt + 1}/{self.max_retry_attempts})")
                    await asyncio.sleep(delay)
                
                # Handle specific error types
                if git_error.error_type == GitErrorType.MERGE_CONFLICT:
                    # Try to resolve conflicts automatically
                    conflicted_files = self._extract_conflicted_files(git_error.stderr)
                    if conflicted_files:
                        success = await self.handle_merge_conflict(
                            conflicted_files, 
                            ConflictResolutionStrategy.AUTO_MERGE
                        )
                        if success:
                            return True
                
                # Retry the original command
                result = await self._run_git_command(git_error.command.split())
                if result["exit_code"] == 0:
                    logger.info(f"Retry successful for error {git_error.error_id}")
                    return True
            
            logger.warning(f"All retry attempts failed for error {git_error.error_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error during retry operation: {e}")
            return False
    
    async def _rollback_operation(self, git_error: GitError) -> bool:
        """Rollback to a safe state"""
        try:
            # Find safe rollback point
            rollback_point = await self._find_safe_rollback_point()
            if not rollback_point:
                logger.error("Could not find safe rollback point")
                return False
            
            return await self._rollback_to_commit(rollback_point)
            
        except Exception as e:
            logger.error(f"Error during rollback operation: {e}")
            return False
    
    async def _reset_hard_operation(self, git_error: GitError) -> bool:
        """Perform hard reset to HEAD"""
        try:
            result = await self._run_git_command(["git", "reset", "--hard", "HEAD"])
            return result["exit_code"] == 0
            
        except Exception as e:
            logger.error(f"Error during reset hard operation: {e}")
            return False
    
    async def _clean_workspace_operation(self, git_error: GitError) -> bool:
        """Clean workspace of untracked files and directories"""
        try:
            # Clean untracked files and directories
            result1 = await self._run_git_command(["git", "clean", "-fd"])
            
            # Reset any changes
            result2 = await self._run_git_command(["git", "reset", "--hard"])
            
            return result1["exit_code"] == 0 and result2["exit_code"] == 0
            
        except Exception as e:
            logger.error(f"Error during clean workspace operation: {e}")
            return False
    
    async def _recreate_branch_operation(self, git_error: GitError) -> bool:
        """Recreate a missing branch"""
        try:
            if not git_error.branch:
                logger.error("No branch information available for recreation")
                return False
            
            # Try to recreate branch from origin
            result = await self._run_git_command([
                "git", "checkout", "-b", git_error.branch, f"origin/{git_error.branch}"
            ])
            
            if result["exit_code"] == 0:
                return True
            
            # If that fails, create from main/master
            for base_branch in ["main", "master"]:
                result = await self._run_git_command([
                    "git", "checkout", "-b", git_error.branch, base_branch
                ])
                if result["exit_code"] == 0:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error during recreate branch operation: {e}")
            return False
    
    async def _clone_fresh_operation(self, git_error: GitError) -> bool:
        """Clone a fresh copy of the repository"""
        try:
            # This is a drastic measure - would need remote URL
            logger.warning("Fresh clone operation not implemented - requires remote URL")
            return False
            
        except Exception as e:
            logger.error(f"Error during clone fresh operation: {e}")
            return False
    
    async def _analyze_conflict(self, file_path: str) -> Optional[MergeConflict]:
        """Analyze a merge conflict in a file"""
        try:
            file_full_path = self.repo_path / file_path
            if not file_full_path.exists():
                return None
            
            with open(file_full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Find conflict markers
            conflict_pattern = r'<<<<<<< .*?\n(.*?)\n=======\n(.*?)\n>>>>>>> .*?\n'
            matches = re.findall(conflict_pattern, content, re.DOTALL)
            
            if not matches:
                return None
            
            # Extract conflict information
            conflict_markers = []
            our_content = ""
            their_content = ""
            
            for match in matches:
                our_part, their_part = match
                our_content += our_part
                their_content += their_part
                conflict_markers.append(f"<<<<<<< ... >>>>>>> conflict")
            
            return MergeConflict(
                file_path=file_path,
                conflict_markers=conflict_markers,
                our_content=our_content.strip(),
                their_content=their_content.strip()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing conflict in {file_path}: {e}")
            return None
    
    async def _resolve_conflict(self, 
                              conflict: MergeConflict, 
                              strategy: ConflictResolutionStrategy) -> bool:
        """Resolve a single merge conflict"""
        try:
            file_full_path = self.repo_path / conflict.file_path
            
            if strategy == ConflictResolutionStrategy.OURS:
                # Keep our version
                resolution_content = conflict.our_content
            
            elif strategy == ConflictResolutionStrategy.THEIRS:
                # Keep their version
                resolution_content = conflict.their_content
            
            elif strategy == ConflictResolutionStrategy.AUTO_MERGE:
                # Attempt automatic merge
                resolution_content = await self._auto_merge_conflict(conflict)
                if not resolution_content:
                    return False
            
            elif strategy == ConflictResolutionStrategy.ABORT:
                # Abort the merge
                result = await self._run_git_command(["git", "merge", "--abort"])
                return result["exit_code"] == 0
            
            else:
                logger.warning(f"Unsupported conflict resolution strategy: {strategy.value}")
                return False
            
            # Write resolved content
            with open(file_full_path, 'w', encoding='utf-8') as f:
                f.write(resolution_content)
            
            # Stage the resolved file
            result = await self._run_git_command(["git", "add", conflict.file_path])
            
            if result["exit_code"] == 0:
                conflict.resolved = True
                conflict.resolution_strategy = strategy
                conflict.resolution_content = resolution_content
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving conflict in {conflict.file_path}: {e}")
            return False
    
    async def _auto_merge_conflict(self, conflict: MergeConflict) -> Optional[str]:
        """Attempt automatic merge of conflict"""
        try:
            # Simple heuristics for auto-merge
            our_lines = conflict.our_content.split('\n')
            their_lines = conflict.their_content.split('\n')
            
            # If one side is empty, use the other
            if not our_lines or (len(our_lines) == 1 and not our_lines[0].strip()):
                return conflict.their_content
            
            if not their_lines or (len(their_lines) == 1 and not their_lines[0].strip()):
                return conflict.our_content
            
            # If both sides are identical, use either
            if conflict.our_content == conflict.their_content:
                return conflict.our_content
            
            # Try to merge line by line (very basic)
            merged_lines = []
            max_lines = max(len(our_lines), len(their_lines))
            
            for i in range(max_lines):
                our_line = our_lines[i] if i < len(our_lines) else ""
                their_line = their_lines[i] if i < len(their_lines) else ""
                
                if our_line == their_line:
                    merged_lines.append(our_line)
                elif not our_line:
                    merged_lines.append(their_line)
                elif not their_line:
                    merged_lines.append(our_line)
                else:
                    # Can't auto-merge this conflict
                    return None
            
            return '\n'.join(merged_lines)
            
        except Exception as e:
            logger.error(f"Error in auto-merge: {e}")
            return None
    
    async def _commit_conflict_resolution(self, conflicts: List[MergeConflict]) -> bool:
        """Commit the resolution of merge conflicts"""
        try:
            # Create commit message
            resolved_files = [c.file_path for c in conflicts if c.resolved]
            commit_message = f"Resolve merge conflicts in {len(resolved_files)} files\n\n"
            commit_message += "Resolved files:\n"
            for file_path in resolved_files:
                commit_message += f"- {file_path}\n"
            
            # Commit the resolution
            result = await self._run_git_command(["git", "commit", "-m", commit_message])
            return result["exit_code"] == 0
            
        except Exception as e:
            logger.error(f"Error committing conflict resolution: {e}")
            return False
    
    async def _create_repository_backup(self, description: str) -> Optional[str]:
        """Create a backup of the repository"""
        try:
            backup_id = f"backup_{int(time.time())}"
            backup_path = self.backup_dir / backup_id
            
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy repository (excluding .git/objects to save space)
            shutil.copytree(
                self.repo_path,
                backup_path / "repo",
                ignore=shutil.ignore_patterns('.git/objects/*', '*.pyc', '__pycache__')
            )
            
            # Get current state info
            current_branch = self._get_current_branch()
            current_commit = self._get_current_commit()
            
            # Calculate backup size
            backup_size = sum(
                f.stat().st_size for f in backup_path.rglob('*') if f.is_file()
            )
            
            # Create backup info
            backup = RepositoryBackup(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_path=str(backup_path),
                original_path=str(self.repo_path),
                branch=current_branch or "unknown",
                commit_hash=current_commit or "unknown",
                description=description,
                size_bytes=backup_size
            )
            
            # Store backup info
            self._backups[backup_id] = backup
            self.stats["backups_created"] += 1
            
            logger.info(f"Created repository backup: {backup_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Error creating repository backup: {e}")
            return None
    
    async def _restore_from_backup(self, backup_id: str) -> bool:
        """Restore repository from backup"""
        try:
            if backup_id not in self._backups:
                logger.error(f"Backup {backup_id} not found")
                return False
            
            backup = self._backups[backup_id]
            backup_repo_path = Path(backup.backup_path) / "repo"
            
            if not backup_repo_path.exists():
                logger.error(f"Backup repository path does not exist: {backup_repo_path}")
                return False
            
            # Remove current repository content (except .git)
            for item in self.repo_path.iterdir():
                if item.name != '.git':
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            
            # Restore from backup
            for item in backup_repo_path.iterdir():
                if item.name != '.git':
                    if item.is_dir():
                        shutil.copytree(item, self.repo_path / item.name)
                    else:
                        shutil.copy2(item, self.repo_path / item.name)
            
            logger.info(f"Restored repository from backup: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup {backup_id}: {e}")
            return False
    
    async def _find_safe_rollback_point(self) -> Optional[str]:
        """Find a safe commit to rollback to"""
        try:
            # Get recent commits
            result = await self._run_git_command([
                "git", "log", "--oneline", "-10", "--format=%H %s"
            ])
            
            if result["exit_code"] != 0:
                return None
            
            commits = result["stdout"].strip().split('\n')
            
            # Look for commits that seem safe (e.g., not merge commits, have good messages)
            for commit_line in commits:
                if not commit_line:
                    continue
                
                parts = commit_line.split(' ', 1)
                if len(parts) < 2:
                    continue
                
                commit_hash, message = parts[0], parts[1]
                
                # Skip merge commits
                if message.lower().startswith('merge'):
                    continue
                
                # This is a potential rollback point
                return commit_hash
            
            # If no good commit found, use HEAD~1
            result = await self._run_git_command(["git", "rev-parse", "HEAD~1"])
            if result["exit_code"] == 0:
                return result["stdout"].strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding safe rollback point: {e}")
            return None
    
    async def _rollback_to_commit(self, commit_hash: str) -> bool:
        """Rollback to a specific commit"""
        try:
            result = await self._run_git_command(["git", "reset", "--hard", commit_hash])
            return result["exit_code"] == 0
            
        except Exception as e:
            logger.error(f"Error rolling back to commit {commit_hash}: {e}")
            return False
    
    async def _recover_via_fresh_clone(self) -> bool:
        """Recover by cloning fresh repository"""
        # This would require knowing the remote URL
        logger.warning("Fresh clone recovery not implemented - requires remote URL configuration")
        return False
    
    async def _recover_via_reset_cleanup(self) -> bool:
        """Recover via reset and cleanup"""
        try:
            # Reset to HEAD
            result1 = await self._run_git_command(["git", "reset", "--hard", "HEAD"])
            
            # Clean workspace
            result2 = await self._run_git_command(["git", "clean", "-fd"])
            
            # Garbage collect
            result3 = await self._run_git_command(["git", "gc", "--prune=now"])
            
            return all(r["exit_code"] == 0 for r in [result1, result2, result3])
            
        except Exception as e:
            logger.error(f"Error in reset cleanup recovery: {e}")
            return False
    
    async def _recover_via_cleanup(self) -> bool:
        """Recover via cleanup only"""
        try:
            # Clean workspace
            result1 = await self._run_git_command(["git", "clean", "-fd"])
            
            # Garbage collect
            result2 = await self._run_git_command(["git", "gc"])
            
            return all(r["exit_code"] == 0 for r in [result1, result2])
            
        except Exception as e:
            logger.error(f"Error in cleanup recovery: {e}")
            return False
    
    async def _run_git_command(self, command: List[str]) -> Dict[str, Any]:
        """Run a Git command and return results"""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "exit_code": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore')
            }
            
        except Exception as e:
            logger.error(f"Error running Git command {' '.join(command)}: {e}")
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e)
            }
    
    def _get_current_branch(self) -> Optional[str]:
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
    
    def _get_current_commit(self) -> Optional[str]:
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
    
    def _extract_affected_files(self, stdout: str, stderr: str) -> List[str]:
        """Extract affected files from command output"""
        files = []
        
        # Look for file patterns in output
        for line in (stdout + stderr).split('\n'):
            # Match common file patterns
            if any(ext in line for ext in ['.py', '.js', '.ts', '.json', '.md', '.txt']):
                # Extract filename
                words = line.split()
                for word in words:
                    if any(ext in word for ext in ['.py', '.js', '.ts', '.json', '.md', '.txt']):
                        files.append(word.strip())
        
        return list(set(files))  # Remove duplicates
    
    def _extract_conflicted_files(self, stderr: str) -> List[str]:
        """Extract conflicted files from Git error output"""
        files = []
        
        for line in stderr.split('\n'):
            if 'conflict' in line.lower():
                # Look for file patterns
                words = line.split()
                for word in words:
                    if '/' in word or '.' in word:
                        files.append(word.strip())
        
        return files
    
    # Public API methods
    
    def get_error_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent error history"""
        errors = list(self._errors.values())
        errors.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit > 0:
            errors = errors[:limit]
        
        return [
            {
                "error_id": error.error_id,
                "timestamp": error.timestamp.isoformat(),
                "error_type": error.error_type.value,
                "command": error.command,
                "resolved": error.resolved,
                "resolution_time": error.resolution_time.isoformat() if error.resolution_time else None,
                "recovery_actions": [action.value for action in error.recovery_actions],
                "resolution_notes": error.resolution_notes
            }
            for error in errors
        ]
    
    def get_conflict_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get merge conflict history"""
        conflicts = self._conflict_history[-limit:] if limit > 0 else self._conflict_history
        
        return [
            {
                "file_path": conflict.file_path,
                "resolved": conflict.resolved,
                "resolution_strategy": conflict.resolution_strategy.value if conflict.resolution_strategy else None,
                "conflict_markers_count": len(conflict.conflict_markers)
            }
            for conflict in conflicts
        ]
    
    def get_backup_info(self) -> List[Dict[str, Any]]:
        """Get information about repository backups"""
        return [
            {
                "backup_id": backup.backup_id,
                "timestamp": backup.timestamp.isoformat(),
                "description": backup.description,
                "branch": backup.branch,
                "commit_hash": backup.commit_hash,
                "size_mb": backup.size_bytes / (1024 * 1024)
            }
            for backup in self._backups.values()
        ]
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get error handler statistics"""
        return self.stats.copy()
    
    async def cleanup_old_backups(self) -> int:
        """Clean up old backups beyond retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.backup_retention_days)
            cleaned_count = 0
            
            backups_to_remove = []
            for backup_id, backup in self._backups.items():
                if backup.timestamp < cutoff_date:
                    backups_to_remove.append(backup_id)
            
            for backup_id in backups_to_remove:
                backup = self._backups[backup_id]
                backup_path = Path(backup.backup_path)
                
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                
                del self._backups[backup_id]
                cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old backups")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            return 0


# Global Git error handler instance
git_error_handler = None

def get_git_error_handler(git_manager: GitWorkflowManager = None) -> GitErrorHandler:
    """Get or create global Git error handler instance"""
    global git_error_handler
    
    if git_error_handler is None and git_manager is not None:
        git_error_handler = GitErrorHandler(git_manager)
    
    return git_error_handler