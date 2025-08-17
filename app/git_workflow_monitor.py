"""
Git Workflow Monitoring and Recovery System

This module provides comprehensive monitoring of Git operations and automated
recovery mechanisms for workflow failures.
"""

import asyncio
import logging
import subprocess
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from .git_workflow_manager import GitWorkflowManager
from .task_git_bridge import TaskGitBridge
from .task_git_models import TaskStatus, MergeStatus, TaskGitMapping


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class RecoveryAction(Enum):
    """Types of recovery actions"""
    RETRY_OPERATION = "retry_operation"
    RESET_BRANCH = "reset_branch"
    MERGE_CONFLICT_RESOLUTION = "merge_conflict_resolution"
    CLEANUP_STALE_BRANCHES = "cleanup_stale_branches"
    REPOSITORY_REPAIR = "repository_repair"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class HealthCheckResult:
    """Result of a health check operation"""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'component': self.component,
            'status': self.status.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'recovery_actions': [action.value for action in self.recovery_actions]
        }


@dataclass
class RecoveryOperation:
    """Represents a recovery operation"""
    operation_id: str
    action: RecoveryAction
    target: str  # task_id, branch_name, etc.
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'operation_id': self.operation_id,
            'action': self.action.value,
            'target': self.target,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


@dataclass
class RepositoryHealth:
    """Overall repository health status"""
    overall_status: HealthStatus
    last_check: datetime
    issues_count: int
    warnings_count: int
    active_recoveries: int
    components: Dict[str, HealthCheckResult] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'overall_status': self.overall_status.value,
            'last_check': self.last_check.isoformat(),
            'issues_count': self.issues_count,
            'warnings_count': self.warnings_count,
            'active_recoveries': self.active_recoveries,
            'components': {name: result.to_dict() for name, result in self.components.items()}
        }


class GitWorkflowMonitor:
    """
    Comprehensive Git workflow monitoring and recovery system
    """
    
    def __init__(self,
                 git_manager: GitWorkflowManager,
                 task_git_bridge: TaskGitBridge,
                 monitoring_config_path: str = ".kiro/workflow_monitoring.json"):
        self.git_manager = git_manager
        self.task_git_bridge = task_git_bridge
        self.config_path = Path(monitoring_config_path)
        
        # Monitoring state
        self.health_history: List[RepositoryHealth] = []
        self.active_recoveries: Dict[str, RecoveryOperation] = {}
        self.monitoring_enabled = True
        
        # Configuration
        self.config = self._load_monitoring_config()
        
        # Thresholds and limits
        self.stale_branch_threshold = timedelta(days=self.config.get('stale_branch_days', 7))
        self.conflict_timeout = timedelta(hours=self.config.get('conflict_timeout_hours', 24))
        self.max_concurrent_recoveries = self.config.get('max_concurrent_recoveries', 3)
        self.health_check_interval = self.config.get('health_check_interval', 300)
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
        logger.info("Git Workflow Monitor initialized")
    
    def _load_monitoring_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default config
                default_config = {
                    "stale_branch_days": 7,
                    "conflict_timeout_hours": 24,
                    "max_concurrent_recoveries": 3,
                    "health_check_interval": 300,
                    "auto_recovery_enabled": True,
                    "repository_checks": {
                        "disk_space_threshold_mb": 1000,
                        "max_branch_count": 100,
                        "max_stale_branches": 10
                    },
                    "alerting": {
                        "enabled": True,
                        "critical_threshold": 5,
                        "warning_threshold": 3
                    }
                }
                self._save_monitoring_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"Failed to load monitoring config: {e}")
            return {}
    
    def _save_monitoring_config(self, config: Dict[str, Any]) -> None:
        """Save monitoring configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save monitoring config: {e}")
    
    async def start_monitoring(self) -> None:
        """Start the monitoring system"""
        logger.info("Starting Git Workflow Monitor")
        
        # Start health monitoring task
        health_monitor = asyncio.create_task(self._health_monitoring_loop())
        self._background_tasks.add(health_monitor)
        
        # Start recovery processing task
        recovery_processor = asyncio.create_task(self._recovery_processing_loop())
        self._background_tasks.add(recovery_processor)
        
        # Start repository maintenance task
        maintenance_task = asyncio.create_task(self._repository_maintenance_loop())
        self._background_tasks.add(maintenance_task)
        
        logger.info("Git Workflow Monitor started")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        logger.info("Stopping Git Workflow Monitor")
        
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.info("Git Workflow Monitor stopped")
    
    async def perform_health_check(self) -> RepositoryHealth:
        """Perform comprehensive repository health check"""
        logger.debug("Performing repository health check")
        
        health_results = {}
        
        # Check Git repository status
        health_results['git_status'] = await self._check_git_status()
        
        # Check task-branch mappings
        health_results['task_mappings'] = await self._check_task_mappings()
        
        # Check for stale branches
        health_results['stale_branches'] = await self._check_stale_branches()
        
        # Check for merge conflicts
        health_results['merge_conflicts'] = await self._check_merge_conflicts()
        
        # Check repository integrity
        health_results['repository_integrity'] = await self._check_repository_integrity()
        
        # Check disk space
        health_results['disk_space'] = await self._check_disk_space()
        
        # Calculate overall health
        overall_status = self._calculate_overall_health(health_results)
        
        # Count issues and warnings
        issues_count = sum(1 for result in health_results.values() 
                          if result.status in [HealthStatus.CRITICAL, HealthStatus.FAILED])
        warnings_count = sum(1 for result in health_results.values() 
                           if result.status == HealthStatus.WARNING)
        
        # Create health report
        health = RepositoryHealth(
            overall_status=overall_status,
            last_check=datetime.now(),
            issues_count=issues_count,
            warnings_count=warnings_count,
            active_recoveries=len(self.active_recoveries),
            components=health_results
        )
        
        # Store in history
        self.health_history.append(health)
        if len(self.health_history) > 100:  # Keep last 100 checks
            self.health_history.pop(0)
        
        # Trigger recovery actions if needed
        if self.config.get('auto_recovery_enabled', True):
            await self._trigger_recovery_actions(health_results)
        
        logger.info(f"Health check completed: {overall_status.value} "
                   f"({issues_count} issues, {warnings_count} warnings)")
        
        return health
    
    async def _check_git_status(self) -> HealthCheckResult:
        """Check Git repository status"""
        try:
            git_status = self.git_manager.get_git_status()
            
            # Check for uncommitted changes in main branch
            current_branch = await self._get_current_branch()
            if current_branch == "main" and (git_status['modified'] or git_status['untracked']):
                return HealthCheckResult(
                    component="git_status",
                    status=HealthStatus.WARNING,
                    message="Uncommitted changes in main branch",
                    details={'uncommitted_files': git_status['modified'] + git_status['untracked']},
                    recovery_actions=[RecoveryAction.MANUAL_INTERVENTION]
                )
            
            return HealthCheckResult(
                component="git_status",
                status=HealthStatus.HEALTHY,
                message="Git repository status is clean",
                details=git_status
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="git_status",
                status=HealthStatus.FAILED,
                message=f"Failed to check Git status: {e}",
                recovery_actions=[RecoveryAction.REPOSITORY_REPAIR]
            )
    
    async def _check_task_mappings(self) -> HealthCheckResult:
        """Check task-branch mappings for consistency"""
        try:
            mappings = self.task_git_bridge.get_all_mappings()
            issues = []
            
            for task_id, mapping in mappings.items():
                # Check if branch exists
                if not await self._branch_exists(mapping.branch_name):
                    issues.append(f"Branch {mapping.branch_name} for task {task_id} does not exist")
                
                # Check for orphaned commits
                for commit_hash in mapping.commits:
                    if not await self._commit_exists(commit_hash):
                        issues.append(f"Commit {commit_hash} for task {task_id} does not exist")
            
            if issues:
                return HealthCheckResult(
                    component="task_mappings",
                    status=HealthStatus.WARNING,
                    message=f"Found {len(issues)} task mapping issues",
                    details={'issues': issues},
                    recovery_actions=[RecoveryAction.CLEANUP_STALE_BRANCHES]
                )
            
            return HealthCheckResult(
                component="task_mappings",
                status=HealthStatus.HEALTHY,
                message=f"All {len(mappings)} task mappings are consistent"
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="task_mappings",
                status=HealthStatus.FAILED,
                message=f"Failed to check task mappings: {e}"
            )
    
    async def _check_stale_branches(self) -> HealthCheckResult:
        """Check for stale branches that need attention"""
        try:
            mappings = self.task_git_bridge.get_all_mappings()
            stale_branches = []
            
            for task_id, mapping in mappings.items():
                if (mapping.status == TaskStatus.IN_PROGRESS and 
                    datetime.now() - mapping.created_at > self.stale_branch_threshold):
                    stale_branches.append({
                        'task_id': task_id,
                        'branch_name': mapping.branch_name,
                        'age_days': (datetime.now() - mapping.created_at).days
                    })
            
            max_stale = self.config.get('repository_checks', {}).get('max_stale_branches', 10)
            
            if len(stale_branches) > max_stale:
                return HealthCheckResult(
                    component="stale_branches",
                    status=HealthStatus.CRITICAL,
                    message=f"Too many stale branches: {len(stale_branches)}",
                    details={'stale_branches': stale_branches},
                    recovery_actions=[RecoveryAction.CLEANUP_STALE_BRANCHES]
                )
            elif stale_branches:
                return HealthCheckResult(
                    component="stale_branches",
                    status=HealthStatus.WARNING,
                    message=f"Found {len(stale_branches)} stale branches",
                    details={'stale_branches': stale_branches},
                    recovery_actions=[RecoveryAction.CLEANUP_STALE_BRANCHES]
                )
            
            return HealthCheckResult(
                component="stale_branches",
                status=HealthStatus.HEALTHY,
                message="No stale branches found"
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="stale_branches",
                status=HealthStatus.FAILED,
                message=f"Failed to check stale branches: {e}"
            )
    
    async def _check_merge_conflicts(self) -> HealthCheckResult:
        """Check for unresolved merge conflicts"""
        try:
            conflicts = self.git_manager.check_for_conflicts()
            mappings = self.task_git_bridge.get_all_mappings()
            
            # Check for tasks with long-standing conflicts
            long_conflicts = []
            for task_id, mapping in mappings.items():
                if (mapping.merge_conflicts and 
                    mapping.created_at < datetime.now() - self.conflict_timeout):
                    long_conflicts.append({
                        'task_id': task_id,
                        'conflicts': mapping.merge_conflicts,
                        'age_hours': (datetime.now() - mapping.created_at).total_seconds() / 3600
                    })
            
            if conflicts:
                return HealthCheckResult(
                    component="merge_conflicts",
                    status=HealthStatus.CRITICAL,
                    message=f"Active merge conflicts in {len(conflicts)} files",
                    details={'conflicted_files': conflicts, 'long_conflicts': long_conflicts},
                    recovery_actions=[RecoveryAction.MERGE_CONFLICT_RESOLUTION]
                )
            elif long_conflicts:
                return HealthCheckResult(
                    component="merge_conflicts",
                    status=HealthStatus.WARNING,
                    message=f"Long-standing conflicts in {len(long_conflicts)} tasks",
                    details={'long_conflicts': long_conflicts},
                    recovery_actions=[RecoveryAction.MERGE_CONFLICT_RESOLUTION]
                )
            
            return HealthCheckResult(
                component="merge_conflicts",
                status=HealthStatus.HEALTHY,
                message="No merge conflicts detected"
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="merge_conflicts",
                status=HealthStatus.FAILED,
                message=f"Failed to check merge conflicts: {e}"
            )
    
    async def _check_repository_integrity(self) -> HealthCheckResult:
        """Check Git repository integrity"""
        try:
            # Run git fsck to check repository integrity
            result = subprocess.run(
                ['git', 'fsck', '--no-progress'],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                return HealthCheckResult(
                    component="repository_integrity",
                    status=HealthStatus.HEALTHY,
                    message="Repository integrity check passed"
                )
            else:
                return HealthCheckResult(
                    component="repository_integrity",
                    status=HealthStatus.CRITICAL,
                    message="Repository integrity issues detected",
                    details={'fsck_output': result.stderr},
                    recovery_actions=[RecoveryAction.REPOSITORY_REPAIR]
                )
                
        except subprocess.TimeoutExpired:
            return HealthCheckResult(
                component="repository_integrity",
                status=HealthStatus.WARNING,
                message="Repository integrity check timed out"
            )
        except Exception as e:
            return HealthCheckResult(
                component="repository_integrity",
                status=HealthStatus.FAILED,
                message=f"Failed to check repository integrity: {e}"
            )
    
    async def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space"""
        try:
            import shutil
            
            # Get disk usage for repository directory
            total, used, free = shutil.disk_usage(self.git_manager.repo_path)
            free_mb = free // (1024 * 1024)
            
            threshold_mb = self.config.get('repository_checks', {}).get('disk_space_threshold_mb', 1000)
            
            if free_mb < threshold_mb:
                return HealthCheckResult(
                    component="disk_space",
                    status=HealthStatus.CRITICAL,
                    message=f"Low disk space: {free_mb}MB available",
                    details={'free_mb': free_mb, 'threshold_mb': threshold_mb},
                    recovery_actions=[RecoveryAction.CLEANUP_STALE_BRANCHES]
                )
            elif free_mb < threshold_mb * 2:
                return HealthCheckResult(
                    component="disk_space",
                    status=HealthStatus.WARNING,
                    message=f"Disk space getting low: {free_mb}MB available",
                    details={'free_mb': free_mb}
                )
            
            return HealthCheckResult(
                component="disk_space",
                status=HealthStatus.HEALTHY,
                message=f"Sufficient disk space: {free_mb}MB available",
                details={'free_mb': free_mb}
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="disk_space",
                status=HealthStatus.FAILED,
                message=f"Failed to check disk space: {e}"
            )
    
    def _calculate_overall_health(self, health_results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Calculate overall health status from component results"""
        if any(result.status == HealthStatus.FAILED for result in health_results.values()):
            return HealthStatus.FAILED
        elif any(result.status == HealthStatus.CRITICAL for result in health_results.values()):
            return HealthStatus.CRITICAL
        elif any(result.status == HealthStatus.WARNING for result in health_results.values()):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    async def _trigger_recovery_actions(self, health_results: Dict[str, HealthCheckResult]) -> None:
        """Trigger recovery actions based on health check results"""
        for component, result in health_results.items():
            for action in result.recovery_actions:
                if len(self.active_recoveries) >= self.max_concurrent_recoveries:
                    logger.warning("Maximum concurrent recoveries reached, skipping recovery action")
                    break
                
                await self._schedule_recovery(action, component, result.details)
    
    async def _schedule_recovery(self, action: RecoveryAction, target: str, details: Dict[str, Any]) -> str:
        """Schedule a recovery operation"""
        operation_id = f"{action.value}_{target}_{int(datetime.now().timestamp())}"
        
        recovery_op = RecoveryOperation(
            operation_id=operation_id,
            action=action,
            target=target
        )
        
        self.active_recoveries[operation_id] = recovery_op
        logger.info(f"Scheduled recovery operation: {operation_id}")
        
        return operation_id
    
    # Recovery action implementations
    
    async def _execute_recovery_operation(self, operation: RecoveryOperation) -> bool:
        """Execute a recovery operation"""
        try:
            operation.status = "running"
            logger.info(f"Executing recovery operation: {operation.operation_id}")
            
            success = False
            
            if operation.action == RecoveryAction.CLEANUP_STALE_BRANCHES:
                success = await self._cleanup_stale_branches()
            elif operation.action == RecoveryAction.MERGE_CONFLICT_RESOLUTION:
                success = await self._resolve_merge_conflicts(operation.target)
            elif operation.action == RecoveryAction.REPOSITORY_REPAIR:
                success = await self._repair_repository()
            elif operation.action == RecoveryAction.RESET_BRANCH:
                success = await self._reset_branch(operation.target)
            else:
                logger.warning(f"Unknown recovery action: {operation.action}")
                success = False
            
            operation.status = "completed" if success else "failed"
            operation.completed_at = datetime.now()
            
            if success:
                logger.info(f"Recovery operation completed successfully: {operation.operation_id}")
            else:
                logger.error(f"Recovery operation failed: {operation.operation_id}")
                operation.retry_count += 1
            
            return success
            
        except Exception as e:
            operation.status = "failed"
            operation.error_message = str(e)
            operation.completed_at = datetime.now()
            operation.retry_count += 1
            logger.error(f"Recovery operation failed with exception: {operation.operation_id}: {e}")
            return False
    
    async def _cleanup_stale_branches(self) -> bool:
        """Clean up stale branches"""
        try:
            mappings = self.task_git_bridge.get_all_mappings()
            cleaned_count = 0
            
            for task_id, mapping in mappings.items():
                if (mapping.status == TaskStatus.IN_PROGRESS and 
                    datetime.now() - mapping.created_at > self.stale_branch_threshold):
                    
                    # Check if branch has any recent activity
                    if await self._has_recent_activity(mapping.branch_name, days=7):
                        continue  # Skip branches with recent activity
                    
                    # Delete the branch
                    if await self._delete_branch(mapping.branch_name):
                        # Update mapping status
                        mapping.status = TaskStatus.ABANDONED
                        cleaned_count += 1
                        logger.info(f"Cleaned up stale branch: {mapping.branch_name}")
            
            logger.info(f"Cleaned up {cleaned_count} stale branches")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup stale branches: {e}")
            return False
    
    async def _resolve_merge_conflicts(self, target: str) -> bool:
        """Attempt to resolve merge conflicts"""
        try:
            # This is a placeholder for conflict resolution logic
            # In practice, this would involve more sophisticated conflict resolution
            conflicts = self.git_manager.check_for_conflicts()
            
            if not conflicts:
                return True  # No conflicts to resolve
            
            # For now, just log the conflicts that need manual resolution
            logger.warning(f"Merge conflicts require manual resolution: {conflicts}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve merge conflicts: {e}")
            return False
    
    async def _repair_repository(self) -> bool:
        """Attempt to repair repository issues"""
        try:
            # Run git gc to clean up repository
            result = subprocess.run(
                ['git', 'gc', '--prune=now'],
                capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                logger.info("Repository garbage collection completed")
                return True
            else:
                logger.error(f"Repository garbage collection failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to repair repository: {e}")
            return False
    
    async def _reset_branch(self, branch_name: str) -> bool:
        """Reset a branch to a clean state"""
        try:
            # This would implement branch reset logic
            logger.info(f"Reset branch operation not implemented for: {branch_name}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to reset branch {branch_name}: {e}")
            return False
    
    # Helper methods
    
    async def _get_current_branch(self) -> Optional[str]:
        """Get current Git branch"""
        try:
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    async def _branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists"""
        try:
            result = subprocess.run(
                ['git', 'branch', '--list', branch_name],
                capture_output=True, text=True, check=True
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False
    
    async def _commit_exists(self, commit_hash: str) -> bool:
        """Check if a commit exists"""
        try:
            result = subprocess.run(
                ['git', 'cat-file', '-e', commit_hash],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    async def _has_recent_activity(self, branch_name: str, days: int = 7) -> bool:
        """Check if a branch has recent activity"""
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            result = subprocess.run([
                'git', 'log', '--since', since_date, '--oneline', branch_name
            ], capture_output=True, text=True, check=True)
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False
    
    async def _delete_branch(self, branch_name: str) -> bool:
        """Delete a Git branch"""
        try:
            result = subprocess.run(
                ['git', 'branch', '-D', branch_name],
                capture_output=True, text=True, check=True
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    # Background monitoring loops
    
    async def _health_monitoring_loop(self) -> None:
        """Main health monitoring loop"""
        logger.info("Started health monitoring loop")
        
        try:
            while True:
                await asyncio.sleep(self.health_check_interval)
                
                if self.monitoring_enabled:
                    try:
                        await self.perform_health_check()
                    except Exception as e:
                        logger.error(f"Health check failed: {e}")
                        
        except asyncio.CancelledError:
            logger.info("Health monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Health monitoring loop failed: {e}")
    
    async def _recovery_processing_loop(self) -> None:
        """Process recovery operations"""
        logger.info("Started recovery processing loop")
        
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Process pending recoveries
                completed_operations = []
                
                for operation_id, operation in self.active_recoveries.items():
                    if operation.status == "pending":
                        await self._execute_recovery_operation(operation)
                    elif operation.status in ["completed", "failed"]:
                        if (operation.status == "failed" and 
                            operation.retry_count < operation.max_retries):
                            # Retry failed operations
                            operation.status = "pending"
                            await asyncio.sleep(60)  # Wait before retry
                        else:
                            completed_operations.append(operation_id)
                
                # Clean up completed operations
                for operation_id in completed_operations:
                    del self.active_recoveries[operation_id]
                    
        except asyncio.CancelledError:
            logger.info("Recovery processing loop cancelled")
        except Exception as e:
            logger.error(f"Recovery processing loop failed: {e}")
    
    async def _repository_maintenance_loop(self) -> None:
        """Perform regular repository maintenance"""
        logger.info("Started repository maintenance loop")
        
        try:
            while True:
                await asyncio.sleep(3600)  # Run every hour
                
                try:
                    # Perform maintenance tasks
                    await self._perform_maintenance()
                except Exception as e:
                    logger.error(f"Repository maintenance failed: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Repository maintenance loop cancelled")
        except Exception as e:
            logger.error(f"Repository maintenance loop failed: {e}")
    
    async def _perform_maintenance(self) -> None:
        """Perform regular maintenance tasks"""
        try:
            # Clean up old health history
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]
            
            # Run git gc periodically (once per day)
            last_gc_file = Path(self.git_manager.repo_path) / ".git" / "last_gc"
            if (not last_gc_file.exists() or 
                datetime.now() - datetime.fromtimestamp(last_gc_file.stat().st_mtime) > timedelta(days=1)):
                
                result = subprocess.run(
                    ['git', 'gc', '--auto'],
                    capture_output=True, text=True, timeout=300
                )
                
                if result.returncode == 0:
                    last_gc_file.touch()
                    logger.debug("Repository garbage collection completed")
            
        except Exception as e:
            logger.error(f"Maintenance task failed: {e}")
    
    # Public API methods
    
    def get_current_health(self) -> Optional[RepositoryHealth]:
        """Get the most recent health status"""
        return self.health_history[-1] if self.health_history else None
    
    def get_health_history(self, limit: int = 24) -> List[Dict[str, Any]]:
        """Get recent health history"""
        recent_health = self.health_history[-limit:] if limit > 0 else self.health_history
        return [health.to_dict() for health in recent_health]
    
    def get_active_recoveries(self) -> List[Dict[str, Any]]:
        """Get currently active recovery operations"""
        return [op.to_dict() for op in self.active_recoveries.values()]
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        current_health = self.get_current_health()
        
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'background_tasks_count': len(self._background_tasks),
            'active_recoveries_count': len(self.active_recoveries),
            'health_checks_performed': len(self.health_history),
            'current_health': current_health.overall_status.value if current_health else None,
            'last_check': current_health.last_check.isoformat() if current_health else None,
            'config': self.config
        }
    
    async def manual_recovery(self, action: str, target: str) -> str:
        """Manually trigger a recovery operation"""
        try:
            recovery_action = RecoveryAction(action)
            return await self._schedule_recovery(recovery_action, target, {})
        except ValueError:
            raise ValueError(f"Invalid recovery action: {action}")