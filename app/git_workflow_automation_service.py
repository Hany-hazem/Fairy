"""
Git Workflow Automation Service

This service provides automated Git workflow triggers and monitoring for task lifecycle events.
It handles automatic branch creation, commit generation, and workflow recovery mechanisms.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import subprocess
from enum import Enum

from .git_workflow_manager import GitWorkflowManager, TaskContext, TaskStatus, BranchType
from .task_git_bridge import TaskGitBridge
from .task_dependency_manager import TaskDependencyManager
from .task_git_models import TaskGitMapping, GitCommit, MergeStatus


logger = logging.getLogger(__name__)


class WorkflowEventType(Enum):
    """Types of workflow events that can trigger Git operations"""
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_ABANDONED = "task_abandoned"
    DEPENDENCY_RESOLVED = "dependency_resolved"
    MERGE_CONFLICT = "merge_conflict"
    BRANCH_READY = "branch_ready"


@dataclass
class WorkflowEvent:
    """Represents a workflow event that can trigger Git operations"""
    event_type: WorkflowEventType
    task_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_type': self.event_type.value,
            'task_id': self.task_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'source': self.source
        }


@dataclass
class WorkflowTrigger:
    """Configuration for automated workflow triggers"""
    event_type: WorkflowEventType
    handler: Callable
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0
    
    def matches_event(self, event: WorkflowEvent) -> bool:
        """Check if this trigger matches the given event"""
        if not self.enabled or event.event_type != self.event_type:
            return False
        
        # Check additional conditions
        for key, expected_value in self.conditions.items():
            if key not in event.metadata or event.metadata[key] != expected_value:
                return False
        
        return True


class GitWorkflowAutomationService:
    """
    Automated Git workflow service that handles task lifecycle events
    and triggers appropriate Git operations.
    """
    
    def __init__(self, 
                 git_manager: GitWorkflowManager,
                 task_git_bridge: TaskGitBridge,
                 dependency_manager: Optional[TaskDependencyManager] = None,
                 config_path: str = ".kiro/workflow_automation.json"):
        self.git_manager = git_manager
        self.task_git_bridge = task_git_bridge
        self.dependency_manager = dependency_manager or TaskDependencyManager(task_git_bridge, git_manager)
        self.config_path = Path(config_path)
        
        # Event handling
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.triggers: List[WorkflowTrigger] = []
        self.event_history: List[WorkflowEvent] = []
        self.max_history_size = 1000
        
        # Monitoring
        self.monitoring_enabled = True
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = datetime.now()
        
        # Configuration
        self.config = self._load_config()
        self._setup_default_triggers()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
        logger.info("Git Workflow Automation Service initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded workflow automation config from {self.config_path}")
                return config
            else:
                # Create default config
                default_config = {
                    "auto_branch_creation": True,
                    "auto_commit_on_progress": True,
                    "auto_merge_ready_branches": False,
                    "conflict_resolution_timeout": 3600,  # 1 hour
                    "health_check_interval": 300,
                    "max_concurrent_operations": 5,
                    "branch_naming_pattern": "task/{task_id}-{task_name}",
                    "commit_message_template": "{type}: {task_name}\n\nTask-ID: {task_id}\nFiles: {files}\nTimestamp: {timestamp}"
                }
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.debug("Saved workflow automation config")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _setup_default_triggers(self) -> None:
        """Setup default workflow triggers"""
        # Task started trigger - create branch
        self.add_trigger(WorkflowTrigger(
            event_type=WorkflowEventType.TASK_STARTED,
            handler=self._handle_task_started,
            priority=10
        ))
        
        # Task progress trigger - commit changes
        self.add_trigger(WorkflowTrigger(
            event_type=WorkflowEventType.TASK_PROGRESS,
            handler=self._handle_task_progress,
            priority=5
        ))
        
        # Task completed trigger - prepare for merge
        self.add_trigger(WorkflowTrigger(
            event_type=WorkflowEventType.TASK_COMPLETED,
            handler=self._handle_task_completed,
            priority=10
        ))
        
        # Dependency resolved trigger - check ready tasks
        self.add_trigger(WorkflowTrigger(
            event_type=WorkflowEventType.DEPENDENCY_RESOLVED,
            handler=self._handle_dependency_resolved,
            priority=3
        ))
        
        logger.info(f"Setup {len(self.triggers)} default workflow triggers")
    
    def add_trigger(self, trigger: WorkflowTrigger) -> None:
        """Add a workflow trigger"""
        self.triggers.append(trigger)
        # Sort by priority (higher priority first)
        self.triggers.sort(key=lambda t: t.priority, reverse=True)
        logger.debug(f"Added workflow trigger for {trigger.event_type.value}")
    
    def remove_trigger(self, event_type: WorkflowEventType) -> bool:
        """Remove workflow triggers for a specific event type"""
        initial_count = len(self.triggers)
        self.triggers = [t for t in self.triggers if t.event_type != event_type]
        removed_count = initial_count - len(self.triggers)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} triggers for {event_type.value}")
            return True
        return False
    
    async def start(self) -> None:
        """Start the automation service"""
        logger.info("Starting Git Workflow Automation Service")
        
        # Start event processing task
        event_processor = asyncio.create_task(self._process_events())
        self._background_tasks.add(event_processor)
        
        # Start health monitoring task
        if self.monitoring_enabled:
            health_monitor = asyncio.create_task(self._health_monitor())
            self._background_tasks.add(health_monitor)
        
        # Start repository monitoring task
        repo_monitor = asyncio.create_task(self._monitor_repository())
        self._background_tasks.add(repo_monitor)
        
        logger.info("Git Workflow Automation Service started")
    
    async def stop(self) -> None:
        """Stop the automation service"""
        logger.info("Stopping Git Workflow Automation Service")
        
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.info("Git Workflow Automation Service stopped")
    
    async def emit_event(self, event: WorkflowEvent) -> None:
        """Emit a workflow event for processing"""
        await self.event_queue.put(event)
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        
        logger.debug(f"Emitted event: {event.event_type.value} for task {event.task_id}")
    
    async def trigger_task_started(self, task_id: str, task_name: str, 
                                 task_description: str = "", 
                                 requirements: List[str] = None) -> None:
        """Trigger task started event"""
        event = WorkflowEvent(
            event_type=WorkflowEventType.TASK_STARTED,
            task_id=task_id,
            metadata={
                'task_name': task_name,
                'task_description': task_description,
                'requirements': requirements or []
            }
        )
        await self.emit_event(event)
    
    async def trigger_task_progress(self, task_id: str, files_changed: List[str], 
                                  progress_notes: str = "") -> None:
        """Trigger task progress event"""
        event = WorkflowEvent(
            event_type=WorkflowEventType.TASK_PROGRESS,
            task_id=task_id,
            metadata={
                'files_changed': files_changed,
                'progress_notes': progress_notes
            }
        )
        await self.emit_event(event)
    
    async def trigger_task_completed(self, task_id: str, completion_notes: str = "",
                                   requirements_addressed: List[str] = None) -> None:
        """Trigger task completed event"""
        event = WorkflowEvent(
            event_type=WorkflowEventType.TASK_COMPLETED,
            task_id=task_id,
            metadata={
                'completion_notes': completion_notes,
                'requirements_addressed': requirements_addressed or []
            }
        )
        await self.emit_event(event)
    
    # Event handlers
    
    async def _handle_task_started(self, event: WorkflowEvent) -> None:
        """Handle task started event - create branch"""
        try:
            if not self.config.get('auto_branch_creation', True):
                logger.debug(f"Auto branch creation disabled, skipping task {event.task_id}")
                return
            
            task_id = event.task_id
            task_name = event.metadata.get('task_name', f'Task {task_id}')
            
            # Create task branch
            branch_name = self.git_manager.create_task_branch(
                task_id=task_id,
                task_name=task_name,
                branch_type=BranchType.TASK
            )
            
            # Link task to branch in bridge
            await self.task_git_bridge.link_task_to_branch(task_id, branch_name)
            
            # Add requirement references if provided
            requirements = event.metadata.get('requirements', [])
            for req in requirements:
                self.task_git_bridge.add_requirement_reference(task_id, req)
            
            logger.info(f"Created branch {branch_name} for task {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle task started event for {event.task_id}: {e}")
    
    async def _handle_task_progress(self, event: WorkflowEvent) -> None:
        """Handle task progress event - commit changes"""
        try:
            if not self.config.get('auto_commit_on_progress', True):
                logger.debug(f"Auto commit on progress disabled, skipping task {event.task_id}")
                return
            
            task_id = event.task_id
            files_changed = event.metadata.get('files_changed', [])
            progress_notes = event.metadata.get('progress_notes', 'Task progress update')
            
            # Generate commit message
            commit_message = self._generate_progress_commit_message(task_id, progress_notes, files_changed)
            
            # Commit changes
            commit_hash = self.git_manager.commit_task_progress(
                task_id=task_id,
                files=files_changed,
                message=commit_message
            )
            
            # Update task status in bridge
            await self.task_git_bridge.update_task_status_from_git(commit_hash)
            
            logger.info(f"Committed progress for task {task_id}: {commit_hash[:8]}")
            
        except Exception as e:
            logger.error(f"Failed to handle task progress event for {event.task_id}: {e}")
    
    async def _handle_task_completed(self, event: WorkflowEvent) -> None:
        """Handle task completed event - prepare for merge"""
        try:
            task_id = event.task_id
            completion_notes = event.metadata.get('completion_notes', 'Task completed')
            
            # Complete task branch
            success = self.git_manager.complete_task_branch(task_id, completion_notes)
            
            if success:
                # Check if task can be automatically merged
                if self.config.get('auto_merge_ready_branches', False):
                    await self._attempt_auto_merge(task_id)
                
                # Emit dependency resolved event for dependent tasks
                await self._emit_dependency_resolved_events(task_id)
                
                logger.info(f"Completed task {task_id} and prepared branch for merge")
            else:
                logger.warning(f"Failed to complete task branch for {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle task completed event for {event.task_id}: {e}")
    
    async def _handle_dependency_resolved(self, event: WorkflowEvent) -> None:
        """Handle dependency resolved event - check for ready tasks"""
        try:
            # Get tasks that are now ready to be worked on
            ready_tasks = self.dependency_manager.get_ready_tasks()
            
            for ready_task_id in ready_tasks:
                mapping = self.task_git_bridge.get_task_mapping(ready_task_id)
                if mapping and mapping.status == TaskStatus.NOT_STARTED:
                    logger.info(f"Task {ready_task_id} is now ready to start (dependencies resolved)")
                    # Could emit a task_ready event here if needed
            
        except Exception as e:
            logger.error(f"Failed to handle dependency resolved event: {e}")
    
    # Helper methods
    
    def _generate_progress_commit_message(self, task_id: str, notes: str, files: List[str]) -> str:
        """Generate commit message for task progress"""
        template = self.config.get('commit_message_template', 
                                 "feat: {task_name}\n\nTask-ID: {task_id}\nFiles: {files}\nTimestamp: {timestamp}")
        
        # Get task name from mapping if available
        mapping = self.task_git_bridge.get_task_mapping(task_id)
        task_name = f"Task {task_id}"  # Default
        
        return template.format(
            type="feat",
            task_name=notes or task_name,
            task_id=task_id,
            files=", ".join(files[:5]) + ("..." if len(files) > 5 else ""),
            timestamp=datetime.now().isoformat()
        )
    
    async def _attempt_auto_merge(self, task_id: str) -> bool:
        """Attempt to automatically merge a completed task"""
        try:
            mapping = self.task_git_bridge.get_task_mapping(task_id)
            if not mapping:
                return False
            
            # Check for merge conflicts
            conflicts = self.git_manager.check_for_conflicts()
            if conflicts:
                logger.warning(f"Cannot auto-merge task {task_id}: conflicts detected in {conflicts}")
                return False
            
            # Attempt merge (this would need to be implemented in GitWorkflowManager)
            # For now, just mark as ready for merge
            mapping.merge_status = MergeStatus.READY
            logger.info(f"Task {task_id} marked as ready for merge")
            return True
            
        except Exception as e:
            logger.error(f"Failed to auto-merge task {task_id}: {e}")
            return False
    
    async def _emit_dependency_resolved_events(self, completed_task_id: str) -> None:
        """Emit dependency resolved events for tasks that depend on the completed task"""
        try:
            dependent_tasks = self.dependency_manager.get_task_dependents(completed_task_id)
            
            for dependent_task_id in dependent_tasks:
                event = WorkflowEvent(
                    event_type=WorkflowEventType.DEPENDENCY_RESOLVED,
                    task_id=dependent_task_id,
                    metadata={
                        'resolved_dependency': completed_task_id
                    }
                )
                await self.emit_event(event)
            
        except Exception as e:
            logger.error(f"Failed to emit dependency resolved events: {e}")
    
    # Background tasks
    
    async def _process_events(self) -> None:
        """Process workflow events from the queue"""
        logger.info("Started event processing task")
        
        try:
            while True:
                try:
                    # Get event from queue with timeout
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    
                    # Process event with matching triggers
                    await self._process_single_event(event)
                    
                    # Mark task as done
                    self.event_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No events to process, continue
                    continue
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Event processing task cancelled")
        except Exception as e:
            logger.error(f"Event processing task failed: {e}")
    
    async def _process_single_event(self, event: WorkflowEvent) -> None:
        """Process a single workflow event"""
        try:
            matching_triggers = [t for t in self.triggers if t.matches_event(event)]
            
            if not matching_triggers:
                logger.debug(f"No triggers found for event {event.event_type.value}")
                return
            
            # Execute matching triggers
            for trigger in matching_triggers:
                try:
                    await trigger.handler(event)
                except Exception as e:
                    logger.error(f"Trigger handler failed for {event.event_type.value}: {e}")
            
            logger.debug(f"Processed event {event.event_type.value} with {len(matching_triggers)} triggers")
            
        except Exception as e:
            logger.error(f"Failed to process event {event.event_type.value}: {e}")
    
    async def _health_monitor(self) -> None:
        """Monitor system health and perform maintenance"""
        logger.info("Started health monitoring task")
        
        try:
            while True:
                await asyncio.sleep(self.config.get('health_check_interval', 300))
                
                try:
                    await self._perform_health_check()
                    self.last_health_check = datetime.now()
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Health monitoring task cancelled")
        except Exception as e:
            logger.error(f"Health monitoring task failed: {e}")
    
    async def _perform_health_check(self) -> None:
        """Perform system health check"""
        try:
            # Check Git repository status
            git_status = self.git_manager.get_git_status()
            
            # Check for stale branches
            await self._check_stale_branches()
            
            # Check for failed operations
            await self._check_failed_operations()
            
            # Clean up old event history
            self._cleanup_event_history()
            
            logger.debug("Health check completed successfully")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def _monitor_repository(self) -> None:
        """Monitor repository for external changes"""
        logger.info("Started repository monitoring task")
        
        try:
            last_check = datetime.now()
            
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                try:
                    # Check for new commits not made by automation
                    await self._check_external_commits(last_check)
                    last_check = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Repository monitoring failed: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Repository monitoring task cancelled")
        except Exception as e:
            logger.error(f"Repository monitoring task failed: {e}")
    
    async def _check_stale_branches(self) -> None:
        """Check for stale task branches that need attention"""
        try:
            mappings = self.task_git_bridge.get_all_mappings()
            stale_threshold = timedelta(days=7)  # 7 days
            
            for task_id, mapping in mappings.items():
                if (mapping.status == TaskStatus.IN_PROGRESS and 
                    datetime.now() - mapping.created_at > stale_threshold):
                    logger.warning(f"Stale branch detected for task {task_id}: {mapping.branch_name}")
                    # Could emit a stale_branch event here
            
        except Exception as e:
            logger.error(f"Failed to check stale branches: {e}")
    
    async def _check_failed_operations(self) -> None:
        """Check for failed Git operations that need recovery"""
        try:
            # Check for branches with merge conflicts
            mappings = self.task_git_bridge.get_all_mappings()
            
            for task_id, mapping in mappings.items():
                if mapping.merge_conflicts:
                    logger.warning(f"Task {task_id} has unresolved merge conflicts: {mapping.merge_conflicts}")
                    # Could emit a merge_conflict event here
            
        except Exception as e:
            logger.error(f"Failed to check failed operations: {e}")
    
    async def _check_external_commits(self, since: datetime) -> None:
        """Check for commits made outside of automation"""
        try:
            # Get recent commits
            result = subprocess.run([
                'git', 'log', '--since', since.isoformat(), '--format=%H|%s|%an'
            ], capture_output=True, text=True, check=True)
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) >= 3:
                    commit_hash, message, author = parts[0], parts[1], parts[2]
                    
                    # Check if this is an automation commit
                    if 'Task-ID:' not in message:
                        logger.info(f"External commit detected: {commit_hash[:8]} by {author}")
                        # Could emit an external_commit event here
            
        except subprocess.CalledProcessError:
            # No commits found or git error, ignore
            pass
        except Exception as e:
            logger.error(f"Failed to check external commits: {e}")
    
    def _cleanup_event_history(self) -> None:
        """Clean up old events from history"""
        if len(self.event_history) > self.max_history_size:
            # Keep only the most recent events
            self.event_history = self.event_history[-self.max_history_size:]
            logger.debug(f"Cleaned up event history, kept {len(self.event_history)} events")
    
    # Public API methods
    
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent event history"""
        recent_events = self.event_history[-limit:] if limit > 0 else self.event_history
        return [event.to_dict() for event in recent_events]
    
    def get_trigger_status(self) -> List[Dict[str, Any]]:
        """Get status of all workflow triggers"""
        return [
            {
                'event_type': trigger.event_type.value,
                'enabled': trigger.enabled,
                'priority': trigger.priority,
                'conditions': trigger.conditions
            }
            for trigger in self.triggers
        ]
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get overall service status"""
        return {
            'running': len(self._background_tasks) > 0,
            'triggers_count': len(self.triggers),
            'events_processed': len(self.event_history),
            'last_health_check': self.last_health_check.isoformat(),
            'monitoring_enabled': self.monitoring_enabled,
            'config': self.config
        }
    
    async def manual_trigger(self, event_type: str, task_id: str, metadata: Dict[str, Any] = None) -> bool:
        """Manually trigger a workflow event"""
        try:
            event = WorkflowEvent(
                event_type=WorkflowEventType(event_type),
                task_id=task_id,
                metadata=metadata or {},
                source="manual"
            )
            await self.emit_event(event)
            return True
        except Exception as e:
            logger.error(f"Failed to manually trigger event: {e}")
            return False