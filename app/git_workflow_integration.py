"""
Git Workflow Integration

This module provides a unified interface for Git workflow automation and monitoring,
integrating all components into a cohesive system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .git_workflow_manager import GitWorkflowManager
from .task_git_bridge import TaskGitBridge
from .task_dependency_manager import TaskDependencyManager
from .git_workflow_automation_service import GitWorkflowAutomationService, WorkflowEventType
from .git_workflow_monitor import GitWorkflowMonitor, HealthStatus, RecoveryAction


logger = logging.getLogger(__name__)


class GitWorkflowIntegration:
    """
    Unified Git workflow system that integrates automation and monitoring
    """
    
    def __init__(self, repo_path: str = "."):
        # Initialize core components
        self.git_manager = GitWorkflowManager(repo_path)
        self.task_git_bridge = TaskGitBridge(self.git_manager)
        self.dependency_manager = TaskDependencyManager(self.task_git_bridge, self.git_manager)
        
        # Initialize automation and monitoring
        self.automation_service = GitWorkflowAutomationService(
            self.git_manager,
            self.task_git_bridge,
            self.dependency_manager
        )
        
        self.monitor = GitWorkflowMonitor(
            self.git_manager,
            self.task_git_bridge
        )
        
        # Integration state
        self.running = False
        
        logger.info("Git Workflow Integration initialized")
    
    async def start(self) -> None:
        """Start the integrated workflow system"""
        if self.running:
            logger.warning("Git workflow system is already running")
            return
        
        logger.info("Starting integrated Git workflow system")
        
        try:
            # Start automation service
            await self.automation_service.start()
            
            # Start monitoring system
            await self.monitor.start_monitoring()
            
            # Set up integration between automation and monitoring
            await self._setup_integration()
            
            self.running = True
            logger.info("Git workflow system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Git workflow system: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the integrated workflow system"""
        if not self.running:
            return
        
        logger.info("Stopping integrated Git workflow system")
        
        try:
            # Stop automation service
            await self.automation_service.stop()
            
            # Stop monitoring system
            await self.monitor.stop_monitoring()
            
            self.running = False
            logger.info("Git workflow system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Git workflow system: {e}")
    
    async def _setup_integration(self) -> None:
        """Set up integration between automation and monitoring"""
        # Add monitoring-triggered automation
        # For example, when monitoring detects issues, trigger recovery events
        pass
    
    # Task lifecycle methods
    
    async def start_task(self, task_id: str, task_name: str, 
                        task_description: str = "", 
                        requirements: List[str] = None,
                        dependencies: List[str] = None) -> bool:
        """Start a new task with full workflow integration"""
        try:
            logger.info(f"Starting task {task_id}: {task_name}")
            
            # Add dependencies if specified
            if dependencies:
                for dep_id in dependencies:
                    self.dependency_manager.add_dependency(task_id, dep_id)
            
            # Trigger task started event
            await self.automation_service.trigger_task_started(
                task_id=task_id,
                task_name=task_name,
                task_description=task_description,
                requirements=requirements or []
            )
            
            logger.info(f"Task {task_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start task {task_id}: {e}")
            return False
    
    async def update_task_progress(self, task_id: str, files_changed: List[str], 
                                 progress_notes: str = "") -> bool:
        """Update task progress with automatic Git operations"""
        try:
            logger.debug(f"Updating progress for task {task_id}")
            
            # Trigger task progress event
            await self.automation_service.trigger_task_progress(
                task_id=task_id,
                files_changed=files_changed,
                progress_notes=progress_notes
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update task progress for {task_id}: {e}")
            return False
    
    async def complete_task(self, task_id: str, completion_notes: str = "",
                          requirements_addressed: List[str] = None) -> bool:
        """Complete a task with full workflow integration"""
        try:
            logger.info(f"Completing task {task_id}")
            
            # Trigger task completed event
            await self.automation_service.trigger_task_completed(
                task_id=task_id,
                completion_notes=completion_notes,
                requirements_addressed=requirements_addressed or []
            )
            
            logger.info(f"Task {task_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False
    
    # Monitoring and health methods
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Get monitoring health
            repo_health = await self.monitor.perform_health_check()
            
            # Get automation service status
            automation_status = self.automation_service.get_service_status()
            
            # Get monitoring status
            monitoring_status = self.monitor.get_monitoring_status()
            
            return {
                'overall_status': repo_health.overall_status.value,
                'repository_health': repo_health.to_dict(),
                'automation_service': automation_status,
                'monitoring_system': monitoring_status,
                'system_running': self.running,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                'overall_status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status for a specific task"""
        try:
            # Get task mapping
            mapping = self.task_git_bridge.get_task_mapping(task_id)
            if not mapping:
                return None
            
            # Get Git metrics
            git_metrics = await self.task_git_bridge.get_task_git_metrics(task_id)
            
            # Get dependencies
            dependencies = self.dependency_manager.get_task_dependencies(task_id)
            dependents = self.dependency_manager.get_task_dependents(task_id)
            
            return {
                'task_id': task_id,
                'mapping': mapping.to_dict(),
                'git_metrics': git_metrics.to_dict() if git_metrics else None,
                'dependencies': dependencies,
                'dependents': dependents,
                'is_ready': task_id in self.dependency_manager.get_ready_tasks(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all tasks"""
        try:
            mappings = self.task_git_bridge.get_all_mappings()
            tasks = []
            
            for task_id, mapping in mappings.items():
                task_info = {
                    'task_id': task_id,
                    'status': mapping.status.value,
                    'branch_name': mapping.branch_name,
                    'created_at': mapping.created_at.isoformat(),
                    'completed_at': mapping.completed_at.isoformat() if mapping.completed_at else None,
                    'commits_count': len(mapping.commits),
                    'has_conflicts': bool(mapping.merge_conflicts),
                    'dependencies_count': len(mapping.dependencies)
                }
                tasks.append(task_info)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to get all tasks: {e}")
            return []
    
    # Recovery and maintenance methods
    
    async def trigger_recovery(self, action: str, target: str) -> str:
        """Manually trigger a recovery operation"""
        try:
            return await self.monitor.manual_recovery(action, target)
        except Exception as e:
            logger.error(f"Failed to trigger recovery: {e}")
            raise
    
    async def cleanup_stale_branches(self) -> bool:
        """Clean up stale branches"""
        try:
            return await self.monitor._cleanup_stale_branches()
        except Exception as e:
            logger.error(f"Failed to cleanup stale branches: {e}")
            return False
    
    # Dependency management methods
    
    def add_task_dependency(self, task_id: str, dependency_id: str) -> bool:
        """Add a dependency between tasks"""
        return self.dependency_manager.add_dependency(task_id, dependency_id)
    
    def remove_task_dependency(self, task_id: str, dependency_id: str) -> bool:
        """Remove a dependency between tasks"""
        return self.dependency_manager.remove_dependency(task_id, dependency_id)
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to be worked on"""
        return self.dependency_manager.get_ready_tasks()
    
    def get_critical_path(self) -> List[str]:
        """Get the critical path through all tasks"""
        return self.dependency_manager.calculate_critical_path()
    
    async def get_merge_strategy(self, task_ids: List[str]) -> Dict[str, Any]:
        """Get merge strategy for a set of tasks"""
        try:
            strategy = await self.dependency_manager.generate_merge_strategy(task_ids)
            return strategy.to_dict()
        except Exception as e:
            logger.error(f"Failed to get merge strategy: {e}")
            return {}
    
    # Event and history methods
    
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent workflow events"""
        return self.automation_service.get_event_history(limit)
    
    def get_health_history(self, limit: int = 24) -> List[Dict[str, Any]]:
        """Get recent health check history"""
        return self.monitor.get_health_history(limit)
    
    def get_active_recoveries(self) -> List[Dict[str, Any]]:
        """Get currently active recovery operations"""
        return self.monitor.get_active_recoveries()
    
    # Manual workflow operations
    
    async def manual_commit(self, task_id: str, message: str, files: List[str] = None) -> Optional[str]:
        """Manually commit changes for a task"""
        try:
            commit_hash = self.git_manager.commit_task_progress(
                task_id=task_id,
                files=files or [],
                message=message
            )
            
            # Update task status
            await self.task_git_bridge.update_task_status_from_git(commit_hash)
            
            return commit_hash
            
        except Exception as e:
            logger.error(f"Failed to manually commit for task {task_id}: {e}")
            return None
    
    async def manual_branch_creation(self, task_id: str, task_name: str) -> Optional[str]:
        """Manually create a branch for a task"""
        try:
            branch_name = self.git_manager.create_task_branch(task_id, task_name)
            await self.task_git_bridge.link_task_to_branch(task_id, branch_name)
            return branch_name
        except Exception as e:
            logger.error(f"Failed to manually create branch for task {task_id}: {e}")
            return None
    
    # Configuration methods
    
    def update_automation_config(self, config: Dict[str, Any]) -> bool:
        """Update automation service configuration"""
        try:
            self.automation_service.config.update(config)
            self.automation_service._save_config(self.automation_service.config)
            return True
        except Exception as e:
            logger.error(f"Failed to update automation config: {e}")
            return False
    
    def update_monitoring_config(self, config: Dict[str, Any]) -> bool:
        """Update monitoring system configuration"""
        try:
            self.monitor.config.update(config)
            self.monitor._save_monitoring_config(self.monitor.config)
            return True
        except Exception as e:
            logger.error(f"Failed to update monitoring config: {e}")
            return False
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get current system configuration"""
        return {
            'automation': self.automation_service.config,
            'monitoring': self.monitor.config,
            'git_manager': {
                'repo_path': str(self.git_manager.repo_path),
                'task_tracking_file': str(self.git_manager.task_tracking_file)
            }
        }
    
    # Utility methods
    
    async def validate_system(self) -> Dict[str, Any]:
        """Validate the entire workflow system"""
        validation_results = {
            'git_repository': False,
            'task_mappings': False,
            'dependencies': False,
            'automation_service': False,
            'monitoring_system': False,
            'overall_valid': False
        }
        
        try:
            # Validate Git repository
            git_status = self.git_manager.get_git_status()
            validation_results['git_repository'] = True
            
            # Validate task mappings
            mappings = self.task_git_bridge.get_all_mappings()
            validation_results['task_mappings'] = True
            
            # Validate dependencies
            ready_tasks = self.dependency_manager.get_ready_tasks()
            validation_results['dependencies'] = True
            
            # Validate automation service
            automation_status = self.automation_service.get_service_status()
            validation_results['automation_service'] = automation_status['running']
            
            # Validate monitoring system
            monitoring_status = self.monitor.get_monitoring_status()
            validation_results['monitoring_system'] = monitoring_status['monitoring_enabled']
            
            # Overall validation
            validation_results['overall_valid'] = all([
                validation_results['git_repository'],
                validation_results['task_mappings'],
                validation_results['dependencies']
            ])
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    async def reset_system(self) -> bool:
        """Reset the workflow system to a clean state"""
        try:
            logger.warning("Resetting Git workflow system")
            
            # Stop the system
            await self.stop()
            
            # Clear task mappings (with backup)
            mappings = self.task_git_bridge.get_all_mappings()
            backup_file = f".kiro/task_mappings_backup_{int(datetime.now().timestamp())}.json"
            
            import json
            with open(backup_file, 'w') as f:
                json.dump({k: v.to_dict() for k, v in mappings.items()}, f, indent=2)
            
            # Clear dependency graph
            self.dependency_manager.dependency_graph.clear()
            
            # Clear event history
            self.automation_service.event_history.clear()
            
            # Clear health history
            self.monitor.health_history.clear()
            
            # Restart the system
            await self.start()
            
            logger.info(f"System reset completed. Backup saved to {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset system: {e}")
            return False