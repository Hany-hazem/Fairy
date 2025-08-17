"""
Task-Git Integration Bridge

This module provides the bridge between task management and Git operations,
enabling automatic tracking of task progress through version control.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import asyncio
import subprocess
from dataclasses import asdict

from .task_git_models import (
    TaskGitMapping, TaskStatus, MergeStatus, GitCommit, 
    TaskGitMetrics, TaskReport, MergeStrategy
)
from .git_workflow_manager import GitWorkflowManager


logger = logging.getLogger(__name__)


class TaskGitBridge:
    """Bridge between task management system and Git operations"""
    
    def __init__(self, 
                 git_manager: GitWorkflowManager,
                 storage_path: str = ".kiro/task_git_mappings.json"):
        self.git_manager = git_manager
        self.storage_path = Path(storage_path)
        self.mappings: Dict[str, TaskGitMapping] = {}
        self._load_mappings()
    
    def _load_mappings(self) -> None:
        """Load task-git mappings from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.mappings = {
                        task_id: TaskGitMapping.from_dict(mapping_data)
                        for task_id, mapping_data in data.items()
                    }
                logger.info(f"Loaded {len(self.mappings)} task-git mappings")
            else:
                # Create directory if it doesn't exist
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("No existing mappings found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load task-git mappings: {e}")
            self.mappings = {}
    
    def _save_mappings(self) -> None:
        """Save task-git mappings to storage"""
        try:
            data = {
                task_id: mapping.to_dict()
                for task_id, mapping in self.mappings.items()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.mappings)} task-git mappings")
        except Exception as e:
            logger.error(f"Failed to save task-git mappings: {e}")
    
    async def link_task_to_branch(self, task_id: str, branch_name: str) -> bool:
        """Link a task to a Git branch"""
        try:
            # Get current commit as start commit
            start_commit = await self._get_current_commit()
            
            # Create or update mapping
            if task_id in self.mappings:
                mapping = self.mappings[task_id]
                mapping.branch_name = branch_name
                mapping.start_commit = start_commit
            else:
                mapping = TaskGitMapping(
                    task_id=task_id,
                    branch_name=branch_name,
                    start_commit=start_commit,
                    status=TaskStatus.IN_PROGRESS
                )
                self.mappings[task_id] = mapping
            
            self._save_mappings()
            logger.info(f"Linked task {task_id} to branch {branch_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to link task {task_id} to branch {branch_name}: {e}")
            return False
    
    async def update_task_status_from_git(self, commit_hash: str) -> bool:
        """Update task status based on Git commit"""
        try:
            # Get commit details
            commit_info = await self._get_commit_info(commit_hash)
            if not commit_info:
                return False
            
            # Extract task ID from commit message
            task_id = self._extract_task_id_from_commit(commit_info['message'])
            if not task_id or task_id not in self.mappings:
                return False
            
            mapping = self.mappings[task_id]
            
            # Add commit to mapping
            if commit_hash not in mapping.commits:
                mapping.commits.append(commit_hash)
            
            # Update status based on commit message patterns
            message = commit_info['message'].lower()
            if 'complete' in message or 'finish' in message:
                mapping.status = TaskStatus.COMPLETED
                mapping.completion_commit = commit_hash
                mapping.completed_at = datetime.now()
            elif 'start' in message or 'begin' in message:
                mapping.status = TaskStatus.IN_PROGRESS
            
            self._save_mappings()
            logger.info(f"Updated task {task_id} status from commit {commit_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update task status from commit {commit_hash}: {e}")
            return False
    
    async def generate_task_completion_report(self, task_id: str) -> Optional[TaskReport]:
        """Generate comprehensive task completion report"""
        try:
            if task_id not in self.mappings:
                logger.error(f"Task {task_id} not found in mappings")
                return None
            
            mapping = self.mappings[task_id]
            
            # Get Git metrics
            git_metrics = await self._calculate_git_metrics(task_id)
            
            # Get commit details
            commits = []
            for commit_hash in mapping.commits:
                commit_info = await self._get_commit_info(commit_hash)
                if commit_info:
                    commits.append(GitCommit(
                        hash=commit_hash,
                        message=commit_info['message'],
                        author=commit_info['author'],
                        timestamp=datetime.fromisoformat(commit_info['timestamp']),
                        files_changed=commit_info['files_changed'],
                        task_id=task_id,
                        requirement_refs=mapping.requirement_refs
                    ))
            
            # Generate report
            report = TaskReport(
                task_id=task_id,
                task_description=f"Task {task_id}",  # Could be enhanced with actual description
                status=mapping.status,
                git_metrics=git_metrics,
                commits=commits,
                requirements_covered=mapping.requirement_refs,
                dependencies_resolved=mapping.dependencies,
                merge_conflicts_resolved=mapping.merge_conflicts,
                completion_notes=f"Task completed with {len(commits)} commits"
            )
            
            logger.info(f"Generated completion report for task {task_id}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate completion report for task {task_id}: {e}")
            return None
    
    async def handle_task_dependency_merge(self, dependent_tasks: List[str]) -> MergeStrategy:
        """Handle merge strategy for dependent tasks"""
        try:
            # Use the more sophisticated dependency manager if available
            try:
                from .task_dependency_manager import TaskDependencyManager
                dependency_manager = TaskDependencyManager(self, self.git_manager)
                return await dependency_manager.generate_merge_strategy(dependent_tasks)
            except ImportError:
                logger.warning("TaskDependencyManager not available, using basic strategy")
            
            # Fallback to basic implementation
            # Analyze dependencies
            dependency_graph = self._build_dependency_graph(dependent_tasks)
            
            # Calculate merge order
            merge_order = self._calculate_merge_order(dependency_graph)
            
            # Identify parallel groups
            parallel_groups = self._identify_parallel_groups(dependency_graph, merge_order)
            
            # Assess conflicts
            conflict_resolution = await self._assess_merge_conflicts(dependent_tasks)
            
            # Estimate duration and risk
            estimated_duration = len(dependent_tasks) * 15  # 15 minutes per task
            risk_level = self._assess_merge_risk(dependent_tasks, conflict_resolution)
            
            strategy = MergeStrategy(
                merge_order=merge_order,
                parallel_groups=parallel_groups,
                conflict_resolution=conflict_resolution,
                estimated_duration=estimated_duration,
                risk_level=risk_level
            )
            
            logger.info(f"Generated merge strategy for {len(dependent_tasks)} tasks")
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to handle task dependency merge: {e}")
            # Return safe fallback strategy
            return MergeStrategy(
                merge_order=dependent_tasks,
                parallel_groups=[],
                conflict_resolution={},
                estimated_duration=len(dependent_tasks) * 20,
                risk_level="high"
            )
    
    async def get_task_git_metrics(self, task_id: str) -> Optional[TaskGitMetrics]:
        """Get Git metrics for a specific task"""
        try:
            return await self._calculate_git_metrics(task_id)
        except Exception as e:
            logger.error(f"Failed to get Git metrics for task {task_id}: {e}")
            return None
    
    def get_task_mapping(self, task_id: str) -> Optional[TaskGitMapping]:
        """Get task-git mapping for a specific task"""
        return self.mappings.get(task_id)
    
    def get_all_mappings(self) -> Dict[str, TaskGitMapping]:
        """Get all task-git mappings"""
        return self.mappings.copy()
    
    def add_task_dependency(self, task_id: str, dependency_task_id: str) -> bool:
        """Add a dependency relationship between tasks"""
        try:
            if task_id not in self.mappings:
                logger.error(f"Task {task_id} not found")
                return False
            
            mapping = self.mappings[task_id]
            if dependency_task_id not in mapping.dependencies:
                mapping.dependencies.append(dependency_task_id)
                self._save_mappings()
                logger.info(f"Added dependency {dependency_task_id} to task {task_id}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to add task dependency: {e}")
            return False
    
    def add_requirement_reference(self, task_id: str, requirement_ref: str) -> bool:
        """Add a requirement reference to a task"""
        try:
            if task_id not in self.mappings:
                logger.error(f"Task {task_id} not found")
                return False
            
            mapping = self.mappings[task_id]
            if requirement_ref not in mapping.requirement_refs:
                mapping.requirement_refs.append(requirement_ref)
                self._save_mappings()
                logger.info(f"Added requirement {requirement_ref} to task {task_id}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to add requirement reference: {e}")
            return False
    
    # Private helper methods
    
    async def _get_current_commit(self) -> Optional[str]:
        """Get current Git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    async def _get_commit_info(self, commit_hash: str) -> Optional[Dict]:
        """Get detailed information about a commit"""
        try:
            # Get commit message and author
            result = subprocess.run([
                'git', 'show', '--format=%H|%an|%ai|%s', '--name-only', commit_hash
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            if not lines:
                return None
            
            # Parse header
            header_parts = lines[0].split('|')
            if len(header_parts) < 4:
                return None
            
            # Get changed files (skip empty lines)
            files_changed = [line for line in lines[1:] if line.strip()]
            
            return {
                'hash': header_parts[0],
                'author': header_parts[1],
                'timestamp': header_parts[2],
                'message': header_parts[3],
                'files_changed': files_changed
            }
        except subprocess.CalledProcessError:
            return None
    
    def _extract_task_id_from_commit(self, commit_message: str) -> Optional[str]:
        """Extract task ID from commit message"""
        # Look for patterns like "Task 1.1", "task-1-1", etc.
        import re
        patterns = [
            r'[Tt]ask[:\s]+(\d+(?:\.\d+)*)',
            r'[Tt]ask-(\d+(?:-\d+)*)',
            r'#(\d+(?:\.\d+)*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, commit_message)
            if match:
                return match.group(1).replace('-', '.')
        
        return None
    
    async def _calculate_git_metrics(self, task_id: str) -> TaskGitMetrics:
        """Calculate Git metrics for a task"""
        if task_id not in self.mappings:
            raise ValueError(f"Task {task_id} not found")
        
        mapping = self.mappings[task_id]
        
        # Calculate basic metrics
        total_commits = len(mapping.commits)
        files_modified = 0
        lines_added = 0
        lines_deleted = 0
        
        # Get detailed stats for each commit
        for commit_hash in mapping.commits:
            try:
                result = subprocess.run([
                    'git', 'show', '--stat', '--format=', commit_hash
                ], capture_output=True, text=True, check=True)
                
                # Parse git stat output
                for line in result.stdout.split('\n'):
                    if '|' in line and ('+' in line or '-' in line):
                        files_modified += 1
                    elif 'insertions' in line or 'deletions' in line:
                        # Parse summary line
                        parts = line.split(',')
                        for part in parts:
                            if 'insertion' in part:
                                lines_added += int(part.strip().split()[0])
                            elif 'deletion' in part:
                                lines_deleted += int(part.strip().split()[0])
            except subprocess.CalledProcessError:
                continue
        
        # Calculate duration
        duration_hours = 0.0
        if mapping.created_at and mapping.completed_at:
            duration = mapping.completed_at - mapping.created_at
            duration_hours = duration.total_seconds() / 3600
        
        # Calculate branch age
        branch_age_days = 0
        if mapping.created_at:
            age = datetime.now() - mapping.created_at
            branch_age_days = age.days
        
        return TaskGitMetrics(
            task_id=task_id,
            total_commits=total_commits,
            files_modified=files_modified,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            duration_hours=duration_hours,
            branch_age_days=branch_age_days,
            merge_conflicts_count=len(mapping.merge_conflicts),
            dependency_count=len(mapping.dependencies)
        )
    
    def _build_dependency_graph(self, task_ids: List[str]) -> Dict[str, List[str]]:
        """Build dependency graph for tasks"""
        graph = {}
        for task_id in task_ids:
            if task_id in self.mappings:
                graph[task_id] = self.mappings[task_id].dependencies
            else:
                graph[task_id] = []
        return graph
    
    def _calculate_merge_order(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Calculate optimal merge order using topological sort"""
        # Simple topological sort implementation
        in_degree = {task: 0 for task in dependency_graph}
        
        # Calculate in-degrees
        for task, deps in dependency_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[task] += 1
        
        # Process tasks with no dependencies first
        queue = [task for task, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            task = queue.pop(0)
            result.append(task)
            
            # Update in-degrees for dependent tasks
            for other_task, deps in dependency_graph.items():
                if task in deps:
                    in_degree[other_task] -= 1
                    if in_degree[other_task] == 0:
                        queue.append(other_task)
        
        return result
    
    def _identify_parallel_groups(self, dependency_graph: Dict[str, List[str]], 
                                merge_order: List[str]) -> List[List[str]]:
        """Identify tasks that can be merged in parallel"""
        parallel_groups = []
        processed = set()
        
        for task in merge_order:
            if task in processed:
                continue
            
            # Find tasks that can be processed in parallel with this one
            parallel_group = [task]
            for other_task in merge_order:
                if (other_task not in processed and 
                    other_task != task and
                    not self._has_dependency_path(dependency_graph, task, other_task) and
                    not self._has_dependency_path(dependency_graph, other_task, task)):
                    parallel_group.append(other_task)
            
            if len(parallel_group) > 1:
                parallel_groups.append(parallel_group)
                processed.update(parallel_group)
            else:
                processed.add(task)
        
        return parallel_groups
    
    def _has_dependency_path(self, graph: Dict[str, List[str]], 
                           start: str, end: str) -> bool:
        """Check if there's a dependency path between two tasks"""
        if start == end:
            return True
        
        visited = set()
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current == end:
                return True
            
            for dep in graph.get(current, []):
                if dep not in visited:
                    queue.append(dep)
        
        return False
    
    async def _assess_merge_conflicts(self, task_ids: List[str]) -> Dict[str, str]:
        """Assess potential merge conflicts between tasks"""
        conflicts = {}
        
        # Simple conflict detection based on file overlap
        file_map = {}
        for task_id in task_ids:
            if task_id in self.mappings:
                mapping = self.mappings[task_id]
                for commit_hash in mapping.commits:
                    commit_info = await self._get_commit_info(commit_hash)
                    if commit_info:
                        for file_path in commit_info['files_changed']:
                            if file_path not in file_map:
                                file_map[file_path] = []
                            file_map[file_path].append(task_id)
        
        # Identify conflicts
        for file_path, tasks in file_map.items():
            if len(tasks) > 1:
                conflict_key = f"{file_path}:{'-'.join(sorted(tasks))}"
                conflicts[conflict_key] = "manual_review"
        
        return conflicts
    
    def _assess_merge_risk(self, task_ids: List[str], 
                         conflicts: Dict[str, str]) -> str:
        """Assess overall merge risk level"""
        if len(conflicts) > 5:
            return "high"
        elif len(conflicts) > 2:
            return "medium"
        elif len(task_ids) > 10:
            return "medium"
        else:
            return "low"