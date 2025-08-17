"""
Task Dependency and Merge Management

This module provides advanced dependency tracking and intelligent merge ordering
for task-based Git workflows.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import asyncio

from .task_git_models import TaskGitMapping, TaskStatus, MergeStrategy, MergeStatus
from .task_git_bridge import TaskGitBridge
from .git_workflow_manager import GitWorkflowManager


logger = logging.getLogger(__name__)


@dataclass
class DependencyNode:
    """Represents a task node in the dependency graph"""
    task_id: str
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.NOT_STARTED
    priority: int = 0
    estimated_duration: int = 60  # minutes
    
    def is_ready_for_merge(self) -> bool:
        """Check if task is ready for merge (all dependencies completed)"""
        return self.status == TaskStatus.COMPLETED


@dataclass
class MergeConflictAnalysis:
    """Analysis of potential merge conflicts"""
    conflicting_files: Dict[str, List[str]]  # file -> list of tasks
    conflict_severity: str  # low, medium, high
    resolution_strategy: str  # auto, manual, staged
    estimated_resolution_time: int  # minutes


class TaskDependencyManager:
    """Manages task dependencies and intelligent merge ordering"""
    
    def __init__(self, task_git_bridge: TaskGitBridge, git_manager: GitWorkflowManager):
        self.task_git_bridge = task_git_bridge
        self.git_manager = git_manager
        self.dependency_graph: Dict[str, DependencyNode] = {}
        self._load_dependencies()
    
    def _load_dependencies(self) -> None:
        """Load existing dependencies from task-git mappings"""
        mappings = self.task_git_bridge.get_all_mappings()
        
        for task_id, mapping in mappings.items():
            node = DependencyNode(
                task_id=task_id,
                dependencies=set(mapping.dependencies),
                status=mapping.status
            )
            self.dependency_graph[task_id] = node
        
        # Build reverse dependencies (dependents)
        for task_id, node in self.dependency_graph.items():
            for dep_id in node.dependencies:
                if dep_id in self.dependency_graph:
                    self.dependency_graph[dep_id].dependents.add(task_id)
        
        logger.info(f"Loaded {len(self.dependency_graph)} tasks with dependencies")
    
    def add_dependency(self, task_id: str, dependency_id: str) -> bool:
        """Add a dependency relationship between tasks"""
        try:
            # Ensure both tasks exist in the graph
            if task_id not in self.dependency_graph:
                self.dependency_graph[task_id] = DependencyNode(task_id=task_id)
            if dependency_id not in self.dependency_graph:
                self.dependency_graph[dependency_id] = DependencyNode(task_id=dependency_id)
            
            # Check for circular dependencies
            if self._would_create_cycle(task_id, dependency_id):
                logger.error(f"Cannot add dependency {dependency_id} to {task_id}: would create cycle")
                return False
            
            # Add the dependency
            self.dependency_graph[task_id].dependencies.add(dependency_id)
            self.dependency_graph[dependency_id].dependents.add(task_id)
            
            # Update task-git bridge
            self.task_git_bridge.add_task_dependency(task_id, dependency_id)
            
            logger.info(f"Added dependency: {task_id} depends on {dependency_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            return False
    
    def remove_dependency(self, task_id: str, dependency_id: str) -> bool:
        """Remove a dependency relationship between tasks"""
        try:
            if (task_id in self.dependency_graph and 
                dependency_id in self.dependency_graph[task_id].dependencies):
                
                self.dependency_graph[task_id].dependencies.remove(dependency_id)
                self.dependency_graph[dependency_id].dependents.remove(task_id)
                
                logger.info(f"Removed dependency: {task_id} no longer depends on {dependency_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove dependency: {e}")
            return False
    
    def get_task_dependencies(self, task_id: str) -> List[str]:
        """Get all dependencies for a task"""
        if task_id in self.dependency_graph:
            return list(self.dependency_graph[task_id].dependencies)
        return []
    
    def get_task_dependents(self, task_id: str) -> List[str]:
        """Get all tasks that depend on this task"""
        if task_id in self.dependency_graph:
            return list(self.dependency_graph[task_id].dependents)
        return []
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to be worked on (all dependencies completed)"""
        ready_tasks = []
        
        for task_id, node in self.dependency_graph.items():
            if node.status in [TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS]:
                # Check if all dependencies are completed
                all_deps_completed = all(
                    self.dependency_graph.get(dep_id, DependencyNode(task_id=dep_id)).status == TaskStatus.COMPLETED
                    for dep_id in node.dependencies
                )
                
                if all_deps_completed:
                    ready_tasks.append(task_id)
        
        return ready_tasks
    
    def calculate_critical_path(self) -> List[str]:
        """Calculate the critical path through the dependency graph"""
        # Use longest path algorithm for critical path
        distances = {}
        predecessors = {}
        
        # Initialize distances
        for task_id in self.dependency_graph:
            distances[task_id] = 0
            predecessors[task_id] = None
        
        # Topological sort with distance calculation
        sorted_tasks = self._topological_sort()
        
        for task_id in sorted_tasks:
            node = self.dependency_graph[task_id]
            current_distance = distances[task_id]
            
            for dependent_id in node.dependents:
                if dependent_id in self.dependency_graph:
                    dependent_node = self.dependency_graph[dependent_id]
                    new_distance = current_distance + node.estimated_duration
                    
                    if new_distance > distances[dependent_id]:
                        distances[dependent_id] = new_distance
                        predecessors[dependent_id] = task_id
        
        # Find the task with maximum distance (end of critical path)
        max_distance = max(distances.values()) if distances else 0
        end_task = None
        for task_id, distance in distances.items():
            if distance == max_distance:
                end_task = task_id
                break
        
        # Reconstruct critical path
        critical_path = []
        current = end_task
        while current is not None:
            critical_path.append(current)
            current = predecessors[current]
        
        critical_path.reverse()
        return critical_path
    
    async def generate_merge_strategy(self, task_ids: List[str]) -> MergeStrategy:
        """Generate intelligent merge strategy for a set of tasks"""
        try:
            # Filter to only include tasks that exist in our graph
            valid_tasks = [tid for tid in task_ids if tid in self.dependency_graph]
            
            if not valid_tasks:
                return MergeStrategy(
                    merge_order=[],
                    parallel_groups=[],
                    conflict_resolution={},
                    estimated_duration=0,
                    risk_level="low"
                )
            
            # Calculate merge order based on dependencies
            merge_order = self._calculate_merge_order(valid_tasks)
            
            # Identify parallel merge groups
            parallel_groups = self._identify_parallel_groups(valid_tasks, merge_order)
            
            # Analyze potential conflicts
            conflict_analysis = await self._analyze_merge_conflicts(valid_tasks)
            
            # Calculate estimated duration
            estimated_duration = self._estimate_merge_duration(valid_tasks, conflict_analysis)
            
            # Assess risk level
            risk_level = self._assess_merge_risk(valid_tasks, conflict_analysis)
            
            strategy = MergeStrategy(
                merge_order=merge_order,
                parallel_groups=parallel_groups,
                conflict_resolution=conflict_analysis.conflicting_files,
                estimated_duration=estimated_duration,
                risk_level=risk_level
            )
            
            logger.info(f"Generated merge strategy for {len(valid_tasks)} tasks")
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to generate merge strategy: {e}")
            return MergeStrategy(
                merge_order=task_ids,
                parallel_groups=[],
                conflict_resolution={},
                estimated_duration=len(task_ids) * 20,
                risk_level="high"
            )
    
    def get_dependency_graph_visualization(self) -> Dict[str, Any]:
        """Get dependency graph data for visualization"""
        nodes = []
        edges = []
        
        for task_id, node in self.dependency_graph.items():
            nodes.append({
                "id": task_id,
                "status": node.status.value,
                "priority": node.priority,
                "estimated_duration": node.estimated_duration
            })
            
            for dep_id in node.dependencies:
                edges.append({
                    "from": dep_id,
                    "to": task_id,
                    "type": "dependency"
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "critical_path": self.calculate_critical_path()
        }
    
    # Private helper methods
    
    def _would_create_cycle(self, task_id: str, dependency_id: str) -> bool:
        """Check if adding a dependency would create a cycle"""
        # Use DFS to check if dependency_id can reach task_id
        visited = set()
        stack = [dependency_id]
        
        while stack:
            current = stack.pop()
            if current == task_id:
                return True
            
            if current in visited:
                continue
            visited.add(current)
            
            if current in self.dependency_graph:
                stack.extend(self.dependency_graph[current].dependencies)
        
        return False
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort of the dependency graph"""
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for task_id, node in self.dependency_graph.items():
            for dep_id in node.dependencies:
                in_degree[task_id] += 1
        
        # Initialize queue with tasks having no dependencies
        queue = deque([task_id for task_id in self.dependency_graph if in_degree[task_id] == 0])
        result = []
        
        while queue:
            task_id = queue.popleft()
            result.append(task_id)
            
            # Update in-degrees for dependents
            if task_id in self.dependency_graph:
                for dependent_id in self.dependency_graph[task_id].dependents:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        queue.append(dependent_id)
        
        return result
    
    def _calculate_merge_order(self, task_ids: List[str]) -> List[str]:
        """Calculate optimal merge order for given tasks"""
        # Create subgraph with only the specified tasks
        subgraph = {tid: self.dependency_graph[tid] for tid in task_ids if tid in self.dependency_graph}
        
        # Perform topological sort on subgraph
        in_degree = defaultdict(int)
        for task_id, node in subgraph.items():
            for dep_id in node.dependencies:
                if dep_id in subgraph:
                    in_degree[task_id] += 1
        
        queue = deque([task_id for task_id in subgraph if in_degree[task_id] == 0])
        result = []
        
        while queue:
            task_id = queue.popleft()
            result.append(task_id)
            
            for dependent_id in subgraph[task_id].dependents:
                if dependent_id in subgraph:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        queue.append(dependent_id)
        
        return result
    
    def _identify_parallel_groups(self, task_ids: List[str], merge_order: List[str]) -> List[List[str]]:
        """Identify tasks that can be merged in parallel"""
        parallel_groups = []
        processed = set()
        
        for task_id in merge_order:
            if task_id in processed:
                continue
            
            # Find tasks that can be processed in parallel
            parallel_group = [task_id]
            
            for other_task in merge_order:
                if (other_task not in processed and 
                    other_task != task_id and
                    not self._has_dependency_path(task_id, other_task) and
                    not self._has_dependency_path(other_task, task_id)):
                    parallel_group.append(other_task)
            
            if len(parallel_group) > 1:
                parallel_groups.append(parallel_group)
                processed.update(parallel_group)
            else:
                processed.add(task_id)
        
        return parallel_groups
    
    def _has_dependency_path(self, start_task: str, end_task: str) -> bool:
        """Check if there's a dependency path between two tasks"""
        if start_task == end_task:
            return True
        
        visited = set()
        stack = [start_task]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            if current == end_task:
                return True
            
            if current in self.dependency_graph:
                stack.extend(self.dependency_graph[current].dependencies)
                stack.extend(self.dependency_graph[current].dependents)
        
        return False
    
    async def _analyze_merge_conflicts(self, task_ids: List[str]) -> MergeConflictAnalysis:
        """Analyze potential merge conflicts between tasks"""
        conflicting_files = defaultdict(list)
        
        # Get file changes for each task
        for task_id in task_ids:
            mapping = self.task_git_bridge.get_task_mapping(task_id)
            if mapping:
                for commit_hash in mapping.commits:
                    try:
                        # Get files changed in this commit
                        import subprocess
                        result = subprocess.run([
                            'git', 'show', '--name-only', '--format=', commit_hash
                        ], capture_output=True, text=True, check=True)
                        
                        files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
                        for file_path in files:
                            conflicting_files[file_path].append(task_id)
                    except subprocess.CalledProcessError:
                        continue
        
        # Identify actual conflicts (files modified by multiple tasks)
        actual_conflicts = {
            file_path: tasks for file_path, tasks in conflicting_files.items()
            if len(tasks) > 1
        }
        
        # Assess conflict severity
        conflict_count = len(actual_conflicts)
        if conflict_count == 0:
            severity = "low"
        elif conflict_count <= 3:
            severity = "medium"
        else:
            severity = "high"
        
        # Determine resolution strategy
        if conflict_count == 0:
            resolution_strategy = "auto"
        elif conflict_count <= 2:
            resolution_strategy = "manual"
        else:
            resolution_strategy = "staged"
        
        # Estimate resolution time
        resolution_time = conflict_count * 15  # 15 minutes per conflict
        
        return MergeConflictAnalysis(
            conflicting_files=actual_conflicts,
            conflict_severity=severity,
            resolution_strategy=resolution_strategy,
            estimated_resolution_time=resolution_time
        )
    
    def _estimate_merge_duration(self, task_ids: List[str], 
                               conflict_analysis: MergeConflictAnalysis) -> int:
        """Estimate total merge duration in minutes"""
        base_time = len(task_ids) * 10  # 10 minutes per task
        conflict_time = conflict_analysis.estimated_resolution_time
        complexity_factor = 1.2 if len(task_ids) > 5 else 1.0
        
        return int((base_time + conflict_time) * complexity_factor)
    
    def _assess_merge_risk(self, task_ids: List[str], 
                         conflict_analysis: MergeConflictAnalysis) -> str:
        """Assess overall merge risk level"""
        risk_factors = 0
        
        # Number of tasks
        if len(task_ids) > 10:
            risk_factors += 2
        elif len(task_ids) > 5:
            risk_factors += 1
        
        # Conflict severity
        if conflict_analysis.conflict_severity == "high":
            risk_factors += 3
        elif conflict_analysis.conflict_severity == "medium":
            risk_factors += 1
        
        # Complex dependency chains
        max_chain_length = max(
            len(self.get_task_dependencies(task_id)) for task_id in task_ids
        ) if task_ids else 0
        
        if max_chain_length > 3:
            risk_factors += 2
        elif max_chain_length > 1:
            risk_factors += 1
        
        # Determine risk level
        if risk_factors >= 5:
            return "high"
        elif risk_factors >= 2:
            return "medium"
        else:
            return "low"