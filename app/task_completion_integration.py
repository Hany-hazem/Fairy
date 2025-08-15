"""
Integration module for task completion with Git workflow automation.

This module provides seamless integration between task completion tracking
and the automated Git workflow system.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.git_workflow_manager import GitWorkflowManager, TaskContext


class TaskCompletionIntegrator:
    """
    Integrates task completion with automated Git workflows.
    
    This class provides high-level methods for completing tasks with
    proper Git integration, spec file updates, and progress tracking.
    """
    
    def __init__(self, repo_path: str = "."):
        """
        Initialize the task completion integrator.
        
        Args:
            repo_path: Path to the Git repository
        """
        self.repo_path = Path(repo_path)
        self.git_manager = GitWorkflowManager(repo_path)
        self.specs_dir = self.repo_path / ".kiro" / "specs"
    
    def find_spec_files(self) -> List[Path]:
        """
        Find all specification files in the project.
        
        Returns:
            List of paths to specification files
        """
        spec_files = []
        if self.specs_dir.exists():
            for spec_file in self.specs_dir.rglob("tasks.md"):
                spec_files.append(spec_file)
        return spec_files
    
    def parse_task_from_spec(self, spec_file: Path, task_id: str) -> Optional[TaskContext]:
        """
        Parse task information from a specification file.
        
        Args:
            spec_file: Path to the specification file
            task_id: ID of the task to parse
            
        Returns:
            TaskContext object if task found, None otherwise
        """
        if not spec_file.exists():
            return None
        
        content = spec_file.read_text()
        lines = content.split('\n')
        
        task_name = ""
        description_parts = []
        requirements = []
        in_task = False
        
        # Pattern to match task lines: - [ ] 1.1 Task name
        task_pattern = rf"^- \[[ x]\] {re.escape(task_id)}\.?\s+(.+)$"
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if this line matches our task
            match = re.match(task_pattern, line_stripped)
            if match:
                in_task = True
                task_name = match.group(1).strip()
                continue
            
            if in_task:
                # Stop when we hit the next task or major section
                if (line_stripped.startswith("- [ ]") or 
                    line_stripped.startswith("- [x]") or
                    line_stripped.startswith("## ")):
                    break
                
                # Collect description lines (indented bullet points)
                if line_stripped.startswith("- ") and not line_stripped.startswith("- [ ]"):
                    desc_text = line_stripped[2:].strip()
                    if not desc_text.startswith("_Requirements:"):
                        description_parts.append(desc_text)
                
                # Extract requirements
                if "_Requirements:" in line_stripped:
                    req_part = line_stripped.split("_Requirements:", 1)[1].strip("_ ")
                    requirements.extend([r.strip() for r in req_part.split(",")])
        
        if not task_name:
            return None
        
        return TaskContext(
            task_id=task_id,
            task_name=task_name,
            description=" ".join(description_parts),
            files_modified=[],  # Will be populated later
            requirements_addressed=requirements,
            completion_time=datetime.now()
        )
    
    def update_task_status_in_spec(self, spec_file: Path, task_id: str, completed: bool = True):
        """
        Update task status in specification file.
        
        Args:
            spec_file: Path to the specification file
            task_id: ID of the task to update
            completed: Whether to mark as completed (True) or in progress (False)
        """
        if not spec_file.exists():
            raise FileNotFoundError(f"Specification file not found: {spec_file}")
        
        content = spec_file.read_text()
        
        # Pattern to match the task line
        if completed:
            old_pattern = f"- [ ] {task_id}"
            new_pattern = f"- [x] {task_id}"
        else:
            old_pattern = f"- [ ] {task_id}"
            new_pattern = f"- [-] {task_id}"  # In progress marker
        
        if old_pattern in content:
            updated_content = content.replace(old_pattern, new_pattern)
            spec_file.write_text(updated_content)
        else:
            # Try to find the task with different status
            for status in ["[ ]", "[x]", "[-]"]:
                pattern = f"- {status} {task_id}"
                if pattern in content:
                    if completed:
                        replacement = f"- [x] {task_id}"
                    else:
                        replacement = f"- [-] {task_id}"
                    updated_content = content.replace(pattern, replacement)
                    spec_file.write_text(updated_content)
                    return
            
            raise ValueError(f"Task {task_id} not found in {spec_file}")
    
    def complete_task_by_id(self, task_id: str, spec_file: Optional[Path] = None, 
                           auto_push: bool = True) -> str:
        """
        Complete a task by its ID, automatically detecting context from spec files.
        
        Args:
            task_id: ID of the task to complete
            spec_file: Specific spec file to use (if None, searches all)
            auto_push: Whether to automatically push to remote
            
        Returns:
            Commit hash of the completion commit
        """
        # Find the task in spec files
        task_context = None
        target_spec_file = None
        
        if spec_file:
            task_context = self.parse_task_from_spec(spec_file, task_id)
            target_spec_file = spec_file
        else:
            # Search all spec files
            for spec_path in self.find_spec_files():
                task_context = self.parse_task_from_spec(spec_path, task_id)
                if task_context:
                    target_spec_file = spec_path
                    break
        
        if not task_context:
            raise ValueError(f"Task {task_id} not found in any specification file")
        
        # Get modified files
        modified_files = self.git_manager.get_modified_files()
        task_context.files_modified = modified_files
        
        # Commit the task completion
        commit_hash = self.git_manager.commit_task_completion(task_context, auto_push)
        
        # Update task status in spec file
        if target_spec_file:
            self.update_task_status_in_spec(target_spec_file, task_id, completed=True)
            
            # Commit the spec file update
            spec_update_context = TaskContext(
                task_id=f"{task_id}-status",
                task_name=f"Update task {task_id} status to completed",
                description=f"Mark task {task_id} as completed in specification",
                files_modified=[str(target_spec_file.relative_to(self.repo_path))],
                requirements_addressed=[],
                completion_time=datetime.now()
            )
            
            self.git_manager.commit_task_completion(spec_update_context, auto_push)
        
        return commit_hash
    
    def start_task(self, task_id: str, spec_file: Optional[Path] = None, 
                   create_branch: bool = False) -> Optional[str]:
        """
        Start working on a task by marking it as in progress.
        
        Args:
            task_id: ID of the task to start
            spec_file: Specific spec file to use (if None, searches all)
            create_branch: Whether to create a feature branch for this task
            
        Returns:
            Branch name if created, None otherwise
        """
        # Find the task in spec files
        target_spec_file = spec_file
        if not target_spec_file:
            for spec_path in self.find_spec_files():
                task_context = self.parse_task_from_spec(spec_path, task_id)
                if task_context:
                    target_spec_file = spec_path
                    break
        
        if not target_spec_file:
            raise ValueError(f"Task {task_id} not found in any specification file")
        
        # Update task status to in progress
        self.update_task_status_in_spec(target_spec_file, task_id, completed=False)
        
        # Create feature branch if requested
        branch_name = None
        if create_branch:
            branch_name = self.git_manager.create_feature_branch(task_id)
        
        # Commit the status update
        status_update_context = TaskContext(
            task_id=f"{task_id}-start",
            task_name=f"Start task {task_id}",
            description=f"Mark task {task_id} as in progress",
            files_modified=[str(target_spec_file.relative_to(self.repo_path))],
            requirements_addressed=[],
            completion_time=datetime.now(),
            branch_name=branch_name
        )
        
        self.git_manager.commit_task_completion(status_update_context, auto_push=True)
        
        return branch_name
    
    def get_task_progress(self) -> Dict[str, Any]:
        """
        Get overall task progress across all specification files.
        
        Returns:
            Dictionary with progress information
        """
        progress = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "in_progress_tasks": 0,
            "pending_tasks": 0,
            "specs": {}
        }
        
        for spec_file in self.find_spec_files():
            spec_name = spec_file.parent.name
            spec_progress = self._analyze_spec_progress(spec_file)
            progress["specs"][spec_name] = spec_progress
            
            progress["total_tasks"] += spec_progress["total"]
            progress["completed_tasks"] += spec_progress["completed"]
            progress["in_progress_tasks"] += spec_progress["in_progress"]
            progress["pending_tasks"] += spec_progress["pending"]
        
        return progress
    
    def _analyze_spec_progress(self, spec_file: Path) -> Dict[str, int]:
        """
        Analyze progress for a single specification file.
        
        Args:
            spec_file: Path to the specification file
            
        Returns:
            Dictionary with progress counts
        """
        if not spec_file.exists():
            return {"total": 0, "completed": 0, "in_progress": 0, "pending": 0}
        
        content = spec_file.read_text()
        
        completed = len(re.findall(r"^- \[x\]", content, re.MULTILINE))
        in_progress = len(re.findall(r"^- \[-\]", content, re.MULTILINE))
        pending = len(re.findall(r"^- \[ \]", content, re.MULTILINE))
        total = completed + in_progress + pending
        
        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending
        }


# Convenience functions for easy usage
def complete_task(task_id: str, spec_file: Optional[str] = None) -> str:
    """
    Complete a task with automated Git workflow.
    
    Args:
        task_id: ID of the task to complete
        spec_file: Optional path to specific spec file
        
    Returns:
        Commit hash of the completion commit
    """
    integrator = TaskCompletionIntegrator()
    spec_path = Path(spec_file) if spec_file else None
    return integrator.complete_task_by_id(task_id, spec_path)


def start_task(task_id: str, spec_file: Optional[str] = None, create_branch: bool = False) -> Optional[str]:
    """
    Start working on a task.
    
    Args:
        task_id: ID of the task to start
        spec_file: Optional path to specific spec file
        create_branch: Whether to create a feature branch
        
    Returns:
        Branch name if created, None otherwise
    """
    integrator = TaskCompletionIntegrator()
    spec_path = Path(spec_file) if spec_file else None
    return integrator.start_task(task_id, spec_path, create_branch)


def get_progress() -> Dict[str, Any]:
    """
    Get overall task progress.
    
    Returns:
        Dictionary with progress information
    """
    integrator = TaskCompletionIntegrator()
    return integrator.get_task_progress()


if __name__ == "__main__":
    # Example usage
    integrator = TaskCompletionIntegrator()
    
    # Show current progress
    progress = integrator.get_task_progress()
    print("📊 Task Progress:")
    print(f"   Total: {progress['total_tasks']}")
    print(f"   Completed: {progress['completed_tasks']}")
    print(f"   In Progress: {progress['in_progress_tasks']}")
    print(f"   Pending: {progress['pending_tasks']}")
    
    for spec_name, spec_progress in progress["specs"].items():
        print(f"\n📋 {spec_name}:")
        print(f"   {spec_progress['completed']}/{spec_progress['total']} completed")
        if spec_progress['in_progress'] > 0:
            print(f"   {spec_progress['in_progress']} in progress")