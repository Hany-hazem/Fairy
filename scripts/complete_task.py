#!/usr/bin/env python3
"""
CLI script for completing tasks with automated Git workflow.

Usage:
    python scripts/complete_task.py --task-id "1.1" --task-name "Implement feature" --description "Added new functionality"
    python scripts/complete_task.py --auto-detect-from-spec .kiro/specs/personal-assistant-enhancement/tasks.md
"""

import argparse
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.git_workflow_manager import GitWorkflowManager, TaskContext
import datetime


def parse_task_from_spec(spec_file: str, task_id: str) -> TaskContext:
    """
    Parse task information from specification file.
    
    Args:
        spec_file: Path to the specification file
        task_id: ID of the task to parse
        
    Returns:
        TaskContext object with parsed information
    """
    spec_path = Path(spec_file)
    if not spec_path.exists():
        raise FileNotFoundError(f"Specification file not found: {spec_file}")
    
    content = spec_path.read_text()
    lines = content.split('\n')
    
    task_name = ""
    description = ""
    requirements = []
    in_task = False
    
    for line in lines:
        line = line.strip()
        
        # Look for the task
        if f"- [ ] {task_id}" in line or f"- [x] {task_id}" in line:
            in_task = True
            # Extract task name
            parts = line.split(task_id, 1)
            if len(parts) > 1:
                task_name = parts[1].strip(". ")
            continue
        
        if in_task:
            # Stop when we hit the next task
            if line.startswith("- [ ]") or line.startswith("- [x]"):
                break
            
            # Extract description and requirements
            if line.startswith("- ") and not line.startswith("- [ ]") and not line.startswith("- [x]"):
                description += line[2:] + " "
            elif "_Requirements:" in line:
                req_part = line.split("_Requirements:", 1)[1].strip("_ ")
                requirements.extend([r.strip() for r in req_part.split(",")])
    
    if not task_name:
        raise ValueError(f"Task {task_id} not found in specification file")
    
    return TaskContext(
        task_id=task_id,
        task_name=task_name,
        description=description.strip(),
        files_modified=[],  # Will be detected automatically
        requirements_addressed=requirements,
        completion_time=datetime.datetime.now()
    )


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Complete a task with automated Git workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete a task manually
  python scripts/complete_task.py --task-id "1.1" --task-name "Implement authentication" --description "Added login system"
  
  # Auto-detect task from specification
  python scripts/complete_task.py --auto-detect --spec-file .kiro/specs/personal-assistant-enhancement/tasks.md --task-id "1.1"
  
  # Complete with requirements
  python scripts/complete_task.py --task-id "1.1" --task-name "Add feature" --description "New feature" --requirements "Req1,Req2"
        """
    )
    
    parser.add_argument("--task-id", required=True, help="Task ID (e.g., '1.1')")
    parser.add_argument("--task-name", help="Human-readable task name")
    parser.add_argument("--description", help="Detailed description of what was accomplished")
    parser.add_argument("--requirements", help="Comma-separated list of requirements addressed")
    parser.add_argument("--auto-detect", action="store_true", help="Auto-detect task info from spec file")
    parser.add_argument("--spec-file", help="Path to specification file for auto-detection")
    parser.add_argument("--no-push", action="store_true", help="Don't automatically push to remote")
    parser.add_argument("--branch", help="Create/switch to specific branch for this task")
    
    args = parser.parse_args()
    
    try:
        # Initialize Git workflow manager
        manager = GitWorkflowManager()
        
        # Create or switch to branch if specified
        if args.branch:
            branch_name = manager.create_feature_branch(args.branch)
            print(f"🌿 Switched to branch: {branch_name}")
        
        # Get task context
        if args.auto_detect:
            if not args.spec_file:
                print("❌ Error: --spec-file is required when using --auto-detect")
                sys.exit(1)
            
            print(f"📖 Parsing task {args.task_id} from {args.spec_file}")
            task_context = parse_task_from_spec(args.spec_file, args.task_id)
        else:
            if not args.task_name or not args.description:
                print("❌ Error: --task-name and --description are required when not using --auto-detect")
                sys.exit(1)
            
            requirements = []
            if args.requirements:
                requirements = [r.strip() for r in args.requirements.split(",")]
            
            task_context = TaskContext(
                task_id=args.task_id,
                task_name=args.task_name,
                description=args.description,
                files_modified=[],  # Will be detected automatically
                requirements_addressed=requirements,
                completion_time=datetime.datetime.now()
            )
        
        # Get modified files
        modified_files = manager.get_modified_files()
        task_context.files_modified = modified_files
        
        if not modified_files:
            print("⚠️  Warning: No modified files detected. Are you sure the task is complete?")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("❌ Task completion cancelled")
                sys.exit(1)
        
        print(f"📝 Task: {task_context.task_name}")
        print(f"📄 Description: {task_context.description}")
        print(f"📁 Files modified: {len(modified_files)}")
        if modified_files:
            for file in modified_files[:5]:  # Show first 5 files
                print(f"   - {file}")
            if len(modified_files) > 5:
                print(f"   ... and {len(modified_files) - 5} more")
        
        if task_context.requirements_addressed:
            print(f"✅ Requirements: {', '.join(task_context.requirements_addressed)}")
        
        # Confirm before committing
        response = input("\n🚀 Commit and push these changes? (Y/n): ")
        if response.lower() == 'n':
            print("❌ Task completion cancelled")
            sys.exit(1)
        
        # Commit task completion
        auto_push = not args.no_push
        commit_hash = manager.commit_task_completion(task_context, auto_push=auto_push)
        
        print(f"\n✅ Task {args.task_id} completed successfully!")
        print(f"📝 Commit: {commit_hash[:8]}")
        if auto_push:
            print(f"🚀 Changes pushed to repository")
        else:
            print(f"📦 Changes committed locally (not pushed)")
        
        # Update task status in spec file if auto-detected
        if args.auto_detect and args.spec_file:
            try:
                update_task_status_in_spec(args.spec_file, args.task_id)
                print(f"📋 Task status updated in {args.spec_file}")
            except Exception as e:
                print(f"⚠️  Warning: Could not update task status: {e}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def update_task_status_in_spec(spec_file: str, task_id: str):
    """
    Update task status from [ ] to [x] in specification file.
    
    Args:
        spec_file: Path to the specification file
        task_id: ID of the task to mark as completed
    """
    spec_path = Path(spec_file)
    content = spec_path.read_text()
    
    # Replace [ ] with [x] for the specific task
    old_pattern = f"- [ ] {task_id}"
    new_pattern = f"- [x] {task_id}"
    
    if old_pattern in content:
        updated_content = content.replace(old_pattern, new_pattern)
        spec_path.write_text(updated_content)
    else:
        raise ValueError(f"Task {task_id} not found or already completed in {spec_file}")


if __name__ == "__main__":
    main()