# Git Workflow Automation

This document describes the automated Git workflow system for task completion and repository management.

## Overview

The Git Workflow Automation system provides:

- **Automated commit generation** with proper task context
- **Intelligent commit message generation** following conventional commit standards
- **Task completion tracking** with Git integration
- **Branch management** for feature development
- **Specification file updates** when tasks are completed

## Components

### 1. GitWorkflowManager (`app/git_workflow_manager.py`)

Core class that handles Git operations:

- **Commit generation**: Creates commits with proper context and messages
- **Branch management**: Creates and manages feature branches
- **Status monitoring**: Tracks repository status and conflicts
- **Task tracking**: Maintains history of completed tasks

### 2. TaskCompletionIntegrator (`app/task_completion_integration.py`)

High-level integration with the task system:

- **Spec file parsing**: Extracts task information from specification files
- **Status updates**: Updates task status in spec files ([ ] → [x])
- **Progress tracking**: Monitors overall project progress
- **Automated workflows**: Combines Git operations with task management

### 3. CLI Script (`scripts/complete_task.py`)

Command-line interface for task completion:

```bash
# Complete a task manually
python scripts/complete_task.py --task-id "1.1" --task-name "Implement feature" --description "Added functionality"

# Auto-detect from spec file
python scripts/complete_task.py --auto-detect --spec-file .kiro/specs/personal-assistant-enhancement/tasks.md --task-id "1.1"
```

## Usage Examples

### Completing a Task

```python
from app.task_completion_integration import complete_task

# Complete task 1.1 from any spec file
commit_hash = complete_task("1.1")
print(f"Task completed with commit: {commit_hash}")
```

### Starting a Task

```python
from app.task_completion_integration import start_task

# Start task 2.1 and create a feature branch
branch_name = start_task("2.1", create_branch=True)
print(f"Started task on branch: {branch_name}")
```

### Checking Progress

```python
from app.task_completion_integration import get_progress

progress = get_progress()
print(f"Progress: {progress['completed_tasks']}/{progress['total_tasks']} tasks completed")
```

### Manual Git Operations

```python
from app.git_workflow_manager import GitWorkflowManager, TaskContext
from datetime import datetime

manager = GitWorkflowManager()

# Create task context
task_context = TaskContext(
    task_id="1.1",
    task_name="Implement authentication",
    description="Added login and registration system",
    files_modified=["app/auth.py", "tests/test_auth.py"],
    requirements_addressed=["User login", "User registration"],
    completion_time=datetime.now()
)

# Commit with automatic push
commit_hash = manager.commit_task_completion(task_context)
```

## Commit Message Format

The system generates commit messages following conventional commit standards:

```
feat: implement user authentication

Task ID: 1.1
Description: Added login and registration system

Files modified:
- app/auth.py
- tests/test_auth.py

Requirements addressed:
- User login
- User registration

Completed: 2024-01-15T10:30:00
```

## Branch Management

### Feature Branches

When creating feature branches, the system uses the pattern:
- `task/1-1` for task 1.1
- `task/2-3` for task 2.3

### Branch Workflow

1. **Start task**: Creates feature branch (optional)
2. **Work on task**: Make changes in the branch
3. **Complete task**: Commits changes and pushes
4. **Merge**: Manual merge to main branch (recommended)

## Task Status Tracking

The system tracks task status in specification files:

- `- [ ] 1.1 Task name` - Pending
- `- [-] 1.1 Task name` - In progress
- `- [x] 1.1 Task name` - Completed

## Configuration

### Environment Variables

- `GIT_WORKFLOW_AUTO_PUSH`: Set to `false` to disable automatic pushing
- `GIT_WORKFLOW_BRANCH_PREFIX`: Custom prefix for feature branches (default: `task/`)

### Task Tracking

Task completion history is stored in `.kiro/task_tracking.json`:

```json
{
  "completed_tasks": [
    {
      "task_id": "1.1",
      "task_name": "Implement authentication",
      "description": "Added login system",
      "files_modified": ["app/auth.py"],
      "requirements_addressed": ["User login"],
      "completion_time": "2024-01-15T10:30:00",
      "commit_hash": "abc123def",
      "branch_name": "task/1-1"
    }
  ],
  "current_branch": "main",
  "last_sync": "2024-01-15T10:30:00"
}
```

## Integration with Existing Workflow

### With Kiro IDE

The system integrates with Kiro's task management:

1. **Task selection**: Choose task from spec file
2. **Implementation**: Work on the task
3. **Completion**: Use automated workflow to commit and push
4. **Status update**: Spec file automatically updated

### With CI/CD

The conventional commit format enables:

- **Automated releases**: Based on commit types
- **Changelog generation**: From commit messages
- **Semantic versioning**: Automatic version bumps

## Best Practices

### 1. Task Granularity

- Keep tasks focused and specific
- One task = one logical change
- Include tests with implementation tasks

### 2. Commit Messages

- Let the system generate messages for consistency
- Add manual details in task descriptions
- Reference requirements explicitly

### 3. Branch Management

- Use feature branches for complex tasks
- Keep branches short-lived
- Merge frequently to avoid conflicts

### 4. Testing

- Run tests before completing tasks
- Include test files in task completion
- Verify functionality before pushing

## Troubleshooting

### Common Issues

**Import errors**: Ensure `PYTHONPATH=.` when running scripts

**Git authentication**: Set up SSH keys or personal access tokens

**Permission errors**: Ensure write access to repository

**Merge conflicts**: Resolve manually before completing tasks

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from app.git_workflow_manager import GitWorkflowManager
manager = GitWorkflowManager()
```

## Future Enhancements

- **Automated testing**: Run tests before committing
- **Code quality checks**: Lint and format code automatically
- **Pull request creation**: Automatic PR creation for feature branches
- **Integration with external tools**: Jira, Trello, etc.
- **Rollback capabilities**: Automatic rollback on failures

## API Reference

See the docstrings in the source files for detailed API documentation:

- `app/git_workflow_manager.py` - Core Git operations
- `app/task_completion_integration.py` - High-level task integration
- `scripts/complete_task.py` - CLI interface