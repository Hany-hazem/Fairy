#!/usr/bin/env python3
"""
Git Workflow Automation Demo

This script demonstrates the Git Workflow Automation Service capabilities
including automated branch creation, commit generation, and monitoring.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.git_workflow_integration import GitWorkflowIntegration


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_basic_workflow():
    """Demonstrate basic workflow automation"""
    print("\n=== Git Workflow Automation Demo ===\n")
    
    # Initialize the integrated workflow system
    workflow = GitWorkflowIntegration()
    
    try:
        # Start the workflow system
        print("1. Starting Git workflow system...")
        await workflow.start()
        
        # Wait a moment for services to initialize
        await asyncio.sleep(2)
        
        # Check system health
        print("2. Checking system health...")
        health = await workflow.get_system_health()
        print(f"   System status: {health['overall_status']}")
        print(f"   Repository health: {health['repository_health']['overall_status']}")
        
        # Start a new task
        print("\n3. Starting a new task...")
        task_id = "8.1"
        task_name = "Build automated Git workflow triggers"
        task_description = "Create task lifecycle event handlers for Git operations"
        requirements = ["4.1", "4.2", "4.3", "8.1", "8.2"]
        
        success = await workflow.start_task(
            task_id=task_id,
            task_name=task_name,
            task_description=task_description,
            requirements=requirements
        )
        
        if success:
            print(f"   ✓ Task {task_id} started successfully")
        else:
            print(f"   ✗ Failed to start task {task_id}")
            return
        
        # Wait for automation to process
        await asyncio.sleep(3)
        
        # Check task status
        print("\n4. Checking task status...")
        task_status = await workflow.get_task_status(task_id)
        if task_status:
            print(f"   Task ID: {task_status['task_id']}")
            print(f"   Status: {task_status['mapping']['status']}")
            print(f"   Branch: {task_status['mapping']['branch_name']}")
            print(f"   Created: {task_status['mapping']['created_at']}")
        
        # Simulate task progress
        print("\n5. Updating task progress...")
        files_changed = [
            "app/git_workflow_automation_service.py",
            "examples/git_workflow_automation_demo.py"
        ]
        
        success = await workflow.update_task_progress(
            task_id=task_id,
            files_changed=files_changed,
            progress_notes="Implemented automated workflow triggers"
        )
        
        if success:
            print(f"   ✓ Task progress updated")
        else:
            print(f"   ✗ Failed to update task progress")
        
        # Wait for automation to process
        await asyncio.sleep(3)
        
        # Complete the task
        print("\n6. Completing task...")
        success = await workflow.complete_task(
            task_id=task_id,
            completion_notes="Successfully implemented automated Git workflow triggers",
            requirements_addressed=requirements
        )
        
        if success:
            print(f"   ✓ Task {task_id} completed successfully")
        else:
            print(f"   ✗ Failed to complete task {task_id}")
        
        # Wait for automation to process
        await asyncio.sleep(3)
        
        # Check final task status
        print("\n7. Final task status...")
        task_status = await workflow.get_task_status(task_id)
        if task_status:
            print(f"   Status: {task_status['mapping']['status']}")
            print(f"   Commits: {len(task_status['mapping']['commits'])}")
            if task_status['git_metrics']:
                print(f"   Files modified: {task_status['git_metrics']['files_modified']}")
                print(f"   Lines added: {task_status['git_metrics']['lines_added']}")
        
        # Show event history
        print("\n8. Recent workflow events...")
        events = workflow.get_event_history(limit=10)
        for event in events[-5:]:  # Show last 5 events
            print(f"   {event['timestamp'][:19]} - {event['event_type']} - {event['task_id']}")
        
        # Show system status
        print("\n9. Final system status...")
        all_tasks = workflow.get_all_tasks()
        print(f"   Total tasks: {len(all_tasks)}")
        
        ready_tasks = workflow.get_ready_tasks()
        print(f"   Ready tasks: {len(ready_tasks)}")
        
        active_recoveries = workflow.get_active_recoveries()
        print(f"   Active recoveries: {len(active_recoveries)}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
    
    finally:
        # Stop the workflow system
        print("\n10. Stopping workflow system...")
        await workflow.stop()
        print("   ✓ Workflow system stopped")


async def demo_dependency_management():
    """Demonstrate dependency management features"""
    print("\n=== Dependency Management Demo ===\n")
    
    workflow = GitWorkflowIntegration()
    
    try:
        await workflow.start()
        await asyncio.sleep(2)
        
        # Create multiple tasks with dependencies
        print("1. Creating tasks with dependencies...")
        
        # Task 8.1 (no dependencies)
        await workflow.start_task("8.1", "Build automated Git workflow triggers")
        
        # Task 8.2 (depends on 8.1)
        await workflow.start_task(
            "8.2", 
            "Implement Git workflow monitoring",
            dependencies=["8.1"]
        )
        
        # Task 8.3 (depends on both 8.1 and 8.2)
        await workflow.start_task(
            "8.3",
            "Create workflow integration tests",
            dependencies=["8.1", "8.2"]
        )
        
        await asyncio.sleep(3)
        
        # Show dependency graph
        print("\n2. Dependency relationships...")
        for task_id in ["8.1", "8.2", "8.3"]:
            deps = workflow.dependency_manager.get_task_dependencies(task_id)
            dependents = workflow.dependency_manager.get_task_dependents(task_id)
            print(f"   Task {task_id}: depends on {deps}, required by {dependents}")
        
        # Show ready tasks
        print("\n3. Ready tasks...")
        ready_tasks = workflow.get_ready_tasks()
        print(f"   Ready to work on: {ready_tasks}")
        
        # Show critical path
        print("\n4. Critical path...")
        critical_path = workflow.get_critical_path()
        print(f"   Critical path: {' -> '.join(critical_path)}")
        
        # Complete task 8.1 and show how it affects ready tasks
        print("\n5. Completing task 8.1...")
        await workflow.complete_task("8.1", "Task 8.1 completed")
        await asyncio.sleep(3)
        
        ready_tasks = workflow.get_ready_tasks()
        print(f"   Now ready to work on: {ready_tasks}")
        
        # Generate merge strategy
        print("\n6. Merge strategy for all tasks...")
        strategy = await workflow.get_merge_strategy(["8.1", "8.2", "8.3"])
        if strategy:
            print(f"   Merge order: {strategy['merge_order']}")
            print(f"   Parallel groups: {strategy['parallel_groups']}")
            print(f"   Risk level: {strategy['risk_level']}")
            print(f"   Estimated duration: {strategy['estimated_duration']} minutes")
        
    except Exception as e:
        logger.error(f"Dependency demo failed: {e}")
        print(f"\n❌ Dependency demo failed: {e}")
    
    finally:
        await workflow.stop()


async def demo_monitoring_and_recovery():
    """Demonstrate monitoring and recovery features"""
    print("\n=== Monitoring and Recovery Demo ===\n")
    
    workflow = GitWorkflowIntegration()
    
    try:
        await workflow.start()
        await asyncio.sleep(2)
        
        # Perform health check
        print("1. Performing health check...")
        health = await workflow.get_system_health()
        
        print(f"   Overall status: {health['overall_status']}")
        print(f"   Issues: {health['repository_health']['issues_count']}")
        print(f"   Warnings: {health['repository_health']['warnings_count']}")
        
        # Show component health
        print("\n2. Component health details...")
        components = health['repository_health']['components']
        for component, details in components.items():
            print(f"   {component}: {details['status']} - {details['message']}")
        
        # Show health history
        print("\n3. Recent health history...")
        health_history = workflow.get_health_history(limit=5)
        for i, health_record in enumerate(health_history[-3:]):
            print(f"   Check {i+1}: {health_record['overall_status']} "
                  f"({health_record['issues_count']} issues)")
        
        # Show active recoveries
        print("\n4. Active recovery operations...")
        recoveries = workflow.get_active_recoveries()
        if recoveries:
            for recovery in recoveries:
                print(f"   {recovery['operation_id']}: {recovery['action']} - {recovery['status']}")
        else:
            print("   No active recovery operations")
        
        # Demonstrate manual recovery trigger
        print("\n5. Manual recovery operations...")
        print("   Available recovery actions:")
        print("   - cleanup_stale_branches")
        print("   - merge_conflict_resolution")
        print("   - repository_repair")
        
        # Trigger cleanup (safe operation)
        try:
            operation_id = await workflow.trigger_recovery("cleanup_stale_branches", "system")
            print(f"   ✓ Triggered cleanup operation: {operation_id}")
        except Exception as e:
            print(f"   ⚠ Recovery trigger failed: {e}")
        
        # Wait and check recovery status
        await asyncio.sleep(5)
        recoveries = workflow.get_active_recoveries()
        if recoveries:
            print(f"   Recovery operations in progress: {len(recoveries)}")
        
    except Exception as e:
        logger.error(f"Monitoring demo failed: {e}")
        print(f"\n❌ Monitoring demo failed: {e}")
    
    finally:
        await workflow.stop()


async def main():
    """Run all demos"""
    print("Git Workflow Automation System Demo")
    print("=" * 50)
    
    # Check if we're in a Git repository
    if not os.path.exists('.git'):
        print("❌ This demo must be run from within a Git repository")
        return
    
    try:
        # Run basic workflow demo
        await demo_basic_workflow()
        
        # Wait between demos
        await asyncio.sleep(2)
        
        # Run dependency management demo
        await demo_dependency_management()
        
        # Wait between demos
        await asyncio.sleep(2)
        
        # Run monitoring and recovery demo
        await demo_monitoring_and_recovery()
        
        print("\n" + "=" * 50)
        print("✅ All demos completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo suite failed: {e}")
        print(f"\n❌ Demo suite failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())