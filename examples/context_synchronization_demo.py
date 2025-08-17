#!/usr/bin/env python3
"""
Context Synchronization Demo

This script demonstrates the Context Synchronization Engine functionality
including context sharing, conflict resolution, and notification system.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock implementations for demo purposes
class MockRedisMCPBackend:
    """Mock Redis backend for demonstration"""
    
    def __init__(self):
        self._connected = False
        self._data = {}
        self._subscriptions = {}
    
    async def connect(self, redis_url: str = None) -> bool:
        """Mock connect"""
        self._connected = True
        logger.info("Mock Redis backend connected")
        return True
    
    async def disconnect(self):
        """Mock disconnect"""
        self._connected = False
        logger.info("Mock Redis backend disconnected")
    
    async def publish_message(self, topic: str, message) -> bool:
        """Mock publish message"""
        logger.info(f"Mock published message to topic: {topic}")
        return True
    
    async def subscribe_to_topic(self, topic: str, callback, message_filter=None) -> str:
        """Mock subscribe to topic"""
        subscription_id = f"sub_{len(self._subscriptions)}"
        self._subscriptions[subscription_id] = {"topic": topic, "callback": callback}
        logger.info(f"Mock subscribed to topic: {topic}")
        return subscription_id
    
    async def unsubscribe_from_topic(self, subscription_id: str) -> bool:
        """Mock unsubscribe from topic"""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.info(f"Mock unsubscribed: {subscription_id}")
            return True
        return False
    
    @property
    def _redis(self):
        """Mock Redis client"""
        return MockRedisClient()


class MockRedisClient:
    """Mock Redis client"""
    
    async def ping(self):
        return True
    
    async def setex(self, key: str, ttl: int, value: str):
        return True
    
    async def get(self, key: str):
        return None
    
    async def sadd(self, key: str, value: str):
        return 1
    
    async def expire(self, key: str, ttl: int):
        return True
    
    async def hset(self, key: str, field: str, value: str):
        return True
    
    async def delete(self, key: str):
        return True


# Mock models for demo
class MockAgentContext:
    """Mock agent context for demonstration"""
    
    def __init__(self, agent_id: str, context_type: str, context_data: Dict[str, Any], 
                 access_level: str = "public", shared_with: list = None, version: str = None):
        self.agent_id = agent_id
        self.context_type = context_type
        self.context_data = context_data
        self.access_level = access_level
        self.shared_with = shared_with or []
        self.version = version or f"v_{datetime.now().timestamp()}"
        self.last_updated = datetime.utcnow()
        self.metadata = {}
        self.ttl = None
    
    def can_access(self, agent_id: str) -> bool:
        """Check if agent can access this context"""
        if agent_id == self.agent_id:
            return True
        if self.access_level == "public":
            return True
        if self.access_level == "private":
            return False
        return agent_id in self.shared_with
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "context_type": self.context_type,
            "context_data": self.context_data,
            "access_level": self.access_level,
            "shared_with": self.shared_with,
            "version": self.version,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }


class ContextSynchronizationDemo:
    """Demonstration of context synchronization functionality"""
    
    def __init__(self):
        self.redis_backend = MockRedisMCPBackend()
        self.contexts = {}
        self.subscriptions = {}
        self.notifications = []
    
    async def start(self):
        """Start the demo"""
        logger.info("Starting Context Synchronization Demo")
        await self.redis_backend.connect()
    
    async def stop(self):
        """Stop the demo"""
        await self.redis_backend.disconnect()
        logger.info("Context Synchronization Demo stopped")
    
    async def demo_context_sharing(self):
        """Demonstrate context sharing mechanisms"""
        logger.info("\n=== Context Sharing Demo ===")
        
        # Create sample contexts
        task_context = MockAgentContext(
            agent_id="ai_assistant",
            context_type="task_context",
            context_data={
                "task_id": "task_123",
                "status": "in_progress",
                "description": "Implement context synchronization",
                "progress": 0.7,
                "files_modified": ["context_synchronizer.py", "notification_system.py"]
            },
            shared_with=["self_improvement_agent", "git_workflow_agent"]
        )
        
        project_context = MockAgentContext(
            agent_id="project_manager",
            context_type="project_context",
            context_data={
                "project_id": "mcp_integration",
                "phase": "implementation",
                "deadline": "2024-02-15",
                "team_members": ["ai_assistant", "self_improvement_agent"],
                "milestones": {
                    "context_sync": "completed",
                    "git_integration": "in_progress",
                    "testing": "pending"
                }
            },
            access_level="public"
        )
        
        # Store contexts
        self.contexts[f"{task_context.agent_id}:{task_context.context_type}"] = task_context
        self.contexts[f"{project_context.agent_id}:{project_context.context_type}"] = project_context
        
        logger.info(f"Created task context: {task_context.context_data['task_id']}")
        logger.info(f"Created project context: {project_context.context_data['project_id']}")
        
        # Simulate context broadcast
        await self.broadcast_context_update(task_context, ["self_improvement_agent", "git_workflow_agent"])
        await self.broadcast_context_update(project_context, ["ai_assistant", "self_improvement_agent"])
        
        return task_context, project_context
    
    async def demo_conflict_resolution(self):
        """Demonstrate context conflict resolution"""
        logger.info("\n=== Conflict Resolution Demo ===")
        
        # Create conflicting contexts
        local_context = MockAgentContext(
            agent_id="ai_assistant",
            context_type="task_context",
            context_data={
                "task_id": "task_123",
                "status": "in_progress",
                "progress": 0.7,
                "last_action": "implemented context sharing"
            }
        )
        
        remote_context = MockAgentContext(
            agent_id="ai_assistant",
            context_type="task_context",
            context_data={
                "task_id": "task_123",
                "status": "completed",
                "progress": 1.0,
                "last_action": "completed all tests"
            }
        )
        
        logger.info("Detected conflict between contexts:")
        logger.info(f"  Local: status={local_context.context_data['status']}, progress={local_context.context_data['progress']}")
        logger.info(f"  Remote: status={remote_context.context_data['status']}, progress={remote_context.context_data['progress']}")
        
        # Simulate conflict resolution strategies
        await self.resolve_conflict_latest_wins(local_context, remote_context)
        await self.resolve_conflict_merge_recursive(local_context, remote_context)
        
        return local_context, remote_context
    
    async def demo_notification_system(self):
        """Demonstrate notification system"""
        logger.info("\n=== Notification System Demo ===")
        
        # Create subscriptions
        await self.create_subscription(
            agent_id="self_improvement_agent",
            context_types=["task_context", "project_context"],
            delivery_method="push"
        )
        
        await self.create_subscription(
            agent_id="git_workflow_agent",
            context_types=["task_context"],
            delivery_method="batch"
        )
        
        # Create context updates
        context_update = MockAgentContext(
            agent_id="ai_assistant",
            context_type="task_context",
            context_data={
                "task_id": "task_124",
                "status": "started",
                "description": "Implement Git workflow automation"
            }
        )
        
        # Send notifications
        await self.notify_context_update(context_update, "created")
        
        # Simulate batch processing
        await self.process_notification_batches()
        
        return context_update
    
    async def demo_access_control(self):
        """Demonstrate access control and permissions"""
        logger.info("\n=== Access Control Demo ===")
        
        # Create contexts with different access levels
        public_context = MockAgentContext(
            agent_id="ai_assistant",
            context_type="public_info",
            context_data={"info": "This is public information"},
            access_level="public"
        )
        
        private_context = MockAgentContext(
            agent_id="ai_assistant",
            context_type="private_info",
            context_data={"secret": "This is private information"},
            access_level="private"
        )
        
        restricted_context = MockAgentContext(
            agent_id="ai_assistant",
            context_type="restricted_info",
            context_data={"data": "This is restricted information"},
            access_level="restricted",
            shared_with=["self_improvement_agent"]
        )
        
        # Test access permissions
        agents = ["self_improvement_agent", "git_workflow_agent", "unauthorized_agent"]
        
        for agent in agents:
            logger.info(f"Access test for {agent}:")
            logger.info(f"  Public context: {public_context.can_access(agent)}")
            logger.info(f"  Private context: {private_context.can_access(agent)}")
            logger.info(f"  Restricted context: {restricted_context.can_access(agent)}")
        
        return public_context, private_context, restricted_context
    
    async def broadcast_context_update(self, context: MockAgentContext, targets: list):
        """Simulate context broadcast"""
        logger.info(f"Broadcasting {context.context_type} from {context.agent_id} to {targets}")
        
        successful_deliveries = []
        failed_deliveries = []
        
        for target in targets:
            if context.can_access(target):
                successful_deliveries.append(target)
                logger.info(f"  ✓ Delivered to {target}")
            else:
                failed_deliveries.append(target)
                logger.info(f"  ✗ Access denied for {target}")
        
        logger.info(f"Broadcast result: {len(successful_deliveries)} successful, {len(failed_deliveries)} failed")
        return successful_deliveries, failed_deliveries
    
    async def resolve_conflict_latest_wins(self, local_context: MockAgentContext, remote_context: MockAgentContext):
        """Simulate latest wins conflict resolution"""
        logger.info("Resolving conflict using 'latest wins' strategy:")
        
        if remote_context.last_updated > local_context.last_updated:
            winner = remote_context
            logger.info("  Remote context wins (newer timestamp)")
        else:
            winner = local_context
            logger.info("  Local context wins (newer timestamp)")
        
        logger.info(f"  Resolved status: {winner.context_data['status']}")
        return winner
    
    async def resolve_conflict_merge_recursive(self, local_context: MockAgentContext, remote_context: MockAgentContext):
        """Simulate recursive merge conflict resolution"""
        logger.info("Resolving conflict using 'merge recursive' strategy:")
        
        merged_data = local_context.context_data.copy()
        
        for key, value in remote_context.context_data.items():
            if key not in merged_data:
                merged_data[key] = value
                logger.info(f"  Added field '{key}' from remote")
            elif merged_data[key] != value:
                merged_data[key] = value  # Remote wins for conflicts
                logger.info(f"  Updated field '{key}' with remote value")
        
        merged_context = MockAgentContext(
            agent_id=local_context.agent_id,
            context_type=local_context.context_type,
            context_data=merged_data
        )
        
        logger.info(f"  Merged context: {json.dumps(merged_data, indent=2)}")
        return merged_context
    
    async def create_subscription(self, agent_id: str, context_types: list, delivery_method: str):
        """Simulate creating a subscription"""
        subscription_id = f"sub_{len(self.subscriptions)}"
        subscription = {
            "subscription_id": subscription_id,
            "agent_id": agent_id,
            "context_types": context_types,
            "delivery_method": delivery_method,
            "created_at": datetime.utcnow()
        }
        
        self.subscriptions[subscription_id] = subscription
        logger.info(f"Created subscription {subscription_id} for {agent_id} ({delivery_method})")
        return subscription_id
    
    async def notify_context_update(self, context: MockAgentContext, notification_type: str):
        """Simulate context update notification"""
        logger.info(f"Notifying context update: {context.context_type} ({notification_type})")
        
        # Find matching subscriptions
        matching_subscriptions = []
        for sub_id, subscription in self.subscriptions.items():
            if context.context_type in subscription["context_types"]:
                if context.can_access(subscription["agent_id"]):
                    matching_subscriptions.append(subscription)
        
        # Create notifications
        for subscription in matching_subscriptions:
            notification = {
                "notification_id": f"notif_{len(self.notifications)}",
                "subscription_id": subscription["subscription_id"],
                "agent_id": subscription["agent_id"],
                "context": context,
                "notification_type": notification_type,
                "created_at": datetime.utcnow(),
                "delivery_method": subscription["delivery_method"]
            }
            
            self.notifications.append(notification)
            logger.info(f"  Created notification for {subscription['agent_id']} ({subscription['delivery_method']})")
        
        return len(matching_subscriptions)
    
    async def process_notification_batches(self):
        """Simulate batch notification processing"""
        logger.info("Processing notification batches:")
        
        # Group notifications by agent and delivery method
        batches = {}
        for notification in self.notifications:
            if notification["delivery_method"] == "batch":
                agent_id = notification["agent_id"]
                if agent_id not in batches:
                    batches[agent_id] = []
                batches[agent_id].append(notification)
        
        # Process batches
        for agent_id, notifications in batches.items():
            if len(notifications) > 0:
                batch_id = f"batch_{agent_id}_{datetime.now().timestamp()}"
                logger.info(f"  Created batch {batch_id} with {len(notifications)} notifications for {agent_id}")
                
                # Simulate batch delivery
                logger.info(f"  Delivered batch {batch_id} to {agent_id}")


async def main():
    """Run the context synchronization demo"""
    demo = ContextSynchronizationDemo()
    
    try:
        await demo.start()
        
        # Run demonstrations
        await demo.demo_context_sharing()
        await demo.demo_conflict_resolution()
        await demo.demo_notification_system()
        await demo.demo_access_control()
        
        logger.info("\n=== Demo Summary ===")
        logger.info(f"Total contexts created: {len(demo.contexts)}")
        logger.info(f"Total subscriptions: {len(demo.subscriptions)}")
        logger.info(f"Total notifications: {len(demo.notifications)}")
        
    finally:
        await demo.stop()


if __name__ == "__main__":
    asyncio.run(main())