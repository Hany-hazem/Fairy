# tests/test_context_synchronization.py
"""
Tests for Context Synchronization Engine

This module tests the context sharing mechanisms, conflict resolution,
and notification system functionality.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.context_synchronizer import (
    ContextSynchronizer, ContextConflict, ContextConflictType, 
    ContextMergeStrategy, ContextResolution, ContextBroadcastResult
)
from app.context_notification_system import (
    ContextNotificationSystem, ContextSubscription, SubscriptionType,
    SubscriptionFilter, NotificationPreferences, NotificationDeliveryMethod,
    ContextNotification, NotificationPriority
)
from app.mcp_models import AgentContext, ContextAccessLevel
from app.redis_mcp_backend import RedisMCPBackend


class TestContextSynchronizer:
    """Test cases for Context Synchronizer"""
    
    @pytest.fixture
    async def mock_redis_backend(self):
        """Mock Redis backend for testing"""
        backend = AsyncMock(spec=RedisMCPBackend)
        backend.connect.return_value = True
        backend.disconnect.return_value = None
        backend.publish_message.return_value = True
        backend._redis = AsyncMock()
        backend._redis.ping.return_value = True
        backend._redis.setex.return_value = True
        backend._redis.get.return_value = None
        backend._redis.sadd.return_value = 1
        backend._redis.expire.return_value = True
        backend._redis.hset.return_value = True
        backend._redis.delete.return_value = True
        return backend
    
    @pytest.fixture
    async def context_synchronizer(self, mock_redis_backend):
        """Create context synchronizer with mocked backend"""
        synchronizer = ContextSynchronizer(mock_redis_backend)
        await synchronizer.start()
        yield synchronizer
        await synchronizer.stop()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample agent context for testing"""
        return AgentContext(
            agent_id="test_agent_1",
            context_type="task_context",
            context_data={"task_id": "task_123", "status": "in_progress", "data": {"key": "value"}},
            access_level=ContextAccessLevel.PUBLIC.value,
            shared_with=["test_agent_2", "test_agent_3"]
        )
    
    @pytest.mark.asyncio
    async def test_context_synchronizer_initialization(self, mock_redis_backend):
        """Test context synchronizer initialization"""
        synchronizer = ContextSynchronizer(mock_redis_backend)
        
        assert synchronizer.redis_backend == mock_redis_backend
        assert synchronizer._running == False
        assert len(synchronizer._local_contexts) == 0
        assert len(synchronizer._active_conflicts) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop_synchronizer(self, mock_redis_backend):
        """Test starting and stopping the synchronizer"""
        synchronizer = ContextSynchronizer(mock_redis_backend)
        
        # Test start
        await synchronizer.start()
        assert synchronizer._running == True
        mock_redis_backend.connect.assert_called_once()
        
        # Test stop
        await synchronizer.stop()
        assert synchronizer._running == False
        mock_redis_backend.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broadcast_context_update(self, context_synchronizer, sample_context):
        """Test broadcasting context updates"""
        # Set up subscription targets
        context_synchronizer._context_subscriptions["task_context"] = {"test_agent_2", "test_agent_3"}
        
        # Broadcast context update
        result = await context_synchronizer.broadcast_context_update(sample_context)
        
        assert isinstance(result, ContextBroadcastResult)
        assert result.success == True
        assert len(result.target_agents) == 2
        assert "test_agent_2" in result.target_agents
        assert "test_agent_3" in result.target_agents
        assert len(result.successful_deliveries) == 2
        assert len(result.failed_deliveries) == 0
    
    @pytest.mark.asyncio
    async def test_broadcast_context_update_with_targets(self, context_synchronizer, sample_context):
        """Test broadcasting context updates with specific targets"""
        targets = ["test_agent_2"]
        
        result = await context_synchronizer.broadcast_context_update(sample_context, targets)
        
        assert isinstance(result, ContextBroadcastResult)
        assert result.success == True
        assert result.target_agents == targets
        assert len(result.successful_deliveries) == 1
        assert "test_agent_2" in result.successful_deliveries
    
    @pytest.mark.asyncio
    async def test_sync_agent_context(self, context_synchronizer, sample_context):
        """Test synchronizing context for specific agent"""
        result = await context_synchronizer.sync_agent_context("test_agent_2", sample_context)
        
        assert result == True
        # Verify context was stored locally
        assert "test_agent_1" in context_synchronizer._local_contexts
        stored_context = context_synchronizer._local_contexts["test_agent_1"]
        assert stored_context.context_type == sample_context.context_type
    
    @pytest.mark.asyncio
    async def test_context_conflict_detection(self, context_synchronizer, sample_context):
        """Test context conflict detection"""
        # Store existing context
        existing_context = AgentContext(
            agent_id="test_agent_1",
            context_type="task_context",
            context_data={"task_id": "task_123", "status": "completed"},
            version="old_version",
            last_updated=datetime.utcnow() - timedelta(minutes=5)
        )
        context_synchronizer._local_contexts["test_agent_2"] = existing_context
        
        # Create newer context with different data
        new_context = sample_context
        new_context.last_updated = datetime.utcnow()
        
        # Detect conflict
        conflict = await context_synchronizer._detect_context_conflict(new_context, "test_agent_2")
        
        assert conflict is not None
        assert conflict.conflict_type == ContextConflictType.DATA_CONFLICT
        assert conflict.agent_id == "test_agent_2"
        assert conflict.local_context == existing_context
        assert conflict.remote_context == new_context
    
    @pytest.mark.asyncio
    async def test_resolve_context_conflict_latest_wins(self, context_synchronizer):
        """Test context conflict resolution with latest wins strategy"""
        # Create conflict
        local_context = AgentContext(
            agent_id="test_agent_1",
            context_type="task_context",
            context_data={"status": "old"},
            last_updated=datetime.utcnow() - timedelta(minutes=5)
        )
        
        remote_context = AgentContext(
            agent_id="test_agent_1",
            context_type="task_context",
            context_data={"status": "new"},
            last_updated=datetime.utcnow()
        )
        
        conflict = ContextConflict(
            conflict_id="test_conflict",
            conflict_type=ContextConflictType.DATA_CONFLICT,
            context_type="task_context",
            agent_id="test_agent_1",
            conflicting_versions=["old", "new"],
            local_context=local_context,
            remote_context=remote_context,
            detected_at=datetime.utcnow(),
            resolution_strategy=ContextMergeStrategy.LATEST_WINS
        )
        
        # Resolve conflict
        resolution = await context_synchronizer.resolve_context_conflict([conflict])
        
        assert resolution.resolved == True
        assert resolution.resolution_strategy == ContextMergeStrategy.LATEST_WINS
        assert resolution.merged_context.context_data["status"] == "new"
    
    @pytest.mark.asyncio
    async def test_resolve_context_conflict_merge_recursive(self, context_synchronizer):
        """Test context conflict resolution with recursive merge strategy"""
        # Create conflict with mergeable data
        local_context = AgentContext(
            agent_id="test_agent_1",
            context_type="task_context",
            context_data={"field1": "local", "shared": {"local_key": "local_value"}}
        )
        
        remote_context = AgentContext(
            agent_id="test_agent_1",
            context_type="task_context",
            context_data={"field2": "remote", "shared": {"remote_key": "remote_value"}}
        )
        
        conflict = ContextConflict(
            conflict_id="test_conflict",
            conflict_type=ContextConflictType.DATA_CONFLICT,
            context_type="task_context",
            agent_id="test_agent_1",
            conflicting_versions=["local", "remote"],
            local_context=local_context,
            remote_context=remote_context,
            detected_at=datetime.utcnow(),
            resolution_strategy=ContextMergeStrategy.MERGE_RECURSIVE
        )
        
        # Resolve conflict
        resolution = await context_synchronizer.resolve_context_conflict([conflict])
        
        assert resolution.resolved == True
        assert resolution.resolution_strategy == ContextMergeStrategy.MERGE_RECURSIVE
        merged_data = resolution.merged_context.context_data
        assert "field1" in merged_data
        assert "field2" in merged_data
        assert "local_key" in merged_data["shared"]
        assert "remote_key" in merged_data["shared"]
    
    @pytest.mark.asyncio
    async def test_get_shared_context(self, context_synchronizer, sample_context):
        """Test getting shared context"""
        # Store context locally
        await context_synchronizer._store_context_locally(sample_context)
        
        # Get shared context
        retrieved_context = await context_synchronizer.get_shared_context("test_agent_2", "task_context")
        
        assert retrieved_context is not None
        assert retrieved_context.context_type == "task_context"
        assert retrieved_context.agent_id == "test_agent_1"
        assert retrieved_context.can_access("test_agent_2")
    
    @pytest.mark.asyncio
    async def test_get_shared_context_access_denied(self, context_synchronizer):
        """Test getting shared context with access denied"""
        # Create context with restricted access
        restricted_context = AgentContext(
            agent_id="test_agent_1",
            context_type="task_context",
            context_data={"sensitive": "data"},
            access_level=ContextAccessLevel.PRIVATE.value
        )
        
        await context_synchronizer._store_context_locally(restricted_context)
        
        # Try to get context as different agent
        retrieved_context = await context_synchronizer.get_shared_context("test_agent_2", "task_context")
        
        assert retrieved_context is None
    
    @pytest.mark.asyncio
    async def test_subscribe_to_context_updates(self, context_synchronizer):
        """Test subscribing to context updates"""
        callback = AsyncMock()
        
        subscription_id = await context_synchronizer.subscribe_to_context_updates(
            "test_agent_1", "task_context", callback
        )
        
        assert subscription_id is not None
        assert "task_context" in context_synchronizer._context_subscriptions
        assert "test_agent_1" in context_synchronizer._context_subscriptions["task_context"]
        assert subscription_id in context_synchronizer._subscription_callbacks
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_context_updates(self, context_synchronizer):
        """Test unsubscribing from context updates"""
        callback = AsyncMock()
        
        # Subscribe first
        subscription_id = await context_synchronizer.subscribe_to_context_updates(
            "test_agent_1", "task_context", callback
        )
        
        # Unsubscribe
        result = await context_synchronizer.unsubscribe_from_context_updates(
            subscription_id, "test_agent_1", "task_context"
        )
        
        assert result == True
        assert subscription_id not in context_synchronizer._subscription_callbacks
    
    @pytest.mark.asyncio
    async def test_context_statistics(self, context_synchronizer, sample_context):
        """Test getting context statistics"""
        # Add some test data
        await context_synchronizer._store_context_locally(sample_context)
        callback = AsyncMock()
        await context_synchronizer.subscribe_to_context_updates("test_agent_1", "task_context", callback)
        
        stats = await context_synchronizer.get_context_statistics()
        
        assert "local_contexts" in stats
        assert "context_subscriptions" in stats
        assert "running" in stats
        assert stats["local_contexts"] >= 1
        assert stats["running"] == True


class TestContextNotificationSystem:
    """Test cases for Context Notification System"""
    
    @pytest.fixture
    async def mock_redis_backend(self):
        """Mock Redis backend for testing"""
        backend = AsyncMock(spec=RedisMCPBackend)
        backend.connect.return_value = True
        backend.disconnect.return_value = None
        backend.publish_message.return_value = True
        backend._redis = AsyncMock()
        backend._redis.setex.return_value = True
        return backend
    
    @pytest.fixture
    async def notification_system(self, mock_redis_backend):
        """Create notification system with mocked backend"""
        system = ContextNotificationSystem(mock_redis_backend)
        await system.start()
        yield system
        await system.stop()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample agent context for testing"""
        return AgentContext(
            agent_id="test_agent_1",
            context_type="task_context",
            context_data={"task_id": "task_123", "status": "in_progress"},
            access_level=ContextAccessLevel.PUBLIC.value
        )
    
    @pytest.fixture
    def subscription_filter(self):
        """Create sample subscription filter"""
        return SubscriptionFilter(
            context_types=["task_context"],
            exclude_self=True
        )
    
    @pytest.mark.asyncio
    async def test_notification_system_initialization(self, mock_redis_backend):
        """Test notification system initialization"""
        system = ContextNotificationSystem(mock_redis_backend)
        
        assert system.redis_backend == mock_redis_backend
        assert system._running == False
        assert len(system._subscriptions) == 0
        assert len(system._notification_queues) == 0
    
    @pytest.mark.asyncio
    async def test_subscribe_to_context_updates(self, notification_system, subscription_filter):
        """Test subscribing to context updates"""
        callback = AsyncMock()
        preferences = NotificationPreferences(delivery_method=NotificationDeliveryMethod.PUSH)
        
        subscription_id = await notification_system.subscribe_to_context_updates(
            "test_agent_1",
            SubscriptionType.CONTEXT_TYPE,
            subscription_filter,
            callback,
            preferences
        )
        
        assert subscription_id is not None
        assert subscription_id in notification_system._subscriptions
        subscription = notification_system._subscriptions[subscription_id]
        assert subscription.agent_id == "test_agent_1"
        assert subscription.subscription_type == SubscriptionType.CONTEXT_TYPE
        assert subscription.callback == callback
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_context_updates(self, notification_system, subscription_filter):
        """Test unsubscribing from context updates"""
        callback = AsyncMock()
        
        # Subscribe first
        subscription_id = await notification_system.subscribe_to_context_updates(
            "test_agent_1",
            SubscriptionType.CONTEXT_TYPE,
            subscription_filter,
            callback
        )
        
        # Unsubscribe
        result = await notification_system.unsubscribe_from_context_updates(subscription_id)
        
        assert result == True
        assert subscription_id not in notification_system._subscriptions
    
    @pytest.mark.asyncio
    async def test_notify_context_update(self, notification_system, sample_context, subscription_filter):
        """Test notifying context updates"""
        # Create subscription
        callback = AsyncMock()
        await notification_system.subscribe_to_context_updates(
            "test_agent_2",  # Different agent to receive notifications
            SubscriptionType.CONTEXT_TYPE,
            subscription_filter,
            callback
        )
        
        # Notify context update
        notification_ids = await notification_system.notify_context_update(
            sample_context,
            "updated",
            NotificationPriority.NORMAL
        )
        
        assert len(notification_ids) == 1
        assert "test_agent_2" in notification_system._notification_queues
        assert len(notification_system._notification_queues["test_agent_2"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_pending_notifications(self, notification_system, sample_context, subscription_filter):
        """Test getting pending notifications"""
        # Create subscription with pull delivery
        preferences = NotificationPreferences(delivery_method=NotificationDeliveryMethod.PULL)
        await notification_system.subscribe_to_context_updates(
            "test_agent_2",
            SubscriptionType.CONTEXT_TYPE,
            subscription_filter,
            preferences=preferences
        )
        
        # Create notification
        await notification_system.notify_context_update(sample_context)
        
        # Get pending notifications
        notifications = await notification_system.get_pending_notifications("test_agent_2")
        
        assert len(notifications) == 1
        notification = notifications[0]
        assert notification.agent_id == "test_agent_2"
        assert notification.context.context_type == "task_context"
        assert notification.delivered == True
    
    @pytest.mark.asyncio
    async def test_create_notification_batch(self, notification_system, sample_context, subscription_filter):
        """Test creating notification batches"""
        # Create subscription with batch delivery
        preferences = NotificationPreferences(
            delivery_method=NotificationDeliveryMethod.BATCH,
            batch_size=2
        )
        await notification_system.subscribe_to_context_updates(
            "test_agent_2",
            SubscriptionType.CONTEXT_TYPE,
            subscription_filter,
            preferences=preferences
        )
        
        # Create multiple notifications
        await notification_system.notify_context_update(sample_context)
        await notification_system.notify_context_update(sample_context)
        
        # Create batch
        batch = await notification_system.create_notification_batch("test_agent_2")
        
        assert batch is not None
        assert batch.agent_id == "test_agent_2"
        assert len(batch.notifications) == 2
        assert batch.delivery_method == NotificationDeliveryMethod.BATCH
    
    @pytest.mark.asyncio
    async def test_access_control(self, notification_system, subscription_filter):
        """Test access control for notifications"""
        # Create restricted context
        restricted_context = AgentContext(
            agent_id="test_agent_1",
            context_type="task_context",
            context_data={"sensitive": "data"},
            access_level=ContextAccessLevel.PRIVATE.value
        )
        
        # Create subscription
        await notification_system.subscribe_to_context_updates(
            "test_agent_2",
            SubscriptionType.CONTEXT_TYPE,
            subscription_filter
        )
        
        # Try to notify - should be blocked by access control
        notification_ids = await notification_system.notify_context_update(restricted_context)
        
        assert len(notification_ids) == 0  # No notifications created due to access control
    
    @pytest.mark.asyncio
    async def test_subscription_statistics(self, notification_system, subscription_filter):
        """Test getting subscription statistics"""
        # Create subscription
        await notification_system.subscribe_to_context_updates(
            "test_agent_1",
            SubscriptionType.CONTEXT_TYPE,
            subscription_filter
        )
        
        # Get global stats
        global_stats = await notification_system.get_subscription_statistics()
        assert "total_subscriptions" in global_stats
        assert global_stats["total_subscriptions"] == 1
        
        # Get agent-specific stats
        agent_stats = await notification_system.get_subscription_statistics("test_agent_1")
        assert "agent_id" in agent_stats
        assert agent_stats["active_subscriptions"] == 1
    
    @pytest.mark.asyncio
    async def test_filter_matching(self, notification_system):
        """Test subscription filter matching"""
        # Test context type filtering
        filter_criteria = SubscriptionFilter(context_types=["task_context"])
        
        # Matching context
        matching_context = AgentContext(
            agent_id="test_agent_1",
            context_type="task_context",
            context_data={}
        )
        
        # Non-matching context
        non_matching_context = AgentContext(
            agent_id="test_agent_1",
            context_type="project_context",
            context_data={}
        )
        
        # Test matching
        matches = await notification_system._matches_filter_criteria(matching_context, filter_criteria)
        assert matches == True
        
        no_matches = await notification_system._matches_filter_criteria(non_matching_context, filter_criteria)
        assert no_matches == False
    
    @pytest.mark.asyncio
    async def test_deduplication(self, notification_system, sample_context, subscription_filter):
        """Test notification deduplication"""
        # Create subscription with deduplication enabled
        preferences = NotificationPreferences(enable_deduplication=True)
        await notification_system.subscribe_to_context_updates(
            "test_agent_2",
            SubscriptionType.CONTEXT_TYPE,
            subscription_filter,
            preferences=preferences
        )
        
        # Send same notification twice
        await notification_system.notify_context_update(sample_context)
        await notification_system.notify_context_update(sample_context)
        
        # Should only have one notification due to deduplication
        notifications = await notification_system.get_pending_notifications("test_agent_2")
        assert len(notifications) == 1


if __name__ == "__main__":
    pytest.main([__file__])