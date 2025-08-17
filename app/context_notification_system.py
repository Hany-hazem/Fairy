# app/context_notification_system.py
"""
Context Subscription and Notification System

This module implements context update subscriptions for agents with efficient
context sharing for large datasets and access control permissions.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .config import settings
from .mcp_models import AgentContext, MCPMessage, MCPMessageType, ContextAccessLevel
from .redis_mcp_backend import RedisMCPBackend

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class SubscriptionType(Enum):
    """Types of context subscriptions"""
    CONTEXT_TYPE = "context_type"  # Subscribe to all contexts of a specific type
    AGENT_CONTEXT = "agent_context"  # Subscribe to contexts from specific agent
    PATTERN_MATCH = "pattern_match"  # Subscribe based on pattern matching
    FIELD_WATCH = "field_watch"  # Subscribe to changes in specific fields
    ACCESS_LEVEL = "access_level"  # Subscribe based on access level


class NotificationDeliveryMethod(Enum):
    """Notification delivery methods"""
    PUSH = "push"  # Real-time push notifications
    PULL = "pull"  # Agent polls for notifications
    BATCH = "batch"  # Batched notifications
    WEBHOOK = "webhook"  # HTTP webhook delivery


@dataclass
class SubscriptionFilter:
    """Filter criteria for context subscriptions"""
    context_types: Optional[List[str]] = None
    agent_ids: Optional[List[str]] = None
    access_levels: Optional[List[str]] = None
    field_patterns: Optional[Dict[str, Any]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    min_priority: Optional[int] = None
    exclude_self: bool = True  # Exclude updates from the subscribing agent


@dataclass
class NotificationPreferences:
    """Agent notification preferences"""
    delivery_method: NotificationDeliveryMethod = NotificationDeliveryMethod.PUSH
    batch_size: int = 10
    batch_timeout: int = 30  # seconds
    max_queue_size: int = 1000
    enable_compression: bool = True
    compression_threshold: int = 1024  # bytes
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_deduplication: bool = True
    deduplication_window: int = 300  # seconds


@dataclass
class ContextSubscription:
    """Context subscription information"""
    subscription_id: str
    agent_id: str
    subscription_type: SubscriptionType
    filter_criteria: SubscriptionFilter
    callback: Optional[Callable] = None
    preferences: NotificationPreferences = field(default_factory=NotificationPreferences)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_notification: Optional[datetime] = None
    notification_count: int = 0
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextNotification:
    """Context update notification"""
    notification_id: str
    subscription_id: str
    agent_id: str
    context: AgentContext
    notification_type: str  # created, updated, deleted, access_changed
    priority: NotificationPriority
    created_at: datetime
    delivered: bool = False
    delivery_attempts: int = 0
    last_delivery_attempt: Optional[datetime] = None
    delivery_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationBatch:
    """Batch of notifications for efficient delivery"""
    batch_id: str
    agent_id: str
    notifications: List[ContextNotification]
    created_at: datetime
    size_bytes: int
    compressed: bool = False
    delivery_method: NotificationDeliveryMethod = NotificationDeliveryMethod.BATCH


class ContextNotificationSystem:
    """
    Context Subscription and Notification System
    
    Provides comprehensive context update notifications with:
    - Flexible subscription management with filtering
    - Multiple delivery methods (push, pull, batch, webhook)
    - Access control and permissions
    - Efficient handling of large datasets
    - Performance optimization and monitoring
    """
    
    def __init__(self, redis_backend: RedisMCPBackend = None):
        self.redis_backend = redis_backend or RedisMCPBackend()
        
        # Subscription management
        self._subscriptions: Dict[str, ContextSubscription] = {}  # subscription_id -> subscription
        self._agent_subscriptions: Dict[str, Set[str]] = {}  # agent_id -> subscription_ids
        self._context_type_subscriptions: Dict[str, Set[str]] = {}  # context_type -> subscription_ids
        
        # Notification queues and batching
        self._notification_queues: Dict[str, List[ContextNotification]] = {}  # agent_id -> notifications
        self._notification_batches: Dict[str, NotificationBatch] = {}  # batch_id -> batch
        self._pending_deliveries: Dict[str, ContextNotification] = {}  # notification_id -> notification
        
        # Access control and permissions
        self._access_control_rules: Dict[str, Dict[str, Any]] = {}  # agent_id -> rules
        self._permission_cache: Dict[str, Dict[str, bool]] = {}  # cache for permission checks
        
        # Performance optimization
        self._notification_cache: Dict[str, ContextNotification] = {}
        self._deduplication_cache: Dict[str, Set[str]] = {}  # agent_id -> notification_hashes
        self._delivery_stats: Dict[str, Dict[str, int]] = {}  # agent_id -> stats
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        # Configuration
        self.max_subscription_per_agent = 100
        self.max_notification_queue_size = 10000
        self.notification_ttl = 86400  # 24 hours
        self.batch_processing_interval = 10  # seconds
        self.cleanup_interval = 300  # 5 minutes
        
        logger.info("Context Notification System initialized")
    
    async def start(self):
        """Start the notification system"""
        try:
            if self._running:
                return
            
            # Connect to Redis backend
            if not await self.redis_backend.connect():
                raise RuntimeError("Failed to connect to Redis backend")
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._running = True
            logger.info("Context Notification System started")
            
        except Exception as e:
            logger.error(f"Failed to start Context Notification System: {e}")
            raise
    
    async def stop(self):
        """Stop the notification system"""
        try:
            self._running = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Disconnect from Redis
            await self.redis_backend.disconnect()
            
            logger.info("Context Notification System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Context Notification System: {e}")
    
    async def subscribe_to_context_updates(
        self,
        agent_id: str,
        subscription_type: SubscriptionType,
        filter_criteria: SubscriptionFilter,
        callback: Optional[Callable] = None,
        preferences: NotificationPreferences = None
    ) -> str:
        """
        Subscribe agent to context updates with filtering and preferences
        
        Args:
            agent_id: Agent ID subscribing
            subscription_type: Type of subscription
            filter_criteria: Filter criteria for notifications
            callback: Optional callback function for push notifications
            preferences: Notification preferences
            
        Returns:
            Subscription ID
        """
        try:
            # Check subscription limits
            agent_subscription_count = len(self._agent_subscriptions.get(agent_id, set()))
            if agent_subscription_count >= self.max_subscription_per_agent:
                raise ValueError(f"Agent {agent_id} has reached maximum subscription limit")
            
            # Create subscription
            subscription_id = str(uuid.uuid4())
            subscription = ContextSubscription(
                subscription_id=subscription_id,
                agent_id=agent_id,
                subscription_type=subscription_type,
                filter_criteria=filter_criteria,
                callback=callback,
                preferences=preferences or NotificationPreferences()
            )
            
            # Store subscription
            self._subscriptions[subscription_id] = subscription
            
            # Update agent subscription tracking
            if agent_id not in self._agent_subscriptions:
                self._agent_subscriptions[agent_id] = set()
            self._agent_subscriptions[agent_id].add(subscription_id)
            
            # Update context type subscription tracking
            if filter_criteria.context_types:
                for context_type in filter_criteria.context_types:
                    if context_type not in self._context_type_subscriptions:
                        self._context_type_subscriptions[context_type] = set()
                    self._context_type_subscriptions[context_type].add(subscription_id)
            
            # Initialize notification queue for agent
            if agent_id not in self._notification_queues:
                self._notification_queues[agent_id] = []
            
            # Initialize delivery stats
            if agent_id not in self._delivery_stats:
                self._delivery_stats[agent_id] = {
                    "total_notifications": 0,
                    "delivered_notifications": 0,
                    "failed_deliveries": 0,
                    "batched_notifications": 0
                }
            
            # Store subscription in Redis for persistence
            await self._store_subscription_in_redis(subscription)
            
            # Set up Redis subscription for real-time notifications
            if subscription.preferences.delivery_method == NotificationDeliveryMethod.PUSH:
                await self._setup_redis_subscription(subscription)
            
            logger.info(f"Created subscription {subscription_id} for agent {agent_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to create subscription for agent {agent_id}: {e}")
            raise
    
    async def unsubscribe_from_context_updates(self, subscription_id: str) -> bool:
        """
        Unsubscribe from context updates
        
        Args:
            subscription_id: Subscription ID to remove
            
        Returns:
            True if unsubscribed successfully
        """
        try:
            if subscription_id not in self._subscriptions:
                logger.warning(f"Subscription {subscription_id} not found")
                return False
            
            subscription = self._subscriptions[subscription_id]
            agent_id = subscription.agent_id
            
            # Remove from tracking
            if agent_id in self._agent_subscriptions:
                self._agent_subscriptions[agent_id].discard(subscription_id)
                if not self._agent_subscriptions[agent_id]:
                    del self._agent_subscriptions[agent_id]
            
            # Remove from context type tracking
            if subscription.filter_criteria.context_types:
                for context_type in subscription.filter_criteria.context_types:
                    if context_type in self._context_type_subscriptions:
                        self._context_type_subscriptions[context_type].discard(subscription_id)
                        if not self._context_type_subscriptions[context_type]:
                            del self._context_type_subscriptions[context_type]
            
            # Remove subscription
            del self._subscriptions[subscription_id]
            
            # Remove from Redis
            await self._remove_subscription_from_redis(subscription_id)
            
            # Clean up Redis subscription
            if subscription.preferences.delivery_method == NotificationDeliveryMethod.PUSH:
                await self._cleanup_redis_subscription(subscription)
            
            logger.info(f"Removed subscription {subscription_id} for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove subscription {subscription_id}: {e}")
            return False
    
    async def notify_context_update(
        self,
        context: AgentContext,
        notification_type: str = "updated",
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> List[str]:
        """
        Notify subscribers about context updates
        
        Args:
            context: Updated context
            notification_type: Type of notification (created, updated, deleted, etc.)
            priority: Notification priority
            
        Returns:
            List of notification IDs created
        """
        try:
            notification_ids = []
            
            # Find matching subscriptions
            matching_subscriptions = await self._find_matching_subscriptions(context)
            
            for subscription in matching_subscriptions:
                # Check access permissions
                if not await self._check_access_permission(subscription.agent_id, context):
                    continue
                
                # Create notification
                notification = ContextNotification(
                    notification_id=str(uuid.uuid4()),
                    subscription_id=subscription.subscription_id,
                    agent_id=subscription.agent_id,
                    context=context,
                    notification_type=notification_type,
                    priority=priority,
                    created_at=datetime.utcnow()
                )
                
                # Check for deduplication
                if subscription.preferences.enable_deduplication:
                    if await self._is_duplicate_notification(notification):
                        continue
                
                # Add to notification queue
                await self._add_notification_to_queue(notification)
                notification_ids.append(notification.notification_id)
                
                # Update subscription stats
                subscription.notification_count += 1
                subscription.last_notification = datetime.utcnow()
            
            logger.debug(f"Created {len(notification_ids)} notifications for context update")
            return notification_ids
            
        except Exception as e:
            logger.error(f"Failed to notify context update: {e}")
            return []
    
    async def get_pending_notifications(self, agent_id: str, limit: int = 100) -> List[ContextNotification]:
        """
        Get pending notifications for an agent (pull method)
        
        Args:
            agent_id: Agent ID to get notifications for
            limit: Maximum number of notifications to return
            
        Returns:
            List of pending notifications
        """
        try:
            if agent_id not in self._notification_queues:
                return []
            
            # Get notifications from queue
            notifications = self._notification_queues[agent_id][:limit]
            
            # Mark as delivered and remove from queue
            delivered_notifications = []
            for notification in notifications:
                notification.delivered = True
                notification.last_delivery_attempt = datetime.utcnow()
                delivered_notifications.append(notification)
                
                # Update stats
                self._delivery_stats[agent_id]["delivered_notifications"] += 1
            
            # Remove delivered notifications from queue
            self._notification_queues[agent_id] = self._notification_queues[agent_id][limit:]
            
            logger.debug(f"Delivered {len(delivered_notifications)} notifications to agent {agent_id}")
            return delivered_notifications
            
        except Exception as e:
            logger.error(f"Failed to get pending notifications for agent {agent_id}: {e}")
            return []
    
    async def create_notification_batch(self, agent_id: str, max_size: int = None) -> Optional[NotificationBatch]:
        """
        Create a batch of notifications for efficient delivery
        
        Args:
            agent_id: Agent ID to create batch for
            max_size: Maximum batch size in bytes
            
        Returns:
            Notification batch if created
        """
        try:
            if agent_id not in self._notification_queues:
                return None
            
            notifications = self._notification_queues[agent_id]
            if not notifications:
                return None
            
            # Get agent preferences
            agent_subscriptions = self._agent_subscriptions.get(agent_id, set())
            if not agent_subscriptions:
                return None
            
            # Use preferences from first subscription (can be enhanced)
            first_subscription = self._subscriptions[next(iter(agent_subscriptions))]
            preferences = first_subscription.preferences
            
            # Determine batch size
            batch_size = min(preferences.batch_size, len(notifications))
            if max_size:
                # Calculate size and adjust batch
                current_size = 0
                actual_batch_size = 0
                for i, notification in enumerate(notifications[:batch_size]):
                    notification_size = len(json.dumps(notification.context.dict()))
                    if current_size + notification_size > max_size and i > 0:
                        break
                    current_size += notification_size
                    actual_batch_size = i + 1
                batch_size = actual_batch_size
            
            if batch_size == 0:
                return None
            
            # Create batch
            batch_notifications = notifications[:batch_size]
            batch = NotificationBatch(
                batch_id=str(uuid.uuid4()),
                agent_id=agent_id,
                notifications=batch_notifications,
                created_at=datetime.utcnow(),
                size_bytes=sum(len(json.dumps(n.context.dict())) for n in batch_notifications),
                delivery_method=NotificationDeliveryMethod.BATCH
            )
            
            # Apply compression if enabled and threshold met
            if (preferences.enable_compression and 
                batch.size_bytes > preferences.compression_threshold):
                batch.compressed = True
            
            # Remove batched notifications from queue
            self._notification_queues[agent_id] = self._notification_queues[agent_id][batch_size:]
            
            # Store batch
            self._notification_batches[batch.batch_id] = batch
            
            # Update stats
            self._delivery_stats[agent_id]["batched_notifications"] += len(batch_notifications)
            
            logger.debug(f"Created notification batch {batch.batch_id} with {len(batch_notifications)} notifications")
            return batch
            
        except Exception as e:
            logger.error(f"Failed to create notification batch for agent {agent_id}: {e}")
            return None
    
    async def set_access_control_rules(self, agent_id: str, rules: Dict[str, Any]):
        """
        Set access control rules for an agent
        
        Args:
            agent_id: Agent ID to set rules for
            rules: Access control rules
        """
        try:
            self._access_control_rules[agent_id] = rules
            
            # Clear permission cache for this agent
            if agent_id in self._permission_cache:
                del self._permission_cache[agent_id]
            
            # Store rules in Redis
            await self._store_access_rules_in_redis(agent_id, rules)
            
            logger.info(f"Set access control rules for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to set access control rules for agent {agent_id}: {e}")
    
    async def check_context_access(self, agent_id: str, context: AgentContext) -> bool:
        """
        Check if agent has access to context
        
        Args:
            agent_id: Agent ID to check
            context: Context to check access for
            
        Returns:
            True if agent has access
        """
        return await self._check_access_permission(agent_id, context)
    
    async def get_subscription_statistics(self, agent_id: str = None) -> Dict[str, Any]:
        """
        Get subscription and notification statistics
        
        Args:
            agent_id: Optional agent ID to get specific stats
            
        Returns:
            Statistics dictionary
        """
        try:
            if agent_id:
                # Get stats for specific agent
                agent_subscriptions = self._agent_subscriptions.get(agent_id, set())
                agent_queue_size = len(self._notification_queues.get(agent_id, []))
                agent_stats = self._delivery_stats.get(agent_id, {})
                
                return {
                    "agent_id": agent_id,
                    "active_subscriptions": len(agent_subscriptions),
                    "pending_notifications": agent_queue_size,
                    "delivery_stats": agent_stats
                }
            else:
                # Get global stats
                total_subscriptions = len(self._subscriptions)
                total_agents = len(self._agent_subscriptions)
                total_pending = sum(len(queue) for queue in self._notification_queues.values())
                total_batches = len(self._notification_batches)
                
                return {
                    "total_subscriptions": total_subscriptions,
                    "total_agents": total_agents,
                    "total_pending_notifications": total_pending,
                    "total_batches": total_batches,
                    "context_type_subscriptions": {
                        ct: len(subs) for ct, subs in self._context_type_subscriptions.items()
                    },
                    "running": self._running
                }
                
        except Exception as e:
            logger.error(f"Failed to get subscription statistics: {e}")
            return {}
    
    async def _find_matching_subscriptions(self, context: AgentContext) -> List[ContextSubscription]:
        """Find subscriptions that match the context update"""
        matching_subscriptions = []
        
        try:
            # Check context type subscriptions
            context_type_subs = self._context_type_subscriptions.get(context.context_type, set())
            
            for subscription_id in context_type_subs:
                if subscription_id not in self._subscriptions:
                    continue
                
                subscription = self._subscriptions[subscription_id]
                
                # Skip inactive subscriptions
                if not subscription.active:
                    continue
                
                # Check filter criteria
                if await self._matches_filter_criteria(context, subscription.filter_criteria):
                    matching_subscriptions.append(subscription)
            
            # Also check agent-specific subscriptions
            for subscription in self._subscriptions.values():
                if subscription.subscription_type == SubscriptionType.AGENT_CONTEXT:
                    if (subscription.filter_criteria.agent_ids and 
                        context.agent_id in subscription.filter_criteria.agent_ids):
                        if await self._matches_filter_criteria(context, subscription.filter_criteria):
                            matching_subscriptions.append(subscription)
            
            return matching_subscriptions
            
        except Exception as e:
            logger.error(f"Error finding matching subscriptions: {e}")
            return [] 
   
    async def _matches_filter_criteria(self, context: AgentContext, filter_criteria: SubscriptionFilter) -> bool:
        """Check if context matches subscription filter criteria"""
        try:
            # Check context types
            if filter_criteria.context_types:
                if context.context_type not in filter_criteria.context_types:
                    return False
            
            # Check agent IDs
            if filter_criteria.agent_ids:
                if context.agent_id not in filter_criteria.agent_ids:
                    return False
            
            # Check access levels
            if filter_criteria.access_levels:
                if context.access_level not in filter_criteria.access_levels:
                    return False
            
            # Check field patterns
            if filter_criteria.field_patterns:
                for field_path, pattern in filter_criteria.field_patterns.items():
                    field_value = self._get_nested_field_value(context.context_data, field_path)
                    if not self._matches_pattern(field_value, pattern):
                        return False
            
            # Check metadata filters
            if filter_criteria.metadata_filters:
                for key, expected_value in filter_criteria.metadata_filters.items():
                    if key not in context.metadata or context.metadata[key] != expected_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching filter criteria: {e}")
            return False
    
    def _get_nested_field_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        try:
            keys = field_path.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except Exception:
            return None
    
    def _matches_pattern(self, value: Any, pattern: Any) -> bool:
        """Check if value matches pattern (supports regex and simple patterns)"""
        try:
            if isinstance(pattern, dict):
                if "$regex" in pattern:
                    import re
                    return bool(re.match(pattern["$regex"], str(value)))
                elif "$in" in pattern:
                    return value in pattern["$in"]
                elif "$gt" in pattern:
                    return value > pattern["$gt"]
                elif "$lt" in pattern:
                    return value < pattern["$lt"]
            else:
                return value == pattern
        except Exception:
            return False
    
    async def _check_access_permission(self, agent_id: str, context: AgentContext) -> bool:
        """Check if agent has permission to access context"""
        try:
            # Check cache first
            cache_key = f"{agent_id}:{context.agent_id}:{context.context_type}"
            if agent_id in self._permission_cache and cache_key in self._permission_cache[agent_id]:
                return self._permission_cache[agent_id][cache_key]
            
            # Use context's built-in access control
            has_access = context.can_access(agent_id)
            
            # Apply additional access control rules if configured
            if agent_id in self._access_control_rules:
                rules = self._access_control_rules[agent_id]
                
                # Check context type restrictions
                if "allowed_context_types" in rules:
                    if context.context_type not in rules["allowed_context_types"]:
                        has_access = False
                
                # Check access level restrictions
                if "max_access_level" in rules:
                    max_level = rules["max_access_level"]
                    access_levels = [level.value for level in ContextAccessLevel]
                    if access_levels.index(context.access_level) > access_levels.index(max_level):
                        has_access = False
                
                # Check agent restrictions
                if "blocked_agents" in rules:
                    if context.agent_id in rules["blocked_agents"]:
                        has_access = False
            
            # Cache result
            if agent_id not in self._permission_cache:
                self._permission_cache[agent_id] = {}
            self._permission_cache[agent_id][cache_key] = has_access
            
            return has_access
            
        except Exception as e:
            logger.error(f"Error checking access permission: {e}")
            return False
    
    async def _is_duplicate_notification(self, notification: ContextNotification) -> bool:
        """Check if notification is a duplicate"""
        try:
            agent_id = notification.agent_id
            
            # Create notification hash
            notification_hash = self._create_notification_hash(notification)
            
            # Check deduplication cache
            if agent_id in self._deduplication_cache:
                if notification_hash in self._deduplication_cache[agent_id]:
                    return True
                
                # Add to cache
                self._deduplication_cache[agent_id].add(notification_hash)
            else:
                self._deduplication_cache[agent_id] = {notification_hash}
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate notification: {e}")
            return False
    
    def _create_notification_hash(self, notification: ContextNotification) -> str:
        """Create hash for notification deduplication"""
        try:
            # Create hash based on context content and type
            import hashlib
            
            hash_data = {
                "context_type": notification.context.context_type,
                "context_version": notification.context.version,
                "notification_type": notification.notification_type,
                "agent_id": notification.context.agent_id
            }
            
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.md5(hash_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error creating notification hash: {e}")
            return str(uuid.uuid4())
    
    async def _add_notification_to_queue(self, notification: ContextNotification):
        """Add notification to agent's queue"""
        try:
            agent_id = notification.agent_id
            
            # Initialize queue if needed
            if agent_id not in self._notification_queues:
                self._notification_queues[agent_id] = []
            
            # Check queue size limits
            queue = self._notification_queues[agent_id]
            if len(queue) >= self.max_notification_queue_size:
                # Remove oldest notification
                removed = queue.pop(0)
                logger.warning(f"Queue full for agent {agent_id}, removed notification {removed.notification_id}")
            
            # Add notification
            queue.append(notification)
            
            # Store in pending deliveries
            self._pending_deliveries[notification.notification_id] = notification
            
            # Update stats
            if agent_id in self._delivery_stats:
                self._delivery_stats[agent_id]["total_notifications"] += 1
            
            # Store notification in Redis for persistence
            await self._store_notification_in_redis(notification)
            
            # Trigger immediate delivery for push notifications
            subscription = self._subscriptions.get(notification.subscription_id)
            if (subscription and 
                subscription.preferences.delivery_method == NotificationDeliveryMethod.PUSH):
                await self._deliver_push_notification(notification)
            
        except Exception as e:
            logger.error(f"Error adding notification to queue: {e}")
    
    async def _deliver_push_notification(self, notification: ContextNotification):
        """Deliver push notification immediately"""
        try:
            subscription = self._subscriptions.get(notification.subscription_id)
            if not subscription:
                return
            
            # Use callback if available
            if subscription.callback:
                try:
                    await subscription.callback(notification)
                    notification.delivered = True
                    notification.last_delivery_attempt = datetime.utcnow()
                    
                    # Update stats
                    agent_id = notification.agent_id
                    if agent_id in self._delivery_stats:
                        self._delivery_stats[agent_id]["delivered_notifications"] += 1
                    
                except Exception as e:
                    logger.error(f"Error in notification callback: {e}")
                    notification.delivery_error = str(e)
                    notification.delivery_attempts += 1
                    
                    # Update stats
                    if notification.agent_id in self._delivery_stats:
                        self._delivery_stats[notification.agent_id]["failed_deliveries"] += 1
            
            else:
                # Use Redis pub/sub for push delivery
                topic = f"mcp:agent:{notification.agent_id}:notifications"
                message_data = {
                    "notification_id": notification.notification_id,
                    "context": notification.context.dict(),
                    "notification_type": notification.notification_type,
                    "priority": notification.priority.value,
                    "created_at": notification.created_at.isoformat()
                }
                
                # Create MCP message
                from .mcp_models import MCPMessage, MCPMessageType
                mcp_message = MCPMessage(
                    type=MCPMessageType.CONTEXT_UPDATE.value,
                    source_agent="context_notification_system",
                    target_agents=[notification.agent_id],
                    payload=message_data,
                    priority=notification.priority.value
                )
                
                success = await self.redis_backend.publish_message(topic, mcp_message)
                if success:
                    notification.delivered = True
                    notification.last_delivery_attempt = datetime.utcnow()
                    
                    # Update stats
                    if notification.agent_id in self._delivery_stats:
                        self._delivery_stats[notification.agent_id]["delivered_notifications"] += 1
                else:
                    notification.delivery_attempts += 1
                    notification.delivery_error = "Failed to publish to Redis"
                    
                    # Update stats
                    if notification.agent_id in self._delivery_stats:
                        self._delivery_stats[notification.agent_id]["failed_deliveries"] += 1
            
        except Exception as e:
            logger.error(f"Error delivering push notification: {e}")
            notification.delivery_error = str(e)
            notification.delivery_attempts += 1
    
    async def _store_subscription_in_redis(self, subscription: ContextSubscription):
        """Store subscription in Redis for persistence"""
        try:
            key = f"mcp:subscription:{subscription.subscription_id}"
            subscription_data = {
                "subscription_id": subscription.subscription_id,
                "agent_id": subscription.agent_id,
                "subscription_type": subscription.subscription_type.value,
                "filter_criteria": {
                    "context_types": subscription.filter_criteria.context_types,
                    "agent_ids": subscription.filter_criteria.agent_ids,
                    "access_levels": subscription.filter_criteria.access_levels,
                    "field_patterns": subscription.filter_criteria.field_patterns,
                    "metadata_filters": subscription.filter_criteria.metadata_filters,
                    "min_priority": subscription.filter_criteria.min_priority,
                    "exclude_self": subscription.filter_criteria.exclude_self
                },
                "preferences": {
                    "delivery_method": subscription.preferences.delivery_method.value,
                    "batch_size": subscription.preferences.batch_size,
                    "batch_timeout": subscription.preferences.batch_timeout,
                    "max_queue_size": subscription.preferences.max_queue_size,
                    "enable_compression": subscription.preferences.enable_compression,
                    "compression_threshold": subscription.preferences.compression_threshold,
                    "retry_attempts": subscription.preferences.retry_attempts,
                    "retry_delay": subscription.preferences.retry_delay,
                    "enable_deduplication": subscription.preferences.enable_deduplication,
                    "deduplication_window": subscription.preferences.deduplication_window
                },
                "created_at": subscription.created_at.isoformat(),
                "active": subscription.active,
                "metadata": subscription.metadata
            }
            
            await self.redis_backend._redis.setex(
                key,
                86400,  # 24 hour TTL
                json.dumps(subscription_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to store subscription in Redis: {e}")
    
    async def _remove_subscription_from_redis(self, subscription_id: str):
        """Remove subscription from Redis"""
        try:
            key = f"mcp:subscription:{subscription_id}"
            await self.redis_backend._redis.delete(key)
            
        except Exception as e:
            logger.error(f"Failed to remove subscription from Redis: {e}")
    
    async def _store_notification_in_redis(self, notification: ContextNotification):
        """Store notification in Redis for persistence"""
        try:
            key = f"mcp:notification:{notification.notification_id}"
            notification_data = {
                "notification_id": notification.notification_id,
                "subscription_id": notification.subscription_id,
                "agent_id": notification.agent_id,
                "context": notification.context.dict(),
                "notification_type": notification.notification_type,
                "priority": notification.priority.value,
                "created_at": notification.created_at.isoformat(),
                "delivered": notification.delivered,
                "delivery_attempts": notification.delivery_attempts,
                "metadata": notification.metadata
            }
            
            await self.redis_backend._redis.setex(
                key,
                self.notification_ttl,
                json.dumps(notification_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to store notification in Redis: {e}")
    
    async def _store_access_rules_in_redis(self, agent_id: str, rules: Dict[str, Any]):
        """Store access control rules in Redis"""
        try:
            key = f"mcp:access_rules:{agent_id}"
            await self.redis_backend._redis.setex(
                key,
                86400,  # 24 hour TTL
                json.dumps(rules)
            )
            
        except Exception as e:
            logger.error(f"Failed to store access rules in Redis: {e}")
    
    async def _setup_redis_subscription(self, subscription: ContextSubscription):
        """Set up Redis subscription for push notifications"""
        try:
            # This would set up Redis pub/sub channels for the subscription
            # Implementation depends on specific Redis pub/sub patterns
            pass
            
        except Exception as e:
            logger.error(f"Failed to setup Redis subscription: {e}")
    
    async def _cleanup_redis_subscription(self, subscription: ContextSubscription):
        """Clean up Redis subscription"""
        try:
            # Clean up Redis pub/sub channels for the subscription
            pass
            
        except Exception as e:
            logger.error(f"Failed to cleanup Redis subscription: {e}")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Batch processing task
        batch_task = asyncio.create_task(self._batch_processing_worker())
        self._background_tasks.add(batch_task)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_worker())
        self._background_tasks.add(cleanup_task)
        
        # Retry failed deliveries task
        retry_task = asyncio.create_task(self._retry_failed_deliveries_worker())
        self._background_tasks.add(retry_task)
        
        # Deduplication cache cleanup task
        dedup_cleanup_task = asyncio.create_task(self._deduplication_cleanup_worker())
        self._background_tasks.add(dedup_cleanup_task)
        
        logger.info("Started notification system background tasks")
    
    async def _batch_processing_worker(self):
        """Process notification batches"""
        while self._running:
            try:
                # Process batches for agents with batch delivery method
                for agent_id, subscriptions in self._agent_subscriptions.items():
                    for subscription_id in subscriptions:
                        subscription = self._subscriptions.get(subscription_id)
                        if (subscription and 
                            subscription.preferences.delivery_method == NotificationDeliveryMethod.BATCH):
                            
                            # Check if batch should be created
                            queue = self._notification_queues.get(agent_id, [])
                            if len(queue) >= subscription.preferences.batch_size:
                                batch = await self.create_notification_batch(agent_id)
                                if batch:
                                    logger.debug(f"Created batch {batch.batch_id} for agent {agent_id}")
                
                await asyncio.sleep(self.batch_processing_interval)
                
            except Exception as e:
                logger.error(f"Error in batch processing worker: {e}")
                await asyncio.sleep(self.batch_processing_interval)
    
    async def _cleanup_worker(self):
        """Clean up expired notifications and subscriptions"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                
                # Clean up expired notifications
                expired_notifications = []
                for notification_id, notification in self._pending_deliveries.items():
                    age = (current_time - notification.created_at).total_seconds()
                    if age > self.notification_ttl:
                        expired_notifications.append(notification_id)
                
                for notification_id in expired_notifications:
                    del self._pending_deliveries[notification_id]
                
                # Clean up expired batches
                expired_batches = []
                for batch_id, batch in self._notification_batches.items():
                    age = (current_time - batch.created_at).total_seconds()
                    if age > self.notification_ttl:
                        expired_batches.append(batch_id)
                
                for batch_id in expired_batches:
                    del self._notification_batches[batch_id]
                
                # Clean up permission cache
                self._permission_cache.clear()
                
                if expired_notifications or expired_batches:
                    logger.debug(f"Cleaned up {len(expired_notifications)} notifications and {len(expired_batches)} batches")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(self.cleanup_interval)
    
    async def _retry_failed_deliveries_worker(self):
        """Retry failed notification deliveries"""
        while self._running:
            try:
                # Find failed notifications that need retry
                for notification in list(self._pending_deliveries.values()):
                    if (not notification.delivered and 
                        notification.delivery_attempts > 0 and
                        notification.delivery_attempts < 3):  # Max retry attempts
                        
                        # Check if enough time has passed for retry
                        if notification.last_delivery_attempt:
                            time_since_last = (datetime.utcnow() - notification.last_delivery_attempt).total_seconds()
                            if time_since_last > 60:  # Retry after 1 minute
                                await self._deliver_push_notification(notification)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in retry failed deliveries worker: {e}")
                await asyncio.sleep(60)
    
    async def _deduplication_cleanup_worker(self):
        """Clean up deduplication cache"""
        while self._running:
            try:
                # Clear deduplication cache periodically to prevent memory growth
                self._deduplication_cache.clear()
                logger.debug("Cleared deduplication cache")
                
                await asyncio.sleep(1800)  # Clear every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in deduplication cleanup worker: {e}")
                await asyncio.sleep(1800)
    
    # Public API methods for management and monitoring
    
    async def get_agent_subscriptions(self, agent_id: str) -> List[ContextSubscription]:
        """Get all subscriptions for an agent"""
        try:
            subscription_ids = self._agent_subscriptions.get(agent_id, set())
            return [self._subscriptions[sid] for sid in subscription_ids if sid in self._subscriptions]
        except Exception as e:
            logger.error(f"Failed to get agent subscriptions: {e}")
            return []
    
    async def update_subscription_preferences(self, subscription_id: str, preferences: NotificationPreferences) -> bool:
        """Update notification preferences for a subscription"""
        try:
            if subscription_id not in self._subscriptions:
                return False
            
            self._subscriptions[subscription_id].preferences = preferences
            
            # Update in Redis
            await self._store_subscription_in_redis(self._subscriptions[subscription_id])
            
            return True
        except Exception as e:
            logger.error(f"Failed to update subscription preferences: {e}")
            return False
    
    async def pause_subscription(self, subscription_id: str) -> bool:
        """Pause a subscription"""
        try:
            if subscription_id not in self._subscriptions:
                return False
            
            self._subscriptions[subscription_id].active = False
            await self._store_subscription_in_redis(self._subscriptions[subscription_id])
            
            return True
        except Exception as e:
            logger.error(f"Failed to pause subscription: {e}")
            return False
    
    async def resume_subscription(self, subscription_id: str) -> bool:
        """Resume a paused subscription"""
        try:
            if subscription_id not in self._subscriptions:
                return False
            
            self._subscriptions[subscription_id].active = True
            await self._store_subscription_in_redis(self._subscriptions[subscription_id])
            
            return True
        except Exception as e:
            logger.error(f"Failed to resume subscription: {e}")
            return False
    
    async def clear_notification_queue(self, agent_id: str) -> int:
        """Clear notification queue for an agent"""
        try:
            if agent_id not in self._notification_queues:
                return 0
            
            count = len(self._notification_queues[agent_id])
            self._notification_queues[agent_id].clear()
            
            logger.info(f"Cleared {count} notifications from queue for agent {agent_id}")
            return count
        except Exception as e:
            logger.error(f"Failed to clear notification queue: {e}")
            return 0


# Global context notification system instance
context_notification_system = ContextNotificationSystem()