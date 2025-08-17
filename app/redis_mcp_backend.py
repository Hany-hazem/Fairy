# app/redis_mcp_backend.py
"""
Redis MCP Backend

This module provides Redis-based message queue and persistence for MCP communication
with connection pooling, failover, and comprehensive error handling.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError, RedisError, ResponseError

from .config import settings
from .mcp_models import MCPMessage

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Redis connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class RedisConfig:
    """Redis configuration settings"""
    url: str
    max_connections: int = 20
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
    socket_connect_timeout: int = 5
    socket_timeout: int = 5
    connection_timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    max_retry_delay: float = 60.0
    # Enhanced security and authentication
    password: Optional[str] = None
    username: Optional[str] = None
    ssl_enabled: bool = False
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    # Enhanced connection management
    health_check_interval: int = 30
    connection_pool_class_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MessageQueueConfig:
    """Message queue configuration"""
    default_ttl: int = 3600  # 1 hour
    max_queue_size: int = 10000
    cleanup_interval: int = 300  # 5 minutes
    batch_size: int = 100
    enable_persistence: bool = True
    compression_threshold: int = 1024  # bytes
    # Enhanced queue management
    priority_queue_enabled: bool = True
    dead_letter_queue_enabled: bool = True
    max_retry_attempts: int = 3
    retry_backoff_multiplier: float = 2.0
    queue_memory_threshold: float = 0.8  # 80% of Redis memory
    auto_scaling_enabled: bool = True
    message_deduplication: bool = True


class RedisMCPBackend:
    """
    Redis-based MCP backend with comprehensive features:
    - Connection pooling and failover
    - Message persistence and replay
    - Pub/sub for real-time messaging
    - Queue management and cleanup
    - Performance monitoring
    """
    
    def __init__(self, config: RedisConfig = None, queue_config: MessageQueueConfig = None):
        self.config = config or RedisConfig(url=settings.REDIS_URL)
        self.queue_config = queue_config or MessageQueueConfig()
        
        # Connection management
        self._connection_pool: Optional[ConnectionPool] = None
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._connection_state = ConnectionState.DISCONNECTED
        self._last_connection_attempt = None
        self._connection_failures = 0
        
        # Subscriptions and handlers
        self._subscriptions: Dict[str, Set[Callable]] = {}
        self._subscription_tasks: Dict[str, asyncio.Task] = {}
        self._message_handlers: Dict[str, Callable] = {}
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        # Statistics
        self.stats = {
            "messages_published": 0,
            "messages_received": 0,
            "connection_failures": 0,
            "reconnection_attempts": 0,
            "queue_operations": 0,
            "cleanup_operations": 0,
            "bytes_transferred": 0
        }
        
        # Performance tracking
        self._start_time = datetime.utcnow()
        
        logger.info("Redis MCP Backend initialized")
    
    async def connect(self, redis_url: str = None) -> bool:
        """
        Connect to Redis with connection pooling and error handling
        
        Args:
            redis_url: Optional Redis URL override
            
        Returns:
            True if connection successful
        """
        try:
            if self._connection_state == ConnectionState.CONNECTED:
                return True
            
            self._connection_state = ConnectionState.CONNECTING
            url = redis_url or self.config.url
            
            # Create connection pool with enhanced configuration
            pool_kwargs = {
                "max_connections": self.config.max_connections,
                "retry_on_timeout": self.config.retry_on_timeout,
                "socket_keepalive": self.config.socket_keepalive,
                "socket_connect_timeout": self.config.socket_connect_timeout,
                "socket_timeout": self.config.socket_timeout
            }
            
            # Add authentication if provided
            if self.config.password:
                pool_kwargs["password"] = self.config.password
            if self.config.username:
                pool_kwargs["username"] = self.config.username
            
            # Add SSL configuration if enabled
            if self.config.ssl_enabled:
                pool_kwargs["ssl"] = True
                pool_kwargs["ssl_cert_reqs"] = self.config.ssl_cert_reqs
                if self.config.ssl_ca_certs:
                    pool_kwargs["ssl_ca_certs"] = self.config.ssl_ca_certs
                if self.config.ssl_certfile:
                    pool_kwargs["ssl_certfile"] = self.config.ssl_certfile
                if self.config.ssl_keyfile:
                    pool_kwargs["ssl_keyfile"] = self.config.ssl_keyfile
            
            # Add any additional connection pool kwargs
            if self.config.connection_pool_class_kwargs:
                pool_kwargs.update(self.config.connection_pool_class_kwargs)
            
            self._connection_pool = ConnectionPool.from_url(url, **pool_kwargs)
            
            # Create Redis client
            self._redis = redis.Redis(
                connection_pool=self._connection_pool,
                decode_responses=True
            )
            
            # Test connection
            await asyncio.wait_for(
                self._redis.ping(),
                timeout=self.config.connection_timeout
            )
            
            # Initialize pub/sub
            self._pubsub = self._redis.pubsub()
            
            # Update state
            self._connection_state = ConnectionState.CONNECTED
            self._connection_failures = 0
            self._last_connection_attempt = datetime.utcnow()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info(f"Connected to Redis at {url}")
            return True
            
        except Exception as e:
            self._connection_state = ConnectionState.FAILED
            self._connection_failures += 1
            self.stats["connection_failures"] += 1
            
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Redis and cleanup resources"""
        try:
            self._running = False
            self._connection_state = ConnectionState.DISCONNECTED
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Cancel subscription tasks
            for task in self._subscription_tasks.values():
                task.cancel()
            
            # Close pub/sub
            if self._pubsub:
                await self._pubsub.close()
            
            # Close Redis connection
            if self._redis:
                await self._redis.close()
            
            # Disconnect connection pool
            if self._connection_pool:
                await self._connection_pool.disconnect()
            
            logger.info("Disconnected from Redis")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
    
    async def publish_message(self, topic: str, message: MCPMessage, retry_on_failure: bool = True) -> bool:
        """
        Publish message to Redis topic with enhanced reliability
        
        Args:
            topic: Topic to publish to
            message: MCP message to publish
            retry_on_failure: Whether to retry on failure
            
        Returns:
            True if message was published successfully
        """
        try:
            if not await self._ensure_connected():
                if retry_on_failure:
                    # Store message for later delivery
                    await self._store_failed_message(topic, message)
                return False
            
            # Serialize message
            message_data = json.dumps(message.to_dict())
            
            # Publish to topic
            result = await self._redis.publish(topic, message_data)
            
            # Store message for persistence if enabled
            if self.queue_config.enable_persistence:
                await self._store_message_for_replay(topic, message)
            
            # Update statistics
            self.stats["messages_published"] += 1
            self.stats["bytes_transferred"] += len(message_data)
            
            logger.debug(f"Published message {message.id} to topic {topic}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to publish message to topic {topic}: {e}")
            
            if retry_on_failure:
                # Store message for retry
                await self._store_failed_message(topic, message)
            
            return False
    
    async def publish_batch_messages(self, topic: str, messages: List[MCPMessage]) -> Dict[str, bool]:
        """
        Publish multiple messages in batch for better performance
        
        Args:
            topic: Topic to publish to
            messages: List of MCP messages to publish
            
        Returns:
            Dictionary mapping message IDs to success status
        """
        results = {}
        
        try:
            if not await self._ensure_connected():
                return {msg.id: False for msg in messages}
            
            # Use Redis pipeline for batch operations
            pipe = self._redis.pipeline()
            
            for message in messages:
                message_data = json.dumps(message.to_dict())
                pipe.publish(topic, message_data)
                
                # Store for persistence if enabled
                if self.queue_config.enable_persistence:
                    await self._store_message_for_replay(topic, message)
            
            # Execute batch
            batch_results = await pipe.execute()
            
            # Process results
            for i, (message, result) in enumerate(zip(messages, batch_results)):
                success = result > 0
                results[message.id] = success
                
                if success:
                    self.stats["messages_published"] += 1
                    self.stats["bytes_transferred"] += len(json.dumps(message.to_dict()))
            
            logger.debug(f"Published batch of {len(messages)} messages to topic {topic}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to publish batch messages to topic {topic}: {e}")
            return {msg.id: False for msg in messages}
    
    async def subscribe_to_topic(self, topic: str, callback: Callable, message_filter: Optional[Dict[str, Any]] = None) -> str:
        """
        Subscribe to Redis topic with callback and optional message filtering
        
        Args:
            topic: Topic to subscribe to (supports wildcards: * and ?)
            callback: Callback function for received messages
            message_filter: Optional filter criteria for messages
            
        Returns:
            Subscription ID
        """
        try:
            if not await self._ensure_connected():
                raise ConnectionError("Not connected to Redis")
            
            # Add callback to subscriptions with filter
            if topic not in self._subscriptions:
                self._subscriptions[topic] = set()
            
            # Store callback with filter information
            callback_info = {
                'callback': callback,
                'filter': message_filter,
                'id': id(callback)
            }
            self._subscriptions[topic].add(callback)
            
            # Store filter information separately
            if not hasattr(self, '_subscription_filters'):
                self._subscription_filters = {}
            self._subscription_filters[f"{topic}:{id(callback)}"] = message_filter
            
            # Start subscription task if not already running
            if topic not in self._subscription_tasks:
                # Check if topic contains wildcards
                if '*' in topic or '?' in topic:
                    task = asyncio.create_task(self._pattern_subscription_worker(topic))
                else:
                    task = asyncio.create_task(self._subscription_worker(topic))
                self._subscription_tasks[topic] = task
            
            # Generate subscription ID
            subscription_id = f"{topic}:{id(callback)}"
            
            logger.info(f"Subscribed to topic {topic} with callback {subscription_id}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {e}")
            raise
    
    async def subscribe_to_multiple_topics(self, topics: List[str], callback: Callable, message_filter: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Subscribe to multiple topics with a single callback
        
        Args:
            topics: List of topics to subscribe to
            callback: Callback function for received messages
            message_filter: Optional filter criteria for messages
            
        Returns:
            List of subscription IDs
        """
        subscription_ids = []
        
        for topic in topics:
            try:
                subscription_id = await self.subscribe_to_topic(topic, callback, message_filter)
                subscription_ids.append(subscription_id)
            except Exception as e:
                logger.error(f"Failed to subscribe to topic {topic}: {e}")
        
        return subscription_ids
    
    async def unsubscribe_from_topic(self, subscription_id: str) -> bool:
        """
        Unsubscribe from Redis topic
        
        Args:
            subscription_id: Subscription ID to remove
            
        Returns:
            True if unsubscribed successfully
        """
        try:
            # Parse subscription ID
            parts = subscription_id.split(":", 1)
            if len(parts) != 2:
                logger.error(f"Invalid subscription ID format: {subscription_id}")
                return False
            
            topic, callback_id = parts
            
            # Remove filter information
            if hasattr(self, '_subscription_filters'):
                self._subscription_filters.pop(subscription_id, None)
            
            # Find and remove callback
            if topic in self._subscriptions:
                callbacks_to_remove = [
                    cb for cb in self._subscriptions[topic]
                    if str(id(cb)) == callback_id
                ]
                
                for callback in callbacks_to_remove:
                    self._subscriptions[topic].discard(callback)
                
                # Clean up empty subscription sets
                if not self._subscriptions[topic]:
                    del self._subscriptions[topic]
                    
                    # Cancel subscription task
                    if topic in self._subscription_tasks:
                        self._subscription_tasks[topic].cancel()
                        del self._subscription_tasks[topic]
                    
                    # Unsubscribe from Redis (handle both regular and pattern subscriptions)
                    if self._pubsub:
                        if '*' in topic or '?' in topic:
                            await self._pubsub.punsubscribe(topic)
                        else:
                            await self._pubsub.unsubscribe(topic)
            
            logger.info(f"Unsubscribed from {subscription_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {subscription_id}: {e}")
            return False
    
    async def unsubscribe_all(self) -> bool:
        """
        Unsubscribe from all topics
        
        Returns:
            True if all unsubscriptions were successful
        """
        try:
            success = True
            
            # Get all subscription IDs
            subscription_ids = []
            for topic, callbacks in self._subscriptions.items():
                for callback in callbacks:
                    subscription_ids.append(f"{topic}:{id(callback)}")
            
            # Unsubscribe from each
            for subscription_id in subscription_ids:
                if not await self.unsubscribe_from_topic(subscription_id):
                    success = False
            
            # Clear all subscription data
            self._subscriptions.clear()
            if hasattr(self, '_subscription_filters'):
                self._subscription_filters.clear()
            
            # Cancel all subscription tasks
            for task in self._subscription_tasks.values():
                task.cancel()
            self._subscription_tasks.clear()
            
            logger.info("Unsubscribed from all topics")
            return success
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from all topics: {e}")
            return False
    
    async def get_message_history(self, topic: str, limit: int = 100, priority_order: bool = False) -> List[MCPMessage]:
        """
        Get message history for a topic
        
        Args:
            topic: Topic to get history for
            limit: Maximum number of messages to return
            priority_order: If True, return messages in priority order
            
        Returns:
            List of historical messages
        """
        try:
            if not await self._ensure_connected():
                return []
            
            messages = []
            
            if priority_order and self.queue_config.priority_queue_enabled:
                # Get messages from priority queue (highest priority first)
                priority_key = f"mcp:priority:{topic}"
                message_data_list = await self._redis.zrevrange(priority_key, 0, limit - 1)
            else:
                # Get messages from history queue (chronological order)
                history_key = f"mcp:history:{topic}"
                message_data_list = await self._redis.lrange(history_key, 0, limit - 1)
            
            for message_data in message_data_list:
                try:
                    message_dict = json.loads(message_data)
                    message = MCPMessage.from_dict(message_dict)
                    messages.append(message)
                except Exception as e:
                    logger.warning(f"Failed to parse historical message: {e}")
            
            logger.debug(f"Retrieved {len(messages)} historical messages for topic {topic}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get message history for topic {topic}: {e}")
            return []
    
    async def get_priority_messages(self, topic: str, min_priority: int = 1, limit: int = 100) -> List[MCPMessage]:
        """
        Get messages from priority queue with minimum priority
        
        Args:
            topic: Topic to get messages for
            min_priority: Minimum priority level
            limit: Maximum number of messages to return
            
        Returns:
            List of priority messages
        """
        try:
            if not await self._ensure_connected():
                return []
            
            if not self.queue_config.priority_queue_enabled:
                logger.warning("Priority queue is not enabled")
                return []
            
            priority_key = f"mcp:priority:{topic}"
            
            # Get messages with priority >= min_priority
            # Note: priorities are stored as negative values for correct sorting
            message_data_list = await self._redis.zrevrangebyscore(
                priority_key, 
                -min_priority,  # max score (higher priority)
                "-inf",         # min score (any priority >= min_priority)
                start=0,
                num=limit
            )
            
            messages = []
            for message_data in message_data_list:
                try:
                    message_dict = json.loads(message_data)
                    message = MCPMessage.from_dict(message_dict)
                    messages.append(message)
                except Exception as e:
                    logger.warning(f"Failed to parse priority message: {e}")
            
            logger.debug(f"Retrieved {len(messages)} priority messages for topic {topic}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get priority messages for topic {topic}: {e}")
            return []
    
    async def get_queue_size(self, topic: str) -> Dict[str, int]:
        """
        Get queue sizes for a topic
        
        Args:
            topic: Topic to check
            
        Returns:
            Dictionary with queue sizes
        """
        try:
            if not await self._ensure_connected():
                return {}
            
            sizes = {}
            
            # Get history queue size
            history_key = f"mcp:history:{topic}"
            sizes["history"] = await self._redis.llen(history_key)
            
            # Get priority queue size if enabled
            if self.queue_config.priority_queue_enabled:
                priority_key = f"mcp:priority:{topic}"
                sizes["priority"] = await self._redis.zcard(priority_key)
            
            return sizes
            
        except Exception as e:
            logger.error(f"Failed to get queue size for topic {topic}: {e}")
            return {}
    
    async def cleanup_expired_messages(self) -> int:
        """
        Clean up expired messages from all queues with enhanced policies
        
        Returns:
            Number of messages cleaned up
        """
        try:
            if not await self._ensure_connected():
                return 0
            
            cleaned_count = 0
            current_time = datetime.utcnow()
            
            # Get all MCP-related keys
            pattern = "mcp:*"
            keys = await self._redis.keys(pattern)
            
            for key in keys:
                try:
                    # Check if key is a list (queue)
                    key_type = await self._redis.type(key)
                    
                    if key_type == "list":
                        # Clean expired messages from list
                        cleaned = await self._cleanup_expired_from_list(key, current_time)
                        cleaned_count += cleaned
                    
                    elif key_type == "zset":
                        # Clean expired messages from sorted set (priority queues)
                        cleaned = await self._cleanup_expired_from_zset(key, current_time)
                        cleaned_count += cleaned
                    
                    elif key_type == "hash":
                        # Clean expired metadata entries
                        cleaned = await self._cleanup_expired_from_hash(key, current_time)
                        cleaned_count += cleaned
                    
                    elif key_type == "string":
                        # Check TTL and remove if expired
                        ttl = await self._redis.ttl(key)
                        if ttl == -1:  # No TTL set
                            await self._redis.expire(key, self.queue_config.default_ttl)
                
                except Exception as e:
                    logger.warning(f"Error cleaning up key {key}: {e}")
            
            # Clean up failed messages that have exceeded retry limits
            await self._cleanup_failed_messages()
            
            # Update statistics
            self.stats["cleanup_operations"] += 1
            
            logger.debug(f"Cleaned up {cleaned_count} expired messages")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired messages: {e}")
            return 0
    
    async def _cleanup_expired_from_zset(self, key: str, current_time: datetime) -> int:
        """Clean up expired messages from a Redis sorted set (priority queue)"""
        try:
            cleaned_count = 0
            
            # Get all messages from sorted set
            messages_with_scores = await self._redis.zrange(key, 0, -1, withscores=True)
            
            for message_data, score in messages_with_scores:
                try:
                    message_dict = json.loads(message_data)
                    message = MCPMessage.from_dict(message_dict)
                    
                    if message.is_expired():
                        # Remove expired message
                        await self._redis.zrem(key, message_data)
                        cleaned_count += 1
                
                except Exception as e:
                    logger.warning(f"Error checking message expiration in zset: {e}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning expired messages from zset {key}: {e}")
            return 0
    
    async def _cleanup_expired_from_hash(self, key: str, current_time: datetime) -> int:
        """Clean up expired metadata from a Redis hash"""
        try:
            cleaned_count = 0
            
            # Get all hash fields
            hash_data = await self._redis.hgetall(key)
            
            for field, value in hash_data.items():
                try:
                    metadata = json.loads(value)
                    timestamp_str = metadata.get("timestamp")
                    
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        age = (current_time - timestamp).total_seconds()
                        
                        if age > self.queue_config.default_ttl:
                            await self._redis.hdel(key, field)
                            cleaned_count += 1
                
                except Exception as e:
                    logger.warning(f"Error checking metadata expiration: {e}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning expired metadata from hash {key}: {e}")
            return 0
    
    async def _cleanup_failed_messages(self):
        """Clean up failed messages that have exceeded retry limits"""
        try:
            failed_keys = await self._redis.keys("mcp:failed:*")
            
            for failed_key in failed_keys:
                failed_messages = await self._redis.lrange(failed_key, 0, -1)
                
                for failed_data in failed_messages:
                    try:
                        failed_info = json.loads(failed_data)
                        retry_count = failed_info.get("retry_count", 0)
                        failed_at_str = failed_info.get("failed_at")
                        
                        # Remove messages that have exceeded retry limits
                        if retry_count >= self.queue_config.max_retry_attempts:
                            await self._redis.lrem(failed_key, 1, failed_data)
                            continue
                        
                        # Remove very old failed messages
                        if failed_at_str:
                            failed_at = datetime.fromisoformat(failed_at_str)
                            age = (datetime.utcnow() - failed_at).total_seconds()
                            
                            if age > self.queue_config.default_ttl * 3:  # 3x normal TTL
                                await self._redis.lrem(failed_key, 1, failed_data)
                    
                    except Exception as e:
                        logger.warning(f"Error cleaning failed message: {e}")
        
        except Exception as e:
            logger.error(f"Error cleaning up failed messages: {e}")
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive Redis queue statistics"""
        try:
            if not await self._ensure_connected():
                return {}
            
            # Get Redis info
            redis_info = await self._redis.info()
            
            # Get MCP-specific stats
            mcp_keys = await self._redis.keys("mcp:*")
            
            # Analyze queue sizes and types
            queue_analysis = await self._analyze_queue_health()
            
            # Get memory usage analysis
            memory_analysis = await self._analyze_memory_usage(redis_info)
            
            queue_stats = {
                "redis_info": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory": redis_info.get("used_memory", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "total_commands_processed": redis_info.get("total_commands_processed", 0),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                    "instantaneous_ops_per_sec": redis_info.get("instantaneous_ops_per_sec", 0),
                    "role": redis_info.get("role", "unknown")
                },
                "mcp_stats": {
                    "total_keys": len(mcp_keys),
                    "connection_state": self._connection_state.value,
                    "active_subscriptions": len(self._subscriptions),
                    "background_tasks": len(self._background_tasks),
                    "queue_health": queue_analysis,
                    "memory_analysis": memory_analysis
                },
                "backend_stats": self.stats.copy(),
                "performance_metrics": {
                    "avg_message_size": self._calculate_avg_message_size(),
                    "messages_per_second": self._calculate_message_rate(),
                    "connection_uptime": self._calculate_connection_uptime()
                }
            }
            
            return queue_stats
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {"error": str(e)}
    
    async def _ensure_connected(self) -> bool:
        """Ensure Redis connection is active, reconnect if needed"""
        if self._connection_state == ConnectionState.CONNECTED:
            try:
                await self._redis.ping()
                return True
            except Exception:
                self._connection_state = ConnectionState.DISCONNECTED
        
        if self._connection_state in [ConnectionState.DISCONNECTED, ConnectionState.FAILED]:
            return await self._reconnect()
        
        return False
    
    async def _reconnect(self) -> bool:
        """Attempt to reconnect to Redis with exponential backoff"""
        if self._connection_state == ConnectionState.RECONNECTING:
            return False
        
        self._connection_state = ConnectionState.RECONNECTING
        self.stats["reconnection_attempts"] += 1
        
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Reconnection attempt {attempt + 1}/{max_retries}")
                
                if await self.connect():
                    logger.info("Successfully reconnected to Redis")
                    return True
                
                # Exponential backoff
                if self.config.exponential_backoff:
                    retry_delay = min(retry_delay * 2, self.config.max_retry_delay)
                
                await asyncio.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        self._connection_state = ConnectionState.FAILED
        logger.error("Failed to reconnect to Redis after all attempts")
        return False
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Message cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_worker())
        self._background_tasks.add(cleanup_task)
        
        # Connection health check task
        health_task = asyncio.create_task(self._health_check_worker())
        self._background_tasks.add(health_task)
        
        logger.info("Started Redis backend background tasks")
    
    async def _cleanup_worker(self):
        """Background worker for message cleanup"""
        while self._running:
            try:
                await self.cleanup_expired_messages()
                await asyncio.sleep(self.queue_config.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(self.queue_config.cleanup_interval)
    
    async def _health_check_worker(self):
        """Background worker for connection health checks"""
        while self._running:
            try:
                if self._connection_state == ConnectionState.CONNECTED:
                    await self._redis.ping()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                self._connection_state = ConnectionState.DISCONNECTED
                await asyncio.sleep(5)
    
    async def _subscription_worker(self, topic: str):
        """Worker for handling topic subscriptions"""
        try:
            # Subscribe to topic
            await self._pubsub.subscribe(topic)
            
            logger.info(f"Started subscription worker for topic: {topic}")
            
            async for message in self._pubsub.listen():
                if not self._running:
                    break
                
                if message["type"] == "message":
                    try:
                        # Parse message
                        message_data = json.loads(message["data"])
                        mcp_message = MCPMessage.from_dict(message_data)
                        
                        # Call all callbacks for this topic with filtering
                        if topic in self._subscriptions:
                            for callback in self._subscriptions[topic].copy():
                                try:
                                    # Check message filter
                                    subscription_id = f"{topic}:{id(callback)}"
                                    message_filter = getattr(self, '_subscription_filters', {}).get(subscription_id)
                                    
                                    if message_filter and not self._message_matches_filter(mcp_message, message_filter):
                                        continue
                                    
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(mcp_message)
                                    else:
                                        callback(mcp_message)
                                except Exception as e:
                                    logger.error(f"Error in subscription callback: {e}")
                        
                        # Update statistics
                        self.stats["messages_received"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing subscription message: {e}")
        
        except Exception as e:
            logger.error(f"Error in subscription worker for topic {topic}: {e}")
        
        finally:
            # Cleanup
            try:
                await self._pubsub.unsubscribe(topic)
            except Exception:
                pass
    
    async def _pattern_subscription_worker(self, pattern: str):
        """Worker for handling pattern-based subscriptions"""
        try:
            # Subscribe to pattern
            await self._pubsub.psubscribe(pattern)
            
            logger.info(f"Started pattern subscription worker for pattern: {pattern}")
            
            async for message in self._pubsub.listen():
                if not self._running:
                    break
                
                if message["type"] == "pmessage":
                    try:
                        # Parse message
                        message_data = json.loads(message["data"])
                        mcp_message = MCPMessage.from_dict(message_data)
                        actual_topic = message["channel"]
                        
                        # Call all callbacks for this pattern with filtering
                        if pattern in self._subscriptions:
                            for callback in self._subscriptions[pattern].copy():
                                try:
                                    # Check message filter
                                    subscription_id = f"{pattern}:{id(callback)}"
                                    message_filter = getattr(self, '_subscription_filters', {}).get(subscription_id)
                                    
                                    if message_filter and not self._message_matches_filter(mcp_message, message_filter):
                                        continue
                                    
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(mcp_message)
                                    else:
                                        callback(mcp_message)
                                except Exception as e:
                                    logger.error(f"Error in pattern subscription callback: {e}")
                        
                        # Update statistics
                        self.stats["messages_received"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing pattern subscription message: {e}")
        
        except Exception as e:
            logger.error(f"Error in pattern subscription worker for pattern {pattern}: {e}")
        
        finally:
            # Cleanup
            try:
                await self._pubsub.punsubscribe(pattern)
            except Exception:
                pass
    
    def _message_matches_filter(self, message: MCPMessage, message_filter: Dict[str, Any]) -> bool:
        """Check if message matches the given filter criteria"""
        try:
            for key, expected_value in message_filter.items():
                # Handle nested keys (e.g., "payload.task_id")
                if '.' in key:
                    keys = key.split('.')
                    actual_value = message.to_dict()
                    
                    for k in keys:
                        if isinstance(actual_value, dict) and k in actual_value:
                            actual_value = actual_value[k]
                        else:
                            return False
                else:
                    # Direct attribute access
                    if hasattr(message, key):
                        actual_value = getattr(message, key)
                    else:
                        return False
                
                # Compare values
                if isinstance(expected_value, dict):
                    # Handle special operators
                    if "$in" in expected_value:
                        if actual_value not in expected_value["$in"]:
                            return False
                    elif "$regex" in expected_value:
                        import re
                        if not re.match(expected_value["$regex"], str(actual_value)):
                            return False
                    elif "$gt" in expected_value:
                        if not (isinstance(actual_value, (int, float)) and actual_value > expected_value["$gt"]):
                            return False
                    elif "$lt" in expected_value:
                        if not (isinstance(actual_value, (int, float)) and actual_value < expected_value["$lt"]):
                            return False
                else:
                    # Direct comparison
                    if actual_value != expected_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching message filter: {e}")
            return False
    
    async def _store_message_for_replay(self, topic: str, message: MCPMessage):
        """Store message for replay capability with enhanced persistence"""
        try:
            history_key = f"mcp:history:{topic}"
            message_data = json.dumps(message.to_dict())
            
            # Check for message deduplication if enabled
            if self.queue_config.message_deduplication:
                message_hash = f"mcp:hash:{message.id}"
                exists = await self._redis.exists(message_hash)
                if exists:
                    logger.debug(f"Duplicate message {message.id} detected, skipping storage")
                    return
                
                # Store message hash for deduplication
                await self._redis.setex(message_hash, self.queue_config.default_ttl, "1")
            
            # Store in priority queue if enabled
            if self.queue_config.priority_queue_enabled:
                priority_key = f"mcp:priority:{topic}"
                # Use negative priority for correct sorting (higher priority first)
                await self._redis.zadd(priority_key, {message_data: -message.priority})
                
                # Trim priority queue to max size
                await self._redis.zremrangebyrank(priority_key, 0, -(self.queue_config.max_queue_size + 1))
                
                # Set TTL on priority queue
                await self._redis.expire(priority_key, self.queue_config.default_ttl)
            
            # Add to history list (for chronological order)
            await self._redis.lpush(history_key, message_data)
            
            # Trim to max size
            await self._redis.ltrim(history_key, 0, self.queue_config.max_queue_size - 1)
            
            # Set TTL
            await self._redis.expire(history_key, self.queue_config.default_ttl)
            
            # Store message metadata for analytics
            metadata_key = f"mcp:metadata:{topic}"
            metadata = {
                "message_id": message.id,
                "timestamp": message.timestamp.isoformat(),
                "source_agent": message.source_agent,
                "message_type": message.type,
                "priority": message.priority,
                "size": len(message_data)
            }
            await self._redis.hset(metadata_key, message.id, json.dumps(metadata))
            await self._redis.expire(metadata_key, self.queue_config.default_ttl)
            
        except Exception as e:
            logger.error(f"Failed to store message for replay: {e}")
    
    async def _cleanup_expired_from_list(self, key: str, current_time: datetime) -> int:
        """Clean up expired messages from a Redis list"""
        try:
            cleaned_count = 0
            
            # Get all messages from list
            messages = await self._redis.lrange(key, 0, -1)
            
            # Check each message for expiration
            for i, message_data in enumerate(messages):
                try:
                    message_dict = json.loads(message_data)
                    message = MCPMessage.from_dict(message_dict)
                    
                    if message.is_expired():
                        # Remove expired message
                        await self._redis.lrem(key, 1, message_data)
                        cleaned_count += 1
                
                except Exception as e:
                    logger.warning(f"Error checking message expiration: {e}")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning expired messages from list {key}: {e}")
            return 0
    
    async def _analyze_queue_health(self) -> Dict[str, Any]:
        """Analyze queue health and performance"""
        try:
            analysis = {
                "total_queues": 0,
                "queue_sizes": {},
                "priority_queues": {},
                "health_status": "healthy",
                "warnings": []
            }
            
            # Get all MCP queue keys
            history_keys = await self._redis.keys("mcp:history:*")
            priority_keys = await self._redis.keys("mcp:priority:*")
            
            analysis["total_queues"] = len(history_keys)
            
            # Analyze history queues
            for key in history_keys:
                topic = key.replace("mcp:history:", "")
                size = await self._redis.llen(key)
                analysis["queue_sizes"][topic] = size
                
                # Check for oversized queues
                if size > self.queue_config.max_queue_size * 0.9:
                    analysis["warnings"].append(f"Queue {topic} is near capacity: {size}/{self.queue_config.max_queue_size}")
                    analysis["health_status"] = "warning"
            
            # Analyze priority queues
            for key in priority_keys:
                topic = key.replace("mcp:priority:", "")
                size = await self._redis.zcard(key)
                analysis["priority_queues"][topic] = size
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing queue health: {e}")
            return {"error": str(e)}
    
    async def _analyze_memory_usage(self, redis_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Redis memory usage"""
        try:
            used_memory = redis_info.get("used_memory", 0)
            max_memory = redis_info.get("maxmemory", 0)
            
            analysis = {
                "used_memory_bytes": used_memory,
                "max_memory_bytes": max_memory,
                "memory_usage_ratio": 0.0,
                "memory_status": "healthy",
                "recommendations": []
            }
            
            if max_memory > 0:
                analysis["memory_usage_ratio"] = used_memory / max_memory
                
                # Check memory thresholds
                if analysis["memory_usage_ratio"] > self.queue_config.queue_memory_threshold:
                    analysis["memory_status"] = "critical"
                    analysis["recommendations"].append("Consider increasing Redis memory or enabling message cleanup")
                elif analysis["memory_usage_ratio"] > 0.7:
                    analysis["memory_status"] = "warning"
                    analysis["recommendations"].append("Monitor memory usage closely")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing memory usage: {e}")
            return {"error": str(e)}
    
    def _calculate_avg_message_size(self) -> float:
        """Calculate average message size"""
        if self.stats["messages_published"] > 0:
            return self.stats["bytes_transferred"] / self.stats["messages_published"]
        return 0.0
    
    def _calculate_message_rate(self) -> float:
        """Calculate messages per second rate"""
        if hasattr(self, '_start_time'):
            uptime = (datetime.utcnow() - self._start_time).total_seconds()
            if uptime > 0:
                return self.stats["messages_published"] / uptime
        return 0.0
    
    def _calculate_connection_uptime(self) -> float:
        """Calculate connection uptime in seconds"""
        if self._last_connection_attempt:
            return (datetime.utcnow() - self._last_connection_attempt).total_seconds()
        return 0.0
    
    async def _store_failed_message(self, topic: str, message: MCPMessage):
        """Store failed message for retry"""
        try:
            if not self.queue_config.dead_letter_queue_enabled:
                return
            
            failed_key = f"mcp:failed:{topic}"
            message_data = json.dumps({
                "message": message.to_dict(),
                "failed_at": datetime.utcnow().isoformat(),
                "retry_count": 0
            })
            
            await self._redis.lpush(failed_key, message_data)
            await self._redis.expire(failed_key, self.queue_config.default_ttl * 2)  # Longer TTL for failed messages
            
            logger.debug(f"Stored failed message {message.id} for topic {topic}")
            
        except Exception as e:
            logger.error(f"Failed to store failed message: {e}")
    
    async def replay_failed_messages(self, topic: str = None) -> Dict[str, int]:
        """
        Replay failed messages
        
        Args:
            topic: Specific topic to replay, or None for all topics
            
        Returns:
            Dictionary with replay statistics
        """
        try:
            if not await self._ensure_connected():
                return {"error": "Not connected to Redis"}
            
            stats = {"replayed": 0, "failed": 0, "topics": 0}
            
            # Get failed message keys
            if topic:
                failed_keys = [f"mcp:failed:{topic}"]
            else:
                failed_keys = await self._redis.keys("mcp:failed:*")
            
            for failed_key in failed_keys:
                topic_name = failed_key.replace("mcp:failed:", "")
                stats["topics"] += 1
                
                # Get all failed messages for this topic
                failed_messages = await self._redis.lrange(failed_key, 0, -1)
                
                for failed_data in failed_messages:
                    try:
                        failed_info = json.loads(failed_data)
                        message_dict = failed_info["message"]
                        retry_count = failed_info.get("retry_count", 0)
                        
                        # Check retry limit
                        if retry_count >= self.queue_config.max_retry_attempts:
                            logger.warning(f"Message exceeded retry limit: {message_dict.get('id')}")
                            await self._redis.lrem(failed_key, 1, failed_data)
                            stats["failed"] += 1
                            continue
                        
                        # Recreate message
                        message = MCPMessage.from_dict(message_dict)
                        
                        # Try to republish
                        success = await self.publish_message(topic_name, message, retry_on_failure=False)
                        
                        if success:
                            # Remove from failed queue
                            await self._redis.lrem(failed_key, 1, failed_data)
                            stats["replayed"] += 1
                            logger.debug(f"Successfully replayed message {message.id}")
                        else:
                            # Update retry count
                            failed_info["retry_count"] = retry_count + 1
                            updated_data = json.dumps(failed_info)
                            
                            # Replace in queue
                            await self._redis.lrem(failed_key, 1, failed_data)
                            await self._redis.lpush(failed_key, updated_data)
                            
                            # Apply exponential backoff delay
                            delay = self.queue_config.retry_backoff_multiplier ** retry_count
                            await asyncio.sleep(min(delay, 60))  # Max 60 second delay
                    
                    except Exception as e:
                        logger.error(f"Error replaying failed message: {e}")
                        stats["failed"] += 1
            
            logger.info(f"Replay completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error during message replay: {e}")
            return {"error": str(e)}
    
    async def get_failed_message_count(self, topic: str = None) -> Dict[str, int]:
        """
        Get count of failed messages
        
        Args:
            topic: Specific topic to check, or None for all topics
            
        Returns:
            Dictionary with failed message counts per topic
        """
        try:
            if not await self._ensure_connected():
                return {}
            
            counts = {}
            
            if topic:
                failed_keys = [f"mcp:failed:{topic}"]
            else:
                failed_keys = await self._redis.keys("mcp:failed:*")
            
            for failed_key in failed_keys:
                topic_name = failed_key.replace("mcp:failed:", "")
                count = await self._redis.llen(failed_key)
                counts[topic_name] = count
            
            return counts
            
        except Exception as e:
            logger.error(f"Error getting failed message count: {e}")
            return {}
    
    async def clear_failed_messages(self, topic: str = None) -> int:
        """
        Clear failed messages
        
        Args:
            topic: Specific topic to clear, or None for all topics
            
        Returns:
            Number of messages cleared
        """
        try:
            if not await self._ensure_connected():
                return 0
            
            cleared = 0
            
            if topic:
                failed_keys = [f"mcp:failed:{topic}"]
            else:
                failed_keys = await self._redis.keys("mcp:failed:*")
            
            for failed_key in failed_keys:
                count = await self._redis.llen(failed_key)
                await self._redis.delete(failed_key)
                cleared += count
            
            logger.info(f"Cleared {cleared} failed messages")
            return cleared
            
        except Exception as e:
            logger.error(f"Error clearing failed messages: {e}")
            return 0


# Global Redis MCP backend instance
redis_mcp_backend = RedisMCPBackend()