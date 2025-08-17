# app/context_synchronizer.py
"""
Context Synchronization Engine

This module implements context sharing mechanisms for agent communication
with versioning, conflict detection, and resolution strategies.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .config import settings
from .mcp_models import AgentContext, MCPMessage, MCPMessageType, create_context_update_message
from .redis_mcp_backend import RedisMCPBackend

logger = logging.getLogger(__name__)


class ContextConflictType(Enum):
    """Types of context conflicts"""
    VERSION_CONFLICT = "version_conflict"
    DATA_CONFLICT = "data_conflict"
    ACCESS_CONFLICT = "access_conflict"
    TIMESTAMP_CONFLICT = "timestamp_conflict"


class ContextMergeStrategy(Enum):
    """Context merge strategies"""
    LATEST_WINS = "latest_wins"
    MERGE_RECURSIVE = "merge_recursive"
    MANUAL_RESOLUTION = "manual_resolution"
    SOURCE_PRIORITY = "source_priority"
    FIELD_LEVEL_MERGE = "field_level_merge"


@dataclass
class ContextConflict:
    """Context conflict information"""
    conflict_id: str
    conflict_type: ContextConflictType
    context_type: str
    agent_id: str
    conflicting_versions: List[str]
    local_context: AgentContext
    remote_context: AgentContext
    detected_at: datetime
    resolution_strategy: Optional[ContextMergeStrategy] = None
    resolved: bool = False
    resolution_data: Optional[Dict[str, Any]] = None


@dataclass
class ContextResolution:
    """Context conflict resolution result"""
    conflict_id: str
    resolved: bool
    merged_context: Optional[AgentContext]
    resolution_strategy: ContextMergeStrategy
    resolution_notes: str
    requires_manual_intervention: bool = False
    affected_agents: List[str] = field(default_factory=list)


@dataclass
class ContextBroadcastResult:
    """Result of context broadcast operation"""
    success: bool
    broadcast_id: str
    target_agents: List[str]
    successful_deliveries: List[str]
    failed_deliveries: List[str]
    conflicts_detected: List[ContextConflict]
    timestamp: datetime


class ContextSynchronizer:
    """
    Context Synchronization Engine for agent communication
    
    Provides comprehensive context sharing with:
    - Context broadcasting and versioning
    - Conflict detection and resolution
    - Merge strategies and manual intervention
    - Performance optimization and caching
    """
    
    def __init__(self, redis_backend: RedisMCPBackend = None):
        self.redis_backend = redis_backend or RedisMCPBackend()
        
        # Context storage and versioning
        self._local_contexts: Dict[str, AgentContext] = {}  # agent_id -> context
        self._context_versions: Dict[str, Dict[str, str]] = {}  # agent_id -> {context_type -> version}
        self._context_history: Dict[str, List[AgentContext]] = {}  # agent_id -> history
        
        # Conflict management
        self._active_conflicts: Dict[str, ContextConflict] = {}
        self._conflict_handlers: Dict[ContextConflictType, Callable] = {}
        self._merge_strategies: Dict[str, Callable] = {}
        
        # Subscription management
        self._context_subscriptions: Dict[str, Set[str]] = {}  # context_type -> agent_ids
        self._agent_subscriptions: Dict[str, Set[str]] = {}  # agent_id -> context_types
        self._subscription_callbacks: Dict[str, Callable] = {}  # subscription_id -> callback
        
        # Performance and caching
        self._context_cache: Dict[str, AgentContext] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Configuration
        self.max_history_size = 50
        self.conflict_timeout = 3600  # 1 hour
        self.broadcast_timeout = 30  # 30 seconds
        self.max_concurrent_broadcasts = 10
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        # Initialize default merge strategies
        self._initialize_merge_strategies()
        
        logger.info("Context Synchronizer initialized")
    
    async def start(self):
        """Start the context synchronizer"""
        try:
            if self._running:
                return
            
            # Connect to Redis backend
            if not await self.redis_backend.connect():
                raise RuntimeError("Failed to connect to Redis backend")
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._running = True
            logger.info("Context Synchronizer started")
            
        except Exception as e:
            logger.error(f"Failed to start Context Synchronizer: {e}")
            raise
    
    async def stop(self):
        """Stop the context synchronizer"""
        try:
            self._running = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Disconnect from Redis
            await self.redis_backend.disconnect()
            
            logger.info("Context Synchronizer stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Context Synchronizer: {e}")
    
    async def broadcast_context_update(self, context: AgentContext, targets: List[str] = None) -> ContextBroadcastResult:
        """
        Broadcast context update to relevant agents
        
        Args:
            context: Agent context to broadcast
            targets: Optional list of target agent IDs (if None, broadcasts to all subscribers)
            
        Returns:
            Broadcast result with delivery status and conflicts
        """
        try:
            broadcast_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            # Determine target agents
            if targets is None:
                targets = list(self._context_subscriptions.get(context.context_type, set()))
            
            # Remove the source agent from targets to avoid self-broadcast
            targets = [t for t in targets if t != context.agent_id]
            
            if not targets:
                logger.debug(f"No targets for context broadcast of type {context.context_type}")
                return ContextBroadcastResult(
                    success=True,
                    broadcast_id=broadcast_id,
                    target_agents=[],
                    successful_deliveries=[],
                    failed_deliveries=[],
                    conflicts_detected=[],
                    timestamp=timestamp
                )
            
            # Update local context storage
            await self._store_context_locally(context)
            
            # Create context update message
            message = create_context_update_message(
                source_agent=context.agent_id,
                target_agents=targets,
                context=context
            )
            
            # Broadcast to targets with conflict detection
            successful_deliveries = []
            failed_deliveries = []
            conflicts_detected = []
            
            # Use semaphore to limit concurrent broadcasts
            semaphore = asyncio.Semaphore(self.max_concurrent_broadcasts)
            
            async def broadcast_to_agent(target_agent: str):
                async with semaphore:
                    try:
                        # Check for conflicts before broadcasting
                        conflict = await self._detect_context_conflict(context, target_agent)
                        if conflict:
                            conflicts_detected.append(conflict)
                            # Still attempt delivery but mark conflict
                        
                        # Broadcast message
                        topic = f"mcp:agent:{target_agent}:context_updates"
                        success = await self.redis_backend.publish_message(topic, message)
                        
                        if success:
                            successful_deliveries.append(target_agent)
                            # Update delivery tracking
                            await self._track_context_delivery(broadcast_id, target_agent, context)
                        else:
                            failed_deliveries.append(target_agent)
                    
                    except Exception as e:
                        logger.error(f"Failed to broadcast context to agent {target_agent}: {e}")
                        failed_deliveries.append(target_agent)
            
            # Execute broadcasts concurrently
            await asyncio.gather(
                *[broadcast_to_agent(target) for target in targets],
                return_exceptions=True
            )
            
            # Store broadcast result
            result = ContextBroadcastResult(
                success=len(failed_deliveries) == 0,
                broadcast_id=broadcast_id,
                target_agents=targets,
                successful_deliveries=successful_deliveries,
                failed_deliveries=failed_deliveries,
                conflicts_detected=conflicts_detected,
                timestamp=timestamp
            )
            
            await self._store_broadcast_result(result)
            
            logger.info(f"Broadcast context {context.context_type} to {len(successful_deliveries)}/{len(targets)} agents")
            return result
            
        except Exception as e:
            logger.error(f"Failed to broadcast context update: {e}")
            return ContextBroadcastResult(
                success=False,
                broadcast_id=str(uuid.uuid4()),
                target_agents=targets or [],
                successful_deliveries=[],
                failed_deliveries=targets or [],
                conflicts_detected=[],
                timestamp=datetime.utcnow()
            )
    
    async def sync_agent_context(self, agent_id: str, context: AgentContext) -> bool:
        """
        Synchronize context for a specific agent
        
        Args:
            agent_id: Target agent ID
            context: Context to synchronize
            
        Returns:
            True if synchronization was successful
        """
        try:
            # Check for conflicts
            conflict = await self._detect_context_conflict(context, agent_id)
            if conflict:
                # Attempt automatic resolution
                resolution = await self.resolve_context_conflict([conflict])
                if not resolution.resolved:
                    logger.warning(f"Context conflict detected for agent {agent_id}, manual resolution required")
                    return False
                
                # Use resolved context
                context = resolution.merged_context
            
            # Create targeted broadcast
            result = await self.broadcast_context_update(context, [agent_id])
            return result.success and agent_id in result.successful_deliveries
            
        except Exception as e:
            logger.error(f"Failed to sync context for agent {agent_id}: {e}")
            return False
    
    async def resolve_context_conflict(self, conflicts: List[ContextConflict]) -> ContextResolution:
        """
        Resolve context conflicts using configured strategies
        
        Args:
            conflicts: List of context conflicts to resolve
            
        Returns:
            Context resolution result
        """
        try:
            if not conflicts:
                return ContextResolution(
                    conflict_id="none",
                    resolved=True,
                    merged_context=None,
                    resolution_strategy=ContextMergeStrategy.LATEST_WINS,
                    resolution_notes="No conflicts to resolve"
                )
            
            # For now, handle single conflict (can be extended for multiple conflicts)
            conflict = conflicts[0]
            
            # Determine resolution strategy
            strategy = conflict.resolution_strategy or self._get_default_strategy(conflict)
            
            # Apply resolution strategy
            if strategy == ContextMergeStrategy.LATEST_WINS:
                merged_context = await self._resolve_latest_wins(conflict)
            elif strategy == ContextMergeStrategy.MERGE_RECURSIVE:
                merged_context = await self._resolve_merge_recursive(conflict)
            elif strategy == ContextMergeStrategy.SOURCE_PRIORITY:
                merged_context = await self._resolve_source_priority(conflict)
            elif strategy == ContextMergeStrategy.FIELD_LEVEL_MERGE:
                merged_context = await self._resolve_field_level_merge(conflict)
            else:
                # Manual resolution required
                return ContextResolution(
                    conflict_id=conflict.conflict_id,
                    resolved=False,
                    merged_context=None,
                    resolution_strategy=strategy,
                    resolution_notes="Manual resolution required",
                    requires_manual_intervention=True,
                    affected_agents=[conflict.agent_id]
                )
            
            # Update conflict status
            conflict.resolved = True
            conflict.resolution_data = {
                "strategy": strategy.value,
                "resolved_at": datetime.utcnow().isoformat(),
                "merged_version": merged_context.version
            }
            
            # Store resolved context
            await self._store_context_locally(merged_context)
            
            return ContextResolution(
                conflict_id=conflict.conflict_id,
                resolved=True,
                merged_context=merged_context,
                resolution_strategy=strategy,
                resolution_notes=f"Resolved using {strategy.value} strategy",
                affected_agents=[conflict.agent_id]
            )
            
        except Exception as e:
            logger.error(f"Failed to resolve context conflict: {e}")
            return ContextResolution(
                conflict_id=conflicts[0].conflict_id if conflicts else "error",
                resolved=False,
                merged_context=None,
                resolution_strategy=ContextMergeStrategy.MANUAL_RESOLUTION,
                resolution_notes=f"Resolution failed: {str(e)}",
                requires_manual_intervention=True
            )
    
    async def get_shared_context(self, agent_id: str, context_type: str) -> Optional[AgentContext]:
        """
        Get shared context for an agent
        
        Args:
            agent_id: Agent ID requesting context
            context_type: Type of context to retrieve
            
        Returns:
            Shared context if available and accessible
        """
        try:
            # Check cache first
            cache_key = f"{agent_id}:{context_type}"
            if cache_key in self._context_cache:
                cached_context = self._context_cache[cache_key]
                cache_time = self._cache_timestamps.get(cache_key)
                
                if cache_time and (datetime.utcnow() - cache_time).total_seconds() < self._cache_ttl:
                    if cached_context.can_access(agent_id):
                        return cached_context
            
            # Look for context in local storage
            for stored_context in self._local_contexts.values():
                if (stored_context.context_type == context_type and 
                    stored_context.can_access(agent_id)):
                    
                    # Update cache
                    self._context_cache[cache_key] = stored_context
                    self._cache_timestamps[cache_key] = datetime.utcnow()
                    
                    return stored_context
            
            # Try to retrieve from Redis
            context = await self._retrieve_context_from_redis(context_type, agent_id)
            if context and context.can_access(agent_id):
                # Update cache and local storage
                self._context_cache[cache_key] = context
                self._cache_timestamps[cache_key] = datetime.utcnow()
                await self._store_context_locally(context)
                
                return context
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get shared context for agent {agent_id}: {e}")
            return None
    
    async def subscribe_to_context_updates(self, agent_id: str, context_type: str, callback: Callable) -> str:
        """
        Subscribe agent to context updates
        
        Args:
            agent_id: Agent ID subscribing
            context_type: Type of context to subscribe to
            callback: Callback function for updates
            
        Returns:
            Subscription ID
        """
        try:
            # Add to subscription tracking
            if context_type not in self._context_subscriptions:
                self._context_subscriptions[context_type] = set()
            self._context_subscriptions[context_type].add(agent_id)
            
            if agent_id not in self._agent_subscriptions:
                self._agent_subscriptions[agent_id] = set()
            self._agent_subscriptions[agent_id].add(context_type)
            
            # Subscribe to Redis topic
            topic = f"mcp:agent:{agent_id}:context_updates"
            subscription_id = await self.redis_backend.subscribe_to_topic(topic, callback)
            
            # Store callback reference
            self._subscription_callbacks[subscription_id] = callback
            
            logger.info(f"Agent {agent_id} subscribed to context type {context_type}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to subscribe agent {agent_id} to context updates: {e}")
            raise
    
    async def unsubscribe_from_context_updates(self, subscription_id: str, agent_id: str, context_type: str) -> bool:
        """
        Unsubscribe agent from context updates
        
        Args:
            subscription_id: Subscription ID to remove
            agent_id: Agent ID unsubscribing
            context_type: Type of context to unsubscribe from
            
        Returns:
            True if unsubscribed successfully
        """
        try:
            # Remove from subscription tracking
            if context_type in self._context_subscriptions:
                self._context_subscriptions[context_type].discard(agent_id)
                if not self._context_subscriptions[context_type]:
                    del self._context_subscriptions[context_type]
            
            if agent_id in self._agent_subscriptions:
                self._agent_subscriptions[agent_id].discard(context_type)
                if not self._agent_subscriptions[agent_id]:
                    del self._agent_subscriptions[agent_id]
            
            # Unsubscribe from Redis
            success = await self.redis_backend.unsubscribe_from_topic(subscription_id)
            
            # Remove callback reference
            self._subscription_callbacks.pop(subscription_id, None)
            
            logger.info(f"Agent {agent_id} unsubscribed from context type {context_type}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe agent {agent_id} from context updates: {e}")
            return False
    
    def _initialize_merge_strategies(self):
        """Initialize default merge strategies"""
        self._merge_strategies = {
            ContextMergeStrategy.LATEST_WINS.value: self._resolve_latest_wins,
            ContextMergeStrategy.MERGE_RECURSIVE.value: self._resolve_merge_recursive,
            ContextMergeStrategy.SOURCE_PRIORITY.value: self._resolve_source_priority,
            ContextMergeStrategy.FIELD_LEVEL_MERGE.value: self._resolve_field_level_merge
        }
    
    def _get_default_strategy(self, conflict: ContextConflict) -> ContextMergeStrategy:
        """Get default resolution strategy for a conflict"""
        if conflict.conflict_type == ContextConflictType.VERSION_CONFLICT:
            return ContextMergeStrategy.LATEST_WINS
        elif conflict.conflict_type == ContextConflictType.DATA_CONFLICT:
            return ContextMergeStrategy.MERGE_RECURSIVE
        elif conflict.conflict_type == ContextConflictType.ACCESS_CONFLICT:
            return ContextMergeStrategy.SOURCE_PRIORITY
        else:
            return ContextMergeStrategy.MANUAL_RESOLUTION
    
    async def _resolve_latest_wins(self, conflict: ContextConflict) -> AgentContext:
        """Resolve conflict using latest timestamp wins strategy"""
        local_time = conflict.local_context.last_updated
        remote_time = conflict.remote_context.last_updated
        
        if remote_time > local_time:
            winner = conflict.remote_context
        else:
            winner = conflict.local_context
        
        # Create new version
        winner.version = str(uuid.uuid4())
        winner.last_updated = datetime.utcnow()
        
        return winner
    
    async def _resolve_merge_recursive(self, conflict: ContextConflict) -> AgentContext:
        """Resolve conflict using recursive merge strategy"""
        # Start with local context as base
        merged_data = conflict.local_context.context_data.copy()
        
        # Recursively merge remote data
        def merge_dict(local_dict: Dict, remote_dict: Dict) -> Dict:
            result = local_dict.copy()
            for key, value in remote_dict.items():
                if key in result:
                    if isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = merge_dict(result[key], value)
                    else:
                        # Use remote value for conflicts (can be customized)
                        result[key] = value
                else:
                    result[key] = value
            return result
        
        merged_data = merge_dict(merged_data, conflict.remote_context.context_data)
        
        # Create merged context
        merged_context = AgentContext(
            agent_id=conflict.local_context.agent_id,
            context_type=conflict.local_context.context_type,
            context_data=merged_data,
            version=str(uuid.uuid4()),
            last_updated=datetime.utcnow(),
            shared_with=list(set(conflict.local_context.shared_with + conflict.remote_context.shared_with)),
            access_level=conflict.local_context.access_level,  # Keep local access level
            metadata={
                **conflict.local_context.metadata,
                **conflict.remote_context.metadata,
                "merged_from": [conflict.local_context.version, conflict.remote_context.version]
            }
        )
        
        return merged_context
    
    async def _resolve_source_priority(self, conflict: ContextConflict) -> AgentContext:
        """Resolve conflict using source priority strategy"""
        # Local context wins (source has priority)
        winner = conflict.local_context
        winner.version = str(uuid.uuid4())
        winner.last_updated = datetime.utcnow()
        
        return winner
    
    async def _resolve_field_level_merge(self, conflict: ContextConflict) -> AgentContext:
        """Resolve conflict using field-level merge strategy"""
        # More sophisticated field-level merging
        local_data = conflict.local_context.context_data
        remote_data = conflict.remote_context.context_data
        merged_data = {}
        
        # Get all unique keys
        all_keys = set(local_data.keys()) | set(remote_data.keys())
        
        for key in all_keys:
            local_value = local_data.get(key)
            remote_value = remote_data.get(key)
            
            if local_value is None:
                merged_data[key] = remote_value
            elif remote_value is None:
                merged_data[key] = local_value
            elif local_value == remote_value:
                merged_data[key] = local_value
            else:
                # Field conflict - use timestamp-based resolution
                local_time = conflict.local_context.last_updated
                remote_time = conflict.remote_context.last_updated
                
                if remote_time > local_time:
                    merged_data[key] = remote_value
                else:
                    merged_data[key] = local_value
        
        # Create merged context
        merged_context = AgentContext(
            agent_id=conflict.local_context.agent_id,
            context_type=conflict.local_context.context_type,
            context_data=merged_data,
            version=str(uuid.uuid4()),
            last_updated=datetime.utcnow(),
            shared_with=list(set(conflict.local_context.shared_with + conflict.remote_context.shared_with)),
            access_level=conflict.local_context.access_level,
            metadata={
                **conflict.local_context.metadata,
                **conflict.remote_context.metadata,
                "field_level_merge": True
            }
        )
        
        return merged_context
    
    async def _detect_context_conflict(self, context: AgentContext, target_agent: str) -> Optional[ContextConflict]:
        """
        Detect context conflicts before broadcasting
        
        Args:
            context: Context being broadcast
            target_agent: Target agent ID
            
        Returns:
            Context conflict if detected, None otherwise
        """
        try:
            # Get existing context for target agent
            existing_context = await self.get_shared_context(target_agent, context.context_type)
            if not existing_context:
                return None  # No conflict if no existing context
            
            # Check for version conflicts
            if existing_context.version != context.version:
                # Check if this is a legitimate update or a conflict
                if existing_context.last_updated > context.last_updated:
                    return ContextConflict(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type=ContextConflictType.VERSION_CONFLICT,
                        context_type=context.context_type,
                        agent_id=target_agent,
                        conflicting_versions=[existing_context.version, context.version],
                        local_context=existing_context,
                        remote_context=context,
                        detected_at=datetime.utcnow()
                    )
            
            # Check for data conflicts
            if existing_context.context_data != context.context_data:
                return ContextConflict(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type=ContextConflictType.DATA_CONFLICT,
                    context_type=context.context_type,
                    agent_id=target_agent,
                    conflicting_versions=[existing_context.version, context.version],
                    local_context=existing_context,
                    remote_context=context,
                    detected_at=datetime.utcnow()
                )
            
            # Check for access conflicts
            if not context.can_access(target_agent):
                return ContextConflict(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type=ContextConflictType.ACCESS_CONFLICT,
                    context_type=context.context_type,
                    agent_id=target_agent,
                    conflicting_versions=[existing_context.version, context.version],
                    local_context=existing_context,
                    remote_context=context,
                    detected_at=datetime.utcnow()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting context conflict: {e}")
            return None
    
    async def _store_context_locally(self, context: AgentContext):
        """Store context in local storage with history"""
        try:
            # Store current context
            self._local_contexts[context.agent_id] = context
            
            # Update version tracking
            if context.agent_id not in self._context_versions:
                self._context_versions[context.agent_id] = {}
            self._context_versions[context.agent_id][context.context_type] = context.version
            
            # Store in history
            if context.agent_id not in self._context_history:
                self._context_history[context.agent_id] = []
            
            self._context_history[context.agent_id].append(context)
            
            # Trim history to max size
            if len(self._context_history[context.agent_id]) > self.max_history_size:
                self._context_history[context.agent_id] = self._context_history[context.agent_id][-self.max_history_size:]
            
            # Store in Redis for persistence
            await self._store_context_in_redis(context)
            
        except Exception as e:
            logger.error(f"Failed to store context locally: {e}")
    
    async def _store_context_in_redis(self, context: AgentContext):
        """Store context in Redis for persistence"""
        try:
            key = f"mcp:context:{context.agent_id}:{context.context_type}"
            context_data = json.dumps(context.dict())
            
            # Store with TTL
            ttl = context.ttl or 3600  # Default 1 hour
            await self.redis_backend._redis.setex(key, ttl, context_data)
            
            # Store in context index
            index_key = f"mcp:context_index:{context.context_type}"
            await self.redis_backend._redis.sadd(index_key, context.agent_id)
            await self.redis_backend._redis.expire(index_key, ttl)
            
        except Exception as e:
            logger.error(f"Failed to store context in Redis: {e}")
    
    async def _retrieve_context_from_redis(self, context_type: str, agent_id: str) -> Optional[AgentContext]:
        """Retrieve context from Redis"""
        try:
            key = f"mcp:context:{agent_id}:{context_type}"
            context_data = await self.redis_backend._redis.get(key)
            
            if context_data:
                context_dict = json.loads(context_data)
                return AgentContext(**context_dict)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve context from Redis: {e}")
            return None
    
    async def _track_context_delivery(self, broadcast_id: str, target_agent: str, context: AgentContext):
        """Track context delivery for monitoring"""
        try:
            delivery_key = f"mcp:delivery:{broadcast_id}"
            delivery_data = {
                "target_agent": target_agent,
                "context_type": context.context_type,
                "context_version": context.version,
                "delivered_at": datetime.utcnow().isoformat()
            }
            
            await self.redis_backend._redis.hset(
                delivery_key,
                target_agent,
                json.dumps(delivery_data)
            )
            await self.redis_backend._redis.expire(delivery_key, 3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Failed to track context delivery: {e}")
    
    async def _store_broadcast_result(self, result: ContextBroadcastResult):
        """Store broadcast result for monitoring"""
        try:
            result_key = f"mcp:broadcast_result:{result.broadcast_id}"
            result_data = {
                "success": result.success,
                "target_agents": result.target_agents,
                "successful_deliveries": result.successful_deliveries,
                "failed_deliveries": result.failed_deliveries,
                "conflicts_count": len(result.conflicts_detected),
                "timestamp": result.timestamp.isoformat()
            }
            
            await self.redis_backend._redis.setex(
                result_key,
                3600,  # 1 hour TTL
                json.dumps(result_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to store broadcast result: {e}")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Conflict cleanup task
        conflict_cleanup_task = asyncio.create_task(self._conflict_cleanup_worker())
        self._background_tasks.add(conflict_cleanup_task)
        
        # Cache cleanup task
        cache_cleanup_task = asyncio.create_task(self._cache_cleanup_worker())
        self._background_tasks.add(cache_cleanup_task)
        
        # Context synchronization health check
        health_check_task = asyncio.create_task(self._health_check_worker())
        self._background_tasks.add(health_check_task)
        
        logger.info("Started context synchronizer background tasks")
    
    async def _conflict_cleanup_worker(self):
        """Clean up expired conflicts"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                expired_conflicts = []
                
                for conflict_id, conflict in self._active_conflicts.items():
                    age = (current_time - conflict.detected_at).total_seconds()
                    if age > self.conflict_timeout:
                        expired_conflicts.append(conflict_id)
                
                # Remove expired conflicts
                for conflict_id in expired_conflicts:
                    del self._active_conflicts[conflict_id]
                    logger.info(f"Cleaned up expired conflict {conflict_id}")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in conflict cleanup worker: {e}")
                await asyncio.sleep(300)
    
    async def _cache_cleanup_worker(self):
        """Clean up expired cache entries"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for cache_key, timestamp in self._cache_timestamps.items():
                    age = (current_time - timestamp).total_seconds()
                    if age > self._cache_ttl:
                        expired_keys.append(cache_key)
                
                # Remove expired cache entries
                for cache_key in expired_keys:
                    self._context_cache.pop(cache_key, None)
                    self._cache_timestamps.pop(cache_key, None)
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in cache cleanup worker: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_worker(self):
        """Monitor context synchronization health"""
        while self._running:
            try:
                # Check Redis connectivity
                if self.redis_backend._redis:
                    await self.redis_backend._redis.ping()
                
                # Log health metrics
                metrics = {
                    "local_contexts": len(self._local_contexts),
                    "active_conflicts": len(self._active_conflicts),
                    "cached_contexts": len(self._context_cache),
                    "context_subscriptions": sum(len(agents) for agents in self._context_subscriptions.values()),
                    "agent_subscriptions": len(self._agent_subscriptions)
                }
                
                logger.debug(f"Context synchronizer health: {metrics}")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Context synchronizer health check failed: {e}")
                await asyncio.sleep(300)
    
    # Public API methods for monitoring and management
    
    async def get_context_statistics(self) -> Dict[str, Any]:
        """Get context synchronization statistics"""
        return {
            "local_contexts": len(self._local_contexts),
            "context_versions": len(self._context_versions),
            "context_history_entries": sum(len(history) for history in self._context_history.values()),
            "active_conflicts": len(self._active_conflicts),
            "cached_contexts": len(self._context_cache),
            "context_subscriptions": dict(self._context_subscriptions),
            "agent_subscriptions": dict(self._agent_subscriptions),
            "running": self._running
        }
    
    async def get_active_conflicts(self) -> List[ContextConflict]:
        """Get list of active conflicts"""
        return list(self._active_conflicts.values())
    
    async def force_context_sync(self, agent_id: str, context_type: str) -> bool:
        """Force context synchronization for an agent"""
        try:
            context = await self.get_shared_context(agent_id, context_type)
            if context:
                result = await self.broadcast_context_update(context)
                return result.success
            return False
            
        except Exception as e:
            logger.error(f"Failed to force context sync: {e}")
            return False
    
    async def clear_context_cache(self):
        """Clear the context cache"""
        self._context_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Context cache cleared")
    
    async def export_context_data(self, agent_id: str = None) -> Dict[str, Any]:
        """Export context data for backup or analysis"""
        try:
            export_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "contexts": {},
                "versions": {},
                "history": {}
            }
            
            if agent_id:
                # Export specific agent data
                if agent_id in self._local_contexts:
                    export_data["contexts"][agent_id] = self._local_contexts[agent_id].dict()
                if agent_id in self._context_versions:
                    export_data["versions"][agent_id] = self._context_versions[agent_id]
                if agent_id in self._context_history:
                    export_data["history"][agent_id] = [ctx.dict() for ctx in self._context_history[agent_id]]
            else:
                # Export all data
                export_data["contexts"] = {aid: ctx.dict() for aid, ctx in self._local_contexts.items()}
                export_data["versions"] = dict(self._context_versions)
                export_data["history"] = {
                    aid: [ctx.dict() for ctx in history] 
                    for aid, history in self._context_history.items()
                }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export context data: {e}")
            return {}


# Global context synchronizer instance
context_synchronizer = ContextSynchronizer()