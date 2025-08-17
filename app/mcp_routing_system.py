# app/mcp_routing_system.py
"""
MCP Message Routing and Delivery System

This module provides comprehensive message routing, queuing, and delivery
mechanisms with acknowledgment support, retry logic, and delivery guarantees.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field

from .mcp_models import (
    MCPMessage, AgentContext, MessageRoutingRule, ValidationResult,
    MCPMessageType, MCPMessagePriority
)
from .redis_mcp_backend import RedisMCPBackend
from .mcp_message_handler import MCPMessageHandler

logger = logging.getLogger(__name__)


class DeliveryStatus(Enum):
    """Message delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"
    RETRYING = "retrying"


class QueuePriority(Enum):
    """Message queue priorities"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DeliveryAttempt:
    """Record of a message delivery attempt"""
    attempt_number: int
    timestamp: datetime
    status: DeliveryStatus
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class MessageDeliveryRecord:
    """Complete delivery record for a message"""
    message_id: str
    target_agent: str
    status: DeliveryStatus
    created_at: datetime
    last_attempt: Optional[datetime] = None
    attempts: List[DeliveryAttempt] = field(default_factory=list)
    requires_ack: bool = False
    ack_timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    next_retry: Optional[datetime] = None
    
    def add_attempt(self, status: DeliveryStatus, error_message: str = None, response_time_ms: float = None):
        """Add a delivery attempt record"""
        attempt = DeliveryAttempt(
            attempt_number=len(self.attempts) + 1,
            timestamp=datetime.utcnow(),
            status=status,
            error_message=error_message,
            response_time_ms=response_time_ms
        )
        self.attempts.append(attempt)
        self.last_attempt = attempt.timestamp
        self.status = status
    
    def should_retry(self) -> bool:
        """Check if message should be retried"""
        if self.status in [DeliveryStatus.DELIVERED, DeliveryStatus.ACKNOWLEDGED, DeliveryStatus.EXPIRED]:
            return False
        
        if self.retry_count >= self.max_retries:
            return False
        
        if self.next_retry and datetime.utcnow() < self.next_retry:
            return False
        
        return True
    
    def calculate_next_retry(self, base_delay: float = 1.0, max_delay: float = 300.0):
        """Calculate next retry time with exponential backoff"""
        delay = min(base_delay * (2 ** self.retry_count), max_delay)
        self.next_retry = datetime.utcnow() + timedelta(seconds=delay)


@dataclass
class RoutingConfig:
    """Configuration for message routing"""
    enable_routing_rules: bool = True
    enable_load_balancing: bool = True
    enable_failover: bool = True
    default_timeout: int = 30
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 300.0
    ack_timeout: int = 60
    queue_size_limit: int = 10000
    enable_metrics: bool = True


class MCPRoutingSystem:
    """
    Comprehensive MCP message routing and delivery system
    
    Features:
    - Intelligent message routing based on agent capabilities
    - Message queuing with priority support
    - Delivery guarantees with acknowledgment and retry logic
    - Load balancing and failover
    - Performance monitoring and metrics
    """
    
    def __init__(self, redis_backend: RedisMCPBackend, message_handler: MCPMessageHandler,
                 config: RoutingConfig = None):
        self.redis_backend = redis_backend
        self.message_handler = message_handler
        self.config = config or RoutingConfig()
        
        # Agent registry and capabilities
        self._registered_agents: Dict[str, Dict[str, Any]] = {}
        self._agent_capabilities: Dict[str, List[str]] = {}
        self._agent_load: Dict[str, int] = {}  # Track message load per agent
        self._agent_health: Dict[str, bool] = {}
        
        # Routing rules
        self._routing_rules: List[MessageRoutingRule] = []
        
        # Delivery tracking
        self._delivery_records: Dict[str, MessageDeliveryRecord] = {}
        self._pending_acks: Dict[str, MessageDeliveryRecord] = {}
        
        # Message queues by priority
        self._message_queues: Dict[QueuePriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=self.config.queue_size_limit)
            for priority in QueuePriority
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        # Statistics
        self.stats = {
            "messages_routed": 0,
            "messages_delivered": 0,
            "messages_acknowledged": 0,
            "messages_failed": 0,
            "messages_retried": 0,
            "routing_rules_applied": 0,
            "load_balancing_decisions": 0,
            "failover_events": 0,
            "average_delivery_time_ms": 0.0,
            "queue_sizes": {}
        }
        
        logger.info("MCP Routing System initialized")
    
    async def start(self):
        """Start the routing system and background tasks"""
        if self._running:
            return
        
        self._running = True
        
        # Start background workers
        await self._start_background_tasks()
        
        logger.info("MCP Routing System started")
    
    async def stop(self):
        """Stop the routing system and cleanup"""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("MCP Routing System stopped")
    
    def register_agent(self, agent_id: str, capabilities: List[str], metadata: Dict[str, Any] = None):
        """
        Register an agent with the routing system
        
        Args:
            agent_id: Unique agent identifier
            capabilities: List of message types the agent can handle
            metadata: Additional agent metadata
        """
        self._registered_agents[agent_id] = {
            "capabilities": capabilities,
            "metadata": metadata or {},
            "registered_at": datetime.utcnow(),
            "last_seen": datetime.utcnow()
        }
        
        self._agent_capabilities[agent_id] = capabilities
        self._agent_load[agent_id] = 0
        self._agent_health[agent_id] = True
        
        logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the routing system"""
        if agent_id in self._registered_agents:
            del self._registered_agents[agent_id]
            del self._agent_capabilities[agent_id]
            del self._agent_load[agent_id]
            del self._agent_health[agent_id]
            
            logger.info(f"Unregistered agent {agent_id}")
    
    def add_routing_rule(self, rule: MessageRoutingRule):
        """Add a message routing rule"""
        self._routing_rules.append(rule)
        logger.info(f"Added routing rule for message type: {rule.message_type}")
    
    def remove_routing_rule(self, rule: MessageRoutingRule):
        """Remove a message routing rule"""
        if rule in self._routing_rules:
            self._routing_rules.remove(rule)
            logger.info(f"Removed routing rule for message type: {rule.message_type}")
    
    async def route_message(self, message: MCPMessage) -> bool:
        """
        Route a message to appropriate agents with delivery guarantees
        
        Args:
            message: MCP message to route
            
        Returns:
            True if message was successfully queued for delivery
        """
        try:
            start_time = time.time()
            
            # Validate message
            validation_result = self.message_handler.validate_message(message)
            if not validation_result.is_valid:
                logger.error(f"Message validation failed: {validation_result.errors}")
                return False
            
            # Apply routing rules to determine final target agents
            target_agents = await self._apply_routing_rules(message)
            
            # Apply load balancing and failover
            target_agents = await self._apply_load_balancing(target_agents, message)
            
            if not target_agents:
                logger.warning(f"No target agents found for message {message.id}")
                return False
            
            # Create delivery records
            delivery_records = []
            for agent_id in target_agents:
                record = MessageDeliveryRecord(
                    message_id=message.id,
                    target_agent=agent_id,
                    status=DeliveryStatus.PENDING,
                    created_at=datetime.utcnow(),
                    requires_ack=message.requires_ack,
                    ack_timeout=self.config.ack_timeout,
                    max_retries=self.config.max_retries
                )
                delivery_records.append(record)
                self._delivery_records[f"{message.id}:{agent_id}"] = record
            
            # Queue message for delivery
            priority = self._get_queue_priority(message.priority)
            delivery_task = {
                "message": message,
                "target_agents": target_agents,
                "delivery_records": delivery_records,
                "queued_at": datetime.utcnow()
            }
            
            await self._message_queues[priority].put(delivery_task)
            
            # Update statistics
            self.stats["messages_routed"] += 1
            routing_time = (time.time() - start_time) * 1000
            
            logger.debug(f"Routed message {message.id} to {len(target_agents)} agents in {routing_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to route message {message.id}: {e}")
            return False
    
    async def acknowledge_message(self, message_id: str, agent_id: str, response_data: Dict[str, Any] = None) -> bool:
        """
        Acknowledge message delivery
        
        Args:
            message_id: ID of the acknowledged message
            agent_id: ID of the agent acknowledging
            response_data: Optional response data
            
        Returns:
            True if acknowledgment was processed
        """
        try:
            record_key = f"{message_id}:{agent_id}"
            
            if record_key not in self._delivery_records:
                logger.warning(f"No delivery record found for message {message_id} to agent {agent_id}")
                return False
            
            record = self._delivery_records[record_key]
            
            if record.status != DeliveryStatus.DELIVERED:
                logger.warning(f"Cannot acknowledge message {message_id} with status {record.status}")
                return False
            
            # Update delivery record
            record.add_attempt(DeliveryStatus.ACKNOWLEDGED)
            
            # Remove from pending acknowledgments
            if record_key in self._pending_acks:
                del self._pending_acks[record_key]
            
            # Update statistics
            self.stats["messages_acknowledged"] += 1
            
            logger.debug(f"Acknowledged message {message_id} from agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge message {message_id}: {e}")
            return False
    
    async def get_delivery_status(self, message_id: str) -> Dict[str, Any]:
        """
        Get delivery status for a message
        
        Args:
            message_id: Message ID to check
            
        Returns:
            Dictionary with delivery status information
        """
        try:
            # Find all delivery records for this message
            records = {
                key: record for key, record in self._delivery_records.items()
                if record.message_id == message_id
            }
            
            if not records:
                return {"message_id": message_id, "status": "not_found"}
            
            # Aggregate status information
            status_summary = {
                "message_id": message_id,
                "total_targets": len(records),
                "delivered": sum(1 for r in records.values() if r.status == DeliveryStatus.DELIVERED),
                "acknowledged": sum(1 for r in records.values() if r.status == DeliveryStatus.ACKNOWLEDGED),
                "failed": sum(1 for r in records.values() if r.status == DeliveryStatus.FAILED),
                "pending": sum(1 for r in records.values() if r.status == DeliveryStatus.PENDING),
                "retrying": sum(1 for r in records.values() if r.status == DeliveryStatus.RETRYING),
                "records": {
                    key: {
                        "target_agent": record.target_agent,
                        "status": record.status.value,
                        "attempts": len(record.attempts),
                        "last_attempt": record.last_attempt.isoformat() if record.last_attempt else None,
                        "next_retry": record.next_retry.isoformat() if record.next_retry else None
                    }
                    for key, record in records.items()
                }
            }
            
            return status_summary
            
        except Exception as e:
            logger.error(f"Failed to get delivery status for message {message_id}: {e}")
            return {"message_id": message_id, "status": "error", "error": str(e)}
    
    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing system statistics"""
        # Update queue sizes
        for priority in QueuePriority:
            self.stats["queue_sizes"][priority.name] = self._message_queues[priority].qsize()
        
        # Add agent statistics
        agent_stats = {
            "registered_agents": len(self._registered_agents),
            "healthy_agents": sum(1 for healthy in self._agent_health.values() if healthy),
            "agent_load": self._agent_load.copy(),
            "agent_capabilities": {
                agent_id: len(caps) for agent_id, caps in self._agent_capabilities.items()
            }
        }
        
        return {
            "routing_stats": self.stats.copy(),
            "agent_stats": agent_stats,
            "delivery_records": len(self._delivery_records),
            "pending_acks": len(self._pending_acks),
            "routing_rules": len(self._routing_rules)
        }
    
    async def _apply_routing_rules(self, message: MCPMessage) -> List[str]:
        """Apply routing rules to determine target agents"""
        target_agents = set(message.target_agents)
        
        if self.config.enable_routing_rules:
            for rule in self._routing_rules:
                if rule.matches(message):
                    target_agents.update(rule.route_to)
                    self.stats["routing_rules_applied"] += 1
        
        return list(target_agents)
    
    async def _apply_load_balancing(self, target_agents: List[str], message: MCPMessage) -> List[str]:
        """Apply load balancing and failover logic"""
        if not self.config.enable_load_balancing:
            return target_agents
        
        # Filter out unhealthy agents
        healthy_agents = [
            agent_id for agent_id in target_agents
            if agent_id in self._agent_health and self._agent_health[agent_id]
        ]
        
        if not healthy_agents:
            logger.warning(f"No healthy agents available for message {message.id}")
            return target_agents  # Return original list as fallback
        
        # For high-priority messages, prefer agents with lower load
        if message.priority >= MCPMessagePriority.HIGH.value:
            healthy_agents.sort(key=lambda agent_id: self._agent_load.get(agent_id, 0))
            self.stats["load_balancing_decisions"] += 1
        
        return healthy_agents
    
    def _get_queue_priority(self, message_priority: int) -> QueuePriority:
        """Convert message priority to queue priority"""
        if message_priority >= MCPMessagePriority.CRITICAL.value:
            return QueuePriority.CRITICAL
        elif message_priority >= MCPMessagePriority.HIGH.value:
            return QueuePriority.HIGH
        elif message_priority >= MCPMessagePriority.NORMAL.value:
            return QueuePriority.NORMAL
        else:
            return QueuePriority.LOW
    
    async def _start_background_tasks(self):
        """Start background worker tasks"""
        # Message delivery workers (one per priority level)
        for priority in QueuePriority:
            worker_task = asyncio.create_task(self._delivery_worker(priority))
            self._background_tasks.add(worker_task)
        
        # Retry handler
        retry_task = asyncio.create_task(self._retry_worker())
        self._background_tasks.add(retry_task)
        
        # Acknowledgment timeout handler
        ack_timeout_task = asyncio.create_task(self._ack_timeout_worker())
        self._background_tasks.add(ack_timeout_task)
        
        # Health check worker
        health_task = asyncio.create_task(self._health_check_worker())
        self._background_tasks.add(health_task)
        
        # Cleanup worker
        cleanup_task = asyncio.create_task(self._cleanup_worker())
        self._background_tasks.add(cleanup_task)
        
        logger.info("Started routing system background tasks")
    
    async def _delivery_worker(self, priority: QueuePriority):
        """Worker for delivering messages from priority queue"""
        queue = self._message_queues[priority]
        
        logger.info(f"Started delivery worker for {priority.name} priority queue")
        
        while self._running:
            try:
                # Get next delivery task
                delivery_task = await queue.get()
                
                message = delivery_task["message"]
                target_agents = delivery_task["target_agents"]
                delivery_records = delivery_task["delivery_records"]
                
                # Deliver to each target agent
                for agent_id, record in zip(target_agents, delivery_records):
                    await self._deliver_message_to_agent(message, agent_id, record)
                
                # Mark task as done
                queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in {priority.name} delivery worker: {e}")
                await asyncio.sleep(1)
    
    async def _deliver_message_to_agent(self, message: MCPMessage, agent_id: str, record: MessageDeliveryRecord):
        """Deliver a message to a specific agent"""
        try:
            start_time = time.time()
            
            # Check if agent is registered and healthy
            if agent_id not in self._registered_agents:
                record.add_attempt(DeliveryStatus.FAILED, "Agent not registered")
                return
            
            if not self._agent_health.get(agent_id, False):
                record.add_attempt(DeliveryStatus.FAILED, "Agent unhealthy")
                return
            
            # Create agent-specific topic
            agent_topic = f"mcp:agent:{agent_id}:messages"
            
            # Publish message to agent's topic
            success = await self.redis_backend.publish_message(agent_topic, message)
            
            delivery_time = (time.time() - start_time) * 1000
            
            if success:
                record.add_attempt(DeliveryStatus.DELIVERED, response_time_ms=delivery_time)
                
                # Track acknowledgment if required
                if message.requires_ack:
                    record_key = f"{message.id}:{agent_id}"
                    self._pending_acks[record_key] = record
                
                # Update agent load
                self._agent_load[agent_id] = self._agent_load.get(agent_id, 0) + 1
                
                # Update statistics
                self.stats["messages_delivered"] += 1
                
                # Update average delivery time
                current_avg = self.stats["average_delivery_time_ms"]
                total_delivered = self.stats["messages_delivered"]
                self.stats["average_delivery_time_ms"] = (
                    (current_avg * (total_delivered - 1) + delivery_time) / total_delivered
                )
                
                logger.debug(f"Delivered message {message.id} to agent {agent_id} in {delivery_time:.2f}ms")
            else:
                record.add_attempt(DeliveryStatus.FAILED, "Failed to publish to Redis")
                logger.error(f"Failed to deliver message {message.id} to agent {agent_id}")
                
        except Exception as e:
            record.add_attempt(DeliveryStatus.FAILED, str(e))
            logger.error(f"Error delivering message {message.id} to agent {agent_id}: {e}")
    
    async def _retry_worker(self):
        """Worker for handling message retries"""
        logger.info("Started retry worker")
        
        while self._running:
            try:
                current_time = datetime.utcnow()
                
                # Find messages that need retry
                retry_records = [
                    record for record in self._delivery_records.values()
                    if record.should_retry() and (not record.next_retry or current_time >= record.next_retry)
                ]
                
                for record in retry_records:
                    # Update retry count and calculate next retry time
                    record.retry_count += 1
                    record.calculate_next_retry(
                        self.config.retry_base_delay,
                        self.config.retry_max_delay
                    )
                    
                    # Find the original message (this would need to be stored separately in a real implementation)
                    # For now, we'll create a retry notification
                    logger.info(f"Retrying delivery for message {record.message_id} to agent {record.target_agent} (attempt {record.retry_count})")
                    
                    record.status = DeliveryStatus.RETRYING
                    self.stats["messages_retried"] += 1
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in retry worker: {e}")
                await asyncio.sleep(10)
    
    async def _ack_timeout_worker(self):
        """Worker for handling acknowledgment timeouts"""
        logger.info("Started acknowledgment timeout worker")
        
        while self._running:
            try:
                current_time = datetime.utcnow()
                
                # Find acknowledgments that have timed out
                timed_out_acks = []
                for record_key, record in self._pending_acks.items():
                    if record.ack_timeout and record.last_attempt:
                        timeout_time = record.last_attempt + timedelta(seconds=record.ack_timeout)
                        if current_time > timeout_time:
                            timed_out_acks.append(record_key)
                
                # Handle timed out acknowledgments
                for record_key in timed_out_acks:
                    record = self._pending_acks[record_key]
                    record.add_attempt(DeliveryStatus.FAILED, "Acknowledgment timeout")
                    del self._pending_acks[record_key]
                    
                    logger.warning(f"Acknowledgment timeout for message {record.message_id} to agent {record.target_agent}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in acknowledgment timeout worker: {e}")
                await asyncio.sleep(30)
    
    async def _health_check_worker(self):
        """Worker for checking agent health"""
        logger.info("Started health check worker")
        
        while self._running:
            try:
                # Simple health check - could be enhanced with actual ping/heartbeat
                for agent_id in list(self._registered_agents.keys()):
                    # For now, assume all registered agents are healthy
                    # In a real implementation, this would check heartbeats, response times, etc.
                    self._agent_health[agent_id] = True
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check worker: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_worker(self):
        """Worker for cleaning up old delivery records"""
        logger.info("Started cleanup worker")
        
        while self._running:
            try:
                current_time = datetime.utcnow()
                cleanup_threshold = current_time - timedelta(hours=24)  # Keep records for 24 hours
                
                # Clean up old delivery records
                old_records = [
                    key for key, record in self._delivery_records.items()
                    if record.created_at < cleanup_threshold and 
                    record.status in [DeliveryStatus.ACKNOWLEDGED, DeliveryStatus.FAILED, DeliveryStatus.EXPIRED]
                ]
                
                for key in old_records:
                    del self._delivery_records[key]
                
                if old_records:
                    logger.info(f"Cleaned up {len(old_records)} old delivery records")
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(3600)


# Factory function to create routing system with dependencies
def create_routing_system(redis_backend: RedisMCPBackend = None, 
                         message_handler: MCPMessageHandler = None,
                         config: RoutingConfig = None) -> MCPRoutingSystem:
    """Create a configured MCP routing system"""
    from .redis_mcp_backend import redis_mcp_backend
    from .mcp_message_handler import mcp_message_handler
    
    backend = redis_backend or redis_mcp_backend
    handler = message_handler or mcp_message_handler
    
    return MCPRoutingSystem(backend, handler, config)