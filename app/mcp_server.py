# app/mcp_server.py
"""
Enhanced MCP Server Core Infrastructure

This module implements a comprehensive Model Context Protocol (MCP) server
with Redis backend integration, connection pooling, message validation,
and robust error handling.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from .config import settings
from .mcp_models import MCPMessage, MCPMessageType, MCPMessagePriority, ValidationResult

logger = logging.getLogger(__name__)


# Message models are now imported from mcp_models.py


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    message_types: List[str]
    parameters: Dict[str, Any]


@dataclass
class RegisteredAgent:
    """Registered agent information"""
    agent_id: str
    capabilities: List[AgentCapability]
    connection_id: str
    last_heartbeat: datetime
    status: str = "active"  # active, inactive, error
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ValidationResult is now imported from mcp_models.py


class ServerStatus:
    """MCP server status information"""
    def __init__(self):
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.connected_agents = 0
        self.total_messages_processed = 0
        self.redis_connected = False
        self.last_error: Optional[str] = None
        self.performance_metrics = {}


class MCPServer:
    """
    Enhanced MCP Server with Redis backend integration
    
    Provides comprehensive MCP protocol implementation with:
    - Redis-based message queuing and persistence
    - Connection pooling and failover
    - Message validation and routing
    - Agent registration and capability management
    - Error handling and recovery
    """
    
    def __init__(self, redis_url: str = None, max_connections: int = 20):
        self.redis_url = redis_url or settings.REDIS_URL
        self.max_connections = max_connections
        
        # Core components
        self._redis_pool: Optional[ConnectionPool] = None
        self._redis: Optional[redis.Redis] = None
        self._registered_agents: Dict[str, RegisteredAgent] = {}
        self._message_handlers: Dict[str, Callable] = {}
        self._subscriptions: Dict[str, Set[str]] = {}  # topic -> agent_ids
        self._running = False
        self._status = ServerStatus()
        
        # Routing system (will be initialized when server starts)
        self._routing_system = None
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.agent_timeout = 120  # seconds
        self.message_ttl = 3600  # 1 hour default TTL
        self.max_queue_size = 10000
        self.retry_attempts = 3
        self.retry_delay = 1.0
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
        logger.info("MCP Server initialized")
    
    async def start_server(self, host: str = "localhost", port: int = 8765) -> bool:
        """
        Start the MCP server
        
        Args:
            host: Server host address
            port: Server port number
            
        Returns:
            True if server started successfully
        """
        try:
            if self._running:
                logger.warning("MCP Server is already running")
                return True
            
            # Initialize Redis connection
            if not await self._initialize_redis():
                logger.error("Failed to initialize Redis connection")
                return False
            
            # Initialize routing system
            await self._initialize_routing_system()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Update status
            self._running = True
            self._status.is_running = True
            self._status.start_time = datetime.utcnow()
            self._status.redis_connected = True
            
            logger.info(f"MCP Server started on {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            self._status.last_error = str(e)
            return False
    
    async def stop_server(self) -> None:
        """Stop the MCP server and cleanup resources"""
        try:
            self._running = False
            self._status.is_running = False
            
            # Stop routing system
            if self._routing_system:
                await self._routing_system.stop()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close Redis connections
            if self._redis:
                await self._redis.close()
            
            if self._redis_pool:
                await self._redis_pool.disconnect()
            
            logger.info("MCP Server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MCP server: {e}")
    
    async def register_agent(self, agent_id: str, capabilities: List[Dict[str, Any]]) -> str:
        """
        Register an agent with the MCP server
        
        Args:
            agent_id: Unique agent identifier
            capabilities: List of agent capabilities
            
        Returns:
            Connection ID for the registered agent
        """
        try:
            # Parse capabilities
            parsed_capabilities = []
            for cap_data in capabilities:
                capability = AgentCapability(
                    name=cap_data.get("name", ""),
                    description=cap_data.get("description", ""),
                    message_types=cap_data.get("message_types", []),
                    parameters=cap_data.get("parameters", {})
                )
                parsed_capabilities.append(capability)
            
            # Generate connection ID
            connection_id = str(uuid.uuid4())
            
            # Create registered agent
            agent = RegisteredAgent(
                agent_id=agent_id,
                capabilities=parsed_capabilities,
                connection_id=connection_id,
                last_heartbeat=datetime.utcnow()
            )
            
            # Store agent registration
            self._registered_agents[agent_id] = agent
            
            # Store in Redis for persistence
            await self._store_agent_registration(agent)
            
            # Register with routing system
            if self._routing_system:
                capability_names = [cap.name for cap in parsed_capabilities]
                self._routing_system.register_agent(agent_id, capability_names, agent.metadata)
            
            # Update status
            self._status.connected_agents = len(self._registered_agents)
            
            logger.info(f"Registered agent {agent_id} with connection {connection_id}")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the MCP server
        
        Args:
            agent_id: Agent identifier to unregister
            
        Returns:
            True if agent was unregistered successfully
        """
        try:
            if agent_id not in self._registered_agents:
                logger.warning(f"Agent {agent_id} not found for unregistration")
                return False
            
            # Remove from local registry
            del self._registered_agents[agent_id]
            
            # Remove from Redis
            await self._remove_agent_registration(agent_id)
            
            # Unregister from routing system
            if self._routing_system:
                self._routing_system.unregister_agent(agent_id)
            
            # Clean up subscriptions
            await self._cleanup_agent_subscriptions(agent_id)
            
            # Update status
            self._status.connected_agents = len(self._registered_agents)
            
            logger.info(f"Unregistered agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def route_message(self, message: MCPMessage) -> bool:
        """
        Route a message to appropriate agents using the routing system
        
        Args:
            message: MCP message to route
            
        Returns:
            True if message was routed successfully
        """
        try:
            # Use routing system if available, otherwise fall back to direct routing
            if self._routing_system:
                success = await self._routing_system.route_message(message)
                if success:
                    self._status.total_messages_processed += 1
                return success
            else:
                # Fallback to direct routing
                return await self._route_message_direct(message)
            
        except Exception as e:
            logger.error(f"Failed to route message {message.id}: {e}")
            await self._send_error_response(message, "Message routing failed", [str(e)])
            return False
    
    async def broadcast_message(self, message: MCPMessage, target_type: str = None) -> int:
        """
        Broadcast a message to multiple agents
        
        Args:
            message: MCP message to broadcast
            target_type: Optional filter by agent capability type
            
        Returns:
            Number of agents the message was sent to
        """
        try:
            # Determine target agents
            target_agents = []
            
            if target_type:
                # Filter by capability type
                for agent_id, agent in self._registered_agents.items():
                    for capability in agent.capabilities:
                        if target_type in capability.message_types:
                            target_agents.append(agent_id)
                            break
            else:
                # Broadcast to all registered agents
                target_agents = list(self._registered_agents.keys())
            
            # Update message targets
            message.target_agents = target_agents
            
            # Route message
            if await self.route_message(message):
                logger.info(f"Broadcast message {message.id} to {len(target_agents)} agents")
                return len(target_agents)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to broadcast message {message.id}: {e}")
            return 0
    
    async def get_server_status(self) -> ServerStatus:
        """Get current server status"""
        # Update dynamic status information
        self._status.connected_agents = len(self._registered_agents)
        
        # Check Redis connection
        try:
            if self._redis:
                await self._redis.ping()
                self._status.redis_connected = True
        except Exception:
            self._status.redis_connected = False
        
        return self._status
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a handler for specific message types"""
        self._message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def acknowledge_message(self, message_id: str, agent_id: str, response_data: Dict[str, Any] = None) -> bool:
        """
        Acknowledge message delivery from an agent
        
        Args:
            message_id: ID of the message being acknowledged
            agent_id: ID of the agent acknowledging the message
            response_data: Optional response data
            
        Returns:
            True if acknowledgment was processed successfully
        """
        try:
            if self._routing_system:
                return await self._routing_system.acknowledge_message(message_id, agent_id, response_data)
            else:
                logger.warning(f"No routing system available for acknowledgment of message {message_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to acknowledge message {message_id}: {e}")
            return False
    
    async def get_message_delivery_status(self, message_id: str) -> Dict[str, Any]:
        """
        Get delivery status for a message
        
        Args:
            message_id: Message ID to check
            
        Returns:
            Dictionary with delivery status information
        """
        try:
            if self._routing_system:
                return await self._routing_system.get_delivery_status(message_id)
            else:
                return {"message_id": message_id, "status": "routing_system_unavailable"}
                
        except Exception as e:
            logger.error(f"Failed to get delivery status for message {message_id}: {e}")
            return {"message_id": message_id, "status": "error", "error": str(e)}
    
    async def _initialize_redis(self) -> bool:
        """Initialize Redis connection with connection pooling"""
        try:
            # Create connection pool
            self._redis_pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            # Create Redis client
            self._redis = redis.Redis(connection_pool=self._redis_pool)
            
            # Test connection
            await self._redis.ping()
            
            logger.info(f"Connected to Redis at {self.redis_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Heartbeat monitoring task
        heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        self._background_tasks.add(heartbeat_task)
        
        # Message cleanup task
        cleanup_task = asyncio.create_task(self._message_cleanup())
        self._background_tasks.add(cleanup_task)
        
        # Performance metrics task
        metrics_task = asyncio.create_task(self._collect_metrics())
        self._background_tasks.add(metrics_task)
        
        logger.info("Started background tasks")
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and handle timeouts"""
        while self._running:
            try:
                current_time = datetime.utcnow()
                timeout_threshold = current_time - timedelta(seconds=self.agent_timeout)
                
                # Check for timed out agents
                timed_out_agents = []
                for agent_id, agent in self._registered_agents.items():
                    if agent.last_heartbeat < timeout_threshold:
                        timed_out_agents.append(agent_id)
                
                # Handle timed out agents
                for agent_id in timed_out_agents:
                    logger.warning(f"Agent {agent_id} timed out, marking as inactive")
                    self._registered_agents[agent_id].status = "inactive"
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _message_cleanup(self):
        """Clean up expired messages and maintain queue sizes"""
        while self._running:
            try:
                # Clean up expired messages from Redis
                if self._redis:
                    # This would be implemented based on specific Redis key patterns
                    # For now, we rely on Redis TTL for automatic cleanup
                    pass
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in message cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _collect_metrics(self):
        """Collect performance metrics"""
        while self._running:
            try:
                # Collect basic metrics
                metrics = {
                    "connected_agents": len(self._registered_agents),
                    "active_agents": len([a for a in self._registered_agents.values() if a.status == "active"]),
                    "total_messages_processed": self._status.total_messages_processed,
                    "redis_connected": self._status.redis_connected,
                    "uptime_seconds": (datetime.utcnow() - self._status.start_time).total_seconds() if self._status.start_time else 0
                }
                
                # Store metrics in Redis for monitoring
                if self._redis:
                    await self._redis.setex(
                        "mcp:server:metrics",
                        300,  # 5 minute TTL
                        json.dumps(metrics)
                    )
                
                self._status.performance_metrics = metrics
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(60)
    
    def _validate_message(self, message: MCPMessage) -> ValidationResult:
        """Validate MCP message format and content"""
        errors = []
        
        # Required fields validation
        if not message.id:
            errors.append("Message ID is required")
        
        if not message.type:
            errors.append("Message type is required")
        
        if not message.source_agent:
            errors.append("Source agent is required")
        
        if not message.target_agents:
            errors.append("Target agents list cannot be empty")
        
        if not isinstance(message.payload, dict):
            errors.append("Payload must be a dictionary")
        
        # Type validation
        if message.type not in [t.value for t in MCPMessageType]:
            errors.append(f"Invalid message type: {message.type}")
        
        # Priority validation
        if message.priority not in [p.value for p in MCPMessagePriority]:
            errors.append(f"Invalid message priority: {message.priority}")
        
        # Target agent validation
        for target_agent in message.target_agents:
            if target_agent not in self._registered_agents:
                errors.append(f"Target agent not registered: {target_agent}")
        
        return ValidationResult(len(errors) == 0, errors)
    
    async def _route_to_agent(self, message: MCPMessage, target_agent: str) -> bool:
        """Route message to a specific agent"""
        try:
            if target_agent not in self._registered_agents:
                logger.error(f"Target agent {target_agent} not registered")
                return False
            
            # Create agent-specific queue key
            queue_key = f"mcp:agent:{target_agent}:messages"
            
            # Serialize message
            message_data = json.dumps(message.to_dict())
            
            # Add to agent's message queue in Redis
            await self._redis.lpush(queue_key, message_data)
            
            # Set TTL on the queue to prevent indefinite growth
            await self._redis.expire(queue_key, self.message_ttl)
            
            # Trim queue to prevent memory issues
            await self._redis.ltrim(queue_key, 0, self.max_queue_size - 1)
            
            # Publish notification to agent's notification channel
            notification_channel = f"mcp:agent:{target_agent}:notifications"
            await self._redis.publish(notification_channel, message.id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to route message to agent {target_agent}: {e}")
            return False
    
    async def _send_error_response(self, original_message: MCPMessage, error_message: str, details: List[str]):
        """Send error response for failed message processing"""
        try:
            error_response = MCPMessage(
                id=str(uuid.uuid4()),
                type=MCPMessageType.ERROR.value,
                source_agent="mcp_server",
                target_agents=[original_message.source_agent],
                payload={
                    "error": error_message,
                    "details": details,
                    "original_message_id": original_message.id
                },
                timestamp=datetime.utcnow(),
                correlation_id=original_message.id
            )
            
            await self.route_message(error_response)
            
        except Exception as e:
            logger.error(f"Failed to send error response: {e}")
    
    async def _store_agent_registration(self, agent: RegisteredAgent):
        """Store agent registration in Redis"""
        try:
            agent_key = f"mcp:agent:{agent.agent_id}:registration"
            agent_data = {
                "agent_id": agent.agent_id,
                "connection_id": agent.connection_id,
                "last_heartbeat": agent.last_heartbeat.isoformat(),
                "status": agent.status,
                "capabilities": [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "message_types": cap.message_types,
                        "parameters": cap.parameters
                    }
                    for cap in agent.capabilities
                ],
                "metadata": agent.metadata
            }
            
            await self._redis.setex(
                agent_key,
                self.agent_timeout * 2,  # Double the timeout for persistence
                json.dumps(agent_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to store agent registration: {e}")
    
    async def _remove_agent_registration(self, agent_id: str):
        """Remove agent registration from Redis"""
        try:
            agent_key = f"mcp:agent:{agent_id}:registration"
            await self._redis.delete(agent_key)
            
        except Exception as e:
            logger.error(f"Failed to remove agent registration: {e}")
    
    async def _cleanup_agent_subscriptions(self, agent_id: str):
        """Clean up subscriptions for an agent"""
        try:
            # Remove agent from all subscription sets
            for topic, agents in self._subscriptions.items():
                agents.discard(agent_id)
            
            # Remove empty subscription sets
            empty_topics = [topic for topic, agents in self._subscriptions.items() if not agents]
            for topic in empty_topics:
                del self._subscriptions[topic]
            
        except Exception as e:
            logger.error(f"Failed to cleanup agent subscriptions: {e}")
    
    async def _initialize_routing_system(self):
        """Initialize the routing system with Redis backend"""
        try:
            from .redis_mcp_backend import RedisMCPBackend
            from .mcp_message_handler import MCPMessageHandler
            from .mcp_routing_system import MCPRoutingSystem, RoutingConfig
            
            # Create Redis backend
            redis_backend = RedisMCPBackend()
            await redis_backend.connect(self.redis_url)
            
            # Create message handler
            message_handler = MCPMessageHandler()
            
            # Create routing system
            routing_config = RoutingConfig()
            self._routing_system = MCPRoutingSystem(redis_backend, message_handler, routing_config)
            
            # Start routing system
            await self._routing_system.start()
            
            logger.info("Routing system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize routing system: {e}")
            # Continue without routing system - server will use fallback routing
    
    async def _route_message_direct(self, message: MCPMessage) -> bool:
        """
        Direct message routing fallback (original implementation)
        
        Args:
            message: MCP message to route
            
        Returns:
            True if message was routed successfully
        """
        try:
            # Validate message
            validation = self._validate_message(message)
            if not validation.is_valid:
                logger.error(f"Message validation failed: {validation.errors}")
                await self._send_error_response(message, "Message validation failed", validation.errors)
                return False
            
            # Check if message has expired
            if message.is_expired():
                logger.warning(f"Message {message.id} has expired, discarding")
                return False
            
            # Route to target agents
            routed_count = 0
            for target_agent in message.target_agents:
                if await self._route_to_agent(message, target_agent):
                    routed_count += 1
            
            # Update metrics
            self._status.total_messages_processed += 1
            
            logger.debug(f"Routed message {message.id} to {routed_count}/{len(message.target_agents)} agents")
            return routed_count > 0
            
        except Exception as e:
            logger.error(f"Failed to route message {message.id}: {e}")
            await self._send_error_response(message, "Message routing failed", [str(e)])
            return False


# Global MCP server instance
mcp_server = MCPServer()

# Import performance integration
try:
    from .mcp_performance_integration import get_enhanced_mcp_server
    # Use enhanced server if available
    enhanced_mcp_server = get_enhanced_mcp_server()
    logger.info("Enhanced MCP server with performance optimizations available")
except ImportError as e:
    logger.warning(f"Performance optimizations not available: {e}")
    enhanced_mcp_server = None