# app/mcp_integration.py
"""
MCP Integration Module

This module integrates all MCP components and provides a unified interface
for the enhanced MCP server infrastructure.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from .mcp_server import MCPServer, MCPMessage, MCPMessageType, RegisteredAgent
from .mcp_message_handler import MCPMessageHandler, MessageRoutingRule
from .redis_mcp_backend import RedisMCPBackend, RedisConfig, MessageQueueConfig
from .mcp_error_handler import MCPErrorHandler, ErrorContext, ErrorCategory, ErrorSeverity
from .config import settings

logger = logging.getLogger(__name__)


class MCPIntegration:
    """
    Unified MCP integration that coordinates all MCP components
    
    Provides a single interface for:
    - MCP server management
    - Message handling and routing
    - Redis backend operations
    - Error handling and recovery
    """
    
    def __init__(self):
        # Initialize components
        self.server = MCPServer(redis_url=settings.REDIS_URL)
        self.message_handler = MCPMessageHandler()
        self.redis_backend = RedisMCPBackend()
        self.error_handler = MCPErrorHandler()
        
        # Integration state
        self._initialized = False
        self._running = False
        
        # Component integration
        self._setup_component_integration()
        
        logger.info("MCP Integration initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize all MCP components
        
        Returns:
            True if initialization successful
        """
        try:
            if self._initialized:
                logger.warning("MCP Integration already initialized")
                return True
            
            # Initialize Redis backend first
            if not await self.redis_backend.connect():
                logger.error("Failed to initialize Redis backend")
                return False
            
            # Initialize MCP server
            if not await self.server.start_server():
                logger.error("Failed to initialize MCP server")
                return False
            
            # Setup message routing integration
            await self._setup_message_routing()
            
            # Setup error handling integration
            await self._setup_error_handling()
            
            self._initialized = True
            self._running = True
            
            logger.info("MCP Integration initialized successfully")
            return True
            
        except Exception as e:
            error_context = ErrorContext(
                component="mcp_integration",
                operation="initialize"
            )
            await self.error_handler.handle_error(e, error_context)
            return False
    
    async def shutdown(self):
        """Shutdown all MCP components"""
        try:
            self._running = False
            
            # Shutdown components in reverse order
            await self.server.stop_server()
            await self.redis_backend.disconnect()
            
            self._initialized = False
            
            logger.info("MCP Integration shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during MCP Integration shutdown: {e}")
    
    async def register_agent(self, agent_id: str, capabilities: List[Dict[str, Any]]) -> Optional[str]:
        """
        Register an agent with comprehensive error handling
        
        Args:
            agent_id: Unique agent identifier
            capabilities: List of agent capabilities
            
        Returns:
            Connection ID if successful, None otherwise
        """
        try:
            if not self._initialized:
                raise RuntimeError("MCP Integration not initialized")
            
            connection_id = await self.server.register_agent(agent_id, capabilities)
            
            logger.info(f"Successfully registered agent {agent_id}")
            return connection_id
            
        except Exception as e:
            error_context = ErrorContext(
                component="mcp_integration",
                operation="register_agent",
                agent_id=agent_id
            )
            await self.error_handler.handle_error(e, error_context)
            return None
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent with error handling
        
        Args:
            agent_id: Agent identifier to unregister
            
        Returns:
            True if successful
        """
        try:
            if not self._initialized:
                raise RuntimeError("MCP Integration not initialized")
            
            success = await self.server.unregister_agent(agent_id)
            
            if success:
                logger.info(f"Successfully unregistered agent {agent_id}")
            
            return success
            
        except Exception as e:
            error_context = ErrorContext(
                component="mcp_integration",
                operation="unregister_agent",
                agent_id=agent_id
            )
            await self.error_handler.handle_error(e, error_context)
            return False
    
    async def send_message(self, message: MCPMessage) -> bool:
        """
        Send a message through the MCP system with full processing
        
        Args:
            message: MCP message to send
            
        Returns:
            True if message was sent successfully
        """
        try:
            if not self._initialized:
                raise RuntimeError("MCP Integration not initialized")
            
            # Validate message
            validation = self.message_handler.validate_message(message)
            if not validation.is_valid:
                raise ValueError(f"Message validation failed: {validation.errors}")
            
            # Route message
            target_agents = self.message_handler.route_message(message)
            message.target_agents = target_agents
            
            # Send through server
            success = await self.server.route_message(message)
            
            if success:
                logger.debug(f"Successfully sent message {message.id}")
            
            return success
            
        except Exception as e:
            error_context = ErrorContext(
                component="mcp_integration",
                operation="send_message",
                message_id=message.id if hasattr(message, 'id') else None
            )
            await self.error_handler.handle_error(e, error_context)
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
            if not self._initialized:
                raise RuntimeError("MCP Integration not initialized")
            
            # Validate message
            validation = self.message_handler.validate_message(message)
            if not validation.is_valid:
                raise ValueError(f"Message validation failed: {validation.errors}")
            
            # Broadcast through server
            count = await self.server.broadcast_message(message, target_type)
            
            logger.info(f"Broadcast message {message.id} to {count} agents")
            return count
            
        except Exception as e:
            error_context = ErrorContext(
                component="mcp_integration",
                operation="broadcast_message",
                message_id=message.id if hasattr(message, 'id') else None
            )
            await self.error_handler.handle_error(e, error_context)
            return 0
    
    async def subscribe_to_messages(self, topic: str, callback: Callable) -> Optional[str]:
        """
        Subscribe to messages on a topic
        
        Args:
            topic: Topic to subscribe to
            callback: Callback function for received messages
            
        Returns:
            Subscription ID if successful
        """
        try:
            if not self._initialized:
                raise RuntimeError("MCP Integration not initialized")
            
            subscription_id = await self.redis_backend.subscribe_to_topic(topic, callback)
            
            logger.info(f"Subscribed to topic {topic}")
            return subscription_id
            
        except Exception as e:
            error_context = ErrorContext(
                component="mcp_integration",
                operation="subscribe_to_messages",
                topic=topic
            )
            await self.error_handler.handle_error(e, error_context)
            return None
    
    async def unsubscribe_from_messages(self, subscription_id: str) -> bool:
        """
        Unsubscribe from messages
        
        Args:
            subscription_id: Subscription ID to remove
            
        Returns:
            True if successful
        """
        try:
            if not self._initialized:
                raise RuntimeError("MCP Integration not initialized")
            
            success = await self.redis_backend.unsubscribe_from_topic(subscription_id)
            
            if success:
                logger.info(f"Unsubscribed from {subscription_id}")
            
            return success
            
        except Exception as e:
            error_context = ErrorContext(
                component="mcp_integration",
                operation="unsubscribe_from_messages"
            )
            await self.error_handler.handle_error(e, error_context)
            return False
    
    def add_message_routing_rule(self, rule: MessageRoutingRule):
        """Add a message routing rule"""
        self.message_handler.add_routing_rule(rule)
    
    def register_message_type_handler(self, message_type: str, handler: Callable):
        """Register a handler for specific message types"""
        self.message_handler.register_type_handler(message_type, handler)
        self.server.register_message_handler(message_type, handler)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            server_status = await self.server.get_server_status()
            redis_stats = await self.redis_backend.get_queue_stats()
            message_stats = self.message_handler.get_statistics()
            error_stats = self.error_handler.get_error_statistics()
            
            return {
                "initialized": self._initialized,
                "running": self._running,
                "timestamp": datetime.utcnow().isoformat(),
                "server_status": {
                    "is_running": server_status.is_running,
                    "connected_agents": server_status.connected_agents,
                    "total_messages_processed": server_status.total_messages_processed,
                    "redis_connected": server_status.redis_connected,
                    "uptime_seconds": (
                        (datetime.utcnow() - server_status.start_time).total_seconds()
                        if server_status.start_time else 0
                    )
                },
                "redis_stats": redis_stats,
                "message_stats": message_stats,
                "error_stats": error_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "initialized": self._initialized,
                "running": self._running,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_health_check(self) -> Dict[str, Any]:
        """Get system health check"""
        try:
            health = {
                "healthy": True,
                "components": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check MCP server
            try:
                server_status = await self.server.get_server_status()
                health["components"]["mcp_server"] = {
                    "healthy": server_status.is_running,
                    "details": {
                        "connected_agents": server_status.connected_agents,
                        "messages_processed": server_status.total_messages_processed
                    }
                }
                if not server_status.is_running:
                    health["healthy"] = False
            except Exception as e:
                health["components"]["mcp_server"] = {
                    "healthy": False,
                    "error": str(e)
                }
                health["healthy"] = False
            
            # Check Redis backend
            try:
                redis_stats = await self.redis_backend.get_queue_stats()
                redis_healthy = "error" not in redis_stats
                health["components"]["redis_backend"] = {
                    "healthy": redis_healthy,
                    "details": redis_stats.get("mcp_stats", {})
                }
                if not redis_healthy:
                    health["healthy"] = False
            except Exception as e:
                health["components"]["redis_backend"] = {
                    "healthy": False,
                    "error": str(e)
                }
                health["healthy"] = False
            
            # Check error rates
            error_stats = self.error_handler.get_error_statistics()
            error_rate = error_stats.get("error_rate_per_hour", 0)
            health["components"]["error_handler"] = {
                "healthy": error_rate < 100,  # Threshold
                "details": {
                    "error_rate_per_hour": error_rate,
                    "unresolved_errors": error_stats.get("unresolved_errors", 0)
                }
            }
            if error_rate >= 100:
                health["healthy"] = False
            
            return health
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _setup_component_integration(self):
        """Setup integration between components"""
        # Register error recovery handlers
        from .mcp_error_handler import RecoveryAction
        
        async def reconnect_handler(error):
            """Handler for reconnection recovery"""
            try:
                return await self.redis_backend.connect()
            except Exception:
                return False
        
        async def retry_handler(error):
            """Handler for retry recovery"""
            try:
                # Implement retry logic based on error context
                return True
            except Exception:
                return False
        
        self.error_handler.register_recovery_handler(RecoveryAction.RECONNECT, reconnect_handler)
        self.error_handler.register_recovery_handler(RecoveryAction.RETRY, retry_handler)
    
    async def _setup_message_routing(self):
        """Setup message routing integration"""
        # Add default routing rules
        from .mcp_message_handler import MessageRoutingRule
        
        # Route context updates to all agents
        context_rule = MessageRoutingRule(
            message_type=MCPMessageType.CONTEXT_UPDATE.value,
            route_to=["*"]  # Broadcast to all
        )
        self.add_message_routing_rule(context_rule)
        
        # Route task notifications to task management agents
        task_rule = MessageRoutingRule(
            message_type=MCPMessageType.TASK_NOTIFICATION.value,
            target_pattern="task_*"
        )
        self.add_message_routing_rule(task_rule)
    
    async def _setup_error_handling(self):
        """Setup error handling integration"""
        # Register custom error handlers for MCP-specific errors
        pass


# Global MCP integration instance
mcp_integration = MCPIntegration()