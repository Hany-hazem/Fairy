# app/mcp_client.py
"""
Enhanced MCP Client for Agent Communication

This module provides a comprehensive MCP client that agents can use to
communicate through the enhanced MCP server infrastructure.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass

from .mcp_models import (
    MCPMessage, MCPMessageType, MCPMessagePriority, AgentContext,
    create_context_update_message, create_task_notification_message,
    create_agent_request_message, create_heartbeat_message
)
from .redis_mcp_backend import RedisMCPBackend
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """MCP client configuration"""
    agent_id: str
    capabilities: List[Dict[str, Any]]
    redis_url: str = None
    heartbeat_interval: int = 30
    message_timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    auto_reconnect: bool = True


class MCPClient:
    """
    Enhanced MCP Client for agent communication
    
    Provides comprehensive MCP client functionality including:
    - Connection management with auto-reconnect
    - Message sending and receiving
    - Context synchronization
    - Heartbeat management
    - Error handling and recovery
    """
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.capabilities = config.capabilities
        
        # Connection state
        self._connected = False
        self._connection_id: Optional[str] = None
        self._redis_backend: Optional[RedisMCPBackend] = None
        
        # Message handling
        self._message_handlers: Dict[str, Callable] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._subscriptions: Set[str] = set()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False
        
        # Context management
        self._local_context: Dict[str, AgentContext] = {}
        self._context_handlers: Dict[str, Callable] = {}
        
        logger.info(f"MCP Client initialized for agent {self.agent_id}")
    
    async def connect(self) -> bool:
        """
        Connect to the MCP server
        
        Returns:
            True if connection successful
        """
        try:
            if self._connected:
                logger.warning(f"Agent {self.agent_id} already connected")
                return True
            
            # Initialize Redis backend
            self._redis_backend = RedisMCPBackend()
            redis_url = self.config.redis_url or settings.REDIS_URL
            
            if not await self._redis_backend.connect(redis_url):
                logger.error(f"Failed to connect to Redis for agent {self.agent_id}")
                return False
            
            # Register with MCP server
            self._connection_id = await self._register_with_server()
            if not self._connection_id:
                logger.error(f"Failed to register agent {self.agent_id} with MCP server")
                return False
            
            # Start message processing
            await self._start_message_processing()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._connected = True
            self._running = True
            
            logger.info(f"Agent {self.agent_id} connected with connection ID {self._connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect agent {self.agent_id}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        try:
            self._running = False
            
            # Stop background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Unregister from server
            if self._connection_id:
                await self._unregister_from_server()
            
            # Close Redis connection
            if self._redis_backend:
                await self._redis_backend.disconnect()
            
            self._connected = False
            self._connection_id = None
            
            logger.info(f"Agent {self.agent_id} disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting agent {self.agent_id}: {e}")
    
    async def send_message(self, message: MCPMessage, wait_for_response: bool = False, timeout: int = None) -> Optional[MCPMessage]:
        """
        Send a message through MCP
        
        Args:
            message: MCP message to send
            wait_for_response: Whether to wait for a response
            timeout: Response timeout in seconds
            
        Returns:
            Response message if wait_for_response is True
        """
        try:
            if not self._connected:
                raise RuntimeError(f"Agent {self.agent_id} not connected to MCP server")
            
            # Set source agent
            message.source_agent = self.agent_id
            
            # Handle response waiting
            response_future = None
            if wait_for_response:
                response_future = asyncio.Future()
                self._pending_responses[message.id] = response_future
            
            # Send message through Redis backend
            success = await self._redis_backend.publish_message("mcp:server:messages", message)
            
            if not success:
                if wait_for_response and message.id in self._pending_responses:
                    del self._pending_responses[message.id]
                raise RuntimeError("Failed to send message")
            
            logger.debug(f"Sent message {message.id} from agent {self.agent_id}")
            
            # Wait for response if requested
            if wait_for_response and response_future:
                try:
                    timeout = timeout or self.config.message_timeout
                    response = await asyncio.wait_for(response_future, timeout=timeout)
                    return response
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for response to message {message.id}")
                    if message.id in self._pending_responses:
                        del self._pending_responses[message.id]
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to send message from agent {self.agent_id}: {e}")
            if wait_for_response and message.id in self._pending_responses:
                del self._pending_responses[message.id]
            raise
    
    async def broadcast_context_update(self, context: AgentContext, target_agents: List[str] = None) -> bool:
        """
        Broadcast a context update to other agents
        
        Args:
            context: Agent context to broadcast
            target_agents: Specific agents to target (None for all)
            
        Returns:
            True if broadcast successful
        """
        try:
            # Update local context
            self._local_context[context.context_type] = context
            
            # Determine target agents
            if target_agents is None:
                target_agents = ["*"]  # Broadcast to all
            
            # Create context update message
            message = create_context_update_message(
                source_agent=self.agent_id,
                target_agents=target_agents,
                context=context,
                priority=MCPMessagePriority.NORMAL.value
            )
            
            # Send message
            await self.send_message(message)
            
            logger.info(f"Broadcast context update {context.context_type} from agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to broadcast context update from agent {self.agent_id}: {e}")
            return False
    
    async def send_task_notification(self, task_id: str, action: str, task_data: Dict[str, Any] = None, target_agents: List[str] = None) -> bool:
        """
        Send a task notification to other agents
        
        Args:
            task_id: Task identifier
            action: Task action (started, completed, failed, etc.)
            task_data: Additional task data
            target_agents: Specific agents to notify
            
        Returns:
            True if notification sent successfully
        """
        try:
            if target_agents is None:
                target_agents = ["*"]  # Notify all agents
            
            message = create_task_notification_message(
                source_agent=self.agent_id,
                target_agents=target_agents,
                task_id=task_id,
                action=action,
                task_data=task_data,
                priority=MCPMessagePriority.HIGH.value
            )
            
            await self.send_message(message)
            
            logger.info(f"Sent task notification {task_id}:{action} from agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send task notification from agent {self.agent_id}: {e}")
            return False
    
    async def request_agent_action(self, target_agents: List[str], request_type: str, request_data: Dict[str, Any] = None, wait_for_response: bool = True) -> Optional[MCPMessage]:
        """
        Request an action from other agents
        
        Args:
            target_agents: Agents to send request to
            request_type: Type of request
            request_data: Request data
            wait_for_response: Whether to wait for response
            
        Returns:
            Response message if wait_for_response is True
        """
        try:
            message = create_agent_request_message(
                source_agent=self.agent_id,
                target_agents=target_agents,
                request_type=request_type,
                request_data=request_data,
                priority=MCPMessagePriority.NORMAL.value
            )
            
            response = await self.send_message(message, wait_for_response=wait_for_response)
            
            logger.info(f"Sent agent request {request_type} from agent {self.agent_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to send agent request from agent {self.agent_id}: {e}")
            return None
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """
        Register a handler for specific message types
        
        Args:
            message_type: Message type to handle
            handler: Handler function
        """
        self._message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type {message_type} in agent {self.agent_id}")
    
    def register_context_handler(self, context_type: str, handler: Callable):
        """
        Register a handler for specific context types
        
        Args:
            context_type: Context type to handle
            handler: Handler function
        """
        self._context_handlers[context_type] = handler
        logger.info(f"Registered context handler for {context_type} in agent {self.agent_id}")
    
    async def get_shared_context(self, context_type: str, agent_id: str = None) -> Optional[AgentContext]:
        """
        Get shared context from another agent
        
        Args:
            context_type: Type of context to retrieve
            agent_id: Specific agent to get context from
            
        Returns:
            Agent context if available
        """
        try:
            # First check local context
            if context_type in self._local_context:
                return self._local_context[context_type]
            
            # Request context from other agents
            request_data = {
                "context_type": context_type,
                "requesting_agent": self.agent_id
            }
            
            target_agents = [agent_id] if agent_id else ["*"]
            
            response = await self.request_agent_action(
                target_agents=target_agents,
                request_type="get_context",
                request_data=request_data,
                wait_for_response=True
            )
            
            if response and "context_data" in response.payload:
                context_data = response.payload["context_data"]
                context = AgentContext(**context_data)
                
                # Cache locally if accessible
                if context.can_access(self.agent_id):
                    self._local_context[context_type] = context
                
                return context
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get shared context {context_type} for agent {self.agent_id}: {e}")
            return None
    
    async def update_local_context(self, context_type: str, context_data: Dict[str, Any], broadcast: bool = True):
        """
        Update local context and optionally broadcast to other agents
        
        Args:
            context_type: Type of context to update
            context_data: New context data
            broadcast: Whether to broadcast update to other agents
        """
        try:
            # Create or update context
            if context_type in self._local_context:
                context = self._local_context[context_type]
                context.update_context(context_data)
            else:
                context = AgentContext(
                    agent_id=self.agent_id,
                    context_type=context_type,
                    context_data=context_data
                )
                self._local_context[context_type] = context
            
            # Broadcast update if requested
            if broadcast:
                await self.broadcast_context_update(context)
            
            logger.debug(f"Updated local context {context_type} for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to update local context for agent {self.agent_id}: {e}")
    
    async def _register_with_server(self) -> Optional[str]:
        """Register agent with MCP server"""
        try:
            # Create registration message
            registration_data = {
                "agent_id": self.agent_id,
                "capabilities": self.capabilities,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send registration through Redis
            success = await self._redis_backend.publish_message(
                "mcp:server:registrations",
                MCPMessage(
                    type="agent_registration",
                    source_agent=self.agent_id,
                    target_agents=["mcp_server"],
                    payload=registration_data
                )
            )
            
            if success:
                # For now, generate a connection ID
                # In a full implementation, this would come from the server response
                return str(uuid.uuid4())
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to register agent {self.agent_id}: {e}")
            return None
    
    async def _unregister_from_server(self):
        """Unregister agent from MCP server"""
        try:
            unregistration_data = {
                "agent_id": self.agent_id,
                "connection_id": self._connection_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._redis_backend.publish_message(
                "mcp:server:unregistrations",
                MCPMessage(
                    type="agent_unregistration",
                    source_agent=self.agent_id,
                    target_agents=["mcp_server"],
                    payload=unregistration_data
                )
            )
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {self.agent_id}: {e}")
    
    async def _start_message_processing(self):
        """Start processing incoming messages"""
        try:
            # Subscribe to agent-specific message queue
            agent_topic = f"mcp:agent:{self.agent_id}:messages"
            
            subscription_id = await self._redis_backend.subscribe_to_topic(
                agent_topic,
                self._handle_incoming_message
            )
            
            if subscription_id:
                self._subscriptions.add(subscription_id)
                logger.info(f"Started message processing for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to start message processing for agent {self.agent_id}: {e}")
    
    async def _handle_incoming_message(self, message_data: Dict[str, Any]):
        """Handle incoming MCP message"""
        try:
            # Parse message
            message = MCPMessage(**message_data)
            
            logger.debug(f"Received message {message.id} for agent {self.agent_id}")
            
            # Handle response messages
            if message.correlation_id and message.correlation_id in self._pending_responses:
                future = self._pending_responses.pop(message.correlation_id)
                if not future.done():
                    future.set_result(message)
                return
            
            # Handle context updates
            if message.type == MCPMessageType.CONTEXT_UPDATE.value:
                await self._handle_context_update(message)
                return
            
            # Handle task notifications
            if message.type == MCPMessageType.TASK_NOTIFICATION.value:
                await self._handle_task_notification(message)
                return
            
            # Handle agent requests
            if message.type == MCPMessageType.AGENT_REQUEST.value:
                await self._handle_agent_request(message)
                return
            
            # Handle with registered handlers
            if message.type in self._message_handlers:
                handler = self._message_handlers[message.type]
                await handler(message)
                return
            
            logger.warning(f"No handler for message type {message.type} in agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error handling incoming message for agent {self.agent_id}: {e}")
    
    async def _handle_context_update(self, message: MCPMessage):
        """Handle context update message"""
        try:
            payload = message.payload
            context_type = payload.get("context_type")
            context_data = payload.get("context_data")
            
            if not context_type or not context_data:
                logger.warning(f"Invalid context update message {message.id}")
                return
            
            # Create context object
            context = AgentContext(
                agent_id=message.source_agent,
                context_type=context_type,
                context_data=context_data,
                version=payload.get("context_version", str(uuid.uuid4())),
                access_level=payload.get("access_level", "public"),
                metadata=payload.get("metadata", {})
            )
            
            # Check access permissions
            if not context.can_access(self.agent_id):
                logger.warning(f"Access denied for context {context_type} from agent {message.source_agent}")
                return
            
            # Update local context if we have access
            self._local_context[context_type] = context
            
            # Call registered context handler
            if context_type in self._context_handlers:
                handler = self._context_handlers[context_type]
                await handler(context)
            
            logger.debug(f"Processed context update {context_type} for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error handling context update for agent {self.agent_id}: {e}")
    
    async def _handle_task_notification(self, message: MCPMessage):
        """Handle task notification message"""
        try:
            payload = message.payload
            task_id = payload.get("task_id")
            action = payload.get("action")
            
            if not task_id or not action:
                logger.warning(f"Invalid task notification message {message.id}")
                return
            
            # Call registered handler if available
            if "task_notification" in self._message_handlers:
                handler = self._message_handlers["task_notification"]
                await handler(message)
            
            logger.debug(f"Processed task notification {task_id}:{action} for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error handling task notification for agent {self.agent_id}: {e}")
    
    async def _handle_agent_request(self, message: MCPMessage):
        """Handle agent request message"""
        try:
            payload = message.payload
            request_type = payload.get("request_type")
            
            if not request_type:
                logger.warning(f"Invalid agent request message {message.id}")
                return
            
            # Handle context requests
            if request_type == "get_context":
                await self._handle_context_request(message)
                return
            
            # Call registered handler if available
            if "agent_request" in self._message_handlers:
                handler = self._message_handlers["agent_request"]
                await handler(message)
            
            logger.debug(f"Processed agent request {request_type} for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error handling agent request for agent {self.agent_id}: {e}")
    
    async def _handle_context_request(self, message: MCPMessage):
        """Handle context request from another agent"""
        try:
            payload = message.payload
            context_type = payload.get("context_type")
            requesting_agent = payload.get("requesting_agent")
            
            if not context_type or not requesting_agent:
                logger.warning(f"Invalid context request message {message.id}")
                return
            
            # Check if we have the requested context
            if context_type not in self._local_context:
                # Send empty response
                response = message.create_response(
                    {"context_data": None, "error": "Context not found"},
                    self.agent_id
                )
                await self.send_message(response)
                return
            
            context = self._local_context[context_type]
            
            # Check access permissions
            if not context.can_access(requesting_agent):
                response = message.create_response(
                    {"context_data": None, "error": "Access denied"},
                    self.agent_id
                )
                await self.send_message(response)
                return
            
            # Send context data
            response = message.create_response(
                {"context_data": context.dict()},
                self.agent_id
            )
            await self.send_message(response)
            
            logger.debug(f"Sent context {context_type} to agent {requesting_agent}")
            
        except Exception as e:
            logger.error(f"Error handling context request for agent {self.agent_id}: {e}")
    
    async def _start_background_tasks(self):
        """Start background tasks"""
        # Heartbeat task
        if self.config.heartbeat_interval > 0:
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._background_tasks.add(heartbeat_task)
        
        # Reconnection task
        if self.config.auto_reconnect:
            reconnect_task = asyncio.create_task(self._reconnection_monitor())
            self._background_tasks.add(reconnect_task)
        
        logger.info(f"Started background tasks for agent {self.agent_id}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self._running:
            try:
                if self._connected:
                    heartbeat = create_heartbeat_message(
                        source_agent=self.agent_id,
                        agent_status="active",
                        status_data={
                            "connection_id": self._connection_id,
                            "local_contexts": list(self._local_context.keys()),
                            "message_handlers": list(self._message_handlers.keys())
                        }
                    )
                    
                    await self.send_message(heartbeat)
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop for agent {self.agent_id}: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _reconnection_monitor(self):
        """Monitor connection and attempt reconnection if needed"""
        while self._running:
            try:
                if not self._connected and self.config.auto_reconnect:
                    logger.info(f"Attempting to reconnect agent {self.agent_id}")
                    await self.connect()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in reconnection monitor for agent {self.agent_id}: {e}")
                await asyncio.sleep(30)