# agents/ai_assistant_agent.py
"""
AI Assistant Agent for handling conversational AI requests through the enhanced MCP system
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from app.ai_assistant_service import ai_assistant
from app.mcp_client import MCPClient, ClientConfig
from app.mcp_models import MCPMessage, MCPMessageType, AgentContext
from app.agent_context_synchronizer import agent_context_synchronizer, AgentContextType, ContextMergeStrategy

logger = logging.getLogger(__name__)


class AIAssistantAgent:
    """
    Enhanced AI Assistant Agent with MCP integration
    
    Provides conversational AI capabilities with context synchronization
    and inter-agent communication through the enhanced MCP system.
    """
    
    def __init__(self):
        self.agent_type = "ai_assistant"
        self.agent_id = "ai_assistant_agent"
        self.service = ai_assistant
        
        # Define agent capabilities for MCP registration
        self.capabilities = [
            {
                "name": "conversational_ai",
                "description": "Process natural language queries and provide AI responses",
                "message_types": ["agent_request", "context_update"],
                "parameters": {
                    "supports_sessions": True,
                    "supports_memory": True,
                    "max_context_length": 4000
                }
            },
            {
                "name": "conversation_management",
                "description": "Manage conversation sessions and history",
                "message_types": ["task_notification", "agent_request"],
                "parameters": {
                    "session_management": True,
                    "history_retrieval": True
                }
            }
        ]
        
        # Initialize MCP client
        client_config = ClientConfig(
            agent_id=self.agent_id,
            capabilities=self.capabilities,
            heartbeat_interval=30,
            auto_reconnect=True
        )
        self.mcp_client = MCPClient(client_config)
        
        # Register message handlers
        self._register_message_handlers()
        
        # Register context handlers with the agent context synchronizer
        self._register_context_handlers()
        
        logger.info("Enhanced AI Assistant Agent initialized")
    
    async def start(self) -> bool:
        """
        Start the AI Assistant Agent and connect to MCP
        
        Returns:
            True if started successfully
        """
        try:
            # Connect to MCP server
            if not await self.mcp_client.connect():
                logger.error("Failed to connect to MCP server")
                return False
            
            # Start agent context synchronizer if not already started
            try:
                await agent_context_synchronizer.start()
            except Exception as e:
                logger.warning(f"Agent context synchronizer already started or failed: {e}")
            
            logger.info("AI Assistant Agent started and connected to MCP")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start AI Assistant Agent: {e}")
            return False
    
    async def stop(self):
        """Stop the AI Assistant Agent"""
        try:
            await self.mcp_client.disconnect()
            logger.info("AI Assistant Agent stopped")
        except Exception as e:
            logger.error(f"Error stopping AI Assistant Agent: {e}")
    
    async def process_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process AI assistant request with enhanced context awareness
        
        Args:
            payload: Request payload containing query and session info
            
        Returns:
            Response dictionary with AI assistant result
        """
        try:
            # Extract request parameters
            query = payload.get("query", payload.get("text", ""))
            session_id = payload.get("session_id")
            user_id = payload.get("user_id")
            temperature = payload.get("temperature")
            max_tokens = payload.get("max_tokens")
            
            if not query:
                return {
                    "error": "No query provided",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            logger.info(f"Processing AI assistant request: {len(query)} chars, session={session_id}")
            
            # Get shared context from other agents if available
            await self._update_context_from_shared_sources(session_id, user_id)
            
            # Process query through AI assistant service
            result = await self.service.process_query(
                query=query,
                session_id=session_id,
                user_id=user_id,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Add agent metadata
            result["agent_type"] = self.agent_type
            result["processed_at"] = datetime.utcnow().isoformat()
            
            # Update conversation context and broadcast to other agents
            await self._update_conversation_context(session_id, query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing AI assistant request: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_conversation_history(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get conversation history for a session"""
        try:
            session_id = payload.get("session_id")
            limit = payload.get("limit", 50)
            
            if not session_id:
                return {
                    "error": "No session_id provided",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            result = await self.service.get_conversation_history(session_id, limit)
            result["agent_type"] = self.agent_type
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def clear_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Clear a conversation session"""
        try:
            session_id = payload.get("session_id")
            
            if not session_id:
                return {
                    "error": "No session_id provided",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            result = await self.service.clear_session(session_id)
            result["agent_type"] = self.agent_type
            
            return result
            
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_user_sessions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get user sessions"""
        try:
            user_id = payload.get("user_id")
            limit = payload.get("limit", 10)
            
            if not user_id:
                return {
                    "error": "No user_id provided",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            result = await self.service.get_user_sessions(user_id, limit)
            result["agent_type"] = self.agent_type
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _register_message_handlers(self):
        """Register MCP message handlers"""
        # Register handler for agent requests
        self.mcp_client.register_message_handler(
            MCPMessageType.AGENT_REQUEST.value,
            self._handle_agent_request
        )
        
        # Register handler for task notifications
        self.mcp_client.register_message_handler(
            MCPMessageType.TASK_NOTIFICATION.value,
            self._handle_task_notification
        )
        
        # Register context handlers
        self.mcp_client.register_context_handler(
            "conversation",
            self._handle_conversation_context
        )
        
        self.mcp_client.register_context_handler(
            "user_session",
            self._handle_user_session_context
        )
    
    def _register_context_handlers(self):
        """Register context handlers with the agent context synchronizer"""
        # Register conversation context handler
        agent_context_synchronizer.register_agent_context_handler(
            agent_id=self.agent_id,
            context_type=AgentContextType.CONVERSATION.value,
            handler=self._handle_conversation_context_sync,
            priority=5,
            auto_resolve_conflicts=True,
            merge_strategy=ContextMergeStrategy.MERGE_RECURSIVE
        )
        
        # Register user session context handler
        agent_context_synchronizer.register_agent_context_handler(
            agent_id=self.agent_id,
            context_type=AgentContextType.USER_SESSION.value,
            handler=self._handle_user_session_context_sync,
            priority=3,
            auto_resolve_conflicts=True,
            merge_strategy=ContextMergeStrategy.LATEST_WINS
        )
        
        # Register task state context handler
        agent_context_synchronizer.register_agent_context_handler(
            agent_id=self.agent_id,
            context_type=AgentContextType.TASK_STATE.value,
            handler=self._handle_task_state_context_sync,
            priority=4,
            auto_resolve_conflicts=True,
            merge_strategy=ContextMergeStrategy.FIELD_LEVEL_MERGE
        )
    
    async def _handle_agent_request(self, message: MCPMessage):
        """Handle incoming agent requests"""
        try:
            payload = message.payload
            request_type = payload.get("request_type")
            
            if request_type == "process_query":
                # Process AI assistant query
                result = await self.process_request(payload)
                
                # Send response back
                response = message.create_response(result, self.agent_id)
                await self.mcp_client.send_message(response)
                
            elif request_type == "get_conversation_history":
                result = await self.get_conversation_history(payload)
                response = message.create_response(result, self.agent_id)
                await self.mcp_client.send_message(response)
                
            elif request_type == "clear_session":
                result = await self.clear_session(payload)
                response = message.create_response(result, self.agent_id)
                await self.mcp_client.send_message(response)
                
            elif request_type == "get_user_sessions":
                result = await self.get_user_sessions(payload)
                response = message.create_response(result, self.agent_id)
                await self.mcp_client.send_message(response)
                
            else:
                logger.warning(f"Unknown request type: {request_type}")
                error_response = message.create_error_response(
                    ValueError(f"Unknown request type: {request_type}"),
                    self.agent_id
                )
                await self.mcp_client.send_message(error_response)
                
        except Exception as e:
            logger.error(f"Error handling agent request: {e}")
            error_response = message.create_error_response(e, self.agent_id)
            await self.mcp_client.send_message(error_response)
    
    async def _handle_task_notification(self, message: MCPMessage):
        """Handle task notifications from other agents"""
        try:
            payload = message.payload
            task_id = payload.get("task_id")
            action = payload.get("action")
            task_data = payload.get("task_data", {})
            
            logger.info(f"Received task notification: {task_id}:{action}")
            
            # Update local context with task information
            await self.mcp_client.update_local_context(
                "active_tasks",
                {task_id: {"action": action, "data": task_data, "timestamp": datetime.utcnow().isoformat()}},
                broadcast=False  # Don't re-broadcast task notifications
            )
            
        except Exception as e:
            logger.error(f"Error handling task notification: {e}")
    
    async def _handle_conversation_context(self, context: AgentContext):
        """Handle conversation context updates from other agents"""
        try:
            logger.info(f"Received conversation context update from {context.agent_id}")
            
            # Process conversation context to enhance AI responses
            conversation_data = context.context_data
            
            # Update local conversation understanding
            # This could be used to maintain conversation state across agents
            
        except Exception as e:
            logger.error(f"Error handling conversation context: {e}")
    
    async def _handle_user_session_context(self, context: AgentContext):
        """Handle user session context updates"""
        try:
            logger.info(f"Received user session context update from {context.agent_id}")
            
            # Update local understanding of user sessions
            session_data = context.context_data
            
            # This could be used to maintain user preferences and history
            
        except Exception as e:
            logger.error(f"Error handling user session context: {e}")
    
    async def _update_context_from_shared_sources(self, session_id: str, user_id: str):
        """Update context from shared sources before processing request"""
        try:
            # Get shared conversation context
            conversation_context = await self.mcp_client.get_shared_context("conversation")
            if conversation_context and session_id:
                # Use conversation context to enhance understanding
                pass
            
            # Get shared user session context
            user_context = await self.mcp_client.get_shared_context("user_session")
            if user_context and user_id:
                # Use user context for personalization
                pass
                
        except Exception as e:
            logger.error(f"Error updating context from shared sources: {e}")
    
    async def _update_conversation_context(self, session_id: str, query: str, result: Dict[str, Any]):
        """Update and broadcast conversation context"""
        try:
            if not session_id:
                return
            
            # Create conversation context
            conversation_data = {
                "session_id": session_id,
                "last_query": query,
                "last_response": result.get("response", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }
            
            # Update local context and broadcast
            await self.mcp_client.update_local_context(
                "conversation",
                conversation_data,
                broadcast=True
            )
            
        except Exception as e:
            logger.error(f"Error updating conversation context: {e}")
    
    async def _handle_conversation_context_sync(self, context: AgentContext):
        """Handle conversation context synchronization from other agents"""
        try:
            logger.info(f"Syncing conversation context from {context.agent_id}")
            
            conversation_data = context.context_data
            session_id = conversation_data.get("session_id")
            
            if session_id:
                # Update local conversation understanding
                # This could be used to maintain conversation continuity across agents
                logger.debug(f"Updated conversation context for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error handling conversation context sync: {e}")
    
    async def _handle_user_session_context_sync(self, context: AgentContext):
        """Handle user session context synchronization"""
        try:
            logger.info(f"Syncing user session context from {context.agent_id}")
            
            session_data = context.context_data
            user_id = session_data.get("user_id")
            
            if user_id:
                # Update local user session understanding
                # This could be used for personalization and user preferences
                logger.debug(f"Updated user session context for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error handling user session context sync: {e}")
    
    async def _handle_task_state_context_sync(self, context: AgentContext):
        """Handle task state context synchronization"""
        try:
            logger.info(f"Syncing task state context from {context.agent_id}")
            
            task_data = context.context_data
            active_tasks = task_data.get("active_tasks", {})
            
            # Update local understanding of active tasks
            # This helps coordinate with other agents working on tasks
            logger.debug(f"Updated task state context with {len(active_tasks)} active tasks")
            
        except Exception as e:
            logger.error(f"Error handling task state context sync: {e}")
    
    async def sync_conversation_context(self, session_id: str, query: str, response: str):
        """Synchronize conversation context with other agents"""
        try:
            context_data = {
                "session_id": session_id,
                "last_query": query,
                "last_response": response,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }
            
            context = AgentContext(
                agent_id=self.agent_id,
                context_type=AgentContextType.CONVERSATION.value,
                context_data=context_data,
                access_level="public"
            )
            
            # Synchronize with other agents
            result = await agent_context_synchronizer.sync_agent_context(
                self.agent_id, context
            )
            
            if result.success:
                logger.debug(f"Successfully synchronized conversation context for session {session_id}")
            else:
                logger.warning(f"Failed to synchronize conversation context: {result.errors}")
            
        except Exception as e:
            logger.error(f"Error synchronizing conversation context: {e}")


# Global AI Assistant Agent instance
ai_assistant_agent = AIAssistantAgent()