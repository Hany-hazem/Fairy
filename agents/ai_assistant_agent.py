# agents/ai_assistant_agent.py
"""
AI Assistant Agent for handling conversational AI requests through the agent registry
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from app.ai_assistant_service import ai_assistant

logger = logging.getLogger(__name__)


class AIAssistantAgent:
    """
    Agent wrapper for AI Assistant Service to integrate with MCP and agent registry
    """
    
    def __init__(self):
        self.agent_type = "ai_assistant"
        self.topic = "ai_assistant_tasks"
        self.service = ai_assistant
        logger.info("AI Assistant Agent initialized")
    
    async def process_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process AI assistant request from MCP
        
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
    
    def start_mcp_listener(self):
        """Start MCP listener for AI assistant tasks"""
        def handle_message(payload: Dict[str, Any]):
            """Handle incoming MCP messages"""
            try:
                action = payload.get("action", "process_request")
                
                if action == "process_request":
                    # Process async request in background
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.process_request(payload))
                    
                    # Publish result back if response_topic provided
                    response_topic = payload.get("response_topic")
                    if response_topic:
                        mcp.publish(response_topic, {"result": result})
                
                elif action == "get_history":
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.get_conversation_history(payload))
                    
                    response_topic = payload.get("response_topic")
                    if response_topic:
                        mcp.publish(response_topic, {"result": result})
                
                elif action == "clear_session":
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.clear_session(payload))
                    
                    response_topic = payload.get("response_topic")
                    if response_topic:
                        mcp.publish(response_topic, {"result": result})
                
                elif action == "get_user_sessions":
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.get_user_sessions(payload))
                    
                    response_topic = payload.get("response_topic")
                    if response_topic:
                        mcp.publish(response_topic, {"result": result})
                
                else:
                    logger.warning(f"Unknown action: {action}")
                    
            except Exception as e:
                logger.error(f"Error handling MCP message: {e}")
        
        # Start MCP subscription in background thread
        import threading
        
        def mcp_listener():
            try:
                mcp.subscribe(self.topic, handle_message)
            except Exception as e:
                logger.error(f"MCP listener error: {e}")
        
        thread = threading.Thread(target=mcp_listener, daemon=True)
        thread.start()
        logger.info(f"Started MCP listener for topic: {self.topic}")


# Global AI Assistant Agent instance
ai_assistant_agent = AIAssistantAgent()