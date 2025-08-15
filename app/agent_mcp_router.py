# app/agent_mcp_router.py
"""
MCP Router for Agent Communication

This module handles routing MCP messages to appropriate agents based on
the agent registry configuration.
"""

import logging
import json
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .agent_registry import registry, AgentType

logger = logging.getLogger(__name__)


class AgentMCPRouter:
    """
    Routes MCP messages to appropriate agents based on agent registry
    """
    
    def __init__(self):
        self.agent_handlers: Dict[str, Callable] = {}
        self.topic_subscriptions: Dict[str, threading.Thread] = {}
        self.running = False
        logger.info("Agent MCP Router initialized")
    
    def register_agent_handler(self, agent_type: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Register a handler function for a specific agent type
        
        Args:
            agent_type: Type of agent (ai_assistant, self_improvement, etc.)
            handler: Async function to handle messages for this agent type
        """
        self.agent_handlers[agent_type] = handler
        logger.info(f"Registered handler for agent type: {agent_type}")
    
    def start_routing(self):
        """Start MCP routing for all registered agents"""
        if not registry:
            logger.error("Agent registry not available, cannot start routing")
            return False
        
        try:
            # Get all agent topics from registry
            agent_topics = registry.get_agent_topics()
            
            for intent, topic in agent_topics.items():
                if topic not in self.topic_subscriptions:
                    self._start_topic_subscription(intent, topic)
            
            self.running = True
            logger.info(f"Started MCP routing for {len(agent_topics)} agent topics")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP routing: {e}")
            return False
    
    def stop_routing(self):
        """Stop all MCP routing"""
        self.running = False
        
        # Note: We can't easily stop Redis subscription threads,
        # but we can mark them as not running
        logger.info("Stopped MCP routing")
    
    def _start_topic_subscription(self, intent: str, topic: str):
        """Start MCP subscription for a specific topic"""
        def topic_handler(payload: Dict[str, Any]):
            """Handle messages for this topic"""
            try:
                if not self.running:
                    return
                
                # Get agent configuration
                agent_config = registry.get_agent(intent)
                agent_type = agent_config.get("type", "llm")
                
                # Add routing metadata
                payload["agent_intent"] = intent
                payload["agent_type"] = agent_type
                payload["topic"] = topic
                payload["routed_at"] = datetime.utcnow().isoformat()
                
                # Route to appropriate handler
                if agent_type in self.agent_handlers:
                    try:
                        # Call the registered handler
                        handler = self.agent_handlers[agent_type]
                        
                        # Handle both sync and async handlers
                        if hasattr(handler, '__call__'):
                            if hasattr(handler, '__await__'):
                                # Async handler
                                import asyncio
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                result = loop.run_until_complete(handler(payload))
                            else:
                                # Sync handler
                                result = handler(payload)
                            
                            # Publish result if response topic provided
                            response_topic = payload.get("response_topic")
                            if response_topic and result:
                                mcp.publish(response_topic, {"result": result})
                                
                        else:
                            logger.warning(f"Invalid handler for agent type: {agent_type}")
                            
                    except Exception as e:
                        logger.error(f"Error in agent handler for {agent_type}: {e}")
                        
                        # Send error response if response topic provided
                        response_topic = payload.get("response_topic")
                        if response_topic:
                            error_result = {
                                "error": str(e),
                                "agent_type": agent_type,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            mcp.publish(response_topic, {"result": error_result})
                else:
                    logger.warning(f"No handler registered for agent type: {agent_type}")
                    
            except Exception as e:
                logger.error(f"Error handling MCP message for topic {topic}: {e}")
        
        def subscription_worker():
            """Worker thread for MCP subscription"""
            try:
                mcp.subscribe(topic, topic_handler)
            except Exception as e:
                logger.error(f"MCP subscription error for topic {topic}: {e}")
        
        # Start subscription thread
        thread = threading.Thread(target=subscription_worker, daemon=True)
        thread.start()
        
        self.topic_subscriptions[topic] = thread
        logger.info(f"Started MCP subscription for topic: {topic} (intent: {intent})")
    
    def send_agent_message(self, intent: str, payload: Dict[str, Any], 
                          response_topic: Optional[str] = None) -> bool:
        """
        Send a message to a specific agent via MCP
        
        Args:
            intent: Agent intent to send message to
            payload: Message payload
            response_topic: Optional topic to receive response
            
        Returns:
            True if message was sent successfully
        """
        try:
            if not registry:
                logger.error("Agent registry not available")
                return False
            
            if not registry.supports_intent(intent):
                logger.error(f"Unknown agent intent: {intent}")
                return False
            
            # Get agent configuration
            agent_config = registry.get_agent(intent)
            topic = agent_config.get("topic")
            
            if not topic:
                logger.error(f"No topic configured for agent intent: {intent}")
                return False
            
            # Add response topic if provided
            if response_topic:
                payload["response_topic"] = response_topic
            
            # Add metadata
            payload["sent_at"] = datetime.utcnow().isoformat()
            payload["sender"] = "agent_mcp_router"
            
            # Publish message
            mcp.publish(topic, payload)
            logger.debug(f"Sent message to agent {intent} on topic {topic}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to agent {intent}: {e}")
            return False
    
    def broadcast_to_agent_type(self, agent_type: str, payload: Dict[str, Any]) -> int:
        """
        Broadcast a message to all agents of a specific type
        
        Args:
            agent_type: Type of agents to broadcast to
            payload: Message payload
            
        Returns:
            Number of agents the message was sent to
        """
        try:
            if not registry:
                logger.error("Agent registry not available")
                return 0
            
            # Get all agents of the specified type
            if AgentType:
                try:
                    agent_type_enum = AgentType(agent_type)
                    agents = registry.get_agents_by_type(agent_type_enum)
                except ValueError:
                    logger.error(f"Invalid agent type: {agent_type}")
                    return 0
            else:
                # Fallback if AgentType enum not available
                all_agents = registry.list_agents()
                agents = {
                    intent: config for intent, config in all_agents.items()
                    if config.get("type") == agent_type
                }
            
            sent_count = 0
            for intent in agents.keys():
                if self.send_agent_message(intent, payload.copy()):
                    sent_count += 1
            
            logger.info(f"Broadcast message to {sent_count} agents of type {agent_type}")
            return sent_count
            
        except Exception as e:
            logger.error(f"Error broadcasting to agent type {agent_type}: {e}")
            return 0
    
    def get_routing_status(self) -> Dict[str, Any]:
        """Get current routing status"""
        try:
            if not registry:
                return {
                    "status": "error",
                    "error": "Agent registry not available"
                }
            
            agent_topics = registry.get_agent_topics()
            
            return {
                "status": "running" if self.running else "stopped",
                "registered_handlers": list(self.agent_handlers.keys()),
                "active_subscriptions": list(self.topic_subscriptions.keys()),
                "agent_topics": agent_topics,
                "topic_count": len(agent_topics),
                "handler_count": len(self.agent_handlers),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting routing status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global MCP router instance
agent_mcp_router = AgentMCPRouter()