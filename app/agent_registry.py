# app/agent_registry.py
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from enum import Enum

from .config import settings

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Supported agent types"""
    LLM = "llm"
    VISION = "vision"
    AI_ASSISTANT = "ai_assistant"
    SELF_IMPROVEMENT = "self_improvement"
    CODE_ANALYSIS = "code_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"


class AgentRegistry:
    """
    Enhanced agent registry with support for AI assistant and self-improvement agents
    """
    
    def __init__(self, path: str):
        self.registry_path = path
        self.registry = self._load_registry()
        logger.info(f"Agent registry loaded with {len(self.registry)} agents")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load agent registry from JSON file with error handling"""
        try:
            with open(self.registry_path) as f:
                registry = json.load(f)
            
            # Validate registry structure
            self._validate_registry(registry)
            return registry
            
        except FileNotFoundError:
            logger.error(f"Agent registry file not found: {self.registry_path}")
            # Return minimal default registry
            return {
                "general_query": {
                    "type": "llm",
                    "topic": "general_queries",
                    "description": "Default LLM agent"
                }
            }
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in agent registry: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading agent registry: {e}")
            raise
    
    def _validate_registry(self, registry: Dict[str, Any]) -> None:
        """Validate registry structure and required fields"""
        required_fields = ["type", "topic", "description"]
        
        for intent, config in registry.items():
            if not isinstance(config, dict):
                raise ValueError(f"Agent config for '{intent}' must be a dictionary")
            
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Agent '{intent}' missing required field: {field}")
            
            # Validate agent type
            agent_type = config["type"]
            try:
                AgentType(agent_type)
            except ValueError:
                logger.warning(f"Unknown agent type '{agent_type}' for intent '{intent}'")
    
    def get_agent(self, intent: str) -> Dict[str, Any]:
        """Get agent configuration by intent"""
        agent_config = self.registry.get(intent)
        
        if not agent_config:
            logger.warning(f"Intent '{intent}' not found, using default agent")
            return self.registry.get("general_query", {
                "type": "llm",
                "topic": "general_queries",
                "description": "Default fallback agent"
            })
        
        return agent_config
    
    def get_agents_by_type(self, agent_type: AgentType) -> Dict[str, Dict[str, Any]]:
        """Get all agents of a specific type"""
        return {
            intent: config for intent, config in self.registry.items()
            if config.get("type") == agent_type.value
        }
    
    def register_agent(self, intent: str, agent_type: AgentType, 
                      topic: str, description: str, **kwargs) -> bool:
        """Register a new agent configuration"""
        try:
            agent_config = {
                "type": agent_type.value,
                "topic": topic,
                "description": description,
                **kwargs
            }
            
            self.registry[intent] = agent_config
            self._save_registry()
            
            logger.info(f"Registered new agent: {intent} ({agent_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent '{intent}': {e}")
            return False
    
    def unregister_agent(self, intent: str) -> bool:
        """Unregister an agent"""
        try:
            if intent in self.registry:
                del self.registry[intent]
                self._save_registry()
                logger.info(f"Unregistered agent: {intent}")
                return True
            else:
                logger.warning(f"Agent '{intent}' not found for unregistration")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister agent '{intent}': {e}")
            return False
    
    def _save_registry(self) -> None:
        """Save registry back to JSON file"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save agent registry: {e}")
            raise
    
    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all registered agents"""
        return self.registry.copy()
    
    def get_agent_topics(self) -> Dict[str, str]:
        """Get mapping of agent intents to MCP topics"""
        return {
            intent: config.get("topic", "general_queries")
            for intent, config in self.registry.items()
        }
    
    def supports_intent(self, intent: str) -> bool:
        """Check if an intent is supported"""
        return intent in self.registry
    
    def reload_registry(self) -> bool:
        """Reload registry from file"""
        try:
            self.registry = self._load_registry()
            logger.info("Agent registry reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload agent registry: {e}")
            return False


# Singleton instance
try:
    registry = AgentRegistry(settings.AGENT_REGISTRY)
except Exception as e:
    logger.error(f"Failed to initialize agent registry: {e}")
    registry = None
