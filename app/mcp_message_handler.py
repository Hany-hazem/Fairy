# app/mcp_message_handler.py
"""
Enhanced MCP Message Handler

This module provides standardized message processing, validation, and serialization
for the Model Context Protocol (MCP) system using Pydantic models.
"""

import json
import gzip
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import ValidationError

from .mcp_models import (
    MCPMessage, AgentContext, MessageRoutingRule, ValidationResult,
    MCPMessageType, MCPMessagePriority, SerializationFormat, ContextAccessLevel
)

logger = logging.getLogger(__name__)


class MessageCompressionLevel(Enum):
    """Compression levels for message serialization"""
    NONE = 0
    LOW = 1
    MEDIUM = 6
    HIGH = 9


class MCPMessageHandler:
    """
    Handles MCP message validation, serialization, and routing logic
    
    Provides comprehensive message processing capabilities including:
    - Message format validation
    - Serialization with compression support
    - Routing rule evaluation
    - Error response generation
    """
    
    def __init__(self):
        self.routing_rules: List[MessageRoutingRule] = []
        self.compression_threshold = 1024  # bytes
        self.max_message_size = 10 * 1024 * 1024  # 10MB
        self.supported_formats = [SerializationFormat.JSON, SerializationFormat.COMPRESSED_JSON]
        
        # Message type handlers
        self.type_handlers: Dict[str, callable] = {}
        
        # Statistics
        self.stats = {
            "messages_processed": 0,
            "messages_validated": 0,
            "validation_errors": 0,
            "serialization_errors": 0,
            "routing_errors": 0,
            "compression_saved_bytes": 0
        }
        
        logger.info("MCP Message Handler initialized")
    
    def validate_message(self, message: Union[MCPMessage, Dict[str, Any]]) -> ValidationResult:
        """
        Validate MCP message format and structure using Pydantic validation
        
        Args:
            message: MCP message to validate (MCPMessage or dict)
            
        Returns:
            ValidationResult with validation status and errors
        """
        try:
            result = ValidationResult(is_valid=True)
            
            # Convert dict to MCPMessage if needed
            if isinstance(message, dict):
                try:
                    message = MCPMessage(**message)
                except ValidationError as e:
                    errors = [f"{error['loc'][0]}: {error['msg']}" for error in e.errors()]
                    return ValidationResult(is_valid=False, errors=errors)
                except Exception as e:
                    return ValidationResult(is_valid=False, errors=[f"Failed to parse message: {str(e)}"])
            
            # Pydantic validation is already done during object creation
            if not isinstance(message, MCPMessage):
                return ValidationResult(is_valid=False, errors=["Invalid message type"])
            
            # Additional business logic validation
            # Payload size validation
            try:
                payload_size = len(message.json())
                if payload_size > self.max_message_size:
                    result.add_error(f"Message size ({payload_size} bytes) exceeds maximum ({self.max_message_size} bytes)")
            except Exception as e:
                result.add_error(f"Failed to serialize message for size check: {str(e)}")
            
            # Check if message has expired
            if message.is_expired():
                result.add_warning("Message has expired based on TTL")
            
            # Custom type-specific validation
            type_errors = self._validate_message_type_content(message)
            for error in type_errors:
                result.add_error(error)
            
            # Update statistics
            self.stats["messages_validated"] += 1
            if result.has_errors():
                self.stats["validation_errors"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating message: {e}")
            self.stats["validation_errors"] += 1
            return ValidationResult(is_valid=False, errors=[f"Validation error: {str(e)}"])
    
    def serialize_message(self, message: MCPMessage, 
                         format_type: SerializationFormat = SerializationFormat.JSON,
                         compression_level: MessageCompressionLevel = MessageCompressionLevel.MEDIUM) -> bytes:
        """
        Serialize MCP message to bytes using Pydantic serialization
        
        Args:
            message: MCP message to serialize
            format_type: Serialization format to use
            compression_level: Compression level for compressed formats
            
        Returns:
            Serialized message as bytes
        """
        try:
            # Use Pydantic's built-in serialization with compression support
            serialized = message.serialize(format_type, self.compression_threshold)
            
            # Track compression savings if compressed
            if format_type == SerializationFormat.COMPRESSED_JSON:
                original_size = len(message.json().encode('utf-8'))
                if len(serialized) < original_size:
                    self.stats["compression_saved_bytes"] += original_size - len(serialized)
            
            # Validate serialized size
            if len(serialized) > self.max_message_size:
                raise ValueError(f"Serialized message size ({len(serialized)} bytes) exceeds maximum")
            
            self.stats["messages_processed"] += 1
            return serialized
            
        except Exception as e:
            logger.error(f"Error serializing message {message.id}: {e}")
            self.stats["serialization_errors"] += 1
            raise
    
    def deserialize_message(self, data: bytes, 
                           format_type: SerializationFormat = SerializationFormat.JSON) -> MCPMessage:
        """
        Deserialize bytes to MCP message using Pydantic deserialization
        
        Args:
            data: Serialized message data
            format_type: Expected serialization format
            
        Returns:
            Deserialized MCP message
        """
        try:
            # Use Pydantic's built-in deserialization
            return MCPMessage.deserialize(data, format_type)
            
        except Exception as e:
            logger.error(f"Error deserializing message: {e}")
            raise
    
    def route_message(self, message: MCPMessage) -> List[str]:
        """
        Determine target agents for message based on routing rules
        
        Args:
            message: MCP message to route
            
        Returns:
            List of agent IDs that should receive the message
        """
        try:
            target_agents = set(message.target_agents)  # Start with explicit targets
            
            # Apply routing rules using Pydantic model methods
            for rule in self.routing_rules:
                if rule.matches(message):
                    target_agents.update(rule.route_to)
            
            return list(target_agents)
            
        except Exception as e:
            logger.error(f"Error routing message {message.id}: {e}")
            self.stats["routing_errors"] += 1
            return message.target_agents  # Fallback to original targets
    
    def create_error_response(self, error: Exception, original_message: MCPMessage) -> MCPMessage:
        """
        Create standardized error response message using Pydantic model methods
        
        Args:
            error: Exception that occurred
            original_message: Original message that caused the error
            
        Returns:
            Error response MCP message
        """
        try:
            # Use the MCPMessage's built-in error response creation
            return original_message.create_error_response(error, "mcp_message_handler")
            
        except Exception as e:
            logger.error(f"Error creating error response: {e}")
            # Create minimal error response
            return MCPMessage(
                type=MCPMessageType.ERROR.value,
                source_agent="mcp_message_handler",
                target_agents=[original_message.source_agent],
                payload={"error": "Failed to create error response", "details": str(e)},
                priority=MCPMessagePriority.HIGH.value
            )
    
    def add_routing_rule(self, rule: MessageRoutingRule):
        """Add a message routing rule"""
        self.routing_rules.append(rule)
        logger.info(f"Added routing rule for message type: {rule.message_type}")
    
    def register_type_handler(self, message_type: str, handler: callable):
        """Register a handler for specific message types"""
        self.type_handlers[message_type] = handler
        logger.info(f"Registered type handler for: {message_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message handler statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset message handler statistics"""
        for key in self.stats:
            self.stats[key] = 0
        logger.info("Message handler statistics reset")
    
    def validate_context(self, context: Union[AgentContext, Dict[str, Any]]) -> ValidationResult:
        """
        Validate agent context using Pydantic validation
        
        Args:
            context: Agent context to validate (AgentContext or dict)
            
        Returns:
            ValidationResult with validation status and errors
        """
        try:
            result = ValidationResult(is_valid=True)
            
            # Convert dict to AgentContext if needed
            if isinstance(context, dict):
                try:
                    context = AgentContext(**context)
                except ValidationError as e:
                    errors = [f"{error['loc'][0]}: {error['msg']}" for error in e.errors()]
                    return ValidationResult(is_valid=False, errors=errors)
                except Exception as e:
                    return ValidationResult(is_valid=False, errors=[f"Failed to parse context: {str(e)}"])
            
            if not isinstance(context, AgentContext):
                return ValidationResult(is_valid=False, errors=["Invalid context type"])
            
            # Check if context has expired
            if context.is_expired():
                result.add_warning("Context has expired based on TTL")
            
            # Additional business logic validation can be added here
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating context: {e}")
            return ValidationResult(is_valid=False, errors=[f"Context validation error: {str(e)}"])
    
    def create_context_update_message(self, context: AgentContext, target_agents: List[str], 
                                    source_agent: str = None) -> MCPMessage:
        """
        Create a context update message from an AgentContext
        
        Args:
            context: Agent context to share
            target_agents: List of target agent IDs
            source_agent: Source agent ID (defaults to context owner)
            
        Returns:
            MCP context update message
        """
        try:
            return context.to_message(target_agents, source_agent)
        except Exception as e:
            logger.error(f"Error creating context update message: {e}")
            raise
    
    def extract_context_from_message(self, message: MCPMessage) -> Optional[AgentContext]:
        """
        Extract AgentContext from a context update message
        
        Args:
            message: MCP context update message
            
        Returns:
            AgentContext if message contains valid context data, None otherwise
        """
        try:
            if message.type != MCPMessageType.CONTEXT_UPDATE.value:
                return None
            
            payload = message.payload
            if 'context_data' not in payload or 'context_type' not in payload:
                return None
            
            # Create AgentContext from message payload
            context_data = {
                'agent_id': message.source_agent,
                'context_type': payload['context_type'],
                'context_data': payload['context_data'],
                'version': payload.get('context_version', message.context_version),
                'last_updated': message.timestamp,
                'access_level': payload.get('access_level', ContextAccessLevel.PUBLIC.value),
                'metadata': payload.get('metadata', {})
            }
            
            return AgentContext(**context_data)
            
        except Exception as e:
            logger.error(f"Error extracting context from message {message.id}: {e}")
            return None
    

    
    def _validate_message_type_content(self, message: MCPMessage) -> List[str]:
        """
        Additional custom validation for message content based on type
        Note: Basic type validation is now handled by Pydantic models
        """
        errors = []
        
        try:
            # Call custom type handler if available
            if message.type in self.type_handlers:
                try:
                    handler_errors = self.type_handlers[message.type](message)
                    if handler_errors:
                        errors.extend(handler_errors)
                except Exception as e:
                    errors.append(f"Type handler error: {str(e)}")
        
        except Exception as e:
            errors.append(f"Content validation error: {str(e)}")
        
        return errors
    



# Global message handler instance
mcp_message_handler = MCPMessageHandler()