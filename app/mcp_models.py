# app/mcp_models.py
"""
Enhanced MCP Message Models and Validation

This module provides Pydantic-based data models for MCP messages and agent context
with comprehensive validation, serialization, and compression support.
"""

import gzip
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator
import logging

logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """Standard MCP message types"""
    CONTEXT_UPDATE = "context_update"
    TASK_NOTIFICATION = "task_notification"
    AGENT_REQUEST = "agent_request"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"


class MCPMessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ContextAccessLevel(Enum):
    """Context access levels"""
    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class SerializationFormat(Enum):
    """Supported message serialization formats"""
    JSON = "json"
    COMPRESSED_JSON = "compressed_json"
    MSGPACK = "msgpack"  # Future implementation


class MCPMessage(BaseModel):
    """
    Enhanced MCP message with Pydantic validation
    
    Provides comprehensive message structure with validation,
    serialization support, and automatic field generation.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message identifier")
    type: str = Field(..., description="Message type from MCPMessageType enum")
    source_agent: str = Field(..., description="ID of the agent sending the message")
    target_agents: List[str] = Field(..., description="List of target agent IDs")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Message payload data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message creation timestamp")
    priority: int = Field(default=MCPMessagePriority.NORMAL.value, description="Message priority level")
    requires_ack: bool = Field(default=False, description="Whether message requires acknowledgment")
    correlation_id: Optional[str] = Field(None, description="ID for correlating request/response messages")
    context_version: Optional[str] = Field(None, description="Version of context when message was created")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    compression_format: Optional[str] = Field(None, description="Compression format used for payload")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True
    
    @validator('type')
    def validate_message_type(cls, v):
        """Validate message type against enum values"""
        valid_types = [t.value for t in MCPMessageType]
        if v not in valid_types:
            raise ValueError(f"Invalid message type: {v}. Must be one of {valid_types}")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority level"""
        valid_priorities = [p.value for p in MCPMessagePriority]
        if v not in valid_priorities:
            raise ValueError(f"Invalid priority: {v}. Must be one of {valid_priorities}")
        return v
    
    @validator('target_agents')
    def validate_target_agents(cls, v):
        """Validate target agents list"""
        if not v:
            raise ValueError("Target agents list cannot be empty")
        if not all(isinstance(agent, str) and agent.strip() for agent in v):
            raise ValueError("All target agents must be non-empty strings")
        return v
    
    @validator('ttl')
    def validate_ttl(cls, v):
        """Validate TTL value"""
        if v is not None and (not isinstance(v, int) or v <= 0):
            raise ValueError("TTL must be a positive integer")
        return v
    
    @root_validator
    def validate_message_content(cls, values):
        """Validate message content based on type"""
        message_type = values.get('type')
        payload = values.get('payload', {})
        
        if message_type == MCPMessageType.CONTEXT_UPDATE.value:
            if 'context_data' not in payload:
                raise ValueError("Context update messages must include 'context_data' in payload")
            if 'context_type' not in payload:
                raise ValueError("Context update messages must include 'context_type' in payload")
        
        elif message_type == MCPMessageType.TASK_NOTIFICATION.value:
            if 'task_id' not in payload:
                raise ValueError("Task notification messages must include 'task_id' in payload")
            if 'action' not in payload:
                raise ValueError("Task notification messages must include 'action' in payload")
        
        elif message_type == MCPMessageType.AGENT_REQUEST.value:
            if 'request_type' not in payload:
                raise ValueError("Agent request messages must include 'request_type' in payload")
        
        elif message_type == MCPMessageType.AGENT_RESPONSE.value:
            if 'response_data' not in payload:
                raise ValueError("Agent response messages must include 'response_data' in payload")
            if not values.get('correlation_id'):
                raise ValueError("Agent response messages must include correlation_id")
        
        elif message_type == MCPMessageType.HEARTBEAT.value:
            if 'agent_status' not in payload:
                raise ValueError("Heartbeat messages must include 'agent_status' in payload")
        
        return values
    
    def is_expired(self) -> bool:
        """Check if message has expired based on TTL"""
        if not self.ttl:
            return False
        return (datetime.utcnow() - self.timestamp).total_seconds() > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return self.dict()
    
    def serialize(self, format_type: SerializationFormat = SerializationFormat.JSON,
                  compression_threshold: int = 1024) -> bytes:
        """
        Serialize message to bytes with optional compression
        
        Args:
            format_type: Serialization format to use
            compression_threshold: Minimum size in bytes to trigger compression
            
        Returns:
            Serialized message as bytes
        """
        try:
            # Convert to JSON
            json_data = self.json().encode('utf-8')
            
            if format_type == SerializationFormat.JSON:
                return json_data
            
            elif format_type == SerializationFormat.COMPRESSED_JSON:
                # Only compress if data is above threshold
                if len(json_data) > compression_threshold:
                    compressed = gzip.compress(json_data, compresslevel=6)
                    # Update compression format in the message
                    self.compression_format = "gzip"
                    return compressed
                else:
                    return json_data
            
            else:
                raise ValueError(f"Unsupported serialization format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error serializing message {self.id}: {e}")
            raise
    
    @classmethod
    def deserialize(cls, data: bytes, format_type: SerializationFormat = SerializationFormat.JSON) -> 'MCPMessage':
        """
        Deserialize bytes to MCP message
        
        Args:
            data: Serialized message data
            format_type: Expected serialization format
            
        Returns:
            Deserialized MCP message
        """
        try:
            if format_type == SerializationFormat.JSON:
                json_str = data.decode('utf-8')
                return cls.parse_raw(json_str)
            
            elif format_type == SerializationFormat.COMPRESSED_JSON:
                # Try decompression first, fallback to direct JSON if not compressed
                try:
                    decompressed = gzip.decompress(data)
                    json_str = decompressed.decode('utf-8')
                except gzip.BadGzipFile:
                    # Data is not compressed, treat as regular JSON
                    json_str = data.decode('utf-8')
                
                return cls.parse_raw(json_str)
            
            else:
                raise ValueError(f"Unsupported deserialization format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error deserializing message: {e}")
            raise
    
    def create_response(self, response_data: Dict[str, Any], source_agent: str) -> 'MCPMessage':
        """
        Create a response message for this message
        
        Args:
            response_data: Response payload data
            source_agent: ID of the agent creating the response
            
        Returns:
            Response MCP message
        """
        return MCPMessage(
            type=MCPMessageType.AGENT_RESPONSE.value,
            source_agent=source_agent,
            target_agents=[self.source_agent],
            payload={"response_data": response_data},
            correlation_id=self.id,
            priority=self.priority
        )
    
    def create_error_response(self, error: Exception, source_agent: str) -> 'MCPMessage':
        """
        Create an error response message for this message
        
        Args:
            error: Exception that occurred
            source_agent: ID of the agent creating the error response
            
        Returns:
            Error response MCP message
        """
        error_payload = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "original_message_id": self.id,
            "original_message_type": self.type,
            "timestamp": datetime.utcnow().isoformat(),
            "recoverable": self._is_recoverable_error(error)
        }
        
        return MCPMessage(
            type=MCPMessageType.ERROR.value,
            source_agent=source_agent,
            target_agents=[self.source_agent],
            payload=error_payload,
            correlation_id=self.id,
            priority=MCPMessagePriority.HIGH.value
        )
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable"""
        recoverable_types = [
            "ConnectionError",
            "TimeoutError",
            "TemporaryFailure",
            "RateLimitError"
        ]
        return type(error).__name__ in recoverable_types


class AgentContext(BaseModel):
    """
    Agent context model for sharing context information between agents
    
    Provides structured context data with versioning, access control,
    and efficient serialization for inter-agent communication.
    """
    
    agent_id: str = Field(..., description="ID of the agent owning this context")
    context_type: str = Field(..., description="Type of context (task, project, conversation, etc.)")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Context data payload")
    version: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Context version identifier")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    shared_with: List[str] = Field(default_factory=list, description="List of agent IDs with access")
    access_level: str = Field(default=ContextAccessLevel.PUBLIC.value, description="Access level for this context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True
    
    @validator('access_level')
    def validate_access_level(cls, v):
        """Validate access level against enum values"""
        valid_levels = [level.value for level in ContextAccessLevel]
        if v not in valid_levels:
            raise ValueError(f"Invalid access level: {v}. Must be one of {valid_levels}")
        return v
    
    @validator('context_type')
    def validate_context_type(cls, v):
        """Validate context type"""
        if not v or not isinstance(v, str):
            raise ValueError("Context type must be a non-empty string")
        return v.lower()
    
    @validator('shared_with')
    def validate_shared_with(cls, v):
        """Validate shared_with list"""
        if not all(isinstance(agent, str) and agent.strip() for agent in v):
            raise ValueError("All shared_with entries must be non-empty strings")
        return v
    
    def is_expired(self) -> bool:
        """Check if context has expired based on TTL"""
        if not self.ttl:
            return False
        return (datetime.utcnow() - self.last_updated).total_seconds() > self.ttl
    
    def can_access(self, agent_id: str) -> bool:
        """
        Check if an agent can access this context
        
        Args:
            agent_id: ID of the agent requesting access
            
        Returns:
            True if agent has access permission
        """
        # Owner always has access
        if agent_id == self.agent_id:
            return True
        
        # Check access level
        if self.access_level == ContextAccessLevel.PUBLIC.value:
            return True
        
        if self.access_level == ContextAccessLevel.PRIVATE.value:
            return False
        
        # For restricted and confidential, check shared_with list
        return agent_id in self.shared_with
    
    def grant_access(self, agent_id: str):
        """Grant access to an agent"""
        if agent_id not in self.shared_with:
            self.shared_with.append(agent_id)
            self.last_updated = datetime.utcnow()
    
    def revoke_access(self, agent_id: str):
        """Revoke access from an agent"""
        if agent_id in self.shared_with:
            self.shared_with.remove(agent_id)
            self.last_updated = datetime.utcnow()
    
    def update_context(self, new_data: Dict[str, Any], merge: bool = True):
        """
        Update context data
        
        Args:
            new_data: New context data
            merge: Whether to merge with existing data or replace
        """
        if merge:
            self.context_data.update(new_data)
        else:
            self.context_data = new_data
        
        self.last_updated = datetime.utcnow()
        self.version = str(uuid.uuid4())
    
    def to_message(self, target_agents: List[str], source_agent: str = None) -> MCPMessage:
        """
        Convert context to MCP context update message
        
        Args:
            target_agents: List of target agent IDs
            source_agent: Source agent ID (defaults to context owner)
            
        Returns:
            MCP context update message
        """
        return MCPMessage(
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent=source_agent or self.agent_id,
            target_agents=target_agents,
            payload={
                "context_data": self.context_data,
                "context_type": self.context_type,
                "context_version": self.version,
                "access_level": self.access_level,
                "metadata": self.metadata
            },
            context_version=self.version
        )


class MessageRoutingRule(BaseModel):
    """
    Rule for routing messages based on content and type
    
    Provides flexible message routing configuration with pattern matching
    and filtering capabilities.
    """
    
    message_type: str = Field(..., description="Message type to match (* for all)")
    source_pattern: Optional[str] = Field(None, description="Source agent pattern (* wildcard supported)")
    target_pattern: Optional[str] = Field(None, description="Target agent pattern (* wildcard supported)")
    payload_filter: Optional[Dict[str, Any]] = Field(None, description="Payload filter criteria")
    priority_threshold: Optional[int] = Field(None, description="Minimum priority level")
    route_to: List[str] = Field(default_factory=list, description="Additional agents to route to")
    enabled: bool = Field(default=True, description="Whether this rule is enabled")
    
    @validator('priority_threshold')
    def validate_priority_threshold(cls, v):
        """Validate priority threshold"""
        if v is not None:
            valid_priorities = [p.value for p in MCPMessagePriority]
            if v not in valid_priorities:
                raise ValueError(f"Invalid priority threshold: {v}. Must be one of {valid_priorities}")
        return v
    
    def matches(self, message: MCPMessage) -> bool:
        """
        Check if message matches this routing rule
        
        Args:
            message: MCP message to check
            
        Returns:
            True if message matches the rule
        """
        if not self.enabled:
            return False
        
        # Check message type
        if self.message_type != "*" and self.message_type != message.type:
            return False
        
        # Check source pattern
        if self.source_pattern and not self._matches_pattern(message.source_agent, self.source_pattern):
            return False
        
        # Check target pattern
        if self.target_pattern:
            target_match = any(
                self._matches_pattern(target, self.target_pattern)
                for target in message.target_agents
            )
            if not target_match:
                return False
        
        # Check priority threshold
        if self.priority_threshold and message.priority < self.priority_threshold:
            return False
        
        # Check payload filter
        if self.payload_filter and not self._matches_payload_filter(message.payload, self.payload_filter):
            return False
        
        return True
    
    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Simple pattern matching (supports * wildcard)"""
        if pattern == "*":
            return True
        
        if "*" not in pattern:
            return value == pattern
        
        # Simple wildcard matching
        import re
        regex_pattern = pattern.replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", value))
    
    def _matches_payload_filter(self, payload: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if payload matches filter criteria"""
        try:
            for key, expected_value in filter_dict.items():
                if key not in payload:
                    return False
                
                actual_value = payload[key]
                
                # Support different comparison types
                if isinstance(expected_value, dict) and "$regex" in expected_value:
                    import re
                    if not re.match(expected_value["$regex"], str(actual_value)):
                        return False
                elif isinstance(expected_value, dict) and "$in" in expected_value:
                    if actual_value not in expected_value["$in"]:
                        return False
                elif actual_value != expected_value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error matching payload filter: {e}")
            return False


class ValidationResult(BaseModel):
    """Message validation result"""
    
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    
    def add_error(self, error: str):
        """Add a validation error"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a validation warning"""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if there are validation errors"""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are validation warnings"""
        return len(self.warnings) > 0


# Utility functions for message creation
def create_context_update_message(
    source_agent: str,
    target_agents: List[str],
    context: AgentContext,
    priority: int = MCPMessagePriority.NORMAL.value
) -> MCPMessage:
    """Create a context update message"""
    return context.to_message(target_agents, source_agent)


def create_task_notification_message(
    source_agent: str,
    target_agents: List[str],
    task_id: str,
    action: str,
    task_data: Dict[str, Any] = None,
    priority: int = MCPMessagePriority.NORMAL.value
) -> MCPMessage:
    """Create a task notification message"""
    payload = {
        "task_id": task_id,
        "action": action
    }
    if task_data:
        payload["task_data"] = task_data
    
    return MCPMessage(
        type=MCPMessageType.TASK_NOTIFICATION.value,
        source_agent=source_agent,
        target_agents=target_agents,
        payload=payload,
        priority=priority
    )


def create_agent_request_message(
    source_agent: str,
    target_agents: List[str],
    request_type: str,
    request_data: Dict[str, Any] = None,
    priority: int = MCPMessagePriority.NORMAL.value
) -> MCPMessage:
    """Create an agent request message"""
    payload = {
        "request_type": request_type
    }
    if request_data:
        payload.update(request_data)
    
    return MCPMessage(
        type=MCPMessageType.AGENT_REQUEST.value,
        source_agent=source_agent,
        target_agents=target_agents,
        payload=payload,
        priority=priority
    )


def create_heartbeat_message(
    source_agent: str,
    agent_status: str,
    status_data: Dict[str, Any] = None
) -> MCPMessage:
    """Create a heartbeat message"""
    payload = {
        "agent_status": agent_status
    }
    if status_data:
        payload["status_data"] = status_data
    
    return MCPMessage(
        type=MCPMessageType.HEARTBEAT.value,
        source_agent=source_agent,
        target_agents=["mcp_server"],  # Heartbeats go to server
        payload=payload,
        priority=MCPMessagePriority.LOW.value
    )