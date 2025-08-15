# app/models.py
"""
Data models for the AI Assistant system
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
import uuid

class Message(BaseModel):
    """Individual message in a conversation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConversationSession(BaseModel):
    """Conversation session containing multiple messages"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Message] = Field(default_factory=list)
    context_summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def add_message(self, role: Literal["user", "assistant", "system"], 
                   content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a new message to the conversation"""
        message = Message(
            session_id=self.id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
        return message
    
    def get_messages_for_llm(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages in OpenAI format for LLM consumption"""
        messages = self.messages
        if max_messages:
            messages = messages[-max_messages:]
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    def get_context_length(self) -> int:
        """Estimate total context length in characters"""
        return sum(len(msg.content) for msg in self.messages)
    
    def get_recent_messages(self, minutes: int = 60) -> List[Message]:
        """Get messages from the last N minutes"""
        cutoff = datetime.utcnow().timestamp() - (minutes * 60)
        return [
            msg for msg in self.messages 
            if msg.timestamp.timestamp() > cutoff
        ]

class SessionSummary(BaseModel):
    """Summary of a conversation session for quick retrieval"""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime
    last_activity: datetime
    message_count: int
    context_summary: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConversationContext(BaseModel):
    """Context information for conversation continuation"""
    session_id: str
    recent_messages: List[Message]
    context_summary: Optional[str] = None
    relevant_history: List[str] = Field(default_factory=list)
    total_context_length: int
    
    def to_llm_messages(self) -> List[Dict[str, str]]:
        """Convert to OpenAI message format"""
        messages = []
        
        # Add context summary as system message if available
        if self.context_summary:
            messages.append({
                "role": "system", 
                "content": f"Previous conversation context: {self.context_summary}"
            })
        
        # Add recent messages
        messages.extend([
            {"role": msg.role, "content": msg.content}
            for msg in self.recent_messages
        ])
        
        return messages