# app/conversation_manager.py
"""
Conversation session management with Redis persistence
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import redis
from .config import settings
from .models import ConversationSession, Message, SessionSummary, ConversationContext
from .conversation_memory import conversation_memory

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation sessions with Redis persistence"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis = None
        self.session_ttl = 86400 * 7  # 7 days in seconds
        self.max_context_length = 8000  # characters
        self.max_messages_in_context = 20
    
    @property
    def redis(self) -> redis.Redis:
        """Lazy Redis connection"""
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self._redis.ping()
                logger.info(f"Connected to Redis at {self.redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                # Fallback to in-memory storage for development
                self._redis = {}
                logger.warning("Using in-memory storage as Redis fallback")
        return self._redis
    
    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"conversation:session:{session_id}"
    
    def _summary_key(self, session_id: str) -> str:
        """Generate Redis key for session summary"""
        return f"conversation:summary:{session_id}"
    
    def _user_sessions_key(self, user_id: str) -> str:
        """Generate Redis key for user's session list"""
        return f"conversation:user:{user_id}:sessions"
    
    def create_session(self, user_id: Optional[str] = None) -> ConversationSession:
        """Create a new conversation session"""
        session = ConversationSession(user_id=user_id)
        
        try:
            # Store session in Redis
            if isinstance(self.redis, dict):
                # In-memory fallback
                self.redis[self._session_key(session.id)] = session.json()
            else:
                self.redis.setex(
                    self._session_key(session.id),
                    self.session_ttl,
                    session.json()
                )
            
            # Add to user's session list if user_id provided
            if user_id:
                self._add_session_to_user(user_id, session.id)
            
            logger.info(f"Created conversation session {session.id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a conversation session"""
        try:
            if isinstance(self.redis, dict):
                # In-memory fallback
                session_data = self.redis.get(self._session_key(session_id))
            else:
                session_data = self.redis.get(self._session_key(session_id))
            
            if session_data:
                return ConversationSession.parse_raw(session_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def update_session(self, session: ConversationSession) -> bool:
        """Update a conversation session"""
        try:
            session.last_activity = datetime.utcnow()
            
            if isinstance(self.redis, dict):
                # In-memory fallback
                self.redis[self._session_key(session.id)] = session.json()
            else:
                self.redis.setex(
                    self._session_key(session.id),
                    self.session_ttl,
                    session.json()
                )
            
            # Update session summary
            self._update_session_summary(session)
            
            # Store conversation context in memory for future retrieval
            # (only if session has enough messages to be meaningful)
            if len(session.messages) >= 2:
                try:
                    conversation_memory.store_conversation_context(session)
                except Exception as e:
                    logger.warning(f"Failed to store conversation context: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session {session.id}: {e}")
            return False
    
    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: Optional[Dict] = None) -> Optional[Message]:
        """Add a message to a conversation session"""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return None
        
        message = session.add_message(role, content, metadata)
        
        # Update session in storage
        if self.update_session(session):
            logger.debug(f"Added message to session {session_id}")
            return message
        
        return None
    
    def get_context(self, session_id: str, max_tokens: int = None, 
                   query: str = None, user_id: str = None) -> ConversationContext:
        """Get enhanced conversation context for LLM consumption"""
        session = self.get_session(session_id)
        if not session:
            return ConversationContext(
                session_id=session_id,
                recent_messages=[],
                total_context_length=0
            )
        
        max_length = max_tokens or self.max_context_length
        
        # Start with most recent messages and work backwards
        recent_messages = []
        total_length = 0
        
        for message in reversed(session.messages):
            message_length = len(message.content)
            # Check if adding this message would exceed the limit
            if total_length + message_length > max_length and recent_messages:
                break
            
            recent_messages.insert(0, message)
            total_length += message_length
            
            if len(recent_messages) >= self.max_messages_in_context:
                break
        
        # Create base context
        context = ConversationContext(
            session_id=session_id,
            recent_messages=recent_messages,
            context_summary=session.context_summary,
            total_context_length=total_length
        )
        
        # Enhance with relevant history if query provided
        if query and user_id:
            try:
                context = conversation_memory.enhance_context_with_history(
                    context, query, user_id
                )
            except Exception as e:
                logger.error(f"Failed to enhance context with history: {e}")
        
        return context
    
    def summarize_old_context(self, session_id: str) -> Optional[str]:
        """Create a summary of older messages using enhanced memory system"""
        session = self.get_session(session_id)
        if not session or len(session.messages) < 10:
            return None
        
        try:
            # Use the enhanced memory system for summarization
            summary = conversation_memory.generate_context_summary(session)
            
            if summary:
                # Update session with summary
                session.context_summary = summary
                self.update_session(session)
                
                # Store conversation context in memory for future retrieval
                conversation_memory.store_conversation_context(session)
                
                logger.info(f"Created enhanced context summary for session {session_id}")
                return summary
            
        except Exception as e:
            logger.error(f"Failed to create enhanced summary: {e}")
        
        # Fallback to simple summary
        old_messages = session.messages[:-self.max_messages_in_context]
        if not old_messages:
            return None
        
        summary_parts = []
        user_messages = [msg for msg in old_messages if msg.role == "user"]
        assistant_messages = [msg for msg in old_messages if msg.role == "assistant"]
        
        if user_messages:
            summary_parts.append(f"User discussed: {len(user_messages)} topics")
        if assistant_messages:
            summary_parts.append(f"Assistant provided {len(assistant_messages)} responses")
        
        summary = "; ".join(summary_parts)
        
        # Update session with summary
        session.context_summary = summary
        self.update_session(session)
        
        logger.info(f"Created fallback context summary for session {session_id}")
        return summary
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a conversation session"""
        try:
            if isinstance(self.redis, dict):
                # In-memory fallback
                self.redis.pop(self._session_key(session_id), None)
                self.redis.pop(self._summary_key(session_id), None)
            else:
                self.redis.delete(self._session_key(session_id))
                self.redis.delete(self._summary_key(session_id))
            
            logger.info(f"Cleared session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[SessionSummary]:
        """Get recent sessions for a user"""
        try:
            if isinstance(self.redis, dict):
                # In-memory fallback - simplified
                return []
            
            session_ids = self.redis.lrange(self._user_sessions_key(user_id), 0, limit - 1)
            summaries = []
            
            for session_id in session_ids:
                summary_data = self.redis.get(self._summary_key(session_id))
                if summary_data:
                    summaries.append(SessionSummary.parse_raw(summary_data))
            
            return summaries
            
        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id}: {e}")
            return []
    
    def _add_session_to_user(self, user_id: str, session_id: str):
        """Add session to user's session list"""
        try:
            if not isinstance(self.redis, dict):
                # Add to front of list (most recent first)
                self.redis.lpush(self._user_sessions_key(user_id), session_id)
                # Keep only last 50 sessions
                self.redis.ltrim(self._user_sessions_key(user_id), 0, 49)
                # Set expiry on user sessions list
                self.redis.expire(self._user_sessions_key(user_id), self.session_ttl)
        except Exception as e:
            logger.error(f"Failed to add session to user list: {e}")
    
    def _update_session_summary(self, session: ConversationSession):
        """Update session summary for quick retrieval"""
        try:
            summary = SessionSummary(
                session_id=session.id,
                user_id=session.user_id,
                created_at=session.created_at,
                last_activity=session.last_activity,
                message_count=len(session.messages),
                context_summary=session.context_summary
            )
            
            if isinstance(self.redis, dict):
                # In-memory fallback
                self.redis[self._summary_key(session.id)] = summary.json()
            else:
                self.redis.setex(
                    self._summary_key(session.id),
                    self.session_ttl,
                    summary.json()
                )
                
        except Exception as e:
            logger.error(f"Failed to update session summary: {e}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions (called periodically)"""
        try:
            if isinstance(self.redis, dict):
                # In-memory fallback - no cleanup needed
                return
            
            # Redis TTL handles expiration automatically
            logger.info("Session cleanup completed (handled by Redis TTL)")
            
        except Exception as e:
            logger.error(f"Failed to cleanup sessions: {e}")

# Global conversation manager instance
conversation_manager = ConversationManager()