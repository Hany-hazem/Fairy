# app/ai_assistant_service.py
"""
Core AI Assistant Service that orchestrates conversation management,
context retrieval, and LLM interactions
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from .config import settings
from .models import ConversationSession, Message, ConversationContext
from .conversation_manager import ConversationManager
from .llm_studio_client import get_studio_client, LMStudioConnectionError
from .safety_filter import safety_filter
from .conversation_memory import conversation_memory
from .performance_monitor import performance_monitor

logger = logging.getLogger(__name__)

class AIAssistantService:
    """
    Core AI Assistant Service that provides conversational AI capabilities
    with context management, memory retrieval, and safety filtering
    """
    
    def __init__(self, conversation_manager: Optional[ConversationManager] = None):
        self.conversation_manager = conversation_manager or ConversationManager()
        self._studio_client = None
        
        # Configuration
        self.max_context_tokens = 4000  # Leave room for response
        self.default_temperature = 0.7
        self.default_max_tokens = 512
        
        logger.info("AI Assistant Service initialized")
    
    @property
    def studio_client(self):
        """Lazy initialization of LM Studio client with health monitoring"""
        if self._studio_client is None:
            try:
                self._studio_client = get_studio_client()
                logger.info("LM Studio client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LM Studio client: {e}")
                raise LMStudioConnectionError(f"Could not initialize LM Studio client: {e}")
        return self._studio_client
    
    @studio_client.setter
    def studio_client(self, value):
        """Allow setting studio client for testing"""
        self._studio_client = value
    
    @studio_client.deleter
    def studio_client(self):
        """Allow deleting studio client for testing"""
        self._studio_client = None
    
    async def process_query(self, query: str, session_id: str = None, 
                          user_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Process a user query and return an AI response with full context
        
        Args:
            query: User's input query
            session_id: Optional existing session ID
            user_id: Optional user identifier
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Dict containing response, session info, and metadata
        """
        request_id = f"query_{session_id or 'new'}_{int(datetime.utcnow().timestamp())}"
        
        with performance_monitor.track_request(
            request_id, 
            "ai_assistant_query",
            user_id=user_id,
            session_id=session_id,
            query_length=len(query)
        ):
            try:
                # Create or retrieve session
                if session_id:
                    session = self.conversation_manager.get_session(session_id)
                    if not session:
                        logger.warning(f"Session {session_id} not found, creating new session")
                        session = self.conversation_manager.create_session(user_id=user_id)
                else:
                    session = self.conversation_manager.create_session(user_id=user_id)
                
                # Add user message to conversation
                user_message = self.conversation_manager.add_message(
                    session.id, "user", query, metadata={"timestamp": datetime.utcnow().isoformat()}
                )
                
                if not user_message:
                    raise Exception("Failed to add user message to session")
                
                # Get enhanced context with relevant history
                context = self.conversation_manager.get_context(
                    session.id,
                    max_tokens=self.max_context_tokens,
                    query=query,
                    user_id=user_id
                )
                
                # Pre-flight check for LM Studio connectivity
                if not self._check_studio_health():
                    logger.warning("LM Studio health check failed, attempting to proceed anyway")
                
                # Generate AI response with performance tracking
                llm_start_time = time.time()
                try:
                    response_text = await self._generate_response(context, query, **kwargs)
                    llm_duration = time.time() - llm_start_time
                    
                    # Record successful LLM performance
                    performance_monitor.record_response_time(
                        "llm_generation", 
                        llm_duration,
                        context_length=context.total_context_length,
                        response_length=len(response_text)
                    )
                    
                except Exception as llm_error:
                    llm_duration = time.time() - llm_start_time
                    logger.error(f"LLM generation failed after {llm_duration:.2f}s: {llm_error}")
                    
                    # Record failed attempt
                    performance_monitor.record_response_time(
                        "llm_generation_failed", 
                        llm_duration,
                        context_length=context.total_context_length,
                        error=str(llm_error)
                    )
                    
                    # Return appropriate error response
                    if isinstance(llm_error, LMStudioConnectionError):
                        response_text = "I apologize, but I'm having trouble connecting to my language model. Please try again later."
                    else:
                        response_text = "I apologize, but I encountered an error while generating a response. Please try again."
                
                # Safety check
                if not self._is_safe_response(response_text):
                    response_text = "I apologize, but I cannot provide that response due to safety guidelines."
                    logger.warning(f"Response filtered for safety in session {session.id}")
                
                # Add assistant response to conversation
                assistant_message = self.conversation_manager.add_message(
                    session.id, "assistant", response_text, 
                    metadata={
                        "timestamp": datetime.utcnow().isoformat(),
                        "context_length": context.total_context_length,
                        "relevant_history_count": len(context.relevant_history)
                    }
                )
                
                # Check if we need to summarize old context
                updated_session = self.conversation_manager.get_session(session.id)
                if updated_session and len(updated_session.messages) > 20:
                    try:
                        self.conversation_manager.summarize_old_context(session.id)
                        logger.info(f"Summarized old context for session {session.id}")
                    except Exception as e:
                        logger.warning(f"Failed to summarize context: {e}")
                
                return {
                    "response": response_text,
                    "session_id": session.id,
                    "user_id": user_id,
                    "message_id": assistant_message.id if assistant_message else None,
                    "context_info": {
                        "context_length": context.total_context_length,
                        "relevant_history_count": len(context.relevant_history),
                        "has_summary": context.context_summary is not None
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                return {
                    "error": str(e),
                    "session_id": session_id,
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
    
    async def _generate_response(self, context: ConversationContext, 
                               query: str, **kwargs) -> str:
        """Generate AI response using LM Studio with enhanced context and prompt engineering"""
        try:
            # Prepare messages for LLM
            messages = context.to_llm_messages()
            
            # Enhanced prompt engineering with system message
            system_prompt = self._build_system_prompt(context, query)
            
            # Insert or merge system message
            if messages and messages[0]["role"] == "system":
                # Merge with existing system message
                messages[0]["content"] = f"{messages[0]['content']}\n\n{system_prompt}"
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            # Extract parameters with validation
            temperature = self._validate_temperature(kwargs.get("temperature", self.default_temperature))
            max_tokens = self._validate_max_tokens(kwargs.get("max_tokens", self.default_max_tokens))
            
            logger.debug(f"Generating response with {len(messages)} messages, temp={temperature}, max_tokens={max_tokens}")
            
            # Generate response using LM Studio with retry logic
            response = self.studio_client.chat(
                prompt="",  # Not used when messages provided
                messages=messages,
                temperature=temperature,
                max_new_tokens=max_tokens
            )
            
            logger.debug(f"Generated response: {len(response)} characters")
            return response
            
        except LMStudioConnectionError as e:
            logger.error(f"LM Studio connection error: {e}")
            return "I apologize, but I'm having trouble connecting to my language model. Please try again later."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _build_system_prompt(self, context: ConversationContext, query: str) -> str:
        """Build enhanced system prompt with context and conversation guidelines"""
        base_prompt = """You are a helpful, knowledgeable, and friendly AI assistant. Your goal is to provide accurate, helpful, and contextually relevant responses to users.

Guidelines:
- Be conversational and engaging while maintaining professionalism
- Use the conversation history to provide personalized responses
- If you're unsure about something, acknowledge it honestly
- Keep responses concise but comprehensive
- Ask clarifying questions when needed"""
        
        # Add relevant history context if available
        if context.relevant_history:
            history_context = "\n".join([
                f"- {item[:150]}..." if len(item) > 150 else f"- {item}"
                for item in context.relevant_history[:3]  # Limit to top 3 most relevant
            ])
            
            context_prompt = f"""

Relevant context from previous conversations:
{history_context}

Use this context to provide more informed and personalized responses, but don't explicitly mention that you're referencing previous conversations unless directly relevant."""
            
            base_prompt += context_prompt
        
        # Add session context if this is a continuing conversation
        if context.recent_messages and len(context.recent_messages) > 1:
            base_prompt += "\n\nThis is a continuing conversation. Maintain consistency with the established context and tone."
        
        return base_prompt
    
    def _validate_temperature(self, temperature: float) -> float:
        """Validate and clamp temperature parameter"""
        if not isinstance(temperature, (int, float)):
            logger.warning(f"Invalid temperature type: {type(temperature)}, using default")
            return self.default_temperature
        
        # Clamp between 0.0 and 2.0
        clamped = max(0.0, min(2.0, float(temperature)))
        if clamped != temperature:
            logger.warning(f"Temperature {temperature} clamped to {clamped}")
        
        return clamped
    
    def _validate_max_tokens(self, max_tokens: int) -> int:
        """Validate and clamp max_tokens parameter"""
        if not isinstance(max_tokens, (int, float)):
            logger.warning(f"Invalid max_tokens type: {type(max_tokens)}, using default")
            return self.default_max_tokens
        
        # Clamp between 1 and 4096
        clamped = max(1, min(4096, int(max_tokens)))
        if clamped != max_tokens:
            logger.warning(f"Max tokens {max_tokens} clamped to {clamped}")
        
        return clamped
    
    def _check_studio_health(self) -> bool:
        """Check LM Studio health with caching to avoid excessive calls"""
        try:
            return self.studio_client.health_check()
        except Exception as e:
            logger.warning(f"Studio health check failed: {e}")
            return False
    
    def _is_safe_response(self, response: str) -> bool:
        """Check if response passes safety filters"""
        try:
            return safety_filter.is_safe(response)
        except Exception as e:
            logger.warning(f"Safety filter error: {e}")
            # Default to safe if filter fails
            return True
    
    async def get_conversation_history(self, session_id: str, 
                                     limit: int = 50) -> Dict[str, Any]:
        """Get conversation history for a session"""
        try:
            session = self.conversation_manager.get_session(session_id)
            if not session:
                return {"error": "Session not found", "session_id": session_id}
            
            # Get recent messages
            messages = session.messages[-limit:] if limit else session.messages
            
            return {
                "session_id": session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "message_count": len(session.messages),
                "context_summary": session.context_summary,
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata
                    }
                    for msg in messages
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return {"error": str(e), "session_id": session_id}
    
    async def clear_session(self, session_id: str) -> Dict[str, Any]:
        """Clear a conversation session"""
        try:
            success = self.conversation_manager.clear_session(session_id)
            return {
                "success": success,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_user_sessions(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get recent sessions for a user"""
        try:
            sessions = self.conversation_manager.get_user_sessions(user_id, limit)
            
            return {
                "user_id": user_id,
                "session_count": len(sessions),
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "created_at": session.created_at.isoformat(),
                        "last_activity": session.last_activity.isoformat(),
                        "message_count": session.message_count,
                        "context_summary": session.context_summary,
                        "topics": session.topics
                    }
                    for session in sessions
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def validate_connection(self) -> Dict[str, Any]:
        """Validate connections to all required services"""
        try:
            # Test LM Studio connection
            studio_status = self.studio_client.validate_connection()
            
            # Test conversation manager (Redis)
            test_session = self.conversation_manager.create_session(user_id="test")
            redis_status = {
                "status": "connected" if test_session else "error",
                "test_session_created": test_session is not None
            }
            if test_session:
                self.conversation_manager.clear_session(test_session.id)
            
            # Test memory system
            try:
                memory_results = conversation_memory.retrieve_relevant_context("test", k=1)
                memory_status = {
                    "status": "connected",
                    "vector_db_available": hasattr(conversation_memory, '_collection') and conversation_memory._collection is not None,
                    "embedder_available": conversation_memory.embedder is not None
                }
            except Exception as e:
                memory_status = {
                    "status": "error",
                    "error": str(e)
                }
            
            return {
                "overall_status": "healthy" if all([
                    studio_status.get("status") == "connected",
                    redis_status.get("status") == "connected",
                    memory_status.get("status") == "connected"
                ]) else "degraded",
                "services": {
                    "lm_studio": studio_status,
                    "redis": redis_status,
                    "memory": memory_status
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error validating connections: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Global AI Assistant Service instance
ai_assistant = AIAssistantService()