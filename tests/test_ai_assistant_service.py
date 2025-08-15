# tests/test_ai_assistant_service.py
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from app.ai_assistant_service import AIAssistantService
from app.models import ConversationSession, Message, ConversationContext
from app.conversation_manager import ConversationManager

class TestAIAssistantService:
    @pytest.fixture
    def mock_conversation_manager(self):
        """Create mock conversation manager"""
        manager = Mock(spec=ConversationManager)
        
        # Mock session
        mock_session = Mock(spec=ConversationSession)
        mock_session.id = "test_session_123"
        mock_session.user_id = "test_user"
        mock_session.messages = []
        
        # Mock message
        mock_message = Mock(spec=Message)
        mock_message.id = "msg_123"
        mock_message.role = "user"
        mock_message.content = "Hello"
        
        # Setup manager methods
        manager.create_session.return_value = mock_session
        manager.get_session.return_value = mock_session
        manager.add_message.return_value = mock_message
        manager.get_context.return_value = ConversationContext(
            session_id="test_session_123",
            recent_messages=[mock_message],
            total_context_length=50
        )
        manager.clear_session.return_value = True
        manager.get_user_sessions.return_value = []
        
        return manager
    
    @pytest.fixture
    def mock_studio_client(self):
        """Create mock LM Studio client"""
        client = Mock()
        client.chat.return_value = "Hello! How can I help you today?"
        client.validate_connection.return_value = {
            "status": "connected",
            "healthy": True,
            "endpoint": "http://localhost:1234"
        }
        return client
    
    @pytest.fixture
    def ai_service(self, mock_conversation_manager):
        """Create AI Assistant Service with mocked dependencies"""
        return AIAssistantService(conversation_manager=mock_conversation_manager)
    
    @pytest.mark.asyncio
    async def test_process_query_new_session(self, ai_service, mock_studio_client):
        """Test processing query with new session creation"""
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            with patch('app.ai_assistant_service.safety_filter') as mock_safety:
                mock_safety.is_safe.return_value = True
                
                result = await ai_service.process_query(
                    query="Hello, how are you?",
                    user_id="test_user"
                )
                
                assert "response" in result
                assert result["response"] == "Hello! How can I help you today?"
                assert "session_id" in result
                assert result["user_id"] == "test_user"
                assert "context_info" in result
                
                # Verify conversation manager was called
                ai_service.conversation_manager.create_session.assert_called_once_with(user_id="test_user")
                ai_service.conversation_manager.add_message.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_query_existing_session(self, ai_service, mock_studio_client):
        """Test processing query with existing session"""
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            with patch('app.ai_assistant_service.safety_filter') as mock_safety:
                mock_safety.is_safe.return_value = True
                
                result = await ai_service.process_query(
                    query="What's the weather like?",
                    session_id="existing_session",
                    user_id="test_user"
                )
                
                assert "response" in result
                assert result["session_id"] == "test_session_123"  # From mock
                
                # Verify session retrieval was attempted first
                ai_service.conversation_manager.get_session.assert_any_call("existing_session")
    
    @pytest.mark.asyncio
    async def test_process_query_with_context_enhancement(self, ai_service, mock_studio_client):
        """Test query processing with enhanced context"""
        from app.models import Message
        
        # Create proper Message instance
        test_message = Message(
            id="msg_test",
            session_id="test_session_123",
            role="user",
            content="Hello"
        )
        
        # Mock enhanced context with relevant history
        enhanced_context = ConversationContext(
            session_id="test_session_123",
            recent_messages=[test_message],
            relevant_history=["Previous conversation about weather"],
            total_context_length=100
        )
        
        ai_service.conversation_manager.get_context.return_value = enhanced_context
        
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            with patch('app.ai_assistant_service.safety_filter') as mock_safety:
                mock_safety.is_safe.return_value = True
                
                result = await ai_service.process_query(
                    query="Tell me about today's weather",
                    session_id="test_session",
                    user_id="test_user"
                )
                
                assert result["context_info"]["relevant_history_count"] == 1
                
                # Verify context was requested with query and user_id
                ai_service.conversation_manager.get_context.assert_called_with(
                    "test_session_123",
                    max_tokens=4000,
                    query="Tell me about today's weather",
                    user_id="test_user"
                )
    
    @pytest.mark.asyncio
    async def test_process_query_safety_filter(self, ai_service, mock_studio_client):
        """Test safety filtering of responses"""
        mock_studio_client.chat.return_value = "This is a harmful response"
        
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            with patch('app.ai_assistant_service.safety_filter') as mock_safety:
                mock_safety.is_safe.return_value = False  # Flag as unsafe
                
                result = await ai_service.process_query(
                    query="Tell me something harmful",
                    user_id="test_user"
                )
                
                assert "I apologize, but I cannot provide that response" in result["response"]
                mock_safety.is_safe.assert_called_once_with("This is a harmful response")
    
    @pytest.mark.asyncio
    async def test_process_query_llm_error(self, ai_service, mock_studio_client):
        """Test handling of LLM errors"""
        from app.llm_studio_client import LMStudioConnectionError
        mock_studio_client.chat.side_effect = LMStudioConnectionError("Connection failed")
        mock_studio_client.health_check.return_value = False
        
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            with patch('app.ai_assistant_service.safety_filter') as mock_safety:
                mock_safety.is_safe.return_value = True
                
                result = await ai_service.process_query(
                    query="Hello",
                    user_id="test_user"
                )
                
                assert "trouble connecting to my language model" in result["response"]
    
    @pytest.mark.asyncio
    async def test_process_query_with_health_check(self, ai_service, mock_studio_client):
        """Test process query with health check integration"""
        mock_studio_client.health_check.return_value = True
        mock_studio_client.chat.return_value = "Hello! How can I help?"
        
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            with patch('app.ai_assistant_service.safety_filter') as mock_safety:
                mock_safety.is_safe.return_value = True
                
                result = await ai_service.process_query(
                    query="Hello",
                    user_id="test_user"
                )
                
                assert result["response"] == "Hello! How can I help?"
                mock_studio_client.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_response_with_history(self, ai_service, mock_studio_client):
        """Test response generation with relevant history"""
        from app.models import Message
        
        # Create proper Message instances
        user_message = Message(
            id="msg_user",
            session_id="test_session",
            role="user",
            content="Hello"
        )
        
        assistant_message = Message(
            id="msg_assistant",
            session_id="test_session",
            role="assistant",
            content="Hi there!"
        )
        
        context = ConversationContext(
            session_id="test_session",
            recent_messages=[user_message, assistant_message],
            relevant_history=["User previously asked about weather", "Discussion about travel"],
            total_context_length=150
        )
        
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            response = await ai_service._generate_response(context, "What's the weather?")
            
            assert response == "Hello! How can I help you today?"
            
            # Verify that messages were passed to studio client
            mock_studio_client.chat.assert_called_once()
            call_args = mock_studio_client.chat.call_args
            
            # Should have system message with relevant history
            messages = call_args[1]['messages']
            assert any(msg['role'] == 'system' for msg in messages)
            
            # System message should contain relevant history
            system_msg = next(msg for msg in messages if msg['role'] == 'system')
            assert "Relevant context from previous conversations" in system_msg['content']
    
    @pytest.mark.asyncio
    async def test_generate_response_parameters(self, ai_service, mock_studio_client):
        """Test response generation with custom parameters"""
        context = ConversationContext(
            session_id="test_session",
            recent_messages=[],
            total_context_length=0
        )
        
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            await ai_service._generate_response(
                context, "Hello", 
                temperature=0.5, 
                max_tokens=256
            )
            
            # Verify parameters were passed correctly
            call_args = mock_studio_client.chat.call_args
            assert call_args[1]['temperature'] == 0.5
            assert call_args[1]['max_new_tokens'] == 256
    
    def test_validate_temperature(self, ai_service):
        """Test temperature parameter validation"""
        # Valid temperatures
        assert ai_service._validate_temperature(0.7) == 0.7
        assert ai_service._validate_temperature(0.0) == 0.0
        assert ai_service._validate_temperature(2.0) == 2.0
        
        # Clamped temperatures
        assert ai_service._validate_temperature(-0.1) == 0.0
        assert ai_service._validate_temperature(2.5) == 2.0
        
        # Invalid types
        assert ai_service._validate_temperature("invalid") == ai_service.default_temperature
        assert ai_service._validate_temperature(None) == ai_service.default_temperature
    
    def test_validate_max_tokens(self, ai_service):
        """Test max_tokens parameter validation"""
        # Valid tokens
        assert ai_service._validate_max_tokens(512) == 512
        assert ai_service._validate_max_tokens(1) == 1
        assert ai_service._validate_max_tokens(4096) == 4096
        
        # Clamped tokens
        assert ai_service._validate_max_tokens(0) == 1
        assert ai_service._validate_max_tokens(5000) == 4096
        
        # Invalid types
        assert ai_service._validate_max_tokens("invalid") == ai_service.default_max_tokens
        assert ai_service._validate_max_tokens(None) == ai_service.default_max_tokens
    
    def test_build_system_prompt(self, ai_service):
        """Test system prompt building"""
        # Basic context without history
        context = ConversationContext(
            session_id="test_session",
            recent_messages=[],
            total_context_length=0
        )
        
        prompt = ai_service._build_system_prompt(context, "Hello")
        assert "helpful, knowledgeable, and friendly AI assistant" in prompt
        assert "previous conversations" not in prompt
        
        # Context with relevant history
        context_with_history = ConversationContext(
            session_id="test_session",
            recent_messages=[],
            relevant_history=["User asked about weather", "Discussion about travel plans"],
            total_context_length=100
        )
        
        prompt_with_history = ai_service._build_system_prompt(context_with_history, "What's the weather?")
        assert "Relevant context from previous conversations" in prompt_with_history
        assert "User asked about weather" in prompt_with_history
        assert "Discussion about travel plans" in prompt_with_history
    
    def test_check_studio_health(self, ai_service, mock_studio_client):
        """Test studio health checking"""
        mock_studio_client.health_check.return_value = True
        
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            assert ai_service._check_studio_health() is True
            mock_studio_client.health_check.assert_called_once()
    
    def test_check_studio_health_failure(self, ai_service, mock_studio_client):
        """Test studio health check failure handling"""
        mock_studio_client.health_check.side_effect = Exception("Connection failed")
        
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            assert ai_service._check_studio_health() is False
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, ai_service):
        """Test getting conversation history"""
        # Mock session with messages
        mock_session = Mock()
        mock_session.id = "test_session"
        mock_session.user_id = "test_user"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        mock_session.context_summary = "Test summary"
        
        mock_message = Mock()
        mock_message.id = "msg_1"
        mock_message.role = "user"
        mock_message.content = "Hello"
        mock_message.timestamp = datetime.utcnow()
        mock_message.metadata = {}
        
        mock_session.messages = [mock_message]
        ai_service.conversation_manager.get_session.return_value = mock_session
        
        result = await ai_service.get_conversation_history("test_session")
        
        assert result["session_id"] == "test_session"
        assert result["user_id"] == "test_user"
        assert result["message_count"] == 1
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_not_found(self, ai_service):
        """Test getting history for non-existent session"""
        ai_service.conversation_manager.get_session.return_value = None
        
        result = await ai_service.get_conversation_history("nonexistent")
        
        assert "error" in result
        assert result["session_id"] == "nonexistent"
    
    @pytest.mark.asyncio
    async def test_clear_session(self, ai_service):
        """Test clearing a session"""
        ai_service.conversation_manager.clear_session.return_value = True
        
        result = await ai_service.clear_session("test_session")
        
        assert result["success"] is True
        assert result["session_id"] == "test_session"
        ai_service.conversation_manager.clear_session.assert_called_once_with("test_session")
    
    @pytest.mark.asyncio
    async def test_get_user_sessions(self, ai_service):
        """Test getting user sessions"""
        from app.models import SessionSummary
        
        mock_summary = Mock(spec=SessionSummary)
        mock_summary.session_id = "session_1"
        mock_summary.created_at = datetime.utcnow()
        mock_summary.last_activity = datetime.utcnow()
        mock_summary.message_count = 5
        mock_summary.context_summary = "Test summary"
        mock_summary.topics = ["weather", "travel"]
        
        ai_service.conversation_manager.get_user_sessions.return_value = [mock_summary]
        
        result = await ai_service.get_user_sessions("test_user")
        
        assert result["user_id"] == "test_user"
        assert result["session_count"] == 1
        assert len(result["sessions"]) == 1
        assert result["sessions"][0]["session_id"] == "session_1"
    
    @pytest.mark.asyncio
    async def test_validate_connection(self, ai_service, mock_studio_client):
        """Test connection validation"""
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            with patch('app.ai_assistant_service.conversation_memory') as mock_memory:
                mock_memory.retrieve_relevant_context.return_value = []
                mock_memory._collection = Mock()  # Simulate ChromaDB available
                mock_memory.embedder = Mock()
                
                result = await ai_service.validate_connection()
                
                assert result["overall_status"] == "healthy"
                assert result["services"]["lm_studio"]["status"] == "connected"
                assert result["services"]["redis"]["status"] == "connected"
                assert result["services"]["memory"]["status"] == "connected"
    
    @pytest.mark.asyncio
    async def test_validate_connection_degraded(self, ai_service, mock_studio_client):
        """Test connection validation with some services down"""
        mock_studio_client.validate_connection.return_value = {
            "status": "error",
            "error": "Connection failed"
        }
        
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            with patch('app.ai_assistant_service.conversation_memory') as mock_memory:
                mock_memory.retrieve_relevant_context.return_value = []
                
                result = await ai_service.validate_connection()
                
                assert result["overall_status"] == "degraded"
                assert result["services"]["lm_studio"]["status"] == "error"
    
    def test_is_safe_response(self, ai_service):
        """Test safety response checking"""
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            assert ai_service._is_safe_response("This is a safe response") is True
            mock_safety.is_safe.assert_called_once_with("This is a safe response")
    
    def test_is_safe_response_filter_error(self, ai_service):
        """Test safety filter error handling"""
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.side_effect = Exception("Filter error")
            
            # Should default to safe when filter fails
            assert ai_service._is_safe_response("Some response") is True
    
    @pytest.mark.asyncio
    async def test_process_query_context_summarization(self, ai_service, mock_studio_client):
        """Test automatic context summarization for long conversations"""
        # Mock a session with many messages
        mock_session = Mock()
        mock_session.id = "long_session_123"
        mock_session.messages = [Mock() for _ in range(25)]  # More than 20 messages
        
        # Set up the mock to return the long session when get_session is called after adding messages
        def get_session_side_effect(session_id):
            if session_id == "long_session":
                return ai_service.conversation_manager.create_session.return_value
            elif session_id == "test_session_123":  # After message is added
                return mock_session
            return None
        
        ai_service.conversation_manager.get_session.side_effect = get_session_side_effect
        
        with patch.object(ai_service, 'studio_client', mock_studio_client):
            with patch('app.ai_assistant_service.safety_filter') as mock_safety:
                mock_safety.is_safe.return_value = True
                
                await ai_service.process_query(
                    query="Hello",
                    session_id="long_session",
                    user_id="test_user"
                )
                
                # Should trigger summarization with the session ID from the created/retrieved session
                ai_service.conversation_manager.summarize_old_context.assert_called_once_with("test_session_123")

class TestAIAssistantServiceIntegration:
    """Integration tests with real components"""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test complete conversation flow with real conversation manager"""
        # Use real conversation manager but mock LLM
        service = AIAssistantService()
        
        with patch.object(service, 'studio_client') as mock_client:
            mock_client.chat.return_value = "Hello! I'm doing well, thank you for asking."
            mock_client.validate_connection.return_value = {"status": "connected"}
            
            with patch('app.ai_assistant_service.safety_filter') as mock_safety:
                mock_safety.is_safe.return_value = True
                
                # First message
                result1 = await service.process_query(
                    query="Hello, how are you?",
                    user_id="integration_test_user"
                )
                
                assert "response" in result1
                session_id = result1["session_id"]
                
                # Second message in same session
                result2 = await service.process_query(
                    query="What's your favorite color?",
                    session_id=session_id,
                    user_id="integration_test_user"
                )
                
                assert result2["session_id"] == session_id
                
                # Get conversation history
                history = await service.get_conversation_history(session_id)
                assert len(history["messages"]) == 4  # 2 user + 2 assistant
                
                # Clean up
                await service.clear_session(session_id)