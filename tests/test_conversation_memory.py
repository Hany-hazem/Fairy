# tests/test_conversation_memory.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np

from app.conversation_memory import ConversationMemoryManager
from app.models import ConversationSession, Message, ConversationContext

class TestConversationMemoryManager:
    @pytest.fixture
    def memory_manager(self):
        """Create memory manager with mocked dependencies"""
        with patch('app.conversation_memory.CHROMADB_AVAILABLE', False):
            manager = ConversationMemoryManager()
            return manager
    
    @pytest.fixture
    def sample_session(self):
        """Create a sample conversation session"""
        session = ConversationSession(user_id="test_user")
        session.add_message("user", "Hello, I want to know about the weather today.")
        session.add_message("assistant", "I'd be happy to help you with weather information! However, I don't have access to real-time weather data.")
        session.add_message("user", "What about cooking recipes?")
        session.add_message("assistant", "I can definitely help with cooking recipes! What type of cuisine are you interested in?")
        return session
    
    def test_initialization_without_chromadb(self, memory_manager):
        """Test initialization when ChromaDB is not available"""
        assert hasattr(memory_manager, '_memory_store')
        assert isinstance(memory_manager._memory_store, list)
        assert memory_manager.embedder is not None
    
    def test_embed_text(self, memory_manager):
        """Test text embedding functionality"""
        with patch.object(memory_manager.embedder, 'encode') as mock_encode:
            mock_encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            
            result = memory_manager._embed_text(["hello", "world"])
            
            assert result.shape == (2, 3)
            mock_encode.assert_called_once_with(["hello", "world"], convert_to_numpy=True)
    
    def test_embed_text_error_handling(self, memory_manager):
        """Test embedding error handling"""
        with patch.object(memory_manager.embedder, 'encode') as mock_encode:
            mock_encode.side_effect = Exception("Embedding failed")
            
            result = memory_manager._embed_text(["hello"])
            
            # Should return zero vector as fallback
            assert result.shape == (1, 384)  # all-MiniLM-L6-v2 dimension
            assert np.all(result == 0)
    
    def test_extract_topics(self, memory_manager):
        """Test topic extraction from text"""
        text = "I want to know about the weather and cooking recipes for dinner"
        topics = memory_manager._extract_topics(text)
        
        assert "weather" in topics
        assert "cooking" in topics
        assert "recipe" in topics
    
    def test_create_context_chunks(self, memory_manager, sample_session):
        """Test creating context chunks from conversation"""
        chunks = memory_manager._create_context_chunks(sample_session)
        
        assert len(chunks) == 2  # Two user messages
        
        # First chunk
        assert "Hello, I want to know about the weather today." in chunks[0]["text"]
        assert "I'd be happy to help you with weather information!" in chunks[0]["text"]
        assert "weather" in chunks[0]["topics"]
        
        # Second chunk
        assert "What about cooking recipes?" in chunks[1]["text"]
        assert "I can definitely help with cooking recipes!" in chunks[1]["text"]
        assert "cooking" in chunks[1]["topics"]
    
    def test_store_conversation_context(self, memory_manager, sample_session):
        """Test storing conversation context"""
        memory_manager.store_conversation_context(sample_session)
        
        # Should have stored chunks in memory
        assert len(memory_manager._memory_store) > 0
        
        # Check stored data structure
        stored_item = memory_manager._memory_store[0]
        assert "text" in stored_item
        assert "embedding" in stored_item
        assert stored_item["session_id"] == sample_session.id
        assert stored_item["user_id"] == sample_session.user_id
    
    def test_retrieve_relevant_context(self, memory_manager, sample_session):
        """Test retrieving relevant context"""
        # Store some context first
        memory_manager.store_conversation_context(sample_session)
        
        # Retrieve context related to weather
        results = memory_manager.retrieve_relevant_context("weather forecast", k=2)
        
        assert len(results) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        
        # Check that results contain text and similarity scores
        text, score = results[0]
        assert isinstance(text, str)
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_retrieve_relevant_context_with_filters(self, memory_manager, sample_session):
        """Test retrieving context with session and user filters"""
        # Store context
        memory_manager.store_conversation_context(sample_session)
        
        # Test session filter
        results = memory_manager.retrieve_relevant_context(
            "cooking", session_id=sample_session.id, k=2
        )
        assert len(results) > 0
        
        # Test user filter
        results = memory_manager.retrieve_relevant_context(
            "cooking", user_id=sample_session.user_id, k=2
        )
        assert len(results) > 0
        
        # Test with non-matching filters
        results = memory_manager.retrieve_relevant_context(
            "cooking", session_id="nonexistent", k=2
        )
        assert len(results) == 0
    
    def test_format_conversation_for_summary(self, memory_manager, sample_session):
        """Test formatting conversation for summarization"""
        formatted = memory_manager._format_conversation_for_summary(sample_session)
        
        assert "User:" in formatted
        assert "Assistant:" in formatted
        assert "weather" in formatted.lower()
        assert "cooking" in formatted.lower()
    
    def test_create_simple_summary(self, memory_manager, sample_session):
        """Test creating simple rule-based summary"""
        summary = memory_manager._create_simple_summary(sample_session, max_length=200)
        
        assert isinstance(summary, str)
        assert len(summary) <= 200
        assert "User discussed" in summary or "Assistant provided" in summary
        assert "weather" in summary or "cooking" in summary
    
    def test_generate_context_summary_fallback(self, memory_manager, sample_session):
        """Test context summary generation with LLM fallback"""
        with patch('app.conversation_memory.get_studio_client') as mock_client:
            # Mock LLM client to raise exception (test fallback)
            mock_client.side_effect = Exception("LLM not available")
            
            summary = memory_manager.generate_context_summary(sample_session)
            
            # Should fall back to simple summary
            assert isinstance(summary, str)
            assert len(summary) > 0
    
    def test_generate_context_summary_with_llm(self, memory_manager, sample_session):
        """Test context summary generation with LLM"""
        with patch('app.conversation_memory.get_studio_client') as mock_get_client:
            mock_client = Mock()
            mock_client.chat.return_value = "User asked about weather and cooking recipes. Assistant provided helpful responses about both topics."
            mock_get_client.return_value = mock_client
            
            summary = memory_manager.generate_context_summary(sample_session)
            
            assert summary == "User asked about weather and cooking recipes. Assistant provided helpful responses about both topics."
            mock_client.chat.assert_called_once()
    
    def test_enhance_context_with_history(self, memory_manager, sample_session):
        """Test enhancing context with relevant history"""
        # Store some context first
        memory_manager.store_conversation_context(sample_session)
        
        # Create base context
        context = ConversationContext(
            session_id="new_session",
            recent_messages=[],
            total_context_length=0
        )
        
        # Enhance with history
        enhanced_context = memory_manager.enhance_context_with_history(
            context, "weather information", sample_session.user_id
        )
        
        assert enhanced_context.session_id == "new_session"
        # Should have relevant history if similarity is high enough
        assert isinstance(enhanced_context.relevant_history, list)
    
    def test_cleanup_old_memories(self, memory_manager, sample_session):
        """Test cleaning up old memories"""
        # Store some context
        memory_manager.store_conversation_context(sample_session)
        original_count = len(memory_manager._memory_store)
        
        # Modify timestamp to make it old
        if memory_manager._memory_store:
            memory_manager._memory_store[0]["timestamp"] = datetime.utcnow() - timedelta(days=35)
        
        # Cleanup memories older than 30 days
        memory_manager.cleanup_old_memories(days=30)
        
        # Should have fewer items
        assert len(memory_manager._memory_store) < original_count
    
    def test_store_chunk_error_handling(self, memory_manager):
        """Test error handling in chunk storage"""
        # Create a chunk that might cause issues
        chunk = {
            "text": "test",
            "user_message_id": "test_id",
            "assistant_message_id": None,
            "timestamp": datetime.utcnow(),
            "topics": ["test"]
        }
        
        # Mock embedding to fail
        with patch.object(memory_manager, '_embed_text') as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")
            
            # Should not raise exception
            memory_manager._store_chunk(chunk, "session_id", "user_id")
            
            # Memory store should remain empty due to error
            assert len(memory_manager._memory_store) == 0

class TestConversationMemoryManagerWithChromaDB:
    """Tests for ChromaDB integration (mocked)"""
    
    @pytest.fixture
    def chromadb_manager(self):
        """Create memory manager with mocked ChromaDB"""
        with patch('app.conversation_memory.CHROMADB_AVAILABLE', True):
            with patch('app.conversation_memory.chromadb.PersistentClient') as mock_client:
                mock_collection = Mock()
                mock_client.return_value.get_or_create_collection.return_value = mock_collection
                
                manager = ConversationMemoryManager()
                manager._collection = mock_collection
                return manager, mock_collection
    
    def test_store_chunk_with_chromadb(self, chromadb_manager):
        """Test storing chunk with ChromaDB"""
        manager, mock_collection = chromadb_manager
        
        chunk = {
            "text": "test conversation",
            "user_message_id": "user_123",
            "assistant_message_id": "assistant_456",
            "timestamp": datetime.utcnow(),
            "topics": ["test", "conversation"]
        }
        
        with patch.object(manager, '_embed_text') as mock_embed:
            mock_embed.return_value = np.array([[0.1, 0.2, 0.3]])
            
            manager._store_chunk(chunk, "session_123", "user_456")
            
            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args
            
            assert call_args[1]["documents"] == ["test conversation"]
            assert len(call_args[1]["embeddings"]) == 1
            assert call_args[1]["ids"][0] == "session_123_user_123"
    
    def test_retrieve_with_chromadb(self, chromadb_manager):
        """Test retrieving context with ChromaDB"""
        manager, mock_collection = chromadb_manager
        
        # Mock ChromaDB query response
        mock_collection.query.return_value = {
            "documents": [["conversation about weather", "discussion about cooking"]],
            "distances": [[0.2, 0.4]]
        }
        
        with patch.object(manager, '_embed_text') as mock_embed:
            mock_embed.return_value = np.array([[0.1, 0.2, 0.3]])
            
            results = manager.retrieve_relevant_context("weather", k=2)
            
            assert len(results) == 2
            assert results[0][0] == "conversation about weather"
            assert results[0][1] == 0.8  # 1 - 0.2
            assert results[1][0] == "discussion about cooking"
            assert results[1][1] == 0.6  # 1 - 0.4