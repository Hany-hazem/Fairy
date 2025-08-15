# tests/test_e2e_conversation_integration.py
"""
End-to-end conversation integration tests with real LM Studio integration
Tests full conversation flows, multi-session management, context persistence,
and performance benchmarking for conversation flows.

Requirements: 1.1, 1.2, 1.4, 2.1, 2.2, 2.3, 2.4
"""

import pytest
import asyncio
import time
import json
import logging
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from app.ai_assistant_service import AIAssistantService
from app.conversation_manager import ConversationManager
from app.llm_studio_client import LMStudioClient, LMStudioConfig, LMStudioConnectionError
from app.models import ConversationSession, Message, ConversationContext
from app.config import settings

logger = logging.getLogger(__name__)

class TestE2EConversationIntegration:
    """End-to-end conversation integration tests"""
    
    @pytest.fixture
    def real_lm_studio_client(self):
        """Create real LM Studio client for integration testing"""
        config = LMStudioConfig(
            endpoint_url="http://localhost:1234",
            model_name="gpt-oss-20b",
            temperature=0.7,
            max_tokens=512,
            timeout=30,
            retry_attempts=2
        )
        return LMStudioClient(config)
    
    @pytest.fixture
    def mock_lm_studio_client(self):
        """Create mock LM Studio client with realistic responses"""
        client = Mock(spec=LMStudioClient)
        
        # Mock realistic conversation responses
        responses = [
            "Hello! I'm doing well, thank you for asking. How can I help you today?",
            "That's a great question! Let me think about that for a moment.",
            "Based on our previous conversation, I remember you were interested in that topic.",
            "I understand what you're looking for. Here's what I can tell you about that.",
            "That's an interesting follow-up to what we discussed earlier."
        ]
        
        response_index = 0
        def mock_chat(*args, **kwargs):
            nonlocal response_index
            response = responses[response_index % len(responses)]
            response_index += 1
            return response
        
        client.chat = mock_chat
        client.health_check.return_value = True
        client.validate_connection.return_value = {
            "status": "connected",
            "healthy": True,
            "endpoint": "http://localhost:1234"
        }
        
        return client
    
    @pytest.fixture
    def ai_service_with_mock_llm(self, mock_lm_studio_client):
        """Create AI service with mocked LLM for reliable testing"""
        service = AIAssistantService()
        service.studio_client = mock_lm_studio_client
        return service
    
    @pytest.fixture
    def ai_service_with_real_llm(self, real_lm_studio_client):
        """Create AI service with real LLM for integration testing"""
        service = AIAssistantService()
        service.studio_client = real_lm_studio_client
        return service
    
    @pytest.mark.asyncio
    async def test_single_conversation_flow(self, ai_service_with_mock_llm):
        """Test complete single conversation flow with context management"""
        service = ai_service_with_mock_llm
        
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            # Start conversation
            result1 = await service.process_query(
                query="Hello, how are you doing today?",
                user_id="test_user_single"
            )
            
            assert "response" in result1
            assert "session_id" in result1
            assert result1["user_id"] == "test_user_single"
            session_id = result1["session_id"]
            
            # Continue conversation with context
            result2 = await service.process_query(
                query="What's the weather like?",
                session_id=session_id,
                user_id="test_user_single"
            )
            
            assert result2["session_id"] == session_id
            assert "response" in result2
            
            # Ask follow-up question
            result3 = await service.process_query(
                query="Can you tell me more about that?",
                session_id=session_id,
                user_id="test_user_single"
            )
            
            assert result3["session_id"] == session_id
            
            # Verify conversation history
            history = await service.get_conversation_history(session_id)
            assert history["session_id"] == session_id
            assert len(history["messages"]) == 6  # 3 user + 3 assistant
            
            # Verify message order and content
            messages = history["messages"]
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "Hello, how are you doing today?"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            assert messages[2]["content"] == "What's the weather like?"
            
            # Clean up
            await service.clear_session(session_id)
    
    @pytest.mark.asyncio
    async def test_multi_session_conversation_management(self, ai_service_with_mock_llm):
        """Test managing multiple concurrent conversation sessions"""
        service = ai_service_with_mock_llm
        
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            # Create multiple sessions for different users
            sessions = {}
            
            # User 1 - Session 1
            result1 = await service.process_query(
                query="Hello, I'm interested in learning about Python",
                user_id="user1"
            )
            sessions["user1_session1"] = result1["session_id"]
            
            # User 2 - Session 1
            result2 = await service.process_query(
                query="Hi, I need help with JavaScript",
                user_id="user2"
            )
            sessions["user2_session1"] = result2["session_id"]
            
            # User 1 - Session 2 (new topic)
            result3 = await service.process_query(
                query="Actually, let's talk about machine learning instead",
                user_id="user1"
            )
            sessions["user1_session2"] = result3["session_id"]
            
            # Verify all sessions are different
            session_ids = list(sessions.values())
            assert len(set(session_ids)) == 3, "All sessions should have unique IDs"
            
            # Continue conversations in each session
            await service.process_query(
                query="Can you explain Python classes?",
                session_id=sessions["user1_session1"],
                user_id="user1"
            )
            
            await service.process_query(
                query="What are JavaScript promises?",
                session_id=sessions["user2_session1"],
                user_id="user2"
            )
            
            await service.process_query(
                query="What's the difference between supervised and unsupervised learning?",
                session_id=sessions["user1_session2"],
                user_id="user1"
            )
            
            # Verify session isolation
            history1 = await service.get_conversation_history(sessions["user1_session1"])
            history2 = await service.get_conversation_history(sessions["user2_session1"])
            history3 = await service.get_conversation_history(sessions["user1_session2"])
            
            # Each session should have its own context
            assert "Python" in str(history1["messages"])
            assert "JavaScript" in str(history2["messages"])
            assert "machine learning" in str(history3["messages"])
            
            # Verify no cross-contamination
            assert "JavaScript" not in str(history1["messages"])
            assert "Python" not in str(history2["messages"])
            assert "Python classes" not in str(history3["messages"])
            
            # Clean up all sessions
            for session_id in sessions.values():
                await service.clear_session(session_id)
    
    @pytest.mark.asyncio
    async def test_context_persistence_and_retrieval(self, ai_service_with_mock_llm):
        """Test context persistence across conversation turns and retrieval validation"""
        service = ai_service_with_mock_llm
        
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            # Start conversation with specific context
            result1 = await service.process_query(
                query="I'm working on a project about renewable energy, specifically solar panels",
                user_id="context_test_user"
            )
            session_id = result1["session_id"]
            
            # Add more context
            await service.process_query(
                query="I'm particularly interested in efficiency improvements over the last decade",
                session_id=session_id,
                user_id="context_test_user"
            )
            
            await service.process_query(
                query="My budget is around $10,000 for a residential installation",
                session_id=session_id,
                user_id="context_test_user"
            )
            
            # Test context retrieval
            context = service.conversation_manager.get_context(
                session_id, 
                max_tokens=2000,
                query="What should I consider for my solar panel project?",
                user_id="context_test_user"
            )
            
            assert context.session_id == session_id
            assert len(context.recent_messages) > 0
            assert context.total_context_length > 0
            
            # Verify context contains relevant information
            context_text = " ".join([msg.content for msg in context.recent_messages])
            assert "renewable energy" in context_text
            assert "solar panels" in context_text
            assert "efficiency" in context_text
            assert "$10,000" in context_text
            
            # Test context-aware response
            result4 = await service.process_query(
                query="Based on what we discussed, what would you recommend?",
                session_id=session_id,
                user_id="context_test_user"
            )
            
            # Verify context information is included in response metadata
            assert result4["context_info"]["context_length"] > 0
            assert result4["context_info"]["relevant_history_count"] >= 0
            
            # Clean up
            await service.clear_session(session_id)
    
    @pytest.mark.asyncio
    async def test_long_conversation_context_management(self, ai_service_with_mock_llm):
        """Test context management for very long conversations with summarization"""
        service = ai_service_with_mock_llm
        
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            # Create a long conversation
            result = await service.process_query(
                query="Let's start a long conversation about various topics",
                user_id="long_conv_user"
            )
            session_id = result["session_id"]
            
            # Add many messages to trigger summarization
            topics = [
                "Tell me about artificial intelligence",
                "What about machine learning algorithms?",
                "How does deep learning work?",
                "Explain neural networks",
                "What are transformers in AI?",
                "Tell me about natural language processing",
                "How does computer vision work?",
                "What is reinforcement learning?",
                "Explain generative AI models",
                "What about large language models?",
                "How do recommendation systems work?",
                "Tell me about data preprocessing",
                "What is feature engineering?",
                "Explain model evaluation metrics",
                "How do you prevent overfitting?",
                "What is cross-validation?",
                "Tell me about ensemble methods",
                "How does gradient descent work?",
                "What are activation functions?",
                "Explain backpropagation",
                "What is transfer learning?",
                "How do GANs work?",
                "Tell me about attention mechanisms"
            ]
            
            for i, topic in enumerate(topics):
                await service.process_query(
                    query=topic,
                    session_id=session_id,
                    user_id="long_conv_user"
                )
                
                # Check if summarization was triggered (after 20+ messages)
                if i > 10:  # After several exchanges
                    history = await service.get_conversation_history(session_id)
                    if history.get("context_summary"):
                        logger.info(f"Context summarization triggered after {i+1} topics")
                        break
            
            # Verify final conversation state
            final_history = await service.get_conversation_history(session_id)
            
            # Should have many messages
            assert len(final_history["messages"]) > 20
            
            # May have context summary if summarization was triggered
            if final_history.get("context_summary"):
                assert len(final_history["context_summary"]) > 0
                logger.info(f"Final context summary: {final_history['context_summary']}")
            
            # Test that context retrieval still works efficiently
            context = service.conversation_manager.get_context(
                session_id,
                max_tokens=4000,
                query="Summarize what we've discussed about AI",
                user_id="long_conv_user"
            )
            
            assert context.total_context_length <= 4000
            assert len(context.recent_messages) > 0
            
            # Clean up
            await service.clear_session(session_id)
    
    @pytest.mark.asyncio
    async def test_conversation_performance_benchmarking(self, ai_service_with_mock_llm):
        """Test and benchmark conversation performance metrics"""
        service = ai_service_with_mock_llm
        
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            # Performance metrics collection
            metrics = {
                "response_times": [],
                "context_lengths": [],
                "session_creation_times": [],
                "memory_usage": []
            }
            
            # Test session creation performance
            session_creation_start = time.time()
            result = await service.process_query(
                query="Hello, let's test performance",
                user_id="perf_test_user"
            )
            session_creation_time = time.time() - session_creation_start
            metrics["session_creation_times"].append(session_creation_time)
            
            session_id = result["session_id"]
            
            # Test multiple query response times
            test_queries = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "Explain deep learning concepts",
                "What are neural networks?",
                "Tell me about natural language processing",
                "How do recommendation systems work?",
                "What is computer vision?",
                "Explain reinforcement learning",
                "What are large language models?",
                "How does transfer learning work?"
            ]
            
            for query in test_queries:
                start_time = time.time()
                
                result = await service.process_query(
                    query=query,
                    session_id=session_id,
                    user_id="perf_test_user"
                )
                
                response_time = time.time() - start_time
                metrics["response_times"].append(response_time)
                
                # Collect context length metrics
                if "context_info" in result:
                    metrics["context_lengths"].append(result["context_info"]["context_length"])
            
            # Analyze performance metrics
            avg_response_time = sum(metrics["response_times"]) / len(metrics["response_times"])
            max_response_time = max(metrics["response_times"])
            min_response_time = min(metrics["response_times"])
            
            avg_context_length = sum(metrics["context_lengths"]) / len(metrics["context_lengths"])
            max_context_length = max(metrics["context_lengths"])
            
            # Performance assertions
            assert avg_response_time < 5.0, f"Average response time too high: {avg_response_time:.2f}s"
            assert max_response_time < 10.0, f"Max response time too high: {max_response_time:.2f}s"
            assert session_creation_time < 2.0, f"Session creation too slow: {session_creation_time:.2f}s"
            
            # Context management assertions
            assert avg_context_length > 0, "Context should be maintained"
            assert max_context_length < 10000, f"Context length too large: {max_context_length}"
            
            # Log performance summary
            logger.info(f"Performance Benchmark Results:")
            logger.info(f"  Average response time: {avg_response_time:.3f}s")
            logger.info(f"  Min/Max response time: {min_response_time:.3f}s / {max_response_time:.3f}s")
            logger.info(f"  Session creation time: {session_creation_time:.3f}s")
            logger.info(f"  Average context length: {avg_context_length:.0f} chars")
            logger.info(f"  Max context length: {max_context_length:.0f} chars")
            
            # Clean up
            await service.clear_session(session_id)
    
    @pytest.mark.asyncio
    async def test_conversation_error_handling_and_recovery(self, ai_service_with_mock_llm):
        """Test conversation error handling and recovery scenarios"""
        service = ai_service_with_mock_llm
        
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            # Test LLM connection error handling
            service.studio_client.chat.side_effect = LMStudioConnectionError("Connection failed")
            
            result = await service.process_query(
                query="This should fail gracefully",
                user_id="error_test_user"
            )
            
            assert "response" in result
            assert "trouble connecting" in result["response"]
            session_id = result["session_id"]
            
            # Test recovery after connection is restored
            service.studio_client.chat.side_effect = None
            service.studio_client.chat.return_value = "Connection restored, I can help you now!"
            
            recovery_result = await service.process_query(
                query="Are you working now?",
                session_id=session_id,
                user_id="error_test_user"
            )
            
            assert recovery_result["session_id"] == session_id
            assert "Connection restored" in recovery_result["response"]
            
            # Test safety filter rejection
            service.studio_client.chat.return_value = "Potentially harmful content"
            mock_safety.is_safe.return_value = False
            
            safety_result = await service.process_query(
                query="Generate harmful content",
                session_id=session_id,
                user_id="error_test_user"
            )
            
            assert "cannot provide that response" in safety_result["response"]
            
            # Verify conversation history includes error handling
            history = await service.get_conversation_history(session_id)
            assert len(history["messages"]) >= 6  # All interactions recorded
            
            # Clean up
            await service.clear_session(session_id)
    
    @pytest.mark.asyncio
    async def test_concurrent_conversation_sessions(self, ai_service_with_mock_llm):
        """Test handling multiple concurrent conversation sessions"""
        service = ai_service_with_mock_llm
        
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            # Create multiple concurrent sessions
            async def create_conversation_session(user_id: str, topic: str):
                result = await service.process_query(
                    query=f"Let's talk about {topic}",
                    user_id=user_id
                )
                session_id = result["session_id"]
                
                # Continue conversation
                for i in range(3):
                    await service.process_query(
                        query=f"Tell me more about {topic} - question {i+1}",
                        session_id=session_id,
                        user_id=user_id
                    )
                
                return session_id, user_id, topic
            
            # Run concurrent conversations
            tasks = [
                create_conversation_session("user1", "Python programming"),
                create_conversation_session("user2", "Machine learning"),
                create_conversation_session("user3", "Web development"),
                create_conversation_session("user4", "Data science"),
                create_conversation_session("user5", "Cloud computing")
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all sessions completed successfully
            assert len(results) == 5
            
            # Verify session isolation
            for session_id, user_id, topic in results:
                history = await service.get_conversation_history(session_id)
                
                assert history["user_id"] == user_id
                assert len(history["messages"]) == 8  # 4 user + 4 assistant
                
                # Verify topic isolation
                history_text = " ".join([msg["content"] for msg in history["messages"]])
                assert topic in history_text
                
                # Clean up
                await service.clear_session(session_id)
    
    @pytest.mark.asyncio
    async def test_conversation_memory_integration(self, ai_service_with_mock_llm):
        """Test integration with conversation memory system for context enhancement"""
        service = ai_service_with_mock_llm
        
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            with patch('app.ai_assistant_service.conversation_memory') as mock_memory:
                # Mock memory system responses
                mock_memory.retrieve_relevant_context.return_value = [
                    "User previously asked about Python programming",
                    "Discussion about machine learning algorithms",
                    "Interest in data science career"
                ]
                
                mock_memory.enhance_context_with_history.return_value = Mock(
                    session_id="test_session",
                    recent_messages=[],
                    relevant_history=[
                        "User previously asked about Python programming",
                        "Discussion about machine learning algorithms"
                    ],
                    total_context_length=200
                )
                
                # Test memory-enhanced conversation
                result = await service.process_query(
                    query="I want to continue learning about programming",
                    user_id="memory_test_user"
                )
                
                session_id = result["session_id"]
                
                # Verify memory integration was called
                mock_memory.enhance_context_with_history.assert_called()
                
                # Test that relevant history is included in context info
                assert result["context_info"]["relevant_history_count"] == 2
                
                # Clean up
                await service.clear_session(session_id)

class TestE2EConversationWithRealLMStudio:
    """Integration tests with real LM Studio (requires running LM Studio)"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_lm_studio_conversation(self, ai_service_with_real_llm):
        """Test conversation with real LM Studio instance"""
        service = ai_service_with_real_llm
        
        # Check if LM Studio is available
        try:
            health_check = service.studio_client.health_check()
            if not health_check:
                pytest.skip("LM Studio not available for integration testing")
        except Exception:
            pytest.skip("LM Studio not available for integration testing")
        
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            # Test real conversation
            result = await service.process_query(
                query="Hello! Can you tell me a brief fact about artificial intelligence?",
                user_id="real_llm_test_user"
            )
            
            assert "response" in result
            assert len(result["response"]) > 10  # Should get a real response
            assert result["user_id"] == "real_llm_test_user"
            
            session_id = result["session_id"]
            
            # Follow-up question
            follow_up = await service.process_query(
                query="That's interesting! Can you elaborate on that?",
                session_id=session_id,
                user_id="real_llm_test_user"
            )
            
            assert follow_up["session_id"] == session_id
            assert len(follow_up["response"]) > 10
            
            # Verify conversation history
            history = await service.get_conversation_history(session_id)
            assert len(history["messages"]) == 4  # 2 user + 2 assistant
            
            logger.info(f"Real LM Studio conversation test completed successfully")
            logger.info(f"First response: {result['response'][:100]}...")
            logger.info(f"Follow-up response: {follow_up['response'][:100]}...")
            
            # Clean up
            await service.clear_session(session_id)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_lm_studio_performance(self, ai_service_with_real_llm):
        """Test performance with real LM Studio"""
        service = ai_service_with_real_llm
        
        # Check if LM Studio is available
        try:
            health_check = service.studio_client.health_check()
            if not health_check:
                pytest.skip("LM Studio not available for performance testing")
        except Exception:
            pytest.skip("LM Studio not available for performance testing")
        
        with patch('app.ai_assistant_service.safety_filter') as mock_safety:
            mock_safety.is_safe.return_value = True
            
            # Measure response time for real LLM
            start_time = time.time()
            
            result = await service.process_query(
                query="What is machine learning in one sentence?",
                user_id="perf_real_llm_user"
            )
            
            response_time = time.time() - start_time
            
            assert "response" in result
            assert response_time < 30.0, f"Real LLM response too slow: {response_time:.2f}s"
            
            logger.info(f"Real LM Studio response time: {response_time:.2f}s")
            logger.info(f"Response length: {len(result['response'])} characters")
            
            # Clean up
            await service.clear_session(result["session_id"])

if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main([__file__ + "::TestE2EConversationIntegration::test_single_conversation_flow", "-v"])