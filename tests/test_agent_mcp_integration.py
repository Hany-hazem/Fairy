# tests/test_agent_mcp_integration.py
"""
Tests for Agent MCP Integration

This module tests the integration between agents and the enhanced MCP system
with context synchronization capabilities.
"""

import asyncio
import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from app.mcp_client import MCPClient, ClientConfig
from app.mcp_models import MCPMessage, MCPMessageType, AgentContext
from app.agent_context_synchronizer import agent_context_synchronizer, AgentContextType
from agents.ai_assistant_agent import AIAssistantAgent
from agents.self_improvement_agent import SelfImprovementAgent


class TestAgentMCPIntegration:
    """Test agent integration with MCP system"""
    
    @pytest.fixture
    def mock_redis_backend(self):
        """Mock Redis backend for testing"""
        mock_backend = Mock()
        mock_backend.connect = AsyncMock(return_value=True)
        mock_backend.disconnect = AsyncMock()
        mock_backend.publish_message = AsyncMock(return_value=True)
        mock_backend.subscribe_to_topic = AsyncMock(return_value="sub_123")
        mock_backend.unsubscribe_from_topic = AsyncMock(return_value=True)
        return mock_backend
    
    @pytest.fixture
    def ai_assistant_agent(self):
        """Create AI Assistant Agent for testing"""
        return AIAssistantAgent()
    
    @pytest.fixture
    def self_improvement_agent(self):
        """Create Self-Improvement Agent for testing"""
        return SelfImprovementAgent()
    
    @pytest.mark.asyncio
    async def test_mcp_client_initialization(self):
        """Test MCP client initialization"""
        config = ClientConfig(
            agent_id="test_agent",
            capabilities=[{
                "name": "test_capability",
                "description": "Test capability",
                "message_types": ["agent_request"],
                "parameters": {}
            }]
        )
        
        client = MCPClient(config)
        
        assert client.agent_id == "test_agent"
        assert len(client.capabilities) == 1
        assert client.capabilities[0]["name"] == "test_capability"
    
    @pytest.mark.asyncio
    async def test_agent_context_creation(self):
        """Test agent context creation and validation"""
        context = AgentContext(
            agent_id="test_agent",
            context_type="test_context",
            context_data={"key": "value"},
            access_level="public"
        )
        
        assert context.agent_id == "test_agent"
        assert context.context_type == "test_context"
        assert context.context_data["key"] == "value"
        assert context.can_access("any_agent")  # Public access
    
    @pytest.mark.asyncio
    async def test_mcp_message_creation(self):
        """Test MCP message creation and validation"""
        message = MCPMessage(
            type=MCPMessageType.AGENT_REQUEST.value,
            source_agent="source_agent",
            target_agents=["target_agent"],
            payload={"request_type": "test_request"}
        )
        
        assert message.type == MCPMessageType.AGENT_REQUEST.value
        assert message.source_agent == "source_agent"
        assert "target_agent" in message.target_agents
        assert message.payload["request_type"] == "test_request"
    
    @pytest.mark.asyncio
    @patch('app.mcp_integration.mcp_integration')
    async def test_ai_assistant_agent_initialization(self, mock_mcp_integration, ai_assistant_agent):
        """Test AI Assistant Agent initialization with MCP"""
        mock_mcp_integration._initialized = True
        
        # Test agent properties
        assert ai_assistant_agent.agent_id == "ai_assistant_agent"
        assert ai_assistant_agent.agent_type == "ai_assistant"
        assert len(ai_assistant_agent.capabilities) == 2
        
        # Check capabilities
        capability_names = [cap["name"] for cap in ai_assistant_agent.capabilities]
        assert "conversational_ai" in capability_names
        assert "conversation_management" in capability_names
    
    @pytest.mark.asyncio
    @patch('app.mcp_integration.mcp_integration')
    async def test_self_improvement_agent_initialization(self, mock_mcp_integration, self_improvement_agent):
        """Test Self-Improvement Agent initialization with MCP"""
        mock_mcp_integration._initialized = True
        
        # Test agent properties
        assert self_improvement_agent.agent_id == "self_improvement_agent"
        assert self_improvement_agent.agent_type == "self_improvement"
        assert len(self_improvement_agent.capabilities) == 3
        
        # Check capabilities
        capability_names = [cap["name"] for cap in self_improvement_agent.capabilities]
        assert "code_analysis" in capability_names
        assert "autonomous_improvement" in capability_names
        assert "performance_monitoring" in capability_names
    
    @pytest.mark.asyncio
    @patch('app.agent_context_synchronizer.agent_context_synchronizer')
    async def test_context_handler_registration(self, mock_synchronizer, ai_assistant_agent):
        """Test context handler registration"""
        mock_synchronizer.register_agent_context_handler = Mock()
        
        # Register context handlers
        ai_assistant_agent._register_context_handlers()
        
        # Verify handlers were registered
        assert mock_synchronizer.register_agent_context_handler.call_count >= 3
        
        # Check specific handler registrations
        calls = mock_synchronizer.register_agent_context_handler.call_args_list
        context_types = [call[1]["context_type"] for call in calls]
        
        assert AgentContextType.CONVERSATION.value in context_types
        assert AgentContextType.USER_SESSION.value in context_types
        assert AgentContextType.TASK_STATE.value in context_types
    
    @pytest.mark.asyncio
    async def test_context_synchronization(self, mock_redis_backend):
        """Test context synchronization between agents"""
        # Create test context
        context = AgentContext(
            agent_id="source_agent",
            context_type=AgentContextType.CONVERSATION.value,
            context_data={
                "session_id": "test_session",
                "last_query": "Hello",
                "last_response": "Hi there!"
            },
            access_level="public"
        )
        
        # Mock the synchronizer
        with patch('app.agent_context_synchronizer.agent_context_synchronizer.base_synchronizer') as mock_base:
            mock_base.sync_agent_context = AsyncMock(return_value=True)
            
            # Test synchronization
            result = await agent_context_synchronizer.sync_agent_context(
                "source_agent", context, ["target_agent"]
            )
            
            # Verify result structure
            assert hasattr(result, 'success')
            assert hasattr(result, 'agent_id')
            assert hasattr(result, 'context_type')
    
    @pytest.mark.asyncio
    async def test_message_handler_registration(self, ai_assistant_agent):
        """Test message handler registration"""
        # Mock MCP client
        ai_assistant_agent.mcp_client.register_message_handler = Mock()
        
        # Register message handlers
        ai_assistant_agent._register_message_handlers()
        
        # Verify handlers were registered
        assert ai_assistant_agent.mcp_client.register_message_handler.call_count >= 2
        
        # Check specific handler registrations
        calls = ai_assistant_agent.mcp_client.register_message_handler.call_args_list
        message_types = [call[0][0] for call in calls]
        
        assert MCPMessageType.AGENT_REQUEST.value in message_types
        assert MCPMessageType.TASK_NOTIFICATION.value in message_types
    
    @pytest.mark.asyncio
    async def test_agent_request_handling(self, ai_assistant_agent):
        """Test agent request message handling"""
        # Create test message
        message = MCPMessage(
            type=MCPMessageType.AGENT_REQUEST.value,
            source_agent="test_agent",
            target_agents=[ai_assistant_agent.agent_id],
            payload={
                "request_type": "process_query",
                "query": "Test query",
                "session_id": "test_session"
            }
        )
        
        # Mock dependencies
        with patch.object(ai_assistant_agent, 'process_request') as mock_process:
            mock_process.return_value = {"response": "Test response"}
            
            with patch.object(ai_assistant_agent.mcp_client, 'send_message') as mock_send:
                mock_send.return_value = True
                
                # Handle the request
                await ai_assistant_agent._handle_agent_request(message)
                
                # Verify processing was called
                mock_process.assert_called_once()
                mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_conflict_resolution(self):
        """Test context conflict resolution"""
        # Create conflicting contexts
        context1 = AgentContext(
            agent_id="agent1",
            context_type="test_context",
            context_data={"value": "old"},
            version="v1"
        )
        
        context2 = AgentContext(
            agent_id="agent2",
            context_type="test_context",
            context_data={"value": "new"},
            version="v2"
        )
        
        # Mock conflict detection and resolution
        with patch('app.agent_context_synchronizer.agent_context_synchronizer._detect_agent_context_conflicts') as mock_detect:
            mock_detect.return_value = []  # No conflicts for this test
            
            with patch('app.agent_context_synchronizer.agent_context_synchronizer.base_synchronizer') as mock_base:
                mock_base.sync_agent_context = AsyncMock(return_value=True)
                
                # Test synchronization without conflicts
                result = await agent_context_synchronizer.sync_agent_context(
                    "agent1", context1, ["agent2"]
                )
                
                # Should succeed without conflicts
                assert result.conflicts_resolved == 0
    
    @pytest.mark.asyncio
    async def test_agent_startup_sequence(self, ai_assistant_agent, mock_redis_backend):
        """Test complete agent startup sequence"""
        # Mock MCP client connection
        with patch.object(ai_assistant_agent.mcp_client, 'connect') as mock_connect:
            mock_connect.return_value = True
            
            # Mock context synchronizer startup
            with patch('app.agent_context_synchronizer.agent_context_synchronizer.start') as mock_start:
                mock_start.return_value = None
                
                # Start the agent
                result = await ai_assistant_agent.start()
                
                # Verify startup sequence
                assert result is True
                mock_connect.assert_called_once()
                mock_start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_access_control(self):
        """Test context access control"""
        # Create private context
        private_context = AgentContext(
            agent_id="owner_agent",
            context_type="private_context",
            context_data={"secret": "data"},
            access_level="private"
        )
        
        # Test access control
        assert private_context.can_access("owner_agent")  # Owner can access
        assert not private_context.can_access("other_agent")  # Others cannot
        
        # Create public context
        public_context = AgentContext(
            agent_id="owner_agent",
            context_type="public_context",
            context_data={"public": "data"},
            access_level="public"
        )
        
        # Test public access
        assert public_context.can_access("owner_agent")
        assert public_context.can_access("other_agent")
        assert public_context.can_access("any_agent")
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self):
        """Test performance metrics tracking in context synchronization"""
        # Mock agent context synchronizer
        with patch('app.agent_context_synchronizer.agent_context_synchronizer._update_sync_metrics') as mock_metrics:
            with patch('app.agent_context_synchronizer.agent_context_synchronizer.base_synchronizer') as mock_base:
                mock_base.sync_agent_context = AsyncMock(return_value=True)
                
                context = AgentContext(
                    agent_id="test_agent",
                    context_type="test_context",
                    context_data={"test": "data"}
                )
                
                # Perform synchronization
                await agent_context_synchronizer.sync_agent_context(
                    "test_agent", context, ["target_agent"]
                )
                
                # Verify metrics were updated
                mock_metrics.assert_called_once()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])