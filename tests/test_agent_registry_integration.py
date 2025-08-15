# tests/test_agent_registry_integration.py
"""
Integration tests for agent registry updates and MCP integration
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.agent_registry import AgentRegistry, AgentType
from app.agent_mcp_router import AgentMCPRouter
from app.agent_integration import (
    initialize_agent_integration,
    ai_assistant_handler,
    self_improvement_handler,
    code_analysis_handler,
    performance_analysis_handler
)


class TestAgentRegistry:
    """Test enhanced agent registry functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary registry file
        self.temp_registry = {
            "general_query": {
                "type": "llm",
                "topic": "general_queries",
                "description": "General purpose LLM agent"
            },
            "ai_assistant": {
                "type": "ai_assistant",
                "topic": "ai_assistant_tasks",
                "description": "Advanced conversational AI assistant"
            },
            "self_improvement": {
                "type": "self_improvement",
                "topic": "self_improvement_tasks",
                "description": "Self-improvement agent"
            }
        }
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.temp_registry, self.temp_file)
        self.temp_file.close()
        
        # Create registry instance
        self.registry = AgentRegistry(self.temp_file.name)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_registry_initialization(self):
        """Test registry loads correctly"""
        assert len(self.registry.registry) == 3
        assert "ai_assistant" in self.registry.registry
        assert "self_improvement" in self.registry.registry
    
    def test_get_agent_by_intent(self):
        """Test getting agent configuration by intent"""
        ai_agent = self.registry.get_agent("ai_assistant")
        assert ai_agent["type"] == "ai_assistant"
        assert ai_agent["topic"] == "ai_assistant_tasks"
        
        # Test fallback for unknown intent
        unknown_agent = self.registry.get_agent("unknown_intent")
        assert unknown_agent["type"] == "llm"  # Should fallback to general_query
    
    def test_get_agents_by_type(self):
        """Test filtering agents by type"""
        ai_agents = self.registry.get_agents_by_type(AgentType.AI_ASSISTANT)
        assert len(ai_agents) == 1
        assert "ai_assistant" in ai_agents
        
        llm_agents = self.registry.get_agents_by_type(AgentType.LLM)
        assert len(llm_agents) == 1
        assert "general_query" in llm_agents
    
    def test_register_new_agent(self):
        """Test registering a new agent"""
        success = self.registry.register_agent(
            intent="test_agent",
            agent_type=AgentType.CODE_ANALYSIS,
            topic="test_topic",
            description="Test agent"
        )
        
        assert success
        assert "test_agent" in self.registry.registry
        assert self.registry.registry["test_agent"]["type"] == "code_analysis"
    
    def test_unregister_agent(self):
        """Test unregistering an agent"""
        # First register an agent
        self.registry.register_agent(
            intent="temp_agent",
            agent_type=AgentType.VISION,
            topic="temp_topic",
            description="Temporary agent"
        )
        
        assert "temp_agent" in self.registry.registry
        
        # Now unregister it
        success = self.registry.unregister_agent("temp_agent")
        assert success
        assert "temp_agent" not in self.registry.registry
    
    def test_get_agent_topics(self):
        """Test getting agent topic mapping"""
        topics = self.registry.get_agent_topics()
        
        assert topics["ai_assistant"] == "ai_assistant_tasks"
        assert topics["self_improvement"] == "self_improvement_tasks"
        assert topics["general_query"] == "general_queries"
    
    def test_supports_intent(self):
        """Test checking if intent is supported"""
        assert self.registry.supports_intent("ai_assistant")
        assert self.registry.supports_intent("self_improvement")
        assert not self.registry.supports_intent("unknown_intent")
    
    def test_invalid_registry_file(self):
        """Test handling of invalid registry file"""
        # Create invalid JSON file
        invalid_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        invalid_file.write("invalid json content")
        invalid_file.close()
        
        try:
            with pytest.raises(json.JSONDecodeError):
                AgentRegistry(invalid_file.name)
        finally:
            os.unlink(invalid_file.name)
    
    def test_missing_registry_file(self):
        """Test handling of missing registry file"""
        registry = AgentRegistry("nonexistent_file.json")
        
        # Should create minimal default registry
        assert "general_query" in registry.registry
        assert registry.registry["general_query"]["type"] == "llm"


class TestAgentMCPRouter:
    """Test MCP router functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.router = AgentMCPRouter()
        
        # Mock MCP
        self.mock_mcp = Mock()
        self.router.mcp = self.mock_mcp
    
    def test_register_agent_handler(self):
        """Test registering agent handlers"""
        mock_handler = Mock()
        
        self.router.register_agent_handler("test_agent", mock_handler)
        
        assert "test_agent" in self.router.agent_handlers
        assert self.router.agent_handlers["test_agent"] == mock_handler
    
    @patch('app.agent_mcp_router.registry')
    @patch('app.agent_mcp_router.mcp')
    def test_send_agent_message(self, mock_mcp_module, mock_registry):
        """Test sending message to specific agent"""
        # Mock registry
        mock_registry.supports_intent.return_value = True
        mock_registry.get_agent.return_value = {
            "type": "ai_assistant",
            "topic": "ai_assistant_tasks"
        }
        
        # Replace the router's mcp with our mock
        self.router.mcp = mock_mcp_module
        
        payload = {"action": "test", "data": "test_data"}
        
        success = self.router.send_agent_message("ai_assistant", payload)
        
        assert success
        mock_mcp_module.publish.assert_called_once()
    
    @patch('app.agent_mcp_router.registry')
    @patch('app.agent_mcp_router.mcp')
    def test_broadcast_to_agent_type(self, mock_mcp_module, mock_registry):
        """Test broadcasting to agent type"""
        # Mock registry
        mock_registry.get_agents_by_type.return_value = {
            "agent1": {"type": "ai_assistant", "topic": "topic1"},
            "agent2": {"type": "ai_assistant", "topic": "topic2"}
        }
        mock_registry.supports_intent.return_value = True
        mock_registry.get_agent.side_effect = [
            {"type": "ai_assistant", "topic": "topic1"},
            {"type": "ai_assistant", "topic": "topic2"}
        ]
        
        # Replace the router's mcp with our mock
        self.router.mcp = mock_mcp_module
        
        payload = {"action": "broadcast_test"}
        
        sent_count = self.router.broadcast_to_agent_type("ai_assistant", payload)
        
        assert sent_count == 2
        assert mock_mcp_module.publish.call_count == 2
    
    def test_get_routing_status(self):
        """Test getting routing status"""
        # Register some handlers
        self.router.register_agent_handler("test1", Mock())
        self.router.register_agent_handler("test2", Mock())
        
        with patch('app.agent_mcp_router.registry') as mock_registry:
            mock_registry.get_agent_topics.return_value = {
                "test1": "topic1",
                "test2": "topic2"
            }
            
            status = self.router.get_routing_status()
            
            assert status["status"] == "stopped"  # Not started yet
            assert len(status["registered_handlers"]) == 2
            assert "test1" in status["registered_handlers"]
            assert "test2" in status["registered_handlers"]


class TestAgentHandlers:
    """Test agent message handlers"""
    
    @pytest.mark.asyncio
    async def test_ai_assistant_handler(self):
        """Test AI assistant handler"""
        with patch('agents.ai_assistant_agent.ai_assistant_agent') as mock_agent:
            mock_agent.process_request = AsyncMock(return_value={
                "response": "Test response",
                "session_id": "test_session"
            })
            
            payload = {
                "action": "process_request",
                "query": "Test query"
            }
            
            result = await ai_assistant_handler(payload)
            
            assert "response" in result
            assert result["response"] == "Test response"
            mock_agent.process_request.assert_called_once_with(payload)
    
    @pytest.mark.asyncio
    async def test_self_improvement_handler(self):
        """Test self-improvement handler"""
        with patch('agents.self_improvement_agent.self_improvement_agent') as mock_agent:
            mock_agent.trigger_improvement_cycle = AsyncMock(return_value={
                "cycle_id": "test_cycle",
                "status": "started"
            })
            
            payload = {
                "action": "trigger_improvement",
                "trigger": "manual"
            }
            
            result = await self_improvement_handler(payload)
            
            assert "cycle_id" in result
            assert result["cycle_id"] == "test_cycle"
            mock_agent.trigger_improvement_cycle.assert_called_once_with(payload)
    
    @pytest.mark.asyncio
    async def test_code_analysis_handler(self):
        """Test code analysis handler"""
        with patch('app.code_analyzer.code_analyzer') as mock_analyzer:
            mock_report = Mock()
            mock_report.to_dict.return_value = {"issues": [], "metrics": {}}
            mock_analyzer.analyze_file.return_value = mock_report
            
            payload = {
                "action": "analyze_code",
                "file_path": "test.py"
            }
            
            result = await code_analysis_handler(payload)
            
            assert "report" in result
            assert result["file_path"] == "test.py"
            mock_analyzer.analyze_file.assert_called_once_with("test.py")
    
    @pytest.mark.asyncio
    async def test_performance_analysis_handler(self):
        """Test performance analysis handler"""
        with patch('app.performance_analyzer.get_performance_analyzer') as mock_get_analyzer:
            mock_analyzer = Mock()
            mock_trend = Mock()
            mock_trend.direction.value = "improving"
            mock_trend.slope = 0.1
            mock_trend.confidence = 0.8
            mock_trend.recent_avg = 100
            mock_trend.historical_avg = 90
            mock_trend.change_percentage = 11.1
            
            mock_analyzer.analyze_trends.return_value = mock_trend
            mock_get_analyzer.return_value = mock_analyzer
            
            payload = {
                "action": "analyze_performance",
                "metric_name": "response_time",
                "hours": 24
            }
            
            result = await performance_analysis_handler(payload)
            
            assert "trend_analysis" in result
            assert result["metric_name"] == "response_time"
            mock_analyzer.analyze_trends.assert_called_once_with("response_time", 24)
    
    @pytest.mark.asyncio
    async def test_handler_error_handling(self):
        """Test error handling in handlers"""
        # Test AI assistant handler with error
        with patch('agents.ai_assistant_agent.ai_assistant_agent') as mock_agent:
            mock_agent.process_request = AsyncMock(side_effect=Exception("Test error"))
            
            payload = {"action": "process_request", "query": "Test"}
            result = await ai_assistant_handler(payload)
            
            assert "error" in result
            assert "Test error" in result["error"]
            assert result["agent_type"] == "ai_assistant"


class TestAgentIntegration:
    """Test overall agent integration"""
    
    @patch('app.agent_integration.registry')
    @patch('app.agent_integration.agent_mcp_router')
    def test_initialize_agent_integration(self, mock_router, mock_registry):
        """Test agent integration initialization"""
        mock_registry.__bool__ = Mock(return_value=True)  # Registry is available
        mock_router.start_routing.return_value = True
        
        success = initialize_agent_integration()
        
        assert success
        
        # Verify handlers were registered
        expected_handlers = [
            "ai_assistant", "self_improvement", "code_analysis",
            "performance_analysis", "vision", "llm"
        ]
        
        assert mock_router.register_agent_handler.call_count == len(expected_handlers)
        mock_router.start_routing.assert_called_once()
    
    @patch('app.agent_integration.registry', None)
    def test_initialize_without_registry(self):
        """Test initialization when registry is not available"""
        success = initialize_agent_integration()
        
        assert not success
    
    @patch('app.agent_integration.registry')
    @patch('app.agent_integration.agent_mcp_router')
    def test_get_integration_status(self, mock_router, mock_registry):
        """Test getting integration status"""
        from app.agent_integration import get_integration_status
        
        # Mock registry
        mock_registry.__bool__ = Mock(return_value=True)
        mock_registry.list_agents.return_value = {
            "ai_assistant": {"type": "ai_assistant"},
            "self_improvement": {"type": "self_improvement"}
        }
        
        # Mock router status
        mock_router.get_routing_status.return_value = {
            "status": "running",
            "registered_handlers": ["ai_assistant", "self_improvement"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        status = get_integration_status()
        
        assert status["integration_status"] == "active"
        assert status["agent_registry"]["available"]
        assert status["agent_registry"]["agent_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__])