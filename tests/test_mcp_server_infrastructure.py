# tests/test_mcp_server_infrastructure.py
"""
Tests for MCP Server Core Infrastructure

This module tests the enhanced MCP server implementation including:
- MCP server functionality
- Message handling and validation
- Redis backend integration
- Error handling and recovery
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import uuid

from app.mcp_server import MCPServer, MCPMessage, MCPMessageType, MCPMessagePriority, RegisteredAgent
from app.mcp_message_handler import MCPMessageHandler, MessageRoutingRule, ValidationResult
from app.redis_mcp_backend import RedisMCPBackend, RedisConfig, MessageQueueConfig
from app.mcp_error_handler import MCPErrorHandler, ErrorContext, ErrorCategory, ErrorSeverity
from app.mcp_integration import MCPIntegration


class TestMCPMessage:
    """Test MCP message functionality"""
    
    def test_message_creation(self):
        """Test MCP message creation and serialization"""
        message = MCPMessage(
            id="test-123",
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="test_agent",
            target_agents=["target_agent"],
            payload={"test": "data"},
            timestamp=datetime.utcnow(),
            priority=MCPMessagePriority.HIGH.value
        )
        
        assert message.id == "test-123"
        assert message.type == MCPMessageType.CONTEXT_UPDATE.value
        assert message.source_agent == "test_agent"
        assert message.target_agents == ["target_agent"]
        assert message.payload == {"test": "data"}
        assert message.priority == MCPMessagePriority.HIGH.value
    
    def test_message_serialization(self):
        """Test message to_dict and from_dict"""
        original_message = MCPMessage(
            id="test-456",
            type=MCPMessageType.TASK_NOTIFICATION.value,
            source_agent="task_agent",
            target_agents=["worker_agent"],
            payload={"task_id": "task-123", "action": "start"},
            timestamp=datetime.utcnow()
        )
        
        # Convert to dict
        message_dict = original_message.to_dict()
        assert isinstance(message_dict, dict)
        assert message_dict["id"] == "test-456"
        assert message_dict["type"] == MCPMessageType.TASK_NOTIFICATION.value
        
        # Convert back to message
        restored_message = MCPMessage.from_dict(message_dict)
        assert restored_message.id == original_message.id
        assert restored_message.type == original_message.type
        assert restored_message.source_agent == original_message.source_agent
        assert restored_message.payload == original_message.payload
    
    def test_message_expiration(self):
        """Test message TTL and expiration"""
        # Create message with short TTL
        message = MCPMessage(
            id="expire-test",
            type=MCPMessageType.HEARTBEAT.value,
            source_agent="test_agent",
            target_agents=["target"],
            payload={},
            timestamp=datetime.utcnow() - timedelta(seconds=10),  # 10 seconds ago
            ttl=5  # 5 second TTL
        )
        
        assert message.is_expired() == True
        
        # Create non-expired message
        fresh_message = MCPMessage(
            id="fresh-test",
            type=MCPMessageType.HEARTBEAT.value,
            source_agent="test_agent",
            target_agents=["target"],
            payload={},
            timestamp=datetime.utcnow(),
            ttl=60  # 60 second TTL
        )
        
        assert fresh_message.is_expired() == False


class TestMCPMessageHandler:
    """Test MCP message handler functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.handler = MCPMessageHandler()
    
    def test_message_validation_valid(self):
        """Test validation of valid messages"""
        message = MCPMessage(
            id="valid-123",
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="test_agent",
            target_agents=["target_agent"],
            payload={"context_data": {"key": "value"}, "context_type": "test"},
            timestamp=datetime.utcnow()
        )
        
        result = self.handler.validate_message(message)
        assert result.is_valid == True
        assert len(result.errors) == 0
    
    def test_message_validation_invalid(self):
        """Test validation of invalid messages"""
        # Missing required fields
        invalid_message = MCPMessage(
            id="",  # Empty ID
            type="invalid_type",  # Invalid type
            source_agent="",  # Empty source
            target_agents=[],  # Empty targets
            payload="not_a_dict",  # Invalid payload type
            timestamp=datetime.utcnow()
        )
        
        result = self.handler.validate_message(invalid_message)
        assert result.is_valid == False
        assert len(result.errors) > 0
    
    def test_message_serialization(self):
        """Test message serialization and deserialization"""
        message = MCPMessage(
            id="serialize-test",
            type=MCPMessageType.AGENT_REQUEST.value,
            source_agent="client_agent",
            target_agents=["server_agent"],
            payload={"request_type": "status", "data": {"key": "value"}},
            timestamp=datetime.utcnow()
        )
        
        # Serialize
        serialized = self.handler.serialize_message(message)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = self.handler.deserialize_message(serialized)
        assert deserialized.id == message.id
        assert deserialized.type == message.type
        assert deserialized.payload == message.payload
    
    def test_message_routing_rules(self):
        """Test message routing with rules"""
        # Add routing rule
        rule = MessageRoutingRule(
            message_type=MCPMessageType.CONTEXT_UPDATE.value,
            route_to=["context_handler", "logger_agent"]
        )
        self.handler.add_routing_rule(rule)
        
        message = MCPMessage(
            id="routing-test",
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="source",
            target_agents=["original_target"],
            payload={},
            timestamp=datetime.utcnow()
        )
        
        targets = self.handler.route_message(message)
        assert "original_target" in targets
        assert "context_handler" in targets
        assert "logger_agent" in targets
    
    def test_error_response_creation(self):
        """Test error response message creation"""
        original_message = MCPMessage(
            id="original-123",
            type=MCPMessageType.AGENT_REQUEST.value,
            source_agent="client",
            target_agents=["server"],
            payload={},
            timestamp=datetime.utcnow()
        )
        
        error = ValueError("Test error")
        error_response = self.handler.create_error_response(error, original_message)
        
        assert error_response.type == MCPMessageType.ERROR.value
        assert error_response.target_agents == ["client"]
        assert error_response.correlation_id == original_message.id
        assert "Test error" in error_response.payload["error_message"]


@pytest.mark.asyncio
class TestMCPServer:
    """Test MCP server functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.server = MCPServer(redis_url="redis://localhost:6379/1")  # Use test DB
    
    async def test_server_lifecycle(self):
        """Test server start and stop"""
        # Mock Redis connection
        with patch('app.mcp_server.redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.Redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            # Start server
            success = await self.server.start_server()
            assert success == True
            
            status = await self.server.get_server_status()
            assert status.is_running == True
            
            # Stop server
            await self.server.stop_server()
            
            status = await self.server.get_server_status()
            assert status.is_running == False
    
    async def test_agent_registration(self):
        """Test agent registration and unregistration"""
        with patch('app.mcp_server.redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.Redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            await self.server.start_server()
            
            # Register agent
            capabilities = [
                {
                    "name": "text_processing",
                    "description": "Process text messages",
                    "message_types": ["text_request"],
                    "parameters": {}
                }
            ]
            
            connection_id = await self.server.register_agent("test_agent", capabilities)
            assert connection_id is not None
            assert isinstance(connection_id, str)
            
            # Check agent is registered
            status = await self.server.get_server_status()
            assert status.connected_agents == 1
            
            # Unregister agent
            success = await self.server.unregister_agent("test_agent")
            assert success == True
            
            status = await self.server.get_server_status()
            assert status.connected_agents == 0
            
            await self.server.stop_server()
    
    async def test_message_routing(self):
        """Test message routing functionality"""
        with patch('app.mcp_server.redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.Redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            await self.server.start_server()
            
            # Register test agent
            await self.server.register_agent("test_agent", [])
            
            # Create test message
            message = MCPMessage(
                id="route-test",
                type=MCPMessageType.AGENT_REQUEST.value,
                source_agent="client",
                target_agents=["test_agent"],
                payload={"request_type": "ping"},
                timestamp=datetime.utcnow()
            )
            
            # Route message
            success = await self.server.route_message(message)
            assert success == True
            
            await self.server.stop_server()


@pytest.mark.asyncio
class TestRedisMCPBackend:
    """Test Redis MCP backend functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        config = RedisConfig(url="redis://localhost:6379/1")  # Use test DB
        self.backend = RedisMCPBackend(config=config)
    
    async def test_connection_lifecycle(self):
        """Test Redis connection and disconnection"""
        # Mock Redis connection
        with patch('app.redis_mcp_backend.redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.Redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            # Connect
            success = await self.backend.connect()
            assert success == True
            
            # Disconnect
            await self.backend.disconnect()
    
    async def test_message_publishing(self):
        """Test message publishing to Redis"""
        with patch('app.redis_mcp_backend.redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.Redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            mock_redis_instance.publish.return_value = 1
            
            await self.backend.connect()
            
            message = MCPMessage(
                id="pub-test",
                type=MCPMessageType.CONTEXT_UPDATE.value,
                source_agent="publisher",
                target_agents=["subscriber"],
                payload={"data": "test"},
                timestamp=datetime.utcnow()
            )
            
            success = await self.backend.publish_message("test_topic", message)
            assert success == True
            
            # Verify publish was called
            mock_redis_instance.publish.assert_called_once()
            
            await self.backend.disconnect()
    
    async def test_message_subscription(self):
        """Test message subscription functionality"""
        with patch('app.redis_mcp_backend.redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.Redis.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True
            
            await self.backend.connect()
            
            # Mock callback
            callback = AsyncMock()
            
            # Subscribe
            subscription_id = await self.backend.subscribe_to_topic("test_topic", callback)
            assert subscription_id is not None
            
            # Unsubscribe
            success = await self.backend.unsubscribe_from_topic(subscription_id)
            assert success == True
            
            await self.backend.disconnect()


class TestMCPErrorHandler:
    """Test MCP error handler functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.error_handler = MCPErrorHandler()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and classification"""
        context = ErrorContext(
            component="test_component",
            operation="test_operation",
            message_id="test-123"
        )
        
        test_exception = ConnectionError("Redis connection failed")
        
        error = await self.error_handler.handle_error(test_exception, context)
        
        assert error.id is not None
        assert error.severity in [s for s in ErrorSeverity]
        assert error.category in [c for c in ErrorCategory]
        assert error.message == str(test_exception)
        assert error.context == context
    
    def test_error_classification(self):
        """Test error severity and category classification"""
        context = ErrorContext(component="test", operation="test")
        
        # Test connection error classification
        conn_error = ConnectionError("Connection failed")
        severity = self.error_handler._classify_severity(conn_error, context)
        category = self.error_handler._classify_category(conn_error, context)
        
        assert severity == ErrorSeverity.HIGH
        assert category == ErrorCategory.CONNECTION
        
        # Test validation error classification
        val_error = ValueError("Invalid input")
        severity = self.error_handler._classify_severity(val_error, context)
        category = self.error_handler._classify_category(val_error, context)
        
        assert severity == ErrorSeverity.MEDIUM
        assert category == ErrorCategory.VALIDATION
    
    def test_error_statistics(self):
        """Test error statistics tracking"""
        initial_stats = self.error_handler.get_error_statistics()
        assert "total_errors" in initial_stats
        assert "errors_by_severity" in initial_stats
        assert "errors_by_category" in initial_stats
    
    @pytest.mark.asyncio
    async def test_error_report_generation(self):
        """Test error report generation"""
        # Generate some test errors
        context = ErrorContext(component="test", operation="test")
        
        await self.error_handler.handle_error(ValueError("Test error 1"), context)
        await self.error_handler.handle_error(ConnectionError("Test error 2"), context)
        
        # Generate report
        report = await self.error_handler.generate_error_report(hours=1)
        
        assert "summary" in report
        assert "error_patterns" in report
        assert "recommendations" in report
        assert report["summary"]["total_errors"] >= 2


@pytest.mark.asyncio
class TestMCPIntegration:
    """Test MCP integration functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.integration = MCPIntegration()
    
    async def test_integration_lifecycle(self):
        """Test integration initialization and shutdown"""
        # Mock all components
        with patch.multiple(
            'app.mcp_integration',
            MCPServer=Mock,
            RedisMCPBackend=Mock,
            MCPMessageHandler=Mock,
            MCPErrorHandler=Mock
        ):
            # Mock component methods
            self.integration.redis_backend.connect = AsyncMock(return_value=True)
            self.integration.server.start_server = AsyncMock(return_value=True)
            self.integration.server.stop_server = AsyncMock()
            self.integration.redis_backend.disconnect = AsyncMock()
            
            # Initialize
            success = await self.integration.initialize()
            assert success == True
            assert self.integration._initialized == True
            
            # Shutdown
            await self.integration.shutdown()
            assert self.integration._initialized == False
    
    async def test_agent_management(self):
        """Test agent registration through integration"""
        with patch.multiple(
            'app.mcp_integration',
            MCPServer=Mock,
            RedisMCPBackend=Mock,
            MCPMessageHandler=Mock,
            MCPErrorHandler=Mock
        ):
            self.integration._initialized = True
            self.integration.server.register_agent = AsyncMock(return_value="conn-123")
            self.integration.server.unregister_agent = AsyncMock(return_value=True)
            
            # Register agent
            connection_id = await self.integration.register_agent("test_agent", [])
            assert connection_id == "conn-123"
            
            # Unregister agent
            success = await self.integration.unregister_agent("test_agent")
            assert success == True
    
    async def test_message_handling(self):
        """Test message sending through integration"""
        with patch.multiple(
            'app.mcp_integration',
            MCPServer=Mock,
            RedisMCPBackend=Mock,
            MCPMessageHandler=Mock,
            MCPErrorHandler=Mock
        ):
            self.integration._initialized = True
            
            # Mock validation and routing
            validation_result = ValidationResult(True, [])
            self.integration.message_handler.validate_message.return_value = validation_result
            self.integration.message_handler.route_message.return_value = ["target_agent"]
            self.integration.server.route_message = AsyncMock(return_value=True)
            
            message = MCPMessage(
                id="integration-test",
                type=MCPMessageType.CONTEXT_UPDATE.value,
                source_agent="source",
                target_agents=["target"],
                payload={},
                timestamp=datetime.utcnow()
            )
            
            success = await self.integration.send_message(message)
            assert success == True
    
    async def test_system_status(self):
        """Test system status reporting"""
        with patch.multiple(
            'app.mcp_integration',
            MCPServer=Mock,
            RedisMCPBackend=Mock,
            MCPMessageHandler=Mock,
            MCPErrorHandler=Mock
        ):
            # Mock status methods
            from app.mcp_server import ServerStatus
            server_status = ServerStatus()
            server_status.is_running = True
            server_status.connected_agents = 2
            
            self.integration.server.get_server_status = AsyncMock(return_value=server_status)
            self.integration.redis_backend.get_queue_stats = AsyncMock(return_value={})
            self.integration.message_handler.get_statistics.return_value = {}
            self.integration.error_handler.get_error_statistics.return_value = {}
            
            status = await self.integration.get_system_status()
            
            assert "initialized" in status
            assert "server_status" in status
            assert "redis_stats" in status
            assert "message_stats" in status
            assert "error_stats" in status
    
    async def test_health_check(self):
        """Test health check functionality"""
        with patch.multiple(
            'app.mcp_integration',
            MCPServer=Mock,
            RedisMCPBackend=Mock,
            MCPMessageHandler=Mock,
            MCPErrorHandler=Mock
        ):
            # Mock healthy components
            from app.mcp_server import ServerStatus
            server_status = ServerStatus()
            server_status.is_running = True
            
            self.integration.server.get_server_status = AsyncMock(return_value=server_status)
            self.integration.redis_backend.get_queue_stats = AsyncMock(return_value={"mcp_stats": {}})
            self.integration.error_handler.get_error_statistics.return_value = {"error_rate_per_hour": 5}
            
            health = await self.integration.get_health_check()
            
            assert "healthy" in health
            assert "components" in health
            assert "mcp_server" in health["components"]
            assert "redis_backend" in health["components"]
            assert "error_handler" in health["components"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])