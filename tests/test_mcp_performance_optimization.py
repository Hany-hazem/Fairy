# tests/test_mcp_performance_optimization.py
"""
Test suite for MCP Performance Optimization and Monitoring

This module provides comprehensive tests for the performance optimization
and monitoring systems including batching, compression, and metrics collection.
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.mcp_performance_optimizer import (
    MessageBatcher, MessageCompressor, EnhancedConnectionPool,
    MCPPerformanceOptimizer, BatchConfig, CompressionConfig, 
    ConnectionPoolConfig, CompressionAlgorithm
)
from app.mcp_monitoring_system import (
    MCPMetricsCollector, GitWorkflowMonitor, AlertManager, 
    SystemHealthDashboard, AlertRule, AlertSeverity, MetricType
)
from app.mcp_performance_integration import (
    EnhancedMCPServer, MCPPerformanceMiddleware
)
from app.mcp_models import MCPMessage, MCPMessageType, MCPMessagePriority


class TestMessageBatcher:
    """Test suite for MessageBatcher"""
    
    @pytest.fixture
    def batch_config(self):
        return BatchConfig(
            max_batch_size=5,
            max_batch_wait_ms=100,
            enable_priority_batching=True,
            batch_by_target=True
        )
    
    @pytest.fixture
    def batcher(self, batch_config):
        return MessageBatcher(batch_config)
    
    @pytest.fixture
    def sample_message(self):
        return MCPMessage(
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="test_agent",
            target_agents=["target_agent_1"],
            payload={"test": "data"}
        )
    
    @pytest.mark.asyncio
    async def test_batcher_initialization(self, batcher):
        """Test batcher initialization"""
        assert batcher.config.max_batch_size == 5
        assert batcher.config.max_batch_wait_ms == 100
        assert not batcher._running
        
        await batcher.start()
        assert batcher._running
        
        await batcher.stop()
        assert not batcher._running
    
    @pytest.mark.asyncio
    async def test_message_batching_by_size(self, batcher, sample_message):
        """Test batching messages by size limit"""
        await batcher.start()
        
        # Mock batch handler
        batch_handler = AsyncMock()
        batcher.add_batch_handler(batch_handler)
        
        # Add messages up to batch size
        for i in range(5):
            message = MCPMessage(
                type=sample_message.type,
                source_agent=sample_message.source_agent,
                target_agents=sample_message.target_agents,
                payload={"index": i}
            )
            await batcher.add_message(message)
        
        # Wait for batch processing
        await asyncio.sleep(0.1)
        
        # Verify batch was processed
        assert batch_handler.call_count == 1
        batch_messages, batch_key = batch_handler.call_args[0]
        assert len(batch_messages) == 5
        
        await batcher.stop()
    
    @pytest.mark.asyncio
    async def test_message_batching_by_time(self, batcher, sample_message):
        """Test batching messages by time limit"""
        await batcher.start()
        
        # Mock batch handler
        batch_handler = AsyncMock()
        batcher.add_batch_handler(batch_handler)
        
        # Add single message
        await batcher.add_message(sample_message)
        
        # Wait for time-based batch processing
        await asyncio.sleep(0.2)  # Wait longer than max_batch_wait_ms
        
        # Verify batch was processed
        assert batch_handler.call_count == 1
        batch_messages, batch_key = batch_handler.call_args[0]
        assert len(batch_messages) == 1
        
        await batcher.stop()
    
    @pytest.mark.asyncio
    async def test_priority_batching(self, batcher):
        """Test priority-based batching"""
        await batcher.start()
        
        batch_handler = AsyncMock()
        batcher.add_batch_handler(batch_handler)
        
        # Add high priority message
        high_priority_msg = MCPMessage(
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="test_agent",
            target_agents=["target_agent_1"],
            payload={"priority": "high"},
            priority=MCPMessagePriority.HIGH.value
        )
        
        await batcher.add_message(high_priority_msg)
        
        # Add normal priority messages
        for i in range(10):
            normal_msg = MCPMessage(
                type=MCPMessageType.CONTEXT_UPDATE.value,
                source_agent="test_agent",
                target_agents=["target_agent_1"],
                payload={"index": i},
                priority=MCPMessagePriority.NORMAL.value
            )
            await batcher.add_message(normal_msg)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # High priority batch should be processed first
        assert batch_handler.call_count >= 1
        
        await batcher.stop()
    
    @pytest.mark.asyncio
    async def test_batch_stats(self, batcher, sample_message):
        """Test batch statistics collection"""
        await batcher.start()
        
        batch_handler = AsyncMock()
        batcher.add_batch_handler(batch_handler)
        
        # Process some messages
        for i in range(3):
            await batcher.add_message(sample_message)
        
        await asyncio.sleep(0.2)
        
        stats = await batcher.get_batch_stats()
        assert "total_messages_batched" in stats
        assert "average_batch_size" in stats
        assert stats["total_messages_batched"] >= 3
        
        await batcher.stop()


class TestMessageCompressor:
    """Test suite for MessageCompressor"""
    
    @pytest.fixture
    def compression_config(self):
        return CompressionConfig(
            algorithm=CompressionAlgorithm.GZIP,
            compression_level=6,
            threshold_bytes=100,
            enable_adaptive_compression=True
        )
    
    @pytest.fixture
    def compressor(self, compression_config):
        return MessageCompressor(compression_config)
    
    @pytest.fixture
    def large_message(self):
        # Create a message with large payload
        large_payload = {"data": "x" * 2000}  # 2KB of data
        return MCPMessage(
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="test_agent",
            target_agents=["target_agent_1"],
            payload=large_payload
        )
    
    @pytest.fixture
    def small_message(self):
        return MCPMessage(
            type=MCPMessageType.HEARTBEAT.value,
            source_agent="test_agent",
            target_agents=["target_agent_1"],
            payload={"status": "ok"}
        )
    
    @pytest.mark.asyncio
    async def test_message_compression(self, compressor, large_message):
        """Test message compression"""
        compressed_data, metadata = await compressor.compress_message(large_message)
        
        assert metadata["algorithm"] == "gzip"
        assert metadata["compressed_size"] < metadata["original_size"]
        assert metadata["compression_ratio"] < 1.0
        
        # Test decompression
        decompressed_message = await compressor.decompress_message(compressed_data, metadata)
        assert decompressed_message.id == large_message.id
        assert decompressed_message.payload == large_message.payload
    
    @pytest.mark.asyncio
    async def test_small_message_no_compression(self, compressor, small_message):
        """Test that small messages are not compressed"""
        compressed_data, metadata = await compressor.compress_message(small_message)
        
        assert metadata["algorithm"] == "none"
        assert "compressed_size" not in metadata or metadata["compressed_size"] == metadata["original_size"]
    
    @pytest.mark.asyncio
    async def test_batch_compression(self, compressor, large_message):
        """Test batch compression"""
        messages = [large_message for _ in range(5)]
        
        compressed_data, metadata = await compressor.compress_batch(messages)
        
        assert metadata["algorithm"] == "gzip"
        assert metadata["batch_size"] == 5
        assert metadata["compression_ratio"] < 1.0
        
        # Test decompression
        decompressed_messages = await compressor.decompress_batch(compressed_data, metadata)
        assert len(decompressed_messages) == 5
        assert all(msg.payload == large_message.payload for msg in decompressed_messages)
    
    def test_compression_stats(self, compressor):
        """Test compression statistics"""
        stats = compressor.get_compression_stats()
        
        assert "messages_compressed" in stats
        assert "total_compression_time" in stats
        assert "average_compression_ratio" in stats


class TestEnhancedConnectionPool:
    """Test suite for EnhancedConnectionPool"""
    
    @pytest.fixture
    def pool_config(self):
        return ConnectionPoolConfig(
            max_connections=10,
            min_connections=2,
            connection_timeout=5,
            health_check_interval=10
        )
    
    @pytest.fixture
    def connection_pool(self, pool_config):
        return EnhancedConnectionPool("redis://localhost:6379", pool_config)
    
    @pytest.mark.asyncio
    async def test_pool_initialization(self, connection_pool):
        """Test connection pool initialization"""
        # Mock Redis connection
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_redis.return_value.ping = AsyncMock()
            
            success = await connection_pool.initialize()
            assert success
            
            await connection_pool.close_all_connections()
    
    @pytest.mark.asyncio
    async def test_connection_reuse(self, connection_pool):
        """Test connection reuse"""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_redis.return_value.ping = AsyncMock(return_value=True)
            
            await connection_pool.initialize()
            
            # Get connection with ID
            conn1 = await connection_pool.get_connection("test_conn")
            conn2 = await connection_pool.get_connection("test_conn")
            
            # Should be the same connection
            assert conn1 is conn2
            
            await connection_pool.close_all_connections()
    
    @pytest.mark.asyncio
    async def test_pool_stats(self, connection_pool):
        """Test connection pool statistics"""
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_redis.return_value.ping = AsyncMock(return_value=True)
            
            await connection_pool.initialize()
            
            # Get some connections
            await connection_pool.get_connection("conn1")
            await connection_pool.get_connection("conn2")
            
            stats = await connection_pool.get_pool_stats()
            
            assert "max_connections" in stats
            assert "active_connections" in stats
            assert "pool_hits" in stats
            assert "pool_misses" in stats
            
            await connection_pool.close_all_connections()


class TestMCPMetricsCollector:
    """Test suite for MCPMetricsCollector"""
    
    @pytest.fixture
    def metrics_collector(self):
        return MCPMetricsCollector()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, metrics_collector):
        """Test basic metrics collection"""
        # Mock Redis connection
        with patch('redis.from_url') as mock_redis:
            mock_redis.return_value.ping = AsyncMock()
            
            await metrics_collector.start()
            
            # Record various metrics
            metrics_collector.record_counter("test_counter", 5)
            metrics_collector.record_gauge("test_gauge", 42.5)
            metrics_collector.record_histogram("test_histogram", 123)
            metrics_collector.record_timer("test_timer", 0.5)
            
            # Get metrics
            counter_metric = metrics_collector.get_metric("test_counter")
            gauge_metric = metrics_collector.get_metric("test_gauge")
            
            assert counter_metric is not None
            assert counter_metric.get_latest_value() == 5
            assert gauge_metric.get_latest_value() == 42.5
            
            await metrics_collector.stop()
    
    @pytest.mark.asyncio
    async def test_metrics_summary(self, metrics_collector):
        """Test metrics summary generation"""
        with patch('redis.from_url') as mock_redis:
            mock_redis.return_value.ping = AsyncMock()
            
            await metrics_collector.start()
            
            # Record some metrics
            metrics_collector.record_counter("messages_total", 100)
            metrics_collector.record_gauge("cpu_usage", 75.5)
            
            summary = metrics_collector.get_metrics_summary()
            
            assert "messages_total" in summary
            assert "cpu_usage" in summary
            assert summary["messages_total"]["latest_value"] == 100
            assert summary["cpu_usage"]["latest_value"] == 75.5
            
            await metrics_collector.stop()


class TestGitWorkflowMonitor:
    """Test suite for GitWorkflowMonitor"""
    
    @pytest.fixture
    def git_monitor(self):
        metrics_collector = MCPMetricsCollector()
        return GitWorkflowMonitor(metrics_collector)
    
    def test_operation_tracking(self, git_monitor):
        """Test Git operation tracking"""
        # Start operation
        git_monitor.start_operation("op1", "commit", {"branch": "main"})
        
        active_ops = git_monitor.get_active_operations()
        assert "op1" in active_ops
        assert active_ops["op1"]["type"] == "commit"
        
        # Complete operation
        git_monitor.complete_operation("op1", success=True)
        
        active_ops = git_monitor.get_active_operations()
        assert "op1" not in active_ops
        
        # Check stats
        stats = git_monitor.get_operation_stats()
        assert "commit" in stats
        assert stats["commit"]["total_operations"] == 1
        assert stats["commit"]["successful_operations"] == 1
    
    def test_performance_thresholds(self, git_monitor):
        """Test performance threshold checking"""
        # Start and complete a slow operation
        git_monitor.start_operation("slow_op", "commit")
        
        # Simulate slow operation by modifying start time
        git_monitor.active_operations["slow_op"]["start_time"] = time.time() - 15  # 15 seconds ago
        
        git_monitor.complete_operation("slow_op", success=True)
        
        # Should trigger performance threshold metrics
        # (This would be verified by checking metrics in a real test)


class TestAlertManager:
    """Test suite for AlertManager"""
    
    @pytest.fixture
    def alert_manager(self):
        metrics_collector = MCPMetricsCollector()
        return AlertManager(metrics_collector)
    
    @pytest.mark.asyncio
    async def test_alert_rule_evaluation(self, alert_manager):
        """Test alert rule evaluation"""
        with patch('redis.from_url') as mock_redis:
            mock_redis.return_value.ping = AsyncMock()
            
            await alert_manager.start()
            
            # Add test metric
            alert_manager.metrics_collector.record_gauge("test_metric", 150)
            
            # Add alert rule
            rule = AlertRule(
                name="test_rule",
                metric_name="test_metric",
                condition=">",
                threshold=100,
                severity=AlertSeverity.WARNING,
                description="Test alert"
            )
            alert_manager.add_alert_rule(rule)
            
            # Evaluate rules
            await alert_manager._evaluate_all_rules()
            
            # Check for active alerts
            active_alerts = alert_manager.get_active_alerts()
            assert len(active_alerts) > 0
            assert active_alerts[0].rule_name == "test_rule"
            
            await alert_manager.stop()
    
    @pytest.mark.asyncio
    async def test_alert_acknowledgment(self, alert_manager):
        """Test alert acknowledgment"""
        with patch('redis.from_url') as mock_redis:
            mock_redis.return_value.ping = AsyncMock()
            mock_redis.return_value.setex = AsyncMock()
            
            await alert_manager.start()
            
            # Create a test alert
            from app.mcp_monitoring_system import Alert, AlertStatus
            alert = Alert(
                id="test_alert",
                rule_name="test_rule",
                metric_name="test_metric",
                severity=AlertSeverity.WARNING,
                status=AlertStatus.ACTIVE,
                message="Test alert message",
                triggered_at=datetime.utcnow()
            )
            
            alert_manager.active_alerts[alert.id] = alert
            
            # Acknowledge alert
            await alert_manager.acknowledge_alert(alert.id, "test_user")
            
            # Check status
            assert alert.status == AlertStatus.ACKNOWLEDGED
            assert alert.acknowledged_by == "test_user"
            
            await alert_manager.stop()


class TestEnhancedMCPServer:
    """Test suite for EnhancedMCPServer"""
    
    @pytest.fixture
    def enhanced_server(self):
        return EnhancedMCPServer()
    
    @pytest.mark.asyncio
    async def test_enhanced_server_initialization(self, enhanced_server):
        """Test enhanced server initialization"""
        with patch.multiple(
            enhanced_server,
            _initialize_monitoring_system=AsyncMock(),
            _initialize_performance_optimizer=AsyncMock()
        ):
            with patch.object(EnhancedMCPServer.__bases__[0], 'start_server', return_value=True):
                success = await enhanced_server.start_server()
                assert success
                
                await enhanced_server.stop_server()
    
    @pytest.mark.asyncio
    async def test_enhanced_message_routing(self, enhanced_server):
        """Test enhanced message routing with metrics"""
        # Mock components
        enhanced_server.metrics_collector = Mock()
        enhanced_server.performance_optimizer = Mock()
        enhanced_server.performance_optimizer.process_message = AsyncMock(return_value=True)
        enhanced_server._registered_agents = {"test_agent": Mock()}
        
        message = MCPMessage(
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="test_agent",
            target_agents=["target_agent"],
            payload={"test": "data"}
        )
        
        success = await enhanced_server.route_message(message)
        assert success
        
        # Verify metrics were recorded
        assert enhanced_server.metrics_collector.record_counter.called
        assert enhanced_server.metrics_collector.record_timer.called
    
    @pytest.mark.asyncio
    async def test_performance_stats(self, enhanced_server):
        """Test performance statistics collection"""
        # Mock components
        enhanced_server.performance_optimizer = Mock()
        enhanced_server.performance_optimizer.get_performance_stats = AsyncMock(return_value={"test": "stats"})
        enhanced_server.health_dashboard = Mock()
        enhanced_server.health_dashboard.get_dashboard_data = AsyncMock(return_value={"dashboard": "data"})
        
        with patch.object(EnhancedMCPServer.__bases__[0], 'get_server_status', return_value={"server": "status"}):
            stats = await enhanced_server.get_performance_stats()
            
            assert "server" in stats
            assert "performance_optimization" in stats
            assert "monitoring" in stats


class TestMCPPerformanceMiddleware:
    """Test suite for MCPPerformanceMiddleware"""
    
    @pytest.fixture
    def middleware(self):
        metrics_collector = MCPMetricsCollector()
        return MCPPerformanceMiddleware(metrics_collector)
    
    def test_operation_timing(self, middleware):
        """Test operation timing"""
        op_id = middleware.start_operation("test_operation")
        assert op_id in middleware.operation_timers
        
        time.sleep(0.1)  # Simulate work
        
        middleware.end_operation(op_id, "test_operation", success=True)
        assert op_id not in middleware.operation_timers
    
    @pytest.mark.asyncio
    async def test_operation_decorator(self, middleware):
        """Test operation decorator"""
        @middleware.operation_decorator("decorated_operation")
        async def test_function():
            await asyncio.sleep(0.1)
            return "result"
        
        result = await test_function()
        assert result == "result"


# Integration tests
class TestPerformanceIntegration:
    """Integration tests for performance optimization system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_message_processing(self):
        """Test end-to-end message processing with all optimizations"""
        # This would be a comprehensive integration test
        # that tests the entire pipeline from message creation
        # to delivery with all optimizations enabled
        pass
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under high load"""
        # This would test the system with high message volumes
        # to verify performance optimizations are working
        pass
    
    @pytest.mark.asyncio
    async def test_monitoring_accuracy(self):
        """Test monitoring system accuracy"""
        # This would verify that monitoring metrics
        # accurately reflect system performance
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])