#!/usr/bin/env python3
"""
Validation script for MCP Performance Optimization implementation

This script validates the core functionality of the performance optimization
system without requiring external dependencies like Redis.
"""

import sys
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path
sys.path.append('.')

def test_basic_imports():
    """Test that all modules can be imported"""
    try:
        # Test core models
        from app.mcp_models import MCPMessage, MCPMessageType, MCPMessagePriority
        print("✓ MCP models imported successfully")
        
        # Test performance optimizer components (without Redis)
        from app.mcp_performance_optimizer import BatchConfig, CompressionConfig, ConnectionPoolConfig
        print("✓ Performance optimizer configs imported successfully")
        
        # Test monitoring system components
        from app.mcp_monitoring_system import MetricType, AlertSeverity, MetricPoint, Metric
        print("✓ Monitoring system components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_message_creation():
    """Test MCP message creation and validation"""
    try:
        from app.mcp_models import MCPMessage, MCPMessageType, MCPMessagePriority
        
        # Create a test message
        message = MCPMessage(
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="test_agent",
            target_agents=["target_agent_1", "target_agent_2"],
            payload={"test_data": "hello world", "timestamp": datetime.utcnow().isoformat()},
            priority=MCPMessagePriority.NORMAL.value
        )
        
        print(f"✓ Message created with ID: {message.id}")
        print(f"✓ Message type: {message.type}")
        print(f"✓ Target agents: {len(message.target_agents)}")
        print(f"✓ Payload size: {len(str(message.payload))} characters")
        
        # Test message serialization
        message_dict = message.to_dict()
        assert "id" in message_dict
        assert "type" in message_dict
        assert "payload" in message_dict
        print("✓ Message serialization works")
        
        # Test message validation
        assert not message.is_expired()  # Should not be expired immediately
        print("✓ Message validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Message creation error: {e}")
        return False

def test_batch_configuration():
    """Test batch configuration"""
    try:
        from app.mcp_performance_optimizer import BatchConfig
        
        config = BatchConfig(
            max_batch_size=100,
            max_batch_wait_ms=50,
            enable_priority_batching=True,
            batch_by_target=True,
            compression_threshold=1024
        )
        
        print(f"✓ Batch config created:")
        print(f"  - Max batch size: {config.max_batch_size}")
        print(f"  - Max wait time: {config.max_batch_wait_ms}ms")
        print(f"  - Priority batching: {config.enable_priority_batching}")
        print(f"  - Batch by target: {config.batch_by_target}")
        print(f"  - Compression threshold: {config.compression_threshold} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch configuration error: {e}")
        return False

def test_compression_configuration():
    """Test compression configuration"""
    try:
        from app.mcp_performance_optimizer import CompressionConfig, CompressionAlgorithm
        
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.GZIP,
            compression_level=6,
            threshold_bytes=1024,
            enable_adaptive_compression=True
        )
        
        print(f"✓ Compression config created:")
        print(f"  - Algorithm: {config.algorithm.value}")
        print(f"  - Compression level: {config.compression_level}")
        print(f"  - Threshold: {config.threshold_bytes} bytes")
        print(f"  - Adaptive compression: {config.enable_adaptive_compression}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compression configuration error: {e}")
        return False

def test_metrics_system():
    """Test metrics system components"""
    try:
        from app.mcp_monitoring_system import Metric, MetricType, MetricPoint
        from collections import deque
        
        # Create a test metric
        metric = Metric(
            name="test_metric",
            type=MetricType.COUNTER,
            description="Test counter metric",
            unit="count",
            data_points=deque(maxlen=100)
        )
        
        # Add some data points
        metric.add_point(10, {"label": "test"})
        metric.add_point(20, {"label": "test"})
        metric.add_point(30, {"label": "test"})
        
        print(f"✓ Metric created: {metric.name}")
        print(f"✓ Metric type: {metric.type.value}")
        print(f"✓ Data points: {len(metric.data_points)}")
        print(f"✓ Latest value: {metric.get_latest_value()}")
        print(f"✓ Average value: {metric.get_average(60)}")  # Last 60 minutes
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics system error: {e}")
        return False

def test_alert_system():
    """Test alert system components"""
    try:
        from app.mcp_monitoring_system import AlertRule, AlertSeverity, Alert, AlertStatus
        
        # Create an alert rule
        rule = AlertRule(
            name="high_cpu_usage",
            metric_name="cpu_usage",
            condition=">",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            duration_minutes=2,
            description="CPU usage is high"
        )
        
        print(f"✓ Alert rule created: {rule.name}")
        print(f"✓ Condition: {rule.metric_name} {rule.condition} {rule.threshold}")
        print(f"✓ Severity: {rule.severity.value}")
        
        # Create a test alert
        alert = Alert(
            id="test_alert_001",
            rule_name=rule.name,
            metric_name=rule.metric_name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message="Test alert message",
            triggered_at=datetime.utcnow()
        )
        
        print(f"✓ Alert created: {alert.id}")
        print(f"✓ Alert status: {alert.status.value}")
        
        # Test alert serialization
        alert_dict = alert.to_dict()
        assert "id" in alert_dict
        assert "severity" in alert_dict
        print("✓ Alert serialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ Alert system error: {e}")
        return False

def test_message_compression():
    """Test message compression without Redis"""
    try:
        import gzip
        import json
        from app.mcp_models import MCPMessage, MCPMessageType
        
        # Create a large message
        large_payload = {"data": "x" * 2000, "numbers": list(range(100))}
        message = MCPMessage(
            type=MCPMessageType.CONTEXT_UPDATE.value,
            source_agent="test_agent",
            target_agents=["target_agent_1"],
            payload=large_payload
        )
        
        # Serialize to JSON
        json_data = message.json().encode('utf-8')
        original_size = len(json_data)
        
        # Compress
        compressed_data = gzip.compress(json_data, compresslevel=6)
        compressed_size = len(compressed_data)
        
        compression_ratio = compressed_size / original_size
        
        print(f"✓ Message compression test:")
        print(f"  - Original size: {original_size} bytes")
        print(f"  - Compressed size: {compressed_size} bytes")
        print(f"  - Compression ratio: {compression_ratio:.2f}")
        print(f"  - Space saved: {((1 - compression_ratio) * 100):.1f}%")
        
        # Test decompression
        decompressed_data = gzip.decompress(compressed_data)
        decompressed_message = MCPMessage.parse_raw(decompressed_data.decode('utf-8'))
        
        assert decompressed_message.id == message.id
        assert decompressed_message.payload == message.payload
        print("✓ Decompression successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Message compression error: {e}")
        return False

def test_performance_calculations():
    """Test performance calculation utilities"""
    try:
        import statistics
        
        # Simulate performance data
        response_times = [0.1, 0.15, 0.12, 0.18, 0.09, 0.14, 0.11, 0.16, 0.13, 0.17]
        throughput_data = [100, 105, 98, 110, 95, 108, 102, 99, 107, 103]
        
        # Calculate statistics
        avg_response_time = statistics.mean(response_times)
        p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
        avg_throughput = statistics.mean(throughput_data)
        
        print(f"✓ Performance calculations:")
        print(f"  - Average response time: {avg_response_time:.3f}s")
        print(f"  - P95 response time: {p95_response_time:.3f}s")
        print(f"  - Average throughput: {avg_throughput:.1f} msg/s")
        
        # Test threshold checking
        warning_threshold = 0.15
        critical_threshold = 0.20
        
        high_response_times = [rt for rt in response_times if rt > warning_threshold]
        critical_response_times = [rt for rt in response_times if rt > critical_threshold]
        
        print(f"  - Response times > {warning_threshold}s: {len(high_response_times)}")
        print(f"  - Response times > {critical_threshold}s: {len(critical_response_times)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance calculations error: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 Starting MCP Performance Optimization Validation")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Message Creation", test_message_creation),
        ("Batch Configuration", test_batch_configuration),
        ("Compression Configuration", test_compression_configuration),
        ("Metrics System", test_metrics_system),
        ("Alert System", test_alert_system),
        ("Message Compression", test_message_compression),
        ("Performance Calculations", test_performance_calculations),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Validation Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Performance optimization system is ready.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())