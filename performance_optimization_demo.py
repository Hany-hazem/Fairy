#!/usr/bin/env python3
"""
MCP Performance Optimization Demo

This script demonstrates the key concepts and algorithms implemented
in the performance optimization system without requiring external dependencies.
"""

import asyncio
import gzip
import json
import time
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


# Simplified message model
@dataclass
class SimpleMessage:
    """Simplified MCP message for demonstration"""
    id: str
    type: str
    source_agent: str
    target_agents: List[str]
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "source_agent": self.source_agent,
            "target_agents": self.target_agents,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority
        }


# Batch configuration
@dataclass
class BatchConfig:
    """Configuration for message batching"""
    max_batch_size: int = 100
    max_batch_wait_ms: int = 50
    enable_priority_batching: bool = True
    batch_by_target: bool = True


# Compression configuration
class CompressionAlgorithm(Enum):
    NONE = "none"
    GZIP = "gzip"


@dataclass
class CompressionConfig:
    """Configuration for message compression"""
    algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    compression_level: int = 6
    threshold_bytes: int = 1024


# Metrics system
class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class SimpleMetric:
    """Simplified metric for demonstration"""
    
    def __init__(self, name: str, metric_type: MetricType, description: str = ""):
        self.name = name
        self.type = metric_type
        self.description = description
        self.data_points = deque(maxlen=1000)
    
    def add_point(self, value: float, labels: Dict[str, str] = None):
        """Add a data point"""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self.data_points.append(point)
    
    def get_latest_value(self) -> Optional[float]:
        """Get the latest value"""
        if self.data_points:
            return self.data_points[-1].value
        return None
    
    def get_average(self, duration_minutes: int = 5) -> Optional[float]:
        """Get average value over duration"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        recent_points = [
            point.value for point in self.data_points
            if point.timestamp >= cutoff_time
        ]
        
        if recent_points:
            return statistics.mean(recent_points)
        return None


class SimpleBatcher:
    """Simplified message batcher for demonstration"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.batches: Dict[str, List[SimpleMessage]] = defaultdict(list)
        self.batch_handlers = []
        self.stats = {
            "messages_batched": 0,
            "batches_processed": 0,
            "total_batch_time": 0.0
        }
    
    def add_batch_handler(self, handler):
        """Add a batch processing handler"""
        self.batch_handlers.append(handler)
    
    async def add_message(self, message: SimpleMessage) -> bool:
        """Add message to batch"""
        try:
            batch_key = self._get_batch_key(message)
            self.batches[batch_key].append(message)
            
            # Check if batch should be processed
            if self._should_process_batch(batch_key):
                await self._process_batch(batch_key)
            
            return True
        except Exception as e:
            print(f"Error adding message to batch: {e}")
            return False
    
    def _get_batch_key(self, message: SimpleMessage) -> str:
        """Generate batch key"""
        key_parts = []
        
        if self.config.batch_by_target:
            targets = sorted(message.target_agents)
            key_parts.append(f"targets:{','.join(targets)}")
        
        if self.config.enable_priority_batching:
            key_parts.append(f"priority:{message.priority}")
        
        return "|".join(key_parts) if key_parts else "default"
    
    def _should_process_batch(self, batch_key: str) -> bool:
        """Check if batch should be processed"""
        batch = self.batches[batch_key]
        return len(batch) >= self.config.max_batch_size
    
    async def _process_batch(self, batch_key: str):
        """Process a batch of messages"""
        start_time = time.time()
        
        batch = self.batches[batch_key]
        self.batches[batch_key] = []
        
        if not batch:
            return
        
        # Update stats
        self.stats["messages_batched"] += len(batch)
        self.stats["batches_processed"] += 1
        
        # Process with handlers
        for handler in self.batch_handlers:
            await handler(batch, batch_key)
        
        # Update timing
        processing_time = time.time() - start_time
        self.stats["total_batch_time"] += processing_time
        
        print(f"Processed batch {batch_key} with {len(batch)} messages in {processing_time:.3f}s")


class SimpleCompressor:
    """Simplified message compressor for demonstration"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.stats = {
            "messages_compressed": 0,
            "total_compression_time": 0.0,
            "compression_ratio_sum": 0.0
        }
    
    async def compress_message(self, message: SimpleMessage) -> tuple:
        """Compress a message"""
        start_time = time.time()
        
        # Serialize message
        json_data = json.dumps(message.to_dict()).encode('utf-8')
        original_size = len(json_data)
        
        # Check if compression is beneficial
        if original_size < self.config.threshold_bytes:
            return json_data, {"algorithm": "none", "original_size": original_size}
        
        # Compress data
        if self.config.algorithm == CompressionAlgorithm.GZIP:
            compressed_data = gzip.compress(json_data, compresslevel=self.config.compression_level)
        else:
            compressed_data = json_data
        
        compression_ratio = len(compressed_data) / original_size
        compression_time = time.time() - start_time
        
        # Update stats
        self.stats["messages_compressed"] += 1
        self.stats["total_compression_time"] += compression_time
        self.stats["compression_ratio_sum"] += compression_ratio
        
        metadata = {
            "algorithm": self.config.algorithm.value,
            "original_size": original_size,
            "compressed_size": len(compressed_data),
            "compression_ratio": compression_ratio,
            "compression_time": compression_time
        }
        
        return compressed_data, metadata
    
    async def decompress_message(self, data: bytes, metadata: Dict[str, Any]) -> SimpleMessage:
        """Decompress message data"""
        algorithm = metadata.get("algorithm", "none")
        
        if algorithm == "gzip":
            decompressed_data = gzip.decompress(data)
            json_str = decompressed_data.decode('utf-8')
        else:
            json_str = data.decode('utf-8')
        
        message_dict = json.loads(json_str)
        return SimpleMessage(
            id=message_dict["id"],
            type=message_dict["type"],
            source_agent=message_dict["source_agent"],
            target_agents=message_dict["target_agents"],
            payload=message_dict["payload"],
            timestamp=datetime.fromisoformat(message_dict["timestamp"]),
            priority=message_dict["priority"]
        )


class SimpleMetricsCollector:
    """Simplified metrics collector for demonstration"""
    
    def __init__(self):
        self.metrics: Dict[str, SimpleMetric] = {}
    
    def record_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Record a counter metric"""
        if name not in self.metrics:
            self.metrics[name] = SimpleMetric(name, MetricType.COUNTER)
        
        current_value = self.metrics[name].get_latest_value() or 0
        self.metrics[name].add_point(current_value + value, labels)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a gauge metric"""
        if name not in self.metrics:
            self.metrics[name] = SimpleMetric(name, MetricType.GAUGE)
        
        self.metrics[name].add_point(value, labels)
    
    def record_timer(self, name: str, duration_seconds: float, labels: Dict[str, str] = None):
        """Record a timer metric"""
        if name not in self.metrics:
            self.metrics[name] = SimpleMetric(name, MetricType.TIMER)
        
        self.metrics[name].add_point(duration_seconds, labels)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        
        for name, metric in self.metrics.items():
            latest_value = metric.get_latest_value()
            avg_5min = metric.get_average(5)
            
            summary[name] = {
                "type": metric.type.value,
                "description": metric.description,
                "latest_value": latest_value,
                "average_5min": avg_5min,
                "data_points_count": len(metric.data_points)
            }
        
        return summary


async def demo_message_batching():
    """Demonstrate message batching functionality"""
    print("\n🔄 Message Batching Demo")
    print("-" * 40)
    
    # Create batcher
    config = BatchConfig(max_batch_size=3, batch_by_target=True)
    batcher = SimpleBatcher(config)
    
    # Add batch handler
    async def batch_handler(messages, batch_key):
        print(f"  📦 Processed batch '{batch_key}' with {len(messages)} messages")
        for msg in messages:
            print(f"    - {msg.id}: {msg.type} -> {msg.target_agents}")
    
    batcher.add_batch_handler(batch_handler)
    
    # Create test messages
    messages = [
        SimpleMessage(
            id=f"msg_{i}",
            type="context_update",
            source_agent="agent_1",
            target_agents=["agent_2", "agent_3"],
            payload={"data": f"message_{i}"},
            timestamp=datetime.utcnow(),
            priority=2 if i % 2 == 0 else 3
        )
        for i in range(7)
    ]
    
    # Add messages to batcher
    for message in messages:
        await batcher.add_message(message)
        await asyncio.sleep(0.01)  # Small delay
    
    print(f"\n📊 Batching Stats:")
    print(f"  - Messages batched: {batcher.stats['messages_batched']}")
    print(f"  - Batches processed: {batcher.stats['batches_processed']}")
    print(f"  - Total batch time: {batcher.stats['total_batch_time']:.3f}s")


async def demo_message_compression():
    """Demonstrate message compression functionality"""
    print("\n🗜️  Message Compression Demo")
    print("-" * 40)
    
    # Create compressor
    config = CompressionConfig(algorithm=CompressionAlgorithm.GZIP, threshold_bytes=100)
    compressor = SimpleCompressor(config)
    
    # Create test messages of different sizes
    messages = [
        SimpleMessage(
            id="small_msg",
            type="heartbeat",
            source_agent="agent_1",
            target_agents=["agent_2"],
            payload={"status": "ok"},
            timestamp=datetime.utcnow()
        ),
        SimpleMessage(
            id="large_msg",
            type="context_update",
            source_agent="agent_1",
            target_agents=["agent_2", "agent_3"],
            payload={
                "large_data": "x" * 2000,
                "numbers": list(range(100)),
                "nested": {"deep": {"data": "y" * 500}}
            },
            timestamp=datetime.utcnow()
        )
    ]
    
    for message in messages:
        compressed_data, metadata = await compressor.compress_message(message)
        
        print(f"\n📄 Message: {message.id}")
        print(f"  - Algorithm: {metadata['algorithm']}")
        print(f"  - Original size: {metadata['original_size']} bytes")
        print(f"  - Compressed size: {metadata.get('compressed_size', metadata['original_size'])} bytes")
        
        if 'compression_ratio' in metadata:
            ratio = metadata['compression_ratio']
            savings = (1 - ratio) * 100
            print(f"  - Compression ratio: {ratio:.3f}")
            print(f"  - Space saved: {savings:.1f}%")
        
        # Test decompression
        decompressed_message = await compressor.decompress_message(compressed_data, metadata)
        assert decompressed_message.id == message.id
        print(f"  - ✓ Decompression successful")
    
    print(f"\n📊 Compression Stats:")
    print(f"  - Messages compressed: {compressor.stats['messages_compressed']}")
    print(f"  - Total compression time: {compressor.stats['total_compression_time']:.3f}s")
    if compressor.stats['messages_compressed'] > 0:
        avg_ratio = compressor.stats['compression_ratio_sum'] / compressor.stats['messages_compressed']
        print(f"  - Average compression ratio: {avg_ratio:.3f}")


def demo_metrics_collection():
    """Demonstrate metrics collection functionality"""
    print("\n📊 Metrics Collection Demo")
    print("-" * 40)
    
    # Create metrics collector
    collector = SimpleMetricsCollector()
    
    # Simulate some operations with metrics
    print("Simulating MCP operations...")
    
    for i in range(10):
        # Record message processing
        collector.record_counter("messages_processed")
        collector.record_timer("message_processing_time", 0.1 + (i * 0.01))
        
        # Record system metrics
        collector.record_gauge("cpu_usage", 50 + (i * 2))
        collector.record_gauge("memory_usage", 60 + (i * 1.5))
        
        # Record some failures occasionally
        if i % 3 == 0:
            collector.record_counter("message_failures")
    
    # Get metrics summary
    summary = collector.get_metrics_summary()
    
    print(f"\n📈 Metrics Summary:")
    for name, data in summary.items():
        print(f"  - {name}:")
        print(f"    Type: {data['type']}")
        print(f"    Latest: {data['latest_value']}")
        print(f"    5min avg: {data['average_5min']}")
        print(f"    Data points: {data['data_points_count']}")


def demo_performance_calculations():
    """Demonstrate performance calculation utilities"""
    print("\n⚡ Performance Calculations Demo")
    print("-" * 40)
    
    # Simulate performance data
    response_times = [0.08, 0.12, 0.15, 0.09, 0.18, 0.11, 0.14, 0.16, 0.10, 0.13,
                     0.20, 0.07, 0.19, 0.12, 0.15, 0.08, 0.17, 0.11, 0.14, 0.09]
    
    throughput_data = [95, 102, 98, 105, 92, 108, 100, 97, 110, 103,
                      88, 112, 99, 106, 94, 109, 101, 96, 107, 104]
    
    error_counts = [0, 1, 0, 0, 2, 0, 1, 0, 0, 1,
                   3, 0, 1, 0, 0, 2, 0, 1, 0, 0]
    
    # Calculate statistics
    avg_response_time = statistics.mean(response_times)
    median_response_time = statistics.median(response_times)
    p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
    p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)]
    
    avg_throughput = statistics.mean(throughput_data)
    max_throughput = max(throughput_data)
    min_throughput = min(throughput_data)
    
    total_requests = len(response_times)
    total_errors = sum(error_counts)
    error_rate = (total_errors / total_requests) * 100
    
    print(f"📊 Response Time Analysis:")
    print(f"  - Average: {avg_response_time:.3f}s")
    print(f"  - Median: {median_response_time:.3f}s")
    print(f"  - P95: {p95_response_time:.3f}s")
    print(f"  - P99: {p99_response_time:.3f}s")
    
    print(f"\n📈 Throughput Analysis:")
    print(f"  - Average: {avg_throughput:.1f} msg/s")
    print(f"  - Maximum: {max_throughput} msg/s")
    print(f"  - Minimum: {min_throughput} msg/s")
    
    print(f"\n❌ Error Analysis:")
    print(f"  - Total requests: {total_requests}")
    print(f"  - Total errors: {total_errors}")
    print(f"  - Error rate: {error_rate:.2f}%")
    
    # Performance thresholds
    warning_threshold = 0.15
    critical_threshold = 0.20
    
    warning_violations = sum(1 for rt in response_times if rt > warning_threshold)
    critical_violations = sum(1 for rt in response_times if rt > critical_threshold)
    
    print(f"\n⚠️  Threshold Analysis:")
    print(f"  - Warning threshold ({warning_threshold}s): {warning_violations} violations")
    print(f"  - Critical threshold ({critical_threshold}s): {critical_violations} violations")
    
    # Health score calculation
    health_score = 100.0
    health_score -= critical_violations * 10  # -10 points per critical violation
    health_score -= warning_violations * 5   # -5 points per warning violation
    health_score -= error_rate * 2           # -2 points per % error rate
    
    if avg_response_time > 0.15:
        health_score -= 10
    
    health_score = max(0, min(100, health_score))
    
    print(f"\n🏥 System Health Score: {health_score:.1f}/100")
    
    if health_score >= 90:
        status = "🟢 Excellent"
    elif health_score >= 75:
        status = "🟡 Good"
    elif health_score >= 60:
        status = "🟠 Warning"
    else:
        status = "🔴 Critical"
    
    print(f"   Status: {status}")


async def main():
    """Run all demonstrations"""
    print("🚀 MCP Performance Optimization Demo")
    print("=" * 60)
    print("This demo showcases the key concepts and algorithms")
    print("implemented in the MCP performance optimization system.")
    print("=" * 60)
    
    # Run demonstrations
    await demo_message_batching()
    await demo_message_compression()
    demo_metrics_collection()
    demo_performance_calculations()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("• Message batching for improved throughput")
    print("• Intelligent compression for bandwidth optimization")
    print("• Comprehensive metrics collection and analysis")
    print("• Performance monitoring and health scoring")
    print("• Threshold-based alerting and violation detection")
    print("\n🎯 The performance optimization system is ready for integration!")


if __name__ == "__main__":
    asyncio.run(main())