# app/mcp_performance_optimizer.py
"""
MCP Performance Optimization Module

This module provides performance optimizations for MCP operations including:
- Message batching for high-throughput scenarios
- Enhanced connection pooling for Redis connections
- Message compression for large payloads
- Performance monitoring and metrics collection
"""

import asyncio
import gzip
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
import statistics

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from .mcp_models import MCPMessage, SerializationFormat
from .config import settings

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Supported compression algorithms"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"  # Future implementation


@dataclass
class BatchConfig:
    """Configuration for message batching"""
    max_batch_size: int = 100
    max_batch_wait_ms: int = 50
    enable_priority_batching: bool = True
    batch_by_target: bool = True
    batch_by_type: bool = False
    compression_threshold: int = 1024
    max_memory_per_batch: int = 1024 * 1024  # 1MB


@dataclass
class ConnectionPoolConfig:
    """Enhanced connection pool configuration"""
    max_connections: int = 50
    min_connections: int = 5
    connection_timeout: int = 10
    socket_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    max_idle_time: int = 300  # 5 minutes
    connection_pool_class_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionConfig:
    """Configuration for message compression"""
    algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    compression_level: int = 6
    threshold_bytes: int = 1024
    enable_adaptive_compression: bool = True
    compression_ratio_threshold: float = 0.7  # Only compress if ratio < 0.7


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    messages_processed: int = 0
    messages_batched: int = 0
    messages_compressed: int = 0
    total_processing_time: float = 0.0
    total_compression_time: float = 0.0
    total_batch_time: float = 0.0
    compression_ratio_sum: float = 0.0
    batch_sizes: List[int] = field(default_factory=list)
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_count: int = 0
    connection_pool_hits: int = 0
    connection_pool_misses: int = 0


class MessageBatcher:
    """
    High-performance message batching system
    
    Provides intelligent batching of messages for improved throughput
    with configurable batching strategies and compression support.
    """
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        
        # Batching state
        self._batches: Dict[str, List[MCPMessage]] = defaultdict(list)
        self._batch_timers: Dict[str, asyncio.Task] = {}
        self._batch_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Callbacks for batch processing
        self._batch_handlers: List[Callable] = []
        
        # Metrics
        self.metrics = PerformanceMetrics()
        
        # Background tasks
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()
        
        logger.info("Message batcher initialized")
    
    async def start(self):
        """Start the message batcher"""
        if self._running:
            return
        
        self._running = True
        
        # Start background monitoring task
        monitor_task = asyncio.create_task(self._monitor_batches())
        self._background_tasks.add(monitor_task)
        
        logger.info("Message batcher started")
    
    async def stop(self):
        """Stop the message batcher and flush remaining messages"""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Flush all remaining batches
        await self._flush_all_batches()
        
        logger.info("Message batcher stopped")
    
    def add_batch_handler(self, handler: Callable[[List[MCPMessage], str], None]):
        """Add a handler for processing batches"""
        self._batch_handlers.append(handler)
    
    async def add_message(self, message: MCPMessage) -> bool:
        """
        Add a message to the batching system
        
        Args:
            message: MCP message to batch
            
        Returns:
            True if message was added successfully
        """
        try:
            # Determine batch key based on configuration
            batch_key = self._get_batch_key(message)
            
            async with self._batch_locks[batch_key]:
                # Add message to batch
                self._batches[batch_key].append(message)
                
                # Check if batch should be processed immediately
                if await self._should_process_batch(batch_key):
                    await self._process_batch(batch_key)
                else:
                    # Set timer for batch processing if not already set
                    if batch_key not in self._batch_timers:
                        timer_task = asyncio.create_task(
                            self._batch_timer(batch_key, self.config.max_batch_wait_ms / 1000.0)
                        )
                        self._batch_timers[batch_key] = timer_task
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding message to batch: {e}")
            self.metrics.error_count += 1
            return False
    
    async def flush_batch(self, batch_key: str) -> bool:
        """
        Manually flush a specific batch
        
        Args:
            batch_key: Key of the batch to flush
            
        Returns:
            True if batch was flushed successfully
        """
        try:
            async with self._batch_locks[batch_key]:
                if batch_key in self._batches and self._batches[batch_key]:
                    await self._process_batch(batch_key)
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Error flushing batch {batch_key}: {e}")
            return False
    
    async def get_batch_stats(self) -> Dict[str, Any]:
        """Get current batching statistics"""
        active_batches = {
            key: len(messages) 
            for key, messages in self._batches.items() 
            if messages
        }
        
        avg_batch_size = (
            statistics.mean(self.metrics.batch_sizes) 
            if self.metrics.batch_sizes else 0
        )
        
        avg_processing_time = (
            statistics.mean(self.metrics.processing_times)
            if self.metrics.processing_times else 0
        )
        
        return {
            "active_batches": active_batches,
            "total_messages_processed": self.metrics.messages_processed,
            "total_messages_batched": self.metrics.messages_batched,
            "average_batch_size": avg_batch_size,
            "average_processing_time_ms": avg_processing_time * 1000,
            "total_processing_time": self.metrics.total_processing_time,
            "error_count": self.metrics.error_count
        }
    
    def _get_batch_key(self, message: MCPMessage) -> str:
        """Generate batch key based on configuration"""
        key_parts = []
        
        if self.config.batch_by_target:
            # Sort targets for consistent key generation
            targets = sorted(message.target_agents)
            key_parts.append(f"targets:{','.join(targets)}")
        
        if self.config.batch_by_type:
            key_parts.append(f"type:{message.type}")
        
        if self.config.enable_priority_batching:
            key_parts.append(f"priority:{message.priority}")
        
        # Default key if no specific batching criteria
        if not key_parts:
            key_parts.append("default")
        
        return "|".join(key_parts)
    
    async def _should_process_batch(self, batch_key: str) -> bool:
        """Check if batch should be processed immediately"""
        batch = self._batches[batch_key]
        
        # Check batch size limit
        if len(batch) >= self.config.max_batch_size:
            return True
        
        # Check memory usage
        batch_memory = sum(len(json.dumps(msg.to_dict())) for msg in batch)
        if batch_memory >= self.config.max_memory_per_batch:
            return True
        
        # Check for high priority messages
        if self.config.enable_priority_batching:
            high_priority_count = sum(1 for msg in batch if msg.priority >= 3)
            if high_priority_count > 0 and len(batch) >= 10:  # Process high priority batches sooner
                return True
        
        return False
    
    async def _process_batch(self, batch_key: str):
        """Process a batch of messages"""
        try:
            start_time = time.time()
            
            # Get and clear the batch
            batch = self._batches[batch_key]
            self._batches[batch_key] = []
            
            # Cancel timer if exists
            if batch_key in self._batch_timers:
                self._batch_timers[batch_key].cancel()
                del self._batch_timers[batch_key]
            
            if not batch:
                return
            
            # Update metrics
            self.metrics.messages_batched += len(batch)
            self.metrics.batch_sizes.append(len(batch))
            
            # Process batch with all handlers
            for handler in self._batch_handlers:
                try:
                    await handler(batch, batch_key)
                except Exception as e:
                    logger.error(f"Error in batch handler: {e}")
                    self.metrics.error_count += 1
            
            # Update timing metrics
            processing_time = time.time() - start_time
            self.metrics.total_batch_time += processing_time
            self.metrics.processing_times.append(processing_time)
            
            logger.debug(f"Processed batch {batch_key} with {len(batch)} messages in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_key}: {e}")
            self.metrics.error_count += 1
    
    async def _batch_timer(self, batch_key: str, delay: float):
        """Timer for batch processing"""
        try:
            await asyncio.sleep(delay)
            
            async with self._batch_locks[batch_key]:
                if batch_key in self._batches and self._batches[batch_key]:
                    await self._process_batch(batch_key)
                    
        except asyncio.CancelledError:
            pass  # Timer was cancelled, which is normal
        except Exception as e:
            logger.error(f"Error in batch timer for {batch_key}: {e}")
    
    async def _monitor_batches(self):
        """Background task to monitor batch health"""
        while self._running:
            try:
                # Check for stale batches
                current_time = time.time()
                stale_batches = []
                
                for batch_key, batch in self._batches.items():
                    if batch:
                        # Check if oldest message in batch is too old
                        oldest_message = min(batch, key=lambda m: m.timestamp)
                        age = (datetime.utcnow() - oldest_message.timestamp).total_seconds()
                        
                        if age > (self.config.max_batch_wait_ms / 1000.0) * 2:  # 2x the normal wait time
                            stale_batches.append(batch_key)
                
                # Process stale batches
                for batch_key in stale_batches:
                    logger.warning(f"Processing stale batch: {batch_key}")
                    await self.flush_batch(batch_key)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in batch monitor: {e}")
                await asyncio.sleep(10)
    
    async def _flush_all_batches(self):
        """Flush all remaining batches"""
        batch_keys = list(self._batches.keys())
        for batch_key in batch_keys:
            await self.flush_batch(batch_key)


class MessageCompressor:
    """
    High-performance message compression system
    
    Provides intelligent compression of messages with adaptive algorithms
    and performance monitoring.
    """
    
    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.metrics = PerformanceMetrics()
        
        # Compression statistics for adaptive compression
        self._compression_stats: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"Message compressor initialized with {self.config.algorithm.value} algorithm")
    
    async def compress_message(self, message: MCPMessage) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress a message with optimal algorithm selection
        
        Args:
            message: MCP message to compress
            
        Returns:
            Tuple of (compressed_data, compression_metadata)
        """
        try:
            start_time = time.time()
            
            # Serialize message to JSON
            json_data = message.json().encode('utf-8')
            original_size = len(json_data)
            
            # Check if compression is beneficial
            if original_size < self.config.threshold_bytes:
                return json_data, {"algorithm": "none", "original_size": original_size}
            
            # Select compression algorithm
            algorithm = await self._select_compression_algorithm(message.type, original_size)
            
            # Compress data
            compressed_data, compression_ratio = await self._compress_data(json_data, algorithm)
            
            # Update metrics
            compression_time = time.time() - start_time
            self.metrics.messages_compressed += 1
            self.metrics.total_compression_time += compression_time
            self.metrics.compression_ratio_sum += compression_ratio
            
            # Store compression statistics for adaptive algorithm
            self._compression_stats[message.type].append(compression_ratio)
            if len(self._compression_stats[message.type]) > 100:
                self._compression_stats[message.type].pop(0)  # Keep only recent stats
            
            metadata = {
                "algorithm": algorithm.value,
                "original_size": original_size,
                "compressed_size": len(compressed_data),
                "compression_ratio": compression_ratio,
                "compression_time": compression_time
            }
            
            logger.debug(f"Compressed message {message.id}: {original_size} -> {len(compressed_data)} bytes "
                        f"({compression_ratio:.2f} ratio) using {algorithm.value}")
            
            return compressed_data, metadata
            
        except Exception as e:
            logger.error(f"Error compressing message {message.id}: {e}")
            # Return uncompressed data on error
            json_data = message.json().encode('utf-8')
            return json_data, {"algorithm": "none", "error": str(e)}
    
    async def decompress_message(self, data: bytes, metadata: Dict[str, Any]) -> MCPMessage:
        """
        Decompress message data
        
        Args:
            data: Compressed message data
            metadata: Compression metadata
            
        Returns:
            Decompressed MCP message
        """
        try:
            algorithm_name = metadata.get("algorithm", "none")
            
            if algorithm_name == "none":
                json_str = data.decode('utf-8')
            else:
                algorithm = CompressionAlgorithm(algorithm_name)
                decompressed_data = await self._decompress_data(data, algorithm)
                json_str = decompressed_data.decode('utf-8')
            
            return MCPMessage.parse_raw(json_str)
            
        except Exception as e:
            logger.error(f"Error decompressing message: {e}")
            raise
    
    async def compress_batch(self, messages: List[MCPMessage]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress a batch of messages together for better compression ratio
        
        Args:
            messages: List of MCP messages to compress
            
        Returns:
            Tuple of (compressed_data, compression_metadata)
        """
        try:
            start_time = time.time()
            
            # Serialize all messages to a single JSON structure
            batch_data = {
                "messages": [msg.to_dict() for msg in messages],
                "batch_size": len(messages),
                "batch_timestamp": datetime.utcnow().isoformat()
            }
            
            json_data = json.dumps(batch_data).encode('utf-8')
            original_size = len(json_data)
            
            # Always compress batches since they're typically large
            algorithm = self.config.algorithm
            compressed_data, compression_ratio = await self._compress_data(json_data, algorithm)
            
            # Update metrics
            compression_time = time.time() - start_time
            self.metrics.messages_compressed += len(messages)
            self.metrics.total_compression_time += compression_time
            
            metadata = {
                "algorithm": algorithm.value,
                "original_size": original_size,
                "compressed_size": len(compressed_data),
                "compression_ratio": compression_ratio,
                "compression_time": compression_time,
                "batch_size": len(messages)
            }
            
            logger.debug(f"Compressed batch of {len(messages)} messages: {original_size} -> {len(compressed_data)} bytes "
                        f"({compression_ratio:.2f} ratio)")
            
            return compressed_data, metadata
            
        except Exception as e:
            logger.error(f"Error compressing message batch: {e}")
            raise
    
    async def decompress_batch(self, data: bytes, metadata: Dict[str, Any]) -> List[MCPMessage]:
        """
        Decompress a batch of messages
        
        Args:
            data: Compressed batch data
            metadata: Compression metadata
            
        Returns:
            List of decompressed MCP messages
        """
        try:
            algorithm_name = metadata.get("algorithm", "none")
            
            if algorithm_name == "none":
                json_str = data.decode('utf-8')
            else:
                algorithm = CompressionAlgorithm(algorithm_name)
                decompressed_data = await self._decompress_data(data, algorithm)
                json_str = decompressed_data.decode('utf-8')
            
            batch_data = json.loads(json_str)
            messages = []
            
            for msg_dict in batch_data["messages"]:
                message = MCPMessage(**msg_dict)
                messages.append(message)
            
            return messages
            
        except Exception as e:
            logger.error(f"Error decompressing message batch: {e}")
            raise
    
    async def _select_compression_algorithm(self, message_type: str, size: int) -> CompressionAlgorithm:
        """Select optimal compression algorithm based on message type and size"""
        if not self.config.enable_adaptive_compression:
            return self.config.algorithm
        
        # Use historical compression ratios to select algorithm
        if message_type in self._compression_stats:
            avg_ratio = statistics.mean(self._compression_stats[message_type])
            
            # If compression isn't very effective, consider not compressing
            if avg_ratio > self.config.compression_ratio_threshold:
                return CompressionAlgorithm.NONE
        
        # For large messages, use higher compression
        if size > 10 * 1024:  # 10KB
            return CompressionAlgorithm.GZIP
        
        return self.config.algorithm
    
    async def _compress_data(self, data: bytes, algorithm: CompressionAlgorithm) -> Tuple[bytes, float]:
        """Compress data with specified algorithm"""
        if algorithm == CompressionAlgorithm.NONE:
            return data, 1.0
        
        elif algorithm == CompressionAlgorithm.GZIP:
            compressed = gzip.compress(data, compresslevel=self.config.compression_level)
            ratio = len(compressed) / len(data)
            return compressed, ratio
        
        elif algorithm == CompressionAlgorithm.ZLIB:
            import zlib
            compressed = zlib.compress(data, level=self.config.compression_level)
            ratio = len(compressed) / len(data)
            return compressed, ratio
        
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")
    
    async def _decompress_data(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Decompress data with specified algorithm"""
        if algorithm == CompressionAlgorithm.NONE:
            return data
        
        elif algorithm == CompressionAlgorithm.GZIP:
            return gzip.decompress(data)
        
        elif algorithm == CompressionAlgorithm.ZLIB:
            import zlib
            return zlib.decompress(data)
        
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        avg_compression_ratio = (
            self.metrics.compression_ratio_sum / self.metrics.messages_compressed
            if self.metrics.messages_compressed > 0 else 0
        )
        
        return {
            "messages_compressed": self.metrics.messages_compressed,
            "total_compression_time": self.metrics.total_compression_time,
            "average_compression_ratio": avg_compression_ratio,
            "compression_stats_by_type": {
                msg_type: {
                    "count": len(ratios),
                    "average_ratio": statistics.mean(ratios),
                    "min_ratio": min(ratios),
                    "max_ratio": max(ratios)
                }
                for msg_type, ratios in self._compression_stats.items()
                if ratios
            }
        }


class EnhancedConnectionPool:
    """
    Enhanced Redis connection pool with advanced features
    
    Provides intelligent connection management with health monitoring,
    automatic scaling, and performance optimization.
    """
    
    def __init__(self, redis_url: str, config: ConnectionPoolConfig = None):
        self.redis_url = redis_url
        self.config = config or ConnectionPoolConfig()
        
        # Connection pool
        self._pool: Optional[ConnectionPool] = None
        self._redis_clients: Dict[str, redis.Redis] = {}
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_health: Dict[str, bool] = {}
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Background tasks
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()
        
        logger.info("Enhanced connection pool initialized")
    
    async def initialize(self) -> bool:
        """Initialize the connection pool"""
        try:
            # Create connection pool with enhanced configuration
            pool_kwargs = {
                "max_connections": self.config.max_connections,
                "retry_on_timeout": self.config.retry_on_timeout,
                "socket_connect_timeout": self.config.connection_timeout,
                "socket_timeout": self.config.socket_timeout,
                "socket_keepalive": True,
                "socket_keepalive_options": {},
                "health_check_interval": self.config.health_check_interval
            }
            
            # Add custom configuration
            pool_kwargs.update(self.config.connection_pool_class_kwargs)
            
            self._pool = ConnectionPool.from_url(self.redis_url, **pool_kwargs)
            
            # Test initial connection
            test_redis = redis.Redis(connection_pool=self._pool)
            await test_redis.ping()
            await test_redis.close()
            
            # Start background tasks
            self._running = True
            await self._start_background_tasks()
            
            logger.info(f"Connection pool initialized with {self.config.max_connections} max connections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            return False
    
    async def get_connection(self, connection_id: str = None) -> redis.Redis:
        """
        Get a Redis connection from the pool
        
        Args:
            connection_id: Optional connection identifier for reuse
            
        Returns:
            Redis connection instance
        """
        try:
            if connection_id and connection_id in self._redis_clients:
                # Reuse existing connection
                client = self._redis_clients[connection_id]
                
                # Test connection health
                if await self._test_connection_health(client):
                    self.metrics.connection_pool_hits += 1
                    return client
                else:
                    # Remove unhealthy connection
                    del self._redis_clients[connection_id]
            
            # Create new connection
            client = redis.Redis(connection_pool=self._pool)
            
            # Store connection if ID provided
            if connection_id:
                self._redis_clients[connection_id] = client
            
            self.metrics.connection_pool_misses += 1
            return client
            
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
            raise
    
    async def release_connection(self, client: redis.Redis, connection_id: str = None):
        """
        Release a connection back to the pool
        
        Args:
            client: Redis client to release
            connection_id: Optional connection identifier
        """
        try:
            if connection_id and connection_id in self._redis_clients:
                # Keep connection for reuse
                return
            
            # Close connection if not being reused
            await client.close()
            
        except Exception as e:
            logger.error(f"Error releasing connection: {e}")
    
    async def close_all_connections(self):
        """Close all connections and cleanup"""
        try:
            self._running = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close all client connections
            for client in self._redis_clients.values():
                await client.close()
            self._redis_clients.clear()
            
            # Disconnect pool
            if self._pool:
                await self._pool.disconnect()
            
            logger.info("All connections closed")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        pool_stats = {}
        
        if self._pool:
            pool_stats = {
                "max_connections": self.config.max_connections,
                "active_connections": len(self._redis_clients),
                "pool_hits": self.metrics.connection_pool_hits,
                "pool_misses": self.metrics.connection_pool_misses,
                "hit_ratio": (
                    self.metrics.connection_pool_hits / 
                    (self.metrics.connection_pool_hits + self.metrics.connection_pool_misses)
                    if (self.metrics.connection_pool_hits + self.metrics.connection_pool_misses) > 0 
                    else 0
                ),
                "healthy_connections": sum(1 for healthy in self._connection_health.values() if healthy),
                "unhealthy_connections": sum(1 for healthy in self._connection_health.values() if not healthy)
            }
        
        return pool_stats
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Health check task
        health_task = asyncio.create_task(self._health_check_worker())
        self._background_tasks.add(health_task)
        
        # Connection cleanup task
        cleanup_task = asyncio.create_task(self._connection_cleanup_worker())
        self._background_tasks.add(cleanup_task)
        
        logger.info("Started connection pool background tasks")
    
    async def _health_check_worker(self):
        """Background task for connection health monitoring"""
        while self._running:
            try:
                # Check health of all connections
                for conn_id, client in list(self._redis_clients.items()):
                    is_healthy = await self._test_connection_health(client)
                    self._connection_health[conn_id] = is_healthy
                    
                    if not is_healthy:
                        logger.warning(f"Connection {conn_id} is unhealthy")
                        # Remove unhealthy connection
                        try:
                            await client.close()
                        except:
                            pass
                        del self._redis_clients[conn_id]
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check worker: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _connection_cleanup_worker(self):
        """Background task for cleaning up idle connections"""
        while self._running:
            try:
                current_time = time.time()
                idle_connections = []
                
                # Find idle connections (this is a simplified implementation)
                # In a real implementation, you'd track last usage time
                if len(self._redis_clients) > self.config.min_connections:
                    # Remove excess connections beyond minimum
                    excess_count = len(self._redis_clients) - self.config.min_connections
                    connection_ids = list(self._redis_clients.keys())
                    idle_connections = connection_ids[:excess_count]
                
                # Close idle connections
                for conn_id in idle_connections:
                    if conn_id in self._redis_clients:
                        try:
                            await self._redis_clients[conn_id].close()
                            del self._redis_clients[conn_id]
                            logger.debug(f"Closed idle connection {conn_id}")
                        except Exception as e:
                            logger.warning(f"Error closing idle connection {conn_id}: {e}")
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in connection cleanup worker: {e}")
                await asyncio.sleep(60)
    
    async def _test_connection_health(self, client: redis.Redis) -> bool:
        """Test if a Redis connection is healthy"""
        try:
            await asyncio.wait_for(client.ping(), timeout=2.0)
            return True
        except Exception:
            return False


class MCPPerformanceOptimizer:
    """
    Main performance optimization coordinator
    
    Integrates all performance optimization components and provides
    a unified interface for high-performance MCP operations.
    """
    
    def __init__(self, 
                 redis_url: str = None,
                 batch_config: BatchConfig = None,
                 compression_config: CompressionConfig = None,
                 pool_config: ConnectionPoolConfig = None):
        
        self.redis_url = redis_url or settings.REDIS_URL
        
        # Initialize components
        self.batcher = MessageBatcher(batch_config)
        self.compressor = MessageCompressor(compression_config)
        self.connection_pool = EnhancedConnectionPool(self.redis_url, pool_config)
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.total_messages_processed = 0
        
        logger.info("MCP Performance Optimizer initialized")
    
    async def start(self) -> bool:
        """Start all performance optimization components"""
        try:
            # Initialize connection pool
            if not await self.connection_pool.initialize():
                return False
            
            # Start message batcher
            await self.batcher.start()
            
            # Add batch handler for processing
            self.batcher.add_batch_handler(self._process_message_batch)
            
            logger.info("MCP Performance Optimizer started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start performance optimizer: {e}")
            return False
    
    async def stop(self):
        """Stop all performance optimization components"""
        try:
            await self.batcher.stop()
            await self.connection_pool.close_all_connections()
            
            logger.info("MCP Performance Optimizer stopped")
            
        except Exception as e:
            logger.error(f"Error stopping performance optimizer: {e}")
    
    async def process_message(self, message: MCPMessage) -> bool:
        """
        Process a single message with all optimizations
        
        Args:
            message: MCP message to process
            
        Returns:
            True if message was processed successfully
        """
        try:
            # Add to batcher for optimized processing
            success = await self.batcher.add_message(message)
            
            if success:
                self.total_messages_processed += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            return False
    
    async def process_messages_batch(self, messages: List[MCPMessage]) -> Dict[str, bool]:
        """
        Process multiple messages in an optimized batch
        
        Args:
            messages: List of MCP messages to process
            
        Returns:
            Dictionary mapping message IDs to success status
        """
        try:
            results = {}
            
            # Add all messages to batcher
            for message in messages:
                success = await self.batcher.add_message(message)
                results[message.id] = success
                
                if success:
                    self.total_messages_processed += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing message batch: {e}")
            return {msg.id: False for msg in messages}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Get stats from all components
        batch_stats = await self.batcher.get_batch_stats()
        compression_stats = self.compressor.get_compression_stats()
        pool_stats = await self.connection_pool.get_pool_stats()
        
        return {
            "uptime_seconds": uptime,
            "total_messages_processed": self.total_messages_processed,
            "messages_per_second": self.total_messages_processed / uptime if uptime > 0 else 0,
            "batching": batch_stats,
            "compression": compression_stats,
            "connection_pool": pool_stats,
            "memory_usage": {
                # Add memory usage stats if needed
                "estimated_memory_mb": 0  # Placeholder
            }
        }
    
    async def _process_message_batch(self, messages: List[MCPMessage], batch_key: str):
        """Process a batch of messages (called by batcher)"""
        try:
            # Compress batch if beneficial
            if len(messages) > 1:
                compressed_data, metadata = await self.compressor.compress_batch(messages)
                
                # Store compressed batch in Redis
                redis_client = await self.connection_pool.get_connection(f"batch_{batch_key}")
                
                try:
                    # Store batch with metadata
                    batch_id = f"mcp:batch:{batch_key}:{int(time.time())}"
                    
                    pipe = redis_client.pipeline()
                    pipe.set(f"{batch_id}:data", compressed_data)
                    pipe.set(f"{batch_id}:metadata", json.dumps(metadata))
                    pipe.expire(f"{batch_id}:data", 3600)  # 1 hour TTL
                    pipe.expire(f"{batch_id}:metadata", 3600)
                    
                    await pipe.execute()
                    
                    # Notify about batch availability
                    await redis_client.publish(f"mcp:batch_ready:{batch_key}", batch_id)
                    
                finally:
                    await self.connection_pool.release_connection(redis_client, f"batch_{batch_key}")
            
            else:
                # Process single message
                message = messages[0]
                compressed_data, metadata = await self.compressor.compress_message(message)
                
                # Store individual message
                redis_client = await self.connection_pool.get_connection(f"msg_{message.id}")
                
                try:
                    message_key = f"mcp:message:{message.id}"
                    
                    pipe = redis_client.pipeline()
                    pipe.set(f"{message_key}:data", compressed_data)
                    pipe.set(f"{message_key}:metadata", json.dumps(metadata))
                    pipe.expire(f"{message_key}:data", 3600)
                    pipe.expire(f"{message_key}:metadata", 3600)
                    
                    await pipe.execute()
                    
                    # Publish to target agents
                    for target_agent in message.target_agents:
                        await redis_client.publish(f"mcp:agent:{target_agent}:messages", message.id)
                
                finally:
                    await self.connection_pool.release_connection(redis_client, f"msg_{message.id}")
            
            logger.debug(f"Processed batch {batch_key} with {len(messages)} messages")
            
        except Exception as e:
            logger.error(f"Error processing message batch {batch_key}: {e}")
            raise