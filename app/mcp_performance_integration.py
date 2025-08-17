# app/mcp_performance_integration.py
"""
MCP Performance Integration

This module integrates the performance optimization and monitoring systems
with the existing MCP infrastructure, providing seamless performance enhancements.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from .mcp_server import MCPServer
from .mcp_performance_optimizer import (
    MCPPerformanceOptimizer, 
    BatchConfig, 
    CompressionConfig, 
    ConnectionPoolConfig
)
from .mcp_monitoring_system import (
    get_monitoring_system,
    MCPMetricsCollector,
    GitWorkflowMonitor,
    AlertManager,
    SystemHealthDashboard
)
from .mcp_models import MCPMessage
from .config import settings

logger = logging.getLogger(__name__)


class EnhancedMCPServer(MCPServer):
    """
    Enhanced MCP Server with integrated performance optimization and monitoring
    
    Extends the base MCP server with performance optimizations, comprehensive
    monitoring, and intelligent alerting capabilities.
    """
    
    def __init__(self, redis_url: str = None, max_connections: int = 20):
        super().__init__(redis_url, max_connections)
        
        # Performance optimization
        self.performance_optimizer: Optional[MCPPerformanceOptimizer] = None
        
        # Monitoring system
        self.metrics_collector: Optional[MCPMetricsCollector] = None
        self.git_monitor: Optional[GitWorkflowMonitor] = None
        self.alert_manager: Optional[AlertManager] = None
        self.health_dashboard: Optional[SystemHealthDashboard] = None
        
        # Performance tracking
        self._message_start_times: Dict[str, float] = {}
        
        logger.info("Enhanced MCP Server initialized")
    
    async def start_server(self, host: str = "localhost", port: int = 8765) -> bool:
        """
        Start the enhanced MCP server with all optimizations and monitoring
        
        Args:
            host: Server host address
            port: Server port number
            
        Returns:
            True if server started successfully
        """
        try:
            # Initialize monitoring system
            await self._initialize_monitoring_system()
            
            # Initialize performance optimizer
            await self._initialize_performance_optimizer()
            
            # Start base server
            if not await super().start_server(host, port):
                return False
            
            # Record server start metrics
            self.metrics_collector.record_counter("mcp_server_starts")
            self.metrics_collector.record_gauge("mcp_server_status", 1)
            
            logger.info("Enhanced MCP Server started with performance optimizations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start enhanced MCP server: {e}")
            if self.metrics_collector:
                self.metrics_collector.record_counter("mcp_server_start_failures")
            return False
    
    async def stop_server(self) -> None:
        """Stop the enhanced MCP server and cleanup all components"""
        try:
            # Record server stop metrics
            if self.metrics_collector:
                self.metrics_collector.record_counter("mcp_server_stops")
                self.metrics_collector.record_gauge("mcp_server_status", 0)
            
            # Stop performance optimizer
            if self.performance_optimizer:
                await self.performance_optimizer.stop()
            
            # Stop monitoring components
            if self.alert_manager:
                await self.alert_manager.stop()
            if self.metrics_collector:
                await self.metrics_collector.stop()
            
            # Stop base server
            await super().stop_server()
            
            logger.info("Enhanced MCP Server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping enhanced MCP server: {e}")
    
    async def route_message(self, message: MCPMessage) -> bool:
        """
        Route message with performance optimization and monitoring
        
        Args:
            message: MCP message to route
            
        Returns:
            True if message was routed successfully
        """
        start_time = time.time()
        self._message_start_times[message.id] = start_time
        
        try:
            # Record message metrics
            self.metrics_collector.record_counter("mcp_messages_total")
            self.metrics_collector.record_counter(
                "mcp_messages_by_type",
                labels={"message_type": message.type}
            )
            self.metrics_collector.record_gauge("mcp_active_connections", len(self._registered_agents))
            
            # Use performance optimizer if available
            if self.performance_optimizer:
                success = await self.performance_optimizer.process_message(message)
            else:
                # Fallback to base implementation
                success = await super().route_message(message)
            
            # Record completion metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_timer("mcp_message_processing_duration", processing_time)
            
            if success:
                self.metrics_collector.record_counter("mcp_messages_successful")
                self.metrics_collector.record_gauge(
                    "mcp_messages_per_second",
                    self._calculate_messages_per_second()
                )
            else:
                self.metrics_collector.record_counter("mcp_failed_messages")
            
            # Clean up tracking
            self._message_start_times.pop(message.id, None)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in enhanced message routing: {e}")
            
            # Record error metrics
            self.metrics_collector.record_counter("mcp_routing_errors")
            processing_time = time.time() - start_time
            self.metrics_collector.record_timer("mcp_message_processing_duration", processing_time)
            
            # Clean up tracking
            self._message_start_times.pop(message.id, None)
            
            return False
    
    async def broadcast_message(self, message: MCPMessage, target_type: str = None) -> int:
        """
        Broadcast message with performance optimization and monitoring
        
        Args:
            message: MCP message to broadcast
            target_type: Optional filter by agent capability type
            
        Returns:
            Number of agents the message was sent to
        """
        start_time = time.time()
        
        try:
            # Record broadcast metrics
            self.metrics_collector.record_counter("mcp_broadcasts_total")
            
            # Use performance optimizer for batch processing if available
            if self.performance_optimizer and len(message.target_agents) > 1:
                # Create individual messages for each target
                individual_messages = []
                for target_agent in message.target_agents:
                    individual_message = MCPMessage(
                        id=f"{message.id}_{target_agent}",
                        type=message.type,
                        source_agent=message.source_agent,
                        target_agents=[target_agent],
                        payload=message.payload,
                        timestamp=message.timestamp,
                        priority=message.priority,
                        requires_ack=message.requires_ack,
                        correlation_id=message.correlation_id,
                        context_version=message.context_version
                    )
                    individual_messages.append(individual_message)
                
                # Process as batch
                results = await self.performance_optimizer.process_messages_batch(individual_messages)
                success_count = sum(1 for success in results.values() if success)
            else:
                # Use base implementation
                success_count = await super().broadcast_message(message, target_type)
            
            # Record metrics
            broadcast_time = time.time() - start_time
            self.metrics_collector.record_timer("mcp_broadcast_duration", broadcast_time)
            self.metrics_collector.record_gauge("mcp_broadcast_success_count", success_count)
            
            return success_count
            
        except Exception as e:
            logger.error(f"Error in enhanced message broadcast: {e}")
            self.metrics_collector.record_counter("mcp_broadcast_errors")
            return 0
    
    async def register_agent(self, agent_id: str, capabilities: List[Dict[str, Any]]) -> str:
        """
        Register agent with enhanced monitoring
        
        Args:
            agent_id: Unique agent identifier
            capabilities: List of agent capabilities
            
        Returns:
            Connection ID for the registered agent
        """
        try:
            connection_id = await super().register_agent(agent_id, capabilities)
            
            # Record registration metrics
            self.metrics_collector.record_counter("mcp_agent_registrations")
            self.metrics_collector.record_gauge("mcp_active_connections", len(self._registered_agents))
            
            # Record capability metrics
            for capability in capabilities:
                self.metrics_collector.record_counter(
                    "mcp_agent_capabilities",
                    labels={"capability": capability.get("name", "unknown")}
                )
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Error in enhanced agent registration: {e}")
            self.metrics_collector.record_counter("mcp_agent_registration_failures")
            raise
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister agent with enhanced monitoring
        
        Args:
            agent_id: Agent identifier to unregister
            
        Returns:
            True if agent was unregistered successfully
        """
        try:
            success = await super().unregister_agent(agent_id)
            
            if success:
                self.metrics_collector.record_counter("mcp_agent_unregistrations")
                self.metrics_collector.record_gauge("mcp_active_connections", len(self._registered_agents))
            
            return success
            
        except Exception as e:
            logger.error(f"Error in enhanced agent unregistration: {e}")
            return False
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            stats = {
                "server": await super().get_server_status(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add performance optimizer stats
            if self.performance_optimizer:
                perf_stats = await self.performance_optimizer.get_performance_stats()
                stats["performance_optimization"] = perf_stats
            
            # Add monitoring stats
            if self.health_dashboard:
                dashboard_data = await self.health_dashboard.get_dashboard_data()
                stats["monitoring"] = dashboard_data
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {"error": str(e)}
    
    async def _initialize_monitoring_system(self):
        """Initialize the monitoring system components"""
        try:
            # Get monitoring system components
            (self.metrics_collector, 
             self.git_monitor, 
             self.alert_manager, 
             self.health_dashboard) = get_monitoring_system()
            
            # Start monitoring components
            await self.metrics_collector.start()
            await self.alert_manager.start()
            
            # Add custom notification handler for alerts
            self.alert_manager.add_notification_handler(self._handle_alert_notification)
            
            logger.info("Monitoring system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {e}")
            raise
    
    async def _initialize_performance_optimizer(self):
        """Initialize the performance optimization system"""
        try:
            # Create performance optimizer with custom configuration
            batch_config = BatchConfig(
                max_batch_size=getattr(settings, 'MCP_BATCH_SIZE', 100),
                max_batch_wait_ms=getattr(settings, 'MCP_BATCH_WAIT_MS', 50),
                enable_priority_batching=True,
                batch_by_target=True
            )
            
            compression_config = CompressionConfig(
                threshold_bytes=getattr(settings, 'MCP_COMPRESSION_THRESHOLD', 1024),
                enable_adaptive_compression=True
            )
            
            pool_config = ConnectionPoolConfig(
                max_connections=self.max_connections,
                min_connections=5,
                health_check_interval=30
            )
            
            self.performance_optimizer = MCPPerformanceOptimizer(
                redis_url=self.redis_url,
                batch_config=batch_config,
                compression_config=compression_config,
                pool_config=pool_config
            )
            
            # Start performance optimizer
            await self.performance_optimizer.start()
            
            logger.info("Performance optimizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance optimizer: {e}")
            raise
    
    async def _handle_alert_notification(self, alert):
        """Handle alert notifications"""
        try:
            logger.warning(f"Alert triggered: {alert.message} (Severity: {alert.severity.value})")
            
            # You could add additional notification logic here:
            # - Send email notifications
            # - Post to Slack/Discord
            # - Trigger automated responses
            # - etc.
            
        except Exception as e:
            logger.error(f"Error handling alert notification: {e}")
    
    def _calculate_messages_per_second(self) -> float:
        """Calculate current messages per second rate"""
        try:
            # Simple implementation - in production you'd use a sliding window
            current_time = time.time()
            recent_messages = [
                start_time for start_time in self._message_start_times.values()
                if current_time - start_time <= 60  # Last minute
            ]
            
            return len(recent_messages) / 60.0
            
        except Exception:
            return 0.0


class MCPPerformanceMiddleware:
    """
    Middleware for adding performance monitoring to existing MCP operations
    
    Can be used to wrap existing MCP components with performance tracking
    without modifying their core implementation.
    """
    
    def __init__(self, metrics_collector: MCPMetricsCollector):
        self.metrics_collector = metrics_collector
        self.operation_timers: Dict[str, float] = {}
    
    def start_operation(self, operation_name: str, operation_id: str = None) -> str:
        """Start timing an operation"""
        op_id = operation_id or f"{operation_name}_{int(time.time() * 1000)}"
        self.operation_timers[op_id] = time.time()
        
        self.metrics_collector.record_counter(
            f"mcp_operation_starts",
            labels={"operation": operation_name}
        )
        
        return op_id
    
    def end_operation(self, operation_id: str, operation_name: str, success: bool = True):
        """End timing an operation and record metrics"""
        if operation_id not in self.operation_timers:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return
        
        duration = time.time() - self.operation_timers.pop(operation_id)
        
        self.metrics_collector.record_timer(
            f"mcp_operation_duration",
            duration,
            labels={"operation": operation_name, "success": str(success)}
        )
        
        self.metrics_collector.record_counter(
            f"mcp_operation_completions",
            labels={"operation": operation_name, "success": str(success)}
        )
    
    def operation_decorator(self, operation_name: str):
        """Decorator for automatically timing operations"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                op_id = self.start_operation(operation_name)
                try:
                    result = await func(*args, **kwargs)
                    self.end_operation(op_id, operation_name, True)
                    return result
                except Exception as e:
                    self.end_operation(op_id, operation_name, False)
                    raise
            return wrapper
        return decorator


# Factory function for creating enhanced MCP server
def create_enhanced_mcp_server(redis_url: str = None, max_connections: int = 20) -> EnhancedMCPServer:
    """
    Factory function to create an enhanced MCP server with all optimizations
    
    Args:
        redis_url: Redis connection URL
        max_connections: Maximum number of connections
        
    Returns:
        Configured enhanced MCP server instance
    """
    return EnhancedMCPServer(redis_url, max_connections)


# Global enhanced server instance
enhanced_mcp_server = None

def get_enhanced_mcp_server() -> EnhancedMCPServer:
    """Get or create the global enhanced MCP server instance"""
    global enhanced_mcp_server
    
    if enhanced_mcp_server is None:
        enhanced_mcp_server = create_enhanced_mcp_server()
    
    return enhanced_mcp_server