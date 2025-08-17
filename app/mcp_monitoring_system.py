# app/mcp_monitoring_system.py
"""
MCP Monitoring and Metrics System

This module provides comprehensive monitoring and metrics collection for MCP operations
including performance metrics, system health dashboards, and intelligent alerting.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from enum import Enum
import statistics
import psutil
import threading

import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError

from .config import settings
from .mcp_models import MCPMessage

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: datetime
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels
        }


@dataclass
class Metric:
    """Metric definition and data storage"""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_point(self, value: Union[int, float], labels: Dict[str, str] = None):
        """Add a data point to the metric"""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self.data_points.append(point)
    
    def get_latest_value(self) -> Optional[Union[int, float]]:
        """Get the latest metric value"""
        if self.data_points:
            return self.data_points[-1].value
        return None
    
    def get_average(self, duration_minutes: int = 5) -> Optional[float]:
        """Get average value over specified duration"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        recent_points = [
            point.value for point in self.data_points
            if point.timestamp >= cutoff_time
        ]
        
        if recent_points:
            return statistics.mean(recent_points)
        return None
    
    def get_percentile(self, percentile: float, duration_minutes: int = 5) -> Optional[float]:
        """Get percentile value over specified duration"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        recent_points = [
            point.value for point in self.data_points
            if point.timestamp >= cutoff_time
        ]
        
        if recent_points:
            sorted_points = sorted(recent_points)
            index = int(len(sorted_points) * percentile / 100)
            return sorted_points[min(index, len(sorted_points) - 1)]
        return None


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    metric_name: str
    condition: str  # e.g., "> 100", "< 0.5", "== 0"
    threshold: Union[int, float]
    severity: AlertSeverity
    duration_minutes: int = 1  # How long condition must be true
    description: str = ""
    enabled: bool = True
    
    def evaluate(self, metric: Metric) -> bool:
        """Evaluate if alert condition is met"""
        if not self.enabled:
            return False
        
        # Get recent values for duration check
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.duration_minutes)
        recent_points = [
            point.value for point in metric.data_points
            if point.timestamp >= cutoff_time
        ]
        
        if not recent_points:
            return False
        
        # Check if condition is met for all recent points
        for value in recent_points:
            if not self._check_condition(value):
                return False
        
        return True
    
    def _check_condition(self, value: Union[int, float]) -> bool:
        """Check if a single value meets the condition"""
        if self.condition.startswith(">="):
            return value >= self.threshold
        elif self.condition.startswith("<="):
            return value <= self.threshold
        elif self.condition.startswith(">"):
            return value > self.threshold
        elif self.condition.startswith("<"):
            return value < self.threshold
        elif self.condition.startswith("=="):
            return value == self.threshold
        elif self.condition.startswith("!="):
            return value != self.threshold
        else:
            return False


@dataclass
class Alert:
    """Active alert instance"""
    id: str
    rule_name: str
    metric_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "metadata": self.metadata
        }


class MCPMetricsCollector:
    """
    Comprehensive metrics collection system for MCP operations
    
    Collects and stores various performance and operational metrics
    with efficient storage and retrieval capabilities.
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        
        # Metrics storage
        self.metrics: Dict[str, Metric] = {}
        self.metric_lock = threading.Lock()
        
        # Redis connection for persistence
        self._redis: Optional[redis.Redis] = None
        
        # Background tasks
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Initialize standard MCP metrics
        self._initialize_standard_metrics()
        
        logger.info("MCP Metrics Collector initialized")
    
    async def start(self) -> bool:
        """Start the metrics collector"""
        try:
            # Connect to Redis
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            
            self._running = True
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("MCP Metrics Collector started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start metrics collector: {e}")
            return False
    
    async def stop(self):
        """Stop the metrics collector"""
        try:
            self._running = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self._redis:
                await self._redis.close()
            
            logger.info("MCP Metrics Collector stopped")
            
        except Exception as e:
            logger.error(f"Error stopping metrics collector: {e}")
    
    def record_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Record a counter metric"""
        with self.metric_lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(
                    name=name,
                    type=MetricType.COUNTER,
                    description=f"Counter metric: {name}"
                )
            
            # For counters, we add to the current value
            current_value = self.metrics[name].get_latest_value() or 0
            self.metrics[name].add_point(current_value + value, labels)
    
    def record_gauge(self, name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Record a gauge metric"""
        with self.metric_lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(
                    name=name,
                    type=MetricType.GAUGE,
                    description=f"Gauge metric: {name}"
                )
            
            self.metrics[name].add_point(value, labels)
    
    def record_histogram(self, name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Record a histogram metric"""
        with self.metric_lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(
                    name=name,
                    type=MetricType.HISTOGRAM,
                    description=f"Histogram metric: {name}"
                )
            
            self.metrics[name].add_point(value, labels)
    
    def record_timer(self, name: str, duration_seconds: float, labels: Dict[str, str] = None):
        """Record a timer metric"""
        with self.metric_lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(
                    name=name,
                    type=MetricType.TIMER,
                    description=f"Timer metric: {name}",
                    unit="seconds"
                )
            
            self.metrics[name].add_point(duration_seconds, labels)
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name"""
        with self.metric_lock:
            return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all metrics"""
        with self.metric_lock:
            return self.metrics.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        
        with self.metric_lock:
            for name, metric in self.metrics.items():
                latest_value = metric.get_latest_value()
                avg_5min = metric.get_average(5)
                
                summary[name] = {
                    "type": metric.type.value,
                    "description": metric.description,
                    "unit": metric.unit,
                    "latest_value": latest_value,
                    "average_5min": avg_5min,
                    "data_points_count": len(metric.data_points)
                }
                
                # Add percentiles for histograms and timers
                if metric.type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                    summary[name].update({
                        "p50": metric.get_percentile(50),
                        "p95": metric.get_percentile(95),
                        "p99": metric.get_percentile(99)
                    })
        
        return summary
    
    async def persist_metrics(self):
        """Persist metrics to Redis"""
        try:
            if not self._redis:
                return
            
            with self.metric_lock:
                for name, metric in self.metrics.items():
                    # Store recent data points
                    recent_points = list(metric.data_points)[-100:]  # Last 100 points
                    
                    metric_data = {
                        "name": metric.name,
                        "type": metric.type.value,
                        "description": metric.description,
                        "unit": metric.unit,
                        "labels": metric.labels,
                        "data_points": [point.to_dict() for point in recent_points]
                    }
                    
                    await self._redis.setex(
                        f"mcp:metrics:{name}",
                        3600,  # 1 hour TTL
                        json.dumps(metric_data)
                    )
            
            logger.debug("Metrics persisted to Redis")
            
        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")
    
    def _initialize_standard_metrics(self):
        """Initialize standard MCP metrics"""
        standard_metrics = [
            ("mcp_messages_total", MetricType.COUNTER, "Total MCP messages processed"),
            ("mcp_messages_per_second", MetricType.GAUGE, "MCP messages processed per second"),
            ("mcp_message_processing_duration", MetricType.HISTOGRAM, "Message processing duration", "seconds"),
            ("mcp_active_connections", MetricType.GAUGE, "Number of active MCP connections"),
            ("mcp_failed_messages", MetricType.COUNTER, "Number of failed message deliveries"),
            ("mcp_queue_size", MetricType.GAUGE, "Size of message queues"),
            ("mcp_compression_ratio", MetricType.HISTOGRAM, "Message compression ratio"),
            ("mcp_batch_size", MetricType.HISTOGRAM, "Size of message batches"),
            ("redis_connection_pool_size", MetricType.GAUGE, "Redis connection pool size"),
            ("redis_connection_pool_hits", MetricType.COUNTER, "Redis connection pool hits"),
            ("redis_connection_pool_misses", MetricType.COUNTER, "Redis connection pool misses"),
            ("system_memory_usage", MetricType.GAUGE, "System memory usage", "bytes"),
            ("system_cpu_usage", MetricType.GAUGE, "System CPU usage", "percent"),
        ]
        
        for name, metric_type, description, *unit in standard_metrics:
            self.metrics[name] = Metric(
                name=name,
                type=metric_type,
                description=description,
                unit=unit[0] if unit else ""
            )
    
    async def _start_background_tasks(self):
        """Start background tasks for metrics collection"""
        # System metrics collection
        system_task = asyncio.create_task(self._collect_system_metrics())
        self._background_tasks.add(system_task)
        
        # Metrics persistence
        persist_task = asyncio.create_task(self._persist_metrics_worker())
        self._background_tasks.add(persist_task)
        
        logger.info("Started metrics collection background tasks")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        while self._running:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_gauge("system_memory_usage", memory.used)
                self.record_gauge("system_memory_percent", memory.percent)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_gauge("system_cpu_usage", cpu_percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.record_gauge("system_disk_usage", disk.used)
                self.record_gauge("system_disk_percent", disk.percent)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(30)
    
    async def _persist_metrics_worker(self):
        """Background worker for metrics persistence"""
        while self._running:
            try:
                await self.persist_metrics()
                await asyncio.sleep(60)  # Persist every minute
                
            except Exception as e:
                logger.error(f"Error in metrics persistence worker: {e}")
                await asyncio.sleep(60)


class GitWorkflowMonitor:
    """
    Git workflow performance monitoring system
    
    Monitors Git operations and workflow performance with
    detailed metrics and performance analysis.
    """
    
    def __init__(self, metrics_collector: MCPMetricsCollector):
        self.metrics_collector = metrics_collector
        
        # Git operation tracking
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.operation_history: deque = deque(maxlen=1000)
        
        # Performance thresholds
        self.performance_thresholds = {
            "commit_duration_warning": 10.0,  # seconds
            "commit_duration_critical": 30.0,
            "branch_creation_warning": 5.0,
            "branch_creation_critical": 15.0,
            "merge_duration_warning": 20.0,
            "merge_duration_critical": 60.0
        }
        
        logger.info("Git Workflow Monitor initialized")
    
    def start_operation(self, operation_id: str, operation_type: str, metadata: Dict[str, Any] = None):
        """Start tracking a Git operation"""
        self.active_operations[operation_id] = {
            "type": operation_type,
            "start_time": time.time(),
            "metadata": metadata or {}
        }
        
        # Record operation start
        self.metrics_collector.record_counter(
            "git_operations_started",
            labels={"operation_type": operation_type}
        )
    
    def complete_operation(self, operation_id: str, success: bool = True, error: str = None):
        """Complete tracking a Git operation"""
        if operation_id not in self.active_operations:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return
        
        operation = self.active_operations.pop(operation_id)
        duration = time.time() - operation["start_time"]
        
        # Record completion metrics
        self.metrics_collector.record_counter(
            "git_operations_completed",
            labels={
                "operation_type": operation["type"],
                "success": str(success)
            }
        )
        
        self.metrics_collector.record_timer(
            f"git_{operation['type']}_duration",
            duration,
            labels={"success": str(success)}
        )
        
        # Store in history
        operation_record = {
            "id": operation_id,
            "type": operation["type"],
            "duration": duration,
            "success": success,
            "error": error,
            "timestamp": datetime.utcnow(),
            "metadata": operation["metadata"]
        }
        self.operation_history.append(operation_record)
        
        # Check performance thresholds
        self._check_performance_thresholds(operation["type"], duration, success)
        
        logger.debug(f"Git operation {operation_id} ({operation['type']}) completed in {duration:.2f}s")
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get Git operation statistics"""
        if not self.operation_history:
            return {}
        
        # Group operations by type
        operations_by_type = defaultdict(list)
        for op in self.operation_history:
            operations_by_type[op["type"]].append(op)
        
        stats = {}
        for op_type, operations in operations_by_type.items():
            durations = [op["duration"] for op in operations]
            success_count = sum(1 for op in operations if op["success"])
            
            stats[op_type] = {
                "total_operations": len(operations),
                "successful_operations": success_count,
                "success_rate": success_count / len(operations),
                "average_duration": statistics.mean(durations),
                "median_duration": statistics.median(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "p95_duration": sorted(durations)[int(len(durations) * 0.95)] if durations else 0
            }
        
        return stats
    
    def get_active_operations(self) -> Dict[str, Any]:
        """Get currently active operations"""
        current_time = time.time()
        active_ops = {}
        
        for op_id, op_data in self.active_operations.items():
            active_ops[op_id] = {
                "type": op_data["type"],
                "duration": current_time - op_data["start_time"],
                "metadata": op_data["metadata"]
            }
        
        return active_ops
    
    def _check_performance_thresholds(self, operation_type: str, duration: float, success: bool):
        """Check if operation duration exceeds performance thresholds"""
        warning_key = f"{operation_type}_duration_warning"
        critical_key = f"{operation_type}_duration_critical"
        
        if warning_key in self.performance_thresholds:
            if duration > self.performance_thresholds[critical_key]:
                self.metrics_collector.record_counter(
                    "git_performance_critical_threshold_exceeded",
                    labels={"operation_type": operation_type}
                )
            elif duration > self.performance_thresholds[warning_key]:
                self.metrics_collector.record_counter(
                    "git_performance_warning_threshold_exceeded",
                    labels={"operation_type": operation_type}
                )


class AlertManager:
    """
    Intelligent alerting system for MCP and Git operations
    
    Provides configurable alert rules, notification management,
    and alert lifecycle tracking.
    """
    
    def __init__(self, metrics_collector: MCPMetricsCollector, redis_url: str = None):
        self.metrics_collector = metrics_collector
        self.redis_url = redis_url or settings.REDIS_URL
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Notification handlers
        self.notification_handlers: List[Callable] = []
        
        # Redis connection
        self._redis: Optional[redis.Redis] = None
        
        # Background tasks
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        logger.info("Alert Manager initialized")
    
    async def start(self) -> bool:
        """Start the alert manager"""
        try:
            # Connect to Redis
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            
            self._running = True
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Alert Manager started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start alert manager: {e}")
            return False
    
    async def stop(self):
        """Stop the alert manager"""
        try:
            self._running = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close Redis connection
            if self._redis:
                await self._redis.close()
            
            logger.info("Alert Manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping alert manager: {e}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            
            # Persist to Redis
            await self._persist_alert(alert)
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            # Persist to Redis
            await self._persist_alert(alert)
            
            logger.info(f"Alert {alert_id} resolved")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
        
        recent_history = [
            alert for alert in self.alert_history
            if alert.triggered_at >= datetime.utcnow() - timedelta(hours=24)
        ]
        
        return {
            "active_alerts": len(self.active_alerts),
            "active_by_severity": dict(active_by_severity),
            "alerts_last_24h": len(recent_history),
            "alert_rules_count": len(self.alert_rules),
            "enabled_rules": sum(1 for rule in self.alert_rules.values() if rule.enabled)
        }
    
    async def _start_background_tasks(self):
        """Start background tasks for alert evaluation"""
        # Alert evaluation task
        eval_task = asyncio.create_task(self._evaluate_alerts_worker())
        self._background_tasks.add(eval_task)
        
        # Alert persistence task
        persist_task = asyncio.create_task(self._persist_alerts_worker())
        self._background_tasks.add(persist_task)
        
        logger.info("Started alert manager background tasks")
    
    async def _evaluate_alerts_worker(self):
        """Background worker for evaluating alert rules"""
        while self._running:
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alert evaluation worker: {e}")
                await asyncio.sleep(30)
    
    async def _evaluate_all_rules(self):
        """Evaluate all alert rules"""
        for rule_name, rule in self.alert_rules.items():
            try:
                metric = self.metrics_collector.get_metric(rule.metric_name)
                if not metric:
                    continue
                
                # Check if rule condition is met
                if rule.evaluate(metric):
                    # Check if alert already exists
                    existing_alert = None
                    for alert in self.active_alerts.values():
                        if alert.rule_name == rule_name and alert.status == AlertStatus.ACTIVE:
                            existing_alert = alert
                            break
                    
                    if not existing_alert:
                        # Create new alert
                        alert = Alert(
                            id=f"{rule_name}_{int(time.time())}",
                            rule_name=rule_name,
                            metric_name=rule.metric_name,
                            severity=rule.severity,
                            status=AlertStatus.ACTIVE,
                            message=f"{rule.description} - Current value: {metric.get_latest_value()}",
                            triggered_at=datetime.utcnow(),
                            metadata={
                                "threshold": rule.threshold,
                                "condition": rule.condition,
                                "current_value": metric.get_latest_value()
                            }
                        )
                        
                        self.active_alerts[alert.id] = alert
                        
                        # Send notifications
                        await self._send_notifications(alert)
                        
                        logger.warning(f"Alert triggered: {alert.message}")
                
                else:
                    # Check if we should resolve any active alerts for this rule
                    alerts_to_resolve = [
                        alert_id for alert_id, alert in self.active_alerts.items()
                        if alert.rule_name == rule_name and alert.status == AlertStatus.ACTIVE
                    ]
                    
                    for alert_id in alerts_to_resolve:
                        await self.resolve_alert(alert_id)
                        
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
    
    async def _persist_alert(self, alert: Alert):
        """Persist alert to Redis"""
        try:
            if self._redis:
                await self._redis.setex(
                    f"mcp:alerts:{alert.id}",
                    86400,  # 24 hour TTL
                    json.dumps(alert.to_dict())
                )
        except Exception as e:
            logger.error(f"Error persisting alert: {e}")
    
    async def _persist_alerts_worker(self):
        """Background worker for alert persistence"""
        while self._running:
            try:
                # Persist all active alerts
                for alert in self.active_alerts.values():
                    await self._persist_alert(alert)
                
                await asyncio.sleep(300)  # Persist every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in alert persistence worker: {e}")
                await asyncio.sleep(300)
    
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                name="high_message_processing_time",
                metric_name="mcp_message_processing_duration",
                condition=">",
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                duration_minutes=2,
                description="Message processing time is high"
            ),
            AlertRule(
                name="low_message_throughput",
                metric_name="mcp_messages_per_second",
                condition="<",
                threshold=1.0,
                severity=AlertSeverity.WARNING,
                duration_minutes=5,
                description="Message throughput is low"
            ),
            AlertRule(
                name="high_failed_messages",
                metric_name="mcp_failed_messages",
                condition=">",
                threshold=10,
                severity=AlertSeverity.ERROR,
                duration_minutes=1,
                description="High number of failed messages"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system_memory_percent",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                duration_minutes=2,
                description="System memory usage is critical"
            ),
            AlertRule(
                name="high_cpu_usage",
                metric_name="system_cpu_usage",
                condition=">",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                duration_minutes=3,
                description="System CPU usage is critical"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule


class SystemHealthDashboard:
    """
    System health dashboard for MCP and Git operations
    
    Provides real-time health status and performance overview
    with web-based dashboard interface.
    """
    
    def __init__(self, 
                 metrics_collector: MCPMetricsCollector,
                 git_monitor: GitWorkflowMonitor,
                 alert_manager: AlertManager):
        
        self.metrics_collector = metrics_collector
        self.git_monitor = git_monitor
        self.alert_manager = alert_manager
        
        # Dashboard state
        self.dashboard_data = {}
        self.last_update = datetime.utcnow()
        
        logger.info("System Health Dashboard initialized")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Get metrics summary
            metrics_summary = self.metrics_collector.get_metrics_summary()
            
            # Get Git operation stats
            git_stats = self.git_monitor.get_operation_stats()
            active_git_ops = self.git_monitor.get_active_operations()
            
            # Get alert summary
            alert_summary = self.alert_manager.get_alert_summary()
            active_alerts = self.alert_manager.get_active_alerts()
            
            # System health indicators
            health_indicators = await self._calculate_health_indicators(metrics_summary)
            
            dashboard_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_health": health_indicators,
                "metrics": {
                    "summary": metrics_summary,
                    "key_metrics": self._extract_key_metrics(metrics_summary)
                },
                "git_operations": {
                    "statistics": git_stats,
                    "active_operations": active_git_ops
                },
                "alerts": {
                    "summary": alert_summary,
                    "active_alerts": [alert.to_dict() for alert in active_alerts[:10]]  # Latest 10
                },
                "performance": {
                    "uptime": self._get_uptime(),
                    "throughput": self._calculate_throughput(metrics_summary),
                    "error_rate": self._calculate_error_rate(metrics_summary)
                }
            }
            
            self.dashboard_data = dashboard_data
            self.last_update = datetime.utcnow()
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard data: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            metrics_summary = self.metrics_collector.get_metrics_summary()
            alert_summary = self.alert_manager.get_alert_summary()
            
            # Calculate overall health score (0-100)
            health_score = await self._calculate_health_score(metrics_summary, alert_summary)
            
            # Determine health status
            if health_score >= 90:
                status = "healthy"
                color = "green"
            elif health_score >= 70:
                status = "warning"
                color = "yellow"
            elif health_score >= 50:
                status = "degraded"
                color = "orange"
            else:
                status = "critical"
                color = "red"
            
            return {
                "status": status,
                "score": health_score,
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
                "details": {
                    "active_alerts": alert_summary.get("active_alerts", 0),
                    "critical_alerts": alert_summary.get("active_by_severity", {}).get("critical", 0),
                    "system_load": metrics_summary.get("system_cpu_usage", {}).get("latest_value", 0),
                    "memory_usage": metrics_summary.get("system_memory_percent", {}).get("latest_value", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating health status: {e}")
            return {
                "status": "unknown",
                "score": 0,
                "color": "gray",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _calculate_health_indicators(self, metrics_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system health indicators"""
        indicators = {}
        
        # MCP Health
        mcp_messages_total = metrics_summary.get("mcp_messages_total", {}).get("latest_value", 0)
        mcp_failed_messages = metrics_summary.get("mcp_failed_messages", {}).get("latest_value", 0)
        
        if mcp_messages_total > 0:
            mcp_success_rate = ((mcp_messages_total - mcp_failed_messages) / mcp_messages_total) * 100
        else:
            mcp_success_rate = 100
        
        indicators["mcp_health"] = {
            "success_rate": mcp_success_rate,
            "status": "healthy" if mcp_success_rate >= 95 else "warning" if mcp_success_rate >= 90 else "critical"
        }
        
        # System Resource Health
        cpu_usage = metrics_summary.get("system_cpu_usage", {}).get("latest_value", 0)
        memory_usage = metrics_summary.get("system_memory_percent", {}).get("latest_value", 0)
        
        indicators["resource_health"] = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "status": "healthy" if cpu_usage < 80 and memory_usage < 80 else "warning" if cpu_usage < 95 and memory_usage < 95 else "critical"
        }
        
        # Connection Health
        active_connections = metrics_summary.get("mcp_active_connections", {}).get("latest_value", 0)
        pool_hits = metrics_summary.get("redis_connection_pool_hits", {}).get("latest_value", 0)
        pool_misses = metrics_summary.get("redis_connection_pool_misses", {}).get("latest_value", 0)
        
        if pool_hits + pool_misses > 0:
            pool_hit_rate = (pool_hits / (pool_hits + pool_misses)) * 100
        else:
            pool_hit_rate = 100
        
        indicators["connection_health"] = {
            "active_connections": active_connections,
            "pool_hit_rate": pool_hit_rate,
            "status": "healthy" if pool_hit_rate >= 80 else "warning" if pool_hit_rate >= 60 else "critical"
        }
        
        return indicators
    
    async def _calculate_health_score(self, metrics_summary: Dict[str, Any], alert_summary: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0
        
        # Deduct points for active alerts
        critical_alerts = alert_summary.get("active_by_severity", {}).get("critical", 0)
        error_alerts = alert_summary.get("active_by_severity", {}).get("error", 0)
        warning_alerts = alert_summary.get("active_by_severity", {}).get("warning", 0)
        
        score -= critical_alerts * 20  # -20 points per critical alert
        score -= error_alerts * 10     # -10 points per error alert
        score -= warning_alerts * 5    # -5 points per warning alert
        
        # Deduct points for high resource usage
        cpu_usage = metrics_summary.get("system_cpu_usage", {}).get("latest_value", 0)
        memory_usage = metrics_summary.get("system_memory_percent", {}).get("latest_value", 0)
        
        if cpu_usage > 90:
            score -= 15
        elif cpu_usage > 80:
            score -= 10
        elif cpu_usage > 70:
            score -= 5
        
        if memory_usage > 90:
            score -= 15
        elif memory_usage > 80:
            score -= 10
        elif memory_usage > 70:
            score -= 5
        
        # Deduct points for failed messages
        failed_messages = metrics_summary.get("mcp_failed_messages", {}).get("latest_value", 0)
        total_messages = metrics_summary.get("mcp_messages_total", {}).get("latest_value", 1)
        
        if total_messages > 0:
            failure_rate = (failed_messages / total_messages) * 100
            if failure_rate > 10:
                score -= 20
            elif failure_rate > 5:
                score -= 10
            elif failure_rate > 1:
                score -= 5
        
        return max(0.0, min(100.0, score))
    
    def _extract_key_metrics(self, metrics_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for dashboard display"""
        key_metrics = {}
        
        important_metrics = [
            "mcp_messages_per_second",
            "mcp_message_processing_duration",
            "mcp_active_connections",
            "mcp_queue_size",
            "system_cpu_usage",
            "system_memory_percent",
            "redis_connection_pool_size"
        ]
        
        for metric_name in important_metrics:
            if metric_name in metrics_summary:
                key_metrics[metric_name] = metrics_summary[metric_name]
        
        return key_metrics
    
    def _get_uptime(self) -> Dict[str, Any]:
        """Get system uptime information"""
        # This is a simplified implementation
        # In a real system, you'd track actual start time
        return {
            "seconds": 0,
            "formatted": "0d 0h 0m 0s"
        }
    
    def _calculate_throughput(self, metrics_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system throughput metrics"""
        messages_per_second = metrics_summary.get("mcp_messages_per_second", {}).get("latest_value", 0)
        
        return {
            "messages_per_second": messages_per_second,
            "messages_per_minute": messages_per_second * 60,
            "messages_per_hour": messages_per_second * 3600
        }
    
    def _calculate_error_rate(self, metrics_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system error rate"""
        total_messages = metrics_summary.get("mcp_messages_total", {}).get("latest_value", 1)
        failed_messages = metrics_summary.get("mcp_failed_messages", {}).get("latest_value", 0)
        
        if total_messages > 0:
            error_rate = (failed_messages / total_messages) * 100
        else:
            error_rate = 0
        
        return {
            "error_rate_percent": error_rate,
            "total_errors": failed_messages,
            "total_messages": total_messages
        }


# Global monitoring system instance
monitoring_system = None

def get_monitoring_system() -> Tuple[MCPMetricsCollector, GitWorkflowMonitor, AlertManager, SystemHealthDashboard]:
    """Get or create the global monitoring system"""
    global monitoring_system
    
    if monitoring_system is None:
        metrics_collector = MCPMetricsCollector()
        git_monitor = GitWorkflowMonitor(metrics_collector)
        alert_manager = AlertManager(metrics_collector)
        dashboard = SystemHealthDashboard(metrics_collector, git_monitor, alert_manager)
        
        monitoring_system = (metrics_collector, git_monitor, alert_manager, dashboard)
    
    return monitoring_system