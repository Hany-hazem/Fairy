# app/performance_monitor.py
"""
Performance metrics collection and monitoring system
"""

import time
import psutil
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import redis
from contextlib import contextmanager

from .config import settings

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "threshold": self.threshold,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetric':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            value=data["value"],
            unit=data["unit"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            threshold=data.get("threshold"),
            metadata=data.get("metadata", {})
        )

@dataclass
class PerformanceReport:
    """Performance analysis report"""
    start_time: datetime
    end_time: datetime
    metrics: List[PerformanceMetric]
    summary: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": self.summary,
            "alerts": self.alerts
        }

class PerformanceCollector:
    """Collects and stores performance metrics"""
    
    def __init__(self, redis_url: str = None, max_metrics: int = 10000):
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis = None
        self.max_metrics = max_metrics
        
        # In-memory storage for recent metrics
        self._metrics = deque(maxlen=max_metrics)
        self._metrics_by_name = defaultdict(lambda: deque(maxlen=1000))
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance thresholds
        self.thresholds = {
            "response_time": 5.0,  # seconds
            "memory_usage": 80.0,  # percentage
            "cpu_usage": 80.0,     # percentage
            "error_rate": 5.0,     # percentage
            "queue_size": 100      # number of items
        }
        
        logger.info("Performance collector initialized")
    
    @property
    def redis(self) -> Optional[redis.Redis]:
        """Lazy Redis connection"""
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                self._redis.ping()
                logger.info(f"Connected to Redis for metrics storage")
            except Exception as e:
                logger.warning(f"Could not connect to Redis for metrics: {e}")
                self._redis = False  # Mark as failed
        
        return self._redis if self._redis is not False else None
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                     threshold: Optional[float] = None, **metadata) -> PerformanceMetric:
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            threshold=threshold or self.thresholds.get(name),
            metadata=metadata
        )
        
        with self._lock:
            # Store in memory
            self._metrics.append(metric)
            self._metrics_by_name[name].append(metric)
            
            # Store in Redis if available
            if self.redis:
                try:
                    # Store in time-series format with timestamp-based keys
                    timestamp_key = int(metric.timestamp.timestamp())
                    
                    # Main metrics list for recent data
                    key = f"metrics:{name}"
                    data = metric.to_dict()
                    self.redis.lpush(key, json.dumps(data))
                    self.redis.ltrim(key, 0, 999)  # Keep last 1000 entries
                    self.redis.expire(key, 86400 * 7)  # 7 days TTL
                    
                    # Time-series aggregation for efficient querying
                    # Store hourly aggregates for longer-term analysis
                    hour_key = f"metrics:hourly:{name}:{timestamp_key // 3600}"
                    self.redis.zadd(hour_key, {json.dumps(data): timestamp_key})
                    self.redis.expire(hour_key, 86400 * 30)  # 30 days TTL
                    
                    # Store daily aggregates for even longer-term trends
                    day_key = f"metrics:daily:{name}:{timestamp_key // 86400}"
                    self.redis.zadd(day_key, {json.dumps(data): timestamp_key})
                    self.redis.expire(day_key, 86400 * 365)  # 1 year TTL
                    
                except Exception as e:
                    logger.warning(f"Failed to store metric in Redis: {e}")
        
        # Check for threshold violations
        if metric.threshold and metric.value > metric.threshold:
            logger.warning(f"Performance threshold exceeded: {name}={value}{unit} > {metric.threshold}{unit}")
        
        return metric
    
    def get_metrics(self, name: Optional[str] = None, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 100) -> List[PerformanceMetric]:
        """Retrieve metrics with optional filtering"""
        with self._lock:
            if name:
                metrics = list(self._metrics_by_name[name])
            else:
                metrics = list(self._metrics)
        
        # Apply time filtering
        if start_time or end_time:
            filtered_metrics = []
            for metric in metrics:
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                filtered_metrics.append(metric)
            metrics = filtered_metrics
        
        # Sort by timestamp (newest first) and limit
        metrics.sort(key=lambda m: m.timestamp, reverse=True)
        return metrics[:limit]
    
    def get_metric_summary(self, name: str, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get statistical summary for a metric"""
        metrics = self.get_metrics(name, start_time, end_time)
        
        if not metrics:
            return {"count": 0}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[0] if values else None,
            "unit": metrics[0].unit if metrics else "",
            "threshold": metrics[0].threshold if metrics else None,
            "threshold_violations": sum(1 for m in metrics if m.threshold and m.value > m.threshold)
        }
    
    def get_time_series_data(self, name: str, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           granularity: str = "raw") -> List[PerformanceMetric]:
        """Get time-series data with specified granularity"""
        if not self.redis:
            # Fallback to in-memory data
            return self.get_metrics(name, start_time, end_time)
        
        try:
            if granularity == "hourly":
                return self._get_aggregated_data(name, start_time, end_time, "hourly")
            elif granularity == "daily":
                return self._get_aggregated_data(name, start_time, end_time, "daily")
            else:
                # Raw data - use existing method
                return self.get_metrics(name, start_time, end_time)
        except Exception as e:
            logger.warning(f"Failed to retrieve time-series data from Redis: {e}")
            return self.get_metrics(name, start_time, end_time)
    
    def _get_aggregated_data(self, name: str, 
                           start_time: Optional[datetime],
                           end_time: Optional[datetime],
                           granularity: str) -> List[PerformanceMetric]:
        """Get aggregated time-series data from Redis"""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=1)
        if not end_time:
            end_time = datetime.utcnow()
        
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        
        metrics = []
        
        if granularity == "hourly":
            # Get data by hour
            for hour_ts in range(start_ts // 3600, end_ts // 3600 + 1):
                key = f"metrics:hourly:{name}:{hour_ts}"
                data = self.redis.zrangebyscore(key, start_ts, end_ts)
                for item in data:
                    try:
                        metric_data = json.loads(item)
                        metrics.append(PerformanceMetric.from_dict(metric_data))
                    except Exception as e:
                        logger.warning(f"Failed to parse metric data: {e}")
        
        elif granularity == "daily":
            # Get data by day
            for day_ts in range(start_ts // 86400, end_ts // 86400 + 1):
                key = f"metrics:daily:{name}:{day_ts}"
                data = self.redis.zrangebyscore(key, start_ts, end_ts)
                for item in data:
                    try:
                        metric_data = json.loads(item)
                        metrics.append(PerformanceMetric.from_dict(metric_data))
                    except Exception as e:
                        logger.warning(f"Failed to parse metric data: {e}")
        
        # Sort by timestamp
        metrics.sort(key=lambda m: m.timestamp, reverse=True)
        return metrics
    
    def clear_metrics(self, name: Optional[str] = None, older_than: Optional[datetime] = None):
        """Clear metrics with optional filtering"""
        with self._lock:
            if name and older_than:
                # Remove specific metrics older than timestamp
                self._metrics_by_name[name] = deque(
                    [m for m in self._metrics_by_name[name] if m.timestamp >= older_than],
                    maxlen=1000
                )
                # Also remove from main metrics collection
                self._metrics = deque(
                    [m for m in self._metrics if not (m.name == name and m.timestamp < older_than)],
                    maxlen=self.max_metrics
                )
            elif name:
                # Clear all metrics for specific name
                self._metrics_by_name[name].clear()
                # Also remove from main metrics collection
                self._metrics = deque(
                    [m for m in self._metrics if m.name != name],
                    maxlen=self.max_metrics
                )
            elif older_than:
                # Remove all metrics older than timestamp
                self._metrics = deque(
                    [m for m in self._metrics if m.timestamp >= older_than],
                    maxlen=self.max_metrics
                )
                # Also clean named metrics
                for metric_name in self._metrics_by_name:
                    self._metrics_by_name[metric_name] = deque(
                        [m for m in self._metrics_by_name[metric_name] if m.timestamp >= older_than],
                        maxlen=1000
                    )
            else:
                # Clear all metrics
                self._metrics.clear()
                self._metrics_by_name.clear()

class SystemMetricsCollector:
    """Collects system-level performance metrics"""
    
    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
        self.process = psutil.Process()
        
    def collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.collector.record_metric("cpu_usage", cpu_percent, "percent")
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.collector.record_metric("memory_usage", memory.percent, "percent")
            self.collector.record_metric("memory_available", memory.available / (1024**3), "GB")
            
            # Process-specific metrics
            process_memory = self.process.memory_info()
            self.collector.record_metric("process_memory", process_memory.rss / (1024**2), "MB")
            self.collector.record_metric("process_cpu", self.process.cpu_percent(), "percent")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.collector.record_metric("disk_usage", (disk.used / disk.total) * 100, "percent")
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, redis_url: str = None):
        self.collector = PerformanceCollector(redis_url)
        self.system_collector = SystemMetricsCollector(self.collector)
        
        # Request tracking
        self._active_requests = {}
        self._request_lock = threading.RLock()
        
        logger.info("Performance monitor initialized")
    
    @contextmanager
    def track_request(self, request_id: str, operation: str = "request", **metadata):
        """Context manager to track request performance"""
        start_time = time.time()
        
        with self._request_lock:
            self._active_requests[request_id] = {
                "operation": operation,
                "start_time": start_time,
                "metadata": metadata
            }
        
        try:
            yield
            # Success
            duration = time.time() - start_time
            self.collector.record_metric(
                f"{operation}_time", 
                duration, 
                "seconds",
                metadata={**metadata, "status": "success"}
            )
            
        except Exception as e:
            # Error
            duration = time.time() - start_time
            self.collector.record_metric(
                f"{operation}_time", 
                duration, 
                "seconds",
                metadata={**metadata, "status": "error", "error": str(e)}
            )
            self.collector.record_metric(
                f"{operation}_error", 
                1, 
                "count",
                error=str(e),
                **metadata
            )
            raise
            
        finally:
            with self._request_lock:
                self._active_requests.pop(request_id, None)
    
    def record_response_time(self, operation: str, duration: float, **metadata):
        """Record response time for an operation"""
        self.collector.record_metric(f"{operation}_response_time", duration, "seconds", **metadata)
    
    def record_error(self, operation: str, error: str, **metadata):
        """Record an error occurrence"""
        self.collector.record_metric(f"{operation}_error", 1, "count", error=error, **metadata)
    
    def record_throughput(self, operation: str, count: int, **metadata):
        """Record throughput metrics"""
        self.collector.record_metric(f"{operation}_throughput", count, "requests", **metadata)
    
    def collect_system_metrics(self):
        """Collect current system metrics"""
        self.system_collector.collect_system_metrics()
    
    def get_performance_report(self, start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> PerformanceReport:
        """Generate comprehensive performance report"""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(hours=1)
        if not end_time:
            end_time = datetime.utcnow()
        
        # Get all metrics in time range
        metrics = self.collector.get_metrics(start_time=start_time, end_time=end_time, limit=10000)
        
        # Generate summary statistics
        summary = {}
        alerts = []
        
        # Group metrics by name for analysis
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric)
        
        for name, metric_list in metrics_by_name.items():
            values = [m.value for m in metric_list]
            if values:
                metric_summary = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[0]
                }
                summary[name] = metric_summary
                
                # Check for alerts
                threshold = metric_list[0].threshold
                if threshold:
                    violations = sum(1 for v in values if v > threshold)
                    if violations > 0:
                        alerts.append(f"{name}: {violations} threshold violations (>{threshold})")
        
        return PerformanceReport(
            start_time=start_time,
            end_time=end_time,
            metrics=metrics,
            summary=summary,
            alerts=alerts
        )
    
    def get_active_requests(self) -> Dict[str, Any]:
        """Get information about currently active requests"""
        with self._request_lock:
            current_time = time.time()
            active = {}
            
            for request_id, info in self._active_requests.items():
                duration = current_time - info["start_time"]
                active[request_id] = {
                    "operation": info["operation"],
                    "duration": duration,
                    "metadata": info["metadata"]
                }
            
            return active
    
    def cleanup_old_metrics(self, days: int = 7):
        """Clean up metrics older than specified days"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        self.collector.clear_metrics(older_than=cutoff_time)
        logger.info(f"Cleaned up metrics older than {days} days")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()