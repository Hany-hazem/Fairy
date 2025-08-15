# tests/test_performance_monitor.py
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from app.performance_monitor import (
    PerformanceMetric, 
    PerformanceReport, 
    PerformanceCollector,
    SystemMetricsCollector,
    PerformanceMonitor
)

class TestPerformanceMetric:
    def test_metric_creation(self):
        """Test basic metric creation"""
        metric = PerformanceMetric(
            name="test_metric",
            value=1.5,
            unit="seconds",
            timestamp=datetime.utcnow(),
            threshold=2.0,
            metadata={"test": "data"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 1.5
        assert metric.unit == "seconds"
        assert metric.threshold == 2.0
        assert metric.metadata["test"] == "data"
    
    def test_metric_serialization(self):
        """Test metric to/from dict conversion"""
        original = PerformanceMetric(
            name="test_metric",
            value=2.5,
            unit="ms",
            timestamp=datetime.utcnow()
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = PerformanceMetric.from_dict(data)
        
        assert restored.name == original.name
        assert restored.value == original.value
        assert restored.unit == original.unit
        assert restored.timestamp == original.timestamp

class TestPerformanceCollector:
    @pytest.fixture
    def collector(self):
        """Create performance collector with mocked Redis"""
        with patch('app.performance_monitor.redis.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Redis not available")
            collector = PerformanceCollector()
            return collector
    
    def test_record_metric(self, collector):
        """Test recording a metric"""
        metric = collector.record_metric("test_response_time", 1.5, "seconds")
        
        assert metric.name == "test_response_time"
        assert metric.value == 1.5
        assert metric.unit == "seconds"
        assert isinstance(metric.timestamp, datetime)
    
    def test_record_metric_with_threshold(self, collector):
        """Test recording metric with threshold"""
        metric = collector.record_metric(
            "response_time", 
            6.0, 
            "seconds", 
            threshold=5.0
        )
        
        assert metric.threshold == 5.0
        assert metric.value > metric.threshold
    
    def test_get_metrics_by_name(self, collector):
        """Test retrieving metrics by name"""
        # Record some metrics
        collector.record_metric("cpu_usage", 50.0, "percent")
        collector.record_metric("memory_usage", 60.0, "percent")
        collector.record_metric("cpu_usage", 55.0, "percent")
        
        # Get CPU metrics only
        cpu_metrics = collector.get_metrics("cpu_usage")
        
        assert len(cpu_metrics) == 2
        assert all(m.name == "cpu_usage" for m in cpu_metrics)
        assert cpu_metrics[0].value == 55.0  # Most recent first
        assert cpu_metrics[1].value == 50.0
    
    def test_get_metrics_with_time_filter(self, collector):
        """Test retrieving metrics with time filtering"""
        now = datetime.utcnow()
        old_time = now - timedelta(hours=2)
        
        # Record metrics at different times
        with patch('app.performance_monitor.datetime') as mock_dt:
            mock_dt.utcnow.return_value = old_time
            collector.record_metric("old_metric", 1.0, "count")
            
            mock_dt.utcnow.return_value = now
            collector.record_metric("new_metric", 2.0, "count")
        
        # Get metrics from last hour only
        recent_metrics = collector.get_metrics(
            start_time=now - timedelta(hours=1)
        )
        
        assert len(recent_metrics) == 1
        assert recent_metrics[0].name == "new_metric"
    
    def test_get_metric_summary(self, collector):
        """Test metric summary statistics"""
        # Record multiple values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            collector.record_metric("test_metric", value, "units")
        
        summary = collector.get_metric_summary("test_metric")
        
        assert summary["count"] == 5
        assert summary["min"] == 1.0
        assert summary["max"] == 5.0
        assert summary["avg"] == 3.0
        assert summary["latest"] == 5.0
        assert summary["unit"] == "units"
    
    def test_clear_metrics(self, collector):
        """Test clearing metrics"""
        # Record some metrics
        collector.record_metric("metric1", 1.0, "count")
        collector.record_metric("metric2", 2.0, "count")
        
        assert len(collector.get_metrics()) == 2
        
        # Clear all metrics
        collector.clear_metrics()
        
        assert len(collector.get_metrics()) == 0
    
    def test_clear_metrics_by_name(self, collector):
        """Test clearing specific metrics"""
        collector.record_metric("keep_metric", 1.0, "count")
        collector.record_metric("clear_metric", 2.0, "count")
        
        # Clear only clear_metric
        collector.clear_metrics("clear_metric")
        
        remaining = collector.get_metrics()
        assert len(remaining) == 1
        assert remaining[0].name == "keep_metric"
    
    def test_get_time_series_data_fallback(self, collector):
        """Test time-series data retrieval fallback to in-memory"""
        # Record some metrics
        collector.record_metric("test_metric", 1.0, "count")
        collector.record_metric("test_metric", 2.0, "count")
        
        # Should fallback to regular get_metrics since Redis is mocked
        data = collector.get_time_series_data("test_metric")
        
        assert len(data) == 2
        assert data[0].value == 2.0  # Most recent first
        assert data[1].value == 1.0

class TestSystemMetricsCollector:
    @pytest.fixture
    def system_collector(self):
        """Create system metrics collector with mocked performance collector"""
        mock_collector = Mock()
        return SystemMetricsCollector(mock_collector)
    
    def test_collect_system_metrics(self, system_collector):
        """Test system metrics collection"""
        with patch('app.performance_monitor.psutil') as mock_psutil:
            # Mock system metrics
            mock_psutil.cpu_percent.return_value = 45.0
            mock_psutil.virtual_memory.return_value = Mock(percent=60.0, available=8*1024**3)
            mock_psutil.disk_usage.return_value = Mock(used=100*1024**3, total=500*1024**3)
            
            # Mock process metrics
            mock_process = Mock()
            mock_process.memory_info.return_value = Mock(rss=256*1024**2)
            mock_process.cpu_percent.return_value = 15.0
            system_collector.process = mock_process
            
            system_collector.collect_system_metrics()
            
            # Verify metrics were recorded
            calls = system_collector.collector.record_metric.call_args_list
            
            # Check that various system metrics were recorded
            metric_names = [call[0][0] for call in calls]
            assert "cpu_usage" in metric_names
            assert "memory_usage" in metric_names
            assert "process_memory" in metric_names
            assert "process_cpu" in metric_names
            assert "disk_usage" in metric_names

class TestPerformanceMonitor:
    @pytest.fixture
    def monitor(self):
        """Create performance monitor with mocked dependencies"""
        with patch('app.performance_monitor.PerformanceCollector') as mock_collector_class:
            with patch('app.performance_monitor.SystemMetricsCollector') as mock_system_class:
                monitor = PerformanceMonitor()
                return monitor
    
    def test_track_request_success(self, monitor):
        """Test successful request tracking"""
        with monitor.track_request("test_request", "test_operation", user_id="test_user"):
            time.sleep(0.01)  # Simulate some work
        
        # Verify metrics were recorded
        monitor.collector.record_metric.assert_called()
        
        # Check the call for success metrics
        calls = monitor.collector.record_metric.call_args_list
        success_call = next(call for call in calls if "test_operation_time" in call[0])
        
        assert success_call[0][0] == "test_operation_time"
        assert success_call[0][2] == "seconds"
        assert success_call[1]["metadata"]["status"] == "success"
        assert success_call[1]["metadata"]["user_id"] == "test_user"
    
    def test_track_request_error(self, monitor):
        """Test request tracking with error"""
        with pytest.raises(ValueError):
            with monitor.track_request("test_request", "test_operation"):
                raise ValueError("Test error")
        
        # Verify error metrics were recorded
        calls = monitor.collector.record_metric.call_args_list
        
        # Should have both time and error metrics
        time_call = next(call for call in calls if "test_operation_time" in call[0])
        error_call = next(call for call in calls if "test_operation_error" in call[0])
        
        assert time_call[1]["metadata"]["status"] == "error"
        assert error_call[0][1] == 1  # Error count
        assert "Test error" in error_call[1]["error"]
    
    def test_record_response_time(self, monitor):
        """Test recording response time"""
        monitor.record_response_time("api_call", 1.5, endpoint="/test")
        
        monitor.collector.record_metric.assert_called_with(
            "api_call_response_time", 1.5, "seconds", endpoint="/test"
        )
    
    def test_record_error(self, monitor):
        """Test recording error"""
        monitor.record_error("database", "Connection timeout", table="users")
        
        monitor.collector.record_metric.assert_called_with(
            "database_error", 1, "count", error="Connection timeout", table="users"
        )
    
    def test_record_throughput(self, monitor):
        """Test recording throughput"""
        monitor.record_throughput("api", 150, endpoint="/chat")
        
        monitor.collector.record_metric.assert_called_with(
            "api_throughput", 150, "requests", endpoint="/chat"
        )
    
    def test_get_performance_report(self, monitor):
        """Test generating performance report"""
        # Mock metrics data
        mock_metrics = [
            Mock(name="response_time", value=1.0, threshold=2.0),
            Mock(name="response_time", value=3.0, threshold=2.0),  # Violation
            Mock(name="cpu_usage", value=50.0, threshold=80.0)
        ]
        monitor.collector.get_metrics.return_value = mock_metrics
        
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        report = monitor.get_performance_report(start_time, end_time)
        
        assert isinstance(report, PerformanceReport)
        assert report.start_time == start_time
        assert report.end_time == end_time
        assert len(report.metrics) == 3
        assert len(report.alerts) > 0  # Should have threshold violation alert
    
    def test_get_active_requests(self, monitor):
        """Test getting active requests"""
        # Simulate active request
        monitor._active_requests["req1"] = {
            "operation": "test_op",
            "start_time": time.time() - 1.0,
            "metadata": {"user": "test"}
        }
        
        active = monitor.get_active_requests()
        
        assert "req1" in active
        assert active["req1"]["operation"] == "test_op"
        assert active["req1"]["duration"] >= 1.0
        assert active["req1"]["metadata"]["user"] == "test"
    
    def test_cleanup_old_metrics(self, monitor):
        """Test cleaning up old metrics"""
        monitor.cleanup_old_metrics(7)
        
        # Verify collector cleanup was called
        monitor.collector.clear_metrics.assert_called_once()
        
        # Check that cutoff time was approximately 7 days ago
        call_args = monitor.collector.clear_metrics.call_args
        cutoff_time = call_args[1]["older_than"]
        expected_cutoff = datetime.utcnow() - timedelta(days=7)
        
        # Allow 1 minute tolerance
        assert abs((cutoff_time - expected_cutoff).total_seconds()) < 60

class TestPerformanceReport:
    def test_report_serialization(self):
        """Test performance report serialization"""
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        metrics = [
            PerformanceMetric("test_metric", 1.0, "seconds", datetime.utcnow())
        ]
        
        report = PerformanceReport(
            start_time=start_time,
            end_time=end_time,
            metrics=metrics,
            summary={"test_metric": {"avg": 1.0}},
            alerts=["Test alert"]
        )
        
        data = report.to_dict()
        
        assert "start_time" in data
        assert "end_time" in data
        assert "metrics" in data
        assert "summary" in data
        assert "alerts" in data
        assert len(data["metrics"]) == 1
        assert data["alerts"][0] == "Test alert"