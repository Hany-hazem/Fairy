# tests/test_performance_analyzer.py
"""
Unit tests for performance analyzer
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from app.performance_analyzer import (
    PerformanceAnalyzer, TrendDirection, AlertSeverity,
    TrendAnalysis, PerformanceAlert, PerformanceInsight, AnalysisReport
)
from app.performance_monitor import PerformanceMonitor, PerformanceMetric

class TestPerformanceAnalyzer:
    """Test performance analyzer functionality"""
    
    @pytest.fixture
    def mock_monitor(self):
        """Create mock performance monitor"""
        monitor = Mock(spec=PerformanceMonitor)
        monitor.collector = Mock()
        return monitor
    
    @pytest.fixture
    def analyzer(self, mock_monitor):
        """Create performance analyzer with mock monitor"""
        return PerformanceAnalyzer(mock_monitor)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing"""
        base_time = datetime.utcnow()
        metrics = []
        
        # Create metrics with increasing values (degrading trend)
        for i in range(20):
            metrics.append(PerformanceMetric(
                name="response_time",
                value=1.0 + (i * 0.1),  # Increasing from 1.0 to 3.0
                unit="seconds",
                timestamp=base_time + timedelta(minutes=i * 5),
                threshold=2.0
            ))
        
        return metrics
    
    def test_analyzer_initialization(self, mock_monitor):
        """Test analyzer initialization"""
        analyzer = PerformanceAnalyzer(mock_monitor)
        
        assert analyzer.monitor == mock_monitor
        assert analyzer.trend_window_hours == 24
        assert analyzer.min_data_points == 10
        assert "response_time" in analyzer.alert_thresholds
    
    def test_trend_analysis_degrading(self, analyzer, sample_metrics):
        """Test trend analysis for degrading performance"""
        # Create historical metrics with lower values
        base_time = datetime.utcnow()
        historical_metrics = []
        for i in range(15):
            historical_metrics.append(PerformanceMetric(
                name="response_time",
                value=0.8 + (i * 0.05),  # Lower baseline values
                unit="seconds",
                timestamp=base_time - timedelta(hours=48) + timedelta(minutes=i * 10),
                threshold=2.0
            ))
        
        # Mock the collector to return different data for different time ranges
        def mock_get_metrics(name=None, start_time=None, end_time=None, limit=1000):
            if start_time and start_time < base_time - timedelta(hours=24):
                return historical_metrics
            else:
                return sample_metrics
        
        analyzer.monitor.collector.get_metrics.side_effect = mock_get_metrics
        
        trend = analyzer.analyze_trends("response_time")
        
        assert trend is not None
        assert trend.metric_name == "response_time"
        assert trend.direction == TrendDirection.DEGRADING
        assert trend.slope > 0  # Positive slope indicates degradation
        assert trend.data_points == 20
        assert trend.recent_avg > trend.historical_avg
    
    def test_trend_analysis_insufficient_data(self, analyzer):
        """Test trend analysis with insufficient data"""
        # Mock insufficient data
        analyzer.monitor.collector.get_metrics.return_value = [
            PerformanceMetric("test_metric", 1.0, "unit", datetime.utcnow())
        ]
        
        trend = analyzer.analyze_trends("test_metric")
        
        assert trend is None
    
    def test_trend_analysis_stable(self, analyzer):
        """Test trend analysis for stable performance"""
        base_time = datetime.utcnow()
        stable_metrics = []
        
        # Create metrics with stable values
        for i in range(15):
            stable_metrics.append(PerformanceMetric(
                name="stable_metric",
                value=2.0 + (0.01 * (i % 2)),  # Very small variation
                unit="seconds",
                timestamp=base_time + timedelta(minutes=i * 5)
            ))
        
        analyzer.monitor.collector.get_metrics.return_value = stable_metrics
        
        trend = analyzer.analyze_trends("stable_metric")
        
        assert trend is not None
        assert trend.direction == TrendDirection.STABLE
        assert abs(trend.slope) < 0.01
    
    def test_trend_analysis_volatile(self, analyzer):
        """Test trend analysis for volatile performance"""
        base_time = datetime.utcnow()
        volatile_metrics = []
        
        # Create metrics with high volatility
        values = [1.0, 5.0, 2.0, 8.0, 1.5, 6.0, 3.0, 7.0, 2.5, 9.0, 1.0, 4.0, 6.0, 2.0, 8.0]
        for i, value in enumerate(values):
            volatile_metrics.append(PerformanceMetric(
                name="volatile_metric",
                value=value,
                unit="seconds",
                timestamp=base_time + timedelta(minutes=i * 5)
            ))
        
        analyzer.monitor.collector.get_metrics.return_value = volatile_metrics
        
        trend = analyzer.analyze_trends("volatile_metric")
        
        assert trend is not None
        assert trend.direction == TrendDirection.VOLATILE
    
    def test_calculate_trend_slope(self, analyzer):
        """Test trend slope calculation"""
        x_values = [0, 1, 2, 3, 4]
        y_values = [1, 2, 3, 4, 5]  # Perfect positive correlation
        
        slope, confidence = analyzer._calculate_trend_slope(x_values, y_values)
        
        assert slope == 1.0  # Perfect slope
        assert confidence > 0.9  # High confidence
    
    def test_calculate_trend_slope_edge_cases(self, analyzer):
        """Test trend slope calculation edge cases"""
        # Empty data
        slope, confidence = analyzer._calculate_trend_slope([], [])
        assert slope == 0.0
        assert confidence == 0.0
        
        # Single point
        slope, confidence = analyzer._calculate_trend_slope([1], [1])
        assert slope == 0.0
        assert confidence == 0.0
    
    def test_generate_alerts_critical(self, analyzer):
        """Test alert generation for critical thresholds"""
        base_time = datetime.utcnow()
        critical_metrics = [
            PerformanceMetric("response_time", 15.0, "seconds", base_time, threshold=5.0),
            PerformanceMetric("memory_usage", 98.0, "percent", base_time, threshold=80.0),
            PerformanceMetric("cpu_usage", 99.0, "percent", base_time, threshold=80.0)
        ]
        
        analyzer.monitor.collector.get_metrics.return_value = critical_metrics
        
        alerts = analyzer.generate_alerts()
        
        assert len(alerts) == 3
        assert all(alert.severity == AlertSeverity.CRITICAL for alert in alerts)
        assert all(len(alert.suggested_actions) > 0 for alert in alerts)
    
    def test_generate_alerts_no_issues(self, analyzer):
        """Test alert generation when no issues exist"""
        base_time = datetime.utcnow()
        normal_metrics = [
            PerformanceMetric("response_time", 0.5, "seconds", base_time, threshold=5.0),
            PerformanceMetric("memory_usage", 45.0, "percent", base_time, threshold=80.0)
        ]
        
        analyzer.monitor.collector.get_metrics.return_value = normal_metrics
        
        alerts = analyzer.generate_alerts()
        
        assert len(alerts) == 0
    
    def test_check_metric_thresholds(self, analyzer):
        """Test metric threshold checking"""
        # Test response time alert
        alert = analyzer._check_metric_thresholds("response_time", 8.0, 5.0)
        
        assert alert is not None
        assert alert.severity == AlertSeverity.HIGH
        assert alert.current_value == 8.0
        assert "response time" in alert.message.lower()
        assert len(alert.suggested_actions) > 0
    
    def test_generate_alert_message_response_time(self, analyzer):
        """Test alert message generation for response time"""
        message, suggestions = analyzer._generate_alert_message(
            "response_time", 5.5, AlertSeverity.HIGH
        )
        
        assert "response time" in message.lower()
        assert "5.5" in message
        assert len(suggestions) > 0
        assert any("database" in s.lower() for s in suggestions)
    
    def test_generate_alert_message_memory(self, analyzer):
        """Test alert message generation for memory usage"""
        message, suggestions = analyzer._generate_alert_message(
            "memory_usage", 85.5, AlertSeverity.HIGH
        )
        
        assert "memory" in message.lower()
        assert "85.5" in message
        assert len(suggestions) > 0
        assert any("memory leak" in s.lower() for s in suggestions)
    
    def test_generate_insights_from_degrading_trend(self, analyzer):
        """Test insight generation from degrading trend"""
        trend = TrendAnalysis(
            metric_name="response_time",
            direction=TrendDirection.DEGRADING,
            slope=0.1,
            confidence=0.8,
            recent_avg=3.0,
            historical_avg=2.0,
            change_percentage=50.0,
            data_points=20
        )
        
        insight = analyzer._generate_insight_from_trend(trend)
        
        assert insight is not None
        assert "Response Time Degradation" in insight.title
        assert "50.0%" in insight.description
        assert insight.priority == AlertSeverity.HIGH
        assert len(insight.recommended_actions) > 0
    
    def test_generate_insights_from_volatile_trend(self, analyzer):
        """Test insight generation from volatile trend"""
        trend = TrendAnalysis(
            metric_name="cpu_usage",
            direction=TrendDirection.VOLATILE,
            slope=0.05,
            confidence=0.2,
            recent_avg=60.0,
            historical_avg=55.0,
            change_percentage=9.1,
            data_points=15
        )
        
        insight = analyzer._generate_insight_from_trend(trend)
        
        assert insight is not None
        assert "Volatility" in insight.title
        assert insight.priority == AlertSeverity.MEDIUM
        assert len(insight.recommended_actions) > 0
    
    def test_generate_system_insights_high_resources(self, analyzer):
        """Test system insight generation for high resource usage"""
        # Mock performance report with high resource usage
        mock_report = Mock()
        mock_report.summary = {
            "cpu_usage": {"avg": 75.0},
            "memory_usage": {"avg": 80.0},
            "response_time": {"avg": 1.5}
        }
        
        analyzer.monitor.get_performance_report.return_value = mock_report
        
        insights = analyzer._generate_system_insights(24)
        
        assert len(insights) > 0
        resource_insight = next((i for i in insights if "Resource Utilization" in i.title), None)
        assert resource_insight is not None
        assert resource_insight.priority == AlertSeverity.HIGH
    
    def test_generate_system_insights_performance_bottleneck(self, analyzer):
        """Test system insight generation for performance bottleneck"""
        # Mock performance report with slow response and high resource usage
        mock_report = Mock()
        mock_report.summary = {
            "cpu_usage": {"avg": 70.0},
            "memory_usage": {"avg": 50.0},
            "response_time": {"avg": 3.0}
        }
        
        analyzer.monitor.get_performance_report.return_value = mock_report
        
        insights = analyzer._generate_system_insights(24)
        
        bottleneck_insight = next((i for i in insights if "Bottleneck" in i.title), None)
        assert bottleneck_insight is not None
        assert bottleneck_insight.priority == AlertSeverity.HIGH
        assert "response_time" in bottleneck_insight.affected_metrics
    
    def test_generate_comprehensive_report(self, analyzer, sample_metrics):
        """Test comprehensive report generation"""
        # Mock all the required methods
        analyzer.monitor.get_performance_report.return_value = Mock(
            metrics=sample_metrics,
            summary={"response_time": {"avg": 2.0}}
        )
        analyzer.monitor.collector.get_metrics.return_value = sample_metrics
        
        # Mock trend analysis
        with patch.object(analyzer, 'analyze_trends') as mock_trends:
            mock_trends.return_value = TrendAnalysis(
                metric_name="response_time",
                direction=TrendDirection.DEGRADING,
                slope=0.1,
                confidence=0.8,
                recent_avg=2.5,
                historical_avg=2.0,
                change_percentage=25.0,
                data_points=20
            )
            
            # Mock alerts and insights
            with patch.object(analyzer, 'generate_alerts') as mock_alerts:
                mock_alerts.return_value = [
                    PerformanceAlert(
                        metric_name="response_time",
                        severity=AlertSeverity.HIGH,
                        message="High response time",
                        current_value=3.0,
                        threshold=2.0
                    )
                ]
                
                with patch.object(analyzer, 'generate_insights') as mock_insights:
                    mock_insights.return_value = [
                        PerformanceInsight(
                            title="Test Insight",
                            description="Test description",
                            impact="Test impact",
                            recommended_actions=["Test action"],
                            priority=AlertSeverity.MEDIUM
                        )
                    ]
                    
                    report = analyzer.generate_comprehensive_report(24)
        
        assert isinstance(report, AnalysisReport)
        assert len(report.trends) > 0
        assert len(report.alerts) > 0
        assert len(report.insights) > 0
        assert len(report.recommendations) > 0
        assert report.summary["analysis_period_hours"] == 24
        assert report.summary["degrading_trends"] > 0
    
    def test_generate_recommendations_critical_alerts(self, analyzer):
        """Test recommendation generation with critical alerts"""
        trends = []
        alerts = [
            PerformanceAlert("test", AlertSeverity.CRITICAL, "Critical issue", 10.0, 5.0)
        ]
        insights = []
        
        recommendations = analyzer._generate_recommendations(trends, alerts, insights)
        
        assert len(recommendations) > 0
        assert any("URGENT" in rec for rec in recommendations)
    
    def test_generate_recommendations_multiple_degrading_trends(self, analyzer):
        """Test recommendation generation with multiple degrading trends"""
        trends = [
            TrendAnalysis("metric1", TrendDirection.DEGRADING, 0.1, 0.8, 2.0, 1.5, 33.0, 10),
            TrendAnalysis("metric2", TrendDirection.DEGRADING, 0.2, 0.7, 3.0, 2.0, 50.0, 10),
            TrendAnalysis("metric3", TrendDirection.DEGRADING, 0.15, 0.9, 4.0, 3.0, 33.0, 10),
            TrendAnalysis("metric4", TrendDirection.DEGRADING, 0.05, 0.6, 1.5, 1.2, 25.0, 10)
        ]
        alerts = []
        insights = []
        
        recommendations = analyzer._generate_recommendations(trends, alerts, insights)
        
        assert any("comprehensive performance review" in rec for rec in recommendations)
    
    def test_generate_recommendations_resource_specific(self, analyzer):
        """Test resource-specific recommendations"""
        trends = []
        alerts = [
            PerformanceAlert("memory_usage", AlertSeverity.HIGH, "High memory", 85.0, 80.0),
            PerformanceAlert("cpu_usage", AlertSeverity.HIGH, "High CPU", 90.0, 80.0)
        ]
        insights = []
        
        recommendations = analyzer._generate_recommendations(trends, alerts, insights)
        
        assert any("scaling" in rec.lower() or "resource optimization" in rec.lower() 
                  for rec in recommendations)
    
    def test_generate_recommendations_stable_system(self, analyzer):
        """Test recommendations for stable system"""
        trends = [
            TrendAnalysis("metric1", TrendDirection.STABLE, 0.001, 0.9, 2.0, 2.0, 0.0, 10)
        ]
        alerts = []
        insights = []
        
        recommendations = analyzer._generate_recommendations(trends, alerts, insights)
        
        assert any("stable" in rec.lower() for rec in recommendations)

class TestPerformanceAnalyzerIntegration:
    """Integration tests for performance analyzer"""
    
    def test_real_analyzer_creation(self):
        """Test creating analyzer with real monitor"""
        from app.performance_analyzer import get_performance_analyzer
        
        analyzer = get_performance_analyzer()
        
        assert isinstance(analyzer, PerformanceAnalyzer)
        assert analyzer.monitor is not None
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis_flow(self):
        """Test complete analysis flow with real components"""
        from app.performance_monitor import PerformanceMonitor
        
        # Create real monitor and analyzer
        monitor = PerformanceMonitor()
        analyzer = PerformanceAnalyzer(monitor)
        
        # Record some test metrics
        for i in range(15):
            monitor.collector.record_metric("test_response_time", 1.0 + (i * 0.1), "seconds")
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Generate analysis
        report = analyzer.generate_comprehensive_report(hours=1)
        
        assert isinstance(report, AnalysisReport)
        assert len(report.summary) > 0
        
        # Test trend analysis
        trend = analyzer.analyze_trends("test_response_time")
        if trend:  # May be None if insufficient data
            assert isinstance(trend, TrendAnalysis)
            assert trend.metric_name == "test_response_time"

if __name__ == "__main__":
    pytest.main([__file__])