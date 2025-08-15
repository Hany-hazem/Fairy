# app/performance_analyzer.py
"""
Performance analysis and reporting system with trend detection and actionable insights
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

from .performance_monitor import PerformanceMonitor, PerformanceMetric, PerformanceReport

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction indicators"""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TrendAnalysis:
    """Trend analysis result for a metric"""
    metric_name: str
    direction: TrendDirection
    slope: float
    confidence: float
    recent_avg: float
    historical_avg: float
    change_percentage: float
    data_points: int

@dataclass
class PerformanceAlert:
    """Performance alert with actionable information"""
    metric_name: str
    severity: AlertSeverity
    message: str
    current_value: float
    threshold: float
    trend: Optional[TrendDirection] = None
    suggested_actions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PerformanceInsight:
    """Actionable performance insight"""
    title: str
    description: str
    impact: str
    recommended_actions: List[str]
    priority: AlertSeverity
    affected_metrics: List[str] = field(default_factory=list)

@dataclass
class AnalysisReport:
    """Comprehensive performance analysis report"""
    start_time: datetime
    end_time: datetime
    trends: List[TrendAnalysis]
    alerts: List[PerformanceAlert]
    insights: List[PerformanceInsight]
    summary: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)

class PerformanceAnalyzer:
    """Advanced performance analysis with trend detection and insights"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        
        # Analysis configuration
        self.trend_window_hours = 24
        self.comparison_window_hours = 168  # 1 week
        self.min_data_points = 10
        self.volatility_threshold = 0.5  # Increased threshold to be less sensitive
        
        # Alert thresholds (can be customized)
        self.alert_thresholds = {
            "response_time": {
                "medium": 2.0,
                "high": 5.0,
                "critical": 10.0
            },
            "memory_usage": {
                "medium": 70.0,
                "high": 85.0,
                "critical": 95.0
            },
            "cpu_usage": {
                "medium": 70.0,
                "high": 85.0,
                "critical": 95.0
            },
            "error_rate": {
                "medium": 1.0,
                "high": 5.0,
                "critical": 10.0
            }
        }
        
        logger.info("Performance analyzer initialized")
    
    def analyze_trends(self, metric_name: str, hours: int = None) -> Optional[TrendAnalysis]:
        """Analyze trends for a specific metric"""
        hours = hours or self.trend_window_hours
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get recent metrics
        recent_metrics = self.monitor.collector.get_metrics(
            name=metric_name,
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        if len(recent_metrics) < self.min_data_points:
            logger.warning(f"Insufficient data points for trend analysis: {metric_name}")
            return None
        
        # Sort by timestamp
        recent_metrics.sort(key=lambda m: m.timestamp)
        
        # Get historical comparison data
        historical_start = start_time - timedelta(hours=self.comparison_window_hours)
        historical_metrics = self.monitor.collector.get_metrics(
            name=metric_name,
            start_time=historical_start,
            end_time=start_time,
            limit=1000
        )
        
        # Calculate trend
        values = [m.value for m in recent_metrics]
        timestamps = [(m.timestamp - recent_metrics[0].timestamp).total_seconds() for m in recent_metrics]
        
        # Linear regression for trend
        if len(values) >= 2:
            slope, confidence = self._calculate_trend_slope(timestamps, values)
        else:
            slope, confidence = 0.0, 0.0
        
        # Calculate averages
        recent_avg = statistics.mean(values)
        historical_avg = statistics.mean([m.value for m in historical_metrics]) if historical_metrics else recent_avg
        
        # Determine trend direction
        direction = self._determine_trend_direction(slope, values, confidence)
        
        # Calculate change percentage
        change_percentage = ((recent_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0.0
        
        return TrendAnalysis(
            metric_name=metric_name,
            direction=direction,
            slope=slope,
            confidence=confidence,
            recent_avg=recent_avg,
            historical_avg=historical_avg,
            change_percentage=change_percentage,
            data_points=len(values)
        )
    
    def _calculate_trend_slope(self, x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
        """Calculate trend slope using linear regression"""
        try:
            if len(x_values) < 2:
                return 0.0, 0.0
            
            # Convert to numpy arrays for calculation
            x = np.array(x_values)
            y = np.array(y_values)
            
            # Calculate slope using least squares
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
            
            # Calculate correlation coefficient as confidence measure
            correlation = np.corrcoef(x, y)[0, 1] if n > 1 else 0.0
            confidence = abs(correlation)
            
            return float(slope), float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating trend slope: {e}")
            return 0.0, 0.0
    
    def _determine_trend_direction(self, slope: float, values: List[float], confidence: float) -> TrendDirection:
        """Determine trend direction based on slope and volatility"""
        if len(values) < 2:
            return TrendDirection.STABLE
        
        # Calculate coefficient of variation for volatility
        mean_val = statistics.mean(values)
        cv = 0
        if mean_val > 0:
            cv = statistics.stdev(values) / mean_val
        
        # Calculate value range as percentage of mean
        value_range = max(values) - min(values)
        range_percentage = (value_range / mean_val) if mean_val > 0 else 0
        
        # Determine if data is volatile based on coefficient of variation
        is_volatile = cv > self.volatility_threshold
        
        # For very low confidence, only call it volatile if CV is very high
        if confidence < 0.1 and cv > 0.8:
            return TrendDirection.VOLATILE
        
        # Calculate slope significance based on data range
        # A slope is significant if it represents a meaningful change over the time period
        # Use a smaller threshold since slope is calculated per second but data may be sparse
        slope_threshold = (value_range * 0.001) / len(values) if value_range > 0 else 0.0001
        
        # Determine trend direction
        if is_volatile and abs(slope) < slope_threshold:
            return TrendDirection.VOLATILE
        elif abs(slope) >= slope_threshold and confidence > 0.2:
            # Clear trend detected
            if slope > 0:
                return TrendDirection.DEGRADING
            else:
                return TrendDirection.IMPROVING
        else:
            # No significant trend
            if is_volatile:
                return TrendDirection.VOLATILE
            else:
                return TrendDirection.STABLE
    
    def generate_alerts(self, hours: int = 1) -> List[PerformanceAlert]:
        """Generate performance alerts based on current metrics"""
        alerts = []
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get recent metrics for analysis
        recent_metrics = self.monitor.collector.get_metrics(
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_name[metric.name].append(metric)
        
        # Analyze each metric type
        for metric_name, metric_list in metrics_by_name.items():
            if not metric_list:
                continue
            
            # Get latest value
            latest_metric = max(metric_list, key=lambda m: m.timestamp)
            current_value = latest_metric.value
            threshold = latest_metric.threshold
            
            # Check against configured thresholds
            alert = self._check_metric_thresholds(metric_name, current_value, threshold)
            if alert:
                # Add trend information
                trend_analysis = self.analyze_trends(metric_name, hours=24)
                if trend_analysis:
                    alert.trend = trend_analysis.direction
                
                alerts.append(alert)
        
        return alerts
    
    def _check_metric_thresholds(self, metric_name: str, value: float, threshold: Optional[float]) -> Optional[PerformanceAlert]:
        """Check if metric value exceeds thresholds and generate alert"""
        # Check for specific metric patterns first
        severity = None
        used_threshold = threshold
        
        # Check configured thresholds based on metric name patterns
        if "response_time" in metric_name or "time" in metric_name:
            thresholds = self.alert_thresholds.get("response_time", {})
            if value > thresholds.get("critical", float('inf')):
                severity = AlertSeverity.CRITICAL
                used_threshold = thresholds["critical"]
            elif value > thresholds.get("high", float('inf')):
                severity = AlertSeverity.HIGH
                used_threshold = thresholds["high"]
            elif value > thresholds.get("medium", float('inf')):
                severity = AlertSeverity.MEDIUM
                used_threshold = thresholds["medium"]
        elif "memory" in metric_name:
            thresholds = self.alert_thresholds.get("memory_usage", {})
            if value > thresholds.get("critical", float('inf')):
                severity = AlertSeverity.CRITICAL
                used_threshold = thresholds["critical"]
            elif value > thresholds.get("high", float('inf')):
                severity = AlertSeverity.HIGH
                used_threshold = thresholds["high"]
            elif value > thresholds.get("medium", float('inf')):
                severity = AlertSeverity.MEDIUM
                used_threshold = thresholds["medium"]
        elif "cpu" in metric_name:
            thresholds = self.alert_thresholds.get("cpu_usage", {})
            if value > thresholds.get("critical", float('inf')):
                severity = AlertSeverity.CRITICAL
                used_threshold = thresholds["critical"]
            elif value > thresholds.get("high", float('inf')):
                severity = AlertSeverity.HIGH
                used_threshold = thresholds["high"]
            elif value > thresholds.get("medium", float('inf')):
                severity = AlertSeverity.MEDIUM
                used_threshold = thresholds["medium"]
        elif "error" in metric_name:
            thresholds = self.alert_thresholds.get("error_rate", {})
            if value > thresholds.get("critical", float('inf')):
                severity = AlertSeverity.CRITICAL
                used_threshold = thresholds["critical"]
            elif value > thresholds.get("high", float('inf')):
                severity = AlertSeverity.HIGH
                used_threshold = thresholds["high"]
            elif value > thresholds.get("medium", float('inf')):
                severity = AlertSeverity.MEDIUM
                used_threshold = thresholds["medium"]
        
        # Fallback to provided threshold
        if not severity and threshold and value > threshold:
            severity = AlertSeverity.LOW
            used_threshold = threshold
        
        if not severity:
            return None
        
        # Generate alert message and suggestions
        message, suggestions = self._generate_alert_message(metric_name, value, severity)
        
        return PerformanceAlert(
            metric_name=metric_name,
            severity=severity,
            message=message,
            current_value=value,
            threshold=used_threshold,
            suggested_actions=suggestions
        )
    
    def _generate_alert_message(self, metric_name: str, value: float, severity: AlertSeverity) -> Tuple[str, List[str]]:
        """Generate alert message and suggested actions"""
        suggestions = []
        
        if "response_time" in metric_name:
            message = f"High response time detected: {value:.2f}s"
            suggestions = [
                "Check for slow database queries",
                "Review API endpoint performance",
                "Consider caching frequently accessed data",
                "Monitor external service dependencies"
            ]
        elif "memory_usage" in metric_name:
            message = f"High memory usage: {value:.1f}%"
            suggestions = [
                "Check for memory leaks in application code",
                "Review large data structures and caching",
                "Consider increasing available memory",
                "Implement memory cleanup routines"
            ]
        elif "cpu_usage" in metric_name:
            message = f"High CPU usage: {value:.1f}%"
            suggestions = [
                "Identify CPU-intensive operations",
                "Optimize algorithms and data processing",
                "Consider load balancing or scaling",
                "Review background task scheduling"
            ]
        elif "error" in metric_name:
            message = f"Elevated error rate: {value} errors"
            suggestions = [
                "Review application logs for error patterns",
                "Check external service connectivity",
                "Validate input data and error handling",
                "Monitor downstream dependencies"
            ]
        else:
            message = f"Performance threshold exceeded: {metric_name}={value}"
            suggestions = [
                "Review metric-specific documentation",
                "Check system resources and dependencies",
                "Consider scaling or optimization"
            ]
        
        return message, suggestions
    
    def generate_insights(self, hours: int = 24) -> List[PerformanceInsight]:
        """Generate actionable performance insights"""
        insights = []
        
        # Analyze trends for key metrics
        key_metrics = ["response_time", "memory_usage", "cpu_usage", "error_rate"]
        
        for base_metric in key_metrics:
            # Find actual metric names that match the base
            all_metrics = self.monitor.collector.get_metrics(limit=1)
            matching_metrics = [m.name for m in all_metrics if base_metric in m.name]
            
            for metric_name in matching_metrics:
                trend = self.analyze_trends(metric_name, hours)
                if trend:
                    insight = self._generate_insight_from_trend(trend)
                    if insight:
                        insights.append(insight)
        
        # Add system-level insights
        system_insights = self._generate_system_insights(hours)
        insights.extend(system_insights)
        
        return insights
    
    def _generate_insight_from_trend(self, trend: TrendAnalysis) -> Optional[PerformanceInsight]:
        """Generate insight from trend analysis"""
        if trend.direction == TrendDirection.DEGRADING and trend.confidence > 0.5:
            if "response_time" in trend.metric_name:
                return PerformanceInsight(
                    title="Response Time Degradation Detected",
                    description=f"{trend.metric_name} has increased by {trend.change_percentage:.1f}% over the analysis period",
                    impact="User experience may be affected by slower response times",
                    recommended_actions=[
                        "Profile slow endpoints and optimize database queries",
                        "Review recent code changes for performance regressions",
                        "Consider implementing caching strategies",
                        "Monitor external service dependencies"
                    ],
                    priority=AlertSeverity.HIGH if trend.change_percentage > 20 else AlertSeverity.MEDIUM,
                    affected_metrics=[trend.metric_name]
                )
            elif "memory_usage" in trend.metric_name:
                return PerformanceInsight(
                    title="Memory Usage Trending Upward",
                    description=f"Memory usage has increased by {trend.change_percentage:.1f}% and may indicate a memory leak",
                    impact="System stability may be at risk if memory usage continues to grow",
                    recommended_actions=[
                        "Review application code for memory leaks",
                        "Analyze object lifecycle and garbage collection",
                        "Implement memory monitoring and cleanup routines",
                        "Consider memory profiling tools"
                    ],
                    priority=AlertSeverity.HIGH,
                    affected_metrics=[trend.metric_name]
                )
        
        elif trend.direction == TrendDirection.VOLATILE:
            return PerformanceInsight(
                title="Performance Volatility Detected",
                description=f"{trend.metric_name} shows high variability, indicating unstable performance",
                impact="Inconsistent performance may affect user experience and system reliability",
                recommended_actions=[
                    "Investigate root causes of performance variability",
                    "Review system load patterns and resource allocation",
                    "Consider implementing performance smoothing mechanisms",
                    "Monitor for external factors affecting performance"
                ],
                priority=AlertSeverity.MEDIUM,
                affected_metrics=[trend.metric_name]
            )
        
        return None
    
    def _generate_system_insights(self, hours: int) -> List[PerformanceInsight]:
        """Generate system-level performance insights"""
        insights = []
        
        # Check for correlation between metrics
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get recent performance report
        report = self.monitor.get_performance_report(start_time, end_time)
        
        # Look for patterns in the summary
        if report.summary:
            cpu_avg = report.summary.get("cpu_usage", {}).get("avg", 0)
            memory_avg = report.summary.get("memory_usage", {}).get("avg", 0)
            response_time_avg = report.summary.get("response_time", {}).get("avg", 0)
            
            # High resource usage correlation
            if cpu_avg > 70 and memory_avg > 70:
                insights.append(PerformanceInsight(
                    title="High Resource Utilization",
                    description="Both CPU and memory usage are consistently high",
                    impact="System may be under-resourced or inefficiently utilizing available resources",
                    recommended_actions=[
                        "Consider scaling up system resources",
                        "Optimize resource-intensive operations",
                        "Implement resource monitoring and alerting",
                        "Review system architecture for bottlenecks"
                    ],
                    priority=AlertSeverity.HIGH,
                    affected_metrics=["cpu_usage", "memory_usage"]
                ))
            
            # Response time and resource correlation
            if response_time_avg > 2.0 and (cpu_avg > 60 or memory_avg > 60):
                insights.append(PerformanceInsight(
                    title="Performance Bottleneck Detected",
                    description="Slow response times correlate with high resource usage",
                    impact="User experience is degraded due to system resource constraints",
                    recommended_actions=[
                        "Profile application performance under load",
                        "Optimize database queries and data access patterns",
                        "Consider implementing request queuing or rate limiting",
                        "Review caching strategies and implementation"
                    ],
                    priority=AlertSeverity.HIGH,
                    affected_metrics=["response_time", "cpu_usage", "memory_usage"]
                ))
        
        return insights
    
    def generate_comprehensive_report(self, hours: int = 24) -> AnalysisReport:
        """Generate comprehensive performance analysis report"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        logger.info(f"Generating performance analysis report for {hours} hours")
        
        # Get base performance report
        base_report = self.monitor.get_performance_report(start_time, end_time)
        
        # Generate trend analysis for all metrics
        trends = []
        metric_names = set(m.name for m in base_report.metrics)
        
        for metric_name in metric_names:
            trend = self.analyze_trends(metric_name, hours)
            if trend:
                trends.append(trend)
        
        # Generate alerts
        alerts = self.generate_alerts(hours=1)  # Recent alerts
        
        # Generate insights
        insights = self.generate_insights(hours)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trends, alerts, insights)
        
        # Enhanced summary
        summary = base_report.summary.copy()
        summary.update({
            "analysis_period_hours": hours,
            "total_metrics_analyzed": len(metric_names),
            "trends_detected": len(trends),
            "active_alerts": len(alerts),
            "insights_generated": len(insights),
            "degrading_trends": len([t for t in trends if t.direction == TrendDirection.DEGRADING]),
            "improving_trends": len([t for t in trends if t.direction == TrendDirection.IMPROVING]),
            "volatile_metrics": len([t for t in trends if t.direction == TrendDirection.VOLATILE])
        })
        
        return AnalysisReport(
            start_time=start_time,
            end_time=end_time,
            trends=trends,
            alerts=alerts,
            insights=insights,
            summary=summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, trends: List[TrendAnalysis], 
                                alerts: List[PerformanceAlert], 
                                insights: List[PerformanceInsight]) -> List[str]:
        """Generate high-level recommendations based on analysis"""
        recommendations = []
        
        # Critical alerts
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            recommendations.append("URGENT: Address critical performance alerts immediately")
        
        # Degrading trends
        degrading_trends = [t for t in trends if t.direction == TrendDirection.DEGRADING]
        if len(degrading_trends) > 3:
            recommendations.append("Multiple metrics showing degradation - conduct comprehensive performance review")
        
        # High volatility
        volatile_metrics = [t for t in trends if t.direction == TrendDirection.VOLATILE]
        if len(volatile_metrics) > 2:
            recommendations.append("High performance volatility detected - investigate system stability")
        
        # Resource-specific recommendations
        memory_issues = any("memory" in a.metric_name for a in alerts)
        cpu_issues = any("cpu" in a.metric_name for a in alerts)
        
        if memory_issues and cpu_issues:
            recommendations.append("Consider system scaling or resource optimization")
        elif memory_issues:
            recommendations.append("Focus on memory optimization and leak detection")
        elif cpu_issues:
            recommendations.append("Optimize CPU-intensive operations and algorithms")
        
        # Response time issues
        response_issues = any("response_time" in a.metric_name for a in alerts)
        if response_issues:
            recommendations.append("Prioritize response time optimization for better user experience")
        
        if not recommendations:
            recommendations.append("System performance appears stable - continue monitoring")
        
        return recommendations

# Global performance analyzer instance
def get_performance_analyzer(monitor: PerformanceMonitor = None) -> PerformanceAnalyzer:
    """Get or create performance analyzer instance"""
    from .performance_monitor import performance_monitor
    return PerformanceAnalyzer(monitor or performance_monitor)