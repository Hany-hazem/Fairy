# app/monitoring_dashboard.py
"""
Web-based Monitoring Dashboard

This module provides a FastAPI-based web dashboard for monitoring
MCP operations and Git workflow performance with real-time updates.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from .mcp_monitoring_system import get_monitoring_system, Alert, AlertSeverity
from .config import settings

logger = logging.getLogger(__name__)

# Create FastAPI app for dashboard
dashboard_app = FastAPI(title="MCP Monitoring Dashboard", version="1.0.0")

# Templates and static files
templates = Jinja2Templates(directory="app/templates")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Get monitoring system components
metrics_collector, git_monitor, alert_manager, health_dashboard = get_monitoring_system()


@dashboard_app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@dashboard_app.get("/api/health")
async def get_health_status():
    """Get overall system health status"""
    try:
        health_status = await health_dashboard.get_health_status()
        return JSONResponse(content=health_status)
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_app.get("/api/dashboard")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        dashboard_data = await health_dashboard.get_dashboard_data()
        return JSONResponse(content=dashboard_data)
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_app.get("/api/metrics")
async def get_metrics():
    """Get all metrics summary"""
    try:
        metrics_summary = metrics_collector.get_metrics_summary()
        return JSONResponse(content=metrics_summary)
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_app.get("/api/metrics/{metric_name}")
async def get_metric_details(metric_name: str):
    """Get detailed information for a specific metric"""
    try:
        metric = metrics_collector.get_metric(metric_name)
        if not metric:
            raise HTTPException(status_code=404, detail="Metric not found")
        
        # Get recent data points
        recent_points = list(metric.data_points)[-100:]  # Last 100 points
        
        metric_details = {
            "name": metric.name,
            "type": metric.type.value,
            "description": metric.description,
            "unit": metric.unit,
            "labels": metric.labels,
            "latest_value": metric.get_latest_value(),
            "average_5min": metric.get_average(5),
            "average_1hour": metric.get_average(60),
            "data_points": [point.to_dict() for point in recent_points]
        }
        
        # Add percentiles for histograms and timers
        if metric.type.value in ["histogram", "timer"]:
            metric_details.update({
                "p50": metric.get_percentile(50),
                "p95": metric.get_percentile(95),
                "p99": metric.get_percentile(99)
            })
        
        return JSONResponse(content=metric_details)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metric details for {metric_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_app.get("/api/git/operations")
async def get_git_operations():
    """Get Git operation statistics"""
    try:
        git_stats = git_monitor.get_operation_stats()
        active_ops = git_monitor.get_active_operations()
        
        return JSONResponse(content={
            "statistics": git_stats,
            "active_operations": active_ops
        })
    except Exception as e:
        logger.error(f"Error getting Git operations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_app.get("/api/alerts")
async def get_alerts():
    """Get alert information"""
    try:
        active_alerts = alert_manager.get_active_alerts()
        alert_summary = alert_manager.get_alert_summary()
        
        return JSONResponse(content={
            "summary": alert_summary,
            "active_alerts": [alert.to_dict() for alert in active_alerts]
        })
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, request: Request):
    """Acknowledge an alert"""
    try:
        body = await request.json()
        acknowledged_by = body.get("acknowledged_by", "dashboard_user")
        
        await alert_manager.acknowledge_alert(alert_id, acknowledged_by)
        
        return JSONResponse(content={"status": "acknowledged"})
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    try:
        await alert_manager.resolve_alert(alert_id)
        return JSONResponse(content={"status": "resolved"})
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            dashboard_data = await health_dashboard.get_dashboard_data()
            await manager.send_personal_message(
                json.dumps({
                    "type": "dashboard_update",
                    "data": dashboard_data
                }),
                websocket
            )
            
            # Wait before next update
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Background task for broadcasting alerts
async def alert_broadcaster():
    """Background task to broadcast new alerts via WebSocket"""
    last_alert_count = 0
    
    while True:
        try:
            current_alerts = alert_manager.get_active_alerts()
            current_count = len(current_alerts)
            
            # Check for new alerts
            if current_count > last_alert_count:
                # Broadcast new alerts
                new_alerts = current_alerts[last_alert_count:]
                for alert in new_alerts:
                    await manager.broadcast(json.dumps({
                        "type": "new_alert",
                        "data": alert.to_dict()
                    }))
            
            last_alert_count = current_count
            await asyncio.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            logger.error(f"Error in alert broadcaster: {e}")
            await asyncio.sleep(10)


# Startup event
@dashboard_app.on_event("startup")
async def startup_event():
    """Initialize monitoring system on startup"""
    try:
        # Start monitoring components
        await metrics_collector.start()
        await alert_manager.start()
        
        # Start background alert broadcaster
        asyncio.create_task(alert_broadcaster())
        
        logger.info("Monitoring dashboard started successfully")
        
    except Exception as e:
        logger.error(f"Error starting monitoring dashboard: {e}")


# Shutdown event
@dashboard_app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await metrics_collector.stop()
        await alert_manager.stop()
        
        logger.info("Monitoring dashboard stopped")
        
    except Exception as e:
        logger.error(f"Error stopping monitoring dashboard: {e}")


# Create dashboard HTML template
DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .status-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        
        .status-card.healthy { border-left-color: #28a745; }
        .status-card.warning { border-left-color: #ffc107; }
        .status-card.critical { border-left-color: #dc3545; }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .alerts-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .alert-item {
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        
        .alert-critical { 
            background-color: #f8d7da; 
            border-left-color: #dc3545; 
        }
        
        .alert-error { 
            background-color: #f8d7da; 
            border-left-color: #fd7e14; 
        }
        
        .alert-warning { 
            background-color: #fff3cd; 
            border-left-color: #ffc107; 
        }
        
        .alert-info { 
            background-color: #d1ecf1; 
            border-left-color: #17a2b8; 
        }
        
        .alert-actions {
            margin-top: 10px;
        }
        
        .btn {
            padding: 5px 15px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            margin-right: 10px;
        }
        
        .btn-primary { background-color: #007bff; color: white; }
        .btn-success { background-color: #28a745; color: white; }
        
        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }
        
        .last-updated {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>MCP Monitoring Dashboard</h1>
        <p>Real-time monitoring of MCP operations and Git workflows</p>
    </div>
    
    <div id="loading" class="loading">
        <p>Loading dashboard data...</p>
    </div>
    
    <div id="dashboard-content" style="display: none;">
        <!-- System Health Status -->
        <div class="status-grid" id="status-grid">
            <!-- Status cards will be populated by JavaScript -->
        </div>
        
        <!-- Performance Charts -->
        <div class="charts-grid">
            <div class="chart-container">
                <h3>Message Throughput</h3>
                <canvas id="throughputChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>System Resources</h3>
                <canvas id="resourceChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Git Operations</h3>
                <canvas id="gitChart"></canvas>
            </div>
            <div class="chart-container">
                <h3>Response Times</h3>
                <canvas id="responseTimeChart"></canvas>
            </div>
        </div>
        
        <!-- Active Alerts -->
        <div class="alerts-section">
            <h3>Active Alerts</h3>
            <div id="alerts-container">
                <!-- Alerts will be populated by JavaScript -->
            </div>
        </div>
    </div>
    
    <div class="last-updated" id="last-updated"></div>
    
    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        // Chart instances
        let charts = {};
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            
            if (message.type === 'dashboard_update') {
                updateDashboard(message.data);
            } else if (message.type === 'new_alert') {
                showNewAlert(message.data);
            }
        };
        
        ws.onopen = function(event) {
            console.log('WebSocket connected');
        };
        
        ws.onclose = function(event) {
            console.log('WebSocket disconnected');
            // Attempt to reconnect after 5 seconds
            setTimeout(() => {
                location.reload();
            }, 5000);
        };
        
        function updateDashboard(data) {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('dashboard-content').style.display = 'block';
            
            // Update status cards
            updateStatusCards(data.system_health);
            
            // Update charts
            updateCharts(data.metrics);
            
            // Update alerts
            updateAlerts(data.alerts);
            
            // Update timestamp
            document.getElementById('last-updated').textContent = 
                `Last updated: ${new Date(data.timestamp).toLocaleString()}`;
        }
        
        function updateStatusCards(healthData) {
            const statusGrid = document.getElementById('status-grid');
            statusGrid.innerHTML = '';
            
            // Overall health
            const overallHealth = calculateOverallHealth(healthData);
            statusGrid.appendChild(createStatusCard(
                'Overall Health',
                overallHealth.status.toUpperCase(),
                overallHealth.status,
                `Score: ${overallHealth.score}/100`
            ));
            
            // MCP Health
            if (healthData.mcp_health) {
                statusGrid.appendChild(createStatusCard(
                    'MCP Operations',
                    `${healthData.mcp_health.success_rate.toFixed(1)}%`,
                    healthData.mcp_health.status,
                    'Success Rate'
                ));
            }
            
            // Resource Health
            if (healthData.resource_health) {
                statusGrid.appendChild(createStatusCard(
                    'System Resources',
                    `CPU: ${healthData.resource_health.cpu_usage.toFixed(1)}%`,
                    healthData.resource_health.status,
                    `Memory: ${healthData.resource_health.memory_usage.toFixed(1)}%`
                ));
            }
            
            // Connection Health
            if (healthData.connection_health) {
                statusGrid.appendChild(createStatusCard(
                    'Connections',
                    healthData.connection_health.active_connections.toString(),
                    healthData.connection_health.status,
                    `Pool Hit Rate: ${healthData.connection_health.pool_hit_rate.toFixed(1)}%`
                ));
            }
        }
        
        function createStatusCard(title, value, status, subtitle) {
            const card = document.createElement('div');
            card.className = `status-card ${status}`;
            card.innerHTML = `
                <div class="metric-value">${value}</div>
                <div class="metric-label">${title}</div>
                <div class="metric-label">${subtitle}</div>
            `;
            return card;
        }
        
        function calculateOverallHealth(healthData) {
            // Simple health calculation
            let score = 100;
            let status = 'healthy';
            
            if (healthData.mcp_health && healthData.mcp_health.success_rate < 95) {
                score -= 20;
                status = 'warning';
            }
            
            if (healthData.resource_health) {
                if (healthData.resource_health.cpu_usage > 90 || healthData.resource_health.memory_usage > 90) {
                    score -= 30;
                    status = 'critical';
                } else if (healthData.resource_health.cpu_usage > 80 || healthData.resource_health.memory_usage > 80) {
                    score -= 15;
                    if (status === 'healthy') status = 'warning';
                }
            }
            
            return { score: Math.max(0, score), status };
        }
        
        function updateCharts(metricsData) {
            // Initialize charts if not already done
            if (Object.keys(charts).length === 0) {
                initializeCharts();
            }
            
            // Update chart data (simplified implementation)
            // In a real implementation, you'd update with actual time series data
        }
        
        function initializeCharts() {
            // Throughput Chart
            const throughputCtx = document.getElementById('throughputChart').getContext('2d');
            charts.throughput = new Chart(throughputCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Messages/sec',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // Resource Chart
            const resourceCtx = document.getElementById('resourceChart').getContext('2d');
            charts.resource = new Chart(resourceCtx, {
                type: 'doughnut',
                data: {
                    labels: ['CPU Usage', 'Memory Usage', 'Available'],
                    datasets: [{
                        data: [30, 45, 25],
                        backgroundColor: ['#ff6384', '#36a2eb', '#4bc0c0']
                    }]
                },
                options: {
                    responsive: true
                }
            });
            
            // Git Operations Chart
            const gitCtx = document.getElementById('gitChart').getContext('2d');
            charts.git = new Chart(gitCtx, {
                type: 'bar',
                data: {
                    labels: ['Commits', 'Branches', 'Merges'],
                    datasets: [{
                        label: 'Operations',
                        data: [12, 5, 3],
                        backgroundColor: ['#4bc0c0', '#9966ff', '#ff9f40']
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // Response Time Chart
            const responseCtx = document.getElementById('responseTimeChart').getContext('2d');
            charts.responseTime = new Chart(responseCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Response Time (ms)',
                        data: [],
                        borderColor: '#ff6384',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        function updateAlerts(alertsData) {
            const alertsContainer = document.getElementById('alerts-container');
            
            if (!alertsData.active_alerts || alertsData.active_alerts.length === 0) {
                alertsContainer.innerHTML = '<p>No active alerts</p>';
                return;
            }
            
            alertsContainer.innerHTML = '';
            
            alertsData.active_alerts.forEach(alert => {
                const alertElement = document.createElement('div');
                alertElement.className = `alert-item alert-${alert.severity}`;
                alertElement.innerHTML = `
                    <strong>${alert.rule_name}</strong>
                    <p>${alert.message}</p>
                    <small>Triggered: ${new Date(alert.triggered_at).toLocaleString()}</small>
                    <div class="alert-actions">
                        <button class="btn btn-primary" onclick="acknowledgeAlert('${alert.id}')">
                            Acknowledge
                        </button>
                        <button class="btn btn-success" onclick="resolveAlert('${alert.id}')">
                            Resolve
                        </button>
                    </div>
                `;
                alertsContainer.appendChild(alertElement);
            });
        }
        
        function showNewAlert(alert) {
            // Show notification for new alert
            if (Notification.permission === 'granted') {
                new Notification(`New ${alert.severity} Alert`, {
                    body: alert.message,
                    icon: '/static/alert-icon.png'
                });
            }
        }
        
        async function acknowledgeAlert(alertId) {
            try {
                const response = await fetch(`/api/alerts/${alertId}/acknowledge`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        acknowledged_by: 'dashboard_user'
                    })
                });
                
                if (response.ok) {
                    console.log('Alert acknowledged');
                }
            } catch (error) {
                console.error('Error acknowledging alert:', error);
            }
        }
        
        async function resolveAlert(alertId) {
            try {
                const response = await fetch(`/api/alerts/${alertId}/resolve`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    console.log('Alert resolved');
                }
            } catch (error) {
                console.error('Error resolving alert:', error);
            }
        }
        
        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
        
        // Initial data load
        fetch('/api/dashboard')
            .then(response => response.json())
            .then(data => updateDashboard(data))
            .catch(error => {
                console.error('Error loading dashboard:', error);
                document.getElementById('loading').innerHTML = 
                    '<p>Error loading dashboard data. Please refresh the page.</p>';
            });
    </script>
</body>
</html>
"""

# Create templates directory and dashboard template
import os
os.makedirs("app/templates", exist_ok=True)

with open("app/templates/dashboard.html", "w") as f:
    f.write(DASHBOARD_HTML_TEMPLATE)


def start_monitoring_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Start the monitoring dashboard server"""
    logger.info(f"Starting monitoring dashboard on {host}:{port}")
    uvicorn.run(dashboard_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_monitoring_dashboard()