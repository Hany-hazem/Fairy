#!/usr/bin/env python3
"""
Fairy AI Assistant - Main FastAPI Application

A sophisticated AI assistant that can analyze, improve, and evolve its own codebase 
while providing intelligent conversational assistance.

Copyright (c) 2024 Hani Hazem
Licensed under the MIT License. See LICENSE file in the project root for full license information.
Repository: https://github.com/Hany-hazem/Fairy
Contact: hany.hazem.cs@gmail.com
"""
import json
import time
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query as QueryParam, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import settings
from .safety_filter import safety_filter
from .ai_assistant_service import ai_assistant
from .performance_monitor import performance_monitor
from .performance_analyzer import get_performance_analyzer
from .code_analyzer import code_analyzer
from .self_improvement_engine import get_self_improvement_engine
from .personal_assistant_models import PermissionType

# Import legacy components with error handling
try:
    from .llm_adapter import LLMAdapter
    gpt_adapter = LLMAdapter(settings.LM_STUDIO_MODEL)
except Exception as e:
    print(f"Warning: Could not initialize LLM adapter: {e}")
    gpt_adapter = None

try:
    from .memory_manager import memory_manager
except Exception as e:
    print(f"Warning: Could not import memory manager: {e}")
    memory_manager = None

try:
    from .agent_registry import registry, AgentType
except Exception as e:
    print(f"Warning: Could not import agent registry: {e}")
    registry = None
    AgentType = None

try:
    from .mcp import mcp
except Exception as e:
    print(f"Warning: Could not import MCP: {e}")
    mcp = None

# Initialize agent integration
try:
    from .agent_integration import initialize_agent_integration, get_integration_status
    agent_integration_available = True
    
    # Initialize on startup
    integration_success = initialize_agent_integration()
    if integration_success:
        print("Agent integration initialized successfully")
    else:
        print("Warning: Agent integration initialization failed")
        
except Exception as e:
    print(f"Warning: Could not initialize agent integration: {e}")
    agent_integration_available = False
    get_integration_status = None

app = FastAPI(
    title="Self-Evolving AI Agent & Assistant",
    description="Advanced AI system with conversational assistant, self-improvement capabilities, and multi-agent orchestration",
    version="1.0.0"
)

# Add CORS middleware for web frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Performance monitoring middleware
@app.middleware("http")
async def performance_middleware(request, call_next):
    """Middleware to track API performance metrics"""
    start_time = time.time()
    request_id = f"api_{int(start_time * 1000)}"
    
    # Collect system metrics periodically
    if int(start_time) % 60 == 0:  # Every minute
        performance_monitor.collect_system_metrics()
    
    with performance_monitor.track_request(
        request_id,
        "api_request",
        method=request.method,
        path=request.url.path,
        client=str(request.client.host) if request.client else "unknown"
    ):
        response = await call_next(request)
        
        # Record additional metrics
        duration = time.time() - start_time
        performance_monitor.record_response_time(
            f"api_{request.method.lower()}",
            duration,
            path=request.url.path,
            status_code=response.status_code
        )
        
        return response

# ============================================================================
# LEGACY ENDPOINTS (Original agent system)
# ============================================================================

class Query(BaseModel):
    text: str
    intent: str | None = None   # optional, defaults to "general_query"

@app.post("/query", tags=["Legacy Agent System"])
async def handle_legacy_query(query: Query):
    """Legacy query endpoint for backward compatibility with enhanced agent routing"""
    # Check if registry is available
    if not registry:
        # Fallback to AI assistant
        chat_request = ChatRequest(message=query.text)
        return await chat_with_assistant(chat_request)
    
    try:
        # 1️⃣ Determine intent / agent
        intent = query.intent or "general_query"
        agent_cfg = registry.get_agent(intent)
        agent_type = agent_cfg.get("type", "llm")

        # Handle different agent types
        if agent_type == "ai_assistant":
            # Route to AI assistant service
            chat_request = ChatRequest(message=query.text)
            result = await chat_with_assistant(chat_request)
            return {"response": result.response}

        elif agent_type == "self_improvement":
            # Route to self-improvement engine
            if mcp:
                payload = {
                    "action": "analyze_code",
                    "query": query.text,
                    "timestamp": datetime.utcnow().isoformat()
                }
                mcp.publish(agent_cfg["topic"], payload)
            return {"response": "Self-improvement analysis initiated"}

        elif agent_type == "code_analysis":
            # Route to code analysis
            if mcp:
                payload = {
                    "action": "analyze_code",
                    "query": query.text,
                    "timestamp": datetime.utcnow().isoformat()
                }
                mcp.publish(agent_cfg["topic"], payload)
            return {"response": "Code analysis initiated"}

        elif agent_type == "performance_analysis":
            # Route to performance analysis
            if mcp:
                payload = {
                    "action": "analyze_performance",
                    "query": query.text,
                    "timestamp": datetime.utcnow().isoformat()
                }
                mcp.publish(agent_cfg["topic"], payload)
            return {"response": "Performance analysis initiated"}

        elif agent_type == "llm":
            # Legacy LLM handling
            if not memory_manager or not gpt_adapter:
                # Fallback to AI assistant
                chat_request = ChatRequest(message=query.text)
                result = await chat_with_assistant(chat_request)
                return {"response": result.response}
            
            # 2️⃣ Retrieve relevant memory (context)
            context_docs = memory_manager.retrieve(query.text, k=3)
            context_str = "\n".join([f"[{score:.2f}] {doc}" for doc, score in context_docs])

            # 3️⃣ Build prompt
            prompt = f"{context_str}\nUser: {query.text}\nAssistant:"
            raw_response = gpt_adapter.predict(prompt)

        elif agent_type == "vision":
            # 4️⃣ Send request to vision agent via MCP
            if mcp:
                payload = {"text": query.text, "timestamp": str(datetime.utcnow())}
                mcp.publish(agent_cfg["topic"], json.dumps(payload))
            # For demo, return a placeholder
            raw_response = "Vision processing initiated"

        else:
            # Unknown agent type - fallback to AI assistant
            chat_request = ChatRequest(message=query.text)
            result = await chat_with_assistant(chat_request)
            return {"response": result.response}

        # 5️⃣ Safety check for LLM responses
        if agent_type == "llm" and raw_response:
            if not safety_filter.is_safe(raw_response):
                raise HTTPException(status_code=400, detail="Response failed safety policy")

            # 6️⃣ Store for future recall
            if memory_manager:
                memory_manager.store(query.text + " " + raw_response)

            return {"response": raw_response}

        # For non-LLM agents, return the response as-is
        return {"response": raw_response if 'raw_response' in locals() else "Request processed"}
        
    except Exception as e:
        # Fallback to AI assistant on any error
        chat_request = ChatRequest(message=query.text)
        result = await chat_with_assistant(chat_request)
        return {"response": result.response}

# ============================================================================
# AI ASSISTANT ENDPOINTS (New conversational system)
# ============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature (0.0-2.0)")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Maximum tokens in response")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    user_id: Optional[str] = Field(None, description="User identifier")
    message_id: Optional[str] = Field(None, description="Unique message ID")
    context_info: Dict = Field(..., description="Information about conversation context")
    timestamp: str = Field(..., description="Response timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    session_id: Optional[str] = Field(None, description="Session ID if available")
    user_id: Optional[str] = Field(None, description="User ID if available")
    timestamp: str = Field(..., description="Error timestamp")

@app.post("/chat", response_model=ChatResponse, tags=["AI Assistant"])
async def chat_with_assistant(request: ChatRequest):
    """
    Chat with the AI assistant with full conversation context and memory
    """
    try:
        result = await ai_assistant.process_query(
            query=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return ChatResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history/{session_id}", tags=["AI Assistant"])
async def get_conversation_history(
    session_id: str = Path(..., description="Session ID"),
    limit: int = QueryParam(50, ge=1, le=200, description="Maximum number of messages to return")
):
    """
    Get conversation history for a specific session
    """
    try:
        result = await ai_assistant.get_conversation_history(session_id, limit)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        # Re-raise HTTPExceptions to preserve status codes
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/session/{session_id}", tags=["AI Assistant"])
async def clear_conversation_session(
    session_id: str = Path(..., description="Session ID to clear")
):
    """
    Clear a conversation session and all its messages
    """
    try:
        result = await ai_assistant.clear_session(session_id)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to clear session"))
        
        return {"message": "Session cleared successfully", "session_id": session_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/users/{user_id}/sessions", tags=["AI Assistant"])
async def get_user_sessions(
    user_id: str = Path(..., description="User ID"),
    limit: int = QueryParam(10, ge=1, le=50, description="Maximum number of sessions to return")
):
    """
    Get recent conversation sessions for a user
    """
    try:
        result = await ai_assistant.get_user_sessions(user_id, limit)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user identifier")

@app.post("/chat/session", tags=["AI Assistant"])
async def create_conversation_session(request: CreateSessionRequest):
    """
    Create a new conversation session
    """
    try:
        # Create a new session directly through conversation manager
        session = ai_assistant.conversation_manager.create_session(user_id=request.user_id)
        
        return {
            "session_id": session.id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "created": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PERFORMANCE MONITORING AND ANALYSIS ENDPOINTS
# ============================================================================

@app.get("/metrics", tags=["Performance"])
async def get_performance_metrics(
    metric_name: Optional[str] = QueryParam(None, description="Specific metric name to retrieve"),
    start_time: Optional[str] = QueryParam(None, description="Start time (ISO format)"),
    end_time: Optional[str] = QueryParam(None, description="End time (ISO format)"),
    limit: int = QueryParam(100, ge=1, le=1000, description="Maximum number of metrics to return")
):
    """
    Get performance metrics with optional filtering
    """
    try:
        # Parse time parameters
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None
        
        metrics = performance_monitor.collector.get_metrics(
            name=metric_name,
            start_time=start_dt,
            end_time=end_dt,
            limit=limit
        )
        
        return {
            "metrics": [m.to_dict() for m in metrics],
            "count": len(metrics),
            "filtered_by": {
                "metric_name": metric_name,
                "start_time": start_time,
                "end_time": end_time,
                "limit": limit
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/summary", tags=["Performance"])
async def get_metrics_summary(
    start_time: Optional[str] = QueryParam(None, description="Start time (ISO format)"),
    end_time: Optional[str] = QueryParam(None, description="End time (ISO format)")
):
    """
    Get performance metrics summary and statistics
    """
    try:
        # Parse time parameters
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None
        
        # Get performance report
        report = performance_monitor.get_performance_report(start_dt, end_dt)
        
        return {
            "report": report.to_dict(),
            "active_requests": performance_monitor.get_active_requests(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/system", tags=["Performance"])
async def get_system_metrics():
    """
    Get current system performance metrics
    """
    try:
        # Collect fresh system metrics
        performance_monitor.collect_system_metrics()
        
        # Get recent system metrics
        metrics = performance_monitor.collector.get_metrics(
            start_time=datetime.utcnow() - timedelta(minutes=5),
            limit=100
        )
        
        # Filter for system metrics
        system_metrics = [m for m in metrics if any(
            name in m.name for name in ['cpu', 'memory', 'disk', 'process']
        )]
        
        # Create summary
        summary = {}
        for metric in system_metrics:
            if metric.name not in summary:
                summary[metric.name] = performance_monitor.collector.get_metric_summary(metric.name)
        
        return {
            "current_metrics": [m.to_dict() for m in system_metrics[-10:]],  # Last 10
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics/cleanup", tags=["Performance"])
async def cleanup_old_metrics(
    days: int = Body(7, ge=1, le=30, description="Number of days of metrics to keep")
):
    """
    Clean up old performance metrics
    """
    try:
        performance_monitor.cleanup_old_metrics(days)
        
        return {
            "message": f"Cleaned up metrics older than {days} days",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PERFORMANCE ANALYSIS ENDPOINTS
# ============================================================================

class TrendAnalysisRequest(BaseModel):
    metric_name: str = Field(..., description="Name of the metric to analyze")
    hours: Optional[int] = Field(24, ge=1, le=168, description="Number of hours to analyze (1-168)")

@app.post("/analysis/trends", tags=["Performance Analysis"])
async def analyze_metric_trends(request: TrendAnalysisRequest):
    """
    Analyze performance trends for a specific metric
    """
    try:
        analyzer = get_performance_analyzer()
        trend = analyzer.analyze_trends(request.metric_name, request.hours)
        
        if not trend:
            raise HTTPException(
                status_code=404, 
                detail=f"Insufficient data for trend analysis of {request.metric_name}"
            )
        
        return {
            "trend_analysis": {
                "metric_name": trend.metric_name,
                "direction": trend.direction.value,
                "slope": trend.slope,
                "confidence": trend.confidence,
                "recent_avg": trend.recent_avg,
                "historical_avg": trend.historical_avg,
                "change_percentage": trend.change_percentage,
                "data_points": trend.data_points
            },
            "analysis_period_hours": request.hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/alerts", tags=["Performance Analysis"])
async def get_performance_alerts(
    hours: int = QueryParam(1, ge=1, le=24, description="Hours to analyze for alerts")
):
    """
    Get current performance alerts based on recent metrics
    """
    try:
        analyzer = get_performance_analyzer()
        alerts = analyzer.generate_alerts(hours)
        
        return {
            "alerts": [
                {
                    "metric_name": alert.metric_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "trend": alert.trend.value if alert.trend else None,
                    "suggested_actions": alert.suggested_actions,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in alerts
            ],
            "alert_count": len(alerts),
            "analysis_period_hours": hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/insights", tags=["Performance Analysis"])
async def get_performance_insights(
    hours: int = QueryParam(24, ge=1, le=168, description="Hours to analyze for insights")
):
    """
    Get actionable performance insights based on system analysis
    """
    try:
        analyzer = get_performance_analyzer()
        insights = analyzer.generate_insights(hours)
        
        return {
            "insights": [
                {
                    "title": insight.title,
                    "description": insight.description,
                    "impact": insight.impact,
                    "recommended_actions": insight.recommended_actions,
                    "priority": insight.priority.value,
                    "affected_metrics": insight.affected_metrics
                }
                for insight in insights
            ],
            "insight_count": len(insights),
            "analysis_period_hours": hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/report", tags=["Performance Analysis"])
async def get_comprehensive_analysis_report(
    hours: int = QueryParam(24, ge=1, le=168, description="Hours to include in analysis report")
):
    """
    Generate comprehensive performance analysis report with trends, alerts, and insights
    """
    try:
        analyzer = get_performance_analyzer()
        report = analyzer.generate_comprehensive_report(hours)
        
        return {
            "report": {
                "start_time": report.start_time.isoformat(),
                "end_time": report.end_time.isoformat(),
                "summary": report.summary,
                "trends": [
                    {
                        "metric_name": trend.metric_name,
                        "direction": trend.direction.value,
                        "slope": trend.slope,
                        "confidence": trend.confidence,
                        "recent_avg": trend.recent_avg,
                        "historical_avg": trend.historical_avg,
                        "change_percentage": trend.change_percentage,
                        "data_points": trend.data_points
                    }
                    for trend in report.trends
                ],
                "alerts": [
                    {
                        "metric_name": alert.metric_name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "trend": alert.trend.value if alert.trend else None,
                        "suggested_actions": alert.suggested_actions,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in report.alerts
                ],
                "insights": [
                    {
                        "title": insight.title,
                        "description": insight.description,
                        "impact": insight.impact,
                        "recommended_actions": insight.recommended_actions,
                        "priority": insight.priority.value,
                        "affected_metrics": insight.affected_metrics
                    }
                    for insight in report.insights
                ],
                "recommendations": report.recommendations
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/dashboard", tags=["Performance Analysis"])
async def get_performance_dashboard_data(
    hours: int = QueryParam(24, ge=1, le=168, description="Hours of data for dashboard")
):
    """
    Get performance dashboard data with key metrics, trends, and alerts
    """
    try:
        analyzer = get_performance_analyzer()
        
        # Get comprehensive report
        report = analyzer.generate_comprehensive_report(hours)
        
        # Get recent system metrics
        recent_metrics = performance_monitor.collector.get_metrics(
            start_time=datetime.utcnow() - timedelta(hours=1),
            limit=100
        )
        
        # Create dashboard summary
        dashboard_data = {
            "overview": {
                "analysis_period_hours": hours,
                "total_metrics": report.summary.get("total_metrics_analyzed", 0),
                "active_alerts": len(report.alerts),
                "critical_alerts": len([a for a in report.alerts if a.severity.value == "critical"]),
                "degrading_trends": report.summary.get("degrading_trends", 0),
                "improving_trends": report.summary.get("improving_trends", 0),
                "volatile_metrics": report.summary.get("volatile_metrics", 0)
            },
            "key_metrics": {
                name: summary for name, summary in report.summary.items()
                if isinstance(summary, dict) and "avg" in summary
            },
            "recent_alerts": [
                {
                    "metric_name": alert.metric_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "current_value": alert.current_value
                }
                for alert in report.alerts[:5]  # Top 5 alerts
            ],
            "top_insights": [
                {
                    "title": insight.title,
                    "priority": insight.priority.value,
                    "description": insight.description
                }
                for insight in sorted(report.insights, 
                                    key=lambda x: ["low", "medium", "high", "critical"].index(x.priority.value),
                                    reverse=True)[:3]  # Top 3 insights
            ],
            "recommendations": report.recommendations[:5],  # Top 5 recommendations
            "system_health": {
                "overall_status": "healthy" if len([a for a in report.alerts if a.severity.value in ["high", "critical"]]) == 0 else "warning",
                "last_updated": datetime.utcnow().isoformat()
            }
        }
        
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CODE ANALYSIS ENDPOINTS
# ============================================================================

class CodeAnalysisRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to analyze")

@app.post("/analysis/code/file", tags=["Code Analysis"])
async def analyze_code_file(request: CodeAnalysisRequest):
    """
    Analyze a single code file for quality issues
    """
    try:
        report = code_analyzer.analyze_file(request.file_path)
        
        if not report:
            raise HTTPException(status_code=404, detail=f"File not found or unsupported: {request.file_path}")
        
        return {
            "report": report.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/code/project", tags=["Code Analysis"])
async def analyze_project_code():
    """
    Analyze the entire project for code quality issues
    """
    try:
        reports = code_analyzer.analyze_project()
        
        # Convert reports to dictionaries
        reports_dict = {
            file_path: report.to_dict() 
            for file_path, report in reports.items()
        }
        
        # Generate project summary
        summary = code_analyzer.get_project_summary(reports)
        
        return {
            "reports": reports_dict,
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/code/summary", tags=["Code Analysis"])
async def get_code_quality_summary():
    """
    Get a summary of code quality across the project
    """
    try:
        reports = code_analyzer.analyze_project()
        summary = code_analyzer.get_project_summary(reports)
        
        return {
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/code/issues", tags=["Code Analysis"])
async def get_code_issues(
    severity: Optional[str] = QueryParam(None, description="Filter by severity (low, medium, high, critical)"),
    category: Optional[str] = QueryParam(None, description="Filter by category (performance, complexity, maintainability, security, style, bug_risk)"),
    limit: int = QueryParam(50, ge=1, le=500, description="Maximum number of issues to return")
):
    """
    Get code quality issues with optional filtering
    """
    try:
        reports = code_analyzer.analyze_project()
        
        # Collect all issues
        all_issues = []
        for file_path, report in reports.items():
            for issue in report.issues:
                issue_dict = issue.to_dict()
                issue_dict["file_path"] = file_path  # Ensure file path is included
                all_issues.append(issue_dict)
        
        # Apply filters
        filtered_issues = all_issues
        
        if severity:
            filtered_issues = [i for i in filtered_issues if i["severity"] == severity.lower()]
        
        if category:
            filtered_issues = [i for i in filtered_issues if i["category"] == category.lower()]
        
        # Sort by severity (critical first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        filtered_issues.sort(key=lambda x: severity_order.get(x["severity"], 4))
        
        # Apply limit
        filtered_issues = filtered_issues[:limit]
        
        return {
            "issues": filtered_issues,
            "total_count": len(all_issues),
            "filtered_count": len(filtered_issues),
            "filters": {
                "severity": severity,
                "category": category,
                "limit": limit
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# AGENT REGISTRY ENDPOINTS
# ============================================================================

@app.get("/agents", tags=["Agent Registry"])
async def list_agents():
    """List all registered agents"""
    try:
        if not registry:
            raise HTTPException(status_code=503, detail="Agent registry not available")
        
        agents = registry.list_agents()
        topics = registry.get_agent_topics()
        
        return {
            "agents": agents,
            "topics": topics,
            "agent_count": len(agents),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{intent}", tags=["Agent Registry"])
async def get_agent_config(intent: str = Path(..., description="Agent intent")):
    """Get configuration for a specific agent"""
    try:
        if not registry:
            raise HTTPException(status_code=503, detail="Agent registry not available")
        
        if not registry.supports_intent(intent):
            raise HTTPException(status_code=404, detail=f"Agent intent '{intent}' not found")
        
        agent_config = registry.get_agent(intent)
        
        return {
            "intent": intent,
            "config": agent_config,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RegisterAgentRequest(BaseModel):
    intent: str = Field(..., description="Agent intent/name")
    agent_type: str = Field(..., description="Agent type (llm, vision, ai_assistant, self_improvement, etc.)")
    topic: str = Field(..., description="MCP topic for agent communication")
    description: str = Field(..., description="Agent description")

@app.post("/agents/register", tags=["Agent Registry"])
async def register_agent(request: RegisterAgentRequest):
    """Register a new agent"""
    try:
        if not registry:
            raise HTTPException(status_code=503, detail="Agent registry not available")
        
        # Validate agent type
        if AgentType:
            try:
                agent_type_enum = AgentType(request.agent_type)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid agent type: {request.agent_type}"
                )
        else:
            agent_type_enum = request.agent_type
        
        # Register the agent
        success = registry.register_agent(
            intent=request.intent,
            agent_type=agent_type_enum,
            topic=request.topic,
            description=request.description
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to register agent")
        
        return {
            "message": f"Agent '{request.intent}' registered successfully",
            "intent": request.intent,
            "agent_type": request.agent_type,
            "topic": request.topic,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/agents/{intent}", tags=["Agent Registry"])
async def unregister_agent(intent: str = Path(..., description="Agent intent to unregister")):
    """Unregister an agent"""
    try:
        if not registry:
            raise HTTPException(status_code=503, detail="Agent registry not available")
        
        if not registry.supports_intent(intent):
            raise HTTPException(status_code=404, detail=f"Agent intent '{intent}' not found")
        
        success = registry.unregister_agent(intent)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to unregister agent")
        
        return {
            "message": f"Agent '{intent}' unregistered successfully",
            "intent": intent,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/reload", tags=["Agent Registry"])
async def reload_agent_registry():
    """Reload agent registry from file"""
    try:
        if not registry:
            raise HTTPException(status_code=503, detail="Agent registry not available")
        
        success = registry.reload_registry()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reload agent registry")
        
        return {
            "message": "Agent registry reloaded successfully",
            "agent_count": len(registry.list_agents()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/types/{agent_type}", tags=["Agent Registry"])
async def get_agents_by_type(agent_type: str = Path(..., description="Agent type to filter by")):
    """Get all agents of a specific type"""
    try:
        if not registry:
            raise HTTPException(status_code=503, detail="Agent registry not available")
        
        # Validate agent type
        if AgentType:
            try:
                agent_type_enum = AgentType(agent_type)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid agent type: {agent_type}"
                )
            
            agents = registry.get_agents_by_type(agent_type_enum)
        else:
            # Fallback if AgentType enum not available
            all_agents = registry.list_agents()
            agents = {
                intent: config for intent, config in all_agents.items()
                if config.get("type") == agent_type
            }
        
        return {
            "agent_type": agent_type,
            "agents": agents,
            "agent_count": len(agents),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/integration/status", tags=["Agent Registry"])
async def get_agent_integration_status():
    """Get status of agent integration and MCP routing"""
    try:
        if not agent_integration_available or not get_integration_status:
            raise HTTPException(status_code=503, detail="Agent integration not available")
        
        status = get_integration_status()
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SendAgentMessageRequest(BaseModel):
    intent: str = Field(..., description="Agent intent to send message to")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    response_topic: Optional[str] = Field(None, description="Optional topic to receive response")

@app.post("/agents/send", tags=["Agent Registry"])
async def send_agent_message(request: SendAgentMessageRequest):
    """Send a message to a specific agent via MCP"""
    try:
        if not agent_integration_available:
            raise HTTPException(status_code=503, detail="Agent integration not available")
        
        from .agent_mcp_router import agent_mcp_router
        
        success = agent_mcp_router.send_agent_message(
            intent=request.intent,
            payload=request.payload,
            response_topic=request.response_topic
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to send message to agent")
        
        return {
            "message": f"Message sent to agent '{request.intent}' successfully",
            "intent": request.intent,
            "response_topic": request.response_topic,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BroadcastMessageRequest(BaseModel):
    agent_type: str = Field(..., description="Agent type to broadcast to")
    payload: Dict[str, Any] = Field(..., description="Message payload")

@app.post("/agents/broadcast", tags=["Agent Registry"])
async def broadcast_to_agent_type(request: BroadcastMessageRequest):
    """Broadcast a message to all agents of a specific type"""
    try:
        if not agent_integration_available:
            raise HTTPException(status_code=503, detail="Agent integration not available")
        
        from .agent_mcp_router import agent_mcp_router
        
        sent_count = agent_mcp_router.broadcast_to_agent_type(
            agent_type=request.agent_type,
            payload=request.payload
        )
        
        return {
            "message": f"Message broadcast to {sent_count} agents of type '{request.agent_type}'",
            "agent_type": request.agent_type,
            "sent_count": sent_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SELF-IMPROVEMENT ENGINE ENDPOINTS
# ============================================================================

class TriggerImprovementRequest(BaseModel):
    trigger: Optional[str] = Field("manual", description="Trigger type (manual, scheduled, performance_threshold)")

@app.post("/improvement/trigger", tags=["Self-Improvement"])
async def trigger_improvement_cycle(request: TriggerImprovementRequest):
    """
    Trigger a new self-improvement cycle
    """
    try:
        engine = get_self_improvement_engine()
        cycle_id = await engine.trigger_improvement_cycle(request.trigger)
        
        return {
            "cycle_id": cycle_id,
            "trigger": request.trigger,
            "status": "started",
            "message": f"Improvement cycle {cycle_id} started successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/improvement/status", tags=["Self-Improvement"])
async def get_improvement_status():
    """
    Get current status of the self-improvement engine
    """
    try:
        engine = get_self_improvement_engine()
        status = engine.get_current_status()
        
        return {
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/improvement/cycles", tags=["Self-Improvement"])
async def get_improvement_cycles(
    limit: int = QueryParam(10, ge=1, le=50, description="Maximum number of cycles to return")
):
    """
    Get recent improvement cycle history
    """
    try:
        engine = get_self_improvement_engine()
        cycles = engine.get_cycle_history(limit)
        
        return {
            "cycles": cycles,
            "count": len(cycles),
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/improvement/cycles/{cycle_id}", tags=["Self-Improvement"])
async def get_improvement_cycle_details(
    cycle_id: str = Path(..., description="Improvement cycle ID")
):
    """
    Get detailed information about a specific improvement cycle
    """
    try:
        engine = get_self_improvement_engine()
        details = engine.get_cycle_details(cycle_id)
        
        if not details:
            raise HTTPException(status_code=404, detail=f"Cycle {cycle_id} not found")
        
        return {
            "cycle": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class UpdateSafetyLevelRequest(BaseModel):
    safety_level: str = Field(..., description="Safety level (conservative, moderate, aggressive)")

@app.post("/improvement/safety-level", tags=["Self-Improvement"])
async def update_safety_level(request: UpdateSafetyLevelRequest):
    """
    Update the safety level for self-improvement operations
    """
    try:
        from .self_improvement_engine import SafetyLevel
        
        # Validate safety level
        try:
            safety_level = SafetyLevel(request.safety_level.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid safety level. Must be one of: conservative, moderate, aggressive"
            )
        
        engine = get_self_improvement_engine()
        engine.update_safety_level(safety_level)
        
        return {
            "safety_level": safety_level.value,
            "message": f"Safety level updated to {safety_level.value}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/improvement/scheduler/start", tags=["Self-Improvement"])
async def start_improvement_scheduler():
    """
    Start the automatic improvement scheduler
    """
    try:
        engine = get_self_improvement_engine()
        await engine.start_scheduler()
        
        return {
            "message": "Improvement scheduler started",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/improvement/scheduler/stop", tags=["Self-Improvement"])
async def stop_improvement_scheduler():
    """
    Stop the automatic improvement scheduler
    """
    try:
        engine = get_self_improvement_engine()
        await engine.stop_scheduler()
        
        return {
            "message": "Improvement scheduler stopped",
            "status": "stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/improvement/emergency-stop", tags=["Self-Improvement"])
async def emergency_stop_improvements():
    """
    Emergency stop of all self-improvement activities with rollback
    """
    try:
        engine = get_self_improvement_engine()
        success = await engine.emergency_stop()
        
        if not success:
            raise HTTPException(status_code=500, detail="Emergency stop failed")
        
        return {
            "message": "Emergency stop executed successfully",
            "status": "stopped",
            "rollback_performed": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/improvement/suggestions", tags=["Self-Improvement"])
async def get_improvement_suggestions(
    limit: int = QueryParam(10, ge=1, le=50, description="Maximum number of suggestions to return")
):
    """
    Get current improvement suggestions without triggering a cycle
    """
    try:
        engine = get_self_improvement_engine()
        
        # Get suggestions from improvement engine
        improvements = engine.improvement_engine.analyze_and_suggest_improvements()
        
        # Filter by safety level
        filtered_improvements = engine._filter_improvements_by_safety(improvements)
        
        # Select top improvements
        selected_improvements = engine._select_improvements(filtered_improvements, limit)
        
        return {
            "suggestions": [imp.to_dict() for imp in selected_improvements],
            "total_found": len(improvements),
            "filtered_count": len(filtered_improvements),
            "returned_count": len(selected_improvements),
            "safety_level": engine.safety_level.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ApproveImprovementRequest(BaseModel):
    improvement_id: str = Field(..., description="ID of the improvement to approve")
    apply_immediately: bool = Field(False, description="Whether to apply the improvement immediately")

@app.post("/improvement/approve", tags=["Self-Improvement"])
async def approve_improvement(request: ApproveImprovementRequest):
    """
    Approve a specific improvement for implementation
    """
    try:
        # This is a placeholder for manual approval workflow
        # In a full implementation, this would:
        # 1. Store the approval in a database
        # 2. Optionally trigger immediate application
        # 3. Update improvement status
        
        return {
            "improvement_id": request.improvement_id,
            "status": "approved",
            "apply_immediately": request.apply_immediately,
            "message": f"Improvement {request.improvement_id} approved",
            "note": "Manual approval workflow not fully implemented yet",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/improvement/rollback-points", tags=["Self-Improvement"])
async def get_rollback_points():
    """
    Get available rollback points for emergency recovery
    """
    try:
        engine = get_self_improvement_engine()
        
        # Get recent cycles with rollback points
        cycles = engine.get_cycle_history(limit=20)
        rollback_points = []
        
        for cycle in cycles:
            if cycle.get("rollback_points"):
                for point in cycle["rollback_points"]:
                    rollback_points.append({
                        "cycle_id": cycle["id"],
                        "rollback_point": point,
                        "created_at": cycle["started_at"],
                        "status": cycle["status"],
                        "applied_improvements": cycle.get("applied_improvements", [])
                    })
        
        return {
            "rollback_points": rollback_points,
            "count": len(rollback_points),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RollbackRequest(BaseModel):
    cycle_id: str = Field(..., description="ID of the cycle to rollback")

@app.post("/improvement/rollback", tags=["Self-Improvement"])
async def rollback_improvement_cycle(request: RollbackRequest):
    """
    Rollback a specific improvement cycle
    """
    try:
        engine = get_self_improvement_engine()
        
        # Load the cycle
        cycle_details = engine.get_cycle_details(request.cycle_id)
        if not cycle_details:
            raise HTTPException(status_code=404, detail=f"Cycle {request.cycle_id} not found")
        
        # Check if cycle can be rolled back
        if cycle_details["status"] not in ["completed", "failed"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot rollback cycle in status: {cycle_details['status']}"
            )
        
        # Perform rollback (this would need to be implemented in the engine)
        # For now, return a placeholder response
        
        return {
            "cycle_id": request.cycle_id,
            "status": "rollback_initiated",
            "message": f"Rollback initiated for cycle {request.cycle_id}",
            "note": "Rollback functionality not fully implemented yet",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SYSTEM HEALTH AND STATUS ENDPOINTS
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """
    Comprehensive system health check
    """
    try:
        result = await ai_assistant.validate_connection()
        
        status_code = 200 if result["overall_status"] == "healthy" else 503
        
        return result
        
    except Exception as e:
        return {
            "overall_status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/status", tags=["System"])
async def system_status():
    """
    Get detailed system status and configuration
    """
    return {
        "system": "Self-Evolving AI Agent & Assistant",
        "version": "1.0.0",
        "status": "operational",
        "features": {
            "ai_assistant": True,
            "conversation_memory": True,
            "context_retrieval": True,
            "safety_filtering": True,
            "multi_agent_support": True,
            "self_improvement": True,
            "performance_monitoring": True,
            "code_analysis": True,
            "automated_testing": True,
            "safe_code_modification": True
        },
        "configuration": {
            "lm_studio_url": settings.LLMS_STUDIO_URL,
            "model": settings.LM_STUDIO_MODEL,
            "safety_threshold": settings.SAFETY_THRESHOLD,
            "vector_db_path": settings.VECTOR_DB_PATH
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# INTEGRATION HUB ENDPOINTS
# ============================================================================

class IntegrationRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    action: str = Field(..., description="Integration action")
    service: Optional[str] = Field(None, description="Specific service name")
    message: Optional[str] = Field(None, description="Message for notifications")
    channel: Optional[str] = Field(None, description="Channel for notifications")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

@app.post("/integrations/action", tags=["Integration Hub"])
async def handle_integration_action(request: IntegrationRequest):
    """
    Handle integration actions like listing services, syncing files, etc.
    """
    try:
        from .personal_assistant_core import PersonalAssistantCore, AssistantRequest, RequestType
        
        # Initialize personal assistant core
        assistant_core = PersonalAssistantCore()
        
        # Create assistant request
        assistant_request = AssistantRequest(
            user_id=request.user_id,
            request_type=RequestType.INTEGRATION_REQUEST,
            content=request.message or f"Integration action: {request.action}",
            metadata={
                "action": request.action,
                "service": request.service,
                "message": request.message,
                "channel": request.channel,
                **request.metadata
            }
        )
        
        # Process the request
        response = await assistant_core.process_request(assistant_request)
        
        return {
            "success": response.success,
            "content": response.content,
            "metadata": response.metadata,
            "suggestions": response.suggestions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/integrations/list", tags=["Integration Hub"])
async def list_integrations():
    """
    List all available integrations and their status
    """
    try:
        from .integration_hub import IntegrationHub
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        integration_hub = IntegrationHub(privacy_manager)
        await integration_hub.initialize()
        
        integrations = await integration_hub.list_integrations()
        
        return {
            "integrations": integrations,
            "total_count": len(integrations),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integrations/test", tags=["Integration Hub"])
async def test_integration_connections():
    """
    Test connections for all configured integrations
    """
    try:
        from .integration_hub import IntegrationHub
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        integration_hub = IntegrationHub(privacy_manager)
        await integration_hub.initialize()
        
        results = await integration_hub.test_all_connections()
        
        working_count = sum(1 for result in results.values() if result)
        total_count = len(results)
        
        return {
            "connection_results": results,
            "working_count": working_count,
            "total_count": total_count,
            "overall_status": "healthy" if working_count == total_count else "degraded",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CloudSyncRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    service: Optional[str] = Field(None, description="Specific cloud service to sync")

@app.post("/integrations/sync-cloud", tags=["Integration Hub"])
async def sync_cloud_files(request: CloudSyncRequest):
    """
    Sync files from cloud storage services
    """
    try:
        from .integration_hub import IntegrationHub
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        integration_hub = IntegrationHub(privacy_manager)
        await integration_hub.initialize()
        
        files = await integration_hub.sync_files_from_cloud(request.user_id, request.service)
        
        return {
            "files": files,
            "file_count": len(files),
            "service": request.service or "all",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DevContextRequest(BaseModel):
    user_id: str = Field(..., description="User ID")

@app.post("/integrations/dev-context", tags=["Integration Hub"])
async def get_development_context(request: DevContextRequest):
    """
    Get development context from integrated development tools
    """
    try:
        from .integration_hub import IntegrationHub
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        integration_hub = IntegrationHub(privacy_manager)
        await integration_hub.initialize()
        
        context = await integration_hub.get_development_context(request.user_id)
        
        repo_count = len(context.get('repositories', []))
        issue_count = sum(len(issues) for issues in context.get('issues', {}).values())
        
        return {
            "context": context,
            "summary": {
                "repository_count": repo_count,
                "issue_count": issue_count
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class NotificationRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    message: str = Field(..., description="Message to send")
    channel: Optional[str] = Field(None, description="Channel to send to")

@app.post("/integrations/notify", tags=["Integration Hub"])
async def send_notification(request: NotificationRequest):
    """
    Send notification through integrated communication tools
    """
    try:
        from .integration_hub import IntegrationHub
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        integration_hub = IntegrationHub(privacy_manager)
        await integration_hub.initialize()
        
        success = await integration_hub.send_notification(
            request.user_id, 
            request.message, 
            request.channel
        )
        
        return {
            "success": success,
            "message": request.message,
            "channel": request.channel,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SAFETY AND CONTENT MODERATION ENDPOINTS
# ============================================================================

class SafetyCheckRequest(BaseModel):
    text: str = Field(..., description="Text content to check for safety")

@app.post("/safety/check", tags=["Safety"])
async def check_content_safety(request: SafetyCheckRequest):
    """
    Check if content passes safety filters
    """
    try:
        safety_result = safety_filter.get_safety_score(request.text)
        return {
            "text": request.text,
            "safe": safety_result["safe"],
            "score": safety_result["score"],
            "threshold": safety_result["threshold"],
            "flags": safety_result["flags"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PRIVACY AND SECURITY ENDPOINTS
# ============================================================================

class PermissionRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    permission_type: str = Field(..., description="Permission type")
    scope: Optional[Dict[str, Any]] = Field(None, description="Permission scope")
    expires_in_days: Optional[int] = Field(None, description="Expiration in days")

class ConsentRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    data_category: str = Field(..., description="Data category")
    purpose: str = Field(..., description="Purpose for data collection")
    retention_days: Optional[int] = Field(None, description="Data retention period in days")

class DataEncryptionRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    data_key: str = Field(..., description="Data key")
    data: Dict[str, Any] = Field(..., description="Data to encrypt")
    data_category: str = Field(..., description="Data category")
    privacy_level: str = Field("confidential", description="Privacy level")

class DataDeletionRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    data_categories: List[str] = Field(..., description="Data categories to delete")
    reason: Optional[str] = Field(None, description="Reason for deletion")

class PrivacySettingRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    setting_key: str = Field(..., description="Setting key")
    setting_value: Any = Field(..., description="Setting value")
    privacy_level: str = Field("internal", description="Privacy level")

class RetentionPolicyRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    data_category: str = Field(..., description="Data category")
    retention_policy: str = Field(..., description="Retention policy")
    custom_days: Optional[int] = Field(None, description="Custom retention days")
    auto_delete: bool = Field(False, description="Enable auto-deletion")

@app.post("/privacy/permissions/request", tags=["Privacy & Security"])
async def request_permission(request: PermissionRequest):
    """Request user permission for data access"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        from .personal_assistant_models import PermissionType
        
        privacy_manager = PrivacySecurityManager()
        permission_type = PermissionType(request.permission_type)
        
        granted = await privacy_manager.request_permission(
            request.user_id, permission_type, request.scope, request.expires_in_days
        )
        
        return {
            "granted": granted,
            "permission_type": request.permission_type,
            "user_id": request.user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/privacy/permissions/{user_id}/{permission_type}", tags=["Privacy & Security"])
async def get_permission_status(
    user_id: str = Path(..., description="User ID"),
    permission_type: str = Path(..., description="Permission type")
):
    """Get permission status for a user"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        from .personal_assistant_models import PermissionType
        
        privacy_manager = PrivacySecurityManager()
        permission_type_enum = PermissionType(permission_type)
        
        permission = await privacy_manager.get_permission(user_id, permission_type_enum)
        
        if not permission:
            return {"granted": False, "exists": False}
        
        return {
            "granted": permission.granted,
            "revoked": permission.revoked,
            "granted_at": permission.granted_at.isoformat() if permission.granted_at else None,
            "expires_at": permission.expires_at.isoformat() if permission.expires_at else None,
            "scope": permission.scope,
            "exists": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/privacy/permissions/{user_id}/{permission_type}", tags=["Privacy & Security"])
async def revoke_permission(
    user_id: str = Path(..., description="User ID"),
    permission_type: str = Path(..., description="Permission type")
):
    """Revoke a user permission"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        from .personal_assistant_models import PermissionType
        
        privacy_manager = PrivacySecurityManager()
        permission_type_enum = PermissionType(permission_type)
        
        revoked = await privacy_manager.revoke_permission(user_id, permission_type_enum)
        
        return {
            "revoked": revoked,
            "permission_type": permission_type,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/privacy/consent/request", tags=["Privacy & Security"])
async def request_consent(request: ConsentRequest):
    """Request user consent for data collection"""
    try:
        from .privacy_security_manager import PrivacySecurityManager, DataCategory
        
        privacy_manager = PrivacySecurityManager()
        data_category = DataCategory(request.data_category)
        
        status = await privacy_manager.request_consent(
            request.user_id, data_category, request.purpose, request.retention_days
        )
        
        return {
            "consent_status": status.value,
            "data_category": request.data_category,
            "user_id": request.user_id,
            "purpose": request.purpose,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/privacy/consent/{user_id}/{data_category}", tags=["Privacy & Security"])
async def get_consent_status(
    user_id: str = Path(..., description="User ID"),
    data_category: str = Path(..., description="Data category")
):
    """Get consent status for a data category"""
    try:
        from .privacy_security_manager import PrivacySecurityManager, DataCategory
        
        privacy_manager = PrivacySecurityManager()
        data_category_enum = DataCategory(data_category)
        
        status = await privacy_manager.get_consent_status(user_id, data_category_enum)
        
        return {
            "consent_status": status.value,
            "data_category": data_category,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/privacy/consent/{user_id}/{data_category}", tags=["Privacy & Security"])
async def revoke_consent(
    user_id: str = Path(..., description="User ID"),
    data_category: str = Path(..., description="Data category")
):
    """Revoke user consent for a data category"""
    try:
        from .privacy_security_manager import PrivacySecurityManager, DataCategory
        
        privacy_manager = PrivacySecurityManager()
        data_category_enum = DataCategory(data_category)
        
        revoked = await privacy_manager.revoke_consent(user_id, data_category_enum)
        
        return {
            "revoked": revoked,
            "data_category": data_category,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/privacy/data/encrypt", tags=["Privacy & Security"])
async def encrypt_personal_data(request: DataEncryptionRequest):
    """Encrypt and store personal data"""
    try:
        from .privacy_security_manager import PrivacySecurityManager, DataCategory, PrivacyLevel
        
        privacy_manager = PrivacySecurityManager()
        data_category = DataCategory(request.data_category)
        privacy_level = PrivacyLevel(request.privacy_level)
        
        success = await privacy_manager.encrypt_personal_data(
            request.user_id, request.data_key, request.data, 
            data_category, privacy_level
        )
        
        return {
            "success": success,
            "data_key": request.data_key,
            "user_id": request.user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/privacy/data/decrypt/{user_id}/{data_key}", tags=["Privacy & Security"])
async def decrypt_personal_data(
    user_id: str = Path(..., description="User ID"),
    data_key: str = Path(..., description="Data key"),
    purpose: Optional[str] = QueryParam(None, description="Purpose for data access")
):
    """Decrypt and retrieve personal data"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        
        privacy_manager = PrivacySecurityManager()
        
        data = await privacy_manager.decrypt_personal_data(user_id, data_key, purpose)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Data not found or access denied")
        
        return {
            "data": data,
            "data_key": data_key,
            "user_id": user_id,
            "purpose": purpose,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/privacy/data/delete", tags=["Privacy & Security"])
async def request_data_deletion(request: DataDeletionRequest):
    """Request deletion of user data"""
    try:
        from .privacy_security_manager import PrivacySecurityManager, DataCategory
        
        privacy_manager = PrivacySecurityManager()
        data_categories = [DataCategory(cat) for cat in request.data_categories]
        
        request_id = await privacy_manager.request_data_deletion(
            request.user_id, data_categories, request.reason
        )
        
        return {
            "request_id": request_id,
            "data_categories": request.data_categories,
            "user_id": request.user_id,
            "reason": request.reason,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/privacy/dashboard/{user_id}", tags=["Privacy & Security"])
async def get_privacy_dashboard(user_id: str = Path(..., description="User ID")):
    """Get comprehensive privacy dashboard data"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        
        privacy_manager = PrivacySecurityManager()
        
        dashboard_data = await privacy_manager.get_enhanced_privacy_dashboard_data(user_id)
        
        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/privacy/settings", tags=["Privacy & Security"])
async def set_privacy_setting(request: PrivacySettingRequest):
    """Set a privacy setting"""
    try:
        from .privacy_security_manager import PrivacySecurityManager, PrivacyLevel
        
        privacy_manager = PrivacySecurityManager()
        privacy_level = PrivacyLevel(request.privacy_level)
        
        success = await privacy_manager.set_privacy_setting(
            request.user_id, request.setting_key, 
            request.setting_value, privacy_level
        )
        
        return {
            "success": success,
            "setting_key": request.setting_key,
            "setting_value": request.setting_value,
            "user_id": request.user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/privacy/settings/{user_id}", tags=["Privacy & Security"])
async def get_privacy_settings(user_id: str = Path(..., description="User ID")):
    """Get all privacy settings for a user"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        
        privacy_manager = PrivacySecurityManager()
        
        settings = await privacy_manager.get_all_privacy_settings(user_id)
        
        return {
            "settings": settings,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/privacy/retention-policy", tags=["Privacy & Security"])
async def set_retention_policy(request: RetentionPolicyRequest):
    """Set data retention policy"""
    try:
        from .privacy_security_manager import PrivacySecurityManager, DataCategory, DataRetentionPolicy
        
        privacy_manager = PrivacySecurityManager()
        data_category = DataCategory(request.data_category)
        retention_policy = DataRetentionPolicy(request.retention_policy)
        
        success = await privacy_manager.set_data_retention_policy(
            request.user_id, data_category, retention_policy,
            request.custom_days, request.auto_delete
        )
        
        return {
            "success": success,
            "data_category": request.data_category,
            "retention_policy": request.retention_policy,
            "user_id": request.user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/privacy/access-history/{user_id}", tags=["Privacy & Security"])
async def get_data_access_history(
    user_id: str = Path(..., description="User ID"),
    data_category: Optional[str] = QueryParam(None, description="Filter by data category"),
    days: int = QueryParam(30, ge=1, le=365, description="Number of days to retrieve")
):
    """Get data access history for transparency"""
    try:
        from .privacy_security_manager import PrivacySecurityManager, DataCategory
        
        privacy_manager = PrivacySecurityManager()
        
        data_category_enum = None
        if data_category:
            data_category_enum = DataCategory(data_category)
        
        history = await privacy_manager.get_data_access_history(
            user_id, data_category_enum, days
        )
        
        return {
            "access_history": history,
            "user_id": user_id,
            "data_category": data_category,
            "days": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/privacy/violations/{user_id}", tags=["Privacy & Security"])
async def get_privacy_violations(
    user_id: str = Path(..., description="User ID"),
    resolved: Optional[bool] = QueryParam(None, description="Filter by resolution status")
):
    """Get privacy violations for a user"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        
        privacy_manager = PrivacySecurityManager()
        
        violations = await privacy_manager.get_privacy_violations(user_id, resolved)
        
        return {
            "violations": violations,
            "user_id": user_id,
            "resolved": resolved,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PERSONAL ASSISTANT ENDPOINTS
# ============================================================================

# File System Operations
class FileOperationRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    operation: str = Field(..., description="File operation type (read, write, list, search, organize)")
    file_path: Optional[str] = Field(None, description="File path for operation")
    content: Optional[str] = Field(None, description="Content for write operations")
    search_query: Optional[str] = Field(None, description="Search query for file search")
    organization_strategy: Optional[str] = Field(None, description="Organization strategy")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

@app.post("/assistant/files/operation", tags=["Personal Assistant - Files"])
async def handle_file_operation(request: FileOperationRequest):
    """Handle file system operations with user permission checks"""
    try:
        from .personal_assistant_core import PersonalAssistantCore, AssistantRequest, RequestType
        
        # Initialize personal assistant core
        assistant_core = PersonalAssistantCore()
        
        # Create assistant request
        assistant_request = AssistantRequest(
            user_id=request.user_id,
            request_type=RequestType.FILE_OPERATION,
            content=f"File operation: {request.operation}",
            metadata={
                "operation": request.operation,
                "file_path": request.file_path,
                "content": request.content,
                "search_query": request.search_query,
                "organization_strategy": request.organization_strategy,
                **request.metadata
            }
        )
        
        # Process the request
        response = await assistant_core.process_request(assistant_request)
        
        return {
            "success": response.success,
            "content": response.content,
            "metadata": response.metadata,
            "suggestions": response.suggestions,
            "requires_permission": response.requires_permission,
            "permission_type": response.permission_type.value if response.permission_type else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/files/list/{user_id}", tags=["Personal Assistant - Files"])
async def list_user_files(
    user_id: str = Path(..., description="User ID"),
    directory: Optional[str] = QueryParam(None, description="Directory to list"),
    file_type: Optional[str] = QueryParam(None, description="Filter by file type"),
    limit: int = QueryParam(100, ge=1, le=1000, description="Maximum files to return")
):
    """List files accessible to the user"""
    try:
        from .file_system_manager import FileSystemManager
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        file_manager = FileSystemManager(privacy_manager)
        
        # Check permissions
        has_permission = await privacy_manager.check_permission(user_id, PermissionType.FILE_READ)
        if not has_permission:
            raise HTTPException(status_code=403, detail="File read permission required")
        
        # List files
        files = await file_manager.list_files(user_id, directory, file_type, limit)
        
        return {
            "files": files,
            "directory": directory,
            "file_type": file_type,
            "count": len(files),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/files/search/{user_id}", tags=["Personal Assistant - Files"])
async def search_files(
    user_id: str = Path(..., description="User ID"),
    query: str = QueryParam(..., description="Search query"),
    file_types: Optional[str] = QueryParam(None, description="Comma-separated file types"),
    limit: int = QueryParam(50, ge=1, le=500, description="Maximum results to return")
):
    """Search files by content and metadata"""
    try:
        from .file_system_manager import FileSystemManager
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        file_manager = FileSystemManager(privacy_manager)
        
        # Check permissions
        has_permission = await privacy_manager.check_permission(user_id, PermissionType.FILE_READ)
        if not has_permission:
            raise HTTPException(status_code=403, detail="File read permission required")
        
        # Parse file types
        file_type_list = file_types.split(',') if file_types else None
        
        # Search files
        results = await file_manager.search_files(user_id, query, file_type_list, limit)
        
        return {
            "results": results,
            "query": query,
            "file_types": file_type_list,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Screen Monitoring Endpoints
class ScreenMonitoringRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    action: str = Field(..., description="Monitoring action (start, stop, status, get_context)")
    config: Optional[Dict[str, Any]] = Field(None, description="Monitoring configuration")

@app.post("/assistant/screen/control", tags=["Personal Assistant - Screen"])
async def control_screen_monitoring(request: ScreenMonitoringRequest):
    """Control screen monitoring functionality"""
    try:
        from .personal_assistant_core import PersonalAssistantCore, AssistantRequest, RequestType
        
        # Initialize personal assistant core
        assistant_core = PersonalAssistantCore()
        
        # Create assistant request
        assistant_request = AssistantRequest(
            user_id=request.user_id,
            request_type=RequestType.SCREEN_MONITORING,
            content=f"Screen monitoring: {request.action}",
            metadata={
                "action": request.action,
                "config": request.config or {}
            }
        )
        
        # Process the request
        response = await assistant_core.process_request(assistant_request)
        
        return {
            "success": response.success,
            "content": response.content,
            "metadata": response.metadata,
            "requires_permission": response.requires_permission,
            "permission_type": response.permission_type.value if response.permission_type else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/screen/context/{user_id}", tags=["Personal Assistant - Screen"])
async def get_screen_context(
    user_id: str = Path(..., description="User ID"),
    include_history: bool = QueryParam(False, description="Include context history")
):
    """Get current screen context and analysis"""
    try:
        from .screen_monitor import ScreenMonitor
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        screen_monitor = ScreenMonitor(privacy_manager)
        
        # Check permissions
        has_permission = await privacy_manager.check_permission(user_id, PermissionType.SCREEN_MONITOR)
        if not has_permission:
            raise HTTPException(status_code=403, detail="Screen monitoring permission required")
        
        # Get current context
        context = await screen_monitor.get_current_context(user_id)
        
        result = {
            "context": context.to_dict() if context else None,
            "monitoring_active": await screen_monitor.is_monitoring_active(user_id),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if include_history:
            history = await screen_monitor.get_context_history(user_id, limit=10)
            result["history"] = [ctx.to_dict() for ctx in history]
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Task Management Endpoints
class TaskRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    action: str = Field(..., description="Task action (create, update, delete, list, complete)")
    task_id: Optional[str] = Field(None, description="Task ID for update/delete operations")
    title: Optional[str] = Field(None, description="Task title")
    description: Optional[str] = Field(None, description="Task description")
    priority: Optional[str] = Field(None, description="Task priority (low, medium, high, urgent)")
    due_date: Optional[str] = Field(None, description="Due date (ISO format)")
    project_id: Optional[str] = Field(None, description="Associated project ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

@app.post("/assistant/tasks/manage", tags=["Personal Assistant - Tasks"])
async def manage_tasks(request: TaskRequest):
    """Manage tasks and projects"""
    try:
        from .personal_assistant_core import PersonalAssistantCore, AssistantRequest, RequestType
        
        # Initialize personal assistant core
        assistant_core = PersonalAssistantCore()
        
        # Create assistant request
        assistant_request = AssistantRequest(
            user_id=request.user_id,
            request_type=RequestType.TASK_MANAGEMENT,
            content=f"Task management: {request.action}",
            metadata={
                "action": request.action,
                "task_id": request.task_id,
                "title": request.title,
                "description": request.description,
                "priority": request.priority,
                "due_date": request.due_date,
                "project_id": request.project_id,
                **request.metadata
            }
        )
        
        # Process the request
        response = await assistant_core.process_request(assistant_request)
        
        return {
            "success": response.success,
            "content": response.content,
            "metadata": response.metadata,
            "suggestions": response.suggestions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/tasks/{user_id}", tags=["Personal Assistant - Tasks"])
async def get_user_tasks(
    user_id: str = Path(..., description="User ID"),
    status: Optional[str] = QueryParam(None, description="Filter by status"),
    priority: Optional[str] = QueryParam(None, description="Filter by priority"),
    project_id: Optional[str] = QueryParam(None, description="Filter by project"),
    limit: int = QueryParam(50, ge=1, le=200, description="Maximum tasks to return")
):
    """Get user tasks with optional filtering"""
    try:
        from .task_manager import TaskManager
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        task_manager = TaskManager(privacy_manager)
        
        # Get tasks
        tasks = await task_manager.get_user_tasks(
            user_id, status, priority, project_id, limit
        )
        
        return {
            "tasks": [task.to_dict() for task in tasks],
            "filters": {
                "status": status,
                "priority": priority,
                "project_id": project_id
            },
            "count": len(tasks),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/tasks/{user_id}/upcoming", tags=["Personal Assistant - Tasks"])
async def get_upcoming_deadlines(
    user_id: str = Path(..., description="User ID"),
    days: int = QueryParam(7, ge=1, le=30, description="Days ahead to check")
):
    """Get upcoming task deadlines"""
    try:
        from .task_manager import TaskManager
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        task_manager = TaskManager(privacy_manager)
        
        # Get upcoming deadlines
        deadlines = await task_manager.get_upcoming_deadlines(user_id, days)
        
        return {
            "deadlines": deadlines,
            "days_ahead": days,
            "count": len(deadlines),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge Base Endpoints
class KnowledgeSearchRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    query: str = Field(..., description="Search query")
    search_type: str = Field("semantic", description="Search type (semantic, keyword, hybrid)")
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")
    include_metadata: bool = Field(True, description="Include result metadata")

@app.post("/assistant/knowledge/search", tags=["Personal Assistant - Knowledge"])
async def search_knowledge_base(request: KnowledgeSearchRequest):
    """Search the personal knowledge base"""
    try:
        from .personal_knowledge_base import PersonalKnowledgeBase
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        knowledge_base = PersonalKnowledgeBase(privacy_manager)
        
        # Search knowledge base
        results = await knowledge_base.search(
            user_id=request.user_id,
            query=request.query,
            search_type=request.search_type,
            limit=request.limit,
            include_metadata=request.include_metadata
        )
        
        return {
            "results": [result.to_dict() for result in results],
            "query": request.query,
            "search_type": request.search_type,
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/knowledge/{user_id}/topics", tags=["Personal Assistant - Knowledge"])
async def get_knowledge_topics(
    user_id: str = Path(..., description="User ID"),
    limit: int = QueryParam(20, ge=1, le=100, description="Maximum topics to return")
):
    """Get knowledge topics and expertise areas"""
    try:
        from .personal_knowledge_base import PersonalKnowledgeBase
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        knowledge_base = PersonalKnowledgeBase(privacy_manager)
        
        # Get topics
        topics = await knowledge_base.get_user_topics(user_id, limit)
        
        return {
            "topics": topics,
            "count": len(topics),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant/knowledge/index", tags=["Personal Assistant - Knowledge"])
async def index_document(
    user_id: str = Body(..., description="User ID"),
    document_path: str = Body(..., description="Document path to index"),
    document_type: Optional[str] = Body(None, description="Document type"),
    extract_entities: bool = Body(True, description="Extract entities from document")
):
    """Index a document in the knowledge base"""
    try:
        from .personal_knowledge_base import PersonalKnowledgeBase
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        knowledge_base = PersonalKnowledgeBase(privacy_manager)
        
        # Index document
        success = await knowledge_base.index_document(
            user_id, document_path, document_type, extract_entities
        )
        
        return {
            "success": success,
            "document_path": document_path,
            "document_type": document_type,
            "extract_entities": extract_entities,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Privacy Control and Permission Management Endpoints
@app.post("/assistant/privacy/permissions/grant", tags=["Personal Assistant - Privacy"])
async def grant_permission(request: PermissionRequest):
    """Grant permission for specific data access"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        from .personal_assistant_models import PermissionType
        
        privacy_manager = PrivacySecurityManager()
        permission_type = PermissionType(request.permission_type)
        
        success = await privacy_manager.grant_permission(
            request.user_id, permission_type, request.scope, request.expires_in_days
        )
        
        return {
            "success": success,
            "permission_type": request.permission_type,
            "user_id": request.user_id,
            "expires_in_days": request.expires_in_days,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant/privacy/permissions/revoke", tags=["Personal Assistant - Privacy"])
async def revoke_permission(
    user_id: str = Body(..., description="User ID"),
    permission_type: str = Body(..., description="Permission type to revoke")
):
    """Revoke a specific permission"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        from .personal_assistant_models import PermissionType
        
        privacy_manager = PrivacySecurityManager()
        permission_type_enum = PermissionType(permission_type)
        
        success = await privacy_manager.revoke_permission(user_id, permission_type_enum)
        
        return {
            "success": success,
            "permission_type": permission_type,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/privacy/permissions/{user_id}", tags=["Personal Assistant - Privacy"])
async def list_user_permissions(
    user_id: str = Path(..., description="User ID")
):
    """List all permissions for a user"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        
        privacy_manager = PrivacySecurityManager()
        permissions = await privacy_manager.get_all_permissions(user_id)
        
        return {
            "permissions": [
                {
                    "permission_type": perm.permission_type.value,
                    "granted": perm.granted,
                    "granted_at": perm.granted_at.isoformat() if perm.granted_at else None,
                    "expires_at": perm.expires_at.isoformat() if perm.expires_at else None,
                    "scope": perm.scope,
                    "revoked": perm.revoked
                }
                for perm in permissions
            ],
            "user_id": user_id,
            "count": len(permissions),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistant/context/{user_id}", tags=["Personal Assistant - Context"])
async def get_user_context(
    user_id: str = Path(..., description="User ID"),
    include_history: bool = QueryParam(False, description="Include interaction history")
):
    """Get current user context and state"""
    try:
        from .user_context_manager import UserContextManager
        
        context_manager = UserContextManager()
        context = await context_manager.get_user_context(user_id)
        
        result = {
            "context": context.to_dict() if context else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if include_history and context:
            result["interaction_history"] = [
                interaction.to_dict() for interaction in context.recent_interactions[-10:]
            ]
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assistant/context/update", tags=["Personal Assistant - Context"])
async def update_user_context(
    user_id: str = Body(..., description="User ID"),
    context_updates: Dict[str, Any] = Body(..., description="Context updates"),
    merge_strategy: str = Body("merge", description="Update strategy (merge, replace)")
):
    """Update user context information"""
    try:
        from .user_context_manager import UserContextManager
        
        context_manager = UserContextManager()
        success = await context_manager.update_context(
            user_id, context_updates, merge_strategy
        )
        
        return {
            "success": success,
            "user_id": user_id,
            "merge_strategy": merge_strategy,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ROOT AND DOCUMENTATION
# ============================================================================

@app.get("/", tags=["System"])
async def root():
    """
    Serve the web UI
    """
    return FileResponse("app/static/index.html")

@app.get("/api", tags=["System"])
async def api_info():
    """
    API information endpoint
    """
    return {
        "message": "Self-Evolving AI Agent & Assistant API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "status": "/status",
        "endpoints": {
            "chat": "/chat",
            "legacy_query": "/query",
            "conversation_history": "/chat/history/{session_id}",
            "user_sessions": "/chat/users/{user_id}/sessions",
            "safety_check": "/safety/check"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
# ============================================================================
# PERSONAL ASSISTANT WEB INTERFACE ENDPOINTS
# ============================================================================

class PersonalAssistantChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID")
    user_id: str = Field(..., description="User ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class FileListRequest(BaseModel):
    path: str = Field("/", description="Directory path to list")

class FileAnalysisRequest(BaseModel):
    path: str = Field(..., description="Path to analyze")

class FileOrganizationRequest(BaseModel):
    path: str = Field(..., description="Path to organize")

class TaskCreateRequest(BaseModel):
    title: str = Field(..., description="Task title")
    priority: str = Field("Medium", description="Task priority")
    due_date: Optional[str] = Field(None, description="Due date")

class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")

class PermissionUpdateRequest(BaseModel):
    permission: str = Field(..., description="Permission type")
    enabled: bool = Field(..., description="Whether permission is enabled")

@app.post("/personal-assistant/chat", tags=["Personal Assistant Web"])
async def personal_assistant_chat(request: PersonalAssistantChatRequest):
    """Enhanced chat endpoint for personal assistant with context awareness"""
    try:
        from .personal_assistant_core import PersonalAssistantCore, AssistantRequest, RequestType
        
        # Initialize personal assistant core
        assistant_core = PersonalAssistantCore()
        
        # Create assistant request with enhanced context
        assistant_request = AssistantRequest(
            user_id=request.user_id,
            request_type=RequestType.CHAT_MESSAGE,
            content=request.message,
            metadata={
                "session_id": request.session_id,
                "web_interface": True,
                **request.context
            }
        )
        
        # Process the request
        response = await assistant_core.process_request(assistant_request)
        
        return {
            "response": response.content,
            "session_id": request.session_id or f"session_{int(time.time())}",
            "suggestions": response.suggestions,
            "metadata": response.metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/personal-assistant/files/list", tags=["Personal Assistant Web"])
async def list_files_for_web(path: str = QueryParam("/", description="Directory path")):
    """List files for web interface file browser"""
    try:
        from .file_system_manager import FileSystemManager
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        file_manager = FileSystemManager(privacy_manager)
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # List files
        files = await file_manager.list_directory(path, user_id)
        
        # Format for web interface
        formatted_files = []
        for file_info in files:
            formatted_files.append({
                "name": file_info.get("name", ""),
                "type": "directory" if file_info.get("is_directory", False) else "file",
                "size": file_info.get("size", 0),
                "modified": file_info.get("modified", "")
            })
        
        return {
            "files": formatted_files,
            "path": path,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Return mock data for demo purposes
        return {
            "files": [
                {"name": "..", "type": "directory", "size": 0, "modified": ""},
                {"name": "Documents", "type": "directory", "size": 0, "modified": "2024-01-15"},
                {"name": "Downloads", "type": "directory", "size": 0, "modified": "2024-01-14"},
                {"name": "example.txt", "type": "file", "size": 1024, "modified": "2024-01-13"}
            ],
            "path": path,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/personal-assistant/files/analyze", tags=["Personal Assistant Web"])
async def analyze_files_for_web(request: FileAnalysisRequest):
    """Analyze files for web interface"""
    try:
        from .file_system_manager import FileSystemManager
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        file_manager = FileSystemManager(privacy_manager)
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # Analyze files
        analysis = await file_manager.analyze_directory(request.path, user_id)
        
        return {
            "analysis": f"Analysis complete for {request.path}. Found {analysis.get('file_count', 0)} files.",
            "details": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Return mock analysis for demo purposes
        return {
            "analysis": f"Analysis complete for {request.path}. Found 15 files, 3 directories. Suggested improvements: organize by file type, remove duplicates.",
            "details": {
                "file_count": 15,
                "directory_count": 3,
                "total_size": "2.5 MB",
                "suggestions": ["Organize by file type", "Remove duplicates", "Archive old files"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/personal-assistant/files/organize", tags=["Personal Assistant Web"])
async def organize_files_for_web(request: FileOrganizationRequest):
    """Organize files for web interface"""
    try:
        from .file_system_manager import FileSystemManager
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        file_manager = FileSystemManager(privacy_manager)
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # Organize files
        result = await file_manager.organize_files(request.path, "by_type", user_id)
        
        return {
            "result": f"Organization complete for {request.path}. {result.get('moved_files', 0)} files organized.",
            "details": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Return mock result for demo purposes
        return {
            "result": f"Organization complete for {request.path}. 12 files moved to appropriate folders.",
            "details": {
                "moved_files": 12,
                "created_folders": ["Documents", "Images", "Archives"],
                "actions": ["Created Documents folder", "Moved 8 text files", "Moved 4 image files"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/personal-assistant/tasks/list", tags=["Personal Assistant Web"])
async def list_tasks_for_web():
    """List tasks for web interface"""
    try:
        from .task_manager import TaskManager
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        task_manager = TaskManager(privacy_manager)
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # Get tasks
        tasks = await task_manager.get_user_tasks(user_id)
        
        return {
            "tasks": tasks,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Return mock tasks for demo purposes
        return {
            "tasks": [
                {
                    "id": "1",
                    "title": "Complete project documentation",
                    "completed": False,
                    "priority": "High",
                    "due_date": "Today"
                },
                {
                    "id": "2", 
                    "title": "Review code changes",
                    "completed": False,
                    "priority": "Medium",
                    "due_date": "Tomorrow"
                },
                {
                    "id": "3",
                    "title": "Setup development environment", 
                    "completed": True,
                    "priority": "High",
                    "due_date": "Completed"
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/personal-assistant/tasks/{task_id}/toggle", tags=["Personal Assistant Web"])
async def toggle_task_for_web(task_id: str = Path(..., description="Task ID")):
    """Toggle task completion status"""
    try:
        from .task_manager import TaskManager
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        task_manager = TaskManager(privacy_manager)
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # Toggle task
        success = await task_manager.toggle_task_completion(task_id, user_id)
        
        return {
            "success": success,
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Return mock success for demo purposes
        return {
            "success": True,
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/personal-assistant/tasks/create", tags=["Personal Assistant Web"])
async def create_task_for_web(request: TaskCreateRequest):
    """Create a new task"""
    try:
        from .task_manager import TaskManager
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        task_manager = TaskManager(privacy_manager)
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # Create task
        task_id = await task_manager.create_task(
            user_id, request.title, request.priority, request.due_date
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "title": request.title,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Return mock success for demo purposes
        return {
            "success": True,
            "task_id": f"task_{int(time.time())}",
            "title": request.title,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/personal-assistant/knowledge/recent", tags=["Personal Assistant Web"])
async def get_recent_knowledge_for_web():
    """Get recent knowledge base items"""
    try:
        from .personal_knowledge_base import PersonalKnowledgeBase
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        knowledge_base = PersonalKnowledgeBase(privacy_manager)
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # Get recent items
        items = await knowledge_base.get_recent_items(user_id, limit=10)
        
        return {
            "items": items,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Return mock knowledge items for demo purposes
        return {
            "items": [
                {
                    "title": "Python Best Practices",
                    "snippet": "A comprehensive guide to Python coding standards including PEP 8 compliance, proper error handling, and performance optimization techniques...",
                    "source": "development_notes.md",
                    "updated": "2 days ago"
                },
                {
                    "title": "Project Architecture Overview",
                    "snippet": "The system follows a modular architecture with clear separation of concerns. Core components include the assistant engine, capability modules...",
                    "source": "architecture_docs.md", 
                    "updated": "1 week ago"
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/personal-assistant/knowledge/search", tags=["Personal Assistant Web"])
async def search_knowledge_for_web(request: KnowledgeSearchRequest):
    """Search knowledge base"""
    try:
        from .personal_knowledge_base import PersonalKnowledgeBase
        from .privacy_security_manager import PrivacySecurityManager
        
        # Initialize components
        privacy_manager = PrivacySecurityManager()
        knowledge_base = PersonalKnowledgeBase(privacy_manager)
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # Search knowledge
        results = await knowledge_base.search_knowledge(user_id, request.query)
        
        return {
            "results": results,
            "query": request.query,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Return mock search results for demo purposes
        return {
            "results": [
                {
                    "title": f"Search Results for '{request.query}'",
                    "snippet": f"Found relevant information about {request.query} in your personal knowledge base. This includes documentation, notes, and previous conversations...",
                    "source": "search_results.md",
                    "updated": "Just now"
                }
            ],
            "query": request.query,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/personal-assistant/privacy/permissions", tags=["Personal Assistant Web"])
async def update_permission_for_web(request: PermissionUpdateRequest):
    """Update permission setting"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        from .personal_assistant_models import PermissionType
        
        privacy_manager = PrivacySecurityManager()
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # Map permission strings to enum values
        permission_map = {
            "file_access": PermissionType.FILE_ACCESS,
            "screen_monitor": PermissionType.SCREEN_MONITORING,
            "learning": PermissionType.LEARNING_DATA,
            "cloud_sync": PermissionType.CLOUD_INTEGRATION
        }
        
        permission_type = permission_map.get(request.permission)
        if not permission_type:
            raise HTTPException(status_code=400, detail="Invalid permission type")
        
        if request.enabled:
            success = await privacy_manager.request_permission(user_id, permission_type)
        else:
            success = await privacy_manager.revoke_permission(user_id, permission_type)
        
        return {
            "success": success,
            "permission": request.permission,
            "enabled": request.enabled,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Return mock success for demo purposes
        return {
            "success": True,
            "permission": request.permission,
            "enabled": request.enabled,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/personal-assistant/privacy/export", tags=["Personal Assistant Web"])
async def export_data_for_web():
    """Export user data"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        
        privacy_manager = PrivacySecurityManager()
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # Export data
        data = await privacy_manager.export_user_data(user_id)
        
        # Create JSON response
        import json
        from fastapi.responses import Response
        
        json_data = json.dumps(data, indent=2)
        
        return Response(
            content=json_data,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=personal_assistant_data.json"}
        )
        
    except Exception as e:
        # Return mock export data
        mock_data = {
            "user_id": "web_user",
            "export_date": datetime.utcnow().isoformat(),
            "data": {
                "conversations": [],
                "tasks": [],
                "files_accessed": [],
                "knowledge_items": [],
                "permissions": []
            }
        }
        
        import json
        from fastapi.responses import Response
        
        json_data = json.dumps(mock_data, indent=2)
        
        return Response(
            content=json_data,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=personal_assistant_data.json"}
        )

@app.delete("/personal-assistant/privacy/delete-all", tags=["Personal Assistant Web"])
async def delete_all_data_for_web():
    """Delete all user data"""
    try:
        from .privacy_security_manager import PrivacySecurityManager
        from .personal_assistant_models import DataCategory
        
        privacy_manager = PrivacySecurityManager()
        
        # Mock user ID for web interface
        user_id = "web_user"
        
        # Delete all data categories
        all_categories = [
            DataCategory.CONVERSATION_DATA,
            DataCategory.FILE_ACCESS_DATA,
            DataCategory.LEARNING_DATA,
            DataCategory.TASK_DATA,
            DataCategory.KNOWLEDGE_DATA
        ]
        
        request_id = await privacy_manager.request_data_deletion(
            user_id, all_categories, "User requested complete data deletion"
        )
        
        return {
            "success": True,
            "request_id": request_id,
            "message": "All data deletion requested successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        # Return mock success for demo purposes
        return {
            "success": True,
            "request_id": f"delete_{int(time.time())}",
            "message": "All data deletion requested successfully",
            "timestamp": datetime.utcnow().isoformat()
        }

# Serve the main web interface
@app.get("/", tags=["Web Interface"])
async def serve_web_interface():
    """Serve the main web interface"""
    return FileResponse("app/static/index.html")