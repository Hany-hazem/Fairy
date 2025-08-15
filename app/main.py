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