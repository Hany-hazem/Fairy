# app/agent_integration.py
"""
Agent Integration Module

This module initializes and coordinates all agent integrations with the
MCP router and agent registry.
"""

import logging
from typing import Dict, Any

from .agent_mcp_router import agent_mcp_router
from .agent_registry import registry, AgentType

logger = logging.getLogger(__name__)


async def ai_assistant_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for AI assistant agent messages"""
    try:
        # Import here to avoid circular imports
        from agents.ai_assistant_agent import ai_assistant_agent
        
        action = payload.get("action", "process_request")
        
        if action == "process_request":
            return await ai_assistant_agent.process_request(payload)
        elif action == "get_history":
            return await ai_assistant_agent.get_conversation_history(payload)
        elif action == "clear_session":
            return await ai_assistant_agent.clear_session(payload)
        elif action == "get_user_sessions":
            return await ai_assistant_agent.get_user_sessions(payload)
        else:
            return {
                "error": f"Unknown action: {action}",
                "agent_type": "ai_assistant",
                "supported_actions": ["process_request", "get_history", "clear_session", "get_user_sessions"]
            }
            
    except Exception as e:
        logger.error(f"Error in AI assistant handler: {e}")
        return {
            "error": str(e),
            "agent_type": "ai_assistant"
        }


async def self_improvement_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for self-improvement agent messages"""
    try:
        # Import here to avoid circular imports
        from agents.self_improvement_agent import self_improvement_agent
        
        action = payload.get("action", "trigger_improvement")
        
        if action == "trigger_improvement":
            return await self_improvement_agent.trigger_improvement_cycle(payload)
        elif action == "get_status":
            return await self_improvement_agent.get_improvement_status(payload)
        elif action == "analyze_code":
            return await self_improvement_agent.analyze_code_quality(payload)
        elif action == "analyze_performance":
            return await self_improvement_agent.analyze_performance(payload)
        elif action == "approve_improvement":
            return await self_improvement_agent.approve_improvement(payload)
        elif action == "emergency_stop":
            return await self_improvement_agent.emergency_stop(payload)
        else:
            return {
                "error": f"Unknown action: {action}",
                "agent_type": "self_improvement",
                "supported_actions": [
                    "trigger_improvement", "get_status", "analyze_code", 
                    "analyze_performance", "approve_improvement", "emergency_stop"
                ]
            }
            
    except Exception as e:
        logger.error(f"Error in self-improvement handler: {e}")
        return {
            "error": str(e),
            "agent_type": "self_improvement"
        }


async def code_analysis_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for code analysis agent messages"""
    try:
        from app.code_analyzer import code_analyzer
        
        action = payload.get("action", "analyze_code")
        
        if action == "analyze_code":
            file_path = payload.get("file_path")
            
            if file_path:
                # Analyze specific file
                report = code_analyzer.analyze_file(file_path)
                if not report:
                    return {
                        "error": f"Could not analyze file: {file_path}",
                        "agent_type": "code_analysis"
                    }
                
                return {
                    "file_path": file_path,
                    "report": report.to_dict(),
                    "agent_type": "code_analysis"
                }
            else:
                # Analyze entire project
                reports = code_analyzer.analyze_project()
                summary = code_analyzer.get_project_summary(reports)
                
                return {
                    "project_analysis": True,
                    "summary": summary,
                    "file_count": len(reports),
                    "agent_type": "code_analysis"
                }
        else:
            return {
                "error": f"Unknown action: {action}",
                "agent_type": "code_analysis",
                "supported_actions": ["analyze_code"]
            }
            
    except Exception as e:
        logger.error(f"Error in code analysis handler: {e}")
        return {
            "error": str(e),
            "agent_type": "code_analysis"
        }


async def performance_analysis_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for performance analysis agent messages"""
    try:
        from app.performance_analyzer import get_performance_analyzer
        
        action = payload.get("action", "analyze_performance")
        
        if action == "analyze_performance":
            hours = payload.get("hours", 24)
            metric_name = payload.get("metric_name")
            
            analyzer = get_performance_analyzer()
            
            if metric_name:
                # Analyze specific metric
                trend = analyzer.analyze_trends(metric_name, hours)
                if not trend:
                    return {
                        "error": f"Insufficient data for metric: {metric_name}",
                        "agent_type": "performance_analysis"
                    }
                
                return {
                    "metric_name": metric_name,
                    "trend_analysis": {
                        "direction": trend.direction.value,
                        "slope": trend.slope,
                        "confidence": trend.confidence,
                        "recent_avg": trend.recent_avg,
                        "historical_avg": trend.historical_avg,
                        "change_percentage": trend.change_percentage
                    },
                    "agent_type": "performance_analysis"
                }
            else:
                # Generate comprehensive performance report
                report = analyzer.generate_comprehensive_report(hours)
                
                return {
                    "comprehensive_analysis": True,
                    "analysis_period_hours": hours,
                    "summary": report.summary,
                    "trends_count": len(report.trends),
                    "alerts_count": len(report.alerts),
                    "insights_count": len(report.insights),
                    "agent_type": "performance_analysis"
                }
        else:
            return {
                "error": f"Unknown action: {action}",
                "agent_type": "performance_analysis",
                "supported_actions": ["analyze_performance"]
            }
            
    except Exception as e:
        logger.error(f"Error in performance analysis handler: {e}")
        return {
            "error": str(e),
            "agent_type": "performance_analysis"
        }


def vision_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for vision agent messages (legacy compatibility)"""
    try:
        # Import vision agent if available
        try:
            from agents.vision_agent import vision_agent
            # Process vision request
            return vision_agent.process_request(payload)
        except ImportError:
            return {
                "error": "Vision agent not available",
                "agent_type": "vision"
            }
            
    except Exception as e:
        logger.error(f"Error in vision handler: {e}")
        return {
            "error": str(e),
            "agent_type": "vision"
        }


def llm_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for legacy LLM agent messages"""
    try:
        # For legacy LLM requests, we can route them through the AI assistant
        # or handle them with the original LLM adapter
        
        query = payload.get("query", payload.get("text", ""))
        if not query:
            return {
                "error": "No query provided",
                "agent_type": "llm"
            }
        
        # Route to AI assistant for better handling
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        ai_payload = {
            "query": query,
            "action": "process_request"
        }
        
        result = loop.run_until_complete(ai_assistant_handler(ai_payload))
        result["agent_type"] = "llm"  # Override agent type for compatibility
        
        return result
        
    except Exception as e:
        logger.error(f"Error in LLM handler: {e}")
        return {
            "error": str(e),
            "agent_type": "llm"
        }


def initialize_agent_integration():
    """Initialize all agent integrations with MCP router"""
    try:
        if not registry:
            logger.error("Agent registry not available, skipping agent integration")
            return False
        
        # Register handlers for each agent type
        agent_mcp_router.register_agent_handler("ai_assistant", ai_assistant_handler)
        agent_mcp_router.register_agent_handler("self_improvement", self_improvement_handler)
        agent_mcp_router.register_agent_handler("code_analysis", code_analysis_handler)
        agent_mcp_router.register_agent_handler("performance_analysis", performance_analysis_handler)
        agent_mcp_router.register_agent_handler("vision", vision_handler)
        agent_mcp_router.register_agent_handler("llm", llm_handler)
        
        # Start MCP routing
        success = agent_mcp_router.start_routing()
        
        if success:
            logger.info("Agent integration initialized successfully")
            return True
        else:
            logger.error("Failed to start MCP routing")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing agent integration: {e}")
        return False


def get_integration_status() -> Dict[str, Any]:
    """Get status of agent integration"""
    try:
        routing_status = agent_mcp_router.get_routing_status()
        
        # Add registry status
        if registry:
            agents = registry.list_agents()
            registry_status = {
                "available": True,
                "agent_count": len(agents),
                "agent_types": list(set(config.get("type", "unknown") for config in agents.values()))
            }
        else:
            registry_status = {
                "available": False,
                "error": "Agent registry not available"
            }
        
        return {
            "integration_status": "active" if routing_status.get("status") == "running" else "inactive",
            "mcp_router": routing_status,
            "agent_registry": registry_status,
            "timestamp": routing_status.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"Error getting integration status: {e}")
        return {
            "integration_status": "error",
            "error": str(e)
        }