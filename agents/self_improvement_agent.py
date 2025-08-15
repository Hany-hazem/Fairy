# agents/self_improvement_agent.py
"""
Self-Improvement Agent for handling autonomous code improvement requests
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from app.self_improvement_engine import get_self_improvement_engine

logger = logging.getLogger(__name__)


class SelfImprovementAgent:
    """
    Agent wrapper for Self-Improvement Engine to integrate with MCP and agent registry
    """
    
    def __init__(self):
        self.agent_type = "self_improvement"
        self.topic = "self_improvement_tasks"
        logger.info("Self-Improvement Agent initialized")
    
    async def trigger_improvement_cycle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger a new self-improvement cycle
        
        Args:
            payload: Request payload containing trigger type and parameters
            
        Returns:
            Response dictionary with cycle information
        """
        try:
            trigger = payload.get("trigger", "manual")
            safety_level = payload.get("safety_level", "conservative")
            
            logger.info(f"Triggering improvement cycle: trigger={trigger}, safety={safety_level}")
            
            # Get self-improvement engine
            engine = get_self_improvement_engine()
            
            # Trigger improvement cycle
            cycle_id = await engine.trigger_improvement_cycle(trigger)
            
            return {
                "cycle_id": cycle_id,
                "trigger": trigger,
                "safety_level": safety_level,
                "status": "started",
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error triggering improvement cycle: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_improvement_status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of improvement cycles"""
        try:
            cycle_id = payload.get("cycle_id")
            
            engine = get_self_improvement_engine()
            
            if cycle_id:
                # Get specific cycle status
                cycle = engine.get_cycle_status(cycle_id)
                if not cycle:
                    return {
                        "error": f"Cycle {cycle_id} not found",
                        "agent_type": self.agent_type,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                return {
                    "cycle": cycle.to_dict(),
                    "agent_type": self.agent_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Get all active cycles
                active_cycles = engine.get_active_cycles()
                recent_cycles = engine.get_recent_cycles(limit=10)
                
                return {
                    "active_cycles": [cycle.to_dict() for cycle in active_cycles],
                    "recent_cycles": [cycle.to_dict() for cycle in recent_cycles],
                    "agent_type": self.agent_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting improvement status: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def analyze_code_quality(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code quality for improvement opportunities"""
        try:
            file_path = payload.get("file_path")
            
            if file_path:
                # Analyze specific file
                report = code_analyzer.analyze_file(file_path)
                if not report:
                    return {
                        "error": f"Could not analyze file: {file_path}",
                        "agent_type": self.agent_type,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                return {
                    "file_path": file_path,
                    "report": report.to_dict(),
                    "agent_type": self.agent_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # Analyze entire project
                reports = code_analyzer.analyze_project()
                summary = code_analyzer.get_project_summary(reports)
                
                return {
                    "project_analysis": True,
                    "summary": summary,
                    "file_count": len(reports),
                    "agent_type": self.agent_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error analyzing code quality: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def analyze_performance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system performance for improvement opportunities"""
        try:
            hours = payload.get("hours", 24)
            metric_name = payload.get("metric_name")
            
            analyzer = get_performance_analyzer()
            
            if metric_name:
                # Analyze specific metric
                trend = analyzer.analyze_trends(metric_name, hours)
                if not trend:
                    return {
                        "error": f"Insufficient data for metric: {metric_name}",
                        "agent_type": self.agent_type,
                        "timestamp": datetime.utcnow().isoformat()
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
                    "agent_type": self.agent_type,
                    "timestamp": datetime.utcnow().isoformat()
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
                    "agent_type": self.agent_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def approve_improvement(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Approve a pending improvement"""
        try:
            cycle_id = payload.get("cycle_id")
            improvement_id = payload.get("improvement_id")
            approved = payload.get("approved", True)
            
            if not cycle_id or not improvement_id:
                return {
                    "error": "cycle_id and improvement_id are required",
                    "agent_type": self.agent_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            engine = get_self_improvement_engine()
            result = await engine.approve_improvement(cycle_id, improvement_id, approved)
            
            return {
                "cycle_id": cycle_id,
                "improvement_id": improvement_id,
                "approved": approved,
                "result": result,
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error approving improvement: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def emergency_stop(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency stop for all improvement activities"""
        try:
            reason = payload.get("reason", "Manual emergency stop")
            
            engine = get_self_improvement_engine()
            result = await engine.emergency_stop(reason)
            
            return {
                "emergency_stop": True,
                "reason": reason,
                "stopped_cycles": result.get("stopped_cycles", []),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
            return {
                "error": str(e),
                "agent_type": self.agent_type,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def start_mcp_listener(self):
        """Start MCP listener for self-improvement tasks"""
        def handle_message(payload: Dict[str, Any]):
            """Handle incoming MCP messages"""
            try:
                action = payload.get("action", "trigger_improvement")
                
                if action == "trigger_improvement":
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.trigger_improvement_cycle(payload))
                
                elif action == "get_status":
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.get_improvement_status(payload))
                
                elif action == "analyze_code":
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.analyze_code_quality(payload))
                
                elif action == "analyze_performance":
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.analyze_performance(payload))
                
                elif action == "approve_improvement":
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.approve_improvement(payload))
                
                elif action == "emergency_stop":
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.emergency_stop(payload))
                
                else:
                    logger.warning(f"Unknown action: {action}")
                    result = {"error": f"Unknown action: {action}"}
                
                # Publish result back if response_topic provided
                response_topic = payload.get("response_topic")
                if response_topic:
                    mcp.publish(response_topic, {"result": result})
                    
            except Exception as e:
                logger.error(f"Error handling MCP message: {e}")
        
        # Start MCP subscription in background thread
        import threading
        
        def mcp_listener():
            try:
                mcp.subscribe(self.topic, handle_message)
            except Exception as e:
                logger.error(f"MCP listener error: {e}")
        
        thread = threading.Thread(target=mcp_listener, daemon=True)
        thread.start()
        logger.info(f"Started MCP listener for topic: {self.topic}")


# Global Self-Improvement Agent instance
self_improvement_agent = SelfImprovementAgent()