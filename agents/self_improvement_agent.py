# agents/self_improvement_agent.py
"""
Enhanced Self-Improvement Agent with MCP integration for autonomous code improvement
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

from app.self_improvement_engine import get_self_improvement_engine
from app.mcp_client import MCPClient, ClientConfig
from app.mcp_models import MCPMessage, MCPMessageType, AgentContext
from app.agent_context_synchronizer import agent_context_synchronizer, AgentContextType, ContextMergeStrategy

logger = logging.getLogger(__name__)


class SelfImprovementAgent:
    """
    Enhanced Self-Improvement Agent with MCP integration
    
    Provides autonomous code improvement capabilities with context synchronization
    and inter-agent communication through the enhanced MCP system.
    """
    
    def __init__(self):
        self.agent_type = "self_improvement"
        self.agent_id = "self_improvement_agent"
        
        # Define agent capabilities for MCP registration
        self.capabilities = [
            {
                "name": "code_analysis",
                "description": "Analyze code quality and identify improvement opportunities",
                "message_types": ["agent_request", "task_notification"],
                "parameters": {
                    "supports_file_analysis": True,
                    "supports_project_analysis": True,
                    "analysis_types": ["quality", "performance", "security"]
                }
            },
            {
                "name": "autonomous_improvement",
                "description": "Execute autonomous code improvements with safety checks",
                "message_types": ["agent_request", "context_update"],
                "parameters": {
                    "safety_levels": ["conservative", "moderate", "aggressive"],
                    "supports_rollback": True,
                    "requires_approval": True
                }
            },
            {
                "name": "performance_monitoring",
                "description": "Monitor system performance and identify optimization opportunities",
                "message_types": ["agent_request", "context_update"],
                "parameters": {
                    "metrics_tracking": True,
                    "trend_analysis": True,
                    "alert_generation": True
                }
            }
        ]
        
        # Initialize MCP client
        client_config = ClientConfig(
            agent_id=self.agent_id,
            capabilities=self.capabilities,
            heartbeat_interval=30,
            auto_reconnect=True
        )
        self.mcp_client = MCPClient(client_config)
        
        # Register message handlers
        self._register_message_handlers()
        
        # Register context handlers with the agent context synchronizer
        self._register_context_handlers()
        
        logger.info("Enhanced Self-Improvement Agent initialized")
    
    async def start(self) -> bool:
        """
        Start the Self-Improvement Agent and connect to MCP
        
        Returns:
            True if started successfully
        """
        try:
            # Connect to MCP server
            if not await self.mcp_client.connect():
                logger.error("Failed to connect to MCP server")
                return False
            
            # Start agent context synchronizer if not already started
            try:
                await agent_context_synchronizer.start()
            except Exception as e:
                logger.warning(f"Agent context synchronizer already started or failed: {e}")
            
            logger.info("Self-Improvement Agent started and connected to MCP")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Self-Improvement Agent: {e}")
            return False
    
    async def stop(self):
        """Stop the Self-Improvement Agent"""
        try:
            await self.mcp_client.disconnect()
            logger.info("Self-Improvement Agent stopped")
        except Exception as e:
            logger.error(f"Error stopping Self-Improvement Agent: {e}")
    
    async def trigger_improvement_cycle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger a new self-improvement cycle with enhanced context awareness
        
        Args:
            payload: Request payload containing trigger type and parameters
            
        Returns:
            Response dictionary with cycle information
        """
        try:
            trigger = payload.get("trigger", "manual")
            safety_level = payload.get("safety_level", "conservative")
            
            logger.info(f"Triggering improvement cycle: trigger={trigger}, safety={safety_level}")
            
            # Get shared context from other agents
            await self._update_context_from_shared_sources()
            
            # Get self-improvement engine
            engine = get_self_improvement_engine()
            
            # Trigger improvement cycle
            cycle_id = await engine.trigger_improvement_cycle(trigger)
            
            # Notify other agents about the improvement cycle
            await self.mcp_client.send_task_notification(
                task_id=cycle_id,
                action="improvement_cycle_started",
                task_data={
                    "trigger": trigger,
                    "safety_level": safety_level,
                    "agent_id": self.agent_id
                }
            )
            
            # Update improvement context
            await self._update_improvement_context(cycle_id, "started", {
                "trigger": trigger,
                "safety_level": safety_level
            })
            
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
    
    def _register_message_handlers(self):
        """Register MCP message handlers"""
        # Register handler for agent requests
        self.mcp_client.register_message_handler(
            MCPMessageType.AGENT_REQUEST.value,
            self._handle_agent_request
        )
        
        # Register handler for task notifications
        self.mcp_client.register_message_handler(
            MCPMessageType.TASK_NOTIFICATION.value,
            self._handle_task_notification
        )
        
        # Register context handlers
        self.mcp_client.register_context_handler(
            "code_analysis",
            self._handle_code_analysis_context
        )
        
        self.mcp_client.register_context_handler(
            "performance_metrics",
            self._handle_performance_context
        )
        
        self.mcp_client.register_context_handler(
            "improvement_cycle",
            self._handle_improvement_context
        )
    
    def _register_context_handlers(self):
        """Register context handlers with the agent context synchronizer"""
        # Register code analysis context handler
        agent_context_synchronizer.register_agent_context_handler(
            agent_id=self.agent_id,
            context_type=AgentContextType.CODE_ANALYSIS.value,
            handler=self._handle_code_analysis_context_sync,
            priority=8,
            auto_resolve_conflicts=True,
            merge_strategy=ContextMergeStrategy.MERGE_RECURSIVE
        )
        
        # Register performance metrics context handler
        agent_context_synchronizer.register_agent_context_handler(
            agent_id=self.agent_id,
            context_type=AgentContextType.PERFORMANCE_METRICS.value,
            handler=self._handle_performance_metrics_context_sync,
            priority=7,
            auto_resolve_conflicts=True,
            merge_strategy=ContextMergeStrategy.LATEST_WINS
        )
        
        # Register improvement cycle context handler
        agent_context_synchronizer.register_agent_context_handler(
            agent_id=self.agent_id,
            context_type=AgentContextType.IMPROVEMENT_CYCLE.value,
            handler=self._handle_improvement_cycle_context_sync,
            priority=9,
            auto_resolve_conflicts=True,
            merge_strategy=ContextMergeStrategy.SOURCE_PRIORITY
        )
        
        # Register system status context handler
        agent_context_synchronizer.register_agent_context_handler(
            agent_id=self.agent_id,
            context_type=AgentContextType.SYSTEM_STATUS.value,
            handler=self._handle_system_status_context_sync,
            priority=6,
            auto_resolve_conflicts=True,
            merge_strategy=ContextMergeStrategy.FIELD_LEVEL_MERGE
        )
    
    async def _handle_agent_request(self, message: MCPMessage):
        """Handle incoming agent requests"""
        try:
            payload = message.payload
            request_type = payload.get("request_type")
            
            if request_type == "trigger_improvement":
                result = await self.trigger_improvement_cycle(payload)
                response = message.create_response(result, self.agent_id)
                await self.mcp_client.send_message(response)
                
            elif request_type == "get_status":
                result = await self.get_improvement_status(payload)
                response = message.create_response(result, self.agent_id)
                await self.mcp_client.send_message(response)
                
            elif request_type == "analyze_code":
                result = await self.analyze_code_quality(payload)
                response = message.create_response(result, self.agent_id)
                await self.mcp_client.send_message(response)
                
            elif request_type == "analyze_performance":
                result = await self.analyze_performance(payload)
                response = message.create_response(result, self.agent_id)
                await self.mcp_client.send_message(response)
                
            elif request_type == "approve_improvement":
                result = await self.approve_improvement(payload)
                response = message.create_response(result, self.agent_id)
                await self.mcp_client.send_message(response)
                
            elif request_type == "emergency_stop":
                result = await self.emergency_stop(payload)
                response = message.create_response(result, self.agent_id)
                await self.mcp_client.send_message(response)
                
            else:
                logger.warning(f"Unknown request type: {request_type}")
                error_response = message.create_error_response(
                    ValueError(f"Unknown request type: {request_type}"),
                    self.agent_id
                )
                await self.mcp_client.send_message(error_response)
                
        except Exception as e:
            logger.error(f"Error handling agent request: {e}")
            error_response = message.create_error_response(e, self.agent_id)
            await self.mcp_client.send_message(error_response)
    
    async def _handle_task_notification(self, message: MCPMessage):
        """Handle task notifications from other agents"""
        try:
            payload = message.payload
            task_id = payload.get("task_id")
            action = payload.get("action")
            task_data = payload.get("task_data", {})
            
            logger.info(f"Received task notification: {task_id}:{action}")
            
            # React to task completions for potential improvement opportunities
            if action == "completed":
                # Analyze completed task for improvement opportunities
                await self._analyze_task_completion(task_id, task_data)
            
            # Update local context with task information
            await self.mcp_client.update_local_context(
                "active_tasks",
                {task_id: {"action": action, "data": task_data, "timestamp": datetime.utcnow().isoformat()}},
                broadcast=False
            )
            
        except Exception as e:
            logger.error(f"Error handling task notification: {e}")
    
    async def _handle_code_analysis_context(self, context: AgentContext):
        """Handle code analysis context updates"""
        try:
            logger.info(f"Received code analysis context from {context.agent_id}")
            
            analysis_data = context.context_data
            
            # Use analysis data to inform improvement decisions
            if "quality_issues" in analysis_data:
                await self._process_quality_issues(analysis_data["quality_issues"])
            
        except Exception as e:
            logger.error(f"Error handling code analysis context: {e}")
    
    async def _handle_performance_context(self, context: AgentContext):
        """Handle performance metrics context updates"""
        try:
            logger.info(f"Received performance context from {context.agent_id}")
            
            performance_data = context.context_data
            
            # Use performance data to identify optimization opportunities
            if "performance_issues" in performance_data:
                await self._process_performance_issues(performance_data["performance_issues"])
            
        except Exception as e:
            logger.error(f"Error handling performance context: {e}")
    
    async def _handle_improvement_context(self, context: AgentContext):
        """Handle improvement cycle context updates"""
        try:
            logger.info(f"Received improvement context from {context.agent_id}")
            
            # Coordinate with other improvement agents if any
            improvement_data = context.context_data
            
        except Exception as e:
            logger.error(f"Error handling improvement context: {e}")
    
    async def _update_context_from_shared_sources(self):
        """Update context from shared sources before processing"""
        try:
            # Get shared code analysis context
            code_context = await self.mcp_client.get_shared_context("code_analysis")
            if code_context:
                # Use code analysis context to inform improvements
                pass
            
            # Get shared performance context
            perf_context = await self.mcp_client.get_shared_context("performance_metrics")
            if perf_context:
                # Use performance context for optimization decisions
                pass
                
        except Exception as e:
            logger.error(f"Error updating context from shared sources: {e}")
    
    async def _update_improvement_context(self, cycle_id: str, status: str, data: Dict[str, Any]):
        """Update and broadcast improvement context"""
        try:
            improvement_data = {
                "cycle_id": cycle_id,
                "status": status,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }
            
            await self.mcp_client.update_local_context(
                "improvement_cycle",
                improvement_data,
                broadcast=True
            )
            
        except Exception as e:
            logger.error(f"Error updating improvement context: {e}")
    
    async def _analyze_task_completion(self, task_id: str, task_data: Dict[str, Any]):
        """Analyze completed task for improvement opportunities"""
        try:
            # This could trigger analysis of the completed task
            # to identify patterns and improvement opportunities
            logger.info(f"Analyzing completed task {task_id} for improvement opportunities")
            
        except Exception as e:
            logger.error(f"Error analyzing task completion: {e}")
    
    async def _process_quality_issues(self, quality_issues: List[Dict[str, Any]]):
        """Process quality issues from code analysis"""
        try:
            # Process quality issues and potentially trigger improvements
            logger.info(f"Processing {len(quality_issues)} quality issues")
            
        except Exception as e:
            logger.error(f"Error processing quality issues: {e}")
    
    async def _process_performance_issues(self, performance_issues: List[Dict[str, Any]]):
        """Process performance issues from monitoring"""
        try:
            # Process performance issues and potentially trigger optimizations
            logger.info(f"Processing {len(performance_issues)} performance issues")
            
        except Exception as e:
            logger.error(f"Error processing performance issues: {e}")
    
    async def _handle_code_analysis_context_sync(self, context: AgentContext):
        """Handle code analysis context synchronization from other agents"""
        try:
            logger.info(f"Syncing code analysis context from {context.agent_id}")
            
            analysis_data = context.context_data
            quality_issues = analysis_data.get("quality_issues", [])
            
            if quality_issues:
                # Process quality issues for potential improvements
                await self._process_quality_issues(quality_issues)
            
            # Update local code analysis understanding
            logger.debug(f"Updated code analysis context with {len(quality_issues)} quality issues")
            
        except Exception as e:
            logger.error(f"Error handling code analysis context sync: {e}")
    
    async def _handle_performance_metrics_context_sync(self, context: AgentContext):
        """Handle performance metrics context synchronization"""
        try:
            logger.info(f"Syncing performance metrics context from {context.agent_id}")
            
            metrics_data = context.context_data
            performance_issues = metrics_data.get("performance_issues", [])
            
            if performance_issues:
                # Process performance issues for optimization opportunities
                await self._process_performance_issues(performance_issues)
            
            # Update local performance understanding
            logger.debug(f"Updated performance metrics context with {len(performance_issues)} issues")
            
        except Exception as e:
            logger.error(f"Error handling performance metrics context sync: {e}")
    
    async def _handle_improvement_cycle_context_sync(self, context: AgentContext):
        """Handle improvement cycle context synchronization"""
        try:
            logger.info(f"Syncing improvement cycle context from {context.agent_id}")
            
            cycle_data = context.context_data
            cycle_id = cycle_data.get("cycle_id")
            status = cycle_data.get("status")
            
            if cycle_id and status:
                # Coordinate with other improvement agents
                logger.debug(f"Updated improvement cycle context: {cycle_id} - {status}")
            
        except Exception as e:
            logger.error(f"Error handling improvement cycle context sync: {e}")
    
    async def _handle_system_status_context_sync(self, context: AgentContext):
        """Handle system status context synchronization"""
        try:
            logger.info(f"Syncing system status context from {context.agent_id}")
            
            status_data = context.context_data
            system_health = status_data.get("system_health", {})
            
            # Use system status to inform improvement decisions
            if system_health:
                logger.debug(f"Updated system status context with health data")
            
        except Exception as e:
            logger.error(f"Error handling system status context sync: {e}")
    
    async def sync_improvement_cycle_context(self, cycle_id: str, status: str, data: Dict[str, Any]):
        """Synchronize improvement cycle context with other agents"""
        try:
            context_data = {
                "cycle_id": cycle_id,
                "status": status,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }
            
            context = AgentContext(
                agent_id=self.agent_id,
                context_type=AgentContextType.IMPROVEMENT_CYCLE.value,
                context_data=context_data,
                access_level="public"
            )
            
            # Synchronize with other agents
            result = await agent_context_synchronizer.sync_agent_context(
                self.agent_id, context
            )
            
            if result.success:
                logger.debug(f"Successfully synchronized improvement cycle context for {cycle_id}")
            else:
                logger.warning(f"Failed to synchronize improvement cycle context: {result.errors}")
            
        except Exception as e:
            logger.error(f"Error synchronizing improvement cycle context: {e}")
    
    async def sync_code_analysis_context(self, analysis_results: Dict[str, Any]):
        """Synchronize code analysis context with other agents"""
        try:
            context_data = {
                "analysis_results": analysis_results,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            }
            
            context = AgentContext(
                agent_id=self.agent_id,
                context_type=AgentContextType.CODE_ANALYSIS.value,
                context_data=context_data,
                access_level="public"
            )
            
            # Synchronize with other agents
            result = await agent_context_synchronizer.sync_agent_context(
                self.agent_id, context
            )
            
            if result.success:
                logger.debug("Successfully synchronized code analysis context")
            else:
                logger.warning(f"Failed to synchronize code analysis context: {result.errors}")
            
        except Exception as e:
            logger.error(f"Error synchronizing code analysis context: {e}")


# Global Self-Improvement Agent instance
self_improvement_agent = SelfImprovementAgent()