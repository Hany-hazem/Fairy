# app/agent_context_synchronizer.py
"""
Agent Context Synchronization System

This module provides enhanced context synchronization specifically designed
for agent-to-agent communication with conflict resolution and context handlers.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from .context_synchronizer import ContextSynchronizer, ContextConflict, ContextResolution, ContextMergeStrategy
from .mcp_models import AgentContext, MCPMessage, MCPMessageType
from .mcp_integration import mcp_integration

logger = logging.getLogger(__name__)


class AgentContextType(Enum):
    """Standard agent context types"""
    CONVERSATION = "conversation"
    TASK_STATE = "task_state"
    USER_SESSION = "user_session"
    PERFORMANCE_METRICS = "performance_metrics"
    CODE_ANALYSIS = "code_analysis"
    IMPROVEMENT_CYCLE = "improvement_cycle"
    SYSTEM_STATUS = "system_status"
    AGENT_CAPABILITIES = "agent_capabilities"


@dataclass
class AgentContextHandler:
    """Agent-specific context handler configuration"""
    agent_id: str
    context_type: str
    handler_function: Callable
    priority: int = 0
    auto_resolve_conflicts: bool = True
    merge_strategy: Optional[ContextMergeStrategy] = None


@dataclass
class ContextSyncResult:
    """Result of context synchronization operation"""
    success: bool
    agent_id: str
    context_type: str
    conflicts_resolved: int
    errors: List[str]
    timestamp: datetime


class AgentContextSynchronizer:
    """
    Enhanced context synchronization for agent communication
    
    Provides agent-specific context synchronization with:
    - Agent-specific context handlers
    - Automatic conflict resolution
    - Context access control
    - Performance optimization
    """
    
    def __init__(self, base_synchronizer: ContextSynchronizer = None):
        self.base_synchronizer = base_synchronizer or ContextSynchronizer()
        
        # Agent-specific handlers and configurations
        self._agent_handlers: Dict[str, Dict[str, AgentContextHandler]] = {}  # agent_id -> {context_type -> handler}
        self._context_access_rules: Dict[str, Dict[str, Set[str]]] = {}  # context_type -> {access_level -> agent_ids}
        self._agent_priorities: Dict[str, int] = {}  # agent_id -> priority
        
        # Context synchronization state
        self._sync_locks: Dict[str, asyncio.Lock] = {}  # context_type -> lock
        self._sync_history: Dict[str, List[ContextSyncResult]] = {}  # agent_id -> sync_history
        
        # Performance tracking
        self._sync_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_sync_retries = 3
        self.sync_timeout = 30
        self.conflict_resolution_timeout = 60
        
        logger.info("Agent Context Synchronizer initialized")
    
    async def start(self):
        """Start the agent context synchronizer"""
        try:
            # Start base synchronizer
            await self.base_synchronizer.start()
            
            # Initialize MCP integration if available
            if mcp_integration._initialized:
                await self._setup_mcp_integration()
            
            logger.info("Agent Context Synchronizer started")
            
        except Exception as e:
            logger.error(f"Failed to start Agent Context Synchronizer: {e}")
            raise
    
    async def stop(self):
        """Stop the agent context synchronizer"""
        try:
            await self.base_synchronizer.stop()
            logger.info("Agent Context Synchronizer stopped")
        except Exception as e:
            logger.error(f"Error stopping Agent Context Synchronizer: {e}")
    
    def register_agent_context_handler(self, agent_id: str, context_type: str, 
                                     handler: Callable, priority: int = 0,
                                     auto_resolve_conflicts: bool = True,
                                     merge_strategy: ContextMergeStrategy = None):
        """
        Register a context handler for a specific agent
        
        Args:
            agent_id: Agent identifier
            context_type: Type of context to handle
            handler: Handler function
            priority: Handler priority (higher = more important)
            auto_resolve_conflicts: Whether to automatically resolve conflicts
            merge_strategy: Preferred merge strategy for conflicts
        """
        try:
            if agent_id not in self._agent_handlers:
                self._agent_handlers[agent_id] = {}
            
            handler_config = AgentContextHandler(
                agent_id=agent_id,
                context_type=context_type,
                handler_function=handler,
                priority=priority,
                auto_resolve_conflicts=auto_resolve_conflicts,
                merge_strategy=merge_strategy
            )
            
            self._agent_handlers[agent_id][context_type] = handler_config
            
            # Set agent priority
            self._agent_priorities[agent_id] = max(
                self._agent_priorities.get(agent_id, 0),
                priority
            )
            
            logger.info(f"Registered context handler for agent {agent_id}, context {context_type}")
            
        except Exception as e:
            logger.error(f"Failed to register context handler: {e}")
    
    def unregister_agent_context_handler(self, agent_id: str, context_type: str):
        """Unregister a context handler for an agent"""
        try:
            if agent_id in self._agent_handlers and context_type in self._agent_handlers[agent_id]:
                del self._agent_handlers[agent_id][context_type]
                
                if not self._agent_handlers[agent_id]:
                    del self._agent_handlers[agent_id]
                    self._agent_priorities.pop(agent_id, None)
                
                logger.info(f"Unregistered context handler for agent {agent_id}, context {context_type}")
            
        except Exception as e:
            logger.error(f"Failed to unregister context handler: {e}")
    
    async def sync_agent_context(self, agent_id: str, context: AgentContext, 
                               target_agents: List[str] = None) -> ContextSyncResult:
        """
        Synchronize context for a specific agent with enhanced conflict resolution
        
        Args:
            agent_id: Source agent ID
            context: Context to synchronize
            target_agents: Specific target agents (None for all subscribers)
            
        Returns:
            Context synchronization result
        """
        try:
            start_time = datetime.utcnow()
            errors = []
            conflicts_resolved = 0
            
            # Get sync lock for this context type
            lock_key = f"{context.context_type}:{agent_id}"
            if lock_key not in self._sync_locks:
                self._sync_locks[lock_key] = asyncio.Lock()
            
            async with self._sync_locks[lock_key]:
                # Validate context access
                if not await self._validate_context_access(agent_id, context):
                    errors.append("Context access validation failed")
                    return ContextSyncResult(
                        success=False,
                        agent_id=agent_id,
                        context_type=context.context_type,
                        conflicts_resolved=0,
                        errors=errors,
                        timestamp=start_time
                    )
                
                # Determine target agents
                if target_agents is None:
                    target_agents = await self._get_context_subscribers(context.context_type)
                
                # Remove source agent from targets
                target_agents = [t for t in target_agents if t != agent_id]
                
                # Process each target agent
                for target_agent in target_agents:
                    try:
                        # Check for conflicts
                        conflicts = await self._detect_agent_context_conflicts(
                            context, target_agent
                        )
                        
                        if conflicts:
                            # Attempt to resolve conflicts
                            for conflict in conflicts:
                                resolution = await self._resolve_agent_context_conflict(
                                    conflict, agent_id, target_agent
                                )
                                
                                if resolution.resolved:
                                    conflicts_resolved += 1
                                    # Use resolved context
                                    context = resolution.merged_context
                                else:
                                    errors.append(f"Failed to resolve conflict for agent {target_agent}")
                        
                        # Synchronize context with target agent
                        success = await self.base_synchronizer.sync_agent_context(
                            target_agent, context
                        )
                        
                        if success:
                            # Call agent-specific handler if registered
                            await self._call_agent_context_handler(
                                target_agent, context
                            )
                        else:
                            errors.append(f"Failed to sync context with agent {target_agent}")
                    
                    except Exception as e:
                        logger.error(f"Error syncing context with agent {target_agent}: {e}")
                        errors.append(f"Error with agent {target_agent}: {str(e)}")
                
                # Update sync metrics
                await self._update_sync_metrics(agent_id, context.context_type, 
                                              len(target_agents), conflicts_resolved, len(errors))
                
                # Create result
                result = ContextSyncResult(
                    success=len(errors) == 0,
                    agent_id=agent_id,
                    context_type=context.context_type,
                    conflicts_resolved=conflicts_resolved,
                    errors=errors,
                    timestamp=start_time
                )
                
                # Store sync history
                await self._store_sync_result(result)
                
                return result
        
        except Exception as e:
            logger.error(f"Failed to sync agent context: {e}")
            return ContextSyncResult(
                success=False,
                agent_id=agent_id,
                context_type=context.context_type,
                conflicts_resolved=0,
                errors=[str(e)],
                timestamp=datetime.utcnow()
            )
    
    async def handle_context_update(self, agent_id: str, context_update: AgentContext) -> bool:
        """
        Handle incoming context update for an agent
        
        Args:
            agent_id: Target agent ID
            context_update: Context update to handle
            
        Returns:
            True if handled successfully
        """
        try:
            # Check if agent has a registered handler
            if (agent_id in self._agent_handlers and 
                context_update.context_type in self._agent_handlers[agent_id]):
                
                handler_config = self._agent_handlers[agent_id][context_update.context_type]
                
                # Check for conflicts if auto-resolution is enabled
                if handler_config.auto_resolve_conflicts:
                    conflicts = await self._detect_agent_context_conflicts(
                        context_update, agent_id
                    )
                    
                    if conflicts:
                        for conflict in conflicts:
                            resolution = await self._resolve_agent_context_conflict(
                                conflict, context_update.agent_id, agent_id
                            )
                            
                            if resolution.resolved:
                                context_update = resolution.merged_context
                            else:
                                logger.warning(f"Could not resolve context conflict for agent {agent_id}")
                                return False
                
                # Call the handler
                await handler_config.handler_function(context_update)
                
                logger.debug(f"Handled context update {context_update.context_type} for agent {agent_id}")
                return True
            
            else:
                # No specific handler, use default processing
                logger.debug(f"No specific handler for context {context_update.context_type} in agent {agent_id}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to handle context update for agent {agent_id}: {e}")
            return False
    
    async def resolve_context_conflicts_for_agents(self, context_type: str, 
                                                 involved_agents: List[str]) -> List[ContextResolution]:
        """
        Resolve context conflicts between multiple agents
        
        Args:
            context_type: Type of context with conflicts
            involved_agents: List of agents involved in conflicts
            
        Returns:
            List of context resolutions
        """
        try:
            resolutions = []
            
            # Get all contexts of this type from involved agents
            agent_contexts = {}
            for agent_id in involved_agents:
                context = await self.base_synchronizer.get_shared_context(agent_id, context_type)
                if context:
                    agent_contexts[agent_id] = context
            
            if len(agent_contexts) < 2:
                return resolutions  # No conflicts to resolve
            
            # Detect conflicts between all pairs
            conflicts = []
            agent_ids = list(agent_contexts.keys())
            
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    agent1, agent2 = agent_ids[i], agent_ids[j]
                    context1, context2 = agent_contexts[agent1], agent_contexts[agent2]
                    
                    if context1.version != context2.version or context1.context_data != context2.context_data:
                        conflict = ContextConflict(
                            conflict_id=str(uuid.uuid4()),
                            conflict_type=self._determine_conflict_type(context1, context2),
                            context_type=context_type,
                            agent_id=agent1,
                            conflicting_versions=[context1.version, context2.version],
                            local_context=context1,
                            remote_context=context2,
                            detected_at=datetime.utcnow()
                        )
                        conflicts.append(conflict)
            
            # Resolve each conflict
            for conflict in conflicts:
                resolution = await self._resolve_agent_context_conflict(
                    conflict, conflict.local_context.agent_id, conflict.agent_id
                )
                resolutions.append(resolution)
                
                # If resolved, update contexts for all involved agents
                if resolution.resolved and resolution.merged_context:
                    for agent_id in involved_agents:
                        await self.sync_agent_context(agent_id, resolution.merged_context, [agent_id])
            
            return resolutions
        
        except Exception as e:
            logger.error(f"Failed to resolve context conflicts for agents: {e}")
            return []
    
    def set_context_access_rules(self, context_type: str, access_rules: Dict[str, List[str]]):
        """
        Set access rules for a context type
        
        Args:
            context_type: Type of context
            access_rules: Dictionary mapping access levels to agent IDs
        """
        try:
            if context_type not in self._context_access_rules:
                self._context_access_rules[context_type] = {}
            
            for access_level, agent_ids in access_rules.items():
                self._context_access_rules[context_type][access_level] = set(agent_ids)
            
            logger.info(f"Set access rules for context type {context_type}")
            
        except Exception as e:
            logger.error(f"Failed to set context access rules: {e}")
    
    async def get_agent_sync_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get synchronization metrics for an agent"""
        try:
            return self._sync_metrics.get(agent_id, {})
        except Exception as e:
            logger.error(f"Failed to get sync metrics for agent {agent_id}: {e}")
            return {}
    
    async def get_agent_sync_history(self, agent_id: str, limit: int = 50) -> List[ContextSyncResult]:
        """Get synchronization history for an agent"""
        try:
            history = self._sync_history.get(agent_id, [])
            return history[-limit:] if limit > 0 else history
        except Exception as e:
            logger.error(f"Failed to get sync history for agent {agent_id}: {e}")
            return []
    
    async def _setup_mcp_integration(self):
        """Setup integration with MCP system"""
        try:
            # Register message handlers for context synchronization
            mcp_integration.register_message_type_handler(
                MCPMessageType.CONTEXT_UPDATE.value,
                self._handle_mcp_context_update
            )
            
            logger.info("MCP integration setup completed for agent context synchronizer")
            
        except Exception as e:
            logger.error(f"Failed to setup MCP integration: {e}")
    
    async def _handle_mcp_context_update(self, message: MCPMessage):
        """Handle MCP context update messages"""
        try:
            payload = message.payload
            context_data = payload.get("context_data", {})
            context_type = payload.get("context_type")
            
            if not context_type or not context_data:
                logger.warning(f"Invalid context update message {message.id}")
                return
            
            # Create context object
            context = AgentContext(
                agent_id=message.source_agent,
                context_type=context_type,
                context_data=context_data,
                version=payload.get("context_version", str(uuid.uuid4())),
                access_level=payload.get("access_level", "public"),
                metadata=payload.get("metadata", {})
            )
            
            # Handle context update for each target agent
            for target_agent in message.target_agents:
                await self.handle_context_update(target_agent, context)
            
        except Exception as e:
            logger.error(f"Error handling MCP context update: {e}")
    
    async def _validate_context_access(self, agent_id: str, context: AgentContext) -> bool:
        """Validate that an agent can access/modify a context"""
        try:
            # Check if agent is the owner
            if context.agent_id == agent_id:
                return True
            
            # Check access rules
            context_type = context.context_type
            if context_type in self._context_access_rules:
                access_level = context.access_level
                if access_level in self._context_access_rules[context_type]:
                    return agent_id in self._context_access_rules[context_type][access_level]
            
            # Default to context's own access control
            return context.can_access(agent_id)
            
        except Exception as e:
            logger.error(f"Error validating context access: {e}")
            return False
    
    async def _get_context_subscribers(self, context_type: str) -> List[str]:
        """Get list of agents subscribed to a context type"""
        try:
            # Get subscribers from base synchronizer
            subscribers = []
            if hasattr(self.base_synchronizer, '_context_subscriptions'):
                subscribers = list(self.base_synchronizer._context_subscriptions.get(context_type, set()))
            
            return subscribers
            
        except Exception as e:
            logger.error(f"Error getting context subscribers: {e}")
            return []
    
    async def _detect_agent_context_conflicts(self, context: AgentContext, 
                                            target_agent: str) -> List[ContextConflict]:
        """Detect context conflicts for a specific agent"""
        try:
            conflicts = []
            
            # Get existing context for target agent
            existing_context = await self.base_synchronizer.get_shared_context(
                target_agent, context.context_type
            )
            
            if existing_context:
                conflict = await self.base_synchronizer._detect_context_conflict(
                    context, target_agent
                )
                if conflict:
                    conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error detecting agent context conflicts: {e}")
            return []
    
    async def _resolve_agent_context_conflict(self, conflict: ContextConflict, 
                                            source_agent: str, target_agent: str) -> ContextResolution:
        """Resolve context conflict between specific agents"""
        try:
            # Check if agents have specific merge strategies
            source_handler = self._agent_handlers.get(source_agent, {}).get(conflict.context_type)
            target_handler = self._agent_handlers.get(target_agent, {}).get(conflict.context_type)
            
            # Determine merge strategy based on agent priorities and preferences
            merge_strategy = None
            
            if source_handler and source_handler.merge_strategy:
                merge_strategy = source_handler.merge_strategy
            elif target_handler and target_handler.merge_strategy:
                merge_strategy = target_handler.merge_strategy
            else:
                # Use agent priorities
                source_priority = self._agent_priorities.get(source_agent, 0)
                target_priority = self._agent_priorities.get(target_agent, 0)
                
                if source_priority > target_priority:
                    merge_strategy = ContextMergeStrategy.SOURCE_PRIORITY
                elif target_priority > source_priority:
                    merge_strategy = ContextMergeStrategy.LATEST_WINS
                else:
                    merge_strategy = ContextMergeStrategy.MERGE_RECURSIVE
            
            # Set strategy on conflict
            conflict.resolution_strategy = merge_strategy
            
            # Resolve using base synchronizer
            resolution = await self.base_synchronizer.resolve_context_conflict([conflict])
            
            return resolution
            
        except Exception as e:
            logger.error(f"Error resolving agent context conflict: {e}")
            return ContextResolution(
                conflict_id=conflict.conflict_id,
                resolved=False,
                merged_context=None,
                resolution_strategy=ContextMergeStrategy.MANUAL_RESOLUTION,
                resolution_notes=f"Resolution failed: {str(e)}",
                requires_manual_intervention=True
            )
    
    async def _call_agent_context_handler(self, agent_id: str, context: AgentContext):
        """Call agent-specific context handler if registered"""
        try:
            if (agent_id in self._agent_handlers and 
                context.context_type in self._agent_handlers[agent_id]):
                
                handler_config = self._agent_handlers[agent_id][context.context_type]
                await handler_config.handler_function(context)
            
        except Exception as e:
            logger.error(f"Error calling agent context handler: {e}")
    
    async def _update_sync_metrics(self, agent_id: str, context_type: str, 
                                 target_count: int, conflicts_resolved: int, error_count: int):
        """Update synchronization metrics"""
        try:
            if agent_id not in self._sync_metrics:
                self._sync_metrics[agent_id] = {
                    "total_syncs": 0,
                    "successful_syncs": 0,
                    "conflicts_resolved": 0,
                    "errors": 0,
                    "context_types": {},
                    "last_sync": None
                }
            
            metrics = self._sync_metrics[agent_id]
            metrics["total_syncs"] += 1
            metrics["conflicts_resolved"] += conflicts_resolved
            metrics["errors"] += error_count
            metrics["last_sync"] = datetime.utcnow().isoformat()
            
            if error_count == 0:
                metrics["successful_syncs"] += 1
            
            # Context type specific metrics
            if context_type not in metrics["context_types"]:
                metrics["context_types"][context_type] = {
                    "syncs": 0,
                    "conflicts": 0,
                    "errors": 0
                }
            
            context_metrics = metrics["context_types"][context_type]
            context_metrics["syncs"] += 1
            context_metrics["conflicts"] += conflicts_resolved
            context_metrics["errors"] += error_count
            
        except Exception as e:
            logger.error(f"Error updating sync metrics: {e}")
    
    async def _store_sync_result(self, result: ContextSyncResult):
        """Store synchronization result in history"""
        try:
            if result.agent_id not in self._sync_history:
                self._sync_history[result.agent_id] = []
            
            self._sync_history[result.agent_id].append(result)
            
            # Trim history to reasonable size
            max_history = 100
            if len(self._sync_history[result.agent_id]) > max_history:
                self._sync_history[result.agent_id] = self._sync_history[result.agent_id][-max_history:]
            
        except Exception as e:
            logger.error(f"Error storing sync result: {e}")
    
    def _determine_conflict_type(self, context1: AgentContext, context2: AgentContext) -> str:
        """Determine the type of conflict between two contexts"""
        if context1.version != context2.version:
            return "version_conflict"
        elif context1.context_data != context2.context_data:
            return "data_conflict"
        elif context1.access_level != context2.access_level:
            return "access_conflict"
        else:
            return "timestamp_conflict"


# Global agent context synchronizer instance
agent_context_synchronizer = AgentContextSynchronizer()