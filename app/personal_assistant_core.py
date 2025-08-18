"""
Personal Assistant Core

This module contains the main PersonalAssistantCore class that orchestrates
all personal assistant capabilities including request routing, user context
management, and privacy/security controls.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .personal_assistant_models import (
    UserContext, Interaction, InteractionType, PermissionType
)
from .user_context_manager import UserContextManager
from .privacy_security_manager import PrivacySecurityManager, DataCategory, ConsentStatus
from .personal_database import PersonalDatabase
from .file_system_manager import FileSystemManager, FileOperation, FileAccessRequest
from .screen_monitor import ScreenMonitor, MonitoringMode, ScreenContext
from .learning_engine import LearningEngine, BehaviorPattern, UserFeedback
from .task_manager import TaskManager, Task, Project, TaskPriority, TaskStatus
from .personal_knowledge_base import PersonalKnowledgeBase, SearchResult
from .interaction_mode_manager import InteractionModeManager, InteractionMode, ModeTransition
from .voice_processor import VoiceProcessor, VoiceCommand, VoiceResponse, VoiceSettings
from .screen_overlay import ScreenOverlay, OverlayConfig, OverlayType, OverlayPosition
from .text_completion import TextCompletion, TextContext, Completion, CompletionSettings
from .accessibility_manager import AccessibilityManager, AccessibilitySettings
from .integration_hub import IntegrationHub
from .proactive_assistant import ProactiveAssistant, ProactiveOpportunity, OpportunityType

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Types of assistant requests"""
    QUERY = "query"
    COMMAND = "command"
    FILE_OPERATION = "file_operation"
    CONTEXT_UPDATE = "context_update"
    PERMISSION_REQUEST = "permission_request"
    PRIVACY_CONTROL = "privacy_control"
    SCREEN_MONITORING = "screen_monitoring"
    TASK_MANAGEMENT = "task_management"
    KNOWLEDGE_SEARCH = "knowledge_search"
    LEARNING_FEEDBACK = "learning_feedback"
    VOICE_COMMAND = "voice_command"
    TEXT_COMPLETION = "text_completion"
    VISUAL_FEEDBACK = "visual_feedback"
    MODE_SWITCH = "mode_switch"
    ACCESSIBILITY_REQUEST = "accessibility_request"
    INTEGRATION_REQUEST = "integration_request"
    PROACTIVE_ASSISTANCE = "proactive_assistance"


@dataclass
class AssistantRequest:
    """Request to the personal assistant"""
    user_id: str
    request_type: RequestType
    content: str
    metadata: Dict[str, Any]
    session_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AssistantResponse:
    """Response from the personal assistant"""
    content: str
    success: bool
    metadata: Dict[str, Any]
    suggestions: List[str]
    requires_permission: bool = False
    permission_type: Optional[PermissionType] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Suggestion:
    """Proactive suggestion from the assistant"""
    title: str
    description: str
    action_type: str
    confidence: float
    metadata: Dict[str, Any]
    expires_at: Optional[datetime] = None


class PersonalAssistantCore:
    """
    Central orchestrator for all personal assistant capabilities.
    
    This class handles request routing, user context management, privacy controls,
    and coordinates between different assistant modules.
    """
    
    def __init__(self, db_path: str = "personal_assistant.db"):
        self.db = PersonalDatabase(db_path)
        self.context_manager = UserContextManager(db_path)
        self.privacy_manager = PrivacySecurityManager(db_path)
        
        # Initialize capability modules
        self._capability_modules = {}
        self._initialize_capability_modules()
        
        # Initialize Integration Hub
        self.integration_hub = IntegrationHub(self.privacy_manager)
        
        logger.info("Personal Assistant Core initialized with all capability modules")
    
    def _initialize_capability_modules(self):
        """Initialize all capability modules"""
        try:
            # Initialize File System Manager
            self._capability_modules['file_system'] = FileSystemManager(
                privacy_manager=self.privacy_manager,
                context_manager=self.context_manager
            )
            logger.info("FileSystemManager initialized")
            
            # Initialize Learning Engine
            self._capability_modules['learning'] = LearningEngine(
                db_path=self.db.db_path,
                context_manager=self.context_manager
            )
            logger.info("LearningEngine initialized")
            
            # Initialize Task Manager
            self._capability_modules['task_manager'] = TaskManager(
                db_path=self.db.db_path
            )
            logger.info("TaskManager initialized")
            
            # Initialize Multi-Modal Interaction Manager
            self._capability_modules['interaction_manager'] = InteractionModeManager(
                voice_settings=VoiceSettings(),
                completion_settings=CompletionSettings(),
                accessibility_settings=AccessibilitySettings()
            )
            logger.info("InteractionModeManager initialized")
            
            # Initialize Integration Hub
            asyncio.create_task(self.integration_hub.initialize())
            logger.info("IntegrationHub initialized")
            
            # Screen Monitor and Knowledge Base will be initialized per-user
            # as they require user-specific configuration
            
        except Exception as e:
            logger.error(f"Error initializing capability modules: {e}")
            # Continue with limited functionality
    
    async def _get_or_create_user_modules(self, user_id: str) -> Dict[str, Any]:
        """Get or create user-specific modules"""
        user_key = f"user_{user_id}"
        
        if user_key not in self._capability_modules:
            self._capability_modules[user_key] = {}
            
            try:
                # Initialize Screen Monitor for user
                self._capability_modules[user_key]['screen_monitor'] = ScreenMonitor(
                    privacy_manager=self.privacy_manager
                )
                
                # Initialize Personal Knowledge Base for user
                self._capability_modules[user_key]['knowledge_base'] = PersonalKnowledgeBase(
                    user_id=user_id
                )
                
                # Initialize Proactive Assistant for user
                self._capability_modules[user_key]['proactive_assistant'] = ProactiveAssistant(
                    learning_engine=self._capability_modules['learning'],
                    task_manager=self._capability_modules['task_manager'],
                    screen_monitor=self._capability_modules[user_key]['screen_monitor'],
                    knowledge_base=self._capability_modules[user_key]['knowledge_base'],
                    context_manager=self.context_manager
                )
                
                logger.info(f"User-specific modules initialized for user {user_id}")
                
            except Exception as e:
                logger.error(f"Error initializing user modules for {user_id}: {e}")
        
        return self._capability_modules.get(user_key, {})
    
    async def process_request(self, request: AssistantRequest) -> AssistantResponse:
        """
        Process an assistant request with proper routing and security checks.
        
        Args:
            request: The assistant request to process
            
        Returns:
            AssistantResponse with the result
        """
        try:
            # Get user context
            context = await self.context_manager.get_user_context(request.user_id)
            
            # Check permissions based on request type
            permission_check = await self._check_request_permissions(request)
            if not permission_check["allowed"]:
                return AssistantResponse(
                    content=permission_check["message"],
                    success=False,
                    metadata={"permission_required": True},
                    suggestions=[],
                    requires_permission=True,
                    permission_type=permission_check.get("permission_type")
                )
            
            # Route request to appropriate handler
            response = await self._route_request(request, context)
            
            # Record interaction
            interaction = Interaction(
                user_id=request.user_id,
                interaction_type=self._map_request_to_interaction_type(request.request_type),
                content=request.content,
                response=response.content,
                context_data=request.metadata,
                metadata={"request_type": request.request_type.value}
            )
            
            await self.context_manager.add_interaction(request.user_id, interaction)
            
            # Update context based on interaction
            await self._update_context_from_interaction(context, request, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request for user {request.user_id}: {e}")
            return AssistantResponse(
                content=f"I encountered an error processing your request: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Please try again or contact support if the issue persists."]
            )
    
    async def get_context(self, user_id: str) -> UserContext:
        """Get current user context"""
        return await self.context_manager.get_user_context(user_id)
    
    async def learn_from_interaction(self, interaction: Interaction) -> None:
        """Learn from user interaction to improve future responses"""
        # This would integrate with the learning engine when implemented
        # For now, just log the learning opportunity
        logger.info(f"Learning opportunity from interaction {interaction.id}")
        
        # Update user context with learning insights
        context = await self.context_manager.get_user_context(interaction.user_id)
        
        # Simple learning: track interaction patterns
        if interaction.feedback_score is not None:
            # Update preferences based on feedback
            if interaction.feedback_score > 0.7:
                # Positive feedback - reinforce this type of response
                pass
            elif interaction.feedback_score < 0.3:
                # Negative feedback - avoid this type of response
                pass
        
        await self.context_manager.update_user_context(context)
    
    async def suggest_proactive_actions(self, context: UserContext) -> List[Suggestion]:
        """Generate proactive suggestions based on user context"""
        suggestions = []
        
        # Analyze context for suggestion opportunities
        current_time = datetime.now()
        
        # Task-based suggestions
        if context.task_context.current_tasks:
            suggestions.append(Suggestion(
                title="Review Current Tasks",
                description=f"You have {len(context.task_context.current_tasks)} active tasks. Would you like to review them?",
                action_type="task_review",
                confidence=0.8,
                metadata={"task_count": len(context.task_context.current_tasks)}
            ))
        
        # File organization suggestions
        if len(context.current_files) > 10:
            suggestions.append(Suggestion(
                title="Organize Files",
                description="You're working with many files. I can help organize them by project or type.",
                action_type="file_organization",
                confidence=0.6,
                metadata={"file_count": len(context.current_files)}
            ))
        
        # Productivity suggestions based on activity patterns
        if context.task_context.work_session_start:
            session_duration = current_time - context.task_context.work_session_start
            if session_duration.total_seconds() > 7200:  # 2 hours
                suggestions.append(Suggestion(
                    title="Take a Break",
                    description="You've been working for over 2 hours. Consider taking a short break.",
                    action_type="break_reminder",
                    confidence=0.7,
                    metadata={"session_duration": session_duration.total_seconds()}
                ))
        
        # Learning suggestions based on knowledge gaps
        if context.knowledge_state.learning_goals:
            suggestions.append(Suggestion(
                title="Continue Learning",
                description=f"You have {len(context.knowledge_state.learning_goals)} learning goals. Would you like to work on one?",
                action_type="learning_continuation",
                confidence=0.5,
                metadata={"learning_goals": context.knowledge_state.learning_goals}
            ))
        
        return suggestions
    
    async def request_permission(self, user_id: str, permission_type: PermissionType,
                               purpose: str, scope: Optional[Dict[str, Any]] = None) -> bool:
        """Request permission from user for specific functionality"""
        return await self.privacy_manager.request_permission(user_id, permission_type, scope)
    
    async def get_privacy_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get privacy dashboard data for user"""
        return await self.privacy_manager.get_privacy_dashboard_data(user_id)
    
    async def handle_privacy_request(self, user_id: str, request_type: str, 
                                   data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle privacy-related requests (consent, data deletion, etc.)"""
        if request_type == "revoke_consent":
            data_category = DataCategory(data["data_category"])
            success = await self.privacy_manager.revoke_consent(user_id, data_category)
            return {"success": success, "message": "Consent revoked" if success else "Failed to revoke consent"}
        
        elif request_type == "delete_data":
            categories = [DataCategory(cat) for cat in data["categories"]]
            request_id = await self.privacy_manager.request_data_deletion(user_id, categories, data.get("reason"))
            return {"success": True, "request_id": request_id, "message": "Data deletion request submitted"}
        
        elif request_type == "revoke_permission":
            permission_type = PermissionType(data["permission_type"])
            success = await self.privacy_manager.revoke_permission(user_id, permission_type)
            return {"success": success, "message": "Permission revoked" if success else "Failed to revoke permission"}
        
        else:
            return {"success": False, "message": "Unknown privacy request type"}
    
    async def _check_request_permissions(self, request: AssistantRequest) -> Dict[str, Any]:
        """Check if user has granted necessary permissions for the request"""
        required_permission = self._get_required_permission(request.request_type)
        
        if not required_permission:
            return {"allowed": True}
        
        has_permission = await self.privacy_manager.check_permission(
            request.user_id, required_permission
        )
        
        if has_permission:
            return {"allowed": True}
        else:
            return {
                "allowed": False,
                "message": f"This action requires {required_permission.value} permission. Would you like to grant it?",
                "permission_type": required_permission
            }
    
    def _get_required_permission(self, request_type: RequestType) -> Optional[PermissionType]:
        """Get the required permission for a request type"""
        permission_map = {
            RequestType.FILE_OPERATION: PermissionType.FILE_READ,
            RequestType.CONTEXT_UPDATE: PermissionType.PERSONAL_DATA,
            RequestType.SCREEN_MONITORING: PermissionType.SCREEN_MONITOR,
            RequestType.TASK_MANAGEMENT: PermissionType.PERSONAL_DATA,
            RequestType.KNOWLEDGE_SEARCH: PermissionType.PERSONAL_DATA,
            RequestType.LEARNING_FEEDBACK: PermissionType.LEARNING,
            RequestType.INTEGRATION_REQUEST: PermissionType.AUTOMATION,
        }
        return permission_map.get(request_type)
    
    async def _route_request(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Route request to appropriate handler based on type and content"""
        if request.request_type == RequestType.QUERY:
            return await self._handle_query(request, context)
        elif request.request_type == RequestType.COMMAND:
            return await self._handle_command(request, context)
        elif request.request_type == RequestType.FILE_OPERATION:
            return await self._handle_file_operation(request, context)
        elif request.request_type == RequestType.CONTEXT_UPDATE:
            return await self._handle_context_update(request, context)
        elif request.request_type == RequestType.PERMISSION_REQUEST:
            return await self._handle_permission_request(request, context)
        elif request.request_type == RequestType.PRIVACY_CONTROL:
            return await self._handle_privacy_control(request, context)
        elif request.request_type == RequestType.SCREEN_MONITORING:
            return await self._handle_screen_monitoring(request, context)
        elif request.request_type == RequestType.TASK_MANAGEMENT:
            return await self._handle_task_management(request, context)
        elif request.request_type == RequestType.KNOWLEDGE_SEARCH:
            return await self._handle_knowledge_search(request, context)
        elif request.request_type == RequestType.LEARNING_FEEDBACK:
            return await self._handle_learning_feedback(request, context)
        elif request.request_type == RequestType.VOICE_COMMAND:
            return await self._handle_voice_command(request, context)
        elif request.request_type == RequestType.TEXT_COMPLETION:
            return await self._handle_text_completion(request, context)
        elif request.request_type == RequestType.VISUAL_FEEDBACK:
            return await self._handle_visual_feedback(request, context)
        elif request.request_type == RequestType.MODE_SWITCH:
            return await self._handle_mode_switch(request, context)
        elif request.request_type == RequestType.ACCESSIBILITY_REQUEST:
            return await self._handle_accessibility_request(request, context)
        elif request.request_type == RequestType.INTEGRATION_REQUEST:
            return await self._handle_integration_request(request, context)
        elif request.request_type == RequestType.PROACTIVE_ASSISTANCE:
            return await self._handle_proactive_assistance(request, context)
        else:
            return AssistantResponse(
                content="I don't understand that type of request.",
                success=False,
                metadata={},
                suggestions=["Try rephrasing your request or ask for help."]
            )
    
    async def _handle_query(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle general queries with knowledge base integration"""
        try:
            # Try to get relevant knowledge from user's knowledge base
            user_modules = await self._get_or_create_user_modules(request.user_id)
            knowledge_base = user_modules.get('knowledge_base')
            
            relevant_knowledge = []
            if knowledge_base:
                try:
                    search_results = await knowledge_base.search_knowledge(
                        request.content, k=3, min_similarity=0.4
                    )
                    relevant_knowledge = [
                        f"From {result.knowledge_item.source_file}: {result.knowledge_item.summary or result.knowledge_item.content[:150]}..."
                        for result in search_results[:2]
                    ]
                except Exception as e:
                    logger.warning(f"Knowledge search failed: {e}")
            
            # Generate response with context
            response_parts = [f"Regarding your question: {request.content}"]
            
            if relevant_knowledge:
                response_parts.append("\nBased on your documents:")
                response_parts.extend([f"• {knowledge}" for knowledge in relevant_knowledge])
                response_parts.append("\nWould you like me to search for more specific information?")
            else:
                response_parts.append("\nI don't have specific information about this in your knowledge base.")
                response_parts.append("Would you like me to help you find relevant documents or information?")
            
            # Get learning insights for personalized response
            learning_engine = self._capability_modules.get('learning')
            if learning_engine:
                try:
                    patterns = await learning_engine.get_user_behavior_patterns(request.user_id)
                    # Use patterns to personalize response (simplified)
                    if patterns:
                        response_parts.append(f"\nBased on your preferences, I can also help with related tasks.")
                except Exception as e:
                    logger.warning(f"Learning pattern retrieval failed: {e}")
            
            suggestions = []
            if relevant_knowledge:
                suggestions.append("Show me more details from these documents")
                suggestions.append("Search for related information")
            else:
                suggestions.append("Help me find relevant documents")
                suggestions.append("Add this topic to my knowledge base")
            
            return AssistantResponse(
                content="\n".join(response_parts),
                success=True,
                metadata={
                    "query_type": "knowledge_enhanced",
                    "knowledge_results": len(relevant_knowledge),
                    "has_context": len(relevant_knowledge) > 0
                },
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            return AssistantResponse(
                content=f"I understand you're asking: {request.content}. I encountered an issue accessing additional context, but I'm here to help!",
                success=True,
                metadata={"query_type": "fallback", "error": str(e)},
                suggestions=["Would you like me to help with something specific?"]
            )
    
    async def _handle_command(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle commands"""
        # This is a placeholder - would integrate with command processing
        return AssistantResponse(
            content=f"I would execute the command: {request.content}",
            success=True,
            metadata={"command_type": "general"},
            suggestions=[]
        )
    
    async def _handle_file_operation(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle file operations using FileSystemManager"""
        try:
            file_manager = self._capability_modules.get('file_system')
            if not file_manager:
                return AssistantResponse(
                    content="File system manager is not available.",
                    success=False,
                    metadata={},
                    suggestions=["Please try again later."]
                )
            
            # Extract operation details from request
            operation = FileOperation(request.metadata.get('operation', 'read'))
            file_path = request.metadata.get('file_path', '')
            
            if not file_path:
                return AssistantResponse(
                    content="File path is required for file operations.",
                    success=False,
                    metadata={},
                    suggestions=["Please specify a file path."]
                )
            
            # Create file access request
            access_request = FileAccessRequest(
                user_id=request.user_id,
                operation=operation,
                file_path=file_path,
                justification=request.content,
                metadata=request.metadata
            )
            
            # Execute the file operation
            if operation == FileOperation.READ:
                result = await file_manager.read_file(access_request)
            elif operation == FileOperation.WRITE:
                content = request.metadata.get('content', '')
                result = await file_manager.write_file(access_request, content)
            elif operation == FileOperation.LIST:
                result = await file_manager.list_directory(access_request)
            elif operation == FileOperation.SEARCH:
                query = request.metadata.get('query', '')
                result = await file_manager.search_files(access_request, query)
            elif operation == FileOperation.ANALYZE:
                result = await file_manager.analyze_file(access_request)
            elif operation == FileOperation.ORGANIZE:
                result = await file_manager.organize_files(access_request)
            else:
                return AssistantResponse(
                    content=f"File operation '{operation.value}' is not supported yet.",
                    success=False,
                    metadata={},
                    suggestions=["Try read, write, list, search, analyze, or organize operations."]
                )
            
            if result.success:
                response_content = f"File operation '{operation.value}' completed successfully."
                if result.content:
                    response_content += f"\n\nResult: {result.content[:500]}..."
                
                return AssistantResponse(
                    content=response_content,
                    success=True,
                    metadata={
                        "operation": operation.value,
                        "file_path": file_path,
                        "result_metadata": result.metadata
                    },
                    suggestions=[]
                )
            else:
                return AssistantResponse(
                    content=f"File operation failed: {result.error_message}",
                    success=False,
                    metadata={"error": result.error_message},
                    suggestions=["Check file permissions and try again."],
                    requires_permission=result.permission_required
                )
                
        except Exception as e:
            logger.error(f"Error handling file operation: {e}")
            return AssistantResponse(
                content=f"An error occurred during file operation: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Please try again or contact support."]
            )
    
    async def _handle_context_update(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle context updates"""
        # Update context based on request
        if "current_activity" in request.metadata:
            context.current_activity = request.metadata["current_activity"]
        
        if "active_applications" in request.metadata:
            context.active_applications = request.metadata["active_applications"]
        
        await self.context_manager.update_user_context(context)
        
        return AssistantResponse(
            content="Context updated successfully.",
            success=True,
            metadata={"updated_fields": list(request.metadata.keys())},
            suggestions=[]
        )
    
    async def _handle_permission_request(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle permission requests"""
        permission_type = PermissionType(request.metadata["permission_type"])
        granted = await self.privacy_manager.request_permission(
            request.user_id, permission_type, request.metadata.get("scope")
        )
        
        return AssistantResponse(
            content=f"Permission {'granted' if granted else 'denied'} for {permission_type.value}.",
            success=granted,
            metadata={"permission_granted": granted},
            suggestions=[]
        )
    
    async def _handle_privacy_control(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle privacy control requests"""
        result = await self.handle_privacy_request(
            request.user_id, request.metadata["privacy_action"], request.metadata
        )
        
        return AssistantResponse(
            content=result["message"],
            success=result["success"],
            metadata=result,
            suggestions=[]
        )
    
    async def _handle_screen_monitoring(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle screen monitoring requests"""
        try:
            user_modules = await self._get_or_create_user_modules(request.user_id)
            screen_monitor = user_modules.get('screen_monitor')
            
            if not screen_monitor:
                return AssistantResponse(
                    content="Screen monitoring is not available.",
                    success=False,
                    metadata={},
                    suggestions=["Please check system requirements."]
                )
            
            action = request.metadata.get('action', 'get_context')
            
            if action == 'start_monitoring':
                from app.screen_monitor import MonitorConfig
                mode = MonitoringMode(request.metadata.get('mode', 'selective'))
                config = MonitorConfig(mode=mode)
                success = await screen_monitor.start_monitoring(request.user_id, config)
                return AssistantResponse(
                    content=f"Screen monitoring {'started' if success else 'failed to start'}.",
                    success=success,
                    metadata={"monitoring_active": success},
                    suggestions=[]
                )
            
            elif action == 'stop_monitoring':
                await screen_monitor.stop_monitoring(request.user_id)
                return AssistantResponse(
                    content="Screen monitoring stopped.",
                    success=True,
                    metadata={"monitoring_active": False},
                    suggestions=[]
                )
            
            elif action == 'get_context':
                screen_context = await screen_monitor.get_current_context(request.user_id)
                if screen_context:
                    return AssistantResponse(
                        content=f"Current screen context: {screen_context.context_summary}",
                        success=True,
                        metadata={
                            "active_application": screen_context.active_application,
                            "window_title": screen_context.window_title,
                            "application_type": screen_context.application_type.value
                        },
                        suggestions=[]
                    )
                else:
                    return AssistantResponse(
                        content="No screen context available.",
                        success=False,
                        metadata={},
                        suggestions=["Start screen monitoring first."]
                    )
            
            else:
                return AssistantResponse(
                    content=f"Unknown screen monitoring action: {action}",
                    success=False,
                    metadata={},
                    suggestions=["Try 'start_monitoring', 'stop_monitoring', or 'get_context'."]
                )
                
        except Exception as e:
            logger.error(f"Error handling screen monitoring request: {e}")
            return AssistantResponse(
                content=f"Screen monitoring error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Please try again."]
            )
    
    async def _handle_task_management(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle task management requests"""
        try:
            task_manager = self._capability_modules.get('task_manager')
            if not task_manager:
                return AssistantResponse(
                    content="Task manager is not available.",
                    success=False,
                    metadata={},
                    suggestions=["Please try again later."]
                )
            
            action = request.metadata.get('action', 'list_tasks')
            
            if action == 'create_task':
                task_data = request.metadata.get('task_data', {})
                task = Task(
                    title=task_data.get('title', request.content),
                    description=task_data.get('description', ''),
                    priority=TaskPriority(task_data.get('priority', 'medium')),
                    user_id=request.user_id
                )
                
                created_task = await task_manager.create_task(task)
                return AssistantResponse(
                    content=f"Task '{created_task.title}' created successfully.",
                    success=True,
                    metadata={"task_id": created_task.id, "task": created_task.to_dict()},
                    suggestions=["Would you like to set a due date or add more details?"]
                )
            
            elif action == 'list_tasks':
                status_filter = request.metadata.get('status')
                tasks = await task_manager.get_user_tasks(
                    request.user_id, 
                    status=TaskStatus(status_filter) if status_filter else None
                )
                
                if tasks:
                    task_list = "\n".join([f"- {task.title} ({task.status.value})" for task in tasks[:10]])
                    return AssistantResponse(
                        content=f"Your tasks:\n{task_list}",
                        success=True,
                        metadata={"task_count": len(tasks), "tasks": [t.to_dict() for t in tasks[:10]]},
                        suggestions=["Would you like to see details for any specific task?"]
                    )
                else:
                    return AssistantResponse(
                        content="You have no tasks.",
                        success=True,
                        metadata={"task_count": 0},
                        suggestions=["Would you like to create a new task?"]
                    )
            
            elif action == 'update_task':
                task_id = request.metadata.get('task_id')
                updates = request.metadata.get('updates', {})
                
                if not task_id:
                    return AssistantResponse(
                        content="Task ID is required for updates.",
                        success=False,
                        metadata={},
                        suggestions=["Please specify which task to update."]
                    )
                
                success = await task_manager.update_task(task_id, updates)
                return AssistantResponse(
                    content=f"Task {'updated' if success else 'update failed'}.",
                    success=success,
                    metadata={"task_id": task_id},
                    suggestions=[]
                )
            
            elif action == 'get_suggestions':
                suggestions = await task_manager.get_productivity_suggestions(request.user_id)
                if suggestions:
                    suggestion_text = "\n".join([f"- {s['title']}: {s['description']}" for s in suggestions[:5]])
                    return AssistantResponse(
                        content=f"Productivity suggestions:\n{suggestion_text}",
                        success=True,
                        metadata={"suggestions": suggestions},
                        suggestions=[]
                    )
                else:
                    return AssistantResponse(
                        content="No productivity suggestions at this time.",
                        success=True,
                        metadata={},
                        suggestions=[]
                    )
            
            else:
                return AssistantResponse(
                    content=f"Unknown task management action: {action}",
                    success=False,
                    metadata={},
                    suggestions=["Try 'create_task', 'list_tasks', 'update_task', or 'get_suggestions'."]
                )
                
        except Exception as e:
            logger.error(f"Error handling task management request: {e}")
            return AssistantResponse(
                content=f"Task management error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Please try again."]
            )
    
    async def _handle_knowledge_search(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle knowledge base search requests"""
        try:
            user_modules = await self._get_or_create_user_modules(request.user_id)
            knowledge_base = user_modules.get('knowledge_base')
            
            if not knowledge_base:
                return AssistantResponse(
                    content="Knowledge base is not available.",
                    success=False,
                    metadata={},
                    suggestions=["Please try again later."]
                )
            
            action = request.metadata.get('action', 'search')
            
            if action == 'search':
                query = request.content
                k = request.metadata.get('max_results', 5)
                min_similarity = request.metadata.get('min_similarity', 0.3)
                
                results = await knowledge_base.search_knowledge(query, k=k, min_similarity=min_similarity)
                
                if results:
                    response_parts = []
                    for i, result in enumerate(results[:3], 1):
                        response_parts.append(
                            f"{i}. {result.knowledge_item.summary or result.knowledge_item.content[:100]}... "
                            f"(from {result.knowledge_item.source_file})"
                        )
                    
                    response_content = f"Found {len(results)} relevant items:\n" + "\n".join(response_parts)
                    
                    return AssistantResponse(
                        content=response_content,
                        success=True,
                        metadata={
                            "result_count": len(results),
                            "results": [
                                {
                                    "content": r.knowledge_item.content[:200],
                                    "source": r.knowledge_item.source_file,
                                    "similarity": r.similarity_score,
                                    "topics": r.matched_topics
                                } for r in results[:5]
                            ]
                        },
                        suggestions=["Would you like more details about any of these items?"]
                    )
                else:
                    return AssistantResponse(
                        content="No relevant knowledge found for your query.",
                        success=True,
                        metadata={"result_count": 0},
                        suggestions=["Try a different search term or add more documents to your knowledge base."]
                    )
            
            elif action == 'index_document':
                file_path = request.metadata.get('file_path')
                content = request.metadata.get('content', '')
                
                if not file_path:
                    return AssistantResponse(
                        content="File path is required for document indexing.",
                        success=False,
                        metadata={},
                        suggestions=["Please specify a file path."]
                    )
                
                success = await knowledge_base.index_document(file_path, content)
                return AssistantResponse(
                    content=f"Document {'indexed' if success else 'indexing failed'}.",
                    success=success,
                    metadata={"file_path": file_path},
                    suggestions=[]
                )
            
            elif action == 'get_statistics':
                stats = await knowledge_base.get_knowledge_statistics()
                return AssistantResponse(
                    content=f"Knowledge base contains {stats.get('total_items', 0)} items across {stats.get('total_topics', 0)} topics.",
                    success=True,
                    metadata=stats,
                    suggestions=[]
                )
            
            else:
                return AssistantResponse(
                    content=f"Unknown knowledge base action: {action}",
                    success=False,
                    metadata={},
                    suggestions=["Try 'search', 'index_document', or 'get_statistics'."]
                )
                
        except Exception as e:
            logger.error(f"Error handling knowledge search request: {e}")
            return AssistantResponse(
                content=f"Knowledge search error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Please try again."]
            )
    
    async def _handle_learning_feedback(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle learning and feedback requests"""
        try:
            learning_engine = self._capability_modules.get('learning')
            if not learning_engine:
                return AssistantResponse(
                    content="Learning engine is not available.",
                    success=False,
                    metadata={},
                    suggestions=["Please try again later."]
                )
            
            action = request.metadata.get('action', 'provide_feedback')
            
            if action == 'provide_feedback':
                interaction_id = request.metadata.get('interaction_id')
                feedback_type = request.metadata.get('feedback_type', 'rating')
                feedback_value = request.metadata.get('feedback_value')
                
                if not interaction_id or feedback_value is None:
                    return AssistantResponse(
                        content="Interaction ID and feedback value are required.",
                        success=False,
                        metadata={},
                        suggestions=["Please provide feedback for a specific interaction."]
                    )
                
                feedback = UserFeedback(
                    feedback_id=f"feedback_{datetime.now().timestamp()}",
                    user_id=request.user_id,
                    interaction_id=interaction_id,
                    feedback_type=feedback_type,
                    feedback_value=feedback_value,
                    timestamp=datetime.now(),
                    context_data=request.metadata
                )
                
                await learning_engine.process_feedback(feedback)
                return AssistantResponse(
                    content="Thank you for your feedback! I'll use it to improve my responses.",
                    success=True,
                    metadata={"feedback_processed": True},
                    suggestions=[]
                )
            
            elif action == 'get_patterns':
                patterns = await learning_engine.get_user_behavior_patterns(request.user_id)
                if patterns:
                    pattern_summary = f"I've identified {len(patterns)} behavior patterns to better assist you."
                    return AssistantResponse(
                        content=pattern_summary,
                        success=True,
                        metadata={"pattern_count": len(patterns)},
                        suggestions=[]
                    )
                else:
                    return AssistantResponse(
                        content="I'm still learning your preferences. Keep interacting with me to help me understand you better!",
                        success=True,
                        metadata={"pattern_count": 0},
                        suggestions=[]
                    )
            
            elif action == 'adapt_preferences':
                preferences = request.metadata.get('preferences', {})
                success = await learning_engine.update_user_preferences(request.user_id, preferences)
                return AssistantResponse(
                    content=f"Preferences {'updated' if success else 'update failed'}.",
                    success=success,
                    metadata={"preferences_updated": success},
                    suggestions=[]
                )
            
            else:
                return AssistantResponse(
                    content=f"Unknown learning action: {action}",
                    success=False,
                    metadata={},
                    suggestions=["Try 'provide_feedback', 'get_patterns', or 'adapt_preferences'."]
                )
                
        except Exception as e:
            logger.error(f"Error handling learning feedback request: {e}")
            return AssistantResponse(
                content=f"Learning feedback error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Please try again."]
            )
    
    async def _handle_voice_command(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle voice command requests"""
        try:
            interaction_manager = self._capability_modules.get('interaction_manager')
            if not interaction_manager:
                return AssistantResponse(
                    content="Voice interaction is not available.",
                    success=False,
                    metadata={},
                    suggestions=["Try using text input instead."]
                )
            
            # Extract voice command data
            command_text = request.metadata.get('command_text', request.content)
            intent = request.metadata.get('intent', 'general_query')
            entities = request.metadata.get('entities', {})
            confidence = request.metadata.get('confidence', 0.5)
            
            # Process voice command through interaction manager
            interaction_data = {
                "type": "voice_command",
                "command": command_text,
                "intent": intent,
                "entities": entities,
                "confidence": confidence,
                "user_id": request.user_id
            }
            
            result = await interaction_manager.process_interaction(interaction_data)
            
            if result.get("success", False):
                responses = result.get("responses", [])
                response_content = []
                
                for resp in responses:
                    if resp["type"] == "text":
                        response_content.append(resp["content"])
                    elif resp["type"] == "voice":
                        response_content.append(f"[Voice] {resp['content']}")
                    elif resp["type"] == "visual":
                        response_content.append(f"[Visual] {resp['content']}")
                
                return AssistantResponse(
                    content="\n".join(response_content) if response_content else "Voice command processed.",
                    success=True,
                    metadata={"voice_processed": True, "responses": responses},
                    suggestions=[]
                )
            else:
                return AssistantResponse(
                    content=f"Voice command processing failed: {result.get('error', 'Unknown error')}",
                    success=False,
                    metadata={"error": result.get('error')},
                    suggestions=["Try rephrasing your voice command."]
                )
                
        except Exception as e:
            logger.error(f"Error handling voice command: {e}")
            return AssistantResponse(
                content=f"Voice command error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Please try again or use text input."]
            )
    
    async def _handle_text_completion(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle text completion requests"""
        try:
            interaction_manager = self._capability_modules.get('interaction_manager')
            if not interaction_manager:
                return AssistantResponse(
                    content="Text completion is not available.",
                    success=False,
                    metadata={},
                    suggestions=["Try typing without completion assistance."]
                )
            
            # Extract text context
            text_before = request.metadata.get('text_before', '')
            text_after = request.metadata.get('text_after', '')
            cursor_position = request.metadata.get('cursor_position', 0)
            context_type = request.metadata.get('context_type', 'general')
            application = request.metadata.get('application')
            file_type = request.metadata.get('file_type')
            
            # Create text context
            from .text_completion import TextContext, ContextType
            text_context = TextContext(
                text_before=text_before,
                text_after=text_after,
                cursor_position=cursor_position,
                context_type=ContextType(context_type) if context_type in [ct.value for ct in ContextType] else ContextType.GENERAL,
                application=application,
                file_type=file_type,
                metadata={"user_id": request.user_id}
            )
            
            # Get completions
            completions = await interaction_manager.get_text_completions(text_context)
            
            completion_data = []
            for completion in completions:
                completion_data.append({
                    "text": completion.text,
                    "type": completion.completion_type.value,
                    "confidence": completion.confidence,
                    "description": completion.description,
                    "source": completion.source
                })
            
            return AssistantResponse(
                content=f"Found {len(completions)} text completions.",
                success=True,
                metadata={
                    "completions": completion_data,
                    "completion_count": len(completions)
                },
                suggestions=[comp.text for comp in completions[:3]]  # Top 3 as suggestions
            )
            
        except Exception as e:
            logger.error(f"Error handling text completion: {e}")
            return AssistantResponse(
                content=f"Text completion error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Try typing without completion assistance."]
            )
    
    async def _handle_visual_feedback(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle visual feedback requests"""
        try:
            interaction_manager = self._capability_modules.get('interaction_manager')
            if not interaction_manager:
                return AssistantResponse(
                    content="Visual feedback is not available.",
                    success=False,
                    metadata={},
                    suggestions=["Visual feedback requires display capabilities."]
                )
            
            # Extract visual feedback data
            message = request.metadata.get('message', request.content)
            feedback_type = request.metadata.get('feedback_type', 'info')
            duration = request.metadata.get('duration', 3.0)
            position = request.metadata.get('position', 'bottom_right')
            
            # Show visual feedback
            overlay_id = await interaction_manager.show_visual_feedback(message, feedback_type)
            
            if overlay_id:
                return AssistantResponse(
                    content=f"Visual feedback displayed: {message}",
                    success=True,
                    metadata={
                        "overlay_id": overlay_id,
                        "feedback_type": feedback_type,
                        "message": message
                    },
                    suggestions=[]
                )
            else:
                return AssistantResponse(
                    content="Failed to display visual feedback.",
                    success=False,
                    metadata={},
                    suggestions=["Check if visual display is available."]
                )
                
        except Exception as e:
            logger.error(f"Error handling visual feedback: {e}")
            return AssistantResponse(
                content=f"Visual feedback error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Try again or use text feedback."]
            )
    
    async def _handle_mode_switch(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle interaction mode switch requests"""
        try:
            interaction_manager = self._capability_modules.get('interaction_manager')
            if not interaction_manager:
                return AssistantResponse(
                    content="Interaction mode switching is not available.",
                    success=False,
                    metadata={},
                    suggestions=["Mode switching requires multi-modal support."]
                )
            
            # Extract mode switch data
            target_mode = request.metadata.get('target_mode', 'text')
            transition_type = request.metadata.get('transition_type', 'user_initiated')
            preserve_context = request.metadata.get('preserve_context', True)
            
            # Map string to enum
            mode_mapping = {
                'text': InteractionMode.TEXT,
                'voice': InteractionMode.VOICE,
                'visual': InteractionMode.VISUAL,
                'mixed': InteractionMode.MIXED,
                'accessibility': InteractionMode.ACCESSIBILITY
            }
            
            transition_mapping = {
                'user_initiated': ModeTransition.USER_INITIATED,
                'automatic': ModeTransition.AUTOMATIC,
                'context_based': ModeTransition.CONTEXT_BASED,
                'accessibility_required': ModeTransition.ACCESSIBILITY_REQUIRED
            }
            
            if target_mode not in mode_mapping:
                return AssistantResponse(
                    content=f"Unknown interaction mode: {target_mode}",
                    success=False,
                    metadata={},
                    suggestions=["Available modes: text, voice, visual, mixed, accessibility"]
                )
            
            # Switch mode
            success = await interaction_manager.switch_mode(
                mode_mapping[target_mode],
                transition_mapping.get(transition_type, ModeTransition.USER_INITIATED),
                preserve_context
            )
            
            if success:
                current_mode = interaction_manager.get_current_mode()
                return AssistantResponse(
                    content=f"Successfully switched to {current_mode.value} mode.",
                    success=True,
                    metadata={
                        "previous_mode": interaction_manager.context.previous_mode.value if interaction_manager.context.previous_mode else None,
                        "current_mode": current_mode.value,
                        "context_preserved": preserve_context
                    },
                    suggestions=[]
                )
            else:
                return AssistantResponse(
                    content=f"Failed to switch to {target_mode} mode.",
                    success=False,
                    metadata={"target_mode": target_mode},
                    suggestions=["Check if the target mode is supported on your system."]
                )
                
        except Exception as e:
            logger.error(f"Error handling mode switch: {e}")
            return AssistantResponse(
                content=f"Mode switch error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Try switching modes again."]
            )
    
    async def _handle_accessibility_request(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle accessibility-related requests"""
        try:
            interaction_manager = self._capability_modules.get('interaction_manager')
            if not interaction_manager:
                return AssistantResponse(
                    content="Accessibility features are not available.",
                    success=False,
                    metadata={},
                    suggestions=["Accessibility support requires proper system configuration."]
                )
            
            accessibility_manager = interaction_manager.accessibility_manager
            action = request.metadata.get('action', 'get_status')
            
            if action == 'get_status':
                status = accessibility_manager.get_accessibility_status()
                return AssistantResponse(
                    content="Accessibility status retrieved.",
                    success=True,
                    metadata={"accessibility_status": status},
                    suggestions=[]
                )
            
            elif action == 'enable_feature':
                feature_name = request.metadata.get('feature')
                if not feature_name:
                    return AssistantResponse(
                        content="Feature name is required to enable accessibility feature.",
                        success=False,
                        metadata={},
                        suggestions=["Specify which accessibility feature to enable."]
                    )
                
                try:
                    from .accessibility_manager import AccessibilityFeature
                    feature = AccessibilityFeature(feature_name)
                    accessibility_manager.enable_feature(feature)
                    
                    return AssistantResponse(
                        content=f"Accessibility feature '{feature_name}' enabled.",
                        success=True,
                        metadata={"enabled_feature": feature_name},
                        suggestions=[]
                    )
                except ValueError:
                    return AssistantResponse(
                        content=f"Unknown accessibility feature: {feature_name}",
                        success=False,
                        metadata={},
                        suggestions=["Available features: screen_reader, keyboard_navigation, high_contrast, large_text, voice_control, reduced_motion"]
                    )
            
            elif action == 'disable_feature':
                feature_name = request.metadata.get('feature')
                if not feature_name:
                    return AssistantResponse(
                        content="Feature name is required to disable accessibility feature.",
                        success=False,
                        metadata={},
                        suggestions=["Specify which accessibility feature to disable."]
                    )
                
                try:
                    from .accessibility_manager import AccessibilityFeature
                    feature = AccessibilityFeature(feature_name)
                    accessibility_manager.disable_feature(feature)
                    
                    return AssistantResponse(
                        content=f"Accessibility feature '{feature_name}' disabled.",
                        success=True,
                        metadata={"disabled_feature": feature_name},
                        suggestions=[]
                    )
                except ValueError:
                    return AssistantResponse(
                        content=f"Unknown accessibility feature: {feature_name}",
                        success=False,
                        metadata={},
                        suggestions=["Available features: screen_reader, keyboard_navigation, high_contrast, large_text, voice_control, reduced_motion"]
                    )
            
            elif action == 'check_compliance':
                compliance = accessibility_manager.check_wcag_compliance()
                return AssistantResponse(
                    content=f"WCAG compliance level: {compliance['compliance_level']}",
                    success=True,
                    metadata={"compliance_check": compliance},
                    suggestions=compliance.get('recommendations', [])
                )
            
            elif action == 'announce':
                text = request.metadata.get('text', request.content)
                priority = request.metadata.get('priority', 'normal')
                
                await accessibility_manager.announce_to_screen_reader(text, priority)
                return AssistantResponse(
                    content=f"Announced to screen reader: {text}",
                    success=True,
                    metadata={"announced_text": text, "priority": priority},
                    suggestions=[]
                )
            
            else:
                return AssistantResponse(
                    content=f"Unknown accessibility action: {action}",
                    success=False,
                    metadata={},
                    suggestions=["Available actions: get_status, enable_feature, disable_feature, check_compliance, announce"]
                )
                
        except Exception as e:
            logger.error(f"Error handling accessibility request: {e}")
            return AssistantResponse(
                content=f"Accessibility request error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Try the accessibility request again."]
            )
    
    async def _handle_integration_request(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle external tool integration requests"""
        try:
            action = request.metadata.get('action', 'list_integrations')
            
            if action == 'list_integrations':
                integrations = await self.integration_hub.list_integrations()
                return AssistantResponse(
                    content=f"Available integrations: {', '.join(integrations.keys())}",
                    success=True,
                    metadata={"integrations": integrations},
                    suggestions=["You can connect to cloud services, development tools, and productivity apps."]
                )
            
            elif action == 'test_connections':
                results = await self.integration_hub.test_all_connections()
                working_count = sum(1 for result in results.values() if result)
                total_count = len(results)
                
                return AssistantResponse(
                    content=f"Connection test complete: {working_count}/{total_count} integrations working",
                    success=True,
                    metadata={"connection_results": results},
                    suggestions=["Check individual integration settings if any connections failed."]
                )
            
            elif action == 'sync_cloud_files':
                service = request.metadata.get('service')
                files = await self.integration_hub.sync_files_from_cloud(request.user_id, service)
                
                return AssistantResponse(
                    content=f"Found {len(files)} files from cloud services",
                    success=True,
                    metadata={"cloud_files": files},
                    suggestions=["I can help you organize or analyze these files."]
                )
            
            elif action == 'get_dev_context':
                context_data = await self.integration_hub.get_development_context(request.user_id)
                
                repo_count = len(context_data.get('repositories', []))
                issue_count = sum(len(issues) for issues in context_data.get('issues', {}).values())
                
                return AssistantResponse(
                    content=f"Development context: {repo_count} repositories, {issue_count} open issues",
                    success=True,
                    metadata={"dev_context": context_data},
                    suggestions=["I can help you prioritize issues or analyze repository activity."]
                )
            
            elif action == 'send_notification':
                message = request.metadata.get('message', request.content)
                channel = request.metadata.get('channel')
                
                success = await self.integration_hub.send_notification(request.user_id, message, channel)
                
                if success:
                    return AssistantResponse(
                        content="Notification sent successfully",
                        success=True,
                        metadata={"notification_sent": True},
                        suggestions=[]
                    )
                else:
                    return AssistantResponse(
                        content="Failed to send notification - check your communication tool connections",
                        success=False,
                        metadata={"notification_sent": False},
                        suggestions=["Verify your Slack or other communication tool integration is working."]
                    )
            
            elif action == 'connect_service':
                service_name = request.metadata.get('service')
                if not service_name:
                    return AssistantResponse(
                        content="Service name is required to connect",
                        success=False,
                        metadata={},
                        suggestions=["Specify which service to connect: google_drive, onedrive, github, slack"]
                    )
                
                # This would typically involve OAuth flow in a real implementation
                return AssistantResponse(
                    content=f"To connect {service_name}, you'll need to provide authentication credentials",
                    success=True,
                    metadata={"service": service_name, "auth_required": True},
                    suggestions=[f"Visit the {service_name} settings to configure API access"]
                )
            
            else:
                return AssistantResponse(
                    content=f"Unknown integration action: {action}",
                    success=False,
                    metadata={},
                    suggestions=["Available actions: list_integrations, test_connections, sync_cloud_files, get_dev_context, send_notification, connect_service"]
                )
                
        except Exception as e:
            logger.error(f"Error handling integration request: {e}")
            return AssistantResponse(
                content=f"Integration request error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Try the integration request again or check your connection settings."]
            )
    
    async def _handle_proactive_assistance(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle proactive assistance requests"""
        try:
            user_modules = await self._get_or_create_user_modules(request.user_id)
            proactive_assistant = user_modules.get('proactive_assistant')
            
            if not proactive_assistant:
                return AssistantResponse(
                    content="Proactive assistance is not available. Please try again later.",
                    success=False,
                    metadata={},
                    suggestions=["Check your assistant configuration."]
                )
            
            action = request.metadata.get('action', 'identify_opportunities')
            
            if action == 'identify_opportunities':
                # Identify new proactive opportunities
                opportunities = await proactive_assistant.identify_opportunities(request.user_id)
                
                if not opportunities:
                    return AssistantResponse(
                        content="I don't see any immediate opportunities for assistance right now. I'll keep monitoring and let you know when I find something helpful!",
                        success=True,
                        metadata={"opportunities_count": 0},
                        suggestions=["Continue working - I'll proactively suggest improvements as I notice patterns."]
                    )
                
                # Format opportunities for response
                opportunity_summaries = []
                for opp in opportunities[:5]:  # Show top 5 opportunities
                    priority_emoji = {
                        "urgent": "🚨",
                        "high": "⚡",
                        "medium": "💡",
                        "low": "💭"
                    }.get(opp.priority.value, "💡")
                    
                    opportunity_summaries.append(
                        f"{priority_emoji} **{opp.title}**\n   {opp.description}\n   *Suggested: {opp.suggested_action}*"
                    )
                
                content = f"I found {len(opportunities)} opportunities to help you:\n\n" + "\n\n".join(opportunity_summaries)
                
                if len(opportunities) > 5:
                    content += f"\n\n... and {len(opportunities) - 5} more suggestions available."
                
                return AssistantResponse(
                    content=content,
                    success=True,
                    metadata={
                        "opportunities_count": len(opportunities),
                        "opportunities": [
                            {
                                "id": opp.opportunity_id,
                                "type": opp.opportunity_type.value,
                                "title": opp.title,
                                "priority": opp.priority.value,
                                "confidence": opp.confidence
                            }
                            for opp in opportunities
                        ]
                    },
                    suggestions=[
                        "Tell me more about a specific suggestion",
                        "Implement one of these suggestions",
                        "Dismiss suggestions I'm not interested in"
                    ]
                )
            
            elif action == 'get_active_opportunities':
                # Get currently active opportunities
                active_opportunities = await proactive_assistant.get_active_opportunities(request.user_id)
                
                if not active_opportunities:
                    return AssistantResponse(
                        content="You don't have any active proactive suggestions right now.",
                        success=True,
                        metadata={"active_count": 0},
                        suggestions=["Ask me to look for new opportunities."]
                    )
                
                summaries = [f"• {opp.title}: {opp.description}" for opp in active_opportunities]
                
                return AssistantResponse(
                    content=f"You have {len(active_opportunities)} active suggestions:\n\n" + "\n".join(summaries),
                    success=True,
                    metadata={
                        "active_count": len(active_opportunities),
                        "opportunities": [opp.__dict__ for opp in active_opportunities]
                    },
                    suggestions=["Act on a suggestion", "Dismiss a suggestion", "Get more details"]
                )
            
            elif action == 'respond_to_opportunity':
                # Process user response to an opportunity
                opportunity_id = request.metadata.get('opportunity_id')
                response_text = request.metadata.get('response', request.content)
                action_taken = request.metadata.get('action_taken', False)
                
                if not opportunity_id:
                    return AssistantResponse(
                        content="Please specify which opportunity you're responding to.",
                        success=False,
                        metadata={},
                        suggestions=["Use the opportunity ID from the suggestion list."]
                    )
                
                await proactive_assistant.process_user_response(
                    request.user_id, opportunity_id, response_text, action_taken
                )
                
                return AssistantResponse(
                    content="Thank you for your feedback! I'll use this to improve future suggestions.",
                    success=True,
                    metadata={"feedback_recorded": True},
                    suggestions=["Ask for new opportunities", "Check your suggestion statistics"]
                )
            
            elif action == 'dismiss_opportunity':
                # Dismiss a specific opportunity
                opportunity_id = request.metadata.get('opportunity_id')
                
                if not opportunity_id:
                    return AssistantResponse(
                        content="Please specify which opportunity to dismiss.",
                        success=False,
                        metadata={},
                        suggestions=["Use the opportunity ID from the suggestion list."]
                    )
                
                success = await proactive_assistant.dismiss_opportunity(request.user_id, opportunity_id)
                
                if success:
                    return AssistantResponse(
                        content="Suggestion dismissed. I'll learn from this preference.",
                        success=True,
                        metadata={"dismissed": True},
                        suggestions=["Ask for new opportunities"]
                    )
                else:
                    return AssistantResponse(
                        content="Could not find that suggestion to dismiss.",
                        success=False,
                        metadata={"dismissed": False},
                        suggestions=["Check the suggestion ID and try again."]
                    )
            
            elif action == 'get_statistics':
                # Get proactive assistance statistics
                stats = proactive_assistant.get_opportunity_statistics(request.user_id)
                
                if not stats or stats.get("total_opportunities", 0) == 0:
                    return AssistantResponse(
                        content="No proactive assistance statistics available yet. I'll start tracking as I make suggestions!",
                        success=True,
                        metadata={"stats": stats},
                        suggestions=["Ask me to look for opportunities to get started."]
                    )
                
                content = f"""**Proactive Assistance Statistics:**

📊 **Overall Activity:**
• Total suggestions made: {stats['total_opportunities']}
• Suggestions you responded to: {stats['responded_opportunities']}
• Actions you took: {stats['actions_taken']}
• Response rate: {stats['response_rate']:.1%}
• Action rate: {stats['action_rate']:.1%}

🎯 **Suggestion Types:**"""
                
                for opp_type, count in stats.get('opportunity_types', {}).items():
                    response_rate = stats.get('type_response_rates', {}).get(opp_type, 0)
                    content += f"\n• {opp_type.replace('_', ' ').title()}: {count} suggestions ({response_rate:.1%} response rate)"
                
                return AssistantResponse(
                    content=content,
                    success=True,
                    metadata={"statistics": stats},
                    suggestions=["Ask for new opportunities", "See what types of suggestions work best for you"]
                )
            
            else:
                return AssistantResponse(
                    content=f"Unknown proactive assistance action: {action}",
                    success=False,
                    metadata={},
                    suggestions=[
                        "Available actions: identify_opportunities, get_active_opportunities, respond_to_opportunity, dismiss_opportunity, get_statistics"
                    ]
                )
                
        except Exception as e:
            logger.error(f"Error handling proactive assistance request: {e}")
            return AssistantResponse(
                content=f"Proactive assistance error: {str(e)}",
                success=False,
                metadata={"error": str(e)},
                suggestions=["Try the proactive assistance request again."]
            )
    
    def _map_request_to_interaction_type(self, request_type: RequestType) -> InteractionType:
        """Map request type to interaction type"""
        mapping = {
            RequestType.QUERY: InteractionType.QUERY,
            RequestType.COMMAND: InteractionType.COMMAND,
            RequestType.FILE_OPERATION: InteractionType.FILE_ACCESS,
            RequestType.CONTEXT_UPDATE: InteractionType.QUERY,
            RequestType.PERMISSION_REQUEST: InteractionType.QUERY,
            RequestType.PRIVACY_CONTROL: InteractionType.QUERY,
            RequestType.SCREEN_MONITORING: InteractionType.SCREEN_CONTEXT,
            RequestType.TASK_MANAGEMENT: InteractionType.COMMAND,
            RequestType.KNOWLEDGE_SEARCH: InteractionType.QUERY,
            RequestType.LEARNING_FEEDBACK: InteractionType.FEEDBACK,
            RequestType.VOICE_COMMAND: InteractionType.COMMAND,
            RequestType.TEXT_COMPLETION: InteractionType.QUERY,
            RequestType.VISUAL_FEEDBACK: InteractionType.COMMAND,
            RequestType.MODE_SWITCH: InteractionType.COMMAND,
            RequestType.ACCESSIBILITY_REQUEST: InteractionType.COMMAND,
            RequestType.INTEGRATION_REQUEST: InteractionType.COMMAND,
            RequestType.PROACTIVE_ASSISTANCE: InteractionType.COMMAND,
        }
        return mapping.get(request_type, InteractionType.QUERY)
    
    async def _update_context_from_interaction(self, context: UserContext, 
                                             request: AssistantRequest, 
                                             response: AssistantResponse) -> None:
        """Update user context based on the interaction"""
        # Update last activity
        context.last_activity = datetime.now()
        
        # Update current activity if relevant
        if request.request_type == RequestType.FILE_OPERATION:
            file_path = request.metadata.get("file_path")
            if file_path and file_path not in context.current_files:
                context.current_files.append(file_path)
                # Keep only recent files (last 20)
                if len(context.current_files) > 20:
                    context.current_files = context.current_files[-20:]
        
        # Update context and save
        await self.context_manager.update_user_context(context)
    
    async def get_capability_status(self, user_id: str) -> Dict[str, Any]:
        """Get status of all capability modules for a user"""
        status = {
            "core_modules": {},
            "user_modules": {},
            "permissions": {},
            "overall_health": "healthy"
        }
        
        try:
            # Check core modules
            status["core_modules"]["file_system"] = "available" if self._capability_modules.get('file_system') else "unavailable"
            status["core_modules"]["learning"] = "available" if self._capability_modules.get('learning') else "unavailable"
            status["core_modules"]["task_manager"] = "available" if self._capability_modules.get('task_manager') else "unavailable"
            
            # Check user-specific modules
            user_modules = await self._get_or_create_user_modules(user_id)
            status["user_modules"]["screen_monitor"] = "available" if user_modules.get('screen_monitor') else "unavailable"
            status["user_modules"]["knowledge_base"] = "available" if user_modules.get('knowledge_base') else "unavailable"
            
            # Check permissions
            for permission in PermissionType:
                has_permission = await self.privacy_manager.check_permission(user_id, permission)
                status["permissions"][permission.value] = "granted" if has_permission else "not_granted"
            
            # Determine overall health
            unavailable_count = sum(1 for v in status["core_modules"].values() if v == "unavailable")
            unavailable_count += sum(1 for v in status["user_modules"].values() if v == "unavailable")
            
            if unavailable_count == 0:
                status["overall_health"] = "healthy"
            elif unavailable_count <= 2:
                status["overall_health"] = "degraded"
            else:
                status["overall_health"] = "critical"
                
        except Exception as e:
            logger.error(f"Error getting capability status: {e}")
            status["overall_health"] = "error"
            status["error"] = str(e)
        
        return status
    
    async def initialize_user_capabilities(self, user_id: str) -> Dict[str, bool]:
        """Initialize all capabilities for a new user"""
        results = {}
        
        try:
            # Initialize user-specific modules
            user_modules = await self._get_or_create_user_modules(user_id)
            results["screen_monitor"] = user_modules.get('screen_monitor') is not None
            results["knowledge_base"] = user_modules.get('knowledge_base') is not None
            
            # Initialize user context if not exists
            context = await self.context_manager.get_user_context(user_id)
            results["user_context"] = context is not None
            
            # Initialize learning model (learning engine initializes automatically)
            learning_engine = self._capability_modules.get('learning')
            results["learning_model"] = learning_engine is not None
            
            logger.info(f"User capabilities initialized for {user_id}: {results}")
            
        except Exception as e:
            logger.error(f"Error initializing user capabilities for {user_id}: {e}")
            results["error"] = str(e)
        
        return results
    
    async def shutdown(self):
        """Shutdown the personal assistant core"""
        logger.info("Shutting down Personal Assistant Core")
        
        # Shutdown user-specific modules
        for user_key, user_modules in self._capability_modules.items():
            if user_key.startswith("user_") and isinstance(user_modules, dict):
                screen_monitor = user_modules.get('screen_monitor')
                if screen_monitor:
                    try:
                        # Extract user_id from user_key
                        user_id = user_key.replace("user_", "")
                        await screen_monitor.stop_monitoring(user_id)
                    except Exception as e:
                        logger.error(f"Error stopping screen monitor: {e}")
        
        # Cleanup resources
        self.db.close()
        logger.info("Personal Assistant Core shutdown complete")