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

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Types of assistant requests"""
    QUERY = "query"
    COMMAND = "command"
    FILE_OPERATION = "file_operation"
    CONTEXT_UPDATE = "context_update"
    PERMISSION_REQUEST = "permission_request"
    PRIVACY_CONTROL = "privacy_control"


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
        
        # Initialize capability modules (placeholders for now)
        self._capability_modules = {}
        
        logger.info("Personal Assistant Core initialized")
    
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
        else:
            return AssistantResponse(
                content="I don't understand that type of request.",
                success=False,
                metadata={},
                suggestions=["Try rephrasing your request or ask for help."]
            )
    
    async def _handle_query(self, request: AssistantRequest, context: UserContext) -> AssistantResponse:
        """Handle general queries"""
        # This is a placeholder - would integrate with actual query processing
        return AssistantResponse(
            content=f"I understand you're asking: {request.content}. This is a placeholder response.",
            success=True,
            metadata={"query_type": "general"},
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
        """Handle file operations"""
        # This is a placeholder - would integrate with file system manager
        return AssistantResponse(
            content=f"File operation requested: {request.content}",
            success=True,
            metadata={"operation_type": "file"},
            suggestions=[]
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
    
    def _map_request_to_interaction_type(self, request_type: RequestType) -> InteractionType:
        """Map request type to interaction type"""
        mapping = {
            RequestType.QUERY: InteractionType.QUERY,
            RequestType.COMMAND: InteractionType.COMMAND,
            RequestType.FILE_OPERATION: InteractionType.FILE_ACCESS,
            RequestType.CONTEXT_UPDATE: InteractionType.QUERY,
            RequestType.PERMISSION_REQUEST: InteractionType.QUERY,
            RequestType.PRIVACY_CONTROL: InteractionType.QUERY,
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
    
    async def shutdown(self):
        """Shutdown the personal assistant core"""
        logger.info("Shutting down Personal Assistant Core")
        # Cleanup resources
        self.db.close()