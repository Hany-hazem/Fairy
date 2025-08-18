"""
Proactive Assistant Engine

This module implements proactive assistance capabilities that identify opportunities
for automation, provide contextual help, detect errors, and recommend learning resources
based on user behavior patterns and context.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import re

from .personal_assistant_models import UserContext, Interaction, InteractionType
from .learning_engine import LearningEngine, BehaviorPattern, UserFeedback
from .task_manager import TaskManager, Task, TaskPriority, TaskStatus
from .screen_monitor import ScreenMonitor, ScreenContext
from .personal_knowledge_base import PersonalKnowledgeBase
from .user_context_manager import UserContextManager

logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    """Types of proactive assistance opportunities"""
    AUTOMATION = "automation"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    ERROR_DETECTION = "error_detection"
    LEARNING_RESOURCE = "learning_resource"
    CONTEXTUAL_HELP = "contextual_help"
    DEADLINE_REMINDER = "deadline_reminder"
    PRODUCTIVITY_TIP = "productivity_tip"


class SuggestionPriority(Enum):
    """Priority levels for proactive suggestions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ProactiveOpportunity:
    """Represents a proactive assistance opportunity"""
    opportunity_id: str
    user_id: str
    opportunity_type: OpportunityType
    title: str
    description: str
    suggested_action: str
    priority: SuggestionPriority
    confidence: float
    context_data: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutomationSuggestion:
    """Represents an automation opportunity"""
    suggestion_id: str
    task_pattern: str
    frequency: int
    time_saved_estimate: int  # in minutes
    automation_script: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)
    difficulty_level: str = "medium"


@dataclass
class WorkflowOptimization:
    """Represents a workflow optimization suggestion"""
    optimization_id: str
    current_workflow: List[str]
    optimized_workflow: List[str]
    efficiency_gain: float  # percentage improvement
    rationale: str
    implementation_steps: List[str] = field(default_factory=list)


@dataclass
class ErrorDetection:
    """Represents detected error or issue"""
    error_id: str
    error_type: str
    error_description: str
    suggested_solutions: List[str]
    severity: str
    context_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningResource:
    """Represents a recommended learning resource"""
    resource_id: str
    title: str
    resource_type: str  # "tutorial", "documentation", "course", "article"
    url: Optional[str] = None
    description: str = ""
    skill_level: str = "beginner"
    estimated_time: int = 0  # in minutes
    relevance_score: float = 0.0


class ProactiveAssistant:
    """
    Proactive Assistant Engine that identifies opportunities for assistance,
    automation, and optimization based on user behavior and context.
    """
    
    def __init__(
        self,
        learning_engine: LearningEngine,
        task_manager: TaskManager,
        screen_monitor: ScreenMonitor,
        knowledge_base: PersonalKnowledgeBase,
        context_manager: UserContextManager
    ):
        self.learning_engine = learning_engine
        self.task_manager = task_manager
        self.screen_monitor = screen_monitor
        self.knowledge_base = knowledge_base
        self.context_manager = context_manager
        
        # Opportunity tracking
        self.active_opportunities: Dict[str, ProactiveOpportunity] = {}
        self.opportunity_history: List[ProactiveOpportunity] = []
        self.user_responses: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Pattern detection thresholds
        self.repetition_threshold = 3  # minimum repetitions to suggest automation
        self.time_window_hours = 24  # time window for pattern detection
        self.confidence_threshold = 0.7  # minimum confidence for suggestions
        
        # Error detection patterns
        self.error_patterns = {
            "file_not_found": r"(file not found|no such file|cannot find)",
            "permission_denied": r"(permission denied|access denied|unauthorized)",
            "syntax_error": r"(syntax error|invalid syntax|parse error)",
            "network_error": r"(connection failed|network error|timeout)",
            "memory_error": r"(out of memory|memory error|insufficient memory)"
        }
        
        # Learning resource database
        self.learning_resources = self._initialize_learning_resources()
        
        logger.info("ProactiveAssistant initialized")
    
    async def identify_opportunities(self, user_id: str) -> List[ProactiveOpportunity]:
        """
        Identify proactive assistance opportunities for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of identified opportunities
        """
        try:
            opportunities = []
            
            # Get current user context
            context = await self.context_manager.get_context(user_id)
            if not context:
                return opportunities
            
            # Identify different types of opportunities
            automation_ops = await self._identify_automation_opportunities(user_id, context)
            workflow_ops = await self._identify_workflow_optimizations(user_id, context)
            error_ops = await self._detect_errors_and_issues(user_id, context)
            learning_ops = await self._recommend_learning_resources(user_id, context)
            contextual_ops = await self._provide_contextual_help(user_id, context)
            deadline_ops = await self._check_deadline_reminders(user_id, context)
            
            opportunities.extend(automation_ops)
            opportunities.extend(workflow_ops)
            opportunities.extend(error_ops)
            opportunities.extend(learning_ops)
            opportunities.extend(contextual_ops)
            opportunities.extend(deadline_ops)
            
            # Filter and prioritize opportunities
            filtered_opportunities = self._filter_and_prioritize(opportunities, user_id)
            
            # Update active opportunities
            for opp in filtered_opportunities:
                self.active_opportunities[opp.opportunity_id] = opp
            
            logger.info(f"Identified {len(filtered_opportunities)} opportunities for user {user_id}")
            return filtered_opportunities
            
        except Exception as e:
            logger.error(f"Error identifying opportunities for user {user_id}: {e}")
            return []
    
    async def _identify_automation_opportunities(
        self, 
        user_id: str, 
        context: UserContext
    ) -> List[ProactiveOpportunity]:
        """Identify repetitive tasks that can be automated"""
        opportunities = []
        
        try:
            # Get user behavior patterns
            patterns = await self.learning_engine.get_behavior_patterns(user_id)
            
            # Look for repetitive task patterns
            for pattern in patterns:
                if pattern.pattern_type == "task_sequence" and pattern.frequency >= self.repetition_threshold:
                    # Create automation suggestion
                    automation = AutomationSuggestion(
                        suggestion_id=f"auto_{pattern.pattern_id}",
                        task_pattern=pattern.pattern_data.get("sequence", ""),
                        frequency=pattern.frequency,
                        time_saved_estimate=self._estimate_time_savings(pattern),
                        automation_script=self._generate_automation_script(pattern)
                    )
                    
                    opportunity = ProactiveOpportunity(
                        opportunity_id=f"automation_{automation.suggestion_id}",
                        user_id=user_id,
                        opportunity_type=OpportunityType.AUTOMATION,
                        title=f"Automate repetitive task: {pattern.pattern_data.get('name', 'Unknown')}",
                        description=f"You've performed this task {pattern.frequency} times. Consider automation.",
                        suggested_action=f"Create automation script: {automation.automation_script}",
                        priority=SuggestionPriority.MEDIUM,
                        confidence=pattern.confidence,
                        context_data={"automation": automation.__dict__},
                        created_at=datetime.now()
                    )
                    
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying automation opportunities: {e}")
            return []
    
    async def _identify_workflow_optimizations(
        self, 
        user_id: str, 
        context: UserContext
    ) -> List[ProactiveOpportunity]:
        """Identify workflow optimization opportunities"""
        opportunities = []
        
        try:
            # Analyze current workflows from task patterns
            patterns = await self.learning_engine.get_behavior_patterns(user_id)
            
            for pattern in patterns:
                if pattern.pattern_type == "workflow" and pattern.confidence > self.confidence_threshold:
                    # Analyze workflow efficiency
                    current_workflow = pattern.pattern_data.get("steps", [])
                    optimized_workflow = self._optimize_workflow(current_workflow)
                    
                    if len(optimized_workflow) < len(current_workflow):
                        efficiency_gain = (len(current_workflow) - len(optimized_workflow)) / len(current_workflow) * 100
                        
                        optimization = WorkflowOptimization(
                            optimization_id=f"workflow_{pattern.pattern_id}",
                            current_workflow=current_workflow,
                            optimized_workflow=optimized_workflow,
                            efficiency_gain=efficiency_gain,
                            rationale=f"Reduce steps by {len(current_workflow) - len(optimized_workflow)}",
                            implementation_steps=self._generate_implementation_steps(optimized_workflow)
                        )
                        
                        opportunity = ProactiveOpportunity(
                            opportunity_id=f"workflow_{optimization.optimization_id}",
                            user_id=user_id,
                            opportunity_type=OpportunityType.WORKFLOW_OPTIMIZATION,
                            title=f"Optimize workflow: {pattern.pattern_data.get('name', 'Unknown')}",
                            description=f"Potential {efficiency_gain:.1f}% efficiency improvement",
                            suggested_action=f"Implement optimized workflow with {len(optimized_workflow)} steps",
                            priority=SuggestionPriority.MEDIUM,
                            confidence=pattern.confidence,
                            context_data={"optimization": optimization.__dict__},
                            created_at=datetime.now()
                        )
                        
                        opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying workflow optimizations: {e}")
            return []
    
    async def _detect_errors_and_issues(
        self, 
        user_id: str, 
        context: UserContext
    ) -> List[ProactiveOpportunity]:
        """Detect errors and issues from screen content and interactions"""
        opportunities = []
        
        try:
            # Get recent screen context
            screen_context = await self.screen_monitor.get_current_context(user_id)
            if not screen_context:
                return opportunities
            
            # Check for error patterns in visible text
            visible_text = screen_context.visible_text.lower()
            
            for error_type, pattern in self.error_patterns.items():
                if re.search(pattern, visible_text, re.IGNORECASE):
                    # Generate error detection
                    error = ErrorDetection(
                        error_id=f"error_{error_type}_{datetime.now().timestamp()}",
                        error_type=error_type,
                        error_description=f"Detected {error_type} in current application",
                        suggested_solutions=self._get_error_solutions(error_type),
                        severity=self._get_error_severity(error_type),
                        context_info={
                            "application": screen_context.active_application,
                            "window_title": screen_context.window_title,
                            "detected_text": visible_text[:200]
                        }
                    )
                    
                    opportunity = ProactiveOpportunity(
                        opportunity_id=f"error_{error.error_id}",
                        user_id=user_id,
                        opportunity_type=OpportunityType.ERROR_DETECTION,
                        title=f"Error detected: {error_type.replace('_', ' ').title()}",
                        description=error.error_description,
                        suggested_action=f"Try: {error.suggested_solutions[0] if error.suggested_solutions else 'Check documentation'}",
                        priority=SuggestionPriority.HIGH if error.severity == "high" else SuggestionPriority.MEDIUM,
                        confidence=0.8,
                        context_data={"error": error.__dict__},
                        created_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(minutes=30)
                    )
                    
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error detecting issues: {e}")
            return []
    
    async def _recommend_learning_resources(
        self, 
        user_id: str, 
        context: UserContext
    ) -> List[ProactiveOpportunity]:
        """Recommend learning resources based on user context and skill gaps"""
        opportunities = []
        
        try:
            # Get user's current activity and skill level
            current_activity = context.current_activity
            if not current_activity:
                return opportunities
            
            # Identify skill gaps from error patterns or new technologies
            skill_gaps = await self._identify_skill_gaps(user_id, context)
            
            for skill in skill_gaps:
                # Find relevant learning resources
                resources = self._find_learning_resources(skill, context)
                
                for resource in resources[:2]:  # Limit to top 2 resources per skill
                    opportunity = ProactiveOpportunity(
                        opportunity_id=f"learning_{resource.resource_id}",
                        user_id=user_id,
                        opportunity_type=OpportunityType.LEARNING_RESOURCE,
                        title=f"Learn: {resource.title}",
                        description=f"Recommended {resource.resource_type} for {skill}",
                        suggested_action=f"Access resource: {resource.url or 'Available in knowledge base'}",
                        priority=SuggestionPriority.LOW,
                        confidence=resource.relevance_score,
                        context_data={"resource": resource.__dict__},
                        created_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(days=7)
                    )
                    
                    opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error recommending learning resources: {e}")
            return []
    
    async def _provide_contextual_help(
        self, 
        user_id: str, 
        context: UserContext
    ) -> List[ProactiveOpportunity]:
        """Provide contextual help based on current user activity"""
        opportunities = []
        
        try:
            # Get screen context for current application
            screen_context = await self.screen_monitor.get_current_context(user_id)
            if not screen_context:
                return opportunities
            
            # Generate contextual help based on application and activity
            help_suggestions = await self._generate_contextual_help(screen_context, context)
            
            for suggestion in help_suggestions:
                opportunity = ProactiveOpportunity(
                    opportunity_id=f"help_{suggestion['id']}",
                    user_id=user_id,
                    opportunity_type=OpportunityType.CONTEXTUAL_HELP,
                    title=suggestion["title"],
                    description=suggestion["description"],
                    suggested_action=suggestion["action"],
                    priority=SuggestionPriority.LOW,
                    confidence=suggestion.get("confidence", 0.6),
                    context_data=suggestion,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(hours=2)
                )
                
                opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error providing contextual help: {e}")
            return []
    
    async def _check_deadline_reminders(
        self, 
        user_id: str, 
        context: UserContext
    ) -> List[ProactiveOpportunity]:
        """Check for upcoming deadlines and provide reminders"""
        opportunities = []
        
        try:
            # Get upcoming deadlines from task manager
            deadlines = await self.task_manager.get_upcoming_deadlines(user_id)
            
            for deadline in deadlines:
                time_until_deadline = deadline.due_date - datetime.now()
                
                # Create reminder based on urgency
                if time_until_deadline.days <= 1:
                    priority = SuggestionPriority.URGENT
                    title = f"URGENT: {deadline.task_name} due soon"
                elif time_until_deadline.days <= 3:
                    priority = SuggestionPriority.HIGH
                    title = f"Reminder: {deadline.task_name} due in {time_until_deadline.days} days"
                else:
                    continue  # Don't remind for distant deadlines
                
                opportunity = ProactiveOpportunity(
                    opportunity_id=f"deadline_{deadline.task_id}",
                    user_id=user_id,
                    opportunity_type=OpportunityType.DEADLINE_REMINDER,
                    title=title,
                    description=f"Task: {deadline.task_name}, Due: {deadline.due_date.strftime('%Y-%m-%d %H:%M')}",
                    suggested_action=f"Work on task: {deadline.task_name}",
                    priority=priority,
                    confidence=1.0,
                    context_data={"deadline": deadline.__dict__},
                    created_at=datetime.now(),
                    expires_at=deadline.due_date
                )
                
                opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error checking deadline reminders: {e}")
            return []
    
    def _filter_and_prioritize(
        self, 
        opportunities: List[ProactiveOpportunity], 
        user_id: str
    ) -> List[ProactiveOpportunity]:
        """Filter and prioritize opportunities based on user preferences and context"""
        try:
            # Filter by confidence threshold
            filtered = [opp for opp in opportunities if opp.confidence >= self.confidence_threshold]
            
            # Remove expired opportunities
            now = datetime.now()
            filtered = [opp for opp in filtered if not opp.expires_at or opp.expires_at > now]
            
            # Remove duplicates based on similar content
            filtered = self._remove_duplicate_opportunities(filtered)
            
            # Sort by priority and confidence
            priority_order = {
                SuggestionPriority.URGENT: 4,
                SuggestionPriority.HIGH: 3,
                SuggestionPriority.MEDIUM: 2,
                SuggestionPriority.LOW: 1
            }
            
            filtered.sort(
                key=lambda x: (priority_order[x.priority], x.confidence),
                reverse=True
            )
            
            # Limit to top 10 opportunities to avoid overwhelming user
            return filtered[:10]
            
        except Exception as e:
            logger.error(f"Error filtering and prioritizing opportunities: {e}")
            return opportunities
    
    def _remove_duplicate_opportunities(
        self, 
        opportunities: List[ProactiveOpportunity]
    ) -> List[ProactiveOpportunity]:
        """Remove duplicate or very similar opportunities"""
        unique_opportunities = []
        seen_titles = set()
        
        for opp in opportunities:
            # Simple deduplication based on title similarity
            title_key = opp.title.lower().strip()
            if title_key not in seen_titles:
                unique_opportunities.append(opp)
                seen_titles.add(title_key)
        
        return unique_opportunities
    
    def _estimate_time_savings(self, pattern: BehaviorPattern) -> int:
        """Estimate time savings from automation in minutes"""
        # Simple heuristic based on pattern frequency and complexity
        base_time = 5  # assume 5 minutes per manual execution
        return pattern.frequency * base_time
    
    def _generate_automation_script(self, pattern: BehaviorPattern) -> str:
        """Generate a simple automation script suggestion"""
        sequence = pattern.pattern_data.get("sequence", [])
        if not sequence:
            return "# Create custom automation script"
        
        script_lines = ["# Automation script suggestion"]
        for i, step in enumerate(sequence[:5]):  # Limit to first 5 steps
            script_lines.append(f"# Step {i+1}: {step}")
        
        return "\n".join(script_lines)
    
    def _optimize_workflow(self, workflow: List[str]) -> List[str]:
        """Optimize a workflow by removing redundant steps"""
        if not workflow:
            return workflow
        
        # Simple optimization: remove duplicate consecutive steps
        optimized = [workflow[0]]
        for step in workflow[1:]:
            if step != optimized[-1]:
                optimized.append(step)
        
        return optimized
    
    def _generate_implementation_steps(self, workflow: List[str]) -> List[str]:
        """Generate implementation steps for optimized workflow"""
        return [f"Implement: {step}" for step in workflow]
    
    def _get_error_solutions(self, error_type: str) -> List[str]:
        """Get suggested solutions for different error types"""
        solutions = {
            "file_not_found": [
                "Check file path and spelling",
                "Verify file exists in expected location",
                "Check file permissions"
            ],
            "permission_denied": [
                "Run with administrator privileges",
                "Check file/folder permissions",
                "Contact system administrator"
            ],
            "syntax_error": [
                "Check code syntax",
                "Review documentation",
                "Use IDE syntax highlighting"
            ],
            "network_error": [
                "Check internet connection",
                "Verify server status",
                "Try again later"
            ],
            "memory_error": [
                "Close unnecessary applications",
                "Increase available memory",
                "Optimize code for memory usage"
            ]
        }
        
        return solutions.get(error_type, ["Check documentation", "Search for solutions online"])
    
    def _get_error_severity(self, error_type: str) -> str:
        """Get severity level for different error types"""
        high_severity = ["permission_denied", "memory_error"]
        return "high" if error_type in high_severity else "medium"
    
    async def _identify_skill_gaps(self, user_id: str, context: UserContext) -> List[str]:
        """Identify skill gaps based on user activity and errors"""
        skill_gaps = []
        
        # Analyze recent interactions for technology mentions
        recent_interactions = context.recent_interactions[-10:]  # Last 10 interactions
        
        technologies = set()
        for interaction in recent_interactions:
            # Extract technology keywords from interaction content
            content = interaction.content.lower()
            tech_keywords = ["python", "javascript", "react", "docker", "kubernetes", "aws", "git"]
            for tech in tech_keywords:
                if tech in content:
                    technologies.add(tech)
        
        # For each technology, check if user seems to be struggling
        for tech in technologies:
            # Simple heuristic: if user asks many questions about a technology, they might need learning resources
            tech_questions = sum(1 for interaction in recent_interactions 
                               if tech in interaction.content.lower() and "?" in interaction.content)
            
            if tech_questions >= 2:  # Asked 2+ questions about this technology
                skill_gaps.append(tech)
        
        return list(skill_gaps)
    
    def _find_learning_resources(self, skill: str, context: UserContext) -> List[LearningResource]:
        """Find learning resources for a specific skill"""
        resources = []
        
        # Get resources from the learning resource database
        skill_resources = self.learning_resources.get(skill.lower(), [])
        
        for resource_data in skill_resources:
            resource = LearningResource(
                resource_id=resource_data["id"],
                title=resource_data["title"],
                resource_type=resource_data["type"],
                url=resource_data.get("url"),
                description=resource_data.get("description", ""),
                skill_level=resource_data.get("level", "beginner"),
                estimated_time=resource_data.get("time", 30),
                relevance_score=resource_data.get("relevance", 0.7)
            )
            resources.append(resource)
        
        return resources
    
    async def _generate_contextual_help(
        self, 
        screen_context, 
        user_context: UserContext
    ) -> List[Dict[str, Any]]:
        """Generate contextual help suggestions based on current screen content"""
        suggestions = []
        
        app_name = screen_context.active_application.lower()
        
        # Application-specific help suggestions
        if "vscode" in app_name or "code" in app_name:
            suggestions.extend([
                {
                    "id": "vscode_shortcuts",
                    "title": "VS Code Shortcuts",
                    "description": "Learn useful keyboard shortcuts for faster coding",
                    "action": "Show VS Code shortcut reference",
                    "confidence": 0.7
                },
                {
                    "id": "vscode_extensions",
                    "title": "Recommended Extensions",
                    "description": "Discover extensions that could improve your workflow",
                    "action": "Browse recommended VS Code extensions",
                    "confidence": 0.6
                }
            ])
        
        elif "terminal" in app_name or "cmd" in app_name:
            suggestions.append({
                "id": "terminal_tips",
                "title": "Terminal Tips",
                "description": "Learn useful command line shortcuts and commands",
                "action": "Show terminal command reference",
                "confidence": 0.7
            })
        
        elif "browser" in app_name or "chrome" in app_name or "firefox" in app_name:
            suggestions.append({
                "id": "browser_productivity",
                "title": "Browser Productivity",
                "description": "Tips for more efficient web browsing and research",
                "action": "Show browser productivity tips",
                "confidence": 0.6
            })
        
        return suggestions
    
    def _initialize_learning_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize the learning resource database"""
        return {
            "python": [
                {
                    "id": "python_basics",
                    "title": "Python Basics Tutorial",
                    "type": "tutorial",
                    "url": "https://docs.python.org/3/tutorial/",
                    "description": "Official Python tutorial covering basics",
                    "level": "beginner",
                    "time": 120,
                    "relevance": 0.9
                },
                {
                    "id": "python_advanced",
                    "title": "Advanced Python Concepts",
                    "type": "course",
                    "description": "Deep dive into advanced Python features",
                    "level": "advanced",
                    "time": 300,
                    "relevance": 0.8
                }
            ],
            "javascript": [
                {
                    "id": "js_fundamentals",
                    "title": "JavaScript Fundamentals",
                    "type": "tutorial",
                    "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
                    "description": "MDN JavaScript guide",
                    "level": "beginner",
                    "time": 180,
                    "relevance": 0.9
                }
            ],
            "react": [
                {
                    "id": "react_intro",
                    "title": "Introduction to React",
                    "type": "tutorial",
                    "url": "https://reactjs.org/tutorial/tutorial.html",
                    "description": "Official React tutorial",
                    "level": "beginner",
                    "time": 90,
                    "relevance": 0.9
                }
            ],
            "docker": [
                {
                    "id": "docker_basics",
                    "title": "Docker Basics",
                    "type": "tutorial",
                    "url": "https://docs.docker.com/get-started/",
                    "description": "Official Docker getting started guide",
                    "level": "beginner",
                    "time": 60,
                    "relevance": 0.9
                }
            ],
            "git": [
                {
                    "id": "git_basics",
                    "title": "Git Basics",
                    "type": "tutorial",
                    "url": "https://git-scm.com/book/en/v2/Getting-Started-Git-Basics",
                    "description": "Official Git documentation",
                    "level": "beginner",
                    "time": 45,
                    "relevance": 0.9
                }
            ]
        }
    
    async def process_user_response(
        self, 
        user_id: str, 
        opportunity_id: str, 
        response: str,
        action_taken: bool = False
    ) -> None:
        """
        Process user response to a proactive suggestion
        
        Args:
            user_id: User identifier
            opportunity_id: ID of the opportunity being responded to
            response: User's response text
            action_taken: Whether user took the suggested action
        """
        try:
            # Record user response
            self.user_responses[user_id][opportunity_id] = {
                "response": response,
                "action_taken": action_taken,
                "timestamp": datetime.now()
            }
            
            # Update learning engine with feedback
            if opportunity_id in self.active_opportunities:
                opportunity = self.active_opportunities[opportunity_id]
                
                feedback = UserFeedback(
                    feedback_id=f"proactive_{opportunity_id}_{datetime.now().timestamp()}",
                    user_id=user_id,
                    interaction_id=opportunity_id,
                    feedback_type="proactive_response",
                    feedback_value={
                        "response": response,
                        "action_taken": action_taken,
                        "opportunity_type": opportunity.opportunity_type.value
                    },
                    timestamp=datetime.now(),
                    context_data=opportunity.context_data
                )
                
                await self.learning_engine.process_feedback(feedback)
                
                # Move to history
                self.opportunity_history.append(opportunity)
                del self.active_opportunities[opportunity_id]
            
            logger.info(f"Processed user response for opportunity {opportunity_id}")
            
        except Exception as e:
            logger.error(f"Error processing user response: {e}")
    
    async def get_active_opportunities(self, user_id: str) -> List[ProactiveOpportunity]:
        """Get currently active opportunities for a user"""
        try:
            user_opportunities = [
                opp for opp in self.active_opportunities.values()
                if opp.user_id == user_id
            ]
            
            # Remove expired opportunities
            now = datetime.now()
            active = []
            for opp in user_opportunities:
                if not opp.expires_at or opp.expires_at > now:
                    active.append(opp)
                else:
                    # Move expired opportunity to history
                    self.opportunity_history.append(opp)
                    if opp.opportunity_id in self.active_opportunities:
                        del self.active_opportunities[opp.opportunity_id]
            
            return active
            
        except Exception as e:
            logger.error(f"Error getting active opportunities: {e}")
            return []
    
    async def dismiss_opportunity(self, user_id: str, opportunity_id: str) -> bool:
        """Dismiss a proactive opportunity"""
        try:
            if opportunity_id in self.active_opportunities:
                opportunity = self.active_opportunities[opportunity_id]
                if opportunity.user_id == user_id:
                    # Record dismissal as feedback
                    await self.process_user_response(
                        user_id, 
                        opportunity_id, 
                        "dismissed", 
                        action_taken=False
                    )
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error dismissing opportunity: {e}")
            return False
    
    def get_opportunity_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about proactive opportunities for a user"""
        try:
            user_history = [opp for opp in self.opportunity_history if opp.user_id == user_id]
            user_responses = self.user_responses.get(user_id, {})
            
            total_opportunities = len(user_history)
            responded_opportunities = len(user_responses)
            actions_taken = sum(1 for resp in user_responses.values() if resp.get("action_taken", False))
            
            # Opportunity type breakdown
            type_counts = Counter(opp.opportunity_type.value for opp in user_history)
            
            # Response rate by type
            type_response_rates = {}
            for opp_type in type_counts:
                type_opportunities = [opp for opp in user_history if opp.opportunity_type.value == opp_type]
                type_responses = sum(1 for opp in type_opportunities if opp.opportunity_id in user_responses)
                type_response_rates[opp_type] = type_responses / len(type_opportunities) if type_opportunities else 0
            
            return {
                "total_opportunities": total_opportunities,
                "responded_opportunities": responded_opportunities,
                "actions_taken": actions_taken,
                "response_rate": responded_opportunities / total_opportunities if total_opportunities > 0 else 0,
                "action_rate": actions_taken / responded_opportunities if responded_opportunities > 0 else 0,
                "opportunity_types": dict(type_counts),
                "type_response_rates": type_response_rates
            }
            
        except Exception as e:
            logger.error(f"Error getting opportunity statistics: {e}")
            return {}