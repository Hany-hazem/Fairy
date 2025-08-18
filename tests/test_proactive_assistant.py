"""
Tests for ProactiveAssistant module

This module contains comprehensive tests for the proactive assistance engine,
including opportunity identification, automation suggestions, error detection,
and learning resource recommendations.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
import sys
import os

# Add the app directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock the config module to avoid configuration issues
with patch.dict('sys.modules', {'app.config': Mock()}):
    from app.proactive_assistant import (
        ProactiveAssistant, ProactiveOpportunity, OpportunityType, SuggestionPriority,
        AutomationSuggestion, WorkflowOptimization, ErrorDetection, LearningResource
    )

# Mock the other modules that might have config dependencies
@pytest.fixture(autouse=True)
def mock_config_dependencies():
    """Mock configuration-dependent modules"""
    with patch.dict('sys.modules', {
        'app.config': Mock(),
        'app.personal_knowledge_base': Mock(),
    }):
        yield


class TestProactiveAssistant:
    """Test cases for ProactiveAssistant class"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for ProactiveAssistant"""
        learning_engine = Mock()
        task_manager = Mock()
        screen_monitor = Mock()
        knowledge_base = Mock()
        context_manager = Mock()
        
        # Setup async methods
        learning_engine.get_behavior_patterns = AsyncMock()
        learning_engine.process_feedback = AsyncMock()
        task_manager.get_upcoming_deadlines = AsyncMock()
        screen_monitor.get_current_context = AsyncMock()
        knowledge_base.search_knowledge = AsyncMock()
        context_manager.get_context = AsyncMock()
        
        return {
            'learning_engine': learning_engine,
            'task_manager': task_manager,
            'screen_monitor': screen_monitor,
            'knowledge_base': knowledge_base,
            'context_manager': context_manager
        }
    
    @pytest.fixture
    def proactive_assistant(self, mock_dependencies):
        """Create ProactiveAssistant instance with mocked dependencies"""
        return ProactiveAssistant(**mock_dependencies)
    
    @pytest.fixture
    def sample_user_context(self):
        """Create sample user context for testing"""
        return UserContext(
            user_id="test_user",
            current_activity="coding",
            active_applications=["vscode", "terminal"],
            current_files=["/home/user/project/main.py"],
            recent_interactions=[
                Interaction(
                    interaction_id="int1",
                    user_id="test_user",
                    interaction_type=InteractionType.QUERY,
                    content="How to fix syntax error in Python?",
                    timestamp=datetime.now() - timedelta(minutes=5)
                ),
                Interaction(
                    interaction_id="int2",
                    user_id="test_user",
                    interaction_type=InteractionType.QUERY,
                    content="Python debugging tips",
                    timestamp=datetime.now() - timedelta(minutes=10)
                )
            ],
            preferences={},
            knowledge_state={},
            task_context={}
        )
    
    @pytest.fixture
    def sample_behavior_patterns(self):
        """Create sample behavior patterns for testing"""
        return [
            BehaviorPattern(
                pattern_id="pattern1",
                user_id="test_user",
                pattern_type="task_sequence",
                pattern_data={
                    "sequence": ["git add .", "git commit -m", "git push"],
                    "name": "Git workflow"
                },
                confidence=0.9,
                frequency=5,
                first_detected=datetime.now() - timedelta(days=7),
                last_updated=datetime.now()
            ),
            BehaviorPattern(
                pattern_id="pattern2",
                user_id="test_user",
                pattern_type="workflow",
                pattern_data={
                    "steps": ["open file", "edit", "save", "test", "commit"],
                    "name": "Development workflow"
                },
                confidence=0.8,
                frequency=10,
                first_detected=datetime.now() - timedelta(days=14),
                last_updated=datetime.now()
            )
        ]
    
    @pytest.fixture
    def sample_screen_context(self):
        """Create sample screen context for testing"""
        return ScreenContext(
            active_application="vscode",
            window_title="main.py - Visual Studio Code",
            visible_text="SyntaxError: invalid syntax on line 42",
            ui_elements=[],
            detected_actions=[],
            context_summary="User editing Python file with syntax error",
            timestamp=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_identify_opportunities_basic(self, proactive_assistant, mock_dependencies, sample_user_context):
        """Test basic opportunity identification"""
        # Setup mocks
        mock_dependencies['context_manager'].get_context.return_value = sample_user_context
        mock_dependencies['learning_engine'].get_behavior_patterns.return_value = []
        mock_dependencies['screen_monitor'].get_current_context.return_value = None
        mock_dependencies['task_manager'].get_upcoming_deadlines.return_value = []
        
        # Test
        opportunities = await proactive_assistant.identify_opportunities("test_user")
        
        # Verify
        assert isinstance(opportunities, list)
        mock_dependencies['context_manager'].get_context.assert_called_once_with("test_user")
    
    @pytest.mark.asyncio
    async def test_identify_automation_opportunities(
        self, 
        proactive_assistant, 
        mock_dependencies, 
        sample_user_context,
        sample_behavior_patterns
    ):
        """Test automation opportunity identification"""
        # Setup mocks
        mock_dependencies['context_manager'].get_context.return_value = sample_user_context
        mock_dependencies['learning_engine'].get_behavior_patterns.return_value = sample_behavior_patterns
        mock_dependencies['screen_monitor'].get_current_context.return_value = None
        mock_dependencies['task_manager'].get_upcoming_deadlines.return_value = []
        
        # Test
        opportunities = await proactive_assistant.identify_opportunities("test_user")
        
        # Verify automation opportunity was created
        automation_opportunities = [
            opp for opp in opportunities 
            if opp.opportunity_type == OpportunityType.AUTOMATION
        ]
        assert len(automation_opportunities) > 0
        
        automation_opp = automation_opportunities[0]
        assert "Git workflow" in automation_opp.title
        assert automation_opp.confidence == 0.9
        assert "automation" in automation_opp.context_data
    
    @pytest.mark.asyncio
    async def test_identify_workflow_optimizations(
        self, 
        proactive_assistant, 
        mock_dependencies, 
        sample_user_context,
        sample_behavior_patterns
    ):
        """Test workflow optimization identification"""
        # Setup mocks
        mock_dependencies['context_manager'].get_context.return_value = sample_user_context
        mock_dependencies['learning_engine'].get_behavior_patterns.return_value = sample_behavior_patterns
        mock_dependencies['screen_monitor'].get_current_context.return_value = None
        mock_dependencies['task_manager'].get_upcoming_deadlines.return_value = []
        
        # Test
        opportunities = await proactive_assistant.identify_opportunities("test_user")
        
        # Verify workflow optimization opportunity was created
        workflow_opportunities = [
            opp for opp in opportunities 
            if opp.opportunity_type == OpportunityType.WORKFLOW_OPTIMIZATION
        ]
        assert len(workflow_opportunities) > 0
        
        workflow_opp = workflow_opportunities[0]
        assert "workflow" in workflow_opp.title.lower()
        assert "optimization" in workflow_opp.context_data
    
    @pytest.mark.asyncio
    async def test_detect_errors_and_issues(
        self, 
        proactive_assistant, 
        mock_dependencies, 
        sample_user_context,
        sample_screen_context
    ):
        """Test error detection from screen content"""
        # Setup mocks
        mock_dependencies['context_manager'].get_context.return_value = sample_user_context
        mock_dependencies['learning_engine'].get_behavior_patterns.return_value = []
        mock_dependencies['screen_monitor'].get_current_context.return_value = sample_screen_context
        mock_dependencies['task_manager'].get_upcoming_deadlines.return_value = []
        
        # Test
        opportunities = await proactive_assistant.identify_opportunities("test_user")
        
        # Verify error detection opportunity was created
        error_opportunities = [
            opp for opp in opportunities 
            if opp.opportunity_type == OpportunityType.ERROR_DETECTION
        ]
        assert len(error_opportunities) > 0
        
        error_opp = error_opportunities[0]
        assert "syntax" in error_opp.title.lower()
        assert error_opp.priority in [SuggestionPriority.HIGH, SuggestionPriority.MEDIUM]
        assert "error" in error_opp.context_data
    
    @pytest.mark.asyncio
    async def test_recommend_learning_resources(
        self, 
        proactive_assistant, 
        mock_dependencies, 
        sample_user_context
    ):
        """Test learning resource recommendations"""
        # Setup mocks
        mock_dependencies['context_manager'].get_context.return_value = sample_user_context
        mock_dependencies['learning_engine'].get_behavior_patterns.return_value = []
        mock_dependencies['screen_monitor'].get_current_context.return_value = None
        mock_dependencies['task_manager'].get_upcoming_deadlines.return_value = []
        
        # Test
        opportunities = await proactive_assistant.identify_opportunities("test_user")
        
        # Verify learning resource opportunities were created
        learning_opportunities = [
            opp for opp in opportunities 
            if opp.opportunity_type == OpportunityType.LEARNING_RESOURCE
        ]
        
        # Should have learning resources for Python based on user's questions
        assert len(learning_opportunities) > 0
        
        learning_opp = learning_opportunities[0]
        assert "python" in learning_opp.title.lower() or "learn" in learning_opp.title.lower()
        assert "resource" in learning_opp.context_data
    
    @pytest.mark.asyncio
    async def test_provide_contextual_help(
        self, 
        proactive_assistant, 
        mock_dependencies, 
        sample_user_context,
        sample_screen_context
    ):
        """Test contextual help provision"""
        # Setup mocks
        mock_dependencies['context_manager'].get_context.return_value = sample_user_context
        mock_dependencies['learning_engine'].get_behavior_patterns.return_value = []
        mock_dependencies['screen_monitor'].get_current_context.return_value = sample_screen_context
        mock_dependencies['task_manager'].get_upcoming_deadlines.return_value = []
        
        # Test
        opportunities = await proactive_assistant.identify_opportunities("test_user")
        
        # Verify contextual help opportunities were created
        help_opportunities = [
            opp for opp in opportunities 
            if opp.opportunity_type == OpportunityType.CONTEXTUAL_HELP
        ]
        
        # Should have VS Code specific help
        assert len(help_opportunities) > 0
        
        help_opp = help_opportunities[0]
        assert "vscode" in help_opp.title.lower() or "code" in help_opp.title.lower()
    
    @pytest.mark.asyncio
    async def test_check_deadline_reminders(self, proactive_assistant, mock_dependencies, sample_user_context):
        """Test deadline reminder functionality"""
        # Create mock deadline
        mock_deadline = Mock()
        mock_deadline.task_id = "task1"
        mock_deadline.task_name = "Complete project"
        mock_deadline.due_date = datetime.now() + timedelta(hours=12)  # Due in 12 hours
        mock_deadline.__dict__ = {
            "task_id": "task1",
            "task_name": "Complete project",
            "due_date": mock_deadline.due_date
        }
        
        # Setup mocks
        mock_dependencies['context_manager'].get_context.return_value = sample_user_context
        mock_dependencies['learning_engine'].get_behavior_patterns.return_value = []
        mock_dependencies['screen_monitor'].get_current_context.return_value = None
        mock_dependencies['task_manager'].get_upcoming_deadlines.return_value = [mock_deadline]
        
        # Test
        opportunities = await proactive_assistant.identify_opportunities("test_user")
        
        # Verify deadline reminder opportunity was created
        deadline_opportunities = [
            opp for opp in opportunities 
            if opp.opportunity_type == OpportunityType.DEADLINE_REMINDER
        ]
        assert len(deadline_opportunities) > 0
        
        deadline_opp = deadline_opportunities[0]
        assert "Complete project" in deadline_opp.title
        assert deadline_opp.priority == SuggestionPriority.URGENT
    
    def test_filter_and_prioritize(self, proactive_assistant):
        """Test opportunity filtering and prioritization"""
        # Create test opportunities with different priorities and confidence
        opportunities = [
            ProactiveOpportunity(
                opportunity_id="opp1",
                user_id="test_user",
                opportunity_type=OpportunityType.AUTOMATION,
                title="Low priority automation",
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.LOW,
                confidence=0.5,  # Below threshold
                context_data={},
                created_at=datetime.now()
            ),
            ProactiveOpportunity(
                opportunity_id="opp2",
                user_id="test_user",
                opportunity_type=OpportunityType.ERROR_DETECTION,
                title="High priority error",
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.HIGH,
                confidence=0.9,
                context_data={},
                created_at=datetime.now()
            ),
            ProactiveOpportunity(
                opportunity_id="opp3",
                user_id="test_user",
                opportunity_type=OpportunityType.DEADLINE_REMINDER,
                title="Urgent deadline",
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.URGENT,
                confidence=1.0,
                context_data={},
                created_at=datetime.now()
            ),
            ProactiveOpportunity(
                opportunity_id="opp4",
                user_id="test_user",
                opportunity_type=OpportunityType.LEARNING_RESOURCE,
                title="Expired opportunity",
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.MEDIUM,
                confidence=0.8,
                context_data={},
                created_at=datetime.now(),
                expires_at=datetime.now() - timedelta(minutes=1)  # Expired
            )
        ]
        
        # Test filtering and prioritization
        filtered = proactive_assistant._filter_and_prioritize(opportunities, "test_user")
        
        # Should filter out low confidence and expired opportunities
        assert len(filtered) == 2
        
        # Should be sorted by priority (URGENT first, then HIGH)
        assert filtered[0].priority == SuggestionPriority.URGENT
        assert filtered[1].priority == SuggestionPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_process_user_response(self, proactive_assistant, mock_dependencies):
        """Test processing user response to opportunities"""
        # Create test opportunity
        opportunity = ProactiveOpportunity(
            opportunity_id="test_opp",
            user_id="test_user",
            opportunity_type=OpportunityType.AUTOMATION,
            title="Test automation",
            description="Test",
            suggested_action="Test",
            priority=SuggestionPriority.MEDIUM,
            confidence=0.8,
            context_data={},
            created_at=datetime.now()
        )
        
        proactive_assistant.active_opportunities["test_opp"] = opportunity
        
        # Test processing response
        await proactive_assistant.process_user_response(
            "test_user", 
            "test_opp", 
            "Yes, this is helpful", 
            action_taken=True
        )
        
        # Verify response was recorded
        assert "test_user" in proactive_assistant.user_responses
        assert "test_opp" in proactive_assistant.user_responses["test_user"]
        
        response_data = proactive_assistant.user_responses["test_user"]["test_opp"]
        assert response_data["response"] == "Yes, this is helpful"
        assert response_data["action_taken"] is True
        
        # Verify opportunity moved to history
        assert "test_opp" not in proactive_assistant.active_opportunities
        assert opportunity in proactive_assistant.opportunity_history
        
        # Verify learning engine was called
        mock_dependencies['learning_engine'].process_feedback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_active_opportunities(self, proactive_assistant):
        """Test getting active opportunities for a user"""
        # Create test opportunities
        active_opp = ProactiveOpportunity(
            opportunity_id="active_opp",
            user_id="test_user",
            opportunity_type=OpportunityType.AUTOMATION,
            title="Active opportunity",
            description="Test",
            suggested_action="Test",
            priority=SuggestionPriority.MEDIUM,
            confidence=0.8,
            context_data={},
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        expired_opp = ProactiveOpportunity(
            opportunity_id="expired_opp",
            user_id="test_user",
            opportunity_type=OpportunityType.LEARNING_RESOURCE,
            title="Expired opportunity",
            description="Test",
            suggested_action="Test",
            priority=SuggestionPriority.LOW,
            confidence=0.7,
            context_data={},
            created_at=datetime.now(),
            expires_at=datetime.now() - timedelta(minutes=1)
        )
        
        other_user_opp = ProactiveOpportunity(
            opportunity_id="other_user_opp",
            user_id="other_user",
            opportunity_type=OpportunityType.ERROR_DETECTION,
            title="Other user opportunity",
            description="Test",
            suggested_action="Test",
            priority=SuggestionPriority.HIGH,
            confidence=0.9,
            context_data={},
            created_at=datetime.now()
        )
        
        proactive_assistant.active_opportunities.update({
            "active_opp": active_opp,
            "expired_opp": expired_opp,
            "other_user_opp": other_user_opp
        })
        
        # Test getting active opportunities
        active_opportunities = await proactive_assistant.get_active_opportunities("test_user")
        
        # Should only return active, non-expired opportunities for the user
        assert len(active_opportunities) == 1
        assert active_opportunities[0].opportunity_id == "active_opp"
        
        # Expired opportunity should be moved to history
        assert expired_opp in proactive_assistant.opportunity_history
        assert "expired_opp" not in proactive_assistant.active_opportunities
    
    @pytest.mark.asyncio
    async def test_dismiss_opportunity(self, proactive_assistant):
        """Test dismissing an opportunity"""
        # Create test opportunity
        opportunity = ProactiveOpportunity(
            opportunity_id="dismiss_test",
            user_id="test_user",
            opportunity_type=OpportunityType.CONTEXTUAL_HELP,
            title="Test help",
            description="Test",
            suggested_action="Test",
            priority=SuggestionPriority.LOW,
            confidence=0.6,
            context_data={},
            created_at=datetime.now()
        )
        
        proactive_assistant.active_opportunities["dismiss_test"] = opportunity
        
        # Test dismissing opportunity
        result = await proactive_assistant.dismiss_opportunity("test_user", "dismiss_test")
        
        # Verify dismissal was successful
        assert result is True
        assert "dismiss_test" not in proactive_assistant.active_opportunities
        assert opportunity in proactive_assistant.opportunity_history
        
        # Verify dismissal was recorded as response
        assert "test_user" in proactive_assistant.user_responses
        assert "dismiss_test" in proactive_assistant.user_responses["test_user"]
        assert proactive_assistant.user_responses["test_user"]["dismiss_test"]["response"] == "dismissed"
    
    def test_get_opportunity_statistics(self, proactive_assistant):
        """Test getting opportunity statistics"""
        # Setup test data
        test_opportunities = [
            ProactiveOpportunity(
                opportunity_id="stat_opp1",
                user_id="test_user",
                opportunity_type=OpportunityType.AUTOMATION,
                title="Test automation",
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.MEDIUM,
                confidence=0.8,
                context_data={},
                created_at=datetime.now()
            ),
            ProactiveOpportunity(
                opportunity_id="stat_opp2",
                user_id="test_user",
                opportunity_type=OpportunityType.ERROR_DETECTION,
                title="Test error",
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.HIGH,
                confidence=0.9,
                context_data={},
                created_at=datetime.now()
            )
        ]
        
        proactive_assistant.opportunity_history.extend(test_opportunities)
        
        # Setup user responses
        proactive_assistant.user_responses["test_user"] = {
            "stat_opp1": {"response": "helpful", "action_taken": True},
            "stat_opp2": {"response": "not relevant", "action_taken": False}
        }
        
        # Test getting statistics
        stats = proactive_assistant.get_opportunity_statistics("test_user")
        
        # Verify statistics
        assert stats["total_opportunities"] == 2
        assert stats["responded_opportunities"] == 2
        assert stats["actions_taken"] == 1
        assert stats["response_rate"] == 1.0
        assert stats["action_rate"] == 0.5
        assert stats["opportunity_types"]["automation"] == 1
        assert stats["opportunity_types"]["error_detection"] == 1
    
    def test_estimate_time_savings(self, proactive_assistant):
        """Test time savings estimation"""
        pattern = BehaviorPattern(
            pattern_id="time_test",
            user_id="test_user",
            pattern_type="task_sequence",
            pattern_data={"sequence": ["step1", "step2", "step3"]},
            confidence=0.8,
            frequency=10,
            first_detected=datetime.now(),
            last_updated=datetime.now()
        )
        
        time_savings = proactive_assistant._estimate_time_savings(pattern)
        
        # Should be frequency * base_time (5 minutes)
        assert time_savings == 50  # 10 * 5
    
    def test_generate_automation_script(self, proactive_assistant):
        """Test automation script generation"""
        pattern = BehaviorPattern(
            pattern_id="script_test",
            user_id="test_user",
            pattern_type="task_sequence",
            pattern_data={"sequence": ["git add .", "git commit -m 'update'", "git push"]},
            confidence=0.9,
            frequency=5,
            first_detected=datetime.now(),
            last_updated=datetime.now()
        )
        
        script = proactive_assistant._generate_automation_script(pattern)
        
        assert "# Automation script suggestion" in script
        assert "git add ." in script
        assert "git commit" in script
        assert "git push" in script
    
    def test_optimize_workflow(self, proactive_assistant):
        """Test workflow optimization"""
        # Test workflow with duplicate steps
        workflow = ["step1", "step2", "step2", "step3", "step3", "step4"]
        
        optimized = proactive_assistant._optimize_workflow(workflow)
        
        # Should remove consecutive duplicates
        expected = ["step1", "step2", "step3", "step4"]
        assert optimized == expected
    
    def test_get_error_solutions(self, proactive_assistant):
        """Test error solution suggestions"""
        solutions = proactive_assistant._get_error_solutions("file_not_found")
        
        assert len(solutions) > 0
        assert any("file path" in solution.lower() for solution in solutions)
        
        # Test unknown error type
        unknown_solutions = proactive_assistant._get_error_solutions("unknown_error")
        assert "documentation" in unknown_solutions[0].lower()
    
    def test_get_error_severity(self, proactive_assistant):
        """Test error severity classification"""
        assert proactive_assistant._get_error_severity("permission_denied") == "high"
        assert proactive_assistant._get_error_severity("memory_error") == "high"
        assert proactive_assistant._get_error_severity("syntax_error") == "medium"
        assert proactive_assistant._get_error_severity("unknown_error") == "medium"
    
    @pytest.mark.asyncio
    async def test_identify_skill_gaps(self, proactive_assistant, sample_user_context):
        """Test skill gap identification"""
        skill_gaps = await proactive_assistant._identify_skill_gaps("test_user", sample_user_context)
        
        # Should identify Python as a skill gap based on user questions
        assert "python" in skill_gaps
    
    def test_find_learning_resources(self, proactive_assistant, sample_user_context):
        """Test finding learning resources for skills"""
        resources = proactive_assistant._find_learning_resources("python", sample_user_context)
        
        assert len(resources) > 0
        assert any("python" in resource.title.lower() for resource in resources)
        assert all(isinstance(resource, LearningResource) for resource in resources)
    
    @pytest.mark.asyncio
    async def test_generate_contextual_help(self, proactive_assistant, sample_screen_context, sample_user_context):
        """Test contextual help generation"""
        help_suggestions = await proactive_assistant._generate_contextual_help(
            sample_screen_context, 
            sample_user_context
        )
        
        assert len(help_suggestions) > 0
        
        # Should have VS Code specific suggestions
        vscode_suggestions = [s for s in help_suggestions if "vscode" in s["title"].lower()]
        assert len(vscode_suggestions) > 0
    
    def test_remove_duplicate_opportunities(self, proactive_assistant):
        """Test duplicate opportunity removal"""
        opportunities = [
            ProactiveOpportunity(
                opportunity_id="dup1",
                user_id="test_user",
                opportunity_type=OpportunityType.AUTOMATION,
                title="Test Automation",
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.MEDIUM,
                confidence=0.8,
                context_data={},
                created_at=datetime.now()
            ),
            ProactiveOpportunity(
                opportunity_id="dup2",
                user_id="test_user",
                opportunity_type=OpportunityType.AUTOMATION,
                title="Test Automation",  # Same title
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.MEDIUM,
                confidence=0.7,
                context_data={},
                created_at=datetime.now()
            ),
            ProactiveOpportunity(
                opportunity_id="unique",
                user_id="test_user",
                opportunity_type=OpportunityType.ERROR_DETECTION,
                title="Unique Error",
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.HIGH,
                confidence=0.9,
                context_data={},
                created_at=datetime.now()
            )
        ]
        
        unique_opportunities = proactive_assistant._remove_duplicate_opportunities(opportunities)
        
        # Should remove one duplicate
        assert len(unique_opportunities) == 2
        
        titles = [opp.title for opp in unique_opportunities]
        assert "Test Automation" in titles
        assert "Unique Error" in titles
    
    @pytest.mark.asyncio
    async def test_error_handling_in_identify_opportunities(self, proactive_assistant, mock_dependencies):
        """Test error handling in opportunity identification"""
        # Setup mock to raise exception
        mock_dependencies['context_manager'].get_context.side_effect = Exception("Test error")
        
        # Test that method handles errors gracefully
        opportunities = await proactive_assistant.identify_opportunities("test_user")
        
        # Should return empty list on error
        assert opportunities == []
    
    @pytest.mark.asyncio
    async def test_no_context_handling(self, proactive_assistant, mock_dependencies):
        """Test handling when no user context is available"""
        # Setup mock to return None
        mock_dependencies['context_manager'].get_context.return_value = None
        
        # Test
        opportunities = await proactive_assistant.identify_opportunities("test_user")
        
        # Should return empty list when no context
        assert opportunities == []


if __name__ == "__main__":
    pytest.main([__file__])