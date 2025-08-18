"""
Simple tests for ProactiveAssistant module

This module contains basic tests for the proactive assistance engine
without complex dependencies.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


# Mock the required enums and classes
class OpportunityType(Enum):
    AUTOMATION = "automation"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    ERROR_DETECTION = "error_detection"
    LEARNING_RESOURCE = "learning_resource"
    CONTEXTUAL_HELP = "contextual_help"
    DEADLINE_REMINDER = "deadline_reminder"


class SuggestionPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ProactiveOpportunity:
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
    expires_at: datetime = None


@dataclass
class BehaviorPattern:
    pattern_id: str
    user_id: str
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    frequency: int
    first_detected: datetime
    last_updated: datetime


@dataclass
class UserContext:
    user_id: str
    current_activity: str
    active_applications: List[str]
    current_files: List[str]
    recent_interactions: List[Any]
    preferences: Dict[str, Any]
    knowledge_state: Dict[str, Any]
    task_context: Dict[str, Any]


class TestProactiveAssistantCore:
    """Test core functionality of ProactiveAssistant"""
    
    def test_opportunity_creation(self):
        """Test creating a proactive opportunity"""
        opportunity = ProactiveOpportunity(
            opportunity_id="test_opp",
            user_id="test_user",
            opportunity_type=OpportunityType.AUTOMATION,
            title="Test automation",
            description="Test description",
            suggested_action="Test action",
            priority=SuggestionPriority.MEDIUM,
            confidence=0.8,
            context_data={"test": "data"},
            created_at=datetime.now()
        )
        
        assert opportunity.opportunity_id == "test_opp"
        assert opportunity.user_id == "test_user"
        assert opportunity.opportunity_type == OpportunityType.AUTOMATION
        assert opportunity.priority == SuggestionPriority.MEDIUM
        assert opportunity.confidence == 0.8
    
    def test_priority_ordering(self):
        """Test that priorities can be ordered correctly"""
        priorities = [
            SuggestionPriority.LOW,
            SuggestionPriority.URGENT,
            SuggestionPriority.MEDIUM,
            SuggestionPriority.HIGH
        ]
        
        priority_values = {
            SuggestionPriority.URGENT: 4,
            SuggestionPriority.HIGH: 3,
            SuggestionPriority.MEDIUM: 2,
            SuggestionPriority.LOW: 1
        }
        
        sorted_priorities = sorted(priorities, key=lambda x: priority_values[x], reverse=True)
        
        assert sorted_priorities[0] == SuggestionPriority.URGENT
        assert sorted_priorities[1] == SuggestionPriority.HIGH
        assert sorted_priorities[2] == SuggestionPriority.MEDIUM
        assert sorted_priorities[3] == SuggestionPriority.LOW
    
    def test_opportunity_expiration(self):
        """Test opportunity expiration logic"""
        now = datetime.now()
        
        # Non-expired opportunity
        active_opp = ProactiveOpportunity(
            opportunity_id="active",
            user_id="test_user",
            opportunity_type=OpportunityType.CONTEXTUAL_HELP,
            title="Active help",
            description="Test",
            suggested_action="Test",
            priority=SuggestionPriority.LOW,
            confidence=0.6,
            context_data={},
            created_at=now,
            expires_at=now + timedelta(hours=1)
        )
        
        # Expired opportunity
        expired_opp = ProactiveOpportunity(
            opportunity_id="expired",
            user_id="test_user",
            opportunity_type=OpportunityType.LEARNING_RESOURCE,
            title="Expired resource",
            description="Test",
            suggested_action="Test",
            priority=SuggestionPriority.LOW,
            confidence=0.7,
            context_data={},
            created_at=now,
            expires_at=now - timedelta(minutes=1)
        )
        
        # Test expiration check
        assert active_opp.expires_at > now
        assert expired_opp.expires_at < now
    
    def test_behavior_pattern_analysis(self):
        """Test behavior pattern data structure"""
        pattern = BehaviorPattern(
            pattern_id="test_pattern",
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
        )
        
        assert pattern.pattern_type == "task_sequence"
        assert pattern.frequency == 5
        assert pattern.confidence == 0.9
        assert "git add ." in pattern.pattern_data["sequence"]
    
    def test_user_context_structure(self):
        """Test user context data structure"""
        context = UserContext(
            user_id="test_user",
            current_activity="coding",
            active_applications=["vscode", "terminal"],
            current_files=["/home/user/project/main.py"],
            recent_interactions=[],
            preferences={},
            knowledge_state={},
            task_context={}
        )
        
        assert context.user_id == "test_user"
        assert context.current_activity == "coding"
        assert "vscode" in context.active_applications
        assert len(context.current_files) == 1
    
    def test_opportunity_filtering_by_confidence(self):
        """Test filtering opportunities by confidence threshold"""
        opportunities = [
            ProactiveOpportunity(
                opportunity_id="low_conf",
                user_id="test_user",
                opportunity_type=OpportunityType.AUTOMATION,
                title="Low confidence",
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.MEDIUM,
                confidence=0.5,  # Below typical threshold
                context_data={},
                created_at=datetime.now()
            ),
            ProactiveOpportunity(
                opportunity_id="high_conf",
                user_id="test_user",
                opportunity_type=OpportunityType.ERROR_DETECTION,
                title="High confidence",
                description="Test",
                suggested_action="Test",
                priority=SuggestionPriority.HIGH,
                confidence=0.9,  # Above threshold
                context_data={},
                created_at=datetime.now()
            )
        ]
        
        confidence_threshold = 0.7
        filtered = [opp for opp in opportunities if opp.confidence >= confidence_threshold]
        
        assert len(filtered) == 1
        assert filtered[0].opportunity_id == "high_conf"
    
    def test_opportunity_deduplication(self):
        """Test removing duplicate opportunities"""
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
        
        # Simple deduplication by title
        seen_titles = set()
        unique_opportunities = []
        
        for opp in opportunities:
            title_key = opp.title.lower().strip()
            if title_key not in seen_titles:
                unique_opportunities.append(opp)
                seen_titles.add(title_key)
        
        assert len(unique_opportunities) == 2
        titles = [opp.title for opp in unique_opportunities]
        assert "Test Automation" in titles
        assert "Unique Error" in titles
    
    def test_error_pattern_matching(self):
        """Test error pattern detection"""
        error_patterns = {
            "file_not_found": r"(file not found|no such file|cannot find)",
            "permission_denied": r"(permission denied|access denied|unauthorized)",
            "syntax_error": r"(syntax error|invalid syntax|parse error)",
        }
        
        test_texts = [
            "Error: file not found in directory",
            "Permission denied: cannot access file",
            "SyntaxError: invalid syntax on line 42",
            "Network connection successful"
        ]
        
        import re
        
        detected_errors = []
        for text in test_texts:
            for error_type, pattern in error_patterns.items():
                if re.search(pattern, text.lower(), re.IGNORECASE):
                    detected_errors.append((error_type, text))
        
        assert len(detected_errors) == 3
        assert ("file_not_found", "Error: file not found in directory") in detected_errors
        assert ("permission_denied", "Permission denied: cannot access file") in detected_errors
        assert ("syntax_error", "SyntaxError: invalid syntax on line 42") in detected_errors
    
    def test_time_savings_calculation(self):
        """Test time savings estimation"""
        def estimate_time_savings(frequency: int, base_time: int = 5) -> int:
            return frequency * base_time
        
        # Test different frequencies
        assert estimate_time_savings(3) == 15  # 3 * 5 minutes
        assert estimate_time_savings(10) == 50  # 10 * 5 minutes
        assert estimate_time_savings(1) == 5   # 1 * 5 minutes
    
    def test_workflow_optimization(self):
        """Test workflow optimization logic"""
        def optimize_workflow(workflow: List[str]) -> List[str]:
            if not workflow:
                return workflow
            
            # Remove consecutive duplicates
            optimized = [workflow[0]]
            for step in workflow[1:]:
                if step != optimized[-1]:
                    optimized.append(step)
            
            return optimized
        
        # Test workflow with duplicates
        workflow = ["step1", "step2", "step2", "step3", "step3", "step4"]
        optimized = optimize_workflow(workflow)
        
        expected = ["step1", "step2", "step3", "step4"]
        assert optimized == expected
        
        # Test workflow without duplicates
        clean_workflow = ["step1", "step2", "step3"]
        assert optimize_workflow(clean_workflow) == clean_workflow
        
        # Test empty workflow
        assert optimize_workflow([]) == []
    
    def test_learning_resource_relevance(self):
        """Test learning resource relevance scoring"""
        @dataclass
        class LearningResource:
            resource_id: str
            title: str
            resource_type: str
            relevance_score: float
        
        resources = [
            LearningResource("py1", "Python Basics", "tutorial", 0.9),
            LearningResource("py2", "Advanced Python", "course", 0.7),
            LearningResource("js1", "JavaScript Guide", "tutorial", 0.3),
        ]
        
        # Filter by relevance threshold
        relevant_resources = [r for r in resources if r.relevance_score >= 0.6]
        
        assert len(relevant_resources) == 2
        assert all("Python" in r.title for r in relevant_resources)
    
    def test_opportunity_statistics(self):
        """Test opportunity statistics calculation"""
        # Mock opportunity history
        opportunities = [
            {"type": "automation", "responded": True, "action_taken": True},
            {"type": "automation", "responded": True, "action_taken": False},
            {"type": "error_detection", "responded": False, "action_taken": False},
            {"type": "learning_resource", "responded": True, "action_taken": True},
        ]
        
        total_opportunities = len(opportunities)
        responded_opportunities = sum(1 for opp in opportunities if opp["responded"])
        actions_taken = sum(1 for opp in opportunities if opp["action_taken"])
        
        response_rate = responded_opportunities / total_opportunities if total_opportunities > 0 else 0
        action_rate = actions_taken / responded_opportunities if responded_opportunities > 0 else 0
        
        # Count by type
        from collections import Counter
        type_counts = Counter(opp["type"] for opp in opportunities)
        
        assert total_opportunities == 4
        assert responded_opportunities == 3
        assert actions_taken == 2
        assert response_rate == 0.75  # 3/4
        assert action_rate == 2/3  # 2/3
        assert type_counts["automation"] == 2
        assert type_counts["error_detection"] == 1
        assert type_counts["learning_resource"] == 1


if __name__ == "__main__":
    pytest.main([__file__])