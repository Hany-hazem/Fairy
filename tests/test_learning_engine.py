"""
Tests for Learning Engine

This module contains comprehensive tests for the LearningEngine class,
including pattern recognition, feedback processing, and model adaptation.
"""

import pytest
import pytest_asyncio
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.learning_engine import (
    LearningEngine, BehaviorPattern, UserFeedback, LearningModel
)
from app.personal_assistant_models import (
    UserContext, Interaction, InteractionType, UserPreferences,
    TaskContext, KnowledgeState
)
from app.user_context_manager import UserContextManager


class TestLearningEngine:
    """Test cases for LearningEngine"""
    
    @pytest_asyncio.fixture
    async def temp_db(self):
        """Create a temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        db_path = temp_file.name
        temp_file.close()
        
        try:
            yield db_path
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    @pytest_asyncio.fixture
    async def learning_engine(self, temp_db):
        """Create a LearningEngine instance for testing"""
        context_manager = Mock(spec=UserContextManager)
        engine = LearningEngine(db_path=temp_db, context_manager=context_manager)
        return engine
    
    @pytest_asyncio.fixture
    async def sample_interaction(self):
        """Create a sample interaction for testing"""
        return Interaction(
            id="test_interaction_1",
            user_id="test_user",
            interaction_type=InteractionType.QUERY,
            content="How do I write a Python function?",
            response="Here's how to write a Python function...",
            timestamp=datetime.now(),
            context_data={"category": "programming", "language": "python"},
            feedback_score=4.5
        )
    
    @pytest_asyncio.fixture
    async def sample_user_context(self):
        """Create a sample user context for testing"""
        return UserContext(
            user_id="test_user",
            current_activity="coding",
            active_applications=["vscode", "terminal"],
            current_files=["main.py", "test.py"],
            preferences=UserPreferences(
                user_id="test_user",
                language="en",
                learning_preferences={"adaptive_suggestions": True}
            ),
            task_context=TaskContext(
                current_tasks=["Implement user authentication", "Write unit tests"],
                active_projects=["web_app"]
            ),
            knowledge_state=KnowledgeState(
                expertise_areas={"python": 0.8, "web_development": 0.6}
            )
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, temp_db):
        """Test LearningEngine initialization"""
        engine = LearningEngine(db_path=temp_db)
        
        assert engine.db_path == temp_db
        assert engine.context_manager is not None
        assert engine.min_interactions_for_pattern == 5
        assert engine.pattern_confidence_threshold == 0.7
        assert isinstance(engine._user_models, dict)
    
    @pytest.mark.asyncio
    async def test_learn_from_interaction(self, learning_engine, sample_interaction):
        """Test learning from user interactions"""
        # Mock the internal methods
        learning_engine._get_user_model = AsyncMock(return_value=LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        ))
        learning_engine._analyze_interaction_patterns = AsyncMock(return_value=[])
        learning_engine._update_behavior_pattern = AsyncMock()
        learning_engine._update_user_model = AsyncMock()
        learning_engine._log_learning_event = AsyncMock()
        
        await learning_engine.learn_from_interaction(sample_interaction)
        
        # Verify methods were called
        learning_engine._get_user_model.assert_called_once_with("test_user")
        learning_engine._analyze_interaction_patterns.assert_called_once_with(sample_interaction)
        learning_engine._update_user_model.assert_called_once()
        learning_engine._log_learning_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_feedback(self, learning_engine):
        """Test processing user feedback"""
        feedback = UserFeedback(
            feedback_id="feedback_1",
            user_id="test_user",
            interaction_id="interaction_1",
            feedback_type="rating",
            feedback_value=4.5,
            timestamp=datetime.now(),
            context_data={"interaction_type": "query"}
        )
        
        # Mock the internal methods
        learning_engine._store_feedback = AsyncMock()
        learning_engine._get_user_model = AsyncMock(return_value=LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        ))
        learning_engine._apply_feedback_to_model = AsyncMock()
        learning_engine._update_preferences_from_feedback = AsyncMock()
        learning_engine._log_learning_event = AsyncMock()
        
        await learning_engine.process_feedback(feedback)
        
        # Verify methods were called
        learning_engine._store_feedback.assert_called_once_with(feedback)
        learning_engine._get_user_model.assert_called_once_with("test_user")
        learning_engine._apply_feedback_to_model.assert_called_once()
        learning_engine._update_preferences_from_feedback.assert_called_once_with(feedback)
        learning_engine._log_learning_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_personalized_suggestions(self, learning_engine, sample_user_context):
        """Test generating personalized suggestions"""
        # Mock the internal methods
        learning_engine._get_user_model = AsyncMock(return_value=LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={"suggestion_time_based": 0.8},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        ))
        learning_engine._get_time_based_suggestions = AsyncMock(return_value=[
            {"type": "time_based", "suggestion": "Time to code!", "confidence": 0.8}
        ])
        learning_engine._get_task_based_suggestions = AsyncMock(return_value=[
            {"type": "task_based", "suggestion": "Work on authentication", "confidence": 0.9}
        ])
        learning_engine._get_pattern_based_suggestions = AsyncMock(return_value=[])
        learning_engine._rank_suggestions = AsyncMock(return_value=[
            {"type": "task_based", "suggestion": "Work on authentication", "confidence": 0.9, "score": 0.9},
            {"type": "time_based", "suggestion": "Time to code!", "confidence": 0.8, "score": 0.8}
        ])
        
        suggestions = await learning_engine.get_personalized_suggestions(sample_user_context)
        
        assert len(suggestions) == 2
        assert suggestions[0]["type"] == "task_based"
        assert suggestions[1]["type"] == "time_based"
        
        # Verify methods were called
        learning_engine._get_user_model.assert_called_once_with("test_user")
        learning_engine._get_time_based_suggestions.assert_called_once()
        learning_engine._get_task_based_suggestions.assert_called_once()
        learning_engine._get_pattern_based_suggestions.assert_called_once()
        learning_engine._rank_suggestions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_adapt_to_feedback_rating(self, learning_engine):
        """Test adapting to rating feedback"""
        feedback = UserFeedback(
            feedback_id="feedback_1",
            user_id="test_user",
            interaction_id="interaction_1",
            feedback_type="rating",
            feedback_value=4.5,
            timestamp=datetime.now()
        )
        
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={"test_preference": 0.5},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        # Mock the internal methods
        learning_engine._get_user_model = AsyncMock(return_value=model)
        learning_engine._adjust_preference_weights = AsyncMock()
        learning_engine._save_user_model = AsyncMock()
        
        await learning_engine.adapt_to_feedback(feedback)
        
        # Verify model was updated
        assert model.model_version == 2
        learning_engine._adjust_preference_weights.assert_called_once()
        learning_engine._save_user_model.assert_called_once_with(model)
    
    @pytest.mark.asyncio
    async def test_adapt_to_feedback_correction(self, learning_engine):
        """Test adapting to correction feedback"""
        feedback = UserFeedback(
            feedback_id="feedback_1",
            user_id="test_user",
            interaction_id="interaction_1",
            feedback_type="correction",
            feedback_value="The correct answer is...",
            timestamp=datetime.now()
        )
        
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        # Mock the internal methods
        learning_engine._get_user_model = AsyncMock(return_value=model)
        learning_engine._learn_from_correction = AsyncMock()
        learning_engine._save_user_model = AsyncMock()
        
        await learning_engine.adapt_to_feedback(feedback)
        
        # Verify model was updated
        assert model.model_version == 2
        learning_engine._learn_from_correction.assert_called_once()
        learning_engine._save_user_model.assert_called_once_with(model)
    
    @pytest.mark.asyncio
    async def test_get_user_behavior_patterns(self, learning_engine, temp_db):
        """Test retrieving user behavior patterns"""
        # Insert test pattern into database
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                INSERT INTO behavior_patterns 
                (pattern_id, user_id, pattern_type, pattern_data, confidence, 
                 frequency, first_detected, last_updated, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "pattern_1",
                "test_user",
                "time_preference",
                json.dumps({"hour": 9, "day_of_week": 1}),
                0.8,
                10,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                json.dumps({})
            ))
            conn.commit()
        
        patterns = await learning_engine.get_user_behavior_patterns("test_user")
        
        assert len(patterns) == 1
        assert patterns[0].pattern_id == "pattern_1"
        assert patterns[0].pattern_type == "time_preference"
        assert patterns[0].confidence == 0.8
        assert patterns[0].frequency == 10
    
    @pytest.mark.asyncio
    async def test_detect_time_pattern(self, learning_engine, sample_interaction, temp_db):
        """Test detecting time-based patterns"""
        # Create the user_interactions table first
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_data TEXT,
                    feedback_score REAL,
                    metadata TEXT
                )
            """)
            
            # Insert multiple interactions at the same time
            for i in range(6):  # Above min_interactions_for_pattern
                conn.execute("""
                    INSERT INTO user_interactions 
                    (id, user_id, interaction_type, content, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    f"interaction_{i}",
                    "test_user",
                    "query",
                    f"Test query {i}",
                    datetime.now().replace(hour=9).isoformat()
                ))
            conn.commit()
        
        # Set interaction time to match
        sample_interaction.timestamp = datetime.now().replace(hour=9)
        
        pattern = await learning_engine._detect_time_pattern(sample_interaction)
        
        assert pattern is not None
        assert pattern.pattern_type == "time_preference"
        assert pattern.pattern_data["hour"] == 9
        assert pattern.confidence > 0
        assert pattern.frequency >= learning_engine.min_interactions_for_pattern
    
    @pytest.mark.asyncio
    async def test_detect_content_pattern(self, learning_engine, sample_interaction, temp_db):
        """Test detecting content-based patterns"""
        # Create the user_interactions table first
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    response TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_data TEXT,
                    feedback_score REAL,
                    metadata TEXT
                )
            """)
            
            # Insert multiple interactions with similar content
            for i in range(6):  # Above min_interactions_for_pattern
                conn.execute("""
                    INSERT INTO user_interactions 
                    (id, user_id, interaction_type, content, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    f"interaction_{i}",
                    "test_user",
                    "query",
                    f"Python programming question {i}",
                    datetime.now().isoformat()
                ))
            conn.commit()
        
        sample_interaction.content = "Python programming help needed"
        
        pattern = await learning_engine._detect_content_pattern(sample_interaction)
        
        assert pattern is not None
        assert pattern.pattern_type == "content_preference"
        assert "python" in pattern.pattern_data["common_keywords"]
        assert pattern.confidence > 0
    
    @pytest.mark.asyncio
    async def test_boost_preference_weights(self, learning_engine):
        """Test boosting preference weights"""
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={"interaction_type_query": 0.5},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        context_data = {"interaction_type": "query", "content_category": "technical"}
        
        learning_engine._boost_preference_weights(model, context_data, 0.2)
        
        assert model.preference_weights["interaction_type_query"] == 0.7
        assert model.preference_weights["content_category_technical"] == 0.7
    
    @pytest.mark.asyncio
    async def test_reduce_preference_weights(self, learning_engine):
        """Test reducing preference weights"""
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={"interaction_type_query": 0.5},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        context_data = {"interaction_type": "query", "content_category": "technical"}
        
        learning_engine._reduce_preference_weights(model, context_data, 0.2)
        
        assert model.preference_weights["interaction_type_query"] == 0.3
        assert model.preference_weights["content_category_technical"] == 0.3
    
    @pytest.mark.asyncio
    async def test_get_time_based_suggestions(self, learning_engine, sample_user_context):
        """Test generating time-based suggestions"""
        # Create a model with time-based patterns
        current_hour = datetime.now().hour
        pattern = BehaviorPattern(
            pattern_id="time_pattern_1",
            user_id="test_user",
            pattern_type="time_preference",
            pattern_data={"hour": current_hour, "interaction_type": "query"},
            confidence=0.8,
            frequency=10,
            first_detected=datetime.now(),
            last_updated=datetime.now()
        )
        
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={"time_pattern_1": pattern},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        suggestions = await learning_engine._get_time_based_suggestions(model, sample_user_context)
        
        assert len(suggestions) == 1
        assert suggestions[0]["type"] == "time_based"
        assert suggestions[0]["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_get_task_based_suggestions(self, learning_engine, sample_user_context):
        """Test generating task-based suggestions"""
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        suggestions = await learning_engine._get_task_based_suggestions(model, sample_user_context)
        
        # Should suggest help with current tasks
        assert len(suggestions) == 2  # Two current tasks in sample context
        assert all(s["type"] == "task_based" for s in suggestions)
        assert "authentication" in suggestions[0]["suggestion"].lower()
    
    @pytest.mark.asyncio
    async def test_get_pattern_based_suggestions(self, learning_engine, sample_user_context):
        """Test generating pattern-based suggestions"""
        # Create a model with content patterns
        pattern = BehaviorPattern(
            pattern_id="content_pattern_1",
            user_id="test_user",
            pattern_type="content_preference",
            pattern_data={"common_keywords": ["python", "programming", "function"]},
            confidence=0.8,
            frequency=15,
            first_detected=datetime.now(),
            last_updated=datetime.now()
        )
        
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={"content_pattern_1": pattern},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        suggestions = await learning_engine._get_pattern_based_suggestions(model, sample_user_context)
        
        assert len(suggestions) == 1
        assert suggestions[0]["type"] == "pattern_based"
        assert "python" in suggestions[0]["suggestion"].lower()
        assert suggestions[0]["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_rank_suggestions(self, learning_engine, sample_user_context):
        """Test ranking suggestions by relevance"""
        suggestions = [
            {"type": "time_based", "confidence": 0.6},
            {"type": "task_based", "confidence": 0.9},
            {"type": "pattern_based", "confidence": 0.7}
        ]
        
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={
                "suggestion_time_based": 0.5,
                "suggestion_task_based": 0.9,
                "suggestion_pattern_based": 0.7
            },
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        ranked = await learning_engine._rank_suggestions(suggestions, model, sample_user_context)
        
        # Should be ranked by score (confidence * preference_weight)
        assert len(ranked) == 3
        assert ranked[0]["type"] == "task_based"  # 0.9 * 0.9 = 0.81
        assert ranked[1]["type"] == "pattern_based"  # 0.7 * 0.7 = 0.49
        assert ranked[2]["type"] == "time_based"  # 0.6 * 0.5 = 0.3
    
    @pytest.mark.asyncio
    async def test_adjust_preference_weights(self, learning_engine):
        """Test adjusting preference weights based on rating"""
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={"interaction_type_query": 0.5},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        feedback = UserFeedback(
            feedback_id="feedback_1",
            user_id="test_user",
            interaction_id="interaction_1",
            feedback_type="rating",
            feedback_value=5.0,  # High rating
            timestamp=datetime.now(),
            context_data={"interaction_type": "query"}
        )
        
        await learning_engine._adjust_preference_weights(model, feedback, 5.0)
        
        # Should increase preference weight for positive rating
        assert model.preference_weights["interaction_type_query"] > 0.5
    
    @pytest.mark.asyncio
    async def test_learn_from_correction(self, learning_engine):
        """Test learning from user corrections"""
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now(),
            metadata={}
        )
        
        feedback = UserFeedback(
            feedback_id="feedback_1",
            user_id="test_user",
            interaction_id="interaction_1",
            feedback_type="correction",
            feedback_value="The correct answer is 42",
            timestamp=datetime.now(),
            context_data={"original_response": "The answer is 24"}
        )
        
        await learning_engine._learn_from_correction(model, feedback)
        
        # Should store correction in metadata
        assert "corrections" in model.metadata
        assert len(model.metadata["corrections"]) == 1
        assert model.metadata["corrections"][0]["corrected_response"] == "The correct answer is 42"
    
    @pytest.mark.asyncio
    async def test_update_performance_metrics(self, learning_engine):
        """Test updating performance metrics"""
        model = LearningModel(
            user_id="test_user",
            model_version=1,
            behavior_patterns={},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        interaction = Interaction(
            id="test_interaction",
            user_id="test_user",
            interaction_type=InteractionType.QUERY,
            content="Test query",
            response="Test response",
            timestamp=datetime.now(),
            feedback_score=4.0
        )
        
        await learning_engine._update_performance_metrics(model, interaction)
        
        # Should update metrics
        assert model.performance_metrics["total_interactions"] == 1
        assert "last_interaction" in model.performance_metrics
        assert model.performance_metrics["avg_feedback_score"] == 4.0
    
    @pytest.mark.asyncio
    async def test_default_preference_weights(self, learning_engine):
        """Test default preference weights"""
        weights = learning_engine._get_default_preference_weights()
        
        assert isinstance(weights, dict)
        assert len(weights) > 0
        assert all(isinstance(v, float) for v in weights.values())
        assert all(0.0 <= v <= 1.0 for v in weights.values())
        
        # Check for expected default weights
        assert "interaction_type_query" in weights
        assert "interaction_type_command" in weights
        assert "time_morning" in weights
    
    @pytest.mark.asyncio
    async def test_error_handling_learn_from_interaction(self, learning_engine, sample_interaction):
        """Test error handling in learn_from_interaction"""
        # Mock method to raise exception
        learning_engine._get_user_model = AsyncMock(side_effect=Exception("Database error"))
        
        # Should not raise exception, but log error
        await learning_engine.learn_from_interaction(sample_interaction)
        
        # Verify it was called despite the error
        learning_engine._get_user_model.assert_called_once_with("test_user")
    
    @pytest.mark.asyncio
    async def test_error_handling_process_feedback(self, learning_engine):
        """Test error handling in process_feedback"""
        feedback = UserFeedback(
            feedback_id="feedback_1",
            user_id="test_user",
            interaction_id="interaction_1",
            feedback_type="rating",
            feedback_value=4.5,
            timestamp=datetime.now()
        )
        
        # Mock method to raise exception
        learning_engine._store_feedback = AsyncMock(side_effect=Exception("Database error"))
        
        # Should not raise exception, but log error
        await learning_engine.process_feedback(feedback)
        
        # Verify it was called despite the error
        learning_engine._store_feedback.assert_called_once_with(feedback)


if __name__ == "__main__":
    pytest.main([__file__])