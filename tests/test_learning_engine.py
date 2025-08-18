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


class TestContinuousLearning:
    """Test cases for continuous learning functionality"""
    
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
    async def continuous_learning_engine(self, temp_db):
        """Create a LearningEngine with continuous learning enabled"""
        context_manager = Mock(spec=UserContextManager)
        engine = LearningEngine(db_path=temp_db, context_manager=context_manager)
        engine.continuous_learning_enabled = True
        return engine
    
    @pytest.mark.asyncio
    async def test_enable_continuous_learning(self, continuous_learning_engine):
        """Test enabling continuous learning"""
        user_id = "test_user"
        
        # Mock the log learning event method
        continuous_learning_engine._log_learning_event = AsyncMock()
        
        await continuous_learning_engine.enable_continuous_learning(user_id)
        
        # Verify continuous learning is enabled
        assert continuous_learning_engine.continuous_learning_enabled is True
        
        # Verify learning event was logged
        continuous_learning_engine._log_learning_event.assert_called_once()
        
        # Check database state
        import sqlite3
        with sqlite3.connect(continuous_learning_engine.db_path) as conn:
            cursor = conn.execute("""
                SELECT user_id, optimization_count FROM continuous_learning_state 
                WHERE user_id = ?
            """, (user_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == user_id
            assert row[1] == 0  # Initial optimization count
    
    @pytest.mark.asyncio
    async def test_disable_continuous_learning(self, continuous_learning_engine):
        """Test disabling continuous learning"""
        user_id = "test_user"
        
        # Mock the log learning event method
        continuous_learning_engine._log_learning_event = AsyncMock()
        
        await continuous_learning_engine.disable_continuous_learning(user_id)
        
        # Verify continuous learning is disabled
        assert continuous_learning_engine.continuous_learning_enabled is False
        
        # Verify learning event was logged
        continuous_learning_engine._log_learning_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_track_performance_metric(self, continuous_learning_engine):
        """Test tracking performance metrics"""
        user_id = "test_user"
        metric_name = "response_time"
        metric_value = 1.5
        context_data = {"interaction_type": "query"}
        
        # Mock methods
        continuous_learning_engine._get_user_model = AsyncMock(return_value=LearningModel(
            user_id=user_id,
            model_version=1,
            behavior_patterns={},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        ))
        continuous_learning_engine._check_optimization_trigger = AsyncMock()
        
        await continuous_learning_engine.track_performance_metric(
            user_id, metric_name, metric_value, context_data
        )
        
        # Verify metric was stored in database
        import sqlite3
        with sqlite3.connect(continuous_learning_engine.db_path) as conn:
            cursor = conn.execute("""
                SELECT metric_name, metric_value, context_data FROM performance_metrics 
                WHERE user_id = ?
            """, (user_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert row[0] == metric_name
            assert row[1] == metric_value
            assert json.loads(row[2]) == context_data
        
        # Verify optimization trigger was checked
        continuous_learning_engine._check_optimization_trigger.assert_called_once_with(
            user_id, metric_name, metric_value
        )
    
    @pytest.mark.asyncio
    async def test_get_performance_trends(self, continuous_learning_engine, temp_db):
        """Test getting performance trends"""
        user_id = "test_user"
        metric_name = "feedback_score"
        
        # Insert test performance metrics
        import sqlite3
        with sqlite3.connect(temp_db) as conn:
            # Insert metrics with improving trend - need more data points
            base_time = datetime.now() - timedelta(days=20)
            for i in range(20):  # More data points
                timestamp = base_time + timedelta(days=i)
                metric_value = 3.0 + (i * 0.1)  # Improving from 3.0 to 5.0
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (user_id, metric_name, metric_value, timestamp, model_version)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, metric_name, metric_value, timestamp.isoformat(), 1))
            conn.commit()
        
        trends = await continuous_learning_engine.get_performance_trends(user_id, metric_name)
        
        assert metric_name in trends
        trend_data = trends[metric_name]
        assert trend_data["direction"] == "improving"
        assert trend_data["magnitude"] > 0
        assert trend_data["total_points"] == 20
        assert trend_data["latest_value"] == 4.9
    
    @pytest.mark.asyncio
    async def test_optimize_learning_parameters(self, continuous_learning_engine):
        """Test optimizing learning parameters"""
        user_id = "test_user"
        
        # Mock methods
        continuous_learning_engine._get_user_model = AsyncMock(return_value=LearningModel(
            user_id=user_id,
            model_version=1,
            behavior_patterns={},  # Few patterns to trigger optimization
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        ))
        
        continuous_learning_engine.get_performance_trends = AsyncMock(return_value={
            "avg_feedback_score": {
                "direction": "declining",
                "magnitude": 0.15,  # Above improvement threshold
                "recent_average": 3.2
            }
        })
        
        continuous_learning_engine._update_continuous_learning_state = AsyncMock()
        continuous_learning_engine._log_learning_event = AsyncMock()
        
        # Store initial learning rate
        initial_learning_rate = continuous_learning_engine.adaptation_learning_rate
        initial_confidence_threshold = continuous_learning_engine.pattern_confidence_threshold
        
        results = await continuous_learning_engine.optimize_learning_parameters(user_id)
        
        # Verify optimizations were applied
        assert "optimizations_applied" in results
        assert len(results["optimizations_applied"]) > 0
        
        # Verify learning rate was increased due to declining feedback
        assert continuous_learning_engine.adaptation_learning_rate > initial_learning_rate
        
        # Verify confidence threshold was lowered due to few patterns
        assert continuous_learning_engine.pattern_confidence_threshold < initial_confidence_threshold
        
        # Verify state was updated
        continuous_learning_engine._update_continuous_learning_state.assert_called_once()
        continuous_learning_engine._log_learning_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_suggest_feature_improvements(self, continuous_learning_engine):
        """Test suggesting feature improvements"""
        user_id = "test_user"
        
        # Create patterns that should trigger feature suggestions
        automation_pattern = BehaviorPattern(
            pattern_id="automation_pattern",
            user_id=user_id,
            pattern_type="content_preference",
            pattern_data={"common_keywords": ["automation", "workflow", "automate"]},
            confidence=0.9,
            frequency=15,
            first_detected=datetime.now(),
            last_updated=datetime.now()
        )
        
        integration_pattern = BehaviorPattern(
            pattern_id="integration_pattern",
            user_id=user_id,
            pattern_type="content_preference",
            pattern_data={"common_keywords": ["integration", "connect", "api"]},
            confidence=0.8,
            frequency=12,
            first_detected=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Mock methods
        continuous_learning_engine._get_user_model = AsyncMock(return_value=LearningModel(
            user_id=user_id,
            model_version=1,
            behavior_patterns={},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        ))
        
        continuous_learning_engine.get_user_behavior_patterns = AsyncMock(return_value=[
            automation_pattern, integration_pattern
        ])
        
        continuous_learning_engine.get_performance_trends = AsyncMock(return_value={
            "response_time": {
                "direction": "declining",
                "magnitude": 0.25
            }
        })
        
        continuous_learning_engine._store_feature_suggestion = AsyncMock()
        
        suggestions = await continuous_learning_engine.suggest_feature_improvements(user_id)
        
        # Verify suggestions were generated
        assert len(suggestions) >= 2  # At least automation and performance suggestions
        
        # Check for automation suggestion
        automation_suggestions = [s for s in suggestions if "automation" in s["suggestion"].lower()]
        assert len(automation_suggestions) > 0
        assert automation_suggestions[0]["confidence"] >= continuous_learning_engine.feature_suggestion_confidence
        
        # Check for performance suggestion
        performance_suggestions = [s for s in suggestions if "performance" in s["suggestion"].lower()]
        assert len(performance_suggestions) > 0
        
        # Verify suggestions were stored
        assert continuous_learning_engine._store_feature_suggestion.call_count >= len(suggestions)
    
    @pytest.mark.asyncio
    async def test_adapt_to_changing_patterns(self, continuous_learning_engine):
        """Test adapting to changing user patterns"""
        user_id = "test_user"
        
        # Create existing pattern
        existing_pattern = BehaviorPattern(
            pattern_id="existing_pattern",
            user_id=user_id,
            pattern_type="time_preference",
            pattern_data={"hour": 9},
            confidence=0.7,
            frequency=10,
            first_detected=datetime.now() - timedelta(days=30),
            last_updated=datetime.now() - timedelta(days=15)
        )
        
        model = LearningModel(
            user_id=user_id,
            model_version=1,
            behavior_patterns={"existing_pattern": existing_pattern},
            preference_weights={},
            adaptation_history=[],
            performance_metrics={},
            last_updated=datetime.now()
        )
        
        # Mock methods
        continuous_learning_engine._get_user_model = AsyncMock(return_value=model)
        continuous_learning_engine.get_user_behavior_patterns = AsyncMock(return_value=[existing_pattern])
        continuous_learning_engine._get_recent_interactions = AsyncMock(return_value=[
            # Mock recent interactions
            Interaction(
                id=f"interaction_{i}",
                user_id=user_id,
                interaction_type=InteractionType.QUERY,
                content=f"Test query {i}",
                response=f"Test response {i}",
                timestamp=datetime.now() - timedelta(days=i),
                context_data={},
                feedback_score=4.0
            ) for i in range(10)
        ])
        continuous_learning_engine._analyze_interaction_patterns = AsyncMock(return_value=[])
        continuous_learning_engine._get_recent_feedback = AsyncMock(return_value=[])
        continuous_learning_engine._save_user_model = AsyncMock()
        continuous_learning_engine._log_learning_event = AsyncMock()
        
        results = await continuous_learning_engine.adapt_to_changing_patterns(user_id)
        
        # Verify adaptation results
        assert "patterns_updated" in results
        assert "patterns_deprecated" in results
        assert "new_patterns_detected" in results
        assert "preference_adjustments" in results
        assert "timestamp" in results
        
        # Verify model was saved
        continuous_learning_engine._save_user_model.assert_called_once()
        continuous_learning_engine._log_learning_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_learning_insights(self, continuous_learning_engine):
        """Test getting learning insights"""
        user_id = "test_user"
        
        # Create test data
        pattern = BehaviorPattern(
            pattern_id="test_pattern",
            user_id=user_id,
            pattern_type="content_preference",
            pattern_data={"keywords": ["python", "programming"]},
            confidence=0.8,
            frequency=15,
            first_detected=datetime.now(),
            last_updated=datetime.now()
        )
        
        model = LearningModel(
            user_id=user_id,
            model_version=3,
            behavior_patterns={"test_pattern": pattern},
            preference_weights={"test_pref": 0.7},
            adaptation_history=[
                {"timestamp": datetime.now().isoformat(), "type": "feedback"},
                {"timestamp": datetime.now().isoformat(), "type": "pattern_update"}
            ],
            performance_metrics={"avg_feedback_score": 4.2, "total_interactions": 50},
            last_updated=datetime.now()
        )
        
        # Mock methods
        continuous_learning_engine._get_user_model = AsyncMock(return_value=model)
        continuous_learning_engine.get_user_behavior_patterns = AsyncMock(return_value=[pattern])
        continuous_learning_engine.get_performance_trends = AsyncMock(return_value={
            "feedback_score": {
                "direction": "improving",
                "magnitude": 0.1,
                "recent_average": 4.2
            }
        })
        
        insights = await continuous_learning_engine.get_learning_insights(user_id)
        
        # Verify insights structure
        assert "model_info" in insights
        assert "pattern_analysis" in insights
        assert "performance_summary" in insights
        assert "learning_effectiveness" in insights
        
        # Verify model info
        model_info = insights["model_info"]
        assert model_info["version"] == 3
        assert model_info["total_patterns"] == 1
        assert model_info["adaptation_events"] == 2
        
        # Verify pattern analysis
        pattern_analysis = insights["pattern_analysis"]
        assert pattern_analysis["average_confidence"] == 0.8
        assert len(pattern_analysis["most_confident_patterns"]) == 1
        assert pattern_analysis["pattern_types"]["content_preference"] == 1
        
        # Verify learning effectiveness metrics
        effectiveness = insights["learning_effectiveness"]
        assert "adaptation_rate" in effectiveness
        assert "pattern_stability" in effectiveness
        assert "feedback_incorporation" in effectiveness


class TestContinuousLearningSimulation:
    """Long-running simulation tests for continuous learning"""
    
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
    async def simulation_engine(self, temp_db):
        """Create a LearningEngine for simulation testing"""
        context_manager = Mock(spec=UserContextManager)
        engine = LearningEngine(db_path=temp_db, context_manager=context_manager)
        engine.continuous_learning_enabled = True
        return engine
    
    @pytest.mark.asyncio
    async def test_long_term_learning_simulation(self, simulation_engine):
        """Test long-term learning behavior with simulated interactions"""
        user_id = "simulation_user"
        
        # Enable continuous learning
        await simulation_engine.enable_continuous_learning(user_id)
        
        # Simulate 100 interactions over time
        interaction_types = [InteractionType.QUERY, InteractionType.COMMAND, InteractionType.FEEDBACK]
        content_themes = ["python programming", "web development", "data analysis", "automation", "testing"]
        
        initial_model = await simulation_engine._get_user_model(user_id)
        initial_patterns = len(initial_model.behavior_patterns)
        
        for i in range(100):
            # Create varied interactions
            interaction_type = interaction_types[i % len(interaction_types)]
            theme = content_themes[i % len(content_themes)]
            
            interaction = Interaction(
                id=f"sim_interaction_{i}",
                user_id=user_id,
                interaction_type=interaction_type,
                content=f"{theme} question {i}",
                response=f"Response about {theme} {i}",
                timestamp=datetime.now() - timedelta(hours=100-i),  # Spread over time
                context_data={"theme": theme, "session": i // 10},
                feedback_score=3.5 + (i % 3) * 0.5  # Varying feedback scores
            )
            
            # Learn from interaction
            await simulation_engine.learn_from_interaction(interaction)
            
            # Occasionally provide feedback
            if i % 10 == 0:
                feedback = UserFeedback(
                    feedback_id=f"sim_feedback_{i}",
                    user_id=user_id,
                    interaction_id=interaction.id,
                    feedback_type="rating",
                    feedback_value=4.0 + (i % 2) * 0.5,
                    timestamp=interaction.timestamp,
                    context_data={"theme": theme}
                )
                await simulation_engine.process_feedback(feedback)
            
            # Track performance metrics
            if i % 5 == 0:
                await simulation_engine.track_performance_metric(
                    user_id, "response_quality", 3.5 + (i / 100.0), {"interaction_count": i}
                )
        
        # Analyze results after simulation
        final_model = await simulation_engine._get_user_model(user_id)
        final_patterns = await simulation_engine.get_user_behavior_patterns(user_id)
        trends = await simulation_engine.get_performance_trends(user_id)
        insights = await simulation_engine.get_learning_insights(user_id)
        
        # Verify learning occurred
        assert len(final_patterns) > initial_patterns, "Should have detected new patterns"
        assert final_model.model_version > initial_model.model_version, "Model should have been updated"
        assert len(final_model.adaptation_history) > 0, "Should have adaptation history"
        
        # Verify performance tracking
        assert "response_quality" in trends, "Should have performance trends"
        assert trends["response_quality"]["total_points"] > 0, "Should have tracked metrics"
        
        # Verify insights are meaningful
        assert insights["model_info"]["total_patterns"] > 0, "Should have detected patterns"
        assert insights["learning_effectiveness"]["adaptation_rate"] > 0, "Should show adaptation activity"
        
        # Test feature suggestions
        suggestions = await simulation_engine.suggest_feature_improvements(user_id)
        assert len(suggestions) > 0, "Should generate feature suggestions based on usage patterns"
        
        # Test pattern adaptation
        adaptation_results = await simulation_engine.adapt_to_changing_patterns(user_id)
        assert "patterns_updated" in adaptation_results, "Should adapt to changing patterns"
        
        # Test parameter optimization
        optimization_results = await simulation_engine.optimize_learning_parameters(user_id)
        assert "optimizations_applied" in optimization_results, "Should optimize parameters"
    
    @pytest.mark.asyncio
    async def test_feedback_adaptation_simulation(self, simulation_engine):
        """Test how the system adapts to different types of feedback over time"""
        user_id = "feedback_user"
        
        await simulation_engine.enable_continuous_learning(user_id)
        
        # Phase 1: Positive feedback period
        for i in range(20):
            interaction = Interaction(
                id=f"pos_interaction_{i}",
                user_id=user_id,
                interaction_type=InteractionType.QUERY,
                content=f"Positive phase query {i}",
                response=f"Good response {i}",
                timestamp=datetime.now() - timedelta(hours=50-i),
                feedback_score=4.5
            )
            await simulation_engine.learn_from_interaction(interaction)
            
            feedback = UserFeedback(
                feedback_id=f"pos_feedback_{i}",
                user_id=user_id,
                interaction_id=interaction.id,
                feedback_type="rating",
                feedback_value=4.5,
                timestamp=interaction.timestamp,
                context_data={"phase": "positive", "interaction_type": "query"}
            )
            await simulation_engine.process_feedback(feedback)
        
        model_after_positive = await simulation_engine._get_user_model(user_id)
        positive_weights = model_after_positive.preference_weights.copy()
        
        # Phase 2: Negative feedback period
        for i in range(20):
            interaction = Interaction(
                id=f"neg_interaction_{i}",
                user_id=user_id,
                interaction_type=InteractionType.QUERY,
                content=f"Negative phase query {i}",
                response=f"Poor response {i}",
                timestamp=datetime.now() - timedelta(hours=25-i),
                feedback_score=2.0
            )
            await simulation_engine.learn_from_interaction(interaction)
            
            feedback = UserFeedback(
                feedback_id=f"neg_feedback_{i}",
                user_id=user_id,
                interaction_id=interaction.id,
                feedback_type="rating",
                feedback_value=2.0,
                timestamp=interaction.timestamp,
                context_data={"phase": "negative", "interaction_type": "query"}
            )
            await simulation_engine.process_feedback(feedback)
        
        model_after_negative = await simulation_engine._get_user_model(user_id)
        negative_weights = model_after_negative.preference_weights.copy()
        
        # Verify adaptation occurred
        assert model_after_negative.model_version > model_after_positive.model_version
        
        # Check that preference weights changed appropriately
        query_weight_key = "interaction_type_query"
        if query_weight_key in positive_weights and query_weight_key in negative_weights:
            # Weight should have decreased due to negative feedback
            assert negative_weights[query_weight_key] < positive_weights[query_weight_key]
        
        # Test recovery with mixed feedback
        for i in range(10):
            interaction = Interaction(
                id=f"mixed_interaction_{i}",
                user_id=user_id,
                interaction_type=InteractionType.QUERY,
                content=f"Mixed phase query {i}",
                response=f"Improving response {i}",
                timestamp=datetime.now() - timedelta(hours=10-i),
                feedback_score=3.5
            )
            await simulation_engine.learn_from_interaction(interaction)
            
            feedback = UserFeedback(
                feedback_id=f"mixed_feedback_{i}",
                user_id=user_id,
                interaction_id=interaction.id,
                feedback_type="rating",
                feedback_value=3.5,
                timestamp=interaction.timestamp,
                context_data={"phase": "recovery", "interaction_type": "query"}
            )
            await simulation_engine.process_feedback(feedback)
        
        # Verify system adapted to changing feedback patterns
        final_insights = await simulation_engine.get_learning_insights(user_id)
        assert final_insights["learning_effectiveness"]["feedback_incorporation"] > 0
        assert final_insights["model_info"]["adaptation_events"] >= 50  # Should have many adaptations
    
    @pytest.mark.asyncio
    async def test_pattern_evolution_simulation(self, simulation_engine):
        """Test how patterns evolve and adapt over time"""
        user_id = "pattern_user"
        
        await simulation_engine.enable_continuous_learning(user_id)
        
        # Phase 1: Establish morning work pattern
        morning_hour = 9
        for i in range(15):
            interaction = Interaction(
                id=f"morning_interaction_{i}",
                user_id=user_id,
                interaction_type=InteractionType.QUERY,
                content=f"Morning work query {i}",
                response=f"Morning response {i}",
                timestamp=datetime.now().replace(hour=morning_hour) - timedelta(days=i),
                feedback_score=4.0
            )
            await simulation_engine.learn_from_interaction(interaction)
        
        # Check that morning pattern was detected
        morning_patterns = await simulation_engine.get_user_behavior_patterns(user_id)
        morning_time_patterns = [p for p in morning_patterns if p.pattern_type == "time_preference" and p.pattern_data.get("hour") == morning_hour]
        assert len(morning_time_patterns) > 0, "Should detect morning work pattern"
        
        # Phase 2: Shift to evening work pattern
        evening_hour = 20
        for i in range(15):
            interaction = Interaction(
                id=f"evening_interaction_{i}",
                user_id=user_id,
                interaction_type=InteractionType.QUERY,
                content=f"Evening work query {i}",
                response=f"Evening response {i}",
                timestamp=datetime.now().replace(hour=evening_hour) - timedelta(days=i),
                feedback_score=4.2
            )
            await simulation_engine.learn_from_interaction(interaction)
        
        # Trigger pattern adaptation
        adaptation_results = await simulation_engine.adapt_to_changing_patterns(user_id)
        
        # Check that new evening pattern was detected
        updated_patterns = await simulation_engine.get_user_behavior_patterns(user_id)
        evening_time_patterns = [p for p in updated_patterns if p.pattern_type == "time_preference" and p.pattern_data.get("hour") == evening_hour]
        assert len(evening_time_patterns) > 0, "Should detect new evening work pattern"
        
        # Verify adaptation occurred
        assert adaptation_results["new_patterns_detected"] > 0 or adaptation_results["patterns_updated"] > 0
        
        # Test that suggestions adapt to new patterns
        suggestions = await simulation_engine.get_personalized_suggestions(UserContext(
            user_id=user_id,
            current_activity="working",
            active_applications=["editor"],
            current_files=[],
            preferences=UserPreferences(user_id=user_id, language="en"),
            task_context=TaskContext(current_tasks=[], active_projects=[]),
            knowledge_state=KnowledgeState(expertise_areas={})
        ))
        
        # Should have suggestions based on detected patterns
        assert len(suggestions) > 0, "Should generate suggestions based on learned patterns"


if __name__ == "__main__":
    # Run all tests including the new continuous learning tests
    pytest.main([__file__, "-v"])