"""
Learning Engine

This module implements personalized learning and adaptation capabilities for the AI assistant.
It analyzes user behavior patterns, processes feedback, and continuously improves assistance
based on user interactions and preferences.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import numpy as np
from pathlib import Path

from .personal_assistant_models import (
    UserContext, Interaction, InteractionType, UserPreferences
)
from .user_context_manager import UserContextManager

logger = logging.getLogger(__name__)


@dataclass
class BehaviorPattern:
    """Represents a detected user behavior pattern"""
    pattern_id: str
    user_id: str
    pattern_type: str  # e.g., "time_preference", "task_sequence", "tool_usage"
    pattern_data: Dict[str, Any]
    confidence: float
    frequency: int
    first_detected: datetime
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserFeedback:
    """Represents user feedback on assistant responses"""
    feedback_id: str
    user_id: str
    interaction_id: str
    feedback_type: str  # "rating", "correction", "preference"
    feedback_value: Any  # rating score, corrected text, preference setting
    timestamp: datetime
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningModel:
    """User-specific learning model"""
    user_id: str
    model_version: int
    behavior_patterns: Dict[str, BehaviorPattern]
    preference_weights: Dict[str, float]
    adaptation_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class LearningEngine:
    """
    Personalized learning and adaptation engine for the AI assistant.
    
    This class analyzes user interactions, identifies behavior patterns,
    processes feedback, and continuously adapts the assistant's responses
    to better serve individual users. It includes continuous improvement
    capabilities for self-optimization and feature suggestions.
    """
    
    def __init__(self, db_path: str = "personal_assistant.db", context_manager: Optional[UserContextManager] = None):
        self.db_path = db_path
        self.context_manager = context_manager or UserContextManager(db_path)
        self._init_database()
        self._user_models: Dict[str, LearningModel] = {}
        
        # Learning parameters
        self.min_interactions_for_pattern = 5
        self.pattern_confidence_threshold = 0.7
        self.adaptation_learning_rate = 0.1
        self.feedback_weight = 0.3
        
        # Continuous learning parameters
        self.performance_tracking_window = 100  # Number of recent interactions to track
        self.improvement_threshold = 0.1  # Minimum improvement to trigger adaptation
        self.feature_suggestion_confidence = 0.8  # Minimum confidence for feature suggestions
        self.continuous_learning_enabled = True
    
    def _init_database(self):
        """Initialize the database schema for learning engine"""
        with sqlite3.connect(self.db_path) as conn:
            # Behavior patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS behavior_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    frequency INTEGER NOT NULL,
                    first_detected TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # User feedback table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    interaction_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    feedback_value TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_data TEXT
                )
            """)
            
            # Learning models table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_models (
                    user_id TEXT PRIMARY KEY,
                    model_version INTEGER NOT NULL,
                    model_data TEXT NOT NULL,
                    performance_metrics TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Learning events table for tracking adaptation history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_version INTEGER
                )
            """)
            
            # Performance metrics table for continuous tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_version INTEGER,
                    context_data TEXT
                )
            """)
            
            # Feature suggestions table for self-improvement
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    suggestion_type TEXT NOT NULL,
                    suggestion_data TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Continuous learning state table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS continuous_learning_state (
                    user_id TEXT PRIMARY KEY,
                    last_optimization TIMESTAMP,
                    optimization_count INTEGER DEFAULT 0,
                    performance_trend TEXT,
                    adaptation_history TEXT,
                    metadata TEXT
                )
            """)
            
            conn.commit()
    
    async def learn_from_interaction(self, interaction: Interaction) -> None:
        """
        Learn from a user interaction by analyzing patterns and updating models.
        
        Args:
            interaction: The user interaction to learn from
        """
        try:
            user_id = interaction.user_id
            
            # Get or create user learning model
            model = await self._get_user_model(user_id)
            
            # Analyze interaction for patterns
            patterns = await self._analyze_interaction_patterns(interaction)
            
            # Update behavior patterns
            for pattern in patterns:
                await self._update_behavior_pattern(pattern)
            
            # Update user model
            await self._update_user_model(model, interaction, patterns)
            
            # Log learning event
            await self._log_learning_event(user_id, "interaction_learning", {
                "interaction_id": interaction.id,
                "patterns_detected": len(patterns),
                "interaction_type": interaction.interaction_type.value
            })
            
            logger.info(f"Learned from interaction {interaction.id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
    
    async def process_feedback(self, feedback: UserFeedback) -> None:
        """
        Process user feedback and adapt the learning model accordingly.
        
        Args:
            feedback: User feedback to process
        """
        try:
            # Store feedback
            await self._store_feedback(feedback)
            
            # Get user model
            model = await self._get_user_model(feedback.user_id)
            
            # Apply feedback to model
            await self._apply_feedback_to_model(model, feedback)
            
            # Update preferences based on feedback
            await self._update_preferences_from_feedback(feedback)
            
            # Log learning event
            await self._log_learning_event(feedback.user_id, "feedback_processing", {
                "feedback_id": feedback.feedback_id,
                "feedback_type": feedback.feedback_type,
                "interaction_id": feedback.interaction_id
            })
            
            logger.info(f"Processed feedback {feedback.feedback_id} for user {feedback.user_id}")
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
    
    async def get_personalized_suggestions(self, user_context: UserContext) -> List[Dict[str, Any]]:
        """
        Generate personalized suggestions based on user patterns and context.
        
        Args:
            user_context: Current user context
            
        Returns:
            List of personalized suggestions
        """
        try:
            model = await self._get_user_model(user_context.user_id)
            suggestions = []
            
            # Time-based suggestions
            time_suggestions = await self._get_time_based_suggestions(model, user_context)
            suggestions.extend(time_suggestions)
            
            # Task-based suggestions
            task_suggestions = await self._get_task_based_suggestions(model, user_context)
            suggestions.extend(task_suggestions)
            
            # Pattern-based suggestions
            pattern_suggestions = await self._get_pattern_based_suggestions(model, user_context)
            suggestions.extend(pattern_suggestions)
            
            # Rank suggestions by relevance
            ranked_suggestions = await self._rank_suggestions(suggestions, model, user_context)
            
            return ranked_suggestions[:10]  # Return top 10 suggestions
            
        except Exception as e:
            logger.error(f"Error generating personalized suggestions: {e}")
            return []
    
    async def adapt_to_feedback(self, user_feedback: UserFeedback) -> None:
        """
        Adapt the learning model based on user feedback.
        
        Args:
            user_feedback: Feedback to adapt to
        """
        try:
            model = await self._get_user_model(user_feedback.user_id)
            
            # Adjust preference weights based on feedback
            if user_feedback.feedback_type == "rating":
                rating = float(user_feedback.feedback_value)
                await self._adjust_preference_weights(model, user_feedback, rating)
            
            elif user_feedback.feedback_type == "correction":
                await self._learn_from_correction(model, user_feedback)
            
            elif user_feedback.feedback_type == "preference":
                await self._update_preference_directly(model, user_feedback)
            
            # Update model version and save
            model.model_version += 1
            model.last_updated = datetime.now()
            await self._save_user_model(model)
            
            logger.info(f"Adapted model for user {user_feedback.user_id} based on feedback")
            
        except Exception as e:
            logger.error(f"Error adapting to feedback: {e}")
    
    async def get_user_behavior_patterns(self, user_id: str) -> List[BehaviorPattern]:
        """
        Get detected behavior patterns for a user.
        
        Args:
            user_id: User ID to get patterns for
            
        Returns:
            List of behavior patterns
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT pattern_id, pattern_type, pattern_data, confidence, 
                           frequency, first_detected, last_updated, metadata
                    FROM behavior_patterns 
                    WHERE user_id = ? 
                    ORDER BY confidence DESC, frequency DESC
                """, (user_id,))
                
                patterns = []
                for row in cursor.fetchall():
                    pattern = BehaviorPattern(
                        pattern_id=row[0],
                        user_id=user_id,
                        pattern_type=row[1],
                        pattern_data=json.loads(row[2]),
                        confidence=row[3],
                        frequency=row[4],
                        first_detected=datetime.fromisoformat(row[5]),
                        last_updated=datetime.fromisoformat(row[6]),
                        metadata=json.loads(row[7]) if row[7] else {}
                    )
                    patterns.append(pattern)
                
                return patterns
                
        except Exception as e:
            logger.error(f"Error getting behavior patterns: {e}")
            return []
    
    async def _get_user_model(self, user_id: str) -> LearningModel:
        """Get or create a learning model for a user"""
        if user_id in self._user_models:
            return self._user_models[user_id]
        
        # Try to load from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT model_version, model_data, performance_metrics, metadata
                FROM learning_models WHERE user_id = ?
            """, (user_id,))
            row = cursor.fetchone()
            
            if row:
                model_data = json.loads(row[1])
                model = LearningModel(
                    user_id=user_id,
                    model_version=row[0],
                    behavior_patterns={},
                    preference_weights=model_data.get("preference_weights", {}),
                    adaptation_history=model_data.get("adaptation_history", []),
                    performance_metrics=json.loads(row[2]) if row[2] else {},
                    last_updated=datetime.now(),
                    metadata=json.loads(row[3]) if row[3] else {}
                )
            else:
                # Create new model
                model = LearningModel(
                    user_id=user_id,
                    model_version=1,
                    behavior_patterns={},
                    preference_weights=self._get_default_preference_weights(),
                    adaptation_history=[],
                    performance_metrics={},
                    last_updated=datetime.now()
                )
                await self._save_user_model(model)
        
        # Load behavior patterns
        patterns = await self.get_user_behavior_patterns(user_id)
        model.behavior_patterns = {p.pattern_id: p for p in patterns}
        
        self._user_models[user_id] = model
        return model
    
    async def _analyze_interaction_patterns(self, interaction: Interaction) -> List[BehaviorPattern]:
        """Analyze an interaction for behavior patterns"""
        patterns = []
        
        # Time-based patterns
        time_pattern = await self._detect_time_pattern(interaction)
        if time_pattern:
            patterns.append(time_pattern)
        
        # Content-based patterns
        content_pattern = await self._detect_content_pattern(interaction)
        if content_pattern:
            patterns.append(content_pattern)
        
        # Sequence patterns
        sequence_pattern = await self._detect_sequence_pattern(interaction)
        if sequence_pattern:
            patterns.append(sequence_pattern)
        
        return patterns
    
    async def _detect_time_pattern(self, interaction: Interaction) -> Optional[BehaviorPattern]:
        """Detect time-based usage patterns"""
        try:
            hour = interaction.timestamp.hour
            day_of_week = interaction.timestamp.weekday()
            
            # Check if this is a consistent time pattern
            with sqlite3.connect(self.db_path) as conn:
                # First check if we have enough interactions at this hour
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM user_interactions 
                    WHERE user_id = ? AND 
                          CAST(strftime('%H', timestamp) AS INTEGER) = ?
                """, (interaction.user_id, hour))
                
                count = cursor.fetchone()[0]
                
                if count >= self.min_interactions_for_pattern:
                    pattern_id = f"time_{interaction.user_id}_{hour}_{day_of_week}"
                    return BehaviorPattern(
                        pattern_id=pattern_id,
                        user_id=interaction.user_id,
                        pattern_type="time_preference",
                        pattern_data={
                            "hour": hour,
                            "day_of_week": day_of_week,
                            "interaction_type": interaction.interaction_type.value
                        },
                        confidence=min(count / 20.0, 1.0),  # Max confidence at 20 interactions
                        frequency=count,
                        first_detected=datetime.now(),
                        last_updated=datetime.now()
                    )
        except Exception as e:
            logger.error(f"Error detecting time pattern: {e}")
        
        return None
    
    async def _detect_content_pattern(self, interaction: Interaction) -> Optional[BehaviorPattern]:
        """Detect content-based patterns in user queries"""
        try:
            # Simple keyword-based pattern detection
            content_lower = interaction.content.lower()
            keywords = content_lower.split()
            
            # Look for recurring keywords
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT content FROM user_interactions 
                    WHERE user_id = ? AND interaction_type = ?
                    ORDER BY timestamp DESC LIMIT 50
                """, (interaction.user_id, interaction.interaction_type.value))
                
                recent_contents = [row[0].lower() for row in cursor.fetchall()]
                
                # Find common keywords
                all_keywords = []
                for content in recent_contents:
                    all_keywords.extend(content.split())
                
                keyword_counts = Counter(all_keywords)
                common_keywords = [k for k, v in keyword_counts.items() 
                                 if v >= self.min_interactions_for_pattern and len(k) > 3]
                
                if common_keywords:
                    pattern_id = f"content_{interaction.user_id}_{hash('_'.join(sorted(common_keywords[:3])))}"
                    return BehaviorPattern(
                        pattern_id=pattern_id,
                        user_id=interaction.user_id,
                        pattern_type="content_preference",
                        pattern_data={
                            "common_keywords": common_keywords[:10],
                            "interaction_type": interaction.interaction_type.value
                        },
                        confidence=min(len(common_keywords) / 10.0, 1.0),
                        frequency=len(recent_contents),
                        first_detected=datetime.now(),
                        last_updated=datetime.now()
                    )
        except Exception as e:
            logger.error(f"Error detecting content pattern: {e}")
        
        return None
    
    async def _detect_sequence_pattern(self, interaction: Interaction) -> Optional[BehaviorPattern]:
        """Detect sequential patterns in user interactions"""
        try:
            # Get recent interactions to detect sequences
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT interaction_type, content FROM user_interactions 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC LIMIT 10
                """, (interaction.user_id,))
                
                recent_interactions = cursor.fetchall()
                
                if len(recent_interactions) >= 3:
                    # Look for common sequences
                    sequence = [row[0] for row in recent_interactions[:3]]
                    sequence_str = "_".join(sequence)
                    
                    # Check if this sequence appears frequently
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM (
                            SELECT interaction_type,
                                   LAG(interaction_type, 1) OVER (ORDER BY timestamp) as prev1,
                                   LAG(interaction_type, 2) OVER (ORDER BY timestamp) as prev2
                            FROM user_interactions 
                            WHERE user_id = ?
                        ) WHERE interaction_type = ? AND prev1 = ? AND prev2 = ?
                    """, (interaction.user_id, sequence[0], sequence[1], sequence[2]))
                    
                    count = cursor.fetchone()[0]
                    
                    if count >= self.min_interactions_for_pattern:
                        pattern_id = f"sequence_{interaction.user_id}_{hash(sequence_str)}"
                        return BehaviorPattern(
                            pattern_id=pattern_id,
                            user_id=interaction.user_id,
                            pattern_type="sequence_pattern",
                            pattern_data={
                                "sequence": sequence,
                                "sequence_length": 3
                            },
                            confidence=min(count / 10.0, 1.0),
                            frequency=count,
                            first_detected=datetime.now(),
                            last_updated=datetime.now()
                        )
        except Exception as e:
            logger.error(f"Error detecting sequence pattern: {e}")
        
        return None
    
    async def _update_behavior_pattern(self, pattern: BehaviorPattern) -> None:
        """Update or insert a behavior pattern in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO behavior_patterns 
                    (pattern_id, user_id, pattern_type, pattern_data, confidence, 
                     frequency, first_detected, last_updated, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.user_id,
                    pattern.pattern_type,
                    json.dumps(pattern.pattern_data),
                    pattern.confidence,
                    pattern.frequency,
                    pattern.first_detected,
                    pattern.last_updated,
                    json.dumps(pattern.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating behavior pattern: {e}")
    
    async def _update_user_model(self, model: LearningModel, interaction: Interaction, patterns: List[BehaviorPattern]) -> None:
        """Update user model based on interaction and detected patterns"""
        try:
            # Update adaptation history
            model.adaptation_history.append({
                "timestamp": datetime.now().isoformat(),
                "interaction_id": interaction.id,
                "patterns_detected": len(patterns),
                "interaction_type": interaction.interaction_type.value
            })
            
            # Keep only recent history (last 100 entries)
            if len(model.adaptation_history) > 100:
                model.adaptation_history = model.adaptation_history[-100:]
            
            # Update performance metrics
            await self._update_performance_metrics(model, interaction)
            
            # Save updated model
            await self._save_user_model(model)
            
        except Exception as e:
            logger.error(f"Error updating user model: {e}")
    
    async def _save_user_model(self, model: LearningModel) -> None:
        """Save user learning model to database"""
        try:
            model_data = {
                "preference_weights": model.preference_weights,
                "adaptation_history": model.adaptation_history
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO learning_models 
                    (user_id, model_version, model_data, performance_metrics, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    model.user_id,
                    model.model_version,
                    json.dumps(model_data),
                    json.dumps(model.performance_metrics),
                    json.dumps(model.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving user model: {e}")
    
    async def _store_feedback(self, feedback: UserFeedback) -> None:
        """Store user feedback in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_feedback 
                    (feedback_id, user_id, interaction_id, feedback_type, 
                     feedback_value, context_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    feedback.feedback_id,
                    feedback.user_id,
                    feedback.interaction_id,
                    feedback.feedback_type,
                    json.dumps(feedback.feedback_value),
                    json.dumps(feedback.context_data)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
    
    async def _apply_feedback_to_model(self, model: LearningModel, feedback: UserFeedback) -> None:
        """Apply feedback to update the learning model"""
        try:
            if feedback.feedback_type == "rating":
                rating = float(feedback.feedback_value)
                # Adjust preference weights based on rating
                if rating >= 4.0:  # Positive feedback
                    self._boost_preference_weights(model, feedback.context_data, 0.1)
                elif rating <= 2.0:  # Negative feedback
                    self._reduce_preference_weights(model, feedback.context_data, 0.1)
            
            # Update performance metrics
            model.performance_metrics["feedback_count"] = model.performance_metrics.get("feedback_count", 0) + 1
            model.performance_metrics["last_feedback"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error applying feedback to model: {e}")
    
    def _boost_preference_weights(self, model: LearningModel, context_data: Dict[str, Any], boost_factor: float) -> None:
        """Boost preference weights based on positive feedback"""
        for key, value in context_data.items():
            if isinstance(value, str) and key in ["interaction_type", "content_category"]:
                current_weight = model.preference_weights.get(f"{key}_{value}", 0.5)
                model.preference_weights[f"{key}_{value}"] = min(1.0, current_weight + boost_factor)
    
    def _reduce_preference_weights(self, model: LearningModel, context_data: Dict[str, Any], reduction_factor: float) -> None:
        """Reduce preference weights based on negative feedback"""
        for key, value in context_data.items():
            if isinstance(value, str) and key in ["interaction_type", "content_category"]:
                current_weight = model.preference_weights.get(f"{key}_{value}", 0.5)
                model.preference_weights[f"{key}_{value}"] = max(0.0, current_weight - reduction_factor)
    
    async def _update_preferences_from_feedback(self, feedback: UserFeedback) -> None:
        """Update user preferences based on feedback"""
        try:
            if feedback.feedback_type == "preference":
                context = await self.context_manager.get_user_context(feedback.user_id)
                
                # Update learning preferences
                if isinstance(feedback.feedback_value, dict):
                    for key, value in feedback.feedback_value.items():
                        if key in context.preferences.learning_preferences:
                            context.preferences.learning_preferences[key] = value
                
                context.preferences.updated_at = datetime.now()
                await self.context_manager.update_user_context(context)
                
        except Exception as e:
            logger.error(f"Error updating preferences from feedback: {e}")
    
    async def _log_learning_event(self, user_id: str, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log a learning event for tracking and debugging"""
        try:
            model = await self._get_user_model(user_id)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO learning_events 
                    (user_id, event_type, event_data, model_version)
                    VALUES (?, ?, ?, ?)
                """, (
                    user_id,
                    event_type,
                    json.dumps(event_data),
                    model.model_version
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging learning event: {e}")
    
    def _get_default_preference_weights(self) -> Dict[str, float]:
        """Get default preference weights for new users"""
        return {
            "interaction_type_query": 0.5,
            "interaction_type_command": 0.5,
            "interaction_type_feedback": 0.5,
            "interaction_type_file_access": 0.5,
            "interaction_type_screen_context": 0.5,
            "time_morning": 0.5,
            "time_afternoon": 0.5,
            "time_evening": 0.5,
            "content_technical": 0.5,
            "content_personal": 0.5,
            "content_creative": 0.5
        }
    
    async def _get_time_based_suggestions(self, model: LearningModel, context: UserContext) -> List[Dict[str, Any]]:
        """Generate time-based suggestions"""
        suggestions = []
        current_hour = datetime.now().hour
        
        # Check for time-based patterns
        for pattern in model.behavior_patterns.values():
            if pattern.pattern_type == "time_preference" and pattern.confidence > self.pattern_confidence_threshold:
                pattern_hour = pattern.pattern_data.get("hour")
                if pattern_hour and abs(current_hour - pattern_hour) <= 1:
                    suggestions.append({
                        "type": "time_based",
                        "suggestion": f"Based on your usage patterns, you often work on {pattern.pattern_data.get('interaction_type', 'tasks')} around this time.",
                        "confidence": pattern.confidence,
                        "pattern_id": pattern.pattern_id
                    })
        
        return suggestions
    
    async def _get_task_based_suggestions(self, model: LearningModel, context: UserContext) -> List[Dict[str, Any]]:
        """Generate task-based suggestions"""
        suggestions = []
        
        # Suggest based on current tasks and patterns
        if context.task_context.current_tasks:
            for task in context.task_context.current_tasks:
                suggestions.append({
                    "type": "task_based",
                    "suggestion": f"Would you like help with your current task: {task}?",
                    "confidence": 0.8,
                    "task": task
                })
        
        return suggestions
    
    async def _get_pattern_based_suggestions(self, model: LearningModel, context: UserContext) -> List[Dict[str, Any]]:
        """Generate pattern-based suggestions"""
        suggestions = []
        
        # Suggest based on detected patterns
        for pattern in model.behavior_patterns.values():
            if pattern.confidence > self.pattern_confidence_threshold:
                if pattern.pattern_type == "content_preference":
                    keywords = pattern.pattern_data.get("common_keywords", [])
                    if keywords:
                        suggestions.append({
                            "type": "pattern_based",
                            "suggestion": f"I notice you often ask about {', '.join(keywords[:3])}. Would you like related resources?",
                            "confidence": pattern.confidence,
                            "pattern_id": pattern.pattern_id
                        })
        
        return suggestions
    
    async def _rank_suggestions(self, suggestions: List[Dict[str, Any]], model: LearningModel, context: UserContext) -> List[Dict[str, Any]]:
        """Rank suggestions by relevance and user preferences"""
        # Simple ranking based on confidence and recency
        for suggestion in suggestions:
            base_score = suggestion.get("confidence", 0.5)
            
            # Boost score based on user preferences
            suggestion_type = suggestion.get("type", "")
            preference_key = f"suggestion_{suggestion_type}"
            preference_weight = model.preference_weights.get(preference_key, 0.5)
            
            suggestion["score"] = base_score * preference_weight
        
        # Sort by score descending
        return sorted(suggestions, key=lambda x: x.get("score", 0), reverse=True)
    
    async def _adjust_preference_weights(self, model: LearningModel, feedback: UserFeedback, rating: float) -> None:
        """Adjust preference weights based on rating feedback"""
        # Normalize rating to -1 to 1 scale
        normalized_rating = (rating - 3.0) / 2.0  # Assuming 1-5 scale
        adjustment = normalized_rating * self.adaptation_learning_rate
        
        # Apply adjustment to relevant preferences
        context_data = feedback.context_data
        for key, value in context_data.items():
            if isinstance(value, str):
                preference_key = f"{key}_{value}"
                current_weight = model.preference_weights.get(preference_key, 0.5)
                new_weight = max(0.0, min(1.0, current_weight + adjustment))
                model.preference_weights[preference_key] = new_weight
    
    async def _learn_from_correction(self, model: LearningModel, feedback: UserFeedback) -> None:
        """Learn from user corrections"""
        # Store correction for future reference
        correction_data = {
            "original_response": feedback.context_data.get("original_response", ""),
            "corrected_response": feedback.feedback_value,
            "timestamp": feedback.timestamp.isoformat()
        }
        
        model.metadata.setdefault("corrections", []).append(correction_data)
        
        # Keep only recent corrections
        if len(model.metadata["corrections"]) > 50:
            model.metadata["corrections"] = model.metadata["corrections"][-50:]
    
    async def _update_preference_directly(self, model: LearningModel, feedback: UserFeedback) -> None:
        """Update preferences directly from user feedback"""
        if isinstance(feedback.feedback_value, dict):
            for key, value in feedback.feedback_value.items():
                model.preference_weights[key] = float(value)
    
    async def _update_performance_metrics(self, model: LearningModel, interaction: Interaction) -> None:
        """Update performance metrics based on interaction"""
        model.performance_metrics["total_interactions"] = model.performance_metrics.get("total_interactions", 0) + 1
        model.performance_metrics["last_interaction"] = interaction.timestamp.isoformat()
        
        # Calculate average response quality if feedback is available
        if interaction.feedback_score is not None:
            current_avg = model.performance_metrics.get("avg_feedback_score", 3.0)
            total_feedback = model.performance_metrics.get("feedback_count", 0)
            
            new_avg = ((current_avg * total_feedback) + interaction.feedback_score) / (total_feedback + 1)
            model.performance_metrics["avg_feedback_score"] = new_avg    

    # Continuous Learning and Improvement Methods
    
    async def enable_continuous_learning(self, user_id: str) -> None:
        """Enable continuous learning for a user"""
        try:
            self.continuous_learning_enabled = True
            
            # Initialize continuous learning state
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO continuous_learning_state 
                    (user_id, last_optimization, optimization_count, performance_trend, adaptation_history)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id,
                    datetime.now().isoformat(),
                    0,
                    json.dumps({"trend": "stable", "direction": "neutral"}),
                    json.dumps([])
                ))
                conn.commit()
            
            await self._log_learning_event(user_id, "continuous_learning_enabled", {
                "enabled": True,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Continuous learning enabled for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error enabling continuous learning: {e}")
    
    async def disable_continuous_learning(self, user_id: str) -> None:
        """Disable continuous learning for a user"""
        try:
            self.continuous_learning_enabled = False
            
            await self._log_learning_event(user_id, "continuous_learning_disabled", {
                "enabled": False,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Continuous learning disabled for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error disabling continuous learning: {e}")
    
    async def track_performance_metric(self, user_id: str, metric_name: str, metric_value: float, context_data: Dict[str, Any] = None) -> None:
        """Track a performance metric for continuous learning"""
        try:
            model = await self._get_user_model(user_id)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (user_id, metric_name, metric_value, model_version, context_data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id,
                    metric_name,
                    metric_value,
                    model.model_version,
                    json.dumps(context_data or {})
                ))
                conn.commit()
            
            # Trigger continuous optimization if needed
            if self.continuous_learning_enabled:
                await self._check_optimization_trigger(user_id, metric_name, metric_value)
            
        except Exception as e:
            logger.error(f"Error tracking performance metric: {e}")
    
    async def get_performance_trends(self, user_id: str, metric_name: str = None, days: int = 30) -> Dict[str, Any]:
        """Get performance trends for a user"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                if metric_name:
                    cursor = conn.execute("""
                        SELECT metric_name, metric_value, timestamp 
                        FROM performance_metrics 
                        WHERE user_id = ? AND metric_name = ? AND timestamp > ?
                        ORDER BY timestamp ASC
                    """, (user_id, metric_name, cutoff_date.isoformat()))
                else:
                    cursor = conn.execute("""
                        SELECT metric_name, metric_value, timestamp 
                        FROM performance_metrics 
                        WHERE user_id = ? AND timestamp > ?
                        ORDER BY timestamp ASC
                    """, (user_id, cutoff_date.isoformat()))
                
                metrics = cursor.fetchall()
            
            # Analyze trends
            trends = {}
            metric_groups = defaultdict(list)
            
            for metric_name, value, timestamp in metrics:
                metric_groups[metric_name].append({
                    "value": value,
                    "timestamp": timestamp
                })
            
            for metric, values in metric_groups.items():
                if len(values) >= 2:
                    # Calculate trend
                    recent_values = [v["value"] for v in values[-10:]]  # Last 10 values
                    older_values = [v["value"] for v in values[:-10]] if len(values) > 10 else []
                    
                    if older_values:
                        recent_avg = np.mean(recent_values)
                        older_avg = np.mean(older_values)
                        trend_direction = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
                        trend_magnitude = abs(recent_avg - older_avg) / older_avg if older_avg != 0 else 0
                    else:
                        trend_direction = "insufficient_data"
                        trend_magnitude = 0
                    
                    trends[metric] = {
                        "direction": trend_direction,
                        "magnitude": trend_magnitude,
                        "recent_average": np.mean(recent_values),
                        "total_points": len(values),
                        "latest_value": values[-1]["value"],
                        "latest_timestamp": values[-1]["timestamp"]
                    }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {}
    
    async def optimize_learning_parameters(self, user_id: str) -> Dict[str, Any]:
        """Optimize learning parameters based on performance trends"""
        try:
            model = await self._get_user_model(user_id)
            trends = await self.get_performance_trends(user_id)
            
            optimization_results = {
                "optimizations_applied": [],
                "performance_before": {},
                "performance_after": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Analyze feedback score trends
            if "avg_feedback_score" in trends:
                feedback_trend = trends["avg_feedback_score"]
                optimization_results["performance_before"]["avg_feedback_score"] = feedback_trend["recent_average"]
                
                if feedback_trend["direction"] == "declining" and feedback_trend["magnitude"] > self.improvement_threshold:
                    # Increase adaptation learning rate
                    old_rate = self.adaptation_learning_rate
                    self.adaptation_learning_rate = min(0.3, self.adaptation_learning_rate * 1.2)
                    optimization_results["optimizations_applied"].append({
                        "parameter": "adaptation_learning_rate",
                        "old_value": old_rate,
                        "new_value": self.adaptation_learning_rate,
                        "reason": "declining_feedback_scores"
                    })
                
                elif feedback_trend["direction"] == "improving":
                    # Slightly decrease learning rate for stability
                    old_rate = self.adaptation_learning_rate
                    self.adaptation_learning_rate = max(0.05, self.adaptation_learning_rate * 0.9)
                    optimization_results["optimizations_applied"].append({
                        "parameter": "adaptation_learning_rate",
                        "old_value": old_rate,
                        "new_value": self.adaptation_learning_rate,
                        "reason": "improving_feedback_scores"
                    })
            
            # Analyze pattern detection effectiveness
            pattern_count = len(model.behavior_patterns)
            if pattern_count < 3:  # Too few patterns detected
                old_threshold = self.pattern_confidence_threshold
                self.pattern_confidence_threshold = max(0.5, self.pattern_confidence_threshold * 0.8)
                optimization_results["optimizations_applied"].append({
                    "parameter": "pattern_confidence_threshold",
                    "old_value": old_threshold,
                    "new_value": self.pattern_confidence_threshold,
                    "reason": "insufficient_patterns_detected"
                })
            
            elif pattern_count > 20:  # Too many patterns, might be noise
                old_threshold = self.pattern_confidence_threshold
                self.pattern_confidence_threshold = min(0.9, self.pattern_confidence_threshold * 1.1)
                optimization_results["optimizations_applied"].append({
                    "parameter": "pattern_confidence_threshold",
                    "old_value": old_threshold,
                    "new_value": self.pattern_confidence_threshold,
                    "reason": "too_many_patterns_detected"
                })
            
            # Update continuous learning state
            await self._update_continuous_learning_state(user_id, optimization_results)
            
            # Log optimization event
            await self._log_learning_event(user_id, "parameter_optimization", optimization_results)
            
            logger.info(f"Optimized learning parameters for user {user_id}: {len(optimization_results['optimizations_applied'])} changes")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing learning parameters: {e}")
            return {"error": str(e)}
    
    async def suggest_feature_improvements(self, user_id: str) -> List[Dict[str, Any]]:
        """Suggest feature improvements based on usage patterns and feedback"""
        try:
            model = await self._get_user_model(user_id)
            patterns = await self.get_user_behavior_patterns(user_id)
            trends = await self.get_performance_trends(user_id)
            
            suggestions = []
            
            # Analyze interaction patterns for feature suggestions
            interaction_types = defaultdict(int)
            for pattern in patterns:
                if pattern.pattern_type == "content_preference":
                    keywords = pattern.pattern_data.get("common_keywords", [])
                    for keyword in keywords:
                        interaction_types[keyword] += pattern.frequency
            
            # Suggest new features based on common keywords
            if "automation" in interaction_types and interaction_types["automation"] > 10:
                suggestions.append({
                    "type": "feature_enhancement",
                    "suggestion": "Add automated workflow creation based on repetitive tasks",
                    "confidence": min(1.0, interaction_types["automation"] / 20.0),
                    "priority": "high",
                    "category": "automation",
                    "evidence": f"User mentioned automation {interaction_types['automation']} times"
                })
            
            if "integration" in interaction_types and interaction_types["integration"] > 8:
                suggestions.append({
                    "type": "feature_enhancement",
                    "suggestion": "Expand third-party tool integrations",
                    "confidence": min(1.0, interaction_types["integration"] / 15.0),
                    "priority": "medium",
                    "category": "integration",
                    "evidence": f"User mentioned integration {interaction_types['integration']} times"
                })
            
            # Analyze performance trends for optimization suggestions
            if "response_time" in trends:
                response_trend = trends["response_time"]
                if response_trend["direction"] == "declining" and response_trend["magnitude"] > 0.2:
                    suggestions.append({
                        "type": "performance_optimization",
                        "suggestion": "Optimize response generation for faster performance",
                        "confidence": 0.9,
                        "priority": "high",
                        "category": "performance",
                        "evidence": f"Response time declining by {response_trend['magnitude']:.2%}"
                    })
            
            # Suggest UI/UX improvements based on interaction patterns
            time_patterns = [p for p in patterns if p.pattern_type == "time_preference"]
            if len(time_patterns) > 3:
                peak_hours = [p.pattern_data.get("hour") for p in time_patterns if p.confidence > 0.8]
                if peak_hours:
                    suggestions.append({
                        "type": "ui_enhancement",
                        "suggestion": f"Add quick access features for peak usage hours ({min(peak_hours)}-{max(peak_hours)})",
                        "confidence": 0.8,
                        "priority": "medium",
                        "category": "user_experience",
                        "evidence": f"Strong time-based usage patterns detected"
                    })
            
            # Store suggestions in database
            for suggestion in suggestions:
                if suggestion["confidence"] >= self.feature_suggestion_confidence:
                    await self._store_feature_suggestion(user_id, suggestion)
            
            # Filter by confidence threshold
            high_confidence_suggestions = [s for s in suggestions if s["confidence"] >= self.feature_suggestion_confidence]
            
            logger.info(f"Generated {len(high_confidence_suggestions)} feature suggestions for user {user_id}")
            
            return high_confidence_suggestions
            
        except Exception as e:
            logger.error(f"Error suggesting feature improvements: {e}")
            return []
    
    async def adapt_to_changing_patterns(self, user_id: str) -> Dict[str, Any]:
        """Adapt to changing user patterns and preferences"""
        try:
            model = await self._get_user_model(user_id)
            current_patterns = await self.get_user_behavior_patterns(user_id)
            
            adaptation_results = {
                "patterns_updated": 0,
                "patterns_deprecated": 0,
                "new_patterns_detected": 0,
                "preference_adjustments": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Check for pattern changes
            cutoff_date = datetime.now() - timedelta(days=30)
            recent_interactions = await self._get_recent_interactions(user_id, cutoff_date)
            
            if len(recent_interactions) >= self.min_interactions_for_pattern:
                # Detect new patterns from recent interactions
                new_patterns = []
                for interaction in recent_interactions[-10:]:  # Analyze last 10 interactions
                    patterns = await self._analyze_interaction_patterns(interaction)
                    new_patterns.extend(patterns)
                
                # Compare with existing patterns
                for new_pattern in new_patterns:
                    existing_pattern = model.behavior_patterns.get(new_pattern.pattern_id)
                    
                    if existing_pattern:
                        # Update existing pattern
                        if abs(new_pattern.confidence - existing_pattern.confidence) > 0.1:
                            existing_pattern.confidence = (existing_pattern.confidence + new_pattern.confidence) / 2
                            existing_pattern.last_updated = datetime.now()
                            await self._update_behavior_pattern(existing_pattern)
                            adaptation_results["patterns_updated"] += 1
                    else:
                        # New pattern detected
                        await self._update_behavior_pattern(new_pattern)
                        model.behavior_patterns[new_pattern.pattern_id] = new_pattern
                        adaptation_results["new_patterns_detected"] += 1
                
                # Deprecate old patterns that are no longer relevant
                for pattern_id, pattern in list(model.behavior_patterns.items()):
                    days_since_update = (datetime.now() - pattern.last_updated).days
                    if days_since_update > 60 and pattern.confidence < 0.5:
                        # Remove deprecated pattern
                        del model.behavior_patterns[pattern_id]
                        adaptation_results["patterns_deprecated"] += 1
                
                # Adjust preferences based on recent feedback
                recent_feedback = await self._get_recent_feedback(user_id, cutoff_date)
                if recent_feedback:
                    for feedback in recent_feedback:
                        if feedback.feedback_type == "rating":
                            rating = float(feedback.feedback_value)
                            if rating != 3.0:  # Not neutral
                                await self._adjust_preference_weights(model, feedback, rating)
                                adaptation_results["preference_adjustments"] += 1
                
                # Update model version and save
                model.model_version += 1
                model.last_updated = datetime.now()
                await self._save_user_model(model)
            
            # Log adaptation event
            await self._log_learning_event(user_id, "pattern_adaptation", adaptation_results)
            
            logger.info(f"Adapted to changing patterns for user {user_id}: {adaptation_results}")
            
            return adaptation_results
            
        except Exception as e:
            logger.error(f"Error adapting to changing patterns: {e}")
            return {"error": str(e)}
    
    async def get_learning_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about the learning process and model performance"""
        try:
            model = await self._get_user_model(user_id)
            patterns = await self.get_user_behavior_patterns(user_id)
            trends = await self.get_performance_trends(user_id)
            
            insights = {
                "model_info": {
                    "version": model.model_version,
                    "last_updated": model.last_updated.isoformat(),
                    "total_patterns": len(patterns),
                    "adaptation_events": len(model.adaptation_history)
                },
                "pattern_analysis": {
                    "most_confident_patterns": [],
                    "pattern_types": defaultdict(int),
                    "average_confidence": 0.0
                },
                "performance_summary": {
                    "trends": trends,
                    "key_metrics": model.performance_metrics
                },
                "learning_effectiveness": {
                    "adaptation_rate": 0.0,
                    "pattern_stability": 0.0,
                    "feedback_incorporation": 0.0
                }
            }
            
            # Analyze patterns
            if patterns:
                insights["pattern_analysis"]["average_confidence"] = np.mean([p.confidence for p in patterns])
                insights["pattern_analysis"]["most_confident_patterns"] = [
                    {
                        "pattern_id": p.pattern_id,
                        "type": p.pattern_type,
                        "confidence": p.confidence,
                        "frequency": p.frequency
                    }
                    for p in sorted(patterns, key=lambda x: x.confidence, reverse=True)[:5]
                ]
                
                for pattern in patterns:
                    insights["pattern_analysis"]["pattern_types"][pattern.pattern_type] += 1
            
            # Calculate learning effectiveness metrics
            if model.adaptation_history:
                recent_adaptations = [a for a in model.adaptation_history if 
                                    datetime.fromisoformat(a["timestamp"]) > datetime.now() - timedelta(days=30)]
                insights["learning_effectiveness"]["adaptation_rate"] = len(recent_adaptations) / 30.0
            
            # Pattern stability (how consistent patterns are over time)
            stable_patterns = [p for p in patterns if p.confidence > 0.7]
            if patterns:
                insights["learning_effectiveness"]["pattern_stability"] = len(stable_patterns) / len(patterns)
            
            # Feedback incorporation rate
            feedback_events = [e for e in model.adaptation_history if "feedback" in e.get("interaction_type", "")]
            if model.adaptation_history:
                insights["learning_effectiveness"]["feedback_incorporation"] = len(feedback_events) / len(model.adaptation_history)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {"error": str(e)}
    
    # Helper methods for continuous learning
    
    async def _check_optimization_trigger(self, user_id: str, metric_name: str, metric_value: float) -> None:
        """Check if optimization should be triggered based on performance metrics"""
        try:
            # Get recent metrics for comparison
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT metric_value FROM performance_metrics 
                    WHERE user_id = ? AND metric_name = ?
                    ORDER BY timestamp DESC LIMIT 10
                """, (user_id, metric_name))
                
                recent_values = [row[0] for row in cursor.fetchall()]
            
            if len(recent_values) >= 5:
                # Check if there's a significant trend
                recent_avg = np.mean(recent_values[:5])
                older_avg = np.mean(recent_values[5:]) if len(recent_values) > 5 else recent_avg
                
                if abs(recent_avg - older_avg) / older_avg > self.improvement_threshold:
                    # Trigger optimization
                    await self.optimize_learning_parameters(user_id)
                    
        except Exception as e:
            logger.error(f"Error checking optimization trigger: {e}")
    
    async def _update_continuous_learning_state(self, user_id: str, optimization_results: Dict[str, Any]) -> None:
        """Update the continuous learning state"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT optimization_count, adaptation_history FROM continuous_learning_state 
                    WHERE user_id = ?
                """, (user_id,))
                row = cursor.fetchone()
                
                if row:
                    optimization_count = row[0] + 1
                    adaptation_history = json.loads(row[1]) if row[1] else []
                else:
                    optimization_count = 1
                    adaptation_history = []
                
                # Add current optimization to history
                adaptation_history.append(optimization_results)
                
                # Keep only recent history
                if len(adaptation_history) > 50:
                    adaptation_history = adaptation_history[-50:]
                
                conn.execute("""
                    INSERT OR REPLACE INTO continuous_learning_state 
                    (user_id, last_optimization, optimization_count, adaptation_history)
                    VALUES (?, ?, ?, ?)
                """, (
                    user_id,
                    datetime.now().isoformat(),
                    optimization_count,
                    json.dumps(adaptation_history)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating continuous learning state: {e}")
    
    async def _store_feature_suggestion(self, user_id: str, suggestion: Dict[str, Any]) -> None:
        """Store a feature suggestion in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO feature_suggestions 
                    (user_id, suggestion_type, suggestion_data, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user_id,
                    suggestion["type"],
                    json.dumps(suggestion),
                    suggestion["confidence"],
                    json.dumps({"priority": suggestion.get("priority", "medium")})
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing feature suggestion: {e}")
    
    async def _get_recent_interactions(self, user_id: str, cutoff_date: datetime) -> List[Interaction]:
        """Get recent interactions for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, interaction_type, content, response, timestamp, context_data, feedback_score
                    FROM user_interactions 
                    WHERE user_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                """, (user_id, cutoff_date.isoformat()))
                
                interactions = []
                for row in cursor.fetchall():
                    interaction = Interaction(
                        id=row[0],
                        user_id=user_id,
                        interaction_type=InteractionType(row[1]),
                        content=row[2],
                        response=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        context_data=json.loads(row[5]) if row[5] else {},
                        feedback_score=row[6]
                    )
                    interactions.append(interaction)
                
                return interactions
                
        except Exception as e:
            logger.error(f"Error getting recent interactions: {e}")
            return []
    
    async def _get_recent_feedback(self, user_id: str, cutoff_date: datetime) -> List[UserFeedback]:
        """Get recent feedback for a user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT feedback_id, interaction_id, feedback_type, feedback_value, timestamp, context_data
                    FROM user_feedback 
                    WHERE user_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                """, (user_id, cutoff_date.isoformat()))
                
                feedback_list = []
                for row in cursor.fetchall():
                    feedback = UserFeedback(
                        feedback_id=row[0],
                        user_id=user_id,
                        interaction_id=row[1],
                        feedback_type=row[2],
                        feedback_value=json.loads(row[3]),
                        timestamp=datetime.fromisoformat(row[4]),
                        context_data=json.loads(row[5]) if row[5] else {}
                    )
                    feedback_list.append(feedback)
                
                return feedback_list
                
        except Exception as e:
            logger.error(f"Error getting recent feedback: {e}")
            return []