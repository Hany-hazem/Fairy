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
    to better serve individual users.
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