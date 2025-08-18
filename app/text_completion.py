"""
Text Completion Module

This module provides intelligent text completion and suggestion capabilities
for the personal assistant, offering context-aware suggestions and learning
from user patterns.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import time

logger = logging.getLogger(__name__)


class CompletionType(Enum):
    """Types of text completions"""
    WORD = "word"
    PHRASE = "phrase"
    SENTENCE = "sentence"
    COMMAND = "command"
    FILE_PATH = "file_path"
    VARIABLE = "variable"
    FUNCTION = "function"
    TEMPLATE = "template"


class ContextType(Enum):
    """Types of text contexts"""
    GENERAL = "general"
    CODE = "code"
    EMAIL = "email"
    DOCUMENT = "document"
    CHAT = "chat"
    SEARCH = "search"
    COMMAND_LINE = "command_line"
    FORM = "form"


@dataclass
class TextContext:
    """Context information for text completion"""
    text_before: str
    text_after: str
    cursor_position: int
    context_type: ContextType
    application: Optional[str] = None
    file_type: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Completion:
    """A text completion suggestion"""
    text: str
    completion_type: CompletionType
    confidence: float
    description: Optional[str] = None
    category: Optional[str] = None
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    last_used: Optional[datetime] = None


@dataclass
class CompletionSettings:
    """Settings for text completion"""
    enabled: bool = True
    max_suggestions: int = 5
    min_confidence: float = 0.3
    auto_complete_threshold: float = 0.9
    learning_enabled: bool = True
    context_aware: bool = True
    application_specific: bool = True
    show_descriptions: bool = True
    completion_delay: float = 0.5  # Seconds to wait before showing suggestions
    max_completion_length: int = 100


class CompletionEngine:
    """Core text completion engine"""
    
    def __init__(self, settings: CompletionSettings = None):
        self.settings = settings or CompletionSettings()
        self.completion_cache: Dict[str, List[Completion]] = {}
        self.user_patterns: Dict[str, Dict[str, Any]] = {}
        self.common_completions: Dict[str, List[Completion]] = {}
        self.application_completions: Dict[str, Dict[str, List[Completion]]] = {}
        
        # Initialize with common completions
        self._initialize_common_completions()
        
        logger.info("Text completion engine initialized")
    
    def _initialize_common_completions(self):
        """Initialize common text completions"""
        # Common words and phrases
        common_words = [
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
            "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
            "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy",
            "did", "its", "let", "put", "say", "she", "too", "use"
        ]
        
        self.common_completions["words"] = [
            Completion(
                text=word,
                completion_type=CompletionType.WORD,
                confidence=0.5,
                source="common"
            ) for word in common_words
        ]
        
        # Common phrases
        common_phrases = [
            "Thank you for",
            "I would like to",
            "Please let me know",
            "Looking forward to",
            "Best regards",
            "Kind regards",
            "Please find attached",
            "I hope this helps",
            "Let me know if you need",
            "I'll get back to you"
        ]
        
        self.common_completions["phrases"] = [
            Completion(
                text=phrase,
                completion_type=CompletionType.PHRASE,
                confidence=0.6,
                source="common"
            ) for phrase in common_phrases
        ]
        
        # Programming completions
        programming_completions = [
            "def __init__(self):",
            "if __name__ == '__main__':",
            "try:\n    \nexcept Exception as e:",
            "for i in range(",
            "while True:",
            "import ",
            "from ",
            "class ",
            "def ",
            "return ",
            "print(",
            "len(",
            "str(",
            "int(",
            "float("
        ]
        
        self.common_completions["programming"] = [
            Completion(
                text=comp,
                completion_type=CompletionType.TEMPLATE,
                confidence=0.7,
                source="programming"
            ) for comp in programming_completions
        ]
    
    async def get_completions(self, context: TextContext) -> List[Completion]:
        """Get text completions for the given context"""
        try:
            if not self.settings.enabled:
                return []
            
            # Extract the current word/phrase being typed
            current_input = self._extract_current_input(context)
            
            if not current_input or len(current_input) < 2:
                return []
            
            # Generate cache key
            cache_key = self._generate_cache_key(context, current_input)
            
            # Check cache first
            if cache_key in self.completion_cache:
                cached_completions = self.completion_cache[cache_key]
                return self._filter_and_rank_completions(cached_completions, current_input)
            
            # Generate new completions
            completions = []
            
            # Add context-specific completions
            completions.extend(await self._get_context_completions(context, current_input))
            
            # Add common completions
            completions.extend(self._get_common_completions(current_input))
            
            # Add user pattern completions
            completions.extend(await self._get_pattern_completions(context, current_input))
            
            # Add application-specific completions
            if context.application:
                completions.extend(self._get_application_completions(context.application, current_input))
            
            # Filter and rank completions
            final_completions = self._filter_and_rank_completions(completions, current_input)
            
            # Cache results
            self.completion_cache[cache_key] = final_completions
            
            return final_completions[:self.settings.max_suggestions]
            
        except Exception as e:
            logger.error(f"Error getting completions: {e}")
            return []
    
    def _extract_current_input(self, context: TextContext) -> str:
        """Extract the current word/phrase being typed"""
        text_before = context.text_before
        
        # Find the start of the current word
        word_chars = re.findall(r'\w+$', text_before)
        if word_chars:
            return word_chars[-1]
        
        # Check for partial phrases (for phrase completion)
        words = text_before.split()
        if len(words) >= 2:
            return ' '.join(words[-2:])
        elif words:
            return words[-1]
        
        return ""
    
    def _generate_cache_key(self, context: TextContext, current_input: str) -> str:
        """Generate cache key for completions"""
        key_parts = [
            current_input.lower(),
            context.context_type.value,
            context.application or "unknown",
            context.file_type or "unknown"
        ]
        return "|".join(key_parts)
    
    async def _get_context_completions(self, context: TextContext, current_input: str) -> List[Completion]:
        """Get completions based on context type"""
        completions = []
        
        if context.context_type == ContextType.CODE:
            completions.extend(self._get_code_completions(context, current_input))
        elif context.context_type == ContextType.EMAIL:
            completions.extend(self._get_email_completions(current_input))
        elif context.context_type == ContextType.COMMAND_LINE:
            completions.extend(self._get_command_completions(current_input))
        elif context.context_type == ContextType.DOCUMENT:
            completions.extend(self._get_document_completions(current_input))
        
        return completions
    
    def _get_code_completions(self, context: TextContext, current_input: str) -> List[Completion]:
        """Get code-specific completions"""
        completions = []
        
        # Language-specific completions
        if context.language == "python":
            python_keywords = [
                "def", "class", "if", "else", "elif", "for", "while", "try", "except",
                "import", "from", "return", "yield", "lambda", "with", "as", "pass",
                "break", "continue", "global", "nonlocal", "assert", "del", "raise"
            ]
            
            for keyword in python_keywords:
                if keyword.startswith(current_input.lower()):
                    completions.append(Completion(
                        text=keyword,
                        completion_type=CompletionType.WORD,
                        confidence=0.8,
                        source="python_keywords"
                    ))
        
        # Common programming patterns
        if current_input.lower().startswith("for"):
            completions.append(Completion(
                text="for i in range(len()):",
                completion_type=CompletionType.TEMPLATE,
                confidence=0.7,
                description="For loop with range and length",
                source="code_template"
            ))
        
        return completions
    
    def _get_email_completions(self, current_input: str) -> List[Completion]:
        """Get email-specific completions"""
        completions = []
        
        email_phrases = [
            "Dear Sir/Madam,",
            "Thank you for your email",
            "I hope this email finds you well",
            "Please let me know if you have any questions",
            "I look forward to hearing from you",
            "Best regards,",
            "Kind regards,",
            "Sincerely,",
            "Thank you for your time and consideration"
        ]
        
        for phrase in email_phrases:
            if phrase.lower().startswith(current_input.lower()):
                completions.append(Completion(
                    text=phrase,
                    completion_type=CompletionType.PHRASE,
                    confidence=0.7,
                    source="email_templates"
                ))
        
        return completions
    
    def _get_command_completions(self, current_input: str) -> List[Completion]:
        """Get command line completions"""
        completions = []
        
        common_commands = [
            "ls -la", "cd ", "mkdir ", "rm -rf ", "cp ", "mv ", "grep ",
            "find ", "chmod ", "chown ", "ps aux", "kill ", "top",
            "git add", "git commit", "git push", "git pull", "git status",
            "python ", "pip install", "npm install", "docker run"
        ]
        
        for command in common_commands:
            if command.startswith(current_input):
                completions.append(Completion(
                    text=command,
                    completion_type=CompletionType.COMMAND,
                    confidence=0.8,
                    source="shell_commands"
                ))
        
        return completions
    
    def _get_document_completions(self, current_input: str) -> List[Completion]:
        """Get document writing completions"""
        completions = []
        
        document_phrases = [
            "In conclusion,",
            "Furthermore,",
            "However,",
            "Therefore,",
            "On the other hand,",
            "As a result,",
            "In addition,",
            "For example,",
            "In summary,",
            "Nevertheless,"
        ]
        
        for phrase in document_phrases:
            if phrase.lower().startswith(current_input.lower()):
                completions.append(Completion(
                    text=phrase,
                    completion_type=CompletionType.PHRASE,
                    confidence=0.6,
                    source="document_writing"
                ))
        
        return completions
    
    def _get_common_completions(self, current_input: str) -> List[Completion]:
        """Get common word/phrase completions"""
        completions = []
        
        # Check common words
        for completion in self.common_completions.get("words", []):
            if completion.text.startswith(current_input.lower()):
                completions.append(completion)
        
        # Check common phrases
        for completion in self.common_completions.get("phrases", []):
            if completion.text.lower().startswith(current_input.lower()):
                completions.append(completion)
        
        return completions
    
    async def _get_pattern_completions(self, context: TextContext, current_input: str) -> List[Completion]:
        """Get completions based on learned user patterns"""
        completions = []
        
        if not self.settings.learning_enabled:
            return completions
        
        # Look for patterns in user's typing history
        user_id = context.metadata.get("user_id", "default")
        if user_id in self.user_patterns:
            patterns = self.user_patterns[user_id]
            
            # Find completions based on previous context
            context_key = f"{context.context_type.value}_{context.application or 'unknown'}"
            if context_key in patterns:
                pattern_data = patterns[context_key]
                
                # Look for completions that follow the current input
                for pattern, frequency in pattern_data.get("sequences", {}).items():
                    if pattern.startswith(current_input.lower()):
                        confidence = min(0.9, frequency / 10.0)  # Scale frequency to confidence
                        completions.append(Completion(
                            text=pattern,
                            completion_type=CompletionType.PHRASE,
                            confidence=confidence,
                            source="user_patterns",
                            usage_count=frequency
                        ))
        
        return completions
    
    def _get_application_completions(self, application: str, current_input: str) -> List[Completion]:
        """Get application-specific completions"""
        completions = []
        
        if application in self.application_completions:
            app_completions = self.application_completions[application]
            
            for category, comp_list in app_completions.items():
                for completion in comp_list:
                    if completion.text.lower().startswith(current_input.lower()):
                        completions.append(completion)
        
        return completions
    
    def _filter_and_rank_completions(self, completions: List[Completion], current_input: str) -> List[Completion]:
        """Filter and rank completions by relevance"""
        if not completions:
            return []
        
        # Filter by minimum confidence
        filtered = [c for c in completions if c.confidence >= self.settings.min_confidence]
        
        # Remove duplicates
        seen = set()
        unique_completions = []
        for completion in filtered:
            if completion.text not in seen:
                seen.add(completion.text)
                unique_completions.append(completion)
        
        # Calculate ranking scores
        for completion in unique_completions:
            score = completion.confidence
            
            # Boost score for exact prefix matches
            if completion.text.lower().startswith(current_input.lower()):
                score += 0.2
            
            # Boost score for frequently used completions
            if completion.usage_count > 0:
                score += min(0.3, completion.usage_count / 20.0)
            
            # Boost score for recently used completions
            if completion.last_used:
                days_since_use = (datetime.now() - completion.last_used).days
                if days_since_use < 7:
                    score += 0.1
            
            completion.confidence = min(1.0, score)
        
        # Sort by confidence (descending)
        unique_completions.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_completions
    
    async def learn_from_selection(self, context: TextContext, selected: Completion):
        """Learn from user's completion selection"""
        if not self.settings.learning_enabled:
            return
        
        try:
            user_id = context.metadata.get("user_id", "default")
            
            # Initialize user patterns if needed
            if user_id not in self.user_patterns:
                self.user_patterns[user_id] = {}
            
            context_key = f"{context.context_type.value}_{context.application or 'unknown'}"
            if context_key not in self.user_patterns[user_id]:
                self.user_patterns[user_id][context_key] = {
                    "sequences": {},
                    "preferences": {},
                    "last_updated": datetime.now()
                }
            
            # Update sequence patterns
            sequences = self.user_patterns[user_id][context_key]["sequences"]
            selected_text = selected.text.lower()
            
            if selected_text in sequences:
                sequences[selected_text] += 1
            else:
                sequences[selected_text] = 1
            
            # Update completion usage
            selected.usage_count += 1
            selected.last_used = datetime.now()
            
            # Update preferences
            preferences = self.user_patterns[user_id][context_key]["preferences"]
            completion_type = selected.completion_type.value
            
            if completion_type in preferences:
                preferences[completion_type] += 1
            else:
                preferences[completion_type] = 1
            
            self.user_patterns[user_id][context_key]["last_updated"] = datetime.now()
            
            logger.debug(f"Learned from selection: {selected.text} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error learning from selection: {e}")
    
    async def update_patterns(self, context: TextContext, suggestions: List[Completion]):
        """Update patterns based on context and suggestions"""
        if not self.settings.learning_enabled:
            return
        
        try:
            user_id = context.metadata.get("user_id", "default")
            
            # Track what suggestions were offered
            if user_id not in self.user_patterns:
                self.user_patterns[user_id] = {}
            
            context_key = f"{context.context_type.value}_{context.application or 'unknown'}"
            if context_key not in self.user_patterns[user_id]:
                self.user_patterns[user_id][context_key] = {
                    "sequences": {},
                    "preferences": {},
                    "suggestions_offered": {},
                    "last_updated": datetime.now()
                }
            
            # Track suggestions offered
            suggestions_offered = self.user_patterns[user_id][context_key]["suggestions_offered"]
            current_input = self._extract_current_input(context)
            
            if current_input:
                if current_input not in suggestions_offered:
                    suggestions_offered[current_input] = []
                
                suggestions_offered[current_input].extend([s.text for s in suggestions])
            
        except Exception as e:
            logger.error(f"Error updating patterns: {e}")
    
    def add_custom_completion(self, completion: Completion, application: str = None):
        """Add a custom completion"""
        if application:
            if application not in self.application_completions:
                self.application_completions[application] = {"custom": []}
            
            self.application_completions[application]["custom"].append(completion)
        else:
            if "custom" not in self.common_completions:
                self.common_completions["custom"] = []
            
            self.common_completions["custom"].append(completion)
        
        logger.info(f"Added custom completion: {completion.text}")
    
    def get_completion_stats(self, user_id: str = "default") -> Dict[str, Any]:
        """Get completion statistics for a user"""
        stats = {
            "total_patterns": 0,
            "contexts": [],
            "most_used_completions": [],
            "completion_types": {}
        }
        
        if user_id in self.user_patterns:
            user_data = self.user_patterns[user_id]
            stats["total_patterns"] = len(user_data)
            stats["contexts"] = list(user_data.keys())
            
            # Aggregate completion types
            for context_data in user_data.values():
                for comp_type, count in context_data.get("preferences", {}).items():
                    if comp_type in stats["completion_types"]:
                        stats["completion_types"][comp_type] += count
                    else:
                        stats["completion_types"][comp_type] = count
            
            # Find most used completions
            all_sequences = {}
            for context_data in user_data.values():
                for sequence, count in context_data.get("sequences", {}).items():
                    if sequence in all_sequences:
                        all_sequences[sequence] += count
                    else:
                        all_sequences[sequence] = count
            
            # Sort by usage count
            sorted_sequences = sorted(all_sequences.items(), key=lambda x: x[1], reverse=True)
            stats["most_used_completions"] = sorted_sequences[:10]
        
        return stats


class TextCompletion:
    """
    Main text completion system that coordinates completion engines and provides
    the interface for the personal assistant.
    """
    
    def __init__(self, settings: CompletionSettings = None):
        self.settings = settings or CompletionSettings()
        self.completion_engine = CompletionEngine(self.settings)
        self.suggestion_provider = SuggestionProvider()
        self.context_analyzer = ContextAnalyzer()
        self.pattern_learner = PatternLearner()
        
        # Active completion sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Text completion system initialized")
    
    async def get_completions(self, context: TextContext) -> List[Completion]:
        """Get text completions for the given context"""
        try:
            # Analyze context
            analyzed_context = await self.context_analyzer.analyze_context(context)
            
            # Get completions from engine
            completions = await self.completion_engine.get_completions(analyzed_context)
            
            # Enhance with suggestions
            enhanced_completions = await self.suggestion_provider.get_suggestions(analyzed_context, completions)
            
            # Update patterns
            await self.pattern_learner.update_patterns(analyzed_context, enhanced_completions)
            
            return enhanced_completions
            
        except Exception as e:
            logger.error(f"Error getting text completions: {e}")
            return []
    
    async def learn_from_selection(self, context: TextContext, selected: Completion):
        """Learn from user's completion selection"""
        try:
            await self.completion_engine.learn_from_selection(context, selected)
            await self.pattern_learner.learn_selection(context, selected)
            
        except Exception as e:
            logger.error(f"Error learning from selection: {e}")
    
    def update_settings(self, new_settings: Dict[str, Any]):
        """Update completion settings"""
        for key, value in new_settings.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        
        logger.info(f"Text completion settings updated: {new_settings}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get text completion system status"""
        return {
            "enabled": self.settings.enabled,
            "learning_enabled": self.settings.learning_enabled,
            "context_aware": self.settings.context_aware,
            "active_sessions": len(self.active_sessions),
            "cache_size": len(self.completion_engine.completion_cache)
        }


class SuggestionProvider:
    """Provides enhanced suggestions based on context"""
    
    async def get_suggestions(self, context: TextContext, base_completions: List[Completion]) -> List[Completion]:
        """Get enhanced suggestions"""
        # For now, just return base completions
        # Could be extended with ML-based suggestions
        return base_completions


class ContextAnalyzer:
    """Analyzes text context for better completions"""
    
    async def analyze_context(self, context: TextContext) -> TextContext:
        """Analyze and enhance context"""
        # For now, just return the original context
        # Could be extended with NLP analysis
        return context


class PatternLearner:
    """Learns from user patterns for better completions"""
    
    async def update_patterns(self, context: TextContext, suggestions: List[Completion]):
        """Update learning patterns"""
        # Placeholder for pattern learning
        pass
    
    async def learn_selection(self, context: TextContext, selected: Completion):
        """Learn from user selection"""
        # Placeholder for selection learning
        pass