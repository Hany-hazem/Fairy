# app/safety_filter.py
"""
Safety filtering for AI responses to ensure content moderation
"""

import logging
import re
from typing import List, Dict, Any
from .config import settings

logger = logging.getLogger(__name__)

class SafetyFilter:
    """
    Content safety filter for AI responses
    """
    
    def __init__(self):
        # Basic harmful content patterns
        self.harmful_patterns = [
            r'\b(?:kill|murder|suicide)\b.*\b(?:yourself|myself|someone|people)\b',
            r'\b(?:harm|hurt)\b.*\b(?:yourself|myself|someone|people)\b',
            r'\b(?:how to|instructions for|guide to)\b.*\b(?:mak|creat|build).*\b(?:bomb|weapon|explosive|poison)\b',
            r'\b(?:illegal|criminal|unlawful)\b.*\b(?:activit|action)',
            r'\b(?:hate|discrimination|racism|sexism)\b.*\b(?:against|towards)\b',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.harmful_patterns]
        
        # Content categories to flag
        self.flagged_categories = [
            "violence", "self-harm", "illegal", "hate", "harassment",
            "adult", "privacy", "misinformation"
        ]
        
        logger.info("Safety filter initialized")
    
    def is_safe(self, text: str) -> bool:
        """
        Check if text content is safe
        
        Args:
            text: Text content to check
            
        Returns:
            True if content is safe, False otherwise
        """
        if not text or not isinstance(text, str):
            return True
        
        try:
            # Basic pattern matching
            if self._contains_harmful_patterns(text):
                logger.warning("Content flagged by pattern matching")
                return False
            
            # Length check (extremely long responses might be problematic)
            if len(text) > 10000:
                logger.warning("Content flagged for excessive length")
                return False
            
            # Check for potential prompt injection
            if self._contains_prompt_injection(text):
                logger.warning("Content flagged for potential prompt injection")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in safety filter: {e}")
            # Default to safe if filter fails
            return True
    
    def _contains_harmful_patterns(self, text: str) -> bool:
        """Check if text contains harmful patterns"""
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _contains_prompt_injection(self, text: str) -> bool:
        """Check for potential prompt injection attempts"""
        injection_indicators = [
            "ignore previous instructions",
            "forget everything above",
            "system prompt",
            "you are now",
            "new instructions:",
            "override your",
            "disregard the above",
            "act as if",
            "pretend to be",
            "roleplay as"
        ]
        
        text_lower = text.lower()
        for indicator in injection_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    def get_safety_score(self, text: str) -> Dict[str, Any]:
        """
        Get detailed safety analysis of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with safety score and details
        """
        try:
            score = 1.0  # Start with safe score
            flags = []
            
            if self._contains_harmful_patterns(text):
                score -= 0.5
                flags.append("harmful_patterns")
            
            if self._contains_prompt_injection(text):
                score -= 0.3
                flags.append("prompt_injection")
            
            if len(text) > 10000:
                score -= 0.2
                flags.append("excessive_length")
            
            # Ensure score doesn't go below 0
            score = max(0.0, score)
            
            return {
                "safe": score >= settings.SAFETY_THRESHOLD,
                "score": score,
                "threshold": settings.SAFETY_THRESHOLD,
                "flags": flags,
                "text_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"Error getting safety score: {e}")
            return {
                "safe": True,
                "score": 1.0,
                "threshold": settings.SAFETY_THRESHOLD,
                "flags": ["error"],
                "error": str(e)
            }
    
    def filter_content(self, text: str) -> str:
        """
        Filter and clean content if possible
        
        Args:
            text: Text to filter
            
        Returns:
            Filtered text or warning message
        """
        if self.is_safe(text):
            return text
        
        # For now, return a generic safe response
        return "I apologize, but I cannot provide that response due to content safety guidelines."

# Global safety filter instance
safety_filter = SafetyFilter()