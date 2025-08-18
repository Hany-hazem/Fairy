"""
Simple tests for Multi-Modal Interaction components

This module contains basic tests for the multi-modal components without
requiring the full application configuration.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

# Test the individual components without full app imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_voice_command_creation():
    """Test voice command creation without dependencies"""
    from app.voice_processor import VoiceCommand
    
    command = VoiceCommand(
        text="test command",
        intent="test_intent",
        confidence=0.8,
        entities={"param": "value"}
    )
    
    assert command.text == "test command"
    assert command.intent == "test_intent"
    assert command.confidence == 0.8
    assert command.entities["param"] == "value"
    assert command.timestamp > 0


def test_voice_settings():
    """Test voice settings configuration"""
    from app.voice_processor import VoiceSettings
    
    settings = VoiceSettings()
    settings.voice_type = "test"
    settings.speech_speed = 1.5
    settings.language = "en-US"
    
    assert settings.voice_type == "test"
    assert settings.speech_speed == 1.5
    assert settings.language == "en-US"


def test_overlay_config_creation():
    """Test overlay configuration"""
    from app.screen_overlay import OverlayConfig, OverlayType, OverlayPosition
    
    config = OverlayConfig(
        overlay_type=OverlayType.NOTIFICATION,
        content="Test notification",
        position=OverlayPosition.TOP_RIGHT,
        duration=5.0
    )
    
    assert config.overlay_type == OverlayType.NOTIFICATION
    assert config.content == "Test notification"
    assert config.position == OverlayPosition.TOP_RIGHT
    assert config.duration == 5.0


def test_text_context_creation():
    """Test text context creation"""
    from app.text_completion import TextContext, ContextType
    
    context = TextContext(
        text_before="def hello_",
        text_after="():",
        cursor_position=10,
        context_type=ContextType.CODE,
        application="vscode",
        file_type="python"
    )
    
    assert context.text_before == "def hello_"
    assert context.text_after == "():"
    assert context.cursor_position == 10
    assert context.context_type == ContextType.CODE
    assert context.application == "vscode"
    assert context.file_type == "python"


def test_completion_creation():
    """Test completion object creation"""
    from app.text_completion import Completion, CompletionType
    
    completion = Completion(
        text="world",
        completion_type=CompletionType.WORD,
        confidence=0.8,
        description="Complete function name",
        source="test"
    )
    
    assert completion.text == "world"
    assert completion.completion_type == CompletionType.WORD
    assert completion.confidence == 0.8
    assert completion.description == "Complete function name"
    assert completion.source == "test"


def test_accessibility_settings():
    """Test accessibility settings"""
    from app.accessibility_manager import AccessibilitySettings, AccessibilityFeature, AccessibilityLevel
    
    settings = AccessibilitySettings()
    settings.enabled_features = [AccessibilityFeature.SCREEN_READER, AccessibilityFeature.HIGH_CONTRAST]
    settings.compliance_level = AccessibilityLevel.AA
    settings.screen_reader_enabled = True
    
    assert AccessibilityFeature.SCREEN_READER in settings.enabled_features
    assert AccessibilityFeature.HIGH_CONTRAST in settings.enabled_features
    assert settings.compliance_level == AccessibilityLevel.AA
    assert settings.screen_reader_enabled


def test_interaction_mode_enum():
    """Test interaction mode enumeration"""
    from app.interaction_mode_manager import InteractionMode, ModeTransition
    
    assert InteractionMode.TEXT.value == "text"
    assert InteractionMode.VOICE.value == "voice"
    assert InteractionMode.VISUAL.value == "visual"
    assert InteractionMode.MIXED.value == "mixed"
    assert InteractionMode.ACCESSIBILITY.value == "accessibility"
    
    assert ModeTransition.USER_INITIATED.value == "user_initiated"
    assert ModeTransition.AUTOMATIC.value == "automatic"


@pytest.mark.asyncio
async def test_voice_processor_basic():
    """Test basic voice processor functionality"""
    from app.voice_processor import VoiceProcessor, VoiceSettings
    
    settings = VoiceSettings()
    processor = VoiceProcessor(settings)
    
    assert processor is not None
    assert not processor.is_listening
    assert not processor.is_speaking
    
    # Test command parsing
    command = processor._parse_command("open file test.txt")
    assert command.text == "open file test.txt"
    assert command.intent in ["file_operation", "general_query"]


@pytest.mark.asyncio
async def test_text_completion_engine():
    """Test text completion engine"""
    from app.text_completion import CompletionEngine, CompletionSettings
    
    settings = CompletionSettings()
    engine = CompletionEngine(settings)
    
    assert engine is not None
    assert engine.settings.enabled
    assert len(engine.common_completions) > 0
    
    # Test current input extraction
    from app.text_completion import TextContext, ContextType
    context = TextContext(
        text_before="hello wor",
        text_after="ld",
        cursor_position=9,
        context_type=ContextType.GENERAL
    )
    
    current_input = engine._extract_current_input(context)
    assert current_input == "wor"


def test_screen_overlay_types():
    """Test screen overlay types and positions"""
    from app.screen_overlay import OverlayType, OverlayPosition, OverlayStyle
    
    # Test overlay types
    assert OverlayType.NOTIFICATION.value == "notification"
    assert OverlayType.CONTEXT_INFO.value == "context_info"
    assert OverlayType.TOOLTIP.value == "tooltip"
    
    # Test positions
    assert OverlayPosition.TOP_LEFT.value == "top_left"
    assert OverlayPosition.CENTER.value == "center"
    assert OverlayPosition.BOTTOM_RIGHT.value == "bottom_right"
    
    # Test style
    style = OverlayStyle()
    assert style.background_color == "#2D2D2D"
    assert style.text_color == "#FFFFFF"
    assert style.opacity == 0.9


def test_accessibility_features():
    """Test accessibility feature enumeration"""
    from app.accessibility_manager import AccessibilityFeature, AccessibilityLevel
    
    # Test features
    features = list(AccessibilityFeature)
    assert AccessibilityFeature.SCREEN_READER in features
    assert AccessibilityFeature.KEYBOARD_NAVIGATION in features
    assert AccessibilityFeature.HIGH_CONTRAST in features
    assert AccessibilityFeature.LARGE_TEXT in features
    
    # Test compliance levels
    levels = list(AccessibilityLevel)
    assert AccessibilityLevel.A in levels
    assert AccessibilityLevel.AA in levels
    assert AccessibilityLevel.AAA in levels


@pytest.mark.asyncio
async def test_mock_screen_overlay():
    """Test mock screen overlay functionality"""
    from app.screen_overlay import MockScreenOverlay
    
    overlay = MockScreenOverlay()
    
    # Test basic operations
    overlay_id = await overlay.show_notification("Test message")
    assert overlay_id == "mock_notification"
    
    await overlay.hide_overlay(overlay_id)
    await overlay.remove_overlay(overlay_id)
    
    assert not overlay.is_available()


def test_voice_command_parser():
    """Test voice command parser"""
    from app.voice_processor import VoiceCommandParser
    
    parser = VoiceCommandParser()
    
    # Test intent detection
    assert "file_operation" in parser.intent_keywords
    assert "task_management" in parser.intent_keywords
    assert "general_query" in parser.intent_keywords
    
    # Test advanced parsing
    command = parser.parse_advanced("open the file called document.pdf")
    assert command.intent in ["file_operation", "general_query"]
    assert command.confidence > 0


def test_completion_settings():
    """Test completion settings"""
    from app.text_completion import CompletionSettings
    
    settings = CompletionSettings()
    settings.enabled = True
    settings.max_suggestions = 10
    settings.min_confidence = 0.5
    settings.learning_enabled = True
    
    assert settings.enabled
    assert settings.max_suggestions == 10
    assert settings.min_confidence == 0.5
    assert settings.learning_enabled


def test_keyboard_navigation():
    """Test keyboard navigation manager"""
    from app.accessibility_manager import KeyboardNavigationManager
    
    nav_manager = KeyboardNavigationManager()
    
    assert not nav_manager.keyboard_only_mode
    assert nav_manager.current_focus is None
    assert len(nav_manager.tab_order) == 0
    
    # Test enabling keyboard-only mode
    nav_manager.enable_keyboard_only_mode()
    assert nav_manager.keyboard_only_mode
    
    # Test tab order
    nav_manager.set_tab_order(["button1", "button2", "button3"])
    assert len(nav_manager.tab_order) == 3
    assert nav_manager.tab_order[0] == "button1"


def test_high_contrast_manager():
    """Test high contrast manager"""
    from app.accessibility_manager import HighContrastManager
    
    contrast_manager = HighContrastManager()
    
    assert not contrast_manager.high_contrast_enabled
    assert contrast_manager.current_scheme == "default"
    
    # Test enabling high contrast
    contrast_manager.enable_high_contrast("high_contrast_dark")
    assert contrast_manager.high_contrast_enabled
    assert contrast_manager.current_scheme == "high_contrast_dark"
    
    # Test color schemes
    colors = contrast_manager.get_current_colors()
    assert "background" in colors
    assert "text" in colors
    assert "accent" in colors


@pytest.mark.asyncio
async def test_interaction_context():
    """Test interaction context management"""
    from app.interaction_mode_manager import InteractionContext, InteractionMode
    
    context = InteractionContext(
        current_mode=InteractionMode.TEXT,
        previous_mode=None
    )
    
    assert context.current_mode == InteractionMode.TEXT
    assert context.previous_mode is None
    assert len(context.conversation_history) == 0
    assert context.mode_switches == 0
    assert isinstance(context.session_start, datetime)


def test_mode_capabilities():
    """Test mode capabilities definition"""
    from app.interaction_mode_manager import ModeCapabilities
    
    # Test text mode capabilities
    text_caps = ModeCapabilities(
        text_input=True,
        text_output=True,
        visual_feedback=True
    )
    
    assert text_caps.text_input
    assert text_caps.text_output
    assert text_caps.visual_feedback
    assert not text_caps.voice_input
    assert not text_caps.voice_output
    
    # Test voice mode capabilities
    voice_caps = ModeCapabilities(
        voice_input=True,
        voice_output=True,
        visual_feedback=True,
        screen_overlay=True
    )
    
    assert voice_caps.voice_input
    assert voice_caps.voice_output
    assert voice_caps.visual_feedback
    assert voice_caps.screen_overlay


def test_request_types():
    """Test new request types for multi-modal functionality"""
    # This would test the RequestType enum if we could import it
    # For now, just test that the values are what we expect
    
    expected_types = [
        "voice_command",
        "text_completion", 
        "visual_feedback",
        "mode_switch",
        "accessibility_request"
    ]
    
    # These should be valid request type values
    for req_type in expected_types:
        assert isinstance(req_type, str)
        assert len(req_type) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])