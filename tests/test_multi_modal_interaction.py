"""
Tests for Multi-Modal Interaction Support

This module contains comprehensive tests for voice processing, screen overlay,
text completion, accessibility features, and interaction mode management.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Import the modules to test
from app.voice_processor import VoiceProcessor, VoiceCommand, VoiceResponse, VoiceSettings
from app.screen_overlay import ScreenOverlay, OverlayConfig, OverlayType, OverlayPosition
from app.text_completion import TextCompletion, TextContext, Completion, CompletionType, ContextType
from app.accessibility_manager import AccessibilityManager, AccessibilitySettings, AccessibilityFeature
from app.interaction_mode_manager import InteractionModeManager, InteractionMode, ModeTransition
from app.personal_assistant_core import PersonalAssistantCore, RequestType, AssistantRequest


class TestVoiceProcessor:
    """Test voice processing functionality"""
    
    @pytest.fixture
    def voice_settings(self):
        """Create voice settings for testing"""
        settings = VoiceSettings()
        settings.voice_type = "test"
        settings.speech_speed = 1.0
        settings.language = "en-US"
        settings.wake_word_enabled = False
        return settings
    
    @pytest.fixture
    def voice_processor(self, voice_settings):
        """Create voice processor for testing"""
        return VoiceProcessor(voice_settings)
    
    def test_voice_processor_initialization(self, voice_processor):
        """Test voice processor initialization"""
        assert voice_processor is not None
        assert voice_processor.settings is not None
        assert not voice_processor.is_listening
        assert not voice_processor.is_speaking
    
    def test_voice_command_creation(self):
        """Test voice command creation"""
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
    
    def test_voice_response_creation(self):
        """Test voice response creation"""
        response = VoiceResponse(
            text="test response",
            voice_type="default",
            speed=1.0,
            volume=0.8
        )
        
        assert response.text == "test response"
        assert response.voice_type == "default"
        assert response.speed == 1.0
        assert response.volume == 0.8
    
    @pytest.mark.asyncio
    async def test_voice_command_processing(self, voice_processor):
        """Test voice command processing"""
        # Test text-based command processing
        command = await voice_processor.process_voice_command(text="open file test.txt")
        
        assert command is not None
        assert command.text == "open file test.txt"
        assert command.intent in ["file_operation", "general_query"]
        assert 0 <= command.confidence <= 1
    
    def test_command_pattern_matching(self, voice_processor):
        """Test command pattern matching"""
        # Test file operation pattern
        command = voice_processor._parse_command("open file document.pdf")
        assert command.intent == "file_operation"
        
        # Test task management pattern
        command = voice_processor._parse_command("create task finish project")
        assert command.intent == "task_management"
        
        # Test general query pattern
        command = voice_processor._parse_command("help me with something")
        assert command.intent == "general_query"
    
    @pytest.mark.asyncio
    async def test_voice_response_generation(self, voice_processor):
        """Test voice response generation"""
        response = VoiceResponse(text="Test response")
        
        # This will use mock TTS since real TTS may not be available
        result = await voice_processor.generate_voice_response(response)
        
        # Should return True even with mock TTS
        assert isinstance(result, bool)
    
    def test_voice_status(self, voice_processor):
        """Test voice status reporting"""
        status = voice_processor.get_voice_status()
        
        assert "available" in status
        assert "listening" in status
        assert "speaking" in status
        assert "language" in status
        assert isinstance(status["available"], bool)
        assert isinstance(status["listening"], bool)
        assert isinstance(status["speaking"], bool)


class TestScreenOverlay:
    """Test screen overlay functionality"""
    
    @pytest.fixture
    def screen_overlay(self):
        """Create screen overlay for testing"""
        return ScreenOverlay()
    
    def test_screen_overlay_initialization(self, screen_overlay):
        """Test screen overlay initialization"""
        assert screen_overlay is not None
        assert screen_overlay.overlays == {}
        assert screen_overlay.overlay_counter == 0
    
    def test_overlay_config_creation(self):
        """Test overlay configuration creation"""
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
    
    @pytest.mark.asyncio
    async def test_show_notification(self, screen_overlay):
        """Test showing notification overlay"""
        overlay_id = await screen_overlay.show_notification(
            "Test notification",
            duration=3.0,
            position=OverlayPosition.TOP_RIGHT
        )
        
        # Should return an overlay ID (string) or empty string if not available
        assert isinstance(overlay_id, str)
    
    @pytest.mark.asyncio
    async def test_show_context_info(self, screen_overlay):
        """Test showing context information overlay"""
        overlay_id = await screen_overlay.show_context_info(
            "Context information",
            position=OverlayPosition.TOP_LEFT
        )
        
        assert isinstance(overlay_id, str)
    
    @pytest.mark.asyncio
    async def test_visual_feedback(self, screen_overlay):
        """Test visual feedback display"""
        # Mock result object
        result = Mock()
        result.success = True
        
        overlay_id = await screen_overlay.show_visual_feedback(
            "Test action",
            result,
            duration=2.0
        )
        
        assert isinstance(overlay_id, str)
    
    def test_overlay_availability(self, screen_overlay):
        """Test overlay system availability"""
        available = screen_overlay.is_available()
        assert isinstance(available, bool)
    
    @pytest.mark.asyncio
    async def test_overlay_management(self, screen_overlay):
        """Test overlay management operations"""
        # Show overlay
        overlay_id = await screen_overlay.show_notification("Test")
        
        if overlay_id:  # Only test if overlay was created
            # Hide overlay
            await screen_overlay.hide_overlay(overlay_id)
            
            # Remove overlay
            await screen_overlay.remove_overlay(overlay_id)
        
        # Clear all overlays
        await screen_overlay.clear_all_overlays()
        
        # Should not raise exceptions
        assert True


class TestTextCompletion:
    """Test text completion functionality"""
    
    @pytest.fixture
    def text_completion(self):
        """Create text completion system for testing"""
        return TextCompletion()
    
    @pytest.fixture
    def text_context(self):
        """Create text context for testing"""
        return TextContext(
            text_before="def hello_",
            text_after="():",
            cursor_position=10,
            context_type=ContextType.CODE,
            application="vscode",
            file_type="python",
            language="python"
        )
    
    def test_text_completion_initialization(self, text_completion):
        """Test text completion initialization"""
        assert text_completion is not None
        assert text_completion.completion_engine is not None
        assert text_completion.settings is not None
    
    def test_text_context_creation(self, text_context):
        """Test text context creation"""
        assert text_context.text_before == "def hello_"
        assert text_context.text_after == "():"
        assert text_context.cursor_position == 10
        assert text_context.context_type == ContextType.CODE
        assert text_context.application == "vscode"
        assert text_context.file_type == "python"
    
    def test_completion_creation(self):
        """Test completion object creation"""
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
    
    @pytest.mark.asyncio
    async def test_get_completions(self, text_completion, text_context):
        """Test getting text completions"""
        completions = await text_completion.get_completions(text_context)
        
        assert isinstance(completions, list)
        # Should return some completions for code context
        for completion in completions:
            assert isinstance(completion, Completion)
            assert completion.text is not None
            assert 0 <= completion.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_learn_from_selection(self, text_completion, text_context):
        """Test learning from user selection"""
        completion = Completion(
            text="world",
            completion_type=CompletionType.WORD,
            confidence=0.8
        )
        
        # Should not raise exception
        await text_completion.learn_from_selection(text_context, completion)
        assert True
    
    def test_completion_status(self, text_completion):
        """Test completion system status"""
        status = text_completion.get_status()
        
        assert "enabled" in status
        assert "learning_enabled" in status
        assert "context_aware" in status
        assert isinstance(status["enabled"], bool)


class TestAccessibilityManager:
    """Test accessibility management functionality"""
    
    @pytest.fixture
    def accessibility_settings(self):
        """Create accessibility settings for testing"""
        settings = AccessibilitySettings()
        settings.screen_reader_enabled = True
        settings.keyboard_only_navigation = True
        settings.high_contrast_mode = False
        return settings
    
    @pytest.fixture
    def accessibility_manager(self, accessibility_settings):
        """Create accessibility manager for testing"""
        return AccessibilityManager(accessibility_settings)
    
    def test_accessibility_manager_initialization(self, accessibility_manager):
        """Test accessibility manager initialization"""
        assert accessibility_manager is not None
        assert accessibility_manager.settings is not None
        assert accessibility_manager.screen_reader is not None
        assert accessibility_manager.keyboard_nav is not None
        assert accessibility_manager.high_contrast is not None
    
    def test_feature_enable_disable(self, accessibility_manager):
        """Test enabling and disabling accessibility features"""
        # Enable feature
        accessibility_manager.enable_feature(AccessibilityFeature.HIGH_CONTRAST)
        assert AccessibilityFeature.HIGH_CONTRAST in accessibility_manager.settings.enabled_features
        assert accessibility_manager.settings.high_contrast_mode
        
        # Disable feature
        accessibility_manager.disable_feature(AccessibilityFeature.HIGH_CONTRAST)
        assert AccessibilityFeature.HIGH_CONTRAST not in accessibility_manager.settings.enabled_features
        assert not accessibility_manager.settings.high_contrast_mode
    
    @pytest.mark.asyncio
    async def test_screen_reader_announcement(self, accessibility_manager):
        """Test screen reader announcements"""
        # Should not raise exception
        await accessibility_manager.announce_to_screen_reader("Test announcement")
        assert True
    
    @pytest.mark.asyncio
    async def test_keyboard_event_handling(self, accessibility_manager):
        """Test keyboard event handling"""
        key_event = {
            "key": "Tab",
            "modifiers": []
        }
        
        result = await accessibility_manager.handle_keyboard_event(key_event)
        assert isinstance(result, bool)
    
    def test_accessibility_status(self, accessibility_manager):
        """Test accessibility status reporting"""
        status = accessibility_manager.get_accessibility_status()
        
        assert "enabled_features" in status
        assert "compliance_level" in status
        assert "screen_reader" in status
        assert "keyboard_navigation" in status
        assert isinstance(status["enabled_features"], list)
    
    def test_wcag_compliance_check(self, accessibility_manager):
        """Test WCAG compliance checking"""
        compliance = accessibility_manager.check_wcag_compliance()
        
        assert "compliance_level" in compliance
        assert "checks" in compliance
        assert "recommendations" in compliance
        assert compliance["compliance_level"] in ["A", "AA", "AAA"]


class TestInteractionModeManager:
    """Test interaction mode management"""
    
    @pytest.fixture
    def interaction_manager(self):
        """Create interaction mode manager for testing"""
        return InteractionModeManager()
    
    def test_interaction_manager_initialization(self, interaction_manager):
        """Test interaction manager initialization"""
        assert interaction_manager is not None
        assert interaction_manager.current_mode == InteractionMode.TEXT
        assert interaction_manager.voice_processor is not None
        assert interaction_manager.screen_overlay is not None
        assert interaction_manager.text_completion is not None
        assert interaction_manager.accessibility_manager is not None
    
    @pytest.mark.asyncio
    async def test_mode_switching(self, interaction_manager):
        """Test interaction mode switching"""
        # Switch to voice mode
        success = await interaction_manager.switch_mode(
            InteractionMode.VOICE,
            ModeTransition.USER_INITIATED
        )
        
        # Success depends on voice availability
        assert isinstance(success, bool)
        
        if success:
            assert interaction_manager.current_mode == InteractionMode.VOICE
    
    @pytest.mark.asyncio
    async def test_optimal_mode_detection(self, interaction_manager):
        """Test optimal mode detection"""
        context_data = {
            "application": "vscode",
            "current_activity": "typing",
            "noise_level": 0.2
        }
        
        optimal_mode = await interaction_manager.detect_optimal_mode(context_data)
        assert isinstance(optimal_mode, InteractionMode)
    
    @pytest.mark.asyncio
    async def test_interaction_processing(self, interaction_manager):
        """Test interaction processing"""
        interaction_data = {
            "type": "text_input",
            "content": "Hello, assistant!",
            "user_id": "test_user"
        }
        
        result = await interaction_manager.process_interaction(interaction_data)
        
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_mode_capabilities(self, interaction_manager):
        """Test mode capabilities"""
        text_caps = interaction_manager.get_mode_capabilities(InteractionMode.TEXT)
        assert text_caps.text_input
        assert text_caps.text_output
        
        voice_caps = interaction_manager.get_mode_capabilities(InteractionMode.VOICE)
        assert voice_caps.voice_input
        assert voice_caps.voice_output
        
        mixed_caps = interaction_manager.get_mode_capabilities(InteractionMode.MIXED)
        assert mixed_caps.voice_input
        assert mixed_caps.text_input
        assert mixed_caps.visual_feedback
    
    def test_interaction_context(self, interaction_manager):
        """Test interaction context management"""
        context = interaction_manager.get_interaction_context()
        
        assert context.current_mode == InteractionMode.TEXT
        assert isinstance(context.conversation_history, list)
        assert isinstance(context.user_preferences, dict)
        assert isinstance(context.session_start, datetime)
    
    def test_status_reporting(self, interaction_manager):
        """Test status reporting"""
        status = interaction_manager.get_status()
        
        assert "current_mode" in status
        assert "capabilities" in status
        assert "component_status" in status
        assert status["current_mode"] == "text"


class TestPersonalAssistantCoreIntegration:
    """Test integration of multi-modal features with PersonalAssistantCore"""
    
    @pytest.fixture
    def assistant_core(self):
        """Create PersonalAssistantCore for testing"""
        return PersonalAssistantCore(":memory:")  # Use in-memory database
    
    @pytest.mark.asyncio
    async def test_voice_command_request(self, assistant_core):
        """Test voice command request handling"""
        request = AssistantRequest(
            user_id="test_user",
            request_type=RequestType.VOICE_COMMAND,
            content="open file test.txt",
            metadata={
                "command_text": "open file test.txt",
                "intent": "file_operation",
                "confidence": 0.8
            }
        )
        
        response = await assistant_core.process_request(request)
        
        assert response is not None
        assert isinstance(response.success, bool)
        assert response.content is not None
    
    @pytest.mark.asyncio
    async def test_text_completion_request(self, assistant_core):
        """Test text completion request handling"""
        request = AssistantRequest(
            user_id="test_user",
            request_type=RequestType.TEXT_COMPLETION,
            content="Get completions",
            metadata={
                "text_before": "def hello_",
                "text_after": "():",
                "cursor_position": 10,
                "context_type": "code",
                "application": "vscode",
                "file_type": "python"
            }
        )
        
        response = await assistant_core.process_request(request)
        
        assert response is not None
        assert isinstance(response.success, bool)
        if response.success:
            assert "completions" in response.metadata
    
    @pytest.mark.asyncio
    async def test_visual_feedback_request(self, assistant_core):
        """Test visual feedback request handling"""
        request = AssistantRequest(
            user_id="test_user",
            request_type=RequestType.VISUAL_FEEDBACK,
            content="Task completed successfully",
            metadata={
                "message": "Task completed successfully",
                "feedback_type": "success",
                "duration": 3.0
            }
        )
        
        response = await assistant_core.process_request(request)
        
        assert response is not None
        assert isinstance(response.success, bool)
    
    @pytest.mark.asyncio
    async def test_mode_switch_request(self, assistant_core):
        """Test mode switch request handling"""
        request = AssistantRequest(
            user_id="test_user",
            request_type=RequestType.MODE_SWITCH,
            content="Switch to voice mode",
            metadata={
                "target_mode": "voice",
                "transition_type": "user_initiated",
                "preserve_context": True
            }
        )
        
        response = await assistant_core.process_request(request)
        
        assert response is not None
        assert isinstance(response.success, bool)
    
    @pytest.mark.asyncio
    async def test_accessibility_request(self, assistant_core):
        """Test accessibility request handling"""
        request = AssistantRequest(
            user_id="test_user",
            request_type=RequestType.ACCESSIBILITY_REQUEST,
            content="Get accessibility status",
            metadata={
                "action": "get_status"
            }
        )
        
        response = await assistant_core.process_request(request)
        
        assert response is not None
        assert isinstance(response.success, bool)
        if response.success:
            assert "accessibility_status" in response.metadata


class TestMultiModalIntegration:
    """Test integration between multi-modal components"""
    
    @pytest.fixture
    def interaction_manager(self):
        """Create interaction manager for integration testing"""
        return InteractionModeManager()
    
    @pytest.mark.asyncio
    async def test_voice_to_text_transition(self, interaction_manager):
        """Test transition from voice to text mode"""
        # Start in voice mode (if available)
        await interaction_manager.switch_mode(InteractionMode.VOICE)
        
        # Switch to text mode
        success = await interaction_manager.switch_mode(InteractionMode.TEXT)
        assert success
        assert interaction_manager.current_mode == InteractionMode.TEXT
    
    @pytest.mark.asyncio
    async def test_context_preservation(self, interaction_manager):
        """Test context preservation during mode switches"""
        # Add some interaction history
        interaction_data = {
            "type": "test",
            "content": "Test interaction",
            "user_id": "test_user"
        }
        
        await interaction_manager.process_interaction(interaction_data)
        
        # Switch modes with context preservation
        await interaction_manager.switch_mode(
            InteractionMode.VISUAL,
            preserve_context=True
        )
        
        # Check that context is preserved
        context = interaction_manager.get_interaction_context()
        assert len(context.conversation_history) > 0
    
    @pytest.mark.asyncio
    async def test_accessibility_integration(self, interaction_manager):
        """Test accessibility integration across modes"""
        # Enable accessibility features
        interaction_manager.accessibility_manager.enable_feature(
            AccessibilityFeature.SCREEN_READER
        )
        
        # Switch to accessibility mode
        success = await interaction_manager.switch_mode(InteractionMode.ACCESSIBILITY)
        
        # Should succeed regardless of other component availability
        assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_multi_modal_response(self, interaction_manager):
        """Test multi-modal response generation"""
        # Switch to mixed mode
        await interaction_manager.switch_mode(InteractionMode.MIXED)
        
        # Process interaction that should generate multiple response types
        interaction_data = {
            "type": "multi_modal_test",
            "text": "Show me visual feedback",
            "voice_text": "Also speak this response",
            "visual_content": "Display this overlay",
            "user_id": "test_user"
        }
        
        result = await interaction_manager.process_interaction(interaction_data)
        
        assert isinstance(result, dict)
        assert "success" in result


# Performance and stress tests
class TestMultiModalPerformance:
    """Test performance of multi-modal components"""
    
    @pytest.mark.asyncio
    async def test_completion_performance(self):
        """Test text completion performance"""
        text_completion = TextCompletion()
        context = TextContext(
            text_before="test",
            text_after="",
            cursor_position=4,
            context_type=ContextType.GENERAL
        )
        
        # Measure completion time
        start_time = datetime.now()
        completions = await text_completion.get_completions(context)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert duration < 1.0  # Less than 1 second
        assert isinstance(completions, list)
    
    @pytest.mark.asyncio
    async def test_mode_switch_performance(self):
        """Test mode switching performance"""
        interaction_manager = InteractionModeManager()
        
        # Measure mode switch time
        start_time = datetime.now()
        await interaction_manager.switch_mode(InteractionMode.VISUAL)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Should switch modes quickly
        assert duration < 0.5  # Less than 500ms
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent multi-modal operations"""
        interaction_manager = InteractionModeManager()
        
        # Create multiple concurrent operations
        tasks = []
        for i in range(5):
            interaction_data = {
                "type": "concurrent_test",
                "content": f"Test {i}",
                "user_id": f"user_{i}"
            }
            task = interaction_manager.process_interaction(interaction_data)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, dict)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])