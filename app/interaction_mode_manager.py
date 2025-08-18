"""
Interaction Mode Manager

This module manages different interaction modes (voice, visual, text) and
coordinates seamless switching between them while preserving context.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .voice_processor import VoiceProcessor, VoiceCommand, VoiceResponse, VoiceSettings
from .screen_overlay import ScreenOverlay, OverlayConfig, OverlayType, OverlayPosition
from .text_completion import TextCompletion, TextContext, Completion, CompletionSettings
from .accessibility_manager import AccessibilityManager, AccessibilitySettings, AccessibilityEvent

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Available interaction modes"""
    TEXT = "text"
    VOICE = "voice"
    VISUAL = "visual"
    MIXED = "mixed"
    ACCESSIBILITY = "accessibility"


class ModeTransition(Enum):
    """Types of mode transitions"""
    USER_INITIATED = "user_initiated"
    AUTOMATIC = "automatic"
    CONTEXT_BASED = "context_based"
    ACCESSIBILITY_REQUIRED = "accessibility_required"


@dataclass
class InteractionContext:
    """Context information for interactions"""
    current_mode: InteractionMode
    previous_mode: Optional[InteractionMode]
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    active_tasks: List[str] = field(default_factory=list)
    application_context: Optional[str] = None
    accessibility_needs: List[str] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.now)
    last_interaction: Optional[datetime] = None
    mode_switches: int = 0


@dataclass
class ModeCapabilities:
    """Capabilities available in each mode"""
    voice_input: bool = False
    voice_output: bool = False
    visual_feedback: bool = False
    text_input: bool = False
    text_output: bool = False
    screen_overlay: bool = False
    keyboard_navigation: bool = False
    screen_reader_support: bool = False


class InteractionModeManager:
    """
    Manages different interaction modes and coordinates seamless switching
    between voice, visual, and text interfaces while preserving context.
    """
    
    def __init__(self, 
                 voice_settings: VoiceSettings = None,
                 completion_settings: CompletionSettings = None,
                 accessibility_settings: AccessibilitySettings = None):
        
        # Initialize components
        self.voice_processor = VoiceProcessor(voice_settings)
        self.screen_overlay = ScreenOverlay()
        self.text_completion = TextCompletion(completion_settings)
        self.accessibility_manager = AccessibilityManager(accessibility_settings)
        
        # Mode management
        self.current_mode = InteractionMode.TEXT
        self.context = InteractionContext(current_mode=self.current_mode)
        self.mode_capabilities = self._initialize_mode_capabilities()
        
        # Event handlers
        self.mode_change_handlers: List[Callable] = []
        self.interaction_handlers: Dict[InteractionMode, Callable] = {}
        
        # Auto-detection settings
        self.auto_mode_detection = True
        self.context_aware_switching = True
        
        logger.info("Interaction mode manager initialized")
    
    def _initialize_mode_capabilities(self) -> Dict[InteractionMode, ModeCapabilities]:
        """Initialize capabilities for each interaction mode"""
        return {
            InteractionMode.TEXT: ModeCapabilities(
                text_input=True,
                text_output=True,
                visual_feedback=True
            ),
            InteractionMode.VOICE: ModeCapabilities(
                voice_input=True,
                voice_output=True,
                visual_feedback=True,
                screen_overlay=True
            ),
            InteractionMode.VISUAL: ModeCapabilities(
                visual_feedback=True,
                screen_overlay=True,
                text_input=True,
                text_output=True
            ),
            InteractionMode.MIXED: ModeCapabilities(
                voice_input=True,
                voice_output=True,
                visual_feedback=True,
                text_input=True,
                text_output=True,
                screen_overlay=True
            ),
            InteractionMode.ACCESSIBILITY: ModeCapabilities(
                voice_input=True,
                voice_output=True,
                text_input=True,
                text_output=True,
                keyboard_navigation=True,
                screen_reader_support=True,
                visual_feedback=True
            )
        }
    
    async def switch_mode(self, new_mode: InteractionMode, 
                         transition_type: ModeTransition = ModeTransition.USER_INITIATED,
                         preserve_context: bool = True) -> bool:
        """Switch to a new interaction mode"""
        try:
            if new_mode == self.current_mode:
                logger.debug(f"Already in mode {new_mode.value}")
                return True
            
            old_mode = self.current_mode
            
            # Validate mode transition
            if not await self._validate_mode_transition(old_mode, new_mode):
                logger.warning(f"Invalid mode transition from {old_mode.value} to {new_mode.value}")
                return False
            
            # Prepare for mode switch
            await self._prepare_mode_switch(old_mode, new_mode, preserve_context)
            
            # Update context
            self.context.previous_mode = old_mode
            self.context.current_mode = new_mode
            self.context.mode_switches += 1
            self.current_mode = new_mode
            
            # Initialize new mode
            await self._initialize_mode(new_mode)
            
            # Cleanup old mode if needed
            await self._cleanup_mode(old_mode, new_mode)
            
            # Notify handlers
            await self._notify_mode_change(old_mode, new_mode, transition_type)
            
            # Announce mode change if accessibility is enabled
            if self.accessibility_manager.settings.screen_reader_enabled:
                await self.accessibility_manager.announce_to_screen_reader(
                    f"Switched to {new_mode.value} interaction mode"
                )
            
            logger.info(f"Successfully switched from {old_mode.value} to {new_mode.value} mode")
            return True
            
        except Exception as e:
            logger.error(f"Error switching interaction mode: {e}")
            return False
    
    async def _validate_mode_transition(self, old_mode: InteractionMode, 
                                      new_mode: InteractionMode) -> bool:
        """Validate if mode transition is allowed"""
        # Check if required components are available
        capabilities = self.mode_capabilities[new_mode]
        
        if capabilities.voice_input or capabilities.voice_output:
            if not self.voice_processor.is_voice_available():
                logger.warning("Voice mode requested but voice processing not available")
                return False
        
        if capabilities.screen_overlay:
            if not self.screen_overlay.is_available():
                logger.warning("Visual mode requested but screen overlay not available")
                return False
        
        return True
    
    async def _prepare_mode_switch(self, old_mode: InteractionMode, 
                                 new_mode: InteractionMode, preserve_context: bool):
        """Prepare for mode switch"""
        if preserve_context:
            # Save current interaction state
            await self._save_interaction_state()
        
        # Stop active processes in old mode
        if old_mode == InteractionMode.VOICE:
            await self.voice_processor.stop_listening()
        
        # Clear temporary overlays if switching away from visual mode
        if old_mode == InteractionMode.VISUAL and new_mode != InteractionMode.MIXED:
            active_overlays = self.screen_overlay.get_active_overlays()
            for overlay_id in active_overlays:
                await self.screen_overlay.hide_overlay(overlay_id)
    
    async def _initialize_mode(self, mode: InteractionMode):
        """Initialize the new interaction mode"""
        capabilities = self.mode_capabilities[mode]
        
        if capabilities.voice_input:
            # Start voice listening
            await self.voice_processor.start_listening(self._handle_voice_command)
        
        if capabilities.screen_overlay:
            # Show mode indicator overlay
            await self.screen_overlay.show_notification(
                f"Switched to {mode.value} mode",
                duration=2.0,
                position=OverlayPosition.TOP_CENTER
            )
        
        if capabilities.screen_reader_support:
            # Ensure accessibility features are active
            if not self.accessibility_manager.settings.screen_reader_enabled:
                self.accessibility_manager.enable_feature(
                    self.accessibility_manager.AccessibilityFeature.SCREEN_READER
                )
    
    async def _cleanup_mode(self, old_mode: InteractionMode, new_mode: InteractionMode):
        """Clean up resources from the old mode"""
        old_capabilities = self.mode_capabilities[old_mode]
        new_capabilities = self.mode_capabilities[new_mode]
        
        # Stop voice processing if not needed in new mode
        if old_capabilities.voice_input and not new_capabilities.voice_input:
            await self.voice_processor.stop_listening()
        
        # Clean up overlays if not needed
        if old_capabilities.screen_overlay and not new_capabilities.screen_overlay:
            # Keep essential overlays, remove temporary ones
            pass
    
    async def _save_interaction_state(self):
        """Save current interaction state for context preservation"""
        state = {
            "mode": self.current_mode.value,
            "timestamp": datetime.now().isoformat(),
            "conversation_history": self.context.conversation_history[-10:],  # Last 10 interactions
            "active_tasks": self.context.active_tasks,
            "application_context": self.context.application_context
        }
        
        # Add to conversation history
        self.context.conversation_history.append({
            "type": "mode_switch",
            "data": state
        })
    
    async def _notify_mode_change(self, old_mode: InteractionMode, 
                                new_mode: InteractionMode, transition_type: ModeTransition):
        """Notify registered handlers of mode change"""
        for handler in self.mode_change_handlers:
            try:
                await handler(old_mode, new_mode, transition_type)
            except Exception as e:
                logger.error(f"Error in mode change handler: {e}")
    
    async def detect_optimal_mode(self, context_data: Dict[str, Any]) -> InteractionMode:
        """Detect the optimal interaction mode based on context"""
        if not self.auto_mode_detection:
            return self.current_mode
        
        try:
            # Check accessibility requirements first
            if context_data.get("accessibility_required"):
                return InteractionMode.ACCESSIBILITY
            
            # Check application context
            application = context_data.get("application", "")
            
            # Voice-friendly applications
            if application.lower() in ["music", "media", "driving", "cooking"]:
                if self.voice_processor.is_voice_available():
                    return InteractionMode.VOICE
            
            # Visual-heavy applications
            elif application.lower() in ["design", "image", "video", "presentation"]:
                if self.screen_overlay.is_available():
                    return InteractionMode.VISUAL
            
            # Code/text editing
            elif application.lower() in ["editor", "ide", "terminal", "code"]:
                return InteractionMode.TEXT
            
            # Check user activity
            activity = context_data.get("current_activity", "")
            if "typing" in activity.lower():
                return InteractionMode.TEXT
            elif "speaking" in activity.lower() or "meeting" in activity.lower():
                return InteractionMode.VOICE
            
            # Check environmental factors
            noise_level = context_data.get("noise_level", 0)
            if noise_level > 0.7:  # High noise environment
                return InteractionMode.TEXT
            
            # Default to mixed mode if multiple capabilities are available
            if (self.voice_processor.is_voice_available() and 
                self.screen_overlay.is_available()):
                return InteractionMode.MIXED
            
            return self.current_mode
            
        except Exception as e:
            logger.error(f"Error detecting optimal mode: {e}")
            return self.current_mode
    
    async def process_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an interaction based on current mode"""
        try:
            self.context.last_interaction = datetime.now()
            
            # Add to conversation history
            self.context.conversation_history.append({
                "type": "interaction",
                "mode": self.current_mode.value,
                "data": interaction_data,
                "timestamp": self.context.last_interaction.isoformat()
            })
            
            # Route to appropriate handler
            if self.current_mode in self.interaction_handlers:
                return await self.interaction_handlers[self.current_mode](interaction_data)
            else:
                return await self._default_interaction_handler(interaction_data)
                
        except Exception as e:
            logger.error(f"Error processing interaction: {e}")
            return {"success": False, "error": str(e)}
    
    async def _default_interaction_handler(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default interaction handler"""
        capabilities = self.mode_capabilities[self.current_mode]
        response = {"success": True, "responses": []}
        
        # Handle text input/output
        if capabilities.text_input and capabilities.text_output:
            text_response = interaction_data.get("text", "")
            if text_response:
                response["responses"].append({
                    "type": "text",
                    "content": f"Processed in {self.current_mode.value} mode: {text_response}"
                })
        
        # Handle voice input/output
        if capabilities.voice_input and capabilities.voice_output:
            voice_text = interaction_data.get("voice_text", "")
            if voice_text:
                voice_response = VoiceResponse(text=f"Voice response: {voice_text}")
                await self.voice_processor.generate_voice_response(voice_response)
                response["responses"].append({
                    "type": "voice",
                    "content": voice_text
                })
        
        # Handle visual feedback
        if capabilities.visual_feedback:
            visual_content = interaction_data.get("visual_content", "")
            if visual_content:
                await self.screen_overlay.show_notification(visual_content)
                response["responses"].append({
                    "type": "visual",
                    "content": visual_content
                })
        
        return response
    
    async def _handle_voice_command(self, command: VoiceCommand):
        """Handle voice commands"""
        try:
            # Check for mode switching commands
            if command.intent == "context_control":
                mode_param = command.entities.get("param_0", "")
                if "text" in mode_param.lower():
                    await self.switch_mode(InteractionMode.TEXT)
                elif "visual" in mode_param.lower():
                    await self.switch_mode(InteractionMode.VISUAL)
                elif "mixed" in mode_param.lower():
                    await self.switch_mode(InteractionMode.MIXED)
                return
            
            # Process command in current context
            interaction_data = {
                "type": "voice_command",
                "command": command.text,
                "intent": command.intent,
                "entities": command.entities,
                "confidence": command.confidence
            }
            
            await self.process_interaction(interaction_data)
            
        except Exception as e:
            logger.error(f"Error handling voice command: {e}")
    
    def register_mode_change_handler(self, handler: Callable):
        """Register a handler for mode changes"""
        self.mode_change_handlers.append(handler)
        logger.debug("Mode change handler registered")
    
    def register_interaction_handler(self, mode: InteractionMode, handler: Callable):
        """Register a handler for specific interaction mode"""
        self.interaction_handlers[mode] = handler
        logger.debug(f"Interaction handler registered for {mode.value} mode")
    
    async def get_text_completions(self, context: TextContext) -> List[Completion]:
        """Get text completions (available in text and mixed modes)"""
        capabilities = self.mode_capabilities[self.current_mode]
        
        if capabilities.text_input:
            return await self.text_completion.get_completions(context)
        else:
            logger.warning("Text completion not available in current mode")
            return []
    
    async def show_visual_feedback(self, message: str, feedback_type: str = "info"):
        """Show visual feedback (available in visual and mixed modes)"""
        capabilities = self.mode_capabilities[self.current_mode]
        
        if capabilities.visual_feedback:
            if feedback_type == "success":
                overlay_type = OverlayType.CONFIRMATION
            elif feedback_type == "error":
                overlay_type = OverlayType.NOTIFICATION
            else:
                overlay_type = OverlayType.CONTEXT_INFO
            
            config = OverlayConfig(
                overlay_type=overlay_type,
                content=message,
                position=OverlayPosition.BOTTOM_RIGHT,
                duration=3.0
            )
            
            return await self.screen_overlay.show_overlay(config)
        else:
            logger.warning("Visual feedback not available in current mode")
            return ""
    
    async def speak_response(self, text: str, interrupt: bool = False):
        """Speak a response (available in voice and mixed modes)"""
        capabilities = self.mode_capabilities[self.current_mode]
        
        if capabilities.voice_output:
            return await self.voice_processor.speak_text(text, interrupt)
        else:
            logger.warning("Voice output not available in current mode")
            return False
    
    def get_current_mode(self) -> InteractionMode:
        """Get current interaction mode"""
        return self.current_mode
    
    def get_mode_capabilities(self, mode: InteractionMode = None) -> ModeCapabilities:
        """Get capabilities for a specific mode or current mode"""
        target_mode = mode or self.current_mode
        return self.mode_capabilities[target_mode]
    
    def get_interaction_context(self) -> InteractionContext:
        """Get current interaction context"""
        return self.context
    
    def get_status(self) -> Dict[str, Any]:
        """Get interaction mode manager status"""
        return {
            "current_mode": self.current_mode.value,
            "previous_mode": self.context.previous_mode.value if self.context.previous_mode else None,
            "mode_switches": self.context.mode_switches,
            "session_duration": (datetime.now() - self.context.session_start).total_seconds(),
            "last_interaction": self.context.last_interaction.isoformat() if self.context.last_interaction else None,
            "conversation_length": len(self.context.conversation_history),
            "capabilities": {
                mode.value: {
                    "voice_input": caps.voice_input,
                    "voice_output": caps.voice_output,
                    "visual_feedback": caps.visual_feedback,
                    "text_input": caps.text_input,
                    "text_output": caps.text_output,
                    "screen_overlay": caps.screen_overlay,
                    "keyboard_navigation": caps.keyboard_navigation,
                    "screen_reader_support": caps.screen_reader_support
                }
                for mode, caps in self.mode_capabilities.items()
            },
            "component_status": {
                "voice_processor": self.voice_processor.get_voice_status(),
                "screen_overlay": {
                    "available": self.screen_overlay.is_available(),
                    "active_overlays": len(self.screen_overlay.get_active_overlays())
                },
                "text_completion": self.text_completion.get_status(),
                "accessibility": self.accessibility_manager.get_accessibility_status()
            }
        }
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update interaction mode settings"""
        if "auto_mode_detection" in settings:
            self.auto_mode_detection = settings["auto_mode_detection"]
        
        if "context_aware_switching" in settings:
            self.context_aware_switching = settings["context_aware_switching"]
        
        # Update component settings
        if "voice_settings" in settings:
            self.voice_processor.update_settings(settings["voice_settings"])
        
        if "completion_settings" in settings:
            self.text_completion.update_settings(settings["completion_settings"])
        
        if "accessibility_settings" in settings:
            self.accessibility_manager.update_settings(settings["accessibility_settings"])
        
        logger.info(f"Interaction mode settings updated: {settings}")
    
    async def cleanup(self):
        """Clean up interaction mode manager resources"""
        try:
            # Stop voice processing
            await self.voice_processor.cleanup()
            
            # Clear overlays
            await self.screen_overlay.cleanup()
            
            # Clean up accessibility manager
            await self.accessibility_manager.cleanup()
            
            logger.info("Interaction mode manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")