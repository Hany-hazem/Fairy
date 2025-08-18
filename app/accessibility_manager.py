"""
Accessibility Manager Module

This module provides accessibility support for the personal assistant,
including screen reader compatibility, keyboard navigation, high contrast modes,
and other accessibility features to ensure inclusive design.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import platform

logger = logging.getLogger(__name__)


class AccessibilityFeature(Enum):
    """Types of accessibility features"""
    SCREEN_READER = "screen_reader"
    KEYBOARD_NAVIGATION = "keyboard_navigation"
    HIGH_CONTRAST = "high_contrast"
    LARGE_TEXT = "large_text"
    VOICE_CONTROL = "voice_control"
    REDUCED_MOTION = "reduced_motion"
    FOCUS_INDICATORS = "focus_indicators"
    ALTERNATIVE_TEXT = "alternative_text"
    CAPTIONS = "captions"
    MAGNIFICATION = "magnification"


class AccessibilityLevel(Enum):
    """Accessibility compliance levels"""
    A = "A"
    AA = "AA"
    AAA = "AAA"


@dataclass
class AccessibilitySettings:
    """Accessibility configuration settings"""
    enabled_features: List[AccessibilityFeature] = field(default_factory=list)
    compliance_level: AccessibilityLevel = AccessibilityLevel.AA
    screen_reader_enabled: bool = False
    keyboard_only_navigation: bool = False
    high_contrast_mode: bool = False
    large_text_mode: bool = False
    voice_control_enabled: bool = False
    reduced_motion: bool = False
    focus_indicators_enhanced: bool = False
    text_scaling_factor: float = 1.0
    contrast_ratio: float = 4.5  # WCAG AA standard
    animation_duration_multiplier: float = 1.0
    keyboard_timeout: float = 5.0
    voice_feedback_enabled: bool = False
    alternative_text_enabled: bool = True


@dataclass
class AccessibilityEvent:
    """Accessibility event for screen readers and other AT"""
    event_type: str
    element_id: Optional[str]
    element_type: str
    content: str
    description: Optional[str] = None
    role: Optional[str] = None
    state: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: __import__('time').time())


class ScreenReaderSupport:
    """Screen reader integration and support"""
    
    def __init__(self):
        self.is_active = False
        self.current_platform = platform.system().lower()
        self.supported_readers = self._get_supported_readers()
        self.active_reader = None
        
        # Try to detect active screen reader
        self._detect_screen_reader()
    
    def _get_supported_readers(self) -> Dict[str, Dict[str, Any]]:
        """Get supported screen readers by platform"""
        return {
            "windows": {
                "nvda": {"name": "NVDA", "api": "nvda_api"},
                "jaws": {"name": "JAWS", "api": "jaws_api"},
                "narrator": {"name": "Windows Narrator", "api": "uia"}
            },
            "darwin": {  # macOS
                "voiceover": {"name": "VoiceOver", "api": "accessibility_api"}
            },
            "linux": {
                "orca": {"name": "Orca", "api": "at_spi"},
                "speakup": {"name": "Speakup", "api": "speakup_api"}
            }
        }
    
    def _detect_screen_reader(self):
        """Detect if a screen reader is active"""
        try:
            if self.current_platform == "windows":
                self._detect_windows_screen_reader()
            elif self.current_platform == "darwin":
                self._detect_macos_screen_reader()
            elif self.current_platform == "linux":
                self._detect_linux_screen_reader()
        except Exception as e:
            logger.error(f"Error detecting screen reader: {e}")
    
    def _detect_windows_screen_reader(self):
        """Detect Windows screen readers"""
        try:
            import winreg
            import ctypes
            
            # Check for NVDA
            try:
                nvda_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                        r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\nvda.exe")
                self.active_reader = "nvda"
                self.is_active = True
                winreg.CloseKey(nvda_key)
                return
            except FileNotFoundError:
                pass
            
            # Check for JAWS
            try:
                jaws_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                        r"SOFTWARE\Freedom Scientific\JAWS")
                self.active_reader = "jaws"
                self.is_active = True
                winreg.CloseKey(jaws_key)
                return
            except FileNotFoundError:
                pass
            
            # Check for Narrator (Windows built-in)
            try:
                # Check if Narrator is running
                user32 = ctypes.windll.user32
                if user32.FindWindowW("Narrator", None):
                    self.active_reader = "narrator"
                    self.is_active = True
                    return
            except:
                pass
                
        except ImportError:
            logger.warning("Windows-specific modules not available")
    
    def _detect_macos_screen_reader(self):
        """Detect macOS VoiceOver"""
        try:
            import subprocess
            
            # Check if VoiceOver is enabled
            result = subprocess.run([
                "defaults", "read", "com.apple.universalaccess", "voiceOverOnOffKey"
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and "1" in result.stdout:
                self.active_reader = "voiceover"
                self.is_active = True
                
        except Exception as e:
            logger.debug(f"Could not detect VoiceOver: {e}")
    
    def _detect_linux_screen_reader(self):
        """Detect Linux screen readers"""
        try:
            import subprocess
            
            # Check for Orca
            result = subprocess.run(["pgrep", "orca"], capture_output=True)
            if result.returncode == 0:
                self.active_reader = "orca"
                self.is_active = True
                return
            
            # Check for other screen readers
            for reader in ["speakup", "speechd"]:
                result = subprocess.run(["pgrep", reader], capture_output=True)
                if result.returncode == 0:
                    self.active_reader = reader
                    self.is_active = True
                    return
                    
        except Exception as e:
            logger.debug(f"Could not detect Linux screen readers: {e}")
    
    async def announce(self, text: str, priority: str = "normal"):
        """Announce text to screen reader"""
        if not self.is_active:
            logger.debug(f"Screen reader announcement (not active): {text}")
            return
        
        try:
            if self.current_platform == "windows":
                await self._announce_windows(text, priority)
            elif self.current_platform == "darwin":
                await self._announce_macos(text, priority)
            elif self.current_platform == "linux":
                await self._announce_linux(text, priority)
        except Exception as e:
            logger.error(f"Error announcing to screen reader: {e}")
    
    async def _announce_windows(self, text: str, priority: str):
        """Announce to Windows screen readers"""
        # This would integrate with specific Windows screen reader APIs
        # For now, we'll use a generic approach
        logger.info(f"Windows screen reader announcement: {text}")
    
    async def _announce_macos(self, text: str, priority: str):
        """Announce to macOS VoiceOver"""
        try:
            import subprocess
            # Use the 'say' command for VoiceOver
            subprocess.Popen(["say", text])
        except Exception as e:
            logger.error(f"Error with macOS announcement: {e}")
    
    async def _announce_linux(self, text: str, priority: str):
        """Announce to Linux screen readers"""
        try:
            import subprocess
            # Use speech-dispatcher if available
            subprocess.Popen(["spd-say", text])
        except Exception as e:
            logger.debug(f"Could not use speech-dispatcher: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get screen reader status"""
        return {
            "active": self.is_active,
            "reader": self.active_reader,
            "platform": self.current_platform,
            "supported_readers": self.supported_readers.get(self.current_platform, {})
        }


class KeyboardNavigationManager:
    """Manages keyboard-only navigation support"""
    
    def __init__(self):
        self.keyboard_only_mode = False
        self.focus_stack: List[str] = []
        self.current_focus = None
        self.tab_order: List[str] = []
        self.keyboard_shortcuts: Dict[str, Callable] = {}
        self.focus_indicators_enabled = True
    
    def enable_keyboard_only_mode(self):
        """Enable keyboard-only navigation mode"""
        self.keyboard_only_mode = True
        logger.info("Keyboard-only navigation mode enabled")
    
    def disable_keyboard_only_mode(self):
        """Disable keyboard-only navigation mode"""
        self.keyboard_only_mode = False
        logger.info("Keyboard-only navigation mode disabled")
    
    def set_tab_order(self, elements: List[str]):
        """Set the tab order for keyboard navigation"""
        self.tab_order = elements
        logger.debug(f"Tab order set: {elements}")
    
    def register_keyboard_shortcut(self, key_combination: str, callback: Callable):
        """Register a keyboard shortcut"""
        self.keyboard_shortcuts[key_combination] = callback
        logger.debug(f"Keyboard shortcut registered: {key_combination}")
    
    async def handle_key_event(self, key_event: Dict[str, Any]) -> bool:
        """Handle keyboard events for navigation"""
        try:
            key = key_event.get("key", "")
            modifiers = key_event.get("modifiers", [])
            
            # Handle tab navigation
            if key == "Tab":
                if "Shift" in modifiers:
                    await self._focus_previous()
                else:
                    await self._focus_next()
                return True
            
            # Handle arrow key navigation
            elif key in ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]:
                await self._handle_arrow_navigation(key)
                return True
            
            # Handle Enter/Space for activation
            elif key in ["Enter", "Space"]:
                await self._activate_current_element()
                return True
            
            # Handle Escape for cancellation
            elif key == "Escape":
                await self._handle_escape()
                return True
            
            # Handle registered shortcuts
            key_combo = self._create_key_combination(key, modifiers)
            if key_combo in self.keyboard_shortcuts:
                await self.keyboard_shortcuts[key_combo]()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling key event: {e}")
            return False
    
    def _create_key_combination(self, key: str, modifiers: List[str]) -> str:
        """Create key combination string"""
        parts = sorted(modifiers) + [key]
        return "+".join(parts)
    
    async def _focus_next(self):
        """Focus next element in tab order"""
        if not self.tab_order:
            return
        
        if self.current_focus is None:
            self.current_focus = self.tab_order[0]
        else:
            try:
                current_index = self.tab_order.index(self.current_focus)
                next_index = (current_index + 1) % len(self.tab_order)
                self.current_focus = self.tab_order[next_index]
            except ValueError:
                self.current_focus = self.tab_order[0]
        
        await self._update_focus()
    
    async def _focus_previous(self):
        """Focus previous element in tab order"""
        if not self.tab_order:
            return
        
        if self.current_focus is None:
            self.current_focus = self.tab_order[-1]
        else:
            try:
                current_index = self.tab_order.index(self.current_focus)
                prev_index = (current_index - 1) % len(self.tab_order)
                self.current_focus = self.tab_order[prev_index]
            except ValueError:
                self.current_focus = self.tab_order[-1]
        
        await self._update_focus()
    
    async def _handle_arrow_navigation(self, key: str):
        """Handle arrow key navigation"""
        # This would be implemented based on the specific UI layout
        logger.debug(f"Arrow navigation: {key}")
    
    async def _activate_current_element(self):
        """Activate the currently focused element"""
        if self.current_focus:
            logger.debug(f"Activating element: {self.current_focus}")
            # This would trigger the element's action
    
    async def _handle_escape(self):
        """Handle escape key press"""
        if self.focus_stack:
            self.current_focus = self.focus_stack.pop()
            await self._update_focus()
        logger.debug("Escape key handled")
    
    async def _update_focus(self):
        """Update focus indicators and announce to screen reader"""
        if self.current_focus:
            logger.debug(f"Focus updated to: {self.current_focus}")
            # This would update visual focus indicators
            # and announce to screen reader if needed


class HighContrastManager:
    """Manages high contrast mode and color accessibility"""
    
    def __init__(self):
        self.high_contrast_enabled = False
        self.contrast_ratio = 4.5  # WCAG AA standard
        self.color_schemes = self._initialize_color_schemes()
        self.current_scheme = "default"
    
    def _initialize_color_schemes(self) -> Dict[str, Dict[str, str]]:
        """Initialize high contrast color schemes"""
        return {
            "default": {
                "background": "#FFFFFF",
                "text": "#000000",
                "accent": "#0066CC",
                "border": "#666666"
            },
            "high_contrast_dark": {
                "background": "#000000",
                "text": "#FFFFFF",
                "accent": "#FFFF00",
                "border": "#FFFFFF"
            },
            "high_contrast_light": {
                "background": "#FFFFFF",
                "text": "#000000",
                "accent": "#0000FF",
                "border": "#000000"
            },
            "yellow_on_black": {
                "background": "#000000",
                "text": "#FFFF00",
                "accent": "#00FFFF",
                "border": "#FFFF00"
            }
        }
    
    def enable_high_contrast(self, scheme: str = "high_contrast_dark"):
        """Enable high contrast mode"""
        if scheme in self.color_schemes:
            self.high_contrast_enabled = True
            self.current_scheme = scheme
            logger.info(f"High contrast mode enabled with scheme: {scheme}")
        else:
            logger.warning(f"Unknown color scheme: {scheme}")
    
    def disable_high_contrast(self):
        """Disable high contrast mode"""
        self.high_contrast_enabled = False
        self.current_scheme = "default"
        logger.info("High contrast mode disabled")
    
    def get_current_colors(self) -> Dict[str, str]:
        """Get current color scheme"""
        return self.color_schemes[self.current_scheme]
    
    def calculate_contrast_ratio(self, color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors"""
        # Simplified contrast ratio calculation
        # In a real implementation, this would use proper color space calculations
        return 4.5  # Placeholder


class AccessibilityManager:
    """
    Main accessibility manager that coordinates all accessibility features
    and ensures WCAG compliance for the personal assistant.
    """
    
    def __init__(self, settings: AccessibilitySettings = None):
        self.settings = settings or AccessibilitySettings()
        self.screen_reader = ScreenReaderSupport()
        self.keyboard_nav = KeyboardNavigationManager()
        self.high_contrast = HighContrastManager()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Initialize based on settings
        self._initialize_features()
        
        logger.info("Accessibility manager initialized")
    
    def _initialize_features(self):
        """Initialize accessibility features based on settings"""
        if AccessibilityFeature.SCREEN_READER in self.settings.enabled_features:
            self.settings.screen_reader_enabled = True
        
        if AccessibilityFeature.KEYBOARD_NAVIGATION in self.settings.enabled_features:
            self.settings.keyboard_only_navigation = True
            self.keyboard_nav.enable_keyboard_only_mode()
        
        if AccessibilityFeature.HIGH_CONTRAST in self.settings.enabled_features:
            self.settings.high_contrast_mode = True
            self.high_contrast.enable_high_contrast()
        
        if AccessibilityFeature.LARGE_TEXT in self.settings.enabled_features:
            self.settings.large_text_mode = True
        
        if AccessibilityFeature.REDUCED_MOTION in self.settings.enabled_features:
            self.settings.reduced_motion = True
    
    async def announce_to_screen_reader(self, text: str, priority: str = "normal"):
        """Announce text to screen reader"""
        if self.settings.screen_reader_enabled:
            await self.screen_reader.announce(text, priority)
    
    async def handle_keyboard_event(self, key_event: Dict[str, Any]) -> bool:
        """Handle keyboard events for accessibility"""
        if self.settings.keyboard_only_navigation:
            return await self.keyboard_nav.handle_key_event(key_event)
        return False
    
    def enable_feature(self, feature: AccessibilityFeature):
        """Enable an accessibility feature"""
        if feature not in self.settings.enabled_features:
            self.settings.enabled_features.append(feature)
        
        if feature == AccessibilityFeature.SCREEN_READER:
            self.settings.screen_reader_enabled = True
        elif feature == AccessibilityFeature.KEYBOARD_NAVIGATION:
            self.settings.keyboard_only_navigation = True
            self.keyboard_nav.enable_keyboard_only_mode()
        elif feature == AccessibilityFeature.HIGH_CONTRAST:
            self.settings.high_contrast_mode = True
            self.high_contrast.enable_high_contrast()
        elif feature == AccessibilityFeature.LARGE_TEXT:
            self.settings.large_text_mode = True
        elif feature == AccessibilityFeature.REDUCED_MOTION:
            self.settings.reduced_motion = True
        
        logger.info(f"Accessibility feature enabled: {feature.value}")
    
    def disable_feature(self, feature: AccessibilityFeature):
        """Disable an accessibility feature"""
        if feature in self.settings.enabled_features:
            self.settings.enabled_features.remove(feature)
        
        if feature == AccessibilityFeature.SCREEN_READER:
            self.settings.screen_reader_enabled = False
        elif feature == AccessibilityFeature.KEYBOARD_NAVIGATION:
            self.settings.keyboard_only_navigation = False
            self.keyboard_nav.disable_keyboard_only_mode()
        elif feature == AccessibilityFeature.HIGH_CONTRAST:
            self.settings.high_contrast_mode = False
            self.high_contrast.disable_high_contrast()
        elif feature == AccessibilityFeature.LARGE_TEXT:
            self.settings.large_text_mode = False
        elif feature == AccessibilityFeature.REDUCED_MOTION:
            self.settings.reduced_motion = False
        
        logger.info(f"Accessibility feature disabled: {feature.value}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an accessibility event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Accessibility event handler registered: {event_type}")
    
    async def emit_accessibility_event(self, event: AccessibilityEvent):
        """Emit an accessibility event"""
        try:
            # Announce to screen reader if appropriate
            if self.settings.screen_reader_enabled and event.content:
                await self.announce_to_screen_reader(event.content)
            
            # Call registered handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Error in accessibility event handler: {e}")
            
        except Exception as e:
            logger.error(f"Error emitting accessibility event: {e}")
    
    def get_accessibility_status(self) -> Dict[str, Any]:
        """Get current accessibility status"""
        return {
            "enabled_features": [f.value for f in self.settings.enabled_features],
            "compliance_level": self.settings.compliance_level.value,
            "screen_reader": {
                "enabled": self.settings.screen_reader_enabled,
                "status": self.screen_reader.get_status()
            },
            "keyboard_navigation": {
                "enabled": self.settings.keyboard_only_navigation,
                "current_focus": self.keyboard_nav.current_focus
            },
            "high_contrast": {
                "enabled": self.settings.high_contrast_mode,
                "scheme": self.high_contrast.current_scheme
            },
            "text_scaling": self.settings.text_scaling_factor,
            "reduced_motion": self.settings.reduced_motion
        }
    
    def update_settings(self, new_settings: Dict[str, Any]):
        """Update accessibility settings"""
        for key, value in new_settings.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        
        # Re-initialize features based on new settings
        self._initialize_features()
        
        logger.info(f"Accessibility settings updated: {new_settings}")
    
    def check_wcag_compliance(self) -> Dict[str, Any]:
        """Check WCAG compliance status"""
        compliance_checks = {
            "contrast_ratio": self.high_contrast.contrast_ratio >= 4.5,
            "keyboard_accessible": self.settings.keyboard_only_navigation,
            "screen_reader_support": self.settings.screen_reader_enabled,
            "alternative_text": self.settings.alternative_text_enabled,
            "focus_indicators": self.settings.focus_indicators_enhanced,
            "text_scaling": self.settings.text_scaling_factor >= 1.0
        }
        
        compliance_level = AccessibilityLevel.A
        if all(compliance_checks.values()):
            compliance_level = AccessibilityLevel.AA
        
        return {
            "compliance_level": compliance_level.value,
            "checks": compliance_checks,
            "recommendations": self._get_compliance_recommendations(compliance_checks)
        }
    
    def _get_compliance_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Get recommendations for improving accessibility compliance"""
        recommendations = []
        
        if not checks["contrast_ratio"]:
            recommendations.append("Increase color contrast ratio to meet WCAG AA standards")
        
        if not checks["keyboard_accessible"]:
            recommendations.append("Enable keyboard navigation support")
        
        if not checks["screen_reader_support"]:
            recommendations.append("Enable screen reader support")
        
        if not checks["alternative_text"]:
            recommendations.append("Provide alternative text for images and media")
        
        if not checks["focus_indicators"]:
            recommendations.append("Enhance focus indicators for better visibility")
        
        return recommendations
    
    async def cleanup(self):
        """Clean up accessibility manager resources"""
        # Clean up any resources
        logger.info("Accessibility manager cleaned up")