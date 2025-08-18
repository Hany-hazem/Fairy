"""
Screen Overlay Module

This module provides screen overlay and annotation capabilities for visual feedback
and contextual information display in the personal assistant.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from datetime import datetime, timedelta

try:
    import tkinter as tk
    from tkinter import ttk
    import tkinter.font as tkFont
    OVERLAY_AVAILABLE = True
except ImportError:
    OVERLAY_AVAILABLE = False
    tk = None
    ttk = None
    tkFont = None

logger = logging.getLogger(__name__)


class OverlayType(Enum):
    """Types of screen overlays"""
    NOTIFICATION = "notification"
    CONTEXT_INFO = "context_info"
    ANNOTATION = "annotation"
    TOOLTIP = "tooltip"
    PROGRESS = "progress"
    CONFIRMATION = "confirmation"
    HELP = "help"


class OverlayPosition(Enum):
    """Overlay positioning options"""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    TOP_CENTER = "top_center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_CENTER = "bottom_center"
    CENTER = "center"
    CURSOR = "cursor"
    CUSTOM = "custom"


@dataclass
class OverlayStyle:
    """Styling configuration for overlays"""
    background_color: str = "#2D2D2D"
    text_color: str = "#FFFFFF"
    border_color: str = "#4A4A4A"
    border_width: int = 1
    font_family: str = "Arial"
    font_size: int = 12
    font_weight: str = "normal"
    padding: int = 10
    margin: int = 5
    opacity: float = 0.9
    corner_radius: int = 5
    shadow: bool = True
    animation: bool = True


@dataclass
class OverlayConfig:
    """Configuration for a screen overlay"""
    overlay_type: OverlayType
    content: str
    position: OverlayPosition = OverlayPosition.TOP_RIGHT
    custom_position: Optional[Tuple[int, int]] = None
    duration: Optional[float] = None  # None = persistent
    style: OverlayStyle = field(default_factory=OverlayStyle)
    interactive: bool = False
    buttons: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority overlays appear on top
    auto_hide: bool = True
    fade_in: bool = True
    fade_out: bool = True


@dataclass
class OverlayAction:
    """Action that can be triggered from an overlay"""
    action_id: str
    label: str
    callback: Optional[Callable] = None
    style: str = "default"  # default, primary, secondary, danger


class OverlayWindow:
    """Individual overlay window implementation"""
    
    def __init__(self, config: OverlayConfig, overlay_id: str):
        self.config = config
        self.overlay_id = overlay_id
        self.window = None
        self.is_visible = False
        self.created_at = datetime.now()
        self.expires_at = None
        
        if config.duration:
            self.expires_at = self.created_at + timedelta(seconds=config.duration)
    
    def create_window(self):
        """Create the overlay window"""
        if not OVERLAY_AVAILABLE:
            logger.warning("Tkinter not available for overlay creation")
            return False
        
        try:
            self.window = tk.Toplevel()
            self.window.withdraw()  # Hide initially
            
            # Configure window properties
            self.window.overrideredirect(True)  # Remove window decorations
            self.window.attributes('-topmost', True)  # Always on top
            self.window.attributes('-alpha', self.config.style.opacity)
            
            # Set window background
            self.window.configure(bg=self.config.style.background_color)
            
            # Create content
            self._create_content()
            
            # Position window
            self._position_window()
            
            # Set up auto-hide timer if needed
            if self.config.duration and self.config.auto_hide:
                self.window.after(int(self.config.duration * 1000), self.hide)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating overlay window: {e}")
            return False
    
    def _create_content(self):
        """Create the overlay content"""
        # Main frame
        main_frame = tk.Frame(
            self.window,
            bg=self.config.style.background_color,
            bd=self.config.style.border_width,
            relief="solid"
        )
        main_frame.pack(fill="both", expand=True, padx=self.config.style.margin, pady=self.config.style.margin)
        
        # Content label
        font = tkFont.Font(
            family=self.config.style.font_family,
            size=self.config.style.font_size,
            weight=self.config.style.font_weight
        )
        
        content_label = tk.Label(
            main_frame,
            text=self.config.content,
            bg=self.config.style.background_color,
            fg=self.config.style.text_color,
            font=font,
            wraplength=400,  # Wrap long text
            justify="left",
            padx=self.config.style.padding,
            pady=self.config.style.padding
        )
        content_label.pack(fill="both", expand=True)
        
        # Add buttons if interactive
        if self.config.interactive and self.config.buttons:
            self._create_buttons(main_frame)
    
    def _create_buttons(self, parent):
        """Create interactive buttons"""
        button_frame = tk.Frame(parent, bg=self.config.style.background_color)
        button_frame.pack(fill="x", padx=self.config.style.padding, pady=(0, self.config.style.padding))
        
        for button_config in self.config.buttons:
            button = tk.Button(
                button_frame,
                text=button_config.get("text", "OK"),
                command=lambda cmd=button_config.get("command"): self._handle_button_click(cmd),
                bg="#4A4A4A",
                fg=self.config.style.text_color,
                relief="flat",
                padx=10,
                pady=5
            )
            button.pack(side="right", padx=(5, 0))
    
    def _handle_button_click(self, command):
        """Handle button click events"""
        if command:
            try:
                if callable(command):
                    command()
                elif isinstance(command, str):
                    # Handle string commands (could be extended)
                    if command == "close":
                        self.hide()
                    elif command == "dismiss":
                        self.hide()
            except Exception as e:
                logger.error(f"Error handling button click: {e}")
        
        # Auto-hide after button click unless specified otherwise
        if self.config.auto_hide:
            self.hide()
    
    def _position_window(self):
        """Position the overlay window"""
        if not self.window:
            return
        
        # Update window to get accurate size
        self.window.update_idletasks()
        
        # Get window dimensions
        width = self.window.winfo_reqwidth()
        height = self.window.winfo_reqheight()
        
        # Get screen dimensions
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        
        # Calculate position based on configuration
        if self.config.position == OverlayPosition.TOP_LEFT:
            x, y = 10, 10
        elif self.config.position == OverlayPosition.TOP_RIGHT:
            x, y = screen_width - width - 10, 10
        elif self.config.position == OverlayPosition.TOP_CENTER:
            x, y = (screen_width - width) // 2, 10
        elif self.config.position == OverlayPosition.BOTTOM_LEFT:
            x, y = 10, screen_height - height - 50
        elif self.config.position == OverlayPosition.BOTTOM_RIGHT:
            x, y = screen_width - width - 10, screen_height - height - 50
        elif self.config.position == OverlayPosition.BOTTOM_CENTER:
            x, y = (screen_width - width) // 2, screen_height - height - 50
        elif self.config.position == OverlayPosition.CENTER:
            x, y = (screen_width - width) // 2, (screen_height - height) // 2
        elif self.config.position == OverlayPosition.CUSTOM and self.config.custom_position:
            x, y = self.config.custom_position
        else:
            x, y = screen_width - width - 10, 10  # Default to top-right
        
        # Set window position
        self.window.geometry(f"{width}x{height}+{x}+{y}")
    
    def show(self):
        """Show the overlay"""
        if not self.window:
            if not self.create_window():
                return False
        
        try:
            if self.config.fade_in and self.config.style.animation:
                self._fade_in()
            else:
                self.window.deiconify()
            
            self.is_visible = True
            return True
            
        except Exception as e:
            logger.error(f"Error showing overlay: {e}")
            return False
    
    def hide(self):
        """Hide the overlay"""
        if not self.window or not self.is_visible:
            return
        
        try:
            if self.config.fade_out and self.config.style.animation:
                self._fade_out()
            else:
                self.window.withdraw()
            
            self.is_visible = False
            
        except Exception as e:
            logger.error(f"Error hiding overlay: {e}")
    
    def destroy(self):
        """Destroy the overlay window"""
        if self.window:
            try:
                self.window.destroy()
                self.window = None
                self.is_visible = False
            except Exception as e:
                logger.error(f"Error destroying overlay: {e}")
    
    def _fade_in(self):
        """Fade in animation"""
        if not self.window:
            return
        
        self.window.attributes('-alpha', 0.0)
        self.window.deiconify()
        
        def animate():
            current_alpha = self.window.attributes('-alpha')
            if current_alpha < self.config.style.opacity:
                new_alpha = min(current_alpha + 0.1, self.config.style.opacity)
                self.window.attributes('-alpha', new_alpha)
                self.window.after(50, animate)
        
        animate()
    
    def _fade_out(self):
        """Fade out animation"""
        if not self.window:
            return
        
        def animate():
            current_alpha = self.window.attributes('-alpha')
            if current_alpha > 0.0:
                new_alpha = max(current_alpha - 0.1, 0.0)
                self.window.attributes('-alpha', new_alpha)
                if new_alpha > 0.0:
                    self.window.after(50, animate)
                else:
                    self.window.withdraw()
            else:
                self.window.withdraw()
        
        animate()
    
    def is_expired(self) -> bool:
        """Check if overlay has expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at
    
    def update_content(self, new_content: str):
        """Update overlay content"""
        self.config.content = new_content
        if self.window and self.is_visible:
            # Recreate content with new text
            for widget in self.window.winfo_children():
                widget.destroy()
            self._create_content()


class ScreenOverlay:
    """
    Screen overlay system for visual feedback and contextual information display.
    
    Manages multiple overlay windows with different types, positioning, and styling.
    """
    
    def __init__(self):
        self.overlays: Dict[str, OverlayWindow] = {}
        self.overlay_counter = 0
        self.is_initialized = False
        self.root_window = None
        self.cleanup_thread = None
        self.running = False
        
        # Default styles for different overlay types
        self.default_styles = self._create_default_styles()
        
        if OVERLAY_AVAILABLE:
            self._initialize_overlay_system()
    
    def _initialize_overlay_system(self):
        """Initialize the overlay system"""
        try:
            # Create root window (hidden)
            self.root_window = tk.Tk()
            self.root_window.withdraw()  # Hide the root window
            self.root_window.title("Personal Assistant Overlays")
            
            self.is_initialized = True
            self.running = True
            
            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            
            logger.info("Screen overlay system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize overlay system: {e}")
            self.is_initialized = False
    
    def _create_default_styles(self) -> Dict[OverlayType, OverlayStyle]:
        """Create default styles for different overlay types"""
        return {
            OverlayType.NOTIFICATION: OverlayStyle(
                background_color="#2D2D2D",
                text_color="#FFFFFF",
                border_color="#4A90E2",
                font_size=12
            ),
            OverlayType.CONTEXT_INFO: OverlayStyle(
                background_color="#1E1E1E",
                text_color="#E0E0E0",
                border_color="#666666",
                font_size=11,
                opacity=0.85
            ),
            OverlayType.ANNOTATION: OverlayStyle(
                background_color="#FFF3CD",
                text_color="#856404",
                border_color="#FFEAA7",
                font_size=10,
                opacity=0.9
            ),
            OverlayType.TOOLTIP: OverlayStyle(
                background_color="#000000",
                text_color="#FFFFFF",
                border_color="#333333",
                font_size=10,
                padding=5,
                opacity=0.8
            ),
            OverlayType.PROGRESS: OverlayStyle(
                background_color="#2D2D2D",
                text_color="#FFFFFF",
                border_color="#4CAF50",
                font_size=12
            ),
            OverlayType.CONFIRMATION: OverlayStyle(
                background_color="#D4EDDA",
                text_color="#155724",
                border_color="#C3E6CB",
                font_size=12
            ),
            OverlayType.HELP: OverlayStyle(
                background_color="#E3F2FD",
                text_color="#0D47A1",
                border_color="#BBDEFB",
                font_size=11
            )
        }
    
    async def show_overlay(self, config: OverlayConfig) -> str:
        """Show a new overlay and return its ID"""
        if not self.is_initialized:
            logger.warning("Overlay system not initialized")
            return ""
        
        try:
            # Generate unique overlay ID
            overlay_id = f"overlay_{self.overlay_counter}"
            self.overlay_counter += 1
            
            # Apply default style if not specified
            if config.overlay_type in self.default_styles:
                default_style = self.default_styles[config.overlay_type]
                # Merge with provided style
                for attr in dir(default_style):
                    if not attr.startswith('_') and not hasattr(config.style, attr):
                        setattr(config.style, attr, getattr(default_style, attr))
            
            # Create overlay window
            overlay = OverlayWindow(config, overlay_id)
            
            # Show overlay
            if overlay.show():
                self.overlays[overlay_id] = overlay
                logger.info(f"Overlay {overlay_id} created and shown")
                return overlay_id
            else:
                logger.error(f"Failed to show overlay {overlay_id}")
                return ""
                
        except Exception as e:
            logger.error(f"Error creating overlay: {e}")
            return ""
    
    async def hide_overlay(self, overlay_id: str):
        """Hide a specific overlay"""
        if overlay_id in self.overlays:
            self.overlays[overlay_id].hide()
            logger.info(f"Overlay {overlay_id} hidden")
    
    async def remove_overlay(self, overlay_id: str):
        """Remove a specific overlay"""
        if overlay_id in self.overlays:
            self.overlays[overlay_id].destroy()
            del self.overlays[overlay_id]
            logger.info(f"Overlay {overlay_id} removed")
    
    async def update_overlay(self, overlay_id: str, new_content: str):
        """Update overlay content"""
        if overlay_id in self.overlays:
            self.overlays[overlay_id].update_content(new_content)
            logger.info(f"Overlay {overlay_id} content updated")
    
    async def show_notification(self, message: str, duration: float = 5.0, 
                              position: OverlayPosition = OverlayPosition.TOP_RIGHT) -> str:
        """Show a simple notification overlay"""
        config = OverlayConfig(
            overlay_type=OverlayType.NOTIFICATION,
            content=message,
            position=position,
            duration=duration,
            auto_hide=True
        )
        return await self.show_overlay(config)
    
    async def show_context_info(self, info: str, 
                              position: OverlayPosition = OverlayPosition.TOP_LEFT) -> str:
        """Show contextual information overlay"""
        config = OverlayConfig(
            overlay_type=OverlayType.CONTEXT_INFO,
            content=info,
            position=position,
            duration=None,  # Persistent
            auto_hide=False
        )
        return await self.show_overlay(config)
    
    async def show_confirmation(self, message: str, callback: Callable = None,
                              duration: float = 10.0) -> str:
        """Show confirmation overlay with buttons"""
        buttons = [
            {"text": "OK", "command": callback or "close"},
            {"text": "Cancel", "command": "close"}
        ]
        
        config = OverlayConfig(
            overlay_type=OverlayType.CONFIRMATION,
            content=message,
            position=OverlayPosition.CENTER,
            duration=duration,
            interactive=True,
            buttons=buttons,
            auto_hide=False
        )
        return await self.show_overlay(config)
    
    async def show_progress(self, message: str, progress: float = 0.0) -> str:
        """Show progress overlay"""
        progress_text = f"{message}\nProgress: {progress:.1%}"
        
        config = OverlayConfig(
            overlay_type=OverlayType.PROGRESS,
            content=progress_text,
            position=OverlayPosition.BOTTOM_CENTER,
            duration=None,
            auto_hide=False
        )
        return await self.show_overlay(config)
    
    async def show_help_tooltip(self, help_text: str, 
                               position: Tuple[int, int] = None) -> str:
        """Show help tooltip at specific position"""
        config = OverlayConfig(
            overlay_type=OverlayType.TOOLTIP,
            content=help_text,
            position=OverlayPosition.CUSTOM if position else OverlayPosition.CURSOR,
            custom_position=position,
            duration=8.0,
            auto_hide=True
        )
        return await self.show_overlay(config)
    
    async def show_contextual_overlay(self, context_data: Dict[str, Any], 
                                    screen_context: Any = None):
        """Show overlay based on screen context"""
        try:
            # Analyze context and create appropriate overlay
            if screen_context and hasattr(screen_context, 'active_application'):
                app_name = screen_context.active_application
                
                # Create context-specific overlay content
                content_parts = [f"Working in: {app_name}"]
                
                if hasattr(screen_context, 'window_title') and screen_context.window_title:
                    content_parts.append(f"Window: {screen_context.window_title}")
                
                if context_data.get('suggestions'):
                    content_parts.append("\nSuggestions:")
                    for suggestion in context_data['suggestions'][:3]:
                        content_parts.append(f"• {suggestion}")
                
                content = "\n".join(content_parts)
                
                return await self.show_context_info(content)
            
            else:
                # Generic context overlay
                content = "Context information available"
                if context_data.get('current_task'):
                    content += f"\nCurrent task: {context_data['current_task']}"
                
                return await self.show_context_info(content)
                
        except Exception as e:
            logger.error(f"Error showing contextual overlay: {e}")
            return ""
    
    async def show_visual_feedback(self, action: str, result: Any, duration: float = 3.0):
        """Show visual feedback for actions"""
        try:
            if hasattr(result, 'success') and result.success:
                message = f"✓ {action} completed successfully"
                overlay_type = OverlayType.CONFIRMATION
            else:
                message = f"✗ {action} failed"
                if hasattr(result, 'error_message'):
                    message += f": {result.error_message}"
                overlay_type = OverlayType.NOTIFICATION
            
            config = OverlayConfig(
                overlay_type=overlay_type,
                content=message,
                position=OverlayPosition.BOTTOM_RIGHT,
                duration=duration,
                auto_hide=True
            )
            
            return await self.show_overlay(config)
            
        except Exception as e:
            logger.error(f"Error showing visual feedback: {e}")
            return ""
    
    def get_active_overlays(self) -> List[str]:
        """Get list of active overlay IDs"""
        return [overlay_id for overlay_id, overlay in self.overlays.items() 
                if overlay.is_visible]
    
    def get_overlay_count(self) -> int:
        """Get total number of overlays"""
        return len(self.overlays)
    
    async def clear_all_overlays(self):
        """Clear all overlays"""
        overlay_ids = list(self.overlays.keys())
        for overlay_id in overlay_ids:
            await self.remove_overlay(overlay_id)
        logger.info("All overlays cleared")
    
    def _cleanup_loop(self):
        """Background cleanup loop for expired overlays"""
        while self.running:
            try:
                expired_overlays = []
                
                for overlay_id, overlay in self.overlays.items():
                    if overlay.is_expired():
                        expired_overlays.append(overlay_id)
                
                # Remove expired overlays
                for overlay_id in expired_overlays:
                    asyncio.create_task(self.remove_overlay(overlay_id))
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in overlay cleanup loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def is_available(self) -> bool:
        """Check if overlay system is available"""
        return OVERLAY_AVAILABLE and self.is_initialized
    
    async def cleanup(self):
        """Clean up overlay system"""
        self.running = False
        
        # Clear all overlays
        await self.clear_all_overlays()
        
        # Clean up root window
        if self.root_window:
            try:
                self.root_window.quit()
                self.root_window.destroy()
            except:
                pass
        
        logger.info("Screen overlay system cleaned up")


# Mock implementation for when GUI libraries are not available
class MockScreenOverlay:
    """Mock screen overlay for testing and fallback"""
    
    def __init__(self):
        self.overlays = {}
        self.overlay_counter = 0
    
    async def show_overlay(self, config):
        overlay_id = f"mock_overlay_{self.overlay_counter}"
        self.overlay_counter += 1
        self.overlays[overlay_id] = config
        logger.info(f"Mock overlay: {config.content}")
        return overlay_id
    
    async def hide_overlay(self, overlay_id):
        logger.info(f"Mock hide overlay: {overlay_id}")
    
    async def remove_overlay(self, overlay_id):
        if overlay_id in self.overlays:
            del self.overlays[overlay_id]
        logger.info(f"Mock remove overlay: {overlay_id}")
    
    async def show_notification(self, message, duration=5.0, position=None):
        logger.info(f"Mock notification: {message}")
        return "mock_notification"
    
    async def show_context_info(self, info, position=None):
        logger.info(f"Mock context info: {info}")
        return "mock_context"
    
    async def show_visual_feedback(self, action, result, duration=3.0):
        logger.info(f"Mock feedback: {action} - {result}")
        return "mock_feedback"
    
    def is_available(self):
        return False
    
    async def cleanup(self):
        pass