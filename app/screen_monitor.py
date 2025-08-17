"""
Screen Monitor Module

This module provides screen capture and analysis capabilities for contextual awareness,
including OCR text extraction, application detection, and privacy controls.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from pathlib import Path

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageGrab
    import pytesseract
    import psutil
    HAS_SCREEN_DEPS = True
except ImportError:
    # Create dummy imports for type hints when dependencies are not available
    import numpy as np
    cv2 = None
    Image = None
    ImageGrab = None
    pytesseract = None
    psutil = None
    HAS_SCREEN_DEPS = False

from .personal_assistant_models import UserContext, Interaction, InteractionType
from .privacy_security_manager import PrivacySecurityManager, PermissionType, DataCategory, ConsentStatus

logger = logging.getLogger(__name__)


class MonitoringMode(Enum):
    """Screen monitoring modes"""
    DISABLED = "disabled"
    SELECTIVE = "selective"  # Only monitor specific applications
    FULL = "full"  # Monitor all screen content
    PRIVACY = "privacy"  # Monitor with content filtering


class ApplicationType(Enum):
    """Types of applications for context detection"""
    BROWSER = "browser"
    IDE = "ide"
    TERMINAL = "terminal"
    OFFICE = "office"
    COMMUNICATION = "communication"
    MEDIA = "media"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class UIElement:
    """Represents a UI element detected on screen"""
    element_type: str
    text: str
    position: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreenContext:
    """Current screen context information"""
    active_application: str
    window_title: str
    visible_text: str
    ui_elements: List[UIElement]
    detected_actions: List[str]
    context_summary: str
    timestamp: datetime
    application_type: ApplicationType = ApplicationType.UNKNOWN
    screenshot_hash: Optional[str] = None
    privacy_filtered: bool = False


@dataclass
class MonitorConfig:
    """Configuration for screen monitoring"""
    mode: MonitoringMode = MonitoringMode.DISABLED
    capture_interval: float = 2.0  # seconds
    ocr_enabled: bool = True
    application_detection: bool = True
    privacy_filtering: bool = True
    excluded_applications: Set[str] = field(default_factory=set)
    included_applications: Set[str] = field(default_factory=set)
    sensitive_keywords: Set[str] = field(default_factory=lambda: {
        "password", "credit card", "ssn", "social security", "bank account"
    })
    max_text_length: int = 10000
    store_screenshots: bool = False


class ScreenAnalysis:
    """Results of screen content analysis"""
    
    def __init__(self, screenshot: np.ndarray, config: MonitorConfig):
        self.screenshot = screenshot
        self.config = config
        self.text_content = ""
        self.ui_elements: List[UIElement] = []
        self.detected_applications: List[str] = []
        self.context_summary = ""
        self.privacy_filtered = False
        self.analysis_timestamp = datetime.now()
    
    def extract_text(self) -> str:
        """Extract text from screenshot using OCR"""
        if not self.config.ocr_enabled or not HAS_SCREEN_DEPS:
            return ""
        
        try:
            # Check if dependencies are actually available
            if cv2 is None or Image is None or pytesseract is None:
                return ""
            
            # Convert to PIL Image for OCR
            pil_image = Image.fromarray(cv2.cvtColor(self.screenshot, cv2.COLOR_BGR2RGB))
            
            # Use pytesseract for OCR
            text = pytesseract.image_to_string(pil_image, config='--psm 6')
            
            # Apply privacy filtering
            if self.config.privacy_filtering:
                text = self._filter_sensitive_content(text)
            
            # Limit text length
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length] + "..."
            
            self.text_content = text.strip()
            return self.text_content
            
        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")
            return ""
    
    def detect_ui_elements(self) -> List[UIElement]:
        """Detect UI elements in the screenshot"""
        if not HAS_SCREEN_DEPS or cv2 is None or Image is None or pytesseract is None:
            return []
        
        try:
            # Simple UI element detection using contours
            gray = cv2.cvtColor(self.screenshot, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            elements = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very small elements
                if w > 20 and h > 10:
                    # Extract text from this region if possible
                    roi = self.screenshot[y:y+h, x:x+w]
                    try:
                        element_text = pytesseract.image_to_string(
                            Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                        ).strip()
                        
                        if element_text and self.config.privacy_filtering:
                            element_text = self._filter_sensitive_content(element_text)
                        
                        element = UIElement(
                            element_type="text_region",
                            text=element_text,
                            position=(x, y, w, h),
                            confidence=0.8,
                            metadata={"area": w * h}
                        )
                        elements.append(element)
                        
                    except Exception:
                        # Skip elements where OCR fails
                        continue
            
            # Sort by area (largest first) and limit results
            elements.sort(key=lambda e: e.metadata.get("area", 0), reverse=True)
            self.ui_elements = elements[:20]  # Limit to top 20 elements
            return self.ui_elements
            
        except Exception as e:
            logger.error(f"UI element detection failed: {e}")
            return []
    
    def _filter_sensitive_content(self, text: str) -> str:
        """Filter out sensitive content from text"""
        if not text:
            return text
        
        filtered_text = text
        for keyword in self.config.sensitive_keywords:
            if keyword.lower() in text.lower():
                # Replace sensitive content with placeholder
                filtered_text = filtered_text.replace(keyword, "[FILTERED]")
                self.privacy_filtered = True
        
        return filtered_text
    
    def generate_context_summary(self) -> str:
        """Generate a summary of the current screen context"""
        summary_parts = []
        
        if self.text_content:
            # Extract key phrases from text content
            words = self.text_content.split()
            if len(words) > 50:
                summary_parts.append(f"Text content: {' '.join(words[:50])}...")
            else:
                summary_parts.append(f"Text content: {self.text_content}")
        
        if self.ui_elements:
            element_count = len(self.ui_elements)
            summary_parts.append(f"UI elements detected: {element_count}")
        
        if self.privacy_filtered:
            summary_parts.append("Content filtered for privacy")
        
        self.context_summary = "; ".join(summary_parts)
        return self.context_summary


class ScreenMonitor:
    """Screen monitoring and analysis system"""
    
    def __init__(self, privacy_manager: PrivacySecurityManager):
        self.privacy_manager = privacy_manager
        self._monitoring_active: Dict[str, bool] = {}
        self._monitoring_threads: Dict[str, threading.Thread] = {}
        self._context_queues: Dict[str, queue.Queue] = {}
        self._user_configs: Dict[str, MonitorConfig] = {}
        self._current_contexts: Dict[str, ScreenContext] = {}
        self._context_callbacks: Dict[str, List[Callable]] = {}
        
        # Application detection patterns
        self._app_patterns = {
            ApplicationType.BROWSER: ["chrome", "firefox", "safari", "edge", "browser"],
            ApplicationType.IDE: ["vscode", "pycharm", "intellij", "sublime", "atom", "vim", "emacs"],
            ApplicationType.TERMINAL: ["terminal", "cmd", "powershell", "bash", "zsh"],
            ApplicationType.OFFICE: ["word", "excel", "powerpoint", "libreoffice", "pages", "numbers"],
            ApplicationType.COMMUNICATION: ["slack", "teams", "discord", "zoom", "skype", "telegram"],
            ApplicationType.MEDIA: ["vlc", "spotify", "youtube", "netflix", "media player"],
        }
        
        if not HAS_SCREEN_DEPS:
            logger.warning("Screen monitoring dependencies not available. Install opencv-python, pytesseract, and pillow.")
    
    async def start_monitoring(self, user_id: str, config: MonitorConfig) -> bool:
        """Start screen monitoring for a user"""
        if not HAS_SCREEN_DEPS:
            logger.error("Screen monitoring dependencies not available")
            return False
        
        # Check permissions
        has_permission = await self.privacy_manager.check_permission(user_id, PermissionType.SCREEN_MONITOR)
        if not has_permission:
            # Request permission
            granted = await self.privacy_manager.request_permission(user_id, PermissionType.SCREEN_MONITOR)
            if not granted:
                logger.warning(f"Screen monitoring permission denied for user {user_id}")
                return False
        
        # Check consent for screen content data
        consent_status = await self.privacy_manager.get_consent_status(user_id, DataCategory.SCREEN_CONTENT)
        if consent_status not in [ConsentStatus.GRANTED]:
            consent_status = await self.privacy_manager.request_consent(
                user_id, DataCategory.SCREEN_CONTENT, 
                "Screen monitoring for contextual assistance", 
                retention_days=30
            )
            if consent_status != ConsentStatus.GRANTED:
                logger.warning(f"Screen content consent not granted for user {user_id}")
                return False
        
        # Stop existing monitoring if active
        if user_id in self._monitoring_active and self._monitoring_active[user_id]:
            await self.stop_monitoring(user_id)
        
        # Store configuration
        self._user_configs[user_id] = config
        self._monitoring_active[user_id] = True
        self._context_queues[user_id] = queue.Queue(maxsize=100)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(user_id, config),
            daemon=True
        )
        monitor_thread.start()
        self._monitoring_threads[user_id] = monitor_thread
        
        logger.info(f"Screen monitoring started for user {user_id}")
        return True
    
    async def stop_monitoring(self, user_id: str) -> bool:
        """Stop screen monitoring for a user"""
        if user_id not in self._monitoring_active:
            return False
        
        # Stop monitoring
        self._monitoring_active[user_id] = False
        
        # Wait for thread to finish
        if user_id in self._monitoring_threads:
            thread = self._monitoring_threads[user_id]
            thread.join(timeout=5.0)  # Wait up to 5 seconds
            del self._monitoring_threads[user_id]
        
        # Clean up
        if user_id in self._context_queues:
            del self._context_queues[user_id]
        if user_id in self._current_contexts:
            del self._current_contexts[user_id]
        
        logger.info(f"Screen monitoring stopped for user {user_id}")
        return True
    
    async def get_current_context(self, user_id: str) -> Optional[ScreenContext]:
        """Get the current screen context for a user"""
        return self._current_contexts.get(user_id)
    
    async def analyze_screen_content(self, screenshot: np.ndarray, config: MonitorConfig) -> ScreenAnalysis:
        """Analyze screen content and extract information"""
        analysis = ScreenAnalysis(screenshot, config)
        
        # Extract text content
        analysis.extract_text()
        
        # Detect UI elements
        analysis.detect_ui_elements()
        
        # Generate context summary
        analysis.generate_context_summary()
        
        return analysis
    
    def register_context_callback(self, user_id: str, callback: Callable[[ScreenContext], None]) -> None:
        """Register a callback for context updates"""
        if user_id not in self._context_callbacks:
            self._context_callbacks[user_id] = []
        self._context_callbacks[user_id].append(callback)
    
    def _monitoring_loop(self, user_id: str, config: MonitorConfig) -> None:
        """Main monitoring loop running in a separate thread"""
        logger.info(f"Starting monitoring loop for user {user_id}")
        
        while self._monitoring_active.get(user_id, False):
            try:
                # Capture screenshot
                screenshot = self._capture_screenshot()
                if screenshot is None:
                    time.sleep(config.capture_interval)
                    continue
                
                # Get active application info
                app_info = self._get_active_application()
                
                # Check if we should monitor this application
                if not self._should_monitor_application(app_info["name"], config):
                    time.sleep(config.capture_interval)
                    continue
                
                # Analyze screen content
                analysis = asyncio.run(self.analyze_screen_content(screenshot, config))
                
                # Create screen context
                context = ScreenContext(
                    active_application=app_info["name"],
                    window_title=app_info["title"],
                    visible_text=analysis.text_content,
                    ui_elements=analysis.ui_elements,
                    detected_actions=[],  # Could be enhanced with action detection
                    context_summary=analysis.context_summary,
                    timestamp=datetime.now(),
                    application_type=self._classify_application(app_info["name"]),
                    privacy_filtered=analysis.privacy_filtered
                )
                
                # Store current context
                self._current_contexts[user_id] = context
                
                # Store context data securely
                asyncio.run(self._store_context_data(user_id, context))
                
                # Notify callbacks
                self._notify_context_callbacks(user_id, context)
                
                # Add to queue for processing
                try:
                    self._context_queues[user_id].put_nowait(context)
                except queue.Full:
                    # Remove oldest item and add new one
                    try:
                        self._context_queues[user_id].get_nowait()
                        self._context_queues[user_id].put_nowait(context)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                logger.error(f"Error in monitoring loop for user {user_id}: {e}")
            
            time.sleep(config.capture_interval)
        
        logger.info(f"Monitoring loop ended for user {user_id}")
    
    def _capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture a screenshot of the current screen"""
        try:
            # Use PIL to capture screenshot
            screenshot = ImageGrab.grab()
            
            # Convert to numpy array for OpenCV
            screenshot_np = np.array(screenshot)
            screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            return screenshot_bgr
            
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    def _get_active_application(self) -> Dict[str, str]:
        """Get information about the currently active application"""
        try:
            # This is a simplified implementation
            # In a real system, you'd use platform-specific APIs
            
            # Get the current process list and find the most likely active app
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] > 0:
                        processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage and get the most active process
            if processes:
                active_proc = max(processes, key=lambda p: p['cpu_percent'])
                return {
                    "name": active_proc['name'],
                    "title": active_proc['name'],  # Simplified - would need window title API
                    "pid": active_proc['pid']
                }
            
            return {"name": "unknown", "title": "unknown", "pid": 0}
            
        except Exception as e:
            logger.error(f"Failed to get active application: {e}")
            return {"name": "unknown", "title": "unknown", "pid": 0}
    
    def _should_monitor_application(self, app_name: str, config: MonitorConfig) -> bool:
        """Check if an application should be monitored based on configuration"""
        app_name_lower = app_name.lower()
        
        if config.mode == MonitoringMode.DISABLED:
            return False
        
        if config.mode == MonitoringMode.SELECTIVE:
            # Only monitor included applications
            if config.included_applications:
                return any(included.lower() in app_name_lower for included in config.included_applications)
            return False
        
        # For FULL and PRIVACY modes, monitor unless excluded
        if config.excluded_applications:
            return not any(excluded.lower() in app_name_lower for excluded in config.excluded_applications)
        
        return True
    
    def _classify_application(self, app_name: str) -> ApplicationType:
        """Classify application type based on name"""
        app_name_lower = app_name.lower()
        
        for app_type, patterns in self._app_patterns.items():
            if any(pattern in app_name_lower for pattern in patterns):
                return app_type
        
        return ApplicationType.UNKNOWN
    
    async def _store_context_data(self, user_id: str, context: ScreenContext) -> None:
        """Store screen context data securely"""
        try:
            # Create data key for this context
            data_key = f"screen_context_{context.timestamp.isoformat()}"
            
            # Prepare context data for storage
            context_data = {
                "active_application": context.active_application,
                "window_title": context.window_title,
                "visible_text": context.visible_text[:1000],  # Limit stored text
                "context_summary": context.context_summary,
                "timestamp": context.timestamp.isoformat(),
                "application_type": context.application_type.value,
                "privacy_filtered": context.privacy_filtered,
                "ui_element_count": len(context.ui_elements)
            }
            
            # Store encrypted context data
            await self.privacy_manager.encrypt_personal_data(
                user_id, data_key, context_data, DataCategory.SCREEN_CONTENT
            )
            
        except Exception as e:
            logger.error(f"Failed to store context data for user {user_id}: {e}")
    
    def _notify_context_callbacks(self, user_id: str, context: ScreenContext) -> None:
        """Notify registered callbacks about context updates"""
        if user_id in self._context_callbacks:
            for callback in self._context_callbacks[user_id]:
                try:
                    callback(context)
                except Exception as e:
                    logger.error(f"Context callback failed: {e}")
    
    async def get_context_history(self, user_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get screen context history for a user"""
        try:
            # This would retrieve stored context data from the privacy manager
            # For now, return empty list as the storage implementation would need
            # to be extended to support querying by time range
            return []
            
        except Exception as e:
            logger.error(f"Failed to get context history for user {user_id}: {e}")
            return []
    
    async def update_monitoring_config(self, user_id: str, config: MonitorConfig) -> bool:
        """Update monitoring configuration for a user"""
        if user_id not in self._monitoring_active:
            return False
        
        # Update stored configuration
        self._user_configs[user_id] = config
        
        # If monitoring is active, restart with new config
        if self._monitoring_active[user_id]:
            await self.stop_monitoring(user_id)
            return await self.start_monitoring(user_id, config)
        
        return True
    
    async def get_monitoring_status(self, user_id: str) -> Dict[str, Any]:
        """Get current monitoring status for a user"""
        return {
            "active": self._monitoring_active.get(user_id, False),
            "config": self._user_configs.get(user_id).__dict__ if user_id in self._user_configs else None,
            "current_context": self._current_contexts.get(user_id).__dict__ if user_id in self._current_contexts else None,
            "has_permission": await self.privacy_manager.check_permission(user_id, PermissionType.SCREEN_MONITOR),
            "consent_status": (await self.privacy_manager.get_consent_status(user_id, DataCategory.SCREEN_CONTENT)).value
        }