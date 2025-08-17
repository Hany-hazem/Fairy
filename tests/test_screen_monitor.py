"""
Unit tests for the Screen Monitor module
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import threading
import queue
import time

from app.screen_monitor import (
    ScreenMonitor, ScreenAnalysis, ScreenContext, MonitorConfig, 
    MonitoringMode, ApplicationType, UIElement
)
from app.privacy_security_manager import PrivacySecurityManager, PermissionType, DataCategory, ConsentStatus
from app.personal_assistant_models import UserContext


class TestScreenAnalysis:
    """Test ScreenAnalysis functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = MonitorConfig(
            mode=MonitoringMode.FULL,
            ocr_enabled=True,
            privacy_filtering=True
        )
        # Create a simple test image (100x100 white image)
        self.test_screenshot = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_screen_analysis_initialization(self):
        """Test ScreenAnalysis initialization"""
        analysis = ScreenAnalysis(self.test_screenshot, self.config)
        
        assert analysis.screenshot is not None
        assert analysis.config == self.config
        assert analysis.text_content == ""
        assert analysis.ui_elements == []
        assert analysis.context_summary == ""
        assert not analysis.privacy_filtered
        assert isinstance(analysis.analysis_timestamp, datetime)
    
    @patch('app.screen_monitor.pytesseract')
    @patch('app.screen_monitor.Image')
    @patch('app.screen_monitor.cv2')
    @patch('app.screen_monitor.HAS_SCREEN_DEPS', True)
    def test_extract_text_success(self, mock_cv2, mock_image, mock_pytesseract):
        """Test successful text extraction"""
        mock_pytesseract.image_to_string.return_value = "Test text content"
        mock_cv2.cvtColor.return_value = self.test_screenshot
        mock_cv2.COLOR_BGR2RGB = 4
        mock_pil_image = Mock()
        mock_image.fromarray.return_value = mock_pil_image
        
        analysis = ScreenAnalysis(self.test_screenshot, self.config)
        result = analysis.extract_text()
        
        assert result == "Test text content"
        assert analysis.text_content == "Test text content"
        mock_pytesseract.image_to_string.assert_called_once()
    
    @patch('app.screen_monitor.pytesseract')
    @patch('app.screen_monitor.Image')
    @patch('app.screen_monitor.cv2')
    @patch('app.screen_monitor.HAS_SCREEN_DEPS', True)
    def test_extract_text_with_privacy_filtering(self, mock_cv2, mock_image, mock_pytesseract):
        """Test text extraction with privacy filtering"""
        mock_pytesseract.image_to_string.return_value = "Enter your password: secret123"
        mock_cv2.cvtColor.return_value = self.test_screenshot
        mock_cv2.COLOR_BGR2RGB = 4
        mock_pil_image = Mock()
        mock_image.fromarray.return_value = mock_pil_image
        
        analysis = ScreenAnalysis(self.test_screenshot, self.config)
        result = analysis.extract_text()
        
        assert "[FILTERED]" in result
        assert "secret123" in result  # Only keyword is filtered, not the actual password
        assert analysis.privacy_filtered
    
    @patch('app.screen_monitor.HAS_SCREEN_DEPS', False)
    def test_extract_text_no_dependencies(self):
        """Test text extraction when dependencies are not available"""
        analysis = ScreenAnalysis(self.test_screenshot, self.config)
        result = analysis.extract_text()
        
        assert result == ""
        assert analysis.text_content == ""
    
    def test_extract_text_ocr_disabled(self):
        """Test text extraction when OCR is disabled"""
        config = MonitorConfig(ocr_enabled=False)
        analysis = ScreenAnalysis(self.test_screenshot, config)
        result = analysis.extract_text()
        
        assert result == ""
        assert analysis.text_content == ""
    
    @patch('app.screen_monitor.pytesseract')
    @patch('app.screen_monitor.Image')
    @patch('app.screen_monitor.cv2')
    @patch('app.screen_monitor.HAS_SCREEN_DEPS', True)
    def test_extract_text_length_limit(self, mock_cv2, mock_image, mock_pytesseract):
        """Test text extraction with length limit"""
        long_text = "A" * 15000  # Longer than default max_text_length
        mock_pytesseract.image_to_string.return_value = long_text
        mock_cv2.cvtColor.return_value = self.test_screenshot
        mock_cv2.COLOR_BGR2RGB = 4
        mock_pil_image = Mock()
        mock_image.fromarray.return_value = mock_pil_image
        
        analysis = ScreenAnalysis(self.test_screenshot, self.config)
        result = analysis.extract_text()
        
        assert len(result) <= self.config.max_text_length + 3  # +3 for "..."
        assert result.endswith("...")
    
    @patch('app.screen_monitor.cv2')
    @patch('app.screen_monitor.pytesseract')
    @patch('app.screen_monitor.Image')
    @patch('app.screen_monitor.HAS_SCREEN_DEPS', True)
    def test_detect_ui_elements(self, mock_image, mock_pytesseract, mock_cv2):
        """Test UI element detection"""
        # Mock OpenCV functions
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.Canny.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.findContours.return_value = ([np.array([[10, 10], [50, 10], [50, 50], [10, 50]])], None)
        mock_cv2.boundingRect.return_value = (10, 10, 40, 40)
        mock_cv2.COLOR_BGR2GRAY = 6
        mock_cv2.COLOR_BGR2RGB = 4
        mock_cv2.RETR_EXTERNAL = 0
        mock_cv2.CHAIN_APPROX_SIMPLE = 2
        
        mock_pytesseract.image_to_string.return_value = "Button text"
        mock_pil_image = Mock()
        mock_image.fromarray.return_value = mock_pil_image
        
        analysis = ScreenAnalysis(self.test_screenshot, self.config)
        elements = analysis.detect_ui_elements()
        
        assert len(elements) > 0
        assert isinstance(elements[0], UIElement)
        assert elements[0].text == "Button text"
        assert elements[0].position == (10, 10, 40, 40)
    
    def test_filter_sensitive_content(self):
        """Test sensitive content filtering"""
        analysis = ScreenAnalysis(self.test_screenshot, self.config)
        
        # Test filtering
        filtered = analysis._filter_sensitive_content("Enter your password here")
        assert "[FILTERED]" in filtered
        assert analysis.privacy_filtered
        
        # Test no filtering needed
        analysis.privacy_filtered = False
        normal = analysis._filter_sensitive_content("Normal text content")
        assert normal == "Normal text content"
        assert not analysis.privacy_filtered
    
    def test_generate_context_summary(self):
        """Test context summary generation"""
        analysis = ScreenAnalysis(self.test_screenshot, self.config)
        analysis.text_content = "Sample text content"
        analysis.ui_elements = [UIElement("button", "Click me", (0, 0, 50, 20), 0.9)]
        analysis.privacy_filtered = True
        
        summary = analysis.generate_context_summary()
        
        assert "Sample text content" in summary
        assert "UI elements detected: 1" in summary
        assert "Content filtered for privacy" in summary
        assert analysis.context_summary == summary


class TestMonitorConfig:
    """Test MonitorConfig functionality"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MonitorConfig()
        
        assert config.mode == MonitoringMode.DISABLED
        assert config.capture_interval == 2.0
        assert config.ocr_enabled
        assert config.application_detection
        assert config.privacy_filtering
        assert len(config.sensitive_keywords) > 0
        assert config.max_text_length == 10000
        assert not config.store_screenshots
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MonitorConfig(
            mode=MonitoringMode.SELECTIVE,
            capture_interval=1.0,
            ocr_enabled=False,
            excluded_applications={"notepad", "calculator"}
        )
        
        assert config.mode == MonitoringMode.SELECTIVE
        assert config.capture_interval == 1.0
        assert not config.ocr_enabled
        assert "notepad" in config.excluded_applications


class TestScreenMonitor:
    """Test ScreenMonitor functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.privacy_manager = Mock(spec=PrivacySecurityManager)
        self.screen_monitor = ScreenMonitor(self.privacy_manager)
        self.user_id = "test_user_123"
        self.config = MonitorConfig(mode=MonitoringMode.FULL)
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test ScreenMonitor initialization"""
        assert self.screen_monitor.privacy_manager == self.privacy_manager
        assert self.screen_monitor._monitoring_active == {}
        assert self.screen_monitor._monitoring_threads == {}
        assert self.screen_monitor._context_queues == {}
        assert self.screen_monitor._user_configs == {}
        assert self.screen_monitor._current_contexts == {}
    
    @pytest.mark.asyncio
    @patch('app.screen_monitor.HAS_SCREEN_DEPS', False)
    async def test_start_monitoring_no_dependencies(self):
        """Test starting monitoring without dependencies"""
        result = await self.screen_monitor.start_monitoring(self.user_id, self.config)
        
        assert not result
        assert self.user_id not in self.screen_monitor._monitoring_active
    
    @pytest.mark.asyncio
    @patch('app.screen_monitor.HAS_SCREEN_DEPS', True)
    async def test_start_monitoring_no_permission(self):
        """Test starting monitoring without permission"""
        self.privacy_manager.check_permission = AsyncMock(return_value=False)
        self.privacy_manager.request_permission = AsyncMock(return_value=False)
        
        result = await self.screen_monitor.start_monitoring(self.user_id, self.config)
        
        assert not result
        self.privacy_manager.check_permission.assert_called_once_with(
            self.user_id, PermissionType.SCREEN_MONITOR
        )
        self.privacy_manager.request_permission.assert_called_once_with(
            self.user_id, PermissionType.SCREEN_MONITOR
        )
    
    @pytest.mark.asyncio
    @patch('app.screen_monitor.HAS_SCREEN_DEPS', True)
    async def test_start_monitoring_no_consent(self):
        """Test starting monitoring without consent"""
        self.privacy_manager.check_permission = AsyncMock(return_value=True)
        self.privacy_manager.get_consent_status = AsyncMock(return_value=ConsentStatus.DENIED)
        self.privacy_manager.request_consent = AsyncMock(return_value=ConsentStatus.DENIED)
        
        result = await self.screen_monitor.start_monitoring(self.user_id, self.config)
        
        assert not result
        self.privacy_manager.get_consent_status.assert_called_once_with(
            self.user_id, DataCategory.SCREEN_CONTENT
        )
    
    @pytest.mark.asyncio
    @patch('app.screen_monitor.HAS_SCREEN_DEPS', True)
    @patch('threading.Thread')
    async def test_start_monitoring_success(self, mock_thread):
        """Test successful monitoring start"""
        self.privacy_manager.check_permission = AsyncMock(return_value=True)
        self.privacy_manager.get_consent_status = AsyncMock(return_value=ConsentStatus.GRANTED)
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        result = await self.screen_monitor.start_monitoring(self.user_id, self.config)
        
        assert result
        assert self.screen_monitor._monitoring_active[self.user_id]
        assert self.user_id in self.screen_monitor._user_configs
        assert self.user_id in self.screen_monitor._context_queues
        assert self.user_id in self.screen_monitor._monitoring_threads
        
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_monitoring_not_active(self):
        """Test stopping monitoring when not active"""
        result = await self.screen_monitor.stop_monitoring(self.user_id)
        assert not result
    
    @pytest.mark.asyncio
    async def test_stop_monitoring_success(self):
        """Test successful monitoring stop"""
        # Set up active monitoring
        self.screen_monitor._monitoring_active[self.user_id] = True
        mock_thread = Mock()
        mock_thread.join = Mock()
        self.screen_monitor._monitoring_threads[self.user_id] = mock_thread
        self.screen_monitor._context_queues[self.user_id] = queue.Queue()
        self.screen_monitor._current_contexts[self.user_id] = Mock()
        
        result = await self.screen_monitor.stop_monitoring(self.user_id)
        
        assert result
        assert not self.screen_monitor._monitoring_active[self.user_id]
        assert self.user_id not in self.screen_monitor._monitoring_threads
        assert self.user_id not in self.screen_monitor._context_queues
        assert self.user_id not in self.screen_monitor._current_contexts
        
        mock_thread.join.assert_called_once_with(timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_get_current_context(self):
        """Test getting current context"""
        mock_context = Mock(spec=ScreenContext)
        self.screen_monitor._current_contexts[self.user_id] = mock_context
        
        result = await self.screen_monitor.get_current_context(self.user_id)
        assert result == mock_context
        
        # Test non-existent user
        result = await self.screen_monitor.get_current_context("non_existent")
        assert result is None
    
    @pytest.mark.asyncio
    @patch('app.screen_monitor.HAS_SCREEN_DEPS', True)
    async def test_analyze_screen_content(self):
        """Test screen content analysis"""
        screenshot = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        with patch.object(ScreenAnalysis, 'extract_text', return_value="Test text"):
            with patch.object(ScreenAnalysis, 'detect_ui_elements', return_value=[]):
                with patch.object(ScreenAnalysis, 'generate_context_summary', return_value="Summary"):
                    analysis = await self.screen_monitor.analyze_screen_content(screenshot, self.config)
                    
                    assert isinstance(analysis, ScreenAnalysis)
                    assert analysis.screenshot is not None
                    assert analysis.config == self.config
    
    def test_register_context_callback(self):
        """Test registering context callbacks"""
        callback = Mock()
        
        self.screen_monitor.register_context_callback(self.user_id, callback)
        
        assert self.user_id in self.screen_monitor._context_callbacks
        assert callback in self.screen_monitor._context_callbacks[self.user_id]
    
    @patch('app.screen_monitor.ImageGrab')
    @patch('app.screen_monitor.cv2')
    def test_capture_screenshot(self, mock_cv2, mock_imagegrab):
        """Test screenshot capture"""
        mock_image = Mock()
        mock_imagegrab.grab.return_value = mock_image
        mock_cv2.cvtColor.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        mock_cv2.COLOR_RGB2BGR = 4
        
        with patch('numpy.array', return_value=np.ones((100, 100, 3), dtype=np.uint8)):
            result = self.screen_monitor._capture_screenshot()
            
            assert result is not None
            mock_imagegrab.grab.assert_called_once()
            mock_cv2.cvtColor.assert_called_once()
    
    @patch('app.screen_monitor.psutil')
    def test_get_active_application(self, mock_psutil):
        """Test getting active application info"""
        mock_proc = Mock()
        mock_proc.info = {'pid': 1234, 'name': 'test_app', 'cpu_percent': 5.0}
        mock_psutil.process_iter.return_value = [mock_proc]
        
        result = self.screen_monitor._get_active_application()
        
        assert result['name'] == 'test_app'
        assert result['pid'] == 1234
        assert 'title' in result
    
    def test_should_monitor_application(self):
        """Test application monitoring decision logic"""
        # Test disabled mode
        config = MonitorConfig(mode=MonitoringMode.DISABLED)
        assert not self.screen_monitor._should_monitor_application("any_app", config)
        
        # Test selective mode with included apps
        config = MonitorConfig(
            mode=MonitoringMode.SELECTIVE,
            included_applications={"vscode", "chrome"}
        )
        assert self.screen_monitor._should_monitor_application("vscode", config)
        assert not self.screen_monitor._should_monitor_application("notepad", config)
        
        # Test full mode with excluded apps
        config = MonitorConfig(
            mode=MonitoringMode.FULL,
            excluded_applications={"notepad"}
        )
        assert self.screen_monitor._should_monitor_application("vscode", config)
        assert not self.screen_monitor._should_monitor_application("notepad", config)
    
    def test_classify_application(self):
        """Test application classification"""
        assert self.screen_monitor._classify_application("chrome") == ApplicationType.BROWSER
        assert self.screen_monitor._classify_application("vscode") == ApplicationType.IDE
        assert self.screen_monitor._classify_application("terminal") == ApplicationType.TERMINAL
        assert self.screen_monitor._classify_application("unknown_app") == ApplicationType.UNKNOWN
    
    @pytest.mark.asyncio
    async def test_store_context_data(self):
        """Test storing context data"""
        context = ScreenContext(
            active_application="test_app",
            window_title="Test Window",
            visible_text="Test text",
            ui_elements=[],
            detected_actions=[],
            context_summary="Test summary",
            timestamp=datetime.now(),
            application_type=ApplicationType.UNKNOWN
        )
        
        self.privacy_manager.encrypt_personal_data = AsyncMock(return_value=True)
        
        await self.screen_monitor._store_context_data(self.user_id, context)
        
        self.privacy_manager.encrypt_personal_data.assert_called_once()
        call_args = self.privacy_manager.encrypt_personal_data.call_args
        assert call_args[0][0] == self.user_id  # user_id
        assert call_args[0][2]['active_application'] == "test_app"  # context data
        assert call_args[0][3] == DataCategory.SCREEN_CONTENT  # data category
    
    def test_notify_context_callbacks(self):
        """Test notifying context callbacks"""
        callback1 = Mock()
        callback2 = Mock()
        context = Mock(spec=ScreenContext)
        
        self.screen_monitor._context_callbacks[self.user_id] = [callback1, callback2]
        
        self.screen_monitor._notify_context_callbacks(self.user_id, context)
        
        callback1.assert_called_once_with(context)
        callback2.assert_called_once_with(context)
    
    def test_notify_context_callbacks_with_error(self):
        """Test context callback notification with error handling"""
        callback_error = Mock(side_effect=Exception("Callback error"))
        callback_success = Mock()
        context = Mock(spec=ScreenContext)
        
        self.screen_monitor._context_callbacks[self.user_id] = [callback_error, callback_success]
        
        # Should not raise exception
        self.screen_monitor._notify_context_callbacks(self.user_id, context)
        
        callback_error.assert_called_once_with(context)
        callback_success.assert_called_once_with(context)
    
    @pytest.mark.asyncio
    async def test_get_context_history(self):
        """Test getting context history"""
        result = await self.screen_monitor.get_context_history(self.user_id, hours=24)
        
        # Currently returns empty list as storage implementation is simplified
        assert result == []
    
    @pytest.mark.asyncio
    async def test_update_monitoring_config(self):
        """Test updating monitoring configuration"""
        # Test non-existent user
        result = await self.screen_monitor.update_monitoring_config(self.user_id, self.config)
        assert not result
        
        # Test existing user
        self.screen_monitor._monitoring_active[self.user_id] = False
        result = await self.screen_monitor.update_monitoring_config(self.user_id, self.config)
        assert result
        assert self.screen_monitor._user_configs[self.user_id] == self.config
    
    @pytest.mark.asyncio
    async def test_get_monitoring_status(self):
        """Test getting monitoring status"""
        self.privacy_manager.check_permission = AsyncMock(return_value=True)
        self.privacy_manager.get_consent_status = AsyncMock(return_value=ConsentStatus.GRANTED)
        
        # Set up some test data
        self.screen_monitor._monitoring_active[self.user_id] = True
        self.screen_monitor._user_configs[self.user_id] = self.config
        
        status = await self.screen_monitor.get_monitoring_status(self.user_id)
        
        assert status['active']
        assert status['config'] is not None
        assert status['has_permission']
        assert status['consent_status'] == ConsentStatus.GRANTED.value


class TestScreenContext:
    """Test ScreenContext data model"""
    
    def test_screen_context_creation(self):
        """Test ScreenContext creation"""
        timestamp = datetime.now()
        ui_element = UIElement("button", "Click me", (0, 0, 50, 20), 0.9)
        
        context = ScreenContext(
            active_application="test_app",
            window_title="Test Window",
            visible_text="Test text",
            ui_elements=[ui_element],
            detected_actions=["click", "scroll"],
            context_summary="Test summary",
            timestamp=timestamp,
            application_type=ApplicationType.BROWSER,
            privacy_filtered=True
        )
        
        assert context.active_application == "test_app"
        assert context.window_title == "Test Window"
        assert context.visible_text == "Test text"
        assert len(context.ui_elements) == 1
        assert context.ui_elements[0] == ui_element
        assert context.detected_actions == ["click", "scroll"]
        assert context.context_summary == "Test summary"
        assert context.timestamp == timestamp
        assert context.application_type == ApplicationType.BROWSER
        assert context.privacy_filtered


class TestUIElement:
    """Test UIElement data model"""
    
    def test_ui_element_creation(self):
        """Test UIElement creation"""
        element = UIElement(
            element_type="button",
            text="Click me",
            position=(10, 20, 100, 30),
            confidence=0.95,
            metadata={"color": "blue", "enabled": True}
        )
        
        assert element.element_type == "button"
        assert element.text == "Click me"
        assert element.position == (10, 20, 100, 30)
        assert element.confidence == 0.95
        assert element.metadata["color"] == "blue"
        assert element.metadata["enabled"]


if __name__ == "__main__":
    pytest.main([__file__])