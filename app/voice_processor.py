"""
Voice Processor Module

This module provides voice command processing capabilities including speech recognition,
text-to-speech synthesis, and natural language command parsing for the personal assistant.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time

try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    sr = None
    pyttsx3 = None

logger = logging.getLogger(__name__)


class VoiceCommand:
    """Represents a parsed voice command"""
    def __init__(self, text: str, intent: str, confidence: float, entities: Dict[str, Any] = None):
        self.text = text
        self.intent = intent
        self.confidence = confidence
        self.entities = entities or {}
        self.timestamp = time.time()


class VoiceSettings:
    """Voice processing settings"""
    def __init__(self):
        self.voice_type = "default"
        self.speech_speed = 1.0
        self.volume = 0.8
        self.language = "en-US"
        self.wake_word_enabled = False
        self.wake_word = "assistant"
        self.noise_threshold = 0.5
        self.timeout = 5.0
        self.phrase_timeout = 1.0


@dataclass
class VoiceResponse:
    """Voice response configuration"""
    text: str
    voice_type: Optional[str] = None
    speed: Optional[float] = None
    volume: Optional[float] = None
    priority: int = 0  # Higher priority responses interrupt lower priority ones


class VoiceProcessor:
    """
    Voice processing system for speech recognition and synthesis.
    
    Provides voice command processing, text-to-speech synthesis, and
    natural language command parsing capabilities.
    """
    
    def __init__(self, settings: VoiceSettings = None):
        self.settings = settings or VoiceSettings()
        self.is_listening = False
        self.is_speaking = False
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.listening_thread = None
        self.speaking_thread = None
        
        # Command parsing patterns
        self.command_patterns = self._initialize_command_patterns()
        
        # Initialize voice components if available
        if VOICE_AVAILABLE:
            self._initialize_voice_components()
        else:
            logger.warning("Voice processing libraries not available. Voice features will be disabled.")
    
    def _initialize_voice_components(self):
        """Initialize speech recognition and TTS components"""
        try:
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Initialize text-to-speech
            self.tts_engine = pyttsx3.init()
            self._configure_tts()
            
            logger.info("Voice components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice components: {e}")
            self.recognizer = None
            self.microphone = None
            self.tts_engine = None
    
    def _configure_tts(self):
        """Configure text-to-speech engine"""
        if not self.tts_engine:
            return
        
        try:
            # Set voice properties
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Try to find a suitable voice
                for voice in voices:
                    if self.settings.language.lower() in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Set speech rate and volume
            self.tts_engine.setProperty('rate', int(200 * self.settings.speech_speed))
            self.tts_engine.setProperty('volume', self.settings.volume)
            
        except Exception as e:
            logger.error(f"Failed to configure TTS engine: {e}")
    
    def _initialize_command_patterns(self) -> Dict[str, List[str]]:
        """Initialize command parsing patterns"""
        return {
            "file_operation": [
                r"(?:open|read|show me) (?:the )?file (.+)",
                r"(?:create|write|save) (?:a )?file (?:called )?(.+)",
                r"(?:list|show) (?:files|documents) (?:in )?(.+)?",
                r"(?:search|find) (?:for )?(.+) (?:in files|in documents)"
            ],
            "task_management": [
                r"(?:create|add|new) (?:a )?task (.+)",
                r"(?:show|list|display) (?:my )?tasks",
                r"(?:complete|finish|done) (?:task )?(.+)",
                r"(?:remind me|set reminder) (?:to )?(.+) (?:at|in|on) (.+)"
            ],
            "screen_monitoring": [
                r"(?:start|begin|enable) screen (?:monitoring|watching)",
                r"(?:stop|end|disable) screen (?:monitoring|watching)",
                r"(?:what|show) (?:am I|is) (?:currently )?(?:working on|doing)",
                r"(?:analyze|check) (?:current|this) (?:screen|window|application)"
            ],
            "knowledge_search": [
                r"(?:search|find|look for) (.+) (?:in my|from my) (?:knowledge|documents|notes)",
                r"(?:what do I know about|tell me about) (.+)",
                r"(?:remember|recall) (.+)",
                r"(?:add|save|remember) (?:this|that) (.+) (?:to|in) (?:my )?(?:knowledge|notes)"
            ],
            "general_query": [
                r"(?:help|assist) (?:me )?(?:with )?(.+)",
                r"(?:how do I|how can I) (.+)",
                r"(?:what is|what are|explain) (.+)",
                r"(?:can you|could you|please) (.+)"
            ],
            "context_control": [
                r"(?:switch to|change to|use) (.+) (?:mode|interface)",
                r"(?:enable|turn on|activate) (.+)",
                r"(?:disable|turn off|deactivate) (.+)",
                r"(?:set|change|update) (?:my )?(?:preferences|settings) (?:for )?(.+)"
            ]
        }
    
    async def start_listening(self, callback: Callable[[VoiceCommand], None] = None):
        """Start continuous voice listening"""
        if not VOICE_AVAILABLE or not self.recognizer or not self.microphone:
            logger.warning("Voice recognition not available")
            return False
        
        if self.is_listening:
            logger.info("Voice listening already active")
            return True
        
        self.is_listening = True
        self.listening_thread = threading.Thread(
            target=self._listening_loop,
            args=(callback,),
            daemon=True
        )
        self.listening_thread.start()
        
        logger.info("Voice listening started")
        return True
    
    async def stop_listening(self):
        """Stop voice listening"""
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join(timeout=2.0)
        logger.info("Voice listening stopped")
    
    def _listening_loop(self, callback: Callable[[VoiceCommand], None] = None):
        """Main listening loop (runs in separate thread)"""
        while self.is_listening:
            try:
                # Listen for audio with timeout
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.settings.timeout,
                        phrase_time_limit=self.settings.phrase_timeout
                    )
                
                # Process the audio
                command = self._process_audio(audio)
                if command:
                    if callback:
                        callback(command)
                    else:
                        self.command_queue.put(command)
                
            except sr.WaitTimeoutError:
                # Normal timeout, continue listening
                continue
            except sr.UnknownValueError:
                # Could not understand audio
                logger.debug("Could not understand audio")
                continue
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                time.sleep(1)  # Brief pause before retrying
            except Exception as e:
                logger.error(f"Unexpected error in listening loop: {e}")
                time.sleep(1)
    
    def _process_audio(self, audio) -> Optional[VoiceCommand]:
        """Process audio data and extract command"""
        try:
            # Convert speech to text
            text = self.recognizer.recognize_google(audio, language=self.settings.language)
            logger.info(f"Recognized speech: {text}")
            
            # Check for wake word if enabled
            if self.settings.wake_word_enabled:
                if self.settings.wake_word.lower() not in text.lower():
                    return None
                # Remove wake word from text
                text = text.lower().replace(self.settings.wake_word.lower(), "").strip()
            
            # Parse command
            return self._parse_command(text)
            
        except sr.UnknownValueError:
            logger.debug("Could not understand the audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Could not request results from speech recognition service: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None
    
    def _parse_command(self, text: str) -> VoiceCommand:
        """Parse text into a structured command"""
        import re
        
        text = text.lower().strip()
        
        # Try to match against command patterns
        for intent, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entities = {}
                    if match.groups():
                        entities = {f"param_{i}": group for i, group in enumerate(match.groups())}
                    
                    return VoiceCommand(
                        text=text,
                        intent=intent,
                        confidence=0.8,  # Simple confidence score
                        entities=entities
                    )
        
        # Default to general query if no specific pattern matches
        return VoiceCommand(
            text=text,
            intent="general_query",
            confidence=0.5,
            entities={"query": text}
        )
    
    async def process_voice_command(self, audio_data: bytes = None, text: str = None) -> VoiceCommand:
        """Process a single voice command from audio data or text"""
        if text:
            return self._parse_command(text)
        
        if audio_data and VOICE_AVAILABLE and self.recognizer:
            # Convert bytes to AudioData (simplified - would need proper implementation)
            # For now, return a mock command
            return VoiceCommand(
                text="mock command from audio",
                intent="general_query",
                confidence=0.5
            )
        
        raise ValueError("Either audio_data or text must be provided")
    
    async def generate_voice_response(self, response: VoiceResponse) -> bool:
        """Generate voice response using text-to-speech"""
        if not VOICE_AVAILABLE or not self.tts_engine:
            logger.warning("Text-to-speech not available")
            return False
        
        try:
            # Configure TTS for this response
            if response.voice_type:
                # Would set specific voice type
                pass
            
            if response.speed:
                rate = int(200 * response.speed)
                self.tts_engine.setProperty('rate', rate)
            
            if response.volume:
                self.tts_engine.setProperty('volume', response.volume)
            
            # Speak the text
            self.is_speaking = True
            self.tts_engine.say(response.text)
            self.tts_engine.runAndWait()
            self.is_speaking = False
            
            logger.info(f"Voice response generated: {response.text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error generating voice response: {e}")
            self.is_speaking = False
            return False
    
    async def speak_text(self, text: str, interrupt: bool = False) -> bool:
        """Simple text-to-speech method"""
        if interrupt and self.is_speaking:
            self.stop_speaking()
        
        response = VoiceResponse(text=text)
        return await self.generate_voice_response(response)
    
    def stop_speaking(self):
        """Stop current speech output"""
        if self.tts_engine and self.is_speaking:
            try:
                self.tts_engine.stop()
                self.is_speaking = False
            except Exception as e:
                logger.error(f"Error stopping speech: {e}")
    
    def get_next_command(self, timeout: float = None) -> Optional[VoiceCommand]:
        """Get the next command from the queue"""
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def is_voice_available(self) -> bool:
        """Check if voice processing is available"""
        return VOICE_AVAILABLE and self.recognizer is not None and self.tts_engine is not None
    
    def get_voice_status(self) -> Dict[str, Any]:
        """Get current voice processing status"""
        return {
            "available": self.is_voice_available(),
            "listening": self.is_listening,
            "speaking": self.is_speaking,
            "wake_word_enabled": self.settings.wake_word_enabled,
            "wake_word": self.settings.wake_word,
            "language": self.settings.language,
            "queue_size": self.command_queue.qsize()
        }
    
    def update_settings(self, new_settings: Dict[str, Any]):
        """Update voice processing settings"""
        for key, value in new_settings.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        
        # Reconfigure TTS if needed
        if self.tts_engine:
            self._configure_tts()
        
        logger.info(f"Voice settings updated: {new_settings}")
    
    async def cleanup(self):
        """Clean up voice processing resources"""
        await self.stop_listening()
        self.stop_speaking()
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        logger.info("Voice processor cleaned up")


class VoiceCommandParser:
    """Advanced voice command parser with natural language understanding"""
    
    def __init__(self):
        self.intent_keywords = {
            "file_operation": ["file", "document", "open", "read", "write", "save", "create"],
            "task_management": ["task", "todo", "reminder", "deadline", "project"],
            "screen_monitoring": ["screen", "monitor", "window", "application", "current"],
            "knowledge_search": ["search", "find", "know", "remember", "knowledge"],
            "general_query": ["help", "how", "what", "explain", "tell"],
            "context_control": ["mode", "switch", "enable", "disable", "settings"]
        }
    
    def parse_advanced(self, text: str) -> VoiceCommand:
        """Advanced command parsing with better intent detection"""
        text = text.lower().strip()
        
        # Calculate intent scores
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                intent_scores[intent] = score / len(keywords)
        
        # Get best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
        else:
            best_intent = "general_query"
            confidence = 0.3
        
        # Extract entities based on intent
        entities = self._extract_entities(text, best_intent)
        
        return VoiceCommand(
            text=text,
            intent=best_intent,
            confidence=confidence,
            entities=entities
        )
    
    def _extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract entities based on intent"""
        entities = {}
        
        if intent == "file_operation":
            # Extract file paths, operations
            import re
            file_match = re.search(r'(?:file|document)\s+(?:called\s+)?([^\s]+)', text)
            if file_match:
                entities["file_name"] = file_match.group(1)
        
        elif intent == "task_management":
            # Extract task descriptions, dates
            task_match = re.search(r'(?:task|reminder)\s+(.+?)(?:\s+(?:at|on|in)\s+(.+))?$', text)
            if task_match:
                entities["task_description"] = task_match.group(1)
                if task_match.group(2):
                    entities["due_date"] = task_match.group(2)
        
        # Add more entity extraction logic as needed
        
        return entities


# Mock implementations for when voice libraries are not available
class MockVoiceProcessor:
    """Mock voice processor for testing and fallback"""
    
    def __init__(self, settings: VoiceSettings = None):
        self.settings = settings or VoiceSettings()
        self.is_listening = False
        self.is_speaking = False
    
    async def start_listening(self, callback=None):
        self.is_listening = True
        logger.info("Mock voice listening started")
        return True
    
    async def stop_listening(self):
        self.is_listening = False
        logger.info("Mock voice listening stopped")
    
    async def process_voice_command(self, audio_data=None, text=None):
        if text:
            return VoiceCommand(text=text, intent="general_query", confidence=0.5)
        return VoiceCommand(text="mock command", intent="general_query", confidence=0.5)
    
    async def generate_voice_response(self, response):
        logger.info(f"Mock TTS: {response.text}")
        return True
    
    async def speak_text(self, text, interrupt=False):
        logger.info(f"Mock speak: {text}")
        return True
    
    def is_voice_available(self):
        return False
    
    def get_voice_status(self):
        return {
            "available": False,
            "listening": self.is_listening,
            "speaking": self.is_speaking,
            "mock": True
        }
    
    async def cleanup(self):
        pass