"""
Action module for the AI agent.
Handles mouse control, keyboard control, and text-to-speech output.
"""
import threading
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MouseController:
    """Controls mouse movements and clicks."""
    
    def __init__(self):
        self.enabled = True
    
    def move_to(self, x: int, y: int):
        """Move mouse to specified coordinates."""
        if not self.enabled:
            return
        
        try:
            import pyautogui
            pyautogui.moveTo(x, y, duration=0.5)
            logger.info(f"Mouse moved to ({x}, {y})")
        except Exception as e:
            logger.error(f"Error moving mouse: {e}")
    
    def click(self, button: str = 'left', clicks: int = 1):
        """Click mouse button."""
        if not self.enabled:
            return
        
        try:
            import pyautogui
            pyautogui.click(button=button, clicks=clicks)
            logger.info(f"Mouse {button} clicked {clicks} time(s)")
        except Exception as e:
            logger.error(f"Error clicking mouse: {e}")
    
    def scroll(self, amount: int):
        """Scroll mouse wheel."""
        if not self.enabled:
            return
        
        try:
            import pyautogui
            pyautogui.scroll(amount)
            logger.info(f"Mouse scrolled by {amount}")
        except Exception as e:
            logger.error(f"Error scrolling mouse: {e}")
    
    def get_position(self) -> tuple:
        """Get current mouse position."""
        try:
            import pyautogui
            return pyautogui.position()
        except Exception as e:
            logger.error(f"Error getting mouse position: {e}")
            return (0, 0)


class KeyboardController:
    """Controls keyboard input."""
    
    def __init__(self):
        self.enabled = True
    
    def press(self, key: str):
        """Press a single key."""
        if not self.enabled:
            return
        
        try:
            import pyautogui
            pyautogui.press(key)
            logger.info(f"Key pressed: {key}")
        except Exception as e:
            logger.error(f"Error pressing key: {e}")
    
    def write(self, text: str, interval: float = 0.1):
        """Type text."""
        if not self.enabled:
            return
        
        try:
            import pyautogui
            pyautogui.write(text, interval=interval)
            logger.info(f"Text typed: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error typing text: {e}")
    
    def hotkey(self, *keys):
        """Press multiple keys together (hotkey)."""
        if not self.enabled:
            return
        
        try:
            import pyautogui
            pyautogui.hotkey(*keys)
            logger.info(f"Hotkey pressed: {keys}")
        except Exception as e:
            logger.error(f"Error pressing hotkey: {e}")


class TextToSpeech:
    """Converts text to speech."""
    
    def __init__(self):
        self.engine = None
        self.enabled = True
        self.speaking = False
        self.lock = threading.Lock()
        self._initialize()
    
    def _initialize(self):
        """Initialize TTS engine."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # Configure voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find Russian voice
                ru_voice = None
                for voice in voices:
                    if 'ru' in voice.id.lower() or 'russian' in voice.name.lower():
                        ru_voice = voice
                        break
                
                if ru_voice:
                    self.engine.setProperty('voice', ru_voice.id)
                else:
                    self.engine.setProperty('voice', voices[0].id)
            
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.9)  # Volume
            
            logger.info("TTS engine initialized")
        except Exception as e:
            logger.warning(f"TTS not available: {e}")
            self.enabled = False
    
    def speak(self, text: str, block: bool = False):
        """Speak text."""
        if not self.enabled or not text:
            return
        
        def _speak_thread():
            with self.lock:
                self.speaking = True
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    logger.error(f"Error speaking: {e}")
                finally:
                    self.speaking = False
        
        if block:
            _speak_thread()
        else:
            thread = threading.Thread(target=_speak_thread, daemon=True)
            thread.start()
    
    def stop(self):
        """Stop current speech."""
        if self.engine and self.speaking:
            try:
                self.engine.stop()
                self.speaking = False
            except Exception as e:
                logger.error(f"Error stopping speech: {e}")


class ActionExecutor:
    """Main action executor combining all output methods."""
    
    def __init__(self, config):
        self.config = config
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.tts = TextToSpeech()
    
    def execute_action(self, action_type: str, **kwargs):
        """Execute an action based on type."""
        try:
            if action_type == 'move_mouse':
                self.mouse.move_to(kwargs.get('x', 0), kwargs.get('y', 0))
            
            elif action_type == 'click':
                self.mouse.click(
                    button=kwargs.get('button', 'left'),
                    clicks=kwargs.get('clicks', 1)
                )
            
            elif action_type == 'scroll':
                self.mouse.scroll(kwargs.get('amount', 0))
            
            elif action_type == 'press_key':
                self.keyboard.press(kwargs.get('key', ''))
            
            elif action_type == 'type_text':
                self.keyboard.write(
                    kwargs.get('text', ''),
                    interval=kwargs.get('interval', 0.1)
                )
            
            elif action_type == 'hotkey':
                keys = kwargs.get('keys', [])
                self.keyboard.hotkey(*keys)
            
            elif action_type == 'speak':
                self.tts.speak(
                    kwargs.get('text', ''),
                    block=kwargs.get('block', False)
                )
            
            else:
                logger.warning(f"Unknown action type: {action_type}")
        
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current action executor state."""
        return {
            'mouse_position': self.mouse.get_position(),
            'tts_enabled': self.tts.enabled,
            'tts_speaking': self.tts.speaking
        }
