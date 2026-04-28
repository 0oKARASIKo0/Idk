"""
Configuration settings for the Qwen AI Agent.
"""
import os
from pathlib import Path


class Config:
    """Configuration class for the AI agent."""
    
    # LLM Settings
    LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1')
    LLM_MODEL = os.getenv('LLM_MODEL', 'qwen2.5')
    LLM_API_KEY = os.getenv('LLM_API_KEY', 'not-needed')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Camera settings
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
    CAMERA_FRAME_WIDTH = int(os.getenv('CAMERA_FRAME_WIDTH', '640'))
    CAMERA_FRAME_HEIGHT = int(os.getenv('CAMERA_FRAME_HEIGHT', '480'))
    
    # Audio settings
    AUDIO_SAMPLE_RATE = int(os.getenv('AUDIO_SAMPLE_RATE', '16000'))
    AUDIO_CHUNK_SIZE = int(os.getenv('AUDIO_CHUNK_SIZE', '1024'))
    
    # Screen capture settings
    SCREEN_CAPTURE_INTERVAL = float(os.getenv('SCREEN_CAPTURE_INTERVAL', '2.0'))
    
    # Voice settings
    VOICE_ENABLED = os.getenv('VOICE_ENABLED', 'true').lower() == 'true'
    SPEECH_RECOGNITION_LANGUAGE = os.getenv('SPEECH_RECOGNITION_LANGUAGE', 'ru-RU')
    
    # Autonomous mode settings
    THINKING_INTERVAL = float(os.getenv('THINKING_INTERVAL', '5.0'))
    ACTION_COOLDOWN = float(os.getenv('ACTION_COOLDOWN', '3.0'))
    
    # File paths
    DATA_DIR = Path(os.getenv('DATA_DIR', Path.home() / '.qwen_agent'))
    CHAT_HISTORY_FILE = Path(os.getenv('CHAT_HISTORY_FILE', DATA_DIR / 'chat_history.json'))
    USER_INFO_FILE = Path(os.getenv('USER_INFO_FILE', 'не забыть.txt'))
    
    def __init__(self):
        """Initialize configuration and ensure data directory exists."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
