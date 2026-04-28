"""
Qwen AI Agent Package
Autonomous AI agent with continuous thinking, mouse/keyboard control,
voice commands, camera vision, and text-to-speech.
"""

__version__ = "1.0.0"

from .config import Config

# Singleton config instance
config = Config()
