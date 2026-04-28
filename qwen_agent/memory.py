"""
Memory module for the AI agent.
Handles chat history, user information file, and conversation management.
"""
import json
import threading
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ChatHistory:
    """Manages chat/conversation history."""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.messages: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self._load()
    
    def _load(self):
        """Load history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.messages = data.get('messages', [])
                logger.info(f"Loaded {len(self.messages)} messages from history")
            else:
                logger.info("No existing chat history found")
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            self.messages = []
    
    def _save(self):
        """Save history to file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({'messages': self.messages}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the history."""
        with self.lock:
            message = {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            self.messages.append(message)
            
            # Auto-save after each message
            self._save()
            
            logger.debug(f"Added {role} message to history")
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages for context."""
        with self.lock:
            return self.messages[-count:] if len(self.messages) > count else self.messages
    
    def clear_history(self):
        """Clear all chat history."""
        with self.lock:
            self.messages = []
            self._save()
        logger.info("Chat history cleared")
    
    def export_to_file(self, output_file: Path):
        """Export chat history to a text file."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("ИСТОРИЯ ЧАТА\n")
                f.write(f"Экспортировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                for msg in self.messages:
                    timestamp = msg.get('timestamp', 'Unknown')
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    
                    role_name = "Пользователь" if role == 'user' else "Ассистент" if role == 'assistant' else role
                    f.write(f"[{timestamp}] {role_name}:\n{content}\n\n")
                    f.write("-" * 40 + "\n")
            
            logger.info(f"Chat history exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting chat history: {e}")
            return False
    
    def get_context_for_llm(self, max_messages: int = 20) -> List[Dict[str, str]]:
        """Get formatted context for LLM API calls."""
        with self.lock:
            messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
            
            formatted = []
            for msg in messages:
                formatted.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            return formatted


class UserInfoManager:
    """Manages the user information file (не забыть.txt)."""
    
    def __init__(self, info_file: Path):
        self.info_file = info_file
        self.content: str = ""
        self.last_modified: float = 0
        self.lock = threading.Lock()
        self._load()
    
    def _load(self):
        """Load user info from file."""
        try:
            if self.info_file.exists():
                with open(self.info_file, 'r', encoding='utf-8') as f:
                    self.content = f.read()
                self.last_modified = self.info_file.stat().st_mtime
                logger.info(f"Loaded user info from {self.info_file}")
            else:
                # Create default file
                self.content = self._get_default_content()
                self._save()
                logger.info(f"Created default user info file at {self.info_file}")
        except Exception as e:
            logger.error(f"Error loading user info: {e}")
            self.content = self._get_default_content()
    
    def _save(self):
        """Save user info to file."""
        try:
            with open(self.info_file, 'w', encoding='utf-8') as f:
                f.write(self.content)
            self.last_modified = time.time()
            logger.debug("User info saved")
        except Exception as e:
            logger.error(f"Error saving user info: {e}")
    
    def _get_default_content(self) -> str:
        """Get default content for user info file."""
        return """# Информация о пользователе
# Редактируйте этот файл для обновления информации о себе

## Личная информация
Имя: Пользователь
Возраст: 
Город: 

## Предпочтения
Любимые цвета: 
Хобби: 
Интересы: 

## Важные заметки
- Здесь можно писать важные напоминания
- Агент будет использовать эту информацию при общении
- Вы можете редактировать этот файл в любое время

## Текущие задачи
- Задача 1
- Задача 2

## Контакты
Email: 
Телефон: 

---
Последнее обновление: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def get_content(self) -> str:
        """Get current user info content."""
        # Check if file was modified externally
        try:
            if self.info_file.exists():
                current_mtime = self.info_file.stat().st_mtime
                if current_mtime > self.last_modified:
                    self._load()
        except Exception:
            pass
        
        with self.lock:
            return self.content
    
    def update_content(self, new_content: str):
        """Update user info content."""
        with self.lock:
            self.content = new_content
            self._save()
        logger.info("User info updated")
    
    def append_note(self, note: str):
        """Append a note to the user info file."""
        with self.lock:
            self.content += f"\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n{note}"
            self._save()
        logger.info("Note appended to user info")
    
    def reload(self):
        """Reload user info from file."""
        self._load()


class Memory:
    """Main memory module combining chat history and user info."""
    
    def __init__(self, config):
        self.config = config
        self.chat_history = ChatHistory(config.CHAT_HISTORY_FILE)
        self.user_info = UserInfoManager(config.USER_INFO_FILE)
    
    def add_user_message(self, content: str, metadata: Optional[Dict] = None):
        """Add a user message to chat history."""
        self.chat_history.add_message('user', content, metadata)
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None):
        """Add an assistant message to chat history."""
        self.chat_history.add_message('assistant', content, metadata)
    
    def add_system_message(self, content: str, metadata: Optional[Dict] = None):
        """Add a system message to chat history."""
        self.chat_history.add_message('system', content, metadata)
    
    def get_context(self, max_messages: int = 20) -> str:
        """Get full context including chat history and user info."""
        chat_messages = self.chat_history.get_recent_messages(max_messages)
        user_info = self.user_info.get_content()
        
        context_parts = []
        
        # Add user info context
        context_parts.append("=== ИНФОРМАЦИЯ О ПОЛЬЗОВАТЕЛЕ ===")
        context_parts.append(user_info)
        context_parts.append("")
        
        # Add recent chat
        context_parts.append("=== ПОСЛЕДНИЕ СООБЩЕНИЯ ===")
        for msg in chat_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def get_llm_messages(self, max_messages: int = 20) -> List[Dict[str, str]]:
        """Get formatted messages for LLM API."""
        return self.chat_history.get_context_for_llm(max_messages)
    
    def export_chat(self, output_file: Optional[Path] = None) -> bool:
        """Export chat history to file."""
        if output_file is None:
            output_file = self.config.DATA_DIR / f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        return self.chat_history.export_to_file(output_file)
    
    def clear_chat(self):
        """Clear chat history."""
        self.chat_history.clear_history()
    
    def save_checkpoint(self):
        """Save a checkpoint of current state."""
        self.chat_history._save()
        self.user_info._save()
        logger.info("Memory checkpoint saved")
