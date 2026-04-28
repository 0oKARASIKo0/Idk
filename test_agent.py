#!/usr/bin/env python3
"""
Тестовый скрипт для проверки AI агента.
Запускает агент в текстовом режиме.
"""
import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from qwen_agent import config
from qwen_agent.main import AIAgent

def main():
    print("\n" + "=" * 60)
    print("AI АГЕНТ - ТЕСТОВЫЙ ЗАПУСК")
    print("=" * 60)
    
    # Создаем агента
    agent = AIAgent(config)
    
    print(f"\n[+] Агент создан успешно")
    print(f"[+] LLM URL: {config.LLM_BASE_URL}")
    print(f"[+] Модель: {config.LLM_MODEL}")
    print(f"[+] TTS: {'Включен' if agent.actions.tts.enabled else 'Выключен'}")
    print(f"[+] Камера: индекс {config.CAMERA_INDEX}")
    print(f"[+] Микрофон: {config.VOICE_ENABLED}")
    print(f"[+] Файл инфо: {config.USER_INFO_FILE}")
    
    print("\n" + "=" * 60)
    print("Для запуска интерактивного режима используйте:")
    print("  python run_agent.py --mode interactive")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
