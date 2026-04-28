import os
import sys
import time
import threading
import json
from datetime import datetime
from pathlib import Path

# Проверка наличия ollama
try:
    import requests
except ImportError:
    print("Установите requests: pip install requests")
    sys.exit(1)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"  # Легкая модель для вашего ПК

class LocalAgent:
    def __init__(self):
        self.memory_file = Path("не забыть.txt")
        self.chat_history = []
        self.running = True
        self.last_action_time = 0
        self.action_interval = 10  # Секунды между проверками действий
        
        # Загрузка памяти
        if self.memory_file.exists():
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.user_info = f.read()
        else:
            self.user_info = "Пользователь: Карась. Компьютер: HP Compaq Elite 8300. ОС: Windows 10."
            self.save_memory()
            
        print(f"🤖 Локальный агент запущен (Модель: {MODEL_NAME})")
        print(f"💾 Память загружена из {self.memory_file}")
        print("Нажмите Ctrl+C для остановки")

    def save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(self.user_info)

    def edit_memory(self):
        print("\n--- Редактирование памяти ---")
        print("Текущая информация:")
        print(self.user_info)
        new_info = input("Введите новую информацию (или нажмите Enter для отмены): ")
        if new_info:
            self.user_info += "\n" + new_info
            self.save_memory()
            print("✅ Память обновлена")

    def ask_llm(self, prompt):
        """Отправка запроса к локальной Ollama"""
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Ошибка API: {response.status_code}"
        except Exception as e:
            return f"Ошибка подключения к Ollama: {e}. Убедитесь, что Ollama запущена."

    def perceive_environment(self):
        """Сбор информации об окружении (эмуляция для локальной версии)"""
        # В полной версии тут был бы код камеры, микрофона и скриншотов
        # Для экономии ресурсов ПК пока собираем только базовую инфу
        status = {
            "time": datetime.now().strftime("%H:%M"),
            "user_info": self.user_info,
            "system": "Windows 10 (Intel i5, 16GB RAM)",
            "status": "Ожидание команд..."
        }
        return status

    def decide_action(self, context):
        """Принятие решения о действии"""
        prompt = f"""
Ты автономный AI-ассистент на ПК пользователя.
Контекст: {json.dumps(context, ensure_ascii=False)}
История чата: {json.dumps(self.chat_history[-3:], ensure_ascii=False)}

Проанализируй ситуацию. Если нужно что-то сказать или сделать - напиши действие.
Формат ответа строго JSON:
{{
  "thought": "о чем думаешь",
  "speak": "текст речи (если надо говорить)",
  "action": "описание действия (если надо делать)",
  "wait": true/false (ждать ли следующего цикла)
}}
Если действий нет, верни wait: true.
"""
        response = self.ask_llm(prompt)
        try:
            # Попытка найти JSON в ответе
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                return json.loads(response[start:end])
        except:
            pass
        
        return {"thought": "Анализ...", "wait": True}

    def speak(self, text):
        if not text:
            return
        print(f"🗣️ Агент говорит: {text}")
        # Простая озвучка через pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            voices = engine.getProperty('voices')
            # Пытаемся найти русский голос
            for voice in voices:
                if 'RU' in voice.languages or 'ru' in voice.id:
                    engine.setProperty('voice', voice.id)
                    break
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"❌ Ошибка синтеза речи: {e}")

    def run_cycle(self):
        """Один цикл работы агента"""
        context = self.perceive_environment()
        decision = self.decide_action(context)
        
        thought = decision.get("thought", "")
        speak_text = decision.get("speak", "")
        action = decision.get("action", "")
        wait = decision.get("wait", True)
        
        print(f"\n🧠 Мысль: {thought}")
        
        if speak_text:
            self.speak(speak_text)
            self.chat_history.append({"role": "assistant", "content": speak_text})
        
        if action and not wait:
            print(f"🖐️ Действие: {action}")
            # Здесь можно добавить реальное управление мышью/клавиатурой
            # pyautogui.click() и т.д.
        
        self.last_action_time = time.time()

    def interactive_mode(self):
        """Режим ручного ввода команд"""
        while self.running:
            try:
                user_input = input("\n👤 Вы: ")
                if user_input.lower() in ["выход", "quit", "exit"]:
                    break
                if user_input.lower() == "память":
                    self.edit_memory()
                    continue
                
                # Добавляем в историю
                self.chat_history.append({"role": "user", "content": user_input})
                
                prompt = f"""
Пользователь сказал: {user_input}
Твоя память: {self.user_info}
Ответь кратко и по делу. Если нужно действие, опиши его.
"""
                response = self.ask_llm(prompt)
                print(f"🤖 Агент: {response}")
                self.speak(response)
                self.chat_history.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                break

    def autonomous_loop(self):
        """Автономный цикл работы"""
        print("🔄 Запуск автономного режима...")
        while self.running:
            try:
                self.run_cycle()
                time.sleep(self.action_interval)
            except KeyboardInterrupt:
                print("\n⏹️ Остановка агента...")
                self.running = False
                break

def main():
    agent = LocalAgent()
    
    print("\nВыберите режим:")
    print("1. Автономный режим (работает постоянно)")
    print("2. Текстовый чат (ручное управление)")
    
    choice = input("Ваш выбор (1/2): ").strip()
    
    if choice == "2":
        agent.interactive_mode()
    else:
        # Запускаем автономный режим в отдельном потоке, чтобы можно было прервать
        thread = threading.Thread(target=agent.autonomous_loop)
        thread.start()
        try:
            while thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            agent.running = False
            thread.join()

if __name__ == "__main__":
    main()