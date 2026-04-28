#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полноценный AI-агент для работы локально через Ollama
Поддержка: камера, микрофон, экран, распознавание лиц/речи, управление мышью/клавиатурой
Работает постоянно, принимает решения автономно
"""

import os
import sys
import time
import threading
import json
import queue
from datetime import datetime
from pathlib import Path

# GUI
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

# Мультимедиа и восприятие
import cv2
import numpy as np
import pyautogui
import pyscreenshot as ImageGrab
from PIL import Image

# Аудио
import speech_recognition as sr
import pyttsx3
import pyaudio

# Управление
from pynput import mouse, keyboard

# Локальная LLM через Ollama
import requests

# Пути
BASE_DIR = Path(__file__).parent
MEMORY_FILE = BASE_DIR / "не забыть.txt"
CHATS_DIR = BASE_DIR / "chats"
CONFIG_FILE = BASE_DIR / "config.json"

# Настройки Ollama
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:0.5b"  # Легкая модель для GTX 1050 Ti

class PerceptionSystem:
    """Система восприятия: камера, микрофон, экран, лица"""
    
    def __init__(self):
        self.camera = None
        self.audio = sr.Recognizer()
        self.mic = None
        self.screen_queue = queue.Queue()
        self.face_cache = None
        self.running = False
        
        # Инициализация камеры
        try:
            self.camera = cv2.VideoCapture(0)
            print("✓ Камера инициализирована")
        except Exception as e:
            print(f"⚠ Камера не доступна: {e}")
        
        # Инициализация микрофона
        try:
            self.mic = sr.Microphone()
            print("✓ Микрофон инициализирован")
        except Exception as e:
            print(f"⚠ Микрофон не доступен: {e}")
    
    def capture_camera(self):
        """Захват кадра с камеры"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None
    
    def detect_faces(self, frame):
        """Простое обнаружение лиц (через каскады Haar)"""
        if frame is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Загрузка каскада
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0
    
    def capture_screen(self):
        """Захват экрана"""
        try:
            screenshot = ImageGrab.grab()
            return np.array(screenshot)
        except Exception as e:
            print(f"Ошибка захвата экрана: {e}")
            return None
    
    def listen_audio(self, timeout=5, phrase_time_limit=10):
        """Распознавание речи"""
        if not self.mic:
            return None
        
        try:
            with self.mic as source:
                self.audio.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.audio.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            # Распознавание через встроенный движок (можно заменить на Whisper локально)
            try:
                text = self.audio.recognize_google(audio, language="ru-RU")
                return text
            except:
                try:
                    text = self.audio.recognize_google(audio, language="en-US")
                    return text
                except:
                    return None
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"Ошибка аудио: {e}")
            return None
    
    def start_continuous_listening(self, callback):
        """Непрерывное прослушивание в отдельном потоке"""
        def listen_loop():
            while self.running:
                text = self.listen_audio(timeout=1, phrase_time_limit=5)
                if text:
                    callback(text)
                time.sleep(0.5)
        
        thread = threading.Thread(target=listen_loop, daemon=True)
        thread.start()
        return thread
    
    def close(self):
        self.running = False
        if self.camera:
            self.camera.release()

class LocalLLM:
    """Локальная LLM через Ollama"""
    
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.base_url = OLLAMA_URL
        self.context = []
        self.max_context = 10
    
    def check_connection(self):
        """Проверка подключения к Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt, system_prompt="Ты полезный AI-ассистент."):
        """Генерация ответа"""
        url = f"{self.base_url}/api/generate"
        
        # Добавляем в контекст
        self.context.append({"role": "user", "content": prompt})
        if len(self.context) > self.max_context:
            self.context = self.context[-self.max_context:]
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "context": [msg["content"] for msg in self.context[-5:]]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "")
                
                # Сохраняем ответ в контекст
                self.context.append({"role": "assistant", "content": answer})
                return answer
            else:
                return f"Ошибка API: {response.status_code}"
        except Exception as e:
            return f"Ошибка подключения: {e}"
    
    def analyze_image(self, image_path, question="Что на изображении?"):
        """Анализ изображения (если модель поддерживает)"""
        # Для мультимодальных моделей типа llava
        url = f"{self.base_url}/api/generate"
        
        try:
            with open(image_path, "rb") as f:
                import base64
                image_data = base64.b64encode(f.read()).decode()
            
            payload = {
                "model": "llava:7b",  # Мультимодальная модель
                "prompt": question,
                "images": [image_data],
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json().get("response", "")
        except:
            pass
        
        return "Не удалось проанализировать изображение"
    
    def clear_context(self):
        self.context = []

class MemoryManager:
    """Управление памятью и файлом 'не забыть.txt'"""
    
    def __init__(self, memory_file=MEMORY_FILE):
        self.memory_file = memory_file
        self.chats_dir = CHATS_DIR
        self.chats_dir.mkdir(exist_ok=True)
        self._init_memory_file()
    
    def _init_memory_file(self):
        if not self.memory_file.exists():
            initial_content = """# Важная информация о пользователе
Имя: Пользователь
Предпочтения: 
- Язык: русский
- Режим работы: автономный

Заметки:
(Добавляйте сюда важную информацию)

Последнее обновление: """ + datetime.now().strftime("%Y-%m-%d %H:%M")
            
            with open(self.memory_file, "w", encoding="utf-8") as f:
                f.write(initial_content)
    
    def read_memory(self):
        with open(self.memory_file, "r", encoding="utf-8") as f:
            return f.read()
    
    def write_memory(self, content):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write(content + "\n\nПоследнее обновление: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    def save_chat(self, chat_history, name=None):
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        chat_file = self.chats_dir / f"{name}.json"
        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        
        return chat_file
    
    def load_chat(self, name):
        chat_file = self.chats_dir / f"{name}.json"
        if chat_file.exists():
            with open(chat_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    
    def list_chats(self):
        return [f.stem for f in self.chats_dir.glob("*.json")]

class VoiceSynthesizer:
    """Синтез речи"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
        # Попытка установить русский голос
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'ru' in voice.languages or 'Russian' in voice.name:
                self.engine.setProperty('voice', voice.id)
                break
    
    def speak(self, text):
        """Озвучивание текста"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Ошибка синтеза речи: {e}")
    
    def speak_async(self, text):
        """Асинхронное озвучивание"""
        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()

class AutonomousAgent:
    """Основной класс автономного агента"""
    
    def __init__(self):
        self.perception = PerceptionSystem()
        self.llm = LocalLLM()
        self.memory = MemoryManager()
        self.voice = VoiceSynthesizer()
        
        self.running = False
        self.decision_interval = 5  # Секунды между решениями
        self.action_queue = queue.Queue()
        
        # Загрузка памяти пользователя
        self.user_memory = self.memory.read_memory()
        
        # Проверка подключения к Ollama
        if not self.llm.check_connection():
            print("⚠ Ollama не запущен! Запустите: ollama serve")
            print("И установите модель: ollama pull qwen2.5:0.5b")
        else:
            print("✓ Ollama подключен")
    
    def get_system_prompt(self):
        """Формирование системного промпта с учетом памяти"""
        return f"""Ты автономный AI-агент, работающий на компьютере пользователя.

{self.user_memory}

Твои возможности:
- Видеть камеру, слышать микрофон, видеть экран
- Управлять мышью и клавиатурой
- Говорить через синтезатор речи
- Принимать решения самостоятельно

Правила:
1. Действуй только если это действительно нужно
2. Не мешай пользователю без причины
3. Если пользователь работает - не вмешивайся
4. Предлагай помощь если видишь проблему
5. Будь дружелюбным и полезным

Отвечай кратко и по делу."""
    
    def perceive_environment(self):
        """Сбор информации об окружении"""
        info = {
            "time": datetime.now().strftime("%H:%M"),
            "camera_face": False,
            "screen_content": "",
            "audio_input": "",
            "mouse_pos": pyautogui.position(),
            "active_window": ""
        }
        
        # Камера
        frame = self.perception.capture_camera()
        if frame is not None:
            info["camera_face"] = self.perception.detect_faces(frame)
        
        # Экран (центральный区域)
        screen = self.perception.capture_screen()
        if screen is not None:
            h, w = screen.shape[:2]
            center = screen[h//3:2*h//3, w//3:2*w//3]
            info["screen_content"] = f"Разрешение: {w}x{h}"
        
        return info
    
    def decide_action(self, perception_info):
        """Принятие решения о действии"""
        prompt = f"""Текущая ситуация:
Время: {perception_info['time']}
Лицо в камере: {'Да' if perception_info['camera_face'] else 'Нет'}
Экран: {perception_info['screen_content']}
Позиция мыши: {perception_info['mouse_pos']}

Нужно ли предпринимать какие-то действия? 
Если да - опиши что именно и почему.
Если нет - просто напиши 'Действий не требуется'."""

        response = self.llm.generate(prompt, self.get_system_prompt())
        
        if "Действий не требуется" not in response and len(response) > 10:
            return response
        return None
    
    def execute_action(self, action_description):
        """Выполнение действия"""
        print(f"🤖 Действие: {action_description}")
        
        # Простая логика выполнения
        if "открыть" in action_description.lower():
            if "браузер" in action_description.lower():
                pyautogui.hotkey('win', 'd')
                time.sleep(0.5)
                pyautogui.typewrite('chrome')
                pyautogui.press('enter')
        
        elif "сказать" in action_description.lower() or "говорить" in action_description.lower():
            # Извлекаем текст для озвучивания
            self.voice.speak_async(action_description)
        
        elif "написать" in action_description.lower():
            # Эмуляция ввода текста
            pass
        
        # Логирование
        with open("agent_actions.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} - {action_description}\n")
    
    def run_autonomous_loop(self):
        """Основной цикл автономной работы"""
        self.running = True
        self.perception.running = True
        
        print("🚀 Автономный режим запущен...")
        
        # Непрерывное прослушивание
        def on_speech(text):
            print(f"🎤 Услышано: {text}")
            self.handle_speech(text)
        
        listen_thread = self.perception.start_continuous_listening(on_speech)
        
        last_decision_time = 0
        
        try:
            while self.running:
                current_time = time.time()
                
                # Принятие решений каждые N секунд
                if current_time - last_decision_time >= self.decision_interval:
                    perception_info = self.perceive_environment()
                    action = self.decide_action(perception_info)
                    
                    if action:
                        self.execute_action(action)
                    
                    last_decision_time = current_time
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n🛑 Остановка агента...")
        finally:
            self.stop()
    
    def handle_speech(self, text):
        """Обработка голосовой команды"""
        prompt = f"Пользователь сказал: '{text}'. Что ответить или сделать?"
        response = self.llm.generate(prompt, self.get_system_prompt())
        
        print(f"💬 Ответ: {response}")
        self.voice.speak_async(response)
    
    def chat_mode(self, text):
        """Текстовый режим чата"""
        response = self.llm.generate(text, self.get_system_prompt())
        return response
    
    def stop(self):
        self.running = False
        self.perception.close()

class AgentGUI:
    """Графический интерфейс управления агентом"""
    
    def __init__(self, agent):
        self.agent = agent
        self.root = tk.Tk()
        self.root.title("AI Агент - Локальный")
        self.root.geometry("900x700")
        
        self.chat_history = []
        self.current_chat_name = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        # Верхняя панель
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(top_frame, text="Статус:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.status_label = ttk.Label(top_frame, text="⏸️ Остановлен", foreground="orange")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(top_frame, text="▶️ Запустить", command=self.start_agent).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="⏹️ Остановить", command=self.stop_agent).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="💾 Сохранить чат", command=self.save_chat).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="📂 Загрузить чат", command=self.load_chat).pack(side=tk.LEFT, padx=5)
        
        # Основная область
        main_frame = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Чат
        chat_frame = ttk.LabelFrame(main_frame, text="Чат с агентом")
        main_frame.add(chat_frame, weight=3)
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Ввод сообщения
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.message_entry = ttk.Entry(input_frame, font=("Arial", 11))
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.message_entry.bind("<Return>", lambda e: self.send_message())
        
        ttk.Button(input_frame, text="Отправить", command=self.send_message).pack(side=tk.RIGHT)
        
        # Память пользователя
        memory_frame = ttk.LabelFrame(main_frame, text="Память (не забыть.txt)")
        main_frame.add(memory_frame, weight=1)
        
        self.memory_display = scrolledtext.ScrolledText(memory_frame, wrap=tk.WORD, font=("Consolas", 9), height=8)
        self.memory_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Button(memory_frame, text="Сохранить изменения", command=self.save_memory).pack(pady=5)
        
        # Загрузка памяти
        self.memory_display.insert(tk.END, self.agent.memory.read_memory())
        
        # Нижняя панель
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(bottom_frame, text="Режим:").pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="chat")
        ttk.Radiobutton(bottom_frame, text="Чат", variable=self.mode_var, value="chat").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(bottom_frame, text="Автономный", variable=self.mode_var, value="autonomous").pack(side=tk.LEFT, padx=5)
        
        # Лог действий
        log_frame = ttk.LabelFrame(self.root, text="Лог действий")
        log_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.log_display = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 8), height=4)
        self.log_display.pack(fill=tk.X, padx=5, pady=5)
    
    def start_agent(self):
        mode = self.mode_var.get()
        
        if mode == "autonomous":
            thread = threading.Thread(target=self.agent.run_autonomous_loop, daemon=True)
            thread.start()
            self.status_label.config(text="🟢 Автономный режим", foreground="green")
            self._log("Автономный режим запущен")
        else:
            self.status_label.config(text="🟢 Готов к чату", foreground="green")
            self._log("Режим чата активен")
    
    def stop_agent(self):
        self.agent.stop()
        self.status_label.config(text="⏸️ Остановлен", foreground="orange")
        self._log("Агент остановлен")
    
    def send_message(self):
        message = self.message_entry.get().strip()
        if not message:
            return
        
        self.message_entry.delete(0, tk.END)
        
        # Отображение сообщения пользователя
        self.chat_display.insert(tk.END, f"\n👤 Вы: {message}\n")
        self.chat_history.append({"role": "user", "content": message})
        
        # Получение ответа
        response = self.agent.chat_mode(message)
        
        # Отображение ответа
        self.chat_display.insert(tk.END, f"🤖 Агент: {response}\n")
        self.chat_history.append({"role": "assistant", "content": response})
        
        self.chat_display.see(tk.END)
        
        # Озвучивание
        self.agent.voice.speak_async(response)
    
    def save_chat(self):
        if not self.chat_history:
            messagebox.showwarning("Предупреждение", "Чат пуст!")
            return
        
        name = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON файлы", "*.json")],
            initialdir=str(self.agent.memory.chats_dir)
        )
        
        if name:
            with open(name, "w", encoding="utf-8") as f:
                json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            
            self._log(f"Чат сохранен: {name}")
            messagebox.showinfo("Успех", "Чат сохранен!")
    
    def load_chat(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON файлы", "*.json")],
            initialdir=str(self.agent.memory.chats_dir)
        )
        
        if filename:
            with open(filename, "r", encoding="utf-8") as f:
                self.chat_history = json.load(f)
            
            self.chat_display.delete(1.0, tk.END)
            for msg in self.chat_history:
                role = "👤 Вы" if msg["role"] == "user" else "🤖 Агент"
                self.chat_display.insert(tk.END, f"{role}: {msg['content']}\n")
            
            self._log(f"Чат загружен: {filename}")
    
    def save_memory(self):
        content = self.memory_display.get(1.0, tk.END).strip()
        self.agent.memory.write_memory(content)
        self.agent.user_memory = content
        self._log("Память обновлена")
        messagebox.showinfo("Успех", "Память сохранена!")
    
    def _log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_display.see(tk.END)
    
    def run(self):
        self.root.mainloop()

def main():
    print("=" * 50)
    print("🤖 Локальный AI Агент")
    print("=" * 50)
    
    # Создание агента
    agent = AutonomousAgent()
    
    # Запуск GUI
    gui = AgentGUI(agent)
    gui.run()

if __name__ == "__main__":
    main()
