"""
Tkinter GUI for the AI Agent.
Provides a graphical interface for controlling the agent, viewing chat, and managing settings.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentGUI:
    """Graphical user interface for the AI agent."""
    
    def __init__(self, agent):
        self.agent = agent
        self.root = None
        self.running = False
        
        # GUI variables
        self.chat_history_var = []
        self.status_var = tk.StringVar()
        self.thought_var = tk.StringVar()
        self.user_input_var = tk.StringVar()
        
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main window
        self.root = tk.Tk()
        self.root.title("AI Агент - Панель управления")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # === Status Bar ===
        status_frame = ttk.LabelFrame(main_frame, text="Статус", padding="5")
        status_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_frame, text="Состояние:").grid(row=0, column=0, sticky="w")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground="red")
        self.status_label.grid(row=0, column=1, sticky="w", padx=5)
        
        ttk.Label(status_frame, text="Мышь:").grid(row=0, column=2, sticky="e", padx=(20, 0))
        self.mouse_label = ttk.Label(status_frame, text="(0, 0)")
        self.mouse_label.grid(row=0, column=3, sticky="w", padx=5)
        
        # === Current Thought ===
        thought_frame = ttk.LabelFrame(main_frame, text="Текущая мысль агента", padding="5")
        thought_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        thought_frame.columnconfigure(0, weight=1)
        
        self.thought_label = ttk.Label(thought_frame, textvariable=self.thought_var, wraplength=850)
        self.thought_label.grid(row=0, column=0, sticky="w")
        
        # === Chat Area ===
        chat_frame = ttk.LabelFrame(main_frame, text="История чата", padding="5")
        chat_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 5))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=20,
            font=('Consolas', 10)
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew")
        
        # Configure tags for different message types
        self.chat_display.tag_configure('user', foreground='blue')
        self.chat_display.tag_configure('assistant', foreground='green')
        self.chat_display.tag_configure('system', foreground='gray', font=('Consolas', 10, 'italic'))
        self.chat_display.tag_configure('timestamp', foreground='gray', font=('Consolas', 8))
        
        # === Input Area ===
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=3, column=0, sticky="ew", pady=(0, 5))
        input_frame.columnconfigure(0, weight=1)
        
        self.input_entry = ttk.Entry(input_frame, textvariable=self.user_input_var, font=('Consolas', 10))
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.input_entry.bind('<Return>', lambda e: self._send_message())
        
        send_btn = ttk.Button(input_frame, text="Отправить", command=self._send_message)
        send_btn.grid(row=0, column=1)
        
        # === Control Buttons ===
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, column=0, sticky="ew")
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Старт", command=self._start_agent)
        self.start_btn.grid(row=0, column=0, padx=2)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Стоп", command=self._stop_agent)
        self.stop_btn.grid(row=0, column=1, padx=2)
        
        save_btn = ttk.Button(btn_frame, text="💾 Сохранить чат", command=self._save_chat)
        save_btn.grid(row=0, column=2, padx=2)
        
        clear_btn = ttk.Button(btn_frame, text="🗑 Очистить", command=self._clear_chat)
        clear_btn.grid(row=0, column=3, padx=2)
        
        info_btn = ttk.Button(btn_frame, text="ℹ Инфо", command=self._show_user_info)
        info_btn.grid(row=0, column=4, padx=2)
        
        edit_info_btn = ttk.Button(btn_frame, text="✏ Ред. инфо", command=self._edit_user_info)
        edit_info_btn.grid(row=0, column=5, padx=2)
        
        export_btn = ttk.Button(btn_frame, text="📤 Экспорт", command=self._export_chat)
        export_btn.grid(row=0, column=6, padx=2)
        
        # === Menu Bar ===
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Сохранить чат", command=self._save_chat)
        file_menu.add_command(label="Экспортировать чат", command=self._export_chat)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self._on_close)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Правка", menu=edit_menu)
        edit_menu.add_command(label="Очистить историю", command=self._clear_chat)
        edit_menu.add_command(label="Информация о пользователе", command=self._show_user_info)
        edit_menu.add_command(label="Редактировать информацию", command=self._edit_user_info)
        
        # Agent menu
        agent_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Агент", menu=agent_menu)
        agent_menu.add_command(label="Запустить", command=self._start_agent)
        agent_menu.add_command(label="Остановить", command=self._stop_agent)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Помощь", menu=help_menu)
        help_menu.add_command(label="О программе", command=self._show_about)
        
        # Load initial chat history
        self._load_chat_history()
        
        # Start status update loop
        self._update_status()
        
        logger.info("GUI initialized")
    
    def _load_chat_history(self):
        """Load chat history from memory."""
        messages = self.agent.memory.chat_history.messages
        
        self.chat_display.configure(state='normal')
        self.chat_display.delete('1.0', tk.END)
        
        for msg in messages:
            timestamp = msg.get('timestamp', '')[:19]
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            self.chat_display.insert(tk.END, f"[{timestamp}] ", 'timestamp')
            
            if role == 'user':
                self.chat_display.insert(tk.END, "Вы: ", 'user')
            elif role == 'assistant':
                self.chat_display.insert(tk.END, "Агент: ", 'assistant')
            else:
                self.chat_display.insert(tk.END, f"{role}: ", 'system')
            
            self.chat_display.insert(tk.END, content + "\n\n")
        
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')
    
    def _add_message_to_chat(self, role: str, content: str):
        """Add a message to the chat display."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"[{timestamp}] ", 'timestamp')
        
        if role == 'user':
            self.chat_display.insert(tk.END, "Вы: ", 'user')
        elif role == 'assistant':
            self.chat_display.insert(tk.END, "Агент: ", 'assistant')
        else:
            self.chat_display.insert(tk.END, f"{role}: ", 'system')
        
        self.chat_display.insert(tk.END, content + "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')
    
    def _send_message(self):
        """Send a message to the agent."""
        message = self.user_input_var.get().strip()
        if not message:
            return
        
        self.user_input_var.set("")
        
        # Add to chat display
        self._add_message_to_chat('user', message)
        
        # Add to memory and get response
        self.agent.memory.add_user_message(message)
        
        # Process in background
        def process_response():
            perception_data = self.agent.perception.get_current_state()
            decision = self.agent._think(perception_data)
            
            response = decision.get('response', decision.get('thought', ''))
            if response:
                self.agent.memory.add_assistant_message(response)
                
                # Update GUI in main thread
                self.root.after(0, lambda: self._add_message_to_chat('assistant', response))
                
                # Speak if enabled
                self.agent.actions.tts.speak(response)
            
            # Execute any actions
            self.agent._execute_decision(decision)
        
        thread = threading.Thread(target=process_response, daemon=True)
        thread.start()
    
    def _start_agent(self):
        """Start the agent."""
        self.agent.start(autonomous=True)
        self.status_var.set("Работает (автономный режим)")
        self.status_label.configure(foreground="green")
    
    def _stop_agent(self):
        """Stop the agent."""
        self.agent.stop()
        self.status_var.set("Остановлен")
        self.status_label.configure(foreground="red")
    
    def _save_chat(self):
        """Save current chat."""
        self.agent.memory.save_checkpoint()
        messagebox.showinfo("Сохранение", "Чат сохранён!")
        self._load_chat_history()
    
    def _clear_chat(self):
        """Clear chat history."""
        if messagebox.askyesno("Подтверждение", "Очистить всю историю чата?"):
            self.agent.memory.clear_chat()
            self.chat_display.configure(state='normal')
            self.chat_display.delete('1.0', tk.END)
            self.chat_display.configure(state='disabled')
    
    def _show_user_info(self):
        """Show user information in a dialog."""
        info = self.agent.memory.user_info.get_content()
        
        info_window = tk.Toplevel(self.root)
        info_window.title("Информация о пользователе")
        info_window.geometry("600x500")
        
        text_widget = scrolledtext.ScrolledText(info_window, wrap=tk.WORD, font=('Consolas', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert('1.0', info)
        text_widget.configure(state='disabled')
    
    def _edit_user_info(self):
        """Open editor for user information file."""
        info = self.agent.memory.user_info.get_content()
        
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Редактирование информации о пользователе")
        edit_window.geometry("600x500")
        
        text_widget = scrolledtext.ScrolledText(edit_window, wrap=tk.NONE, font=('Consolas', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert('1.0', info)
        
        def save_info():
            new_content = text_widget.get('1.0', tk.END)
            self.agent.memory.user_info.update_content(new_content)
            messagebox.showinfo("Сохранение", "Информация обновлена!")
            edit_window.destroy()
        
        btn_frame = ttk.Frame(edit_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        save_btn = ttk.Button(btn_frame, text="Сохранить", command=save_info)
        save_btn.pack(side=tk.RIGHT, padx=5)
        
        cancel_btn = ttk.Button(btn_frame, text="Отмена", command=edit_window.destroy)
        cancel_btn.pack(side=tk.RIGHT)
    
    def _export_chat(self):
        """Export chat to file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if file_path:
            success = self.agent.memory.export_chat(Path(file_path))
            if success:
                messagebox.showinfo("Экспорт", f"Чат экспортирован в:\n{file_path}")
            else:
                messagebox.showerror("Ошибка", "Не удалось экспортировать чат")
    
    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "О программе",
            "AI Агент v1.0\n\n"
            "Автономный AI-агент с возможностями:\n"
            "- Распознавание видео с камеры\n"
            "- Распознавание речи с микрофона\n"
            "- Детекция лиц\n"
            "- Захват экрана\n"
            "- Управление мышью и клавиатурой\n"
            "- Синтез речи (TTS)\n"
            "- Постоянная работа в реальном времени\n"
            "- Сохранение истории чата\n"
            "- Файл пользовательской информации"
        )
    
    def _update_status(self):
        """Periodically update status display."""
        if self.running and self.root:
            try:
                status = self.agent.get_status()
                
                if status['running']:
                    if status['autonomous_mode']:
                        self.status_var.set("Работает (автономный)")
                        self.status_label.configure(foreground="green")
                    else:
                        self.status_var.set("Работает (ожидание)")
                        self.status_label.configure(foreground="orange")
                else:
                    self.status_var.set("Остановлен")
                    self.status_label.configure(foreground="red")
                
                # Update thought
                if status['current_thought']:
                    thought = status['current_thought'][:100] + "..." if len(status['current_thought']) > 100 else status['current_thought']
                    self.thought_var.set(thought)
                
                # Update mouse position
                action_state = status.get('action_executor_state', {})
                mouse_pos = action_state.get('mouse_position', (0, 0))
                self.mouse_label.configure(text=f"({mouse_pos[0]}, {mouse_pos[1]})")
                
            except Exception as e:
                logger.error(f"Error updating status: {e}")
            
            # Schedule next update
            self.root.after(1000, self._update_status)
    
    def _on_close(self):
        """Handle window close event."""
        if messagebox.askyesno("Выход", "Остановить агента и выйти?"):
            self.running = False
            self.agent.stop()
            self.root.destroy()
    
    def run(self):
        """Run the GUI main loop."""
        self.running = True
        self._create_widgets()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start main loop
        self.root.mainloop()
        
        self.running = False
