"""
Main AI Agent module.
Combines perception, memory, actions, and LLM for autonomous operation.
"""
import threading
import time
import logging
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class AIAgent:
    """
    Autonomous AI Agent with continuous perception, thinking, and action.
    
    Features:
    - Real-time camera video capture
    - Microphone audio capture and speech recognition
    - Face detection
    - Screen capture
    - Mouse and keyboard control
    - Text-to-speech output
    - Continuous autonomous operation
    - Chat history management
    - User information file (не забыть.txt)
    - Tkinter GUI for control
    - Text mode interface
    """
    
    def __init__(self, config):
        self.config = config
        self.running = False
        self.autonomous_mode = False
        self.last_action_time = 0
        self.last_thought_time = 0
        
        # Initialize modules
        from .perception import Perception
        from .actions import ActionExecutor
        from .memory import Memory
        
        self.perception = Perception(config)
        self.actions = ActionExecutor(config)
        self.memory = Memory(config)
        
        # LLM client
        self.llm_client = None
        self._initialize_llm()
        
        # Threads
        self.main_thread: Optional[threading.Thread] = None
        self.gui_thread: Optional[threading.Thread] = None
        
        # State
        self.current_thought = ""
        self.decision_made = False
        self.state_lock = threading.Lock()
        
        logger.info("AI Agent initialized")
    
    def _initialize_llm(self):
        """Initialize LLM client."""
        try:
            from openai import OpenAI
            
            self.llm_client = OpenAI(
                base_url=self.config.LLM_BASE_URL,
                api_key=self.config.LLM_API_KEY
            )
            logger.info(f"LLM client initialized: {self.config.LLM_BASE_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
    
    def _think(self, perception_data: Dict[str, Any]) -> str:
        """Generate thought/decision based on perception and memory."""
        try:
            # Build context
            context = self.memory.get_context(max_messages=15)
            
            # Prepare perception summary
            perception_summary = []
            
            if perception_data.get('faces'):
                face_count = len(perception_data['faces'])
                perception_summary.append(f"Обнаружено лиц: {face_count}")
            
            if perception_data.get('speech'):
                perception_summary.append(f"Распознана речь: {perception_data['speech']}")
            
            if perception_data.get('camera_frame') is not None:
                perception_summary.append("Камера активна, изображение получено")
            
            if perception_data.get('screen') is not None:
                perception_summary.append("Экран захвачен")
            
            # Build prompt
            system_prompt = """Ты автономный AI-агент с возможностью восприятия окружения и выполнения действий.
Ты постоянно анализируешь ситуацию и решаешь, нужно ли что-то делать или говорить.
Если думаешь, что действие не требуется - просто наблюдай.
Если видишь необходимость в действии - выполни его.

Доступные действия:
- move_mouse(x, y) - переместить мышь
- click(button='left', clicks=1) - клик мышью
- press_key(key) - нажать клавишу
- type_text(text) - напечатать текст
- speak(text) - сказать текст вслух
- hotkey(keys) - комбинация клавиш

Отвечай в формате JSON:
{
    "thought": "твои размышления о ситуации",
    "action_needed": true/false,
    "action_type": "тип действия или null",
    "action_params": {},
    "response": "ответ пользователю или null"
}"""
            
            user_prompt = f"{context}\n\nТекущее восприятие:\n" + "\n".join(perception_summary)
            
            if self.llm_client:
                response = self.llm_client.chat.completions.create(
                    model=self.config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Try to parse as JSON
                try:
                    # Extract JSON from response
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = result_text[start_idx:end_idx]
                        result = json.loads(json_str)
                    else:
                        result = {
                            "thought": result_text,
                            "action_needed": False,
                            "action_type": None,
                            "action_params": {},
                            "response": None
                        }
                except json.JSONDecodeError:
                    result = {
                        "thought": result_text,
                        "action_needed": False,
                        "action_type": None,
                        "action_params": {},
                        "response": result_text
                    }
                
                return result
            else:
                # Fallback without LLM
                return {
                    "thought": "LLM недоступен, работа в ограниченном режиме",
                    "action_needed": False,
                    "action_type": None,
                    "action_params": {},
                    "response": "LLM клиент не подключен. Пожалуйста, запустите Ollama или другой совместимый сервер."
                }
        
        except Exception as e:
            logger.error(f"Error in thinking: {e}")
            return {
                "thought": f"Ошибка при анализе: {e}",
                "action_needed": False,
                "action_type": None,
                "action_params": {},
                "response": None
            }
    
    def _execute_decision(self, decision: Dict[str, Any]):
        """Execute the decided action."""
        if not decision.get('action_needed', False):
            return
        
        action_type = decision.get('action_type')
        action_params = decision.get('action_params', {})
        
        if action_type and action_params:
            logger.info(f"Executing action: {action_type} with params {action_params}")
            self.actions.execute_action(action_type, **action_params)
            self.last_action_time = time.time()
    
    def _autonomous_loop(self):
        """Main autonomous operation loop."""
        logger.info("Starting autonomous loop")
        
        while self.running and self.autonomous_mode:
            try:
                # Get perception data
                perception_data = self.perception.get_current_state()
                
                # Check for user speech input
                if perception_data.get('speech'):
                    speech_text = perception_data['speech']
                    logger.info(f"User said: {speech_text}")
                    self.memory.add_user_message(speech_text, {'source': 'speech'})
                
                # Think and decide
                current_time = time.time()
                if current_time - self.last_thought_time >= self.config.THINKING_INTERVAL:
                    decision = self._think(perception_data)
                    
                    with self.state_lock:
                        self.current_thought = decision.get('thought', '')
                        self.decision_made = True
                    
                    # Store thought
                    if self.current_thought:
                        logger.info(f"Thought: {self.current_thought}")
                    
                    # Execute action if needed
                    self._execute_decision(decision)
                    
                    # Add response to memory if any
                    response = decision.get('response')
                    if response:
                        self.memory.add_assistant_message(response)
                        # Speak response if TTS enabled
                        self.actions.tts.speak(response)
                    
                    self.last_thought_time = current_time
                
                # Check cooldown before next action
                time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                time.sleep(1)
        
        logger.info("Autonomous loop stopped")
    
    def start(self, autonomous: bool = True):
        """Start the agent."""
        if self.running:
            logger.warning("Agent already running")
            return
        
        self.running = True
        self.autonomous_mode = autonomous
        
        # Start perception
        self.perception.start()
        
        if autonomous:
            # Start autonomous thread
            self.main_thread = threading.Thread(target=self._autonomous_loop, daemon=True)
            self.main_thread.start()
        
        logger.info("Agent started")
    
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.autonomous_mode = False
        
        # Stop perception
        self.perception.stop()
        
        # Wait for threads
        if self.main_thread:
            self.main_thread.join(timeout=3.0)
        
        # Save state
        self.memory.save_checkpoint()
        
        logger.info("Agent stopped")
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop()
        logger.info("Agent cleaned up")
    
    def run_interactive(self):
        """Run in interactive text mode with console input."""
        print("\n" + "=" * 60)
        print("AI АГЕНТ - Интерактивный режим")
        print("=" * 60)
        print("Команды:")
        print("  /start - Запустить автономный режим")
        print("  /stop - Остановить автономный режим")
        print("  /save - Сохранить текущий чат")
        print("  /clear - Очистить историю чата")
        print("  /info - Показать информацию о пользователе")
        print("  /gui - Запустить графический интерфейс")
        print("  /quit - Выход")
        print("=" * 60 + "\n")
        
        self.start(autonomous=False)
        
        try:
            while self.running:
                try:
                    user_input = input("\nВы: ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.startswith('/'):
                        command = user_input.lower()
                        
                        if command == '/quit' or command == '/exit':
                            break
                        
                        elif command == '/start':
                            self.autonomous_mode = True
                            self.main_thread = threading.Thread(target=self._autonomous_loop, daemon=True)
                            self.main_thread.start()
                            print("[*] Автономный режим запущен")
                        
                        elif command == '/stop':
                            self.autonomous_mode = False
                            print("[*] Автономный режим остановлен")
                        
                        elif command == '/save':
                            export_file = self.memory.export_chat()
                            if export_file:
                                print(f"[+] Чат сохранён в: {export_file}")
                        
                        elif command == '/clear':
                            self.memory.clear_chat()
                            print("[+] История чата очищена")
                        
                        elif command == '/info':
                            info = self.memory.user_info.get_content()
                            print("\n" + info + "\n")
                        
                        elif command == '/gui':
                            self.start_gui()
                        
                        else:
                            print(f"[!] Неизвестная команда: {command}")
                    
                    else:
                        # Regular message - process through LLM
                        self.memory.add_user_message(user_input)
                        
                        # Get response
                        perception_data = self.perception.get_current_state()
                        decision = self._think(perception_data)
                        
                        response = decision.get('response', decision.get('thought', ''))
                        if response:
                            print(f"\nАгент: {response}")
                            self.memory.add_assistant_message(response)
                            
                            # Speak if TTS enabled
                            self.actions.tts.speak(response)
                        
                        # Execute any actions
                        self._execute_decision(decision)
                
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
        
        finally:
            self.stop()
    
    def start_gui(self):
        """Start the Tkinter GUI in a separate thread."""
        from .gui import AgentGUI
        
        def run_gui():
            gui = AgentGUI(self)
            gui.run()
        
        self.gui_thread = threading.Thread(target=run_gui, daemon=True)
        self.gui_thread.start()
        print("[*] Графический интерфейс запущен")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        with self.state_lock:
            return {
                'running': self.running,
                'autonomous_mode': self.autonomous_mode,
                'current_thought': self.current_thought,
                'last_action_time': self.last_action_time,
                'last_thought_time': self.last_thought_time,
                'action_executor_state': self.actions.get_state(),
                'chat_message_count': len(self.memory.chat_history.messages)
            }
    
    def save_status(self, status_file: str):
        """Save current status to file."""
        try:
            status = self.get_status()
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving status: {e}")


def create_agent(base_url: str = None, api_key: str = None, model: str = None):
    """Factory function to create an agent instance."""
    from . import config
    
    # Override config if provided
    if base_url:
        config.LLM_BASE_URL = base_url
    if api_key:
        config.LLM_API_KEY = api_key
    if model:
        config.LLM_MODEL = model
    
    return AIAgent(config)
