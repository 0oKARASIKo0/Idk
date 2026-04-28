#!/usr/bin/env python3
"""
Qwen AI Agent - Main Entry Point
Autonomous AI agent with continuous thinking, mouse/keyboard control,
voice commands, camera vision, and text-to-speech.
"""
import sys
import os
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_agent.main import create_agent
from qwen_agent import config


def setup_logging(log_level: str = None, log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level or config.LOG_LEVEL)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Qwen AI Agent - Autonomous AI with continuous thinking"
    )
    parser.add_argument(
        '--llm-url',
        type=str,
        default=None,
        help='LLM base URL (default: http://localhost:11434/v1 for Ollama)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='LLM model name (default: qwen2.5)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='LLM API key (not needed for local Ollama)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'autonomous'],
        default='interactive',
        help='Running mode (default: interactive)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=None,
        help='Logging level'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file path'
    )
    parser.add_argument(
        '--status-file',
        type=str,
        default=None,
        help='File to save status periodically'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Create agent
    agent = create_agent(
        base_url=args.llm_url,
        api_key=args.api_key,
        model=args.model
    )
    
    try:
        if args.mode == 'interactive':
            # Run in interactive mode with console input
            agent.run_interactive()
        
        elif args.mode == 'autonomous':
            # Run fully autonomously
            agent.start()
            
            # Keep running until interrupted
            import time
            while True:
                time.sleep(1)
                
                # Optionally save status
                if args.status_file:
                    agent.save_status(args.status_file)
    
    except KeyboardInterrupt:
        print("\n\n[!] Прервано пользователем")
        logging.info("Agent interrupted by user")
    except ImportError as e:
        print(f"\n\n[ERROR] Ошибка импорта модуля: {e}")
        print("Запустите: pip install -r requirements.txt")
        input("\nНажмите Enter для выхода...")
        sys.exit(1)
    except ConnectionError as e:
        print(f"\n\n[ERROR] Ошибка подключения: {e}")
        print("Проверьте что Ollama запущен (ollama serve)")
        input("\nНажмите Enter для выхода...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        input("\nНажмите Enter для выхода...")
        sys.exit(1)
    finally:
        agent.cleanup()
        print("\nАгент остановлен.")


if __name__ == "__main__":
    main()
