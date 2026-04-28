#!/usr/bin/env python3
"""
Download a local LLM model from HuggingFace.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from qwen_agent.local_llm import LocalLLM


def main():
    parser = argparse.ArgumentParser(description="Download LLM model")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5-3b",
        help="Model to download"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    llm = LocalLLM()
    
    if args.list:
        print("Available models:")
        for m in llm.list_models():
            print(f"  {m['id']}: {m['description']} ({m['size_gb']}GB)")
        return
    
    print(f"Model: {args.model}")
    print(f"Loading: {llm.MODELS.get(args.model, {}).get('name', 'unknown')}")
    print()
    
    success = llm.download_model()
    
    if success:
        print("\n" + "="*50)
        print("MODEL DOWNLOADED SUCCESSFULLY!")
        print("="*50)
        print(f"\nModel: {args.model}")
        print(f"Location: {llm.cache_dir}")
        print("\nYou can now run the agent:")
        print("  python run_agent.py --mode interactive")
        print("\nThe agent will use the local model automatically.")
    else:
        print("\n" + "="*50)
        print("DOWNLOAD FAILED!")
        print("="*50)
        print("\nCheck your internet connection and try again.")
        print("Or try a smaller model: smollm2-135m")


if __name__ == "__main__":
    main()
