"""Configuration settings for the Hybrid Search Agent"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Model settings
CONTEXT_WINDOW = 6000
MODELS_DIR = Path("./models")  # Папка для скачанных моделей

# Available models catalog
AVAILABLE_MODELS = {
    # Russian models
    "qvikhr-3b": {
        "filename": "QVikhr-3-4B-Instruction-Q3_K_S.gguf",
        "url": "https://huggingface.co/IlyaGusev/QVikhr-3-4B-Instruction-GGUF/resolve/main/QVikhr-3-4B-Instruction-Q3_K_S.gguf",
        "description": "QVikhr 3.4B Russian instruction model",
        "size_gb": 2.1,
        "language": "ru",
        "context": 6000,
        "default": False,
    },
    "qvikhr-7b": {
        "filename": "QVikhr-7B-Instruction-Q4_K_M.gguf",
        "url": "https://huggingface.co/IlyaGusev/QVikhr-7B-Instruct-GGUF/resolve/main/QVikhr-7B-Instruct-Q4_K_M.gguf",
        "description": "QVikhr 7B Russian instruction model",
        "size_gb": 4.3,
        "language": "ru",
        "context": 8000,
        "default": False,
    },
    "saiga-7b": {
        "filename": "saiga-7b-q4_K_M.gguf",
        "url": "https://huggingface.co/IlyaGusev/saiga_7b_lora_gguf/resolve/main/saiga-7b-q4_K_M.gguf",
        "description": "Saiga 7B Russian chatbot model",
        "size_gb": 4.1,
        "language": "ru",
        "context": 4000,
        "default": False,
    },
    "saiga-13b": {
        "filename": "saiga-13b-q4_K_M.gguf",
        "url": "https://huggingface.co/IlyaGusev/saiga_13b_lora_gguf/resolve/main/saiga-13b-q4_K_M.gguf",
        "description": "Saiga 13B Russian chatbot model",
        "size_gb": 7.9,
        "language": "ru",
        "context": 4000,
        "default": False,
    },
    # English models
    "tinyllama": {
        "filename": "tinyllama-1.1b-chat.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "description": "TinyLlama 1.1B Chat (lightweight, fast)",
        "size_gb": 0.7,
        "language": "en",
        "context": 2048,
        "default": True,
    },
    "llama2-7b": {
        "filename": "llama-2-7b-chat.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
        "description": "Llama 2 7B Chat (balanced)",
        "size_gb": 4.1,
        "language": "en",
        "context": 4096,
        "default": False,
    },
    "llama2-13b": {
        "filename": "llama-2-13b-chat.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf",
        "description": "Llama 2 13B Chat (high quality)",
        "size_gb": 7.9,
        "language": "en",
        "context": 4096,
        "default": False,
    },
    "mistral-7b": {
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "Mistral 7B Instruct v0.2 (excellent quality)",
        "size_gb": 4.1,
        "language": "en",
        "context": 8192,
        "default": False,
    },
    "zephyr-7b": {
        "filename": "zephyr-7b-beta.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf",
        "description": "Zephyr 7B Beta (instruct tuned)",
        "size_gb": 4.1,
        "language": "en",
        "context": 4096,
        "default": False,
    },
    "phi-2": {
        "filename": "phi-2.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "description": "Microsoft Phi-2 2.7B (compact, capable)",
        "size_gb": 1.6,
        "language": "en",
        "context": 2048,
        "default": False,
    },
    "neural-chat-7b": {
        "filename": "neural-chat-7b-v3-1.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/neural-chat-7b-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf",
        "description": "Neural Chat 7B v3.1 (Intel)",
        "size_gb": 4.1,
        "language": "en",
        "context": 4096,
        "default": False,
    },
    "openchat-7b": {
        "filename": "openchat-3.5-7b.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/openchat-3.5-7b-GGUF/resolve/main/openchat-3.5-7b.Q4_K_M.gguf",
        "description": "OpenChat 3.5 7B",
        "size_gb": 4.1,
        "language": "en",
        "context": 8192,
        "default": False,
    },
    "dolphin-7b": {
        "filename": "dolphin-2.2.1-mistral-7b.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7b-GGUF/resolve/main/dolphin-2.2.1-mistral-7b.Q4_K_M.gguf",
        "description": "Dolphin 2.2.1 Mistral 7B",
        "size_gb": 4.1,
        "language": "en",
        "context": 8192,
        "default": False,
    },
}

DEFAULT_MODEL_PATH = ""
# Tavily API
TAVILY_API_KEY = "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"

# Directory settings
DATA_DIR = Path("./data")
STORAGE_DIR = Path("./storage")
LOGS_DIR = Path("./logs")
SCREENSHOTS_DIR = Path("./screenshots")
PDFS_DIR = Path("./pdfs")
STEP_HISTORY_DIR = Path("./step_history")

# Embedding settings
EMBEDDING_MODEL = "deepvk/USER2-small"
# EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 32

# LLM settings
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 1024
GPU_BATCH_SIZE = 512
CPU_THREADS = 8

# Agent settings
MAX_ITERATIONS = 4
PLAYWRIGHT_SLOW_MO = 50
HEADLESS_BROWSER = True

# Tracing settings
ENABLE_PHOENIX_TRACING = True


# Create directories
def create_directories():
    """Create all necessary directories"""
    directories = [
        DATA_DIR,
        STORAGE_DIR,
        LOGS_DIR,
        SCREENSHOTS_DIR,
        PDFS_DIR,
        STEP_HISTORY_DIR,
        MODELS_DIR,
    ]
    for directory in directories:
        directory.mkdir(exist_ok=True)
    return directories


__all__ = [
    "CONTEXT_WINDOW",
    "MODELS_DIR",
    "AVAILABLE_MODELS",
    "TAVILY_API_KEY",
    "DATA_DIR",
    "STORAGE_DIR",
    "LOGS_DIR",
    "SCREENSHOTS_DIR",
    "PDFS_DIR",
    "STEP_HISTORY_DIR",
    "EMBEDDING_MODEL",
    "FALLBACK_EMBEDDING_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "EMBED_BATCH_SIZE",
    "TEMPERATURE",
    "MAX_NEW_TOKENS",
    "GPU_BATCH_SIZE",
    "CPU_THREADS",
    "MAX_ITERATIONS",
    "PLAYWRIGHT_SLOW_MO",
    "HEADLESS_BROWSER",
    "ENABLE_PHOENIX_TRACING",
    "create_directories",
]
