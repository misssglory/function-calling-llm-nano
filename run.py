#!/usr/bin/env python3
"""
Root runner script for Hybrid Search Agent
"""

import sys
import torch
import asyncio
from pathlib import Path
from loguru import logger

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hybrid_search_agent.main import main
from hybrid_search_agent.utils.model_utils import (
    list_downloaded_models,
    list_available_models,
    download_model,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Search Agent")

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="tinyllama",
        help="Model name/key (e.g., tinyllama, mistral-7b, qvikhr-3b)",
    )

    parser.add_argument(
        "--list-models", "-lm", action="store_true", help="List downloaded models"
    )

    parser.add_argument(
        "--available-models",
        "-am",
        action="store_true",
        help="List all available models for download",
    )

    parser.add_argument("--download", type=str, help="Download a model by key")

    parser.add_argument(
        "--language", "-lang", choices=["ru", "en"], help="Filter models by language"
    )

    parser.add_argument(
        "--step-by-step",
        "-s",
        action="store_true",
        help="Enable step-by-step execution mode",
    )

    parser.add_argument(
        "--visible", action="store_false", dest="headless", help="Show browser window"
    )

    parser.add_argument(
        "--query", "-q", type=str, help="Single query to execute (non-interactive mode)"
    )

    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable automatic model download",
    )

    parser.add_argument(
        "--colab",
        action="store_true",
        dest="colab",
        help="Option to run in colab environment",
    )

    args, unknown = parser.parse_known_args()

    logger.info(f"Args: {args}")

    if args.colab:
        import nest_asyncio

        nest_asyncio.apply()
        logger.info("Next asyncio applied")

    if args.list_models:
        list_downloaded_models()
    elif args.available_models:
        list_available_models(args.language)
    elif args.download:
        model_path = download_model(args.download)
        if model_path:
            print(f"\n✅ Model downloaded: {model_path}")
    else:
        # Pass args to main
        sys.argv = [sys.argv[0]] + unknown
        sys.argv.extend(["--model", args.model])
        if args.step_by_step:
            sys.argv.append("--step-by-step")
        if not args.headless:
            sys.argv.append("--visible")
        if args.query:
            sys.argv.extend(["--query", args.query])
        if args.no_auto_download:
            sys.argv.append("--no-auto-download")

        asyncio.run(main())
