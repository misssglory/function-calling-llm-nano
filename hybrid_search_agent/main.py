#!/usr/bin/env python3
"""Main entry point for Hybrid Search Agent"""

import asyncio
import argparse
from pathlib import Path

from hybrid_search_agent.sessions.interactive import (
    create_new_session,
    interactive_chat_session,
    quick_query,
)
from hybrid_search_agent.utils.tracing import setup_tracing
from hybrid_search_agent.utils.model_utils import (
    list_downloaded_models,
    list_available_models,
)
from loguru import logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hybrid Search Agent with Step-by-Step Execution"
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="tinyllama",
        help="Model name/key (e.g., tinyllama, mistral-7b, qvikhr-3b)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing local documents",
    )

    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./storage",
        help="Directory for storing vector index",
    )

    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU acceleration"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode",
    )

    parser.add_argument(
        "--visible", action="store_false", dest="headless", help="Show browser window"
    )

    parser.add_argument(
        "--slow-mo", type=int, default=50, help="Slow down Playwright operations (ms)"
    )

    parser.add_argument(
        "--step-by-step", action="store_true", help="Enable step-by-step execution mode"
    )

    parser.add_argument(
        "--query", type=str, help="Single query to execute (non-interactive mode)"
    )

    parser.add_argument(
        "--no-tracing", action="store_true", help="Disable Phoenix tracing"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable automatic model download",
    )

    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()

    # Setup tracing
    setup_tracing(enable_phoenix=not args.no_tracing, log_level=args.log_level)

    try:
        if args.query:
            # Quick query mode
            response = await quick_query(
                args.query,
                model=args.model,  # Передаем model, не model_path
                data_dir=args.data_dir,
                persist_dir=args.persist_dir,
                use_gpu=not args.no_gpu,
                headless_browser=args.headless,
                playwright_slow_mo=args.slow_mo,
                step_by_step_mode=args.step_by_step,
                auto_download=not args.no_auto_download,
            )
            print("\n" + "=" * 60)
            print(response)
            print("=" * 60)
        else:
            # Interactive session mode
            agent = await create_new_session(
                model=args.model,  # Передаем model, не model_path
                data_dir=args.data_dir,
                persist_dir=args.persist_dir,
                use_gpu=not args.no_gpu,
                display_banner=True,
                headless_browser=args.headless,
                playwright_slow_mo=args.slow_mo,
                step_by_step_mode=args.step_by_step,
                auto_download=not args.no_auto_download,
            )

            await interactive_chat_session(agent)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
