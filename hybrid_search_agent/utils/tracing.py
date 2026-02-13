"""Tracing utilities for Phoenix and Loguru"""

import sys
from pathlib import Path
from loguru import logger

from trace_context import setup_logger, TraceContext, trace_function
from phoenix_client import setup_phoenix_tracing


def setup_tracing(
    enable_phoenix: bool = True,
    log_level: str = "INFO",
    log_file: str = "./logs/hybrid_agent.log",
):
    """Setup tracing and logging"""

    # Setup Loguru
    logger.remove()  # Remove default handler

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="1 week",
        compression="zip",
    )

    # Setup trace context
    setup_logger(logger)

    # Setup Phoenix
    phoenix_url = None
    if enable_phoenix:
        phoenix_url = setup_phoenix_tracing(auto_start_server=True)
        logger.info(f"Phoenix tracing enabled at {phoenix_url}")

    return phoenix_url


__all__ = ["setup_tracing", "TraceContext", "trace_function"]
