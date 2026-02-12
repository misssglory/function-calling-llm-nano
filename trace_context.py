from loguru import logger
import time
import uuid
import sys


class TraceContext:
    """Менеджер контекста для трассировки операций."""

    def __init__(self, operation_name: str, **kwargs):
        self.operation_name = operation_name
        self.kwargs = kwargs
        self.start_time = None
        self.trace_id = str(uuid.uuid4())[:8]

    def __enter__(self):
        self.start_time = time.time()
        logger.bind(trace_id=self.trace_id).info(
            f"Начинаем операцию: {self.operation_name}", extra={"kwargs": self.kwargs}
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type:
            logger.bind(trace_id=self.trace_id).error(
                f"Ошибка операции: {self.operation_name} | Ошибка: {exc_val} | Время: {elapsed:.2f}с"
            )
        else:
            logger.bind(trace_id=self.trace_id).info(
                f"Завершена операция: {self.operation_name} | Время: {elapsed:.2f}с"
            )


def trace_function(func):
    """Декоратор для трассировки выполнения функций."""

    def wrapper(*args, **kwargs):
        func_name = func.__name__
        class_name = args[0].__class__.__name__ if args else ""

        with TraceContext(
            f"{class_name}.{func_name}" if class_name else func_name,
            args=len(args),
            kwargs_count=len(kwargs),
        ):
            return func(*args, **kwargs)

    return wrapper


def setup_logger(logger):
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file.path}:{line}</cyan>:<green>{function}</green> | {message}",
        level="DEBUG",
    )
    logger.add(
        "logs/hybrid_search_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {file.path}:{line}:{function} | {message}",
        level="DEBUG",
        compression="zip",
    )
