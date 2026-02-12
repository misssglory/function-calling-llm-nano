#!/usr/bin/env python3
"""
Phoenix Tracing Server
Запускает Phoenix сервер для трассировки.
"""
import os
import sys
import time
import argparse
import socket
import webbrowser
from pathlib import Path
from loguru import logger

# Настройка логирования
logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    level="DEBUG",
)
logger.add(
    "logs/phoenix_server_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="DEBUG",
)


class PhoenixServer:
    """Управление Phoenix сервером."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6006):
        self.host = host
        self.port = port
        self.ui_port = 6006  # UI порт
        self.trace_port = (
            6007  # Порт для трассировки (иногда Phoenix использует другой порт)
        )
        self.session = None
        self.running = False

    def start(self) -> tuple:
        """
        Запускает Phoenix сервер.

        Returns:
            tuple: (ui_url, trace_endpoint)
        """
        try:
            import phoenix as px

            # Устанавливаем переменные окружения для Phoenix
            os.environ["PHOENIX_HOST"] = self.host
            os.environ["PHOENIX_PORT"] = str(self.port)
            os.environ["PHOENIX_UI_PORT"] = str(self.ui_port)

            logger.info(f"Запуск Phoenix сервера...")
            logger.info(f"  Host: {self.host}")
            logger.info(f"  UI Port: {self.ui_port}")

            # Запускаем Phoenix приложение
            self.session = px.launch_app(
                host=self.host,
                port=self.ui_port,  # UI порт
                run_in_thread=False,  # Блокирующий режим
            )

            self.running = True

            # Формируем URL
            ui_url = f"http://{self.host}:{self.ui_port}"
            trace_endpoint = f"http://{self.host}:{self.ui_port}/v1/traces"

            logger.success(f"✓ Phoenix сервер запущен успешно!")
            logger.info(f"📊 UI Dashboard: {ui_url}")
            logger.info(f"📡 Traces Endpoint: {trace_endpoint}")

            return ui_url, trace_endpoint

        except ImportError as e:
            logger.error(f"Ошибка импорта Phoenix: {e}")
            logger.info("Установка: pip install arize-phoenix")
            raise
        except Exception as e:
            logger.error(f"Ошибка запуска Phoenix сервера: {e}")
            raise

    def start_background(self) -> tuple:
        """Запускает Phoenix сервер в фоновом режиме."""
        try:
            import phoenix as px

            # Устанавливаем переменные окружения
            os.environ["PHOENIX_HOST"] = self.host
            os.environ["PHOENIX_PORT"] = str(self.port)
            os.environ["PHOENIX_UI_PORT"] = str(self.ui_port)

            logger.info(f"Запуск Phoenix сервера в фоновом режиме...")

            self.session = px.launch_app(
                host=self.host, port=self.ui_port, run_in_thread=True  # Фоновый режим
            )

            self.running = True

            ui_url = f"http://{self.host}:{self.ui_port}"
            trace_endpoint = f"http://{self.host}:{self.ui_port}/v1/traces"

            logger.success(f"✓ Phoenix сервер запущен в фоновом режиме")
            logger.info(f"📊 UI Dashboard: {ui_url}")
            logger.info(f"📡 Traces Endpoint: {trace_endpoint}")

            # Проверяем доступность UI
            time.sleep(2)
            if self._check_ui_available():
                logger.success(f"✓ UI доступен по адресу: {ui_url}")
            else:
                logger.warning(
                    f"⚠ UI пока не доступен, возможно требуется время для запуска"
                )

            return ui_url, trace_endpoint

        except Exception as e:
            logger.error(f"Ошибка запуска Phoenix сервера в фоновом режиме: {e}")
            raise

    def _check_ui_available(self) -> bool:
        """Проверяет доступность UI."""
        try:
            import requests

            response = requests.get(f"http://{self.host}:{self.ui_port}", timeout=2)
            return response.status_code == 200
        except:
            return False

    def stop(self):
        """Останавливает Phoenix сервер."""
        if self.session:
            logger.info("Остановка Phoenix сервера...")
            self.session = None
            self.running = False
            logger.info("✓ Phoenix сервер остановлен")


def main():
    parser = argparse.ArgumentParser(description="Phoenix Tracing Server")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Хост для сервера"
    )
    parser.add_argument("--port", type=int, default=6006, help="Порт для UI")
    parser.add_argument(
        "--background", action="store_true", help="Запустить в фоновом режиме"
    )
    parser.add_argument(
        "--open-browser", action="store_true", help="Открыть браузер после запуска"
    )

    args = parser.parse_args()

    # Создаем директории
    Path("./logs").mkdir(exist_ok=True)
    Path("./phoenix_storage").mkdir(exist_ok=True)

    # Устанавливаем рабочую директорию для Phoenix
    os.environ["PHOENIX_WORKING_DIR"] = str(Path("./phoenix_storage").absolute())

    # Создаем и запускаем сервер
    server = PhoenixServer(host=args.host, port=args.port)

    try:
        if args.background:
            ui_url, _ = server.start_background()
        else:
            ui_url, _ = server.start()

        # Открываем браузер если нужно
        if args.open_browser and ui_url:
            logger.info(f"Открываем браузер: {ui_url}")
            webbrowser.open(ui_url)

        # Держим сервер запущенным
        if args.background:
            logger.info(
                "Сервер запущен в фоновом режиме. Нажмите Ctrl+C для остановки."
            )

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nПолучен сигнал остановки...")
        server.stop()
        logger.info("Phoenix сервер завершил работу")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        server.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
