#!/usr/bin/env python3
"""
Phoenix Tracing Client
Клиент для подключения к запущенному Phoenix серверу и настройки трассировки.
"""
import os
import time
import subprocess
import sys
import socket
from pathlib import Path
from typing import Optional
from loguru import logger
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

def check_phoenix_server(host: str = "127.0.0.1", port: int = 6006, timeout: float = 0.5) -> bool:
    """
    Проверяет, запущен ли Phoenix сервер через сокет.
    
    Returns:
        bool: True если сервер доступен, иначе False
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def wait_for_phoenix_server(host: str = "127.0.0.1", port: int = 6006, max_attempts: int = 30) -> bool:
    """
    Ожидает запуска Phoenix сервера с повторными попытками.
    
    Returns:
        bool: True если сервер запустился, иначе False
    """
    logger.info(f"Ожидание запуска Phoenix сервера на {host}:{port}...")
    
    for attempt in range(max_attempts):
        if check_phoenix_server(host, port):
            logger.success(f"Phoenix сервер доступен (попытка {attempt + 1}/{max_attempts})")
            # Даем серверу дополнительное время для полной инициализации
            time.sleep(1)
            return True
        
        if attempt < max_attempts - 1:
            logger.debug(f"Сервер еще не доступен, ожидание... ({attempt + 1}/{max_attempts})")
            time.sleep(1)
    
    logger.error(f"Phoenix сервер не запустился после {max_attempts} попыток")
    return False

def setup_phoenix_tracing(
    host: str = "127.0.0.1",
    port: int = 6006,
    auto_start_server: bool = False,
    background: bool = True,
    wait_for_server: bool = True,
    max_wait_attempts: int = 30
) -> Optional[str]:
    """
    Настройка трассировки Phoenix с подключением к существующему серверу.
    
    Args:
        host: Хост Phoenix сервера
        port: Порт Phoenix сервера
        auto_start_server: Автоматически запускать сервер, если он не запущен
        background: Запускать сервер в фоновом режиме (если auto_start_server=True)
        wait_for_server: Ожидать запуска сервера
        max_wait_attempts: Максимальное количество попыток ожидания сервера
        
    Returns:
        str: URL Phoenix UI или None в случае ошибки
    """
    try:
        # Проверяем доступность сервера
        server_running = check_phoenix_server(host, port)
        
        if not server_running:
            if auto_start_server:
                logger.warning(f"Phoenix сервер не найден на {host}:{port}")
                logger.info("Запускаем Phoenix сервер в фоновом режиме...")
                
                server_script = Path(__file__).parent / "phoenix_server.py"
                if server_script.exists():
                    # Запускаем Phoenix сервер как отдельный процесс
                    cmd = [sys.executable, str(server_script), 
                           "--host", host, "--port", str(port)]
                    
                    if background:
                        cmd.append("--background")
                    
                    logger.info(f"Запуск команды: {' '.join(cmd)}")
                    
                    if sys.platform == "win32":
                        # Windows
                        process = subprocess.Popen(
                            cmd,
                            creationflags=subprocess.CREATE_NEW_CONSOLE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    else:
                        # Linux/Mac
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True
                        )
                    
                    logger.info(f"PID процесса: {process.pid}")
                    
                    if wait_for_server:
                        # Ждем запуска сервера
                        if not wait_for_phoenix_server(host, port, max_wait_attempts):
                            logger.error("Не удалось дождаться запуска Phoenix сервера")
                            return None
                    else:
                        # Даем немного времени на инициализацию
                        time.sleep(2)
                else:
                    logger.error(f"Скрипт сервера не найден: {server_script}")
                    logger.info(f"Создайте файл phoenix_server.py или запустите сервер вручную")
                    return None
            else:
                logger.warning(f"Phoenix сервер не доступен на {host}:{port}")
                logger.info(f"Запустите сервер вручную: python phoenix_server.py")
                return None
        elif wait_for_server:
            # Сервер запущен, но даем ему время на полную инициализацию
            logger.info("Phoenix сервер уже запущен, проверяем доступность...")
            if not wait_for_phoenix_server(host, port, 5):  # 5 попыток
                logger.warning("Сервер отвечает, но возможно не полностью инициализирован")
        
        # Настройка endpoint для трассировок
        endpoint = f"http://{host}:{port}/v1/traces"
        
        # Дополнительная проверка HTTP доступности
        try:
            import requests
            response = requests.get(f"http://{host}:{port}", timeout=2)
            if response.status_code == 200:
                logger.success(f"Phoenix UI доступен по адресу: http://{host}:{port}")
            else:
                logger.warning(f"Phoenix UI вернул статус: {response.status_code}")
        except Exception as e:
            logger.warning(f"Не удалось проверить UI Phoenix: {e}")
        
        # Настройка трассировки OpenTelemetry
        try:
            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint))
            )
            
            # Инструментируем LlamaIndex
            LlamaIndexInstrumentor().instrument(
                skip_dep_check=True,
                tracer_provider=tracer_provider
            )
            
            phoenix_url = f"http://{host}:{port}"
            logger.success(f"✓ Трассировка Phoenix настроена успешно")
            logger.info(f"📊 Phoenix UI: {phoenix_url}")
            logger.info(f"📡 Endpoint трейсов: {endpoint}")
            
            return phoenix_url
            
        except Exception as e:
            logger.error(f"Ошибка настройки трассировки OpenTelemetry: {e}")
            return None
        
    except ImportError as e:
        logger.warning(f"Не удалось настроить Phoenix трассировку: {e}")
        logger.warning("Установите зависимости: pip install arize-phoenix opentelemetry-exporter-otlp openinference-instrumentation-llama-index")
        return None
    except Exception as e:
        logger.warning(f"Не удалось настроить Phoenix трассировку: {e}")
        return None

# Функция для отложенной инициализации трассировки
def setup_phoenix_tracing_deferred(
    host: str = "127.0.0.1",
    port: int = 6006,
    auto_start_server: bool = True,
    background: bool = True
) -> Optional[str]:
    """
    Настройка трассировки с отложенной инициализацией.
    Не блокирует запуск приложения, если сервер недоступен.
    """
    try:
        # Пробуем подключиться без ожидания
        if check_phoenix_server(host, port):
            return setup_phoenix_tracing(
                host=host,
                port=port,
                auto_start_server=False,
                wait_for_server=False
            )
        elif auto_start_server:
            # Запускаем сервер в фоне и не ждем
            server_script = Path(__file__).parent / "phoenix_server.py"
            if server_script.exists():
                cmd = [sys.executable, str(server_script), 
                       "--host", host, "--port", str(port)]
                if background:
                    cmd.append("--background")
                
                if sys.platform == "win32":
                    subprocess.Popen(
                        cmd,
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True
                    )
                
                logger.info(f"Phoenix сервер запускается в фоне")
                # Не ждем, возвращаем URL
                return f"http://{host}:{port}"
        
        return None
    except Exception as e:
        logger.debug(f"Отложенная инициализация Phoenix: {e}")
        return None

# Пример использования
if __name__ == "__main__":
    # Настройка трассировки с автоматическим запуском сервера и ожиданием
    phoenix_url = setup_phoenix_tracing(
        host="127.0.0.1",
        port=6006,
        auto_start_server=True,
        background=True,
        wait_for_server=True,
        max_wait_attempts=30
    )
    
    if phoenix_url:
        print(f"✅ Phoenix запущен: {phoenix_url}")
    else:
        print("❌ Не удалось запустить Phoenix")