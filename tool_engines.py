from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
# from llama_index.core.agent import ReActAgent, FunctionAgent
# from llama_index.core.workflow import Context
# from llama_index.llms.llama_cpp import LlamaCPP
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
# from llama_index.core.agent.workflow import ToolCallResult, AgentStream
from pathlib import Path
from loguru import logger

from trace_context import trace_function, TraceContext
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.tools.playwright import PlaywrightToolSpec


class LocalSearchEngine:
    """Обрабатывает функционал поиска по локальным документам."""

    def __init__(self, data_dir: str = "./data", persist_dir: str = "./storage"):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.index = None
        self._setup_index()

    @trace_function
    def _setup_index(self):
        """Настройка или загрузка векторного индекса для локальных документов."""
        try:
            if self.persist_dir.exists():
                logger.info(f"Загружаем индекс из {self.persist_dir}")
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.persist_dir)
                )
                self.index = load_index_from_storage(storage_context)
                logger.info(f"Индекс загружен из {self.persist_dir}")
            else:
                logger.info(f"Создаем новый индекс из {self.data_dir}")

                # Загружаем документы
                if not self.data_dir.exists():
                    logger.warning(
                        f"Директория с данными {self.data_dir} не существует. Создаем пустой индекс."
                    )
                    self._create_empty_index()
                    return

                with TraceContext("load_documents"):
                    documents = SimpleDirectoryReader(
                        input_dir=str(self.data_dir),
                        recursive=True,
                        required_exts=[".txt", ".pdf", ".md", ".docx", ".pptx", ".csv"],
                    ).load_data()

                if documents:
                    with TraceContext("create_index"):
                        # Создаем индекс
                        self.index = VectorStoreIndex.from_documents(
                            documents,
                            transformations=[
                                SentenceSplitter(
                                    chunk_size=512,
                                    chunk_overlap=50,
                                )
                            ],
                        )

                        # Сохраняем индекс
                        self.index.storage_context.persist(
                            persist_dir=str(self.persist_dir)
                        )
                        logger.info(f"Индекс создан с {len(documents)} документами")
                else:
                    self._create_empty_index()

        except Exception as e:
            logger.error(f"Ошибка настройки индекса: {e}")
            self._create_empty_index()

    def _create_empty_index(self):
        """Создает пустой индекс, если документы недоступны."""
        with TraceContext("create_empty_index"):
            from llama_index.core import Document

            empty_doc = Document(text="Локальные документы недоступны.")
            self.index = VectorStoreIndex.from_documents([empty_doc])
            logger.warning("Создан пустой индекс - документы не найдены")

    @trace_function
    def get_query_engine(self):
        """Получить движок запросов для локальных документов."""
        if not self.index:
            self._setup_index()

        with TraceContext("create_query_engine"):
            query_engine = self.index.as_query_engine(
                similarity_top_k=3, response_mode="compact"
            )
            logger.debug("Создан локальный движок запросов")
            return query_engine


class DuckDuckGoSearchEngine:
    """Обрабатывает функционал веб-поиска через DuckDuckGo."""

    def __init__(self, translate_tool_description=False):
        """
        Инициализация DuckDuckGo поиска.
        DuckDuckGo не требует API ключей, это бесплатный и приватный поиск.
        """
        self.ddg_tools = None
        self._translate_tool_description = translate_tool_description
        self._setup_duckduckgo_search()

    @trace_function
    def _setup_duckduckgo_search(self):
        """Настройка инструментов поиска DuckDuckGo."""
        try:

            ddg_spec = DuckDuckGoSearchToolSpec()

            # Получаем список инструментов
            self.ddg_tools = ddg_spec.to_tool_list()

            if self._translate_tool_description:
                # Переводим описания инструментов на русский
                for tool in self.ddg_tools:
                    if hasattr(tool, "_metadata") and tool._metadata:
                        # Обновляем описание на русском
                        tool._metadata.description = self._translate_tool_description(
                            tool._metadata
                        )
                        logger.debug(f"Обновлено описание для: {tool._metadata.name}")

            logger.info(
                f"DuckDuckGo поиск инициализирован с {len(self.ddg_tools)} инструментами"
            )

        except ImportError as e:
            logger.error(f"Ошибка импорта llama_index.tools.duckduckgo.base: {e}")
            logger.info(
                "Установите необходимые зависимости: pip install llama-index-tools-duckduckgo"
            )
            raise
        except Exception as e:
            logger.error(f"Ошибка настройки DuckDuckGo Search: {e}")
            raise

    def _translate_tool_description(self, metadata) -> str:
        """Перевод описания инструмента на русский язык."""
        tool_name = metadata.name.lower()
        original_desc = metadata.description

        if "search" in tool_name or "duckduckgo" in original_desc.lower():
            return """Выполняет поиск в интернете через DuckDuckGo.
            
            Параметры:
            - query: Поисковый запрос (строка)
            - num_results: Количество возвращаемых результатов (по умолчанию 10, максимум 20)
            - region: Регион для поиска (например, 'ru-ru' для России, 'wt-wt' для международного)
            - safesearch: Безопасный поиск ('moderate', 'strict', 'off')
            - time: Период времени ('d' - день, 'w' - неделя, 'm' - месяц, 'y' - год)
            
            Возвращает список результатов поиска с заголовками, ссылками и описаниями.
            DuckDuckGo - приватный поисковик, не требует API ключей.
            Идеально для получения актуальной информации, новостей, фактов и справочных данных."""

        elif "news" in tool_name:
            return """Выполняет поиск новостей через DuckDuckGo.
            
            Параметры:
            - query: Поисковый запрос (строка)
            - num_results: Количество возвращаемых результатов
            
            Возвращает последние новости по теме."""

        elif "image" in tool_name:
            return """Выполняет поиск изображений через DuckDuckGo.
            
            Параметры:
            - query: Поисковый запрос (строка)
            - num_results: Количество возвращаемых изображений
            
            Возвращает ссылки на изображения по теме."""

        else:
            # Общий перевод для остальных инструментов
            translated = original_desc.replace("Search", "Поиск").replace(
                "search", "поиск"
            )
            translated = translated.replace("DuckDuckGo", "DuckDuckGo").replace(
                "duckduckgo", "DuckDuckGo"
            )
            return translated

    def get_tools(self):
        """Получить инструменты поиска DuckDuckGo."""
        if not self.ddg_tools:
            self._setup_duckduckgo_search()
        return self.ddg_tools


def create_sync_playwright_browser(headless=True):
    browser = asyncio.get_event_loop().run_until_complete(
        PlaywrightToolSpec.create_async_playwright_browser(headless=headless)
    )
    return browser


class PlaywrightWebScraperEngine:
    """Обрабатывает веб-скрапинг и навигацию через Playwright."""

    def __init__(
        self, headless: bool = True, slow_mo: int = 50, translate_tool_description=False
    ):
        """
        Инициализация Playwright инструментов.

        Args:
            headless: Запускать браузер в фоновом режиме
            slow_mo: Замедление операций в миллисекундах
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.playwright_tools = None
        self._translate_tool_description = translate_tool_description
        # self._setup_playwright()

    @trace_function
    async def _setup_playwright(self, headless):
        """Настройка инструментов Playwright."""
        try:
            # self.browser = create_sync_playwright_browser(self.headless)
            logger.debug(f"Setup browser. Headless: {headless}")
            self.browser = await PlaywrightToolSpec.create_async_playwright_browser(
                headless=headless
            )

            playwright_spec = PlaywrightToolSpec.from_async_browser(self.browser)
            self.playwright_tools = playwright_spec.to_tool_list()
            logger.debug(f"Playwright tools: {self.playwright_tools}")
            logger.debug(f"Playwright tools: {len(self.playwright_tools)}")

            for tool in self.playwright_tools:
                if hasattr(tool, "_metadata") and tool._metadata:
                    # Обновляем описание на русском
                    logger.debug(f"Описание инструмента: {tool._metadata}")
                    if self._translate_tool_description:
                        tool._metadata.description = self._translate_tool_description(
                            tool._metadata
                        )
                    logger.debug(f"Обновлено описание для: {tool._metadata.name}")

            logger.info(
                f"Playwright инструменты инициализированы с {len(self.playwright_tools)} инструментами"
            )

        except ImportError as e:
            logger.error(f"Ошибка импорта llama_index.tools.playwright.base: {e}")
            logger.info(
                """
            Установите необходимые зависимости:
            pip install llama-index-tools-playwright playwright
            playwright install chromium
            """
            )
            raise
        except Exception as e:
            logger.error(f"Ошибка настройки Playwright: {e}")
            raise

    def _translate_tool_description(self, metadata) -> str:
        """Перевод описания инструмента Playwright на русский язык."""
        tool_name = metadata.name.lower()
        original_desc = metadata.description

        if "navigate" in tool_name or "goto" in tool_name:
            return """Переходит по указанному URL в браузере.
            
            Параметры:
            - url: Полный URL веб-страницы для перехода (обязательно)
            - timeout: Таймаут ожидания загрузки в миллисекундах (по умолчанию 30000)
            - wait_until: Состояние загрузки ('load', 'domcontentloaded', 'networkidle')
            
            Возвращает заголовок страницы и статус загрузки.
            Используйте для навигации по веб-сайтам."""

        elif "click" in tool_name:
            return """Нажимает на элемент веб-страницы по селектору.
            
            Параметры:
            - selector: CSS или XPath селектор элемента (обязательно)
            - timeout: Таймаут ожидания элемента в миллисекундах
            
            Возвращает результат клика.
            Используйте для взаимодействия с кнопками и ссылками."""

        elif "type" in tool_name or "fill" in tool_name or "input" in tool_name:
            return """Вводит текст в поле ввода на веб-странице.
            
            Параметры:
            - selector: CSS или XPath селектор поля ввода (обязательно)
            - text: Текст для ввода (обязательно)
            - timeout: Таймаут ожидания элемента
            
            Возвращает подтверждение ввода текста.
            Используйте для заполнения форм и поисковых запросов."""

        elif "extract" in tool_name or "scrape" in tool_name or "get_text" in tool_name:
            return """Извлекает текст и содержимое с текущей веб-страницы.
            
            Параметры:
            - selector: Опциональный CSS селектор для извлечения конкретных элементов
            - include_html: Включать HTML в результат (по умолчанию False)
            
            Возвращает текст и заголовок страницы.
            Используйте для получения содержимого веб-страниц."""

        elif "screenshot" in tool_name:
            return """Делает скриншот текущей веб-страницы.
            
            Параметры:
            - path: Путь для сохранения скриншота (обязательно)
            - full_page: Делать скриншот всей страницы (по умолчанию True)
            
            Возвращает путь к сохраненному скриншоту."""

        elif "pdf" in tool_name:
            return """Сохраняет текущую веб-страницу как PDF.
            
            Параметры:
            - path: Путь для сохранения PDF файла (обязательно)
            
            Возвращает путь к сохраненному PDF."""

        elif "wait" in tool_name:
            return """Ожидает появления элемента на странице или таймаута.
            
            Параметры:
            - selector: CSS селектор элемента для ожидания
            - timeout: Максимальное время ожидания в миллисекундах
            
            Возвращает статус ожидания."""

        elif "evaluate" in tool_name:
            return """Выполняет JavaScript код на странице.
            
            Параметры:
            - script: JavaScript код для выполнения (обязательно)
            
            Возвращает результат выполнения скрипта.
            Используйте для сложных взаимодействий с динамическим контентом."""

        elif "close" in tool_name:
            return """Закрывает браузер и освобождает ресурсы.
            
            Вызывайте этот инструмент после завершения работы с веб-страницами."""

        else:
            # Общий перевод
            translated = original_desc.replace("Navigate", "Перейти").replace(
                "Click", "Нажать"
            )
            translated = translated.replace("Type", "Ввести").replace(
                "Extract", "Извлечь"
            )
            translated = translated.replace("Screenshot", "Скриншот").replace(
                "PDF", "PDF"
            )
            translated = translated.replace("Wait", "Ожидать").replace(
                "Evaluate", "Выполнить"
            )
            return translated

    async def get_tools(self):
        """Получить инструменты Playwright."""
        if not self.playwright_tools:
            self._setup_playwright(self.headless)
        return self.playwright_tools
