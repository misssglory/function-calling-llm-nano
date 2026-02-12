"""Core Hybrid Search Agent implementation"""

import os
import sys
import time
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import contextmanager
from datetime import datetime

from loguru import logger
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.workflow import Context
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from tool_engines import (
    LocalSearchEngine,
    DuckDuckGoSearchEngine,
    PlaywrightWebScraperEngine,
)
from trace_context import TraceContext, trace_function

from hybrid_search_agent.config import (
    CONTEXT_WINDOW,
    TAVILY_API_KEY,
    EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_BATCH_SIZE,
    TEMPERATURE,
    MAX_NEW_TOKENS,
    GPU_BATCH_SIZE,
    CPU_THREADS,
    MAX_ITERATIONS,
    ENABLE_PHOENIX_TRACING,
    DATA_DIR,
    STORAGE_DIR,
    LOGS_DIR,
    SCREENSHOTS_DIR,
    PDFS_DIR,
    MODELS_DIR,
)
from hybrid_search_agent.utils.model_utils import (
    get_model_manager,
    resolve_model_path,
    list_downloaded_models,
    list_available_models,
    verify_model_file,
)
from hybrid_search_agent.agents.step_by_step_agent import StepByStepAgent
from hybrid_search_agent.core.step_history import StepHistory
from phoenix_client import setup_phoenix_tracing

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


class HybridSearchAgent:
    """Combines local search, web search, and web scraping capabilities."""

    def __init__(
        self,
        model_path: str,  # Полный путь к файлу модели
        model_name: Optional[str] = None,  # Имя модели для логирования
        data_dir: str = str(DATA_DIR),
        persist_dir: str = str(STORAGE_DIR),
        use_gpu: bool = True,
        enable_phoenix_tracing: bool = ENABLE_PHOENIX_TRACING,
        headless_browser: bool = True,
        playwright_slow_mo: int = 50,
        step_by_step_mode: bool = False,
    ):
        start_time = time.perf_counter()
        self.session_id = str(uuid.uuid4())[:8]
        logger.bind(session_id=self.session_id).info("Initializing HybridSearchAgent")

        self.model_path = model_path
        self.model_name = model_name or Path(model_path).stem
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.use_gpu = use_gpu
        self.headless_browser = headless_browser
        self.playwright_slow_mo = playwright_slow_mo
        self.step_by_step_mode = step_by_step_mode

        logger.info(f"Model: {self.model_name}")
        logger.info(f"Model path: {self.model_path}")

        # Initialize Phoenix tracing
        self.phoenix_url = None
        if enable_phoenix_tracing:
            self.phoenix_url = setup_phoenix_tracing(auto_start_server=True)

        # Initialize step history for step-by-step mode
        if step_by_step_mode:
            self.step_history = StepHistory()
        else:
            self.step_history = None

        # Components to initialize in init()
        self.llm = None
        self.embed_model = None
        self.local_engine = None
        self.ddg_engine = None
        self.playwright_engine = None
        self.agent = None
        self.ctx = None

        logger.info(
            f"HybridSearchEngine initialized: {time.perf_counter() - start_time:.2f}s"
        )

    async def init(self):
        """Initialize all components asynchronously"""
        try:
            # Setup embedding and LLM in parallel
            setup_embedding_task = asyncio.create_task(self._setup_embedding_model())
            setup_llm_task = asyncio.create_task(
                self._setup_llm(self.model_path, self.use_gpu)
            )

            self.llm = await setup_llm_task
            Settings.llm = self.llm
            self.embed_model = await setup_embedding_task
            Settings.embed_model = self.embed_model
            Settings.chunk_size = CHUNK_SIZE
            Settings.chunk_overlap = CHUNK_OVERLAP

            # Setup search engines
            self.local_engine = LocalSearchEngine(self.data_dir, self.persist_dir)
            self.ddg_engine = DuckDuckGoSearchEngine()
            self.playwright_engine = PlaywrightWebScraperEngine(
                headless=self.headless_browser, slow_mo=self.playwright_slow_mo
            )
            await self.playwright_engine._setup_playwright(self.headless_browser)

            # Create agent
            self.agent, self.ctx = await self._create_agent()

            logger.bind(session_id=self.session_id).info(
                "HybridSearchAgent initialized successfully"
            )

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise

    @trace_function
    async def _setup_llm(self, model_path: str, use_gpu: bool) -> LlamaCPP:
        """Setup GGUF model with LlamaCPP with better error handling."""
        logger.info(f"Loading GGUF model from {model_path}")
        start_time = time.perf_counter()

        # Final verification before loading
        if not verify_model_file(model_path):
            logger.error(f"Model file verification failed: {model_path}")
            raise ValueError(f"Invalid or corrupted model file: {model_path}")

        gpu_layers = -1 if use_gpu else 0

        try:
            with TraceContext("load_llama_cpp_model"):
                llm = LlamaCPP(
                    model_path=model_path,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_NEW_TOKENS,
                    context_window=CONTEXT_WINDOW,
                    model_kwargs={
                        "n_gpu_layers": gpu_layers,
                        "n_batch": GPU_BATCH_SIZE,
                        "n_threads": CPU_THREADS,
                        "f16_kv": True,
                        "verbose": False,
                    },
                    generate_kwargs={},
                    verbose=False,
                )

                # Test the model with a simple prompt
                logger.info("Testing model with simple prompt...")
                test_response = llm.complete("Say 'OK'")
                logger.debug(f"Model test response: {str(test_response)[:50]}...")

                logger.success(
                    f"GGUF model loaded successfully: {Path(model_path).name}"
                )
                logger.info(f"Loading time: {time.perf_counter() - start_time:.2f}s")
                return llm

        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            logger.info("\nPossible solutions:")
            logger.info(
                "1. Check if the model file is corrupted - try downloading again"
            )
            logger.info(
                "2. For CPU: try a smaller model like 'tinyllama-1.1b-chat.Q4_K_M.gguf'"
            )
            logger.info("3. For GPU: ensure CUDA is properly installed")
            logger.info("4. Check available RAM/VRAM")

            # Suggest lightweight model
            if "QVikhr" in model_path:
                logger.info("\n💡 Suggestion: Try the lightweight TinyLlama model:")
                logger.info("   model_path = 'tinyllama-1.1b-chat.Q4_K_M.gguf'")

            raise

    @trace_function
    async def _setup_embedding_model(self) -> HuggingFaceEmbedding:
        """Setup embedding model with fallback options."""
        with TraceContext("load_embedding_model"):
            start_time = time.perf_counter()

            # Try primary embedding model
            try:
                embed_model = HuggingFaceEmbedding(
                    model_name=EMBEDDING_MODEL,
                    embed_batch_size=EMBED_BATCH_SIZE,
                )
                logger.info(f"Embedding model loaded: {embed_model.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {EMBEDDING_MODEL}: {e}")
                logger.info("Falling back to BAAI/bge-small-en-v1.5")

                # Fallback to a more reliable embedding model
                embed_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    embed_batch_size=EMBED_BATCH_SIZE,
                )

            logger.info(
                f"Embedding model loaded. Time: {time.perf_counter() - start_time:.2f}s"
            )
            return embed_model

    @trace_function
    async def _create_agent(self) -> tuple[ReActAgent, Context]:
        """Create agent with local search, web search, and scraping tools."""
        with TraceContext("create_agent"):
            # Get query engines
            local_query_engine = self.local_engine.get_query_engine()
            ddg_tools = self.ddg_engine.get_tools()
            playwright_tools = await self.playwright_engine.get_tools()

            logger.debug(f"Local query engine ready")
            logger.debug(f"DuckDuckGo tools: {len(ddg_tools)}")
            logger.debug(f"Playwright tools: {len(playwright_tools)}")

            # Create local search tool
            local_tool = QueryEngineTool(
                query_engine=local_query_engine,
                metadata=ToolMetadata(
                    name="local_document_search",
                    description=(
                        "Use this tool to search through local documents and files. "
                        "Useful for finding information from your own documents, reports, "
                        "PDF files, or text files. Use when you need specific information "
                        "from uploaded documents."
                    ),
                ),
            )

            # Combine all tools
            all_tools = [local_tool] + ddg_tools + playwright_tools

            # Create base agent
            base_agent = ReActAgent(
                tools=all_tools,
                llm=self.llm,
                verbose=True,
                max_iterations=MAX_ITERATIONS,
                system_prompt=self._get_system_prompt(),
            )

            # Wrap with step-by-step agent if in that mode
            if self.step_by_step_mode:
                from hybrid_search_agent.agents.step_by_step_agent import (
                    StepByStepAgent,
                )

                agent = StepByStepAgent(base_agent, self.step_history)
            else:
                agent = base_agent

            context = Context(agent)
            logger.info(f"Agent created with {len(all_tools)} tools")
            return (agent, context)

    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        # ... (keep existing system prompt)
        return (
            """You are a helpful AI assistant with access to three types of tools..."""
        )

    @trace_function
    async def query(self, question: str) -> str:
        """Send query to agent in standard mode."""
        if self.step_by_step_mode:
            raise ValueError("Use query_step_by_step() for step-by-step mode")

        query_id = str(uuid.uuid4())[:8]

        try:
            logger.bind(query_id=query_id, session_id=self.session_id).info(
                f"Processing query: {question[:100]}..."
            )

            with TraceContext(
                "agent_chat", query_id=query_id, question_length=len(question)
            ):
                handler = self.agent.run(question, ctx=self.ctx)

                async for ev in handler.stream_events():
                    if isinstance(ev, AgentStream):
                        print(f"{ev.delta}", end="", flush=True)

                response = await handler
                response_str = str(response)

            logger.bind(query_id=query_id, session_id=self.session_id).info(
                f"Query completed | Response length: {len(response_str)} chars"
            )

            return response_str

        except Exception as e:
            logger.bind(query_id=query_id, session_id=self.session_id).error(
                f"Error processing query: {e}"
            )
            return f"Sorry, an error occurred: {str(e)}"

    async def query_step_by_step(
        self, question: str, auto_execute: bool = False
    ) -> AsyncGenerator:
        """Execute query step by step with user control."""
        if not self.step_by_step_mode or not isinstance(self.agent, StepByStepAgent):
            raise ValueError("Agent not configured for step-by-step mode")

        self.agent.auto_execute = auto_execute

        async for event in self.agent.run_step_by_step(question, self.ctx):
            yield event

    @trace_function
    async def add_document(self, file_path: str) -> bool:
        """Add a new document to local search index."""
        # ... (keep existing add_document method)
        pass

    async def close_browser(self):
        """Close Playwright browser."""
        if hasattr(self, "playwright_engine") and self.playwright_engine:
            await self.playwright_engine.close()

    def get_phoenix_url(self) -> str:
        """Get Phoenix URL for viewing traces."""
        if self.phoenix_url:
            return self.phoenix_url
        return "Phoenix tracing not enabled"
