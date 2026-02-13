"""Interactive chat session management for Google Colab"""

import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

from loguru import logger

# Colab-specific imports
try:
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

from hybrid_search_agent.core.hybrid_agent import HybridSearchAgent
from hybrid_search_agent.utils.setup import (
    prepare_data_directory,
    display_welcome_banner,
)
from hybrid_search_agent.utils.display import (
    display_step_by_step_instructions,
    display_standard_instructions,
    show_execution_plan,
    show_history,
)
from hybrid_search_agent.models.step_models import StepStatus
from hybrid_search_agent.config import DEFAULT_MODEL_PATH
from trace_context import TraceContext
from hybrid_search_agent.utils.model_utils import resolve_model_path


def colab_input(prompt: str = "") -> str:
    """Cross-compatible input function for both Colab and standard Python"""
    if IN_COLAB:
        from google.colab import output

        # Use output.eval_js for non-blocking input in Colab
        try:
            return output.eval_js(f'prompt("{prompt}")') or ""
        except:
            # Fallback to standard input
            return input(prompt)
    else:
        return input(prompt)


async def create_new_session(
    model: str = "tinyllama",
    data_dir: str = "./data",
    persist_dir: str = "./storage",
    use_gpu: bool = True,
    display_banner: bool = True,
    headless_browser: bool = True,
    playwright_slow_mo: int = 50,
    step_by_step_mode: bool = False,
    auto_download: bool = True,
    colab_mode: bool = False,
) -> HybridSearchAgent:
    """
    Create a new hybrid search agent session with step-by-step execution mode.
    Now with Google Colab compatibility.

    Args:
        model: Model name/key or path to GGUF model file
        data_dir: Directory containing local documents
        persist_dir: Directory for storing vector index
        use_gpu: Whether to use GPU acceleration
        display_banner: Whether to show welcome banner
        headless_browser: Run browser in headless mode
        playwright_slow_mo: Slow down Playwright operations (ms)
        step_by_step_mode: Enable step-by-step execution mode
        auto_download: Automatically download model if not found
        colab_mode: Enable Colab-specific optimizations

    Returns:
        Initialized HybridSearchAgent instance
    """

    try:
        # Colab-specific setup
        if IN_COLAB or colab_mode:
            print("📒 Running in Google Colab mode")

            # Mount Google Drive for persistent storage
            try:
                from google.colab import drive

                drive.mount("/content/drive")

                # Use Drive for persistent storage if available
                if Path("/content/drive/MyDrive").exists():
                    persist_dir = "/content/drive/MyDrive/hybrid_search_storage"
                    Path(persist_dir).mkdir(parents=True, exist_ok=True)
                    print(f"💾 Using Google Drive for persistence: {persist_dir}")
            except:
                print("ℹ️ Google Drive not mounted, using local storage")

            # Setup Playwright in Colab
            if headless_browser:
                print("🔄 Installing Playwright browsers (this may take a minute)...")
                import subprocess

                subprocess.run(
                    ["playwright", "install", "chromium"],
                    capture_output=True,
                    text=True,
                )
                print("✅ Playwright browsers installed")

        # Prepare data directory
        prepare_data_directory()

        # Display banner if requested
        if display_banner:
            display_welcome_banner()

        logger.info(f"Creating new Hybrid Search Agent session with Playwright...")
        logger.info(f"Step-by-step mode: {'ON' if step_by_step_mode else 'OFF'}")
        logger.info(f"Model: {model}")

        # Resolve model path with Colab optimizations
        try:
            resolved_model_path = resolve_model_path(model, auto_download=auto_download)
            logger.info(f"Resolved model path: {resolved_model_path}")
        except FileNotFoundError as e:
            if auto_download:
                logger.error(f"Failed to resolve/download model: {e}")
            else:
                logger.error(f"Model not found and auto_download is disabled: {e}")
            print("\n" + "=" * 60)
            print("❌ MODEL NOT FOUND")
            print("=" * 60)
            print(f"\nCould not find model: {model}")
            print("\n📦 Downloaded models:")
            print("\n💡 Solutions:")
            print("1. Enable auto_download=True (default)")
            print("2. Download a model manually:")
            print("   from hybrid_search_agent.utils.model_utils import download_model")
            print("   download_model('tinyllama')")
            print("3. Specify full path to your existing model")
            print("=" * 60)
            raise

        start_time = time.perf_counter()

        with TraceContext("initialize_hybrid_search_agent"):
            agent = HybridSearchAgent(
                model_path=resolved_model_path,
                model_name=model,
                data_dir=data_dir,
                persist_dir=persist_dir,
                use_gpu=use_gpu,
                headless_browser=headless_browser,
                playwright_slow_mo=playwright_slow_mo,
                step_by_step_mode=step_by_step_mode,
                # Add Colab-specific optimizations
                colab_mode=(IN_COLAB or colab_mode),
            )

            await agent.init()

        logger.success(
            f"Agent session created successfully! Time: {time.perf_counter() - start_time:.2f}s"
        )

        # Show instructions based on mode
        if step_by_step_mode:
            display_step_by_step_instructions()
        else:
            display_standard_instructions()

        return agent

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise

    except ValueError as e:
        logger.error(f"Model error: {e}")
        print("\n" + "=" * 60)
        print("❌ MODEL ERROR")
        print("=" * 60)
        print(f"\nError loading model: {e}")
        print("\n💡 Try using a smaller model for testing:")
        print("   agent = await create_new_session(model='tinyllama')")
        print("=" * 60)
        raise

    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise


async def interactive_chat_session(agent: HybridSearchAgent):
    """Start an interactive chat session with support for Google Colab"""

    chat_history = []
    session_start = datetime.now()
    logger.info(
        f"Chat session started at: {session_start.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print("\n" + "=" * 60)
    print("💬 INTERACTIVE CHAT MODE")
    if IN_COLAB:
        print("🔄 Running in Google Colab - press Ctrl+M then Enter to interrupt")
    if agent.step_by_step_mode:
        print("⚡ Step-by-step execution: ENABLED")
        print("   Step-by-step commands:")
        print("   - /next - execute next step")
        print("   - /skip - skip current step")
        print("   - /auto - toggle auto-execution")
        print("   - /plan - show current plan")
        print("   - /history - show history")
        print("   - /resume - resume paused execution")
    print("=" * 60 + "\n")

    try:
        while True:
            try:
                # Get user input - Colab compatible
                user_input = colab_input("\n🎯 Your question: ").strip()

                # Handle empty input (especially in Colab)
                if not user_input:
                    continue

                # Handle exit commands
                if user_input.lower() in ["quit", "exit", "q"]:
                    logger.info("Session ended by user")
                    await agent.close_browser()
                    break

                # Handle clear command
                if user_input.lower() in ["clear", "reset"]:
                    with TraceContext("reset_agent"):
                        agent.agent.reset()
                        chat_history = []
                    logger.info("Chat history cleared")
                    print("✅ Chat history cleared")
                    continue

                # Handle add document command
                if user_input.lower().startswith("add "):
                    file_path = user_input[4:].strip()

                    # Handle Colab file upload
                    if IN_COLAB and file_path.lower() == "upload":
                        from google.colab import files

                        print("📁 Please upload your file...")
                        uploaded = files.upload()
                        if uploaded:
                            file_path = next(iter(uploaded.keys()))
                            print(f"✅ File uploaded: {file_path}")

                    if Path(file_path).exists():
                        logger.info(f"Attempting to add document: {file_path}")
                        success = await agent.add_document(file_path)
                        if success:
                            logger.info(f"Document added: {file_path}")
                            print(f"✅ Document added: {file_path}")
                        else:
                            logger.error(f"Failed to add document: {file_path}")
                            print(f"❌ Failed to add document: {file_path}")
                    else:
                        logger.error(f"File not found: {file_path}")
                        print(f"❌ File not found: {file_path}")
                        if IN_COLAB:
                            print("   Use 'add upload' to upload files in Colab")
                    continue

                # Handle special commands in step-by-step mode
                if agent.step_by_step_mode and user_input.startswith("/"):
                    await _handle_step_command(agent, user_input, chat_history)
                    continue

                # Process query
                if agent.step_by_step_mode:
                    await _handle_step_by_step_query(agent, user_input, chat_history)
                else:
                    await _handle_standard_query(agent, user_input, chat_history)

            except KeyboardInterrupt:
                logger.warning("Session interrupted by user")
                await agent.close_browser()
                break
            except EOFError:  # Colab specific
                logger.warning("Input stream closed")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"❌ Error: {e}")

    finally:
        # Ensure browser is closed
        await agent.close_browser()
        session_duration = datetime.now() - session_start
        logger.info(f"Session duration: {session_duration}")
        print(f"\n👋 Session ended. Duration: {session_duration}")


async def _handle_step_command(
    agent: HybridSearchAgent, command: str, chat_history: list
):
    """Handle special commands in step-by-step mode"""

    cmd = command[1:].lower()

    if cmd == "next":
        if hasattr(agent.agent, "resume_execution"):
            await agent.agent.resume_execution()
        print("▶️ Continuing execution...")

    elif cmd == "skip":
        if hasattr(agent.agent, "skip_current_step"):
            await agent.agent.skip_current_step()
        print("⏭️ Skipping current step...")

    elif cmd == "auto":
        if hasattr(agent.agent, "toggle_auto_execute"):
            auto_mode = await agent.agent.toggle_auto_execute()
            print(
                f"{'▶️' if auto_mode else '⏸️'} Auto-execution: {'ON' if auto_mode else 'OFF'}"
            )

    elif cmd == "plan":
        await show_execution_plan(agent)

    elif cmd == "history":
        await show_history(agent)

    elif cmd == "resume":
        if hasattr(agent.agent, "resume_execution"):
            await agent.agent.resume_execution()
        print("▶️ Resuming execution...")

    elif cmd == "stop":
        if hasattr(agent.agent, "stop_execution"):
            await agent.agent.stop_execution()
        print("⏹️ Stopping execution...")

    else:
        print(f"❌ Unknown command: {command}")


async def _handle_step_by_step_query(
    agent: HybridSearchAgent, query: str, chat_history: list
):
    """Handle query in step-by-step mode with Colab compatibility"""

    print("\n" + "=" * 60)
    print("📋 EXECUTION PLANNING")
    print("=" * 60)

    auto_mode = False

    async for event in agent.query_step_by_step(query, auto_execute=auto_mode):
        event_type = event.get("type")

        if event_type == "plan_created":
            plan = event["plan"]
            print(f"\n📋 Plan ID: {plan.id}")
            print(f"📝 Query: {plan.query[:100]}...")

        elif event_type == "step_created":
            step = event["step"]
            print(f"\n📌 Step {step.id}: {step.description}")

        elif event_type == "step_start":
            step = event["step"]
            print(f"\n⚡ Executing: {step.description}")

        elif event_type == "step_complete":
            step = event["step"]
            result = event["result"]

            print(f"\n✅ Step {step.id} completed successfully!")

            logger.info(f"Step: {event['step']}")
            if result and len(str(result)) > 200:
                logger.info(f"📊 Result: {str(result)[:200]}...")
            elif result:
                logger.info(f"📊 Result: {result}")

            if step.tool_calls:
                print(f"🔧 Tools used:")
                for tool_call in step.tool_calls[:3]:
                    print(f"   - {tool_call['tool_name']}")
                if len(step.tool_calls) > 3:
                    print(f"     ... and {len(step.tool_calls) - 3} more")

            chat_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "step_id": step.id,
                    "step_description": step.description,
                    "result": str(result)[:500] if result else None,
                }
            )

        elif event_type == "step_failed":
            step = event["step"]
            error = event["error"]

            print(f"\n❌ Step {step.id} failed!")
            print(f"   Error: {error}")

        elif event_type == "step_pause":
            step = event["step"]

            if not auto_mode:
                print(f"\n⏸️ Step {step.id} completed. Continue?")
                print("   [Enter] - continue")
                print("   [n] - next step")
                print("   [s] - skip step")
                print("   [a] - enable auto-execution")
                print("   [p] - show plan")
                print("   [h] - show history")
                print("   [q] - stop execution")

                # Use Colab-compatible input
                cmd = colab_input("Command: ").strip().lower()

                if cmd == "q":
                    print("⏹️ Execution stopped by user")
                    break
                elif cmd == "s":
                    print("⏭️ Step skipped")
                    if hasattr(agent.agent, "skip_step"):
                        agent.agent.skip_step = True
                elif cmd == "a":
                    auto_mode = True
                    if hasattr(agent.agent, "toggle_auto_execute"):
                        await agent.agent.toggle_auto_execute()
                    print("▶️ Auto-execution ENABLED")
                elif cmd == "p":
                    await show_execution_plan(agent)
                elif cmd == "h":
                    await show_history(agent)

        elif event_type == "ask_continue":
            step = event["step"]
            error = event["error"]

            print(f"\n⚠️ Error on step {step.id}: {error}")
            print("Continue execution? (y/n): ", end="")
            sys.stdout.flush()  # Important for Colab

            decision = colab_input().strip().lower()
            if decision == "y":
                event["decision"] = {"continue": True}
            else:
                event["decision"] = {"continue": False}
                break

        elif event_type == "plan_complete":
            plan = event["plan"]
            print("\n" + "=" * 60)
            print("✅ PLAN COMPLETED SUCCESSFULLY!")
            print(f"📈 Total steps: {len(plan.steps)}")
            print(f"✅ Completed: {plan.completed_steps}")
            print(f"❌ Failed: {plan.failed_steps}")
            if plan.completed_at and plan.created_at:
                duration = (plan.completed_at - plan.created_at).total_seconds()
                print(f"⏱️ Execution time: {duration:.1f}s")
            print("=" * 60)

            print("\n📋 Execution summary:")
            for step in plan.steps:
                status_icon = (
                    "✅"
                    if step.status == StepStatus.COMPLETED
                    else (
                        "❌"
                        if step.status == StepStatus.FAILED
                        else "⏭️" if step.status == StepStatus.SKIPPED else "⏳"
                    )
                )
                duration = f" ({step.duration:.1f}s)" if step.duration else ""
                print(f"{status_icon} {step.description}{duration}")


async def _handle_standard_query(
    agent: HybridSearchAgent, query: str, chat_history: list
):
    """Handle query in standard mode with Colab compatibility"""
    logger.info("Processing query...")

    with TraceContext("process_user_query", question=query[:50]):
        response = await agent.query(query)

    print("\n" + "=" * 60)
    print(
        f"💡 Answer: {response[:500]}..."
        if len(response) > 500
        else f"💡 Answer: {response}"
    )
    print("=" * 60)

    chat_history.append(
        {
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "answer": response[:500],
        }
    )


async def quick_query(question: str, **kwargs) -> str:
    """Quick one-off query without interactive session."""
    colab_mode = kwargs.pop("colab_mode", IN_COLAB)

    agent = await create_new_session(
        display_banner=False, colab_mode=colab_mode, **kwargs
    )
    try:
        start_time = time.perf_counter()
        response = await agent.query(question)
        await agent.close_browser()
        logger.info(f"Response time: {time.perf_counter() - start_time:.2f}s")
        return response
    except Exception as e:
        logger.trace(f"Error in quick query: {e}")
        return f"Error: {str(e)}"
    finally:
        await agent.close_browser()


# Colab-specific helper functions
async def setup_colab_environment():
    """One-time setup for Google Colab environment"""
    if not IN_COLAB:
        print("❌ This function is only for Google Colab")
        return

    print("🔄 Setting up environment for Google Colab...")

    import subprocess
    import sys

    packages = [
        "playwright",
        "nest-asyncio",
        "ipywidgets",
    ]

    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    subprocess.run(
        ["playwright", "install", "chromium"], capture_output=True, text=True
    )

    print("✅ Colab environment setup complete!")


def run_chat_session(agent):
    """Synchronous wrapper for running chat session in Colab"""
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(interactive_chat_session(agent))
    except RuntimeError:
        # If we're already in an event loop
        asyncio.create_task(interactive_chat_session(agent))
