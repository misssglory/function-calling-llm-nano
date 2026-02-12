"""Display utilities for console output"""

from loguru import logger

from hybrid_search_agent.models.step_models import StepStatus


def display_welcome_banner():
    """Display welcome banner."""
    banner = """
    =================================================================
          Hybrid Search Agent: Local Documents + DuckDuckGo + Playwright   
                          With Phoenix Tracing                     
    =================================================================
    
    Features:
    ✓ Local document search
    ✓ Web search via DuckDuckGo (no API keys)
    ✓ Web scraping and navigation via Playwright
    
    Playwright tools:
    - Navigate to URL
    - Extract text from pages
    - Click on elements
    - Form filling
    - Screenshots and PDFs
    - Execute JavaScript
    """
    logger.info(banner)
    print(banner)


def display_step_by_step_instructions():
    """Display instructions for step-by-step mode"""
    instructions = """
    =================================================================
                    STEP-BY-STEP EXECUTION MODE
    =================================================================
    
    Mode features:
    
    1️⃣ PLANNING
       - System automatically breaks down your request into logical steps
       - Each step is one specific action with a tool
    
    2️⃣ EXECUTION CONTROL
       - See each step before execution
       - Skip, continue, or stop steps
       - View results of each step separately
    
    3️⃣ DEBUGGING AND HISTORY
       - Detailed tool call information
       - History of all executions saved
       - View previous execution plans
    
    Commands during execution:
    - [Enter] - continue to next step
    - [n] - next step
    - [s] - skip current step
    - [a] - enable auto-execution
    - [p] - show current plan
    - [h] - show execution history
    - [q] - stop execution
    
    Example requests for step-by-step mode:
    🔍 "Find recent AI advancements, open the top link, and save a screenshot"
    📄 "Find information about projects in documents, then search for similar solutions online"
    🌐 "Go to iqdoc.ai homepage, extract text and hyperlinks"
    
    =================================================================
    """
    print(instructions)


def display_standard_instructions():
    """Display instructions for standard mode"""
    instructions = """
    =================================================================
                    STANDARD EXECUTION MODE
    =================================================================
    
    You can ask questions normally. The system will execute them entirely.
    
    For step-by-step mode use:
    create_new_session(step_by_step_mode=True)
    
    =================================================================
    """
    print(instructions)


async def show_execution_plan(agent):
    """Show current execution plan"""
    if agent.step_history and agent.step_history.current_plan:
        plan = agent.step_history.current_plan
        
        print("\n" + "="*60)
        print("📋 CURRENT EXECUTION PLAN")
        print("="*60)
        print(f"Plan ID: {plan.id}")
        print(f"Query: {plan.query[:100]}...")
        print(f"Created: {plan.created_at.strftime('%H:%M:%S')}")
        print()
        
        for i, step in enumerate(plan.steps, 1):
            status_icon = {
                StepStatus.PENDING: "⏳",
                StepStatus.RUNNING: "⚡",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️"
            }.get(step.status, "⏳")
            
            print(f"{i}. {status_icon} {step.description}")
            
            if step.status == StepStatus.COMPLETED and step.result:
                result_preview = str(step.result)[:100] + "..." if len(str(step.result)) > 100 else str(step.result)
                print(f"   📊 Result: {result_preview}")
            if step.status == StepStatus.FAILED and step.error:
                print(f"   ❌ Error: {step.error[:100]}...")
            if step.tool_calls:
                print(f"   🔧 Tools: {', '.join([t['tool_name'] for t in step.tool_calls[:3]])}")
        
        print("="*60)
    else:
        print("\n📭 No active execution plan")


async def show_history(agent):
    """Show execution history"""
    if not agent.step_history:
        print("History not available in current mode")
        return
    
    history = agent.step_history.get_history(limit=5)
    
    if not history:
        print("\n📭 Execution history is empty")
        return
    
    print("\n" + "="*60)
    print("📚 EXECUTION HISTORY")
    print("="*60)
    
    for i, plan_summary in enumerate(history, 1):
        print(f"\n{i}. Query: {plan_summary['query'][:80]}...")
        print(f"   Time: {plan_summary['created_at']}")
        print(f"   Steps: {plan_summary['total_steps']} "
              f"(✅ {plan_summary['completed']}, "
              f"❌ {plan_summary['failed']}, "
              f"⏭️ {plan_summary.get('skipped', 0)})")
        
        # Show first few steps
        if i <= 3:  # Show details for last 3 plans
            print(f"   Steps:")
            for step in plan_summary['steps'][:3]:
                status_icon = "✅" if step['status'] == 'completed' else "❌" if step['status'] == 'failed' else "⏭️"
                print(f"     {status_icon} {step['description'][:60]}...")
            if len(plan_summary['steps']) > 3:
                print(f"     ... and {len(plan_summary['steps']) - 3} more steps")
    
    print("\n" + "="*60)