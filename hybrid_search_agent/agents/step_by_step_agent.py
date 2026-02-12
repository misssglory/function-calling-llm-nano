"""Step-by-step execution agent using composition with critic validation"""

from typing import List, Dict, Any, AsyncGenerator, Optional
from datetime import datetime
import asyncio
import uuid
import json
import re

from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow import ToolCallResult, AgentStream
from llama_index.core.workflow import Context

from hybrid_search_agent.models.step_models import Step, StepStatus, ExecutionPlan
from hybrid_search_agent.core.step_history import StepHistory
from trace_context import trace_function, TraceContext
from loguru import logger


class StepByStepAgent:
    """Agent with step-by-step execution, history preservation, and critic validation"""

    def __init__(self, agent: ReActAgent, step_history: StepHistory):
        self.agent = agent
        self.step_history = step_history
        self.current_step: Optional[Step] = None
        self.auto_execute = False
        self.skip_step = False
        self.continue_execution = True
        self.resume_execution_flag = False

    @trace_function
    async def run_step_by_step(self, query: str, ctx: Context) -> AsyncGenerator:
        """Execute query step by step with user control and critic validation"""

        # Reset flags
        self.skip_step = False
        self.continue_execution = True
        self.resume_execution_flag = False

        # Create execution plan
        plan = self.step_history.create_plan(query)
        yield {"type": "plan_created", "plan": plan}

        # Generate execution plan from LLM
        plan_steps = await self._generate_execution_plan(query)

        for step_desc in plan_steps:
            step = plan.add_step(step_desc)
            yield {"type": "step_created", "step": step}

        # Execute steps sequentially
        for step in plan.steps:
            if not self.continue_execution:
                break

            if self.skip_step:
                step.status = StepStatus.SKIPPED
                self.skip_step = False
                yield {"type": "step_skipped", "step": step}
                continue

            self.current_step = step
            step.status = StepStatus.RUNNING
            step.start_time = datetime.now()

            yield {"type": "step_start", "step": step}

            try:
                # Execute current step using the agent's run method
                result = await self._execute_step(step.description, ctx)

                # CRITIC STEP: Validate if the step was actually completed successfully
                validation_result = await self._critic_step(
                    step_description=step.description,
                    original_query=query,
                    step_result=result,
                    ctx=ctx,
                )

                if validation_result["success"]:
                    step.status = StepStatus.COMPLETED
                    step.result = str(result)[:1000]
                    step.critic_analysis = validation_result.get("analysis", "")

                    yield {
                        "type": "step_complete",
                        "step": step,
                        "result": result,
                        "critic_analysis": validation_result.get("analysis", ""),
                    }
                else:
                    # Step failed critic validation
                    step.status = StepStatus.FAILED
                    step.error = validation_result.get(
                        "error", "Step failed validation"
                    )
                    step.critic_analysis = validation_result.get("analysis", "")

                    yield {
                        "type": "step_failed",
                        "step": step,
                        "error": step.error,
                        "critic_analysis": validation_result.get("analysis", ""),
                    }

                    # Ask user whether to continue on validation failure
                    if not self.auto_execute:
                        decision = yield {
                            "type": "ask_continue",
                            "step": step,
                            "error": step.error,
                            "critic_analysis": validation_result.get("analysis", ""),
                        }
                        if decision and not decision.get("continue", False):
                            break

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)

                # CRITIC STEP: Analyze the failure
                failure_analysis = await self._analyze_failure(
                    step_description=step.description, error=str(e), ctx=ctx
                )
                step.critic_analysis = failure_analysis

                yield {
                    "type": "step_failed",
                    "step": step,
                    "error": str(e),
                    "critic_analysis": failure_analysis,
                }

                # Ask user whether to continue
                if not self.auto_execute:
                    decision = yield {
                        "type": "ask_continue",
                        "step": step,
                        "error": str(e),
                        "critic_analysis": failure_analysis,
                    }
                    if decision and not decision.get("continue", False):
                        break

            finally:
                step.end_time = datetime.now()

            # Pause between steps if not in auto mode
            if not self.auto_execute and step != plan.steps[-1]:
                while not self.resume_execution_flag and not self.auto_execute:
                    yield {"type": "step_pause", "step": step}
                    await asyncio.sleep(0.1)

                self.resume_execution_flag = False

        # Save the plan
        self.step_history.save_plan()

        yield {"type": "plan_complete", "plan": plan}

    @trace_function
    async def _critic_step(
        self, step_description: str, original_query: str, step_result: Any, ctx: Context
    ) -> Dict[str, Any]:
        """Critic step that validates if the step was successfully completed"""
        critic_id = str(uuid.uuid4())[:8]

        critic_prompt = f"""
        You are a step critic that validates if an execution step was successfully completed.
        
        Original user query: {original_query}
        
        Step that was executed: {step_description}
        
        Step result: {step_result}
        
        Your task: Analyze if this step was ACTUALLY completed successfully.
        
        Consider:
        1. Does the result contain meaningful data relevant to the step?
        2. Are there any error indicators in the result (e.g., "error", "failed", "not found", "unable to")?
        3. Was the expected action actually performed?
        4. Is the result complete or truncated?
        5. Does the result make sense for the step?
        
        Return a JSON object with:
        1. "success": boolean (true if step completed successfully)
        2. "analysis": string (brief explanation of your reasoning)
        3. "error": string (only if success=false, describe what went wrong)
        
        Return ONLY the JSON object, no other text.
        """

        try:
            with TraceContext(
                "critic_step", critic_id=critic_id, step=step_description[:50]
            ):
                handler = self.agent.run(critic_prompt, ctx=ctx)
                response = await handler
                response_text = str(response)

            # Parse JSON response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                validation = json.loads(json_match.group())
            else:
                # Fallback validation
                validation = self._fallback_critic(step_result)

            logger.debug(
                f"Critic validation for '{step_description[:50]}...': {validation.get('success', False)}"
            )
            return validation

        except Exception as e:
            logger.error(f"Error in critic step: {e}")
            return self._fallback_critic(step_result)

    @trace_function
    async def _analyze_failure(
        self, step_description: str, error: str, ctx: Context
    ) -> str:
        """Analyze why a step failed"""
        analysis_id = str(uuid.uuid4())[:8]

        analysis_prompt = f"""
        Analyze why this step failed:
        
        Step: {step_description}
        
        Error: {error}
        
        Provide a brief, helpful analysis of:
        1. What went wrong
        2. Possible reasons for the failure
        3. Suggestions for fixing or alternative approaches
        
        Keep it concise (2-3 sentences).
        """

        try:
            with TraceContext("analyze_failure", analysis_id=analysis_id):
                handler = self.agent.run(analysis_prompt, ctx=ctx)
                response = await handler
                return str(response).strip()
        except:
            return f"Step failed with error: {error}"

    def _fallback_critic(self, step_result: Any) -> Dict[str, Any]:
        """Fallback critic when LLM validation fails"""
        result_str = str(step_result).lower()

        # Heuristic-based validation
        error_indicators = [
            "error",
            "failed",
            "exception",
            "not found",
            "unable to",
            "cannot",
            "no results",
            "empty",
            "none",
            "null",
        ]

        success = True
        error_msg = None

        for indicator in error_indicators:
            if indicator in result_str:
                success = False
                error_msg = f"Step result contains error indicator: '{indicator}'"
                break

        return {
            "success": success,
            "analysis": "Heuristic-based validation"
            + (f": {error_msg}" if error_msg else ""),
            "error": error_msg,
        }

    @trace_function
    async def _generate_execution_plan(self, query: str) -> List[str]:
        """Generate execution plan from query using agent.run()"""
        plan_query_id = str(uuid.uuid4())[:8]

        prompt = f"""
        Break down the following request into a sequence of logical steps:
        
        Request: {query}
        
        Plan requirements:
        1. Each step should be a specific action (e.g., "Search for X using duckduckgo_search", "Navigate to Y", "Extract text from page")
        2. Steps should be in logical sequence
        3. Each step must use available tools
        4. Steps should be granular enough for control
        5. Maximum 5-7 steps per plan
        
        Available tools:
        - local_document_search: search local documents
        - duckduckgo_search: search the internet
        - navigate_to: navigate to URL
        - extract_text: extract text from page
        - click: click on elements
        - extract_hyperlinks: extract hyperlinks from page
        - scrape_text: scrape text from current page
        - take_screenshot: take screenshot of current page
        - save_as_pdf: save current page as PDF
        
        Format: simple list of steps, one per line, no numbering, no additional text.
        Example:
        Search for recent AI news using duckduckgo_search
        Navigate to the first result URL
        Extract text from the page
        Extract hyperlinks from the page
        
        Return only the list of steps, nothing else. Do not include "Thought:" or "Answer:" or any other prefixes.
        """

        try:
            with TraceContext("generate_execution_plan", query_id=plan_query_id):
                handler = self.agent.run(prompt, ctx=None)
                response = await handler
                response_text = str(response)

            # Parse steps from response
            steps = []
            for line in response_text.split("\n"):
                line = line.strip()

                # Skip empty lines and markdown code blocks
                if not line or line.startswith(("```", "`", "#")):
                    continue

                # Remove common prefixes
                line = re.sub(r"^(Step\s*\d+[:.]\s*)", "", line, flags=re.IGNORECASE)
                line = re.sub(r"^(\d+[.)]\s*)", "", line)
                line = re.sub(r"^[-*]\s+", "", line)

                # Skip lines that are just "Thought:" or "Answer:" or contain instruction templates
                if (
                    line.startswith(("Thought:", "Answer:", "Action:", "Observation:"))
                    or "[your answer here" in line.lower()
                    or "use the appropriate tool" in line.lower()
                ):
                    continue

                # Only add lines that look like actual step descriptions (more than 10 chars, not just instructions)
                if len(line) > 15 and not any(
                    x in line.lower()
                    for x in [
                        "use the appropriate tool",
                        "be specific with tool",
                        "return only the result",
                        "keep the response concise",
                        "execute now",
                        "previous steps history",
                        "available tools",
                    ]
                ):
                    steps.append(line)

            # If we didn't get any valid steps, use default plan
            if not steps:
                logger.warning("No valid steps generated, using default plan")
                steps = [
                    f"Search for information about '{query}' using duckduckgo_search",
                    "Navigate to the first relevant result",
                    "Extract text from the page",
                    "Summarize the extracted information",
                ]

            logger.debug(f"Generated {len(steps)} steps for plan")
            return steps[:7]

        except Exception as e:
            logger.error(f"Error generating execution plan: {e}")
            return [
                f"Search for '{query}' using duckduckgo_search",
                "Navigate to the first relevant result",
                "Extract text from the page",
                "Summarize the extracted information",
            ]

    @trace_function
    async def _execute_step(self, step_description: str, ctx: Context) -> str:
        """Execute a single step using agent.run()"""
        step_id = str(uuid.uuid4())[:8]

        # Get previous steps context
        steps_history = self._get_steps_history()

        # CRITICAL FIX: Provide clear instructions WITHOUT including the step description in a way
        # that gets misinterpreted as the actual output
        step_context = f"""
        Current task: Execute this specific step: "{step_description}"
        
        Previous steps:
        {steps_history}
        
        Instructions:
        1. Use the appropriate tool for this step
        2. Execute ONLY this step, do not continue to next steps
        3. Return the RESULT of executing this step, not the instructions
        4. Be concise
        
        Execute the step and return the result:
        """

        try:
            with TraceContext(
                "execute_step", step_id=step_id, step_description=step_description[:50]
            ):
                handler = self.agent.run(step_context, ctx=ctx)

                async for ev in handler.stream_events():
                    if isinstance(ev, AgentStream):
                        pass

                response = await handler
                response_str = str(response)

                # Clean up response - remove any instruction text that might have leaked
                response_str = self._clean_step_result(response_str)

            # Save tool call information
            tool_calls = []
            if hasattr(handler, "events"):
                for event in handler.events:
                    if hasattr(event, "tool_name"):
                        tool_calls.append(
                            {
                                "tool_name": event.tool_name,
                                "tool_input": getattr(event, "tool_kwargs", {}),
                                "result": str(getattr(event, "tool_output", ""))[:200],
                            }
                        )

            if self.current_step:
                self.current_step.tool_calls = tool_calls

            return response_str[:2000]

        except Exception as e:
            logger.error(f"Error executing step '{step_description[:50]}...': {e}")
            raise

    def _clean_step_result(self, result: str) -> str:
        """Remove instruction text and templates from step results"""
        # Remove common instruction patterns
        patterns_to_remove = [
            r"1\.\s*Use the appropriate tool.*?(?=\n|$)",
            r"2\.\s*Be specific with tool.*?(?=\n|$)",
            r"3\.\s*Return ONLY the result.*?(?=\n|$)",
            r"4\.\s*Keep the response concise.*?(?=\n|$)",
            r"\[your answer here.*?\]",
            r"Execute now and return the result:",
            r"Previous steps history:.*?(?=\n\n)",
            r"Available tools:.*?(?=\n\n)",
            r"Instructions:.*?(?=\n\n)",
        ]

        cleaned = result
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove multiple newlines
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned)

        return cleaned.strip()

    def _get_steps_history(self) -> str:
        """Get history of executed steps"""
        if not self.step_history.current_plan:
            return "No steps executed yet"

        history = []
        for step in self.step_history.current_plan.steps[-5:]:
            if step.status == StepStatus.COMPLETED:
                history.append(f"✅ {step.description}")
                if step.result:
                    # Truncate result for display
                    result_preview = step.result[:100].replace("\n", " ")
                    history.append(f"   Result: {result_preview}...")
                if step.critic_analysis:
                    history.append(f"   Analysis: {step.critic_analysis[:100]}...")
            elif step.status == StepStatus.FAILED:
                history.append(f"❌ {step.description}")
                if step.error:
                    history.append(f"   Error: {step.error[:100]}...")
                if step.critic_analysis:
                    history.append(f"   Analysis: {step.critic_analysis[:100]}...")
            elif step.status == StepStatus.SKIPPED:
                history.append(f"⏭️ {step.description}")
            elif step.status == StepStatus.RUNNING:
                history.append(f"⚡ {step.description}")
            else:
                history.append(f"⏳ {step.description}")

        return "\n".join(history)

    async def toggle_auto_execute(self) -> bool:
        """Toggle auto-execution mode"""
        self.auto_execute = not self.auto_execute
        return self.auto_execute

    async def skip_current_step(self):
        """Skip current step"""
        self.skip_step = True

    async def resume_execution(self):
        """Resume paused execution"""
        self.resume_execution_flag = True

    async def stop_execution(self):
        """Stop execution"""
        self.continue_execution = False

    def __getattr__(self, name):
        """Delegate attribute access to the underlying agent"""
        return getattr(self.agent, name)
