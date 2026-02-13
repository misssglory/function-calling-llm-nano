"""Step-by-step execution agent using composition with detailed actor-critic validation"""

from typing import List, Dict, Any, AsyncGenerator, Optional
from datetime import datetime
import asyncio
import uuid
import json
import re
import time

from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow import ToolCallResult, AgentStream, AgentOutput
from llama_index.core.workflow import Context
from llama_index.core.workflow.events import StopEvent

from hybrid_search_agent.core import step_history
from hybrid_search_agent.models.step_models import (
    Step,
    StepStatus,
    ExecutionPlan,
    ActorCriticStep,
)
from hybrid_search_agent.core.step_history import StepHistory
from trace_context import trace_function, TraceContext
from loguru import logger


class StepByStepAgent:
    """Agent with step-by-step execution and detailed actor-critic analysis"""

    def __init__(self, agent: ReActAgent, step_history: StepHistory):
        self.agent = agent
        self.step_history = step_history
        self.current_step: Optional[ActorCriticStep] = None
        self.auto_execute = False
        self.skip_step = False
        self.continue_execution = True
        self.resume_execution_flag = False

    @trace_function
    async def run_step_by_step(self, query: str, ctx: Context) -> AsyncGenerator:
        """Execute query step by step with detailed actor-critic analysis"""

        # Reset flags
        self.skip_step = False
        self.continue_execution = True
        self.resume_execution_flag = False

        # Create execution plan
        plan = self.step_history.create_plan(query)
        yield {"type": "plan_created", "plan": plan}

        # Generate execution plan from LLM
        plan_steps = await self._generate_execution_plan(query, ctx)

        for i, step_desc in enumerate(plan_steps):
            step = ActorCriticStep(
                id=f"step_{i+1}",
                description=step_desc,
                status=StepStatus.PENDING,
                actor_result=None,
                actor_metadata={},
                critic_analysis=None,
                critic_metadata={},
            )
            plan.add_step(step)
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
                # ACTOR: Execute current step
                actor_start = time.perf_counter()
                actor_result, tool_calls = await self._execute_step(
                    step.description, ctx
                )
                actor_end = time.perf_counter()

                step.actor_result = actor_result
                step.actor_metadata = {
                    "execution_start": datetime.now().isoformat(),
                    "execution_duration": round(actor_end - actor_start, 3),
                    "tool_calls": tool_calls,
                    "timestamp": datetime.now().isoformat(),
                }
                step.tool_calls = tool_calls

                yield {
                    "type": "actor_complete",
                    "step": step,
                    "actor_result": actor_result,
                    "actor_metadata": step.actor_metadata,
                }

                # CRITIC: Validate the step
                critic_start = time.perf_counter()
                validation_result = await self._critic_step(
                    step_description=step.description,
                    original_query=query,
                    step_result=actor_result,
                    tool_calls=tool_calls,
                    ctx=ctx,
                )
                critic_end = time.perf_counter()

                # Enhance critic result with metadata
                step.critic_analysis = validation_result
                step.critic_metadata = {
                    "execution_start": datetime.now().isoformat(),
                    "execution_duration": round(critic_end - critic_start, 3),
                    "validation_success": validation_result.get("success", False),
                    "confidence_score": validation_result.get("confidence", 0.7),
                    "validation_method": validation_result.get("method", "llm"),
                    "timestamp": datetime.now().isoformat(),
                }

                # Determine step status based on critic
                if validation_result["success"]:
                    step.status = StepStatus.COMPLETED

                    yield {
                        "type": "critic_validation_passed",
                        "step": step,
                        "critic_analysis": validation_result,
                        "critic_metadata": step.critic_metadata,
                    }
                else:
                    step.status = StepStatus.FAILED
                    step.error = validation_result.get(
                        "error", "Step failed validation"
                    )

                    yield {
                        "type": "critic_validation_failed",
                        "step": step,
                        "critic_analysis": validation_result,
                        "critic_metadata": step.critic_metadata,
                    }

                    if not self.auto_execute:
                        decision = yield {
                            "type": "ask_continue_after_failure",
                            "step": step,
                            "critic_analysis": validation_result,
                            "critic_metadata": step.critic_metadata,
                        }
                        if decision and not decision.get("continue", False):
                            break

            except Exception as e:
                # Handle execution errors with critic analysis
                step.status = StepStatus.FAILED
                step.error = str(e)

                # CRITIC: Analyze the failure
                critic_start = time.perf_counter()

                logger.error("run_step_by_step error")
                logger.exception(e)

                failure_analysis = await self._analyze_failure(
                    step_description=step.description, error=str(e), ctx=ctx
                )
                critic_end = time.perf_counter()

                step.critic_analysis = {
                    "success": False,
                    "error": str(e),
                    "analysis": failure_analysis,
                    "recommendations": self._generate_recommendations(
                        step.description, str(e)
                    ),
                }
                step.critic_metadata = {
                    "execution_start": datetime.now().isoformat(),
                    "execution_duration": round(critic_end - critic_start, 3),
                    "validation_success": False,
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat(),
                }

                yield {
                    "type": "step_failed_with_analysis",
                    "step": step,
                    "error": str(e),
                    "critic_analysis": step.critic_analysis,
                    "critic_metadata": step.critic_metadata,
                }

                if not self.auto_execute:
                    decision = yield {
                        "type": "ask_continue_after_error",
                        "step": step,
                        "error": str(e),
                        "critic_analysis": step.critic_analysis,
                    }
                    if decision and not decision.get("continue", False):
                        break

            finally:
                step.end_time = datetime.now()
                step.duration = (step.end_time - step.start_time).total_seconds()

            # Pause between steps if not in auto mode
            if not self.auto_execute and step != plan.steps[-1]:
                while not self.resume_execution_flag and not self.auto_execute:
                    yield {"type": "step_pause", "step": step}
                    await asyncio.sleep(0.1)

                self.resume_execution_flag = False

        # Save the plan
        self.step_history.save_plan()

        # Generate execution summary
        summary = self._generate_execution_summary(plan)
        yield {"type": "plan_complete", "plan": plan, "summary": summary}

    @trace_function
    async def _critic_step(
        self,
        step_description: str,
        original_query: str,
        step_result: Any,
        tool_calls: List[Dict],
        ctx: Context,
        remaining_trials=3,
    ) -> Dict[str, Any]:
        """Enhanced critic with detailed validation and confidence scoring"""
        critic_id = str(uuid.uuid4())[:8]

        critic_prompt = f"""
        You are a rigorous step critic analyzing if an execution step was successfully completed.
        
        ORIGINAL USER QUERY: {original_query}
        
        STEP EXECUTED: {step_description}
        
        STEP RESULT: {step_result}
        
        TOOL CALLS MADE: {json.dumps(tool_calls, indent=2, ensure_ascii=False)}
        
        Your task: Perform a detailed validation of this step.
        
        VALIDATION CRITERIA:
        1. **Relevance**: Does the result directly address what the step intended to do?
        2. **Completeness**: Is the result complete or truncated?
        3. **Accuracy**: Does the result contain the expected type of information?
        4. **Error Detection**: Are there any error indicators in the result?
        5. **Tool Usage**: Were the appropriate tools used correctly?
        6. **Data Quality**: Is the result meaningful and usable?
        
        Return a JSON object with the following structure:
        {{
            "success": boolean,
            "confidence": float (0.0-1.0),
            "analysis": {{
                "summary": "Brief overall assessment",
                "relevance": {{"score": 0-10, "reasoning": "..."}},
                "completeness": {{"score": 0-10, "reasoning": "..."}},
                "accuracy": {{"score": 0-10, "reasoning": "..."}},
                "error_check": {{"found_errors": boolean, "details": "..."}},
                "tool_usage": {{"appropriate": boolean, "reasoning": "..."}},
                "data_quality": {{"score": 0-10, "reasoning": "..."}}
            }},
            "error": "Detailed error description if success=false, otherwise null",
            "recommendations": ["Suggestion 1", "Suggestion 2"],
            "next_steps": ["Recommended next step 1", "Recommended next step 2"],
            "method": "llm"
        }}
        
        Return ONLY the JSON object, no other text.
        """

        try:
            with TraceContext(
                "critic_step", critic_id=critic_id, step=step_description[:50]
            ):
                handler = self.agent.run(critic_prompt, ctx=ctx)
                _events = []
                logger.debug(critic_prompt)
                async for ev in handler.stream_events():
                    if isinstance(ev, AgentStream):
                        print(ev.delta, end="", flush=True)
                        continue
                    logger.debug(type(ev))
                    _events.append(ev)

                    if isinstance(ev, AgentOutput):
                        logger.debug("critic_step: AgentOutput in chain")
                        logger.trace(ev.response.content)
                        # break
                    if isinstance(ev, ToolCallResult):
                        logger.debug("critic_step: ToolCallResult in chain")
                        ctx.send_event(StopEvent(result="ok"))
                        await handler.cancel_run()
                        break
                response = await handler
                response_text = str(response)

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                validation = json.loads(json_match.group())
            else:
                validation = self._fallback_critic(step_result, tool_calls)

            logger.debug(
                f"Critic validation for '{step_description[:50]}...': {validation.get('success', False)}"
            )
            return validation

        except Exception as e:
            logger.exception(f"Error in critic step: {e}")
            ctx.send_event(StopEvent(result="ok"))
            await handler.cancel_run()
            try:
                resp = await self._critic_step(
                    step_description,
                    original_query,
                    step_result,
                    tool_calls,
                    ctx=Context(self.agent),
                    remaining_trials=remaining_trials - 1,
                )
                return resp
            except:
                # raise
                return self._fallback_critic(step_result, tool_calls)

    @trace_function
    async def _execute_step(
        self, step_description: str, ctx: Context, remaining_trials=3
    ) -> tuple[str, List[Dict]]:
        """Execute step and return result with detailed tool call information"""
        step_id = str(uuid.uuid4())[:8]

        # Get previous steps context with actor-critic history
        steps_history = self._get_detailed_steps_history()

        step_context = f"""
        EXECUTE THIS SPECIFIC STEP: "{step_description}"
        
        EXECUTION CONTEXT:
        {steps_history}
        
        EXECUTION REQUIREMENTS:
        1. Use ONLY the tools necessary for this specific step
        2. Do NOT continue to next steps
        3. Return ONLY the result of this step
        4. Be concise but complete
        5. Include ALL data retrieved/generated
        
        AVAILABLE TOOLS:
        - local_document_search(query) - Search local documents
        - duckduckgo_search(query) - Search the internet
        - navigate_to(url) - Navigate to URL
        - extract_text() - Extract text from current page
        - click(selector) - Click on elements
        - extract_hyperlinks() - Extract hyperlinks from page  
        - navigate_back() - Go back to previous page
        
        Execute now and return the result:
        """

        try:
            _events = []
            with TraceContext(
                "execute_step", step_id=step_id, step_description=step_description[:50]
            ):

                logger.debug(f"Step context: {step_context}")
                handler = self.agent.run(step_context, ctx=ctx)

                async for ev in handler.stream_events():
                    if isinstance(ev, AgentStream):
                        print(ev.delta, end="", flush=True)
                        continue
                    logger.debug(type(ev))
                    _events.append(ev)
                    # if isinstance(ev, ToolCallResult):
                    #     logger.debug(f"Tool Used: {ev.tool_name}")
                    #     logger.debug(f"Inputs: {ev.tool_kwargs}")
                    #     logger.debug(f"Output: {ev.tool_output}")
                    if isinstance(ev, AgentOutput):
                        logger.debug("execute_step: AgentOutput in chain")
                        logger.trace(ev.response.content)
                        # break
                    if isinstance(ev, ToolCallResult):
                        logger.debug("execute_step: ToolCallResult in chain")
                        ctx.send_event(StopEvent(result="ok"))
                        await handler.cancel_run()
                        break

                # try:
                # response = await handler
                # except Exception as e:
                # logger.exception(e)
                response = ""
                for ev in _events:
                    if isinstance(ev, AgentOutput):
                        response = response + ev.response.content

                response_str = str(response)

                response_str = self._clean_step_result(response_str)

            tool_calls = []
            for ev in _events:
                if isinstance(ev, ToolCallResult):
                    tool_calls.append(
                        {
                            "tool_name": ev.tool_name,
                            "tool_input": ev.tool_kwargs,
                            "tool_output": ev.tool_output.content,
                            "timestamp": datetime.now().isoformat(),
                            "success": ev.tool_output.is_error,
                        }
                    )

            return response_str[:5000], tool_calls

        except Exception as e:
            logger.exception(f"Error executing step '{step_description[:50]}...': {e}")
            ctx.send_event(StopEvent(result="ok"))
            await handler.cancel_run()

            try:
                resp = await self._execute_step(
                    step_description,
                    ctx=Context(self.agent),
                    remaining_trials=remaining_trials - 1,
                )
                return resp
            except:
                raise
            # raise

    async def _analyze_failure(
        self, step_description: str, error: str, ctx: Context
    ) -> str:
        """Detailed failure analysis with root cause identification"""
        analysis_id = str(uuid.uuid4())[:8]

        analysis_prompt = f"""
        Perform a detailed failure analysis:
        
        FAILED STEP: {step_description}
        
        ERROR: {error}
        
        Analyze:
        1. ROOT CAUSE: What fundamentally caused this failure?
        2. CONTEXT FACTORS: What conditions contributed?
        3. IMPACT: What functionality is affected?
        4. SOLUTION: Specific steps to fix
        5. PREVENTION: How to avoid in future
        
        Provide concise, actionable analysis:
        """

        logger.debug(f"Analyze error: description: {step_description} error: {error}")

        try:
            with TraceContext("analyze_failure", analysis_id=analysis_id):
                handler = self.agent.run(analysis_prompt, ctx=ctx)
                response = await handler

                logger.debug(f"Analyze error response: {str(response)}")

                return str(response).strip()
        except:
            return f"Step failed with error: {error}"

    def _generate_recommendations(self, step_description: str, error: str) -> List[str]:
        """Generate specific recommendations for failure recovery"""
        recommendations = []

        if "timeout" in error.lower() or "navigation" in error.lower():
            recommendations.extend(
                [
                    "Check if the URL is accessible",
                    "Verify internet connection",
                    "Try increasing timeout settings",
                ]
            )
        elif "not found" in error.lower() or "no results" in error.lower():
            recommendations.extend(
                [
                    "Try different search terms",
                    "Check if the element/page exists",
                    "Verify the URL is correct",
                ]
            )
        elif "tool" in error.lower() or "function" in error.lower():
            recommendations.extend(
                [
                    "Verify tool parameters are correct",
                    "Check if required dependencies are installed",
                    "Ensure tool is properly initialized",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Review step requirements",
                    "Check for typos or incorrect parameters",
                    "Try alternative approach",
                ]
            )

        return recommendations

    def _fallback_critic(
        self, step_result: Any, tool_calls: List[Dict]
    ) -> Dict[str, Any]:
        """Enhanced fallback critic with heuristic-based validation"""
        result_str = str(step_result).lower()

        # Error indicators
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
            "404",
            "timeout",
            "forbidden",
            "unauthorized",
        ]

        success = True
        error_msg = None
        confidence = 0.7
        analysis_scores = {}

        # Check for error indicators
        for indicator in error_indicators:
            if indicator in result_str:
                success = False
                error_msg = f"Step result contains error indicator: '{indicator}'"
                confidence = 0.3
                break

        # Check tool usage
        tool_usage_appropriate = bool(tool_calls)
        if not tool_calls:
            confidence -= 0.2
            tool_usage_appropriate = False

        # Check result length for completeness
        completeness_score = min(10, len(result_str) / 100)

        return {
            "success": success,
            "confidence": confidence,
            "analysis": {
                "summary": "Heuristic-based validation",
                "relevance": {"score": 5, "reasoning": "Heuristic assessment"},
                "completeness": {
                    "score": completeness_score,
                    "reasoning": f"Result length: {len(result_str)} chars",
                },
                "accuracy": {"score": 5, "reasoning": "Heuristic assessment"},
                "error_check": {
                    "found_errors": not success,
                    "details": error_msg if error_msg else "No errors detected",
                },
                "tool_usage": {
                    "appropriate": tool_usage_appropriate,
                    "reasoning": f"{len(tool_calls)} tools called",
                },
                "data_quality": {"score": 5, "reasoning": "Heuristic assessment"},
            },
            "error": error_msg,
            "recommendations": [
                "Review step execution",
                "Check tool usage",
                "Verify result completeness",
            ],
            "next_steps": [
                "Continue to next step" if success else "Retry step or skip"
            ],
            "method": "heuristic",
        }

    async def _generate_execution_plan(self, query: str, ctx: Context) -> List[str]:
        """Generate detailed execution plan with specific tool actions"""
        plan_query_id = str(uuid.uuid4())[:8]

        tools_prompt = """AVAILABLE TOOLS:
        🔍 LOCAL_SEARCH: local_document_search(query)
        🌐 WEB_SEARCH: duckduckgo_search(query)
        🧭 NAVIGATION: navigate_to(url)
        📝 TEXT: extract_text()
        🖱️ INTERACTION: click(selector)
        🔗 LINKS: extract_hyperlinks()
        ↩️ BACK: navigate_back()
        """

        tools_prompt = ""

        prompt = f"""
        Create a detailed execution plan for: {query}
        
        REQUIREMENTS:
        1. Each step must specify EXACT tool and action
        2. Steps must be in logical sequence
        3. Include specific parameters where known
        4. Maximum 5 steps
        
        {tools_prompt}
        
        FORMAT: One step per line, no numbers, no prefixes
        EXAMPLE:
Search for iqdoc.ai website using duckduckgo_search
Navigate to https://iqdoc.ai
Extract all text from the page
Save extracted text as response
        
        Return ONLY the steps, nothing else:
        """

        try:
            _events = []
            with TraceContext("generate_execution_plan", query_id=plan_query_id):
                logger.info(prompt)
                logger.debug(ctx)
                handler = self.agent.run(prompt, ctx=ctx)
                async for ev in handler.stream_events():
                    if isinstance(ev, AgentStream):
                        continue
                    logger.debug(type(ev))
                    _events.append(ev)
                    if isinstance(ev, ToolCallResult):
                        logger.debug(f"Tool Used: {ev.tool_name}")
                        logger.debug(f"Inputs: {ev.tool_kwargs}")
                        logger.debug(f"Output: {ev.tool_output}")

                response = await handler
                response_text = str(response)
                # async for ev in handler.stream_events():

            # for ev in self.agent.events:
            #     if isinstance(ev, ToolCallResult):
            #         logger.debug(f"Tool Used: {ev.tool_name}")
            #         logger.debug(f"Inputs: {ev.tool_kwargs}")
            #         logger.debug(f"Output: {ev.tool_output}")

            logger.debug(
                f"generate_execution_plan: Response: <green>{response_text}</green>"
            )
            # Parse and clean steps
            steps = []
            for line in response_text.split("\n"):
                line = line.strip()

                # Skip invalid lines
                if not line or len(line) < 10:
                    continue
                if line.startswith(("```", "`", "#", "Step", "step")):
                    continue
                if "[your answer" in line.lower():
                    continue

                # Clean formatting
                line = re.sub(r"^(\d+[.)]\s*|\-\s*|\*\s*)", "", line)

                # Add if it looks like a valid step
                if any(
                    tool in line.lower()
                    for tool in [
                        "search",
                        "navigate",
                        "extract",
                        "click",
                        "scrape",
                        "save",
                        "take",
                    ]
                ):
                    steps.append(line)

            logger.debug(f"Steps: <green>{steps}</green>")

            if not steps:
                raise Exception("Steps are empty")

            return (
                steps[:7]
                # if steps
                # else [
                #     f"Search for {query} using duckduckgo_search",
                #     "Navigate to the main website URL",
                #     "Extract all text content from the page",
                #     "Return the extracted text as result",
                # ]
            )

        except Exception as e:
            logger.exception(f"Error generating plan: {e}")
            return [
                f"Search for {query} using duckduckgo_search",
                "Navigate to the main website URL",
                "Extract all text content from the page",
                "Return the extracted text as result",
            ]

    def _clean_step_result(self, result: str) -> str:
        """Clean and format step result"""
        # Remove instruction patterns
        patterns = [
            r"1\.\s*Use the appropriate tool.*?(?=\n|$)",
            r"2\.\s*Be specific with tool.*?(?=\n|$)",
            r"3\.\s*Return ONLY the result.*?(?=\n|$)",
            r"4\.\s*Keep the response concise.*?(?=\n|$)",
            r"Execute now and return the result:",
            r"Previous steps history:.*?(?=\n\n)",
            r"Available tools:.*?(?=\n\n)",
            r"EXECUTION REQUIREMENTS:.*?(?=\n\n)",
        ]

        cleaned = result
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove repeated content
        lines = cleaned.split("\n")
        unique_lines = []
        seen = set()

        for line in lines:
            line_strip = line.strip()
            if line_strip and line_strip not in seen:
                seen.add(line_strip)
                unique_lines.append(line)

        return "\n".join(unique_lines).strip()

    def _get_detailed_steps_history(self) -> str:
        """Get detailed step history with actor-critic information"""
        if not self.step_history.current_plan:
            return "No previous steps"

        history = []
        for i, step in enumerate(self.step_history.current_plan.steps[-3:], 1):
            if step.status == StepStatus.COMPLETED:
                history.append(f"Step {i}: ✅ {step.description}")
                if step.actor_result:
                    preview = step.actor_result[:200].replace("\n", " ")
                    history.append(f"   Result: {preview}...")
                if step.critic_analysis:
                    success = step.critic_analysis.get("success", False)
                    confidence = step.critic_analysis.get("confidence", 0)
                    history.append(
                        f"   Validation: {'✓' if success else '✗'} (confidence: {confidence})"
                    )
            elif step.status == StepStatus.FAILED:
                history.append(f"Step {i}: ❌ {step.description}")
                if step.error:
                    history.append(f"   Error: {step.error[:100]}...")

        return "\n".join(history)

    def _generate_execution_summary(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Generate comprehensive execution summary"""
        total_steps = len(plan.steps)
        completed = sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in plan.steps if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in plan.steps if s.status == StepStatus.SKIPPED)

        total_duration = 0
        total_actor_time = 0
        total_critic_time = 0

        for step in plan.steps:
            if step.duration:
                total_duration += step.duration
            if step.actor_metadata:
                total_actor_time += step.actor_metadata.get("execution_duration", 0)
            if step.critic_metadata:
                total_critic_time += step.critic_metadata.get("execution_duration", 0)

        return {
            "query": plan.query,
            "execution_summary": {
                "total_steps": total_steps,
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
                "success_rate": round(
                    (completed / total_steps * 100) if total_steps > 0 else 0, 1
                ),
            },
            "performance": {
                "total_execution_time": round(total_duration, 2),
                "total_actor_time": round(total_actor_time, 2),
                "total_critic_time": round(total_critic_time, 2),
                "average_step_time": (
                    round(total_duration / total_steps, 2) if total_steps > 0 else 0
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }

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
