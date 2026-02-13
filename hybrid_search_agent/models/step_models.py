"""Data models for step-by-step execution"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
from enum import Enum


class StepStatus(Enum):
    """Status of a single execution step"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    """Represents a single step in execution plan"""

    id: str
    description: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tool_calls: List[Dict] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Get step duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict:
        """Convert step to dictionary"""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "duration": self.duration,
            "tool_calls": self.tool_calls,
            "dependencies": self.dependencies,
        }


class ActorCriticStep:
    """Step model with separate actor and critic information"""

    def __init__(
        self,
        id: str,
        description: str,
        status: StepStatus = StepStatus.PENDING,
        actor_result: Optional[str] = None,
        actor_metadata: Optional[Dict[str, Any]] = None,
        critic_analysis: Optional[Dict[str, Any]] = None,
        critic_metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict]] = None,
        error: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        duration: Optional[float] = None,
    ):
        self.id = id
        self.description = description
        self.status = status
        self.actor_result = actor_result
        self.actor_metadata = actor_metadata or {}
        self.critic_analysis = critic_analysis or {}
        self.critic_metadata = critic_metadata or {}
        self.tool_calls = tool_calls or []
        self.error = error
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization"""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "actor_result": self.actor_result,
            "actor_metadata": self.actor_metadata,
            "critic_analysis": self.critic_analysis,
            "critic_metadata": self.critic_metadata,
            "tool_calls": self.tool_calls,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
        }

    def __str__(self) -> str:
        """Returns a readable string representation of the step"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def create_mock(
        cls,
        *,
        id: Optional[str] = None,
        description: Optional[str] = None,
        status: StepStatus = StepStatus.COMPLETED,
        with_actor_result: bool = True,
        with_critic_analysis: bool = True,
        with_tool_calls: bool = False,
        with_error: bool = False,
        with_timings: bool = True,
    ) -> "ActorCriticStep":
        """Create a mock ActorCriticStep instance for testing.

        Args:
            id: Custom ID (auto-generated if None)
            description: Custom description (auto-generated if None)
            status: Step status (default: COMPLETED)
            with_actor_result: Include actor result
            with_critic_analysis: Include critic analysis
            with_tool_calls: Include tool calls
            with_error: Include error message
            with_timings: Include start/end times and duration

        Returns:
            ActorCriticStep instance populated with mock data
        """
        step_id = id or f"test_step_{uuid.uuid4().hex[:6]}"
        desc = description or f"Mock step {step_id}"

        # Base step
        step = cls(id=step_id, description=desc, status=status)

        # Add actor result
        if with_actor_result:
            step.actor_result = f"Successfully executed {desc}"
            step.actor_metadata = {
                "model": "gpt-4",
                "tokens": 150,
                "confidence": 0.95,
                "processing_time_ms": 234,
            }

        # Add critic analysis
        if with_critic_analysis:
            step.critic_analysis = {
                "score": 0.85,
                "feedback": "Step executed correctly",
                "suggestions": ["Consider adding more validation"],
                "issues": [],
            }
            step.critic_metadata = {
                "model": "gpt-4",
                "analysis_time_ms": 156,
                "tokens": 89,
            }

        # Add tool calls
        if with_tool_calls:
            step.tool_calls = [
                {
                    "name": "search_web",
                    "arguments": {"query": "test query"},
                    "result": "Search completed",
                    "duration_ms": 450,
                },
                {
                    "name": "extract_text",
                    "arguments": {"url": "https://example.com"},
                    "result": "Text extracted",
                    "duration_ms": 320,
                },
            ]

        # Add error
        if with_error:
            step.error = "Failed to execute step: timeout after 30s"
            step.status = StepStatus.FAILED

        # Add timings
        if with_timings:
            now = datetime.now()
            step.start_time = now - timedelta(seconds=5)
            step.end_time = now
            step.duration = 5.0

        return step


class StepEncoder(json.JSONEncoder):
    def default(self, obj):
        # Check if the object has our to_dict method
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        # Handle other types like UUID if necessary
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


class ExecutionPlan:
    """Execution plan with steps"""

    def __init__(self, query: str, id: str | None):
        if not id is None:
            self.id = id
        else:
            self.id = str(uuid.uuid4())[:8]
        self.query = query
        self.steps: List[ActorCriticStep] = []
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None

    def add_step(self, description: Union[str, ActorCriticStep]) -> ActorCriticStep:
        """Add step to plan"""
        if isinstance(description, ActorCriticStep):
            step = description
        else:
            step = ActorCriticStep(
                id=f"step_{len(self.steps) + 1}", description=description
            )

        self.steps.append(step)
        return step

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for serialization"""
        return {
            "id": self.id,
            "query": self.query,
            "created_at": self.created_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "steps": [step.to_dict() for step in self.steps],
        }
