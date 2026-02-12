"""Data models for step-by-step execution"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import json
import uuid


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
            "dependencies": self.dependencies
        }


@dataclass
class ExecutionPlan:
    """Execution plan with sequence of steps"""
    id: str
    query: str
    steps: List[Step] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, description: str, dependencies: List[str] = None) -> Step:
        """Add a new step to the plan"""
        step = Step(
            id=f"step_{len(self.steps) + 1}",
            description=description,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        return step
    
    def to_dict(self) -> Dict:
        """Convert execution plan to dictionary"""
        return {
            "id": self.id,
            "query": self.query,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "steps": [step.to_dict() for step in self.steps],
            "context": self.context
        }
    
    @property
    def completed_steps(self) -> int:
        """Get number of completed steps"""
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
    
    @property
    def failed_steps(self) -> int:
        """Get number of failed steps"""
        return sum(1 for s in self.steps if s.status == StepStatus.FAILED)
    
    @property
    def total_steps(self) -> int:
        """Get total number of steps"""
        return len(self.steps)