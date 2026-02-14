from pydantic import BaseModel, Field, validator
from typing import List
import json
import re


class ExecutionPlan(BaseModel):
    """Simple execution plan validation"""

    steps: List[str] = Field(
        ..., description="List of execution steps", min_items=1, max_items=5
    )

    @validator("steps")
    def validate_steps(cls, v):
        """Basic validation of steps"""
        # Check each step has minimum length
        for i, step in enumerate(v):
            if len(step) < 10:
                raise ValueError(f"Step {i+1} is too short: {step}")

            # Check step contains a verb (simple check)
            # verbs = [
            #     "search",
            #     "navigate",
            #     "extract",
            #     "click",
            #     "scrape",
            #     "save",
            #     "take",
            #     "go",
            #     "find",
            # ]
            # if not any(verb in step.lower() for verb in verbs):
            #     raise ValueError(f"Step {i+1} missing action verb: {step}")

        return v
