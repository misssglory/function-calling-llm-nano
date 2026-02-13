"""Step history management for execution plans"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from loguru import logger

from hybrid_search_agent.models.step_models import ExecutionPlan, Step, StepStatus
from hybrid_search_agent.config import STEP_HISTORY_DIR


class StepHistory:
    """Manages history of step-by-step executions"""

    def __init__(self, storage_dir: str = str(STEP_HISTORY_DIR)):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        self.current_plan: Optional[ExecutionPlan] = None

    def create_plan(self, query: str) -> ExecutionPlan:
        """Create a new execution plan"""
        plan_id = str(uuid.uuid4())[:8]
        plan = ExecutionPlan(id=plan_id, query=query)
        self.execution_plans[plan_id] = plan
        self.current_plan = plan
        logger.debug(f"Created execution plan {plan_id}: {query[:50]}...")
        return plan

    def update_step(self, step_id: str, **kwargs):
        """Update step state"""
        if self.current_plan:
            for step in self.current_plan.steps:
                if step.id == step_id:
                    for key, value in kwargs.items():
                        if hasattr(step, key):
                            setattr(step, key, value)
                    break

    def save_plan(self, plan_id: Optional[str] = None):
        """Save execution plan to file"""
        plan = self.execution_plans.get(plan_id) if plan_id else self.current_plan
        if plan:
            plan.completed_at = datetime.now()
            file_path = (
                self.storage_dir
                / f"plan_{plan.id}_{plan.created_at.strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(plan.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved execution plan to {file_path}")

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get history of all executed plans"""
        history = []
        files = sorted(self.storage_dir.glob("plan_*.json"), reverse=True)[:limit]

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    history.append(json.load(f))
            except Exception as e:
                logger.error(f"Error reading history file {file_path}: {e}")

        return history

    def get_plan(self, plan_id: str) -> Optional[Dict]:
        """Get specific plan by ID"""
        # First check in memory
        if plan_id in self.execution_plans:
            return self.execution_plans[plan_id].to_dict()

        # Then check in files
        for file_path in self.storage_dir.glob(f"plan_{plan_id}_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading plan file {file_path}: {e}")

        return None

    def get_plan_summary(self, plan_id: str) -> Dict:
        """Get summary of a specific plan"""
        plan = self.get_plan(plan_id)
        if plan:
            return {
                "id": plan["id"],
                "query": plan["query"],
                "created_at": plan["created_at"],
                "completed_at": plan["completed_at"],
                "total_steps": len(plan["steps"]),
                "completed": sum(
                    1 for s in plan["steps"] if s["status"] == "completed"
                ),
                "failed": sum(1 for s in plan["steps"] if s["status"] == "failed"),
                "skipped": sum(1 for s in plan["steps"] if s["status"] == "skipped"),
            }
        return {}

    def clear_history(self):
        """Clear all history"""
        self.execution_plans.clear()
        self.current_plan = None
        for file_path in self.storage_dir.glob("plan_*.json"):
            file_path.unlink()
        logger.info("Step history cleared")
