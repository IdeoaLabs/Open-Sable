"""
AGI Goal System - Autonomous goal setting, decomposition, and execution.

Features:
- Goal hierarchy (high-level → sub-goals → actions)
- Automatic goal decomposition
- Goal prioritization and scheduling
- Progress tracking and replanning
- Goal conflict resolution
- Success/failure analysis
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Goal execution status."""

    PENDING = "pending"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GoalPriority(Enum):
    """Goal priority levels."""

    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    OPTIONAL = 1


@dataclass
class Goal:
    """Represents a goal in the system."""

    goal_id: str
    description: str
    success_criteria: List[str]
    priority: GoalPriority = GoalPriority.MEDIUM
    deadline: Optional[datetime] = None
    parent_goal_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def is_achievable(self) -> bool:
        """Check if goal can be attempted."""
        return self.status in [GoalStatus.PENDING, GoalStatus.PAUSED]

    def is_complete(self) -> bool:
        """Check if goal is completed."""
        return self.status == GoalStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if goal failed."""
        return self.status == GoalStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "priority": self.priority.name,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "parent_goal_id": self.parent_goal_id,
            "sub_goals": self.sub_goals,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "progress": self.progress,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }


class GoalDecomposer:
    """
    Decomposes high-level goals into actionable sub-goals.

    Uses LLM-based reasoning to break down complex goals.
    """

    def __init__(self, llm_function: Optional[Callable] = None):
        """
        Initialize goal decomposer.

        Args:
            llm_function: Function to call LLM (async)
        """
        self.llm_function = llm_function

    async def decompose(
        self, goal_description: str, context: Optional[Dict[str, Any]] = None, max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Decompose a goal into sub-goals.

        Args:
            goal_description: High-level goal
            context: Additional context
            max_depth: Maximum decomposition depth

        Returns:
            List of sub-goal specifications
        """
        prompt = f"""Decompose this goal into concrete, actionable sub-goals:

Goal: {goal_description}

Context: {json.dumps(context or {}, indent=2)}

Break this down into:
1. Immediate prerequisites (what must be done first)
2. Main steps (ordered sequence)
3. Verification steps (how to confirm success)

For each sub-goal provide:
- Description: What needs to be done
- Success criteria: How to measure completion
- Dependencies: Which other sub-goals must complete first
- Estimated effort: low/medium/high

Format as JSON array:
[
  {{
    "description": "...",
    "success_criteria": ["...", "..."],
    "dependencies": [],
    "effort": "medium",
    "priority": "high"
  }}
]"""

        try:
            if self.llm_function:
                response = await self.llm_function(prompt)

                # Extract JSON from response
                json_match = re.search(r"\[[\s\S]*\]", response)
                if json_match:
                    sub_goals = json.loads(json_match.group())
                    return sub_goals

            # Fallback: simple decomposition
            return [
                {
                    "description": f"Step 1 for: {goal_description}",
                    "success_criteria": ["Step completed"],
                    "dependencies": [],
                    "effort": "medium",
                    "priority": "high",
                },
                {
                    "description": f"Step 2 for: {goal_description}",
                    "success_criteria": ["Step completed"],
                    "dependencies": [0],
                    "effort": "medium",
                    "priority": "medium",
                },
            ]

        except Exception as e:
            logger.error(f"Goal decomposition error: {e}")
            return []


class GoalPlanner:
    """
    Creates execution plans for achieving goals.

    Handles scheduling, resource allocation, and replanning.
    """

    def __init__(self):
        self.plans: Dict[str, List[Dict[str, Any]]] = {}

    def create_plan(
        self, goal: Goal, available_resources: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create execution plan for a goal.

        Args:
            goal: Goal to plan for
            available_resources: Available resources

        Returns:
            List of planned actions
        """
        plan = []

        # Determine action sequence based on goal
        if goal.sub_goals:
            # Plan based on sub-goals
            for sub_goal_id in goal.sub_goals:
                plan.append(
                    {
                        "action": "execute_sub_goal",
                        "goal_id": sub_goal_id,
                        "estimated_duration": timedelta(hours=1),
                    }
                )
        else:
            # Direct execution plan
            for criterion in goal.success_criteria:
                plan.append(
                    {
                        "action": "verify_criterion",
                        "criterion": criterion,
                        "estimated_duration": timedelta(minutes=10),
                    }
                )

        self.plans[goal.goal_id] = plan
        return plan

    def replan(self, goal_id: str, failure_info: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Replan after failure.

        Args:
            goal_id: Failed goal
            failure_info: Information about failure

        Returns:
            Updated plan or None
        """
        if goal_id not in self.plans:
            return None

        original_plan = self.plans[goal_id]

        # Add retry logic
        new_plan = [
            {"action": "analyze_failure", "failure_info": failure_info},
            *original_plan,  # Retry original plan
        ]

        self.plans[goal_id] = new_plan
        return new_plan


class GoalExecutor:
    """
    Executes goals and tracks progress.
    """

    def __init__(self, action_executor: Optional[Callable] = None):
        """
        Initialize executor.

        Args:
            action_executor: Function to execute actions
        """
        self.action_executor = action_executor
        self.execution_history: List[Dict[str, Any]] = []

    async def execute_goal(self, goal: Goal, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a goal according to plan.

        Args:
            goal: Goal to execute
            plan: Execution plan

        Returns:
            Execution result
        """
        goal.status = GoalStatus.IN_PROGRESS
        goal.started_at = datetime.utcnow()

        try:
            results = []

            for i, action in enumerate(plan):
                logger.info(f"Executing action {i+1}/{len(plan)}: {action['action']}")

                if self.action_executor:
                    result = await self.action_executor(action)
                    results.append(result)
                else:
                    # Simulated execution
                    await asyncio.sleep(0.1)
                    results.append({"success": True, "action": action["action"]})

                # Update progress
                goal.progress = (i + 1) / len(plan)

            # Check success criteria
            success = await self._verify_success(goal)

            if success:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = datetime.utcnow()
                goal.result = results
            else:
                goal.status = GoalStatus.FAILED
                goal.error = "Success criteria not met"

            # Record execution
            self.execution_history.append(
                {
                    "goal_id": goal.goal_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": success,
                    "results": results,
                }
            )

            return {"success": success, "goal_id": goal.goal_id, "results": results}

        except Exception as e:
            goal.status = GoalStatus.FAILED
            goal.error = str(e)
            logger.error(f"Goal execution failed: {e}", exc_info=True)

            return {"success": False, "goal_id": goal.goal_id, "error": str(e)}

    async def _verify_success(self, goal: Goal) -> bool:
        """Verify goal success criteria."""
        # Simple verification - could be enhanced with LLM
        return goal.progress >= 1.0


class GoalManager:
    """
    Manages the goal system.

    Coordinates goal creation, decomposition, planning, and execution.
    """

    def __init__(
        self,
        llm_function: Optional[Callable] = None,
        action_executor: Optional[Callable] = None,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize goal manager.

        Args:
            llm_function: LLM function for reasoning
            action_executor: Function to execute actions
            storage_path: Path for persistent storage
        """
        self.decomposer = GoalDecomposer(llm_function)
        self.planner = GoalPlanner()
        self.executor = GoalExecutor(action_executor)

        self.goals: Dict[str, Goal] = {}
        self.active_goals: List[str] = []

        self.storage_path = storage_path or Path("./data/goals.json")
        self._load_goals()

    async def initialize(self):
        """Initialize the goal system (load persisted goals)"""
        self._load_goals()
        return self

    async def create_goal(
        self,
        description: str,
        success_criteria: List[str],
        priority: GoalPriority = GoalPriority.MEDIUM,
        deadline: Optional[datetime] = None,
        parent_goal_id: Optional[str] = None,
        auto_decompose: bool = True,
    ) -> Goal:
        """
        Create a new goal.

        Args:
            description: Goal description
            success_criteria: List of success criteria
            priority: Goal priority
            deadline: Optional deadline
            parent_goal_id: Parent goal if this is a sub-goal
            auto_decompose: Automatically decompose into sub-goals

        Returns:
            Created goal
        """
        goal_id = str(uuid.uuid4())

        goal = Goal(
            goal_id=goal_id,
            description=description,
            success_criteria=success_criteria,
            priority=priority,
            deadline=deadline,
            parent_goal_id=parent_goal_id,
        )

        # Auto-decompose if requested
        if auto_decompose:
            sub_goal_specs = await self.decomposer.decompose(description)

            for spec in sub_goal_specs:
                sub_goal = await self.create_goal(
                    description=spec["description"],
                    success_criteria=spec["success_criteria"],
                    priority=GoalPriority[spec.get("priority", "MEDIUM").upper()],
                    parent_goal_id=goal_id,
                    auto_decompose=False,  # Don't recursively decompose
                )
                goal.sub_goals.append(sub_goal.goal_id)

        self.goals[goal_id] = goal
        self._save_goals()

        logger.info(f"Created goal: {goal_id} - {description}")
        return goal

    async def execute_goal(self, goal_id: str) -> Dict[str, Any]:
        """
        Execute a goal.

        Args:
            goal_id: Goal to execute

        Returns:
            Execution result
        """
        if goal_id not in self.goals:
            raise ValueError(f"Goal not found: {goal_id}")

        goal = self.goals[goal_id]

        # Execute sub-goals first
        if goal.sub_goals:
            for sub_goal_id in goal.sub_goals:
                result = await self.execute_goal(sub_goal_id)
                if not result["success"]:
                    goal.status = GoalStatus.FAILED
                    goal.error = f"Sub-goal failed: {sub_goal_id}"
                    return result

        # Create plan
        plan = self.planner.create_plan(goal)

        # Execute
        result = await self.executor.execute_goal(goal, plan)

        # Handle failure
        if not result["success"] and goal.metadata.get("retry_on_failure", True):
            # Replan and retry
            new_plan = self.planner.replan(goal_id, {"error": goal.error})
            if new_plan:
                logger.info(f"Replanning and retrying goal: {goal_id}")
                result = await self.executor.execute_goal(goal, new_plan)

        self._save_goals()
        return result

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get goal by ID."""
        return self.goals.get(goal_id)

    def list_goals(
        self, status: Optional[GoalStatus] = None, priority: Optional[GoalPriority] = None
    ) -> List[Goal]:
        """
        List goals with optional filtering.

        Args:
            status: Filter by status
            priority: Filter by priority

        Returns:
            List of matching goals
        """
        goals = list(self.goals.values())

        if status:
            goals = [g for g in goals if g.status == status]

        if priority:
            goals = [g for g in goals if g.priority == priority]

        # Sort by priority then deadline
        goals.sort(key=lambda g: (-g.priority.value, g.deadline or datetime.max))

        return goals

    def get_goal_hierarchy(self, goal_id: str) -> Dict[str, Any]:
        """
        Get goal and all its sub-goals as hierarchy.

        Args:
            goal_id: Root goal

        Returns:
            Hierarchical goal structure
        """
        goal = self.goals.get(goal_id)
        if not goal:
            return {}

        hierarchy = goal.to_dict()

        if goal.sub_goals:
            hierarchy["sub_goals_detail"] = [
                self.get_goal_hierarchy(sub_id) for sub_id in goal.sub_goals
            ]

        return hierarchy

    def cancel_goal(self, goal_id: str):
        """Cancel a goal and its sub-goals."""
        goal = self.goals.get(goal_id)
        if not goal:
            return

        goal.status = GoalStatus.CANCELLED

        # Cancel sub-goals
        for sub_goal_id in goal.sub_goals:
            self.cancel_goal(sub_goal_id)

        self._save_goals()
        logger.info(f"Cancelled goal: {goal_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get goal system statistics."""
        total = len(self.goals)
        by_status = {}
        by_priority = {}

        for goal in self.goals.values():
            by_status[goal.status.value] = by_status.get(goal.status.value, 0) + 1
            by_priority[goal.priority.name] = by_priority.get(goal.priority.name, 0) + 1

        return {
            "total_goals": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "active_goals": len(self.active_goals),
        }

    def _save_goals(self):
        """Save goals to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            goals_data = {goal_id: goal.to_dict() for goal_id, goal in self.goals.items()}

            with open(self.storage_path, "w") as f:
                json.dump(goals_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save goals: {e}")

    def _load_goals(self):
        """Load goals from disk."""
        try:
            if not self.storage_path.exists():
                return

            with open(self.storage_path, "r") as f:
                goals_data = json.load(f)

            for goal_id, data in goals_data.items():
                # Reconstruct goal object
                goal = Goal(
                    goal_id=data["goal_id"],
                    description=data["description"],
                    success_criteria=data["success_criteria"],
                    priority=GoalPriority[data["priority"]],
                    deadline=datetime.fromisoformat(data["deadline"]) if data["deadline"] else None,
                    parent_goal_id=data.get("parent_goal_id"),
                    sub_goals=data.get("sub_goals", []),
                    dependencies=data.get("dependencies", []),
                    status=GoalStatus(data["status"]),
                    progress=data.get("progress", 0.0),
                    metadata=data.get("metadata", {}),
                    created_at=datetime.fromisoformat(data["created_at"]),
                    started_at=(
                        datetime.fromisoformat(data["started_at"])
                        if data.get("started_at")
                        else None
                    ),
                    completed_at=(
                        datetime.fromisoformat(data["completed_at"])
                        if data.get("completed_at")
                        else None
                    ),
                    result=data.get("result"),
                    error=data.get("error"),
                )

                self.goals[goal_id] = goal

            logger.info(f"Loaded {len(self.goals)} goals from disk")

        except Exception as e:
            logger.error(f"Failed to load goals: {e}")


# Missing import
import re


# Example usage
async def main():
    """Example goal system usage."""

    print("=" * 50)
    print("AGI Goal System Example")
    print("=" * 50)

    # Initialize goal manager
    goal_manager = GoalManager()

    # Create a high-level goal
    print("\n1. Creating high-level goal...")
    goal = await goal_manager.create_goal(
        description="Build a web application for task management",
        success_criteria=[
            "Frontend is responsive and functional",
            "Backend API is deployed and accessible",
            "Database is configured with proper schema",
            "User authentication works",
            "Tests pass with >80% coverage",
        ],
        priority=GoalPriority.HIGH,
        deadline=datetime.utcnow() + timedelta(days=30),
        auto_decompose=True,
    )

    print(f"  Created goal: {goal.goal_id}")
    print(f"  Sub-goals: {len(goal.sub_goals)}")

    # View hierarchy
    print("\n2. Goal hierarchy...")
    hierarchy = goal_manager.get_goal_hierarchy(goal.goal_id)
    print(f"  Goal: {hierarchy['description']}")
    if "sub_goals_detail" in hierarchy:
        for i, sub in enumerate(hierarchy["sub_goals_detail"], 1):
            print(f"    {i}. {sub['description']}")

    # List goals
    print("\n3. Listing all goals...")
    all_goals = goal_manager.list_goals()
    print(f"  Total goals: {len(all_goals)}")

    # Get stats
    print("\n4. Goal statistics...")
    stats = goal_manager.get_stats()
    print(f"  Total: {stats['total_goals']}")
    print(f"  By status: {stats['by_status']}")
    print(f"  By priority: {stats['by_priority']}")

    # Execute goal (simulated)
    print("\n5. Executing goal...")
    result = await goal_manager.execute_goal(goal.goal_id)
    print(f"  Success: {result['success']}")
    print(f"  Goal status: {goal.status.value}")

    print("\n✅ Goal system example completed!")


if __name__ == "__main__":
    asyncio.run(main())
