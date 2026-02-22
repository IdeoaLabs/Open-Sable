"""
Workflow Persistence and Recovery.

Features:
- Save and resume workflows
- Automatic error recovery
- Workflow templates
- State management
- Checkpoint/restore
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib


class WorkflowStatus(Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""

    RETRY = "retry"
    SKIP = "skip"
    FAIL = "fail"
    ROLLBACK = "rollback"


@dataclass
class WorkflowStep:
    """Single step in workflow."""

    id: str
    name: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["recovery_strategy"] = self.recovery_strategy.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        data = data.copy()
        if "recovery_strategy" in data:
            data["recovery_strategy"] = RecoveryStrategy(data["recovery_strategy"])
        return cls(**data)


@dataclass
class StepResult:
    """Result of step execution."""

    step_id: str
    status: WorkflowStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0

    @property
    def duration_seconds(self) -> float:
        """Get execution duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class Checkpoint:
    """Workflow checkpoint for recovery."""

    checkpoint_id: str
    workflow_id: str
    timestamp: datetime
    status: WorkflowStatus
    current_step: Optional[str]
    completed_steps: List[str]
    step_results: Dict[str, StepResult]
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "step_results": {
                step_id: result.to_dict() for step_id, result in self.step_results.items()
            },
            "context": self.context,
        }


@dataclass
class WorkflowDefinition:
    """Workflow definition/template."""

    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDefinition":
        data = data.copy()
        data["steps"] = [WorkflowStep.from_dict(s) for s in data["steps"]]
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class WorkflowEngine:
    """
    Persistent workflow execution engine.

    Features:
    - Execute workflows with steps
    - Save/resume workflows
    - Automatic checkpointing
    - Error recovery
    - Rollback support
    """

    def __init__(self, storage_dir: Optional[str] = None, checkpoint_interval: int = 5):
        """
        Initialize workflow engine.

        Args:
            storage_dir: Directory for storing workflows
            checkpoint_interval: Checkpoint every N steps
        """
        self.storage_dir = (
            Path(storage_dir) if storage_dir else Path.home() / ".opensable" / "workflows"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_interval = checkpoint_interval
        self.action_handlers: Dict[str, Callable] = {}
        self.active_workflows: Dict[str, WorkflowStatus] = {}

    def register_action(self, action_name: str, handler: Callable):
        """Register action handler."""
        self.action_handlers[action_name] = handler

    async def execute(
        self,
        workflow: WorkflowDefinition,
        context: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute workflow.

        Args:
            workflow: Workflow definition
            context: Execution context
            resume_from: Resume from checkpoint ID

        Returns:
            Execution results
        """
        context = context or {}

        # Load checkpoint if resuming
        if resume_from:
            checkpoint = self._load_checkpoint(resume_from)
            if checkpoint:
                return await self._resume_from_checkpoint(workflow, checkpoint)

        # Initialize execution state
        workflow_id = workflow.id
        self.active_workflows[workflow_id] = WorkflowStatus.RUNNING

        step_results: Dict[str, StepResult] = {}
        completed_steps: List[str] = []
        current_step: Optional[str] = None

        try:
            # Execute steps in dependency order
            execution_order = self._resolve_dependencies(workflow.steps)

            for i, step in enumerate(execution_order):
                current_step = step.id

                # Check if step should be executed
                if not self._check_dependencies(step, completed_steps):
                    continue

                # Execute step with retries
                result = await self._execute_step(step, context)
                step_results[step.id] = result

                if result.status == WorkflowStatus.COMPLETED:
                    completed_steps.append(step.id)
                elif result.status == WorkflowStatus.FAILED:
                    # Handle failure based on recovery strategy
                    if not await self._handle_failure(step, result, context):
                        self.active_workflows[workflow_id] = WorkflowStatus.FAILED
                        break

                # Create checkpoint periodically
                if (i + 1) % self.checkpoint_interval == 0:
                    await self._create_checkpoint(
                        workflow_id, current_step, completed_steps, step_results, context
                    )

            # Final status
            if all(step.id in completed_steps for step in workflow.steps):
                self.active_workflows[workflow_id] = WorkflowStatus.COMPLETED

            # Final checkpoint
            await self._create_checkpoint(workflow_id, None, completed_steps, step_results, context)

            return {
                "workflow_id": workflow_id,
                "status": self.active_workflows[workflow_id].value,
                "completed_steps": completed_steps,
                "results": {step_id: result.to_dict() for step_id, result in step_results.items()},
            }

        except Exception as e:
            self.active_workflows[workflow_id] = WorkflowStatus.FAILED

            await self._create_checkpoint(
                workflow_id, current_step, completed_steps, step_results, context
            )

            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "completed_steps": completed_steps,
            }

    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> StepResult:
        """Execute a single step."""
        result = StepResult(
            step_id=step.id, status=WorkflowStatus.RUNNING, started_at=datetime.now()
        )

        # Get action handler
        handler = self.action_handlers.get(step.action)
        if not handler:
            result.status = WorkflowStatus.FAILED
            result.error = f"No handler for action: {step.action}"
            result.completed_at = datetime.now()
            return result

        # Retry loop
        for attempt in range(step.max_retries + 1):
            try:
                # Execute with timeout
                if step.timeout:
                    step_result = await asyncio.wait_for(
                        handler(**step.params, context=context), timeout=step.timeout
                    )
                else:
                    step_result = await handler(**step.params, context=context)

                result.status = WorkflowStatus.COMPLETED
                result.result = step_result
                result.completed_at = datetime.now()
                return result

            except asyncio.TimeoutError:
                result.error = f"Step timed out after {step.timeout}s"
                result.retry_count = attempt

                if attempt < step.max_retries:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                    continue

            except Exception as e:
                result.error = str(e)
                result.retry_count = attempt

                if attempt < step.max_retries:
                    await asyncio.sleep(2**attempt)
                    continue

        result.status = WorkflowStatus.FAILED
        result.completed_at = datetime.now()
        return result

    async def _handle_failure(
        self, step: WorkflowStep, result: StepResult, context: Dict[str, Any]
    ) -> bool:
        """
        Handle step failure.

        Returns:
            True if workflow should continue, False otherwise
        """
        if step.recovery_strategy == RecoveryStrategy.SKIP:
            return True
        elif step.recovery_strategy == RecoveryStrategy.FAIL:
            return False
        elif step.recovery_strategy == RecoveryStrategy.ROLLBACK:
            await self._rollback(step, context)
            return False

        return False

    async def _rollback(self, step: WorkflowStep, context: Dict[str, Any]):
        """Rollback step changes."""
        # Check for rollback handler
        rollback_handler = self.action_handlers.get(f"{step.action}_rollback")
        if rollback_handler:
            await rollback_handler(**step.params, context=context)

    def _resolve_dependencies(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Resolve step dependencies and return execution order."""
        # Topological sort
        step_map = {step.id: step for step in steps}
        in_degree = {step.id: len(step.depends_on) for step in steps}
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            step_id = queue.pop(0)
            result.append(step_map[step_id])

            # Reduce in-degree for dependent steps
            for step in steps:
                if step_id in step.depends_on:
                    in_degree[step.id] -= 1
                    if in_degree[step.id] == 0:
                        queue.append(step.id)

        return result

    def _check_dependencies(self, step: WorkflowStep, completed_steps: List[str]) -> bool:
        """Check if step dependencies are satisfied."""
        return all(dep in completed_steps for dep in step.depends_on)

    async def _create_checkpoint(
        self,
        workflow_id: str,
        current_step: Optional[str],
        completed_steps: List[str],
        step_results: Dict[str, StepResult],
        context: Dict[str, Any],
    ):
        """Create workflow checkpoint."""
        checkpoint_id = hashlib.sha256(
            f"{workflow_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            timestamp=datetime.now(),
            status=self.active_workflows.get(workflow_id, WorkflowStatus.RUNNING),
            current_step=current_step,
            completed_steps=completed_steps,
            step_results=step_results,
            context=context,
        )

        # Save checkpoint
        checkpoint_file = self.storage_dir / f"{checkpoint_id}.json"
        checkpoint_file.write_text(json.dumps(checkpoint.to_dict(), indent=2))

    def _load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load checkpoint from storage."""
        checkpoint_file = self.storage_dir / f"{checkpoint_id}.json"

        if not checkpoint_file.exists():
            return None

        try:
            data = json.loads(checkpoint_file.read_text())

            # Reconstruct checkpoint
            step_results = {}
            for step_id, result_data in data["step_results"].items():
                result = StepResult(
                    step_id=result_data["step_id"],
                    status=WorkflowStatus(result_data["status"]),
                    result=result_data.get("result"),
                    error=result_data.get("error"),
                    retry_count=result_data.get("retry_count", 0),
                )
                if result_data.get("started_at"):
                    result.started_at = datetime.fromisoformat(result_data["started_at"])
                if result_data.get("completed_at"):
                    result.completed_at = datetime.fromisoformat(result_data["completed_at"])

                step_results[step_id] = result

            return Checkpoint(
                checkpoint_id=data["checkpoint_id"],
                workflow_id=data["workflow_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                status=WorkflowStatus(data["status"]),
                current_step=data.get("current_step"),
                completed_steps=data["completed_steps"],
                step_results=step_results,
                context=data["context"],
            )

        except Exception:
            return None

    async def _resume_from_checkpoint(
        self, workflow: WorkflowDefinition, checkpoint: Checkpoint
    ) -> Dict[str, Any]:
        """Resume workflow from checkpoint."""
        # Continue from where we left off
        remaining_steps = [
            step for step in workflow.steps if step.id not in checkpoint.completed_steps
        ]

        # Create new workflow with remaining steps
        resumed_workflow = WorkflowDefinition(
            id=workflow.id,
            name=f"{workflow.name} (resumed)",
            description=workflow.description,
            steps=remaining_steps,
        )

        return await self.execute(resumed_workflow, context=checkpoint.context)


class WorkflowLibrary:
    """
    Library of reusable workflow templates.

    Features:
    - Store and retrieve templates
    - Workflow categories
    - Import/export templates
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize workflow library."""
        self.storage_dir = (
            Path(storage_dir) if storage_dir else Path.home() / ".opensable" / "workflow_templates"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.templates: Dict[str, WorkflowDefinition] = {}
        self._load_templates()
        self._load_default_templates()

    def _load_templates(self):
        """Load templates from storage."""
        for file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                template = WorkflowDefinition.from_dict(data)
                self.templates[template.id] = template
            except Exception:
                pass

    def _load_default_templates(self):
        """Load default workflow templates."""
        # Data processing pipeline
        data_pipeline = WorkflowDefinition(
            id="data_pipeline",
            name="Data Processing Pipeline",
            description="Extract, transform, and load data",
            steps=[
                WorkflowStep(
                    id="extract",
                    name="Extract Data",
                    action="extract_data",
                    params={"source": "database"},
                ),
                WorkflowStep(
                    id="transform",
                    name="Transform Data",
                    action="transform_data",
                    params={"rules": []},
                    depends_on=["extract"],
                ),
                WorkflowStep(
                    id="load",
                    name="Load Data",
                    action="load_data",
                    params={"destination": "warehouse"},
                    depends_on=["transform"],
                ),
            ],
        )

        # Agent deployment
        agent_deployment = WorkflowDefinition(
            id="agent_deployment",
            name="Agent Deployment",
            description="Deploy agent to production",
            steps=[
                WorkflowStep(
                    id="test", name="Run Tests", action="run_tests", params={"test_suite": "all"}
                ),
                WorkflowStep(
                    id="build", name="Build Agent", action="build_agent", depends_on=["test"]
                ),
                WorkflowStep(
                    id="deploy",
                    name="Deploy to Production",
                    action="deploy_agent",
                    params={"environment": "production"},
                    depends_on=["build"],
                ),
                WorkflowStep(
                    id="verify",
                    name="Verify Deployment",
                    action="verify_deployment",
                    depends_on=["deploy"],
                ),
            ],
        )

        for template in [data_pipeline, agent_deployment]:
            if template.id not in self.templates:
                self.templates[template.id] = template
                self._save_template(template)

    def get(self, template_id: str) -> Optional[WorkflowDefinition]:
        """Get template by ID."""
        return self.templates.get(template_id)

    def add(self, template: WorkflowDefinition):
        """Add template to library."""
        self.templates[template.id] = template
        self._save_template(template)

    def list_templates(self) -> List[str]:
        """List all template IDs."""
        return list(self.templates.keys())

    def _save_template(self, template: WorkflowDefinition):
        """Save template to storage."""
        template_file = self.storage_dir / f"{template.id}.json"
        template_file.write_text(json.dumps(template.to_dict(), indent=2))


# Example usage
async def main():
    """Example workflow persistence."""

    print("=" * 50)
    print("Workflow Persistence Examples")
    print("=" * 50)

    # Create workflow engine
    engine = WorkflowEngine()

    # Register action handlers
    async def fetch_data(**kwargs):
        await asyncio.sleep(0.1)
        return {"records": 100}

    async def process_data(**kwargs):
        await asyncio.sleep(0.1)
        return {"processed": 100}

    async def save_results(**kwargs):
        await asyncio.sleep(0.1)
        return {"saved": True}

    engine.register_action("fetch_data", fetch_data)
    engine.register_action("process_data", process_data)
    engine.register_action("save_results", save_results)

    # Create workflow
    workflow = WorkflowDefinition(
        id="example_workflow",
        name="Example Workflow",
        description="Fetch, process, and save data",
        steps=[
            WorkflowStep(
                id="step1", name="Fetch Data", action="fetch_data", params={"source": "api"}
            ),
            WorkflowStep(
                id="step2",
                name="Process Data",
                action="process_data",
                params={"algorithm": "ml"},
                depends_on=["step1"],
            ),
            WorkflowStep(
                id="step3",
                name="Save Results",
                action="save_results",
                params={"destination": "db"},
                depends_on=["step2"],
            ),
        ],
    )

    print("\n1. Execute Workflow")
    print(f"  Workflow: {workflow.name}")
    print(f"  Steps: {len(workflow.steps)}")

    # Execute workflow
    result = await engine.execute(workflow)

    print(f"\n  Status: {result['status']}")
    print(f"  Completed steps: {len(result['completed_steps'])}/{len(workflow.steps)}")

    for step_id, step_result in result["results"].items():
        print(f"    - {step_id}: {step_result['status']} ({step_result['duration_seconds']:.2f}s)")

    # Workflow Library
    print("\n2. Workflow Templates")
    library = WorkflowLibrary()

    templates = library.list_templates()
    print(f"  Available templates: {len(templates)}")

    for template_id in templates:
        template = library.get(template_id)
        if template:
            print(f"    - {template.name}: {len(template.steps)} steps")

    # Load and execute template
    print("\n3. Execute Template")
    template = library.get("data_pipeline")

    if template:
        print(f"  Template: {template.name}")
        print(f"  Description: {template.description}")
        print("  Steps:")
        for step in template.steps:
            deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
            print(f"    {step.id}: {step.name}{deps}")

    print("\n✅ Workflow persistence examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
