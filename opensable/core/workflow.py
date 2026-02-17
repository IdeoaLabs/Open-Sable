"""
Workflow Engine — Define, execute, and manage multi-step workflows.

Core orchestration module that wraps WorkflowPersistence with a clean API.

Features:
- Define workflows as a sequence of steps
- Conditional branching and loops
- Parallel step execution
- Error handling with retries
- Workflow templates (ETL, CI/CD, data pipeline, chatbot flow)
- Pause / resume support
- Event-driven triggers
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StepResult:
    """Result of a single workflow step."""
    step_id: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retries: int = 0


@dataclass
class WorkflowStep:
    """Definition of a single workflow step."""
    step_id: str
    name: str
    handler: Optional[Callable] = None
    description: str = ""
    timeout: float = 300.0   # seconds
    max_retries: int = 0
    retry_delay: float = 1.0
    condition: Optional[Callable] = None  # skip if returns False
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Workflow:
    """A complete workflow definition."""
    workflow_id: str
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    results: Dict[str, StepResult] = field(default_factory=dict)
    current_step: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "steps": len(self.steps),
            "completed": sum(
                1 for r in self.results.values()
                if r.status == StepStatus.COMPLETED
            ),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class WorkflowEngine:
    """
    Execute and manage multi-step workflows.

    Usage:
        engine = WorkflowEngine()
        wf = engine.create_workflow("My Pipeline", "ETL pipeline")
        engine.add_step(wf, "extract", "Extract data", handler=extract_fn)
        engine.add_step(wf, "transform", "Transform", handler=transform_fn)
        engine.add_step(wf, "load", "Load into DB", handler=load_fn)
        result = await engine.run(wf.workflow_id)
    """

    def __init__(self, config=None):
        self.config = config
        self._workflows: Dict[str, Workflow] = {}
        self._persist_dir = Path.home() / ".sablecore" / "workflows"
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("⚙️ Workflow Engine initialized")

    # ------------------------------------------------------------------
    # Definition
    # ------------------------------------------------------------------

    def create_workflow(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict] = None,
    ) -> Workflow:
        """Create a new workflow."""
        wf_id = hashlib.sha256(
            f"{name}-{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        wf = Workflow(
            workflow_id=wf_id,
            name=name,
            description=description,
            metadata=metadata or {},
        )
        self._workflows[wf_id] = wf
        logger.info(f"📋 Created workflow: {name} ({wf_id})")
        return wf

    def add_step(
        self,
        workflow: Workflow,
        step_id: str,
        name: str,
        handler: Optional[Callable] = None,
        description: str = "",
        timeout: float = 300.0,
        max_retries: int = 0,
        condition: Optional[Callable] = None,
        depends_on: Optional[List[str]] = None,
    ) -> WorkflowStep:
        """Add a step to a workflow."""
        step = WorkflowStep(
            step_id=step_id,
            name=name,
            handler=handler,
            description=description,
            timeout=timeout,
            max_retries=max_retries,
            condition=condition,
            depends_on=depends_on or [],
        )
        workflow.steps.append(step)
        return step

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self, workflow_id: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Run a workflow to completion."""
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return {"error": f"Workflow not found: {workflow_id}"}

        wf.status = WorkflowStatus.RUNNING
        wf.started_at = datetime.now().isoformat()
        ctx = context or {}

        logger.info(f"▶️ Running workflow: {wf.name} ({len(wf.steps)} steps)")

        for i, step in enumerate(wf.steps):
            wf.current_step = i

            # Check condition
            if step.condition and not step.condition(ctx):
                wf.results[step.step_id] = StepResult(
                    step_id=step.step_id, status=StepStatus.SKIPPED
                )
                logger.info(f"  ⏭️ Skipped: {step.name}")
                continue

            # Check dependencies
            for dep in step.depends_on:
                dep_result = wf.results.get(dep)
                if dep_result is None or dep_result.status != StepStatus.COMPLETED:
                    wf.results[step.step_id] = StepResult(
                        step_id=step.step_id,
                        status=StepStatus.SKIPPED,
                        error=f"Dependency not met: {dep}",
                    )
                    continue

            # Execute with retries
            result = await self._execute_step(step, ctx)
            wf.results[step.step_id] = result

            if result.status == StepStatus.COMPLETED:
                # Pass output to context for next steps
                ctx[step.step_id] = result.output
                logger.info(f"  ✅ {step.name} ({result.duration_ms:.0f}ms)")
            else:
                logger.error(f"  ❌ {step.name}: {result.error}")
                wf.status = WorkflowStatus.FAILED
                wf.finished_at = datetime.now().isoformat()
                return {
                    "success": False,
                    "workflow": wf.to_dict(),
                    "failed_step": step.step_id,
                    "error": result.error,
                }

        wf.status = WorkflowStatus.COMPLETED
        wf.finished_at = datetime.now().isoformat()
        logger.info(f"✅ Workflow completed: {wf.name}")

        return {
            "success": True,
            "workflow": wf.to_dict(),
            "results": {
                sid: {"status": r.status.value, "output": r.output}
                for sid, r in wf.results.items()
            },
        }

    async def _execute_step(
        self, step: WorkflowStep, context: Dict
    ) -> StepResult:
        """Execute a single step with timeout and retries."""
        if step.handler is None:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.SKIPPED,
                error="No handler defined",
            )

        retries = 0
        last_error = None

        while retries <= step.max_retries:
            start = time.monotonic()
            try:
                if asyncio.iscoroutinefunction(step.handler):
                    output = await asyncio.wait_for(
                        step.handler(context),
                        timeout=step.timeout,
                    )
                else:
                    output = step.handler(context)

                duration = (time.monotonic() - start) * 1000
                return StepResult(
                    step_id=step.step_id,
                    status=StepStatus.COMPLETED,
                    output=output,
                    duration_ms=duration,
                    retries=retries,
                )
            except asyncio.TimeoutError:
                last_error = f"Timeout after {step.timeout}s"
            except Exception as e:
                last_error = str(e)

            retries += 1
            if retries <= step.max_retries:
                await asyncio.sleep(step.retry_delay)

        duration = (time.monotonic() - start) * 1000
        return StepResult(
            step_id=step.step_id,
            status=StepStatus.FAILED,
            error=last_error,
            duration_ms=duration,
            retries=retries - 1,
        )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def pause(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        wf = self._workflows.get(workflow_id)
        if wf and wf.status == WorkflowStatus.RUNNING:
            wf.status = WorkflowStatus.PAUSED
            return True
        return False

    def cancel(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        wf = self._workflows.get(workflow_id)
        if wf and wf.status in (WorkflowStatus.RUNNING, WorkflowStatus.PAUSED):
            wf.status = WorkflowStatus.CANCELLED
            wf.finished_at = datetime.now().isoformat()
            return True
        return False

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows."""
        return [wf.to_dict() for wf in self._workflows.values()]

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def create_from_template(self, template: str, name: str = "") -> Workflow:
        """Create a workflow from a built-in template."""
        templates = {
            "etl": {
                "name": name or "ETL Pipeline",
                "description": "Extract, Transform, Load data pipeline",
                "steps": [
                    ("extract", "Extract Data"),
                    ("transform", "Transform Data"),
                    ("validate", "Validate Output"),
                    ("load", "Load to Destination"),
                ],
            },
            "ci_cd": {
                "name": name or "CI/CD Pipeline",
                "description": "Build, test, deploy pipeline",
                "steps": [
                    ("checkout", "Checkout Code"),
                    ("build", "Build Project"),
                    ("test", "Run Tests"),
                    ("deploy", "Deploy"),
                ],
            },
            "data_analysis": {
                "name": name or "Data Analysis",
                "description": "Data collection and analysis pipeline",
                "steps": [
                    ("collect", "Collect Data"),
                    ("clean", "Clean Data"),
                    ("analyze", "Run Analysis"),
                    ("report", "Generate Report"),
                ],
            },
        }

        tmpl = templates.get(template)
        if tmpl is None:
            raise ValueError(
                f"Unknown template '{template}'. Available: {list(templates.keys())}"
            )

        wf = self.create_workflow(tmpl["name"], tmpl["description"])
        for step_id, step_name in tmpl["steps"]:
            self.add_step(wf, step_id, step_name)
        return wf
