"""
Tests for Workflow Persistence - Save, resume, recover workflows.
"""

import pytest
from datetime import datetime

from opensable.core.workflow_persistence import (
    WorkflowEngine, WorkflowDefinition, WorkflowStep,
    WorkflowStatus, RecoveryStrategy, Checkpoint, StepResult
)


class TestWorkflowStatus:
    """Test WorkflowStatus enum."""

    def test_members(self):
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"
        assert WorkflowStatus.PAUSED.value == "paused"


class TestRecoveryStrategy:
    """Test RecoveryStrategy enum."""

    def test_members(self):
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.SKIP.value == "skip"
        assert RecoveryStrategy.FAIL.value == "fail"
        assert RecoveryStrategy.ROLLBACK.value == "rollback"


class TestWorkflowStep:
    """Test WorkflowStep dataclass."""

    def test_create(self):
        step = WorkflowStep(id="s1", name="Step 1", action="do_thing")
        assert step.id == "s1"
        assert step.name == "Step 1"
        assert step.action == "do_thing"

    def test_defaults(self):
        step = WorkflowStep(id="s", name="S", action="a")
        assert step.params == {}
        assert step.depends_on == []
        assert step.retry_count == 0
        assert step.max_retries == 3
        assert step.timeout is None
        assert step.recovery_strategy == RecoveryStrategy.RETRY

    def test_with_deps(self):
        step = WorkflowStep(id="s2", name="S2", action="b", depends_on=["s1"])
        assert "s1" in step.depends_on

    def test_to_dict(self):
        step = WorkflowStep(id="s", name="S", action="a", params={"k": "v"})
        d = step.to_dict()
        assert d["id"] == "s"
        assert d["params"] == {"k": "v"}
        assert d["recovery_strategy"] == "retry"

    def test_from_dict(self):
        data = {"id": "s1", "name": "S1", "action": "act", "params": {},
                "depends_on": [], "retry_count": 0, "max_retries": 3,
                "timeout": None, "recovery_strategy": "skip"}
        step = WorkflowStep.from_dict(data)
        assert step.recovery_strategy == RecoveryStrategy.SKIP


class TestWorkflowDefinition:
    """Test WorkflowDefinition dataclass."""

    def test_create(self):
        steps = [
            WorkflowStep(id="a", name="A", action="do_a"),
            WorkflowStep(id="b", name="B", action="do_b", depends_on=["a"]),
        ]
        defn = WorkflowDefinition(
            id="wf1", name="Test", description="A test", steps=steps
        )
        assert defn.id == "wf1"
        assert len(defn.steps) == 2

    def test_metadata(self):
        defn = WorkflowDefinition(
            id="wf2", name="Meta", description="", steps=[],
            metadata={"author": "test"}
        )
        assert defn.metadata["author"] == "test"

    def test_to_dict(self):
        defn = WorkflowDefinition(
            id="wf", name="N", description="D",
            steps=[WorkflowStep(id="s", name="S", action="a")]
        )
        d = defn.to_dict()
        assert d["id"] == "wf"
        assert len(d["steps"]) == 1

    def test_from_dict(self):
        data = {
            "id": "wf1", "name": "N", "description": "D",
            "steps": [{"id": "s1", "name": "S1", "action": "a", "params": {},
                       "depends_on": [], "retry_count": 0, "max_retries": 3,
                       "timeout": None, "recovery_strategy": "retry"}],
            "metadata": {},
            "created_at": "2024-01-01T00:00:00"
        }
        defn = WorkflowDefinition.from_dict(data)
        assert defn.id == "wf1"
        assert len(defn.steps) == 1


class TestStepResult:
    """Test StepResult dataclass."""

    def test_create(self):
        r = StepResult(step_id="s1", status=WorkflowStatus.COMPLETED, result="ok")
        assert r.step_id == "s1"
        assert r.status == WorkflowStatus.COMPLETED

    def test_duration(self):
        now = datetime.now()
        from datetime import timedelta
        r = StepResult(
            step_id="s1", status=WorkflowStatus.COMPLETED,
            started_at=now, completed_at=now + timedelta(seconds=5)
        )
        assert r.duration_seconds == pytest.approx(5.0, abs=0.1)

    def test_to_dict(self):
        r = StepResult(step_id="s1", status=WorkflowStatus.FAILED, error="boom")
        d = r.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "boom"


class TestCheckpoint:
    """Test Checkpoint dataclass."""

    def test_create(self):
        cp = Checkpoint(
            checkpoint_id="cp1",
            workflow_id="wf1",
            timestamp=datetime.now(),
            status=WorkflowStatus.RUNNING,
            current_step="s2",
            completed_steps=["s1"],
            step_results={},
            context={"key": "val"},
        )
        assert cp.checkpoint_id == "cp1"
        assert cp.workflow_id == "wf1"
        assert cp.current_step == "s2"
        assert "s1" in cp.completed_steps

    def test_to_dict(self):
        cp = Checkpoint(
            checkpoint_id="cp2",
            workflow_id="wf2",
            timestamp=datetime.now(),
            status=WorkflowStatus.COMPLETED,
            current_step=None,
            completed_steps=["s1"],
            step_results={},
            context={},
        )
        d = cp.to_dict()
        assert d["checkpoint_id"] == "cp2"
        assert d["status"] == "completed"


class TestWorkflowEngine:
    """Test workflow execution engine."""

    @pytest.fixture
    def engine(self, tmp_path):
        return WorkflowEngine(storage_dir=str(tmp_path / "workflows"))

    def test_init(self, engine):
        assert engine.action_handlers == {}
        assert engine.active_workflows == {}

    def test_register_action(self, engine):
        async def handler(**kwargs):
            return "done"
        engine.register_action("my_action", handler)
        assert "my_action" in engine.action_handlers

    @pytest.mark.asyncio
    async def test_execute_with_handler(self, engine):
        async def handler(**kwargs):
            return {"result": "ok"}

        engine.register_action("task1", handler)
        defn = WorkflowDefinition(
            id="wf1", name="Test", description="",
            steps=[WorkflowStep(id="s1", name="S1", action="task1")]
        )
        result = await engine.execute(defn)
        assert isinstance(result, dict)
        assert result["status"] == "completed"
        assert "s1" in result["completed_steps"]

    @pytest.mark.asyncio
    async def test_execute_no_handler_fails(self, engine):
        """Steps with no handler should fail."""
        defn = WorkflowDefinition(
            id="wf2", name="Fail", description="",
            steps=[WorkflowStep(id="s1", name="S1", action="nonexistent")]
        )
        result = await engine.execute(defn)
        assert isinstance(result, dict)
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_multi_step(self, engine):
        async def handler(**kwargs):
            return "ok"

        engine.register_action("a", handler)
        engine.register_action("b", handler)

        defn = WorkflowDefinition(
            id="wf3", name="Multi", description="",
            steps=[
                WorkflowStep(id="s1", name="S1", action="a"),
                WorkflowStep(id="s2", name="S2", action="b", depends_on=["s1"]),
            ]
        )
        result = await engine.execute(defn)
        assert result["status"] == "completed"
        assert "s1" in result["completed_steps"]
        assert "s2" in result["completed_steps"]
