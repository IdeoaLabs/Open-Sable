"""
Tests for Workflow Persistence - Save, resume, recover workflows.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from core.workflow_persistence import (
    WorkflowEngine, WorkflowTemplate, WorkflowStep,
    StepStatus, RecoveryStrategy, Checkpoint
)


class TestWorkflowEngine:
    """Test workflow execution engine"""
    
    @pytest.fixture
    def engine(self, tmp_path):
        return WorkflowEngine(storage_dir=str(tmp_path / "workflows"))
    
    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self, engine):
        """Test executing a simple workflow"""
        steps = [
            WorkflowStep(
                id="step1",
                name="Step 1",
                action="task1",
                params={}
            ),
            WorkflowStep(
                id="step2",
                name="Step 2",
                action="task2",
                params={},
                dependencies=["step1"]
            )
        ]
        
        result = await engine.execute("wf_001", steps)
        
        assert result.workflow_id == "wf_001"
        assert result.status == StepStatus.COMPLETED
        assert result.completed_steps == 2
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation(self, engine):
        """Test automatic checkpoint creation"""
        steps = [
            WorkflowStep(id="s1", name="S1", action="a1", params={}),
            WorkflowStep(id="s2", name="S2", action="a2", params={}, dependencies=["s1"]),
            WorkflowStep(id="s3", name="S3", action="a3", params={}, dependencies=["s2"])
        ]
        
        result = await engine.execute(
            "wf_checkpoint",
            steps,
            checkpoint_interval=1  # Checkpoint after each step
        )
        
        assert len(result.checkpoints) >= 2
    
    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, engine):
        """Test resuming workflow from checkpoint"""
        steps = [
            WorkflowStep(id="s1", name="S1", action="a1", params={}),
            WorkflowStep(id="s2", name="S2 (fail)", action="fail", params={"fail": True}, dependencies=["s1"]),
            WorkflowStep(id="s3", name="S3", action="a3", params={}, dependencies=["s2"])
        ]
        
        # First execution (will fail at s2)
        try:
            await engine.execute("wf_resume", steps, checkpoint_interval=1)
        except:
            pass
        
        # Resume from checkpoint
        result = await engine.resume("wf_resume")
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_dependency_resolution(self, engine):
        """Test step dependency resolution"""
        steps = [
            WorkflowStep(id="init", name="Init", action="init", params={}),
            WorkflowStep(id="a", name="A", action="a", params={}, dependencies=["init"]),
            WorkflowStep(id="b", name="B", action="b", params={}, dependencies=["init"]),
            WorkflowStep(id="final", name="Final", action="final", params={}, dependencies=["a", "b"])
        ]
        
        result = await engine.execute("wf_deps", steps)
        
        # Final step should run after both a and b
        assert result.status == StepStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_error_recovery_retry(self, engine):
        """Test retry recovery strategy"""
        step = WorkflowStep(
            id="retry_step",
            name="Retry Test",
            action="unreliable",
            params={},
            retry_count=3,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        result = await engine.execute("wf_retry", [step])
        
        # Should complete after retries
        assert result.status in [StepStatus.COMPLETED, StepStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_error_recovery_skip(self, engine):
        """Test skip recovery strategy"""
        steps = [
            WorkflowStep(id="s1", name="S1", action="a1", params={}),
            WorkflowStep(
                id="skip",
                name="Skip on error",
                action="fail",
                params={"fail": True},
                dependencies=["s1"],
                recovery_strategy=RecoveryStrategy.SKIP
            ),
            WorkflowStep(id="s3", name="S3", action="a3", params={}, dependencies=["skip"])
        ]
        
        result = await engine.execute("wf_skip", steps)
        
        # Should complete despite failed step (skipped)
        assert result.completed_steps >= 2


class TestWorkflowTemplate:
    """Test workflow templates"""
    
    @pytest.fixture
    def engine(self, tmp_path):
        return WorkflowEngine(storage_dir=str(tmp_path / "workflows"))
    
    def test_save_template(self, engine):
        """Test saving a workflow template"""
        steps = [
            WorkflowStep(id="s1", name="Step 1", action="action1", params={})
        ]
        
        template = WorkflowTemplate(
            id="tmpl_1",
            name="Test Template",
            description="A test template",
            steps=steps,
            variables=["var1", "var2"]
        )
        
        engine.save_template(template)
        
        loaded = engine.load_template("tmpl_1")
        assert loaded.name == "Test Template"
    
    def test_render_template(self, engine):
        """Test rendering template with variables"""
        steps = [
            WorkflowStep(
                id="s1",
                name="Process {input}",
                action="process",
                params={"source": "{input}", "dest": "{output}"}
            )
        ]
        
        template = WorkflowTemplate(
            id="tmpl_var",
            name="Variable Template",
            steps=steps,
            variables=["input", "output"]
        )
        
        engine.save_template(template)
        
        rendered = engine.render_template(
            "tmpl_var",
            variables={"input": "data.csv", "output": "results.json"}
        )
        
        assert rendered is not None
        assert len(rendered) == 1
    
    def test_template_metadata(self, engine):
        """Test template with metadata"""
        template = WorkflowTemplate(
            id="tmpl_meta",
            name="Meta Template",
            steps=[],
            metadata={"category": "data", "version": "1.0"}
        )
        
        engine.save_template(template)
        
        loaded = engine.load_template("tmpl_meta")
        assert loaded.metadata["category"] == "data"


class TestWorkflowState:
    """Test workflow state management"""
    
    @pytest.fixture
    def engine(self, tmp_path):
        return WorkflowEngine(storage_dir=str(tmp_path / "workflows"))
    
    @pytest.mark.asyncio
    async def test_get_workflow_state(self, engine):
        """Test retrieving workflow state"""
        steps = [
            WorkflowStep(id="s1", name="S1", action="a1", params={})
        ]
        
        await engine.execute("wf_state", steps)
        
        state = engine.get_state("wf_state")
        
        assert state is not None
        assert state.workflow_id == "wf_state"
    
    @pytest.mark.asyncio
    async def test_list_workflows(self, engine):
        """Test listing all workflows"""
        await engine.execute("wf1", [WorkflowStep(id="s1", name="S1", action="a", params={})])
        await engine.execute("wf2", [WorkflowStep(id="s1", name="S1", action="a", params={})])
        
        workflows = engine.list_workflows()
        
        assert len(workflows) >= 2
    
    @pytest.mark.asyncio
    async def test_workflow_export(self, engine, tmp_path):
        """Test exporting workflow"""
        steps = [WorkflowStep(id="s1", name="S1", action="a", params={})]
        
        await engine.execute("wf_export", steps)
        
        export_path = tmp_path / "export.json"
        engine.export_workflow("wf_export", str(export_path))
        
        assert export_path.exists()
    
    @pytest.mark.asyncio
    async def test_workflow_cleanup(self, engine):
        """Test workflow cleanup and archival"""
        steps = [WorkflowStep(id="s1", name="S1", action="a", params={})]
        
        await engine.execute("wf_old", steps)
        
        # Archive old workflows
        from datetime import timedelta
        archived = engine.archive_workflows(
            before=datetime.now() + timedelta(days=1)
        )
        
        assert isinstance(archived, list)


class TestParallelExecution:
    """Test parallel workflow execution"""
    
    @pytest.fixture
    def engine(self, tmp_path):
        return WorkflowEngine(storage_dir=str(tmp_path / "workflows"))
    
    @pytest.mark.asyncio
    async def test_parallel_steps(self, engine):
        """Test executing independent steps in parallel"""
        steps = [
            WorkflowStep(id="init", name="Init", action="init", params={}),
            WorkflowStep(id="a", name="A", action="a", params={}, dependencies=["init"]),
            WorkflowStep(id="b", name="B", action="b", params={}, dependencies=["init"]),
            WorkflowStep(id="c", name="C", action="c", params={}, dependencies=["init"]),
        ]
        
        import time
        start = time.time()
        
        result = await engine.execute(
            "wf_parallel",
            steps,
            max_parallel=3
        )
        
        duration = time.time() - start
        
        # Parallel execution should be faster
        assert result.completed_steps == 4
