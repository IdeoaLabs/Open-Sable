"""
Workflow Persistence Examples - Save, resume, and recover workflows.

Demonstrates workflow checkpointing, state persistence, error recovery, and templates.
"""

import asyncio
from core.workflow_persistence import (
    WorkflowEngine,
    WorkflowTemplate,
    WorkflowStep,
    StepStatus,
    RecoveryStrategy,
)


async def main():
    """Run workflow persistence examples."""

    print("=" * 60)
    print("Workflow Persistence Examples")
    print("=" * 60)

    # Example 1: Workflow engine setup
    print("\n1. Workflow Engine Setup")
    print("-" * 40)

    engine = WorkflowEngine(storage_dir="/tmp/opensable_workflows")

    print("Initialized workflow engine")
    print(f"Storage: {engine.storage_dir}")

    # Example 2: Create workflow steps
    print("\n2. Create Workflow Steps")
    print("-" * 40)

    steps = [
        WorkflowStep(
            id="step_1",
            name="Validate Input",
            action="validate_data",
            params={"schema": "user_input"},
            retry_count=3,
        ),
        WorkflowStep(
            id="step_2",
            name="Process Data",
            action="transform_data",
            params={"format": "json"},
            dependencies=["step_1"],
        ),
        WorkflowStep(
            id="step_3",
            name="Store Results",
            action="save_to_db",
            params={"table": "processed_data"},
            dependencies=["step_2"],
        ),
        WorkflowStep(
            id="step_4",
            name="Send Notification",
            action="notify_user",
            params={"channel": "email"},
            dependencies=["step_3"],
        ),
    ]

    print(f"Created {len(steps)} workflow steps:")
    for step in steps:
        deps = f" (depends on: {', '.join(step.dependencies)})" if step.dependencies else ""
        print(f"  {step.id}: {step.name}{deps}")

    # Example 3: Execute workflow with checkpoints
    print("\n3. Execute Workflow with Checkpoints")
    print("-" * 40)

    workflow_id = "wf_data_processing_001"

    result = await engine.execute(
        workflow_id=workflow_id, steps=steps, checkpoint_interval=1  # Checkpoint after each step
    )

    print("Workflow execution:")
    print(f"  ID: {result.workflow_id}")
    print(f"  Status: {result.status}")
    print(f"  Completed steps: {result.completed_steps}/{result.total_steps}")
    print(f"  Duration: {result.duration_ms:.2f}ms")
    print(f"  Checkpoints saved: {len(result.checkpoints)}")

    # Example 4: View step results
    print("\n4. Step Execution Results")
    print("-" * 40)

    for step_result in result.step_results:
        status_emoji = "✅" if step_result.status == StepStatus.COMPLETED else "❌"
        print(f"{status_emoji} {step_result.step_id}: {step_result.status.value}")
        print(f"   Duration: {step_result.duration_ms:.2f}ms")
        if step_result.output:
            print(f"   Output: {step_result.output}")

    # Example 5: Resume from checkpoint
    print("\n5. Resume from Checkpoint")
    print("-" * 40)

    # Simulate interruption - create new workflow that fails midway
    interrupted_steps = [
        WorkflowStep(id="step_a", name="Step A", action="task_a", params={}),
        WorkflowStep(
            id="step_b",
            name="Step B (will fail)",
            action="task_b",
            params={"fail": True},
            dependencies=["step_a"],
        ),
        WorkflowStep(
            id="step_c", name="Step C", action="task_c", params={}, dependencies=["step_b"]
        ),
    ]

    interrupted_id = "wf_interrupted_001"

    # First attempt (will fail at step B)
    try:
        await engine.execute(interrupted_id, interrupted_steps, checkpoint_interval=1)
    except Exception as e:
        print(f"Workflow failed: {str(e)[:50]}...")

    # Check last checkpoint
    last_checkpoint = engine.get_checkpoint(interrupted_id)
    if last_checkpoint:
        print("\nLast checkpoint:")
        print(f"  Workflow: {last_checkpoint.workflow_id}")
        print(f"  Completed: {last_checkpoint.completed_steps}")
        print(f"  Timestamp: {last_checkpoint.timestamp}")

    # Resume from checkpoint
    print("\nResuming from checkpoint...")
    resumed_result = await engine.resume(interrupted_id)

    print("Resumed workflow:")
    print(f"  Status: {resumed_result.status}")
    print(f"  Completed: {resumed_result.completed_steps}/{resumed_result.total_steps}")

    # Example 6: Error recovery strategies
    print("\n6. Error Recovery Strategies")
    print("-" * 40)

    recovery_steps = [
        WorkflowStep(
            id="retry_step",
            name="Retry on failure",
            action="unreliable_api",
            params={},
            retry_count=3,
            recovery_strategy=RecoveryStrategy.RETRY,
        ),
        WorkflowStep(
            id="skip_step",
            name="Skip on failure",
            action="optional_task",
            params={},
            recovery_strategy=RecoveryStrategy.SKIP,
        ),
        WorkflowStep(
            id="rollback_step",
            name="Rollback on failure",
            action="critical_update",
            params={},
            recovery_strategy=RecoveryStrategy.ROLLBACK,
            rollback_action="undo_update",
        ),
    ]

    print("Recovery strategies configured:")
    for step in recovery_steps:
        print(f"  {step.name}: {step.recovery_strategy.value}")

    # Example 7: Workflow templates
    print("\n7. Workflow Templates")
    print("-" * 40)

    # Create template
    template = WorkflowTemplate(
        id="tmpl_data_pipeline",
        name="Standard Data Pipeline",
        description="ETL workflow for data processing",
        steps=steps,
        variables=["input_source", "output_table"],
        metadata={"category": "data", "version": "1.0"},
    )

    # Save template
    engine.save_template(template)
    print(f"Saved template: {template.name}")
    print(f"  Variables: {', '.join(template.variables)}")
    print(f"  Steps: {len(template.steps)}")

    # Load template
    loaded = engine.load_template("tmpl_data_pipeline")
    if loaded:
        print(f"\nLoaded template: {loaded.name}")
        print(f"  Description: {loaded.description}")

    # Example 8: Execute from template
    print("\n8. Execute from Template")
    print("-" * 40)

    # Render template with variables
    rendered_steps = engine.render_template(
        "tmpl_data_pipeline",
        variables={
            "input_source": "s3://data-bucket/input",
            "output_table": "analytics.processed_data",
        },
    )

    if rendered_steps:
        print(f"Rendered {len(rendered_steps)} steps from template")

        # Execute rendered workflow
        template_result = await engine.execute(
            workflow_id="wf_from_template_001", steps=rendered_steps
        )

        print("Template workflow executed:")
        print(f"  Status: {template_result.status}")
        print(f"  Completed: {template_result.completed_steps}/{template_result.total_steps}")

    # Example 9: Parallel step execution
    print("\n9. Parallel Step Execution")
    print("-" * 40)

    parallel_steps = [
        WorkflowStep(id="init", name="Initialize", action="init", params={}),
        # These can run in parallel (no mutual dependencies)
        WorkflowStep(
            id="task_a", name="Task A", action="process_a", params={}, dependencies=["init"]
        ),
        WorkflowStep(
            id="task_b", name="Task B", action="process_b", params={}, dependencies=["init"]
        ),
        WorkflowStep(
            id="task_c", name="Task C", action="process_c", params={}, dependencies=["init"]
        ),
        # Final step waits for all parallel tasks
        WorkflowStep(
            id="finalize",
            name="Finalize",
            action="finalize",
            params={},
            dependencies=["task_a", "task_b", "task_c"],
        ),
    ]

    import time

    start = time.time()

    parallel_result = await engine.execute(
        workflow_id="wf_parallel_001",
        steps=parallel_steps,
        max_parallel=3,  # Run up to 3 steps in parallel
    )

    duration = time.time() - start

    print(f"Parallel workflow completed in {duration:.2f}s")
    print(f"Steps executed: {parallel_result.completed_steps}")
    print("Note: task_a, task_b, task_c ran in parallel")

    # Example 10: Workflow state persistence
    print("\n10. Workflow State Persistence")
    print("-" * 40)

    # Get workflow state
    state = engine.get_state(workflow_id)

    if state:
        print(f"Workflow state for {workflow_id}:")
        print(f"  Current step: {state.current_step}")
        print(f"  Progress: {state.progress_percent:.1f}%")
        print(f"  Started: {state.started_at}")
        print(f"  Data size: {len(state.data)} bytes")

    # Example 11: List all workflows
    print("\n11. List All Workflows")
    print("-" * 40)

    all_workflows = engine.list_workflows()

    print(f"Total workflows: {len(all_workflows)}")
    for wf in all_workflows[:5]:
        print(f"  - {wf.id}: {wf.status.value} ({wf.completed_steps}/{wf.total_steps} steps)")

    # Example 12: Cleanup and archival
    print("\n12. Workflow Cleanup")
    print("-" * 40)

    from datetime import datetime, timedelta

    # Archive old workflows (older than 30 days)
    archive_before = datetime.now() - timedelta(days=30)
    archived = engine.archive_workflows(before=archive_before)

    print(f"Archived {len(archived)} old workflows")

    # Delete completed workflows
    deleted = engine.delete_workflows(status=StepStatus.COMPLETED, limit=10)
    print(f"Deleted {len(deleted)} completed workflows")

    # Export workflow for backup
    export_path = f"/tmp/workflow_{workflow_id}.json"
    engine.export_workflow(workflow_id, export_path)
    print(f"Exported workflow to {export_path}")

    print("\n" + "=" * 60)
    print("✅ Workflow persistence examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
