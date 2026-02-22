"""
AGI Integration - Combines all AGI capabilities into unified system.

Integrates:
- Goal System: Autonomous goal setting and planning
- Advanced Memory: Episodic, semantic, and working memory
- Meta-Learning: Self-improvement and strategy learning
- Tool Synthesis: Dynamic capability creation
- World Model: Environment understanding and prediction
- Metacognition: Self-monitoring and error recovery
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from opensable.core.goal_system import GoalManager, GoalPriority
from opensable.core.advanced_memory import AdvancedMemorySystem, MemoryImportance
from opensable.core.meta_learning import MetaLearningSystem, PerformanceMetric
from opensable.core.tool_synthesis import ToolSynthesizer, ToolSpecification, ToolType
from opensable.core.world_model import WorldModel
from opensable.core.metacognition import MetacognitiveSystem

logger = logging.getLogger(__name__)


class AGIAgent:
    """
    Advanced General Intelligence Agent.

    Autonomous agent with:
    - Goal-directed behavior
    - Long-term memory
    - Self-improvement
    - Tool creation
    - World understanding
    - Self-awareness
    """

    def __init__(
        self,
        llm_function: Optional[Any] = None,
        action_executor: Optional[Any] = None,
        storage_dir: Optional[Path] = None,
    ):
        """
        Initialize AGI agent.

        Args:
            llm_function: LLM for reasoning
            action_executor: Function to execute actions
            storage_dir: Directory for persistent storage
        """
        storage_dir = storage_dir or Path("./data/agi")
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all subsystems
        logger.info("Initializing AGI subsystems...")

        self.goals = GoalManager(
            llm_function=llm_function,
            action_executor=action_executor,
            storage_path=storage_dir / "goals.json",
        )

        self.memory = AdvancedMemorySystem(
            storage_path=storage_dir / "memory.json",
            episodic_size=10000,
            semantic_size=50000,
            working_capacity=7,
        )

        self.meta_learning = MetaLearningSystem(
            storage_dir=storage_dir / "meta_learning", llm_function=llm_function
        )

        self.tool_synthesis = ToolSynthesizer(
            llm_function=llm_function, storage_path=storage_dir / "tools.json"
        )

        self.world_model = WorldModel(storage_path=storage_dir / "world.json")

        self.metacognition = MetacognitiveSystem(storage_path=storage_dir / "metacognition.json")

        # Agent state
        self.active_goal_id: Optional[str] = None
        self.current_trace_id: Optional[str] = None

        logger.info("AGI agent initialized successfully")

    async def set_goal(
        self,
        description: str,
        success_criteria: List[str],
        priority: GoalPriority = GoalPriority.MEDIUM,
        auto_execute: bool = True,
    ) -> Dict[str, Any]:
        """
        Set a goal for the agent.

        Args:
            description: Goal description
            success_criteria: How to measure success
            priority: Goal priority
            auto_execute: Auto-execute if True

        Returns:
            Goal creation and execution result
        """
        # Start metacognitive monitoring
        trace_id = self.metacognition.start_monitoring_task(f"Goal: {description}")
        self.current_trace_id = trace_id

        # Record in working memory
        self.memory.add_to_working_memory(
            f"New goal: {description}", context={"priority": priority.name}
        )

        # Create goal
        goal = await self.goals.create_goal(
            description=description,
            success_criteria=success_criteria,
            priority=priority,
            auto_decompose=True,
        )

        self.active_goal_id = goal.goal_id

        # Record in episodic memory
        self.memory.store_experience(
            event=f"Set goal: {description}",
            context={
                "goal_id": goal.goal_id,
                "priority": priority.name,
                "sub_goals": len(goal.sub_goals),
            },
            importance=MemoryImportance.HIGH,
        )

        # Update world model
        self.world_model.add_observation(
            observation=f"Agent set goal: {description}",
            entities=[
                {
                    "type": "event",
                    "name": f"Goal: {description[:30]}",
                    "properties": {"goal_id": goal.goal_id, "status": goal.status.value},
                }
            ],
        )

        result = {"goal_id": goal.goal_id, "sub_goals": len(goal.sub_goals), "trace_id": trace_id}

        # Execute if requested
        if auto_execute:
            execution_result = await self.execute_goal(goal.goal_id)
            result["execution"] = execution_result

        return result

    async def execute_goal(self, goal_id: str) -> Dict[str, Any]:
        """
        Execute a goal using all AGI capabilities.

        Args:
            goal_id: Goal to execute

        Returns:
            Execution result
        """
        from datetime import datetime

        # Get goal
        goal = self.goals.get_goal(goal_id)
        if not goal:
            return {"success": False, "error": "Goal not found"}

        logger.info(f"Executing goal: {goal.description}")

        # Start timer
        start_time = datetime.utcnow()

        # Get best strategy from meta-learning
        strategy = await self.meta_learning.get_strategy_for_task(goal.description)

        if strategy:
            logger.info(
                f"Using learned strategy: {strategy.name} (success rate: {strategy.success_rate:.2%})"
            )
            self.metacognition.record_thought_step(
                self.current_trace_id,
                "strategy_selection",
                f"Selected strategy: {strategy.name}",
                confidence=strategy.success_rate,
            )

        # Execute goal
        try:
            result = await self.goals.execute_goal(goal_id)
            success = result.get("success", False)
            duration = datetime.utcnow() - start_time

            # Record performance for meta-learning
            self.meta_learning.record_task_performance(
                task_id=goal_id,
                task_type=goal.description,
                success=success,
                duration=duration,
                metrics={PerformanceMetric.SUCCESS_RATE: 1.0 if success else 0.0},
                strategy_id=strategy.strategy_id if strategy else None,
            )

            # Store in episodic memory
            self.memory.store_experience(
                event=f"{'Completed' if success else 'Failed'} goal: {goal.description}",
                context={
                    "goal_id": goal_id,
                    "duration_seconds": duration.total_seconds(),
                    "success": success,
                },
                importance=MemoryImportance.HIGH if success else MemoryImportance.MEDIUM,
            )

            # Complete metacognitive trace
            await self.metacognition.complete_task(
                self.current_trace_id,
                final_answer=result,
                raw_confidence=0.9 if success else 0.3,
                actual_correctness=success,
            )

            # Consolidate memories
            await self.memory.consolidate_memories()

            return result

        except Exception as e:
            logger.error(f"Goal execution error: {e}")

            # Record failure
            duration = datetime.utcnow() - start_time
            self.meta_learning.record_task_performance(
                task_id=goal_id, task_type=goal.description, success=False, duration=duration
            )

            return {"success": False, "error": str(e)}

    async def create_tool_for_task(
        self,
        task_description: str,
        expected_inputs: List[Dict[str, str]],
        expected_outputs: List[Dict[str, str]],
    ) -> Optional[str]:
        """
        Synthesize a new tool for a task.

        Args:
            task_description: What the tool should do
            expected_inputs: Input specifications
            expected_outputs: Output specifications

        Returns:
            Tool ID if successful
        """
        logger.info(f"Synthesizing tool: {task_description}")

        # Create tool specification
        spec = ToolSpecification(
            name=task_description.replace(" ", "_").lower()[:30],
            description=task_description,
            tool_type=ToolType.DATA_PROCESSOR,
            inputs=expected_inputs,
            outputs=expected_outputs,
        )

        # Synthesize tool
        tool = await self.tool_synthesis.synthesize_tool(spec, auto_validate=True)

        if tool and tool.validated:
            # Store knowledge about new tool
            self.memory.store_knowledge(
                fact=f"Created tool: {tool.name} - {task_description}",
                concepts=["tool", "capability", tool.name],
                importance=MemoryImportance.HIGH,
            )

            logger.info(f"Tool synthesized successfully: {tool.tool_id}")
            return tool.tool_id

        return None

    async def predict_and_plan(self, scenario: str, time_horizon: int = 60) -> Dict[str, Any]:
        """
        Predict future state and plan accordingly.

        Args:
            scenario: Scenario to analyze
            time_horizon: Minutes into future to predict

        Returns:
            Prediction and plan
        """
        from datetime import timedelta

        # Predict future world state
        future_state = await self.world_model.predict_future(timedelta(minutes=time_horizon))

        # Store prediction in memory
        self.memory.add_to_working_memory(
            f"Predicted state for: {scenario}", context={"time_horizon": time_horizon}
        )

        return {
            "scenario": scenario,
            "predicted_entities": len(future_state.entities),
            "predicted_relations": len(future_state.relations),
            "prediction_time": future_state.timestamp.isoformat(),
        }

    async def self_improve(self) -> Dict[str, Any]:
        """
        Run self-improvement cycle.

        Returns:
            Improvement results
        """
        logger.info("Running self-improvement cycle...")

        # Meta-learning improvement
        ml_result = await self.meta_learning.self_improve()

        # Memory consolidation and cleanup
        await self.memory.consolidate_memories()
        self.memory.apply_decay()
        self.memory.forget_old_memories()

        # Get metacognition report
        metacog_report = self.metacognition.get_introspection_report()

        # Get learning report
        learning_report = self.meta_learning.get_learning_report()

        result = {
            "meta_learning": ml_result,
            "metacognition": metacog_report,
            "learning_stats": learning_report,
            "memory_stats": self.memory.get_stats(),
        }

        logger.info(f"Self-improvement complete: {result}")
        return result

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        return {
            "subsystems": {
                "goals": self.goals.get_stats(),
                "memory": self.memory.get_stats(),
                "meta_learning": self.meta_learning.get_learning_report(),
                "tools": self.tool_synthesis.get_stats(),
                "world_model": self.world_model.get_stats(),
                "metacognition": self.metacognition.get_introspection_report(),
            },
            "active_goal": self.active_goal_id,
            "current_trace": self.current_trace_id,
        }

    async def start_autonomous_operation(self, improvement_interval_hours: int = 24):
        """
        Start autonomous operation mode.

        Agent will:
        - Continuously improve itself
        - Consolidate memories
        - Monitor and adapt
        """
        logger.info("Starting autonomous operation mode...")

        # Start background tasks
        await self.memory.start_background_consolidation(interval_hours=1)
        await self.meta_learning.start_continuous_improvement(
            interval_hours=improvement_interval_hours
        )

        logger.info("Autonomous operation mode active")

    async def stop_autonomous_operation(self):
        """Stop autonomous operation."""
        await self.memory.stop_background_consolidation()
        await self.meta_learning.stop_continuous_improvement()

        logger.info("Autonomous operation stopped")


# Example usage
async def main():
    """Example AGI agent usage."""

    print("=" * 60)
    print("AGI Agent - Complete System Example")
    print("=" * 60)

    # Initialize AGI agent
    print("\n🤖 Initializing AGI agent...")
    agent = AGIAgent()
    print("  ✅ Agent ready")

    # Set a goal
    print("\n🎯 Setting goal...")
    goal_result = await agent.set_goal(
        description="Learn to analyze data efficiently",
        success_criteria=[
            "Can process data in < 10 seconds",
            "Accuracy > 90%",
            "Can explain findings clearly",
        ],
        priority=GoalPriority.HIGH,
        auto_execute=False,
    )
    print(f"  Goal ID: {goal_result['goal_id']}")
    print(f"  Sub-goals: {goal_result['sub_goals']}")

    # Create a tool
    print("\n🔧 Creating custom tool...")
    tool_id = await agent.create_tool_for_task(
        task_description="Calculate average of numbers",
        expected_inputs=[
            {"name": "numbers", "type": "List[float]", "description": "List of numbers"}
        ],
        expected_outputs=[{"name": "average", "type": "float", "description": "Average value"}],
    )
    if tool_id:
        print(f"  Tool created: {tool_id}")

    # Make predictions
    print("\n🔮 Predicting future...")
    prediction = await agent.predict_and_plan(scenario="Data analysis workflow", time_horizon=30)
    print(f"  Predicted entities: {prediction['predicted_entities']}")

    # Self-improve
    print("\n📈 Running self-improvement...")
    improvement = await agent.self_improve()
    print(f"  Status: {improvement['meta_learning'].get('status', 'N/A')}")
    print(f"  Overall success rate: {improvement['learning_stats']['overall_success_rate']:.2%}")

    # Get status
    print("\n📊 Agent status...")
    status = agent.get_status()
    print(f"  Goals: {status['subsystems']['goals']['total_goals']}")
    print(f"  Memories (episodic): {status['subsystems']['memory']['episodic_count']}")
    print(f"  Memories (semantic): {status['subsystems']['memory']['semantic_count']}")
    print(f"  Tools synthesized: {status['subsystems']['tools']['total_tools']}")
    print(f"  World entities: {status['subsystems']['world_model']['total_entities']}")
    print(f"  Metacog traces: {status['subsystems']['metacognition']['recent_traces']}")

    print("\n✅ AGI system demonstration complete!")
    print("\n💡 The agent now has:")
    print("  • Autonomous goal setting and planning")
    print("  • Multi-layered memory (episodic, semantic, working)")
    print("  • Self-improvement through meta-learning")
    print("  • Dynamic tool creation")
    print("  • World understanding and prediction")
    print("  • Self-monitoring and error recovery")


if __name__ == "__main__":
    asyncio.run(main())
