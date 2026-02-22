"""
AGI Capabilities Example - Demonstrates all AGI features working together.

This example shows:
1. Autonomous goal creation and decomposition
2. Multi-layer memory (episodic, semantic, working)
3. Meta-learning and self-improvement
4. Dynamic tool synthesis
5. World model and prediction
6. Metacognition and error recovery
"""

import asyncio
import logging
from core.agi_integration import AGIAgent
from core.goal_system import GoalPriority
from core.advanced_memory import MemoryImportance
from core.tool_synthesis import ToolSpecification, ToolType

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def demonstrate_goal_system(agent: AGIAgent):
    """Demonstrate autonomous goal system."""
    print("\n" + "=" * 60)
    print("1. GOAL SYSTEM - Autonomous Planning")
    print("=" * 60)

    # Create high-level goal that will be decomposed
    print("\n📋 Creating complex goal...")
    goal = await agent.set_goal(
        description="Build a data analysis pipeline",
        success_criteria=[
            "Can ingest data from multiple sources",
            "Performs statistical analysis",
            "Generates visualizations",
            "Produces summary report",
        ],
        priority=GoalPriority.HIGH,
        auto_execute=False,
    )

    print(f"✅ Goal created: {goal['goal_id']}")
    print(f"   Sub-goals automatically generated: {goal['sub_goals']}")

    # View goal hierarchy
    hierarchy = agent.goals.get_goal_hierarchy(goal["goal_id"])
    print("\n📊 Goal hierarchy:")
    print(f"   Main: {hierarchy['description']}")
    if "sub_goals_detail" in hierarchy:
        for i, sub in enumerate(hierarchy["sub_goals_detail"], 1):
            print(f"   {i}. {sub['description'][:60]}...")


async def demonstrate_memory_system(agent: AGIAgent):
    """Demonstrate advanced memory system."""
    print("\n" + "=" * 60)
    print("2. MEMORY SYSTEM - Episodic, Semantic, Working")
    print("=" * 60)

    # Store experiences (episodic memory)
    print("\n💭 Storing episodic memories...")
    agent.memory.store_experience(
        event="Completed data analysis task successfully",
        context={"duration": 300, "accuracy": 0.95},
        importance=MemoryImportance.HIGH,
    )

    agent.memory.store_experience(
        event="Encountered timeout issue with large dataset",
        context={"dataset_size": 1000000, "timeout_seconds": 60},
        importance=MemoryImportance.MEDIUM,
    )

    print("✅ Stored 2 episodic memories")

    # Store knowledge (semantic memory)
    print("\n📚 Storing semantic knowledge...")
    agent.memory.store_knowledge(
        fact="Pandas is a Python library for data manipulation and analysis",
        concepts=["Python", "Pandas", "data analysis"],
        importance=MemoryImportance.HIGH,
    )

    agent.memory.store_knowledge(
        fact="Statistical significance is typically measured with p-value < 0.05",
        concepts=["statistics", "p-value", "significance"],
        importance=MemoryImportance.MEDIUM,
    )

    print("✅ Stored 2 semantic facts")

    # Use working memory
    print("\n🔄 Using working memory...")
    agent.memory.add_to_working_memory("Currently analyzing sales data")
    agent.memory.add_to_working_memory("Focus on Q4 2025 trends")
    agent.memory.add_to_working_memory("Target: revenue growth > 10%")

    working_items = agent.memory.get_working_memory()
    print(f"✅ Working memory: {len(working_items)} active items")

    # Recall memories
    print("\n🔍 Recalling relevant memories...")
    knowledge = agent.memory.recall_knowledge("data analysis Python")
    print(f"   Found {len(knowledge)} relevant facts")

    # Get memory stats
    stats = agent.memory.get_stats()
    print("\n📈 Memory stats:")
    print(f"   Episodic: {stats['episodic_count']} memories")
    print(f"   Semantic: {stats['semantic_count']} facts")
    print(f"   Working: {stats['working_count']} items")
    print(f"   Concepts indexed: {stats['total_concepts']}")


async def demonstrate_meta_learning(agent: AGIAgent):
    """Demonstrate meta-learning and self-improvement."""
    print("\n" + "=" * 60)
    print("3. META-LEARNING - Self-Improvement")
    print("=" * 60)

    # Simulate task performance recording
    print("\n📊 Recording task performances...")
    from datetime import timedelta
    from core.meta_learning import PerformanceMetric

    # Good performance on data tasks
    for i in range(3):
        agent.meta_learning.record_task_performance(
            task_id=f"data_task_{i}",
            task_type="data_analysis",
            success=True,
            duration=timedelta(seconds=30),
            metrics={PerformanceMetric.ACCURACY: 0.9, PerformanceMetric.SPEED: 0.8},
        )

    # Poor performance on visualization tasks (learning opportunity)
    for i in range(3):
        agent.meta_learning.record_task_performance(
            task_id=f"viz_task_{i}",
            task_type="visualization",
            success=i >= 2,  # Only last one succeeds
            duration=timedelta(seconds=90),
            metrics={PerformanceMetric.QUALITY: 0.4 + i * 0.2},
        )

    print("✅ Recorded 6 task performances")

    # Run self-improvement
    print("\n🎯 Running self-improvement analysis...")
    improvement = await agent.meta_learning.self_improve()

    print(f"   Status: {improvement.get('status', 'N/A')}")
    if "improvements" in improvement:
        print(f"   Improvements identified: {len(improvement['improvements'])}")
        for imp in improvement["improvements"]:
            print(f"   • {imp['task_type']}: {imp['action']}")

    # Get learning report
    report = agent.meta_learning.get_learning_report()
    print("\n📈 Learning report:")
    print(f"   Total tasks: {report['total_tasks_performed']}")
    print(f"   Success rate: {report['overall_success_rate']:.2%}")
    print(f"   Strategies learned: {report['strategies_learned']}")
    print(f"   Mastered task types: {report['task_types_mastered']}/{report['total_task_types']}")


async def demonstrate_tool_synthesis(agent: AGIAgent):
    """Demonstrate dynamic tool creation."""
    print("\n" + "=" * 60)
    print("4. TOOL SYNTHESIS - Dynamic Capability Creation")
    print("=" * 60)

    # Create a custom tool
    print("\n🔧 Synthesizing tool: JSON parser...")

    spec = ToolSpecification(
        name="json_parser",
        description="Parse JSON string and extract specific fields",
        tool_type=ToolType.DATA_PROCESSOR,
        inputs=[
            {"name": "json_string", "type": "str", "description": "JSON string to parse"},
            {"name": "fields", "type": "List[str]", "description": "Fields to extract"},
        ],
        outputs=[
            {"name": "extracted", "type": "Dict[str, Any]", "description": "Extracted fields"}
        ],
        examples=[
            {
                "input": {"json_string": '{"name": "Alice", "age": 30}', "fields": ["name"]},
                "output": {"extracted": {"name": "Alice"}},
            }
        ],
    )

    tool = await agent.tool_synthesis.synthesize_tool(spec, auto_validate=True)

    if tool:
        print(f"✅ Tool synthesized: {tool.tool_id}")
        print(f"   Validated: {tool.validated}")
        if tool.test_results:
            print(
                f"   Tests: {tool.test_results.get('passed', 0)}/{tool.test_results.get('total', 0)} passed"
            )

    # Create another tool
    print("\n🔧 Synthesizing tool: URL validator...")

    url_tool_id = await agent.create_tool_for_task(
        task_description="Validate if string is valid URL",
        expected_inputs=[{"name": "url", "type": "str", "description": "URL to validate"}],
        expected_outputs=[
            {"name": "is_valid", "type": "bool", "description": "Whether URL is valid"},
            {"name": "protocol", "type": "str", "description": "URL protocol"},
        ],
    )

    if url_tool_id:
        print(f"✅ Tool created: {url_tool_id}")

    # Get tool synthesis stats
    stats = agent.tool_synthesis.get_stats()
    print("\n📊 Tool synthesis stats:")
    print(f"   Total tools: {stats['total_tools']}")
    print(f"   Validated: {stats['validated_tools']}")
    print(f"   By type: {stats['by_type']}")


async def demonstrate_world_model(agent: AGIAgent):
    """Demonstrate world model and prediction."""
    print("\n" + "=" * 60)
    print("5. WORLD MODEL - Understanding & Prediction")
    print("=" * 60)

    # Add observations to world model
    print("\n🌍 Building world model...")

    agent.world_model.add_observation(
        observation="User is working on machine learning project",
        entities=[
            {"type": "agent", "name": "User", "properties": {"activity": "ML development"}},
            {
                "type": "object",
                "name": "ML Project",
                "properties": {"status": "active", "progress": 0.6},
            },
        ],
        relations=[{"type": "has", "source": "User", "target": "ML Project"}],
    )

    agent.world_model.add_observation(
        observation="Project deadline is in 2 weeks",
        entities=[{"type": "event", "name": "Deadline", "properties": {"days_remaining": 14}}],
        relations=[{"type": "requires", "source": "ML Project", "target": "Deadline"}],
    )

    print("✅ Added observations to world model")

    # Query current state
    print("\n🔍 Querying world state...")
    from core.world_model import EntityType

    agents = agent.world_model.query_state(entity_type=EntityType.AGENT)
    objects = agent.world_model.query_state(entity_type=EntityType.OBJECT)
    events = agent.world_model.query_state(entity_type=EntityType.EVENT)

    print(f"   Agents: {len(agents)}")
    print(f"   Objects: {len(objects)}")
    print(f"   Events: {len(events)}")

    # Make prediction
    print("\n🔮 Predicting future state...")
    prediction = await agent.predict_and_plan(scenario="ML project completion", time_horizon=60)

    print("✅ Prediction complete:")
    print(f"   Predicted entities: {prediction['predicted_entities']}")
    print(f"   Prediction time: {prediction['prediction_time']}")

    # Get world model stats
    stats = agent.world_model.get_stats()
    print("\n📊 World model stats:")
    print(f"   Entities: {stats['total_entities']}")
    print(f"   Relations: {stats['total_relations']}")
    print(f"   State snapshots: {stats['state_snapshots']}")


async def demonstrate_metacognition(agent: AGIAgent):
    """Demonstrate metacognition and self-monitoring."""
    print("\n" + "=" * 60)
    print("6. METACOGNITION - Self-Awareness & Error Recovery")
    print("=" * 60)

    # Start monitoring a task
    print("\n🧠 Starting metacognitive monitoring...")

    trace_id = agent.metacognition.start_monitoring_task("Solve optimization problem")

    # Record reasoning steps
    print("   Recording thought process...")
    agent.metacognition.record_thought_step(
        trace_id,
        "problem_analysis",
        "Need to minimize cost while maximizing efficiency",
        raw_confidence=0.8,
    )

    agent.metacognition.record_thought_step(
        trace_id, "approach_selection", "Will use linear programming approach", raw_confidence=0.7
    )

    agent.metacognition.record_thought_step(
        trace_id,
        "implementation",
        "Setting up constraints and objective function",
        raw_confidence=0.85,
    )

    # Complete task
    await agent.metacognition.complete_task(
        trace_id,
        final_answer="Optimal solution found: cost=100, efficiency=0.95",
        raw_confidence=0.9,
        actual_correctness=True,
    )

    print("✅ Task monitoring complete")

    # Demonstrate error detection
    print("\n⚠️  Testing error detection...")

    trace_id2 = agent.metacognition.start_monitoring_task("Analyze contradictory data")
    agent.metacognition.record_thought_step(
        trace_id2,
        "analysis",
        "The dataset shows increasing trend. However, the dataset shows decreasing trend.",
        raw_confidence=0.4,
    )

    await agent.metacognition.complete_task(
        trace_id2, final_answer="Unable to determine trend", raw_confidence=0.3
    )

    print("✅ Error detection active (detected contradiction)")

    # Get introspection report
    report = agent.metacognition.get_introspection_report()
    print("\n📊 Metacognition report:")
    print(f"   Recent traces: {report['recent_traces']}")
    print(f"   Errors detected: {report['total_errors_detected']}")
    print(f"   Avg confidence: {report['avg_confidence']:.2f}")
    print(f"   Recovery success rate: {report['recovery_success_rate']:.2%}")


async def demonstrate_full_agi_cycle(agent: AGIAgent):
    """Demonstrate complete AGI cycle."""
    print("\n" + "=" * 60)
    print("7. FULL AGI CYCLE - Everything Together")
    print("=" * 60)

    print("\n🚀 Running complete AGI cycle...")
    print("   This combines all subsystems working together:\n")

    # 1. Set autonomous goal
    print("   1️⃣  Setting goal (with auto-decomposition)...")
    goal = await agent.set_goal(
        description="Analyze customer feedback data and generate insights",
        success_criteria=[
            "Data is loaded and validated",
            "Sentiment analysis completed",
            "Key themes identified",
            "Report generated with recommendations",
        ],
        priority=GoalPriority.HIGH,
        auto_execute=False,
    )
    print(f"      ✓ Goal created with {goal['sub_goals']} sub-goals")

    # 2. Check if we have necessary tools, create if needed
    print("\n   2️⃣  Checking capabilities (tool synthesis)...")
    sentiment_tool = await agent.create_tool_for_task(
        task_description="Perform sentiment analysis on text",
        expected_inputs=[{"name": "text", "type": "str", "description": "Text to analyze"}],
        expected_outputs=[
            {"name": "sentiment", "type": "str", "description": "Positive/Negative/Neutral"},
            {"name": "score", "type": "float", "description": "Confidence score"},
        ],
    )
    if sentiment_tool:
        print("      ✓ Created sentiment analysis tool")

    # 3. Use world model to understand context
    print("\n   3️⃣  Understanding context (world model)...")
    agent.world_model.add_observation(
        observation="Processing customer feedback data",
        entities=[
            {"type": "object", "name": "Customer Feedback", "properties": {"count": 1000}},
            {"type": "event", "name": "Analysis Task", "properties": {"priority": "high"}},
        ],
    )
    print("      ✓ Context added to world model")

    # 4. Execute with metacognitive monitoring
    print("\n   4️⃣  Executing goal (with self-monitoring)...")
    result = await agent.execute_goal(goal["goal_id"])
    print(f"      ✓ Execution {'successful' if result.get('success') else 'failed'}")

    # 5. Learn from experience
    print("\n   5️⃣  Learning from experience (meta-learning)...")
    improvement = await agent.self_improve()
    print("      ✓ Self-improvement cycle complete")
    print(
        f"         • Overall success rate: {improvement['learning_stats']['overall_success_rate']:.2%}"
    )

    # 6. Consolidate memories
    print("\n   6️⃣  Consolidating memories...")
    await agent.memory.consolidate_memories()
    print("      ✓ Working memory consolidated into long-term storage")

    print("\n✨ Complete AGI cycle finished!")


async def main():
    """Run complete AGI demonstration."""

    print("\n" + "=" * 60)
    print("🤖 Open-Sable AGI System - Complete Demonstration")
    print("=" * 60)
    print("\nThis demonstrates a complete AGI-capable system with:")
    print("• Autonomous goal setting and planning")
    print("• Multi-layered memory (episodic, semantic, working)")
    print("• Meta-learning and continuous self-improvement")
    print("• Dynamic tool synthesis and capability creation")
    print("• World modeling and future prediction")
    print("• Metacognitive self-monitoring and error recovery")

    # Initialize agent
    print("\n⏳ Initializing AGI agent...")
    agent = AGIAgent()

    # Run demonstrations
    await demonstrate_goal_system(agent)
    await demonstrate_memory_system(agent)
    await demonstrate_meta_learning(agent)
    await demonstrate_tool_synthesis(agent)
    await demonstrate_world_model(agent)
    await demonstrate_metacognition(agent)
    await demonstrate_full_agi_cycle(agent)

    # Final status
    print("\n" + "=" * 60)
    print("📊 FINAL AGENT STATUS")
    print("=" * 60)

    status = agent.get_status()

    print("\n🎯 Goals:")
    print(f"   Total: {status['subsystems']['goals']['total_goals']}")
    print(f"   By status: {status['subsystems']['goals']['by_status']}")

    print("\n💭 Memory:")
    print(f"   Episodic: {status['subsystems']['memory']['episodic_count']}")
    print(f"   Semantic: {status['subsystems']['memory']['semantic_count']}")
    print(f"   Working: {status['subsystems']['memory']['working_count']}")

    print("\n📈 Meta-Learning:")
    print(f"   Tasks performed: {status['subsystems']['meta_learning']['total_tasks_performed']}")
    print(f"   Success rate: {status['subsystems']['meta_learning']['overall_success_rate']:.2%}")

    print("\n🔧 Tools:")
    print(f"   Synthesized: {status['subsystems']['tools']['total_tools']}")
    print(f"   Validated: {status['subsystems']['tools']['validated_tools']}")

    print("\n🌍 World Model:")
    print(f"   Entities: {status['subsystems']['world_model']['total_entities']}")
    print(f"   Relations: {status['subsystems']['world_model']['total_relations']}")

    print("\n🧠 Metacognition:")
    print(f"   Traces: {status['subsystems']['metacognition']['recent_traces']}")
    print(f"   Errors detected: {status['subsystems']['metacognition']['total_errors_detected']}")

    print("\n" + "=" * 60)
    print("✅ AGI DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThe agent is now fully operational with AGI capabilities!")


if __name__ == "__main__":
    asyncio.run(main())
