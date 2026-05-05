"""
Open-Sable Multi-Agent Orchestration

Coordinates multiple AI agents working together on complex tasks.
Supports agent delegation, parallel execution, result aggregation,
shared memory (blackboard), crew-style high-level API, and
distributed coordination across network nodes via the Gateway.
"""

import asyncio
import logging
import time as _time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
import json
import uuid as _uuid

from opensable.core.agent import SableAgent
from opensable.core.config import Config
from opensable.core.session_manager import SessionManager

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in multi-agent system"""

    COORDINATOR = "coordinator"  # Orchestrates other agents
    RESEARCHER = "researcher"  # Searches and gathers information
    ANALYST = "analyst"  # Analyzes data and provides insights
    WRITER = "writer"  # Generates content
    CODER = "coder"  # Writes code
    REVIEWER = "reviewer"  # Reviews and validates output
    EXECUTOR = "executor"  # Executes tasks
    PLANNER = "planner"  # Breaks down complex goals into tasks
    CUSTOM = "custom"  # User-defined role


@dataclass
class AgentTask:
    """Represents a task for an agent"""

    task_id: str
    role: AgentRole
    description: str
    input_data: Any
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    max_iterations: int = 6  # guardrail: max LLM turns per task
    output_schema: Optional[Dict] = None  # expected JSON output structure
    allow_delegation: bool = False  # can this agent delegate sub-tasks?


# ─── Shared Blackboard ──────────────────────────────────────────────────────


class SharedBlackboard:
    """
    Shared memory space for agents in a workflow or crew.

    Agents can read/write arbitrary keys, enabling data flow beyond
    simple dependency chains.  Thread-safe via asyncio.Lock.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._history: List[Dict[str, Any]] = []  # audit trail

    async def write(self, key: str, value: Any, *, author: str = "system"):
        async with self._lock:
            self._data[key] = value
            self._history.append(
                {"action": "write", "key": key, "author": author, "ts": _time.time()}
            )

    async def read(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._data.get(key, default)

    async def read_all(self) -> Dict[str, Any]:
        async with self._lock:
            return dict(self._data)

    async def append_list(self, key: str, item: Any, *, author: str = "system"):
        """Append to a list value (create if missing)."""
        async with self._lock:
            lst = self._data.setdefault(key, [])
            lst.append(item)
            self._history.append(
                {"action": "append", "key": key, "author": author, "ts": _time.time()}
            )

    def snapshot(self) -> Dict[str, Any]:
        """Non-async snapshot (for logging)."""
        return dict(self._data)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


class AgentPool:
    """Pool of specialized agents"""

    # Role-specific system prompts
    ROLE_PROMPTS = {
        AgentRole.COORDINATOR: (
            "You are the Coordinator agent. Your job is to synthesize "
            "results from multiple specialist agents into a coherent, "
            "comprehensive final response. Be concise and clear."
        ),
        AgentRole.RESEARCHER: (
            "You are the Research agent. You excel at finding information, "
            "searching the web, and gathering relevant data. Always cite "
            "your sources and be thorough."
        ),
        AgentRole.ANALYST: (
            "You are the Analyst agent. You break down complex problems, "
            "identify patterns, evaluate data quality, and provide "
            "structured insights with pros/cons analysis."
        ),
        AgentRole.WRITER: (
            "You are the Writer agent. You create clear, well-structured "
            "content. Focus on readability, proper formatting, and "
            "engaging language appropriate to the context."
        ),
        AgentRole.CODER: (
            "You are the Coder agent. You write clean, efficient code "
            "with proper error handling. Include comments for complex "
            "logic and follow best practices for the language."
        ),
        AgentRole.REVIEWER: (
            "You are the Reviewer agent. You critically evaluate work "
            "for accuracy, completeness, and quality. Point out issues "
            "and suggest specific improvements."
        ),
        AgentRole.PLANNER: (
            "You are the Planner agent. You decompose complex goals into "
            "ordered, actionable sub-tasks, identify dependencies between "
            "them, and assign each to the most appropriate specialist role."
        ),
    }

    def __init__(self, config: Config):
        self.config = config
        self.agents: Dict[AgentRole, SableAgent] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the agent pool,  agents are created lazily on first use"""
        self._initialized = True
        logger.info(f"Initialized {len(self.ROLE_PROMPTS)} specialized agents")
        return self

    async def _get_or_create_agent(self, role: AgentRole) -> SableAgent:
        """Lazily create and initialize an agent for the given role"""
        if role not in self.agents:
            agent = SableAgent(self.config)
            await agent.initialize()
            self.agents[role] = agent
        return self.agents[role]

    def get_agent(self, role: AgentRole) -> Optional[SableAgent]:
        """Get agent by role (may be None if not yet created)"""
        return self.agents.get(role)

    def get_role_prompt(self, role: AgentRole) -> str:
        """Get the system prompt for a role"""
        return self.ROLE_PROMPTS.get(role, "")


class MultiAgentOrchestrator:
    """Orchestrates multiple agents working together"""

    def __init__(self, config: Config):
        self.config = config
        self.agent_pool = AgentPool(config)
        self.session_manager = SessionManager()
        self.blackboard: Optional[SharedBlackboard] = None

        # Task tracking
        self.tasks: Dict[str, AgentTask] = {}

        # Execution stats
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_agents_used": 0,
        }

    async def execute_workflow(
        self, tasks: List[AgentTask], session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow of agent tasks

        Args:
            tasks: List of tasks to execute
            session_id: Optional session ID for context

        Returns:
            Dictionary with results and metadata
        """
        logger.info(f"Starting workflow with {len(tasks)} tasks")

        # Store tasks
        for task in tasks:
            self.tasks[task.task_id] = task
            self.stats["total_tasks"] += 1

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(tasks)

        # Execute tasks in dependency order
        results = {}

        for level in dependency_graph:
            # Execute all tasks in this level in parallel
            level_tasks = [self.tasks[task_id] for task_id in level]

            logger.info(f"Executing {len(level_tasks)} tasks in parallel")

            level_results = await asyncio.gather(
                *[self._execute_task(task, results, session_id) for task in level_tasks]
            )

            # Store results
            for task, result in zip(level_tasks, level_results):
                results[task.task_id] = result

        # Aggregate results
        successful = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        logger.info(f"Workflow completed: {successful} successful, {failed} failed")

        return {
            "success": failed == 0,
            "total_tasks": len(tasks),
            "successful_tasks": successful,
            "failed_tasks": failed,
            "results": results,
            "tasks": [self._task_to_dict(t) for t in tasks],
        }

    def _build_dependency_graph(self, tasks: List[AgentTask]) -> List[List[str]]:
        """Build execution order based on dependencies"""
        # Simple topological sort
        task_map = {t.task_id: t for t in tasks}
        levels = []
        processed = set()

        while len(processed) < len(tasks):
            # Find tasks with all dependencies satisfied
            current_level = []

            for task in tasks:
                if task.task_id in processed:
                    continue

                # Check if all dependencies are processed
                deps_satisfied = all(dep in processed for dep in task.dependencies)

                if deps_satisfied:
                    current_level.append(task.task_id)

            if not current_level:
                # Circular dependency or error
                logger.error("Circular dependency detected!")
                break

            levels.append(current_level)
            processed.update(current_level)

        return levels

    async def _execute_task(
        self, task: AgentTask, previous_results: Dict[str, Any], session_id: Optional[str]
    ) -> Any:
        """Execute a single agent task"""
        task.status = "running"
        task.started_at = datetime.now(timezone.utc)

        try:
            # Get or create agent for this role (lazy init)
            agent = await self.agent_pool._get_or_create_agent(task.role)

            if not agent:
                raise ValueError(f"No agent found for role: {task.role}")

            # Build context from dependencies + blackboard
            context = self._build_context(task, previous_results)

            # Build prompt with role-specific instructions (+ backstory/goal from CrewMember)
            role_prompt = self.agent_pool.get_role_prompt(task.role)
            input_data = task.input_data if isinstance(task.input_data, dict) else {"data": task.input_data}

            backstory = input_data.get("backstory", "")
            goal = input_data.get("goal", "")

            prompt_parts = [role_prompt]
            if backstory:
                prompt_parts.append(f"\nBackstory: {backstory}")
            if goal:
                prompt_parts.append(f"Goal: {goal}")
            prompt_parts.append(f"\nTask: {task.description}")
            if input_data:
                # Strip internal meta keys for the prompt
                display_data = {k: v for k, v in input_data.items() if k not in ("backstory", "goal", "context_keys")}
                if display_data:
                    prompt_parts.append(f"\nInput Data:\n{json.dumps(display_data, indent=2)}")
            if context:
                prompt_parts.append(f"\n{context}")
            prompt_parts.append("\nComplete this task and provide your output.")

            prompt = "\n".join(prompt_parts)

            # Get session
            session = None
            if session_id:
                session = self.session_manager.get_session(session_id)

            # Execute task
            logger.info(f"Agent {task.role.value} executing: {task.task_id}")

            result = await agent.run(prompt, session)

            # Mark completed
            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc)
            task.result = result

            # Write result to shared blackboard
            if self.blackboard:
                await self.blackboard.write(
                    f"result:{task.task_id}", result, author=task.role.value
                )

            self.stats["completed_tasks"] += 1
            self.stats["total_agents_used"] += 1

            logger.info(f"Task completed: {task.task_id}")

            return result

        except Exception as e:
            logger.error(f"Task failed: {task.task_id} - {e}", exc_info=True)

            task.status = "failed"
            task.completed_at = datetime.now(timezone.utc)
            task.error = str(e)

            self.stats["failed_tasks"] += 1

            return None

    def _build_context(self, task: AgentTask, previous_results: Dict[str, Any]) -> str:
        """Build context from dependency results and shared blackboard."""
        parts: List[str] = []

        if task.dependencies:
            parts.append("Previous Task Results:\n")
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                result = previous_results.get(dep_id)
                if dep_task and result:
                    parts.append(f"Task: {dep_task.description}\nResult: {result}\n")

        # Inject shared blackboard keys requested by the task
        context_keys = (task.input_data or {}).get("context_keys", []) if isinstance(task.input_data, dict) else []
        if context_keys and self.blackboard:
            bb = self.blackboard.snapshot()
            parts.append("Shared Memory (Blackboard):\n")
            for key in context_keys:
                val = bb.get(key)
                if val is not None:
                    parts.append(f"  {key}: {json.dumps(val, default=str)[:2000]}\n")

        return "\n".join(parts)

    def _task_to_dict(self, task: AgentTask) -> dict:
        """Convert task to dictionary"""
        return {
            "task_id": task.task_id,
            "role": task.role.value,
            "description": task.description,
            "status": task.status,
            "dependencies": task.dependencies,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error": task.error,
        }

    async def delegate_task(
        self, description: str, role: AgentRole, context: Optional[str] = None
    ) -> str:
        """Delegate a single task to an agent"""
        import uuid

        task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=role,
            description=description,
            input_data={"context": context} if context else {},
        )

        result = await self._execute_task(task, {}, None)

        return result

    async def collaborative_task(self, description: str, roles: List[AgentRole]) -> Dict[str, Any]:
        """
        Have multiple agents collaborate on a task
        Each agent provides their perspective/contribution
        """
        logger.info(f"Collaborative task with {len(roles)} agents")

        results = {}

        # Execute in parallel
        tasks = await asyncio.gather(*[self.delegate_task(description, role) for role in roles])

        for role, result in zip(roles, tasks):
            results[role.value] = result

        # Synthesize results
        synthesis_prompt = f"""Task: {description}

Multiple agents have provided their perspectives:

{json.dumps(results, indent=2)}

Synthesize these perspectives into a comprehensive, coherent response."""

        # Use coordinator agent to synthesize
        coordinator = self.agent_pool.get_agent(AgentRole.COORDINATOR)
        final_result = await coordinator.run(synthesis_prompt, None)

        return {"individual_results": results, "synthesized_result": final_result}

    def get_stats(self) -> dict:
        """Get orchestrator statistics"""
        return self.stats.copy()

    async def route_complex_task(
        self, task_description: str, user_id: str = "default"
    ) -> Optional[str]:
        """
        Auto-route a complex task to the appropriate specialist(s).
        Returns the final synthesized result, or None if not applicable.

        Routing heuristics:
        - Code-related → CODER + REVIEWER
        - Research/analysis → RESEARCHER + ANALYST
        - Writing → WRITER + REVIEWER
        - Multi-step → full workflow
        """
        desc_lower = task_description.lower()

        # Detect task type
        code_keywords = [
            "write code",
            "implement",
            "programa",
            "function",
            "class",
            "script",
            "debug",
            "fix the code",
            "refactor",
            "code review",
            "algoritmo",
        ]
        research_keywords = [
            "research",
            "investigate",
            "analiza",
            "compare",
            "investiga",
            "pros and cons",
            "report on",
            "informe sobre",
        ]
        write_keywords = [
            "write an article",
            "write a blog",
            "escribe",
            "draft",
            "redacta",
            "essay",
            "summary of",
            "resumen",
        ]

        roles = []
        if any(kw in desc_lower for kw in code_keywords):
            roles = [AgentRole.CODER, AgentRole.REVIEWER]
        elif any(kw in desc_lower for kw in research_keywords):
            roles = [AgentRole.RESEARCHER, AgentRole.ANALYST]
        elif any(kw in desc_lower for kw in write_keywords):
            roles = [AgentRole.WRITER, AgentRole.REVIEWER]

        if not roles:
            return None  # Not a multi-agent task

        logger.info(f"🤝 Multi-agent routing: {[r.value for r in roles]}")
        result = await self.collaborative_task(task_description, roles)
        return result.get("synthesized_result")


# Example workflow builders
class WorkflowBuilder:
    """Helper to build common workflows"""

    @staticmethod
    def research_and_write(topic: str) -> List[AgentTask]:
        """Build workflow: research → analyze → write → review"""
        import uuid

        research_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.RESEARCHER,
            description=f"Research information about: {topic}",
            input_data={"topic": topic},
        )

        analysis_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.ANALYST,
            description=f"Analyze research findings about: {topic}",
            input_data={"topic": topic},
            dependencies=[research_task.task_id],
        )

        writing_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.WRITER,
            description=f"Write comprehensive article about: {topic}",
            input_data={"topic": topic},
            dependencies=[analysis_task.task_id],
        )

        review_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.REVIEWER,
            description=f"Review and improve article about: {topic}",
            input_data={"topic": topic},
            dependencies=[writing_task.task_id],
        )

        return [research_task, analysis_task, writing_task, review_task]

    @staticmethod
    def code_development(requirement: str) -> List[AgentTask]:
        """Build workflow: analyze → code → review"""
        import uuid

        analysis_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.ANALYST,
            description=f"Analyze requirements and design solution for: {requirement}",
            input_data={"requirement": requirement},
        )

        coding_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.CODER,
            description=f"Implement solution for: {requirement}",
            input_data={"requirement": requirement},
            dependencies=[analysis_task.task_id],
        )

        review_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.REVIEWER,
            description=f"Review code for: {requirement}",
            input_data={"requirement": requirement},
            dependencies=[coding_task.task_id],
        )

        return [analysis_task, coding_task, review_task]


# ─── Crew API (CrewAI-style high-level builder) ─────────────────────────────


@dataclass
class CrewMember:
    """Declarative definition of a crew member."""

    role: AgentRole
    goal: str  # what this member is trying to achieve
    backstory: str = ""  # extra context injected into the system prompt
    max_iterations: int = 6
    allow_delegation: bool = False
    output_schema: Optional[Dict] = None  # JSON schema for structured output
    custom_prompt: Optional[str] = None  # override the default role prompt


@dataclass
class CrewTask:
    """Declarative task for a crew."""

    description: str
    assigned_to: AgentRole  # which crew member handles this
    expected_output: str = ""  # human description of what to produce
    depends_on: List[str] = field(default_factory=list)  # indexes (0-based) or labels
    context_keys: List[str] = field(default_factory=list)  # blackboard keys to inject
    label: str = ""  # optional human-readable label for dependency referencing


class Crew:
    """
    High-level orchestration builder (inspired by CrewAI).

    Example::

        crew = Crew(
            config=config,
            members=[
                CrewMember(role=AgentRole.RESEARCHER, goal="Find relevant papers"),
                CrewMember(role=AgentRole.WRITER, goal="Draft an article"),
                CrewMember(role=AgentRole.REVIEWER, goal="Ensure quality"),
            ],
            tasks=[
                CrewTask(description="Research quantum computing", assigned_to=AgentRole.RESEARCHER, label="research"),
                CrewTask(description="Write article from research", assigned_to=AgentRole.WRITER, depends_on=["research"], label="draft"),
                CrewTask(description="Review the draft", assigned_to=AgentRole.REVIEWER, depends_on=["draft"]),
            ],
        )
        result = await crew.kickoff()
    """

    def __init__(
        self,
        config: Config,
        members: List[CrewMember],
        tasks: List[CrewTask],
        *,
        verbose: bool = True,
        shared_memory: Optional[SharedBlackboard] = None,
        on_task_complete: Optional[Callable[[AgentTask, Any], Awaitable[None]]] = None,
    ):
        self.config = config
        self.members = {m.role: m for m in members}
        self.crew_tasks = tasks
        self.verbose = verbose
        self.blackboard = shared_memory or SharedBlackboard()
        self._on_task_complete = on_task_complete
        self._orchestrator = MultiAgentOrchestrator(config)
        self._orchestrator.blackboard = self.blackboard

    async def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start the crew workflow.

        Args:
            inputs: Optional initial data to seed the blackboard.

        Returns:
            Dict with ``results``, ``blackboard``, and per-task metadata.
        """
        # Seed blackboard
        if inputs:
            for k, v in inputs.items():
                await self.blackboard.write(k, v, author="crew.kickoff")

        # Convert CrewTasks → AgentTasks with label-based dependency resolution
        label_to_id: Dict[str, str] = {}
        agent_tasks: List[AgentTask] = []

        for idx, ct in enumerate(self.crew_tasks):
            task_id = str(_uuid.uuid4())
            label = ct.label or f"task_{idx}"
            label_to_id[label] = task_id
            label_to_id[str(idx)] = task_id

        for idx, ct in enumerate(self.crew_tasks):
            label = ct.label or f"task_{idx}"
            task_id = label_to_id[label]

            member = self.members.get(ct.assigned_to)
            deps = [label_to_id.get(d, d) for d in ct.depends_on]

            at = AgentTask(
                task_id=task_id,
                role=ct.assigned_to,
                description=ct.description,
                input_data={
                    "expected_output": ct.expected_output,
                    "context_keys": ct.context_keys,
                    "goal": member.goal if member else "",
                    "backstory": member.backstory if member else "",
                },
                dependencies=deps,
                max_iterations=member.max_iterations if member else 10,
                output_schema=ct.expected_output if isinstance(ct.expected_output, dict) else None,
                allow_delegation=member.allow_delegation if member else False,
            )
            agent_tasks.append(at)

        if self.verbose:
            roles = [ct.assigned_to.value for ct in self.crew_tasks]
            logger.info(f"🚀 Crew kickoff,  {len(self.crew_tasks)} tasks, roles: {roles}")

        # Execute the workflow
        result = await self._orchestrator.execute_workflow(agent_tasks)

        # Write final results to blackboard
        for task_id, res in result.get("results", {}).items():
            await self.blackboard.write(f"result:{task_id}", res, author="crew")

        result["blackboard"] = self.blackboard.snapshot()
        return result


# ─── Distributed Multi-Agent Coordination ────────────────────────────────────


@dataclass
class RemoteNode:
    """Represents a remote agent node reachable over the Gateway."""

    node_id: str
    capabilities: List[str]
    host: str = "local"  # "local" | hostname/IP
    last_seen: float = 0.0
    latency_ms: float = 0.0
    active_tasks: int = 0


class DistributedCoordinator:
    """
    Extends the local MultiAgentOrchestrator with distributed coordination.

    Nodes register via the Gateway's node.register protocol (Unix socket
    or TCP). The coordinator can delegate tasks to remote nodes, monitor
    health, and aggregate results.

    Architecture:
      Coordinator (this)
          │
          ├── Local AgentPool   (in-process agents for fast tasks)
          │
          └── Gateway node bus  (remote nodes on same machine or LAN)
                 ├── Node A  [coder, reviewer]
                 ├── Node B  [researcher]
                 └── Node C  [analyst, writer]
    """

    def __init__(self, config: Config, orchestrator: MultiAgentOrchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self._nodes: Dict[str, RemoteNode] = {}
        self._pending: Dict[str, asyncio.Future] = {}  # request_id → Future
        self._gateway = None  # set when gateway is available
        self._heartbeat_task: Optional[asyncio.Task] = None

    # ── Node registry ─────────────────────────────────────────────────────────

    def register_node(self, node_id: str, capabilities: List[str], host: str = "local"):
        """Register a node (called when Gateway receives node.register)."""
        import time

        node = RemoteNode(
            node_id=node_id,
            capabilities=capabilities,
            host=host,
            last_seen=time.time(),
        )
        self._nodes[node_id] = node
        logger.info(f"🌐 Distributed: node '{node_id}' registered, caps={capabilities}")
        return node

    def unregister_node(self, node_id: str):
        self._nodes.pop(node_id, None)
        logger.info(f"🌐 Distributed: node '{node_id}' removed")

    def list_nodes(self) -> List[Dict[str, Any]]:
        import time

        now = time.time()
        return [
            {
                "node_id": n.node_id,
                "capabilities": n.capabilities,
                "host": n.host,
                "alive": (now - n.last_seen) < 120,
                "active_tasks": n.active_tasks,
                "latency_ms": round(n.latency_ms, 1),
            }
            for n in self._nodes.values()
        ]

    # ── Task routing ──────────────────────────────────────────────────────────

    def _find_node_for_capability(self, capability: str) -> Optional[RemoteNode]:
        """Find the best node for a given capability (least loaded + alive)."""
        import time

        now = time.time()
        candidates = [
            n
            for n in self._nodes.values()
            if capability in n.capabilities and (now - n.last_seen) < 120
        ]
        if not candidates:
            return None
        # Pick least loaded
        return min(candidates, key=lambda n: n.active_tasks)

    async def delegate_to_node(
        self,
        node_id: str,
        capability: str,
        args: Dict[str, Any],
        timeout: float = 60.0,
    ) -> Any:
        """
        Send a task to a remote node via the Gateway and wait for the result.
        """
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Node '{node_id}' not found")

        request_id = str(_uuid.uuid4())
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future
        node.active_tasks += 1

        # Send invocation via gateway
        if self._gateway:
            # Broadcast to the node through the gateway's dispatch
            for client in self._gateway._clients:
                if getattr(client, "node_id", None) == node_id:
                    await client.send(
                        {
                            "type": "node.invoke",
                            "capability": capability,
                            "args": args,
                            "request_id": request_id,
                        }
                    )
                    break

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            import time

            node.last_seen = time.time()
            return result
        except asyncio.TimeoutError:
            logger.error(f"Node '{node_id}' timed out on '{capability}'")
            raise
        finally:
            node.active_tasks -= 1
            self._pending.pop(request_id, None)

    def receive_result(self, request_id: str, output: Any):
        """Called by Gateway when a node.result arrives."""
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(output)

    # ── Distributed workflow execution ────────────────────────────────────────

    async def execute_distributed(
        self,
        tasks: List[AgentTask],
        session_id: Optional[str] = None,
        prefer_remote: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a workflow, distributing tasks to remote nodes when possible.

        Falls back to local execution when no remote node matches.
        """
        logger.info(
            f"🌐 Distributed workflow: {len(tasks)} tasks, " f"{len(self._nodes)} nodes available"
        )

        # Map roles to capabilities
        role_to_cap = {
            AgentRole.RESEARCHER: "research",
            AgentRole.ANALYST: "analyze",
            AgentRole.CODER: "code",
            AgentRole.WRITER: "write",
            AgentRole.REVIEWER: "review",
            AgentRole.EXECUTOR: "execute",
            AgentRole.COORDINATOR: "coordinate",
        }

        results = {}
        dependency_graph = self.orchestrator._build_dependency_graph(tasks)

        for level in dependency_graph:
            level_tasks = [
                self.orchestrator.tasks.get(tid) or next(t for t in tasks if t.task_id == tid)
                for tid in level
            ]

            coros = []
            for task in level_tasks:
                cap = role_to_cap.get(task.role, task.role.value)
                node = self._find_node_for_capability(cap) if prefer_remote else None

                if node:
                    # Remote execution
                    logger.info(f"🌐 Delegating '{task.task_id}' to node '{node.node_id}'")
                    coros.append(
                        self.delegate_to_node(
                            node.node_id,
                            cap,
                            {
                                "description": task.description,
                                "input": task.input_data,
                                "context": self.orchestrator._build_context(task, results),
                            },
                        )
                    )
                else:
                    # Local execution
                    logger.info(f"💻 Executing '{task.task_id}' locally")
                    coros.append(self.orchestrator._execute_task(task, results, session_id))

            level_results = await asyncio.gather(*coros, return_exceptions=True)
            for task, result in zip(level_tasks, level_results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.error = str(result)
                    results[task.task_id] = None
                else:
                    results[task.task_id] = result

        successful = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        remote_count = sum(1 for n in self._nodes.values() if n.active_tasks >= 0)

        return {
            "success": failed == 0,
            "total_tasks": len(tasks),
            "successful_tasks": successful,
            "failed_tasks": failed,
            "nodes_used": remote_count,
            "results": results,
        }

    # ── Health monitoring ─────────────────────────────────────────────────────

    async def start_health_monitor(self, interval: float = 30.0):
        """Periodically ping nodes and remove dead ones."""
        self._heartbeat_task = asyncio.create_task(self._health_loop(interval))

    async def _health_loop(self, interval: float):
        import time

        while True:
            try:
                await asyncio.sleep(interval)
                now = time.time()
                dead = [nid for nid, n in self._nodes.items() if (now - n.last_seen) > interval * 4]
                for nid in dead:
                    logger.warning(f"🌐 Node '{nid}' presumed dead, removing")
                    self.unregister_node(nid)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    def get_cluster_status(self) -> Dict[str, Any]:
        """Return cluster-wide status."""
        return {
            "total_nodes": len(self._nodes),
            "alive_nodes": sum(1 for n in self._nodes.values()),
            "total_capabilities": list({c for n in self._nodes.values() for c in n.capabilities}),
            "nodes": self.list_nodes(),
            "orchestrator_stats": self.orchestrator.get_stats() if self.orchestrator else {},
        }


if __name__ == "__main__":
    from opensable.core.config import load_config

    config = load_config()
    orchestrator = MultiAgentOrchestrator(config)

    async def test():
        # Build workflow
        workflow = WorkflowBuilder.research_and_write("quantum computing")

        # Execute
        result = await orchestrator.execute_workflow(workflow)

        print(f"Workflow result: {json.dumps(result, indent=2)}")
        print(f"Stats: {orchestrator.get_stats()}")

    asyncio.run(test())
