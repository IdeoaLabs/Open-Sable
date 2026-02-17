"""
Open-Sable Multi-Agent Orchestration

Coordinates multiple AI agents working together on complex tasks.
Supports agent delegation, parallel execution, and result aggregation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import json

from opensable.core.agent import SableAgent
from opensable.core.config import Config
from opensable.core.session_manager import SessionManager

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in multi-agent system"""
    COORDINATOR = "coordinator"  # Orchestrates other agents
    RESEARCHER = "researcher"    # Searches and gathers information
    ANALYST = "analyst"          # Analyzes data and provides insights
    WRITER = "writer"            # Generates content
    CODER = "coder"              # Writes code
    REVIEWER = "reviewer"        # Reviews and validates output
    EXECUTOR = "executor"        # Executes tasks


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


class AgentPool:
    """Pool of specialized agents"""
    
    def __init__(self, config: Config):
        self.config = config
        self.agents: Dict[AgentRole, SableAgent] = {}
        self._initialize_agents()
    
    async def initialize(self):
        """Initialize agent pool"""
        return self
    
    def _initialize_agents(self):
        """Initialize specialized agents"""
        # Each role gets a dedicated agent with specialized system prompt
        
        self.agents[AgentRole.COORDINATOR] = SableAgent(self.config)
        self.agents[AgentRole.RESEARCHER] = SableAgent(self.config)
        self.agents[AgentRole.ANALYST]     = SableAgent(self.config)
        self.agents[AgentRole.WRITER]      = SableAgent(self.config)
        self.agents[AgentRole.CODER]       = SableAgent(self.config)
        self.agents[AgentRole.REVIEWER]    = SableAgent(self.config)
        
        logger.info(f"Initialized {len(self.agents)} specialized agents")
    
    def get_agent(self, role: AgentRole) -> SableAgent:
        """Get agent by role"""
        return self.agents.get(role)


class MultiAgentOrchestrator:
    """Orchestrates multiple agents working together"""
    
    def __init__(self, config: Config):
        self.config = config
        self.agent_pool = AgentPool(config)
        self.session_manager = SessionManager()
        
        # Task tracking
        self.tasks: Dict[str, AgentTask] = {}
        
        # Execution stats
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_agents_used': 0
        }
    
    async def execute_workflow(self, 
                               tasks: List[AgentTask],
                               session_id: Optional[str] = None) -> Dict[str, Any]:
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
            self.stats['total_tasks'] += 1
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(tasks)
        
        # Execute tasks in dependency order
        results = {}
        
        for level in dependency_graph:
            # Execute all tasks in this level in parallel
            level_tasks = [self.tasks[task_id] for task_id in level]
            
            logger.info(f"Executing {len(level_tasks)} tasks in parallel")
            
            level_results = await asyncio.gather(*[
                self._execute_task(task, results, session_id)
                for task in level_tasks
            ])
            
            # Store results
            for task, result in zip(level_tasks, level_results):
                results[task.task_id] = result
        
        # Aggregate results
        successful = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        
        logger.info(f"Workflow completed: {successful} successful, {failed} failed")
        
        return {
            'success': failed == 0,
            'total_tasks': len(tasks),
            'successful_tasks': successful,
            'failed_tasks': failed,
            'results': results,
            'tasks': [self._task_to_dict(t) for t in tasks]
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
                deps_satisfied = all(
                    dep in processed
                    for dep in task.dependencies
                )
                
                if deps_satisfied:
                    current_level.append(task.task_id)
            
            if not current_level:
                # Circular dependency or error
                logger.error("Circular dependency detected!")
                break
            
            levels.append(current_level)
            processed.update(current_level)
        
        return levels
    
    async def _execute_task(self,
                           task: AgentTask,
                           previous_results: Dict[str, Any],
                           session_id: Optional[str]) -> Any:
        """Execute a single agent task"""
        task.status = "running"
        task.started_at = datetime.utcnow()
        
        try:
            # Get agent for this role
            agent = self.agent_pool.get_agent(task.role)
            
            if not agent:
                raise ValueError(f"No agent found for role: {task.role}")
            
            # Build context from dependencies
            context = self._build_context(task, previous_results)
            
            # Build prompt
            prompt = f"""Task: {task.description}

Input Data:
{json.dumps(task.input_data, indent=2)}

{context}

Please complete this task and provide your output."""
            
            # Get session
            session = None
            if session_id:
                session = self.session_manager.get_session(session_id)
            
            # Execute task
            logger.info(f"Agent {task.role.value} executing: {task.task_id}")
            
            result = await agent.run(prompt, session)
            
            # Mark completed
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            task.result = result
            
            self.stats['completed_tasks'] += 1
            self.stats['total_agents_used'] += 1
            
            logger.info(f"Task completed: {task.task_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Task failed: {task.task_id} - {e}", exc_info=True)
            
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            task.error = str(e)
            
            self.stats['failed_tasks'] += 1
            
            return None
    
    def _build_context(self, task: AgentTask, previous_results: Dict[str, Any]) -> str:
        """Build context from dependency results"""
        if not task.dependencies:
            return ""
        
        context = "Previous Task Results:\n\n"
        
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            result = previous_results.get(dep_id)
            
            if dep_task and result:
                context += f"Task: {dep_task.description}\n"
                context += f"Result: {result}\n\n"
        
        return context
    
    def _task_to_dict(self, task: AgentTask) -> dict:
        """Convert task to dictionary"""
        return {
            'task_id': task.task_id,
            'role': task.role.value,
            'description': task.description,
            'status': task.status,
            'dependencies': task.dependencies,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'error': task.error
        }
    
    async def delegate_task(self,
                           description: str,
                           role: AgentRole,
                           context: Optional[str] = None) -> str:
        """Delegate a single task to an agent"""
        import uuid
        
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=role,
            description=description,
            input_data={'context': context} if context else {}
        )
        
        result = await self._execute_task(task, {}, None)
        
        return result
    
    async def collaborative_task(self,
                                description: str,
                                roles: List[AgentRole]) -> Dict[str, Any]:
        """
        Have multiple agents collaborate on a task
        Each agent provides their perspective/contribution
        """
        logger.info(f"Collaborative task with {len(roles)} agents")
        
        results = {}
        
        # Execute in parallel
        tasks = await asyncio.gather(*[
            self.delegate_task(description, role)
            for role in roles
        ])
        
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
        
        return {
            'individual_results': results,
            'synthesized_result': final_result
        }
    
    def get_stats(self) -> dict:
        """Get orchestrator statistics"""
        return self.stats.copy()


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
            input_data={'topic': topic}
        )
        
        analysis_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.ANALYST,
            description=f"Analyze research findings about: {topic}",
            input_data={'topic': topic},
            dependencies=[research_task.task_id]
        )
        
        writing_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.WRITER,
            description=f"Write comprehensive article about: {topic}",
            input_data={'topic': topic},
            dependencies=[analysis_task.task_id]
        )
        
        review_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.REVIEWER,
            description=f"Review and improve article about: {topic}",
            input_data={'topic': topic},
            dependencies=[writing_task.task_id]
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
            input_data={'requirement': requirement}
        )
        
        coding_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.CODER,
            description=f"Implement solution for: {requirement}",
            input_data={'requirement': requirement},
            dependencies=[analysis_task.task_id]
        )
        
        review_task = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.REVIEWER,
            description=f"Review code for: {requirement}",
            input_data={'requirement': requirement},
            dependencies=[coding_task.task_id]
        )
        
        return [analysis_task, coding_task, review_task]


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
