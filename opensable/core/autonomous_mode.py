"""
Autonomous Agent Mode - Continuous operation
Allows the agent to run autonomously, taking actions without user prompts
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .agent import SableAgent
from .goal_system import GoalManager, GoalPriority

logger = logging.getLogger(__name__)


class AutonomousMode:
    """
    Autonomous operation mode
    Agent runs continuously, setting its own goals and executing tasks
    """
    
    def __init__(self, agent: SableAgent, config):
        self.agent = agent
        self.config = config
        self.running = False
        self.task_queue: List[Dict] = []
        self.completed_tasks: List[Dict] = []
        self.goal_manager: Optional[GoalManager] = None
        
        # Autonomous operation settings
        self.check_interval = getattr(config, 'autonomous_check_interval', 60)  # seconds
        self.max_concurrent_tasks = getattr(config, 'autonomous_max_tasks', 3)
        self.enabled_sources = getattr(config, 'autonomous_sources', [
            'calendar', 'email', 'system_monitoring'
        ])
        
    async def start(self):
        """Start autonomous operation"""
        logger.info("🤖 Starting autonomous mode...")
        
        # Initialize goal manager if AGI is available
        try:
            from opensable.core.agi_integration import AGIAgent
            if hasattr(self.agent, 'agi'):
                self.goal_manager = self.agent.agi.goals
                logger.info("AGI goal system available")
        except ImportError:
            logger.warning("AGI not available, using basic autonomous mode")
        
        self.running = True
        
        # Start main loop
        await self._autonomous_loop()
    
    async def stop(self):
        """Stop autonomous operation"""
        logger.info("Stopping autonomous mode...")
        self.running = False
    
    async def _autonomous_loop(self):
        """Main autonomous operation loop"""
        logger.info(f"Autonomous loop started (check interval: {self.check_interval}s)")
        
        while self.running:
            try:
                # 1. Check for new tasks from various sources
                await self._discover_tasks()
                
                # 2. Prioritize tasks
                await self._prioritize_tasks()
                
                # 3. Execute tasks (up to max_concurrent)
                await self._execute_tasks()
                
                # 4. Self-improvement (if AGI available)
                if self.goal_manager:
                    await self._self_improve()
                
                # 5. System maintenance
                await self._perform_maintenance()
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _discover_tasks(self):
        """Discover tasks from various sources"""
        
        # Check calendar for upcoming events
        if 'calendar' in self.enabled_sources:
            await self._check_calendar()
        
        # Check email for action items
        if 'email' in self.enabled_sources:
            await self._check_email()
        
        # Monitor system resources
        if 'system_monitoring' in self.enabled_sources:
            await self._check_system()
        
        # Check for scheduled goals (if AGI available)
        if self.goal_manager:
            await self._check_goals()
    
    async def _check_calendar(self):
        """Check calendar for upcoming events requiring action"""
        try:
            # Get upcoming events in next 24 hours
            result = await self.agent.tools.execute("calendar", {
                "action": "list",
                "timeframe": "24h"
            })
            
            # Parse events and create tasks
            # (This would parse the actual calendar data)
            logger.debug("Checked calendar for tasks")
            
        except Exception as e:
            logger.error(f"Failed to check calendar: {e}")
    
    async def _check_email(self):
        """Check email for action items"""
        try:
            # Check for unread emails with high priority
            result = await self.agent.tools.execute("email", {
                "action": "read",
                "filter": "unread",
                "limit": 10
            })
            
            # Analyze emails for tasks (using LLM)
            # Example: "Meeting at 3pm tomorrow" -> Create reminder task
            logger.debug("Checked email for tasks")
            
        except Exception as e:
            logger.error(f"Failed to check email: {e}")
    
    async def _check_system(self):
        """Monitor system resources and create maintenance tasks"""
        try:
            # Get system info
            result = await self.agent.tools.execute("system_info", {})
            
            # Check if action needed (e.g., disk space low, high CPU)
            # This would parse the system info and create tasks if needed
            logger.debug("Checked system resources")
            
        except Exception as e:
            logger.error(f"Failed to check system: {e}")
    
    async def _check_goals(self):
        """Check for pending goals (AGI mode)"""
        if not self.goal_manager:
            return
        
        try:
            # Get active goals
            active_goals = [g for g in self.goal_manager.goals.values() 
                          if g.status == "active"]
            
            # Add to task queue if not already there
            for goal in active_goals:
                task = {
                    'id': goal.goal_id,
                    'type': 'goal',
                    'description': goal.description,
                    'priority': goal.priority.value,
                    'created_at': datetime.now()
                }
                
                if not any(t['id'] == task['id'] for t in self.task_queue):
                    self.task_queue.append(task)
                    logger.info(f"Added goal to queue: {goal.description}")
                    
        except Exception as e:
            logger.error(f"Failed to check goals: {e}")
    
    async def _prioritize_tasks(self):
        """Prioritize tasks in queue"""
        # Sort by priority (higher first)
        self.task_queue.sort(key=lambda t: t.get('priority', 0), reverse=True)
    
    async def _execute_tasks(self):
        """Execute tasks from queue"""
        # Get tasks to execute (up to max_concurrent)
        tasks_to_execute = self.task_queue[:self.max_concurrent_tasks]
        
        if not tasks_to_execute:
            return
        
        logger.info(f"Executing {len(tasks_to_execute)} task(s)...")
        
        # Execute tasks concurrently
        results = await asyncio.gather(
            *[self._execute_single_task(task) for task in tasks_to_execute],
            return_exceptions=True
        )
        
        # Process results
        for task, result in zip(tasks_to_execute, results):
            if isinstance(result, Exception):
                logger.error(f"Task {task['id']} failed: {result}")
            else:
                logger.info(f"Task {task['id']} completed successfully")
                
                # Move to completed
                self.task_queue.remove(task)
                task['completed_at'] = datetime.now()
                task['result'] = result
                self.completed_tasks.append(task)
    
    async def _execute_single_task(self, task: Dict) -> Any:
        """Execute a single task"""
        task_type = task.get('type')
        
        if task_type == 'goal':
            # Execute goal using AGI
            if self.goal_manager:
                goal_id = task['id']
                return await self.goal_manager.execute_goal(goal_id)
        
        elif task_type == 'command':
            # Execute command
            command = task.get('command')
            return await self.agent.tools.execute("execute_command", {"command": command})
        
        elif task_type == 'reminder':
            # Send reminder
            message = task.get('message')
            # This would send notification to configured channels
            logger.info(f"Reminder: {message}")
            return message
        
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return None
    
    async def _self_improve(self):
        """Run self-improvement (AGI meta-learning)"""
        if not self.goal_manager:
            return
        
        try:
            # Run self-improvement every 24 hours
            if not hasattr(self, '_last_improvement'):
                self._last_improvement = datetime.now() - timedelta(hours=25)
            
            if datetime.now() - self._last_improvement >= timedelta(hours=24):
                logger.info("Running self-improvement...")
                
                # Use AGI meta-learning
                if hasattr(self.agent, 'agi'):
                    improvement = await self.agent.agi.self_improve()
                    logger.info(f"Self-improvement complete: {improvement}")
                    
                self._last_improvement = datetime.now()
                
        except Exception as e:
            logger.error(f"Self-improvement failed: {e}")
    
    async def _perform_maintenance(self):
        """Perform system maintenance tasks"""
        try:
            # Memory consolidation
            if hasattr(self.agent, 'memory'):
                # This would be handled by the advanced memory system
                pass
            
            # Clean up old completed tasks (keep last 100)
            if len(self.completed_tasks) > 100:
                self.completed_tasks = self.completed_tasks[-100:]
            
            # Save state
            await self._save_state()
            
        except Exception as e:
            logger.error(f"Maintenance failed: {e}")
    
    async def _save_state(self):
        """Save autonomous agent state"""
        try:
            state_file = Path(getattr(self.config, 'data_dir', './data')) / 'autonomous_state.json'
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                'task_queue': self.task_queue,
                'completed_tasks': self.completed_tasks[-50:],  # Keep last 50
                'last_update': datetime.now().isoformat()
            }
            
            state_file.write_text(json.dumps(state, indent=2, default=str))
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def _load_state(self):
        """Load autonomous agent state"""
        try:
            state_file = Path(getattr(self.config, 'data_dir', './data')) / 'autonomous_state.json'
            
            if state_file.exists():
                state = json.loads(state_file.read_text())
                self.task_queue = state.get('task_queue', [])
                self.completed_tasks = state.get('completed_tasks', [])
                logger.info("Loaded autonomous state")
                
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current autonomous agent status"""
        return {
            'running': self.running,
            'tasks_queued': len(self.task_queue),
            'tasks_completed': len(self.completed_tasks),
            'enabled_sources': self.enabled_sources,
            'check_interval': self.check_interval,
            'current_tasks': self.task_queue[:5]  # Show next 5 tasks
        }
    
    def add_task(self, task: Dict):
        """Manually add a task to the queue"""
        if 'id' not in task:
            task['id'] = f"manual_{datetime.now().timestamp()}"
        if 'created_at' not in task:
            task['created_at'] = datetime.now()
        
        self.task_queue.append(task)
        logger.info(f"Manually added task: {task.get('description', task['id'])}")
