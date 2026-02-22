"""
Open-Sable Task Queue

Asynchronous task queue for background processing, scheduled tasks,
and distributed work across multiple workers.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import json
from enum import Enum
from dataclasses import dataclass, field
import uuid

from opensable.core.config import Config

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority"""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """Represents a queued task"""

    task_id: str
    func_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: int = 60  # seconds

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "func_name": self.func_name,
            "args": str(self.args),  # Simplified for JSON
            "kwargs": str(self.kwargs),
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }


class TaskQueue:
    """Async task queue with priority and persistence"""

    def __init__(self, config: Config):
        self.config = config
        self.storage_dir = Path.home() / ".opensable" / "queue"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Task storage
        self.tasks: Dict[str, Task] = {}
        self.pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Worker configuration
        self.num_workers = getattr(config, "queue_workers", 4)
        self.workers: List[asyncio.Task] = []
        self.running = False

        # Task handlers registry
        self.handlers: Dict[str, Callable] = {}

        # Stats
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "cancelled_tasks": 0,
        }

    def register_handler(self, name: str, handler: Callable):
        """Register task handler function"""
        self.handlers[name] = handler
        logger.info(f"Registered task handler: {name}")

    async def enqueue(
        self,
        func_name: str,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """Add task to queue"""
        task = Task(
            task_id=str(uuid.uuid4()),
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
        )

        self.tasks[task.task_id] = task
        self.stats["total_tasks"] += 1

        # Add to priority queue (lower number = higher priority)
        # Invert priority so CRITICAL (3) comes first
        await self.pending_queue.put(
            (
                -task.priority.value,  # Negative for reverse sort
                task.created_at.timestamp(),
                task.task_id,
            )
        )

        logger.info(f"Enqueued task: {task.task_id} ({func_name}) with priority {priority.name}")

        # Save to disk
        self._save_task(task)

        return task.task_id

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel pending task"""
        task = self.tasks.get(task_id)

        if not task:
            return False

        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            self.stats["cancelled_tasks"] += 1
            self._save_task(task)
            logger.info(f"Cancelled task: {task_id}")
            return True

        return False

    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        task = self.tasks.get(task_id)

        if not task or task.status != TaskStatus.FAILED:
            return False

        if task.retry_count >= task.max_retries:
            logger.warning(f"Task {task_id} exceeded max retries")
            return False

        task.status = TaskStatus.PENDING
        task.retry_count += 1
        task.error = None

        # Re-enqueue
        await self.pending_queue.put(
            (-task.priority.value, datetime.utcnow().timestamp(), task.task_id)
        )

        logger.info(f"Retrying task: {task_id} (attempt {task.retry_count}/{task.max_retries})")
        return True

    async def _worker(self, worker_id: int):
        """Worker process"""
        logger.info(f"Worker {worker_id} started")

        while self.running:
            try:
                # Get next task (with timeout to allow checking running flag)
                try:
                    priority, timestamp, task_id = await asyncio.wait_for(
                        self.pending_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                task = self.tasks.get(task_id)

                if not task or task.status != TaskStatus.PENDING:
                    continue

                # Execute task
                await self._execute_task(worker_id, task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

        logger.info(f"Worker {worker_id} stopped")

    async def _execute_task(self, worker_id: int, task: Task):
        """Execute a task"""
        logger.info(f"Worker {worker_id} executing task: {task.task_id} ({task.func_name})")

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()

        try:
            # Get handler
            if task.func_name not in self.handlers:
                raise ValueError(f"Unknown task handler: {task.func_name}")

            handler = self.handlers[task.func_name]

            # Execute
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*task.args, **task.kwargs)
            else:
                result = handler(*task.args, **task.kwargs)

            # Mark completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = result
            self.stats["completed_tasks"] += 1

            logger.info(f"Task completed: {task.task_id}")

        except Exception as e:
            logger.error(f"Task failed: {task.task_id} - {e}", exc_info=True)

            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error = str(e)
            self.stats["failed_tasks"] += 1

            # Auto-retry if retries remaining
            if task.retry_count < task.max_retries:
                logger.info(f"Scheduling retry for task {task.task_id} in {task.retry_delay}s")
                asyncio.create_task(self._schedule_retry(task))

        finally:
            self._save_task(task)

    async def _schedule_retry(self, task: Task):
        """Schedule task retry after delay"""
        await asyncio.sleep(task.retry_delay)
        await self.retry_task(task.task_id)

    def _save_task(self, task: Task):
        """Save task to disk"""
        try:
            task_file = self.storage_dir / f"{task.task_id}.json"
            with open(task_file, "w") as f:
                json.dump(task.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving task: {e}")

    def _load_tasks(self):
        """Load tasks from disk"""
        try:
            for task_file in self.storage_dir.glob("*.json"):
                try:
                    with open(task_file) as f:
                        data = json.load(f)

                    # Reconstruct task (simplified - won't have actual args/kwargs)
                    task_id = data["task_id"]
                    if task_id not in self.tasks:
                        # Only load for status tracking
                        logger.debug(f"Loaded task metadata: {task_id}")

                except Exception as e:
                    logger.error(f"Error loading task {task_file}: {e}")

        except Exception as e:
            logger.error(f"Error loading tasks: {e}")

    async def start(self):
        """Start queue workers"""
        logger.info(f"Starting task queue with {self.num_workers} workers")

        self.running = True
        self._load_tasks()

        # Start workers
        self.workers = [asyncio.create_task(self._worker(i)) for i in range(self.num_workers)]

    async def stop(self):
        """Stop queue workers"""
        logger.info("Stopping task queue")

        self.running = False

        # Cancel workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers
        await asyncio.gather(*self.workers, return_exceptions=True)

        logger.info("Task queue stopped")

    def get_stats(self) -> dict:
        """Get queue statistics"""
        pending_count = sum(1 for task in self.tasks.values() if task.status == TaskStatus.PENDING)
        running_count = sum(1 for task in self.tasks.values() if task.status == TaskStatus.RUNNING)

        return {
            **self.stats,
            "pending_tasks": pending_count,
            "running_tasks": running_count,
            "queue_size": self.pending_queue.qsize(),
        }


# Example task handlers
async def example_send_email(to: str, subject: str, body: str):
    """Example: send email task"""
    logger.info(f"Sending email to {to}: {subject}")
    await asyncio.sleep(2)  # Simulate work
    logger.info("Email sent!")
    return {"status": "sent", "to": to}


async def example_process_image(image_path: str):
    """Example: image processing task"""
    logger.info(f"Processing image: {image_path}")
    await asyncio.sleep(5)  # Simulate work
    logger.info("Image processed!")
    return {"status": "processed", "path": image_path}


if __name__ == "__main__":
    from opensable.core.config import load_config

    config = load_config()
    queue = TaskQueue(config)

    # Register handlers
    queue.register_handler("send_email", example_send_email)
    queue.register_handler("process_image", example_process_image)

    async def test():
        await queue.start()

        # Enqueue tasks
        task1 = await queue.enqueue(
            "send_email", "user@example.com", "Test", "Hello!", priority=TaskPriority.HIGH
        )

        task2 = await queue.enqueue(
            "process_image", "/path/to/image.jpg", priority=TaskPriority.NORMAL
        )

        # Wait a bit
        await asyncio.sleep(10)

        # Check stats
        stats = queue.get_stats()
        print(f"Queue stats: {stats}")

        await queue.stop()

    asyncio.run(test())
