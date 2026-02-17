"""
Open-Sable Integration Tests

Tests for core components, interfaces, and workflows.
"""

import asyncio
import pytest
from pathlib import Path
import tempfile
import shutil

from core.config import Config, load_config
from core.session_manager import SessionManager
from core.commands import CommandHandler
from core.rate_limiter import RateLimiter, RateLimitExceeded
from core.analytics import Analytics
from core.cache import MultiLayerCache
from core.task_queue import TaskQueue, TaskPriority
from core.webhooks import WebhookManager
from core.plugins import PluginManager
from core.multi_agent import MultiAgentOrchestrator, AgentTask, AgentRole, WorkflowBuilder


class TestSessionManager:
    """Test session management"""
    
    @pytest.fixture
    def session_manager(self):
        """Create session manager"""
        return SessionManager()
    
    def test_create_session(self, session_manager):
        """Test session creation"""
        session = session_manager.get_or_create_session('telegram', 'user123')
        
        assert session is not None
        assert session.channel == 'telegram'
        assert session.user_id == 'user123'
        assert len(session.messages) == 0
    
    def test_add_message(self, session_manager):
        """Test adding messages to session"""
        session = session_manager.get_or_create_session('telegram', 'user123')
        
        session.add_message('user', 'Hello!')
        session.add_message('assistant', 'Hi there!')
        
        assert len(session.messages) == 2
        assert session.messages[0].role == 'user'
        assert session.messages[1].role == 'assistant'
    
    def test_reset_session(self, session_manager):
        """Test session reset"""
        session = session_manager.get_or_create_session('telegram', 'user123')
        session.add_message('user', 'Hello!')
        
        session_manager.reset_session(session.session_id)
        
        assert len(session.messages) == 0


class TestCommands:
    """Test command handling"""
    
    @pytest.fixture
    def command_handler(self):
        """Create command handler"""
        session_manager = SessionManager()
        return CommandHandler(session_manager)
    
    @pytest.fixture
    def session(self):
        """Create test session"""
        manager = SessionManager()
        return manager.get_or_create_session('test', 'user123')
    
    def test_help_command(self, command_handler, session):
        """Test /help command"""
        result = command_handler.handle_command('/help', session)
        
        assert result is not None
        assert result.success is True
        assert 'commands' in result.message.lower()
    
    def test_status_command(self, command_handler, session):
        """Test /status command"""
        session.add_message('user', 'Test message')
        
        result = command_handler.handle_command('/status', session)
        
        assert result is not None
        assert result.success is True
        assert 'messages' in result.message.lower()


class TestRateLimiter:
    """Test rate limiting"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter"""
        config = Config()
        config.user_message_max = 5
        config.user_message_window = 60
        return RateLimiter(config)
    
    @pytest.mark.asyncio
    async def test_message_limit(self, rate_limiter):
        """Test message rate limiting"""
        user_id = "test_user"
        
        # Should allow first 5 messages
        for i in range(5):
            result = await rate_limiter.check_message_limit(user_id)
            assert result is True
        
        # 6th message should fail
        with pytest.raises(RateLimitExceeded):
            await rate_limiter.check_message_limit(user_id)
    
    @pytest.mark.asyncio
    async def test_vip_bypass(self, rate_limiter):
        """Test VIP user bypass"""
        user_id = "vip_user"
        rate_limiter.add_vip_user(user_id)
        
        # Should allow unlimited messages
        for i in range(20):
            result = await rate_limiter.check_message_limit(user_id)
            assert result is True


class TestAnalytics:
    """Test analytics tracking"""
    
    @pytest.fixture
    def analytics(self):
        """Create analytics"""
        config = Config()
        return Analytics(config)
    
    def test_track_message(self, analytics):
        """Test message tracking"""
        analytics.track_message_received('telegram', 'user123', 'Hello!')
        
        assert analytics.metrics['messages_received'] == 1
        assert analytics.channel_stats['telegram'] == 1
    
    def test_track_command(self, analytics):
        """Test command tracking"""
        analytics.track_command('/help', 'user123', True)
        
        assert analytics.metrics['commands_executed'] == 1
    
    def test_get_summary(self, analytics):
        """Test analytics summary"""
        analytics.track_message_received('telegram', 'user123', 'Test')
        analytics.track_message_sent('telegram', 'user123', 'Response', tokens=10)
        
        summary = analytics.get_summary()
        
        assert 'metrics' in summary
        assert summary['metrics']['messages_received'] == 1
        assert summary['metrics']['messages_sent'] == 1


class TestCache:
    """Test caching system"""
    
    @pytest.fixture
    def cache(self):
        """Create cache"""
        config = Config()
        return MultiLayerCache(config)
    
    @pytest.mark.asyncio
    async def test_set_get(self, cache):
        """Test cache set/get"""
        await cache.set('key1', 'value1', ttl=60)
        
        value = await cache.get('key1')
        assert value == 'value1'
    
    @pytest.mark.asyncio
    async def test_expiration(self, cache):
        """Test cache expiration"""
        await cache.set('key1', 'value1', ttl=1)
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        value = await cache.get('key1')
        assert value is None
    
    @pytest.mark.asyncio
    async def test_get_or_compute(self, cache):
        """Test get_or_compute"""
        compute_count = [0]
        
        def compute():
            compute_count[0] += 1
            return "computed_value"
        
        # First call should compute
        value1 = await cache.get_or_compute('key1', compute, ttl=60)
        assert value1 == "computed_value"
        assert compute_count[0] == 1
        
        # Second call should use cache
        value2 = await cache.get_or_compute('key1', compute, ttl=60)
        assert value2 == "computed_value"
        assert compute_count[0] == 1  # Not incremented


class TestTaskQueue:
    """Test task queue"""
    
    @pytest.fixture
    def task_queue(self):
        """Create task queue"""
        config = Config()
        config.queue_workers = 2
        return TaskQueue(config)
    
    @pytest.mark.asyncio
    async def test_enqueue_execute(self, task_queue):
        """Test task enqueue and execution"""
        # Register handler
        async def test_task(x, y):
            return x + y
        
        task_queue.register_handler('add', test_task)
        
        # Start queue
        await task_queue.start()
        
        # Enqueue task
        task_id = await task_queue.enqueue('add', 5, 10)
        
        # Wait for completion
        await asyncio.sleep(2)
        
        # Check result
        task = await task_queue.get_task(task_id)
        assert task is not None
        assert task.status.value == "completed"
        assert task.result == 15
        
        # Stop queue
        await task_queue.stop()


class TestWebhooks:
    """Test webhook system"""
    
    @pytest.fixture
    def webhook_manager(self):
        """Create webhook manager"""
        config = Config()
        return WebhookManager(config)
    
    @pytest.mark.asyncio
    async def test_register_webhook(self, webhook_manager):
        """Test webhook registration"""
        await webhook_manager.start()
        
        webhook_id = webhook_manager.register_webhook(
            url="https://example.com/webhook",
            events=["message.received"]
        )
        
        assert webhook_id is not None
        webhook = webhook_manager.get_webhook(webhook_id)
        assert webhook is not None
        assert webhook.url == "https://example.com/webhook"
        
        await webhook_manager.stop()


class TestMultiAgent:
    """Test multi-agent orchestration"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator"""
        config = Config()
        return MultiAgentOrchestrator(config)
    
    @pytest.mark.asyncio
    async def test_simple_delegation(self, orchestrator):
        """Test delegating task to single agent"""
        result = await orchestrator.delegate_task(
            "What is 2 + 2?",
            AgentRole.ANALYST
        )
        
        assert result is not None
        # Result should contain answer
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, orchestrator):
        """Test workflow execution"""
        import uuid
        
        # Create simple workflow
        task1 = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.RESEARCHER,
            description="Research Python programming",
            input_data={'topic': 'Python'}
        )
        
        task2 = AgentTask(
            task_id=str(uuid.uuid4()),
            role=AgentRole.WRITER,
            description="Write summary of Python",
            input_data={'topic': 'Python'},
            dependencies=[task1.task_id]
        )
        
        result = await orchestrator.execute_workflow([task1, task2])
        
        assert result['success'] is True
        assert result['total_tasks'] == 2


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
