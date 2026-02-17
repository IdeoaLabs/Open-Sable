"""
Open-Sable End-to-End Tests

Tests for complete workflows and integrations.
"""

import asyncio
import pytest
import time
from pathlib import Path

from core.config import Config, load_config
from core.session_manager import SessionManager
from core.commands import CommandHandler
from core.gateway import GatewayServer
from core.analytics import Analytics
from core.rate_limiter import RateLimiter
from core.cache import MultiLayerCache
from core.task_queue import TaskQueue
from core.webhooks import WebhookManager
from core.multi_agent import MultiAgentOrchestrator, WorkflowBuilder


class TestCompleteWorkflow:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_message_flow_with_caching(self):
        """Test complete message flow with caching"""
        # Initialize components
        config = Config()
        session_manager = SessionManager()
        cache = MultiLayerCache(config)
        
        # Create session
        session = session_manager.get_or_create_session('test', 'user123')
        
        # Add message
        session.add_message('user', 'What is Python?')
        
        # Check cache
        cache_key = f"response:{session.session_id}:What is Python?"
        cached_response = await cache.get(cache_key)
        
        if cached_response:
            response = cached_response
        else:
            # Simulate agent response
            response = "Python is a high-level programming language."
            await cache.set(cache_key, response, ttl=3600)
        
        session.add_message('assistant', response)
        
        assert len(session.messages) == 2
        assert session.messages[1].content == response
    
    @pytest.mark.asyncio
    async def test_command_with_analytics(self):
        """Test command execution with analytics tracking"""
        # Initialize
        config = Config()
        session_manager = SessionManager()
        command_handler = CommandHandler(session_manager)
        analytics = Analytics(config)
        
        # Create session
        session = session_manager.get_or_create_session('test', 'user123')
        
        # Execute command
        start_time = time.time()
        result = command_handler.handle_command('/help', session)
        response_time = (time.time() - start_time) * 1000
        
        # Track in analytics
        analytics.track_command('/help', 'user123', result.success)
        analytics.track_response_time('test', response_time)
        
        # Verify
        assert result.success is True
        assert analytics.metrics['commands_executed'] == 1
        assert len(analytics.response_times['test']) == 1
    
    @pytest.mark.asyncio
    async def test_rate_limited_message_flow(self):
        """Test message flow with rate limiting"""
        # Initialize
        config = Config()
        config.user_message_max = 3
        config.user_message_window = 60
        
        session_manager = SessionManager()
        rate_limiter = RateLimiter(config)
        
        user_id = "test_user"
        
        # Send 3 messages (should work)
        for i in range(3):
            allowed = await rate_limiter.check_message_limit(user_id)
            assert allowed is True
            
            session = session_manager.get_or_create_session('test', user_id)
            session.add_message('user', f'Message {i}')
        
        # 4th message should be rate limited
        from core.rate_limiter import RateLimitExceeded
        with pytest.raises(RateLimitExceeded):
            await rate_limiter.check_message_limit(user_id)
    
    @pytest.mark.asyncio
    async def test_multi_agent_research_workflow(self):
        """Test multi-agent research and write workflow"""
        config = Config()
        orchestrator = MultiAgentOrchestrator(config)
        
        # Build research workflow
        builder = WorkflowBuilder(orchestrator)
        
        # Execute research workflow
        result = await builder.research_and_write(
            topic="Python programming",
            audience="beginners"
        )
        
        assert result['success'] is True
        assert 'final_result' in result
        assert len(result['task_results']) >= 3  # research, analyze, write
    
    @pytest.mark.asyncio
    async def test_background_task_processing(self):
        """Test background task queue processing"""
        config = Config()
        config.queue_workers = 2
        task_queue = TaskQueue(config)
        
        # Register handler
        results = []
        
        async def process_data(data):
            results.append(data)
            return f"Processed: {data}"
        
        task_queue.register_handler('process', process_data)
        
        # Start queue
        await task_queue.start()
        
        # Enqueue tasks
        task_ids = []
        for i in range(5):
            task_id = await task_queue.enqueue('process', f'item_{i}')
            task_ids.append(task_id)
        
        # Wait for completion
        await asyncio.sleep(3)
        
        # Check results
        assert len(results) == 5
        assert task_queue.stats['completed_tasks'] == 5
        
        # Stop queue
        await task_queue.stop()
    
    @pytest.mark.asyncio
    async def test_webhook_notification_on_message(self):
        """Test webhook firing on message event"""
        config = Config()
        webhook_manager = WebhookManager(config)
        session_manager = SessionManager()
        
        await webhook_manager.start()
        
        # Register webhook
        webhook_id = webhook_manager.register_webhook(
            url="http://example.com/webhook",
            events=["message.received"],
            secret="test_secret"
        )
        
        # Create session and message
        session = session_manager.get_or_create_session('test', 'user123')
        session.add_message('user', 'Test message')
        
        # Emit webhook event
        await webhook_manager.emit('message.received', {
            'session_id': session.session_id,
            'user_id': 'user123',
            'message': 'Test message'
        })
        
        # Wait for delivery
        await asyncio.sleep(2)
        
        # Check webhook history
        webhook = webhook_manager.get_webhook(webhook_id)
        assert webhook is not None
        
        await webhook_manager.stop()


class TestGatewayIntegration:
    """Test gateway server integration"""
    
    @pytest.mark.asyncio
    async def test_gateway_session_routing(self):
        """Test gateway routing messages to sessions"""
        # This would require running the actual gateway
        # For now, test the core routing logic
        
        config = Config()
        session_manager = SessionManager()
        
        # Simulate multiple sessions
        session1 = session_manager.get_or_create_session('telegram', 'user1')
        session2 = session_manager.get_or_create_session('discord', 'user2')
        
        # Add messages to each
        session1.add_message('user', 'Hello from Telegram')
        session2.add_message('user', 'Hello from Discord')
        
        # Verify routing
        assert session1.channel == 'telegram'
        assert session2.channel == 'discord'
        assert len(session1.messages) == 1
        assert len(session2.messages) == 1


class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_task_retry_on_failure(self):
        """Test task queue retry logic"""
        config = Config()
        task_queue = TaskQueue(config)
        
        # Handler that fails first 2 times
        attempt_count = [0]
        
        async def flaky_task():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise Exception("Temporary failure")
            return "Success"
        
        task_queue.register_handler('flaky', flaky_task)
        
        await task_queue.start()
        
        # Enqueue task
        task_id = await task_queue.enqueue('flaky')
        
        # Wait for retries
        await asyncio.sleep(5)
        
        # Should succeed after retries
        task = await task_queue.get_task(task_id)
        assert task.status.value == "completed"
        assert task.result == "Success"
        
        await task_queue.stop()
    
    def test_invalid_command_handling(self):
        """Test handling invalid commands"""
        config = Config()
        session_manager = SessionManager()
        command_handler = CommandHandler(session_manager)
        
        session = session_manager.get_or_create_session('test', 'user123')
        
        # Invalid command
        result = command_handler.handle_command('/invalid', session)
        
        assert result is not None
        assert result.success is False
        assert 'unknown' in result.message.lower()


class TestPerformance:
    """Test performance and scalability"""
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache hit performance"""
        config = Config()
        cache = MultiLayerCache(config)
        
        # Warm up cache
        for i in range(100):
            await cache.set(f'key_{i}', f'value_{i}', ttl=300)
        
        # Measure hit performance
        start = time.time()
        for i in range(100):
            value = await cache.get(f'key_{i}')
            assert value == f'value_{i}'
        duration = time.time() - start
        
        # Should be fast (< 1 second for 100 hits)
        assert duration < 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self):
        """Test handling multiple concurrent sessions"""
        session_manager = SessionManager()
        
        # Create 100 sessions
        sessions = []
        for i in range(100):
            session = session_manager.get_or_create_session('test', f'user_{i}')
            session.add_message('user', f'Message from user {i}')
            sessions.append(session)
        
        # Verify all created
        assert len(sessions) == 100
        
        # Verify each has correct data
        for i, session in enumerate(sessions):
            assert session.user_id == f'user_{i}'
            assert len(session.messages) == 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
