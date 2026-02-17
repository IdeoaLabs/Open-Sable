"""
Open-Sable Unit Tests

Unit tests for individual components.
"""

import pytest
from datetime import datetime, timedelta

from core.session_manager import Session, Message
from core.commands import CommandResult
from core.rate_limiter import TokenBucket, SlidingWindow
from core.context_manager import ContextWindow
from core.config import Config


class TestSession:
    """Test Session class"""
    
    def test_create_session(self):
        """Test session creation"""
        session = Session('telegram', 'user123')
        
        assert session.channel == 'telegram'
        assert session.user_id == 'user123'
        assert len(session.messages) == 0
        assert session.state == 'active'
    
    def test_add_message(self):
        """Test adding messages"""
        session = Session('telegram', 'user123')
        
        session.add_message('user', 'Hello!')
        session.add_message('assistant', 'Hi there!')
        
        assert len(session.messages) == 2
        assert isinstance(session.messages[0], Message)
        assert session.messages[0].content == 'Hello!'
    
    def test_session_serialization(self):
        """Test session to/from dict"""
        session = Session('telegram', 'user123')
        session.add_message('user', 'Test')
        
        data = session.to_dict()
        
        assert 'session_id' in data
        assert 'channel' in data
        assert 'messages' in data
        
        # Reconstruct
        new_session = Session.from_dict(data)
        assert new_session.channel == session.channel
        assert len(new_session.messages) == len(session.messages)


class TestMessage:
    """Test Message class"""
    
    def test_create_message(self):
        """Test message creation"""
        msg = Message('user', 'Hello!')
        
        assert msg.role == 'user'
        assert msg.content == 'Hello!'
        assert isinstance(msg.timestamp, datetime)
    
    def test_message_serialization(self):
        """Test message to/from dict"""
        msg = Message('user', 'Test message')
        
        data = msg.to_dict()
        
        assert data['role'] == 'user'
        assert data['content'] == 'Test message'
        assert 'timestamp' in data


class TestCommandResult:
    """Test CommandResult class"""
    
    def test_create_result(self):
        """Test result creation"""
        result = CommandResult(
            success=True,
            message="Command executed successfully",
            data={'count': 5}
        )
        
        assert result.success is True
        assert 'successfully' in result.message
        assert result.data['count'] == 5


class TestTokenBucket:
    """Test TokenBucket rate limiter"""
    
    @pytest.mark.asyncio
    async def test_consume_tokens(self):
        """Test token consumption"""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Should allow consumption
        assert await bucket.consume(5) is True
        assert await bucket.consume(5) is True
        
        # Should reject (empty)
        assert await bucket.consume(1) is False
    
    @pytest.mark.asyncio
    async def test_refill(self):
        """Test token refill"""
        import asyncio
        
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/sec
        
        # Consume all
        await bucket.consume(10)
        
        # Wait for refill
        await asyncio.sleep(1)
        
        # Should have refilled
        assert await bucket.consume(5) is True


class TestSlidingWindow:
    """Test SlidingWindow rate limiter"""
    
    @pytest.mark.asyncio
    async def test_window_limit(self):
        """Test sliding window"""
        window = SlidingWindow(max_requests=3, window_seconds=60)
        
        # Should allow 3 requests
        assert await window.is_allowed() is True
        assert await window.is_allowed() is True
        assert await window.is_allowed() is True
        
        # Should reject 4th
        assert await window.is_allowed() is False


class TestContextWindow:
    """Test ContextWindow"""
    
    def test_add_message(self):
        """Test adding messages to context"""
        config = Config()
        context = ContextWindow(config)
        
        context.add_message('user', 'Hello')
        context.add_message('assistant', 'Hi')
        
        assert len(context.recent_messages) == 2
    
    def test_build_context(self):
        """Test building context"""
        config = Config()
        context = ContextWindow(config)
        
        context.set_system_prompt("You are a helpful assistant")
        context.add_message('user', 'Hello')
        
        built_context = context.build_context()
        
        assert len(built_context) >= 2
        assert built_context[0]['role'] == 'system'
        assert built_context[1]['role'] == 'user'
    
    def test_token_estimation(self):
        """Test token estimation"""
        config = Config()
        context = ContextWindow(config)
        
        tokens = context.estimate_tokens("Hello world!")
        
        assert tokens > 0
        assert tokens == len("Hello world!") // 4


class TestConfig:
    """Test Config class"""
    
    def test_config_defaults(self):
        """Test default config values"""
        config = Config()
        
        # Should have default values
        assert hasattr(config, '__dict__')
    
    def test_config_from_dict(self):
        """Test creating config from dict"""
        config = Config()
        config.custom_value = "test"
        
        assert config.custom_value == "test"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, '-v'])
