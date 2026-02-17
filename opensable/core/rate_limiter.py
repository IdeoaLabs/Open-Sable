"""
Open-Sable Rate Limiter

Implements rate limiting for API calls, message processing, and resource usage.
Prevents abuse and ensures fair usage across users and channels.
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import deque
import time

from opensable.core.config import Config

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, limit_type: str, retry_after: float):
        self.limit_type = limit_type
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {limit_type}. Retry after {retry_after:.1f}s")


class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket
        
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        async with self.lock:
            # Refill tokens based on time passed
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now
            
            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None):
        """
        Wait until tokens are available
        
        Args:
            tokens: Number of tokens needed
            timeout: Maximum time to wait (None for infinite)
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        start_time = time.time()
        
        while True:
            if await self.consume(tokens):
                return
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                raise asyncio.TimeoutError("Timeout waiting for rate limit tokens")
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens will be available"""
        if self.tokens >= tokens:
            return 0.0
        
        needed = tokens - self.tokens
        return needed / self.refill_rate


class SlidingWindow:
    """Sliding window rate limiter"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        """
        Args:
            max_requests: Maximum requests in window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = asyncio.Lock()
    
    async def is_allowed(self) -> bool:
        """Check if request is allowed"""
        async with self.lock:
            now = time.time()
            cutoff = now - self.window_seconds
            
            # Remove old requests
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            # Check limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def time_until_allowed(self) -> float:
        """Calculate time until next request is allowed"""
        if len(self.requests) < self.max_requests:
            return 0.0
        
        # Time until oldest request falls out of window
        oldest = self.requests[0]
        now = time.time()
        cutoff = now - self.window_seconds
        
        if oldest < cutoff:
            return 0.0
        
        return oldest - cutoff


class RateLimiter:
    """Comprehensive rate limiting system"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Per-user rate limits
        self.user_message_limits: Dict[str, SlidingWindow] = {}
        self.user_command_limits: Dict[str, SlidingWindow] = {}
        
        # Global rate limits
        self.global_message_limit = TokenBucket(
            capacity=getattr(config, 'global_message_capacity', 1000),
            refill_rate=getattr(config, 'global_message_refill_rate', 10.0)
        )
        
        self.global_api_limit = TokenBucket(
            capacity=getattr(config, 'global_api_capacity', 100),
            refill_rate=getattr(config, 'global_api_refill_rate', 1.0)
        )
        
        # Configuration
        self.user_message_max = getattr(config, 'user_message_max', 20)
        self.user_message_window = getattr(config, 'user_message_window', 60)
        self.user_command_max = getattr(config, 'user_command_max', 10)
        self.user_command_window = getattr(config, 'user_command_window', 60)
        
        # VIP users (no rate limiting)
        self.vip_users = set(getattr(config, 'vip_users', []))
    
    def _get_user_message_limiter(self, user_id: str) -> SlidingWindow:
        """Get or create message rate limiter for user"""
        if user_id not in self.user_message_limits:
            self.user_message_limits[user_id] = SlidingWindow(
                self.user_message_max,
                self.user_message_window
            )
        return self.user_message_limits[user_id]
    
    def _get_user_command_limiter(self, user_id: str) -> SlidingWindow:
        """Get or create command rate limiter for user"""
        if user_id not in self.user_command_limits:
            self.user_command_limits[user_id] = SlidingWindow(
                self.user_command_max,
                self.user_command_window
            )
        return self.user_command_limits[user_id]
    
    async def check_message_limit(self, user_id: str) -> bool:
        """
        Check if user can send a message
        
        Returns:
            True if allowed, False if rate limited
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        # VIP users bypass limits
        if user_id in self.vip_users:
            return True
        
        # Check global limit
        if not await self.global_message_limit.consume(1):
            retry_after = self.global_message_limit.time_until_available(1)
            raise RateLimitExceeded('global_message', retry_after)
        
        # Check user limit
        user_limiter = self._get_user_message_limiter(user_id)
        if not await user_limiter.is_allowed():
            retry_after = user_limiter.time_until_allowed()
            raise RateLimitExceeded('user_message', retry_after)
        
        return True
    
    async def check_command_limit(self, user_id: str) -> bool:
        """
        Check if user can execute a command
        
        Returns:
            True if allowed, False if rate limited
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        # VIP users bypass limits
        if user_id in self.vip_users:
            return True
        
        # Check user limit
        user_limiter = self._get_user_command_limiter(user_id)
        if not await user_limiter.is_allowed():
            retry_after = user_limiter.time_until_allowed()
            raise RateLimitExceeded('user_command', retry_after)
        
        return True
    
    async def check_api_limit(self) -> bool:
        """
        Check global API rate limit
        
        Returns:
            True if allowed, False if rate limited
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        if not await self.global_api_limit.consume(1):
            retry_after = self.global_api_limit.time_until_available(1)
            raise RateLimitExceeded('global_api', retry_after)
        
        return True
    
    def add_vip_user(self, user_id: str):
        """Add user to VIP list (no rate limiting)"""
        self.vip_users.add(user_id)
        logger.info(f"Added VIP user: {user_id}")
    
    def remove_vip_user(self, user_id: str):
        """Remove user from VIP list"""
        self.vip_users.discard(user_id)
        logger.info(f"Removed VIP user: {user_id}")
    
    def reset_user_limits(self, user_id: str):
        """Reset rate limits for a user"""
        if user_id in self.user_message_limits:
            del self.user_message_limits[user_id]
        if user_id in self.user_command_limits:
            del self.user_command_limits[user_id]
        
        logger.info(f"Reset rate limits for user: {user_id}")
    
    def get_user_status(self, user_id: str) -> Dict[str, any]:
        """Get rate limit status for user"""
        status = {
            'user_id': user_id,
            'is_vip': user_id in self.vip_users,
            'message_limit': {
                'max': self.user_message_max,
                'window': self.user_message_window
            },
            'command_limit': {
                'max': self.user_command_max,
                'window': self.user_command_window
            }
        }
        
        # Get current usage
        if user_id in self.user_message_limits:
            limiter = self.user_message_limits[user_id]
            status['message_limit']['used'] = len(limiter.requests)
            status['message_limit']['retry_after'] = limiter.time_until_allowed()
        
        if user_id in self.user_command_limits:
            limiter = self.user_command_limits[user_id]
            status['command_limit']['used'] = len(limiter.requests)
            status['command_limit']['retry_after'] = limiter.time_until_allowed()
        
        return status


# Decorator for rate-limited functions
def rate_limited(limiter: RateLimiter, limit_type: str = 'message'):
    """
    Decorator to apply rate limiting to async functions
    
    Args:
        limiter: RateLimiter instance
        limit_type: Type of limit ('message', 'command', 'api')
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user_id from args/kwargs
            user_id = kwargs.get('user_id')
            if not user_id and args:
                user_id = args[0] if isinstance(args[0], str) else None
            
            # Check rate limit
            if limit_type == 'message' and user_id:
                await limiter.check_message_limit(user_id)
            elif limit_type == 'command' and user_id:
                await limiter.check_command_limit(user_id)
            elif limit_type == 'api':
                await limiter.check_api_limit()
            
            # Call function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


if __name__ == "__main__":
    from opensable.core.config import load_config
    
    config = load_config()
    limiter = RateLimiter(config)
    
    # Test
    async def test():
        user_id = "test_user"
        
        # Try sending messages
        for i in range(25):
            try:
                await limiter.check_message_limit(user_id)
                print(f"Message {i+1}: ✓ Allowed")
            except RateLimitExceeded as e:
                print(f"Message {i+1}: ✗ {e}")
                break
        
        # Check status
        status = limiter.get_user_status(user_id)
        print(f"\nUser status: {status}")
    
    asyncio.run(test())
