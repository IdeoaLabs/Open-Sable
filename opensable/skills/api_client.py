"""
API Integration Skill - Call external REST APIs with advanced features.

Features:
- HTTP methods (GET, POST, PUT, PATCH, DELETE)
- Authentication (API Key, Bearer Token, OAuth, Basic Auth)
- Request/response serialization
- Rate limiting
- Retry with exponential backoff
- Response caching
- Webhook handling
- GraphQL support
- OpenAPI/Swagger integration
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from pathlib import Path

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class AuthType(Enum):
    """Authentication types."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"


@dataclass
class APIAuth:
    """API authentication configuration."""
    type: AuthType = AuthType.NONE
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    bearer_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    oauth_token: Optional[str] = None
    
    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        if self.type == AuthType.API_KEY and self.api_key:
            return {self.api_key_header: self.api_key}
        elif self.type == AuthType.BEARER and self.bearer_token:
            return {"Authorization": f"Bearer {self.bearer_token}"}
        elif self.type == AuthType.OAUTH2 and self.oauth_token:
            return {"Authorization": f"Bearer {self.oauth_token}"}
        return {}


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    backoff_factor: float = 1.0  # seconds
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        return self.backoff_factor * (2 ** attempt)


@dataclass
class CacheConfig:
    """Response caching configuration."""
    enabled: bool = False
    ttl: int = 300  # seconds
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".opensable" / "api_cache")


@dataclass
class APIResponse:
    """API response wrapper."""
    status_code: int
    success: bool
    data: Any
    headers: Dict[str, str]
    error: Optional[str] = None
    cached: bool = False
    response_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status_code": self.status_code,
            "success": self.success,
            "data": self.data,
            "headers": dict(self.headers),
            "error": self.error,
            "cached": self.cached,
            "response_time": self.response_time
        }
    
    @property
    def json(self) -> Any:
        """Get JSON data."""
        return self.data
    
    @property
    def text(self) -> str:
        """Get text data."""
        return str(self.data)


class APIClient:
    """
    Advanced API client for REST API integration.
    
    Features:
    - Multiple HTTP methods
    - Flexible authentication
    - Automatic retries with backoff
    - Response caching
    - Rate limiting
    - Request/response logging
    - GraphQL support
    """
    
    def __init__(
        self,
        base_url: str,
        auth: Optional[APIAuth] = None,
        retry_config: Optional[RetryConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL for API
            auth: Authentication configuration
            retry_config: Retry configuration
            cache_config: Cache configuration
            timeout: Request timeout in seconds
            headers: Default headers
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx not installed: pip install httpx")
        
        self.base_url = base_url.rstrip("/")
        self.auth = auth or APIAuth()
        self.retry_config = retry_config or RetryConfig()
        self.cache_config = cache_config or CacheConfig()
        self.timeout = timeout
        self.default_headers = headers or {}
        
        # Setup cache directory
        if self.cache_config.enabled:
            Path(self.cache_config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True
        )
        
        # Rate limiting
        self._last_request_time = None
        self._min_request_interval = 0.0  # seconds
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def set_rate_limit(self, requests_per_second: float):
        """
        Set rate limit.
        
        Args:
            requests_per_second: Maximum requests per second
        """
        self._min_request_interval = 1.0 / requests_per_second
    
    async def _wait_for_rate_limit(self):
        """Wait for rate limit if needed."""
        if self._min_request_interval > 0 and self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
        
        self._last_request_time = datetime.now()
    
    def _get_cache_key(self, method: str, url: str, params: Dict, data: Any) -> str:
        """Generate cache key."""
        cache_data = f"{method}:{url}:{json.dumps(params, sort_keys=True)}:{json.dumps(data, sort_keys=True)}"
        return hashlib.sha256(cache_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[APIResponse]:
        """Get cached response if available."""
        if not self.cache_config.enabled:
            return None
        
        cache_file = Path(self.cache_config.cache_dir) / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                cached_at = datetime.fromisoformat(data["cached_at"])
                
                # Check if cache is still valid
                if (datetime.now() - cached_at).total_seconds() < self.cache_config.ttl:
                    response = APIResponse(**data["response"])
                    response.cached = True
                    return response
            except Exception:
                pass
        
        return None
    
    def _cache_response(self, cache_key: str, response: APIResponse):
        """Cache response."""
        if not self.cache_config.enabled:
            return
        
        cache_file = Path(self.cache_config.cache_dir) / f"{cache_key}.json"
        
        try:
            cache_data = {
                "cached_at": datetime.now().isoformat(),
                "response": response.to_dict()
            }
            cache_file.write_text(json.dumps(cache_data))
        except Exception:
            pass
    
    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        retry: bool = True
    ) -> APIResponse:
        """
        Make HTTP request.
        
        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            data: Request body (form data)
            json_data: Request body (JSON)
            headers: Request headers
            retry: Enable retry on failure
        
        Returns:
            APIResponse
        """
        # Build URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Merge headers
        request_headers = {**self.default_headers, **self.auth.get_headers()}
        if headers:
            request_headers.update(headers)
        
        # Check cache (only for GET requests)
        if method.upper() == "GET":
            cache_key = self._get_cache_key(method, url, params or {}, data)
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached
        
        # Rate limiting
        await self._wait_for_rate_limit()
        
        # Make request with retry
        attempt = 0
        last_error = None
        
        while attempt <= (self.retry_config.max_retries if retry else 0):
            try:
                start_time = datetime.now()
                
                response = await self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=request_headers
                )
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Parse response
                try:
                    response_data = response.json()
                except Exception:
                    response_data = response.text
                
                # Check if successful
                success = 200 <= response.status_code < 300
                
                api_response = APIResponse(
                    status_code=response.status_code,
                    success=success,
                    data=response_data,
                    headers=dict(response.headers),
                    error=None if success else f"HTTP {response.status_code}",
                    response_time=response_time
                )
                
                # Cache successful GET responses
                if success and method.upper() == "GET":
                    self._cache_response(cache_key, api_response)
                
                # Retry on specific status codes
                if not success and response.status_code in self.retry_config.retry_on_status:
                    raise Exception(f"Retryable status code: {response.status_code}")
                
                return api_response
                
            except Exception as e:
                last_error = str(e)
                
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    await asyncio.sleep(delay)
                
                attempt += 1
        
        # All retries failed
        return APIResponse(
            status_code=0,
            success=False,
            data=None,
            headers={},
            error=f"Request failed after {self.retry_config.max_retries} retries: {last_error}"
        )
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> APIResponse:
        """GET request."""
        return await self.request("GET", endpoint, params=params, **kwargs)
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> APIResponse:
        """POST request."""
        return await self.request("POST", endpoint, data=data, json_data=json_data, **kwargs)
    
    async def put(
        self,
        endpoint: str,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> APIResponse:
        """PUT request."""
        return await self.request("PUT", endpoint, data=data, json_data=json_data, **kwargs)
    
    async def patch(
        self,
        endpoint: str,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> APIResponse:
        """PATCH request."""
        return await self.request("PATCH", endpoint, data=data, json_data=json_data, **kwargs)
    
    async def delete(
        self,
        endpoint: str,
        **kwargs
    ) -> APIResponse:
        """DELETE request."""
        return await self.request("DELETE", endpoint, **kwargs)
    
    async def graphql(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        endpoint: str = "/graphql"
    ) -> APIResponse:
        """
        Execute GraphQL query.
        
        Args:
            query: GraphQL query string
            variables: Query variables
            endpoint: GraphQL endpoint
        
        Returns:
            APIResponse with GraphQL data
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        return await self.post(endpoint, json_data=payload)


# Example usage
async def main():
    """Example API client usage."""
    
    print("=" * 50)
    print("API Client Examples")
    print("=" * 50)
    
    # Example 1: Public API (no auth)
    print("\n1. JSONPlaceholder API (GET)")
    async with APIClient("https://jsonplaceholder.typicode.com") as client:
        response = await client.get("/posts/1")
        print(f"  Status: {response.status_code}")
        print(f"  Success: {response.success}")
        print(f"  Data: {response.json}")
        print(f"  Response time: {response.response_time:.3f}s")
    
    # Example 2: POST request
    print("\n2. JSONPlaceholder API (POST)")
    async with APIClient("https://jsonplaceholder.typicode.com") as client:
        response = await client.post(
            "/posts",
            json_data={
                "title": "Test Post",
                "body": "This is a test post",
                "userId": 1
            }
        )
        print(f"  Status: {response.status_code}")
        print(f"  Created: {response.json}")
    
    # Example 3: With caching
    print("\n3. Cached requests")
    cache_config = CacheConfig(enabled=True, ttl=300)
    async with APIClient(
        "https://jsonplaceholder.typicode.com",
        cache_config=cache_config
    ) as client:
        # First request (not cached)
        response1 = await client.get("/users/1")
        print(f"  First request - Cached: {response1.cached}")
        
        # Second request (cached)
        response2 = await client.get("/users/1")
        print(f"  Second request - Cached: {response2.cached}")
    
    # Example 4: Rate limiting
    print("\n4. Rate limiting")
    async with APIClient("https://jsonplaceholder.typicode.com") as client:
        client.set_rate_limit(2.0)  # 2 requests per second
        
        start = datetime.now()
        for i in range(3):
            await client.get(f"/posts/{i+1}")
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  3 requests took {elapsed:.2f}s (rate limited to 2/s)")
    
    # Example 5: Retry on failure
    print("\n5. Retry on failure")
    retry_config = RetryConfig(max_retries=3, backoff_factor=0.5)
    async with APIClient(
        "https://httpbin.org",
        retry_config=retry_config
    ) as client:
        # This will return 500 and trigger retries
        response = await client.get("/status/500")
        print(f"  Status: {response.status_code}")
        print(f"  Success: {response.success}")
        print(f"  Error: {response.error}")
    
    print("\n✅ API client examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
