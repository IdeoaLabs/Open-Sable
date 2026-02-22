"""
API Integration Examples - REST API client with authentication.

Demonstrates HTTP methods, auth, retry, caching, and rate limiting.
"""

import asyncio
from opensable.skills.api_client import APIClient, APIAuth, AuthType


async def main():
    """Run API integration examples."""

    print("=" * 60)
    print("API Integration Examples")
    print("=" * 60)

    # Example 1: Simple GET request
    print("\n1. Simple GET Request")
    print("-" * 40)

    client = APIClient(base_url="https://jsonplaceholder.typicode.com")

    response = await client.get("/posts/1")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.data}")

    # Example 2: POST request
    print("\n2. POST Request")
    print("-" * 40)

    new_post = {
        "title": "Open-Sable Test",
        "body": "This is a test post from Open-Sable",
        "userId": 1,
    }

    response = await client.post("/posts", json=new_post)
    print(f"Created post ID: {response.data.get('id')}")
    print(f"Title: {response.data.get('title')}")

    # Example 3: PUT request
    print("\n3. PUT Request")
    print("-" * 40)

    updated_post = {"id": 1, "title": "Updated Title", "body": "Updated body", "userId": 1}

    response = await client.put("/posts/1", json=updated_post)
    print(f"Updated post: {response.data.get('title')}")

    # Example 4: DELETE request
    print("\n4. DELETE Request")
    print("-" * 40)

    response = await client.delete("/posts/1")
    print(f"Delete status: {response.status_code}")

    # Example 5: Authentication - API Key
    print("\n5. API Key Authentication")
    print("-" * 40)

    auth_client = APIClient(
        base_url="https://api.example.com",
        auth=APIAuth(type=AuthType.API_KEY, api_key="sk-test-1234567890"),
    )

    print(f"Auth type: {auth_client.auth.type.value}")
    print("Headers will include API key")

    # Example 6: Authentication - Bearer Token
    print("\n6. Bearer Token Authentication")
    print("-" * 40)

    bearer_client = APIClient(
        base_url="https://api.example.com",
        auth=APIAuth(type=AuthType.BEARER, token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."),
    )

    print(f"Auth type: {bearer_client.auth.type.value}")
    print("Bearer token configured")

    # Example 7: Retry with backoff
    print("\n7. Retry Logic")
    print("-" * 40)

    retry_client = APIClient(
        base_url="https://jsonplaceholder.typicode.com", max_retries=3, retry_delay=1.0
    )

    print(f"Max retries: {retry_client.max_retries}")
    print(f"Retry delay: {retry_client.retry_delay}s")

    # Example 8: Response caching
    print("\n8. Response Caching")
    print("-" * 40)

    cache_client = APIClient(
        base_url="https://jsonplaceholder.typicode.com", cache_ttl=60  # Cache for 60 seconds
    )

    # First request (not cached)
    import time

    start = time.time()
    response1 = await cache_client.get("/posts/1")
    time1 = time.time() - start
    print(f"First request: {time1:.3f}s")

    # Second request (from cache)
    start = time.time()
    response2 = await cache_client.get("/posts/1")
    time2 = time.time() - start
    print(f"Second request (cached): {time2:.3f}s")
    print(f"Cache speedup: {time1/time2:.1f}x faster")

    # Example 9: Rate limiting
    print("\n9. Rate Limiting")
    print("-" * 40)

    rate_client = APIClient(
        base_url="https://jsonplaceholder.typicode.com", rate_limit=5  # 5 requests per second
    )

    print(f"Rate limit: {rate_client.rate_limit} req/s")

    # Make multiple requests
    start = time.time()
    for i in range(5):
        await rate_client.get(f"/posts/{i+1}")
    elapsed = time.time() - start

    print(f"Made 5 requests in {elapsed:.2f}s")

    # Example 10: Query parameters
    print("\n10. Query Parameters")
    print("-" * 40)

    response = await client.get("/posts", params={"userId": 1, "_limit": 3})
    print(f"Found {len(response.data)} posts for user 1")

    # Example 11: Custom headers
    print("\n11. Custom Headers")
    print("-" * 40)

    response = await client.get(
        "/posts/1", headers={"User-Agent": "Open-Sable/1.0", "X-Custom-Header": "value"}
    )
    print(f"Request with custom headers: {response.status_code}")

    # Example 12: Error handling
    print("\n12. Error Handling")
    print("-" * 40)

    try:
        response = await client.get("/nonexistent-endpoint")
    except Exception as e:
        print(f"Error (expected): {type(e).__name__}")

    # Example 13: GraphQL query
    print("\n13. GraphQL Query")
    print("-" * 40)

    graphql_query = """
    query GetUser($id: ID!) {
        user(id: $id) {
            id
            name
            email
        }
    }
    """

    print("GraphQL query prepared")
    print("Variables: {'id': '1'}")

    # Example 14: Batch requests
    print("\n14. Batch Requests")
    print("-" * 40)

    tasks = [client.get(f"/posts/{i}") for i in range(1, 6)]

    responses = await asyncio.gather(*tasks)
    print(f"Fetched {len(responses)} posts concurrently")
    for i, resp in enumerate(responses, 1):
        print(f"  Post {i}: {resp.data.get('title', '')[:40]}...")

    # Cleanup
    await client.close()
    await auth_client.close()
    await bearer_client.close()
    await retry_client.close()
    await cache_client.close()
    await rate_client.close()

    print("\n" + "=" * 60)
    print("✅ API integration examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
