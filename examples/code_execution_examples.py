"""
Code Execution Examples - Safe sandboxed code execution.

Demonstrates Python, JavaScript, and Bash execution with Docker isolation.
"""

import asyncio
from skills.code_executor import CodeExecutor, ExecutionConfig, Language


async def main():
    """Run code execution examples."""

    print("=" * 60)
    print("Code Execution Examples")
    print("=" * 60)

    executor = CodeExecutor()

    # Example 1: Python execution
    print("\n1. Python Code Execution")
    print("-" * 40)

    python_code = """
import math

def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

for i in range(10):
    print(f"fib({i}) = {calculate_fibonacci(i)}")
"""

    result = await executor.execute(
        code=python_code, language=Language.PYTHON, config=ExecutionConfig(timeout=5)
    )

    print(f"Status: {result.status}")
    print(f"Output:\n{result.output}")

    # Example 2: JavaScript execution
    print("\n2. JavaScript Code Execution")
    print("-" * 40)

    js_code = """
function isPrime(num) {
    if (num <= 1) return false;
    for (let i = 2; i <= Math.sqrt(num); i++) {
        if (num % i === 0) return false;
    }
    return true;
}

console.log("Prime numbers up to 50:");
for (let i = 2; i <= 50; i++) {
    if (isPrime(i)) {
        console.log(i);
    }
}
"""

    result = await executor.execute(
        code=js_code, language=Language.JAVASCRIPT, config=ExecutionConfig(timeout=5)
    )

    print(f"Status: {result.status}")
    print(f"Output:\n{result.output}")

    # Example 3: Bash script execution
    print("\n3. Bash Script Execution")
    print("-" * 40)

    bash_code = """
#!/bin/bash
echo "System Information:"
echo "==================="
echo "Date: $(date)"
echo "Current directory: $(pwd)"
echo "Environment variables:"
env | head -5
"""

    result = await executor.execute(
        code=bash_code, language=Language.BASH, config=ExecutionConfig(timeout=3)
    )

    print(f"Status: {result.status}")
    print(f"Output:\n{result.output}")

    # Example 4: Resource limits
    print("\n4. Resource Limits Test")
    print("-" * 40)

    memory_test = """
# This will fail due to memory limits
data = 'x' * (1024 * 1024 * 100)  # Try to allocate 100MB
print("Allocated memory")
"""

    result = await executor.execute(
        code=memory_test,
        language=Language.PYTHON,
        config=ExecutionConfig(timeout=2, memory_limit="50M"),
    )

    print(f"Status: {result.status}")
    if result.error:
        print(f"Error (expected): {result.error[:100]}...")

    # Example 5: Timeout handling
    print("\n5. Timeout Handling")
    print("-" * 40)

    infinite_loop = """
import time
while True:
    time.sleep(0.1)
    print("Running...")
"""

    result = await executor.execute(
        code=infinite_loop, language=Language.PYTHON, config=ExecutionConfig(timeout=2)
    )

    print(f"Status: {result.status}")
    print(f"Error: {result.error}")

    # Example 6: Caching
    print("\n6. Execution Caching")
    print("-" * 40)

    cached_code = "print('Hello from cache!')"

    # First execution
    result1 = await executor.execute(cached_code, Language.PYTHON)
    print(f"First execution time: {result1.execution_time:.3f}s")

    # Second execution (should be faster - from cache)
    result2 = await executor.execute(cached_code, Language.PYTHON)
    print(f"Second execution time: {result2.execution_time:.3f}s")
    print(f"Cached: {result2.execution_time < result1.execution_time}")

    print("\n" + "=" * 60)
    print("✅ Code execution examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
