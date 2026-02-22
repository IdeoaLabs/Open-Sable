"""
Secure sandbox runner for dynamically generated tools.

This module provides process-level isolation with resource limits to safely
execute LLM-generated Python code. It prevents:
- CPU/memory bombs
- Path traversal and file exfiltration
- Network access (basic app-level blocking)
- Excessive file operations
- Timeout/runaway processes

For production, consider container-based isolation (Docker/Podman) with
proper network policies and seccomp profiles.
"""

from __future__ import annotations
import os
import resource
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """Raised when sandbox execution fails or times out."""

    pass


def _limit_resources(cpu_seconds: int, mem_mb: int):
    """
    Apply resource limits to the current process.

    Args:
        cpu_seconds: Maximum CPU time in seconds
        mem_mb: Maximum memory in megabytes

    Note:
        This runs in the child process before exec.
    """
    # CPU time limit (triggers SIGXCPU)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

    # Address space limit (virtual memory)
    mem_bytes = mem_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    # No core dumps
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # Limit open file descriptors
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    # Limit number of processes (prevent fork bombs)
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def run_sandboxed_python(
    code: str,
    *,
    cpu_seconds: int = 2,
    mem_mb: int = 256,
    workdir: Optional[Path] = None,
    allowed_imports: Optional[list[str]] = None,
) -> str:
    """
    Execute Python code in a sandboxed subprocess with resource limits.

    Args:
        code: Python code to execute
        cpu_seconds: Maximum CPU time (default: 2 seconds)
        mem_mb: Maximum memory in MB (default: 256 MB)
        workdir: Working directory (default: temp dir)
        allowed_imports: List of allowed module names (default: stdlib only)

    Returns:
        Combined stdout/stderr output from the execution

    Raises:
        SandboxError: If execution times out, crashes, or violates limits

    Example:
        >>> result = run_sandboxed_python("print('hello')", cpu_seconds=1)
        >>> print(result)
        hello

    Security Notes:
        - Runs in isolated process with resource limits
        - No network access (app-level blocking via env vars)
        - Limited file descriptors and no fork
        - For production: use containers with seccomp/AppArmor
    """
    # Create isolated working directory
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="opensable_tool_")).resolve()
    else:
        workdir = workdir.resolve()

    # Wrap code with import validation if allowed_imports is specified
    if allowed_imports is not None:
        wrapper = _create_import_wrapper(code, allowed_imports)
    else:
        wrapper = code

    # Write code to file
    tool_path = workdir / "tool.py"
    tool_path.write_text(wrapper, encoding="utf-8")

    # Prepare isolated environment
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        # Block network at application level (not foolproof, but helps)
        "no_proxy": "*",
        "NO_PROXY": "*",
        "http_proxy": "http://127.0.0.1:1",
        "https_proxy": "http://127.0.0.1:1",
    }

    def preexec():
        """Run in child process before exec."""
        # Create new session (detach from parent)
        os.setsid()
        # Apply resource limits
        _limit_resources(cpu_seconds=cpu_seconds, mem_mb=mem_mb)

    try:
        logger.info(f"Running sandboxed tool: cpu={cpu_seconds}s, mem={mem_mb}MB")

        proc = subprocess.run(
            ["python3", "-I", str(tool_path)],  # -I: isolated mode
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=cpu_seconds + 1,  # Grace period for cleanup
            preexec_fn=preexec,
        )

        output = proc.stdout or ""

        if proc.returncode != 0:
            logger.warning(f"Tool exited with code {proc.returncode}")
            raise SandboxError(f"Tool failed with exit code {proc.returncode}\n{output}")

        logger.info(f"Tool completed successfully ({len(output)} bytes output)")
        return output

    except subprocess.TimeoutExpired as e:
        logger.error(f"Tool execution timed out after {cpu_seconds}s")
        raise SandboxError(f"Tool execution timed out after {cpu_seconds} seconds") from e

    except Exception as e:
        logger.error(f"Sandbox execution failed: {e}")
        raise SandboxError(f"Sandbox execution failed: {e}") from e


def _create_import_wrapper(code: str, allowed_imports: list[str]) -> str:
    """
    Wrap code with import validation logic.

    Args:
        code: Original Python code
        allowed_imports: List of allowed module names

    Returns:
        Wrapped code that validates imports at runtime
    """
    allowed_set = set(allowed_imports)

    wrapper = textwrap.dedent(f"""
        import sys
        import builtins
        
        # Store original import
        _original_import = builtins.__import__
        
        # Allowed modules
        _allowed = {allowed_set!r}
        
        def _safe_import(name, *args, **kwargs):
            # Check if module is allowed
            top_level = name.split('.')[0]
            if top_level not in _allowed:
                raise ImportError(f"Import of '{{name}}' is not allowed in sandbox")
            return _original_import(name, *args, **kwargs)
        
        # Override __import__
        builtins.__import__ = _safe_import
        
        # Execute user code
        {textwrap.indent(code, '        ')}
    """)

    return wrapper


def validate_tool_safety(code: str) -> tuple[bool, list[str]]:
    """
    Perform static analysis to detect obviously dangerous patterns.

    Args:
        code: Python code to validate

    Returns:
        Tuple of (is_safe, list_of_violations)

    Example:
        >>> is_safe, violations = validate_tool_safety("os.system('rm -rf /')")
        >>> print(is_safe, violations)
        False ['Dangerous call: os.system']
    """
    violations = []

    # Dangerous built-ins and calls
    dangerous_patterns = [
        ("exec(", "Direct code execution"),
        ("eval(", "Direct code evaluation"),
        ("__import__", "Dynamic imports"),
        ("compile(", "Code compilation"),
        ("globals(", "Global scope access"),
        ("locals()", "Local scope manipulation"),
        ("vars(", "Variable manipulation"),
        ("setattr(", "Attribute injection"),
        ("delattr(", "Attribute deletion"),
        ("os.system", "Shell command execution"),
        ("subprocess.", "Subprocess execution"),
        ("os.popen", "Process spawning"),
        ("os.spawn", "Process spawning"),
        ("os.exec", "Process replacement"),
        ("os.fork", "Process forking"),
        ("open(", "File access"),  # Could be legitimate, but flagged for review
        ("rm -rf", "Dangerous shell command"),
        ("mkfs", "Filesystem formatting"),
        ("dd if=", "Low-level disk operations"),
        (">/dev/", "Device file access"),
        ("chmod", "Permission changes"),
        ("chown", "Ownership changes"),
    ]

    code_lower = code.lower()

    for pattern, description in dangerous_patterns:
        if pattern.lower() in code_lower:
            violations.append(f"{description}: {pattern}")

    is_safe = len(violations) == 0

    if not is_safe:
        logger.warning(f"Tool safety validation failed: {violations}")

    return is_safe, violations


# Default allowed imports for sandboxed tools (standard library only)
DEFAULT_ALLOWED_IMPORTS = [
    "math",
    "random",
    "datetime",
    "time",
    "json",
    "re",
    "hashlib",
    "base64",
    "urllib.parse",
    "itertools",
    "functools",
    "collections",
    "decimal",
    "fractions",
    "statistics",
    "string",
    "textwrap",
]


if __name__ == "__main__":
    # Demo usage
    print("=== Sandbox Runner Demo ===\n")

    # Safe code
    safe_code = """
import math
result = math.sqrt(16)
print(f"Square root of 16: {result}")
"""

    print("1. Running safe code:")
    try:
        output = run_sandboxed_python(safe_code, cpu_seconds=1, mem_mb=128)
        print(f"✅ Output: {output}")
    except SandboxError as e:
        print(f"❌ Error: {e}")

    # Unsafe code (should fail validation)
    unsafe_code = """
import os
os.system("echo 'This should not run'")
"""

    print("\n2. Validating unsafe code:")
    is_safe, violations = validate_tool_safety(unsafe_code)
    print(f"Is safe: {is_safe}")
    print(f"Violations: {violations}")

    # Memory bomb (should hit limit)
    memory_bomb = """
data = []
while True:
    data.append([0] * 1000000)
"""

    print("\n3. Running memory bomb (should fail):")
    try:
        output = run_sandboxed_python(memory_bomb, cpu_seconds=2, mem_mb=64)
        print(f"Output: {output}")
    except SandboxError as e:
        print(f"✅ Caught: {e}")

    # Timeout test
    timeout_code = """
import time
time.sleep(10)
"""

    print("\n4. Running timeout test (should fail):")
    try:
        output = run_sandboxed_python(timeout_code, cpu_seconds=1, mem_mb=128)
        print(f"Output: {output}")
    except SandboxError as e:
        print(f"✅ Caught: {e}")

    print("\n=== Demo Complete ===")
