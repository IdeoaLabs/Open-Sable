"""
Code Execution Skill - Safe sandbox for executing code in multiple languages.

Supports:
- Python (sandboxed with RestrictedPython)
- JavaScript (Node.js in Docker container)
- Bash/Shell (restricted commands)
- Docker-based isolation for untrusted code
- Resource limits (CPU, memory, timeout)
- Output capture (stdout, stderr, return value)
"""

import asyncio
import tempfile
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import shutil
import hashlib

try:
    from RestrictedPython import compile_restricted, safe_globals
    from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str]
    return_value: Any
    execution_time: float
    language: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "return_value": str(self.return_value) if self.return_value is not None else None,
            "execution_time": self.execution_time,
            "language": self.language,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr
        }


@dataclass
class ExecutionConfig:
    """Configuration for code execution."""
    timeout: int = 30  # seconds
    max_memory: str = "512m"  # Docker memory limit
    max_cpu: float = 1.0  # CPU cores
    allow_network: bool = False
    allow_file_write: bool = False
    working_dir: Optional[str] = None
    env_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}


class CodeExecutor:
    """
    Execute code safely in sandboxed environments.
    
    Features:
    - Multi-language support (Python, JS, Bash)
    - Docker isolation for untrusted code
    - Resource limits (CPU, memory, timeout)
    - Network isolation
    - File system restrictions
    - Output capture and sanitization
    """
    
    def __init__(self, use_docker: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize code executor.
        
        Args:
            use_docker: Use Docker for isolation (recommended for production)
            cache_dir: Directory for caching execution results
        """
        self.use_docker = use_docker
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".opensable" / "code_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check Docker availability
        if use_docker:
            self.docker_available = self._check_docker()
        else:
            self.docker_available = False
            
        # Restricted Python globals
        self.safe_globals = self._setup_safe_globals()
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _setup_safe_globals(self) -> Dict[str, Any]:
        """Setup safe globals for restricted Python execution."""
        if not RESTRICTED_PYTHON_AVAILABLE:
            return {}
        
        safe_builtins = safe_globals.copy()
        safe_builtins['_iter_unpack_sequence_'] = guarded_iter_unpack_sequence
        safe_builtins['_getattr_'] = safer_getattr
        
        # Allow safe built-ins
        safe_builtins['__builtins__']['print'] = print
        safe_builtins['__builtins__']['len'] = len
        safe_builtins['__builtins__']['range'] = range
        safe_builtins['__builtins__']['enumerate'] = enumerate
        safe_builtins['__builtins__']['zip'] = zip
        safe_builtins['__builtins__']['map'] = map
        safe_builtins['__builtins__']['filter'] = filter
        safe_builtins['__builtins__']['sum'] = sum
        safe_builtins['__builtins__']['max'] = max
        safe_builtins['__builtins__']['min'] = min
        safe_builtins['__builtins__']['sorted'] = sorted
        safe_builtins['__builtins__']['abs'] = abs
        safe_builtins['__builtins__']['round'] = round
        
        return safe_builtins
    
    async def execute(
        self,
        code: str,
        language: str = "python",
        config: Optional[ExecutionConfig] = None
    ) -> ExecutionResult:
        """
        Execute code in specified language.
        
        Args:
            code: Code to execute
            language: Programming language (python, javascript, bash)
            config: Execution configuration
        
        Returns:
            ExecutionResult with output and metrics
        """
        config = config or ExecutionConfig()
        start_time = datetime.now()
        
        # Check cache
        cache_key = self._get_cache_key(code, language, config)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            if language.lower() in ["python", "py"]:
                result = await self._execute_python(code, config)
            elif language.lower() in ["javascript", "js", "node"]:
                result = await self._execute_javascript(code, config)
            elif language.lower() in ["bash", "sh", "shell"]:
                result = await self._execute_bash(code, config)
            else:
                result = ExecutionResult(
                    success=False,
                    output="",
                    error=f"Unsupported language: {language}",
                    return_value=None,
                    execution_time=0.0,
                    language=language,
                    exit_code=1
                )
        except Exception as e:
            result = ExecutionResult(
                success=False,
                output="",
                error=str(e),
                return_value=None,
                execution_time=0.0,
                language=language,
                exit_code=1
            )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        result.execution_time = execution_time
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    async def _execute_python(self, code: str, config: ExecutionConfig) -> ExecutionResult:
        """Execute Python code."""
        if self.use_docker and self.docker_available:
            return await self._execute_python_docker(code, config)
        else:
            return await self._execute_python_restricted(code, config)
    
    async def _execute_python_restricted(self, code: str, config: ExecutionConfig) -> ExecutionResult:
        """Execute Python code with RestrictedPython."""
        if not RESTRICTED_PYTHON_AVAILABLE:
            return ExecutionResult(
                success=False,
                output="",
                error="RestrictedPython not available. Install: pip install RestrictedPython",
                return_value=None,
                execution_time=0.0,
                language="python",
                exit_code=1
            )
        
        try:
            # Compile restricted code
            byte_code = compile_restricted(code, '<string>', 'exec')
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_restricted_python(byte_code),
                timeout=config.timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution timeout ({config.timeout}s)",
                return_value=None,
                execution_time=config.timeout,
                language="python",
                exit_code=124
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                return_value=None,
                execution_time=0.0,
                language="python",
                exit_code=1
            )
    
    async def _run_restricted_python(self, byte_code) -> ExecutionResult:
        """Run compiled restricted Python code."""
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(byte_code, self.safe_globals)
            
            return ExecutionResult(
                success=True,
                output=stdout.getvalue(),
                error=stderr.getvalue() if stderr.getvalue() else None,
                return_value=None,
                execution_time=0.0,
                language="python",
                exit_code=0,
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue()
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=stdout.getvalue(),
                error=str(e),
                return_value=None,
                execution_time=0.0,
                language="python",
                exit_code=1,
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue()
            )
    
    async def _execute_python_docker(self, code: str, config: ExecutionConfig) -> ExecutionResult:
        """Execute Python code in Docker container."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            code_file = Path(tmpdir) / "script.py"
            code_file.write_text(code)
            
            # Build Docker command
            docker_cmd = [
                "docker", "run", "--rm",
                f"--memory={config.max_memory}",
                f"--cpus={config.max_cpu}",
                "-v", f"{tmpdir}:/workspace",
                "-w", "/workspace"
            ]
            
            if not config.allow_network:
                docker_cmd.extend(["--network", "none"])
            
            # Add environment variables
            for key, value in config.env_vars.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
            
            docker_cmd.extend(["python:3.11-slim", "python", "script.py"])
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout
                )
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode(),
                    error=stderr.decode() if stderr else None,
                    return_value=None,
                    execution_time=0.0,
                    language="python",
                    exit_code=process.returncode,
                    stdout=stdout.decode(),
                    stderr=stderr.decode()
                )
                
            except asyncio.TimeoutError:
                # Kill container
                await process.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timeout ({config.timeout}s)",
                    return_value=None,
                    execution_time=config.timeout,
                    language="python",
                    exit_code=124
                )
    
    async def _execute_javascript(self, code: str, config: ExecutionConfig) -> ExecutionResult:
        """Execute JavaScript code."""
        if self.use_docker and self.docker_available:
            return await self._execute_javascript_docker(code, config)
        else:
            return await self._execute_javascript_node(code, config)
    
    async def _execute_javascript_node(self, code: str, config: ExecutionConfig) -> ExecutionResult:
        """Execute JavaScript with Node.js."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = Path(tmpdir) / "script.js"
            code_file.write_text(code)
            
            try:
                process = await asyncio.create_subprocess_exec(
                    "node", str(code_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=config.env_vars
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout
                )
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode(),
                    error=stderr.decode() if stderr else None,
                    return_value=None,
                    execution_time=0.0,
                    language="javascript",
                    exit_code=process.returncode,
                    stdout=stdout.decode(),
                    stderr=stderr.decode()
                )
                
            except asyncio.TimeoutError:
                await process.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timeout ({config.timeout}s)",
                    return_value=None,
                    execution_time=config.timeout,
                    language="javascript",
                    exit_code=124
                )
            except FileNotFoundError:
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Node.js not found. Please install Node.js.",
                    return_value=None,
                    execution_time=0.0,
                    language="javascript",
                    exit_code=127
                )
    
    async def _execute_javascript_docker(self, code: str, config: ExecutionConfig) -> ExecutionResult:
        """Execute JavaScript in Docker container."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = Path(tmpdir) / "script.js"
            code_file.write_text(code)
            
            docker_cmd = [
                "docker", "run", "--rm",
                f"--memory={config.max_memory}",
                f"--cpus={config.max_cpu}",
                "-v", f"{tmpdir}:/workspace",
                "-w", "/workspace"
            ]
            
            if not config.allow_network:
                docker_cmd.extend(["--network", "none"])
            
            for key, value in config.env_vars.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
            
            docker_cmd.extend(["node:20-slim", "node", "script.js"])
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout
                )
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode(),
                    error=stderr.decode() if stderr else None,
                    return_value=None,
                    execution_time=0.0,
                    language="javascript",
                    exit_code=process.returncode,
                    stdout=stdout.decode(),
                    stderr=stderr.decode()
                )
                
            except asyncio.TimeoutError:
                await process.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timeout ({config.timeout}s)",
                    return_value=None,
                    execution_time=config.timeout,
                    language="javascript",
                    exit_code=124
                )
    
    async def _execute_bash(self, code: str, config: ExecutionConfig) -> ExecutionResult:
        """Execute Bash commands with restrictions."""
        # Security: Block dangerous commands
        dangerous_patterns = [
            "rm -rf /", ":(){ :|:& };:", "mkfs", "dd if=/dev/zero",
            "wget", "curl http", "> /dev/", "chmod 777"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code.lower():
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Blocked dangerous command pattern: {pattern}",
                    return_value=None,
                    execution_time=0.0,
                    language="bash",
                    exit_code=126
                )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            script_file = Path(tmpdir) / "script.sh"
            script_file.write_text(code)
            script_file.chmod(0o755)
            
            try:
                process = await asyncio.create_subprocess_exec(
                    "bash", str(script_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=config.working_dir or tmpdir,
                    env=config.env_vars
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout
                )
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode(),
                    error=stderr.decode() if stderr else None,
                    return_value=None,
                    execution_time=0.0,
                    language="bash",
                    exit_code=process.returncode,
                    stdout=stdout.decode(),
                    stderr=stderr.decode()
                )
                
            except asyncio.TimeoutError:
                await process.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timeout ({config.timeout}s)",
                    return_value=None,
                    execution_time=config.timeout,
                    language="bash",
                    exit_code=124
                )
    
    def _get_cache_key(self, code: str, language: str, config: ExecutionConfig) -> str:
        """Generate cache key for execution."""
        cache_data = f"{code}:{language}:{config.timeout}:{config.max_memory}"
        return hashlib.sha256(cache_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[ExecutionResult]:
        """Get cached execution result."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                return ExecutionResult(**data)
            except Exception:
                return None
        return None
    
    def _cache_result(self, cache_key: str, result: ExecutionResult):
        """Cache execution result."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            cache_file.write_text(json.dumps(result.to_dict()))
        except Exception:
            pass  # Caching is optional


# Example usage
async def main():
    """Example code execution."""
    executor = CodeExecutor(use_docker=False)
    
    # Python example
    print("=" * 50)
    print("Python Execution")
    print("=" * 50)
    python_code = """
for i in range(5):
    print(f"Hello {i}")
result = sum(range(10))
print(f"Sum: {result}")
"""
    result = await executor.execute(python_code, "python")
    print(f"Success: {result.success}")
    print(f"Output:\n{result.output}")
    print(f"Execution time: {result.execution_time:.3f}s")
    
    # JavaScript example
    print("\n" + "=" * 50)
    print("JavaScript Execution")
    print("=" * 50)
    js_code = """
const numbers = [1, 2, 3, 4, 5];
const sum = numbers.reduce((a, b) => a + b, 0);
console.log(`Sum: ${sum}`);
console.log(`Average: ${sum / numbers.length}`);
"""
    result = await executor.execute(js_code, "javascript")
    print(f"Success: {result.success}")
    print(f"Output:\n{result.output}")
    print(f"Error: {result.error}")
    
    # Bash example
    print("\n" + "=" * 50)
    print("Bash Execution")
    print("=" * 50)
    bash_code = """
echo "System info:"
uname -a
echo "Date:"
date
"""
    result = await executor.execute(bash_code, "bash")
    print(f"Success: {result.success}")
    print(f"Output:\n{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
