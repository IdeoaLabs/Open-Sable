"""
Computer Control Tools - Autonomous system control
Enables the agent to execute commands, modify files, and control the computer
"""
import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import shutil

logger = logging.getLogger(__name__)


class ComputerTools:
    """
    Computer control tools
    Allows agent to execute shell commands, modify files, and control the system
    """
    
    def __init__(self, config, sandbox_mode: bool = False):
        self.config = config
        self.sandbox_mode = sandbox_mode
        self.command_history: List[Dict] = []
        
    async def execute_command(
        self, 
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 30,
        capture_output: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a shell command
        
        Args:
            command: Shell command to execute
            cwd: Working directory (default: current)
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr
            
        Returns:
            {
                'success': bool,
                'stdout': str,
                'stderr': str,
                'exit_code': int,
                'command': str
            }
        """
        logger.info(f"Executing command: {command}")
        
        # Security: Blocked dangerous commands in sandbox mode
        if self.sandbox_mode:
            dangerous = ['rm -rf /', 'mkfs', 'dd', ':(){:|:&};:', 'chmod -R 777']
            if any(d in command for d in dangerous):
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': 'Command blocked by security policy',
                    'exit_code': 1,
                    'command': command
                }
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
                cwd=cwd or os.getcwd()
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            result = {
                'success': process.returncode == 0,
                'stdout': stdout.decode('utf-8', errors='ignore') if stdout else '',
                'stderr': stderr.decode('utf-8', errors='ignore') if stderr else '',
                'exit_code': process.returncode,
                'command': command
            }
            
            # Store in history
            self.command_history.append(result)
            
            logger.info(f"Command completed with exit code: {process.returncode}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Command timed out after {timeout}s: {command}")
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'exit_code': -1,
                'command': command
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1,
                'command': command
            }
    
    async def read_file(self, path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Read file contents
        
        Args:
            path: File path (absolute or relative)
            encoding: Text encoding
            
        Returns:
            {
                'success': bool,
                'content': str,
                'size': int,
                'path': str,
                'error': str (if failed)
            }
        """
        try:
            file_path = Path(path).resolve()
            
            if not file_path.exists():
                return {
                    'success': False,
                    'content': '',
                    'size': 0,
                    'path': str(file_path),
                    'error': 'File not found'
                }
            
            if file_path.is_dir():
                return {
                    'success': False,
                    'content': '',
                    'size': 0,
                    'path': str(file_path),
                    'error': 'Path is a directory'
                }
            
            content = file_path.read_text(encoding=encoding)
            
            logger.info(f"Read file: {file_path} ({len(content)} chars)")
            
            return {
                'success': True,
                'content': content,
                'size': len(content),
                'path': str(file_path),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return {
                'success': False,
                'content': '',
                'size': 0,
                'path': path,
                'error': str(e)
            }
    
    async def write_file(
        self, 
        path: str, 
        content: str,
        mode: str = 'w',
        encoding: str = 'utf-8',
        create_dirs: bool = True
    ) -> Dict[str, Any]:
        """
        Write content to file
        
        Args:
            path: File path
            content: Content to write
            mode: Write mode ('w' = overwrite, 'a' = append)
            encoding: Text encoding
            create_dirs: Create parent directories if needed
            
        Returns:
            {
                'success': bool,
                'path': str,
                'bytes_written': int,
                'error': str (if failed)
            }
        """
        try:
            file_path = Path(path).resolve()
            
            # Create parent directories if needed
            if create_dirs and not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directories: {file_path.parent}")
            
            # Write file
            if mode == 'w':
                file_path.write_text(content, encoding=encoding)
            elif mode == 'a':
                with file_path.open('a', encoding=encoding) as f:
                    f.write(content)
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            bytes_written = len(content.encode(encoding))
            
            logger.info(f"Wrote file: {file_path} ({bytes_written} bytes)")
            
            return {
                'success': True,
                'path': str(file_path),
                'bytes_written': bytes_written,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            return {
                'success': False,
                'path': path,
                'bytes_written': 0,
                'error': str(e)
            }
    
    async def edit_file(
        self,
        path: str,
        old_content: str,
        new_content: str,
        encoding: str = 'utf-8'
    ) -> Dict[str, Any]:
        """
        Edit file by replacing old_content with new_content
        Similar to VSCode's find-and-replace
        
        Args:
            path: File path
            old_content: Content to find
            new_content: Content to replace with
            encoding: Text encoding
            
        Returns:
            {
                'success': bool,
                'path': str,
                'replacements': int,
                'error': str (if failed)
            }
        """
        try:
            file_path = Path(path).resolve()
            
            if not file_path.exists():
                return {
                    'success': False,
                    'path': str(file_path),
                    'replacements': 0,
                    'error': 'File not found'
                }
            
            # Read current content
            content = file_path.read_text(encoding=encoding)
            
            # Replace
            new_file_content = content.replace(old_content, new_content)
            replacements = content.count(old_content)
            
            if replacements == 0:
                return {
                    'success': False,
                    'path': str(file_path),
                    'replacements': 0,
                    'error': 'Old content not found in file'
                }
            
            # Write back
            file_path.write_text(new_file_content, encoding=encoding)
            
            logger.info(f"Edited file: {file_path} ({replacements} replacements)")
            
            return {
                'success': True,
                'path': str(file_path),
                'replacements': replacements,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to edit file {path}: {e}")
            return {
                'success': False,
                'path': path,
                'replacements': 0,
                'error': str(e)
            }
    
    async def list_directory(
        self,
        path: str = '.',
        include_hidden: bool = False,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        List directory contents
        
        Args:
            path: Directory path
            include_hidden: Include hidden files (starting with .)
            recursive: Recursively list subdirectories
            
        Returns:
            {
                'success': bool,
                'path': str,
                'files': List[Dict],
                'error': str (if failed)
            }
        """
        try:
            dir_path = Path(path).resolve()
            
            if not dir_path.exists():
                return {
                    'success': False,
                    'path': str(dir_path),
                    'files': [],
                    'error': 'Directory not found'
                }
            
            if not dir_path.is_dir():
                return {
                    'success': False,
                    'path': str(dir_path),
                    'files': [],
                    'error': 'Path is not a directory'
                }
            
            files = []
            
            if recursive:
                pattern = '**/*'
            else:
                pattern = '*'
            
            for item in dir_path.glob(pattern):
                # Skip hidden files if not requested
                if not include_hidden and item.name.startswith('.'):
                    continue
                
                files.append({
                    'name': item.name,
                    'path': str(item),
                    'type': 'directory' if item.is_dir() else 'file',
                    'size': item.stat().st_size if item.is_file() else 0
                })
            
            logger.info(f"Listed directory: {dir_path} ({len(files)} items)")
            
            return {
                'success': True,
                'path': str(dir_path),
                'files': files,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to list directory {path}: {e}")
            return {
                'success': False,
                'path': path,
                'files': [],
                'error': str(e)
            }
    
    async def create_directory(
        self,
        path: str,
        parents: bool = True
    ) -> Dict[str, Any]:
        """
        Create a directory
        
        Args:
            path: Directory path
            parents: Create parent directories if needed
            
        Returns:
            {
                'success': bool,
                'path': str,
                'error': str (if failed)
            }
        """
        try:
            dir_path = Path(path).resolve()
            dir_path.mkdir(parents=parents, exist_ok=True)
            
            logger.info(f"Created directory: {dir_path}")
            
            return {
                'success': True,
                'path': str(dir_path),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return {
                'success': False,
                'path': path,
                'error': str(e)
            }
    
    async def delete_file(self, path: str) -> Dict[str, Any]:
        """
        Delete a file or directory
        
        Args:
            path: Path to delete
            
        Returns:
            {
                'success': bool,
                'path': str,
                'error': str (if failed)
            }
        """
        try:
            file_path = Path(path).resolve()
            
            if not file_path.exists():
                return {
                    'success': False,
                    'path': str(file_path),
                    'error': 'Path not found'
                }
            
            if file_path.is_dir():
                shutil.rmtree(file_path)
            else:
                file_path.unlink()
            
            logger.info(f"Deleted: {file_path}")
            
            return {
                'success': True,
                'path': str(file_path),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return {
                'success': False,
                'path': path,
                'error': str(e)
            }
    
    async def move_file(
        self,
        source: str,
        destination: str
    ) -> Dict[str, Any]:
        """
        Move/rename a file or directory
        
        Args:
            source: Source path
            destination: Destination path
            
        Returns:
            {
                'success': bool,
                'source': str,
                'destination': str,
                'error': str (if failed)
            }
        """
        try:
            src_path = Path(source).resolve()
            dst_path = Path(destination).resolve()
            
            if not src_path.exists():
                return {
                    'success': False,
                    'source': str(src_path),
                    'destination': str(dst_path),
                    'error': 'Source not found'
                }
            
            shutil.move(str(src_path), str(dst_path))
            
            logger.info(f"Moved: {src_path} -> {dst_path}")
            
            return {
                'success': True,
                'source': str(src_path),
                'destination': str(dst_path),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to move {source} to {destination}: {e}")
            return {
                'success': False,
                'source': source,
                'destination': destination,
                'error': str(e)
            }
    
    async def copy_file(
        self,
        source: str,
        destination: str
    ) -> Dict[str, Any]:
        """
        Copy a file or directory
        
        Args:
            source: Source path
            destination: Destination path
            
        Returns:
            {
                'success': bool,
                'source': str,
                'destination': str,
                'error': str (if failed)
            }
        """
        try:
            src_path = Path(source).resolve()
            dst_path = Path(destination).resolve()
            
            if not src_path.exists():
                return {
                    'success': False,
                    'source': str(src_path),
                    'destination': str(dst_path),
                    'error': 'Source not found'
                }
            
            if src_path.is_dir():
                shutil.copytree(str(src_path), str(dst_path))
            else:
                shutil.copy2(str(src_path), str(dst_path))
            
            logger.info(f"Copied: {src_path} -> {dst_path}")
            
            return {
                'success': True,
                'source': str(src_path),
                'destination': str(dst_path),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to copy {source} to {destination}: {e}")
            return {
                'success': False,
                'source': source,
                'destination': destination,
                'error': str(e)
            }
    
    async def search_files(
        self,
        path: str,
        pattern: str,
        content_search: bool = False,
        case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """
        Search for files by name or content
        
        Args:
            path: Directory to search
            pattern: Search pattern (filename or content regex)
            content_search: Search file contents instead of names
            case_sensitive: Case-sensitive search
            
        Returns:
            {
                'success': bool,
                'matches': List[Dict],
                'error': str (if failed)
            }
        """
        try:
            import re
            
            dir_path = Path(path).resolve()
            matches = []
            
            if not dir_path.exists():
                return {
                    'success': False,
                    'matches': [],
                    'error': 'Directory not found'
                }
            
            regex_flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, regex_flags)
            
            for item in dir_path.rglob('*'):
                if item.is_file():
                    if content_search:
                        # Search in file content
                        try:
                            content = item.read_text(errors='ignore')
                            if regex.search(content):
                                matches.append({
                                    'path': str(item),
                                    'type': 'content',
                                    'size': item.stat().st_size
                                })
                        except:
                            pass
                    else:
                        # Search by filename
                        if regex.search(item.name):
                            matches.append({
                                'path': str(item),
                                'type': 'filename',
                                'size': item.stat().st_size
                            })
            
            logger.info(f"Search completed: {len(matches)} matches")
            
            return {
                'success': True,
                'matches': matches,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                'success': False,
                'matches': [],
                'error': str(e)
            }
    
    def get_command_history(self, limit: int = 10) -> List[Dict]:
        """Get recent command execution history"""
        return self.command_history[-limit:]
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        import psutil
        
        try:
            return {
                'success': True,
                'system': platform.system(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
