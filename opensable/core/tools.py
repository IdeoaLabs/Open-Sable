"""
Tool registry for Open-Sable - manages all available actions
"""

import logging
import json
import aiohttp
from typing import Dict, Any, Callable, List
from pathlib import Path
from datetime import datetime

from .computer_tools import ComputerTools
from .browser import BrowserEngine
from .skill_creator import SkillCreator

try:
    from ..skills import VoiceSkill, ImageSkill, DatabaseSkill, RAGSkill, CodeExecutor, APIClient
except ImportError:
    # Graceful fallback if a skill is not available
    from ..skills import VoiceSkill, DatabaseSkill, RAGSkill, CodeExecutor, APIClient
    from ..skills.image_skill import ImageAnalyzer as ImageSkill  # type: ignore

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of all available tools/actions"""

    # Map tool schema names → security ActionType for RBAC checking
    _TOOL_PERMISSIONS = {
        "execute_command": "system_command",
        "read_file": "file_read",
        "write_file": "file_write",
        "edit_file": "file_write",
        "delete_file": "file_delete",
        "move_file": "file_write",
        "copy_file": "file_write",
        "create_directory": "file_write",
        "browser_search": "browser_navigate",
        "browser_scrape": "browser_navigate",
        "browser_snapshot": "browser_navigate",
        "browser_action": "browser_navigate",
        "execute_code": "system_command",
        "desktop_screenshot": "system_command",
        "desktop_click": "system_command",
        "desktop_type": "system_command",
        "desktop_hotkey": "system_command",
    }

    def __init__(self, config):
        self.config = config
        self.tools: Dict[str, Callable] = {}
        self._permission_manager = None

        # Initialize permission manager for RBAC
        try:
            from .security import PermissionManager

            self._permission_manager = PermissionManager(config)
            self._permission_manager.initialize()
            logger.info("✅ Permission manager loaded for tool RBAC")
        except Exception as e:
            logger.debug(f"Permission manager not available: {e}")

        # Calendar storage
        self.calendar_file = Path.home() / ".opensable" / "calendar.json"
        self.calendar_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.calendar_file.exists():
            self.calendar_file.write_text(json.dumps([], indent=2))

        # Initialize browser engine
        self.browser_engine = BrowserEngine()

        # Initialize computer control tools
        self.computer = ComputerTools(config, sandbox_mode=getattr(config, "sandbox_mode", False))

        # Initialize skill creator
        self.skill_creator = SkillCreator(config)

        # Initialize advanced skills
        self.voice = VoiceSkill(config)
        self.image = ImageSkill(config)
        self.database = DatabaseSkill(config)
        self.rag = RAGSkill(config)
        self.code_executor = CodeExecutor(config)
        self.api_client = APIClient(config)

    async def initialize(self):
        """Initialize all tools"""
        # Register computer control tools (CRITICAL)
        self.register("execute_command", self._execute_command_tool)
        self.register("read_file", self._read_file_tool)
        self.register("write_file", self._write_file_tool)
        self.register("edit_file", self._edit_file_tool)
        self.register("list_directory", self._list_directory_tool)
        self.register("create_directory", self._create_directory_tool)
        self.register("delete_file", self._delete_file_tool)
        self.register("move_file", self._move_file_tool)
        self.register("copy_file", self._copy_file_tool)
        self.register("search_files", self._search_files_tool)
        self.register("system_info", self._system_info_tool)

        # Register built-in tools
        self.register("email", self._email_tool)
        self.register("calendar", self._calendar_tool)
        self.register("browser", self._browser_tool)
        self.register("web_action", self._web_action_tool)
        self.register("weather", self._weather_tool)

        # Register advanced skills
        self.register("voice_speak", self._voice_speak_tool)
        self.register("voice_listen", self._voice_listen_tool)
        self.register("generate_image", self._generate_image_tool)
        self.register("analyze_image", self._analyze_image_tool)
        self.register("ocr", self._ocr_tool)
        self.register("database_query", self._database_query_tool)
        self.register("vector_search", self._vector_search_tool)
        self.register("execute_code", self._execute_code_tool)
        self.register("api_request", self._api_request_tool)

        # Register skill creation
        self.register("create_skill", self._create_skill_tool)
        self.register("list_skills", self._list_skills_tool)

        # Register desktop control tools
        self.register("desktop_screenshot", self._desktop_screenshot_tool)
        self.register("desktop_click", self._desktop_click_tool)
        self.register("desktop_type", self._desktop_type_tool)
        self.register("desktop_hotkey", self._desktop_hotkey_tool)
        self.register("desktop_scroll", self._desktop_scroll_tool)
        self.register("desktop_mouse_move", self._desktop_mouse_move_tool)

        # Initialize advanced skills
        try:
            await self.voice.initialize()
        except Exception as e:
            logger.warning(f"Voice skill initialization failed: {e}")

        try:
            await self.image.initialize()
        except Exception as e:
            logger.warning(f"Image skill initialization failed: {e}")

        try:
            await self.database.initialize()
        except Exception as e:
            logger.warning(f"Database skill initialization failed: {e}")

        try:
            await self.rag.initialize()
        except Exception as e:
            logger.warning(f"RAG skill initialization failed: {e}")

        logger.info(f"Initialized {len(self.tools)} tools")

    def register(self, name: str, func: Callable):
        """Register a new tool"""
        self.tools[name] = func
        logger.debug(f"Registered tool: {name}")

    def list_tools(self) -> List[str]:
        """List all available tools"""
        return list(self.tools.keys())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return Ollama-compatible tool schemas (OpenAI function calling format)"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "browser_search",
                    "description": "Search the web for information about any topic, person, place, news, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"},
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results (default 5)",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_scrape",
                    "description": "Fetch and read the content of a specific web page URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The full URL to scrape (must start with http:// or https://)",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_snapshot",
                    "description": "Take an accessibility snapshot of a web page to see interactive elements (buttons, inputs, links) with stable refs for automation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL to snapshot"}
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_action",
                    "description": "Interact with a web page: click buttons, type text, submit forms. Use after browser_snapshot to get refs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action: click, type, hover, select, press, submit, evaluate, wait",
                            },
                            "ref": {
                                "type": "string",
                                "description": "Element ref from snapshot (e.g. 'e3')",
                            },
                            "selector": {
                                "type": "string",
                                "description": "CSS selector (alternative to ref)",
                            },
                            "value": {
                                "type": "string",
                                "description": "Value for type/fill/select actions",
                            },
                            "url": {
                                "type": "string",
                                "description": "URL to navigate to before action (optional)",
                            },
                        },
                        "required": ["action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Run a shell command on the system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute",
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Working directory (optional)",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default 30)",
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Absolute or relative file path",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "content": {"type": "string", "description": "Content to write"},
                            "mode": {
                                "type": "string",
                                "description": "'w' to overwrite (default), 'a' to append",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and folders in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path (default '.')",
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name, country, or address",
                            }
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calendar",
                    "description": "Manage calendar events: list upcoming events, add new events, or delete events",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "description": "list, add, or delete"},
                            "title": {"type": "string", "description": "Event title (for add)"},
                            "date": {
                                "type": "string",
                                "description": "Date/time in YYYY-MM-DD HH:MM format (for add)",
                            },
                            "description": {
                                "type": "string",
                                "description": "Event description (optional)",
                            },
                            "id": {"type": "integer", "description": "Event ID (for delete)"},
                        },
                        "required": ["action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Execute Python or other code in a sandbox",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Code to execute"},
                            "language": {
                                "type": "string",
                                "description": "Programming language (default: python)",
                            },
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vector_search",
                    "description": "Semantic search through stored documents and knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results (default 5)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_skill",
                    "description": "Create a new dynamic skill/extension for the agent. The skill will be validated, saved, and loaded automatically.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Skill name (snake_case, e.g. 'weather_check')",
                            },
                            "description": {"type": "string", "description": "What the skill does"},
                            "code": {"type": "string", "description": "Python code for the skill"},
                            "author": {
                                "type": "string",
                                "description": "Author name (default: 'sable')",
                            },
                        },
                        "required": ["name", "description", "code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_skills",
                    "description": "List all custom skills created by the agent",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            # ── File & system tools ─────────────────────────────
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit a file by replacing specific text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"},
                            "old_content": {"type": "string", "description": "Text to find"},
                            "new_content": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["path", "old_content", "new_content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file or empty directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "description": "Path to delete"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "move_file",
                    "description": "Move or rename a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Source path"},
                            "destination": {"type": "string", "description": "Destination path"},
                        },
                        "required": ["source", "destination"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for files by name pattern or content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "Search pattern (glob or text)",
                            },
                            "path": {
                                "type": "string",
                                "description": "Directory to search in (default '.')",
                            },
                            "content": {
                                "type": "string",
                                "description": "Search inside file contents for this text",
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "system_info",
                    "description": "Get system information: OS, CPU, memory, disk usage",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            # ── Desktop control tools ─────────────────────────────
            {
                "type": "function",
                "function": {
                    "name": "desktop_screenshot",
                    "description": "Take a screenshot of the screen. Returns base64 PNG image and dimensions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "save_path": {
                                "type": "string",
                                "description": "Optional file path to save the PNG to instead of returning base64",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "desktop_click",
                    "description": "Click the mouse at screen coordinates (x, y)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "description": "X pixel coordinate"},
                            "y": {"type": "integer", "description": "Y pixel coordinate"},
                            "button": {
                                "type": "string",
                                "description": "'left' (default), 'right', or 'middle'",
                            },
                            "clicks": {
                                "type": "integer",
                                "description": "Number of clicks (2 = double-click)",
                            },
                        },
                        "required": ["x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "desktop_type",
                    "description": "Type text via the keyboard at the current cursor position",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The text to type"}
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "desktop_hotkey",
                    "description": "Press a key or key combination (e.g. 'enter', 'ctrl+c', 'alt+f4', 'ctrl+shift+t')",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Key or combo like 'enter', 'ctrl+c', 'alt+tab'",
                            }
                        },
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "desktop_scroll",
                    "description": "Scroll the mouse wheel. Positive = up, negative = down.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "integer",
                                "description": "Scroll amount (positive=up, negative=down)",
                            },
                            "x": {
                                "type": "integer",
                                "description": "Optional X coordinate to scroll at",
                            },
                            "y": {
                                "type": "integer",
                                "description": "Optional Y coordinate to scroll at",
                            },
                        },
                        "required": ["amount"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "desktop_mouse_move",
                    "description": "Move the mouse to screen coordinates (x, y)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "integer", "description": "X pixel coordinate"},
                            "y": {"type": "integer", "description": "Y pixel coordinate"},
                        },
                        "required": ["x", "y"],
                    },
                },
            },
        ]

    # Tool schema → internal tool name mapping
    _SCHEMA_TO_TOOL = {
        "browser_search": ("browser", lambda a: {"action": "search", **a}),
        "browser_scrape": ("browser", lambda a: {"action": "scrape", **a}),
        "browser_snapshot": ("browser", lambda a: {"action": "snapshot", **a}),
        "browser_action": ("web_action", lambda a: a),
        "execute_command": ("execute_command", lambda a: a),
        "read_file": ("read_file", lambda a: a),
        "write_file": ("write_file", lambda a: a),
        "list_directory": ("list_directory", lambda a: a),
        "weather": ("weather", lambda a: a),
        "calendar": ("calendar", lambda a: a),
        "execute_code": ("execute_code", lambda a: a),
        "vector_search": ("vector_search", lambda a: a),
        "create_skill": ("create_skill", lambda a: a),
        "list_skills": ("list_skills", lambda a: a),
        # File & system tools
        "edit_file": ("edit_file", lambda a: a),
        "delete_file": ("delete_file", lambda a: a),
        "move_file": ("move_file", lambda a: a),
        "search_files": ("search_files", lambda a: a),
        "system_info": ("system_info", lambda a: a),
        # Desktop control
        "desktop_screenshot": ("desktop_screenshot", lambda a: a),
        "desktop_click": ("desktop_click", lambda a: a),
        "desktop_type": ("desktop_type", lambda a: a),
        "desktop_hotkey": ("desktop_hotkey", lambda a: a),
        "desktop_scroll": ("desktop_scroll", lambda a: a),
        "desktop_mouse_move": ("desktop_mouse_move", lambda a: a),
    }

    async def execute_schema_tool(
        self, schema_name: str, arguments: Dict[str, Any], user_id: str = "default"
    ) -> str:
        """Execute a tool by its schema name (as returned by Ollama tool calling)"""
        if schema_name not in self._SCHEMA_TO_TOOL:
            return f"⚠️ Unknown tool: {schema_name}"

        # RBAC check — if a permission manager is loaded and the tool is mapped
        if self._permission_manager and schema_name in self._TOOL_PERMISSIONS:
            from .security import ActionType

            action_str = self._TOOL_PERMISSIONS[schema_name]
            try:
                action = ActionType(action_str)
                allowed = await self._permission_manager.check_permission(
                    user_id, action, {"tool": schema_name, "arguments": arguments}
                )
                if not allowed:
                    logger.warning(f"🔒 RBAC denied {schema_name} for user {user_id}")
                    return f"🔒 Permission denied: {schema_name} requires '{action_str}' permission"
            except Exception as e:
                # Don't block tools if RBAC check itself fails
                logger.debug(f"RBAC check error (allowing): {e}")

        internal_name, arg_mapper = self._SCHEMA_TO_TOOL[schema_name]
        mapped_args = arg_mapper(arguments)
        return await self.execute(internal_name, mapped_args)

    async def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Execute a tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        logger.info(f"Executing tool: {tool_name}")
        try:
            result = await self.tools[tool_name](tool_input)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            raise

    # ========== COMPUTER CONTROL TOOLS ==========

    async def _execute_command_tool(self, params: Dict) -> str:
        """Execute shell command"""
        command = params.get("command", "")
        cwd = params.get("cwd")
        timeout = params.get("timeout", 30)

        result = await self.computer.execute_command(command, cwd=cwd, timeout=timeout)

        if result["success"]:
            output = result["stdout"] if result["stdout"] else "(no output)"
            return f"✅ Command executed successfully\n\n```\n{output}\n```\n\nExit code: {result['exit_code']}"
        else:
            return (
                f"❌ Command failed\n\nError: {result['stderr']}\nExit code: {result['exit_code']}"
            )

    async def _read_file_tool(self, params: Dict) -> str:
        """Read file contents"""
        path = params.get("path", "")

        result = await self.computer.read_file(path)

        if result["success"]:
            content = result["content"]
            # Truncate very long files
            if len(content) > 10000:
                content = content[:10000] + f"\n... (truncated, total size: {result['size']} bytes)"
            return f"📄 File: {result['path']}\n\n```\n{content}\n```"
        else:
            return f"❌ Failed to read file: {result['error']}"

    async def _write_file_tool(self, params: Dict) -> str:
        """Write content to file"""
        path = params.get("path", "")
        content = params.get("content", "")
        mode = params.get("mode", "w")

        result = await self.computer.write_file(path, content, mode=mode)

        if result["success"]:
            return f"✅ Wrote {result['bytes_written']} bytes to {result['path']}"
        else:
            return f"❌ Failed to write file: {result['error']}"

    async def _edit_file_tool(self, params: Dict) -> str:
        """Edit file by replacing content"""
        path = params.get("path", "")
        old_content = params.get("old_content", "")
        new_content = params.get("new_content", "")

        result = await self.computer.edit_file(path, old_content, new_content)

        if result["success"]:
            return f"✅ Made {result['replacements']} replacement(s) in {result['path']}"
        else:
            return f"❌ Failed to edit file: {result['error']}"

    async def _list_directory_tool(self, params: Dict) -> str:
        """List directory contents"""
        path = params.get("path", ".")
        include_hidden = params.get("include_hidden", False)
        recursive = params.get("recursive", False)

        result = await self.computer.list_directory(path, include_hidden, recursive)

        if result["success"]:
            files = result["files"]
            output = f"📁 Directory: {result['path']}\n\n"

            if not files:
                return output + "(empty directory)"

            for f in files[:50]:  # Limit to 50 items
                icon = "📁" if f["type"] == "directory" else "📄"
                size = f"({f['size']} bytes)" if f["type"] == "file" else ""
                output += f"{icon} {f['name']} {size}\n"

            if len(files) > 50:
                output += f"\n... and {len(files) - 50} more items"

            return output
        else:
            return f"❌ Failed to list directory: {result['error']}"

    async def _create_directory_tool(self, params: Dict) -> str:
        """Create directory"""
        path = params.get("path", "")

        result = await self.computer.create_directory(path)

        if result["success"]:
            return f"✅ Created directory: {result['path']}"
        else:
            return f"❌ Failed to create directory: {result['error']}"

    async def _delete_file_tool(self, params: Dict) -> str:
        """Delete file or directory"""
        path = params.get("path", "")

        result = await self.computer.delete_file(path)

        if result["success"]:
            return f"✅ Deleted: {result['path']}"
        else:
            return f"❌ Failed to delete: {result['error']}"

    async def _move_file_tool(self, params: Dict) -> str:
        """Move/rename file"""
        source = params.get("source", "")
        destination = params.get("destination", "")

        result = await self.computer.move_file(source, destination)

        if result["success"]:
            return f"✅ Moved: {result['source']} → {result['destination']}"
        else:
            return f"❌ Failed to move: {result['error']}"

    async def _copy_file_tool(self, params: Dict) -> str:
        """Copy file or directory"""
        source = params.get("source", "")
        destination = params.get("destination", "")

        result = await self.computer.copy_file(source, destination)

        if result["success"]:
            return f"✅ Copied: {result['source']} → {result['destination']}"
        else:
            return f"❌ Failed to copy: {result['error']}"

    async def _search_files_tool(self, params: Dict) -> str:
        """Search for files"""
        path = params.get("path", ".")
        pattern = params.get("pattern", "")
        content_search = params.get("content_search", False)

        result = await self.computer.search_files(path, pattern, content_search)

        if result["success"]:
            matches = result["matches"]
            output = f"🔍 Search results for '{pattern}' in {path}\n\n"

            if not matches:
                return output + "No matches found"

            for m in matches[:20]:  # Limit to 20 results
                output += f"• {m['path']}\n"

            if len(matches) > 20:
                output += f"\n... and {len(matches) - 20} more matches"

            return output
        else:
            return f"❌ Search failed: {result['error']}"

    async def _system_info_tool(self, params: Dict) -> str:
        """Get system information"""
        result = await self.computer.get_system_info()

        if result["success"]:
            return f"""💻 System Information

**Platform:** {result['system']} ({result['platform']})
**Python:** {result['python_version']}

**CPU:**
- Cores: {result['cpu_count']}
- Usage: {result['cpu_percent']}%

**Memory:**
- Total: {result['memory_total'] / (1024**3):.2f} GB
- Available: {result['memory_available'] / (1024**3):.2f} GB
- Usage: {result['memory_percent']}%

**Disk:**
- Total: {result['disk_usage']['total'] / (1024**3):.2f} GB
- Used: {result['disk_usage']['used'] / (1024**3):.2f} GB
- Free: {result['disk_usage']['free'] / (1024**3):.2f} GB
- Usage: {result['disk_usage']['percent']}%
"""
        else:
            return f"❌ Failed to get system info: {result['error']}"

    # ========== ORIGINAL TOOLS ==========

    # Built-in tools (simplified implementations)

    async def _email_tool(self, params: Dict) -> str:
        """Email operations via SMTP/IMAP"""
        action = params.get("action", "read")

        if action == "send":
            host = getattr(self.config, "smtp_host", None)
            if not host:
                return (
                    "⚠️ SMTP not configured. Add to .env:\n"
                    "  SMTP_HOST=smtp.gmail.com\n"
                    "  SMTP_USER=you@gmail.com\n"
                    "  SMTP_PASSWORD=your-app-password"
                )

            to = params.get("to", "")
            subject = params.get("subject", "(no subject)")
            body = params.get("body", "")
            if not to:
                return "⚠️ Missing 'to' field — who should I send the email to?"

            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart

                msg = MIMEMultipart()
                msg["From"] = getattr(self.config, "smtp_from", None) or self.config.smtp_user
                msg["To"] = to
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "plain"))

                port = int(getattr(self.config, "smtp_port", 587))
                with smtplib.SMTP(host, port) as server:
                    server.starttls()
                    server.login(self.config.smtp_user, self.config.smtp_password)
                    server.send_message(msg)

                logger.info(f"📧 Email sent to {to}: {subject}")
                return f"✅ Email sent to **{to}**\nSubject: {subject}"
            except Exception as e:
                logger.error(f"Email send failed: {e}")
                return f"❌ Failed to send email: {e}"

        elif action == "read":
            host = getattr(self.config, "imap_host", None)
            if not host:
                return (
                    "⚠️ IMAP not configured. Add to .env:\n"
                    "  IMAP_HOST=imap.gmail.com\n"
                    "  IMAP_USER=you@gmail.com\n"
                    "  IMAP_PASSWORD=your-app-password"
                )

            count = int(params.get("count", 5))
            folder = params.get("folder", "INBOX")

            try:
                import imaplib
                import email as email_lib
                from email.header import decode_header

                port = int(getattr(self.config, "imap_port", 993))
                with imaplib.IMAP4_SSL(host, port) as imap:
                    imap.login(
                        getattr(self.config, "imap_user", None) or self.config.smtp_user,
                        getattr(self.config, "imap_password", None) or self.config.smtp_password,
                    )
                    imap.select(folder, readonly=True)
                    _, data = imap.search(None, "ALL")
                    ids = data[0].split()
                    if not ids:
                        return f"📧 No emails in {folder}."

                    latest = ids[-count:]
                    latest.reverse()
                    results = []
                    for mid in latest:
                        _, msg_data = imap.fetch(mid, "(RFC822)")
                        raw = msg_data[0][1]
                        msg = email_lib.message_from_bytes(raw)
                        subj = ""
                        for part, enc in decode_header(msg["Subject"] or ""):
                            subj += (
                                part.decode(enc or "utf-8")
                                if isinstance(part, bytes)
                                else str(part)
                            )
                        frm = msg["From"] or ""
                        date = msg["Date"] or ""
                        results.append(f"• **{subj}**\n  From: {frm}\n  Date: {date}")

                return f"📧 **Latest {len(results)} emails ({folder}):**\n\n" + "\n\n".join(results)
            except Exception as e:
                logger.error(f"Email read failed: {e}")
                return f"❌ Failed to read email: {e}"

        else:
            return f"Unknown email action: {action}. Use: send, read"

    async def _calendar_tool(self, params: Dict) -> str:
        """Internal calendar operations (stored locally)"""
        action = params.get("action", "list")

        try:
            # Load calendar events
            events = json.loads(self.calendar_file.read_text())

            if action == "list":
                # Show upcoming events
                now = datetime.now()
                upcoming = [e for e in events if datetime.fromisoformat(e["datetime"]) >= now]
                upcoming.sort(key=lambda x: x["datetime"])

                if not upcoming:
                    return "📅 No upcoming events in your calendar."

                result = "📅 **Upcoming Events:**\n\n"
                for event in upcoming[:10]:  # Show next 10
                    dt = datetime.fromisoformat(event["datetime"])
                    result += f"• **{event['title']}**\n"
                    result += f"  📆 {dt.strftime('%Y-%m-%d %H:%M')}\n"
                    if event.get("description"):
                        result += f"  📝 {event['description']}\n"
                    result += "\n"
                return result.strip()

            elif action == "add":
                title = params.get("title", "Untitled Event")
                date_str = params.get("date", "")
                description = params.get("description", "")

                if not date_str:
                    return "⚠️ Please provide a date/time (e.g., '2026-02-20 15:00' or 'tomorrow at 3pm')"

                # Parse date (simple ISO format support)
                try:
                    # Try ISO format first
                    event_dt = datetime.fromisoformat(date_str)
                except ValueError:
                    # Try common formats
                    for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"]:
                        try:
                            event_dt = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        return f"⚠️ Could not parse date '{date_str}'. Use format: YYYY-MM-DD HH:MM"

                # Add event
                new_event = {
                    "id": len(events) + 1,
                    "title": title,
                    "datetime": event_dt.isoformat(),
                    "description": description,
                    "created_at": datetime.now().isoformat(),
                }
                events.append(new_event)

                # Save
                self.calendar_file.write_text(json.dumps(events, indent=2))

                return f"✅ Event added: **{title}** on {event_dt.strftime('%Y-%m-%d %H:%M')}"

            elif action == "delete":
                event_id = params.get("id")
                if not event_id:
                    return "⚠️ Please provide event ID to delete"

                events = [e for e in events if e["id"] != int(event_id)]
                self.calendar_file.write_text(json.dumps(events, indent=2))
                return f"✅ Event {event_id} deleted"

            else:
                return f"Unknown calendar action: {action}. Use: list, add, delete"

        except Exception as e:
            logger.error(f"Calendar error: {e}")
            return f"⚠️ Calendar error: {str(e)}"

    async def _browser_tool(self, params: Dict) -> str:
        """Browser automation and web scraping using Playwright"""
        action = params.get("action", "scrape")

        if action == "snapshot":
            url = params.get("url", "")
            format_type = params.get("format", "aria")

            result = await self.browser_engine.snapshot(url, format_type)
            if result.get("success"):
                refs_text = f"📸 Snapshot of {result.get('url')}\n"
                refs_text += f"Found {result.get('count', 0)} interactive elements:\n\n"

                for ref_data in result.get("refs", [])[:20]:  # Limit to first 20
                    ref_text = f"{ref_data['ref']}: {ref_data['role']}"
                    if ref_data.get("name"):
                        ref_text += f" '{ref_data['name']}'"
                    refs_text += ref_text + "\n"

                if result.get("count", 0) > 20:
                    refs_text += f"\n... and {result.get('count') - 20} more elements"

                return refs_text
            else:
                return f"❌ Snapshot failed: {result.get('error', 'Unknown error')}"

        elif action == "scrape":
            url = params.get("url")
            result = await self.browser_engine.scrape_page(url)

            if not result.get("success"):
                return f"⚠️ {result.get('error', 'Unknown error')}"

            return f"🌐 **{result['title']}**\n\nURL: {result['url']}\n\n{result['content']}"

        elif action == "search":
            query = params.get("query")
            if not query:
                return "⚠️ Please provide a search query"

            logger.info(f"🔍 Searching web for: '{query}'")
            num_results = params.get("num_results", 5)
            result = await self.browser_engine.search_web(query, num_results)

            logger.info(
                f"Search returned: success={result.get('success')}, count={result.get('count', 0)}"
            )

            if not result.get("success"):
                return f"⚠️ {result.get('error', 'Unknown error')}"

            # Check if we got results
            results_list = result.get("results", [])
            if not results_list or len(results_list) == 0:
                logger.warning(f"No search results found for '{query}'")
                return f"🔍 No search results found for '{query}'. The search engine returned 0 results."

            # Format results
            response = f"🔍 **Search Results for: {query}**\n\n"
            for i, res in enumerate(results_list, 1):
                response += f"**{i}. {res.get('title', 'Untitled')}**\n"
                response += f"{res.get('snippet', 'No description')}\n"
                response += f"🔗 {res.get('url', '')}\n\n"

            return response.strip()

        elif action == "screenshot":
            url = params.get("url")
            if not url:
                return "⚠️ Please provide a URL for screenshot"

            result = await self.browser_engine.get_page_screenshot(url)

            if not result.get("success"):
                return f"⚠️ {result.get('error', 'Unknown error')}"

            return f"📸 Screenshot saved: {result['path']}"

        else:
            return f"Unknown browser action: {action}. Available: scrape, search, screenshot"

    async def _web_action_tool(self, params: Dict) -> str:
        """Execute interactive web actions using refs or selectors

        Actions: click, type, hover, drag, select, fill, press, evaluate, wait, submit
        Use refs from snapshot for stable automation
        """
        url = params.get("url")
        action = params.get("action", "")
        ref = params.get("ref")
        selector = params.get("selector")
        value = params.get("value")

        if not action:
            return "⚠️ Missing action parameter"

        try:
            result = await self.browser_engine.execute_action(
                url=url, action=action, ref=ref, selector=selector, value=value
            )

            if result.get("success"):
                action_name = result.get("action", action).capitalize()
                details = ""

                if "value" in result:
                    details = f": '{result['value']}'"
                elif "key" in result:
                    details = f": {result['key']}"
                elif "result" in result:
                    details = f" -> {result['result']}"

                return f"✅ {action_name}{details}"
            else:
                return f"❌ {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    async def _file_tool(self, params: Dict) -> str:
        """
        DEPRECATED: Use specific file tools instead
        (read_file, write_file, edit_file, list_directory, etc.)
        """
        action = params.get("action", "list")

        if action == "list":
            return await self._list_directory_tool({"path": params.get("path", ".")})
        elif action == "read":
            return await self._read_file_tool({"path": params.get("path", "")})
        else:
            return "⚠️ Use specific file tools: read_file, write_file, edit_file, list_directory"

    async def _weather_tool(self, params: Dict) -> str:
        """Weather information using wttr.in (no API key required)"""
        location = params.get("location", "")

        if not location:
            # Use IP-based auto-detection
            location = ""

        try:
            # Call wttr.in API (free, no API key needed)
            async with aiohttp.ClientSession() as session:
                # Format: ?format=j1 for JSON, ?m for metric
                url = f"https://wttr.in/{location}?format=j1&m"
                headers = {"User-Agent": "curl/7.68.0"}  # wttr.in prefers curl user agent

                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        # Parse weather data
                        current = data["current_condition"][0]
                        location_info = data["nearest_area"][0]

                        temp = current["temp_C"]
                        feels_like = current["FeelsLikeC"]
                        humidity = current["humidity"]
                        description = current["weatherDesc"][0]["value"]
                        wind_speed = current["windspeedKmph"]
                        city_name = location_info.get("areaName", [{}])[0].get(
                            "value", location or "Your location"
                        )

                        # Weather emoji mapping
                        weather_code = int(current["weatherCode"])
                        weather_emojis = {
                            113: "☀️",  # Clear/Sunny
                            116: "⛅",  # Partly cloudy
                            119: "☁️",  # Cloudy
                            122: "☁️",  # Overcast
                            143: "🌫️",  # Mist
                            176: "🌦️",  # Patchy rain possible
                            200: "⛈️",  # Thundery outbreaks possible
                            248: "🌫️",  # Fog
                            263: "🌧️",  # Patchy light drizzle
                            266: "🌧️",  # Light drizzle
                            281: "🌧️",  # Freezing drizzle
                            284: "🌧️",  # Heavy freezing drizzle
                            293: "🌧️",  # Patchy light rain
                            296: "🌧️",  # Light rain
                            299: "🌧️",  # Moderate rain at times
                            302: "🌧️",  # Moderate rain
                            305: "🌧️",  # Heavy rain at times
                            308: "🌧️",  # Heavy rain
                            311: "🌧️",  # Light freezing rain
                            314: "🌧️",  # Moderate or heavy freezing rain
                            317: "🌨️",  # Light sleet
                            320: "🌨️",  # Moderate or heavy sleet
                            323: "❄️",  # Patchy light snow
                            326: "❄️",  # Light snow
                            329: "❄️",  # Patchy moderate snow
                            332: "❄️",  # Moderate snow
                            335: "❄️",  # Patchy heavy snow
                            338: "❄️",  # Heavy snow
                            350: "🌨️",  # Ice pellets
                            353: "🌧️",  # Light rain shower
                            356: "🌧️",  # Moderate or heavy rain shower
                            359: "🌧️",  # Torrential rain shower
                            362: "🌨️",  # Light sleet showers
                            365: "🌨️",  # Moderate or heavy sleet showers
                            368: "❄️",  # Light snow showers
                            371: "❄️",  # Moderate or heavy snow showers
                            374: "🌨️",  # Light showers of ice pellets
                            377: "🌨️",  # Moderate or heavy showers of ice pellets
                            386: "⛈️",  # Patchy light rain with thunder
                            389: "⛈️",  # Moderate or heavy rain with thunder
                            392: "⛈️",  # Patchy light snow with thunder
                            395: "⛈️",  # Moderate or heavy snow with thunder
                        }

                        emoji = weather_emojis.get(weather_code, "🌤️")

                        return (
                            f"{emoji} **Weather in {city_name}**\n"
                            f"🌡️ Temperature: {temp}°C (feels like {feels_like}°C)\n"
                            f"💧 Humidity: {humidity}%\n"
                            f"💨 Wind: {wind_speed} km/h\n"
                            f"📝 Conditions: {description}"
                        )
                    else:
                        return f"⚠️ Weather service error: {resp.status}"
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return f"⚠️ Failed to fetch weather data: {str(e)}"

    # ========== VOICE TOOLS ==========

    async def _voice_speak_tool(self, params: Dict) -> str:
        """Text-to-speech conversion"""
        text = params.get("text", "")
        output_file = params.get("output_file")

        try:
            audio_path = await self.voice.speak(text, output_file=output_file)
            return f"🔊 Audio generated: {audio_path}"
        except Exception as e:
            return f"❌ TTS failed: {str(e)}"

    async def _voice_listen_tool(self, params: Dict) -> str:
        """Speech-to-text conversion"""
        audio_file = params.get("audio_file")

        try:
            text = await self.voice.listen(audio_file=audio_file)
            return f"📝 Transcription:\n{text}"
        except Exception as e:
            return f"❌ STT failed: {str(e)}"

    # ========== IMAGE TOOLS ==========

    async def _generate_image_tool(self, params: Dict) -> str:
        """Generate images from text prompts"""
        prompt = params.get("prompt", "")
        model = params.get("model", "dall-e-3")
        size = params.get("size", "1024x1024")
        output_path = params.get("output_path", "generated_image.png")

        try:
            result = await self.image.generate(
                prompt=prompt, model=model, size=size, output_path=output_path
            )

            if result.get("success"):
                return f"🎨 Image generated: {result.get('path')}\nPrompt: {prompt}"
            else:
                return f"❌ Generation failed: {result.get('error')}"
        except Exception as e:
            return f"❌ Image generation error: {str(e)}"

    async def _analyze_image_tool(self, params: Dict) -> str:
        """Analyze image content"""
        image_path = params.get("image_path", "")

        try:
            result = await self.image.analyze(image_path)

            if result.get("success"):
                labels = ", ".join(result.get("labels", []))
                description = result.get("description", "No description")

                return f"🔍 Image Analysis:\n{description}\n\nDetected: {labels}"
            else:
                return f"❌ Analysis failed: {result.get('error')}"
        except Exception as e:
            return f"❌ Image analysis error: {str(e)}"

    async def _ocr_tool(self, params: Dict) -> str:
        """Extract text from images"""
        image_path = params.get("image_path", "")
        language = params.get("language", "eng")

        try:
            result = await self.image.ocr(image_path, language=language)

            if result.get("success"):
                text = result.get("text", "")
                confidence = result.get("confidence", 0)

                return f"📄 OCR Results (confidence: {confidence}%):\n\n{text}"
            else:
                return f"❌ OCR failed: {result.get('error')}"
        except Exception as e:
            return f"❌ OCR error: {str(e)}"

    # ========== DATABASE TOOLS ==========

    async def _database_query_tool(self, params: Dict) -> str:
        """Execute database queries"""
        query = params.get("query", "")
        db_type = params.get("db_type", "sqlite")
        database = params.get("database", "default.db")

        try:
            result = await self.database.execute(query=query, db_type=db_type, database=database)

            if result.get("success"):
                rows = result.get("rows", [])
                row_count = len(rows)

                return f"✅ Query executed successfully\nRows returned: {row_count}\n\n{json.dumps(rows[:10], indent=2)}"
            else:
                return f"❌ Query failed: {result.get('error')}"
        except Exception as e:
            return f"❌ Database error: {str(e)}"

    # ========== RAG TOOLS ==========

    async def _vector_search_tool(self, params: Dict) -> str:
        """Semantic search using vector database, falls back to web search if unavailable"""
        query = params.get("query", "")
        collection = params.get("collection", "default")
        top_k = int(params.get("top_k", 5))  # ensure int, not string

        try:
            results = await self.rag.search(query=query, collection=collection, top_k=top_k)

            if results:
                formatted = "\n\n".join(
                    [
                        f"**Result {i+1}** (score: {r.get('score', 0):.2f}):\n{r.get('content', '')}"
                        for i, r in enumerate(results)
                    ]
                )
                return f"🔍 Found {len(results)} results:\n\n{formatted}"
            else:
                # Local knowledge base is empty — fall back to web search
                logger.info(f"Vector DB empty for '{query}', falling back to browser_search")
                return await self._browser_tool(
                    {"action": "search", "query": query, "num_results": int(top_k)}
                )
        except Exception as e:
            # Embedding model not available — fall back to web search
            logger.warning(f"Vector search unavailable ({e}), falling back to browser_search")
            return await self._browser_tool({"action": "search", "query": query, "num_results": 5})

    # ========== CODE EXECUTION TOOLS ==========

    async def _execute_code_tool(self, params: Dict) -> str:
        """Execute code in sandbox"""
        code = params.get("code", "")
        language = params.get("language", "python")
        timeout = params.get("timeout", 30)

        try:
            result = await self.code_executor.execute(code=code, language=language, timeout=timeout)

            if result.get("success"):
                output = result.get("output", "")
                return f"✅ Code executed successfully:\n\n```\n{output}\n```"
            else:
                error = result.get("error", "Unknown error")
                return f"❌ Execution failed:\n{error}"
        except Exception as e:
            return f"❌ Code execution error: {str(e)}"

    # ========== API CLIENT TOOLS ==========

    async def _api_request_tool(self, params: Dict) -> str:
        """Make HTTP API requests"""
        url = params.get("url", "")
        method = params.get("method", "GET")
        headers = params.get("headers", {})
        data = params.get("data")

        try:
            result = await self.api_client.request(
                url=url, method=method, headers=headers, data=data
            )

            if result.get("success"):
                response_data = result.get("data", "")
                status_code = result.get("status_code", 200)

                return f"✅ API request successful (status: {status_code}):\n\n{json.dumps(response_data, indent=2)[:500]}"
            else:
                return f"❌ API request failed: {result.get('error')}"
        except Exception as e:
            return f"❌ API request error: {str(e)}"

    # ========== SKILL CREATION TOOLS ==========

    async def _create_skill_tool(self, params: Dict) -> str:
        """Create a new dynamic skill"""
        name = params.get("name", "")
        description = params.get("description", "")
        code = params.get("code", "")
        author = params.get("author", "sable")

        metadata = {"author": author, "created_at": datetime.utcnow().isoformat()}

        try:
            result = await self.skill_creator.create_skill(name, description, code, metadata)

            if result.get("success"):
                return f"✅ Skill '{name}' created successfully!\n\nPath: {result.get('path')}\n\nThe skill has been validated and is ready to use."
            else:
                return f"❌ Failed to create skill: {result.get('error')}"
        except Exception as e:
            return f"❌ Skill creation error: {str(e)}"

    async def _list_skills_tool(self, params: Dict) -> str:
        """List all custom skills"""
        try:
            skills = await self.skill_creator.list_skills()

            if not skills:
                return (
                    "📦 No custom skills created yet.\n\nUse create_skill to add new functionality!"
                )

            formatted = "\n".join(
                [
                    f"• **{s['name']}** - {s['description']}\n  Status: {'✅ Enabled' if s.get('enabled', True) else '❌ Disabled'}\n  Author: {s.get('metadata', {}).get('author', 'unknown')}"
                    for s in skills
                ]
            )

            return f"📦 Custom Skills ({len(skills)}):\n\n{formatted}"
        except Exception as e:
            return f"❌ Error listing skills: {str(e)}"

    # ========== DESKTOP CONTROL TOOLS ==========

    async def _desktop_screenshot_tool(self, params: Dict) -> str:
        """Take a screenshot of the screen"""
        save_path = params.get("save_path")
        result = await self.computer.screenshot(save_path=save_path)
        if result.get("success"):
            w, h = result.get("width"), result.get("height")
            if result.get("path"):
                return f"📸 Screenshot saved: {result['path']} ({w}x{h})"
            return f"📸 Screenshot captured ({w}x{h}). Image data returned as base64 PNG."
        return f"❌ Screenshot failed: {result.get('error')}"

    async def _desktop_click_tool(self, params: Dict) -> str:
        """Click mouse at coordinates"""
        x = params.get("x", 0)
        y = params.get("y", 0)
        button = params.get("button", "left")
        clicks = params.get("clicks", 1)
        result = await self.computer.mouse_click(x, y, button=button, clicks=clicks)
        if result.get("success"):
            return f"🖱️ Clicked ({button} x{clicks}) at ({x}, {y})"
        return f"❌ Click failed: {result.get('error')}"

    async def _desktop_type_tool(self, params: Dict) -> str:
        """Type text via keyboard"""
        text = params.get("text", "")
        result = await self.computer.keyboard_type(text)
        if result.get("success"):
            return f"⌨️ Typed {result.get('length', len(text))} characters"
        return f"❌ Type failed: {result.get('error')}"

    async def _desktop_hotkey_tool(self, params: Dict) -> str:
        """Press key or key combination"""
        key = params.get("key", "")
        result = await self.computer.keyboard_press(key)
        if result.get("success"):
            return f"⌨️ Pressed: {key}"
        return f"❌ Hotkey failed: {result.get('error')}"

    async def _desktop_scroll_tool(self, params: Dict) -> str:
        """Scroll the mouse wheel"""
        amount = params.get("amount", 0)
        x = params.get("x")
        y = params.get("y")
        result = await self.computer.mouse_scroll(amount, x=x, y=y)
        if result.get("success"):
            return f"🖱️ Scrolled {amount}"
        return f"❌ Scroll failed: {result.get('error')}"

    async def _desktop_mouse_move_tool(self, params: Dict) -> str:
        """Move mouse to coordinates"""
        x = params.get("x", 0)
        y = params.get("y", 0)
        result = await self.computer.mouse_move(x, y)
        if result.get("success"):
            return f"🖱️ Mouse moved to ({x}, {y})"
        return f"❌ Move failed: {result.get('error')}"
