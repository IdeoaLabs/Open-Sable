"""
Tool schemas for System domain.
"""

SCHEMAS = [
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
    "description": "List files and folders in a directory. Use RELATIVE paths inside the project (e.g. '.', 'data/', 'opensable/core'). Do NOT browse system directories like /home, /usr, /etc.",
    "parameters": {
    "type": "object",
    "properties": {
    "path": {
    "type": "string",
    "description": "Relative path inside the project (default '.')",
    }
    },
    "required": [],
    },
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
    "description": "Search for files by name pattern or content. Always use RELATIVE paths inside the project. Do NOT search system directories.",
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

]
