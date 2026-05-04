"""
Agent management tool schemas,  create, stop, destroy, and list sub-agents.
"""

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "agent_create",
            "description": (
                "Create a new sub-agent and start it. The sub-agent gets its own "
                "profile directory, soul, tools config, and runs as a separate process "
                "with an auto-assigned port. It inherits your LLM configuration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Short name for the sub-agent (e.g. 'sweaters', 'analytics'). "
                            "Will be prefixed with your profile name automatically."
                        ),
                    },
                    "soul": {
                        "type": "string",
                        "description": (
                            "The soul/identity text for the sub-agent (its soul.md). "
                            "Describe who the agent is, what it does, its responsibilities, "
                            "behavioral rules, and any domain-specific knowledge."
                        ),
                    },
                    "tools_mode": {
                        "type": "string",
                        "enum": ["all", "allowlist", "denylist"],
                        "description": "Tool access mode: 'all' (unrestricted), 'allowlist' (only listed), 'denylist' (all except listed). Default: 'allowlist'.",
                    },
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of tool names or groups for the mode. "
                            "Groups: 'group:web', 'group:comms', 'group:documents', "
                            "'group:fs', 'group:runtime', 'group:social', 'group:trading', etc."
                        ),
                    },
                    "env_overrides": {
                        "type": "object",
                        "description": "Optional environment variable overrides (key-value pairs).",
                        "additionalProperties": {"type": "string"},
                    },
                    "auto_start": {
                        "type": "boolean",
                        "description": "Start the agent immediately after creation. Default: true.",
                    },
                },
                "required": ["name", "soul"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_stop",
            "description": "Stop a running sub-agent. The agent's profile directory is preserved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the sub-agent to stop (e.g. 'sweaters' or 'nano-sweaters').",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_start",
            "description": "Start (or restart) an existing sub-agent that has been stopped.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the sub-agent to start (e.g. 'sweaters' or 'nano-sweaters').",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_destroy",
            "description": (
                "Stop and permanently delete a sub-agent. Removes the agent's profile "
                "directory and all its data. This cannot be undone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the sub-agent to destroy (e.g. 'sweaters' or 'nano-sweaters').",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_list",
            "description": "List all sub-agents of the current agent with their running status.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_message",
            "description": "Send a message/instruction to a running sub-agent and get its response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the sub-agent to message (e.g. 'sweaters' or 'nano-sweaters').",
                    },
                    "message": {
                        "type": "string",
                        "description": "The message or instruction to send to the sub-agent.",
                    },
                },
                "required": ["name", "message"],
            },
        },
    },
]
