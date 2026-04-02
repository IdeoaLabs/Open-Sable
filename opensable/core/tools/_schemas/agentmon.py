"""
Tool schemas for the AgentMon League (Pokémon Red) skill.

OpenSable (https://opensable.com) — AI agents playing Pokémon Red
on a Game Boy emulator via the AgentMon League platform.
"""

SCHEMAS = [
    # ── Session ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "agentmon_start",
            "description": (
                "Start a Pokémon Red game session on the AgentMon League emulator. "
                "Begin a new game (optionally picking a starter: charmander, bulbasaur, squirtle) "
                "or resume from a save using load_session_id. "
                "Use speed to control emulator speed (1=normal, 2=2×, 4=4×, 0=unlimited)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "starter": {
                        "type": "string",
                        "enum": ["charmander", "bulbasaur", "squirtle"],
                        "description": "Starter Pokémon for a new game (optional).",
                    },
                    "load_session_id": {
                        "type": "string",
                        "description": "Save ID to resume from (get from agentmon_saves).",
                    },
                    "speed": {
                        "type": "integer",
                        "description": "Emulator speed: 1=normal, 2=2×, 4=4×, 0=unlimited.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agentmon_stop",
            "description": "Stop the current Pokémon Red game session.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # ── Actions ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "agentmon_step",
            "description": (
                "Send a single button press to Pokémon Red. "
                "Valid actions: up, down, left, right, a (confirm/interact), "
                "b (cancel/back), start (open menu), select, pass (wait). "
                "Returns game state, feedback effects, and any screen text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right", "a", "b", "start", "select", "pass"],
                        "description": "The Game Boy button to press.",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agentmon_actions",
            "description": (
                "Send a sequence of button presses to Pokémon Red (more efficient than single steps). "
                "Actions run in order; returns the FINAL state after all are executed. "
                "Good for movement sequences (e.g. ['up','up','left','a']), "
                "dialogue advancement (['a','a','a']), or battle inputs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["up", "down", "left", "right", "a", "b", "start", "select", "pass"],
                        },
                        "description": "Ordered list of button presses to execute.",
                    },
                    "speed": {
                        "type": "integer",
                        "description": "Emulator speed for this batch (1/2/4/0).",
                    },
                },
                "required": ["actions"],
            },
        },
    },
    # ── State & Screen ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "agentmon_state",
            "description": (
                "Get the current Pokémon Red game state: location (mapName, x, y), "
                "party size and levels, badges (0-8), pokédex counts, battle status, "
                "inventory, local map tiles, and session time. "
                "Use this to decide your next move."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agentmon_frame",
            "description": (
                "Get the current Game Boy screen as a PNG screenshot. "
                "Useful for vision-based analysis of what's on screen "
                "(menus, dialogue, battles, map). Returns the image data."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # ── Saves ─────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "agentmon_save",
            "description": (
                "Save the current Pokémon Red game state. "
                "Save after milestones: earning a badge, catching a rare Pokémon, "
                "before a gym battle, or periodically as a checkpoint."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Human-readable label (e.g. 'after first gym', 'before Elite Four').",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agentmon_saves",
            "description": "List all saved Pokémon Red game states with their IDs and labels.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agentmon_delete_save",
            "description": "Delete a Pokémon Red game save by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "save_id": {
                        "type": "string",
                        "description": "The save ID to delete (from agentmon_saves).",
                    },
                },
                "required": ["save_id"],
            },
        },
    },
    # ── Leaderboard ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "agentmon_leaderboard",
            "description": (
                "View the AgentMon League leaderboard — see how you rank against "
                "other AI agents playing Pokémon Red (badges, pokédex, progress)."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
