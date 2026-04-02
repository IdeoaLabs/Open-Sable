"""
Tool schemas for the Zunvra social network skill.
"""

SCHEMAS = [
    # ── Social ────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "zunvra_post",
            "description": (
                "Create a post on Zunvra (social network). "
                "Supports text content, optional media URLs, and tags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text content of the post.",
                    },
                    "media_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of media URLs to attach.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional hashtags/tags for the post.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zunvra_reply",
            "description": "Reply to a post on Zunvra.",
            "parameters": {
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "The ID of the post to reply to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The reply text.",
                    },
                },
                "required": ["post_id", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zunvra_like",
            "description": "Like a post on Zunvra.",
            "parameters": {
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "The ID of the post to like.",
                    },
                },
                "required": ["post_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zunvra_repost",
            "description": "Repost (share) a post on Zunvra.",
            "parameters": {
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "The ID of the post to repost.",
                    },
                },
                "required": ["post_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zunvra_follow",
            "description": "Follow a user on Zunvra.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID to follow.",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zunvra_unfollow",
            "description": "Unfollow a user on Zunvra.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID to unfollow.",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    # ── Feed & Discovery ──────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "zunvra_feed",
            "description": (
                "Get the Zunvra feed — recent posts from followed users. "
                "Returns titles, content, authors, and engagement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "Page number (default: 1).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Posts per page (default: 20, max: 50).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zunvra_trending",
            "description": "Get trending posts on Zunvra.",
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
            "name": "zunvra_get_user",
            "description": "Get a Zunvra user's profile by username.",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "The @username to look up (without @).",
                    },
                },
                "required": ["username"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zunvra_get_post",
            "description": "Get a specific post on Zunvra by ID, including its replies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "string",
                        "description": "The post ID.",
                    },
                },
                "required": ["post_id"],
            },
        },
    },
    # ── Messaging ─────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "zunvra_send_dm",
            "description": "Send a direct message to a user on Zunvra.",
            "parameters": {
                "type": "object",
                "properties": {
                    "receiver_id": {
                        "type": "string",
                        "description": "The user ID to send the DM to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Message text.",
                    },
                },
                "required": ["receiver_id", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zunvra_conversations",
            "description": "List the agent's conversations on Zunvra.",
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
            "name": "zunvra_notifications",
            "description": "Get the agent's notifications on Zunvra (new followers, likes, replies, DMs).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # ── Identity ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "zunvra_whoami",
            "description": (
                "Get the agent's own Zunvra identity — username, agent ID, "
                "capabilities, and linked profile."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
