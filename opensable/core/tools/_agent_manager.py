"""
Agent Manager tools mixin,  sub-agent lifecycle management.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

import aiohttp

logger = logging.getLogger(__name__)


class AgentManagerToolsMixin:
    """Tools for creating, managing, and communicating with sub-agents."""

    async def _agent_create_tool(self, args: Dict[str, Any]) -> str:
        mgr = getattr(self.agent, "_agent_manager", None)
        if not mgr:
            return json.dumps({"success": False, "error": "Agent manager not available"})
        try:
            result = await mgr.create_agent(
                name=args["name"],
                soul=args["soul"],
                tools_mode=args.get("tools_mode", "allowlist"),
                tools=args.get("tools"),
                env_overrides=args.get("env_overrides"),
                auto_start=args.get("auto_start", True),
            )
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _agent_stop_tool(self, args: Dict[str, Any]) -> str:
        mgr = getattr(self.agent, "_agent_manager", None)
        if not mgr:
            return json.dumps({"success": False, "error": "Agent manager not available"})
        try:
            ok = await mgr.stop_child(args["name"])
            return json.dumps({"success": ok, "name": args["name"]})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _agent_start_tool(self, args: Dict[str, Any]) -> str:
        mgr = getattr(self.agent, "_agent_manager", None)
        if not mgr:
            return json.dumps({"success": False, "error": "Agent manager not available"})
        try:
            name = args["name"]
            # Ensure prefixed
            if not name.startswith(f"{mgr.parent}-"):
                name = f"{mgr.parent}-{name}"
            info = await mgr.start_child(name)
            if info:
                return json.dumps({"success": True, "name": name, "pid": info["pid"], "port": info["port"]})
            return json.dumps({"success": False, "error": f"Failed to start {name}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _agent_destroy_tool(self, args: Dict[str, Any]) -> str:
        mgr = getattr(self.agent, "_agent_manager", None)
        if not mgr:
            return json.dumps({"success": False, "error": "Agent manager not available"})
        try:
            result = await mgr.destroy_agent(args["name"])
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _agent_list_tool(self, args: Dict[str, Any]) -> str:
        mgr = getattr(self.agent, "_agent_manager", None)
        if not mgr:
            return json.dumps({"success": False, "error": "Agent manager not available"})
        children = mgr.list_children()
        return json.dumps({"success": True, "parent": mgr.parent, "children": children})

    async def _agent_message_tool(self, args: Dict[str, Any]) -> str:
        """Send a message to a sub-agent via its Unix socket and return the response."""
        mgr = getattr(self.agent, "_agent_manager", None)
        if not mgr:
            return json.dumps({"success": False, "error": "Agent manager not available"})

        name = args["name"]
        message = args["message"]

        # Ensure prefixed
        if not name.startswith(f"{mgr.parent}-"):
            name = f"{mgr.parent}-{name}"

        from pathlib import Path
        socket_path = f"/tmp/sable-{name}.sock"
        if not Path(socket_path).exists():
            return json.dumps({"success": False, "error": f"Agent '{name}' is not running (no socket)"})

        try:
            conn = aiohttp.UnixConnector(path=socket_path)
            async with aiohttp.ClientSession(connector=conn) as session:
                async with session.ws_connect("http://localhost/") as ws:
                    # Send message
                    await ws.send_json({
                        "type": "message",
                        "text": message,
                        "session_id": f"parent-{mgr.parent}",
                        "user_id": mgr.parent,
                    })

                    # Wait for response (message.done)
                    response_text = ""
                    try:
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                if data.get("type") == "message.done":
                                    response_text = data.get("text", "")
                                    break
                                elif data.get("type") == "error":
                                    return json.dumps({
                                        "success": False,
                                        "error": data.get("text", "Unknown error from sub-agent"),
                                    })
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
                    except asyncio.TimeoutError:
                        return json.dumps({"success": False, "error": "Timeout waiting for sub-agent response"})

            return json.dumps({
                "success": True,
                "agent": name,
                "response": response_text,
            })
        except Exception as exc:
            return json.dumps({"success": False, "error": f"Failed to message {name}: {exc}"})
