"""
Dynamic Agent Manager — spawn, stop, and destroy sub-agents at runtime.

Features:
  - Auto-start:   When a parent starts, all ``{parent}-*`` children start too.
  - Auto-port:    Children with ``WEBCHAT_PORT=0`` or no port get a free port.
  - Runtime CRUD: Agents can create / stop / destroy sub-agents via tools.

Children are standard Open-Sable profiles (``agents/{name}/``) launched as
separate OS processes with their own Unix socket and TCP gateway.

Convention: ``agents/nano-sweaters`` is a child of ``agents/nano``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENTS_DIR = _PROJECT_ROOT / "agents"
_TEMPLATE_DIR = _AGENTS_DIR / "_template"


class AgentManager:
    """Manages child agent lifecycle for a parent profile."""

    def __init__(self, parent_profile: str, gateway=None):
        self.parent = parent_profile
        self.gateway = gateway
        # {profile_name: {"proc": Process, "name": str, "pid": int, "port": int}}
        self._children: Dict[str, dict] = {}

    # ── Discovery ─────────────────────────────────────────────────────────

    def _find_children(self) -> List[str]:
        """Return profile names that are children of the current parent.

        Convention: ``agents/{parent}-{child_suffix}/`` is a child.
        """
        prefix = f"{self.parent}-"
        if not _AGENTS_DIR.exists():
            return []
        return sorted(
            d.name
            for d in _AGENTS_DIR.iterdir()
            if d.is_dir()
            and d.name.startswith(prefix)
            and not d.name.startswith("_")
            and (d / "soul.md").exists()          # must have a soul
        )

    # ── Auto-start all children ───────────────────────────────────────────

    async def auto_start_children(self) -> List[dict]:
        """Discover and start all child agents. Returns info about started children."""
        children = self._find_children()
        started = []
        for name in children:
            try:
                info = await self.start_child(name)
                if info:
                    started.append(info)
            except Exception as exc:
                logger.warning(f"[AgentManager] Failed to start child {name}: {exc}")
        return started

    # ── Start a single child ──────────────────────────────────────────────

    async def start_child(self, profile_name: str) -> Optional[dict]:
        """Start a child agent process. Returns info dict or None on failure."""
        if profile_name in self._children:
            info = self._children[profile_name]
            # Check if still running
            if info.get("proc") and info["proc"].returncode is None:
                logger.info(f"[AgentManager] {profile_name} already running (PID {info['pid']})")
                return info
            else:
                # Clean up stale entry
                del self._children[profile_name]

        profile_dir = _AGENTS_DIR / profile_name
        if not profile_dir.exists():
            logger.warning(f"[AgentManager] Profile dir not found: {profile_dir}")
            return None

        # Determine port: read profile.env, if WEBCHAT_PORT is 0/empty/auto → find free port
        port = self._resolve_port(profile_dir)

        # Clean up stale socket
        sock_path = Path(f"/tmp/sable-{profile_name}.sock")
        if sock_path.exists():
            try:
                sock_path.unlink()
            except OSError:
                pass

        # Build env overrides for the child
        child_env = {**os.environ}
        child_env["WEBCHAT_PORT"] = str(port)
        child_env["SABLE_PROFILE"] = profile_name
        child_env["_SABLE_PARENT_PROFILE"] = self.parent

        # Log file for the child (same convention as start.sh)
        log_dir = _PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"sable-{profile_name}.log"

        # Launch as subprocess with output redirected to log file
        log_fh = open(log_file, "a")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "opensable", "--profile", profile_name,
            cwd=str(_PROJECT_ROOT),
            env=child_env,
            stdout=log_fh,
            stderr=log_fh,
            start_new_session=True,       # own process group
        )

        info = {
            "name": profile_name,
            "pid": proc.pid,
            "port": port,
            "proc": proc,
            "log_fh": log_fh,
        }
        self._children[profile_name] = info

        # Wait briefly for socket to appear (agent can take 15-30s to initialize)
        for _ in range(60):  # up to 30 seconds
            if sock_path.exists():
                break
            # Also check if process died
            if proc.returncode is not None:
                logger.warning(f"[AgentManager] {profile_name} exited with code {proc.returncode}")
                log_fh.close()
                return None
            await asyncio.sleep(0.5)

        if sock_path.exists():
            logger.info(f"[AgentManager] ✅ Started {profile_name} (PID {proc.pid}, port {port})")
        else:
            logger.warning(f"[AgentManager] ⚠️  {profile_name} started but socket not yet ready (PID {proc.pid})")

        return info

    # ── Stop a single child ───────────────────────────────────────────────

    async def stop_child(self, profile_name: str) -> bool:
        """Stop a running child agent. Returns True if stopped."""
        info = self._children.get(profile_name)
        if not info:
            # Try to find by PID file
            pid_file = _PROJECT_ROOT / f".sable-{profile_name}.pid"
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, signal.SIGTERM)
                    await asyncio.sleep(2)
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    pid_file.unlink(missing_ok=True)
                    logger.info(f"[AgentManager] Stopped {profile_name} via PID file (PID {pid})")
                    return True
                except Exception:
                    pass
            return False

        proc = info.get("proc")
        if proc and proc.returncode is None:
            try:
                # Kill the process group
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            logger.info(f"[AgentManager] Stopped {profile_name} (PID {info['pid']})")

        # Clean up socket
        sock_path = Path(f"/tmp/sable-{profile_name}.sock")
        sock_path.unlink(missing_ok=True)

        # Clean up PID file
        pid_file = _PROJECT_ROOT / f".sable-{profile_name}.pid"
        pid_file.unlink(missing_ok=True)

        # Close log file handle
        log_fh = info.get("log_fh")
        if log_fh:
            try:
                log_fh.close()
            except Exception:
                pass

        del self._children[profile_name]
        return True

    # ── Stop all children ─────────────────────────────────────────────────

    async def stop_all_children(self):
        """Stop all running child agents (called on parent shutdown)."""
        names = list(self._children.keys())
        for name in names:
            try:
                await self.stop_child(name)
            except Exception as exc:
                logger.warning(f"[AgentManager] Error stopping {name}: {exc}")
        logger.info(f"[AgentManager] All child agents stopped ({len(names)} total)")

    # ── Create a new sub-agent at runtime ─────────────────────────────────

    async def create_agent(
        self,
        name: str,
        soul: str,
        *,
        tools_mode: str = "allowlist",
        tools: Optional[List[str]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        auto_start: bool = True,
    ) -> Dict[str, Any]:
        """Create a new sub-agent profile and optionally start it.

        Args:
            name: Profile name (will be prefixed with ``{parent}-`` if not already).
            soul: The agent's soul.md content.
            tools_mode: "all", "allowlist", or "denylist".
            tools: Tool list for allowlist/denylist.
            env_overrides: Extra env vars for profile.env.
            auto_start: Start the agent immediately after creation.

        Returns:
            Dict with agent info.
        """
        # Ensure name is prefixed
        if not name.startswith(f"{self.parent}-"):
            name = f"{self.parent}-{name}"

        profile_dir = _AGENTS_DIR / name
        if profile_dir.exists():
            return {"success": False, "error": f"Agent '{name}' already exists", "name": name}

        # Create directory structure
        profile_dir.mkdir(parents=True)
        (profile_dir / "data").mkdir(exist_ok=True)

        # Write soul.md
        (profile_dir / "soul.md").write_text(soul.strip() + "\n", encoding="utf-8")

        # Write tools.json
        tools_json = {"mode": tools_mode, "tools": tools or []}
        (profile_dir / "tools.json").write_text(
            json.dumps(tools_json, indent=2) + "\n", encoding="utf-8"
        )

        # Write profile.env (inherit parent's LLM config, set port to auto)
        env_lines = [
            f"# Auto-created sub-agent of {self.parent}",
            f"# Created at runtime by {self.parent}",
            "",
            "# Port: auto-assigned (0 = find free port)",
            "WEBCHAT_PORT=0",
            "",
            "# Interfaces",
            "CLI_ENABLED=false",
            "PIXEL_BRIDGE_ENABLED=true",
            "DESKTOP_ENABLED=false",
            "",
        ]

        # Inherit parent's LLM keys if available
        for key in ("OPENWEBUI_API_KEY", "OPENWEBUI_API_URL", "OPENWEBUI_MODEL",
                     "OLLAMA_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            val = os.environ.get(key, "")
            if val:
                env_lines.append(f"{key}={val}")

        # Add custom overrides
        if env_overrides:
            env_lines.append("")
            env_lines.append("# Custom overrides")
            for k, v in env_overrides.items():
                env_lines.append(f"{k}={v}")

        (profile_dir / "profile.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

        logger.info(f"[AgentManager] Created sub-agent: {name}")

        result: Dict[str, Any] = {
            "success": True,
            "name": name,
            "profile_dir": str(profile_dir),
        }

        if auto_start:
            info = await self.start_child(name)
            if info:
                result["running"] = True
                result["pid"] = info["pid"]
                result["port"] = info["port"]
            else:
                result["running"] = False
                result["error"] = "Created but failed to start"

        return result

    # ── Destroy a sub-agent ───────────────────────────────────────────────

    async def destroy_agent(self, name: str) -> Dict[str, Any]:
        """Stop and remove a sub-agent entirely.

        Args:
            name: Profile name (will be prefixed with ``{parent}-`` if not already).

        Returns:
            Dict with result.
        """
        if not name.startswith(f"{self.parent}-"):
            name = f"{self.parent}-{name}"

        # Safety: cannot destroy parent or non-children
        if name == self.parent:
            return {"success": False, "error": "Cannot destroy self"}

        profile_dir = _AGENTS_DIR / name
        if not profile_dir.exists():
            return {"success": False, "error": f"Agent '{name}' not found"}

        # Stop if running
        await self.stop_child(name)

        # Remove directory
        import shutil
        shutil.rmtree(profile_dir, ignore_errors=True)
        logger.info(f"[AgentManager] Destroyed sub-agent: {name}")

        return {"success": True, "name": name, "destroyed": True}

    # ── List children ─────────────────────────────────────────────────────

    def list_children(self) -> List[Dict[str, Any]]:
        """List all child agents and their status."""
        children = self._find_children()
        result = []
        for name in children:
            sock = Path(f"/tmp/sable-{name}.sock")
            info = self._children.get(name, {})
            result.append({
                "name": name,
                "running": sock.exists(),
                "pid": info.get("pid"),
                "port": info.get("port"),
            })
        return result

    # ── Helper: resolve port ──────────────────────────────────────────────

    @staticmethod
    def _resolve_port(profile_dir: Path) -> int:
        """Read WEBCHAT_PORT from profile.env. If 0/empty/auto → find free port."""
        env_path = profile_dir / "profile.env"
        port_str = "0"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("WEBCHAT_PORT="):
                    port_str = line.split("=", 1)[1].strip()
                    break

        if not port_str or port_str == "0" or port_str.lower() == "auto":
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]
        return int(port_str)
