"""
Sable Node System — Internal IPC (Unix socket, zero TCP ports)

A Node is any script running on the same machine that registers its
capabilities with the Sable Gateway.  Because everything goes through
the Unix socket (/tmp/sable.sock), nodes cannot be reached from the
network — OS file permissions enforce that.

Built-in local node
───────────────────
LocalNode exposes system-level capabilities on the same host:

  system.run      — run a shell command, return stdout/stderr
  system.notify   — send a desktop notification (notify-send / osascript)
  system.info     — return OS/hardware info dict
  fs.read         — read a file (size-limited)
  fs.write        — write a file
  fs.list         — list a directory

Node client SDK
───────────────
GatewayNodeClient lets any Python script register itself as a node:

    from opensable.core.nodes import GatewayNodeClient

    async def handle_invoke(cap, args):
        if cap == "myapp.do_thing":
            return {"result": do_thing(**args)}
        return {"error": "unknown capability"}

    async def main():
        node = GatewayNodeClient(
            node_id="myapp",
            capabilities=["myapp.do_thing"],
            handler=handle_invoke,
        )
        await node.run()   # connects and listens forever

    asyncio.run(main())

The built-in LocalNode is started automatically by __main__.py when
gateway_enabled=True and local_node_enabled=True (default: True).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

SOCKET_PATH = Path("/tmp/sable.sock")
MAX_FILE_READ = 1024 * 512  # 512 KB safety limit for fs.read


# ─── Node Client SDK ──────────────────────────────────────────────────────────


class GatewayNodeClient:
    """
    Connects to the Sable Gateway's Unix socket and registers as a node.

    The handler coroutine is called for every node.invoke received.
    It receives (capability: str, args: dict) and must return a dict.
    """

    WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(
        self,
        node_id: str,
        capabilities: List[str],
        handler: Callable[[str, dict], Coroutine[Any, Any, dict]],
        reconnect_delay: float = 5.0,
    ):
        self.node_id = node_id
        self.capabilities = capabilities
        self.handler = handler
        self.reconnect_delay = reconnect_delay
        self._running = False

    async def run(self):
        """Connect to the gateway and serve requests, reconnecting on drops."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_serve()
            except Exception as e:
                logger.warning(f"[Node:{self.node_id}] disconnected: {e}")
            if self._running:
                logger.info(f"[Node:{self.node_id}] reconnecting in {self.reconnect_delay}s…")
                await asyncio.sleep(self.reconnect_delay)

    async def stop(self):
        self._running = False

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _connect_and_serve(self):
        reader, writer = await asyncio.open_unix_connection(str(SOCKET_PATH))
        ws = await self._handshake(reader, writer)

        # Register
        await ws.send(
            json.dumps(
                {
                    "type": "node.register",
                    "node_id": self.node_id,
                    "capabilities": self.capabilities,
                }
            )
        )

        logger.info(f"[Node:{self.node_id}] registered  caps={self.capabilities}")

        try:
            while self._running:
                raw = await ws.recv()
                if raw is None:
                    break
                msg = json.loads(raw)
                await self._handle(ws, msg)
        finally:
            writer.close()

    async def _handle(self, ws: "_WSClient", msg: dict):
        mtype = msg.get("type", "")

        if mtype == "node.invoke":
            cap = msg.get("capability", "")
            args = msg.get("args", {})
            req_id = msg.get("request_id", "")
            rto = msg.get("reply_to", "")

            try:
                result = await self.handler(cap, args)
            except Exception as e:
                result = {"error": str(e)}

            await ws.send(
                json.dumps(
                    {
                        "type": "node.result",
                        "request_id": req_id,
                        "reply_to": rto,
                        "node_id": self.node_id,
                        "output": result,
                    }
                )
            )

        elif mtype in ("heartbeat", "pong", "node.registered"):
            pass  # silently ignored

        else:
            logger.debug(f"[Node:{self.node_id}] unhandled msg type: {mtype}")

    async def _handshake(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> "_WSClient":
        """Minimal HTTP→WebSocket upgrade as a client."""
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET / HTTP/1.1\r\n"
            f"Host: localhost\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        writer.write(request.encode())
        await writer.drain()

        # Read response headers
        raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
        status_line = raw.decode(errors="replace").split("\r\n")[0]
        if "101" not in status_line:
            raise ConnectionError(f"WebSocket upgrade failed: {status_line}")

        return _WSClient(reader, writer)


class _WSClient:
    """Minimal client-side WebSocket framing (masked frames, per RFC 6455)."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer

    async def recv(self) -> Optional[str]:
        while True:
            try:
                hdr = await self.reader.readexactly(2)
            except Exception:
                return None

            opcode = hdr[0] & 0x0F
            length = hdr[1] & 0x7F

            if opcode == 0x8:
                return None
            if opcode == 0x9:
                await self._pong()
                continue

            if length == 126:
                length = int.from_bytes(await self.reader.readexactly(2), "big")
            elif length == 127:
                length = int.from_bytes(await self.reader.readexactly(8), "big")

            payload = await self.reader.readexactly(length)
            return payload.decode("utf-8", errors="replace")

    async def send(self, text: str):
        """Send a masked text frame (client→server direction requires masking)."""
        payload = text.encode("utf-8")
        mask_key = os.urandom(4)
        masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        n = len(payload)
        if n < 126:
            hdr = bytes([0x81, 0x80 | n])
        elif n < 65536:
            hdr = bytes([0x81, 0xFE]) + n.to_bytes(2, "big")
        else:
            hdr = bytes([0x81, 0xFF]) + n.to_bytes(8, "big")
        self.writer.write(hdr + mask_key + masked)
        await self.writer.drain()

    async def _pong(self):
        self.writer.write(bytes([0x8A, 0x80]) + os.urandom(4))
        await self.writer.drain()


# ─── Built-in Local Node ──────────────────────────────────────────────────────


class LocalNode:
    """
    Built-in node that runs on the same host as the Sable Gateway.

    Capabilities:
      system.run      Run a shell command (whitelist-enforced)
      system.notify   Desktop notification
      system.info     OS/hardware information
      fs.read         Read file content (512 KB limit)
      fs.write        Write file content
      fs.list         List directory entries
    """

    # Only these shell commands may be executed via system.run.
    # Add more carefully — this is a security boundary.
    ALLOWED_COMMANDS = {
        "ls",
        "pwd",
        "echo",
        "date",
        "whoami",
        "uname",
        "df",
        "du",
        "free",
        "uptime",
        "ps",
        "top",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "wc",
        "sort",
        "uniq",
        "cut",
        "awk",
        "sed",
        "tr",
        "curl",
        "wget",
        "ping",
        "netstat",
        "ss",
        "ip",
        "python3",
        "python",
        "pip",
        "pip3",
        "git",
        "make",
        "cmake",
        "systemctl",
        "journalctl",
    }

    CAPABILITIES = [
        "system.run",
        "system.notify",
        "system.info",
        "fs.read",
        "fs.write",
        "fs.list",
    ]

    def __init__(self, config=None):
        self.config = config
        self._client = GatewayNodeClient(
            node_id="local",
            capabilities=self.CAPABILITIES,
            handler=self._handle,
        )

    async def start(self):
        """Start the local node in a background task."""
        asyncio.create_task(self._client.run())
        logger.info("[LocalNode] Started")

    async def stop(self):
        await self._client.stop()

    # ── Capability handlers ───────────────────────────────────────────────────

    async def _handle(self, capability: str, args: dict) -> dict:
        if capability == "system.run":
            return await self._system_run(args)
        elif capability == "system.notify":
            return self._system_notify(args)
        elif capability == "system.info":
            return self._system_info()
        elif capability == "fs.read":
            return self._fs_read(args)
        elif capability == "fs.write":
            return self._fs_write(args)
        elif capability == "fs.list":
            return self._fs_list(args)
        else:
            return {"error": f"Unknown capability: {capability}"}

    async def _system_run(self, args: dict) -> dict:
        """Run a whitelisted shell command and return stdout/stderr."""
        cmd = args.get("command", "").strip()
        if not cmd:
            return {"error": "No command specified"}

        # Security: only allow whitelisted base commands
        base = cmd.split()[0].lstrip("./")
        if base not in self.ALLOWED_COMMANDS:
            return {"error": f"Command '{base}' is not in the allowed list"}

        timeout = min(int(args.get("timeout", 30)), 120)
        cwd = args.get("cwd") or str(Path.home())

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
                "returncode": proc.returncode,
            }
        except asyncio.TimeoutError:
            return {"error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    def _system_notify(self, args: dict) -> dict:
        """Send a desktop notification."""
        title = args.get("title", "Sable")
        message = args.get("message", "")
        try:
            if shutil.which("notify-send"):  # Linux
                subprocess.Popen(["notify-send", title, message])
            elif sys.platform == "darwin":  # macOS
                script = f'display notification "{message}" with title "{title}"'
                subprocess.Popen(["osascript", "-e", script])
            else:
                return {"error": "No notification backend found"}
            return {"sent": True}
        except Exception as e:
            return {"error": str(e)}

    def _system_info(self) -> dict:
        """Return OS/hardware information."""
        info: Dict[str, Any] = {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
            "hostname": platform.node(),
            "cwd": str(Path.cwd()),
            "home": str(Path.home()),
        }
        try:
            import psutil

            info["cpu_count"] = psutil.cpu_count()
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            info["mem_total_gb"] = round(mem.total / 1e9, 1)
            info["mem_used_pct"] = mem.percent
            disk = psutil.disk_usage("/")
            info["disk_total_gb"] = round(disk.total / 1e9, 1)
            info["disk_used_pct"] = disk.percent
        except ImportError:
            pass
        return info

    def _fs_read(self, args: dict) -> dict:
        """Read a file and return its content (512 KB limit)."""
        path = Path(args.get("path", "")).expanduser()
        if not path.exists():
            return {"error": f"File not found: {path}"}
        if path.stat().st_size > MAX_FILE_READ:
            return {"error": f"File too large (max {MAX_FILE_READ // 1024} KB)"}
        try:
            return {"content": path.read_text(errors="replace"), "path": str(path)}
        except Exception as e:
            return {"error": str(e)}

    def _fs_write(self, args: dict) -> dict:
        """Write content to a file."""
        path = Path(args.get("path", "")).expanduser()
        content = args.get("content", "")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return {"written": True, "path": str(path), "bytes": len(content)}
        except Exception as e:
            return {"error": str(e)}

    def _fs_list(self, args: dict) -> dict:
        """List directory entries."""
        path = Path(args.get("path", ".")).expanduser()
        if not path.is_dir():
            return {"error": f"Not a directory: {path}"}
        try:
            entries = []
            for entry in sorted(path.iterdir()):
                entries.append(
                    {
                        "name": entry.name,
                        "type": "dir" if entry.is_dir() else "file",
                        "size": entry.stat().st_size if entry.is_file() else None,
                    }
                )
            return {"path": str(path), "entries": entries}
        except Exception as e:
            return {"error": str(e)}
