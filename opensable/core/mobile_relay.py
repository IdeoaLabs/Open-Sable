"""
Sable Mobile Relay — Secure bridge between the mobile app and the gateway

Architecture overview
─────────────────────

  ┌─────────────────────────────────────────────────────────────────┐
  │  Your PC / VPS                                                  │
  │                                                                 │
  │  ┌──────────────────────┐    Unix socket    ┌───────────────┐  │
  │  │  MobileRelay         │ ◄───────────────► │   Gateway     │  │
  │  │  (this module)       │   /tmp/sable.sock │   (WS server) │  │
  │  └──────────────────────┘                   └───────────────┘  │
  │           │                                                     │
  │    127.0.0.1:7891  (loopback — never 0.0.0.0)                 │
  │           │                                                     │
  │    ┌──────┴────────┐                                           │
  │    │ Tor hidden    │  OR  Tailscale mesh VPN                   │
  │    │ service       │      (recommended for app)                │
  │    └──────┬────────┘                                           │
  └───────────┼──────────────────────────────────────────────────  ┘
              │ .onion address (Tor) or 100.x.x.x (Tailscale)
              ▼
  ┌─────────────────────────┐
  │  Sable Mobile App       │
  │  (Expo / React Native)  │
  │                         │
  │  ① Scan QR code once    │
  │  ② WebSocket connect    │
  │  ③ Send HMAC token      │
  │  ④ Encrypted chat       │
  └─────────────────────────┘

Connection options (choose ONE, ordered by privacy)
────────────────────────────────────────────────────

  OPTION A — Tailscale (RECOMMENDED for most users)
  ─────────────────────────────────────────────────
  • Install Tailscale on both PC and phone (free for personal use)
  • They get private IPs like 100.x.x.x — zero ports open to internet
  • Relay listens on 127.0.0.1 but Tailscale routes the phone's traffic
  • Setup: `tailscale up` → Sable prints its Tailscale IP on startup
  • App connects to: ws://100.x.x.x:7891/mobile?token=<secret>

  OPTION B — Tor Hidden Service (max privacy, no account needed)
  ──────────────────────────────────────────────────────────────
  • Relay listens on 127.0.0.1:7891
  • Tor maps it to a .onion address — only you know the address
  • /etc/tor/torrc:
      HiddenServiceDir /var/lib/tor/sable/
      HiddenServicePort 7891 127.0.0.1:7891
  • `systemctl restart tor` → cat /var/lib/tor/sable/hostname
  • App (using Orbot) connects to: ws://<onion>.onion:7891/mobile?token=<secret>
  • Enable with: MOBILE_RELAY_TOR=true in .env

  OPTION C — WireGuard (nerds / self-hosters)
  ────────────────────────────────────────────
  • Run your own WireGuard server on the VPS
  • Phone gets a VPN IP, connects exactly like Tailscale

Security model
──────────────
  1. HMAC-SHA256 token  — shared secret in .env, rotatable
  2. TLS not needed over Tor/.onion (end-to-end encrypted by Tor)
     For Tailscale, WireGuard layer encrypts all traffic
  3. QR code contains:  server address + HMAC secret (base64 encoded)
  4. If token invalid → connection closed immediately, no response

Protocol
────────
  Same JSON-over-WebSocket protocol as the main Gateway, plus:

  Mobile → Relay:
    {"type": "auth", "token": "<hmac_token>"}   ← must be first frame
    then same as gateway protocol (message, command, etc.)

  Relay → Mobile:
    {"type": "auth.ok", "agent": "Sable", "version": "2.0"}
    {"type": "auth.fail", "reason": "Invalid token"}
    then same as gateway protocol (message.start/chunk/done, etc.)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RELAY_VERSION = "2.0.0"
WS_MAGIC      = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
QR_PATH       = Path.home() / ".opensable" / "mobile_qr.png"
SECRET_PATH   = Path.home() / ".opensable" / "mobile_secret.txt"
TOKEN_TTL     = 300  # HMAC tokens are valid for 5 minutes (rolling window)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_or_create_secret() -> str:
    """Load persisted secret or generate a new one."""
    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SECRET_PATH.exists():
        return SECRET_PATH.read_text().strip()
    secret = secrets.token_hex(32)   # 256-bit
    SECRET_PATH.write_text(secret)
    SECRET_PATH.chmod(0o600)
    logger.info(f"[Relay] Generated new mobile secret → {SECRET_PATH}")
    return secret


def make_token(secret: str) -> str:
    """
    Create a 5-minute rolling HMAC token.
    Time window = floor(unix_ts / TOKEN_TTL) — so both sides agree.
    """
    window = str(int(time.time()) // TOKEN_TTL)
    return hmac.new(secret.encode(), window.encode(), hashlib.sha256).hexdigest()


def verify_token(secret: str, token: str) -> bool:
    """Accept current window and previous window (clock skew tolerance)."""
    now = int(time.time()) // TOKEN_TTL
    for w in [now, now - 1]:
        expected = hmac.new(secret.encode(), str(w).encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, token):
            return True
    return False


def generate_qr(address: str, secret: str) -> str:
    """
    Encode connection info as a URL the mobile app can scan.

    Format:  sable://connect?host=<address>&secret=<secret_b64>
    The mobile app reads this, stores it, and connects automatically.
    """
    secret_b64 = base64.urlsafe_b64encode(secret.encode()).decode()
    url = f"sable://connect?host={address}&secret={secret_b64}"

    try:
        import qrcode  # type: ignore
        img = qrcode.make(url)
        QR_PATH.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(QR_PATH))
        logger.info(f"[Relay] QR code saved → {QR_PATH}")
    except ImportError:
        logger.info("[Relay] Install 'qrcode' to generate a PNG QR code")

    return url


# ─── Minimal WebSocket framing (reused from gateway.py pattern) ───────────────

class _WS:
    """Server-side WebSocket over asyncio streams."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer

    @classmethod
    async def upgrade(cls, reader, writer) -> Optional["_WS"]:
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
        except Exception:
            return None
        lines   = raw.decode(errors="replace").split("\r\n")
        headers = {}
        for ln in lines[1:]:
            if ": " in ln:
                k, v = ln.split(": ", 1)
                headers[k.lower()] = v.strip()
        if "upgrade" not in headers:
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            writer.close()
            return None
        key    = headers.get("sec-websocket-key", "")
        accept = base64.b64encode(
            hashlib.sha1((key + WS_MAGIC).encode()).digest()
        ).decode()
        writer.write((
            f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode())
        await writer.drain()
        return cls(reader, writer)

    async def recv(self) -> Optional[str]:
        while True:
            try:
                hdr = await self.reader.readexactly(2)
            except Exception:
                return None
            opcode = hdr[0] & 0x0F
            masked = bool(hdr[1] & 0x80)
            length = hdr[1] & 0x7F
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self.writer.write(bytes([0x8A, 0]))
                await self.writer.drain()
                continue
            if length == 126:
                length = int.from_bytes(await self.reader.readexactly(2), "big")
            elif length == 127:
                length = int.from_bytes(await self.reader.readexactly(8), "big")
            mask_key = await self.reader.readexactly(4) if masked else b""
            payload  = await self.reader.readexactly(length)
            if masked:
                payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
            return payload.decode("utf-8", errors="replace")

    async def send(self, text: str):
        payload = text.encode("utf-8")
        n = len(payload)
        hdr = bytes([0x81, n]) if n < 126 else (
              bytes([0x81, 126]) + n.to_bytes(2, "big") if n < 65536 else
              bytes([0x81, 127]) + n.to_bytes(8, "big"))
        self.writer.write(hdr + payload)
        await self.writer.drain()

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass


# ─── Gateway bridge (forwards messages to /tmp/sable.sock) ───────────────────

class _GatewayBridge:
    """
    Proxies an authenticated mobile WebSocket connection to the Gateway
    Unix socket.  Each mobile client gets its own gateway connection.
    """

    WS_MAGIC = WS_MAGIC

    def __init__(self, mobile_ws: _WS):
        self.mobile_ws = mobile_ws
        self._gw_reader: Optional[asyncio.StreamReader] = None
        self._gw_writer: Optional[asyncio.StreamWriter] = None
        self._gw_ws:     Optional[_WS] = None

    async def connect_gateway(self) -> bool:
        """Open a connection to the local Unix socket Gateway."""
        from opensable.core.gateway import SOCKET_PATH
        try:
            self._gw_reader, self._gw_writer = await asyncio.open_unix_connection(
                str(SOCKET_PATH)
            )
            # Do client-side WS handshake with the gateway
            key     = base64.b64encode(os.urandom(16)).decode()
            request = (
                f"GET /mobile HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                f"Sec-WebSocket-Version: 13\r\n\r\n"
            )
            self._gw_writer.write(request.encode())
            await self._gw_writer.drain()
            raw = await asyncio.wait_for(
                self._gw_reader.readuntil(b"\r\n\r\n"), timeout=5
            )
            if b"101" not in raw:
                return False
            self._gw_ws = _WS(self._gw_reader, self._gw_writer)
            return True
        except Exception as e:
            logger.warning(f"[Relay] Gateway bridge failed: {e}")
            return False

    async def run(self):
        """
        Bidirectional proxy: mobile ↔ gateway.
        Runs until either side disconnects.
        """
        if not await self.connect_gateway():
            await self.mobile_ws.send(json.dumps({
                "type": "error",
                "text": "Gateway not reachable — is Sable running?"
            }))
            return

        async def mobile_to_gateway():
            while True:
                raw = await self.mobile_ws.recv()
                if raw is None:
                    break
                try:
                    await self._gw_ws.send(raw)
                except Exception:
                    break

        async def gateway_to_mobile():
            while True:
                raw = await self._gw_ws.recv()
                if raw is None:
                    break
                try:
                    await self.mobile_ws.send(raw)
                except Exception:
                    break

        tasks = [
            asyncio.create_task(mobile_to_gateway()),
            asyncio.create_task(gateway_to_mobile()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()

        if self._gw_writer:
            self._gw_writer.close()


# ─── Mobile Relay server ──────────────────────────────────────────────────────

class MobileRelay:
    """
    TCP server on 127.0.0.1:<port> that:
      1. Accepts WebSocket connections (from Tailscale / Tor / WireGuard)
      2. Validates the HMAC token on the first frame
      3. Proxies authenticated connections to the Gateway Unix socket

    Security guarantee: 127.0.0.1 means even a VPS can't be hit from
    the internet.  Only Tailscale / Tor makes it reachable from the phone.
    """

    def __init__(self, config):
        self.config  = config
        self.host    = getattr(config, "mobile_relay_host", "127.0.0.1")
        self.port    = getattr(config, "mobile_relay_port", 7891)
        self.secret  = getattr(config, "mobile_relay_secret", None) or _load_or_create_secret()

        self._server: Optional[asyncio.AbstractServer] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        self._server  = await asyncio.start_server(
            self._on_connect, self.host, self.port
        )
        self._running = True
        self._print_connection_info()
        logger.info(f"[Relay] Mobile relay listening on {self.host}:{self.port}")

    async def stop(self):
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    def _print_connection_info(self):
        """Print startup info so the user knows how to connect."""
        import socket

        # Try to detect Tailscale IP
        tailscale_ip = self._get_tailscale_ip()

        if tailscale_ip:
            address = f"{tailscale_ip}:{self.port}"
            connect_via = f"Tailscale  →  {address}"
        else:
            address = f"localhost:{self.port}"
            connect_via = f"SSH tunnel  →  ssh -L {self.port}:127.0.0.1:{self.port} user@host"

        qr_url = generate_qr(address, self.secret)

        logger.info(
            f"\n"
            f"{'─'*60}\n"
            f"  📱 Sable Mobile Connection\n"
            f"{'─'*60}\n"
            f"  Connect via: {connect_via}\n"
            f"  QR code:     {QR_PATH}\n"
            f"  Manual URL:  {qr_url}\n"
            f"{'─'*60}"
        )

        if getattr(self.config, "mobile_relay_tor_enabled", False):
            logger.info(
                "[Relay] Tor mode enabled — add to /etc/tor/torrc:\n"
                f"  HiddenServiceDir /var/lib/tor/sable/\n"
                f"  HiddenServicePort {self.port} 127.0.0.1:{self.port}\n"
                "then: systemctl restart tor && "
                "cat /var/lib/tor/sable/hostname"
            )

    @staticmethod
    def _get_tailscale_ip() -> Optional[str]:
        """Return the Tailscale IP (100.x.x.x) if available."""
        import subprocess, re
        try:
            result = subprocess.run(
                ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=2
            )
            ip = result.stdout.strip()
            if re.match(r"^100\.\d+\.\d+\.\d+$", ip):
                return ip
        except Exception:
            pass
        return None

    # ── Connection handler ────────────────────────────────────────────────────

    async def _on_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        peer = writer.get_extra_info("peername", "?")
        ws   = await _WS.upgrade(reader, writer)
        if ws is None:
            return

        # ── Step 1: wait for auth frame ────────────────────────────────────────
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
        except asyncio.TimeoutError:
            await ws.send(json.dumps({"type": "auth.fail", "reason": "Timeout"}))
            ws.close()
            return

        if raw is None:
            ws.close()
            return

        try:
            msg   = json.loads(raw)
            token = msg.get("token", "")
        except Exception:
            await ws.send(json.dumps({"type": "auth.fail", "reason": "Bad frame"}))
            ws.close()
            return

        if msg.get("type") != "auth" or not verify_token(self.secret, token):
            logger.warning(f"[Relay] Auth failed from {peer}")
            await ws.send(json.dumps({"type": "auth.fail", "reason": "Invalid token"}))
            ws.close()
            return

        # ── Step 2: authenticated — proxy to gateway ───────────────────────────
        logger.info(f"[Relay] Mobile client authenticated: {peer}")
        await ws.send(json.dumps({
            "type": "auth.ok",
            "agent": "Sable",
            "version": RELAY_VERSION,
        }))

        bridge = _GatewayBridge(ws)
        try:
            await bridge.run()
        finally:
            ws.close()
            logger.info(f"[Relay] Mobile client disconnected: {peer}")
