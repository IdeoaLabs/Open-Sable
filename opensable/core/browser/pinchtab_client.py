"""
PinchTab client — HTTP adapter for the PinchTab browser control server.

PinchTab provides token-efficient, stealth-capable browser automation via a
lightweight HTTP API.  This module wraps its REST endpoints so BrowserEngine
can transparently delegate to PinchTab when available, falling back to
Playwright when it isn't.

Architecture:
    PinchTab runs two HTTP layers:
      * **Management server** (default :9867) — ``/health``, ``/instances``,
        ``/instances/launch``.  Used to start/stop headless browser instances.
      * **Instance server** (e.g. :9868) — ``/navigate``, ``/text``,
        ``/snapshot``, ``/screenshot``, ``/action``, ``/tabs``, ``/tab``.
        This is the browser-control plane.

Ref: https://github.com/pinchtab/pinchtab
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Management server (launches/manages browser instances)
_DEFAULT_MGMT_URL = "http://127.0.0.1:9867"
# Instance server (browser control) — port assigned at launch
_DEFAULT_INSTANCE_URL = "http://127.0.0.1:9868"
_HEALTH_TIMEOUT = 3  # seconds
_NAV_TIMEOUT = 25  # seconds


def _find_pinchtab_binary() -> Optional[str]:
    """Locate the pinchtab binary (project bin/ first, then PATH)."""
    project_bin = Path(__file__).resolve().parents[3] / "bin" / "pinchtab"
    if project_bin.exists() and os.access(str(project_bin), os.X_OK):
        return str(project_bin)
    return shutil.which("pinchtab")


class PinchTabClient:
    """Async HTTP client for a PinchTab server.

    Lifecycle::

        client = PinchTabClient()
        await client.connect()          # management health + ensure instance
        await client.navigate("https://example.com")
        text = await client.text()      # ~800 tokens
        snap = await client.snapshot()  # accessibility tree
        await client.click("e5")        # click by element ref
        await client.shutdown()
    """

    def __init__(
        self,
        mgmt_url: Optional[str] = None,
        auto_start: bool = True,
    ):
        self._mgmt_url = (
            mgmt_url
            or os.environ.get("PINCHTAB_URL")
            or _DEFAULT_MGMT_URL
        ).rstrip("/")
        self._auto_start = auto_start

        # Internals
        self._process: Optional[subprocess.Popen] = None
        self._connected = False
        self._instance_id: Optional[str] = None
        self._instance_url: Optional[str] = None  # e.g. http://127.0.0.1:9868
        self._default_tab: Optional[str] = None
        self._session = None  # lazy aiohttp.ClientSession

    # ── Connection ────────────────────────────────────────────────────────

    async def _get_session(self):
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def connect(self) -> bool:
        """Connect to PinchTab management server; ensure an instance exists."""
        if self._connected:
            return True

        # 1. Already running?
        if await self._health_check(self._mgmt_url):
            self._connected = True
            await self._ensure_instance()
            logger.info(f"PinchTab connected (mgmt={self._mgmt_url}, instance={self._instance_url})")
            return True

        # 2. Auto-start
        if self._auto_start:
            binary = _find_pinchtab_binary()
            if binary:
                return await self._start_server(binary)

        logger.debug("PinchTab not available")
        return False

    async def _health_check(self, base_url: str) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{base_url}/health", timeout=_HEALTH_TIMEOUT) as r:
                return r.status == 200
        except Exception:
            return False

    async def _start_server(self, binary: str) -> bool:
        logger.info(f"Starting PinchTab server: {binary}")
        try:
            self._process = subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            for _ in range(16):
                await asyncio.sleep(0.5)
                if await self._health_check(self._mgmt_url):
                    self._connected = True
                    await self._ensure_instance()
                    logger.info(f"PinchTab started (PID {self._process.pid})")
                    return True
            logger.warning("PinchTab started but not responding in time")
            self._kill_server()
            return False
        except Exception as e:
            logger.warning(f"Failed to start PinchTab: {e}")
            return False

    async def _ensure_instance(self):
        """Reuse or launch a headless browser instance, resolve its port."""
        try:
            session = await self._get_session()

            # Check existing instances
            async with session.get(f"{self._mgmt_url}/instances") as r:
                if r.status == 200:
                    instances = await r.json()
                    if isinstance(instances, dict):
                        instances = instances.get("instances", [])
                    for inst in instances:
                        status = inst.get("status", "")
                        port = inst.get("port")
                        if status in ("running", "starting") and port:
                            self._instance_id = inst.get("id")
                            self._instance_url = f"http://127.0.0.1:{port}"
                            # Verify instance is actually reachable
                            if await self._health_check(self._instance_url):
                                logger.debug(f"PinchTab: reusing instance {self._instance_id} on :{port}")
                                return
                            else:
                                # Stale instance — stop it
                                logger.warning(f"PinchTab: stale instance {self._instance_id}, stopping...")
                                try:
                                    async with session.post(
                                        f"{self._mgmt_url}/instances/{self._instance_id}/stop"
                                    ) as _:
                                        pass
                                except Exception:
                                    pass
                                await asyncio.sleep(1)

            # Launch new headless instance
            async with session.post(
                f"{self._mgmt_url}/instances/launch",
                json={"name": "sable", "mode": "headless"},
            ) as r:
                if r.status in (200, 201):
                    data = await r.json()
                    self._instance_id = data.get("id")
                    port = data.get("port")
                    self._instance_url = f"http://127.0.0.1:{port}"
                    logger.info(f"PinchTab: launched instance {self._instance_id} on :{port}")
                    # Wait for instance to be ready
                    for _ in range(10):
                        await asyncio.sleep(0.5)
                        if await self._health_check(self._instance_url):
                            return
                    logger.warning("PinchTab instance launched but not reachable")
                else:
                    body = await r.text()
                    logger.warning(f"PinchTab launch failed ({r.status}): {body}")
        except Exception as e:
            logger.debug(f"PinchTab instance setup error: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _inst_get(self, path: str, **kwargs) -> Dict[str, Any]:
        """GET on the instance server."""
        if not self._instance_url:
            return {"success": False, "error": "No PinchTab instance"}
        session = await self._get_session()
        async with session.get(f"{self._instance_url}{path}", **kwargs) as r:
            data = await r.json()
            if r.status == 200:
                return {"success": True, **data}
            return {"success": False, "status": r.status, **data}

    async def _inst_post(self, path: str, body: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """POST on the instance server."""
        if not self._instance_url:
            return {"success": False, "error": "No PinchTab instance"}
        session = await self._get_session()
        async with session.post(f"{self._instance_url}{path}", json=body, **kwargs) as r:
            data = await r.json()
            if r.status == 200:
                return {"success": True, **data}
            return {"success": False, "status": r.status, **data}

    # ── Core API (all hit the instance port) ──────────────────────────────

    async def navigate(self, url: str) -> Dict[str, Any]:
        """Navigate the active tab to *url*. Opens a new tab if needed."""
        try:
            result = await self._inst_post(
                "/navigate", {"url": url}, timeout=_NAV_TIMEOUT,
            )
            if result.get("success"):
                self._default_tab = result.get("tabId")
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def snapshot(self) -> Dict[str, Any]:
        """Accessibility snapshot of the active tab (element refs for click/fill)."""
        try:
            return await self._inst_get("/snapshot")
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def text(self) -> Dict[str, Any]:
        """Token-efficient page text (~800 tokens/page)."""
        try:
            return await self._inst_get("/text")
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def screenshot(self) -> Optional[bytes]:
        """Screenshot of active tab. Returns PNG bytes or None."""
        try:
            result = await self._inst_get("/screenshot")
            b64 = result.get("base64")
            if b64:
                return base64.b64decode(b64)
        except Exception:
            pass
        return None

    async def click(self, ref: str) -> Dict[str, Any]:
        """Click element by ref (from snapshot)."""
        try:
            return await self._inst_post("/action", {"kind": "click", "ref": ref})
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def fill(self, ref: str, value: str) -> Dict[str, Any]:
        """Fill an input field by ref."""
        try:
            return await self._inst_post("/action", {"kind": "fill", "ref": ref, "value": value})
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def press(self, key: str, ref: Optional[str] = None) -> Dict[str, Any]:
        """Press a keyboard key, optionally on a specific element."""
        body: Dict[str, Any] = {"kind": "press", "key": key}
        if ref:
            body["ref"] = ref
        try:
            return await self._inst_post("/action", body)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def type_text(self, text: str, ref: Optional[str] = None) -> Dict[str, Any]:
        """Type text character by character."""
        body: Dict[str, Any] = {"kind": "type", "text": text}
        if ref:
            body["ref"] = ref
        try:
            return await self._inst_post("/action", body)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def scroll(self, direction: str = "down", amount: int = 800) -> Dict[str, Any]:
        """Scroll the page. direction: 'up' or 'down'."""
        body: Dict[str, Any] = {"kind": "scroll"}
        if direction == "up":
            body["y"] = -amount
        try:
            return await self._inst_post("/action", body)
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def hover(self, ref: str) -> Dict[str, Any]:
        """Hover over an element by ref."""
        try:
            return await self._inst_post("/action", {"kind": "hover", "ref": ref})
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def select(self, ref: str, value: str) -> Dict[str, Any]:
        """Select a value from a dropdown by ref."""
        try:
            return await self._inst_post("/action", {"kind": "select", "ref": ref, "value": value})
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def cookies(self) -> Dict[str, Any]:
        """Get cookies for the active tab."""
        try:
            return await self._inst_get("/cookies")
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Tab management ────────────────────────────────────────────────────

    async def list_tabs(self) -> List[Dict[str, Any]]:
        """List open tabs."""
        try:
            result = await self._inst_get("/tabs")
            return result.get("tabs", [])
        except Exception:
            return []

    async def close_tab(self, tab_id: Optional[str] = None) -> bool:
        """Close a tab by ID (default: last opened tab)."""
        tid = tab_id or self._default_tab
        if not tid:
            tabs = await self.list_tabs()
            if not tabs:
                return False
            tid = tabs[-1].get("id")
        try:
            result = await self._inst_post("/tab", {"tabId": tid, "action": "close"})
            if result.get("closed"):
                if tid == self._default_tab:
                    self._default_tab = None
                return True
            return False
        except Exception:
            return False

    async def new_tab(self, url: str = "about:blank") -> Dict[str, Any]:
        """Open a new tab and navigate to url."""
        return await self.navigate(url)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._connected and self._instance_url is not None

    def _kill_server(self):
        if self._process:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            self._process = None

    async def shutdown(self):
        """Close session, optionally kill managed server."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._kill_server()
        self._connected = False
        self._instance_id = None
        self._instance_url = None
        self._default_tab = None
