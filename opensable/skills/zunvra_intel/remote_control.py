"""
Zunvra Central Intelligence — Remote Control Module

Gives Sable direct control over the Central Intelligence/Zunvra dashboard UI.
Sends commands via REST API → backend → WebSocket → frontend → map reacts.

Supports:
  - Camera navigation (flyTo with lat/lng/zoom)
  - Entity selection & tracking
  - Layer & style toggling
  - Highlight markers / attention rings
  - Toast notifications
  - Data filter manipulation
  - Batch command sequences (e.g. fly + select + style in one call)

Usage:
    rc = RemoteControl()
    await rc.fly_to(33.8938, 35.5018, zoom=10)   # Fly to Beirut
    await rc.select_entity("ship", "MMSI-123456") # Open ship panel
    await rc.set_style("NVG")                      # Switch to night vision
    await rc.sequence([                             # Compound action
        ("flyTo", {"lat": 51.5, "lng": -0.12}),
        ("toast", {"message": "Monitoring London", "severity": "info"}),
    ])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Try aiohttp (preferred), fall back to httpx, then urllib
try:
    import aiohttp
    _HTTP_LIB = "aiohttp"
except ImportError:
    try:
        import httpx
        _HTTP_LIB = "httpx"
    except ImportError:
        _HTTP_LIB = "urllib"


class RemoteControl:
    """Sends commands to the Central Intelligence dashboard via the Agent Command Channel.

    Two transport options:
      1. REST POST /api/command (default — simple, stateless)
      2. WebSocket ws://host:port/ws/commands (low-latency, persistent)
    """

    DEFAULT_BASE = "http://localhost:8000"
    VALID_STYLES = {"DEFAULT", "SATELLITE", "NVG", "FLIR", "CRT", "THERMAL",
                    "MIDNIGHT", "RADAR", "TOPO", "DARKOPS"}
    VALID_COMMANDS = {"flyTo", "selectEntity", "trackEntity", "setLayers",
                      "setStyle", "highlight", "toast", "setFilters",
                      "openLiveStream", "addDiaryEntry", "addDossierEntry"}

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        source: str = "sable-agent",
        timeout: float = 10.0,
    ):
        self.base_url = (base_url or self.DEFAULT_BASE).rstrip("/")
        self.api_key = api_key or os.environ.get("ZUNVRA_API_KEY", "").strip()
        self.source = source
        self.timeout = timeout
        self._session = None
        self._command_count = 0
        self._last_error: Optional[str] = None
        if not self.api_key:
            logger.warning("RemoteControl: no API key set — protected endpoints will reject requests")

    # ── Low-level transport ──────────────────────────────────────────

    async def _ensure_session(self):
        _auth_headers = {}
        if self.api_key:
            _auth_headers["Authorization"] = f"Bearer {self.api_key}"
        if _HTTP_LIB == "aiohttp":
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers=_auth_headers,
                )
        elif _HTTP_LIB == "httpx":
            if self._session is None:
                self._session = httpx.AsyncClient(
                    timeout=self.timeout,
                    headers=_auth_headers,
                )

    async def _post(self, path: str, payload: dict) -> dict:
        """POST JSON to the backend and return the response."""
        url = f"{self.base_url}{path}"
        await self._ensure_session()

        try:
            if _HTTP_LIB == "aiohttp":
                async with self._session.post(url, json=payload) as resp:
                    result = await resp.json()
                    return result
            elif _HTTP_LIB == "httpx":
                resp = await self._session.post(url, json=payload)
                return resp.json()
            else:
                # Fallback: synchronous urllib (not ideal but works)
                import urllib.request
                data = json.dumps(payload).encode()
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                req = urllib.request.Request(
                    url, data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            logger.warning("RemoteControl POST failed: %s → %s", url, e)
            return {"status": "error", "message": str(e)}

    async def _send(self, cmd_type: str, payload: dict) -> dict:
        """Send a single command via REST."""
        body = {
            "type": cmd_type,
            "payload": payload,
            "source": self.source,
        }
        result = await self._post("/api/command", body)
        self._command_count += 1
        logger.debug("Command #%d: %s → %s", self._command_count, cmd_type, result.get("status"))
        return result

    # ── High-level actions ───────────────────────────────────────────

    async def fly_to(
        self,
        lat: float,
        lng: float,
        zoom: Optional[float] = None,
        duration: Optional[int] = None,
    ) -> dict:
        """Move the map camera to a specific location.

        Args:
            lat: Latitude (-90 to 90)
            lng: Longitude (-180 to 180)
            zoom: Optional zoom level (1-22, default ~8)
            duration: Optional animation duration in ms
        """
        payload: Dict[str, Any] = {"lat": lat, "lng": lng}
        if zoom is not None:
            payload["zoom"] = zoom
        if duration is not None:
            payload["duration"] = duration
        return await self._send("flyTo", payload)

    async def select_entity(
        self,
        entity_type: str,
        entity_id: Union[str, int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Open the detail panel for a specific entity.

        Args:
            entity_type: Type of entity (e.g. "ship", "flight", "military_flight",
                         "satellite", "earthquake", "fire", "carrier")
            entity_id: The entity's unique identifier (MMSI, hex, NORAD, etc.)
            extra: Additional context data to pass to the detail panel
        """
        payload: Dict[str, Any] = {"type": entity_type, "id": entity_id}
        if extra:
            payload["extra"] = extra
        return await self._send("selectEntity", payload)

    async def track_entity(
        self,
        entity_type: str,
        entity_id: Union[str, int],
    ) -> dict:
        """Start tracking an entity (camera follows it).

        Args:
            entity_type: Type of entity
            entity_id: The entity's unique identifier
        """
        return await self._send("trackEntity", {"type": entity_type, "id": entity_id})

    async def stop_tracking(self) -> dict:
        """Stop tracking any entity."""
        return await self._send("trackEntity", {})

    async def set_layers(self, **layers: bool) -> dict:
        """Toggle data layers on/off.

        Args:
            **layers: Layer name → enabled, e.g.:
                set_layers(ships=True, flights=False, fires=True)

        Known layers (varies by config):
            ships, flights, military_flights, satellites, earthquakes, fires,
            gdelt, cyber, gps_jamming, nuclear, carriers, ais, dark_intel,
            internet_outages, ransomware, weather, notams, radio, sigint,
            highres_satellite, etc.
        """
        return await self._send("setLayers", dict(layers))

    async def set_style(self, style: str) -> dict:
        """Change the map visual style.

        Args:
            style: One of DEFAULT, SATELLITE, NVG, FLIR, CRT, THERMAL,
                   MIDNIGHT, RADAR, TOPO, DARKOPS
        """
        style = style.upper()
        if style not in self.VALID_STYLES:
            logger.warning("Unknown style '%s' — sending anyway", style)
        return await self._send("setStyle", {"style": style})

    async def highlight(
        self,
        lat: float,
        lng: float,
        radius: float = 50_000,
        color: str = "#ff0000",
        label: Optional[str] = None,
        duration: Optional[int] = None,
    ) -> dict:
        """Draw an attention marker/ring on the map.

        Args:
            lat: Center latitude
            lng: Center longitude
            radius: Radius in meters (default 50km)
            color: CSS color string
            label: Optional text label
            duration: How long to show in ms (None = permanent until cleared)
        """
        payload: Dict[str, Any] = {"lat": lat, "lng": lng, "radius": radius, "color": color}
        if label:
            payload["label"] = label
        if duration:
            payload["duration"] = duration
        return await self._send("highlight", payload)

    async def toast(
        self,
        message: str,
        severity: str = "info",
    ) -> dict:
        """Show a notification toast on the dashboard.

        Args:
            message: The notification text
            severity: One of "info", "warning", "error", "success"
        """
        return await self._send("toast", {"message": message, "severity": severity})

    async def set_filters(self, **filters: Any) -> dict:
        """Set data filters on the dashboard.

        Args:
            **filters: Filter name → value(s), e.g.:
                set_filters(country=["US","RU"], altitude_min=30000)
        """
        return await self._send("setFilters", dict(filters))

    async def open_live_stream(
        self,
        name: str,
        url: str,
        lat: float,
        lng: float,
        platform: str = "youtube",
        city: str = "",
        country: str = "",
    ) -> dict:
        """Open a live camera stream on the dashboard.

        Args:
            name: Camera/stream display name
            url: Embeddable URL (YouTube embed, EarthCam, etc.)
            lat: Camera latitude
            lng: Camera longitude
            platform: youtube | earthcam | skyline | windy | other
            city: Optional city name
            country: Optional ISO country code
        """
        return await self._send("openLiveStream", {
            "name": name,
            "url": url,
            "lat": lat,
            "lng": lng,
            "platform": platform,
            "city": city,
            "country": country,
        })

    async def add_diary_entry(
        self,
        title: str,
        content: str,
        severity: str = "info",
        lat: float = 0.0,
        lng: float = 0.0,
        domain: str = "",
        tags: Optional[List[str]] = None,
    ) -> dict:
        """Add an entry to the operation diary on the dashboard.

        Args:
            title: Short heading for the entry
            content: Full analysis text / opinion / observation
            severity: info | warning | critical | success
            lat: Location latitude
            lng: Location longitude
            domain: Intelligence domain
            tags: Optional keywords
        """
        return await self._send("addDiaryEntry", {
            "title": title,
            "content": content,
            "severity": severity,
            "lat": lat,
            "lng": lng,
            "domain": domain,
            "tags": tags or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def push_intel_brief(self, brief: dict) -> dict:
        """Push a full intelligence brief to the dashboard overlay.

        The brief contains threat matrix, wargame scenarios, live intel events,
        and world-state statistics for display in the IntelOverlay panel.

        Args:
            brief: Dict matching the IntelBrief schema with keys:
                timestamp, overallThreat, threatName, cycleNumber,
                domains, wargameScenarios, intelEvents, stats
        """
        return await self._send("pushIntelBrief", brief)

    async def push_intel_event(self, event: dict) -> dict:
        """Push a single intel event to the live feed ticker.

        Args:
            event: Dict with id, timestamp, domain, severity, title,
                   detail, lat, lng
        """
        return await self._send("pushIntelEvent", event)

    async def add_dossier_entry(
        self,
        category: str,
        title: str,
        data: dict,
        severity: str = "info",
        lat: float = 0.0,
        lng: float = 0.0,
    ) -> dict:
        """Add an entry to the operation dossier.

        Args:
            category: cctv | aircraft | ship | attack | cyber | nuclear | custom
            title: Display title
            data: Structured data dict (depends on category)
            severity: info | warning | critical
            lat: Location latitude
            lng: Location longitude
        """
        return await self._send("addDossierEntry", {
            "category": category,
            "title": title,
            "data": data,
            "severity": severity,
            "lat": lat,
            "lng": lng,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ── Compound actions ─────────────────────────────────────────────

    async def sequence(self, commands: List[Tuple[str, dict]]) -> dict:
        """Send multiple commands as an atomic batch.

        Args:
            commands: List of (type, payload) tuples, e.g.:
                [("flyTo", {"lat": 33.9, "lng": 35.5}),
                 ("selectEntity", {"type": "ship", "id": "123"})]
        """
        batch = [
            {"type": cmd_type, "payload": payload, "source": self.source}
            for cmd_type, payload in commands[:20]
        ]
        result = await self._post("/api/commands/batch", batch)
        self._command_count += len(batch)
        return result

    async def fly_and_select(
        self,
        lat: float,
        lng: float,
        entity_type: str,
        entity_id: Union[str, int],
        zoom: Optional[float] = None,
    ) -> dict:
        """Convenience: fly to a location and select an entity in one call."""
        fly_payload: Dict[str, Any] = {"lat": lat, "lng": lng}
        if zoom:
            fly_payload["zoom"] = zoom
        return await self.sequence([
            ("flyTo", fly_payload),
            ("selectEntity", {"type": entity_type, "id": entity_id}),
        ])

    async def surveillance_mode(
        self,
        lat: float,
        lng: float,
        entity_type: str,
        entity_id: Union[str, int],
    ) -> dict:
        """Switch to NVG style, fly to location, and start tracking an entity."""
        return await self.sequence([
            ("setStyle", {"style": "NVG"}),
            ("flyTo", {"lat": lat, "lng": lng, "zoom": 12}),
            ("trackEntity", {"type": entity_type, "id": entity_id}),
            ("toast", {"message": f"Surveillance active: {entity_type} {entity_id}", "severity": "warning"}),
        ])

    # ── Status ───────────────────────────────────────────────────────

    async def get_connected_clients(self) -> int:
        """Check how many frontend dashboards are connected."""
        await self._ensure_session()
        try:
            if _HTTP_LIB == "aiohttp":
                async with self._session.get(f"{self.base_url}/api/commands/clients") as resp:
                    data = await resp.json()
                    return data.get("connected_clients", 0)
            elif _HTTP_LIB == "httpx":
                resp = await self._session.get(f"{self.base_url}/api/commands/clients")
                return resp.json().get("connected_clients", 0)
        except Exception:
            return 0

    async def get_command_log(self, limit: int = 50) -> List[dict]:
        """Get recent command history."""
        await self._ensure_session()
        try:
            if _HTTP_LIB == "aiohttp":
                async with self._session.get(
                    f"{self.base_url}/api/commands/log", params={"limit": limit}
                ) as resp:
                    data = await resp.json()
                    return data.get("log", [])
            elif _HTTP_LIB == "httpx":
                resp = await self._session.get(
                    f"{self.base_url}/api/commands/log", params={"limit": limit}
                )
                return resp.json().get("log", [])
        except Exception:
            return []

    def status(self) -> dict:
        """Return local status summary."""
        return {
            "base_url": self.base_url,
            "source": self.source,
            "commands_sent": self._command_count,
            "last_error": self._last_error,
            "http_library": _HTTP_LIB,
            "heartbeat_active": self._heartbeat_task is not None,
        }

    # ── Heartbeat ────────────────────────────────────────────────────

    _heartbeat_task: Optional[asyncio.Task] = None

    async def start_heartbeat(self, interval: float = 10.0):
        """Start sending periodic heartbeats to keep the operator indicator green.

        Args:
            interval: Seconds between heartbeats (default 10s)
        """
        if self._heartbeat_task and not self._heartbeat_task.done():
            return  # Already running

        async def _heartbeat_loop():
            while True:
                try:
                    await self._post("/api/agent/heartbeat", {"source": self.source})
                except Exception:
                    pass
                await asyncio.sleep(interval)

        self._heartbeat_task = asyncio.create_task(_heartbeat_loop())
        logger.info("Heartbeat started (interval=%ss)", interval)

    async def stop_heartbeat(self):
        """Stop the periodic heartbeat."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.info("Heartbeat stopped")

    async def send_heartbeat(self):
        """Send a single heartbeat (manual)."""
        return await self._post("/api/agent/heartbeat", {"source": self.source})

    # ── Cleanup ──────────────────────────────────────────────────────

    async def close(self):
        """Close the HTTP session and stop heartbeat."""
        await self.stop_heartbeat()
        if self._session:
            if _HTTP_LIB == "aiohttp":
                await self._session.close()
            elif _HTTP_LIB == "httpx":
                await self._session.aclose()
            self._session = None

    def __repr__(self) -> str:
        return f"RemoteControl(base={self.base_url}, sent={self._command_count})"
