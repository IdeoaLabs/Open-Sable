"""
Zunvra Central Intelligence — API Connector

Persistent async client for all Zunvra/Central Intelligence backend endpoints.
Handles GZip, ETag caching, SSE streaming, and automatic reconnection.
Every other module in this skill package depends on this connector.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data snapshot that flows through the rest of the skill
# ---------------------------------------------------------------------------

@dataclass
class IntelSnapshot:
    """Single point-in-time capture of the full Zunvra data payload.

    Every field served by /api/live-data/fast and /api/live-data/slow is
    typed here so analysis modules can access them without digging into `raw`.
    """

    timestamp: str = ""

    # ── Position / fast-refresh layers ─────────────────────────────
    flights: List[Dict[str, Any]] = field(default_factory=list)
    military_flights: List[Dict[str, Any]] = field(default_factory=list)
    tracked_flights: List[Dict[str, Any]] = field(default_factory=list)
    ships: List[Dict[str, Any]] = field(default_factory=list)
    satellites: List[Dict[str, Any]] = field(default_factory=list)
    uavs: List[Dict[str, Any]] = field(default_factory=list)
    liveuamap: List[Dict[str, Any]] = field(default_factory=list)
    gps_jamming: List[Dict[str, Any]] = field(default_factory=list)
    gtfs_vehicles: Dict[str, Any] = field(default_factory=dict)
    cctv: List[Dict[str, Any]] = field(default_factory=list)

    # ── Context / slow-refresh layers ──────────────────────────────
    earthquakes: List[Dict[str, Any]] = field(default_factory=list)
    fires: List[Dict[str, Any]] = field(default_factory=list)
    gdelt_events: List[Dict[str, Any]] = field(default_factory=list)
    cyber_threats: List[Dict[str, Any]] = field(default_factory=list)
    carriers: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    dark_intel: Dict[str, Any] = field(default_factory=dict)
    internet_outages: List[Dict[str, Any]] = field(default_factory=list)
    nuclear_facilities: List[Dict[str, Any]] = field(default_factory=list)
    ransomware: List[Dict[str, Any]] = field(default_factory=list)
    prediction_markets: List[Dict[str, Any]] = field(default_factory=list)
    news_feed: List[Dict[str, Any]] = field(default_factory=list)
    economics: Dict[str, Any] = field(default_factory=dict)

    # ── World Monitor layers ───────────────────────────────────────
    health_security: Dict[str, Any] = field(default_factory=dict)
    environmental: Dict[str, Any] = field(default_factory=dict)
    infrastructure: Dict[str, Any] = field(default_factory=dict)
    maritime_intel: Dict[str, Any] = field(default_factory=dict)
    trending: List[Dict[str, Any]] = field(default_factory=list)

    # ── Weather / environmental ────────────────────────────────────
    weather: Dict[str, Any] = field(default_factory=dict)
    space_weather: Dict[str, Any] = field(default_factory=dict)
    air_quality: List[Dict[str, Any]] = field(default_factory=list)
    carbon_intensity: Dict[str, Any] = field(default_factory=dict)
    noaa_alerts: List[Dict[str, Any]] = field(default_factory=list)
    aviation_weather: List[Dict[str, Any]] = field(default_factory=list)

    # ── Markets / economics ────────────────────────────────────────
    stocks: Dict[str, Any] = field(default_factory=dict)
    oil: Dict[str, Any] = field(default_factory=dict)
    exchange_rates: Dict[str, Any] = field(default_factory=dict)

    # ── Traffic & transport ────────────────────────────────────────
    traffic: List[Dict[str, Any]] = field(default_factory=list)
    traffic_flow: List[Dict[str, Any]] = field(default_factory=list)
    airports: List[Dict[str, Any]] = field(default_factory=list)
    citybikes: List[Dict[str, Any]] = field(default_factory=list)

    # ── Infrastructure / telecom ───────────────────────────────────
    datacenters: List[Dict[str, Any]] = field(default_factory=list)
    cable_landings: List[Dict[str, Any]] = field(default_factory=list)
    kiwisdr: List[Dict[str, Any]] = field(default_factory=list)

    # ── Military / conflict ────────────────────────────────────────
    frontlines: Any = field(default=None)
    emergency_squawks: List[Dict[str, Any]] = field(default_factory=list)
    notams_tfr: List[Dict[str, Any]] = field(default_factory=list)

    # ── Dark intel / OSINT ─────────────────────────────────────────
    tor_exit_nodes: List[Dict[str, Any]] = field(default_factory=list)
    sanctions: List[Dict[str, Any]] = field(default_factory=list)
    piracy: List[Dict[str, Any]] = field(default_factory=list)
    interpol_notices: List[Dict[str, Any]] = field(default_factory=list)
    radiation_monitors: List[Dict[str, Any]] = field(default_factory=list)
    fbi_wanted: List[Dict[str, Any]] = field(default_factory=list)
    uk_crimes: List[Dict[str, Any]] = field(default_factory=list)

    # ── Cameras (DOT / open) ───────────────────────────────────────
    open_cameras: List[Dict[str, Any]] = field(default_factory=list)

    # ── Cyber / threat intel ───────────────────────────────────────
    threatfox_iocs: List[Dict[str, Any]] = field(default_factory=list)
    cisa_kev: List[Dict[str, Any]] = field(default_factory=list)
    phishing_sites: List[Dict[str, Any]] = field(default_factory=list)
    dshield_top_ips: List[Dict[str, Any]] = field(default_factory=list)
    recent_cves: List[Dict[str, Any]] = field(default_factory=list)
    urlhaus_active: List[Dict[str, Any]] = field(default_factory=list)
    sslbl_botnet: List[Dict[str, Any]] = field(default_factory=list)

    # ── Disaster / humanitarian ────────────────────────────────────
    gdacs_disasters: List[Dict[str, Any]] = field(default_factory=list)
    reliefweb_crises: List[Dict[str, Any]] = field(default_factory=list)
    eonet_events: List[Dict[str, Any]] = field(default_factory=list)

    # ── Space ──────────────────────────────────────────────────────
    space_launches: List[Dict[str, Any]] = field(default_factory=list)
    spaceflight_news: List[Dict[str, Any]] = field(default_factory=list)
    iss_position: Dict[str, Any] = field(default_factory=dict)

    # ── Risk / geopolitical ────────────────────────────────────────
    country_risk: List[Dict[str, Any]] = field(default_factory=list)
    usgs_earthquakes: List[Dict[str, Any]] = field(default_factory=list)

    # ── Raw payload (always kept for fallback) ─────────────────────
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, data: Dict[str, Any]) -> "IntelSnapshot":
        """Parse the raw /api/live-data/* JSON into a structured snapshot.

        Handles real Central Intelligence data shapes:
         - /fast: commercial_flights, military_flights, ships, gps_jamming ...
         - /slow: news, cyber_threats, conflicts={acled,ucdp}, economics, ...
         - Merged payload: superset of both.
        """
        now = datetime.now(timezone.utc).isoformat()

        # ── Flights ────────────────────────────────────────────────
        flights = data.get("flights", [])
        if not flights:
            flights = data.get("commercial_flights", [])
            flights += data.get("private_flights", [])
            flights += data.get("private_jets", [])

        # ── Military flights ───────────────────────────────────────
        military = data.get("military", data.get("military_flights", []))

        # ── Ships ──────────────────────────────────────────────────
        ships = data.get("ships", data.get("vessels", []))
        # Supplement from maritime_intel if present
        mi = data.get("maritime_intel", {})
        if isinstance(mi, dict):
            usni = mi.get("usni_fleet", {})
            if isinstance(usni, dict):
                usni_vessels = usni.get("vessels", [])
                if usni_vessels and isinstance(usni_vessels, list):
                    ships = ships + [v for v in usni_vessels if isinstance(v, dict)]

        # ── Conflicts (flatten nested {acled: [], ucdp: []}) ──────
        raw_conflicts = data.get("conflicts", data.get("acled", []))
        if isinstance(raw_conflicts, dict):
            # Extract and flatten sub-lists
            flat_conflicts: list = []
            for key in ("acled", "ucdp", "events", "data"):
                sub = raw_conflicts.get(key, [])
                if isinstance(sub, list):
                    flat_conflicts.extend(
                        item for item in sub if isinstance(item, dict)
                    )
            raw_conflicts = flat_conflicts
        elif isinstance(raw_conflicts, list):
            raw_conflicts = [c for c in raw_conflicts if isinstance(c, dict)]
        else:
            raw_conflicts = []

        # ── Carriers ───────────────────────────────────────────────
        carriers = data.get("carriers", [])
        if not carriers and isinstance(mi, dict):
            usni = mi.get("usni_fleet", {})
            if isinstance(usni, dict):
                # Try to extract carrier entries from USNI data
                for v in usni.get("vessels", []):
                    if isinstance(v, dict) and any(
                        kw in str(v.get("type", "")).lower()
                        for kw in ("carrier", "cvn", "cv-", "aircraft")
                    ):
                        carriers.append(v)

        # ── Dark Intel aggregate ───────────────────────────────────
        dark_intel = data.get("dark_intel", {})
        if not dark_intel:
            dark_intel = {
                "tor_exit_nodes": data.get("tor_exit_nodes", []),
                "sanctions": data.get("sanctions", []),
                "piracy": data.get("piracy", []),
                "interpol_notices": data.get("interpol_notices", []),
            }

        # ── Fires ─────────────────────────────────────────────────
        fires = data.get("fires", [])
        if not fires:
            fires = data.get("firms_fires", [])
        fires = [f for f in fires if isinstance(f, dict)]

        # ── Earthquakes (merge both sources) ───────────────────────
        eq = data.get("earthquakes", [])
        usgs = data.get("usgs_earthquakes", [])
        if usgs and isinstance(usgs, list):
            # Merge by id to avoid duplicates
            eq_ids = {e.get("id") for e in eq if isinstance(e, dict)}
            for u in usgs:
                if isinstance(u, dict) and u.get("id") not in eq_ids:
                    eq.append(u)
        earthquakes = [e for e in eq if isinstance(e, dict)]

        # ── GPS Jamming ────────────────────────────────────────────
        gps_jamming = data.get("gps_jamming", data.get("jamming_zones", []))
        if not isinstance(gps_jamming, list):
            gps_jamming = []

        # Helper — safely extract a list or dict, falling back to default
        def _lst(key: str, alt: str = "", default=None):
            """Get a list field, trying key then alt."""
            val = data.get(key, data.get(alt, [] if default is None else default))
            if default is None:
                return val if isinstance(val, list) else []
            return val

        def _dct(key: str, alt: str = ""):
            val = data.get(key, data.get(alt, {}))
            return val if isinstance(val, dict) else {}

        return cls(
            timestamp=now,
            # Position / fast
            flights=flights,
            military_flights=military if isinstance(military, list) else [],
            tracked_flights=_lst("tracked_flights"),
            ships=ships if isinstance(ships, list) else [],
            satellites=_lst("satellites"),
            uavs=_lst("uavs"),
            liveuamap=_lst("liveuamap"),
            gps_jamming=gps_jamming,
            gtfs_vehicles=_dct("gtfs_vehicles"),
            cctv=_lst("cctv"),
            # Context / slow
            earthquakes=earthquakes,
            fires=fires,
            gdelt_events=_lst("gdelt", "gdelt_events"),
            cyber_threats=_lst("cyber_threats", "cyber"),
            carriers=carriers if isinstance(carriers, list) else [],
            conflicts=raw_conflicts,
            dark_intel=dark_intel if isinstance(dark_intel, dict) else {},
            internet_outages=_lst("internet_outages", "outages"),
            nuclear_facilities=_lst("nuclear_facilities"),
            ransomware=_lst("ransomware"),
            prediction_markets=_lst("prediction_markets"),
            news_feed=_lst("news", "news_feed"),
            economics=_dct("economics"),
            # World Monitor
            health_security=_dct("health_security"),
            environmental=_dct("environmental"),
            infrastructure=_dct("infrastructure"),
            maritime_intel=_dct("maritime_intel"),
            trending=_lst("trending"),
            # Weather / environmental
            weather=_dct("weather"),
            space_weather=_dct("space_weather"),
            air_quality=_lst("air_quality"),
            carbon_intensity=_dct("carbon_intensity"),
            noaa_alerts=_lst("noaa_alerts"),
            aviation_weather=_lst("aviation_weather"),
            # Markets
            stocks=_dct("stocks"),
            oil=_dct("oil"),
            exchange_rates=_dct("exchange_rates"),
            # Traffic & transport
            traffic=_lst("traffic"),
            traffic_flow=_lst("traffic_flow"),
            airports=_lst("airports"),
            citybikes=_lst("citybikes"),
            # Infrastructure / telecom
            datacenters=_lst("datacenters"),
            cable_landings=_lst("cable_landings"),
            kiwisdr=_lst("kiwisdr"),
            # Military / conflict
            frontlines=data.get("frontlines"),
            emergency_squawks=_lst("emergency_squawks"),
            notams_tfr=_lst("notams_tfr"),
            # Dark intel / OSINT
            tor_exit_nodes=_lst("tor_exit_nodes"),
            sanctions=_lst("sanctions"),
            piracy=_lst("piracy"),
            interpol_notices=_lst("interpol_notices"),
            radiation_monitors=_lst("radiation_monitors"),
            fbi_wanted=_lst("fbi_wanted"),
            uk_crimes=_lst("uk_crimes"),
            # Cameras (DOT / open)
            open_cameras=_lst("open_cameras"),
            # Cyber / threat intel
            threatfox_iocs=_lst("threatfox_iocs"),
            cisa_kev=_lst("cisa_kev"),
            phishing_sites=_lst("phishing_sites"),
            dshield_top_ips=_lst("dshield_top_ips"),
            recent_cves=_lst("recent_cves"),
            urlhaus_active=_lst("urlhaus_active"),
            sslbl_botnet=_lst("sslbl_botnet"),
            # Disaster / humanitarian
            gdacs_disasters=_lst("gdacs_disasters"),
            reliefweb_crises=_lst("reliefweb_crises"),
            eonet_events=_lst("eonet_events"),
            # Space
            space_launches=_lst("space_launches"),
            spaceflight_news=_lst("spaceflight_news"),
            iss_position=_dct("iss_position"),
            # Risk / geopolitical
            country_risk=_lst("country_risk"),
            usgs_earthquakes=_lst("usgs_earthquakes"),
            # Raw
            raw=data,
        )

    @property
    def total_entities(self) -> int:
        count = 0
        for attr_name in (
            "flights", "military_flights", "tracked_flights",
            "ships", "satellites", "uavs", "liveuamap",
            "earthquakes", "fires", "gdelt_events", "cyber_threats",
            "gps_jamming", "carriers", "conflicts",
            "internet_outages", "nuclear_facilities", "ransomware",
            "prediction_markets", "news_feed", "cctv",
            "trending", "air_quality", "noaa_alerts", "aviation_weather",
            "traffic", "traffic_flow", "airports", "citybikes",
            "datacenters", "cable_landings", "kiwisdr",
            "emergency_squawks", "notams_tfr",
            "tor_exit_nodes", "sanctions", "piracy", "interpol_notices",
            "radiation_monitors", "fbi_wanted", "uk_crimes",
            "open_cameras",
            "threatfox_iocs", "cisa_kev", "phishing_sites",
            "dshield_top_ips", "recent_cves", "urlhaus_active", "sslbl_botnet",
            "gdacs_disasters", "reliefweb_crises", "eonet_events",
            "space_launches", "spaceflight_news", "country_risk",
            "usgs_earthquakes",
        ):
            val = getattr(self, attr_name, [])
            if isinstance(val, list):
                count += len(val)
        return count

    def entities_near(self, lat: float, lon: float, radius_km: float = 50.0) -> Dict[str, List[Dict]]:
        """Return all entities within *radius_km* of (lat, lon)."""
        import math
        results: Dict[str, List[Dict]] = {}
        # All list-type fields that may contain geo-located entities
        _GEO_FIELDS = [
            "flights", "military_flights", "tracked_flights",
            "ships", "satellites", "uavs", "liveuamap",
            "earthquakes", "fires", "gdelt_events", "cyber_threats",
            "gps_jamming", "cctv", "open_cameras",
            "conflicts", "carriers", "internet_outages",
            "nuclear_facilities", "ransomware",
            "traffic", "traffic_flow", "airports",
            "datacenters", "cable_landings", "kiwisdr",
            "emergency_squawks", "notams_tfr",
            "tor_exit_nodes", "piracy", "radiation_monitors",
            "fbi_wanted", "uk_crimes", "air_quality",
            "noaa_alerts", "gdacs_disasters", "eonet_events",
            "citybikes", "interpol_notices",
        ]
        for label in _GEO_FIELDS:
            items = getattr(self, label, [])
            if not isinstance(items, list):
                continue
            near = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                ilat = item.get("lat") or item.get("latitude") or 0
                ilon = item.get("lon") or item.get("longitude") or 0
                try:
                    d = _haversine(lat, lon, float(ilat), float(ilon))
                    if d <= radius_km:
                        near.append(item)
                except (ValueError, TypeError):
                    continue
            if near:
                results[label] = near
        return results

    def summary_text(self, max_chars: int = 3000) -> str:
        """Human-readable summary for LLM consumption."""
        lines = [f"Zunvra Intel Snapshot @ {self.timestamp}"]
        lines.append(f"  Flights: {len(self.flights)} | Military: {len(self.military_flights)} | Tracked: {len(self.tracked_flights)}")
        lines.append(f"  Ships: {len(self.ships)} | Satellites: {len(self.satellites)} | UAVs: {len(self.uavs)}")
        lines.append(f"  Earthquakes: {len(self.earthquakes)} | USGS: {len(self.usgs_earthquakes)} | Fires: {len(self.fires)}")
        lines.append(f"  GDELT events: {len(self.gdelt_events)} | Cyber threats: {len(self.cyber_threats)}")
        lines.append(f"  GPS jamming zones: {len(self.gps_jamming)} | Carriers: {len(self.carriers)}")
        lines.append(f"  Conflicts: {len(self.conflicts)} | Internet outages: {len(self.internet_outages)}")
        lines.append(f"  Ransomware: {len(self.ransomware)} | Nuclear facilities: {len(self.nuclear_facilities)}")
        lines.append(f"  CCTV cameras: {len(self.cctv)} | Open cameras (DOT): {len(self.open_cameras)}")
        lines.append(f"  LiveUAMap: {len(self.liveuamap)} | Emergency squawks: {len(self.emergency_squawks)}")
        lines.append(f"  Airports: {len(self.airports)} | NOTAMs/TFR: {len(self.notams_tfr)}")
        lines.append(f"  Radiation monitors: {len(self.radiation_monitors)} | Cable landings: {len(self.cable_landings)}")
        lines.append(f"  Traffic: {len(self.traffic)} | Traffic flow: {len(self.traffic_flow)}")
        lines.append(f"  Datacenters: {len(self.datacenters)} | KiwiSDR: {len(self.kiwisdr)}")
        lines.append(f"  Tor exit nodes: {len(self.tor_exit_nodes)} | Piracy: {len(self.piracy)}")
        lines.append(f"  FBI wanted: {len(self.fbi_wanted)} | UK crimes: {len(self.uk_crimes)}")
        lines.append(f"  GDACS disasters: {len(self.gdacs_disasters)} | EONET events: {len(self.eonet_events)}")
        lines.append(f"  ThreatFox IOCs: {len(self.threatfox_iocs)} | CISA KEV: {len(self.cisa_kev)}")
        lines.append(f"  Phishing sites: {len(self.phishing_sites)} | Recent CVEs: {len(self.recent_cves)}")
        lines.append(f"  Space launches: {len(self.space_launches)} | Country risk: {len(self.country_risk)}")
        lines.append(f"  News: {len(self.news_feed)} | Trending: {len(self.trending)}")
        lines.append(f"  TOTAL entities: {self.total_entities}")
        text = "\n".join(lines)
        return text[:max_chars]


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class ZunvraConnector:
    """
    Async HTTP client for the Zunvra Central Intelligence backend.

    Supports:
      - Full snapshot via /api/live-data/slow
      - Fast positions via /api/live-data/fast
      - SSE progressive stream via /api/live-data/stream
      - Individual domain endpoints (carriers, dark-intel, etc.)
      - ETag caching (304 not modified)
      - Automatic retry with exponential back-off
    """

    DEFAULT_BASE = "http://localhost:8000"

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        snapshot_dir: Optional[Path] = None,
    ):
        self.base_url = (base_url or self.DEFAULT_BASE).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._session = None
        self._etags: Dict[str, str] = {}
        self._last_snapshot: Optional[IntelSnapshot] = None
        self._connected = False
        self._total_requests = 0
        self._total_bytes = 0

    # ── lifecycle ─────────────────────────────────────────────────────

    async def connect(self):
        """Create the aiohttp session."""
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": "OpenSable-ZunvraIntel/1.0"},
            )
            self._connected = True
            logger.info("ZunvraConnector connected → %s", self.base_url)
        except Exception as e:
            logger.error("ZunvraConnector failed to connect: %s", e)
            self._connected = False

    async def disconnect(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None and not self._session.closed

    # ── core fetchers ─────────────────────────────────────────────────

    async def _get(self, path: str, **kwargs) -> Optional[Dict[str, Any]]:
        """GET with ETag caching and retry."""
        if not self.is_connected:
            await self.connect()
        url = f"{self.base_url}{path}"
        headers = {}
        if path in self._etags:
            headers["If-None-Match"] = self._etags[path]

        for attempt in range(self.max_retries):
            try:
                assert self._session is not None
                async with self._session.get(url, headers=headers, **kwargs) as resp:
                    self._total_requests += 1
                    if resp.status == 304:
                        return None  # unchanged
                    if resp.status == 200:
                        etag = resp.headers.get("ETag")
                        if etag:
                            self._etags[path] = etag
                        body = await resp.read()
                        self._total_bytes += len(body)
                        return json.loads(body)
                    logger.warning("Zunvra %s returned %s", path, resp.status)
            except Exception as e:
                wait = 2 ** attempt
                logger.warning("Zunvra request %s failed (attempt %d): %s — retry in %ds",
                               path, attempt + 1, e, wait)
                await asyncio.sleep(wait)
        return None

    async def fetch_full(self) -> Optional[IntelSnapshot]:
        """Fetch the complete data payload by merging /fast + /slow endpoints.

        /fast has positional data (flights, military, ships, GPS jamming).
        /slow has context data (news, cyber, economics, conflicts, etc.).
        """
        # Fetch both in parallel
        fast_data, slow_data = await asyncio.gather(
            self._get("/api/live-data/fast"),
            self._get("/api/live-data/slow"),
        )

        # Build merged payload
        merged: Dict[str, Any] = {}
        if slow_data:
            merged.update(slow_data)
        if fast_data:
            merged.update(fast_data)

        if not merged:
            if self._last_snapshot:
                return self._last_snapshot  # Use cache
            return None

        snap = IntelSnapshot.from_payload(merged)
        self._last_snapshot = snap
        if self.snapshot_dir:
            self._persist_snapshot(snap)
        return snap

    async def fetch_fast(self) -> Optional[IntelSnapshot]:
        """Fetch positions-only payload (/api/live-data/fast)."""
        data = await self._get("/api/live-data/fast")
        if data is None and self._last_snapshot:
            return self._last_snapshot
        if data is None:
            return None
        snap = IntelSnapshot.from_payload(data)
        self._last_snapshot = snap
        return snap

    async def fetch_endpoint(self, path: str) -> Optional[Dict[str, Any]]:
        """Fetch any arbitrary backend endpoint."""
        return await self._get(path)

    async def stream_sse(self) -> AsyncIterator[Dict[str, Any]]:
        """Connect to /api/live-data/stream and yield parsed SSE chunks."""
        if not self.is_connected:
            await self.connect()
        url = f"{self.base_url}/api/live-data/stream"
        try:
            assert self._session is not None
            async with self._session.get(url) as resp:
                buffer = ""
                async for chunk in resp.content:
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        for line in event_str.split("\n"):
                            if line.startswith("data:"):
                                raw = line[5:].strip()
                                if raw:
                                    try:
                                        yield json.loads(raw)
                                    except json.JSONDecodeError:
                                        pass
        except Exception as e:
            logger.warning("SSE stream error: %s", e)

    @property
    def last_snapshot(self) -> Optional[IntelSnapshot]:
        return self._last_snapshot

    # ── persistence ───────────────────────────────────────────────────

    def _persist_snapshot(self, snap: IntelSnapshot):
        """Write a snapshot to disk for the temporal analysis modules."""
        if not self.snapshot_dir:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fp = self.snapshot_dir / f"snap_{ts}.json"
        payload = {
            "timestamp": snap.timestamp,
            "total_entities": snap.total_entities,
            "flights_count": len(snap.flights),
            "military_count": len(snap.military_flights),
            "ships_count": len(snap.ships),
            "earthquakes_count": len(snap.earthquakes),
            "fires_count": len(snap.fires),
            "gdelt_count": len(snap.gdelt_events),
            "cyber_count": len(snap.cyber_threats),
            "gps_jamming_count": len(snap.gps_jamming),
            "carriers": snap.carriers,
            "conflicts_count": len(snap.conflicts),
            "raw_keys": list(snap.raw.keys()),
        }
        fp.write_text(json.dumps(payload, default=str), encoding="utf-8")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "connected": self.is_connected,
            "total_requests": self._total_requests,
            "total_bytes": self._total_bytes,
            "cached_etags": len(self._etags),
            "has_snapshot": self._last_snapshot is not None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two lat/lon points."""
    import math
    R = 6371.0
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
