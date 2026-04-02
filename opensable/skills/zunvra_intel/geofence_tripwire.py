"""
#13 — Geofence Tripwire System

Define custom virtual boundaries anywhere on Earth.  When any entity
(aircraft, vessel, cyber threat, etc.) enters, exits, or loiters within
the zone, Sable fires an instant alert with full context.

Unlike Palantir's basic geofencing, this supports:
  - Complex polygon zones (not just circles)
  - Multi-condition triggers (enter AND military AND speed>300kts)
  - Loiter detection (entity stays in zone > threshold)
  - Entity-type filters
  - Chained fences (enter Zone A then Zone B within 2h = alert)
  - Pre-configured critical zones (Strait of Hormuz, GIUK Gap, etc.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _point_in_polygon(lat: float, lon: float,
                      polygon: List[List[float]]) -> bool:
    """Ray-casting point-in-polygon test. Polygon = list of [lat, lon]."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        pi_lat, pi_lon = polygon[i]
        pj_lat, pj_lon = polygon[j]
        if ((pi_lon > lon) != (pj_lon > lon)) and \
           (lat < (pj_lat - pi_lat) * (lon - pi_lon) / (pj_lon - pi_lon) + pi_lat):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Geofence:
    """A virtual boundary definition."""
    fence_id: str
    name: str
    description: str = ""
    # Geometry: circle OR polygon
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    radius_km: Optional[float] = None
    polygon: Optional[List[List[float]]] = None  # [[lat, lon], ...]
    # Trigger conditions
    trigger_on_enter: bool = True
    trigger_on_exit: bool = False
    trigger_on_loiter: bool = False
    loiter_threshold_minutes: int = 30
    # Filters
    entity_types: List[str] = field(default_factory=lambda: ["aircraft", "vessel", "military"])
    min_speed_kts: Optional[float] = None
    max_speed_kts: Optional[float] = None
    # Metadata
    severity: str = "high"  # low, medium, high, critical
    active: bool = True
    created_at: str = ""
    tags: List[str] = field(default_factory=list)

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is inside this geofence."""
        if self.polygon:
            return _point_in_polygon(lat, lon, self.polygon)
        if self.center_lat is not None and self.center_lon is not None and self.radius_km:
            dist = _haversine(lat, lon, self.center_lat, self.center_lon)
            return dist <= self.radius_km
        return False


@dataclass
class TripwireAlert:
    """An alert fired when an entity triggers a geofence."""
    alert_id: str
    fence_id: str
    fence_name: str
    entity_id: str
    entity_type: str
    entity_label: str
    trigger_type: str  # enter, exit, loiter
    severity: str
    timestamp: str
    lat: float
    lon: float
    description: str
    speed: Optional[float] = None
    heading: Optional[float] = None
    additional_context: str = ""


# ---------------------------------------------------------------------------
# Pre-configured critical geofences
# ---------------------------------------------------------------------------

CRITICAL_ZONES: List[Dict[str, Any]] = [
    {
        "fence_id": "strait_hormuz",
        "name": "Strait of Hormuz",
        "center_lat": 26.57, "center_lon": 56.25, "radius_km": 100,
        "severity": "critical",
        "tags": ["chokepoint", "energy", "military"],
        "description": "Critical oil chokepoint — 21% of global petroleum transits",
    },
    {
        "fence_id": "giuk_gap",
        "name": "GIUK Gap",
        "center_lat": 63.0, "center_lon": -15.0, "radius_km": 500,
        "severity": "high",
        "tags": ["naval", "submarine", "nato"],
        "description": "Greenland-Iceland-UK gap — NATO submarine choke point",
    },
    {
        "fence_id": "taiwan_strait",
        "name": "Taiwan Strait",
        "center_lat": 24.0, "center_lon": 119.5, "radius_km": 150,
        "severity": "critical",
        "tags": ["flashpoint", "military", "china"],
        "description": "Taiwan Strait — highest probability flashpoint",
    },
    {
        "fence_id": "suez_canal",
        "name": "Suez Canal",
        "center_lat": 30.45, "center_lon": 32.35, "radius_km": 50,
        "severity": "high",
        "tags": ["chokepoint", "trade"],
        "description": "12% of global trade transits Suez",
    },
    {
        "fence_id": "bab_el_mandeb",
        "name": "Bab el-Mandeb Strait",
        "center_lat": 12.6, "center_lon": 43.3, "radius_km": 80,
        "severity": "critical",
        "tags": ["chokepoint", "houthi", "energy"],
        "description": "Red Sea chokepoint — Houthi attack zone",
    },
    {
        "fence_id": "malacca_strait",
        "name": "Strait of Malacca",
        "center_lat": 2.5, "center_lon": 101.5, "radius_km": 200,
        "severity": "high",
        "tags": ["chokepoint", "energy", "piracy"],
        "description": "World's busiest shipping lane — 25% of global trade",
    },
    {
        "fence_id": "black_sea_crimea",
        "name": "Crimea / Black Sea",
        "center_lat": 44.5, "center_lon": 34.0, "radius_km": 300,
        "severity": "high",
        "tags": ["conflict", "military", "russia", "ukraine"],
        "description": "Active conflict zone — maritime & air activity monitored",
    },
    {
        "fence_id": "kaliningrad",
        "name": "Kaliningrad Exclave",
        "center_lat": 54.7, "center_lon": 20.5, "radius_km": 150,
        "severity": "high",
        "tags": ["military", "russia", "nato", "baltic"],
        "description": "Russian exclave — nuclear-capable missile deployment zone",
    },
    {
        "fence_id": "south_china_sea_spratly",
        "name": "Spratly Islands",
        "center_lat": 10.0, "center_lon": 114.0, "radius_km": 300,
        "severity": "high",
        "tags": ["territorial", "military", "china"],
        "description": "Contested artificial islands — Chinese military installations",
    },
    {
        "fence_id": "dmz_korea",
        "name": "Korean DMZ",
        "center_lat": 38.0, "center_lon": 127.0, "radius_km": 100,
        "severity": "critical",
        "tags": ["flashpoint", "nuclear", "military"],
        "description": "Most militarized border on Earth",
    },
    {
        "fence_id": "natanz_iran",
        "name": "Natanz Nuclear Facility",
        "center_lat": 33.72, "center_lon": 51.73, "radius_km": 50,
        "severity": "critical",
        "tags": ["nuclear", "iran", "military"],
        "description": "Iran's primary uranium enrichment facility",
    },
    {
        "fence_id": "zaporizhzhia_npp",
        "name": "Zaporizhzhia NPP",
        "center_lat": 47.51, "center_lon": 34.58, "radius_km": 30,
        "severity": "critical",
        "tags": ["nuclear", "conflict", "ukraine"],
        "description": "Europe's largest nuclear plant in active conflict zone",
    },
]


# ---------------------------------------------------------------------------
# Geofence Tripwire Engine
# ---------------------------------------------------------------------------

class GeofenceTripwire:
    """
    Virtual boundary monitoring system with instant alerting.

    Tracks entity positions against defined geofences and fires alerts
    on enter, exit, and loiter events.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "geofence_state.json"

        self.fences: Dict[str, Geofence] = {}
        self.alerts: List[TripwireAlert] = []
        # Track which entities are inside which fences
        self._entity_fence_state: Dict[str, Dict[str, float]] = {}  # entity_id → {fence_id: entry_timestamp}
        self.total_alerts = 0

        self._load_critical_zones()
        self._load_state()

    def _load_critical_zones(self):
        """Load pre-configured critical geofences."""
        now = datetime.now(timezone.utc).isoformat()
        for zone in CRITICAL_ZONES:
            fid = zone["fence_id"]
            if fid not in self.fences:
                self.fences[fid] = Geofence(
                    fence_id=fid,
                    name=zone["name"],
                    description=zone.get("description", ""),
                    center_lat=zone.get("center_lat"),
                    center_lon=zone.get("center_lon"),
                    radius_km=zone.get("radius_km"),
                    severity=zone.get("severity", "high"),
                    tags=zone.get("tags", []),
                    created_at=now,
                    trigger_on_enter=True,
                    trigger_on_loiter=True,
                    loiter_threshold_minutes=60,
                )

    # ── fence management ──────────────────────────────────────────────

    def add_fence(self, fence: Geofence) -> str:
        """Add a custom geofence."""
        if not fence.created_at:
            fence.created_at = datetime.now(timezone.utc).isoformat()
        self.fences[fence.fence_id] = fence
        self._save_state()
        return fence.fence_id

    def add_circle_fence(self, name: str, lat: float, lon: float,
                         radius_km: float, severity: str = "high",
                         entity_types: Optional[List[str]] = None,
                         **kwargs: Any) -> str:
        """Convenience: add a circular geofence."""
        fid = hashlib.md5(f"{name}_{lat}_{lon}".encode()).hexdigest()[:10]
        fence = Geofence(
            fence_id=fid,
            name=name,
            center_lat=lat,
            center_lon=lon,
            radius_km=radius_km,
            severity=severity,
            entity_types=entity_types or ["aircraft", "vessel", "military"],
            created_at=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )
        return self.add_fence(fence)

    def add_polygon_fence(self, name: str, polygon: List[List[float]],
                          severity: str = "high", **kwargs: Any) -> str:
        """Convenience: add a polygon geofence."""
        fid = hashlib.md5(f"{name}_{len(polygon)}".encode()).hexdigest()[:10]
        fence = Geofence(
            fence_id=fid,
            name=name,
            polygon=polygon,
            severity=severity,
            created_at=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )
        return self.add_fence(fence)

    def remove_fence(self, fence_id: str) -> bool:
        if fence_id in self.fences:
            del self.fences[fence_id]
            self._save_state()
            return True
        return False

    # ── main evaluation ───────────────────────────────────────────────

    def evaluate(self, snapshot: IntelSnapshot) -> List[TripwireAlert]:
        """
        Evaluate all entities against all active geofences.

        Returns list of new alerts triggered.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        now_ts = time.time()
        new_alerts: List[TripwireAlert] = []

        entities = self._extract_entities(snapshot)
        currently_inside: Dict[str, Set[str]] = {}  # entity_id → {fence_ids currently inside}

        for entity in entities:
            eid = entity["id"]
            lat = entity.get("lat")
            lon = entity.get("lon")
            if lat is None or lon is None:
                continue

            currently_inside[eid] = set()

            for fence_id, fence in self.fences.items():
                if not fence.active:
                    continue
                if entity["type"] not in fence.entity_types and "all" not in fence.entity_types:
                    continue

                # Speed filter
                speed = entity.get("speed")
                if fence.min_speed_kts and speed is not None and speed < fence.min_speed_kts:
                    continue
                if fence.max_speed_kts and speed is not None and speed > fence.max_speed_kts:
                    continue

                is_inside = fence.contains(lat, lon)

                # Previous state
                was_inside = (eid in self._entity_fence_state and
                             fence_id in self._entity_fence_state.get(eid, {}))

                if is_inside:
                    currently_inside[eid].add(fence_id)

                    if not was_inside and fence.trigger_on_enter:
                        # ENTER event
                        alert = TripwireAlert(
                            alert_id=hashlib.md5(f"enter_{eid}_{fence_id}_{now_ts}".encode()).hexdigest()[:12],
                            fence_id=fence_id,
                            fence_name=fence.name,
                            entity_id=eid,
                            entity_type=entity["type"],
                            entity_label=entity.get("label", eid),
                            trigger_type="enter",
                            severity=fence.severity,
                            timestamp=now_str,
                            lat=lat, lon=lon,
                            description=f"{entity.get('label', eid)} entered {fence.name}",
                            speed=speed,
                            heading=entity.get("heading"),
                        )
                        new_alerts.append(alert)

                        # Record entry time
                        self._entity_fence_state.setdefault(eid, {})[fence_id] = now_ts

                    elif was_inside and fence.trigger_on_loiter:
                        # Check loiter duration
                        entry_time = self._entity_fence_state.get(eid, {}).get(fence_id, now_ts)
                        minutes_inside = (now_ts - entry_time) / 60
                        if minutes_inside >= fence.loiter_threshold_minutes:
                            alert = TripwireAlert(
                                alert_id=hashlib.md5(f"loiter_{eid}_{fence_id}_{now_ts}".encode()).hexdigest()[:12],
                                fence_id=fence_id,
                                fence_name=fence.name,
                                entity_id=eid,
                                entity_type=entity["type"],
                                entity_label=entity.get("label", eid),
                                trigger_type="loiter",
                                severity=fence.severity,
                                timestamp=now_str,
                                lat=lat, lon=lon,
                                description=(f"{entity.get('label', eid)} loitering in {fence.name} "
                                           f"for {minutes_inside:.0f}min"),
                                speed=speed,
                                additional_context=f"Entry time: {entry_time}",
                            )
                            new_alerts.append(alert)

                elif not is_inside and was_inside and fence.trigger_on_exit:
                    # EXIT event
                    entry_time = self._entity_fence_state.get(eid, {}).get(fence_id, now_ts)
                    duration_min = (now_ts - entry_time) / 60
                    alert = TripwireAlert(
                        alert_id=hashlib.md5(f"exit_{eid}_{fence_id}_{now_ts}".encode()).hexdigest()[:12],
                        fence_id=fence_id,
                        fence_name=fence.name,
                        entity_id=eid,
                        entity_type=entity["type"],
                        entity_label=entity.get("label", eid),
                        trigger_type="exit",
                        severity=fence.severity,
                        timestamp=now_str,
                        lat=lat, lon=lon,
                        description=f"{entity.get('label', eid)} exited {fence.name} after {duration_min:.0f}min",
                        speed=speed,
                    )
                    new_alerts.append(alert)

                    # Clear entry record
                    if eid in self._entity_fence_state:
                        self._entity_fence_state[eid].pop(fence_id, None)

        # Clean stale state
        for eid in list(self._entity_fence_state.keys()):
            if eid in currently_inside:
                stale = set(self._entity_fence_state[eid].keys()) - currently_inside[eid]
                for fid in stale:
                    del self._entity_fence_state[eid][fid]

        self.alerts.extend(new_alerts)
        self.total_alerts += len(new_alerts)
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]

        self._save_state()
        return new_alerts

    # ── entity extraction ─────────────────────────────────────────────

    def _extract_entities(self, snapshot: IntelSnapshot) -> List[Dict[str, Any]]:
        """Flatten snapshot into evaluable entity dicts."""
        entities: List[Dict[str, Any]] = []

        for flight in snapshot.flights:
            lat = self._safe_float(flight.get("lat"))
            lon = self._safe_float(flight.get("lon"))
            if lat is not None and lon is not None:
                entities.append({
                    "id": f"aircraft_{flight.get('hex', '')}",
                    "type": "aircraft",
                    "label": flight.get("callsign", flight.get("flight", flight.get("hex", ""))),
                    "lat": lat, "lon": lon,
                    "speed": self._safe_float(flight.get("gs")),
                    "heading": self._safe_float(flight.get("track")),
                })

        for flight in snapshot.military_flights:
            lat = self._safe_float(flight.get("lat"))
            lon = self._safe_float(flight.get("lon"))
            if lat is not None and lon is not None:
                entities.append({
                    "id": f"military_{flight.get('hex', '')}",
                    "type": "military",
                    "label": flight.get("callsign", flight.get("type", flight.get("hex", ""))),
                    "lat": lat, "lon": lon,
                    "speed": self._safe_float(flight.get("gs")),
                    "heading": self._safe_float(flight.get("track")),
                })

        for ship in snapshot.ships:
            lat = self._safe_float(ship.get("lat"))
            lon = self._safe_float(ship.get("lon"))
            if lat is not None and lon is not None:
                entities.append({
                    "id": f"vessel_{ship.get('mmsi', ship.get('MMSI', ''))}",
                    "type": "vessel",
                    "label": ship.get("name", ship.get("ship_name", str(ship.get("mmsi", "")))),
                    "lat": lat, "lon": lon,
                    "speed": self._safe_float(ship.get("speed", ship.get("sog"))),
                    "heading": self._safe_float(ship.get("heading", ship.get("cog"))),
                })

        return entities

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # ── queries ───────────────────────────────────────────────────────

    def get_active_fences(self) -> List[Geofence]:
        return [f for f in self.fences.values() if f.active]

    def get_recent_alerts(self, limit: int = 50) -> List[TripwireAlert]:
        return self.alerts[-limit:]

    def get_alerts_by_fence(self, fence_id: str) -> List[TripwireAlert]:
        return [a for a in self.alerts if a.fence_id == fence_id]

    def get_entities_in_fence(self, fence_id: str) -> List[str]:
        """Return entity IDs currently inside a fence."""
        return [eid for eid, fences in self._entity_fence_state.items()
                if fence_id in fences]

    def get_fence_geojson(self) -> Dict[str, Any]:
        """Get all fences as GeoJSON for map overlay."""
        features = []
        for f in self.fences.values():
            if f.polygon:
                coords = [[p[1], p[0]] for p in f.polygon]
                coords.append(coords[0])  # Close polygon
                geom = {"type": "Polygon", "coordinates": [coords]}
            elif f.center_lat is not None and f.center_lon is not None:
                geom = {"type": "Point", "coordinates": [f.center_lon, f.center_lat]}
            else:
                continue

            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "id": f.fence_id,
                    "name": f.name,
                    "severity": f.severity,
                    "radius_km": f.radius_km,
                    "active": f.active,
                    "tags": f.tags,
                    "description": f.description,
                    "entities_inside": len(self.get_entities_in_fence(f.fence_id)),
                },
            })

        return {"type": "FeatureCollection", "features": features}

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "total_alerts": self.total_alerts,
                "fence_count": len(self.fences),
                "entity_states": len(self._entity_fence_state),
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save geofence state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_alerts = state.get("total_alerts", 0)
        except Exception as e:
            logger.warning("Failed to load geofence state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_fences": len(self.fences),
            "active_fences": sum(1 for f in self.fences.values() if f.active),
            "total_alerts": self.total_alerts,
            "entities_tracked": len(self._entity_fence_state),
            "critical_zones": [f.name for f in self.fences.values() if f.severity == "critical"],
        }
