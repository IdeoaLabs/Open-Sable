"""
#25 — Force Projection Tracker

ONI / DIA / STRATCOM capability.  Tracks carrier strike groups, monitors
force deployments, predicts power projection patterns, and assesses
naval/air force disposition globally.

Data sources: snapshot.carriers, snapshot.military_flights, snapshot.ships
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .connector import IntelSnapshot

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategic naval chokepoints & areas of interest
# ---------------------------------------------------------------------------

STRATEGIC_AREAS: Dict[str, Dict[str, Any]] = {
    "strait_of_hormuz": {"lat": 26.56, "lon": 56.25, "radius_km": 150,
                         "importance": "oil_transit"},
    "taiwan_strait": {"lat": 24.0, "lon": 118.5, "radius_km": 200,
                      "importance": "geopolitical_flashpoint"},
    "south_china_sea": {"lat": 12.0, "lon": 114.0, "radius_km": 500,
                        "importance": "territorial_dispute"},
    "giuk_gap": {"lat": 63.0, "lon": -15.0, "radius_km": 300,
                 "importance": "submarine_chokepoint"},
    "suez_canal": {"lat": 30.45, "lon": 32.35, "radius_km": 100,
                   "importance": "trade_chokepoint"},
    "bab_el_mandeb": {"lat": 12.6, "lon": 43.3, "radius_km": 100,
                      "importance": "red_sea_access"},
    "malacca_strait": {"lat": 2.5, "lon": 101.0, "radius_km": 150,
                       "importance": "pacific_trade"},
    "baltic_sea": {"lat": 57.5, "lon": 19.0, "radius_km": 200,
                   "importance": "nato_russia_interface"},
    "persian_gulf": {"lat": 27.0, "lon": 51.0, "radius_km": 300,
                     "importance": "energy_security"},
    "sea_of_japan": {"lat": 40.0, "lon": 135.0, "radius_km": 300,
                     "importance": "north_korea_deterrence"},
    "eastern_med": {"lat": 34.0, "lon": 32.0, "radius_km": 250,
                    "importance": "middle_east_projection"},
    "arctic_passage": {"lat": 75.0, "lon": 40.0, "radius_km": 500,
                       "importance": "emerging_route"},
}

# Known carrier designators / hull numbers
CARRIER_KEYWORDS = [
    "nimitz", "ford", "eisenhower", "lincoln", "washington",
    "roosevelt", "truman", "reagan", "stennis", "vinson",
    "kuznetsov", "liaoning", "shandong", "fujian",
    "charles de gaulle", "queen elizabeth", "prince of wales",
    "cavour", "vikramaditya", "vikrant",
    "cvn-", "cv-", "r-", "admiral",
]


@dataclass
class CarrierGroup:
    """Tracked carrier strike group."""
    group_id: str
    name: str
    country: str
    lat: float
    lon: float
    heading: float = 0.0
    speed_knots: float = 0.0
    current_area: str = ""
    status: str = "transit"  # transit, station, exercise, deployment, surge
    escort_count: int = 0
    air_activity: int = 0  # nearby military flights
    first_tracked: str = ""
    last_seen: str = ""
    prev_lat: float = 0.0
    prev_lon: float = 0.0
    days_deployed: int = 0
    track_history: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class ForceAlert:
    """Force projection alert."""
    alert_id: str
    alert_type: str   # carrier_movement, force_buildup, surge_deployment,
                       # chokepoint_transit, area_denial, power_vacuum
    severity: str
    timestamp: str
    title: str
    description: str
    area: str = ""
    countries: List[str] = field(default_factory=list)
    coordinates: Optional[Tuple[float, float]] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(rlat2)
    y = (math.cos(rlat1) * math.sin(rlat2) -
         math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


class ForceProjectionTracker:
    """
    Global force projection tracking engine.

    Monitors carrier strike groups, military deployments, and
    force disposition to assess power projection and detect
    surge deployments or area denial operations.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.carrier_groups: Dict[str, CarrierGroup] = {}
        self.alerts: List[ForceAlert] = []
        self.area_presence: Dict[str, Dict[str, int]] = {
            a: {} for a in STRATEGIC_AREAS
        }  # area -> { country: count }
        self.total_detections = 0

    def observe(self, snapshot: IntelSnapshot) -> List[ForceAlert]:
        """Analyze snapshot for force projection indicators."""
        now_str = datetime.now(timezone.utc).isoformat()
        new_alerts: List[ForceAlert] = []

        # 1. Track carrier groups
        new_alerts.extend(self._track_carriers(snapshot, now_str))

        # 2. Analyze military flight patterns near strategic areas
        new_alerts.extend(self._analyze_air_projection(snapshot, now_str))

        # 3. Detect force buildups
        new_alerts.extend(self._detect_buildups(now_str))

        self.total_detections += len(new_alerts)
        self.alerts.extend(new_alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-250:]

        return new_alerts

    # ── carrier tracking ──────────────────────────────────────────────

    def _track_carriers(self, snapshot: IntelSnapshot,
                        now_str: str) -> List[ForceAlert]:
        alerts: List[ForceAlert] = []

        for carrier in snapshot.carriers:
            name = carrier.get("name", carrier.get("vessel_name", ""))
            country = carrier.get("country", carrier.get("flag", ""))
            lat = carrier.get("lat", carrier.get("latitude"))
            lon = carrier.get("lon", carrier.get("longitude"))
            speed = carrier.get("speed", carrier.get("sog"))
            heading = carrier.get("heading", carrier.get("cog"))

            if not name or lat is None or lon is None:
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
                speed_f = float(speed) if speed is not None else 0
                heading_f = float(heading) if heading is not None else 0
            except (ValueError, TypeError):
                continue

            gid = f"csg_{name.lower().replace(' ', '_')}"

            # Determine current strategic area
            current_area = ""
            for aname, ainfo in STRATEGIC_AREAS.items():
                dist = _haversine(lat_f, lon_f, ainfo["lat"], ainfo["lon"])
                if dist <= ainfo["radius_km"]:
                    current_area = aname
                    break

            existing = self.carrier_groups.get(gid)
            if existing:
                # Movement analysis
                dist_moved = _haversine(existing.lat, existing.lon, lat_f, lon_f)
                prev_area = existing.current_area

                existing.prev_lat = existing.lat
                existing.prev_lon = existing.lon
                existing.lat = lat_f
                existing.lon = lon_f
                existing.speed_knots = speed_f
                existing.heading = heading_f
                existing.current_area = current_area
                existing.last_seen = now_str
                existing.track_history.append((lat_f, lon_f))
                if len(existing.track_history) > 100:
                    existing.track_history = existing.track_history[-50:]

                # Detect area transition
                if current_area and current_area != prev_area:
                    area_info = STRATEGIC_AREAS[current_area]
                    alerts.append(ForceAlert(
                        alert_id=hashlib.md5(f"csg_{gid}_{current_area}_{time.time()}".encode()).hexdigest()[:10],
                        alert_type="chokepoint_transit" if "chokepoint" in area_info.get("importance", "") else "carrier_movement",
                        severity="high",
                        timestamp=now_str,
                        title=f"Carrier movement: {name} entered {current_area}",
                        description=(f"{name} ({country}) entered {current_area} "
                                    f"(importance: {area_info['importance']}). "
                                    f"Speed: {speed_f:.0f}kn, heading: {heading_f:.0f}°. "
                                    f"{'From: ' + prev_area if prev_area else 'Previous area unknown.'}"),
                        area=current_area,
                        countries=[country],
                        coordinates=(lat_f, lon_f),
                        evidence={"carrier": name, "from_area": prev_area,
                                  "to_area": current_area, "speed": speed_f},
                    ))

                # Detect surge (high speed)
                if speed_f > 25:
                    brg = _bearing(existing.prev_lat, existing.prev_lon, lat_f, lon_f)
                    alerts.append(ForceAlert(
                        alert_id=hashlib.md5(f"surge_{gid}_{time.time()}".encode()).hexdigest()[:10],
                        alert_type="surge_deployment",
                        severity="high",
                        timestamp=now_str,
                        title=f"Carrier surge: {name} at {speed_f:.0f} knots",
                        description=(f"{name} ({country}) surging at {speed_f:.0f}kn, "
                                    f"bearing {brg:.0f}°. Possible rapid deployment."),
                        countries=[country],
                        coordinates=(lat_f, lon_f),
                        evidence={"speed_knots": speed_f, "bearing": brg},
                    ))

            else:
                # New carrier detection
                self.carrier_groups[gid] = CarrierGroup(
                    group_id=gid,
                    name=name,
                    country=country,
                    lat=lat_f, lon=lon_f,
                    heading=heading_f,
                    speed_knots=speed_f,
                    current_area=current_area,
                    first_tracked=now_str,
                    last_seen=now_str,
                    track_history=[(lat_f, lon_f)],
                )
                alerts.append(ForceAlert(
                    alert_id=hashlib.md5(f"new_{gid}".encode()).hexdigest()[:10],
                    alert_type="carrier_movement",
                    severity="medium",
                    timestamp=now_str,
                    title=f"New carrier tracked: {name} ({country})",
                    description=f"Carrier {name} ({country}) now tracked at {lat_f:.2f},{lon_f:.2f}. Area: {current_area or 'open ocean'}.",
                    countries=[country],
                    coordinates=(lat_f, lon_f),
                    evidence={"area": current_area},
                ))

            # Update area presence
            if current_area and country:
                self.area_presence[current_area][country] = (
                    self.area_presence[current_area].get(country, 0) + 1
                )

        # Also check regular ships that match carrier patterns
        for ship in snapshot.ships:
            sname = (ship.get("name") or ship.get("ship_name") or "").lower()
            stype = (ship.get("type") or ship.get("ship_type") or "").lower()
            if any(kw in sname for kw in CARRIER_KEYWORDS) or "carrier" in stype:
                # Already tracked as carrier? skip
                gid = f"csg_{sname.replace(' ', '_')}"
                if gid not in self.carrier_groups:
                    lat = ship.get("lat", ship.get("latitude"))
                    lon = ship.get("lon", ship.get("longitude"))
                    flag = ship.get("flag", ship.get("flag_state", ""))
                    if lat is not None and lon is not None:
                        try:
                            lat_f, lon_f = float(lat), float(lon)
                            self.carrier_groups[gid] = CarrierGroup(
                                group_id=gid, name=sname, country=flag,
                                lat=lat_f, lon=lon_f,
                                first_tracked=now_str, last_seen=now_str,
                                track_history=[(lat_f, lon_f)],
                            )
                        except (ValueError, TypeError):
                            pass

        return alerts

    # ── air force projection ──────────────────────────────────────────

    def _analyze_air_projection(self, snapshot: IntelSnapshot,
                                now_str: str) -> List[ForceAlert]:
        alerts: List[ForceAlert] = []
        area_flights: Dict[str, List[str]] = {a: [] for a in STRATEGIC_AREAS}

        for flight in snapshot.military_flights:
            lat = flight.get("lat", flight.get("latitude"))
            lon = flight.get("lon", flight.get("longitude"))
            callsign = flight.get("callsign", "")
            country = flight.get("country", flight.get("origin", ""))

            if lat is None or lon is None:
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
            except (ValueError, TypeError):
                continue

            for aname, ainfo in STRATEGIC_AREAS.items():
                dist = _haversine(lat_f, lon_f, ainfo["lat"], ainfo["lon"])
                if dist <= ainfo["radius_km"]:
                    area_flights[aname].append(callsign or country or "unknown")
                    if country:
                        self.area_presence[aname][country] = (
                            self.area_presence[aname].get(country, 0) + 1
                        )

        # Update carrier groups with nearby air activity
        for gid, cg in self.carrier_groups.items():
            air_count = 0
            for flight in snapshot.military_flights:
                flat = flight.get("lat", flight.get("latitude"))
                flon = flight.get("lon", flight.get("longitude"))
                if flat is None or flon is None:
                    continue
                try:
                    dist = _haversine(cg.lat, cg.lon, float(flat), float(flon))
                    if dist < 200:
                        air_count += 1
                except (ValueError, TypeError):
                    pass
            cg.air_activity = air_count

        return alerts

    # ── force buildup detection ───────────────────────────────────────

    def _detect_buildups(self, now_str: str) -> List[ForceAlert]:
        alerts: List[ForceAlert] = []

        for aname, countries in self.area_presence.items():
            total = sum(countries.values())
            if total < 5:
                continue

            # Multiple countries = tension indicator
            nations = [c for c, n in countries.items() if n > 0]
            if len(nations) >= 2:
                alerts.append(ForceAlert(
                    alert_id=hashlib.md5(f"buildup_{aname}_{total}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="force_buildup",
                    severity="critical" if total >= 10 else "high",
                    timestamp=now_str,
                    title=f"Force buildup: {aname} ({len(nations)} nations, {total} assets)",
                    description=(f"Multi-national force presence in {aname}: "
                                f"{', '.join(nations)}. {total} total military assets. "
                                f"Importance: {STRATEGIC_AREAS[aname]['importance']}."),
                    area=aname,
                    countries=nations,
                    coordinates=(STRATEGIC_AREAS[aname]["lat"],
                                STRATEGIC_AREAS[aname]["lon"]),
                    evidence={"nations": nations, "total_assets": total,
                              "breakdown": dict(countries)},
                ))

        # Reset area presence for next cycle
        self.area_presence = {a: {} for a in STRATEGIC_AREAS}

        return alerts

    # ── queries ───────────────────────────────────────────────────────

    def get_carrier_positions(self) -> List[CarrierGroup]:
        return sorted(self.carrier_groups.values(),
                      key=lambda c: c.last_seen, reverse=True)

    def get_area_disposition(self) -> Dict[str, Dict[str, int]]:
        return self.area_presence

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_detections": self.total_detections,
            "carriers_tracked": len(self.carrier_groups),
            "carriers_by_country": {},  # Populate below
            "strategic_areas_active": sum(1 for a in self.area_presence.values()
                                          if sum(a.values()) > 0),
            "total_alerts": len(self.alerts),
        }
