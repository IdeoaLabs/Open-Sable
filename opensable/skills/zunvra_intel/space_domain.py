"""
#22,  Space Domain Awareness (SDA)

NRO / Space Force / NORAD capability.  Tracks satellite movements,
detects orbital anomalies, identifies potential ASAT activities, and
correlates space activity with ground events.

Data sources: snapshot.satellites, snapshot.military_flights
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
# Orbital regime definitions
# ---------------------------------------------------------------------------

ORBITAL_REGIMES = {
    "LEO":  (160, 2000),      # Low Earth Orbit
    "MEO":  (2000, 35786),    # Medium Earth Orbit (GPS, navigation)
    "GEO":  (35786, 35900),   # Geostationary
    "HEO":  (35900, 100000),  # Highly Elliptical Orbit
}

# Known military / intelligence satellite operators
MILITARY_SAT_KEYWORDS = [
    "usa-", "cosmos", "yaogan", "gaofen", "beidou", "glonass",
    "lacrosse", "keyhole", "onyx", "mentor", "orion", "trumpet",
    "nrol-", "sds-", "milstar", "aehf", "sbirs", "wgs-",
    "muos-", "geodss", "topaz", "pion", "lotos", "persona",
]

# Countries of concern for ASAT capability
ASAT_CAPABLE = {"China", "Russia", "India", "United States"}


@dataclass
class SpaceObject:
    """Tracked space object."""
    object_id: str
    name: str
    norad_id: Optional[str] = None
    country: str = ""
    orbit_regime: str = "unknown"
    altitude_km: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    velocity_kms: float = 0.0
    is_military: bool = False
    anomalous: bool = False
    first_seen: str = ""
    last_seen: str = ""
    maneuver_count: int = 0
    prev_altitude: float = 0.0
    prev_lat: float = 0.0
    prev_lon: float = 0.0


@dataclass
class SpaceAlert:
    """Space domain awareness alert."""
    alert_id: str
    alert_type: str  # orbital_maneuver, proximity_event, asat_indicator,
                     # new_launch, debris_risk, ground_correlation, constellation_change
    severity: str
    timestamp: str
    title: str
    description: str
    objects: List[str] = field(default_factory=list)
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


class SpaceDomainAwareness:
    """
    Space domain awareness engine.

    Tracks satellites, detects orbital anomalies, identifies ASAT
    indicators, and correlates space events with ground activities.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.catalog: Dict[str, SpaceObject] = {}
        self.alerts: List[SpaceAlert] = []
        self.total_detections = 0

    def observe(self, snapshot: IntelSnapshot) -> List[SpaceAlert]:
        """Analyze snapshot for space domain awareness indicators."""
        now_str = datetime.now(timezone.utc).isoformat()
        new_alerts: List[SpaceAlert] = []

        # 1. Update catalog and detect maneuvers
        new_alerts.extend(self._update_catalog(snapshot, now_str))

        # 2. Proximity analysis (potential ASAT / rendezvous)
        new_alerts.extend(self._detect_proximity(now_str))

        # 3. Military satellite correlation with ground events
        new_alerts.extend(self._correlate_ground(snapshot, now_str))

        self.total_detections += len(new_alerts)
        self.alerts.extend(new_alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-250:]

        return new_alerts

    # ── catalog management ────────────────────────────────────────────

    def _update_catalog(self, snapshot: IntelSnapshot,
                        now_str: str) -> List[SpaceAlert]:
        alerts: List[SpaceAlert] = []

        for sat in snapshot.satellites:
            name = sat.get("name", sat.get("satellite_name", ""))
            norad = str(sat.get("norad_id", sat.get("id", "")))
            lat = sat.get("lat", sat.get("latitude"))
            lon = sat.get("lon", sat.get("longitude"))
            alt = sat.get("altitude", sat.get("altitude_km"))
            country = sat.get("country", sat.get("origin", ""))
            velocity = sat.get("velocity", sat.get("speed"))

            if not name and not norad:
                continue

            oid = f"sat_{norad}" if norad else f"sat_{name}"

            try:
                lat_f = float(lat) if lat is not None else 0.0
                lon_f = float(lon) if lon is not None else 0.0
                alt_f = float(alt) if alt is not None else 0.0
                vel_f = float(velocity) if velocity is not None else 0.0
            except (ValueError, TypeError):
                lat_f = lon_f = alt_f = vel_f = 0.0

            # Classify orbit
            regime = "unknown"
            for regime_name, (lo, hi) in ORBITAL_REGIMES.items():
                if lo <= alt_f <= hi:
                    regime = regime_name
                    break

            # Classify military
            is_mil = any(kw in name.lower() for kw in MILITARY_SAT_KEYWORDS)
            if country and country.lower() in ("russia", "china", "iran", "north korea"):
                is_mil = True

            existing = self.catalog.get(oid)

            if existing:
                # Check for orbital maneuver
                if existing.altitude_km > 0 and alt_f > 0:
                    delta_alt = abs(alt_f - existing.altitude_km)
                    if delta_alt > 10:  # >10km altitude change = deliberate maneuver
                        existing.maneuver_count += 1
                        existing.anomalous = True
                        alerts.append(SpaceAlert(
                            alert_id=hashlib.md5(f"man_{oid}_{time.time()}".encode()).hexdigest()[:10],
                            alert_type="orbital_maneuver",
                            severity="high" if delta_alt > 50 else "medium",
                            timestamp=now_str,
                            title=f"Orbital maneuver: {name} ({country})",
                            description=(f"Altitude change: {existing.altitude_km:.0f}km → "
                                        f"{alt_f:.0f}km (Δ{delta_alt:.0f}km). "
                                        f"Maneuver #{existing.maneuver_count}. "
                                        f"Regime: {regime}."),
                            objects=[oid],
                            evidence={"delta_alt_km": delta_alt, "regime": regime,
                                      "is_military": is_mil, "country": country},
                        ))

                existing.prev_altitude = existing.altitude_km
                existing.prev_lat = existing.lat
                existing.prev_lon = existing.lon
                existing.altitude_km = alt_f
                existing.lat = lat_f
                existing.lon = lon_f
                existing.velocity_kms = vel_f
                existing.last_seen = now_str
                existing.orbit_regime = regime

            else:
                # New object
                self.catalog[oid] = SpaceObject(
                    object_id=oid,
                    name=name,
                    norad_id=norad,
                    country=country,
                    orbit_regime=regime,
                    altitude_km=alt_f,
                    lat=lat_f,
                    lon=lon_f,
                    velocity_kms=vel_f,
                    is_military=is_mil,
                    first_seen=now_str,
                    last_seen=now_str,
                )

                if is_mil:
                    alerts.append(SpaceAlert(
                        alert_id=hashlib.md5(f"new_{oid}".encode()).hexdigest()[:10],
                        alert_type="new_launch",
                        severity="medium",
                        timestamp=now_str,
                        title=f"New military satellite: {name} ({country})",
                        description=f"New military satellite detected: {name}, {regime} orbit at {alt_f:.0f}km",
                        objects=[oid],
                        evidence={"regime": regime, "altitude_km": alt_f, "country": country},
                    ))

        # Trim catalog to 10K objects
        if len(self.catalog) > 10000:
            sorted_sats = sorted(self.catalog.values(), key=lambda s: s.last_seen)
            for s in sorted_sats[:len(self.catalog) - 5000]:
                del self.catalog[s.object_id]

        return alerts

    # ── proximity / ASAT detection ────────────────────────────────────

    def _detect_proximity(self, now_str: str) -> List[SpaceAlert]:
        """Detect close approaches between objects (potential ASAT / rendezvous)."""
        alerts: List[SpaceAlert] = []
        mil_sats = [s for s in self.catalog.values() if s.is_military and s.altitude_km > 0]

        checked = set()
        for i, sat_a in enumerate(mil_sats):
            for sat_b in mil_sats[i + 1:]:
                pair = tuple(sorted([sat_a.object_id, sat_b.object_id]))
                if pair in checked:
                    continue
                checked.add(pair)

                # Same orbital regime and close ground track
                if sat_a.orbit_regime == sat_b.orbit_regime:
                    alt_diff = abs(sat_a.altitude_km - sat_b.altitude_km)
                    ground_dist = _haversine(sat_a.lat, sat_a.lon, sat_b.lat, sat_b.lon)

                    if alt_diff < 20 and ground_dist < 100:
                        # Different countries = more suspicious
                        cross_nation = (sat_a.country != sat_b.country and
                                       sat_a.country and sat_b.country)

                        alerts.append(SpaceAlert(
                            alert_id=hashlib.md5(f"prox_{pair}_{time.time()}".encode()).hexdigest()[:10],
                            alert_type="asat_indicator" if cross_nation else "proximity_event",
                            severity="critical" if cross_nation else "high",
                            timestamp=now_str,
                            title=(f"Satellite proximity: {sat_a.name} ↔ {sat_b.name}"),
                            description=(f"Close approach in {sat_a.orbit_regime}: "
                                        f"Δalt={alt_diff:.0f}km, ground={ground_dist:.0f}km. "
                                        f"{sat_a.country} vs {sat_b.country}."),
                            objects=[sat_a.object_id, sat_b.object_id],
                            evidence={"alt_diff_km": alt_diff, "ground_dist_km": ground_dist,
                                      "cross_nation": cross_nation},
                        ))

            if len(alerts) > 20:
                break

        return alerts

    # ── ground correlation ────────────────────────────────────────────

    def _correlate_ground(self, snapshot: IntelSnapshot,
                          now_str: str) -> List[SpaceAlert]:
        """Correlate satellite positions with military ground activity."""
        alerts: List[SpaceAlert] = []
        recon_sats = [s for s in self.catalog.values()
                      if s.is_military and s.orbit_regime == "LEO"]

        for flight in snapshot.military_flights:
            flat = flight.get("lat", flight.get("latitude"))
            flon = flight.get("lon", flight.get("longitude"))
            callsign = flight.get("callsign", "")

            if flat is None or flon is None:
                continue
            try:
                flat_f, flon_f = float(flat), float(flon)
            except (ValueError, TypeError):
                continue

            for sat in recon_sats:
                if sat.lat == 0 and sat.lon == 0:
                    continue
                dist = _haversine(flat_f, flon_f, sat.lat, sat.lon)
                if dist < 200:  # Satellite footprint over military activity
                    alerts.append(SpaceAlert(
                        alert_id=hashlib.md5(f"gc_{sat.object_id}_{callsign}_{time.time()}".encode()).hexdigest()[:10],
                        alert_type="ground_correlation",
                        severity="medium",
                        timestamp=now_str,
                        title=f"Space-ground correlation: {sat.name} over {callsign}",
                        description=(f"Recon satellite {sat.name} ({sat.country}) within "
                                    f"{dist:.0f}km of military flight {callsign}. "
                                    f"Possible persistent surveillance."),
                        objects=[sat.object_id],
                        coordinates=(flat_f, flon_f),
                        evidence={"satellite": sat.name, "flight": callsign,
                                  "distance_km": dist},
                    ))

            if len(alerts) > 15:
                break

        return alerts

    # ── queries ───────────────────────────────────────────────────────

    def get_catalog_summary(self) -> Dict[str, Any]:
        regime_count: Dict[str, int] = {}
        for s in self.catalog.values():
            regime_count[s.orbit_regime] = regime_count.get(s.orbit_regime, 0) + 1
        return {
            "total_objects": len(self.catalog),
            "military": sum(1 for s in self.catalog.values() if s.is_military),
            "anomalous": sum(1 for s in self.catalog.values() if s.anomalous),
            "by_regime": regime_count,
        }

    def get_military_sats(self) -> List[SpaceObject]:
        return [s for s in self.catalog.values() if s.is_military]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_detections": self.total_detections,
            "catalog_size": len(self.catalog),
            "military_objects": sum(1 for s in self.catalog.values() if s.is_military),
            "anomalous_objects": sum(1 for s in self.catalog.values() if s.anomalous),
            "total_maneuvers": sum(s.maneuver_count for s in self.catalog.values()),
            "total_alerts": len(self.alerts),
        }
