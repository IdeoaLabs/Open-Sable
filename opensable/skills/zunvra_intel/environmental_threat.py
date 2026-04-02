"""
#24 — Environmental Threat Intelligence

Military METOC / NGA / FEMA capability.  Correlates fire and earthquake
data with conflict zones to identify scorched-earth campaigns, detect
natural disasters impacting military operations, and assess climate-
driven conflict triggers.

Data sources: snapshot.fires, snapshot.earthquakes, snapshot.conflicts
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
# Conflict-zone fire thresholds
# ---------------------------------------------------------------------------

CONFLICT_ZONES: Dict[str, Dict[str, Any]] = {
    "ukraine_front": {"lat": 48.5, "lon": 37.5, "radius_km": 200,
                      "parties": ["Russia", "Ukraine"]},
    "gaza": {"lat": 31.4, "lon": 34.4, "radius_km": 40,
             "parties": ["Israel", "Hamas"]},
    "sudan": {"lat": 15.6, "lon": 32.5, "radius_km": 300,
              "parties": ["SAF", "RSF"]},
    "myanmar": {"lat": 19.7, "lon": 96.2, "radius_km": 250,
                "parties": ["Junta", "NUG"]},
    "sahel": {"lat": 14.0, "lon": 2.0, "radius_km": 500,
              "parties": ["JNIM", "ISGS", "Wagner"]},
    "ethiopia_tigray": {"lat": 13.5, "lon": 39.5, "radius_km": 200,
                        "parties": ["ENDF", "TPLF"]},
    "syria_idlib": {"lat": 35.8, "lon": 36.8, "radius_km": 100,
                    "parties": ["SAA", "HTS", "Turkey"]},
    "drc_east": {"lat": -1.5, "lon": 29.0, "radius_km": 200,
                 "parties": ["FARDC", "M23", "Rwanda"]},
}

# Natural disaster thresholds
MAJOR_QUAKE_MAGNITUDE = 6.0
CATASTROPHIC_QUAKE_MAGNITUDE = 7.5
FIRE_CLUSTER_THRESHOLD = 10  # fires within a region = suspicious


@dataclass
class EnvAlert:
    """Environmental threat alert."""
    alert_id: str
    alert_type: str   # scorched_earth, natural_disaster, seismic_threat,
                       # fire_corridor, climate_conflict, humanitarian
    severity: str
    timestamp: str
    title: str
    description: str
    region: str = ""
    coordinates: Optional[Tuple[float, float]] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FireCluster:
    """Cluster of fire detections in a region."""
    cluster_id: str
    center_lat: float
    center_lon: float
    fire_count: int
    region: str = ""
    in_conflict_zone: bool = False
    conflict_name: str = ""
    first_seen: str = ""
    last_seen: str = ""
    spread_km: float = 0.0


@dataclass
class SeismicEvent:
    """Significant seismic event."""
    event_id: str
    lat: float
    lon: float
    magnitude: float
    depth_km: float
    timestamp: str
    near_infrastructure: bool = False
    near_population: bool = False


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class EnvironmentalThreatIntel:
    """
    Environmental threat intelligence engine.

    Correlates fires, earthquakes, and conflicts to detect scorched-earth
    campaigns, assess natural disaster impact on military operations,
    and identify climate-driven conflict triggers.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.alerts: List[EnvAlert] = []
        self.fire_clusters: Dict[str, FireCluster] = {}
        self.seismic_events: List[SeismicEvent] = []
        self.total_detections = 0

    def observe(self, snapshot: IntelSnapshot) -> List[EnvAlert]:
        """Analyze snapshot for environmental threat indicators."""
        now_str = datetime.now(timezone.utc).isoformat()
        new_alerts: List[EnvAlert] = []

        # 1. Fire analysis (scorched earth, fire corridors)
        new_alerts.extend(self._analyze_fires(snapshot, now_str))

        # 2. Seismic / earthquake analysis
        new_alerts.extend(self._analyze_earthquakes(snapshot, now_str))

        # 3. Cross-correlation with conflict data
        new_alerts.extend(self._correlate_conflicts(snapshot, now_str))

        self.total_detections += len(new_alerts)
        self.alerts.extend(new_alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-250:]

        return new_alerts

    # ── fire analysis ─────────────────────────────────────────────────

    def _analyze_fires(self, snapshot: IntelSnapshot,
                       now_str: str) -> List[EnvAlert]:
        alerts: List[EnvAlert] = []
        zone_fires: Dict[str, List[Dict]] = {z: [] for z in CONFLICT_ZONES}
        unzoned: List[Dict] = []

        for fire in snapshot.fires:
            lat = fire.get("lat", fire.get("latitude"))
            lon = fire.get("lon", fire.get("longitude"))
            brightness = fire.get("brightness", fire.get("frp"))
            confidence = fire.get("confidence", "")

            if lat is None or lon is None:
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
            except (ValueError, TypeError):
                continue

            # Check if in conflict zone
            in_zone = False
            for zname, zinfo in CONFLICT_ZONES.items():
                dist = _haversine(lat_f, lon_f, zinfo["lat"], zinfo["lon"])
                if dist <= zinfo["radius_km"]:
                    zone_fires[zname].append({"lat": lat_f, "lon": lon_f,
                                              "brightness": brightness})
                    in_zone = True
                    break

            if not in_zone:
                unzoned.append({"lat": lat_f, "lon": lon_f, "brightness": brightness})

        # Detect scorched-earth campaigns
        for zname, fires in zone_fires.items():
            if len(fires) >= FIRE_CLUSTER_THRESHOLD:
                zinfo = CONFLICT_ZONES[zname]
                # Calculate spread
                if len(fires) >= 2:
                    max_dist = 0
                    for i, f1 in enumerate(fires):
                        for f2 in fires[i + 1:min(i + 5, len(fires))]:
                            d = _haversine(f1["lat"], f1["lon"],
                                          f2["lat"], f2["lon"])
                            max_dist = max(max_dist, d)
                else:
                    max_dist = 0

                cid = f"cluster_{zname}"
                self.fire_clusters[cid] = FireCluster(
                    cluster_id=cid,
                    center_lat=zinfo["lat"], center_lon=zinfo["lon"],
                    fire_count=len(fires),
                    region=zname,
                    in_conflict_zone=True,
                    conflict_name=zname,
                    first_seen=now_str,
                    last_seen=now_str,
                    spread_km=max_dist,
                )

                alerts.append(EnvAlert(
                    alert_id=hashlib.md5(f"fire_{zname}_{len(fires)}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="scorched_earth",
                    severity="critical" if len(fires) >= 20 else "high",
                    timestamp=now_str,
                    title=f"Scorched earth indicator: {zname} ({len(fires)} fires)",
                    description=(f"{len(fires)} fire detections in {zname} conflict zone "
                                f"(parties: {', '.join(zinfo['parties'])}). "
                                f"Spread: {max_dist:.0f}km. "
                                f"Possible deliberate scorched-earth campaign."),
                    region=zname,
                    coordinates=(zinfo["lat"], zinfo["lon"]),
                    evidence={"fire_count": len(fires), "spread_km": max_dist,
                              "parties": zinfo["parties"]},
                ))

        return alerts

    # ── earthquake analysis ───────────────────────────────────────────

    def _analyze_earthquakes(self, snapshot: IntelSnapshot,
                             now_str: str) -> List[EnvAlert]:
        alerts: List[EnvAlert] = []

        for quake in snapshot.earthquakes:
            lat = quake.get("lat", quake.get("latitude"))
            lon = quake.get("lon", quake.get("longitude"))
            mag = quake.get("mag", quake.get("magnitude"))
            depth = quake.get("depth", quake.get("depth_km"))
            place = quake.get("place", quake.get("location", ""))

            if lat is None or lon is None or mag is None:
                continue
            try:
                lat_f, lon_f, mag_f = float(lat), float(lon), float(mag)
                depth_f = float(depth) if depth is not None else 10
            except (ValueError, TypeError):
                continue

            eid = hashlib.md5(f"{lat_f}_{lon_f}_{mag_f}".encode()).hexdigest()[:10]
            event = SeismicEvent(
                event_id=eid, lat=lat_f, lon=lon_f,
                magnitude=mag_f, depth_km=depth_f, timestamp=now_str,
            )
            self.seismic_events.append(event)

            # Catastrophic event
            if mag_f >= CATASTROPHIC_QUAKE_MAGNITUDE:
                alerts.append(EnvAlert(
                    alert_id=hashlib.md5(f"quake_{eid}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="natural_disaster",
                    severity="critical",
                    timestamp=now_str,
                    title=f"CATASTROPHIC EARTHQUAKE: M{mag_f:.1f} {place or f'{lat_f:.1f},{lon_f:.1f}'}",
                    description=(f"M{mag_f:.1f} at {depth_f:.0f}km depth. "
                                f"Humanitarian crisis likely. Military/NGO response required."),
                    coordinates=(lat_f, lon_f),
                    evidence={"magnitude": mag_f, "depth_km": depth_f, "place": place},
                ))
            elif mag_f >= MAJOR_QUAKE_MAGNITUDE:
                # Check if near conflict zone
                in_conflict = False
                for zname, zinfo in CONFLICT_ZONES.items():
                    dist = _haversine(lat_f, lon_f, zinfo["lat"], zinfo["lon"])
                    if dist <= zinfo["radius_km"]:
                        in_conflict = True
                        alerts.append(EnvAlert(
                            alert_id=hashlib.md5(f"quake_cz_{eid}_{time.time()}".encode()).hexdigest()[:10],
                            alert_type="humanitarian",
                            severity="high",
                            timestamp=now_str,
                            title=f"Major quake in conflict zone: M{mag_f:.1f} near {zname}",
                            description=(f"M{mag_f:.1f} in {zname} conflict zone. "
                                        f"Humanitarian response complicated by conflict."),
                            region=zname,
                            coordinates=(lat_f, lon_f),
                            evidence={"magnitude": mag_f, "conflict_zone": zname},
                        ))
                        break

                if not in_conflict:
                    alerts.append(EnvAlert(
                        alert_id=hashlib.md5(f"quake_maj_{eid}_{time.time()}".encode()).hexdigest()[:10],
                        alert_type="seismic_threat",
                        severity="high",
                        timestamp=now_str,
                        title=f"Major earthquake: M{mag_f:.1f} {place or f'{lat_f:.1f},{lon_f:.1f}'}",
                        description=f"M{mag_f:.1f} at {depth_f:.0f}km depth. {place}.",
                        coordinates=(lat_f, lon_f),
                        evidence={"magnitude": mag_f, "depth_km": depth_f},
                    ))

        if len(self.seismic_events) > 1000:
            self.seismic_events = self.seismic_events[-500:]

        return alerts

    # ── conflict correlation ──────────────────────────────────────────

    def _correlate_conflicts(self, snapshot: IntelSnapshot,
                             now_str: str) -> List[EnvAlert]:
        """Detect environmental events correlated with conflict activity."""
        alerts: List[EnvAlert] = []

        for conflict in snapshot.conflicts:
            lat = conflict.get("lat", conflict.get("latitude"))
            lon = conflict.get("lon", conflict.get("longitude"))
            event_type = conflict.get("event_type", conflict.get("type", ""))
            description = conflict.get("description", conflict.get("notes", ""))

            if lat is None or lon is None:
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
            except (ValueError, TypeError):
                continue

            full_text = f"{event_type} {description}".lower()

            # Check for environmental warfare indicators
            env_keywords = ["dam", "flood", "water", "forest fire", "burn",
                           "chemical", "toxic", "nuclear", "radiation",
                           "crop", "harvest", "deforest", "mine"]

            matched = [k for k in env_keywords if k in full_text]
            if matched:
                alerts.append(EnvAlert(
                    alert_id=hashlib.md5(f"env_war_{lat_f}_{lon_f}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="climate_conflict",
                    severity="high",
                    timestamp=now_str,
                    title=f"Environmental warfare indicator: {', '.join(matched[:3])}",
                    description=(f"Conflict event with environmental dimensions: "
                                f"{description[:150] if description else event_type}. "
                                f"Keywords: {', '.join(matched)}"),
                    coordinates=(lat_f, lon_f),
                    evidence={"keywords": matched, "event_type": event_type},
                ))

        return alerts

    # ── queries ───────────────────────────────────────────────────────

    def get_fire_map(self) -> List[FireCluster]:
        return sorted(self.fire_clusters.values(),
                      key=lambda c: c.fire_count, reverse=True)

    def get_conflict_zone_fires(self) -> List[FireCluster]:
        return [c for c in self.fire_clusters.values() if c.in_conflict_zone]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_detections": self.total_detections,
            "fire_clusters": len(self.fire_clusters),
            "conflict_zone_fires": sum(1 for c in self.fire_clusters.values()
                                       if c.in_conflict_zone),
            "seismic_events": len(self.seismic_events),
            "major_quakes": sum(1 for e in self.seismic_events
                               if e.magnitude >= MAJOR_QUAKE_MAGNITUDE),
            "total_alerts": len(self.alerts),
        }
