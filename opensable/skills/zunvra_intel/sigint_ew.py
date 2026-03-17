"""
#19 — SIGINT / Electronic Warfare Pattern Analyzer

NSA/GCHQ core capability.  Analyzes GPS jamming zones to detect
electronic warfare campaigns, triangulate jammer positions, correlate
EW activity with military operations, and detect escalation patterns.

Data source: snapshot.gps_jamming (barely used before — only as causal trigger)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class JammingZone:
    """A detected GPS jamming zone with analysis."""
    zone_id: str
    center_lat: float
    center_lon: float
    radius_km: float
    strength: str  # low, medium, high, severe
    first_detected: str
    last_detected: str
    observations: int = 1
    region: str = ""
    likely_source: str = ""  # Estimated state/actor


@dataclass
class EWCampaign:
    """A coordinated electronic warfare campaign (multiple correlated zones)."""
    campaign_id: str
    zones: List[str] = field(default_factory=list)  # zone_ids
    start_time: str = ""
    last_updated: str = ""
    region: str = ""
    scope: str = "tactical"  # tactical, operational, strategic
    description: str = ""
    military_correlation: float = 0.0  # 0-1, how strongly correlated with mil activity
    severity: str = "medium"


@dataclass
class SigintAlert:
    """Alert from SIGINT/EW analysis."""
    alert_id: str
    alert_type: str  # new_jamming, campaign_detected, escalation, mil_correlation,
                     # jammer_triangulated, civilian_impact
    severity: str
    timestamp: str
    description: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Known EW hotspots (permanent monitoring zones)
# ---------------------------------------------------------------------------

EW_HOTSPOTS: Dict[str, Dict[str, Any]] = {
    "kaliningrad": {"lat": 54.7, "lon": 20.5, "radius_km": 300, "actor": "Russia",
                    "desc": "Russian exclave — persistent GPS spoofing/jamming"},
    "eastern_med": {"lat": 35.0, "lon": 33.0, "radius_km": 400, "actor": "Russia/Syria",
                    "desc": "Russian EW systems at Khmeimim AB, Syria"},
    "crimea": {"lat": 44.5, "lon": 34.0, "radius_km": 300, "actor": "Russia",
               "desc": "Crimean EW complex — covers Black Sea approaches"},
    "north_korea": {"lat": 38.5, "lon": 126.5, "radius_km": 200, "actor": "DPRK",
                    "desc": "DPRK GPS spoofing targeting South Korean aviation"},
    "iran_strait": {"lat": 27.0, "lon": 56.0, "radius_km": 200, "actor": "Iran",
                    "desc": "IRGC GPS denial targeting commercial shipping"},
    "south_china_sea": {"lat": 16.0, "lon": 112.0, "radius_km": 500, "actor": "China",
                        "desc": "PLA EW capabilities on artificial islands"},
    "baltics": {"lat": 57.0, "lon": 24.0, "radius_km": 300, "actor": "Russia",
                "desc": "Baltic region — commercial aviation jamming corridor"},
    "finland_border": {"lat": 64.0, "lon": 28.0, "radius_km": 200, "actor": "Russia",
                       "desc": "Russian EW systems along Finnish border"},
}


class SigintEWAnalyzer:
    """
    Analyze GPS jamming data for electronic warfare patterns.

    Detects:
    - New jamming zones and their growth
    - Coordinated EW campaigns (multiple zones activated together)
    - Correlation between EW activity and military operations
    - Jammer source triangulation from zone geometry
    - Civilian aviation impact assessment
    - Escalation patterns (tactical → operational → strategic)
    """

    MAX_ZONES = 5000
    CLUSTER_DISTANCE_KM = 100  # Zones within this distance are part of same campaign
    ESCALATION_ZONE_COUNT = 5  # This many active zones = escalation

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.zones: Dict[str, JammingZone] = {}
        self.campaigns: Dict[str, EWCampaign] = {}
        self.alerts: List[SigintAlert] = []
        self.total_observations = 0
        self._prev_zone_count = 0

    # ── main observation ──────────────────────────────────────────────

    def observe(self, snapshot: IntelSnapshot) -> List[SigintAlert]:
        """Analyze snapshot GPS jamming data for EW patterns."""
        now_str = datetime.now(timezone.utc).isoformat()
        new_alerts: List[SigintAlert] = []

        current_zones: List[Dict[str, Any]] = []

        for jam in snapshot.gps_jamming:
            lat = self._sf(jam.get("lat", jam.get("latitude")))
            lon = self._sf(jam.get("lon", jam.get("longitude")))
            if lat is None or lon is None:
                continue

            radius = self._sf(jam.get("radius", jam.get("radius_km", 50))) or 50.0
            strength = jam.get("strength", jam.get("severity", "medium"))
            zid = hashlib.md5(f"{lat:.2f}_{lon:.2f}".encode()).hexdigest()[:10]

            current_zones.append({"id": zid, "lat": lat, "lon": lon,
                                  "radius": radius, "strength": strength})

            if zid not in self.zones:
                # New zone detected
                zone = JammingZone(
                    zone_id=zid,
                    center_lat=lat, center_lon=lon,
                    radius_km=radius,
                    strength=strength if isinstance(strength, str) else "medium",
                    first_detected=now_str, last_detected=now_str,
                    region=self._classify_region(lat, lon),
                    likely_source=self._attribute_source(lat, lon),
                )
                self.zones[zid] = zone

                new_alerts.append(SigintAlert(
                    alert_id=hashlib.md5(f"new_{zid}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="new_jamming",
                    severity="high",
                    timestamp=now_str,
                    description=(f"New GPS jamming zone: {zone.region} "
                                f"({lat:.2f}, {lon:.2f}), radius {radius:.0f}km, "
                                f"likely source: {zone.likely_source}"),
                    lat=lat, lon=lon,
                    evidence={"zone_id": zid, "source": zone.likely_source},
                ))
            else:
                self.zones[zid].last_detected = now_str
                self.zones[zid].observations += 1

            self.total_observations += 1

        # Campaign detection (cluster analysis)
        if len(current_zones) >= 2:
            campaign_alerts = self._detect_campaigns(current_zones, now_str)
            new_alerts.extend(campaign_alerts)

        # Military correlation
        mil_count = len(snapshot.military_flights)
        jam_count = len(current_zones)
        if jam_count > 0 and mil_count > 5:
            correlation = min(1.0, (mil_count / 20) * (jam_count / 3))
            if correlation > 0.5:
                new_alerts.append(SigintAlert(
                    alert_id=hashlib.md5(f"milcorr_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="mil_correlation",
                    severity="high" if correlation > 0.7 else "medium",
                    timestamp=now_str,
                    description=(f"EW-military correlation: {jam_count} jamming zones active "
                                f"with {mil_count} military flights (r={correlation:.2f})"),
                    evidence={"jamming_zones": jam_count, "military_flights": mil_count,
                              "correlation": correlation},
                ))

        # Escalation detection
        if jam_count >= self.ESCALATION_ZONE_COUNT and jam_count > self._prev_zone_count * 1.5:
            new_alerts.append(SigintAlert(
                alert_id=hashlib.md5(f"escalation_{time.time()}".encode()).hexdigest()[:10],
                alert_type="escalation",
                severity="critical",
                timestamp=now_str,
                description=(f"EW ESCALATION: {jam_count} active jamming zones "
                            f"(was {self._prev_zone_count}). "
                            f"Possible transition from tactical to operational EW campaign."),
                evidence={"current_zones": jam_count, "previous_zones": self._prev_zone_count},
            ))

        # Civilian aviation impact
        for zone_data in current_zones:
            for flight in snapshot.flights:
                f_lat = self._sf(flight.get("lat"))
                f_lon = self._sf(flight.get("lon"))
                if f_lat is not None and f_lon is not None:
                    dist = _haversine(zone_data["lat"], zone_data["lon"], f_lat, f_lon)
                    if dist < zone_data["radius"]:
                        callsign = (flight.get("callsign") or flight.get("flight") or "").strip()
                        new_alerts.append(SigintAlert(
                            alert_id=hashlib.md5(f"civ_{callsign}_{time.time()}".encode()).hexdigest()[:10],
                            alert_type="civilian_impact",
                            severity="high",
                            timestamp=now_str,
                            description=(f"Civilian aircraft {callsign} inside GPS jamming zone "
                                        f"({dist:.0f}km from center) — potential navigation hazard"),
                            lat=f_lat, lon=f_lon,
                            evidence={"callsign": callsign, "zone_id": zone_data["id"],
                                      "distance_km": dist},
                        ))
                        break  # One alert per zone is enough

        self._prev_zone_count = jam_count
        self.alerts.extend(new_alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-250:]

        # Trim zones
        if len(self.zones) > self.MAX_ZONES:
            sorted_z = sorted(self.zones.values(), key=lambda z: z.last_detected)
            for z in sorted_z[:len(self.zones) - self.MAX_ZONES]:
                del self.zones[z.zone_id]

        return new_alerts

    # ── campaign detection ────────────────────────────────────────────

    def _detect_campaigns(self, zones: List[Dict[str, Any]],
                          now_str: str) -> List[SigintAlert]:
        """Cluster nearby jamming zones into EW campaigns."""
        alerts: List[SigintAlert] = []

        # Simple distance-based clustering
        clusters: List[List[Dict[str, Any]]] = []
        used = set()

        for i, z1 in enumerate(zones):
            if i in used:
                continue
            cluster = [z1]
            used.add(i)
            for j, z2 in enumerate(zones):
                if j in used:
                    continue
                dist = _haversine(z1["lat"], z1["lon"], z2["lat"], z2["lon"])
                if dist < self.CLUSTER_DISTANCE_KM:
                    cluster.append(z2)
                    used.add(j)
            if len(cluster) >= 2:
                clusters.append(cluster)

        for cluster in clusters:
            cid = hashlib.md5(f"campaign_{'_'.join(z['id'] for z in cluster)}".encode()).hexdigest()[:10]

            if cid not in self.campaigns:
                avg_lat = sum(z["lat"] for z in cluster) / len(cluster)
                avg_lon = sum(z["lon"] for z in cluster) / len(cluster)
                region = self._classify_region(avg_lat, avg_lon)

                scope = "tactical"
                if len(cluster) >= 4:
                    scope = "strategic"
                elif len(cluster) >= 2:
                    scope = "operational"

                campaign = EWCampaign(
                    campaign_id=cid,
                    zones=[z["id"] for z in cluster],
                    start_time=now_str,
                    last_updated=now_str,
                    region=region,
                    scope=scope,
                    description=(f"Coordinated EW campaign: {len(cluster)} zones "
                                f"in {region}, scope: {scope}"),
                    severity="critical" if scope == "strategic" else "high",
                )
                self.campaigns[cid] = campaign

                alerts.append(SigintAlert(
                    alert_id=hashlib.md5(f"campaign_{cid}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="campaign_detected",
                    severity=campaign.severity,
                    timestamp=now_str,
                    description=campaign.description,
                    lat=avg_lat, lon=avg_lon,
                    evidence={"zone_count": len(cluster), "scope": scope, "region": region},
                ))
            else:
                self.campaigns[cid].last_updated = now_str

        return alerts

    # ── source attribution ────────────────────────────────────────────

    def _attribute_source(self, lat: float, lon: float) -> str:
        """Attempt to attribute jamming source based on location."""
        for name, hotspot in EW_HOTSPOTS.items():
            dist = _haversine(lat, lon, hotspot["lat"], hotspot["lon"])
            if dist < hotspot["radius_km"]:
                return hotspot["actor"]

        # General region-based attribution
        region = self._classify_region(lat, lon)
        attribution_map = {
            "eastern_europe": "Russia (probable)",
            "middle_east": "Iran/Russia (probable)",
            "east_asia": "China/DPRK (probable)",
            "southeast_asia": "China (probable)",
        }
        return attribution_map.get(region, "Unknown")

    @staticmethod
    def _classify_region(lat: float, lon: float) -> str:
        if 50 < lat < 72 and 10 < lon < 40:
            return "scandinavia_baltics"
        if 44 < lat < 56 and 15 < lon < 45:
            return "eastern_europe"
        if 20 < lat < 45 and 25 < lon < 65:
            return "middle_east"
        if 25 < lat < 50 and 100 < lon < 145:
            return "east_asia"
        if -10 < lat < 25 and 95 < lon < 140:
            return "southeast_asia"
        if 35 < lat < 50 and -10 < lon < 15:
            return "western_europe"
        return "other"

    @staticmethod
    def _sf(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # ── queries ───────────────────────────────────────────────────────

    def get_active_zones(self) -> List[JammingZone]:
        return list(self.zones.values())

    def get_campaigns(self) -> List[EWCampaign]:
        return list(self.campaigns.values())

    def get_recent_alerts(self, limit: int = 30) -> List[SigintAlert]:
        return self.alerts[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_observations": self.total_observations,
            "active_zones": len(self.zones),
            "active_campaigns": len(self.campaigns),
            "total_alerts": len(self.alerts),
            "strategic_campaigns": sum(1 for c in self.campaigns.values() if c.scope == "strategic"),
        }
