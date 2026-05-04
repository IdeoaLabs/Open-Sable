"""
#21,  Nuclear Proliferation Monitor

IAEA / DIA / NGA capability.  Tracks nuclear facility status, detects
seismic events correlated to weapons tests, monitors enrichment
indicators, and flags treaty-violation signals.

Data sources: snapshot.nuclear_facilities, snapshot.earthquakes,
              snapshot.military_flights, snapshot.satellites
"""

from __future__ import annotations

import hashlib
import json
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
# Known nuclear-capable / monitored sites
# ---------------------------------------------------------------------------

KNOWN_NUCLEAR_SITES: Dict[str, Dict[str, Any]] = {
    "natanz": {"lat": 33.72, "lon": 51.73, "country": "Iran", "type": "enrichment"},
    "fordow": {"lat": 34.88, "lon": 51.60, "country": "Iran", "type": "enrichment"},
    "yongbyon": {"lat": 39.79, "lon": 125.75, "country": "North Korea", "type": "reactor"},
    "punggye_ri": {"lat": 41.28, "lon": 129.10, "country": "North Korea", "type": "test_site"},
    "zaporizhzhia": {"lat": 47.51, "lon": 34.59, "country": "Ukraine", "type": "power"},
    "bushehr": {"lat": 28.83, "lon": 50.89, "country": "Iran", "type": "power"},
    "dimona": {"lat": 31.00, "lon": 35.15, "country": "Israel", "type": "research"},
    "kahuta": {"lat": 33.60, "lon": 73.39, "country": "Pakistan", "type": "enrichment"},
    "khushab": {"lat": 32.02, "lon": 72.22, "country": "Pakistan", "type": "reactor"},
    "bhabha": {"lat": 19.00, "lon": 72.92, "country": "India", "type": "research"},
    "sarov": {"lat": 54.93, "lon": 43.32, "country": "Russia", "type": "weapons_lab"},
    "lop_nur": {"lat": 41.55, "lon": 88.35, "country": "China", "type": "test_site"},
}

SEISMIC_TEST_SIGNATURES = {
    "depth_km_max": 10.0,       # Nuclear tests are shallow
    "magnitude_min": 4.0,       # Detect above 4.0
    "mb_ms_ratio_flag": 1.5,    # Body-wave to surface-wave ratio
}

PROXIMITY_KM_FACILITY = 150.0   # Alert if quake within 150 km of nuclear site
PROXIMITY_KM_TEST = 50.0        # Correlate as possible test within 50 km


@dataclass
class NuclearAlert:
    """Nuclear proliferation alert."""
    alert_id: str
    alert_type: str  # seismic_test, facility_anomaly, enrichment_indicator,
                     # missile_correlation, safety_risk, treaty_violation
    severity: str   # critical, high, medium, low
    timestamp: str
    title: str
    description: str
    facility: Optional[str] = None
    country: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FacilityStatus:
    """Tracked nuclear facility status."""
    name: str
    country: str
    facility_type: str
    lat: float
    lon: float
    status: str = "normal"     # normal, alert, elevated, critical
    last_activity: str = ""
    alerts_count: int = 0
    nearby_seismic: int = 0
    nearby_military: int = 0


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class NuclearMonitor:
    """
    Nuclear proliferation monitoring engine.

    Combines nuclear facility data, seismic events, military flight
    patterns, and satellite imagery indicators to detect proliferation
    activities, weapons tests, and safety risks.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.alerts: List[NuclearAlert] = []
        self.facilities: Dict[str, FacilityStatus] = {}
        self._seismic_history: List[Dict[str, Any]] = []
        self.total_detections = 0

        # Initialize known sites
        for name, info in KNOWN_NUCLEAR_SITES.items():
            self.facilities[name] = FacilityStatus(
                name=name,
                country=info["country"],
                facility_type=info["type"],
                lat=info["lat"],
                lon=info["lon"],
            )

    def observe(self, snapshot: IntelSnapshot) -> List[NuclearAlert]:
        """Analyze snapshot for nuclear proliferation indicators."""
        now_str = datetime.now(timezone.utc).isoformat()
        new_alerts: List[NuclearAlert] = []

        # 1. Seismic-nuclear correlation
        new_alerts.extend(self._analyze_seismic(snapshot, now_str))

        # 2. Nuclear facility status changes
        new_alerts.extend(self._analyze_facilities(snapshot, now_str))

        # 3. Military flight correlation near nuclear sites
        new_alerts.extend(self._check_military_proximity(snapshot, now_str))

        # 4. Satellite overpass correlation
        new_alerts.extend(self._check_satellite_interest(snapshot, now_str))

        self.total_detections += len(new_alerts)
        self.alerts.extend(new_alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-250:]

        return new_alerts

    # ── seismic / weapons test ────────────────────────────────────────

    def _analyze_seismic(self, snapshot: IntelSnapshot,
                         now_str: str) -> List[NuclearAlert]:
        alerts: List[NuclearAlert] = []

        for quake in snapshot.earthquakes:
            lat = quake.get("lat", quake.get("latitude"))
            lon = quake.get("lon", quake.get("longitude"))
            mag = quake.get("mag", quake.get("magnitude"))
            depth = quake.get("depth", quake.get("depth_km"))

            if lat is None or lon is None or mag is None:
                continue
            try:
                lat, lon, mag = float(lat), float(lon), float(mag)
                depth = float(depth) if depth is not None else 999
            except (ValueError, TypeError):
                continue

            # Check proximity to known nuclear sites
            for fname, fac in self.facilities.items():
                dist = _haversine(lat, lon, fac.lat, fac.lon)

                if dist <= PROXIMITY_KM_TEST:
                    # Shallow + near test site = possible nuclear test
                    if (depth <= SEISMIC_TEST_SIGNATURES["depth_km_max"] and
                            mag >= SEISMIC_TEST_SIGNATURES["magnitude_min"] and
                            fac.facility_type in ("test_site", "weapons_lab")):
                        fac.nearby_seismic += 1
                        fac.status = "critical"
                        fac.alerts_count += 1
                        alerts.append(NuclearAlert(
                            alert_id=hashlib.md5(f"test_{fname}_{mag}_{time.time()}".encode()).hexdigest()[:10],
                            alert_type="seismic_test",
                            severity="critical",
                            timestamp=now_str,
                            title=(f"POSSIBLE NUCLEAR TEST: M{mag:.1f} at {depth:.0f}km depth "
                                   f"near {fname} ({fac.country})"),
                            description=(f"Seismic event M{mag:.1f} at depth {depth:.0f}km, "
                                        f"{dist:.0f}km from {fname}. Signature consistent "
                                        f"with underground nuclear detonation."),
                            facility=fname,
                            country=fac.country,
                            coordinates=(lat, lon),
                            evidence={"magnitude": mag, "depth_km": depth,
                                      "distance_km": dist, "facility_type": fac.facility_type},
                        ))
                    elif dist <= PROXIMITY_KM_FACILITY:
                        # General seismic near nuclear facility (safety concern)
                        fac.nearby_seismic += 1
                        if mag >= 5.0:
                            fac.status = "elevated"
                            alerts.append(NuclearAlert(
                                alert_id=hashlib.md5(f"quake_{fname}_{mag}".encode()).hexdigest()[:10],
                                alert_type="safety_risk",
                                severity="high" if mag >= 6.0 else "medium",
                                timestamp=now_str,
                                title=f"Seismic risk: M{mag:.1f} near {fname} nuclear facility",
                                description=(f"M{mag:.1f} earthquake {dist:.0f}km from {fname} "
                                            f"({fac.facility_type} - {fac.country}). "
                                            f"Safety assessment required."),
                                facility=fname,
                                country=fac.country,
                                coordinates=(lat, lon),
                                evidence={"magnitude": mag, "depth_km": depth,
                                          "distance_km": dist},
                            ))

            self._seismic_history.append({
                "lat": lat, "lon": lon, "mag": mag, "depth": depth, "time": now_str
            })

        if len(self._seismic_history) > 1000:
            self._seismic_history = self._seismic_history[-500:]

        return alerts

    # ── facility data ─────────────────────────────────────────────────

    def _analyze_facilities(self, snapshot: IntelSnapshot,
                            now_str: str) -> List[NuclearAlert]:
        alerts: List[NuclearAlert] = []

        for nf in snapshot.nuclear_facilities:
            name = (nf.get("name") or nf.get("facility_name") or "").lower().replace(" ", "_")
            status = nf.get("status", nf.get("operational_status", ""))
            lat = nf.get("lat", nf.get("latitude"))
            lon = nf.get("lon", nf.get("longitude"))
            country = nf.get("country", "")

            if not name:
                continue

            # Register new facility if unknown
            if name not in self.facilities and lat is not None and lon is not None:
                try:
                    self.facilities[name] = FacilityStatus(
                        name=name, country=country,
                        facility_type=nf.get("type", "unknown"),
                        lat=float(lat), lon=float(lon),
                    )
                except (ValueError, TypeError):
                    continue

            fac = self.facilities.get(name)
            if not fac:
                continue

            fac.last_activity = now_str

            # Check for status changes
            if status:
                status_lower = status.lower()
                if any(kw in status_lower for kw in ("alert", "incident", "scram",
                                                        "shutdown", "emergency")):
                    fac.status = "alert"
                    fac.alerts_count += 1
                    alerts.append(NuclearAlert(
                        alert_id=hashlib.md5(f"fac_{name}_{status}_{time.time()}".encode()).hexdigest()[:10],
                        alert_type="facility_anomaly",
                        severity="high",
                        timestamp=now_str,
                        title=f"Nuclear facility alert: {name} ({country})",
                        description=f"Facility {name} status: {status}. Potential safety event.",
                        facility=name,
                        country=fac.country,
                        evidence={"status": status, "raw": nf},
                    ))

            # Enrichment indicator: new construction / expansion
            description = json.dumps(nf).lower()
            if any(kw in description for kw in ("centrifuge", "enrichment",
                                                   "construction", "expansion",
                                                   "new cascade", "ir-6", "ir-9")):
                alerts.append(NuclearAlert(
                    alert_id=hashlib.md5(f"enrich_{name}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="enrichment_indicator",
                    severity="high",
                    timestamp=now_str,
                    title=f"Enrichment activity: {name} ({country})",
                    description=f"Data indicates enrichment/construction activity at {name}.",
                    facility=name,
                    country=fac.country,
                    evidence={"keywords_matched": True, "raw": nf},
                ))

        return alerts

    # ── military flights near nuclear sites ───────────────────────────

    def _check_military_proximity(self, snapshot: IntelSnapshot,
                                  now_str: str) -> List[NuclearAlert]:
        alerts: List[NuclearAlert] = []
        mil_near_sites: Dict[str, int] = {}

        for flight in snapshot.military_flights:
            lat = flight.get("lat", flight.get("latitude"))
            lon = flight.get("lon", flight.get("longitude"))
            callsign = flight.get("callsign", "")
            ftype = flight.get("type", flight.get("aircraft_type", ""))

            if lat is None or lon is None:
                continue
            try:
                lat, lon = float(lat), float(lon)
            except (ValueError, TypeError):
                continue

            for fname, fac in self.facilities.items():
                dist = _haversine(lat, lon, fac.lat, fac.lon)
                if dist <= PROXIMITY_KM_FACILITY:
                    fac.nearby_military += 1
                    key = fname
                    mil_near_sites[key] = mil_near_sites.get(key, 0) + 1

        # Alert on significant military presence near nuclear sites
        for fname, count in mil_near_sites.items():
            if count >= 3:
                fac = self.facilities[fname]
                alerts.append(NuclearAlert(
                    alert_id=hashlib.md5(f"mil_{fname}_{count}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="missile_correlation",
                    severity="high" if count >= 5 else "medium",
                    timestamp=now_str,
                    title=f"Military activity near {fname}: {count} flights",
                    description=(f"{count} military flights detected within "
                                f"{PROXIMITY_KM_FACILITY}km of {fname} ({fac.country}). "
                                f"Possible surveillance, strike prep, or exercise."),
                    facility=fname,
                    country=fac.country,
                    evidence={"military_flights": count},
                ))

        return alerts

    # ── satellite interest ────────────────────────────────────────────

    def _check_satellite_interest(self, snapshot: IntelSnapshot,
                                  now_str: str) -> List[NuclearAlert]:
        alerts: List[NuclearAlert] = []

        for sat in snapshot.satellites:
            lat = sat.get("lat", sat.get("latitude"))
            lon = sat.get("lon", sat.get("longitude"))
            sat_name = sat.get("name", sat.get("satellite_name", ""))

            if lat is None or lon is None:
                continue
            try:
                lat, lon = float(lat), float(lon)
            except (ValueError, TypeError):
                continue

            for fname, fac in self.facilities.items():
                dist = _haversine(lat, lon, fac.lat, fac.lon)
                if dist <= 50:  # Satellite directly over nuclear site
                    alerts.append(NuclearAlert(
                        alert_id=hashlib.md5(f"sat_{fname}_{sat_name}_{time.time()}".encode()).hexdigest()[:10],
                        alert_type="facility_anomaly",
                        severity="low",
                        timestamp=now_str,
                        title=f"Satellite overpass: {sat_name} over {fname}",
                        description=(f"Satellite {sat_name} detected within 50km of "
                                    f"{fname} ({fac.country}). Possible IMINT collection."),
                        facility=fname,
                        country=fac.country,
                        evidence={"satellite": sat_name, "distance_km": dist},
                    ))

        return alerts

    # ── queries ───────────────────────────────────────────────────────

    def get_facility_status(self) -> Dict[str, FacilityStatus]:
        return self.facilities

    def get_critical_facilities(self) -> List[FacilityStatus]:
        return [f for f in self.facilities.values() if f.status in ("alert", "critical")]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_detections": self.total_detections,
            "tracked_facilities": len(self.facilities),
            "facilities_on_alert": sum(1 for f in self.facilities.values()
                                       if f.status in ("alert", "critical", "elevated")),
            "seismic_events_tracked": len(self._seismic_history),
            "total_alerts": len(self.alerts),
        }
