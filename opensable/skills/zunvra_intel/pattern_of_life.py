"""
#12,  Pattern of Life Analyzer

Track behavioral baselines for every entity over time.  "This aircraft
normally flies Mon-Fri between Zurich and London at FL350.  Today it
deviated to Tripoli at FL420,  ANOMALOUS."

This is what NSA's SKYNET program does for cell phone metadata, and what
Palantir Gotham does for vessel tracking.  We do it across ALL entity types
simultaneously with fully automated baseline computation.

Detects:
  - Route deviations (aircraft, vessels)
  - Schedule anomalies (appears at unusual times)
  - Speed/altitude anomalies
  - Zone violations (entity enters zone it's never been in)
  - Behavioral clustering changes
  - Ghost periods (entity disappears then reappears)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class BehaviorBaseline:
    """Learned behavioral baseline for one entity."""
    entity_id: str
    entity_type: str
    label: str
    total_observations: int = 0

    # Position history (last N lat/lon pairs)
    position_history: List[Tuple[float, float]] = field(default_factory=list)
    # Typical zones
    typical_zones: List[str] = field(default_factory=list)
    # Altitude/speed baselines (for aircraft)
    avg_altitude: float = 0.0
    avg_speed: float = 0.0
    altitude_std: float = 0.0
    speed_std: float = 0.0
    # Timing patterns
    hours_active: List[int] = field(default_factory=list)  # histogram of hours [0-23]
    days_active: List[int] = field(default_factory=list)   # histogram of days [0-6]
    # Route patterns (top destinations / callsigns)
    common_routes: Dict[str, int] = field(default_factory=dict)
    # Centroid of normal operating area
    centroid_lat: float = 0.0
    centroid_lon: float = 0.0
    max_range_km: float = 0.0  # max observed distance from centroid
    # Ghost tracking
    last_seen: str = ""
    consecutive_misses: int = 0  # How many cycles entity not seen
    max_gap_hours: float = 0.0


@dataclass
class PoLAnomaly:
    """A deviation from normal pattern of life."""
    anomaly_id: str
    entity_id: str
    entity_type: str
    label: str
    anomaly_type: str  # route_deviation, schedule_anomaly, speed_anomaly,
                       # altitude_anomaly, zone_violation, ghost_reappearance,
                       # behavioral_shift
    severity: str      # low, medium, high, critical
    description: str
    normal_value: str
    observed_value: str
    deviation_sigma: float = 0.0
    timestamp: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None


# ---------------------------------------------------------------------------
# Pattern of Life Engine
# ---------------------------------------------------------------------------

class PatternOfLifeAnalyzer:
    """
    Builds behavioral baselines for every tracked entity and detects
    deviations from normal patterns.

    This is the core of what makes Palantir's vessel/aircraft tracking
    so powerful,  and what NSA programs like SKYNET use for pattern
    analysis of communication metadata.
    """

    MAX_ENTITIES = 10000
    MAX_POSITION_HISTORY = 100
    MIN_OBS_FOR_BASELINE = 5  # Need at least 5 observations

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "pattern_of_life.json"

        self.baselines: Dict[str, BehaviorBaseline] = {}
        self.total_anomalies_detected = 0
        self.total_observations = 0
        self._load_state()

    # ── main entry point ──────────────────────────────────────────────

    def observe(self, snapshot: IntelSnapshot) -> List[PoLAnomaly]:
        """
        Observe a new snapshot: update all baselines and detect anomalies.

        Returns list of detected deviations from normal behavior.
        """
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        hour = now.hour
        day = now.weekday()
        anomalies: List[PoLAnomaly] = []

        # Process aircraft
        for flight in snapshot.flights + snapshot.military_flights:
            icao = flight.get("hex", flight.get("icao", ""))
            if not icao:
                continue
            eid = f"aircraft_{icao}"
            callsign = flight.get("callsign", flight.get("flight", "")).strip()
            lat = self._safe_float(flight.get("lat"))
            lon = self._safe_float(flight.get("lon"))
            alt = self._safe_float(flight.get("alt_baro", flight.get("altitude")))
            speed = self._safe_float(flight.get("gs", flight.get("speed")))

            baseline = self._get_or_create(eid, "aircraft", callsign or icao)
            new_anomalies = self._update_aircraft(baseline, lat, lon, alt, speed,
                                                   callsign, hour, day, now_str)
            anomalies.extend(new_anomalies)

        # Process vessels
        for ship in snapshot.ships:
            mmsi = str(ship.get("mmsi", ship.get("MMSI", "")))
            if not mmsi:
                continue
            eid = f"vessel_{mmsi}"
            name = ship.get("name", ship.get("ship_name", ""))
            lat = self._safe_float(ship.get("lat"))
            lon = self._safe_float(ship.get("lon"))
            speed = self._safe_float(ship.get("speed", ship.get("sog")))
            dest = ship.get("destination", "")

            baseline = self._get_or_create(eid, "vessel", name or mmsi)
            new_anomalies = self._update_vessel(baseline, lat, lon, speed, dest,
                                                 hour, day, now_str)
            anomalies.extend(new_anomalies)

        # Detect ghost entities (entities that disappeared)
        seen_ids = set()
        for flight in snapshot.flights + snapshot.military_flights:
            icao = flight.get("hex", flight.get("icao", ""))
            if icao:
                seen_ids.add(f"aircraft_{icao}")
        for ship in snapshot.ships:
            mmsi = str(ship.get("mmsi", ship.get("MMSI", "")))
            if mmsi:
                seen_ids.add(f"vessel_{mmsi}")

        for eid, baseline in self.baselines.items():
            if eid not in seen_ids and baseline.total_observations >= self.MIN_OBS_FOR_BASELINE:
                baseline.consecutive_misses += 1
                if baseline.consecutive_misses == 5:  # First time going dark
                    anomalies.append(PoLAnomaly(
                        anomaly_id=hashlib.md5(f"ghost_{eid}_{now_str}".encode()).hexdigest()[:10],
                        entity_id=eid,
                        entity_type=baseline.entity_type,
                        label=baseline.label,
                        anomaly_type="ghost_period",
                        severity="medium",
                        description=f"{baseline.label} has gone dark (not seen for {baseline.consecutive_misses} cycles)",
                        normal_value=f"Seen every cycle for {baseline.total_observations} observations",
                        observed_value="Not present",
                        timestamp=now_str,
                    ))
            elif eid in seen_ids and baseline.consecutive_misses >= 5:
                # Ghost reappearance
                anomalies.append(PoLAnomaly(
                    anomaly_id=hashlib.md5(f"reappear_{eid}_{now_str}".encode()).hexdigest()[:10],
                    entity_id=eid,
                    entity_type=baseline.entity_type,
                    label=baseline.label,
                    anomaly_type="ghost_reappearance",
                    severity="high",
                    description=f"{baseline.label} reappeared after {baseline.consecutive_misses} cycles dark",
                    normal_value="Present every cycle",
                    observed_value=f"Dark for {baseline.consecutive_misses} cycles, now back",
                    timestamp=now_str,
                ))
                baseline.consecutive_misses = 0

        self.total_observations += 1
        self.total_anomalies_detected += len(anomalies)

        # Enforce limits
        if len(self.baselines) > self.MAX_ENTITIES:
            oldest = sorted(self.baselines.items(),
                          key=lambda x: x[1].total_observations)
            for eid, _ in oldest[:1000]:
                del self.baselines[eid]

        self._save_state()
        return anomalies

    # ── entity-specific update logic ──────────────────────────────────

    def _update_aircraft(self, bl: BehaviorBaseline,
                         lat: Optional[float], lon: Optional[float],
                         alt: Optional[float], speed: Optional[float],
                         callsign: str, hour: int, day: int,
                         now: str) -> List[PoLAnomaly]:
        anomalies: List[PoLAnomaly] = []
        bl.total_observations += 1
        bl.last_seen = now

        # Update timing
        if len(bl.hours_active) < 24:
            bl.hours_active = [0] * 24
        if len(bl.days_active) < 7:
            bl.days_active = [0] * 7
        bl.hours_active[hour] += 1
        bl.days_active[day] += 1

        # Position-based anomalies
        if lat is not None and lon is not None:
            bl.position_history.append((lat, lon))
            if len(bl.position_history) > self.MAX_POSITION_HISTORY:
                bl.position_history = bl.position_history[-self.MAX_POSITION_HISTORY:]

            # Update centroid
            if len(bl.position_history) >= 3:
                avg_lat = sum(p[0] for p in bl.position_history) / len(bl.position_history)
                avg_lon = sum(p[1] for p in bl.position_history) / len(bl.position_history)
                bl.centroid_lat = avg_lat
                bl.centroid_lon = avg_lon

                dist = _haversine(lat, lon, avg_lat, avg_lon)
                bl.max_range_km = max(bl.max_range_km, dist)

                # Zone violation: entity far from centroid
                if bl.total_observations >= self.MIN_OBS_FOR_BASELINE and bl.max_range_km > 0:
                    if dist > bl.max_range_km * 2.0 and dist > 500:
                        anomalies.append(PoLAnomaly(
                            anomaly_id=hashlib.md5(f"zone_{bl.entity_id}_{now}".encode()).hexdigest()[:10],
                            entity_id=bl.entity_id,
                            entity_type="aircraft",
                            label=bl.label,
                            anomaly_type="zone_violation",
                            severity="high",
                            description=(f"{bl.label} is {dist:.0f}km from normal operating area "
                                       f"(centroid: {avg_lat:.1f}, {avg_lon:.1f})"),
                            normal_value=f"Max range: {bl.max_range_km:.0f}km",
                            observed_value=f"Current distance: {dist:.0f}km",
                            deviation_sigma=dist / bl.max_range_km if bl.max_range_km > 0 else 0,
                            timestamp=now,
                            lat=lat, lon=lon,
                        ))

        # Altitude anomaly
        if alt is not None and alt > 0:
            if bl.total_observations >= self.MIN_OBS_FOR_BASELINE and bl.avg_altitude > 0:
                z_score = abs(alt - bl.avg_altitude) / bl.altitude_std if bl.altitude_std > 0 else 0
                if z_score > 3.0:
                    anomalies.append(PoLAnomaly(
                        anomaly_id=hashlib.md5(f"alt_{bl.entity_id}_{now}".encode()).hexdigest()[:10],
                        entity_id=bl.entity_id,
                        entity_type="aircraft",
                        label=bl.label,
                        anomaly_type="altitude_anomaly",
                        severity="medium",
                        description=f"{bl.label} at unusual altitude",
                        normal_value=f"Avg: {bl.avg_altitude:.0f}ft ± {bl.altitude_std:.0f}",
                        observed_value=f"{alt:.0f}ft (z={z_score:.1f}σ)",
                        deviation_sigma=z_score,
                        timestamp=now,
                        lat=lat, lon=lon,
                    ))
            # Update running stats
            n = bl.total_observations
            old_avg = bl.avg_altitude
            bl.avg_altitude = old_avg + (alt - old_avg) / n
            if n > 1:
                bl.altitude_std = math.sqrt(
                    ((n - 2) * bl.altitude_std ** 2 + (alt - old_avg) * (alt - bl.avg_altitude)) / (n - 1)
                ) if n > 2 else abs(alt - bl.avg_altitude)

        # Speed anomaly
        if speed is not None and speed > 0:
            if bl.total_observations >= self.MIN_OBS_FOR_BASELINE and bl.avg_speed > 0:
                z_score = abs(speed - bl.avg_speed) / bl.speed_std if bl.speed_std > 0 else 0
                if z_score > 3.0:
                    anomalies.append(PoLAnomaly(
                        anomaly_id=hashlib.md5(f"spd_{bl.entity_id}_{now}".encode()).hexdigest()[:10],
                        entity_id=bl.entity_id,
                        entity_type="aircraft",
                        label=bl.label,
                        anomaly_type="speed_anomaly",
                        severity="low",
                        description=f"{bl.label} at unusual speed",
                        normal_value=f"Avg: {bl.avg_speed:.0f}kts ± {bl.speed_std:.0f}",
                        observed_value=f"{speed:.0f}kts (z={z_score:.1f}σ)",
                        deviation_sigma=z_score,
                        timestamp=now,
                    ))
            n = bl.total_observations
            old_avg = bl.avg_speed
            bl.avg_speed = old_avg + (speed - old_avg) / n
            if n > 1:
                bl.speed_std = math.sqrt(
                    ((n - 2) * bl.speed_std ** 2 + (speed - old_avg) * (speed - bl.avg_speed)) / (n - 1)
                ) if n > 2 else abs(speed - bl.avg_speed)

        # Route tracking
        if callsign:
            bl.common_routes[callsign] = bl.common_routes.get(callsign, 0) + 1

        # Schedule anomaly: active at unusual hour
        if bl.total_observations >= self.MIN_OBS_FOR_BASELINE * 3:
            total_hour_obs = sum(bl.hours_active)
            if total_hour_obs > 0:
                hour_pct = bl.hours_active[hour] / total_hour_obs
                if hour_pct < 0.02 and total_hour_obs > 20:  # <2% of observations
                    anomalies.append(PoLAnomaly(
                        anomaly_id=hashlib.md5(f"sched_{bl.entity_id}_{now}".encode()).hexdigest()[:10],
                        entity_id=bl.entity_id,
                        entity_type="aircraft",
                        label=bl.label,
                        anomaly_type="schedule_anomaly",
                        severity="medium",
                        description=f"{bl.label} active at unusual hour ({hour}:00 UTC)",
                        normal_value=f"Only {hour_pct:.1%} of activity at this hour",
                        observed_value=f"Active at {hour}:00 UTC",
                        timestamp=now,
                    ))

        return anomalies

    def _update_vessel(self, bl: BehaviorBaseline,
                       lat: Optional[float], lon: Optional[float],
                       speed: Optional[float], destination: str,
                       hour: int, day: int, now: str) -> List[PoLAnomaly]:
        anomalies: List[PoLAnomaly] = []
        bl.total_observations += 1
        bl.last_seen = now

        if len(bl.hours_active) < 24:
            bl.hours_active = [0] * 24
        if len(bl.days_active) < 7:
            bl.days_active = [0] * 7
        bl.hours_active[hour] += 1
        bl.days_active[day] += 1

        # Position tracking
        if lat is not None and lon is not None:
            bl.position_history.append((lat, lon))
            if len(bl.position_history) > self.MAX_POSITION_HISTORY:
                bl.position_history = bl.position_history[-self.MAX_POSITION_HISTORY:]

            if len(bl.position_history) >= 3:
                avg_lat = sum(p[0] for p in bl.position_history) / len(bl.position_history)
                avg_lon = sum(p[1] for p in bl.position_history) / len(bl.position_history)
                bl.centroid_lat = avg_lat
                bl.centroid_lon = avg_lon
                dist = _haversine(lat, lon, avg_lat, avg_lon)
                bl.max_range_km = max(bl.max_range_km, dist)

                if bl.total_observations >= self.MIN_OBS_FOR_BASELINE and bl.max_range_km > 0:
                    if dist > bl.max_range_km * 2.5 and dist > 200:
                        anomalies.append(PoLAnomaly(
                            anomaly_id=hashlib.md5(f"vzone_{bl.entity_id}_{now}".encode()).hexdigest()[:10],
                            entity_id=bl.entity_id,
                            entity_type="vessel",
                            label=bl.label,
                            anomaly_type="zone_violation",
                            severity="high",
                            description=f"{bl.label} is {dist:.0f}km from normal operating area",
                            normal_value=f"Max range: {bl.max_range_km:.0f}km",
                            observed_value=f"Current distance: {dist:.0f}km",
                            deviation_sigma=dist / bl.max_range_km,
                            timestamp=now, lat=lat, lon=lon,
                        ))

        # Speed anomaly for vessels
        if speed is not None and speed > 0:
            if bl.total_observations >= self.MIN_OBS_FOR_BASELINE and bl.avg_speed > 0 and bl.speed_std > 0:
                z_score = abs(speed - bl.avg_speed) / bl.speed_std
                if z_score > 3.0:
                    anomalies.append(PoLAnomaly(
                        anomaly_id=hashlib.md5(f"vspd_{bl.entity_id}_{now}".encode()).hexdigest()[:10],
                        entity_id=bl.entity_id,
                        entity_type="vessel",
                        label=bl.label,
                        anomaly_type="speed_anomaly",
                        severity="medium",
                        description=f"{bl.label} at unusual speed",
                        normal_value=f"Avg: {bl.avg_speed:.1f}kts ± {bl.speed_std:.1f}",
                        observed_value=f"{speed:.1f}kts (z={z_score:.1f}σ)",
                        deviation_sigma=z_score,
                        timestamp=now,
                    ))
            n = bl.total_observations
            old_avg = bl.avg_speed
            bl.avg_speed = old_avg + (speed - old_avg) / n
            if n > 2:
                bl.speed_std = math.sqrt(
                    ((n - 2) * bl.speed_std ** 2 + (speed - old_avg) * (speed - bl.avg_speed)) / (n - 1)
                )

        # Route deviation
        if destination and destination.strip().upper() not in ("", "UNKNOWN", "N/A"):
            dest_upper = destination.strip().upper()
            bl.common_routes[dest_upper] = bl.common_routes.get(dest_upper, 0) + 1

            if bl.total_observations >= self.MIN_OBS_FOR_BASELINE and bl.common_routes:
                total_route_obs = sum(bl.common_routes.values())
                if total_route_obs > 5:
                    route_pct = bl.common_routes.get(dest_upper, 0) / total_route_obs
                    if route_pct < 0.05 and bl.common_routes.get(dest_upper, 0) <= 1:
                        anomalies.append(PoLAnomaly(
                            anomaly_id=hashlib.md5(f"route_{bl.entity_id}_{now}".encode()).hexdigest()[:10],
                            entity_id=bl.entity_id,
                            entity_type="vessel",
                            label=bl.label,
                            anomaly_type="route_deviation",
                            severity="high",
                            description=f"{bl.label} heading to unusual destination: {dest_upper}",
                            normal_value=f"Top routes: {self._top_routes(bl, 3)}",
                            observed_value=f"Destination: {dest_upper} (first time)",
                            timestamp=now, lat=lat, lon=lon,
                        ))

        return anomalies

    # ── queries ───────────────────────────────────────────────────────

    def get_baseline(self, entity_id: str) -> Optional[BehaviorBaseline]:
        return self.baselines.get(entity_id)

    def get_anomalous_entities(self, min_observations: int = 10) -> List[str]:
        """Return entity IDs that have established baselines."""
        return [eid for eid, bl in self.baselines.items()
                if bl.total_observations >= min_observations]

    def get_entity_profile(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get a human-readable profile of an entity's normal behavior."""
        bl = self.baselines.get(entity_id)
        if not bl:
            return None

        profile = {
            "entity_id": bl.entity_id,
            "type": bl.entity_type,
            "label": bl.label,
            "observations": bl.total_observations,
            "operating_area": {
                "centroid": (bl.centroid_lat, bl.centroid_lon),
                "max_range_km": bl.max_range_km,
            },
            "common_routes": self._top_routes(bl, 5),
        }
        if bl.entity_type == "aircraft":
            profile["avg_altitude_ft"] = bl.avg_altitude
            profile["avg_speed_kts"] = bl.avg_speed
        elif bl.entity_type == "vessel":
            profile["avg_speed_kts"] = bl.avg_speed

        if bl.hours_active:
            peak_hour = bl.hours_active.index(max(bl.hours_active))
            profile["peak_activity_hour"] = f"{peak_hour}:00 UTC"

        return profile

    # ── helpers ───────────────────────────────────────────────────────

    def _get_or_create(self, entity_id: str, entity_type: str,
                       label: str) -> BehaviorBaseline:
        if entity_id not in self.baselines:
            self.baselines[entity_id] = BehaviorBaseline(
                entity_id=entity_id,
                entity_type=entity_type,
                label=label,
            )
        return self.baselines[entity_id]

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _top_routes(bl: BehaviorBaseline, n: int = 3) -> str:
        if not bl.common_routes:
            return "None"
        sorted_routes = sorted(bl.common_routes.items(), key=lambda x: x[1], reverse=True)
        return ", ".join(f"{r}({c})" for r, c in sorted_routes[:n])

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "total_observations": self.total_observations,
                "total_anomalies": self.total_anomalies_detected,
                "tracked_entities": len(self.baselines),
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save PoL state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_observations = state.get("total_observations", 0)
                self.total_anomalies_detected = state.get("total_anomalies", 0)
        except Exception as e:
            logger.warning("Failed to load PoL state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "tracked_entities": len(self.baselines),
            "total_observations": self.total_observations,
            "total_anomalies_detected": self.total_anomalies_detected,
            "entities_with_baseline": sum(
                1 for bl in self.baselines.values()
                if bl.total_observations >= self.MIN_OBS_FOR_BASELINE
            ),
            "by_type": {
                t: sum(1 for bl in self.baselines.values() if bl.entity_type == t)
                for t in ("aircraft", "vessel")
            },
        }
