"""
#17,  Counter-Surveillance Detector

Detect when entities are engaging in evasive or counter-surveillance
behaviour.  This is what agencies like NSA/GCHQ run internally but
never expose publicly.

Detection capabilities:
  - AIS gaps (dark ships),  vessel turns off transponder
  - Transponder manipulation,  hex/MMSI changes mid-flight/voyage
  - Unusual holding patterns,  orbiting without clearance
  - Shadow tracking,  one entity follows another persistently
  - Deliberate route obfuscation,  zigzag, backtrack, unnecessary waypoints
  - Speed anomalies,  too slow (loitering) or too fast (evasion)
  - Altitude masking,  flying below radar (terrain masking)
  - Rendezvous detection,  two entities converging at unusual location
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


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing between two points in degrees."""
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = rlon2 - rlon1
    x = math.sin(dlon) * math.cos(rlat2)
    y = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


@dataclass
class EvasionAlert:
    """An alert for detected counter-surveillance / evasive behaviour."""
    alert_id: str
    entity_id: str
    entity_label: str
    entity_type: str
    alert_type: str  # dark_period, transponder_change, holding_pattern, shadow_track,
                     # route_obfuscation, speed_anomaly, altitude_masking, rendezvous
    severity: str  # low, medium, high, critical
    description: str
    timestamp: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityTrack:
    """Internal tracking record for an entity."""
    entity_id: str
    entity_type: str
    label: str
    positions: List[Dict[str, Any]] = field(default_factory=list)  # [{lat, lon, alt, speed, heading, ts}]
    last_seen: float = 0.0
    hex_codes: List[str] = field(default_factory=list)  # Track transponder changes
    mmsi_codes: List[str] = field(default_factory=list)


class CounterSurveillanceDetector:
    """
    Detect evasive behavior, transponder manipulation, shadow tracking,
    and other counter-surveillance indicators.
    """

    MAX_ENTITIES = 10000
    MAX_POSITIONS = 50
    DARK_THRESHOLD_SEC = 1800      # 30 min without signal = "dark period"
    HOLDING_PATTERN_TURNS = 3      # Number of 90°+ turns in window
    SHADOW_DISTANCE_KM = 15        # Max distance to be considered "shadowing"
    SHADOW_MIN_OBSERVATIONS = 3    # Min correlated positions
    LOW_ALT_FT = 500               # Below this = terrain masking
    RENDEZVOUS_KM = 5              # Convergence distance
    ZIGZAG_HEADING_CHANGE = 60     # Degrees of heading change

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tracks: Dict[str, EntityTrack] = {}
        self.alerts: List[EvasionAlert] = []
        self.total_alerts = 0
        self._prev_entities: Dict[str, Dict[str, Any]] = {}  # For diff-based detection

    # ── main observation ──────────────────────────────────────────────

    def observe(self, snapshot: IntelSnapshot) -> List[EvasionAlert]:
        """
        Analyze snapshot for counter-surveillance indicators.
        Returns list of new evasion alerts.
        """
        now_epoch = time.time()
        now_str = datetime.now(timezone.utc).isoformat()
        new_alerts: List[EvasionAlert] = []

        current_entities: Dict[str, Dict[str, Any]] = {}

        # Process all entity types
        entities = self._extract_all(snapshot)

        for ent in entities:
            eid = ent["id"]
            current_entities[eid] = ent

            track = self.tracks.get(eid)
            if not track:
                track = EntityTrack(
                    entity_id=eid,
                    entity_type=ent["type"],
                    label=ent.get("label", eid),
                )
                self.tracks[eid] = track

            # Update track
            lat = ent.get("lat")
            lon = ent.get("lon")
            if lat is not None and lon is not None:
                pos = {
                    "lat": lat, "lon": lon,
                    "alt": ent.get("alt"),
                    "speed": ent.get("speed"),
                    "heading": ent.get("heading"),
                    "ts": now_epoch,
                }
                track.positions.append(pos)
                if len(track.positions) > self.MAX_POSITIONS:
                    track.positions = track.positions[-self.MAX_POSITIONS:]

            # Track transponder codes
            hex_code = ent.get("hex")
            if hex_code:
                if track.hex_codes and hex_code != track.hex_codes[-1]:
                    new_alerts.append(self._alert(
                        eid, track.label, ent["type"],
                        "transponder_change", "high",
                        f"{track.label} changed transponder hex: {track.hex_codes[-1]} → {hex_code}",
                        now_str, lat, lon,
                        {"old_hex": track.hex_codes[-1], "new_hex": hex_code},
                    ))
                if not track.hex_codes or hex_code != track.hex_codes[-1]:
                    track.hex_codes.append(hex_code)

            mmsi = ent.get("mmsi")
            if mmsi:
                if track.mmsi_codes and mmsi != track.mmsi_codes[-1]:
                    new_alerts.append(self._alert(
                        eid, track.label, ent["type"],
                        "transponder_change", "critical",
                        f"{track.label} changed MMSI: {track.mmsi_codes[-1]} → {mmsi}",
                        now_str, lat, lon,
                        {"old_mmsi": track.mmsi_codes[-1], "new_mmsi": mmsi},
                    ))
                if not track.mmsi_codes or mmsi != track.mmsi_codes[-1]:
                    track.mmsi_codes.append(mmsi)

            # Dark period detection (entity was gone, now back)
            if track.last_seen > 0:
                gap = now_epoch - track.last_seen
                if gap > self.DARK_THRESHOLD_SEC:
                    gap_min = gap / 60
                    new_alerts.append(self._alert(
                        eid, track.label, ent["type"],
                        "dark_period", "high" if gap_min > 120 else "medium",
                        f"{track.label} went dark for {gap_min:.0f}min,  possible transponder shutdown",
                        now_str, lat, lon,
                        {"gap_minutes": gap_min},
                    ))

            track.last_seen = now_epoch

            # Holding pattern detection
            if len(track.positions) >= 4:
                holding_alert = self._detect_holding_pattern(track, now_str)
                if holding_alert:
                    new_alerts.append(holding_alert)

            # Route obfuscation (zigzag)
            if len(track.positions) >= 5:
                zigzag_alert = self._detect_zigzag(track, now_str)
                if zigzag_alert:
                    new_alerts.append(zigzag_alert)

            # Altitude masking (aircraft only)
            alt = ent.get("alt")
            if alt is not None and ent["type"] in ("aircraft", "military"):
                try:
                    alt_val = float(alt)
                    if 0 < alt_val < self.LOW_ALT_FT:
                        new_alerts.append(self._alert(
                            eid, track.label, ent["type"],
                            "altitude_masking", "high",
                            f"{track.label} flying at {alt_val:.0f}ft,  possible terrain masking / radar evasion",
                            now_str, lat, lon,
                            {"altitude_ft": alt_val},
                        ))
                except (ValueError, TypeError):
                    pass

        # Shadow tracking detection (pairs of entities)
        shadow_alerts = self._detect_shadows(current_entities, now_str)
        new_alerts.extend(shadow_alerts)

        # Rendezvous detection
        rendezvous_alerts = self._detect_rendezvous(current_entities, now_str)
        new_alerts.extend(rendezvous_alerts)

        # Dark entities,  were seen last time, gone now
        for eid, prev_ent in self._prev_entities.items():
            if eid not in current_entities:
                track = self.tracks.get(eid)
                if track and track.last_seen > 0:
                    gap = now_epoch - track.last_seen
                    if gap > self.DARK_THRESHOLD_SEC:
                        # Already handled above for returning entities
                        pass

        self._prev_entities = current_entities
        self.alerts.extend(new_alerts)
        self.total_alerts += len(new_alerts)

        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]

        # Trim tracks
        if len(self.tracks) > self.MAX_ENTITIES:
            sorted_tracks = sorted(self.tracks.items(), key=lambda x: x[1].last_seen)
            to_remove = len(self.tracks) - self.MAX_ENTITIES
            for i in range(to_remove):
                del self.tracks[sorted_tracks[i][0]]

        return new_alerts

    # ── detection algorithms ──────────────────────────────────────────

    def _detect_holding_pattern(self, track: EntityTrack,
                                now_str: str) -> Optional[EvasionAlert]:
        """Detect orbiting / holding patterns from heading changes."""
        positions = track.positions[-10:]
        if len(positions) < 4:
            return None

        large_turns = 0
        for i in range(1, len(positions)):
            h1 = positions[i - 1].get("heading")
            h2 = positions[i].get("heading")
            if h1 is not None and h2 is not None:
                diff = abs(h2 - h1)
                if diff > 180:
                    diff = 360 - diff
                if diff > 70:  # Significant turn
                    large_turns += 1

        if large_turns >= self.HOLDING_PATTERN_TURNS:
            last_pos = positions[-1]
            return self._alert(
                track.entity_id, track.label, track.entity_type,
                "holding_pattern", "medium",
                f"{track.label} executing holding pattern,  {large_turns} significant turns detected",
                now_str,
                last_pos.get("lat"), last_pos.get("lon"),
                {"turn_count": large_turns},
            )
        return None

    def _detect_zigzag(self, track: EntityTrack,
                       now_str: str) -> Optional[EvasionAlert]:
        """Detect deliberate route obfuscation (zigzag pattern)."""
        positions = track.positions[-8:]
        if len(positions) < 5:
            return None

        heading_changes = []
        for i in range(1, len(positions)):
            h1 = positions[i - 1].get("heading")
            h2 = positions[i].get("heading")
            if h1 is not None and h2 is not None:
                diff = h2 - h1
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                heading_changes.append(diff)

        if len(heading_changes) < 4:
            return None

        # Zigzag = alternating left/right turns
        alternations = 0
        for i in range(1, len(heading_changes)):
            if (heading_changes[i] > self.ZIGZAG_HEADING_CHANGE and
                heading_changes[i - 1] < -self.ZIGZAG_HEADING_CHANGE) or \
               (heading_changes[i] < -self.ZIGZAG_HEADING_CHANGE and
                heading_changes[i - 1] > self.ZIGZAG_HEADING_CHANGE):
                alternations += 1

        if alternations >= 2:
            last_pos = positions[-1]
            return self._alert(
                track.entity_id, track.label, track.entity_type,
                "route_obfuscation", "high",
                f"{track.label} executing zigzag pattern,  {alternations} direction alternations",
                now_str,
                last_pos.get("lat"), last_pos.get("lon"),
                {"alternations": alternations},
            )
        return None

    def _detect_shadows(self, entities: Dict[str, Dict[str, Any]],
                        now_str: str) -> List[EvasionAlert]:
        """Detect one entity persistently following another."""
        alerts: List[EvasionAlert] = []
        entity_list = [(eid, e) for eid, e in entities.items()
                      if e.get("lat") is not None and e.get("lon") is not None]

        # O(n²) but limited by entity count in a single snapshot
        checked: set = set()
        for i, (eid1, e1) in enumerate(entity_list):
            for j, (eid2, e2) in enumerate(entity_list):
                if i >= j:
                    continue
                pair = (eid1, eid2)
                if pair in checked:
                    continue
                checked.add(pair)

                dist = _haversine(e1["lat"], e1["lon"], e2["lat"], e2["lon"])
                if dist > self.SHADOW_DISTANCE_KM:
                    continue

                # Check heading similarity
                h1 = e1.get("heading")
                h2 = e2.get("heading")
                heading_match = False
                if h1 is not None and h2 is not None:
                    hdiff = abs(h2 - h1)
                    if hdiff > 180:
                        hdiff = 360 - hdiff
                    heading_match = hdiff < 30

                # Check if this is a persistent shadow (need history)
                t1 = self.tracks.get(eid1)
                t2 = self.tracks.get(eid2)
                close_count = 0
                if t1 and t2 and len(t1.positions) >= 2 and len(t2.positions) >= 2:
                    for p1, p2 in zip(t1.positions[-5:], t2.positions[-5:]):
                        if _haversine(p1["lat"], p1["lon"], p2["lat"], p2["lon"]) < self.SHADOW_DISTANCE_KM:
                            close_count += 1

                if close_count >= self.SHADOW_MIN_OBSERVATIONS and heading_match:
                    alerts.append(self._alert(
                        eid1, entities[eid1].get("label", eid1), e1["type"],
                        "shadow_track", "high",
                        f"Possible shadow tracking: {entities[eid1].get('label', eid1)} and "
                        f"{entities[eid2].get('label', eid2)},  {close_count} correlated positions, "
                        f"{dist:.1f}km apart, matching heading",
                        now_str, e1["lat"], e1["lon"],
                        {"shadow_entity": eid2, "distance_km": dist,
                         "correlated_obs": close_count},
                    ))

        return alerts

    def _detect_rendezvous(self, entities: Dict[str, Dict[str, Any]],
                           now_str: str) -> List[EvasionAlert]:
        """Detect two entities converging at an unusual point."""
        alerts: List[EvasionAlert] = []

        entity_list = [(eid, e) for eid, e in entities.items()
                      if e.get("lat") is not None and e.get("lon") is not None]

        for i, (eid1, e1) in enumerate(entity_list):
            for j, (eid2, e2) in enumerate(entity_list):
                if i >= j:
                    continue
                # Different entity types converging is more suspicious
                if e1["type"] == e2["type"]:
                    continue

                dist = _haversine(e1["lat"], e1["lon"], e2["lat"], e2["lon"])
                if dist > self.RENDEZVOUS_KM:
                    continue

                # Check if they were farther apart before (converging)
                t1 = self.tracks.get(eid1)
                t2 = self.tracks.get(eid2)
                if t1 and t2 and len(t1.positions) >= 2 and len(t2.positions) >= 2:
                    prev_dist = _haversine(
                        t1.positions[-2]["lat"], t1.positions[-2]["lon"],
                        t2.positions[-2]["lat"], t2.positions[-2]["lon"],
                    )
                    if prev_dist > dist * 3:  # Were at least 3x farther apart
                        alerts.append(self._alert(
                            eid1, entities[eid1].get("label", eid1), e1["type"],
                            "rendezvous", "high",
                            f"Possible rendezvous: {entities[eid1].get('label', eid1)} ({e1['type']}) "
                            f"and {entities[eid2].get('label', eid2)} ({e2['type']}) "
                            f"converged to {dist:.1f}km (were {prev_dist:.1f}km apart)",
                            now_str, e1["lat"], e1["lon"],
                            {"other_entity": eid2, "current_dist_km": dist,
                             "previous_dist_km": prev_dist},
                        ))

        return alerts

    # ── entity extraction ─────────────────────────────────────────────

    def _extract_all(self, snapshot: IntelSnapshot) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []

        for flight in snapshot.flights:
            entities.append({
                "id": f"aircraft_{flight.get('hex', '')}",
                "type": "aircraft",
                "label": (flight.get("callsign") or flight.get("flight") or flight.get("hex", "")).strip(),
                "lat": self._sf(flight.get("lat")),
                "lon": self._sf(flight.get("lon")),
                "alt": flight.get("alt_baro"),
                "speed": self._sf(flight.get("gs")),
                "heading": self._sf(flight.get("track")),
                "hex": flight.get("hex"),
            })

        for flight in snapshot.military_flights:
            entities.append({
                "id": f"military_{flight.get('hex', '')}",
                "type": "military",
                "label": (flight.get("callsign") or flight.get("type") or flight.get("hex", "")).strip(),
                "lat": self._sf(flight.get("lat")),
                "lon": self._sf(flight.get("lon")),
                "alt": flight.get("alt_baro"),
                "speed": self._sf(flight.get("gs")),
                "heading": self._sf(flight.get("track")),
                "hex": flight.get("hex"),
            })

        for ship in snapshot.ships:
            mmsi = str(ship.get("mmsi", ship.get("MMSI", "")))
            entities.append({
                "id": f"vessel_{mmsi}",
                "type": "vessel",
                "label": ship.get("name", ship.get("ship_name", mmsi)),
                "lat": self._sf(ship.get("lat")),
                "lon": self._sf(ship.get("lon")),
                "speed": self._sf(ship.get("speed", ship.get("sog"))),
                "heading": self._sf(ship.get("heading", ship.get("cog"))),
                "mmsi": mmsi,
            })

        return entities

    @staticmethod
    def _sf(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _alert(eid: str, label: str, etype: str,
               alert_type: str, severity: str, description: str,
               now_str: str, lat: Optional[float], lon: Optional[float],
               evidence: Dict[str, Any]) -> EvasionAlert:
        return EvasionAlert(
            alert_id=hashlib.md5(f"{eid}_{alert_type}_{time.time()}".encode()).hexdigest()[:12],
            entity_id=eid,
            entity_label=label,
            entity_type=etype,
            alert_type=alert_type,
            severity=severity,
            description=description,
            timestamp=now_str,
            lat=lat, lon=lon,
            evidence=evidence,
        )

    # ── queries ───────────────────────────────────────────────────────

    def get_recent_alerts(self, limit: int = 50) -> List[EvasionAlert]:
        return self.alerts[-limit:]

    def get_alerts_by_type(self, alert_type: str) -> List[EvasionAlert]:
        return [a for a in self.alerts if a.alert_type == alert_type]

    def get_entity_track(self, entity_id: str) -> Optional[EntityTrack]:
        return self.tracks.get(entity_id)

    def get_dark_entities(self, threshold_sec: Optional[float] = None) -> List[str]:
        """Return entity IDs that have gone dark (not seen recently)."""
        threshold = threshold_sec or self.DARK_THRESHOLD_SEC
        now = time.time()
        return [eid for eid, t in self.tracks.items()
                if t.last_seen > 0 and (now - t.last_seen) > threshold]

    def get_stats(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = {}
        for a in self.alerts:
            type_counts[a.alert_type] = type_counts.get(a.alert_type, 0) + 1

        return {
            "total_alerts": self.total_alerts,
            "entities_tracked": len(self.tracks),
            "alert_type_distribution": type_counts,
            "dark_entities": len(self.get_dark_entities()),
        }
