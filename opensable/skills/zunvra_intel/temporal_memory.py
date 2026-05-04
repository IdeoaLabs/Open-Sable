"""
#6,  Temporal Pattern Memory

Leverages OpenSable's Time Crystal + Déjà Vu concepts to detect recurring
temporal patterns in Zunvra's OSINT data stream.

After running for days/weeks, recognizes patterns like:
  "Current GPS jamming pattern in Eastern Med matches pattern from March 2025
   (confidence 0.82). Last time this preceded Israeli military ops 48h later."

Uses multi-dimensional fingerprints: entities, geography, timing, intensity,
domains affected, and geopolitical context.
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
from typing import Any, Dict, List, Optional, Tuple

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


@dataclass
class SituationFingerprint:
    """6-dimensional fingerprint of a situation at a point in time."""
    fingerprint_id: str
    timestamp: str
    # Dimension 1: Entity composition (what types of entities are active)
    entity_profile: Dict[str, float] = field(default_factory=dict)  # domain → normalized count
    # Dimension 2: Geographic focus (where activity is concentrated)
    geo_center_lat: float = 0.0
    geo_center_lon: float = 0.0
    geo_spread_km: float = 0.0
    # Dimension 3: Timing (hour of day, day of week patterns)
    hour_of_day: int = 0
    day_of_week: int = 0
    # Dimension 4: Intensity (how much above baseline)
    intensity_score: float = 0.0
    # Dimension 5: Domain count (how many domains active)
    active_domains: int = 0
    # Dimension 6: Threat signature (which threat types present)
    threat_types: List[str] = field(default_factory=list)

    def to_vector(self) -> List[float]:
        """Convert to a flat vector for cosine similarity."""
        vec = []
        # Entity profile (13 dimensions)
        for domain in ["flights", "military", "ships", "satellites", "earthquakes",
                        "fires", "gdelt_events", "cyber_threats", "gps_jamming",
                        "carriers", "conflicts", "internet_outages", "ransomware"]:
            vec.append(self.entity_profile.get(domain, 0.0))
        # Geography (3 dims)
        vec.extend([self.geo_center_lat / 90.0, self.geo_center_lon / 180.0, self.geo_spread_km / 5000.0])
        # Timing (2 dims)
        vec.extend([self.hour_of_day / 24.0, self.day_of_week / 7.0])
        # Intensity + active domains (2 dims)
        vec.extend([self.intensity_score, self.active_domains / 13.0])
        # Threat types (binary encoding for top types)
        threat_set = set(self.threat_types)
        for tt in ["military", "cyber", "nuclear", "conflict", "natural", "economic"]:
            vec.append(1.0 if tt in threat_set else 0.0)
        return vec


@dataclass
class TemporalPattern:
    """A recurring pattern detected across time."""
    pattern_id: str
    description: str
    fingerprints: List[str] = field(default_factory=list)  # fingerprint IDs
    occurrences: int = 1
    first_seen: str = ""
    last_seen: str = ""
    avg_interval_hours: float = 0.0
    what_followed: List[str] = field(default_factory=list)  # descriptions of what happened after
    confidence: float = 0.5

    def to_text(self) -> str:
        return (
            f"TEMPORAL PATTERN {self.pattern_id}\n"
            f"  {self.description}\n"
            f"  Occurrences: {self.occurrences} | Confidence: {self.confidence:.0%}\n"
            f"  Avg interval: {self.avg_interval_hours:.0f}h\n"
            f"  First: {self.first_seen} | Last: {self.last_seen}\n"
            + ("\n".join(f"  After last: {w}" for w in self.what_followed[:3]) if self.what_followed else "")
        )


@dataclass
class DejaVuMatch:
    """A current situation matching a historical pattern."""
    current_fingerprint: str
    matched_pattern_id: str
    similarity: float
    pattern_description: str
    what_followed_before: List[str]
    prediction: str = ""
    confidence: float = 0.0

    def to_text(self) -> str:
        return (
            f"DÉJÀ VU,  {self.similarity:.0%} match with pattern {self.matched_pattern_id}\n"
            f"  Pattern: {self.pattern_description}\n"
            f"  Historical aftermath: {'; '.join(self.what_followed_before[:3])}\n"
            f"  Prediction: {self.prediction} [{self.confidence:.0%}]"
        )


# ---------------------------------------------------------------------------
# Temporal Pattern Memory
# ---------------------------------------------------------------------------

class TemporalPatternMemory:
    """
    Long-term situational pattern memory. Detects recurring situations
    using multi-dimensional fingerprinting and cosine similarity matching.
    """

    MAX_FINGERPRINTS = 2000
    MAX_PATTERNS = 300
    SIMILARITY_THRESHOLD = 0.75

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "temporal_memory.json"

        self.fingerprints: List[SituationFingerprint] = []
        self.patterns: List[TemporalPattern] = []
        self.total_observations = 0
        self._load_state()

    # ── fingerprinting ────────────────────────────────────────────────

    def fingerprint(self, snapshot: IntelSnapshot) -> SituationFingerprint:
        """Create a 6-dimensional fingerprint from a snapshot."""
        now = datetime.now(timezone.utc)
        fid = hashlib.sha256(f"{now.isoformat()}_{self.total_observations}".encode()).hexdigest()[:12]

        # Entity profile,  normalize by typical maximums
        maxima = {
            "flights": 5000, "military": 50, "ships": 25000, "satellites": 2000,
            "earthquakes": 100, "fires": 5000, "gdelt_events": 1000, "cyber_threats": 50,
            "gps_jamming": 20, "carriers": 11, "conflicts": 200,
            "internet_outages": 50, "ransomware": 30,
        }
        entity_profile = {
            "flights": len(snapshot.flights) / maxima["flights"],
            "military": len(snapshot.military_flights) / maxima["military"],
            "ships": len(snapshot.ships) / maxima["ships"],
            "satellites": len(snapshot.satellites) / maxima["satellites"],
            "earthquakes": len(snapshot.earthquakes) / maxima["earthquakes"],
            "fires": len(snapshot.fires) / maxima["fires"],
            "gdelt_events": len(snapshot.gdelt_events) / maxima["gdelt_events"],
            "cyber_threats": len(snapshot.cyber_threats) / maxima["cyber_threats"],
            "gps_jamming": len(snapshot.gps_jamming) / maxima["gps_jamming"],
            "carriers": len(snapshot.carriers) / maxima["carriers"],
            "conflicts": len(snapshot.conflicts) / maxima["conflicts"],
            "internet_outages": len(snapshot.internet_outages) / maxima["internet_outages"],
            "ransomware": len(snapshot.ransomware) / maxima["ransomware"],
        }

        # Geographic center (simple centroid of military + GDELT events)
        lats, lons = [], []
        for item in snapshot.military_flights + snapshot.gdelt_events[:50]:
            lat = item.get("lat") or item.get("latitude")
            lon = item.get("lon") or item.get("longitude")
            if lat and lon:
                try:
                    lats.append(float(lat))
                    lons.append(float(lon))
                except (ValueError, TypeError):
                    pass
        geo_lat = sum(lats) / len(lats) if lats else 0.0
        geo_lon = sum(lons) / len(lons) if lons else 0.0

        # Active domain count
        active = sum(1 for v in entity_profile.values() if v > 0.05)

        # Intensity: sum of all normalized values
        intensity = sum(entity_profile.values())

        # Threat types present
        threat_types = []
        if entity_profile.get("military", 0) > 0.1: threat_types.append("military")
        if entity_profile.get("cyber_threats", 0) > 0.1: threat_types.append("cyber")
        if entity_profile.get("gps_jamming", 0) > 0.05: threat_types.append("electronic_warfare")
        if entity_profile.get("gdelt_events", 0) > 0.1: threat_types.append("conflict")
        if entity_profile.get("earthquakes", 0) > 0.1: threat_types.append("natural")
        if entity_profile.get("ransomware", 0) > 0.1: threat_types.append("cyber")

        fp = SituationFingerprint(
            fingerprint_id=fid,
            timestamp=now.isoformat(),
            entity_profile=entity_profile,
            geo_center_lat=geo_lat,
            geo_center_lon=geo_lon,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            intensity_score=min(1.0, intensity / 5.0),
            active_domains=active,
            threat_types=threat_types,
        )

        return fp

    # ── observation + matching ────────────────────────────────────────

    def observe(self, snapshot: IntelSnapshot) -> List[DejaVuMatch]:
        """
        Create a fingerprint, store it, and check for déjà vu matches.
        Returns matches with historical patterns if found.
        """
        fp = self.fingerprint(snapshot)
        matches = self._find_matches(fp)

        self.fingerprints.append(fp)
        if len(self.fingerprints) > self.MAX_FINGERPRINTS:
            self.fingerprints = self.fingerprints[-self.MAX_FINGERPRINTS:]

        self.total_observations += 1

        # Update or create patterns for strong matches
        for match in matches:
            self._reinforce_pattern(match, fp)

        self._save_state()
        return matches

    def _find_matches(self, current: SituationFingerprint) -> List[DejaVuMatch]:
        """Compare current fingerprint against all historical ones."""
        matches: List[DejaVuMatch] = []
        current_vec = current.to_vector()

        for historical in self.fingerprints[:-1]:  # exclude very latest
            hist_vec = historical.to_vector()
            sim = self._cosine_similarity(current_vec, hist_vec)

            if sim >= self.SIMILARITY_THRESHOLD:
                # Find the pattern this fingerprint belongs to
                pattern = self._find_pattern_for(historical.fingerprint_id)
                what_followed = pattern.what_followed if pattern else []
                pattern_desc = pattern.description if pattern else f"Similar situation on {historical.timestamp}"
                pattern_id = pattern.pattern_id if pattern else historical.fingerprint_id

                matches.append(DejaVuMatch(
                    current_fingerprint=current.fingerprint_id,
                    matched_pattern_id=pattern_id,
                    similarity=sim,
                    pattern_description=pattern_desc,
                    what_followed_before=what_followed,
                    prediction=f"Based on {len(what_followed)} prior occurrences, similar conditions may recur",
                    confidence=sim * 0.9,
                ))

        # Sort by similarity descending, return top 5
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:5]

    def _reinforce_pattern(self, match: DejaVuMatch, new_fp: SituationFingerprint):
        """Strengthen the matched pattern or create a new one."""
        for pattern in self.patterns:
            if pattern.pattern_id == match.matched_pattern_id:
                pattern.occurrences += 1
                pattern.last_seen = new_fp.timestamp
                pattern.fingerprints.append(new_fp.fingerprint_id)
                pattern.confidence = min(0.95, pattern.confidence + 0.05)
                return

        # Create new pattern
        pid = hashlib.sha256(f"pattern_{self.total_observations}".encode()).hexdigest()[:10]
        new_pattern = TemporalPattern(
            pattern_id=pid,
            description=match.pattern_description,
            fingerprints=[match.matched_pattern_id, new_fp.fingerprint_id],
            occurrences=2,
            first_seen=match.matched_pattern_id,  # approximate
            last_seen=new_fp.timestamp,
            confidence=match.similarity * 0.8,
        )
        self.patterns.append(new_pattern)
        if len(self.patterns) > self.MAX_PATTERNS:
            self.patterns.sort(key=lambda p: p.confidence * p.occurrences, reverse=True)
            self.patterns = self.patterns[:self.MAX_PATTERNS]

    def _find_pattern_for(self, fingerprint_id: str) -> Optional[TemporalPattern]:
        for p in self.patterns:
            if fingerprint_id in p.fingerprints:
                return p
        return None

    def record_aftermath(self, pattern_id: str, description: str):
        """Record what happened after a pattern was observed (for future prediction)."""
        for p in self.patterns:
            if p.pattern_id == pattern_id:
                p.what_followed.append(description)
                if len(p.what_followed) > 20:
                    p.what_followed = p.what_followed[-20:]
                self._save_state()
                return

    # ── similarity ────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    # ── queries ───────────────────────────────────────────────────────

    def get_strongest_patterns(self, n: int = 10) -> List[TemporalPattern]:
        return sorted(self.patterns, key=lambda p: p.confidence * p.occurrences, reverse=True)[:n]

    def get_report(self) -> str:
        lines = [f"TEMPORAL PATTERN MEMORY,  {self.total_observations} observations, {len(self.patterns)} patterns"]
        for p in self.get_strongest_patterns(5):
            lines.append(p.to_text())
        return "\n".join(lines)

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "fingerprints": [asdict(fp) for fp in self.fingerprints[-200:]],  # save last 200
                "patterns": [asdict(p) for p in self.patterns],
                "total_observations": self.total_observations,
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save temporal memory: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                for fpd in state.get("fingerprints", []):
                    self.fingerprints.append(SituationFingerprint(
                        **{k: v for k, v in fpd.items() if k in SituationFingerprint.__dataclass_fields__}
                    ))
                for pd in state.get("patterns", []):
                    self.patterns.append(TemporalPattern(
                        **{k: v for k, v in pd.items() if k in TemporalPattern.__dataclass_fields__}
                    ))
                self.total_observations = state.get("total_observations", 0)
                logger.info("Loaded %d fingerprints, %d patterns",
                           len(self.fingerprints), len(self.patterns))
        except Exception as e:
            logger.warning("Failed to load temporal memory: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_observations": self.total_observations,
            "fingerprints_stored": len(self.fingerprints),
            "patterns_detected": len(self.patterns),
            "strongest_pattern": self.get_strongest_patterns(1)[0].to_text() if self.patterns else None,
        }
