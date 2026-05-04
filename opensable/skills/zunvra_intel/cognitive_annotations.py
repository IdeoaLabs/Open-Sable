"""
#9,  Cognitive Annotations (Map Observer)

Sable observes the map continuously and leaves floating intelligence notes
on zones with detected anomalies.  Each annotation is a self-contained
micro-report: what was detected, when, severity, narrative context.

Example annotation:
  "Black Sea,  GPS-jamming cluster detected.  6 ADS-B positions lost
   around Crimea in 12 min window.  Pattern matches REB-era electronic
   warfare.  Monitoring."
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    """A cognitive annotation pinned to the map."""
    annotation_id: str
    lat: float
    lon: float
    category: str        # anomaly | trend | resolved | warning | insight
    severity: str        # info | low | medium | high | critical
    headline: str        # max ~80 chars
    narrative: str       # 1-3 sentences
    domains: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    ttl_seconds: int = 3600   # auto-expire after 1 hour by default
    resolved: bool = False
    update_count: int = 0

    def is_expired(self) -> bool:
        try:
            created = datetime.fromisoformat(self.created_at)
            elapsed = (datetime.now(timezone.utc) - created).total_seconds()
            return elapsed > self.ttl_seconds
        except Exception:
            return False

    def to_map_marker(self) -> Dict[str, Any]:
        color_map = {
            "info": "#3498db",
            "low": "#2ecc71",
            "medium": "#f39c12",
            "high": "#e67e22",
            "critical": "#e74c3c",
        }
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [self.lon, self.lat]},
            "properties": {
                "id": self.annotation_id,
                "category": self.category,
                "severity": self.severity,
                "headline": self.headline,
                "narrative": self.narrative,
                "color": color_map.get(self.severity, "#95a5a6"),
                "resolved": self.resolved,
                "created_at": self.created_at,
                "ttl": self.ttl_seconds,
            },
        }


# ---------------------------------------------------------------------------
# Anomaly detection rules
# ---------------------------------------------------------------------------

ZONE_DEFINITIONS = {
    "Black Sea": {"lat": 43.5, "lon": 34.0, "radius_km": 500},
    "South China Sea": {"lat": 15.0, "lon": 115.0, "radius_km": 800},
    "Taiwan Strait": {"lat": 24.0, "lon": 119.5, "radius_km": 300},
    "Persian Gulf": {"lat": 26.5, "lon": 52.0, "radius_km": 400},
    "Eastern Mediterranean": {"lat": 34.5, "lon": 33.0, "radius_km": 500},
    "Baltic Sea": {"lat": 58.0, "lon": 20.0, "radius_km": 500},
    "Arctic": {"lat": 75.0, "lon": 40.0, "radius_km": 1000},
    "Horn of Africa": {"lat": 10.0, "lon": 50.0, "radius_km": 600},
    "Korean Peninsula": {"lat": 37.5, "lon": 127.0, "radius_km": 300},
    "Red Sea": {"lat": 20.0, "lon": 38.5, "radius_km": 500},
}


class CognitiveAnnotator:
    """
    Observes the live Zunvra feed and generates floating intelligence
    annotations on regions with detected anomalies or interesting patterns.
    """

    MAX_ANNOTATIONS = 200

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "annotations.json"

        self.annotations: Dict[str, Annotation] = {}
        self.total_created = 0
        self._prev_metrics: Dict[str, float] = {}
        self._load_state()

    # ── main entry point ──────────────────────────────────────────────

    async def observe(
        self,
        snapshot: IntelSnapshot,
        anomalies: Optional[List[Dict]] = None,
        llm=None,
    ) -> List[Annotation]:
        """
        Observe a snapshot and generate/update annotations.

        Parameters
        ----------
        snapshot : IntelSnapshot
            Current Zunvra snapshot
        anomalies : list, optional
            Pre-detected anomalies from SwarmThreatAssessor
        llm : optional
            LLM for narrative generation
        """
        new_annotations: List[Annotation] = []
        now = datetime.now(timezone.utc).isoformat()

        # 1. Zone-level analysis
        for zone_name, zone_def in ZONE_DEFINITIONS.items():
            nearby = snapshot.entities_near(zone_def["lat"], zone_def["lon"],
                                           zone_def["radius_km"])
            annotation = self._analyze_zone(zone_name, zone_def, nearby, now)
            if annotation:
                new_annotations.append(annotation)

        # 2. Convert external anomalies to annotations
        if anomalies:
            for anomaly in anomalies:
                annotation = self._anomaly_to_annotation(anomaly, now)
                if annotation:
                    new_annotations.append(annotation)

        # 3. Detect cross-snapshot changes
        metrics = self._extract_metrics(snapshot)
        for key, val in metrics.items():
            prev = self._prev_metrics.get(key, val)
            if prev > 0 and abs(val - prev) / prev > 0.3:  # >30% change
                direction = "surge" if val > prev else "drop"
                ann = self._create(
                    lat=0, lon=0,
                    category="trend",
                    severity="medium",
                    headline=f"Global {key} {direction}: {prev:.0f} → {val:.0f}",
                    narrative=(f"{key.replace('_', ' ').title()} showed a {abs(val-prev)/prev*100:.0f}% "
                              f"{direction} in the latest observation window."),
                    domains=[key],
                    now=now,
                )
                new_annotations.append(ann)
        self._prev_metrics = metrics

        # 4. LLM narrative enrichment for high-severity annotations
        if llm:
            for ann in new_annotations:
                if ann.severity in ("high", "critical"):
                    try:
                        await self._enrich_narrative(llm, ann)
                    except Exception as e:
                        logger.debug("LLM annotation enrichment failed: %s", e)

        # 5. Auto-expire old annotations
        self._expire_old()

        # 6. Merge into store
        for ann in new_annotations:
            existing = self.annotations.get(ann.annotation_id)
            if existing and not existing.resolved:
                existing.narrative = ann.narrative
                existing.severity = ann.severity
                existing.updated_at = now
                existing.update_count += 1
            else:
                self.annotations[ann.annotation_id] = ann
                self.total_created += 1

        # Cap
        if len(self.annotations) > self.MAX_ANNOTATIONS:
            sorted_ids = sorted(self.annotations.keys(),
                               key=lambda k: self.annotations[k].created_at)
            for aid in sorted_ids[:50]:
                del self.annotations[aid]

        self._save_state()
        return new_annotations

    # ── zone analysis ─────────────────────────────────────────────────

    def _analyze_zone(self, zone_name: str, zone_def: Dict,
                      nearby: Dict[str, list], now: str) -> Optional[Annotation]:
        """Analyze entities near a named zone and generate annotation if notable."""
        total_entities = sum(len(v) for v in nearby.values())
        if total_entities == 0:
            return None

        # Military concentration
        mil_count = len(nearby.get("military", []))
        ship_count = len(nearby.get("ships", []))
        flight_count = len(nearby.get("flights", []))

        # GPS jamming in zone
        jamming = len(nearby.get("gps_jamming", []))

        findings = []
        severity = "info"
        category = "insight"

        if mil_count > 5:
            findings.append(f"{mil_count} military assets detected")
            severity = "high"
            category = "anomaly"
        if jamming > 0:
            findings.append(f"{jamming} GPS jamming events")
            severity = "high" if jamming > 3 else "medium"
            category = "anomaly"
        if ship_count > 100:
            findings.append(f"Dense maritime traffic ({ship_count} vessels)")
            if severity == "info":
                severity = "low"
        if flight_count > 200:
            findings.append(f"Heavy air traffic ({flight_count} flights)")
            if severity == "info":
                severity = "low"

        if not findings or severity == "info":
            return None

        headline = f"{zone_name}: {findings[0]}"
        narrative = f"{zone_name} zone activity,  " + "; ".join(findings) + "."

        domains = []
        if mil_count > 0: domains.append("military")
        if jamming > 0: domains.append("gps_jamming")
        if ship_count > 0: domains.append("maritime")
        if flight_count > 0: domains.append("aviation")

        return self._create(
            lat=zone_def["lat"],
            lon=zone_def["lon"],
            category=category,
            severity=severity,
            headline=headline[:80],
            narrative=narrative[:300],
            domains=domains,
            now=now,
            ttl=7200,  # Zone annotations last 2 hours
        )

    # ── anomaly conversion ────────────────────────────────────────────

    def _anomaly_to_annotation(self, anomaly: Dict, now: str) -> Optional[Annotation]:
        """Convert a detected anomaly dict to a map annotation."""
        desc = anomaly.get("description", anomaly.get("title", ""))
        if not desc:
            return None

        severity_map = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high", "CRITICAL": "critical"}
        severity = severity_map.get(anomaly.get("severity", ""), "medium")

        lat = anomaly.get("lat", 0)
        lon = anomaly.get("lon", 0)

        return self._create(
            lat=lat, lon=lon,
            category="anomaly",
            severity=severity,
            headline=desc[:80],
            narrative=anomaly.get("detail", desc)[:300],
            domains=anomaly.get("domains", []),
            now=now,
        )

    # ── helpers ───────────────────────────────────────────────────────

    def _create(self, lat: float, lon: float, category: str, severity: str,
                headline: str, narrative: str, domains: List[str],
                now: str, ttl: int = 3600) -> Annotation:
        aid = hashlib.sha256(f"{headline}_{lat}_{lon}".encode()).hexdigest()[:12]
        return Annotation(
            annotation_id=aid,
            lat=lat, lon=lon,
            category=category,
            severity=severity,
            headline=headline,
            narrative=narrative,
            domains=domains,
            created_at=now,
            updated_at=now,
            ttl_seconds=ttl,
        )

    def _extract_metrics(self, snapshot: IntelSnapshot) -> Dict[str, float]:
        summary = snapshot.summary_text()
        metrics = {}
        for pair in summary.split(","):
            pair = pair.strip()
            if ":" in pair:
                key, val = pair.split(":", 1)
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    pass
        return metrics

    def _expire_old(self):
        expired = [aid for aid, ann in self.annotations.items() if ann.is_expired()]
        for aid in expired:
            self.annotations[aid].resolved = True

    async def _enrich_narrative(self, llm, annotation: Annotation):
        prompt = (
            "You are a military intelligence briefer. Write a 2-sentence annotation "
            "for a live map display. Be specific and authoritative.\n\n"
            f"Zone: ({annotation.lat}, {annotation.lon})\n"
            f"Headline: {annotation.headline}\n"
            f"Current data: {annotation.narrative}\n\n"
            "Write the annotation (2 sentences, under 200 characters):"
        )
        raw = await llm.chat_raw(prompt, max_tokens=100)
        if raw and raw.strip():
            annotation.narrative = raw.strip()[:300]

    # ── queries ───────────────────────────────────────────────────────

    def get_active_annotations(self) -> List[Annotation]:
        return [a for a in self.annotations.values() if not a.resolved and not a.is_expired()]

    def get_geojson(self) -> Dict:
        features = [a.to_map_marker() for a in self.get_active_annotations()]
        return {"type": "FeatureCollection", "features": features}

    def resolve(self, annotation_id: str):
        if annotation_id in self.annotations:
            self.annotations[annotation_id].resolved = True
            self._save_state()

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "annotations": {k: asdict(v) for k, v in list(self.annotations.items())[-100:]},
                "total_created": self.total_created,
                "prev_metrics": self._prev_metrics,
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save annotations state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_created = state.get("total_created", 0)
                self._prev_metrics = state.get("prev_metrics", {})
        except Exception as e:
            logger.warning("Failed to load annotations state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        active = self.get_active_annotations()
        return {
            "total_created": self.total_created,
            "active_annotations": len(active),
            "by_severity": {s: sum(1 for a in active if a.severity == s)
                           for s in ("info", "low", "medium", "high", "critical")},
        }
