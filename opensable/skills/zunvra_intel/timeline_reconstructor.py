"""
#15 — Timeline Reconstructor

Chronological event sequencing for incident investigation.
Ingests events from ALL intelligence domains (air, maritime, cyber,
economic, conflict, narrative) and produces a unified ordered timeline.

Key abilities:
  - Auto-extract events from snapshots with domain classification
  - Merge overlapping timelines from multiple snapshots
  - Time-window queries ("what happened between 0200-0400 UTC?")
  - Entity-focused timelines ("show me everything about entity X")
  - Event clustering — group related events into incidents
  - Gap detection — identify suspicious gaps in activity
  - LLM narrative generation — produce human-readable incident reports
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """A single event on the timeline."""
    event_id: str
    timestamp: str  # ISO 8601
    timestamp_epoch: float
    domain: str  # air, maritime, cyber, conflict, economic, narrative, unknown
    event_type: str  # detection, movement, alert, disappearance, reappearance, etc.
    title: str
    description: str
    entity_ids: List[str] = field(default_factory=list)
    location: Optional[Dict[str, float]] = None  # {"lat": ..., "lon": ...}
    severity: str = "info"  # info, low, medium, high, critical
    source: str = ""
    tags: List[str] = field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class Incident:
    """Clustered group of related timeline events."""
    incident_id: str
    title: str
    description: str = ""
    events: List[str] = field(default_factory=list)  # event_ids
    entities: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    severity: str = "info"
    tags: List[str] = field(default_factory=list)


@dataclass
class TimelineGap:
    """A suspicious gap in activity."""
    entity_id: str
    last_seen: str
    resumed_at: str
    gap_minutes: float
    description: str


class TimelineReconstructor:
    """
    Build and query chronological event timelines across all domains.
    """

    MAX_EVENTS = 50000
    CLUSTER_WINDOW_SEC = 600  # 10 minute window for incident clustering

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "timeline_state.json"

        self.events: List[TimelineEvent] = []
        self.incidents: List[Incident] = []
        self._last_seen: Dict[str, float] = {}  # entity_id → last epoch
        self._event_index: Dict[str, int] = {}  # event_id → index in self.events
        self.total_events_ingested = 0

        self._load_state()

    # ── main ingestion ────────────────────────────────────────────────

    def ingest(self, snapshot: IntelSnapshot) -> List[TimelineEvent]:
        """
        Extract events from snapshot and add to timeline.
        Returns list of new events added.
        """
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        now_epoch = now.timestamp()
        new_events: List[TimelineEvent] = []

        # Process flights
        for flight in snapshot.flights:
            hex_code = flight.get("hex", "")
            callsign = (flight.get("callsign") or flight.get("flight") or hex_code).strip()
            lat = self._safe_float(flight.get("lat"))
            lon = self._safe_float(flight.get("lon"))
            eid = f"aircraft_{hex_code}"

            evt = TimelineEvent(
                event_id=hashlib.md5(f"{eid}_{now_epoch}".encode()).hexdigest()[:12],
                timestamp=now_str,
                timestamp_epoch=now_epoch,
                domain="air",
                event_type="detection",
                title=f"Aircraft {callsign} detected",
                description=f"Callsign: {callsign}, Alt: {flight.get('alt_baro', '?')}ft, "
                           f"Speed: {flight.get('gs', '?')}kts",
                entity_ids=[eid],
                location={"lat": lat, "lon": lon} if lat and lon else None,
                severity="info",
                source="adsb",
                tags=["aircraft"],
            )
            new_events.append(evt)
            self._check_gap(eid, now_epoch, new_events)

        # Process military flights
        for flight in snapshot.military_flights:
            hex_code = flight.get("hex", "")
            callsign = (flight.get("callsign") or flight.get("type") or hex_code).strip()
            lat = self._safe_float(flight.get("lat"))
            lon = self._safe_float(flight.get("lon"))
            eid = f"military_{hex_code}"

            evt = TimelineEvent(
                event_id=hashlib.md5(f"{eid}_{now_epoch}".encode()).hexdigest()[:12],
                timestamp=now_str,
                timestamp_epoch=now_epoch,
                domain="air",
                event_type="detection",
                title=f"Military aircraft {callsign} detected",
                description=f"Type: {flight.get('type', '?')}, Alt: {flight.get('alt_baro', '?')}ft",
                entity_ids=[eid],
                location={"lat": lat, "lon": lon} if lat and lon else None,
                severity="medium",
                source="adsb_military",
                tags=["military", "aircraft"],
            )
            new_events.append(evt)
            self._check_gap(eid, now_epoch, new_events)

        # Process ships
        for ship in snapshot.ships:
            mmsi = str(ship.get("mmsi", ship.get("MMSI", "")))
            name = ship.get("name", ship.get("ship_name", mmsi))
            lat = self._safe_float(ship.get("lat"))
            lon = self._safe_float(ship.get("lon"))
            eid = f"vessel_{mmsi}"

            evt = TimelineEvent(
                event_id=hashlib.md5(f"{eid}_{now_epoch}".encode()).hexdigest()[:12],
                timestamp=now_str,
                timestamp_epoch=now_epoch,
                domain="maritime",
                event_type="detection",
                title=f"Vessel {name} detected",
                description=f"MMSI: {mmsi}, Speed: {ship.get('speed', ship.get('sog', '?'))}kts",
                entity_ids=[eid],
                location={"lat": lat, "lon": lon} if lat and lon else None,
                severity="info",
                source="ais",
                tags=["vessel"],
            )
            new_events.append(evt)
            self._check_gap(eid, now_epoch, new_events)

        # Process conflicts
        for conflict in snapshot.conflicts:
            title = conflict.get("title") or conflict.get("name") or "Unknown conflict"
            evt = TimelineEvent(
                event_id=hashlib.md5(f"conflict_{title}_{now_epoch}".encode()).hexdigest()[:12],
                timestamp=now_str,
                timestamp_epoch=now_epoch,
                domain="conflict",
                event_type="report",
                title=title[:120],
                description=(conflict.get("description") or "")[:300],
                severity="high",
                source="conflict_monitor",
                tags=["conflict"],
            )
            new_events.append(evt)

        # Add to master timeline
        for evt in new_events:
            self.events.append(evt)
            self._event_index[evt.event_id] = len(self.events) - 1
            # Update last seen
            for eid in evt.entity_ids:
                self._last_seen[eid] = now_epoch

        self.total_events_ingested += len(new_events)

        # Trim if needed
        if len(self.events) > self.MAX_EVENTS:
            trim = len(self.events) - self.MAX_EVENTS
            self.events = self.events[trim:]
            self._rebuild_index()

        self._save_state()
        return new_events

    def _check_gap(self, eid: str, now_epoch: float,
                   events_list: List[TimelineEvent]):
        """Detect gaps in entity activity and add reappearance events."""
        GAP_THRESHOLD = 3600  # 1 hour
        last = self._last_seen.get(eid)
        if last is not None:
            gap_sec = now_epoch - last
            if gap_sec > GAP_THRESHOLD:
                gap_min = gap_sec / 60
                evt = TimelineEvent(
                    event_id=hashlib.md5(f"gap_{eid}_{now_epoch}".encode()).hexdigest()[:12],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    timestamp_epoch=now_epoch,
                    domain="unknown",
                    event_type="reappearance",
                    title=f"{eid} reappeared after {gap_min:.0f}min gap",
                    description=f"Entity {eid} was missing for {gap_min:.0f} minutes — possible transponder manipulation or covert movement",
                    entity_ids=[eid],
                    severity="high" if gap_min > 120 else "medium",
                    tags=["gap", "suspicious"],
                )
                events_list.append(evt)

    # ── queries ───────────────────────────────────────────────────────

    def query_time_range(self, start_epoch: float,
                          end_epoch: float) -> List[TimelineEvent]:
        """Get events within a time window."""
        return [e for e in self.events
                if start_epoch <= e.timestamp_epoch <= end_epoch]

    def query_entity(self, entity_id: str, limit: int = 100) -> List[TimelineEvent]:
        """Get all events involving an entity."""
        results = [e for e in self.events if entity_id in e.entity_ids]
        return results[-limit:]

    def query_domain(self, domain: str, limit: int = 200) -> List[TimelineEvent]:
        """Get events by intelligence domain."""
        results = [e for e in self.events if e.domain == domain]
        return results[-limit:]

    def query_severity(self, min_severity: str = "medium",
                       limit: int = 100) -> List[TimelineEvent]:
        """Get events above a severity threshold."""
        levels = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        threshold = levels.get(min_severity, 2)
        results = [e for e in self.events
                   if levels.get(e.severity, 0) >= threshold]
        return results[-limit:]

    def search(self, query: str, limit: int = 50) -> List[TimelineEvent]:
        """Full-text search across timeline events."""
        q = query.lower()
        results = [e for e in self.events
                   if q in e.title.lower() or q in e.description.lower()
                   or any(q in t for t in e.tags)]
        return results[-limit:]

    # ── incident clustering ───────────────────────────────────────────

    def detect_incidents(self) -> List[Incident]:
        """
        Cluster related events into incidents based on temporal
        and entity proximity.
        """
        if not self.events:
            return []

        # Sort by time
        sorted_events = sorted(self.events, key=lambda e: e.timestamp_epoch)

        # Simple single-pass clustering
        clusters: List[List[TimelineEvent]] = []
        current_cluster: List[TimelineEvent] = [sorted_events[0]]

        for evt in sorted_events[1:]:
            last_evt = current_cluster[-1]
            time_diff = evt.timestamp_epoch - last_evt.timestamp_epoch

            # Same cluster if within time window AND shares entity or domain
            shared_entity = bool(set(evt.entity_ids) & set(last_evt.entity_ids))
            same_domain = evt.domain == last_evt.domain

            if time_diff <= self.CLUSTER_WINDOW_SEC and (shared_entity or same_domain):
                current_cluster.append(evt)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [evt]

        if len(current_cluster) >= 2:
            clusters.append(current_cluster)

        # Convert to Incidents
        self.incidents = []
        for i, cluster in enumerate(clusters):
            all_entities: Set[str] = set()
            all_domains: Set[str] = set()
            max_severity = "info"
            severity_order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

            for evt in cluster:
                all_entities.update(evt.entity_ids)
                all_domains.add(evt.domain)
                if severity_order.get(evt.severity, 0) > severity_order.get(max_severity, 0):
                    max_severity = evt.severity

            incident = Incident(
                incident_id=hashlib.md5(f"incident_{i}_{cluster[0].timestamp_epoch}".encode()).hexdigest()[:10],
                title=f"Incident: {cluster[0].title[:60]}",
                events=[e.event_id for e in cluster],
                entities=list(all_entities),
                domains=list(all_domains),
                start_time=cluster[0].timestamp,
                end_time=cluster[-1].timestamp,
                severity=max_severity,
            )
            self.incidents.append(incident)

        return self.incidents

    # ── gap analysis ──────────────────────────────────────────────────

    def detect_gaps(self, min_gap_minutes: float = 60) -> List[TimelineGap]:
        """Find entities with suspicious activity gaps."""
        # Build per-entity timeline
        entity_times: Dict[str, List[float]] = {}
        for evt in self.events:
            for eid in evt.entity_ids:
                entity_times.setdefault(eid, []).append(evt.timestamp_epoch)

        gaps: List[TimelineGap] = []
        for eid, times in entity_times.items():
            times.sort()
            for i in range(1, len(times)):
                gap_sec = times[i] - times[i - 1]
                gap_min = gap_sec / 60
                if gap_min >= min_gap_minutes:
                    gaps.append(TimelineGap(
                        entity_id=eid,
                        last_seen=datetime.fromtimestamp(times[i - 1], tz=timezone.utc).isoformat(),
                        resumed_at=datetime.fromtimestamp(times[i], tz=timezone.utc).isoformat(),
                        gap_minutes=gap_min,
                        description=f"{eid} went dark for {gap_min:.0f}min",
                    ))

        gaps.sort(key=lambda g: g.gap_minutes, reverse=True)
        return gaps[:100]

    # ── LLM narrative ─────────────────────────────────────────────────

    async def generate_narrative(self, events: List[TimelineEvent],
                                 llm: Any) -> str:
        """Generate a human-readable narrative from timeline events."""
        if not events:
            return "No events to narrate."

        if not llm:
            return self._rule_narrative(events)

        events_text = "\n".join(
            f"[{e.timestamp}] ({e.domain}/{e.severity}) {e.title}: {e.description}"
            for e in events[:30]
        )

        prompt = (
            "You are an intelligence analyst. Write a concise narrative "
            "report synthesizing these chronological events into a coherent "
            "intelligence briefing. Focus on patterns, escalation, and "
            "implications.\n\n"
            f"EVENTS:\n{events_text}\n\n"
            "Write the briefing (200-400 words):"
        )

        try:
            return await llm.ask(prompt)
        except Exception as e:
            logger.warning("LLM narrative generation failed: %s", e)
            return self._rule_narrative(events)

    def _rule_narrative(self, events: List[TimelineEvent]) -> str:
        """Fallback rule-based narrative."""
        lines = [f"Timeline Report — {len(events)} events"]
        if events:
            lines.append(f"Period: {events[0].timestamp} to {events[-1].timestamp}")

        domain_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        for e in events:
            domain_counts[e.domain] = domain_counts.get(e.domain, 0) + 1
            severity_counts[e.severity] = severity_counts.get(e.severity, 0) + 1

        lines.append(f"Domains: {domain_counts}")
        lines.append(f"Severity: {severity_counts}")

        high_events = [e for e in events if e.severity in ("high", "critical")]
        if high_events:
            lines.append(f"\nHigh-priority events ({len(high_events)}):")
            for e in high_events[:10]:
                lines.append(f"  [{e.timestamp}] {e.title}")

        return "\n".join(lines)

    # ── utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _rebuild_index(self):
        self._event_index = {e.event_id: i for i, e in enumerate(self.events)}

    def _save_state(self):
        try:
            state = {
                "total_events_ingested": self.total_events_ingested,
                "current_events": len(self.events),
                "entities_tracked": len(self._last_seen),
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save timeline state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_events_ingested = state.get("total_events_ingested", 0)
        except Exception as e:
            logger.warning("Failed to load timeline state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        domain_dist: Dict[str, int] = {}
        for e in self.events:
            domain_dist[e.domain] = domain_dist.get(e.domain, 0) + 1

        return {
            "total_events_ingested": self.total_events_ingested,
            "current_events": len(self.events),
            "entities_tracked": len(self._last_seen),
            "incidents": len(self.incidents),
            "domain_distribution": domain_dist,
        }
