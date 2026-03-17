"""
#14 — Kill Chain Tracker (MITRE ATT&CK-Inspired)

Maps observed intelligence events onto an adversary kill-chain model.
Tracks how threats progress through phases, detects multi-indicator
alignment, and raises alarms when a chain advances beyond threshold.

Phases (hybrid MITRE ATT&CK + intelligence cycle):
  1. Reconnaissance — probing, scanning, unusual OSINT collection
  2. Staging        — force buildup, logistics, pre-positioning
  3. Mobilisation   — active movement to target area
  4. Engagement     — weapons release, interference, blockade
  5. Exploitation   — territory seizure, infrastructure capture
  6. Consolidation  — occupation, narrative control, fortification
  7. Sustainment    — supply chains, long-term presence

This goes far beyond Palantir — their kill chain is IT/cyber focused.
Ours is GEOPOLITICAL + MILITARY + CYBER across all domains.
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


# ---------------------------------------------------------------------------
# Kill chain model
# ---------------------------------------------------------------------------

KILL_CHAIN_PHASES = [
    "reconnaissance",
    "staging",
    "mobilisation",
    "engagement",
    "exploitation",
    "consolidation",
    "sustainment",
]

PHASE_INDEX = {p: i for i, p in enumerate(KILL_CHAIN_PHASES)}


@dataclass
class KillChainIndicator:
    """A single observed indicator mapped to a kill chain phase."""
    indicator_id: str
    phase: str
    description: str
    source_domain: str  # air, maritime, cyber, economic, narrative, conflict
    confidence: float  # 0.0-1.0
    timestamp: str  # ISO
    entity_id: Optional[str] = None
    raw_evidence: Optional[str] = None


@dataclass
class KillChainTrack:
    """An adversary/scenario tracked through kill chain phases."""
    track_id: str
    name: str
    description: str = ""
    adversary: str = ""
    target: str = ""
    created_at: str = ""
    last_updated: str = ""
    # Phase progression — each phase has list of indicators
    phases: Dict[str, List[KillChainIndicator]] = field(default_factory=dict)
    # Highest phase reached
    max_phase: str = "reconnaissance"
    max_phase_index: int = 0
    # Alert level (auto-computed)
    threat_level: str = "low"  # low, elevated, high, critical
    active: bool = True

    def __post_init__(self):
        if not self.phases:
            self.phases = {p: [] for p in KILL_CHAIN_PHASES}

    def total_indicators(self) -> int:
        return sum(len(indics) for indics in self.phases.values())


@dataclass
class KillChainAlert:
    """Alert when kill chain advances to a new phase."""
    alert_id: str
    track_id: str
    track_name: str
    previous_phase: str
    new_phase: str
    new_phase_index: int
    threat_level: str
    timestamp: str
    description: str
    indicators_in_phase: int


# ---------------------------------------------------------------------------
# Phase detection rules (rule-based, no ML required)
# ---------------------------------------------------------------------------

PHASE_RULES: Dict[str, List[Dict[str, Any]]] = {
    "reconnaissance": [
        {"domain": "air", "pattern": "unusual_surveillance",
         "desc": "Surveillance aircraft (RC-135, P-8, E-3) operating near adversary"},
        {"domain": "maritime", "pattern": "intel_vessel",
         "desc": "Intelligence gathering vessel (AGI) deployed"},
        {"domain": "cyber", "pattern": "scanning",
         "desc": "Port scanning or probing of critical infrastructure"},
        {"domain": "narrative", "pattern": "disinfo_probe",
         "desc": "Disinformation testing — trial narratives planted"},
    ],
    "staging": [
        {"domain": "air", "pattern": "tanker_deployment",
         "desc": "Air-to-air refueling tankers repositioned forward"},
        {"domain": "maritime", "pattern": "fleet_assembly",
         "desc": "Naval vessels assembling outside normal pattern"},
        {"domain": "conflict", "pattern": "troop_buildup",
         "desc": "Troop or equipment movement toward border"},
        {"domain": "economic", "pattern": "sanctions_evasion",
         "desc": "Pre-conflict financial restructuring detected"},
    ],
    "mobilisation": [
        {"domain": "air", "pattern": "combat_sortie_surge",
         "desc": "Combat aircraft sortie rate exceeds baseline by >3x"},
        {"domain": "maritime", "pattern": "fleet_movement",
         "desc": "Naval task group underway toward objective"},
        {"domain": "air", "pattern": "awacs_cap",
         "desc": "AWACS/AEW establishing Combat Air Patrol orbit"},
        {"domain": "conflict", "pattern": "mobilization_order",
         "desc": "Military mobilization announcement or reserve call-up"},
    ],
    "engagement": [
        {"domain": "conflict", "pattern": "kinetic_action",
         "desc": "Confirmed weapons release or kinetic action"},
        {"domain": "maritime", "pattern": "blockade",
         "desc": "Naval blockade established on shipping lane"},
        {"domain": "cyber", "pattern": "active_attack",
         "desc": "Active cyber attack on infrastructure"},
        {"domain": "air", "pattern": "no_fly_violation",
         "desc": "No-fly zone established or violated"},
    ],
    "exploitation": [
        {"domain": "conflict", "pattern": "territory_seizure",
         "desc": "Territory or infrastructure seized"},
        {"domain": "maritime", "pattern": "port_control",
         "desc": "Port or waterway under hostile control"},
        {"domain": "cyber", "pattern": "infrastructure_compromise",
         "desc": "Critical infrastructure compromised and controlled"},
    ],
    "consolidation": [
        {"domain": "narrative", "pattern": "propaganda_surge",
         "desc": "State media propaganda surge to justify action"},
        {"domain": "conflict", "pattern": "fortification",
         "desc": "Defensive positions being constructed"},
        {"domain": "economic", "pattern": "asset_seizure",
         "desc": "Economic assets seized or nationalized"},
    ],
    "sustainment": [
        {"domain": "maritime", "pattern": "logistics_chain",
         "desc": "Supply chain established for sustained operations"},
        {"domain": "conflict", "pattern": "occupation_governance",
         "desc": "Occupation governance structures established"},
        {"domain": "air", "pattern": "persistent_cap",
         "desc": "Persistent combat air patrol maintained"},
    ],
}

# Keywords that help match events to phases (for automatic classification)
PHASE_KEYWORDS: Dict[str, List[str]] = {
    "reconnaissance": [
        "surveillance", "recon", "probe", "scanning", "RC-135", "P-8",
        "E-3", "AGI", "patrol", "monitoring", "intel", "ELINT", "SIGINT",
    ],
    "staging": [
        "tanker", "KC-135", "staging", "preposition", "buildup", "assembly",
        "logistic", "reserve", "deploy", "forward", "amass",
    ],
    "mobilisation": [
        "sortie", "surge", "mobiliz", "underway", "task group", "AWACS",
        "combat air patrol", "CAP", "battle group", "carrier strike",
    ],
    "engagement": [
        "strike", "attack", "launch", "missile", "engag", "kinetic",
        "blockade", "intercept", "shoot", "destroy", "weapon",
    ],
    "exploitation": [
        "seizure", "captured", "occupy", "control", "breach", "penetrat",
        "compromise", "taken",
    ],
    "consolidation": [
        "fortif", "consolidat", "propaganda", "justify", "narrative",
        "construct", "establish",
    ],
    "sustainment": [
        "supply", "sustain", "maintain", "resupply", "logistics chain",
        "occupation", "garrison",
    ],
}


# ---------------------------------------------------------------------------
# Kill Chain Tracker Engine
# ---------------------------------------------------------------------------

SURVEILLANCE_TYPES = {
    "RC135", "RC-135", "P8", "P-8", "E3", "E-3", "E8", "E-8",
    "RIVET", "COBRA", "JSTARS", "AWACS", "SENTRY", "POSEIDON",
    "ORION", "TRITON", "GLOBAL HAWK", "RQ-4", "MQ-4C", "MQ-9",
    "U2", "U-2", "EP-3", "RC-12",
}

TANKER_TYPES = {"KC135", "KC-135", "KC10", "KC-10", "KC46", "KC-46", "A330MRTT", "VOYAGER"}


class KillChainTracker:
    """
    Track adversary progression through kill chain phases.

    Auto-detects indicators from snapshots and maps them.
    Also accepts manual indicator injection for analyst-driven tracking.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "killchain_state.json"

        self.tracks: Dict[str, KillChainTrack] = {}
        self.alerts: List[KillChainAlert] = []
        self.total_indicators = 0

        self._load_state()

    # ── track management ──────────────────────────────────────────────

    def create_track(self, name: str, adversary: str = "",
                     target: str = "", description: str = "") -> str:
        """Create a new kill chain tracking scenario."""
        tid = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:10]
        now = datetime.now(timezone.utc).isoformat()
        self.tracks[tid] = KillChainTrack(
            track_id=tid,
            name=name,
            adversary=adversary,
            target=target,
            description=description,
            created_at=now,
            last_updated=now,
        )
        self._save_state()
        return tid

    def add_indicator(self, track_id: str, phase: str,
                      description: str, source_domain: str,
                      confidence: float = 0.7,
                      entity_id: Optional[str] = None,
                      raw_evidence: Optional[str] = None) -> Optional[KillChainAlert]:
        """
        Manually add an indicator to a track.
        Returns a KillChainAlert if the chain advanced to a new phase.
        """
        track = self.tracks.get(track_id)
        if not track or phase not in PHASE_INDEX:
            return None

        now = datetime.now(timezone.utc).isoformat()
        iid = hashlib.md5(f"{track_id}_{phase}_{description}_{time.time()}".encode()).hexdigest()[:10]

        indicator = KillChainIndicator(
            indicator_id=iid,
            phase=phase,
            description=description,
            source_domain=source_domain,
            confidence=confidence,
            timestamp=now,
            entity_id=entity_id,
            raw_evidence=raw_evidence,
        )
        track.phases[phase].append(indicator)
        track.last_updated = now
        self.total_indicators += 1

        return self._update_track_phase(track)

    # ── auto-classification from snapshots ────────────────────────────

    def observe(self, snapshot: IntelSnapshot,
                track_id: Optional[str] = None) -> List[KillChainAlert]:
        """
        Auto-detect kill chain indicators from a snapshot.

        If track_id is given, adds indicators to that track.
        Otherwise creates/uses a default global track.
        """
        if track_id and track_id in self.tracks:
            target_track = self.tracks[track_id]
        else:
            if "__global__" not in self.tracks:
                self.create_track("Global Threat Monitor",
                                  description="Auto-generated global threat tracking")
                self.tracks["__global__"] = self.tracks[
                    list(self.tracks.keys())[-1]
                ]
                # Re-key it
                old_id = list(self.tracks.keys())[-2] if len(self.tracks) > 1 else list(self.tracks.keys())[0]
                if old_id != "__global__":
                    track = self.tracks.pop(old_id)
                    track.track_id = "__global__"
                    self.tracks["__global__"] = track
            target_track = self.tracks["__global__"]

        indicators = self._extract_indicators(snapshot)
        alerts: List[KillChainAlert] = []

        now = datetime.now(timezone.utc).isoformat()
        for phase, desc, domain, conf, eid in indicators:
            iid = hashlib.md5(f"{phase}_{desc}_{time.time()}".encode()).hexdigest()[:10]
            ind = KillChainIndicator(
                indicator_id=iid,
                phase=phase,
                description=desc,
                source_domain=domain,
                confidence=conf,
                timestamp=now,
                entity_id=eid,
            )
            target_track.phases[phase].append(ind)
            self.total_indicators += 1

        target_track.last_updated = now
        alert = self._update_track_phase(target_track)
        if alert:
            alerts.append(alert)

        self._save_state()
        return alerts

    def _extract_indicators(self, snapshot: IntelSnapshot) -> List[tuple]:
        """
        Auto-detect indicators from snapshot data.
        Returns list of (phase, description, domain, confidence, entity_id).
        """
        indicators: List[tuple] = []

        # Air domain
        for flight in snapshot.military_flights:
            callsign = (flight.get("callsign") or flight.get("type") or "").upper()
            aircraft_type = (flight.get("type") or "").upper()
            hex_code = flight.get("hex", "")

            # Reconnaissance phase — surveillance aircraft
            for st in SURVEILLANCE_TYPES:
                if st in callsign or st in aircraft_type:
                    indicators.append((
                        "reconnaissance",
                        f"Surveillance asset detected: {callsign} ({aircraft_type})",
                        "air", 0.8,
                        f"military_{hex_code}",
                    ))
                    break

            # Staging phase — tanker activity
            for tt in TANKER_TYPES:
                if tt in callsign or tt in aircraft_type:
                    indicators.append((
                        "staging",
                        f"Tanker asset deployed: {callsign} ({aircraft_type})",
                        "air", 0.6,
                        f"military_{hex_code}",
                    ))
                    break

        # Military flights count as mobilisation signal if count exceeds threshold
        mil_count = len(snapshot.military_flights)
        if mil_count > 20:
            indicators.append((
                "mobilisation",
                f"Elevated military air activity: {mil_count} concurrent military flights",
                "air", min(0.9, 0.5 + mil_count / 100),
                None,
            ))

        # Maritime domain
        ship_count = 0
        dark_ships = 0
        for ship in snapshot.ships:
            ship_count += 1
            ship_type = (ship.get("type") or ship.get("ship_type") or "").upper()
            speed = ship.get("speed", ship.get("sog"))

            # Vessels with no AIS transponder = dark ship
            if not ship.get("mmsi") and not ship.get("MMSI"):
                dark_ships += 1

            # Naval vessels in unusual areas
            if any(t in ship_type for t in ["MILITARY", "WARSHIP", "NAVAL", "PATROL"]):
                indicators.append((
                    "staging",
                    f"Military vessel detected: {ship.get('name', 'UNKNOWN')} ({ship_type})",
                    "maritime", 0.6,
                    f"vessel_{ship.get('mmsi', ship.get('MMSI', ''))}",
                ))

        if dark_ships > 5:
            indicators.append((
                "reconnaissance",
                f"Cluster of {dark_ships} dark vessels (no AIS) — possible covert monitoring",
                "maritime", 0.7,
                None,
            ))

        # Cyber domain — from snapshot conflicts / cyber threats
        for conflict in snapshot.conflicts:
            conflict_type = (conflict.get("type") or conflict.get("category") or "").lower()
            conflict_desc = (conflict.get("description") or conflict.get("title") or "").lower()

            if "cyber" in conflict_type or "cyber" in conflict_desc:
                if any(w in conflict_desc for w in ["scan", "probe", "recon"]):
                    indicators.append(("reconnaissance", f"Cyber: {conflict_desc[:100]}", "cyber", 0.7, None))
                elif any(w in conflict_desc for w in ["attack", "ddos", "ransomware"]):
                    indicators.append(("engagement", f"Cyber attack: {conflict_desc[:100]}", "cyber", 0.8, None))
            elif any(w in conflict_desc for w in ["mobiliz", "troops", "buildup"]):
                indicators.append(("mobilisation", f"Conflict: {conflict_desc[:100]}", "conflict", 0.7, None))
            elif any(w in conflict_desc for w in ["strike", "attack", "missile"]):
                indicators.append(("engagement", f"Kinetic: {conflict_desc[:100]}", "conflict", 0.8, None))

        return indicators

    # ── phase progression ─────────────────────────────────────────────

    def _update_track_phase(self, track: KillChainTrack) -> Optional[KillChainAlert]:
        """
        Recalculate highest phase and threat level.
        Returns alert if phase advanced.
        """
        old_max = track.max_phase_index
        old_phase = track.max_phase

        # Find highest phase with indicators
        new_max = 0
        for phase, indics in track.phases.items():
            if indics and PHASE_INDEX[phase] > new_max:
                new_max = PHASE_INDEX[phase]

        track.max_phase_index = new_max
        track.max_phase = KILL_CHAIN_PHASES[new_max]

        # Compute threat level
        if new_max >= 5:
            track.threat_level = "critical"
        elif new_max >= 3:
            track.threat_level = "high"
        elif new_max >= 2:
            track.threat_level = "elevated"
        else:
            track.threat_level = "low"

        # Fire alert on phase advance
        if new_max > old_max:
            now = datetime.now(timezone.utc).isoformat()
            alert = KillChainAlert(
                alert_id=hashlib.md5(f"{track.track_id}_{new_max}_{time.time()}".encode()).hexdigest()[:10],
                track_id=track.track_id,
                track_name=track.name,
                previous_phase=old_phase,
                new_phase=track.max_phase,
                new_phase_index=new_max,
                threat_level=track.threat_level,
                timestamp=now,
                description=f"Kill chain advanced: {old_phase} → {track.max_phase} ({track.name})",
                indicators_in_phase=len(track.phases[track.max_phase]),
            )
            self.alerts.append(alert)
            if len(self.alerts) > 500:
                self.alerts = self.alerts[-250:]
            return alert
        return None

    # ── queries ───────────────────────────────────────────────────────

    def get_track(self, track_id: str) -> Optional[KillChainTrack]:
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> List[KillChainTrack]:
        return list(self.tracks.values())

    def get_track_summary(self, track_id: str) -> Optional[Dict[str, Any]]:
        track = self.tracks.get(track_id)
        if not track:
            return None
        return {
            "track_id": track.track_id,
            "name": track.name,
            "adversary": track.adversary,
            "max_phase": track.max_phase,
            "max_phase_index": track.max_phase_index,
            "threat_level": track.threat_level,
            "total_indicators": track.total_indicators(),
            "phase_breakdown": {p: len(indics) for p, indics in track.phases.items()},
            "active": track.active,
        }

    def get_recent_alerts(self, limit: int = 20) -> List[KillChainAlert]:
        return self.alerts[-limit:]

    def search_indicators(self, query: str) -> List[KillChainIndicator]:
        """Full-text search across all indicators."""
        q = query.lower()
        results: List[KillChainIndicator] = []
        for track in self.tracks.values():
            for phase_indics in track.phases.values():
                for ind in phase_indics:
                    if q in ind.description.lower() or q in (ind.source_domain or ""):
                        results.append(ind)
        return results[:100]

    # ── LLM enrichment ────────────────────────────────────────────────

    async def classify_event(self, event_text: str, llm: Any) -> Optional[KillChainIndicator]:
        """Use LLM to classify a free-text event into kill chain phase."""
        if not llm:
            return self._rule_classify(event_text)

        phases_str = ", ".join(KILL_CHAIN_PHASES)
        prompt = (
            f"Classify this intelligence event into ONE kill chain phase.\n"
            f"Phases: {phases_str}\n\n"
            f"Event: {event_text}\n\n"
            f"Reply ONLY with JSON: {{\"phase\": \"...\", \"confidence\": 0.X, "
            f"\"domain\": \"air|maritime|cyber|conflict|economic|narrative\"}}"
        )

        try:
            resp = await llm.ask(prompt)
            data = json.loads(resp)
            phase = data.get("phase", "").lower()
            if phase in PHASE_INDEX:
                now = datetime.now(timezone.utc).isoformat()
                return KillChainIndicator(
                    indicator_id=hashlib.md5(f"llm_{event_text[:50]}_{time.time()}".encode()).hexdigest()[:10],
                    phase=phase,
                    description=event_text[:200],
                    source_domain=data.get("domain", "unknown"),
                    confidence=float(data.get("confidence", 0.5)),
                    timestamp=now,
                )
        except Exception as e:
            logger.warning("LLM kill chain classification failed: %s", e)

        return self._rule_classify(event_text)

    def _rule_classify(self, text: str) -> Optional[KillChainIndicator]:
        """Fallback rule-based classification."""
        text_lower = text.lower()
        best_phase = None
        best_score = 0

        for phase, keywords in PHASE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_phase = phase

        if best_phase and best_score > 0:
            now = datetime.now(timezone.utc).isoformat()
            return KillChainIndicator(
                indicator_id=hashlib.md5(f"rule_{text[:50]}_{time.time()}".encode()).hexdigest()[:10],
                phase=best_phase,
                description=text[:200],
                source_domain="unknown",
                confidence=min(0.8, 0.3 + best_score * 0.15),
                timestamp=now,
            )
        return None

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "total_indicators": self.total_indicators,
                "track_count": len(self.tracks),
                "alert_count": len(self.alerts),
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save kill chain state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_indicators = state.get("total_indicators", 0)
        except Exception as e:
            logger.warning("Failed to load kill chain state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tracks": len(self.tracks),
            "active_tracks": sum(1 for t in self.tracks.values() if t.active),
            "total_indicators": self.total_indicators,
            "total_alerts": len(self.alerts),
            "highest_threat": max((t.threat_level for t in self.tracks.values()),
                                  default="none", key=lambda x: ["none", "low", "elevated", "high", "critical"].index(x)
                                  if x in ["none", "low", "elevated", "high", "critical"] else 0),
        }
