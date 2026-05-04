"""
#3,  Swarm Intelligence Threat Assessment

When an anomaly is detected in Zunvra data, launches multiple parallel
"thought-agents" that each develop a competing hypothesis about what's
happening. They vote and produce a consensus assessment with confidence.

Output example:
  SABLE SWARM ASSESSMENT,  Anomaly: Eastern Mediterranean
  Hypothesis A (military exercise): 0.73 confidence,  Agent #1
  Hypothesis B (escalation): 0.21,  Agent #2
  Hypothesis C (sensor noise): 0.06,  Agent #3
  CONSENSUS: Most likely a scheduled military exercise (73%)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ThoughtAgent:
    agent_id: str
    hypothesis: str
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "exploring"  # exploring | concluded | discarded


@dataclass
class SwarmAssessment:
    assessment_id: str
    trigger: str
    trigger_region: Optional[str] = None
    trigger_lat: Optional[float] = None
    trigger_lon: Optional[float] = None
    agents: List[ThoughtAgent] = field(default_factory=list)
    consensus: str = ""
    consensus_confidence: float = 0.0
    threat_level: str = "LOW"  # LOW | MODERATE | ELEVATED | HIGH | CRITICAL
    timestamp: str = ""
    domains_involved: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"SABLE SWARM ASSESSMENT,  {self.assessment_id}",
            f"Trigger: {self.trigger}",
            f"Threat Level: {self.threat_level}",
            f"Timestamp: {self.timestamp}",
            "",
        ]
        for agent in sorted(self.agents, key=lambda a: a.confidence, reverse=True):
            lines.append(f"  [{agent.confidence:.0%}] {agent.hypothesis}")
            if agent.evidence:
                for ev in agent.evidence[:3]:
                    lines.append(f"       Evidence: {ev}")
        lines.append("")
        lines.append(f"CONSENSUS: {self.consensus} (confidence {self.consensus_confidence:.0%})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anomaly detection rules
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    anomaly_type: str
    description: str
    region: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    severity: float = 0.5
    domains: List[str] = field(default_factory=list)
    data_context: Dict[str, Any] = field(default_factory=dict)


def detect_anomalies(
    current: IntelSnapshot,
    previous: Optional[IntelSnapshot] = None,
) -> List[Anomaly]:
    """Rule-based anomaly detection from snapshot data."""
    anomalies: List[Anomaly] = []

    # 1. Unusual military flight density
    if len(current.military_flights) > 20:
        anomalies.append(Anomaly(
            anomaly_type="military_surge",
            description=f"Unusually high military flight count: {len(current.military_flights)}",
            severity=min(1.0, len(current.military_flights) / 50),
            domains=["military_flights"],
            data_context={"count": len(current.military_flights)},
        ))

    # 2. GPS jamming active
    if len(current.gps_jamming) > 0:
        anomalies.append(Anomaly(
            anomaly_type="gps_jamming",
            description=f"Active GPS jamming: {len(current.gps_jamming)} zones",
            severity=0.7,
            domains=["gps_jamming", "military_flights"],
        ))

    # 3. Cyber spike
    if len(current.cyber_threats) > 10:
        anomalies.append(Anomaly(
            anomaly_type="cyber_spike",
            description=f"Elevated cyber threat count: {len(current.cyber_threats)}",
            severity=min(1.0, len(current.cyber_threats) / 30),
            domains=["cyber_threats"],
        ))

    # 4. Multi-domain co-occurrence
    active_domains = 0
    if len(current.military_flights) > 5: active_domains += 1
    if len(current.gps_jamming) > 0: active_domains += 1
    if len(current.cyber_threats) > 5: active_domains += 1
    if len(current.gdelt_events) > 30: active_domains += 1
    if len(current.carriers) > 0: active_domains += 1
    if active_domains >= 3:
        anomalies.append(Anomaly(
            anomaly_type="multi_domain_convergence",
            description=f"Convergence across {active_domains} threat domains",
            severity=0.8 + (active_domains - 3) * 0.05,
            domains=["military_flights", "gps_jamming", "cyber_threats", "gdelt_events"],
        ))

    # 5. Significant earthquake
    for eq in current.earthquakes:
        mag = eq.get("magnitude") or eq.get("mag") or 0
        try:
            if float(mag) >= 6.0:
                anomalies.append(Anomaly(
                    anomaly_type="major_earthquake",
                    description=f"Major earthquake M{mag}",
                    lat=float(eq.get("lat") or eq.get("latitude") or 0),
                    lon=float(eq.get("lon") or eq.get("longitude") or 0),
                    severity=min(1.0, float(mag) / 9.0),
                    domains=["earthquakes"],
                ))
        except (ValueError, TypeError):
            continue

    # 6. Delta-based: sudden increase if previous snapshot available
    if previous:
        mil_delta = len(current.military_flights) - len(previous.military_flights)
        if mil_delta > 10:
            anomalies.append(Anomaly(
                anomaly_type="military_surge_delta",
                description=f"Military flights increased by {mil_delta} since last snapshot",
                severity=min(1.0, mil_delta / 30),
                domains=["military_flights"],
            ))

        cyber_delta = len(current.cyber_threats) - len(previous.cyber_threats)
        if cyber_delta > 5:
            anomalies.append(Anomaly(
                anomaly_type="cyber_surge_delta",
                description=f"Cyber threats increased by {cyber_delta} since last snapshot",
                severity=min(1.0, cyber_delta / 20),
                domains=["cyber_threats"],
            ))

    return anomalies


# ---------------------------------------------------------------------------
# Swarm engine
# ---------------------------------------------------------------------------

class SwarmThreatAssessor:
    """
    Uses LLM-powered parallel thought-agents to assess OSINT anomalies.
    Falls back to rule-based assessment when no LLM is available.
    """

    MAX_ASSESSMENTS = 200

    def __init__(self, data_dir: Optional[Path] = None, num_agents: int = 3):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "swarm_assessments.json"
        self.num_agents = num_agents

        self.assessments: List[SwarmAssessment] = []
        self.total_assessments = 0
        self._load_state()

    async def assess(
        self,
        anomaly: Anomaly,
        snapshot: IntelSnapshot,
        llm=None,
    ) -> SwarmAssessment:
        """
        Launch a swarm of thought-agents to assess an anomaly.
        Each agent develops an independent hypothesis, then they vote.
        """
        now = datetime.now(timezone.utc).isoformat()
        aid = hashlib.sha256(f"{now}_{anomaly.anomaly_type}".encode()).hexdigest()[:10]

        assessment = SwarmAssessment(
            assessment_id=aid,
            trigger=anomaly.description,
            trigger_region=anomaly.region,
            trigger_lat=anomaly.lat,
            trigger_lon=anomaly.lon,
            timestamp=now,
            domains_involved=anomaly.domains,
        )

        if llm:
            agents = await self._llm_swarm(llm, anomaly, snapshot)
        else:
            agents = self._rule_swarm(anomaly, snapshot)

        assessment.agents = agents

        # Vote: weighted by confidence
        if agents:
            winner = max(agents, key=lambda a: a.confidence)
            assessment.consensus = winner.hypothesis
            assessment.consensus_confidence = winner.confidence

            # Threat level from consensus confidence + severity
            combined = (anomaly.severity + winner.confidence) / 2
            if combined >= 0.8:
                assessment.threat_level = "CRITICAL"
            elif combined >= 0.65:
                assessment.threat_level = "HIGH"
            elif combined >= 0.5:
                assessment.threat_level = "ELEVATED"
            elif combined >= 0.35:
                assessment.threat_level = "MODERATE"
            else:
                assessment.threat_level = "LOW"

        self.assessments.append(assessment)
        if len(self.assessments) > self.MAX_ASSESSMENTS:
            self.assessments = self.assessments[-self.MAX_ASSESSMENTS:]
        self.total_assessments += 1
        self._save_state()

        return assessment

    async def assess_all(
        self,
        snapshot: IntelSnapshot,
        previous: Optional[IntelSnapshot] = None,
        llm=None,
    ) -> List[SwarmAssessment]:
        """Detect anomalies and assess each one."""
        anomalies = detect_anomalies(snapshot, previous)
        results = []
        for anomaly in anomalies[:5]:  # cap to avoid LLM overuse
            assessment = await self.assess(anomaly, snapshot, llm)
            results.append(assessment)
        return results

    # ── LLM swarm ─────────────────────────────────────────────────────

    async def _llm_swarm(
        self, llm, anomaly: Anomaly, snapshot: IntelSnapshot
    ) -> List[ThoughtAgent]:
        """Launch thought-agents via LLM for competing hypotheses."""
        prompt = (
            "You are a swarm of intelligence analysts assessing an anomaly.\n\n"
            f"ANOMALY: {anomaly.description}\n"
            f"Domains: {', '.join(anomaly.domains)}\n"
            f"Severity: {anomaly.severity:.2f}\n\n"
            f"Global context:\n{snapshot.summary_text(1000)}\n\n"
            f"Generate {self.num_agents} COMPETING hypotheses. Each must be different.\n"
            "Return JSON array:\n"
            "[\n"
            '  {"hypothesis": "brief hypothesis", "reasoning": "why this could be true",\n'
            '   "evidence": ["evidence point 1", "evidence point 2"],\n'
            '   "confidence": 0.0-1.0}\n'
            "]\n\n"
            "Hypotheses should range from benign (routine/error) to serious (escalation/attack).\n"
            "Confidences must sum to approximately 1.0."
        )

        try:
            raw = await llm.chat_raw(prompt, max_tokens=600)
            import re
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if not m:
                return self._rule_swarm(anomaly, snapshot)

            items = json.loads(m.group())
            agents = []
            for i, item in enumerate(items[:self.num_agents]):
                agents.append(ThoughtAgent(
                    agent_id=f"TA-{i+1}",
                    hypothesis=item.get("hypothesis", f"Hypothesis {i+1}"),
                    reasoning=item.get("reasoning", ""),
                    evidence=item.get("evidence", []),
                    confidence=float(item.get("confidence", 1.0 / self.num_agents)),
                    status="concluded",
                ))
            return agents
        except Exception as e:
            logger.warning("LLM swarm failed: %s,  falling back to rules", e)
            return self._rule_swarm(anomaly, snapshot)

    # ── Rule-based fallback ───────────────────────────────────────────

    def _rule_swarm(self, anomaly: Anomaly, snapshot: IntelSnapshot) -> List[ThoughtAgent]:
        """Generate deterministic hypotheses without LLM."""
        agents = []

        # Always generate 3 competing views
        if "military" in anomaly.anomaly_type or "military_flights" in anomaly.domains:
            agents.append(ThoughtAgent(
                agent_id="TA-1", hypothesis="Scheduled military exercise",
                reasoning="Periodic exercises produce similar patterns",
                evidence=[f"Military flight count: {len(snapshot.military_flights)}"],
                confidence=0.50, status="concluded"))
            agents.append(ThoughtAgent(
                agent_id="TA-2", hypothesis="Genuine escalation or force repositioning",
                reasoning="Pattern doesn't match typical exercise schedule",
                evidence=[f"GDELT conflict events: {len(snapshot.gdelt_events)}"],
                confidence=0.30, status="concluded"))
            agents.append(ThoughtAgent(
                agent_id="TA-3", hypothesis="ADS-B data artifact or sensor error",
                reasoning="Occasionally duplicate transponders inflate counts",
                confidence=0.20, status="concluded"))
        elif "cyber" in anomaly.anomaly_type:
            agents.append(ThoughtAgent(
                agent_id="TA-1", hypothesis="Coordinated cyber campaign targeting infrastructure",
                confidence=0.45, status="concluded"))
            agents.append(ThoughtAgent(
                agent_id="TA-2", hypothesis="Botnet scan noise,  not targeted",
                confidence=0.35, status="concluded"))
            agents.append(ThoughtAgent(
                agent_id="TA-3", hypothesis="C2 feed false positives",
                confidence=0.20, status="concluded"))
        elif "earthquake" in anomaly.anomaly_type:
            agents.append(ThoughtAgent(
                agent_id="TA-1", hypothesis="Natural tectonic event,  no threat escalation",
                confidence=0.60, status="concluded"))
            agents.append(ThoughtAgent(
                agent_id="TA-2", hypothesis="Possible nuclear test disguised as natural event",
                confidence=0.15, status="concluded"))
            agents.append(ThoughtAgent(
                agent_id="TA-3", hypothesis="Potential cascade risk to nearby infrastructure",
                confidence=0.25, status="concluded"))
        else:
            agents.append(ThoughtAgent(
                agent_id="TA-1", hypothesis="Routine activity within normal parameters",
                confidence=0.55, status="concluded"))
            agents.append(ThoughtAgent(
                agent_id="TA-2", hypothesis="Emerging situation requiring closer monitoring",
                confidence=0.30, status="concluded"))
            agents.append(ThoughtAgent(
                agent_id="TA-3", hypothesis="Data irregularity,  sensor or feed issue",
                confidence=0.15, status="concluded"))

        return agents

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "assessments": [asdict(a) for a in self.assessments[-50:]],
                "total_assessments": self.total_assessments,
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save swarm state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_assessments = state.get("total_assessments", 0)
        except Exception as e:
            logger.warning("Failed to load swarm state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_assessments": self.total_assessments,
            "recent_count": len(self.assessments),
            "last_assessment": self.assessments[-1].to_text() if self.assessments else None,
        }
