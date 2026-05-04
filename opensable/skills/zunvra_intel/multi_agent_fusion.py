"""
#10,  Multi-Agent Intelligence Fusion

Three virtual analyst profiles (SIGINT, GEOINT, HUMINT) work the same data
in parallel with different collection focuses.  They communicate through an
inter-agent bridge and produce a MULTI-INT ASSESSMENT when at least two
converge on the same conclusion.

Example:
  SIGINT-Analyst: "GPS jamming signatures detected in Eastern Med."
  GEOINT-Analyst: "New military convoy visible on satellite imagery route."
  HUMINT-Analyst: "Open-source telegram traffic matches pre-exercise chatter."
  FUSION:         CONVERGENCE,  3/3 analysts agree: military exercise imminent
                  in Eastern Mediterranean.  Confidence: HIGH.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analyst profiles,  each focuses on different domains
# ---------------------------------------------------------------------------

@dataclass
class AnalystProfile:
    """Defines the collection focus and personality of a virtual analyst."""
    name: str
    role: str
    domains: List[str]
    analysis_style: str
    priority_keywords: List[str]

ANALYST_PROFILES = {
    "SIGINT": AnalystProfile(
        name="SIGINT-Analyst",
        role="Signals Intelligence",
        domains=["gps_jamming", "cyber_events", "internet_outages", "ransomware"],
        analysis_style="electronic_signals",
        priority_keywords=["jamming", "interference", "cyber", "outage", "encryption",
                          "signal", "frequency", "electronic", "ransomware", "malware"],
    ),
    "GEOINT": AnalystProfile(
        name="GEOINT-Analyst",
        role="Geospatial Intelligence",
        domains=["flights", "ships", "military", "fires", "earthquakes", "satellites"],
        analysis_style="spatial_patterns",
        priority_keywords=["position", "movement", "cluster", "convoy", "fleet",
                          "trajectory", "orbit", "imagery", "terrain", "deployment"],
    ),
    "HUMINT": AnalystProfile(
        name="HUMINT-Analyst",
        role="Human-Source Intelligence",
        domains=["interpol", "dark_intel", "news", "social_media"],
        analysis_style="narrative_context",
        priority_keywords=["report", "source", "chatter", "claim", "arrest",
                          "wanted", "notice", "threat", "group", "organization"],
    ),
}


@dataclass
class AnalystFinding:
    """A single finding from one analyst."""
    analyst: str
    timestamp: str
    domain: str
    assessment: str
    confidence: float  # 0.0 - 1.0
    evidence: List[str] = field(default_factory=list)
    region: str = ""
    threat_level: str = "LOW"


@dataclass
class FusionReport:
    """Multi-INT fusion product when analysts converge."""
    report_id: str
    timestamp: str
    convergence_count: int  # how many analysts agree
    converged_topic: str
    threat_level: str
    confidence: float
    findings: List[AnalystFinding] = field(default_factory=list)
    synthesis: str = ""  # Combined narrative
    recommended_actions: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            "═══ MULTI-INT FUSION REPORT ═══",
            f"  Topic: {self.converged_topic}",
            f"  Convergence: {self.convergence_count}/3 analysts",
            f"  Threat Level: {self.threat_level}",
            f"  Confidence: {self.confidence:.0%}",
            "",
        ]
        for f in self.findings:
            lines.append(f"  [{f.analyst}] {f.assessment}")
            if f.evidence:
                for e in f.evidence[:3]:
                    lines.append(f"    - {e}")
            lines.append("")
        if self.synthesis:
            lines.append(f"  SYNTHESIS: {self.synthesis}")
        if self.recommended_actions:
            lines.append("  RECOMMENDED ACTIONS:")
            for a in self.recommended_actions:
                lines.append(f"    → {a}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual analyst engines
# ---------------------------------------------------------------------------

class VirtualAnalyst:
    """A virtual analyst that focuses on specific intelligence domains."""

    def __init__(self, profile: AnalystProfile):
        self.profile = profile
        self.recent_findings: List[AnalystFinding] = []
        self.finding_history: List[AnalystFinding] = []

    def analyze(self, snapshot: IntelSnapshot) -> List[AnalystFinding]:
        """Analyze snapshot through this analyst's lens."""
        now = datetime.now(timezone.utc).isoformat()
        findings: List[AnalystFinding] = []
        summary = snapshot.summary_text()

        for domain in self.profile.domains:
            finding = self._analyze_domain(snapshot, domain, now)
            if finding:
                findings.append(finding)

        # Cross-domain pattern within this analyst's focus
        if len(findings) >= 2:
            cross = self._cross_domain_insight(findings, now)
            if cross:
                findings.append(cross)

        self.recent_findings = findings
        self.finding_history.extend(findings)
        # Keep history bounded
        if len(self.finding_history) > 200:
            self.finding_history = self.finding_history[-100:]

        return findings

    def _analyze_domain(self, snapshot: IntelSnapshot, domain: str,
                        now: str) -> Optional[AnalystFinding]:
        """Produce a finding for a single domain."""
        # Count entities in this domain from snapshot data
        count = self._count_domain(snapshot, domain)
        if count == 0:
            return None

        # Determine threat level based on thresholds
        threat, confidence = self._assess_threat(domain, count)
        if threat == "NONE":
            return None

        evidence = self._gather_evidence(snapshot, domain)

        return AnalystFinding(
            analyst=self.profile.name,
            timestamp=now,
            domain=domain,
            assessment=self._compose_assessment(domain, count, threat),
            confidence=confidence,
            evidence=evidence,
            threat_level=threat,
        )

    def _count_domain(self, snapshot: IntelSnapshot, domain: str) -> int:
        counts = snapshot.raw.get("counts", snapshot.raw.get("summary", {}))
        # Try exact and fuzzy match
        for key in [domain, domain.replace("_", " "), domain.replace("_", "")]:
            if key in counts:
                try:
                    return int(counts[key])
                except (ValueError, TypeError):
                    pass
        # Try nested structures
        if domain in snapshot.raw:
            val = snapshot.raw[domain]
            if isinstance(val, list):
                return len(val)
            if isinstance(val, dict):
                return len(val)
        return 0

    def _assess_threat(self, domain: str, count: int) -> Tuple[str, float]:
        """Classify threat level for a domain count."""
        thresholds = {
            "gps_jamming":       [(1, "MEDIUM", 0.7), (5, "HIGH", 0.85), (10, "CRITICAL", 0.95)],
            "cyber_events":      [(5, "LOW", 0.5), (20, "MEDIUM", 0.7), (50, "HIGH", 0.85)],
            "internet_outages":  [(1, "LOW", 0.5), (5, "MEDIUM", 0.7), (10, "HIGH", 0.85)],
            "ransomware":        [(1, "MEDIUM", 0.7), (5, "HIGH", 0.85), (10, "CRITICAL", 0.95)],
            "military":          [(5, "LOW", 0.5), (15, "MEDIUM", 0.7), (30, "HIGH", 0.85)],
            "fires":             [(10, "LOW", 0.5), (50, "MEDIUM", 0.6), (100, "HIGH", 0.7)],
            "earthquakes":       [(1, "LOW", 0.5), (3, "MEDIUM", 0.7), (5, "HIGH", 0.8)],
            "interpol":          [(1, "LOW", 0.6), (5, "MEDIUM", 0.7), (10, "HIGH", 0.85)],
        }

        rules = thresholds.get(domain, [(1, "LOW", 0.5), (10, "MEDIUM", 0.7), (50, "HIGH", 0.85)])
        level, conf = "NONE", 0.3

        for threshold, threat, confidence in rules:
            if count >= threshold:
                level, conf = threat, confidence

        return level, conf

    def _gather_evidence(self, snapshot: IntelSnapshot, domain: str) -> List[str]:
        """Extract up to 3 evidence items for a domain."""
        evidence = []
        data = snapshot.raw.get(domain, [])
        if isinstance(data, list):
            for item in data[:3]:
                if isinstance(item, dict):
                    desc = item.get("description", item.get("name", item.get("callsign", "")))
                    if desc:
                        evidence.append(str(desc)[:100])
                elif isinstance(item, str):
                    evidence.append(item[:100])
        return evidence

    def _compose_assessment(self, domain: str, count: int, threat: str) -> str:
        templates = {
            "SIGINT-Analyst": {
                "gps_jamming": f"{count} GPS/GNSS jamming events detected. Electronic warfare activity {'likely' if threat != 'LOW' else 'possible'}.",
                "cyber_events": f"{count} cyber events in monitoring window. {'Coordinated campaign suspected.' if threat in ('HIGH', 'CRITICAL') else 'Normal activity levels.'}",
                "internet_outages": f"{count} internet outage(s) detected. {'Possible state-level disruption.' if threat != 'LOW' else 'Isolated incidents.'}",
                "ransomware": f"{count} ransomware incidents tracked. {'Elevated threat posture.' if threat != 'LOW' else 'Baseline activity.'}",
            },
            "GEOINT-Analyst": {
                "flights": f"{count} flights tracked. {'Unusual density pattern detected.' if count > 3000 else 'Normal traffic.'}",
                "ships": f"{count} vessels monitored. {'High-density maritime corridor.' if count > 15000 else 'Standard traffic.'}",
                "military": f"{count} military assets visible. {'Significant force posture observed.' if count > 10 else 'Routine presence.'}",
                "fires": f"{count} active fire detections. {'Large-scale event in progress.' if count > 50 else 'Monitoring.'}",
                "earthquakes": f"{count} seismic events recorded. {'Significant activity.' if count > 3 else 'Minor activity.'}",
            },
            "HUMINT-Analyst": {
                "interpol": f"{count} active Interpol notices relevant. {'Heightened law enforcement activity.' if count > 5 else 'Routine monitoring.'}",
                "dark_intel": f"{count} dark intelligence items collected. {'Active collection environment.' if count > 3 else 'Baseline.'}",
            },
        }
        analyst_templates = templates.get(self.profile.name, {})
        return analyst_templates.get(domain, f"{domain}: {count} items detected. Threat level: {threat}.")

    def _cross_domain_insight(self, findings: List[AnalystFinding],
                               now: str) -> Optional[AnalystFinding]:
        """Generate insight from multiple findings within the same analyst."""
        high_findings = [f for f in findings if f.threat_level in ("HIGH", "CRITICAL")]
        if len(high_findings) < 2:
            return None

        domains = [f.domain for f in high_findings]
        combo = " + ".join(domains)

        return AnalystFinding(
            analyst=self.profile.name,
            timestamp=now,
            domain="cross_domain",
            assessment=(f"MULTI-DOMAIN ALERT: Simultaneous elevated activity across {combo}. "
                       f"Combined assessment suggests coordinated or escalatory pattern."),
            confidence=0.85,
            evidence=[f.assessment for f in high_findings],
            threat_level="HIGH",
        )


# ---------------------------------------------------------------------------
# Fusion Engine
# ---------------------------------------------------------------------------

class MultiAgentFusion:
    """
    Orchestrates three virtual analysts and produces MULTI-INT fusion
    reports when findings converge.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "fusion_reports.json"

        self.analysts = {
            name: VirtualAnalyst(profile)
            for name, profile in ANALYST_PROFILES.items()
        }
        self.fusion_reports: List[FusionReport] = []
        self.total_reports = 0
        self._load_state()

    # ── main pipeline ─────────────────────────────────────────────────

    async def analyze(
        self,
        snapshot: IntelSnapshot,
        llm=None,
    ) -> Optional[FusionReport]:
        """
        Run all three analysts, then attempt fusion.

        Returns a FusionReport if analysts converge, None otherwise.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Phase 1: Each analyst processes independently
        all_findings: Dict[str, List[AnalystFinding]] = {}
        for name, analyst in self.analysts.items():
            findings = analyst.analyze(snapshot)
            all_findings[name] = findings
            logger.debug("%s produced %d findings", name, len(findings))

        # Phase 2: Check for convergence
        convergences = self._detect_convergence(all_findings)

        if not convergences:
            return None

        # Phase 3: Build fusion report for strongest convergence
        best = max(convergences, key=lambda c: c[2])  # highest confidence
        report = self._build_fusion(best, all_findings, now)

        # Phase 4: LLM synthesis
        if llm:
            try:
                await self._llm_synthesize(llm, report)
            except Exception as e:
                logger.debug("LLM fusion synthesis failed: %s", e)

        # Store
        self.fusion_reports.append(report)
        self.total_reports += 1
        if len(self.fusion_reports) > 100:
            self.fusion_reports = self.fusion_reports[-50:]
        self._save_state()

        return report

    # ── convergence detection ─────────────────────────────────────────

    def _detect_convergence(
        self,
        all_findings: Dict[str, List[AnalystFinding]],
    ) -> List[Tuple[str, int, float]]:
        """
        Detect when multiple analysts agree on a threat.

        Returns list of (topic, analyst_count, confidence).
        """
        convergences = []

        # Strategy 1: Shared high-threat domains
        high_by_analyst: Dict[str, List[str]] = {}
        for name, findings in all_findings.items():
            high_domains = {f.domain for f in findings
                          if f.threat_level in ("HIGH", "CRITICAL")}
            if high_domains:
                high_by_analyst[name] = list(high_domains)

        if len(high_by_analyst) >= 2:
            # All analysts flagging high = convergence
            topic_parts = []
            conf_sum = 0.0
            count = 0
            for name, domains in high_by_analyst.items():
                topic_parts.extend(domains)
                count += 1
                max_conf = max((f.confidence for f in all_findings[name]
                               if f.threat_level in ("HIGH", "CRITICAL")), default=0.7)
                conf_sum += max_conf
            topic = "Multi-domain threat: " + ", ".join(set(topic_parts))
            convergences.append((topic, count, conf_sum / count))

        # Strategy 2: Overlapping domains across analysts
        domain_analysts: Dict[str, List[str]] = {}
        for name, findings in all_findings.items():
            for f in findings:
                if f.domain != "cross_domain" and f.threat_level != "NONE":
                    domain_analysts.setdefault(f.domain, []).append(name)

        for domain, analysts in domain_analysts.items():
            if len(set(analysts)) >= 2:
                conf = 0.75  # Cross-analyst agreement gets base confidence bonus
                convergences.append((f"Shared domain: {domain}", len(set(analysts)), conf))

        # Strategy 3: Threat level alignment
        threat_levels = {}
        for name, findings in all_findings.items():
            max_threat = "NONE"
            for f in findings:
                for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"):
                    if f.threat_level == level and level > max_threat:
                        max_threat = level
            threat_levels[name] = max_threat

        high_analysts = [n for n, t in threat_levels.items() if t in ("HIGH", "CRITICAL")]
        if len(high_analysts) >= 2:
            topic = f"Aligned high-threat assessment ({len(high_analysts)}/3 analysts)"
            convergences.append((topic, len(high_analysts), 0.8))

        return convergences

    # ── fusion report builder ─────────────────────────────────────────

    def _build_fusion(
        self,
        convergence: Tuple[str, int, float],
        all_findings: Dict[str, List[AnalystFinding]],
        now: str,
    ) -> FusionReport:
        """Build a fusion report from a detected convergence."""
        topic, count, confidence = convergence

        # Determine overall threat level
        all_f = [f for findings in all_findings.values() for f in findings]
        threat_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for f in all_f:
            if f.threat_level in threat_counts:
                threat_counts[f.threat_level] += 1

        if threat_counts["CRITICAL"] > 0:
            threat = "CRITICAL"
        elif threat_counts["HIGH"] >= 2:
            threat = "HIGH"
        elif threat_counts["MEDIUM"] >= 2:
            threat = "MEDIUM"
        else:
            threat = "LOW"

        # Collect top findings
        top_findings = sorted(all_f, key=lambda f: f.confidence, reverse=True)[:6]

        # Generate recommended actions
        actions = self._recommend_actions(threat, top_findings)

        # Build synthesis from findings
        synthesis_parts = []
        for name in ("SIGINT", "GEOINT", "HUMINT"):
            findings = all_findings.get(name, [])
            if findings:
                best = max(findings, key=lambda f: f.confidence)
                synthesis_parts.append(f"{name}: {best.assessment}")
        synthesis = " | ".join(synthesis_parts)

        rid = f"FUSION-{int(time.time())}"

        return FusionReport(
            report_id=rid,
            timestamp=now,
            convergence_count=count,
            converged_topic=topic,
            threat_level=threat,
            confidence=confidence,
            findings=top_findings,
            synthesis=synthesis[:500],
            recommended_actions=actions,
        )

    def _recommend_actions(self, threat: str,
                           findings: List[AnalystFinding]) -> List[str]:
        """Generate recommended actions based on threat level."""
        actions = []

        domains_involved = {f.domain for f in findings}

        if threat in ("CRITICAL", "HIGH"):
            actions.append("Escalate to duty officer immediately")
            actions.append("Increase collection tempo on affected region")

        if "gps_jamming" in domains_involved:
            actions.append("Task SIGINT collection on affected frequencies")
        if "military" in domains_involved:
            actions.append("Cross-reference with known exercise schedules")
        if "cyber_events" in domains_involved or "ransomware" in domains_involved:
            actions.append("Alert cyber defense partners")
        if "fires" in domains_involved:
            actions.append("Check for correlation with military activity")
        if "ships" in domains_involved:
            actions.append("Monitor AIS gaps for potential dark vessels")

        if threat == "MEDIUM":
            actions.append("Continue monitoring, reassess in 1 hour")
        elif threat == "LOW":
            actions.append("Log for pattern analysis")

        return actions[:5]

    # ── LLM synthesis ─────────────────────────────────────────────────

    async def _llm_synthesize(self, llm, report: FusionReport):
        """Use LLM to produce a narrative synthesis."""
        findings_text = "\n".join(
            f"  [{f.analyst}] {f.assessment} (conf: {f.confidence:.0%})"
            for f in report.findings
        )
        prompt = (
            "You are a senior all-source intelligence analyst producing a MULTI-INT "
            "fusion assessment. Three virtual analysts (SIGINT, GEOINT, HUMINT) have "
            "independently analyzed the same real-time data and converged.\n\n"
            f"Topic: {report.converged_topic}\n"
            f"Threat Level: {report.threat_level}\n"
            f"Convergence: {report.convergence_count}/3 analysts agree\n\n"
            f"Individual findings:\n{findings_text}\n\n"
            "Write a 3-4 sentence professional intelligence synthesis that:\n"
            "1. Summarizes what the converging data means\n"
            "2. Identifies the most likely scenario\n"
            "3. Notes critical uncertainties\n"
            "Be precise and professional. No hedging language."
        )

        raw = await llm.chat_raw(prompt, max_tokens=250)
        if raw and raw.strip():
            report.synthesis = raw.strip()[:500]

    # ── queries ───────────────────────────────────────────────────────

    def get_latest_report(self) -> Optional[FusionReport]:
        return self.fusion_reports[-1] if self.fusion_reports else None

    def get_analyst_status(self) -> Dict[str, Any]:
        status = {}
        for name, analyst in self.analysts.items():
            status[name] = {
                "role": analyst.profile.role,
                "recent_findings": len(analyst.recent_findings),
                "total_findings": len(analyst.finding_history),
                "domains": analyst.profile.domains,
            }
        return status

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "total_reports": self.total_reports,
                "reports": [asdict(r) for r in self.fusion_reports[-20:]],
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save fusion state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_reports = state.get("total_reports", 0)
        except Exception as e:
            logger.warning("Failed to load fusion state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_reports": self.total_reports,
            "analysts": self.get_analyst_status(),
            "latest_threat": self.fusion_reports[-1].threat_level if self.fusion_reports else None,
        }
