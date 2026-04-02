"""
#26 — Threat Fusion Dashboard

NCTC / DIA / Joint Intel Center capability.  Aggregates outputs from ALL
intelligence modules into a unified threat assessment with DEFCON-style
escalation levels, early warning indicators, and priority intelligence
requirements (PIRs).

This module consumes the outputs from every other module to produce a
single fused picture.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threat level definitions (inspired by DEFCON but for OSINT)
# ---------------------------------------------------------------------------

THREAT_LEVELS = {
    1: {"name": "MAXIMUM", "color": "red",    "description": "Active global crisis. Multiple domains at critical."},
    2: {"name": "SEVERE",  "color": "orange", "description": "Major escalation underway. Multiple high-severity alerts."},
    3: {"name": "ELEVATED","color": "yellow", "description": "Elevated threat across key domains."},
    4: {"name": "GUARDED", "color": "blue",   "description": "General awareness. Isolated alerts."},
    5: {"name": "NORMAL",  "color": "green",  "description": "Baseline operations. No significant threats."},
}

# Domain weights for fusion scoring
DOMAIN_WEIGHTS = {
    "sigint_ew": 1.2,
    "finint": 1.0,
    "nuclear": 2.0,        # Nuclear always highest weight
    "space": 0.8,
    "infrastructure": 1.3,
    "environmental": 0.7,
    "force_projection": 1.5,
    "causal": 0.9,
    "swarm": 1.1,
    "predictive": 1.0,
    "pattern_of_life": 0.8,
    "geofence": 1.2,
    "kill_chain": 1.5,
    "counter_surveillance": 0.9,
    "narrative_warfare": 0.8,
    "knowledge_graph": 0.5,
    "temporal": 0.6,
    "cognitive": 0.6,
    "multi_agent": 0.7,
}

SEVERITY_SCORES = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}


@dataclass
class DomainAssessment:
    """Assessment of a single intelligence domain."""
    domain: str
    threat_score: float = 0.0      # 0-10
    alert_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    top_alerts: List[str] = field(default_factory=list)  # Alert titles
    status: str = "green"          # green, yellow, orange, red
    trend: str = "stable"          # improving, stable, degrading, escalating


@dataclass
class FusionAlert:
    """Fused cross-domain alert."""
    alert_id: str
    timestamp: str
    title: str
    description: str
    contributing_domains: List[str]
    severity: str
    fusion_score: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatPicture:
    """Complete fused threat picture."""
    timestamp: str
    threat_level: int              # 1-5 (1=MAXIMUM)
    threat_name: str
    threat_color: str
    overall_score: float           # 0-100
    domain_assessments: Dict[str, DomainAssessment] = field(default_factory=dict)
    fusion_alerts: List[FusionAlert] = field(default_factory=list)
    priority_intel_requirements: List[str] = field(default_factory=list)
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    escalation_indicators: List[str] = field(default_factory=list)


class ThreatFusionDashboard:
    """
    Unified threat fusion engine.

    Aggregates outputs from all intelligence modules into a single
    threat picture with DEFCON-style levels and prioritized indicators.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.current_picture: Optional[ThreatPicture] = None
        self.history: List[ThreatPicture] = []
        self.total_fusions = 0

    def fuse(self, module_outputs: Dict[str, Any]) -> ThreatPicture:
        """
        Produce a unified threat picture from all module outputs.

        Args:
            module_outputs: Dict mapping module names to their output.
                Each value should be a list of alert-like objects or
                dicts with at minimum 'severity' and 'title' fields,
                OR a dict with a 'stats' key.

        Returns:
            ThreatPicture with unified assessment.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        assessments: Dict[str, DomainAssessment] = {}
        all_fusion_alerts: List[FusionAlert] = []
        total_score = 0.0
        total_weight = 0.0

        for domain, output in module_outputs.items():
            assessment = self._assess_domain(domain, output)
            assessments[domain] = assessment

            weight = DOMAIN_WEIGHTS.get(domain, 1.0)
            total_score += assessment.threat_score * weight
            total_weight += weight

        # Normalize to 0-100
        overall = (total_score / total_weight * 10) if total_weight > 0 else 0
        overall = min(100, overall)

        # Determine threat level
        if overall >= 70:
            level = 1
        elif overall >= 50:
            level = 2
        elif overall >= 30:
            level = 3
        elif overall >= 15:
            level = 4
        else:
            level = 5

        level_info = THREAT_LEVELS[level]

        # Generate cross-domain fusion alerts
        all_fusion_alerts = self._cross_correlate(assessments, now_str)

        # Generate priority intelligence requirements
        pirs = self._generate_pirs(assessments)

        # Identify hotspots
        hotspots = self._identify_hotspots(module_outputs)

        # Escalation indicators
        escalation = self._check_escalation(assessments)

        picture = ThreatPicture(
            timestamp=now_str,
            threat_level=level,
            threat_name=level_info["name"],
            threat_color=level_info["color"],
            overall_score=overall,
            domain_assessments=assessments,
            fusion_alerts=all_fusion_alerts,
            priority_intel_requirements=pirs,
            hotspots=hotspots,
            escalation_indicators=escalation,
        )

        self.current_picture = picture
        self.history.append(picture)
        if len(self.history) > 100:
            self.history = self.history[-50:]

        self.total_fusions += 1
        return picture

    # ── domain assessment ─────────────────────────────────────────────

    def _assess_domain(self, domain: str,
                       output: Any) -> DomainAssessment:
        """Score a single domain from its output."""
        assessment = DomainAssessment(domain=domain)

        alerts = []
        if isinstance(output, list):
            alerts = output
        elif isinstance(output, dict):
            alerts = output.get("alerts", output.get("results", []))
            if not isinstance(alerts, list):
                alerts = []

        assessment.alert_count = len(alerts)
        score = 0.0

        for alert in alerts:
            if isinstance(alert, dict):
                sev = alert.get("severity", "low")
                title = alert.get("title", "")
            elif hasattr(alert, "severity"):
                sev = alert.severity
                title = getattr(alert, "title", "")
            else:
                continue

            sev_lower = sev.lower() if isinstance(sev, str) else "low"
            sev_score = SEVERITY_SCORES.get(sev_lower, 1.0)
            score += sev_score

            if sev_lower == "critical":
                assessment.critical_count += 1
            elif sev_lower == "high":
                assessment.high_count += 1

            if len(assessment.top_alerts) < 5:
                assessment.top_alerts.append(str(title)[:120])

        # Normalize score (0-10)
        if alerts:
            assessment.threat_score = min(10.0, score / max(len(alerts), 1) * 2)
        else:
            assessment.threat_score = 0.0

        # Boost for critical alerts
        if assessment.critical_count > 0:
            assessment.threat_score = min(10.0,
                assessment.threat_score + assessment.critical_count * 1.5)

        # Status
        if assessment.threat_score >= 7:
            assessment.status = "red"
        elif assessment.threat_score >= 4:
            assessment.status = "orange"
        elif assessment.threat_score >= 2:
            assessment.status = "yellow"

        return assessment

    # ── cross-domain correlation ──────────────────────────────────────

    def _cross_correlate(self, assessments: Dict[str, DomainAssessment],
                         now_str: str) -> List[FusionAlert]:
        """Find patterns across domains."""
        alerts: List[FusionAlert] = []
        red_domains = [d for d, a in assessments.items() if a.status == "red"]

        if len(red_domains) >= 2:
            alerts.append(FusionAlert(
                alert_id=hashlib.md5(f"multi_red_{len(red_domains)}_{time.time()}".encode()).hexdigest()[:10],
                timestamp=now_str,
                title=f"MULTI-DOMAIN CRISIS: {len(red_domains)} domains critical",
                description=(f"Domains at critical: {', '.join(red_domains)}. "
                            f"Correlated multi-domain escalation detected."),
                contributing_domains=red_domains,
                severity="critical",
                fusion_score=min(100, sum(assessments[d].threat_score
                                         for d in red_domains) * 5),
            ))

        # Nuclear + military correlation
        nuc = assessments.get("nuclear")
        force = assessments.get("force_projection")
        if nuc and force and nuc.threat_score > 3 and force.threat_score > 3:
            alerts.append(FusionAlert(
                alert_id=hashlib.md5(f"nuc_force_{time.time()}".encode()).hexdigest()[:10],
                timestamp=now_str,
                title="NUCLEAR-MILITARY CORRELATION",
                description="Simultaneous nuclear and force projection alerts. Strategic escalation risk.",
                contributing_domains=["nuclear", "force_projection"],
                severity="critical",
                fusion_score=(nuc.threat_score + force.threat_score) * 5,
            ))

        # Cyber + Infrastructure + SIGINT correlation (hybrid warfare)
        infra = assessments.get("infrastructure")
        sigint = assessments.get("sigint_ew")
        if infra and sigint and infra.threat_score > 3 and sigint.threat_score > 3:
            alerts.append(FusionAlert(
                alert_id=hashlib.md5(f"hybrid_{time.time()}".encode()).hexdigest()[:10],
                timestamp=now_str,
                title="HYBRID WARFARE INDICATORS",
                description="Concurrent SIGINT/EW and infrastructure disruption. Hybrid warfare pattern.",
                contributing_domains=["infrastructure", "sigint_ew"],
                severity="critical",
                fusion_score=(infra.threat_score + sigint.threat_score) * 5,
            ))

        # FININT + Narrative (economic warfare + information warfare)
        finint = assessments.get("finint")
        narrative = assessments.get("narrative_warfare")
        if finint and narrative and finint.threat_score > 3 and narrative.threat_score > 3:
            alerts.append(FusionAlert(
                alert_id=hashlib.md5(f"econ_info_{time.time()}".encode()).hexdigest()[:10],
                timestamp=now_str,
                title="ECONOMIC-INFORMATION WARFARE",
                description="Coordinated financial and narrative attacks detected. Gray zone operation.",
                contributing_domains=["finint", "narrative_warfare"],
                severity="high",
                fusion_score=(finint.threat_score + narrative.threat_score) * 4,
            ))

        return alerts

    # ── PIR generation ────────────────────────────────────────────────

    def _generate_pirs(self,
                       assessments: Dict[str, DomainAssessment]) -> List[str]:
        """Generate Priority Intelligence Requirements."""
        pirs: List[str] = []

        # Sort by threat score
        sorted_domains = sorted(assessments.items(),
                               key=lambda x: x[1].threat_score, reverse=True)

        for domain, assessment in sorted_domains[:5]:
            if assessment.threat_score > 2:
                pirs.append(
                    f"[{domain.upper()}] Score {assessment.threat_score:.1f}/10 — "
                    f"{assessment.critical_count} critical, {assessment.high_count} high. "
                    f"Top: {assessment.top_alerts[0] if assessment.top_alerts else 'N/A'}"
                )

        if not pirs:
            pirs.append("No elevated intelligence requirements at this time.")

        return pirs

    # ── hotspot identification ────────────────────────────────────────

    def _identify_hotspots(self,
                           module_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify geographic hotspots across all modules."""
        region_mentions: Dict[str, int] = {}

        for domain, output in module_outputs.items():
            items = output if isinstance(output, list) else (
                output.get("alerts", []) if isinstance(output, dict) else []
            )
            for item in items:
                if isinstance(item, dict):
                    region = (item.get("region") or item.get("area") or
                             item.get("country") or "")
                elif hasattr(item, "region"):
                    region = getattr(item, "region", "") or getattr(item, "area", "") or ""
                else:
                    continue

                if region:
                    region_mentions[region] = region_mentions.get(region, 0) + 1

        hotspots = [
            {"region": r, "mention_count": c, "priority": "high" if c >= 5 else "medium"}
            for r, c in sorted(region_mentions.items(),
                              key=lambda x: x[1], reverse=True)[:10]
        ]
        return hotspots

    # ── escalation check ──────────────────────────────────────────────

    def _check_escalation(self,
                          assessments: Dict[str, DomainAssessment]) -> List[str]:
        """Check for escalation indicators vs previous picture."""
        indicators: List[str] = []

        if self.current_picture:
            prev = self.current_picture
            for domain, curr_assess in assessments.items():
                prev_assess = prev.domain_assessments.get(domain)
                if prev_assess:
                    delta = curr_assess.threat_score - prev_assess.threat_score
                    if delta > 2:
                        indicators.append(
                            f"{domain}: threat score surged {prev_assess.threat_score:.1f} → "
                            f"{curr_assess.threat_score:.1f} (+{delta:.1f})"
                        )
                        curr_assess.trend = "escalating"
                    elif delta > 0.5:
                        curr_assess.trend = "degrading"
                    elif delta < -0.5:
                        curr_assess.trend = "improving"

        return indicators

    # ── queries ───────────────────────────────────────────────────────

    def get_current_picture(self) -> Optional[ThreatPicture]:
        return self.current_picture

    def get_threat_level(self) -> int:
        return self.current_picture.threat_level if self.current_picture else 5

    def get_domain_status(self) -> Dict[str, str]:
        if not self.current_picture:
            return {}
        return {d: a.status for d, a in self.current_picture.domain_assessments.items()}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_fusions": self.total_fusions,
            "current_threat_level": self.get_threat_level(),
            "current_threat_name": (self.current_picture.threat_name
                                    if self.current_picture else "NORMAL"),
            "current_score": (self.current_picture.overall_score
                             if self.current_picture else 0),
            "history_length": len(self.history),
            "domains_assessed": (len(self.current_picture.domain_assessments)
                                if self.current_picture else 0),
        }
