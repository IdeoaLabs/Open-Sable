"""
#23 — Infrastructure Resilience Monitor

DHS / CISA / NSA-CSD capability.  Monitors critical infrastructure
status by correlating internet outages, ransomware campaigns, and
cyber threat indicators to predict cascading failures and attribute
attacks.

Data sources: snapshot.internet_outages, snapshot.ransomware,
              snapshot.cyber_threats
"""

from __future__ import annotations

import hashlib
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
# Critical infrastructure sectors (CISA-defined)
# ---------------------------------------------------------------------------

CRITICAL_SECTORS = {
    "energy", "water", "transportation", "communications", "healthcare",
    "financial", "government", "defense", "food", "chemical",
    "nuclear", "manufacturing", "dams", "emergency_services",
    "information_technology",
}

# High-value target keywords
HVT_KEYWORDS = [
    "power grid", "electrical grid", "water treatment", "hospital",
    "banking", "stock exchange", "telecom", "airport", "port",
    "railway", "pipeline", "refinery", "dam", "nuclear",
    "military", "government", "election", "voting",
]

# Known APT groups associated with infrastructure attacks
APT_INFRA_GROUPS = {
    "sandworm": {"country": "Russia", "targets": ["energy", "government"]},
    "volt typhoon": {"country": "China", "targets": ["communications", "energy", "water"]},
    "lazarus": {"country": "North Korea", "targets": ["financial", "defense"]},
    "apt33": {"country": "Iran", "targets": ["energy", "defense"]},
    "apt41": {"country": "China", "targets": ["healthcare", "telecom"]},
    "lockbit": {"country": "multi", "targets": ["healthcare", "manufacturing"]},
    "cl0p": {"country": "Russia", "targets": ["financial", "government"]},
    "blackcat": {"country": "Russia", "targets": ["energy", "healthcare"]},
    "akira": {"country": "unknown", "targets": ["manufacturing", "healthcare"]},
}


@dataclass
class InfraAlert:
    """Infrastructure resilience alert."""
    alert_id: str
    alert_type: str   # outage_cascade, ransomware_campaign, apt_activity,
                       # sector_degraded, multi_vector, supply_chain
    severity: str
    timestamp: str
    title: str
    description: str
    sector: str = ""
    country: str = ""
    affected_systems: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectorStatus:
    """Health status of a critical infrastructure sector."""
    sector: str
    status: str = "green"     # green, yellow, orange, red
    outages: int = 0
    cyber_incidents: int = 0
    ransomware_hits: int = 0
    last_incident: str = ""
    threat_level: float = 0.0  # 0-10


@dataclass
class RansomwareCampaign:
    """Tracked ransomware campaign."""
    campaign_id: str
    group: str
    first_seen: str
    last_seen: str
    victim_count: int = 0
    sectors_hit: List[str] = field(default_factory=list)
    countries_hit: List[str] = field(default_factory=list)
    escalating: bool = False


class InfrastructureResilience:
    """
    Critical infrastructure resilience monitoring engine.

    Correlates internet outages, ransomware campaigns, and cyber threats
    to detect coordinated attacks, predict cascading failures, and
    maintain sector health dashboards.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.alerts: List[InfraAlert] = []
        self.sectors: Dict[str, SectorStatus] = {
            s: SectorStatus(sector=s) for s in CRITICAL_SECTORS
        }
        self.campaigns: Dict[str, RansomwareCampaign] = {}
        self._prev_outage_count = 0
        self.total_detections = 0

    def observe(self, snapshot: IntelSnapshot) -> List[InfraAlert]:
        """Analyze snapshot for infrastructure threat indicators."""
        now_str = datetime.now(timezone.utc).isoformat()
        new_alerts: List[InfraAlert] = []

        # 1. Internet outage analysis
        new_alerts.extend(self._analyze_outages(snapshot, now_str))

        # 2. Ransomware campaign tracking
        new_alerts.extend(self._analyze_ransomware(snapshot, now_str))

        # 3. Cyber threat / APT correlation
        new_alerts.extend(self._analyze_cyber_threats(snapshot, now_str))

        # 4. Multi-vector correlation
        new_alerts.extend(self._detect_coordinated(now_str))

        self.total_detections += len(new_alerts)
        self.alerts.extend(new_alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-250:]

        self._prev_outage_count = len(snapshot.internet_outages)

        return new_alerts

    # ── outage analysis ───────────────────────────────────────────────

    def _analyze_outages(self, snapshot: IntelSnapshot,
                         now_str: str) -> List[InfraAlert]:
        alerts: List[InfraAlert] = []
        country_outages: Dict[str, int] = {}

        for outage in snapshot.internet_outages:
            country = outage.get("country", outage.get("location", ""))
            provider = outage.get("provider", outage.get("asn_name", ""))
            severity = outage.get("severity", "")
            description = outage.get("description", outage.get("details", ""))

            if country:
                country_outages[country] = country_outages.get(country, 0) + 1

            # Classify sector impact
            full_text = f"{provider} {description}".lower()
            for sector in CRITICAL_SECTORS:
                if sector in full_text or any(kw in full_text for kw in HVT_KEYWORDS
                                               if sector in kw):
                    self.sectors[sector].outages += 1
                    self.sectors[sector].last_incident = now_str

        # Cascade detection: multiple outages in same country
        for country, count in country_outages.items():
            if count >= 3:
                alerts.append(InfraAlert(
                    alert_id=hashlib.md5(f"cascade_{country}_{count}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="outage_cascade",
                    severity="critical" if count >= 5 else "high",
                    timestamp=now_str,
                    title=f"Internet outage cascade: {country} ({count} outages)",
                    description=(f"{count} concurrent internet outages in {country}. "
                                f"Possible coordinated disruption or infrastructure failure."),
                    country=country,
                    evidence={"outage_count": count},
                ))

        # Surge detection vs previous cycle
        current = len(snapshot.internet_outages)
        if self._prev_outage_count > 0 and current > self._prev_outage_count * 2:
            alerts.append(InfraAlert(
                alert_id=hashlib.md5(f"surge_{current}_{time.time()}".encode()).hexdigest()[:10],
                alert_type="outage_cascade",
                severity="high",
                timestamp=now_str,
                title=f"Global outage surge: {self._prev_outage_count} → {current}",
                description=f"Internet outages doubled ({self._prev_outage_count} → {current}). Global disruption event possible.",
                evidence={"previous": self._prev_outage_count, "current": current},
            ))

        return alerts

    # ── ransomware ────────────────────────────────────────────────────

    def _analyze_ransomware(self, snapshot: IntelSnapshot,
                            now_str: str) -> List[InfraAlert]:
        alerts: List[InfraAlert] = []

        for rw in snapshot.ransomware:
            group = (rw.get("group") or rw.get("gang_name") or
                     rw.get("threat_actor") or "unknown").lower()
            victim = rw.get("victim", rw.get("organization", ""))
            country = rw.get("country", "")
            sector = rw.get("sector", rw.get("industry", "")).lower()
            published = rw.get("published", rw.get("date", ""))

            cid = f"campaign_{group}"
            campaign = self.campaigns.get(cid)

            if campaign:
                campaign.victim_count += 1
                campaign.last_seen = now_str
                if sector and sector not in campaign.sectors_hit:
                    campaign.sectors_hit.append(sector)
                if country and country not in campaign.countries_hit:
                    campaign.countries_hit.append(country)
                # Escalating = hitting new sectors
                campaign.escalating = len(campaign.sectors_hit) > 2
            else:
                campaign = RansomwareCampaign(
                    campaign_id=cid,
                    group=group,
                    first_seen=now_str,
                    last_seen=now_str,
                    victim_count=1,
                    sectors_hit=[sector] if sector else [],
                    countries_hit=[country] if country else [],
                )
                self.campaigns[cid] = campaign

            # Update sector status
            for s in CRITICAL_SECTORS:
                if s in sector or s in (victim or "").lower():
                    self.sectors[s].ransomware_hits += 1
                    self.sectors[s].last_incident = now_str
                    self.sectors[s].threat_level = min(10,
                        self.sectors[s].threat_level + 1.0)

            # Check for APT group
            apt_info = APT_INFRA_GROUPS.get(group)
            if apt_info:
                alerts.append(InfraAlert(
                    alert_id=hashlib.md5(f"apt_rw_{group}_{victim}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="apt_activity",
                    severity="critical",
                    timestamp=now_str,
                    title=f"Known APT ransomware: {group.upper()} hit {victim or 'unknown'}",
                    description=(f"APT group {group} (attributed: {apt_info['country']}) "
                                f"hit {victim} in {sector}. "
                                f"Known targets: {', '.join(apt_info['targets'])}"),
                    sector=sector,
                    country=country,
                    evidence={"group": group, "attribution": apt_info, "victim": victim},
                ))
            elif campaign.victim_count >= 3:
                alerts.append(InfraAlert(
                    alert_id=hashlib.md5(f"rw_{group}_{campaign.victim_count}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="ransomware_campaign",
                    severity="high" if campaign.escalating else "medium",
                    timestamp=now_str,
                    title=f"Ransomware campaign: {group} ({campaign.victim_count} victims)",
                    description=(f"Active campaign by {group}: {campaign.victim_count} victims "
                                f"across {', '.join(campaign.sectors_hit) or 'unknown sectors'}. "
                                f"{'ESCALATING - new sectors targeted.' if campaign.escalating else ''}"),
                    sector=sector,
                    evidence={"group": group, "victim_count": campaign.victim_count,
                              "sectors": campaign.sectors_hit},
                ))

        return alerts

    # ── cyber threats / APT ───────────────────────────────────────────

    def _analyze_cyber_threats(self, snapshot: IntelSnapshot,
                               now_str: str) -> List[InfraAlert]:
        alerts: List[InfraAlert] = []

        for threat in snapshot.cyber_threats:
            name = (threat.get("name") or threat.get("title") or
                    threat.get("threat_name") or "")
            cve = threat.get("cve", threat.get("cve_id", ""))
            severity = threat.get("severity", threat.get("cvss", ""))
            affected = threat.get("affected", threat.get("products", ""))
            actor = (threat.get("actor") or threat.get("threat_actor") or "").lower()

            full_text = f"{name} {affected}".lower()

            # Critical infrastructure targeting
            sectors_affected = [s for s in CRITICAL_SECTORS
                               if s in full_text or
                               any(kw in full_text for kw in HVT_KEYWORDS)]

            for sector in sectors_affected:
                self.sectors[sector].cyber_incidents += 1
                self.sectors[sector].last_incident = now_str
                self.sectors[sector].threat_level = min(10,
                    self.sectors[sector].threat_level + 0.5)

            # Check APT attribution
            apt_match = None
            for apt_name in APT_INFRA_GROUPS:
                if apt_name in actor or apt_name in full_text:
                    apt_match = apt_name
                    break

            if apt_match and sectors_affected:
                apt_info = APT_INFRA_GROUPS[apt_match]
                alerts.append(InfraAlert(
                    alert_id=hashlib.md5(f"apt_{apt_match}_{name}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="apt_activity",
                    severity="critical",
                    timestamp=now_str,
                    title=f"APT targeting infrastructure: {apt_match.upper()}",
                    description=(f"APT {apt_match} ({apt_info['country']}) targeting "
                                f"{', '.join(sectors_affected)}. {name}. CVE: {cve or 'N/A'}"),
                    sector=sectors_affected[0],
                    affected_systems=[affected] if affected else [],
                    evidence={"apt": apt_match, "cve": cve, "severity": severity},
                ))

            # High-severity CVEs targeting critical infra
            try:
                sev_val = float(severity)
            except (ValueError, TypeError):
                sev_val = 0

            if sev_val >= 9.0 and sectors_affected:
                alerts.append(InfraAlert(
                    alert_id=hashlib.md5(f"cve_{cve}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="supply_chain",
                    severity="critical",
                    timestamp=now_str,
                    title=f"Critical CVE affecting infrastructure: {cve or name}",
                    description=(f"CVSS {sev_val}: {name}. Affects: {affected}. "
                                f"Sectors at risk: {', '.join(sectors_affected)}"),
                    affected_systems=[affected] if affected else [],
                    evidence={"cvss": sev_val, "cve": cve, "sectors": sectors_affected},
                ))

        return alerts

    # ── multi-vector correlation ──────────────────────────────────────

    def _detect_coordinated(self, now_str: str) -> List[InfraAlert]:
        """Detect coordinated multi-vector attacks on same sector."""
        alerts: List[InfraAlert] = []

        for sector, status in self.sectors.items():
            vectors = 0
            if status.outages > 0:
                vectors += 1
            if status.ransomware_hits > 0:
                vectors += 1
            if status.cyber_incidents > 0:
                vectors += 1

            # Update status color
            if vectors >= 3:
                status.status = "red"
            elif vectors >= 2:
                status.status = "orange"
            elif vectors >= 1:
                status.status = "yellow"

            if vectors >= 2:
                alerts.append(InfraAlert(
                    alert_id=hashlib.md5(f"multi_{sector}_{vectors}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="multi_vector",
                    severity="critical" if vectors >= 3 else "high",
                    timestamp=now_str,
                    title=f"Multi-vector attack on {sector}: {vectors} vectors",
                    description=(f"Sector {sector} under {vectors}-vector pressure: "
                                f"outages={status.outages}, ransomware={status.ransomware_hits}, "
                                f"cyber={status.cyber_incidents}. Coordinated attack likely."),
                    sector=sector,
                    evidence={"vectors": vectors, "outages": status.outages,
                              "ransomware": status.ransomware_hits,
                              "cyber": status.cyber_incidents},
                ))

        return alerts

    # ── queries ───────────────────────────────────────────────────────

    def get_sector_dashboard(self) -> Dict[str, SectorStatus]:
        return self.sectors

    def get_degraded_sectors(self) -> List[SectorStatus]:
        return [s for s in self.sectors.values() if s.status != "green"]

    def get_active_campaigns(self) -> List[RansomwareCampaign]:
        return sorted(self.campaigns.values(), key=lambda c: c.victim_count, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_detections": self.total_detections,
            "sectors_monitored": len(self.sectors),
            "sectors_degraded": sum(1 for s in self.sectors.values() if s.status != "green"),
            "sectors_red": sum(1 for s in self.sectors.values() if s.status == "red"),
            "active_campaigns": len(self.campaigns),
            "total_alerts": len(self.alerts),
        }
