"""
#2,  Cross-Domain Causal Correlation Engine

Ingests periodic Zunvra snapshots and uses LLM-assisted causal extraction
to discover cause→effect chains that span multiple intelligence domains.

Example output:
  GPS jamming Eastern Med → military flights increase → AIS gaps near Syria
  → cyber C2 spike in same region → THREAT CHAIN (confidence 0.78)

Persists a causal graph to disk so patterns accumulate over time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CausalLink:
    cause_domain: str
    cause_description: str
    effect_domain: str
    effect_description: str
    strength: float = 0.5       # 0-1
    observations: int = 1
    first_seen: str = ""
    last_seen: str = ""

    def key(self) -> str:
        return hashlib.sha256(
            f"{self.cause_domain}:{self.cause_description}|{self.effect_domain}:{self.effect_description}".encode()
        ).hexdigest()[:16]


@dataclass
class ThreatChain:
    """Multi-hop causal chain across domains."""
    chain_id: str
    links: List[CausalLink]
    domains_involved: List[str]
    total_strength: float
    timestamp: str
    summary: str = ""
    confidence: float = 0.0

    def to_text(self) -> str:
        parts = []
        for link in self.links:
            parts.append(f"[{link.cause_domain}] {link.cause_description}")
        if self.links:
            last = self.links[-1]
            parts.append(f"[{last.effect_domain}] {last.effect_description}")
        chain_str = " → ".join(parts)
        return (
            f"THREAT CHAIN {self.chain_id} (confidence {self.confidence:.0%})\n"
            f"  {chain_str}\n"
            f"  Domains: {', '.join(self.domains_involved)}\n"
            f"  {self.summary}"
        )


# ---------------------------------------------------------------------------
# Rule-based cross-domain patterns (fire without LLM)
# ---------------------------------------------------------------------------

STATIC_PATTERNS: List[Dict[str, Any]] = [
    {
        "name": "ewarf",
        "cause": ("gps_jamming", "GPS jamming zones detected"),
        "effect": ("military_flights", "Increased military air activity"),
        "condition": lambda snap: len(snap.gps_jamming) > 0 and len(snap.military_flights) > 5,
        "base_strength": 0.7,
    },
    {
        "name": "naval_posture",
        "cause": ("carriers", "Carrier strike group repositioning"),
        "effect": ("military_flights", "Elevated military flight density"),
        "condition": lambda snap: len(snap.carriers) > 0 and len(snap.military_flights) > 10,
        "base_strength": 0.65,
    },
    {
        "name": "cyber_sigint",
        "cause": ("gps_jamming", "Electronic warfare / GPS disruption"),
        "effect": ("cyber_threats", "Coordinated cyber C2 activity"),
        "condition": lambda snap: len(snap.gps_jamming) > 0 and len(snap.cyber_threats) > 3,
        "base_strength": 0.6,
    },
    {
        "name": "conflict_escalation",
        "cause": ("gdelt_events", "Rising GDELT conflict event density"),
        "effect": ("military_flights", "Military air response"),
        "condition": lambda snap: len(snap.gdelt_events) > 50 and len(snap.military_flights) > 5,
        "base_strength": 0.55,
    },
    {
        "name": "infra_attack",
        "cause": ("cyber_threats", "Cyber infrastructure targeting"),
        "effect": ("internet_outages", "Regional internet degradation"),
        "condition": lambda snap: len(snap.cyber_threats) > 5 and len(snap.internet_outages) > 0,
        "base_strength": 0.6,
    },
    {
        "name": "seismic_nuclear",
        "cause": ("earthquakes", "Seismic event near nuclear facility"),
        "effect": ("nuclear_facilities", "Nuclear facility proximity alert"),
        "condition": lambda snap: _quakes_near_nukes(snap),
        "base_strength": 0.8,
    },
    {
        "name": "maritime_conflict",
        "cause": ("gdelt_events", "Maritime conflict incidents reported"),
        "effect": ("ships", "Vessel route anomalies in conflict zone"),
        "condition": lambda snap: len(snap.gdelt_events) > 20 and len(snap.ships) > 100,
        "base_strength": 0.5,
    },
    {
        "name": "ransomware_infra",
        "cause": ("ransomware", "Ransomware campaign active"),
        "effect": ("internet_outages", "Downstream infrastructure disruption"),
        "condition": lambda snap: len(snap.ransomware) > 3 and len(snap.internet_outages) > 0,
        "base_strength": 0.55,
    },
]


def _quakes_near_nukes(snap: IntelSnapshot) -> bool:
    """Check if any earthquake is within 100km of a nuclear facility."""
    from .connector import _haversine
    for eq in snap.earthquakes:
        eq_lat = eq.get("lat") or eq.get("latitude") or 0
        eq_lon = eq.get("lon") or eq.get("longitude") or 0
        for nf in snap.nuclear_facilities:
            nf_lat = nf.get("lat") or nf.get("latitude") or 0
            nf_lon = nf.get("lon") or nf.get("longitude") or 0
            try:
                if _haversine(float(eq_lat), float(eq_lon), float(nf_lat), float(nf_lon)) < 100:
                    return True
            except (ValueError, TypeError):
                continue
    return False


# ---------------------------------------------------------------------------
# Causal Correlation Engine
# ---------------------------------------------------------------------------

class CausalCorrelationEngine:
    """
    Discovers and persists cross-domain causal links from Zunvra snapshots.
    Operates in two modes:
      1. Rule-based (always available, no LLM needed)
      2. LLM-assisted (richer extraction when an LLM is supplied)
    """

    MAX_LINKS = 500
    MAX_CHAINS = 200

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "causal_graph.json"

        self.links: List[CausalLink] = []
        self.chains: List[ThreatChain] = []
        self.total_analyses = 0
        self._load_state()

    # ── core analysis ─────────────────────────────────────────────────

    async def analyze(
        self,
        snapshot: IntelSnapshot,
        llm=None,
    ) -> List[ThreatChain]:
        """
        Run causal correlation on a snapshot.
        Returns newly discovered threat chains.
        """
        new_chains: List[ThreatChain] = []
        now = datetime.now(timezone.utc).isoformat()

        # Phase 1: Static rule-based detection
        triggered_links: List[CausalLink] = []
        for pattern in STATIC_PATTERNS:
            try:
                if pattern["condition"](snapshot):
                    cause_domain, cause_desc = pattern["cause"]
                    effect_domain, effect_desc = pattern["effect"]
                    link = CausalLink(
                        cause_domain=cause_domain,
                        cause_description=cause_desc,
                        effect_domain=effect_domain,
                        effect_description=effect_desc,
                        strength=pattern["base_strength"],
                        first_seen=now,
                        last_seen=now,
                    )
                    self._upsert_link(link)
                    triggered_links.append(link)
            except Exception as e:
                logger.debug("Pattern %s failed: %s", pattern["name"], e)

        # Phase 2: LLM-assisted extraction
        if llm:
            try:
                llm_links = await self._llm_extract(llm, snapshot)
                for link in llm_links:
                    self._upsert_link(link)
                    triggered_links.append(link)
            except Exception as e:
                logger.warning("LLM causal extraction failed: %s", e)

        # Phase 3: Chain building,  find multi-hop sequences
        if len(triggered_links) >= 2:
            chain = self._build_chain(triggered_links, now)
            if chain:
                self.chains.append(chain)
                if len(self.chains) > self.MAX_CHAINS:
                    self.chains = self.chains[-self.MAX_CHAINS:]
                new_chains.append(chain)

        self.total_analyses += 1
        self._save_state()
        return new_chains

    # ── link management ───────────────────────────────────────────────

    def _upsert_link(self, link: CausalLink):
        """Update existing link or insert new one."""
        key = link.key()
        now = datetime.now(timezone.utc).isoformat()
        for existing in self.links:
            if existing.key() == key:
                existing.observations += 1
                existing.last_seen = now
                existing.strength = min(1.0, existing.strength + 0.05)
                return
        link.first_seen = now
        link.last_seen = now
        self.links.append(link)
        if len(self.links) > self.MAX_LINKS:
            self.links.sort(key=lambda l: l.observations, reverse=True)
            self.links = self.links[:self.MAX_LINKS]

    def _build_chain(self, links: List[CausalLink], timestamp: str) -> Optional[ThreatChain]:
        """Try to build a multi-hop chain from triggered links."""
        if len(links) < 2:
            return None

        # Simple greedy chain: connect links where effect_domain == cause_domain
        chain_links: List[CausalLink] = [links[0]]
        used = {0}
        for _ in range(len(links)):
            last_effect = chain_links[-1].effect_domain
            for i, link in enumerate(links):
                if i not in used and link.cause_domain == last_effect:
                    chain_links.append(link)
                    used.add(i)
                    break

        # Also just include all remaining as parallel triggers
        for i, link in enumerate(links):
            if i not in used:
                chain_links.append(link)

        domains = list(set(
            [l.cause_domain for l in chain_links] + [l.effect_domain for l in chain_links]
        ))
        total_strength = sum(l.strength for l in chain_links) / len(chain_links)
        chain_id = hashlib.sha256(f"{timestamp}_{len(self.chains)}".encode()).hexdigest()[:10]

        return ThreatChain(
            chain_id=chain_id,
            links=chain_links,
            domains_involved=domains,
            total_strength=total_strength,
            timestamp=timestamp,
            confidence=min(0.95, total_strength * (1 + 0.1 * (len(domains) - 2))),
        )

    # ── LLM extraction ───────────────────────────────────────────────

    async def _llm_extract(self, llm, snapshot: IntelSnapshot) -> List[CausalLink]:
        """Use LLM to discover non-obvious causal links from snapshot data."""
        prompt = (
            "You are a causal reasoning engine for the Zunvra OSINT intelligence platform.\n"
            "Given the current state of global intelligence feeds, extract cause→effect relationships.\n\n"
            f"Current state:\n{snapshot.summary_text(1500)}\n\n"
            "Output ONLY valid JSON,  an array of objects:\n"
            "[\n"
            '  {"cause_domain": "gps_jamming", "cause": "GPS gaps in Black Sea",\n'
            '   "effect_domain": "military_flights", "effect": "Russian military surge",\n'
            '   "strength": 0.7}\n'
            "]\n\n"
            "Rules:\n"
            "- Extract 2-6 cross-domain causal links\n"
            "- Domains: flights, military_flights, ships, satellites, earthquakes, fires, "
            "gdelt_events, cyber_threats, gps_jamming, carriers, conflicts, internet_outages, "
            "ransomware, nuclear_facilities\n"
            "- Focus on cross-domain links (different cause/effect domains)\n"
            "- strength 1.0 = certain, 0.1 = weak correlation\n"
            "- Be specific and evidence-based"
        )

        raw = await llm.chat_raw(prompt, max_tokens=500)
        import re
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if not m:
            return []

        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            return []

        links = []
        now = datetime.now(timezone.utc).isoformat()
        for item in items[:6]:
            links.append(CausalLink(
                cause_domain=item.get("cause_domain", "unknown"),
                cause_description=item.get("cause", ""),
                effect_domain=item.get("effect_domain", "unknown"),
                effect_description=item.get("effect", ""),
                strength=float(item.get("strength", 0.5)),
                first_seen=now,
                last_seen=now,
            ))
        return links

    # ── queries ───────────────────────────────────────────────────────

    def get_active_chains(self, min_confidence: float = 0.4) -> List[ThreatChain]:
        return [c for c in self.chains if c.confidence >= min_confidence]

    def get_links_for_domain(self, domain: str) -> List[CausalLink]:
        return [l for l in self.links if l.cause_domain == domain or l.effect_domain == domain]

    def get_strongest_links(self, n: int = 10) -> List[CausalLink]:
        return sorted(self.links, key=lambda l: l.strength * l.observations, reverse=True)[:n]

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "links": [asdict(l) for l in self.links],
                "chains": [
                    {
                        "chain_id": c.chain_id,
                        "links": [asdict(l) for l in c.links],
                        "domains_involved": c.domains_involved,
                        "total_strength": c.total_strength,
                        "timestamp": c.timestamp,
                        "summary": c.summary,
                        "confidence": c.confidence,
                    }
                    for c in self.chains[-self.MAX_CHAINS:]
                ],
                "total_analyses": self.total_analyses,
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save causal state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                for ld in state.get("links", []):
                    self.links.append(CausalLink(**{k: v for k, v in ld.items() if k in CausalLink.__dataclass_fields__}))
                self.total_analyses = state.get("total_analyses", 0)
                logger.info("Loaded %d causal links, %d analyses", len(self.links), self.total_analyses)
        except Exception as e:
            logger.warning("Failed to load causal state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_links": len(self.links),
            "total_chains": len(self.chains),
            "total_analyses": self.total_analyses,
            "strongest_link": asdict(self.get_strongest_links(1)[0]) if self.links else None,
        }
