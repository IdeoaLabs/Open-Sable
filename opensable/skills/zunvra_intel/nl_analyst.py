"""
#1 — Natural Language Intelligence Analyst

Chat-based interface that lets a user query the live Zunvra OSINT data
in plain English. Uses OpenSable's ReAct-style tool chaining to:
  - Parse the user's intent (location, entity type, time window)
  - Query the relevant Zunvra endpoints
  - Cross-reference across domains
  - Return a structured intelligence answer with map coordinates

Example queries:
  "What military aircraft are near the Taiwan Strait right now?"
  "Show me all ships within 100 km of the Suez Canal"
  "Are there any cyber threats correlated with the GPS jamming in Ukraine?"
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .connector import ZunvraConnector, IntelSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Well-known regions for fast geo-lookup when no LLM is available
# ---------------------------------------------------------------------------

KNOWN_REGIONS: Dict[str, Tuple[float, float, float]] = {
    # name -> (lat, lon, radius_km)
    "taiwan strait": (24.5, 119.5, 200),
    "south china sea": (12.0, 114.0, 500),
    "strait of hormuz": (26.5, 56.3, 150),
    "suez canal": (30.5, 32.3, 50),
    "black sea": (43.5, 34.0, 400),
    "baltic sea": (58.0, 20.0, 400),
    "mediterranean": (35.0, 18.0, 800),
    "persian gulf": (26.0, 52.0, 300),
    "red sea": (20.0, 38.5, 300),
    "gulf of aden": (12.5, 47.0, 200),
    "east china sea": (30.0, 126.0, 300),
    "sea of japan": (40.0, 135.0, 400),
    "arctic": (80.0, 0.0, 1000),
    "north atlantic": (50.0, -30.0, 1000),
    "english channel": (50.5, 0.0, 100),
    "bosphorus": (41.1, 29.0, 30),
    "malacca strait": (3.0, 101.0, 200),
    "ukraine": (49.0, 32.0, 500),
    "gaza": (31.4, 34.4, 50),
    "beirut": (33.9, 35.5, 30),
    "north korea": (40.0, 127.0, 200),
    "crimea": (45.3, 34.0, 150),
    "kaliningrad": (54.7, 20.5, 100),
}

# Intent classification keywords
DOMAIN_KEYWORDS = {
    "flights": ["flight", "aircraft", "plane", "airplane", "jet", "aviation", "airline"],
    "military_flights": ["military", "fighter", "tanker", "bomber", "reconnaissance", "isr", "awacs", "c-17", "f-35", "f-16", "b-52"],
    "ships": ["ship", "vessel", "tanker", "cargo", "container", "maritime", "ais", "boat", "yacht"],
    "satellites": ["satellite", "orbital", "space", "iss", "tle"],
    "earthquakes": ["earthquake", "seismic", "quake", "tremor", "magnitude"],
    "fires": ["fire", "wildfire", "hotspot", "burn", "firms", "thermal"],
    "gdelt_events": ["conflict", "incident", "event", "gdelt", "attack", "protest", "unrest"],
    "cyber_threats": ["cyber", "c2", "malware", "ransomware", "threat", "hack", "botnet"],
    "gps_jamming": ["gps", "jamming", "spoofing", "navigation", "interference", "sigint"],
    "carriers": ["carrier", "navy", "warship", "strike group", "fleet", "destroyer", "frigate"],
}


@dataclass
class AnalystQuery:
    """Parsed representation of a user's NL query."""

    original: str
    domains: List[str] = field(default_factory=list)
    region_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    radius_km: float = 200.0
    time_window_hours: float = 24.0
    entity_filter: Optional[str] = None  # callsign, MMSI, name, etc.


@dataclass
class AnalystResponse:
    """Structured intelligence answer."""

    query: str
    timestamp: str
    summary: str
    entities_found: int = 0
    results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    cross_domain_insights: List[str] = field(default_factory=list)
    map_focus: Optional[Dict[str, Any]] = None  # {lat, lon, zoom}
    confidence: float = 0.0

    def to_text(self, max_chars: int = 4000) -> str:
        lines = [f"INTELLIGENCE RESPONSE — {self.timestamp}"]
        lines.append(f"Query: {self.query}")
        lines.append(f"Entities found: {self.entities_found}")
        lines.append(f"Confidence: {self.confidence:.0%}")
        lines.append("")
        lines.append(self.summary)
        if self.cross_domain_insights:
            lines.append("")
            lines.append("CROSS-DOMAIN INSIGHTS:")
            for i, ins in enumerate(self.cross_domain_insights, 1):
                lines.append(f"  {i}. {ins}")
        if self.map_focus:
            lines.append("")
            lines.append(f"MAP FOCUS: {self.map_focus.get('lat')}, {self.map_focus.get('lon')} (zoom {self.map_focus.get('zoom', 8)})")
        return "\n".join(lines)[:max_chars]


# ---------------------------------------------------------------------------
# Analyst engine
# ---------------------------------------------------------------------------

class NLIntelAnalyst:
    """
    Natural-language intelligence analyst for the Zunvra OSINT dashboard.

    Flow:
      1. Parse the user query to extract intent, domains, and geography
      2. Fetch the latest snapshot from ZunvraConnector
      3. Filter and correlate data
      4. (Optional) Use LLM for deeper analysis
      5. Return a structured AnalystResponse
    """

    def __init__(self, connector: ZunvraConnector, data_dir: Optional[Path] = None):
        self.connector = connector
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.query_log: List[Dict[str, Any]] = []

    # ── query parsing ─────────────────────────────────────────────────

    def parse_query(self, text: str) -> AnalystQuery:
        """Extract structured intent from free-text query."""
        q = AnalystQuery(original=text)
        lower = text.lower()

        # Detect geographic region
        for region, (lat, lon, radius) in KNOWN_REGIONS.items():
            if region in lower:
                q.region_name = region
                q.lat = lat
                q.lon = lon
                q.radius_km = radius
                break

        # Detect lat/lon if provided directly (e.g., "near 34.5, 32.1")
        coords = re.findall(r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)', text)
        if coords and q.lat is None:
            try:
                q.lat, q.lon = float(coords[0][0]), float(coords[0][1])
            except ValueError:
                pass

        # Detect radius override (e.g., "within 300 km")
        radius_match = re.search(r'within\s+(\d+)\s*km', lower)
        if radius_match:
            q.radius_km = float(radius_match.group(1))

        # Detect domain intent
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in lower:
                    if domain not in q.domains:
                        q.domains.append(domain)
                    break

        # If no domain detected, default to all
        if not q.domains:
            q.domains = list(DOMAIN_KEYWORDS.keys())

        # Detect time window (e.g., "last 6 hours", "past 48h")
        time_match = re.search(r'(?:last|past)\s+(\d+)\s*h(?:ours?)?', lower)
        if time_match:
            q.time_window_hours = float(time_match.group(1))

        # Detect entity filter (callsign, name, etc.)
        for marker in ["callsign", "name", "mmsi", "icao", "registration"]:
            m = re.search(rf'{marker}\s+["\']?(\w+)["\']?', lower)
            if m:
                q.entity_filter = m.group(1)
                break

        return q

    # ── main analysis ─────────────────────────────────────────────────

    async def analyze(
        self,
        text: str,
        llm=None,
        snapshot: Optional[IntelSnapshot] = None,
    ) -> AnalystResponse:
        """
        Run a full NL intelligence query against live data.

        Parameters
        ----------
        text : str
            Free-text question from the user.
        llm : optional
            An LLM instance (with `.chat_raw()`) for deeper analysis.
        snapshot : optional
            Pre-fetched IntelSnapshot; if None, we fetch a fresh one.
        """
        query = self.parse_query(text)
        now = datetime.now(timezone.utc).isoformat()

        # Get data
        snap = snapshot or await self.connector.fetch_full()
        if not snap:
            return AnalystResponse(
                query=text, timestamp=now,
                summary="Unable to connect to Zunvra backend. No data available.",
                confidence=0.0,
            )

        # Filter by geography if we have a location
        if query.lat is not None and query.lon is not None:
            nearby = snap.entities_near(query.lat, query.lon, query.radius_km)
            # Filter to only requested domains
            results = {d: nearby.get(d, []) for d in query.domains if d in nearby}
        else:
            # Return aggregate counts per domain
            results = {}
            domain_map = {
                "flights": snap.flights,
                "military_flights": snap.military_flights,
                "ships": snap.ships,
                "satellites": snap.satellites,
                "earthquakes": snap.earthquakes,
                "fires": snap.fires,
                "gdelt_events": snap.gdelt_events,
                "cyber_threats": snap.cyber_threats,
                "gps_jamming": snap.gps_jamming,
                "carriers": snap.carriers,
            }
            for d in query.domains:
                if d in domain_map and domain_map[d]:
                    results[d] = domain_map[d]

        # Apply entity filter if present
        if query.entity_filter:
            filtered = {}
            ef = query.entity_filter.upper()
            for domain, items in results.items():
                matched = [
                    i for i in items
                    if ef in json.dumps(i).upper()
                ]
                if matched:
                    filtered[domain] = matched
            results = filtered

        total = sum(len(v) for v in results.values())

        # Build summary
        summary_parts = []
        for domain, items in results.items():
            summary_parts.append(f"{domain.replace('_', ' ').title()}: {len(items)} entities")

        summary = "; ".join(summary_parts) if summary_parts else "No matching entities found."
        if query.region_name:
            summary = f"Region: {query.region_name.title()} — {summary}"

        # Cross-domain insights (rule-based)
        insights = self._cross_domain_insights(results, query)

        # Map focus
        map_focus = None
        if query.lat is not None:
            map_focus = {"lat": query.lat, "lon": query.lon, "zoom": 8}

        # LLM enrichment if available
        if llm and total > 0:
            try:
                llm_summary = await self._llm_analyze(llm, text, results, insights)
                if llm_summary:
                    summary = llm_summary
            except Exception as e:
                logger.warning("LLM enrichment failed: %s", e)

        response = AnalystResponse(
            query=text,
            timestamp=now,
            summary=summary,
            entities_found=total,
            results={k: v[:50] for k, v in results.items()},  # cap for sanity
            cross_domain_insights=insights,
            map_focus=map_focus,
            confidence=min(0.95, 0.3 + (0.1 * len(results)) + (0.05 * min(total, 10))),
        )

        # Log
        self.query_log.append({
            "query": text, "timestamp": now,
            "entities_found": total, "domains": list(results.keys()),
        })

        return response

    # ── cross-domain checks ───────────────────────────────────────────

    def _cross_domain_insights(
        self,
        results: Dict[str, List[Dict]],
        query: AnalystQuery,
    ) -> List[str]:
        """Generate rule-based cross-domain observations."""
        insights: List[str] = []

        mil = results.get("military_flights", [])
        ships = results.get("ships", [])
        jamming = results.get("gps_jamming", [])
        cyber = results.get("cyber_threats", [])
        carriers = results.get("carriers", [])

        if mil and ships:
            insights.append(
                f"Co-location detected: {len(mil)} military aircraft and "
                f"{len(ships)} vessels in same area — possible naval exercise or deployment."
            )

        if mil and jamming:
            insights.append(
                f"SIGINT correlation: {len(jamming)} GPS jamming zones coincide with "
                f"{len(mil)} military flights — possible electronic warfare activity."
            )

        if cyber and jamming:
            insights.append(
                f"Multi-domain threat: {len(cyber)} cyber threats and {len(jamming)} "
                f"GPS jamming zones active simultaneously — potential coordinated operation."
            )

        if carriers and mil:
            insights.append(
                f"Carrier activity: {len(carriers)} carrier strike groups in area with "
                f"{len(mil)} military aircraft — elevated military posture."
            )

        if len(results) >= 4:
            insights.append(
                "High multi-domain activity: 4+ intelligence domains active in this area. "
                "Recommend continuous monitoring."
            )

        return insights

    # ── LLM-powered deep analysis ────────────────────────────────────

    async def _llm_analyze(
        self,
        llm,
        query: str,
        results: Dict[str, List[Dict]],
        existing_insights: List[str],
    ) -> Optional[str]:
        """Use an LLM to generate a deeper intelligence assessment."""
        # Build a compact data summary for the prompt
        data_summary = []
        for domain, items in results.items():
            sample = items[:5]  # limit context
            data_summary.append(f"[{domain}] {len(items)} total. Sample: {json.dumps(sample, default=str)[:500]}")

        prompt = (
            "You are an elite OSINT intelligence analyst for the Zunvra Central Intelligence platform.\n"
            f"The operator asked: \"{query}\"\n\n"
            "Live data from Zunvra feeds:\n"
            + "\n".join(data_summary) + "\n\n"
            "Existing rule-based insights:\n"
            + "\n".join(f"- {i}" for i in existing_insights) + "\n\n"
            "Provide a concise (3-6 sentence) intelligence assessment. Include:\n"
            "1. Direct answer to the query\n"
            "2. Any anomalies or patterns worth noting\n"
            "3. Threat level assessment (LOW/MODERATE/ELEVATED/HIGH/CRITICAL)\n"
            "4. Recommended next actions for the analyst\n"
            "Be specific with numbers and entity names. No speculation without evidence."
        )

        raw = await llm.chat_raw(prompt, max_tokens=400)
        return raw.strip() if raw else None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_queries": len(self.query_log),
            "last_query": self.query_log[-1] if self.query_log else None,
        }
