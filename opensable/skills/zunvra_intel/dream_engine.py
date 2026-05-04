"""
#7,  Dream Engine Retrospective

During idle time (no active queries), Sable replays the last 24h of OSINT
data in a "corrupted remix" mode,  shuffling timelines, amplifying weak
signals, and looking for correlations that nobody searched for.

Output: DREAM REPORT,  a memo of creative insights discovered overnight.

"While processing yesterday's data in dream mode, discovered: 3 separate
 ransomware attacks targeted orgs in countries where GPS jamming was
 detected. Possible correlation between EW testing and cyber campaigns."
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


@dataclass
class DreamInsight:
    """A single creative insight from a dream session."""
    insight_id: str
    title: str
    description: str
    domains_connected: List[str]
    novelty_score: float = 0.0  # 0-1, how unexpected
    evidence: List[str] = field(default_factory=list)
    actionable: bool = False


@dataclass
class DreamReport:
    """The full output of one dream session."""
    report_id: str
    session_start: str
    session_end: str
    snapshots_processed: int
    insights: List[DreamInsight] = field(default_factory=list)
    total_correlations_tested: int = 0
    novel_discoveries: int = 0

    def to_text(self) -> str:
        lines = [
            f"DREAM REPORT,  {self.report_id}",
            f"Session: {self.session_start} → {self.session_end}",
            f"Snapshots processed: {self.snapshots_processed}",
            f"Correlations tested: {self.total_correlations_tested}",
            f"Novel discoveries: {self.novel_discoveries}",
            "",
        ]
        for i, ins in enumerate(self.insights, 1):
            lines.append(f"  {i}. [{ins.novelty_score:.0%} novel] {ins.title}")
            lines.append(f"     {ins.description}")
            if ins.evidence:
                for ev in ins.evidence[:3]:
                    lines.append(f"       - {ev}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dream strategies,  different "remixing" approaches
# ---------------------------------------------------------------------------

class _DreamStrategies:
    """Collection of data remixing strategies for creative pattern discovery."""

    @staticmethod
    def temporal_shuffle(snapshots: List[IntelSnapshot]) -> List[Dict[str, Any]]:
        """Shuffle timeline and compare non-adjacent snapshots."""
        if len(snapshots) < 2:
            return []

        insights = []
        indices = list(range(len(snapshots)))
        random.shuffle(indices)

        for i in range(0, len(indices) - 1, 2):
            s1 = snapshots[indices[i]]
            s2 = snapshots[indices[i + 1]]

            # Compare domains that changed significantly between two random points
            for domain in ["military_flights", "cyber_threats", "gps_jamming", "gdelt_events"]:
                count1 = len(getattr(s1, domain, []))
                count2 = len(getattr(s2, domain, []))
                if count1 > 0 and count2 > 0:
                    ratio = max(count1, count2) / max(min(count1, count2), 1)
                    if ratio > 3.0:
                        insights.append({
                            "title": f"Temporal anomaly in {domain}",
                            "description": f"{domain} varied {ratio:.1f}x between non-adjacent snapshots ({count1} vs {count2})",
                            "domains": [domain],
                            "novelty": min(1.0, ratio / 10.0),
                        })

        return insights

    @staticmethod
    def domain_mashup(snapshots: List[IntelSnapshot]) -> List[Dict[str, Any]]:
        """Look for unlikely correlations between unrelated domains."""
        if not snapshots:
            return []

        insights = []
        # Aggregate all snapshots
        totals: Dict[str, List[int]] = {}
        for snap in snapshots:
            for domain in ["military_flights", "cyber_threats", "fires", "earthquakes",
                          "ransomware", "gps_jamming", "internet_outages", "gdelt_events"]:
                totals.setdefault(domain, []).append(len(getattr(snap, domain, [])))

        # Check correlation between unlikely pairs
        unlikely_pairs = [
            ("fires", "cyber_threats"),
            ("earthquakes", "ransomware"),
            ("gps_jamming", "internet_outages"),
            ("fires", "military_flights"),
            ("ransomware", "gps_jamming"),
            ("earthquakes", "military_flights"),
        ]

        for d1, d2 in unlikely_pairs:
            if d1 in totals and d2 in totals:
                v1 = totals[d1]
                v2 = totals[d2]
                if len(v1) == len(v2) and len(v1) >= 3:
                    corr = _pearson(v1, v2)
                    if abs(corr) > 0.5:
                        insights.append({
                            "title": f"Unexpected correlation: {d1} ↔ {d2}",
                            "description": f"Pearson correlation {corr:.2f} between {d1} and {d2} over {len(v1)} snapshots. This is unusual and warrants investigation.",
                            "domains": [d1, d2],
                            "novelty": abs(corr),
                        })

        return insights

    @staticmethod
    def signal_amplification(snapshots: List[IntelSnapshot]) -> List[Dict[str, Any]]:
        """Amplify weak signals that are normally below threshold."""
        insights = []

        for snap in snapshots:
            # Look for "1 or 2" occurrences,  too few to trigger normal alerts
            if 1 <= len(snap.gps_jamming) <= 2:
                insights.append({
                    "title": "Low-level GPS jamming detected",
                    "description": f"Only {len(snap.gps_jamming)} jamming zone(s),  below normal alert threshold but still active.",
                    "domains": ["gps_jamming"],
                    "novelty": 0.4,
                })

            if 1 <= len(snap.ransomware) <= 2:
                insights.append({
                    "title": "Isolated ransomware incident",
                    "description": f"{len(snap.ransomware)} ransomware incident(s),  check if targets share geographic or industry sector.",
                    "domains": ["ransomware"],
                    "novelty": 0.3,
                })

            # Nuclear + earthquake proximity (rarely checked)
            if snap.earthquakes and snap.nuclear_facilities:
                insights.append({
                    "title": "Seismic activity near nuclear infrastructure",
                    "description": f"{len(snap.earthquakes)} earthquakes active with {len(snap.nuclear_facilities)} nuclear facilities in database. Proximity check recommended.",
                    "domains": ["earthquakes", "nuclear_facilities"],
                    "novelty": 0.6,
                })

        # Deduplicate by title
        seen = set()
        unique = []
        for ins in insights:
            if ins["title"] not in seen:
                seen.add(ins["title"])
                unique.append(ins)

        return unique

    @staticmethod
    def geographic_overlay(snapshots: List[IntelSnapshot]) -> List[Dict[str, Any]]:
        """Overlay different domain events geographically to find hot zones."""
        if not snapshots:
            return []

        # Grid the world into 10° cells and count multi-domain hits
        grid: Dict[str, Dict[str, int]] = {}  # "lat_lon" → {domain: count}

        for snap in snapshots:
            for domain, items in [
                ("military_flights", snap.military_flights),
                ("cyber_threats", snap.cyber_threats),
                ("gdelt_events", snap.gdelt_events[:100]),
                ("fires", snap.fires[:100]),
            ]:
                for item in items:
                    lat = item.get("lat") or item.get("latitude")
                    lon = item.get("lon") or item.get("longitude")
                    if lat and lon:
                        try:
                            cell = f"{int(float(lat) / 10) * 10}_{int(float(lon) / 10) * 10}"
                            grid.setdefault(cell, {}).setdefault(domain, 0)
                            grid[cell][domain] += 1
                        except (ValueError, TypeError):
                            continue

        insights = []
        for cell, domains in grid.items():
            if len(domains) >= 3:
                lat, lon = cell.split("_")
                insights.append({
                    "title": f"Multi-domain hotzone at ({lat}°, {lon}°)",
                    "description": f"{len(domains)} domains converge in this grid cell: {', '.join(domains.keys())}",
                    "domains": list(domains.keys()),
                    "novelty": min(1.0, len(domains) / 5.0),
                })

        return insights


# ---------------------------------------------------------------------------
# Dream Engine
# ---------------------------------------------------------------------------

class DreamEngine:
    """
    Replays OSINT data during idle time in creative remix mode.
    Discovers non-obvious correlations and generates Dream Reports.
    """

    MAX_REPORTS = 100

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "dream_reports.json"

        self.reports: List[DreamReport] = []
        self.total_sessions = 0
        self._strategies = _DreamStrategies()
        self._load_state()

    async def dream(
        self,
        snapshots: List[IntelSnapshot],
        llm=None,
    ) -> DreamReport:
        """
        Run a dream session over a list of historical snapshots.
        Each strategy is applied, insights are collected, and optionally
        enriched by LLM for deeper creative analysis.
        """
        now = datetime.now(timezone.utc).isoformat()
        rid = hashlib.sha256(f"dream_{now}_{self.total_sessions}".encode()).hexdigest()[:10]

        report = DreamReport(
            report_id=rid,
            session_start=now,
            session_end="",
            snapshots_processed=len(snapshots),
        )

        all_raw_insights: List[Dict[str, Any]] = []

        # Apply each dream strategy
        all_raw_insights.extend(self._strategies.temporal_shuffle(snapshots))
        all_raw_insights.extend(self._strategies.domain_mashup(snapshots))
        all_raw_insights.extend(self._strategies.signal_amplification(snapshots))
        all_raw_insights.extend(self._strategies.geographic_overlay(snapshots))

        report.total_correlations_tested = len(all_raw_insights) + len(snapshots) * 6

        # Convert raw insights to DreamInsights
        for raw in all_raw_insights:
            iid = hashlib.sha256(raw["title"].encode()).hexdigest()[:8]
            insight = DreamInsight(
                insight_id=iid,
                title=raw["title"],
                description=raw["description"],
                domains_connected=raw.get("domains", []),
                novelty_score=raw.get("novelty", 0.5),
                actionable=raw.get("novelty", 0) > 0.5,
            )
            report.insights.append(insight)

        # LLM creative synthesis
        if llm and all_raw_insights:
            try:
                creative = await self._llm_dream(llm, all_raw_insights, snapshots)
                if creative:
                    report.insights.append(creative)
            except Exception as e:
                logger.debug("LLM dream failed: %s", e)

        report.novel_discoveries = sum(1 for i in report.insights if i.novelty_score > 0.5)
        report.session_end = datetime.now(timezone.utc).isoformat()

        self.reports.append(report)
        if len(self.reports) > self.MAX_REPORTS:
            self.reports = self.reports[-self.MAX_REPORTS:]
        self.total_sessions += 1
        self._save_state()

        logger.info("Dream session complete: %d insights, %d novel",
                    len(report.insights), report.novel_discoveries)

        return report

    async def _llm_dream(
        self, llm, raw_insights: List[Dict], snapshots: List[IntelSnapshot]
    ) -> Optional[DreamInsight]:
        """LLM-powered creative synthesis from dream data."""
        insight_text = "\n".join(f"- {r['title']}: {r['description']}" for r in raw_insights[:10])

        prompt = (
            "You are the Dream Engine of an autonomous OSINT intelligence agent.\n"
            "You are replaying the last 24 hours of global intelligence data in a creative 'dream' state.\n"
            "Your goal: find correlations nobody would look for. Be creative but evidence-based.\n\n"
            f"Raw observations from dream replay:\n{insight_text}\n\n"
            f"Total snapshots reviewed: {len(snapshots)}\n\n"
            "Generate ONE creative synthesis insight that connects multiple observations.\n"
            "Return JSON:\n"
            "{\n"
            '  "title": "creative insight title",\n'
            '  "description": "2-3 sentence explanation of the unexpected connection",\n'
            '  "domains": ["domain1", "domain2"],\n'
            '  "evidence": ["evidence point 1", "evidence point 2"],\n'
            '  "novelty": 0.0-1.0\n'
            "}"
        )

        raw = await llm.chat_raw(prompt, max_tokens=300)
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            return None

        try:
            data = json.loads(m.group())
            return DreamInsight(
                insight_id=hashlib.sha256(data.get("title", "").encode()).hexdigest()[:8],
                title=f"[DREAM] {data.get('title', 'Creative synthesis')}",
                description=data.get("description", ""),
                domains_connected=data.get("domains", []),
                novelty_score=float(data.get("novelty", 0.7)),
                evidence=data.get("evidence", []),
                actionable=True,
            )
        except (json.JSONDecodeError, KeyError):
            return None

    # ── queries ───────────────────────────────────────────────────────

    def get_latest_report(self) -> Optional[DreamReport]:
        return self.reports[-1] if self.reports else None

    def get_all_novel_insights(self) -> List[DreamInsight]:
        insights = []
        for r in self.reports:
            insights.extend(i for i in r.insights if i.novelty_score > 0.5)
        return insights

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "reports": [asdict(r) for r in self.reports[-20:]],
                "total_sessions": self.total_sessions,
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save dream state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_sessions = state.get("total_sessions", 0)
        except Exception as e:
            logger.warning("Failed to load dream state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_sessions": self.total_sessions,
            "total_reports": len(self.reports),
            "total_novel_insights": len(self.get_all_novel_insights()),
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _pearson(x: "List[float] | List[int]", y: "List[float] | List[int]") -> float:
    """Simple Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = sum((xi - mx) ** 2 for xi in x)
    sy = sum((yi - my) ** 2 for yi in y)
    if sx == 0 or sy == 0:
        return 0.0
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    import math
    return sxy / (math.sqrt(sx) * math.sqrt(sy))
