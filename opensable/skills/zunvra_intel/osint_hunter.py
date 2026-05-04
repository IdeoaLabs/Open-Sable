"""
#4,  Autonomous OSINT Hunter

Proactive intelligence gathering,  Sable doesn't wait for data to arrive,
it actively scans OSINT sources (RSS, social, Telegram channels) for new
events, geolocates them, cross-references with live Zunvra feeds, and
generates FLASH INTEL REPORTS injected as new events on the map.

Cycle:
  1. Monitor RSS / web sources continuously
  2. Detect new mention of geopolitical event
  3. Geocode the event location
  4. Pull nearby entities from Zunvra (CCTV, flights, ships, sat imagery)
  5. Generate structured FLASH INTEL REPORT
  6. Emit event for map rendering
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .connector import ZunvraConnector, IntelSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSS / OSINT source definitions
# ---------------------------------------------------------------------------

DEFAULT_OSINT_SOURCES = [
    {"name": "GDELT GKG Live", "url": "https://api.gdeltproject.org/api/v2/doc/doc?query=conflict OR explosion OR attack&mode=artlist&maxrecords=20&format=json", "type": "json"},
    {"name": "USGS Earthquakes", "url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_hour.geojson", "type": "geojson"},
    {"name": "NASA FIRMS Active Fires", "url": "https://firms.modaps.eosdis.nasa.gov/api/area/csv/VIIRS_SNPP_NRT/world/1", "type": "csv"},
    {"name": "BBC World RSS", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "type": "rss"},
    {"name": "Reuters World", "url": "https://feeds.reuters.com/Reuters/worldNews", "type": "rss"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml", "type": "rss"},
    {"name": "The War Zone", "url": "https://www.thedrive.com/the-war-zone/feed", "type": "rss"},
    {"name": "Janes Defence", "url": "https://www.janes.com/feeds/news", "type": "rss"},
    {"name": "Bellingcat", "url": "https://www.bellingcat.com/feed/", "type": "rss"},
]

# Keywords that trigger event extraction
EVENT_KEYWORDS = [
    "explosion", "attack", "missile", "airstrike", "bombing", "shooting",
    "military", "deployed", "carrier", "warship", "fighter jet", "drone strike",
    "earthquake", "tsunami", "eruption", "wildfire", "flooding",
    "cyberattack", "ransomware", "breach", "hack", "outage",
    "protest", "coup", "martial law", "sanctions", "blockade",
    "nuclear", "radiation", "chemical", "biological",
    "jamming", "gps", "electronic warfare",
    "crash", "collision", "shipwreck", "hijack",
]

# Simple geocoding from text (known place names → lat/lon)
# In production this would call Nominatim
FAST_GEOCODE: Dict[str, tuple] = {
    "kyiv": (50.45, 30.52), "moscow": (55.75, 37.62), "beijing": (39.91, 116.40),
    "tehran": (35.69, 51.39), "taipei": (25.03, 121.57), "gaza": (31.50, 34.47),
    "beirut": (33.89, 35.50), "damascus": (33.51, 36.29), "baghdad": (33.31, 44.37),
    "kabul": (34.53, 69.17), "pyongyang": (39.02, 125.75), "seoul": (37.57, 126.98),
    "tokyo": (35.68, 139.69), "london": (51.51, -0.13), "washington": (38.90, -77.04),
    "new york": (40.71, -74.01), "paris": (48.86, 2.35), "berlin": (52.52, 13.41),
    "jerusalem": (31.77, 35.23), "riyadh": (24.69, 46.72), "ankara": (39.93, 32.86),
    "cairo": (30.04, 31.24), "tripoli": (32.90, 13.18), "khartoum": (15.59, 32.53),
    "nairobi": (-1.29, 36.82), "mumbai": (19.08, 72.88), "islamabad": (33.69, 73.04),
    "hanoi": (21.03, 105.85), "manila": (14.60, 120.98), "singapore": (1.35, 103.82),
    "sydney": (-33.87, 151.21), "odesa": (46.48, 30.73), "kharkiv": (49.99, 36.23),
    "crimea": (45.30, 34.00), "donbas": (48.00, 38.00), "suez": (29.97, 32.55),
    "hormuz": (26.50, 56.30), "taiwan": (23.70, 120.96), "south china sea": (12.00, 114.00),
    "black sea": (43.50, 34.00), "baltic": (58.00, 20.00), "mediterranean": (35.00, 18.00),
    "red sea": (20.00, 38.50), "persian gulf": (26.00, 52.00),
}


@dataclass
class FlashReport:
    """A proactive intelligence flash report."""
    report_id: str
    title: str
    summary: str
    source_name: str
    source_url: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_name: Optional[str] = None
    severity: str = "INFO"  # INFO | WARNING | CRITICAL
    timestamp: str = ""
    nearby_entities: Dict[str, int] = field(default_factory=dict)
    keywords_matched: List[str] = field(default_factory=list)
    cross_domain_context: str = ""

    def to_text(self) -> str:
        loc = f" @ {self.location_name} ({self.lat:.2f}, {self.lon:.2f})" if self.lat else ""
        lines = [
            f"⚡ FLASH INTEL,  {self.severity}{loc}",
            f"Source: {self.source_name}",
            f"Title: {self.title}",
            f"Summary: {self.summary}",
        ]
        if self.nearby_entities:
            lines.append(f"Nearby: {self.nearby_entities}")
        if self.cross_domain_context:
            lines.append(f"Context: {self.cross_domain_context}")
        return "\n".join(lines)

    def to_map_event(self) -> Optional[Dict[str, Any]]:
        """Convert to a GeoJSON-like event for map injection."""
        if self.lat is None:
            return None
        return {
            "type": "flash_intel",
            "id": self.report_id,
            "lat": self.lat,
            "lon": self.lon,
            "title": self.title,
            "severity": self.severity,
            "summary": self.summary[:200],
            "timestamp": self.timestamp,
            "source": self.source_name,
        }


# ---------------------------------------------------------------------------
# Autonomous OSINT Hunter
# ---------------------------------------------------------------------------

class AutonomousOSINTHunter:
    """
    Proactive OSINT scanner that monitors open sources, detects events,
    geolocates them, and cross-references with live Zunvra data.
    """

    MAX_REPORTS = 500
    MAX_SEEN = 2000

    def __init__(
        self,
        connector: ZunvraConnector,
        data_dir: Optional[Path] = None,
        sources: Optional[List[Dict]] = None,
    ):
        self.connector = connector
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "osint_hunter_state.json"

        self.sources = sources or DEFAULT_OSINT_SOURCES
        self.reports: List[FlashReport] = []
        self.seen_hashes: set = set()
        self.total_scans = 0
        self.total_events_detected = 0
        self._load_state()

    # ── main scan cycle ───────────────────────────────────────────────

    async def scan(self, llm=None) -> List[FlashReport]:
        """
        Run one complete scan cycle across all sources.
        Returns a list of new FlashReports (events not seen before).
        """
        new_reports: List[FlashReport] = []
        now = datetime.now(timezone.utc).isoformat()

        for source in self.sources:
            try:
                items = await self._fetch_source(source)
                for item in items:
                    title = item.get("title", "")
                    text = item.get("text", item.get("description", ""))
                    url = item.get("url", item.get("link", ""))

                    # Dedup
                    h = hashlib.sha256(f"{title}:{url}".encode()).hexdigest()[:16]
                    if h in self.seen_hashes:
                        continue

                    # Check for event keywords
                    combined = f"{title} {text}".lower()
                    matched_kw = [kw for kw in EVENT_KEYWORDS if kw in combined]
                    if not matched_kw:
                        continue

                    self.seen_hashes.add(h)
                    if len(self.seen_hashes) > self.MAX_SEEN:
                        self.seen_hashes = set(list(self.seen_hashes)[-self.MAX_SEEN:])

                    # Geocode
                    lat, lon, loc_name = self._geocode(combined)

                    # Determine severity
                    crit_words = {"explosion", "missile", "airstrike", "nuclear", "tsunami", "hijack"}
                    warn_words = {"attack", "military", "protest", "cyberattack", "earthquake", "eruption"}
                    if any(w in combined for w in crit_words):
                        severity = "CRITICAL"
                    elif any(w in combined for w in warn_words):
                        severity = "WARNING"
                    else:
                        severity = "INFO"

                    # Cross-reference with Zunvra
                    nearby_entities: Dict[str, int] = {}
                    cross_context = ""
                    if lat is not None:
                        snap = self.connector.last_snapshot
                        if snap:
                            nearby = snap.entities_near(lat, lon, radius_km=100)
                            nearby_entities = {k: len(v) for k, v in nearby.items()}
                            if nearby_entities:
                                parts = [f"{k}: {v}" for k, v in nearby_entities.items()]
                                cross_context = f"Nearby Zunvra entities within 100km: {', '.join(parts)}"

                    # LLM summary if available
                    summary = text[:500]
                    if llm and len(text) > 100:
                        try:
                            summary = await self._llm_summarize(llm, title, text, matched_kw)
                        except Exception as e:
                            logger.debug("LLM summary skipped: %s", e)

                    report = FlashReport(
                        report_id=h,
                        title=title[:200],
                        summary=summary[:500],
                        source_name=source["name"],
                        source_url=url,
                        lat=lat,
                        lon=lon,
                        location_name=loc_name,
                        severity=severity,
                        timestamp=now,
                        nearby_entities=nearby_entities,
                        keywords_matched=matched_kw[:5],
                        cross_domain_context=cross_context,
                    )

                    new_reports.append(report)
                    self.reports.append(report)
                    self.total_events_detected += 1

            except Exception as e:
                logger.debug("Source %s failed: %s", source.get("name"), e)

        if len(self.reports) > self.MAX_REPORTS:
            self.reports = self.reports[-self.MAX_REPORTS:]

        self.total_scans += 1
        self._save_state()

        if new_reports:
            logger.info("OSINT Hunter: %d new events detected across %d sources",
                       len(new_reports), len(self.sources))

        return new_reports

    # ── source fetching ───────────────────────────────────────────────

    async def _fetch_source(self, source: Dict) -> List[Dict[str, str]]:
        """Fetch and normalize items from a single source."""
        import aiohttp

        items: List[Dict[str, str]] = []
        src_type = source.get("type", "rss")
        url = source["url"]

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "OpenSable-OSINTHunter/1.0"},
        ) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                raw = await resp.text()

        if src_type == "rss":
            items = self._parse_rss(raw)
        elif src_type == "json":
            try:
                data = json.loads(raw)
                articles = data if isinstance(data, list) else data.get("articles", [])
                for a in articles[:20]:
                    items.append({
                        "title": a.get("title", ""),
                        "text": a.get("seendate", "") + " " + a.get("title", ""),
                        "url": a.get("url", a.get("link", "")),
                    })
            except json.JSONDecodeError:
                pass
        elif src_type == "geojson":
            try:
                data = json.loads(raw)
                for feat in data.get("features", [])[:20]:
                    props = feat.get("properties", {})
                    coords = (feat.get("geometry", {}).get("coordinates", [0, 0]))
                    items.append({
                        "title": props.get("title", props.get("place", "")),
                        "text": json.dumps(props)[:500],
                        "url": props.get("url", props.get("detail", "")) or "",
                        "lat": str(coords[1]) if len(coords) >= 2 else "",
                        "lon": str(coords[0]) if len(coords) >= 2 else "",
                    })
            except json.JSONDecodeError:
                pass

        return items

    def _parse_rss(self, xml_text: str) -> List[Dict[str, str]]:
        """Minimal RSS/Atom parser."""
        import xml.etree.ElementTree as ET
        items = []
        try:
            root = ET.fromstring(xml_text)
            # RSS 2.0
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                desc = (item.findtext("description") or "").strip()
                link = (item.findtext("link") or "").strip()
                items.append({"title": title, "text": desc, "url": link})
            # Atom fallback
            if not items:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                    title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
                    summary = (entry.findtext("{http://www.w3.org/2005/Atom}summary") or "").strip()
                    link_el = entry.find("{http://www.w3.org/2005/Atom}link")
                    link = link_el.get("href", "") if link_el is not None else ""
                    items.append({"title": title, "text": summary, "url": link})
        except ET.ParseError:
            pass
        return items[:20]

    # ── geocoding ─────────────────────────────────────────────────────

    def _geocode(self, text: str) -> tuple:
        """Fast geocoding from known place names. Returns (lat, lon, name) or (None, None, None)."""
        lower = text.lower()
        for place, (lat, lon) in FAST_GEOCODE.items():
            if place in lower:
                return lat, lon, place.title()
        return None, None, None

    # ── LLM enrichment ────────────────────────────────────────────────

    async def _llm_summarize(self, llm, title: str, text: str, keywords: List[str]) -> str:
        prompt = (
            "You are an OSINT intelligence analyst. Summarize this event in 2-3 sentences.\n"
            "Focus on: what happened, where, who is involved, and potential implications.\n\n"
            f"Title: {title}\n"
            f"Content: {text[:1000]}\n"
            f"Matched keywords: {', '.join(keywords)}\n\n"
            "Summary:"
        )
        raw = await llm.chat_raw(prompt, max_tokens=200)
        return raw.strip() if raw else text[:500]

    # ── queries ───────────────────────────────────────────────────────

    def get_critical_reports(self) -> List[FlashReport]:
        return [r for r in self.reports if r.severity == "CRITICAL"]

    def get_reports_near(self, lat: float, lon: float, radius_km: float = 200) -> List[FlashReport]:
        from .connector import _haversine
        results = []
        for r in self.reports:
            if r.lat is not None and r.lon is not None:
                try:
                    if _haversine(lat, lon, r.lat, r.lon) <= radius_km:
                        results.append(r)
                except (ValueError, TypeError):
                    continue
        return results

    def get_map_events(self) -> List[Dict[str, Any]]:
        """Return all reports as map-ready events."""
        return [e for r in self.reports if (e := r.to_map_event()) is not None]

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "reports": [asdict(r) for r in self.reports[-100:]],
                "seen_hashes": list(self.seen_hashes)[-self.MAX_SEEN:],
                "total_scans": self.total_scans,
                "total_events_detected": self.total_events_detected,
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save OSINT hunter state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.seen_hashes = set(state.get("seen_hashes", []))
                self.total_scans = state.get("total_scans", 0)
                self.total_events_detected = state.get("total_events_detected", 0)
        except Exception as e:
            logger.warning("Failed to load OSINT hunter state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_scans": self.total_scans,
            "total_events_detected": self.total_events_detected,
            "active_reports": len(self.reports),
            "critical_reports": len(self.get_critical_reports()),
            "sources": len(self.sources),
        }
