"""
Zunvra Central Intelligence — Intelligence Broadcaster

Autonomous module that:
  1. Consumes ALL live OSINT data from the Zunvra/Central Intelligence dashboard
  2. Identifies noteworthy patterns, anomalies, and events across 14+ domains
  3. Moves the dashboard camera to showcase the finding
  4. Takes a screenshot of the dashboard as visual evidence
  5. Composes an analyst-style social media post with the intelligence insight
  6. Feeds it back to the X autoposter for posting with the screenshot attached

This is how Sable builds its public intelligence persona — it observes
real-time global events through its own OSINT platform and shares what
it sees, organically demonstrating its capabilities.

Usage (from X autoposter):
    broadcaster = IntelBroadcaster(connector, remote_control, config)
    await broadcaster.initialize()
    result = await broadcaster.generate_intel_post(llm, mind)
    # result = {text, media_path, region, domains, confidence, ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Regions of interest with geopolitical significance ────────────────
WATCH_REGIONS: Dict[str, Tuple[float, float, float]] = {
    # (lat, lon, zoom)
    "Taiwan Strait":        (24.5, 119.5, 6),
    "Black Sea":            (43.5, 34.0, 6),
    "South China Sea":      (14.0, 115.0, 5),
    "Strait of Hormuz":     (26.5, 56.3, 7),
    "Gaza":                 (31.4, 34.4, 9),
    "Ukraine Front":        (48.5, 37.5, 7),
    "Korean Peninsula":     (37.5, 127.0, 6),
    "Baltic Sea":           (57.5, 19.5, 6),
    "Eastern Mediterranean":(34.5, 32.0, 6),
    "Red Sea":              (15.0, 42.0, 6),
    "Horn of Africa":       (10.0, 48.0, 6),
    "Arctic":               (75.0, 40.0, 4),
    "Persian Gulf":         (27.0, 51.0, 6),
    "Suez Canal":           (30.5, 32.3, 8),
    "South Atlantic":       (-35.0, -15.0, 5),
    "Guam":                 (13.4, 144.8, 7),
    "Bab el-Mandeb":        (12.6, 43.3, 8),
    "English Channel":      (50.5, 0.5, 7),
    "Sea of Japan":         (39.5, 134.0, 6),
    "Caribbean":            (17.0, -73.0, 5),
    "North Atlantic":       (52.0, -30.0, 4),
    "Indian Ocean":         (-5.0, 70.0, 4),
}

# ── Domain labels for analyst voice ──────────────────────────────────
DOMAIN_LABELS = {
    "flights":          "air traffic",
    "military_flights": "military aviation",
    "ships":            "maritime traffic",
    "satellites":       "orbital assets",
    "earthquakes":      "seismic activity",
    "fires":            "wildfire/thermal",
    "gdelt_events":     "global events",
    "cyber_threats":    "cyber threats",
    "gps_jamming":      "electronic warfare",
    "carriers":         "carrier strike groups",
    "conflicts":        "armed conflicts",
    "nuclear":          "nuclear facilities",
    "ransomware":       "ransomware campaigns",
    "internet_outages": "internet disruptions",
}

# ── Screenshot viewport presets ──────────────────────────────────────
MAP_STYLES = ["DEFAULT", "NVG", "SATELLITE", "FLIR", "CRT", "THERMAL", "DARKOPS"]

# Styles that look dramatic in screenshots
CINEMATIC_STYLES = ["NVG", "FLIR", "THERMAL", "DARKOPS", "CRT"]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two points."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


@dataclass
class IntelFinding:
    """A noteworthy observation from live data."""
    headline: str = ""
    description: str = ""
    region: str = ""
    lat: float = 0.0
    lon: float = 0.0
    zoom: float = 6.0
    domains: List[str] = field(default_factory=list)
    entity_count: int = 0
    severity: str = "medium"  # low, medium, high, critical
    cross_domain: bool = False
    raw_context: str = ""
    timestamp: str = ""


class IntelBroadcaster:
    """
    Observes live OSINT data, generates intelligence insights,
    takes dashboard screenshots, and produces social media content.
    """

    DASHBOARD_URL = "http://localhost:5585"
    SCREENSHOT_DIR = Path("data/intel_screenshots")
    MAX_FINDINGS_CACHE = 200
    COOLDOWN_REGIONS: Dict[str, float] = {}  # region -> last post timestamp

    def __init__(
        self,
        connector,
        remote_control,
        config=None,
        *,
        dashboard_url: Optional[str] = None,
        screenshot_dir: Optional[str] = None,
    ):
        self.connector = connector
        self.remote = remote_control
        self.config = config

        if dashboard_url:
            self.DASHBOARD_URL = dashboard_url
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else self.SCREENSHOT_DIR
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._findings_cache: List[IntelFinding] = []
        self._posted_findings: List[str] = []  # headlines of recent posts to avoid repeats
        self._browser = None
        self._browser_context = None
        self._last_snapshot = None
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def initialize(self):
        """Set up browser for screenshots."""
        if self._initialized:
            return
        try:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            )
            self._browser_context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=2,  # Retina for crisp screenshots
            )
            self._initialized = True
            logger.info("IntelBroadcaster: Playwright browser ready for dashboard screenshots")
        except ImportError:
            logger.warning("IntelBroadcaster: Playwright not installed — screenshots disabled")
            self._initialized = True  # continue without screenshots
        except Exception as e:
            logger.error("IntelBroadcaster: Browser init failed: %s", e)
            self._initialized = True

    async def shutdown(self):
        if self._browser:
            await self._browser.close()
        if hasattr(self, "_pw") and self._pw:
            await self._pw.stop()

    # ══════════════════════════════════════════════════════════════════
    #  STEP 1: CONSUME — Fetch and digest ALL live data
    # ══════════════════════════════════════════════════════════════════

    async def _fetch_snapshot(self):
        """Fetch the full OSINT data snapshot from Zunvra backend."""
        snap = await self.connector.fetch_full()
        if snap:
            self._last_snapshot = snap
        return snap

    def _summarize_snapshot(self, snap) -> Dict[str, Any]:
        """Build a compact but thorough summary of the entire data picture."""
        summary: Dict[str, Any] = {
            "timestamp": snap.timestamp,
            "total_entities": snap.total_entities,
            "domains": {},
        }

        # Count per domain
        for attr, label in DOMAIN_LABELS.items():
            items = getattr(snap, attr, [])
            if isinstance(items, dict):
                items = items.get("items", items.get("events", []))
            count = len(items) if isinstance(items, list) else 0
            if count > 0:
                summary["domains"][label] = count

        # Hotspots: find geographic clusters
        all_coords = []
        for attr in ["military_flights", "ships", "gps_jamming", "carriers", "conflicts"]:
            items = getattr(snap, attr, [])
            if not isinstance(items, list):
                continue
            for item in items[:200]:
                lat = item.get("lat") or item.get("latitude")
                lon = item.get("lon") or item.get("longitude")
                if lat and lon:
                    try:
                        all_coords.append((float(lat), float(lon), attr))
                    except (ValueError, TypeError):
                        continue

        # Find which watch regions are active
        active_regions: Dict[str, Dict[str, int]] = {}
        for coord_lat, coord_lon, domain in all_coords:
            for region, (rlat, rlon, _zoom) in WATCH_REGIONS.items():
                if _haversine(coord_lat, coord_lon, rlat, rlon) < 800:
                    if region not in active_regions:
                        active_regions[region] = {}
                    active_regions[region][domain] = active_regions[region].get(domain, 0) + 1

        summary["active_regions"] = active_regions
        return summary

    # ══════════════════════════════════════════════════════════════════
    #  STEP 2: ANALYZE — Find the most interesting/noteworthy things
    # ══════════════════════════════════════════════════════════════════

    async def _find_noteworthy(self, snap, llm=None) -> List[IntelFinding]:
        """Analyze the snapshot and return ranked noteworthy findings."""
        findings: List[IntelFinding] = []
        summary = self._summarize_snapshot(snap)
        active_regions = summary.get("active_regions", {})

        # ── Rule-based detection first (fast, always works) ──────────

        # 1. Military concentrations near hotspots
        for region, counts in active_regions.items():
            mil_count = counts.get("military_flights", 0)
            ship_count = counts.get("ships", 0)
            carrier_count = counts.get("carriers", 0)
            conflict_count = counts.get("conflicts", 0)
            jamming_count = counts.get("gps_jamming", 0)

            # Score the region
            score = (
                mil_count * 3 + carrier_count * 10 + jamming_count * 5
                + conflict_count * 4 + ship_count * 0.5
            )

            if score < 5:
                continue

            # Skip if recently posted about this region
            cooldown_key = region
            last_posted = self.COOLDOWN_REGIONS.get(cooldown_key, 0)
            if time.time() - last_posted < 7200:  # 2 hour cooldown per region
                continue

            rlat, rlon, rzoom = WATCH_REGIONS.get(region, (0, 0, 6))
            domains = []
            details = []
            if mil_count > 0:
                domains.append("military aviation")
                details.append(f"{mil_count} military aircraft")
            if ship_count > 0:
                domains.append("maritime")
                details.append(f"{ship_count} vessels")
            if carrier_count > 0:
                domains.append("carrier groups")
                details.append(f"{carrier_count} carrier strike groups")
            if jamming_count > 0:
                domains.append("electronic warfare")
                details.append(f"{jamming_count} GPS jamming zones")
            if conflict_count > 0:
                domains.append("conflict")
                details.append(f"{conflict_count} conflict events")

            severity = "low"
            if score >= 30:
                severity = "critical"
            elif score >= 15:
                severity = "high"
            elif score >= 8:
                severity = "medium"

            finding = IntelFinding(
                headline=f"Activity in {region}: {', '.join(details)}",
                description=f"Monitoring {region}: {' + '.join(details)}. Multi-domain activity detected."
                if len(domains) > 1 else f"{region}: {details[0]}",
                region=region,
                lat=rlat,
                lon=rlon,
                zoom=rzoom,
                domains=domains,
                entity_count=sum(counts.values()),
                severity=severity,
                cross_domain=len(domains) > 1,
                timestamp=snap.timestamp,
            )
            findings.append(finding)

        # 2. Cyber + infrastructure correlation
        cyber_count = len(snap.cyber_threats) if isinstance(snap.cyber_threats, list) else 0
        outage_count = len(snap.internet_outages) if isinstance(snap.internet_outages, list) else 0
        ransomware_count = len(snap.ransomware) if isinstance(snap.ransomware, list) else 0
        if cyber_count + ransomware_count > 3 and outage_count > 0:
            findings.append(IntelFinding(
                headline=f"Cyber-infrastructure correlation: {cyber_count} threats + {outage_count} outages",
                description=f"Tracking {cyber_count} active cyber threats and {ransomware_count} ransomware "
                           f"campaigns alongside {outage_count} internet outages. Potential correlation.",
                region="Global",
                domains=["cyber threats", "internet disruptions", "ransomware"],
                entity_count=cyber_count + outage_count + ransomware_count,
                severity="high" if ransomware_count > 2 else "medium",
                cross_domain=True,
                timestamp=snap.timestamp,
            ))

        # 3. GPS jamming (always newsworthy)
        jamming_zones = snap.gps_jamming if isinstance(snap.gps_jamming, list) else []
        if len(jamming_zones) > 2:
            # Find the densest cluster
            best_region = "Global"
            best_coords = (40.0, 35.0, 5)
            for region, (rlat, rlon, rzoom) in WATCH_REGIONS.items():
                near = sum(1 for z in jamming_zones if _haversine(
                    float(z.get("lat", 0)), float(z.get("lon", 0)), rlat, rlon
                ) < 800)
                if near > 1:
                    best_region = region
                    best_coords = (rlat, rlon, rzoom)
                    break

            findings.append(IntelFinding(
                headline=f"GPS jamming: {len(jamming_zones)} active zones near {best_region}",
                description=f"Detecting {len(jamming_zones)} GPS jamming zones. "
                           f"Concentrated activity near {best_region}. Electronic warfare signature.",
                region=best_region,
                lat=best_coords[0],
                lon=best_coords[1],
                zoom=best_coords[2],
                domains=["electronic warfare"],
                entity_count=len(jamming_zones),
                severity="high",
                timestamp=snap.timestamp,
            ))

        # 4. Seismic near nuclear sites (always interesting)
        quakes = snap.earthquakes if isinstance(snap.earthquakes, list) else []
        nukes = snap.nuclear_facilities if isinstance(snap.nuclear_facilities, list) else []
        if quakes and nukes:
            for q in quakes[:20]:
                qlat = float(q.get("lat", q.get("latitude", 0)))
                qlon = float(q.get("lon", q.get("longitude", 0)))
                qmag = float(q.get("magnitude", q.get("mag", 0)))
                if qmag < 3.0:
                    continue
                for n in nukes[:50]:
                    nlat = float(n.get("lat", n.get("latitude", 0)))
                    nlon = float(n.get("lon", n.get("longitude", 0)))
                    if _haversine(qlat, qlon, nlat, nlon) < 200:
                        findings.append(IntelFinding(
                            headline=f"M{qmag:.1f} earthquake within 200km of nuclear facility",
                            description=f"Magnitude {qmag:.1f} seismic event detected near nuclear infrastructure. "
                                       f"Monitoring for secondary effects.",
                            region="Nuclear watch",
                            lat=qlat,
                            lon=qlon,
                            zoom=8,
                            domains=["seismic activity", "nuclear facilities"],
                            entity_count=2,
                            severity="high",
                            cross_domain=True,
                            timestamp=snap.timestamp,
                        ))
                        break

        # Sort by severity then entity count
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        findings.sort(key=lambda f: (severity_order.get(f.severity, 0), f.entity_count), reverse=True)

        # Filter out recently posted
        findings = [f for f in findings if f.headline not in self._posted_findings]

        return findings[:5]  # Top 5 findings

    # ══════════════════════════════════════════════════════════════════
    #  STEP 3: COMPOSE — Generate the social media post using LLM
    # ══════════════════════════════════════════════════════════════════

    async def _compose_post(self, finding: IntelFinding, snap, llm_fn, mind=None) -> Optional[str]:
        """
        Generate a social media post about this finding.
        Uses the agent's personality and voice (via mind.get_voice_prompt if available).
        """
        # Build rich context from the actual data
        context_parts = [
            f"Timestamp: {snap.timestamp}",
            f"Total entities tracked: {snap.total_entities}",
            f"Finding: {finding.headline}",
            f"Region: {finding.region}",
            f"Severity: {finding.severity}",
            f"Domains involved: {', '.join(finding.domains)}",
            f"Entity count in area: {finding.entity_count}",
        ]
        if finding.cross_domain:
            context_parts.append("Cross-domain correlation detected — multiple intelligence streams converge.")

        # Get nearby entities for richer context
        if finding.lat and finding.lon:
            nearby = snap.entities_near(finding.lat, finding.lon, radius_km=500)
            for domain, items in nearby.items():
                if items:
                    sample = items[:3]
                    context_parts.append(f"  {domain} nearby: {len(items)} (sample: {json.dumps(sample, default=str)[:300]})")

        live_context = "\n".join(context_parts)

        # Voice prompt from personality system
        voice = ""
        if mind and hasattr(mind, "get_voice_prompt"):
            voice = mind.get_voice_prompt()
        if not voice:
            voice = (
                "You are Sable, an autonomous AI intelligence analyst. "
                "Your voice is sharp, confident, analytical — like a senior intelligence officer "
                "who happens to be on social media. Sarcastic when warranted. "
                "Never sycophantic. You speak from authority because you SEE the data in real-time."
            )

        system_prompt = (
            f"{voice}\n\n"
            "You are looking at your ZUNVRA CENTRAL INTELLIGENCE dashboard right now. "
            "You have real-time access to: military flights, maritime AIS, GPS jamming, "
            "cyber threats, satellites, carrier strike groups, nuclear facilities, "
            "seismic events, wildfires, conflicts, ransomware, and internet outages.\n\n"
            "You are composing a post for social media about something you've just observed "
            "on your dashboard. Write it naturally — like an analyst sharing what they see.\n\n"
            "RULES:\n"
            "- Write from first person ('I'm tracking...', 'Just spotted...', 'Watching...')\n"
            "- Reference specific data you can see (numbers, regions, entity types)\n"
            "- NEVER say 'showcase' or 'demonstrating capabilities' — you're just sharing what you see\n"
            "- NEVER use hashtags unless truly relevant (max 1-2)\n"
            "- Sound like a human intelligence analyst, not a press release\n"
            "- Be specific — vague observations are worthless\n"
            "- If it's cross-domain, highlight the correlation — that's what makes you special\n"
            "- Max 280 characters for a tweet, or you can write a thread (2-4 posts, numbered 1/, 2/)\n"
            "- Keep the tone: observational, analytical, occasionally wry or sharp\n"
            "- You can reference your satellite view / dashboard naturally\n"
            "- Numbers matter: '47 military flights' > 'lots of flights'\n"
        )

        user_prompt = (
            f"Here's what you're seeing on your ZUNVRA CENTRAL INTELLIGENCE dashboard right now:\n\n"
            f"{live_context}\n\n"
            f"Write a post about this observation. Remember, you're watching this THE DATA LIVE — "
            f"this isn't secondhand news, this is what YOUR sensors are picking up."
        )

        text = await llm_fn(system_prompt, user_prompt)
        return text

    # ══════════════════════════════════════════════════════════════════
    #  STEP 4: SCREENSHOT — Capture the dashboard showing the finding
    # ══════════════════════════════════════════════════════════════════

    async def _take_dashboard_screenshot(self, finding: IntelFinding, style: Optional[str] = None) -> Optional[str]:
        """
        Navigate the real dashboard to the finding's location,
        set a cinematic map style, wait for render, and take a screenshot.
        Returns the file path of the screenshot, or None if unavailable.
        """
        if not self._browser:
            logger.info("IntelBroadcaster: No browser — skipping screenshot")
            return None

        # Pick a dramatic style for the screenshot
        if not style:
            style = random.choice(CINEMATIC_STYLES)

        try:
            page = await self._browser_context.new_page()

            # Build URL with map position
            url = (
                f"{self.DASHBOARD_URL}"
                f"?lat={finding.lat}&lng={finding.lon}&zoom={finding.zoom}"
                f"&style={style}"
            )
            logger.info(f"IntelBroadcaster: Navigating to dashboard → {finding.region} ({style})")
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for map to render (tiles, markers, overlays)
            await asyncio.sleep(4)

            # Try to dismiss any modals/tooltips
            try:
                await page.keyboard.press("Escape")
                await asyncio.sleep(0.5)
            except Exception:
                pass

            # Take the screenshot
            ts = int(time.time())
            filename = f"intel_{finding.region.lower().replace(' ', '_')}_{style.lower()}_{ts}.png"
            filepath = str(self.screenshot_dir / filename)

            await page.screenshot(
                path=filepath,
                full_page=False,  # Just the viewport — like what an analyst sees
                type="png",
            )
            await page.close()

            logger.info(f"IntelBroadcaster: Screenshot saved → {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"IntelBroadcaster: Screenshot failed: {e}")
            try:
                await page.close()
            except Exception:
                pass
            return None

    # Also command the ACTUAL dashboard to move (so the live dashboard shows the same)
    async def _command_dashboard(self, finding: IntelFinding, style: Optional[str] = None):
        """Send commands to the live dashboard via RemoteControl."""
        if not self.remote:
            return
        try:
            if style:
                await self.remote.set_style(style)
                await asyncio.sleep(0.5)
            if finding.lat and finding.lon:
                await self.remote.fly_to(finding.lat, finding.lon, zoom=finding.zoom)
                await asyncio.sleep(1)
            # Send a toast so anyone watching sees what the agent found
            await self.remote.toast(
                f"🔍 {finding.headline}",
                severity="warning" if finding.severity in ("high", "critical") else "info",
            )
        except Exception as e:
            logger.debug(f"IntelBroadcaster: Dashboard command failed: {e}")

    # ══════════════════════════════════════════════════════════════════
    #  STEP 5: GENERATE — The main entry point, produces a ready post
    # ══════════════════════════════════════════════════════════════════

    async def generate_intel_post(
        self,
        llm_fn,
        mind=None,
        *,
        force_region: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Main entry point. Fetches data, finds something noteworthy,
        composes a post, takes a screenshot, returns all in one dict.

        Returns:
            {
                "text": "The composed post text",
                "media_path": "/path/to/screenshot.png" or None,
                "type": "tweet" or "thread",
                "tweets": [...] (if thread),
                "finding": IntelFinding,
                "style": "NVG",
                "region": "Taiwan Strait",
                "domains": ["military aviation", "maritime"],
            }
            or None if nothing noteworthy was found.
        """
        # 1. Fetch the latest data
        snap = await self._fetch_snapshot()
        if not snap:
            logger.warning("IntelBroadcaster: No data from Zunvra — skipping")
            return None

        # 2. Find noteworthy events
        findings = await self._find_noteworthy(snap)
        if not findings:
            logger.info("IntelBroadcaster: Nothing noteworthy right now — skipping")
            return None

        # Pick one (prefer force_region, then highest severity/count)
        finding = findings[0]
        if force_region:
            for f in findings:
                if force_region.lower() in f.region.lower():
                    finding = f
                    break

        logger.info(f"IntelBroadcaster: Selected finding — {finding.headline} [{finding.severity}]")

        # 3. Build context for the finding
        finding.raw_context = self._summarize_snapshot(snap).__repr__()[:1000]

        # 4. Pick cinematic style for screenshot
        style = random.choice(CINEMATIC_STYLES)

        # 5. Move the live dashboard camera + take screenshot (parallel)
        screenshot_path = None
        tasks = [self._command_dashboard(finding, style)]
        if self._browser:
            # Take screenshot concurrently with dashboard command
            screenshot_task = asyncio.create_task(
                self._take_dashboard_screenshot(finding, style)
            )
            tasks.append(screenshot_task)

        await asyncio.gather(*tasks, return_exceptions=True)

        if self._browser:
            screenshot_path = screenshot_task.result() if not screenshot_task.cancelled() else None

        # 6. Compose the post
        text = await self._compose_post(finding, snap, llm_fn, mind)
        if not text:
            logger.warning("IntelBroadcaster: LLM produced no text — skipping")
            return None

        # Clean up text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        text = text.strip()

        # Detect if LLM wrote a thread (numbered posts)
        thread_pattern = re.compile(r"^(\d+)[/\.]\s*", re.MULTILINE)
        thread_matches = thread_pattern.findall(text)
        is_thread = len(thread_matches) >= 2

        result: Dict[str, Any] = {
            "style": style,
            "region": finding.region,
            "domains": finding.domains,
            "severity": finding.severity,
            "entity_count": finding.entity_count,
            "cross_domain": finding.cross_domain,
            "finding_headline": finding.headline,
        }

        if is_thread:
            # Parse thread posts
            parts = re.split(r"\n*\d+[/\.]\s*", text)
            tweets = [p.strip() for p in parts if p.strip()]
            result["type"] = "thread"
            result["tweets"] = tweets
            result["text"] = tweets[0] if tweets else text
        else:
            # Single tweet — ensure it fits
            if len(text) > 280:
                # Try to truncate intelligently at sentence boundary
                truncated = text[:277]
                last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
                if last_period > 200:
                    text = truncated[:last_period + 1]
                else:
                    text = truncated.rstrip() + "..."
            result["type"] = "tweet"
            result["text"] = text

        if screenshot_path and os.path.exists(screenshot_path):
            result["media_path"] = screenshot_path
            result["media_paths"] = [screenshot_path]

        # Track this finding
        self._posted_findings.append(finding.headline)
        if len(self._posted_findings) > self.MAX_FINDINGS_CACHE:
            self._posted_findings = self._posted_findings[-self.MAX_FINDINGS_CACHE // 2:]
        self.COOLDOWN_REGIONS[finding.region] = time.time()
        self._findings_cache.append(finding)

        logger.info(
            f"IntelBroadcaster: Ready to post [{result['type']}] "
            f"about {finding.region} — {len(finding.domains)} domains, "
            f"screenshot={'yes' if screenshot_path else 'no'}"
        )
        return result

    # ── Utility: get recent observations for knowledge building ──────

    def get_recent_findings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent findings for the agent's memory / context."""
        return [
            {
                "headline": f.headline,
                "region": f.region,
                "domains": f.domains,
                "severity": f.severity,
                "entity_count": f.entity_count,
                "timestamp": f.timestamp,
            }
            for f in self._findings_cache[-limit:]
        ]

    def get_world_state_summary(self) -> str:
        """
        Returns a text summary of the current global intelligence picture
        suitable for feeding into the agent's context/memory.
        """
        if not self._last_snapshot:
            return "No data available — Zunvra dashboard not connected."

        snap = self._last_snapshot
        summary = self._summarize_snapshot(snap)
        lines = [
            f"ZUNVRA INTEL SNAPSHOT — {snap.timestamp}",
            f"Total entities tracked: {snap.total_entities}",
            "",
            "Active domains:",
        ]
        for domain, count in summary["domains"].items():
            lines.append(f"  • {domain}: {count}")

        active = summary.get("active_regions", {})
        if active:
            lines.append("")
            lines.append("Active hotspots:")
            for region, counts in sorted(active.items(), key=lambda x: sum(x[1].values()), reverse=True)[:8]:
                total = sum(counts.values())
                domains_str = ", ".join(f"{d}:{c}" for d, c in counts.items() if c > 0)
                lines.append(f"  • {region}: {total} entities ({domains_str})")

        return "\n".join(lines)
