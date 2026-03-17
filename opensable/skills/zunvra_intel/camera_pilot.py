"""
Zunvra Intelligence — Autonomous Camera Pilot

Sable takes control of the Central Intelligence dashboard map autonomously,
flying to regions of interest, changing zoom, switching visual styles,
highlighting alerts, and narrating its investigation as it goes.

The camera moves with purpose — it's not random. Each move is driven by
the intelligence findings from the 26 analysis modules. Sable acts like
a human analyst who drags the map, zooms in, switches to thermal overlay,
and leans forward when something catches their eye.

Movement behaviors:
  PATROL    — Slow orbit across strategic watch regions
  REACT     — Snap to a high-severity alert location
  DWELL     — Hold position and zoom tight on a finding
  SWEEP     — Progressive scan across a geographic cluster
  CINEMATIC — Dramatic zoom out → style change → zoom in sequence

Usage:
    pilot = AutonomousCameraPilot(remote_control)
    await pilot.narrate_cycle(run_cycle_results, snapshot)
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .live_cameras import LiveCamera, find_cameras_near, generate_camera_for_location, camera_to_dict
from .operation_diary import get_diary

logger = logging.getLogger(__name__)


# ── Style palette — ALWAYS SATELLITE for clean imagery ─────────────
# The user wants ONLY satellite view — no NVG, CRT, THERMAL, etc.
# All domains map to SATELLITE for a consistent real-world view.

DOMAIN_STYLE_MAP: Dict[str, str] = {k: "SATELLITE" for k in [
    "military", "kill_chain", "force_projection", "counter_surveillance",
    "cyber", "sigint_ew", "infrastructure", "narrative_warfare",
    "nuclear", "environmental", "fires", "space", "satellites",
    "finint", "ships", "piracy", "wargaming", "causal",
    "threat_fusion", "default",
]}

# ── Domain → appropriate zoom level ──────────────────────────────────

# Zoom levels raised — minimum 5.0 so Sable never stares at empty ocean
DOMAIN_ZOOM_MAP: Dict[str, float] = {
    "nuclear": 10.0,      # Facility level
    "fires": 7.0,         # Cluster level
    "sigint_ew": 8.0,     # Jamming zone
    "military": 8.0,      # Theater level
    "force_projection": 6.0,  # Fleet level (raised from 5)
    "space": 5.0,         # Orbital ground track (raised from 3)
    "ships": 9.0,         # Chokepoint
    "kill_chain": 8.0,
    "environmental": 7.0,
    "infrastructure": 6.0,
    "wargaming": 5.0,     # Strategic overview (raised from 4)
    "narrative_warfare": 5.0,
    "cyber": 5.0,
    "finint": 7.0,
    "default": 6.0,
}

# ── Strategic watch regions for patrol mode ──────────────────────────
# Comprehensive global coverage — every continent, every ocean, every hotspot.
# Sable is the Eye of God; it sees EVERYTHING.

PATROL_WAYPOINTS: List[Dict[str, Any]] = [
    # ── Eurasia flashpoints ──
    {"name": "Black Sea / Crimea",        "lat": 44.5, "lng": 34.0,  "zoom": 7},
    {"name": "Taiwan Strait",             "lat": 24.0, "lng": 121.0, "zoom": 7},
    {"name": "Strait of Hormuz",          "lat": 26.6, "lng": 56.3,  "zoom": 8},
    {"name": "Red Sea / Bab el-Mandeb",   "lat": 13.0, "lng": 43.0,  "zoom": 7},
    {"name": "South China Sea",           "lat": 12.0, "lng": 114.5, "zoom": 6},
    {"name": "Korean DMZ",                "lat": 38.0, "lng": 127.0, "zoom": 8},
    {"name": "Baltic Sea",                "lat": 58.0, "lng": 20.0,  "zoom": 6},
    {"name": "Eastern Mediterranean",     "lat": 34.5, "lng": 33.5,  "zoom": 7},
    {"name": "Persian Gulf",              "lat": 27.0, "lng": 51.0,  "zoom": 7},
    {"name": "Arctic / GIUK Gap",         "lat": 64.0, "lng": -20.0, "zoom": 5},
    {"name": "Suez Canal",                "lat": 30.5, "lng": 32.3,  "zoom": 10},
    {"name": "Horn of Africa",            "lat": 10.0, "lng": 48.0,  "zoom": 6},
    # ── Europe ──
    {"name": "London",                    "lat": 51.5, "lng": -0.12, "zoom": 11},
    {"name": "Paris",                     "lat": 48.85, "lng": 2.35, "zoom": 11},
    {"name": "Berlin",                    "lat": 52.52, "lng": 13.4, "zoom": 11},
    {"name": "Moscow",                    "lat": 55.75, "lng": 37.6, "zoom": 10},
    {"name": "Istanbul / Bosphorus",      "lat": 41.0,  "lng": 29.0, "zoom": 10},
    {"name": "Kaliningrad",               "lat": 54.7,  "lng": 20.5, "zoom": 9},
    {"name": "Svalbard",                  "lat": 78.0,  "lng": 16.0, "zoom": 6},
    {"name": "Gibraltar Strait",          "lat": 36.0,  "lng": -5.5, "zoom": 9},
    {"name": "North Sea Platforms",       "lat": 57.5,  "lng": 1.5,  "zoom": 7},
    {"name": "Ramstein / Rhine",          "lat": 49.4,  "lng": 7.6,  "zoom": 10},
    # ── Americas ──
    {"name": "Washington DC",             "lat": 38.9,  "lng": -77.0, "zoom": 11},
    {"name": "New York",                  "lat": 40.7,  "lng": -74.0, "zoom": 11},
    {"name": "Panama Canal",              "lat": 9.1,   "lng": -79.7, "zoom": 10},
    {"name": "US Gulf Coast",             "lat": 29.0,  "lng": -90.0, "zoom": 7},
    {"name": "San Diego / SOCAL Op Area", "lat": 32.7,  "lng": -117.2,"zoom": 9},
    {"name": "Norfolk Naval Base",        "lat": 36.9,  "lng": -76.3, "zoom": 10},
    {"name": "Guantánamo Bay",            "lat": 20.0,  "lng": -75.1, "zoom": 9},
    {"name": "South Atlantic",            "lat": -35.0, "lng": -25.0, "zoom": 5},
    {"name": "Caribbean Sea",             "lat": 15.0,  "lng": -70.0, "zoom": 5},
    {"name": "Falkland Islands",          "lat": -51.8, "lng": -59.0, "zoom": 7},
    {"name": "São Paulo",                 "lat": -23.5, "lng": -46.6, "zoom": 10},
    {"name": "Mexico City",              "lat": 19.4,  "lng": -99.1, "zoom": 10},
    # ── Middle East / Central Asia ──
    {"name": "Baghdad",                   "lat": 33.3,  "lng": 44.4, "zoom": 10},
    {"name": "Damascus / Golan",          "lat": 33.5,  "lng": 36.3, "zoom": 9},
    {"name": "Yemen / Sanaa",             "lat": 15.4,  "lng": 44.2, "zoom": 8},
    {"name": "Kabul / Bagram",            "lat": 34.5,  "lng": 69.2, "zoom": 9},
    {"name": "Natanz / Isfahan",          "lat": 32.7,  "lng": 51.7, "zoom": 9},
    {"name": "Astana / Central Asia",     "lat": 51.1,  "lng": 71.4, "zoom": 7},
    # ── Asia-Pacific ──
    {"name": "Tokyo",                     "lat": 35.7,  "lng": 139.7, "zoom": 10},
    {"name": "Shanghai",                  "lat": 31.2,  "lng": 121.5, "zoom": 10},
    {"name": "Pyongyang",                 "lat": 39.0,  "lng": 125.8, "zoom": 10},
    {"name": "Singapore Strait",          "lat": 1.3,   "lng": 103.8, "zoom": 10},
    {"name": "Malacca Strait",            "lat": 3.0,   "lng": 101.0, "zoom": 7},
    {"name": "Mumbai / Arabian Sea",      "lat": 19.1,  "lng": 72.9,  "zoom": 9},
    {"name": "Diego Garcia",              "lat": -7.3,  "lng": 72.4,  "zoom": 8},
    {"name": "Guam",                      "lat": 13.4,  "lng": 144.8, "zoom": 8},
    {"name": "Okinawa / Kadena",          "lat": 26.4,  "lng": 127.8, "zoom": 9},
    {"name": "Sydney / Tasman Sea",       "lat": -33.9, "lng": 151.2, "zoom": 9},
    {"name": "Pearl Harbor",              "lat": 21.3,  "lng": -157.9,"zoom": 9},
    # ── Africa ──
    {"name": "Djibouti / Camp Lemonnier", "lat": 11.5,  "lng": 43.1,  "zoom": 9},
    {"name": "Lagos / Gulf of Guinea",    "lat": 6.5,   "lng": 3.4,   "zoom": 8},
    {"name": "Sahel / Niger",             "lat": 14.0,  "lng": 2.0,   "zoom": 6},
    {"name": "Cape of Good Hope",         "lat": -34.4, "lng": 18.5,  "zoom": 7},
    {"name": "Mogadishu / Somalia",       "lat": 2.0,   "lng": 45.3,  "zoom": 8},
    {"name": "Libya / Tripoli",           "lat": 32.9,  "lng": 13.2,  "zoom": 8},
    {"name": "Eastern Congo",             "lat": -1.7,  "lng": 29.2,  "zoom": 7},
    {"name": "Sudan / Khartoum",          "lat": 15.6,  "lng": 32.5,  "zoom": 8},
    # ── Polar / Remote ──
    {"name": "North Pole / Arctic",       "lat": 80.0,  "lng": 0.0,   "zoom": 5},
    {"name": "Antarctica / McMurdo",      "lat": -77.8, "lng": 166.7, "zoom": 5},
    {"name": "Midway Atoll",              "lat": 28.2,  "lng": -177.4,"zoom": 7},
    {"name": "Ascension Island",          "lat": -7.9,  "lng": -14.4, "zoom": 8},
]


@dataclass
class CameraMove:
    """A single camera movement instruction."""
    lat: float
    lng: float
    zoom: float = 6.0
    style: str = "DEFAULT"
    label: str = ""
    domain: str = "default"
    severity: str = "info"     # info, warning, error, critical
    dwell_seconds: float = 3.0  # How long to stay before next move
    highlight_radius: float = 0.0  # Highlight ring radius in meters (0 = none)
    highlight_color: str = "#ff0000"
    toast_message: str = ""
    media_url: str = ""        # If set, open this URL in the LiveStreamViewer


class AutonomousCameraPilot:
    """
    Drives the Central Intelligence dashboard camera autonomously based on
    intelligence findings.

    The pilot receives analysis results from run_cycle() and the raw
    IntelSnapshot, then generates a sequence of purposeful camera
    movements — each one matching a finding to a location, zoom level,
    and visual style.
    """

    def __init__(
        self,
        remote,  # RemoteControl instance
        *,
        move_delay: float = 2.5,       # Seconds between moves (minimum)
        dwell_default: float = 4.0,    # Default seconds to stay on a finding
        max_moves_per_cycle: int = 25,  # Cap — raised to cover all data sources
        enable_style_changes: bool = True,
        enable_highlights: bool = True,
        enable_toasts: bool = True,
        patrol_on_idle: bool = True,
        video_searcher=None,            # async (query, count) → [{url, title, thumbnail}]
        news_searcher=None,             # async (query, max_items) → [{title, link, source}]
    ):
        self.remote = remote
        self.move_delay = move_delay
        self.dwell_default = dwell_default
        self.max_moves = max_moves_per_cycle
        self.enable_styles = enable_style_changes
        self.enable_highlights = enable_highlights
        self.enable_toasts = enable_toasts
        self.patrol_on_idle = patrol_on_idle

        # ── Web research callbacks (injected by ZunvraIntelSkill) ────
        self._video_searcher = video_searcher    # YouTube search
        self._news_searcher = news_searcher      # GDELT / RSS news search

        self._current_style: str = "DEFAULT"
        self._current_lat: float = 0.0
        self._current_lng: float = 0.0
        self._patrol_index: int = 0
        self._total_moves: int = 0
        self._last_move_time: float = 0.0
        self._move_history: List[Dict[str, Any]] = []
        # Eye of God — diversity tracking (PERSISTENT across restarts)
        self._visited_cells_path = Path(__file__).parent / "data" / "visited_cells.json"
        self._visited_cells: Set[Tuple[int, int]] = self._load_visited_cells()
        self._cycle_count: int = 0  # How many narrate_cycle() invocations
        self._last_opening_coords: Tuple[float, float] = (0.0, 0.0)
        # Live camera tracking — avoid showing the same cam repeatedly
        self._shown_live_cameras: Set[str] = set()  # URLs already shown this session
        self._live_camera_cooldown: float = 0.0      # Timestamp of last live stream open
        # Operation diary / dossier tracking (persistent via OperationDiary)
        self._op_diary = get_diary()
        self._diary_entries: List[Dict[str, Any]] = []
        self._dossier_entries: List[Dict[str, Any]] = []

        # ── Story tracking — follow-up system across cycles ──
        # Maps story_id → {title, domain, severity, first_seen, last_seen,
        #   cycle_count, lat, lng, keywords, last_detail}
        self._tracked_stories: Dict[str, Dict[str, Any]] = {}
        self._tracked_stories_path = Path(__file__).parent / "data" / "tracked_stories.json"
        self._load_tracked_stories()

    # ── Main entry point — narrate a full analysis cycle ─────────────

    async def narrate_cycle(
        self,
        results: Dict[str, Any],
        snapshot,  # IntelSnapshot
        *,
        module_alerts: Optional[Dict[str, list]] = None,
    ) -> Dict[str, Any]:
        """
        Drive the dashboard camera through the findings of a run_cycle().

        Eye of God mode: Sable NEVER follows the same path twice.
        It mines ALL live data for coordinates, randomizes its patrol,
        and scans every corner of the globe across successive cycles.
        """
        self._cycle_count += 1
        # Reset live camera tracking each cycle so streams keep opening
        self._shown_live_cameras.clear()
        self._live_camera_cooldown = 0.0
        self._eligible_move_count = 0
        moves: List[CameraMove] = []

        # ── 0. ADAPTIVE move budget — Sable decides how many moves based on data ──
        # Instead of a hardcoded cap, the budget scales with threat level
        # and number of critical/error-severity findings.
        # Low threat (0-1): 3-5 moves (quick scan, leave camera free for user)
        # Moderate (2-3):   5-8 moves
        # High (4-5):       8-12 moves (more to cover, but still leaves room)
        threat_level = results.get("threat_level", 0)
        adaptive_budget = self._calculate_move_budget(results, snapshot)
        logger.info("Adaptive camera budget: %d moves (threat=%d/5)", adaptive_budget, threat_level)

        # ── 1. Opening move — RANDOMIZED, never the same spot ──
        opening = self._pick_random_opening(snapshot)
        moves.append(CameraMove(
            lat=opening[0], lng=opening[1], zoom=4.0,
            style="SATELLITE",
            label="Strategic overview",
            domain="threat_fusion",
            dwell_seconds=2.0,
            toast_message=f"SABLE initiating intelligence sweep #{self._cycle_count}...",
            severity="info",
        ))

        # ── 2. Generate moves from ALL analysis results + live data ──
        finding_moves = self._generate_finding_moves(results, snapshot, module_alerts)
        moves.extend(finding_moves)

        # ── 3. Patrol — only if budget allows and findings are sparse ──
        remaining = adaptive_budget - len(moves)
        if remaining > 1:
            patrol_count = min(2, remaining - 1)  # Leave 1 slot for closing
            patrol_moves = self._generate_patrol_moves(patrol_count, snapshot)
            moves.extend(patrol_moves)

        # ── 4. Closing move — RANDOMIZED location ──
        # Close at a different region than opening
        closing = self._pick_random_closing(snapshot, opening)
        moves.append(CameraMove(
            lat=closing[0], lng=closing[1], zoom=3.5,
            style="SATELLITE",
            label="Intelligence sweep complete",
            domain="threat_fusion",
            dwell_seconds=2.0,
            toast_message=f"Sweep #{self._cycle_count} complete — Threat Level {threat_level}/5: {results.get('threat_name', '?')}",
            severity="warning" if threat_level >= 3 else "info",
        ))

        # ── 5. Smart trim: prioritize by severity, keep within adaptive budget ──
        moves = self._prioritize_and_trim(moves, adaptive_budget)

        # ── 5b. Build & push intelligence brief to dashboard overlay ──
        try:
            brief = self._build_intel_brief(results, snapshot, moves, module_alerts)

            # ── 5b-ii. Active web research — YouTube + news search ──
            if brief.get("intelEvents"):
                try:
                    enriched = await self._enrich_with_web_research(brief["intelEvents"])
                    brief["intelEvents"] = enriched
                except Exception as web_exc:
                    logger.warning("Web research enrichment failed: %s", web_exc)

            await self.remote.push_intel_brief(brief)
            logger.info("Pushed intel brief: threat=%s/5, domains=%d, wargames=%d, events=%d",
                        brief.get("overallThreat"), len(brief.get("domains", [])),
                        len(brief.get("wargameScenarios", [])), len(brief.get("intelEvents", [])))
        except Exception as exc:
            logger.warning("Failed to push intel brief: %s", exc)

        # ── 5c. Send missile/kinetic trajectory arcs to dashboard ──
        try:
            await self._send_trajectory_arcs()
        except Exception as exc:
            logger.warning("Failed to send trajectory arcs: %s", exc)

        # ── 6. Execute the moves + push live intel events ──
        executed = await self._execute_moves(moves)

        return {
            "total_moves": executed,
            "planned_moves": len(moves),
            "styles_used": list(set(m.style for m in moves)),
            "regions_visited": [m.label for m in moves if m.label],
            "total_dwell_time": sum(m.dwell_seconds for m in moves),
            "pilot_total_moves": self._total_moves,
            "cycle_number": self._cycle_count,
            "unique_cells_visited": len(self._visited_cells),
        }

    # ── Generate camera moves from intelligence findings ─────────────

    def _generate_finding_moves(
        self,
        results: Dict[str, Any],
        snapshot,
        module_alerts: Optional[Dict[str, list]] = None,
    ) -> List[CameraMove]:
        """Extract locations from module results and snapshot, sorted by severity."""
        moves: List[CameraMove] = []

        # --- GPS Jamming (SIGINT/EW) → fly to each jamming zone ---
        if snapshot.gps_jamming:
            for zone in snapshot.gps_jamming[:2]:  # Top 2 (leave room for other sources)
                lat = zone.get("lat", zone.get("latitude"))
                lng = zone.get("lon", zone.get("lng", zone.get("longitude")))
                if lat and lng:
                    moves.append(CameraMove(
                        lat=float(lat), lng=float(lng), zoom=8.0,
                        style="SATELLITE",
                        label=f"GPS Jamming Zone",
                        domain="sigint_ew",
                        severity="warning",
                        dwell_seconds=4.0,
                        highlight_radius=100_000,
                        highlight_color="#ff6600",
                        toast_message=f"EW activity — GPS jamming detected",
                    ))

        # --- Military flight concentrations ---
        mil_clusters = self._cluster_entities(snapshot.military_flights, "Military cluster")
        for cluster in mil_clusters[:2]:
            moves.append(CameraMove(
                lat=cluster["lat"], lng=cluster["lng"], zoom=8.0,
                style="SATELLITE",
                label=f"Military activity — {cluster['count']} aircraft",
                domain="military",
                severity="warning" if cluster["count"] > 10 else "info",
                dwell_seconds=5.0,
                highlight_radius=200_000,
                highlight_color="#00ff00",
                toast_message=f"Analyzing {cluster['count']} military aircraft in region",
            ))

        # --- Nuclear alerts ---
        if results.get("nuclear_alerts", 0) > 0 and module_alerts and "nuclear" in module_alerts:
            for alert in module_alerts["nuclear"][:2]:
                lat = getattr(alert, "lat", None) or (alert.get("lat") if isinstance(alert, dict) else None)
                lng = getattr(alert, "lng", None) or getattr(alert, "lon", None) or (alert.get("lng", alert.get("lon")) if isinstance(alert, dict) else None)
                if lat and lng:
                    moves.append(CameraMove(
                        lat=float(lat), lng=float(lng), zoom=10.0,
                        style="SATELLITE",
                        label="Nuclear facility alert",
                        domain="nuclear",
                        severity="error",
                        dwell_seconds=6.0,
                        highlight_radius=50_000,
                        highlight_color="#ff0000",
                        toast_message="Nuclear activity anomaly detected",
                    ))
        # Fallback: use nuclear facility positions from snapshot
        elif results.get("nuclear_alerts", 0) > 0 and snapshot.nuclear_facilities:
            fac = random.choice(snapshot.nuclear_facilities[:10])
            lat = fac.get("lat", fac.get("latitude"))
            lng = fac.get("lon", fac.get("lng", fac.get("longitude")))
            if lat and lng:
                moves.append(CameraMove(
                    lat=float(lat), lng=float(lng), zoom=10.0,
                    style="SATELLITE",
                    label=f"Nuclear facility: {fac.get('name', 'unknown')}",
                    domain="nuclear",
                    severity="warning",
                    dwell_seconds=5.0,
                    highlight_radius=50_000,
                    highlight_color="#ff4400",
                    toast_message=f"Monitoring nuclear facility — {fac.get('name', 'unknown')}",
                ))

        # --- Wargaming live scenarios ---
        if results.get("wargame_live_scenarios", 0) > 0:
            scenarios = results.get("wargame_escalation_risks", [])
            for sc in scenarios[:2]:
                # Match scenario trigger to a known region
                region_coords = self._match_scenario_to_coords(sc.get("trigger", ""))
                if region_coords:
                    moves.append(CameraMove(
                        lat=region_coords[0], lng=region_coords[1],
                        zoom=5.0,
                        style="SATELLITE",
                        label=f"Wargame: {sc.get('risk', '?').upper()}",
                        domain="wargaming",
                        severity="error" if sc.get("risk") in ("high", "extreme") else "warning",
                        dwell_seconds=5.0,
                        toast_message=f"Wargame scenario active: {sc.get('trigger', '')[:60]}",
                    ))

        # --- Kill chain alerts ---
        if results.get("kill_chain_alerts", 0) > 0:
            # Use military flight positions as proxy
            if snapshot.military_flights:
                for f in snapshot.military_flights[:2]:
                    lat = f.get("lat")
                    lng = f.get("lon", f.get("lng"))
                    if lat and lng:
                        moves.append(CameraMove(
                            lat=float(lat), lng=float(lng), zoom=9.0,
                            style="SATELLITE",
                            label="Kill chain phase advancement",
                            domain="kill_chain",
                            severity="error",
                            dwell_seconds=5.0,
                            highlight_radius=150_000,
                            highlight_color="#ff0044",
                            toast_message="Kill chain tracker — phase advancement detected",
                        ))
                        break  # Just one

        # --- Narrative warfare clusters ---
        if results.get("narrative_alerts", 0) > 0:
            # Can't pinpoint narrative on map, use overview
            moves.append(CameraMove(
                lat=48.0, lng=37.0, zoom=5.5,  # Eastern Europe as proxy
                style="SATELLITE",
                label="Information operations detected",
                domain="narrative_warfare",
                dwell_seconds=4.0,
                toast_message=f"Narrative warfare: {results.get('narrative_alerts', 0)} coordinated activity alerts",
                severity="warning",
            ))

        # --- Infrastructure alerts ---
        if results.get("infra_alerts", 0) > 5:
            moves.append(CameraMove(
                lat=40.0, lng=-20.0, zoom=5.0,  # Atlantic infrastructure corridor
                style="SATELLITE",
                label=f"Infrastructure: {results.get('infra_alerts', 0)} alerts",
                domain="infrastructure",
                dwell_seconds=4.0,
                toast_message=f"Critical infrastructure: {results.get('infra_alerts', 0)} alerts across sectors",
                severity="warning",
            ))

        # --- Internet outages ---
        if snapshot.internet_outages:
            for outage in snapshot.internet_outages[:2]:
                lat = outage.get("lat", outage.get("latitude"))
                lng = outage.get("lon", outage.get("lng", outage.get("longitude")))
                if lat and lng:
                    moves.append(CameraMove(
                        lat=float(lat), lng=float(lng), zoom=6.0,
                        style="SATELLITE",
                        label=f"Internet outage: {outage.get('country', outage.get('location', '?'))}",
                        domain="infrastructure",
                        severity="warning",
                        dwell_seconds=4.0,
                        highlight_radius=300_000,
                        highlight_color="#ff8800",
                        toast_message=f"Internet outage detected — {outage.get('country', '?')}",
                    ))

        # --- Ship concentrations (chokepoints) ---
        if len(snapshot.ships) > 20:
            ship_clusters = self._cluster_entities(snapshot.ships, "Maritime cluster")
            for cluster in ship_clusters[:1]:
                moves.append(CameraMove(
                    lat=cluster["lat"], lng=cluster["lng"], zoom=9.0,
                    style="SATELLITE",
                    label=f"Maritime concentration — {cluster['count']} vessels",
                    domain="ships",
                    severity="info",
                    dwell_seconds=4.0,
                    toast_message=f"Analyzing {cluster['count']} vessels in chokepoint",
                ))

        # --- Environmental — major earthquakes ---
        major_quakes = [q for q in snapshot.earthquakes
                        if float(q.get("mag", q.get("magnitude", 0))) >= 5.5]
        for quake in major_quakes[:2]:
            lat = quake.get("lat", quake.get("latitude"))
            lng = quake.get("lng", quake.get("lon", quake.get("longitude")))
            mag = quake.get("mag", quake.get("magnitude", "?"))
            if lat and lng:
                moves.append(CameraMove(
                    lat=float(lat), lng=float(lng), zoom=8.0,
                    style="SATELLITE",
                    label=f"Seismic event M{mag}",
                    domain="environmental",
                    severity="warning" if float(str(mag)) >= 6.0 else "info",
                    dwell_seconds=4.0,
                    highlight_radius=100_000,
                    highlight_color="#ff6600",
                    toast_message=f"Seismic event — magnitude {mag}",
                ))

        # --- FININT dark fleet ---
        if results.get("finint_alerts", 0) > 0 and snapshot.ships:
            # Suspicious vessels — look at low-speed ships
            suspicious = [s for s in snapshot.ships
                          if float(s.get("speed", s.get("sog", 99))) < 2.0]
            if suspicious:
                s = suspicious[0]
                lat = s.get("lat", s.get("latitude"))
                lng = s.get("lon", s.get("lng", s.get("longitude")))
                if lat and lng:
                    moves.append(CameraMove(
                        lat=float(lat), lng=float(lng), zoom=11.0,
                        style="SATELLITE",
                        label="Dark fleet suspect vessel",
                        domain="finint",
                        severity="warning",
                        dwell_seconds=5.0,
                        highlight_radius=20_000,
                        highlight_color="#aa00ff",
                        toast_message="FININT: Possible sanctions-evading vessel",
                    ))

        # --- Fire clusters (environmental) ---
        if len(snapshot.fires) > 100:
            fire_clusters = self._cluster_entities(snapshot.fires, "Fire cluster", grid_size=2.0)
            for cluster in fire_clusters[:1]:
                if cluster["count"] > 20:
                    moves.append(CameraMove(
                        lat=cluster["lat"], lng=cluster["lng"], zoom=7.0,
                        style="SATELLITE",
                        label=f"Fire cluster — {cluster['count']} hotspots",
                        domain="fires",
                        severity="info",
                        dwell_seconds=3.0,
                        toast_message=f"Monitoring {cluster['count']} active fire hotspots",
                    ))

        # ═══════════════════════════════════════════════════════════════
        # EYE OF GOD — New data sources for complete global coverage
        # ═══════════════════════════════════════════════════════════════

        # --- CCTV cameras — fly to and inspect ---
        if snapshot.cctv:
            cctv_with_coords = [c for c in snapshot.cctv if self._extract_coords(c)]
            if cctv_with_coords:
                sampled = random.sample(cctv_with_coords, min(3, len(cctv_with_coords)))
                for cam in sampled:
                    lat, lng = self._extract_coords(cam)
                    cam_id = cam.get("id", cam.get("name", "unknown"))
                    cam_name = cam.get("name", cam.get("location", cam_id))
                    cam_url = cam.get("media_url", cam.get("url", ""))
                    moves.append(CameraMove(
                        lat=lat, lng=lng, zoom=14.0,
                        style="SATELLITE",
                        label=f"CCTV: {cam_name}",
                        domain="cyber",
                        severity="info",
                        dwell_seconds=5.0,
                        toast_message=f"Inspecting CCTV camera — {cam_name}",
                        media_url=cam_url,
                    ))

        # --- Open cameras (DOT / Webcams) — fly to and show feed ---
        if snapshot.open_cameras:
            oc_with_coords = [c for c in snapshot.open_cameras if self._extract_coords(c)]
            if oc_with_coords:
                sampled = random.sample(oc_with_coords, min(3, len(oc_with_coords)))
                for cam in sampled:
                    lat, lng = self._extract_coords(cam)
                    cam_name = cam.get("name", cam.get("location", cam.get("id", "DOT cam")))
                    cam_url = cam.get("url", cam.get("media_url", cam.get("image_url", "")))
                    source = cam.get("source", cam.get("agency", "DOT"))
                    moves.append(CameraMove(
                        lat=lat, lng=lng, zoom=13.0,
                        style="SATELLITE",
                        label=f"Webcam: {cam_name} ({source})",
                        domain="infrastructure",
                        severity="info",
                        dwell_seconds=4.0,
                        toast_message=f"Viewing DOT webcam — {cam_name}",
                        media_url=cam_url,
                    ))

        # --- Conflicts / ACLED events ---
        if snapshot.conflicts:
            conflict_with_coords = [c for c in snapshot.conflicts if self._extract_coords(c)]
            if conflict_with_coords:
                top_conflicts = random.sample(conflict_with_coords, min(3, len(conflict_with_coords)))
                for con in top_conflicts:
                    lat, lng = self._extract_coords(con)
                    event_type = con.get("event_type", con.get("type", "conflict"))
                    location = con.get("location", con.get("country", "unknown"))
                    moves.append(CameraMove(
                        lat=lat, lng=lng, zoom=8.0,
                        style="SATELLITE",
                        label=f"Conflict: {event_type} — {location}",
                        domain="military",
                        severity="warning",
                        dwell_seconds=4.0,
                        highlight_radius=100_000,
                        highlight_color="#ff2200",
                        toast_message=f"Active conflict — {event_type} in {location}",
                    ))

        # --- Carrier / USNI fleet positions ---
        if snapshot.carriers:
            for carrier in snapshot.carriers[:3]:
                coords = self._extract_coords(carrier)
                if coords:
                    lat, lng = coords
                    name = carrier.get("name", carrier.get("vessel", "carrier group"))
                    moves.append(CameraMove(
                        lat=lat, lng=lng, zoom=9.0,
                        style="SATELLITE",
                        label=f"Carrier: {name}",
                        domain="force_projection",
                        severity="info",
                        dwell_seconds=5.0,
                        highlight_radius=200_000,
                        highlight_color="#0066ff",
                        toast_message=f"Tracking carrier group — {name}",
                    ))

        # --- Cyber threats (geographic origin) ---
        if snapshot.cyber_threats:
            cyber_with_coords = [c for c in snapshot.cyber_threats if self._extract_coords(c)]
            if cyber_with_coords:
                sampled = random.sample(cyber_with_coords, min(2, len(cyber_with_coords)))
                for threat in sampled:
                    lat, lng = self._extract_coords(threat)
                    threat_type = threat.get("type", threat.get("category", "cyber threat"))
                    moves.append(CameraMove(
                        lat=lat, lng=lng, zoom=7.0,
                        style="SATELLITE",
                        label=f"Cyber: {threat_type}",
                        domain="cyber",
                        severity="warning",
                        dwell_seconds=4.0,
                        highlight_radius=150_000,
                        highlight_color="#00ffcc",
                        toast_message=f"Cyber threat origin — {threat_type}",
                    ))

        # --- Ransomware clusters ---
        if snapshot.ransomware:
            ransom_with_coords = [r for r in snapshot.ransomware if self._extract_coords(r)]
            if ransom_with_coords:
                r = random.choice(ransom_with_coords)
                lat, lng = self._extract_coords(r)
                group_name = r.get("group", r.get("name", "unknown"))
                victim = r.get("victim", r.get("target", "?"))
                moves.append(CameraMove(
                    lat=lat, lng=lng, zoom=8.0,
                    style="SATELLITE",
                    label=f"Ransomware: {group_name}",
                    domain="cyber",
                    severity="error",
                    dwell_seconds=5.0,
                    highlight_radius=100_000,
                    highlight_color="#ff00ff",
                    toast_message=f"Ransomware activity — {group_name} targeting {victim}",
                ))

        # --- GDELT events ---
        if snapshot.gdelt_events:
            gdelt_with_coords = [e for e in snapshot.gdelt_events if self._extract_coords(e)]
            if gdelt_with_coords:
                sampled = random.sample(gdelt_with_coords, min(2, len(gdelt_with_coords)))
                for event in sampled:
                    lat, lng = self._extract_coords(event)
                    # GDELT uses GeoJSON Feature format
                    props = event.get("properties", {}) if isinstance(event.get("properties"), dict) else {}
                    title = (props.get("name") or props.get("title") or
                             event.get("title") or event.get("headline") or
                             event.get("event_type") or "geopolitical event")
                    if len(str(title)) > 60:
                        title = str(title)[:57] + "..."
                    moves.append(CameraMove(
                        lat=lat, lng=lng, zoom=6.0,
                        style="SATELLITE",
                        label=f"GDELT: {title}",
                        domain="causal",
                        severity="info",
                        dwell_seconds=3.0,
                        toast_message=f"Global event — {title}",
                    ))

        # --- News feed geolocations ---
        if snapshot.news_feed:
            news_with_coords = [n for n in snapshot.news_feed if self._extract_coords(n)]
            if news_with_coords:
                sampled = random.sample(news_with_coords, min(2, len(news_with_coords)))
                for article in sampled:
                    lat, lng = self._extract_coords(article)
                    headline = article.get("title", article.get("headline", "breaking news"))
                    if len(str(headline)) > 60:
                        headline = str(headline)[:57] + "..."
                    moves.append(CameraMove(
                        lat=lat, lng=lng, zoom=7.0,
                        style="SATELLITE",
                        label=f"News: {headline}",
                        domain="narrative_warfare",
                        severity="info",
                        dwell_seconds=3.0,
                        toast_message=f"News — {headline}",
                    ))

        # --- Prediction markets (if geographic data) ---
        if snapshot.prediction_markets:
            pred_with_coords = [p for p in snapshot.prediction_markets if self._extract_coords(p)]
            if pred_with_coords:
                p = random.choice(pred_with_coords)
                lat, lng = self._extract_coords(p)
                question = p.get("question", p.get("title", "market signal"))
                if len(str(question)) > 50:
                    question = str(question)[:47] + "..."
                moves.append(CameraMove(
                    lat=lat, lng=lng, zoom=5.0,
                    style="SATELLITE",
                    label=f"Prediction: {question}",
                    domain="wargaming",
                    severity="info",
                    dwell_seconds=3.0,
                    toast_message=f"Prediction market signal — {question}",
                ))

        # --- Dark intel (dict — may contain nested lists with coords) ---
        if snapshot.dark_intel and isinstance(snapshot.dark_intel, dict):
            dark_items = []
            for _dk, _dv in snapshot.dark_intel.items():
                if isinstance(_dv, list):
                    dark_items.extend([i for i in _dv[:20] if isinstance(i, dict)])
            dark_with_coords = [d for d in dark_items if self._extract_coords(d)]
            if dark_with_coords:
                d = random.choice(dark_with_coords)
                lat, lng = self._extract_coords(d)
                moves.append(CameraMove(
                    lat=lat, lng=lng, zoom=7.0,
                    style="SATELLITE",
                    label="Dark web intelligence",
                    domain="cyber",
                    severity="error",
                    dwell_seconds=5.0,
                    highlight_radius=100_000,
                    highlight_color="#8800ff",
                    toast_message="Dark web activity detected — analyzing origin",
                ))

        # --- Mine snapshot.raw for ANY remaining datasets with coordinates ---
        raw_extra_moves = self._mine_raw_for_coords(snapshot, max_extra=3)
        moves.extend(raw_extra_moves)

        # --- LiveUAMap — active conflict events ---
        if snapshot.liveuamap:
            lua_with_coords = [e for e in snapshot.liveuamap if self._extract_coords(e)]
            if lua_with_coords:
                sampled = random.sample(lua_with_coords, min(2, len(lua_with_coords)))
                for ev in sampled:
                    lat, lng = self._extract_coords(ev)
                    title = ev.get("title", ev.get("description", "conflict event"))[:50]
                    moves.append(CameraMove(
                        lat=lat, lng=lng, zoom=9.0, style="SATELLITE",
                        label=f"LiveUA: {title}", domain="military",
                        severity="warning", dwell_seconds=4.0,
                        highlight_radius=50_000, highlight_color="#ff4400",
                        toast_message=f"LiveUAMap conflict event — {title}",
                    ))

        # --- Emergency squawks (7500/7600/7700) ---
        if snapshot.emergency_squawks:
            for sq in snapshot.emergency_squawks[:2]:
                coords = self._extract_coords(sq)
                if coords:
                    lat, lng = coords
                    code = sq.get("squawk", sq.get("code", "7700"))
                    callsign = sq.get("callsign", sq.get("flight", "unknown"))
                    moves.append(CameraMove(
                        lat=lat, lng=lng, zoom=10.0, style="SATELLITE",
                        label=f"SQUAWK {code}: {callsign}", domain="military",
                        severity="critical", dwell_seconds=5.0,
                        highlight_radius=80_000, highlight_color="#ff0000",
                        toast_message=f"Emergency squawk {code} — {callsign}",
                    ))

        # --- Radiation monitors ---
        if snapshot.radiation_monitors:
            rad_with_coords = [r for r in snapshot.radiation_monitors if self._extract_coords(r)]
            if rad_with_coords:
                r = random.choice(rad_with_coords)
                lat, lng = self._extract_coords(r)
                name = r.get("name", r.get("station", "radiation monitor"))
                moves.append(CameraMove(
                    lat=lat, lng=lng, zoom=10.0, style="SATELLITE",
                    label=f"Radiation: {name}", domain="nuclear",
                    severity="warning", dwell_seconds=4.0,
                    highlight_radius=50_000, highlight_color="#ffff00",
                    toast_message=f"Radiation monitoring — {name}",
                ))

        # --- GDACS disasters ---
        if snapshot.gdacs_disasters:
            gdacs_with_coords = [g for g in snapshot.gdacs_disasters if self._extract_coords(g)]
            if gdacs_with_coords:
                for dis in random.sample(gdacs_with_coords, min(2, len(gdacs_with_coords))):
                    lat, lng = self._extract_coords(dis)
                    dtype = dis.get("type", dis.get("eventtype", "disaster"))
                    title = dis.get("title", dis.get("name", dtype))[:50]
                    moves.append(CameraMove(
                        lat=lat, lng=lng, zoom=7.0, style="SATELLITE",
                        label=f"Disaster: {title}", domain="environmental",
                        severity="error", dwell_seconds=4.0,
                        highlight_radius=150_000, highlight_color="#ff6600",
                        toast_message=f"GDACS disaster — {title}",
                    ))

        # --- EONET events (NASA natural events) ---
        if snapshot.eonet_events:
            eonet_with_coords = [e for e in snapshot.eonet_events if self._extract_coords(e)]
            if eonet_with_coords:
                ev = random.choice(eonet_with_coords)
                lat, lng = self._extract_coords(ev)
                title = ev.get("title", "natural event")[:50]
                moves.append(CameraMove(
                    lat=lat, lng=lng, zoom=7.0, style="SATELLITE",
                    label=f"EONET: {title}", domain="environmental",
                    severity="warning", dwell_seconds=3.0,
                    toast_message=f"NASA EONET event — {title}",
                ))

        # --- NOTAMs / TFR (airspace restrictions) ---
        if snapshot.notams_tfr:
            notam_with_coords = [n for n in snapshot.notams_tfr if self._extract_coords(n)]
            if notam_with_coords:
                n = random.choice(notam_with_coords)
                lat, lng = self._extract_coords(n)
                desc = n.get("description", n.get("notam_id", "TFR"))[:40]
                moves.append(CameraMove(
                    lat=lat, lng=lng, zoom=9.0, style="SATELLITE",
                    label=f"NOTAM: {desc}", domain="military",
                    severity="info", dwell_seconds=3.0,
                    highlight_radius=80_000, highlight_color="#ff8800",
                    toast_message=f"Airspace restriction — {desc}",
                ))

        # --- Piracy incidents ---
        if snapshot.piracy:
            piracy_with_coords = [p for p in snapshot.piracy if self._extract_coords(p)]
            if piracy_with_coords:
                p = random.choice(piracy_with_coords)
                lat, lng = self._extract_coords(p)
                desc = p.get("description", p.get("title", "piracy incident"))[:40]
                moves.append(CameraMove(
                    lat=lat, lng=lng, zoom=8.0, style="SATELLITE",
                    label=f"Piracy: {desc}", domain="piracy",
                    severity="warning", dwell_seconds=4.0,
                    highlight_radius=100_000, highlight_color="#ff0066",
                    toast_message=f"Maritime piracy — {desc}",
                ))

        # --- ISS position ---
        if snapshot.iss_position and isinstance(snapshot.iss_position, dict):
            iss_lat = snapshot.iss_position.get("latitude") or snapshot.iss_position.get("lat")
            iss_lng = snapshot.iss_position.get("longitude") or snapshot.iss_position.get("lon")
            if iss_lat and iss_lng:
                try:
                    moves.append(CameraMove(
                        lat=float(iss_lat), lng=float(iss_lng), zoom=5.0,
                        style="SATELLITE",
                        label="ISS — International Space Station",
                        domain="space", severity="info", dwell_seconds=3.0,
                        highlight_radius=300_000, highlight_color="#00ccff",
                        toast_message="Tracking ISS ground position",
                    ))
                except (ValueError, TypeError):
                    pass

        # Sort by severity (critical > error > warning > info)
        severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
        moves.sort(key=lambda m: severity_order.get(m.severity, 3))

        return moves

    # ── Patrol mode — dynamic global sweep, never repeating ────────────

    def _generate_patrol_moves(self, count: int = 3, snapshot=None) -> List[CameraMove]:
        """
        Eye of God patrol: combines static waypoints with data-driven
        locations, shuffled and deduplicated so Sable never visits the
        same region twice in a session.
        """
        candidates: List[Dict[str, Any]] = []

        # 1. Static waypoints — shuffled, not sequential
        shuffled_wp = list(PATROL_WAYPOINTS)
        random.shuffle(shuffled_wp)
        candidates.extend(shuffled_wp)

        # 2. Data-driven waypoints from live snapshot
        if snapshot:
            data_wp = self._extract_data_waypoints(snapshot)
            random.shuffle(data_wp)
            # Interleave data waypoints with static for variety
            candidates = self._interleave(data_wp, candidates)

        # 3. Filter out already-visited cells
        novel_candidates = [c for c in candidates if self._is_novel_region(c["lat"], c["lng"])]

        # If we've visited everywhere, reset tracking and reshuffle
        if len(novel_candidates) < count:
            self._visited_cells.clear()
            self._save_visited_cells()
            novel_candidates = candidates

        moves: List[CameraMove] = []
        for wp in novel_candidates[:count]:
            style = "SATELLITE"
            moves.append(CameraMove(
                lat=wp["lat"], lng=wp["lng"], zoom=wp.get("zoom", 6.0),
                style=style,
                label=f"Patrol: {wp['name']}",
                domain="default",
                dwell_seconds=3.0,
                toast_message=f"Eye of God — scanning {wp['name']}",
                severity="info",
            ))
            # Mark as visited
            self._mark_visited(wp["lat"], wp["lng"])
        return moves

    # ── Eye of God — helper methods for global coverage ──────────────

    @staticmethod
    def _extract_coords(entity: Any) -> Optional[Tuple[float, float]]:
        """Universal coordinate extractor — handles every format we've seen."""
        if entity is None:
            return None
        if isinstance(entity, dict):
            # Try common lat/lng key names
            lat = entity.get("lat") or entity.get("latitude") or entity.get("y")
            lng = (entity.get("lng") or entity.get("lon") or entity.get("longitude")
                   or entity.get("x"))
            # Some datasets use 'coords': [lat, lng]
            if lat is None and "coords" in entity:
                coords_val = entity["coords"]
                if isinstance(coords_val, (list, tuple)) and len(coords_val) >= 2:
                    lat, lng = coords_val[0], coords_val[1]
            # Some datasets nest coords inside 'geometry' (GeoJSON)
            if lat is None and "geometry" in entity:
                geo = entity["geometry"]
                if isinstance(geo, dict):
                    coords = geo.get("coordinates", [])
                    if len(coords) >= 2:
                        lng, lat = coords[0], coords[1]  # GeoJSON is [lng, lat]
            # Some datasets nest inside 'location'
            if lat is None and "location" in entity:
                loc = entity["location"]
                if isinstance(loc, dict):
                    lat = loc.get("lat") or loc.get("latitude")
                    lng = loc.get("lng") or loc.get("lon") or loc.get("longitude")
            # Some datasets nest inside 'properties' (GeoJSON features)
            if lat is None and "properties" in entity:
                props = entity.get("properties", {})
                if isinstance(props, dict):
                    lat = props.get("lat") or props.get("latitude")
                    lng = props.get("lng") or props.get("lon") or props.get("longitude")
            if lat is not None and lng is not None:
                try:
                    flat, flng = float(lat), float(lng)
                    if -90 <= flat <= 90 and -180 <= flng <= 180:
                        return (flat, flng)
                except (ValueError, TypeError):
                    pass
        return None

    def _is_novel_region(self, lat: float, lng: float, grid_size: float = 5.0) -> bool:
        """Check if this region hasn't been visited recently."""
        cell = (int(lat / grid_size), int(lng / grid_size))
        return cell not in self._visited_cells

    def _mark_visited(self, lat: float, lng: float, grid_size: float = 5.0) -> None:
        """Mark a region as visited and persist to disk."""
        cell = (int(lat / grid_size), int(lng / grid_size))
        self._visited_cells.add(cell)
        self._save_visited_cells()

    # ── Visited cells persistence ────────────────────────────────────

    def _load_visited_cells(self) -> Set[Tuple[int, int]]:
        """Load visited cells from disk so Sable never revisits after restart."""
        try:
            if self._visited_cells_path.exists():
                data = _json.loads(self._visited_cells_path.read_text(encoding="utf-8"))
                cells = {(c[0], c[1]) for c in data if isinstance(c, (list, tuple)) and len(c) >= 2}
                logger.info("Loaded %d visited cells from disk", len(cells))
                return cells
        except Exception as e:
            logger.warning("Failed to load visited cells: %s", e)
        return set()

    def _save_visited_cells(self) -> None:
        """Persist visited cells to disk."""
        try:
            self._visited_cells_path.parent.mkdir(parents=True, exist_ok=True)
            data = [list(c) for c in self._visited_cells]
            self._visited_cells_path.write_text(
                _json.dumps(data), encoding="utf-8",
            )
        except Exception as e:
            logger.debug("Failed to save visited cells: %s", e)

    # ─────────────────────────────────────────────────────────────
    #  Adaptive move budget & prioritization
    # ─────────────────────────────────────────────────────────────

    _SEVERITY_ORDER = {"critical": 0, "error": 1, "warning": 2, "info": 3}

    def _calculate_move_budget(self, results: dict, snapshot) -> int:
        """Decide how many moves Sable should execute this cycle.

        The budget scales with threat level and the density of
        critical/error findings so the user retains camera freedom
        when the world is calm.

        Returns a move count between 3 and 15 (hard ceiling).
        """
        threat = results.get("threat_level", 0)

        # Base budget from threat level
        if threat <= 1:
            base = 4      # Quick scan — leave camera to user
        elif threat == 2:
            base = 6
        elif threat == 3:
            base = 8
        elif threat == 4:
            base = 10
        else:               # threat 5
            base = 12

        # Bonus for high-severity findings (max +3)
        critical_count = 0
        for alert_list in results.get("module_alerts", {}).values():
            if isinstance(alert_list, list):
                for a in alert_list:
                    sev = a.get("severity", "") if isinstance(a, dict) else getattr(a, "severity", "")
                    if sev in ("critical", "error"):
                        critical_count += 1

        bonus = min(3, critical_count)
        budget = base + bonus

        # Hard ceiling — never exceed self.max_moves (env override) or 15
        ceiling = min(self.max_moves, 15)
        budget = min(budget, ceiling)
        # Hard floor — at least 3 (open + 1 finding + close)
        budget = max(budget, 3)
        return budget

    def _prioritize_and_trim(self, moves: List["CameraMove"], budget: int) -> List["CameraMove"]:
        """Sort by severity and trim to *budget* moves.

        Keeps opening (first) and closing (last) moves fixed.
        Middle moves are sorted by severity (critical first) and
        only the top ones survive.
        """
        if len(moves) <= budget:
            return moves

        # Separate opening, middle, closing
        opening = moves[0]
        closing = moves[-1]
        middle = moves[1:-1]

        # Sort middle by severity (ascending = most critical first)
        middle.sort(key=lambda m: self._SEVERITY_ORDER.get(m.severity, 99))

        # Take the top (budget - 2) middle moves (2 reserved for open/close)
        allowed_middle = max(0, budget - 2)
        trimmed = [opening] + middle[:allowed_middle] + [closing]
        logger.info("Trimmed %d → %d moves (budget=%d)", len(moves), len(trimmed), budget)
        return trimmed

    def _pick_random_opening(self, snapshot) -> Tuple[float, float]:
        """Pick a random opening location — never the same as last time."""
        # Gather candidate openings from data
        candidates: List[Tuple[float, float]] = []

        # Use live data centroids
        for source in [snapshot.military_flights, snapshot.ships, snapshot.conflicts,
                       snapshot.earthquakes, snapshot.cctv, snapshot.open_cameras,
                       snapshot.liveuamap, snapshot.emergency_squawks,
                       snapshot.gdacs_disasters, snapshot.piracy]:
            if source:
                sample = random.sample(source, min(3, len(source)))
                for item in sample:
                    coords = self._extract_coords(item)
                    if coords:
                        candidates.append(coords)

        # Add some patrol waypoints for variety (always on land)
        patrol_sample = random.sample(PATROL_WAYPOINTS, min(5, len(PATROL_WAYPOINTS)))
        candidates.extend([(wp["lat"], wp["lng"]) for wp in patrol_sample])

        # Filter: not too close to last opening
        last = self._last_opening_coords
        far_enough = [c for c in candidates
                      if abs(c[0] - last[0]) > 15 or abs(c[1] - last[1]) > 15]
        if not far_enough:
            far_enough = candidates

        choice = random.choice(far_enough) if far_enough else (random.uniform(-40, 60), random.uniform(-120, 150))
        self._last_opening_coords = choice
        return choice

    def _pick_random_closing(self, snapshot, opening: Tuple[float, float]) -> Tuple[float, float]:
        """Pick a closing location on LAND — never leave the camera staring at ocean.

        Priority order:
          1. A real data point from the snapshot (city, event, facility, camera)
          2. A patrol waypoint (always named cities / chokepoints on land)
          3. A known major city as absolute fallback
        Always picks something far from *opening* for geographic variety.
        """
        # ── 1. Collect candidate locations from live data ──
        candidates: List[Tuple[float, float, str]] = []

        # Data sources most likely to be on land
        land_sources = [
            (snapshot.cctv, "CCTV"),
            (snapshot.conflicts, "Conflict"),
            (snapshot.nuclear_facilities, "Nuclear"),
            (snapshot.fires, "Fire"),
            (snapshot.earthquakes, "Earthquake"),
            (snapshot.internet_outages, "Outage"),
            (snapshot.cyber_threats, "Cyber"),
            (snapshot.news_feed, "News"),
            (snapshot.gdelt_events, "Event"),
            (snapshot.military_flights, "MilFlight"),
            (snapshot.open_cameras, "Camera"),
        ]
        for source, label in land_sources:
            if not source or not isinstance(source, list):
                continue
            sample = random.sample(source, min(3, len(source)))
            for item in sample:
                coords = self._extract_coords(item)
                if coords:
                    candidates.append((coords[0], coords[1], label))

        # ── 2. Add patrol waypoints (curated — always on land) ──
        for wp in PATROL_WAYPOINTS:
            candidates.append((wp["lat"], wp["lng"], wp["name"]))

        # ── 3. Filter: must be far enough from opening for variety ──
        far = [(c[0], c[1], c[2]) for c in candidates
               if abs(c[0] - opening[0]) > 10 or abs(c[1] - opening[1]) > 10]
        if not far:
            far = candidates

        # ── 4. Filter: reject obvious ocean coordinates ──
        on_land = [(c[0], c[1], c[2]) for c in far if self._likely_on_land(c[0], c[1])]
        if not on_land:
            on_land = far  # All candidates from patrol waypoints should be on land anyway

        if on_land:
            pick = random.choice(on_land)
            logger.debug("Closing move on land: %s (%.2f, %.2f)", pick[2], pick[0], pick[1])
            return (pick[0], pick[1])

        # ── 5. Absolute fallback — major world cities ──
        fallback_cities = [
            (51.5, -0.12),   # London
            (40.7, -74.0),   # New York
            (35.7, 139.7),   # Tokyo
            (48.85, 2.35),   # Paris
            (55.75, 37.6),   # Moscow
            (-33.9, 151.2),  # Sydney
            (19.4, -99.1),   # Mexico City
            (1.3, 103.8),    # Singapore
            (28.6, 77.2),    # New Delhi
            (-23.5, -46.6),  # São Paulo
        ]
        return random.choice(fallback_cities)

    @staticmethod
    def _likely_on_land(lat: float, lng: float) -> bool:
        """Quick heuristic: reject coordinates that are almost certainly ocean.

        This is NOT a full land/water mask — it's a cheap check for the most
        common ocean pitfalls (mid-Atlantic, mid-Pacific, Southern Ocean, etc.).
        False positives (saying "ocean" for coastal land) are acceptable because
        we have plenty of candidates; false negatives (saying "land" for ocean)
        are tolerable since patrol waypoints are curated anyway.
        """
        # Southern Ocean / Antarctica below -60
        if lat < -60:
            return False
        # Deep Arctic above 75 (mostly ice/water except a few islands)
        if lat > 75:
            return False
        # Mid-Pacific: lat -50..50, lng 150..(-120) → huge empty ocean
        if -50 < lat < 50 and (lng > 150 or lng < -130):
            # Allow Hawaii, NZ, Japan, Australia vicinities
            if 18 < lat < 25 and -162 < lng < -153:
                return True   # Hawaii
            if -47 < lat < -33 and 165 < lng < 179:
                return True   # New Zealand
            if -45 < lat < -10 and 112 < lng < 155:
                return True   # Australia
            if 24 < lat < 46 and 122 < lng < 150:
                return True   # Japan / Korea
            return False
        # Mid-Atlantic: lat -50..50, lng -40..-10 (except some islands)
        if -50 < lat < 0 and -40 < lng < -5:
            return False   # South Atlantic
        if 0 < lat < 50 and -40 < lng < -10:
            # Allow Azores / Cape Verde / Canaries vicinities
            if 27 < lat < 40 and -32 < lng < -13:
                return True  # Azores / Canaries
            return False
        # Indian Ocean core: lat -40..-5, lng 50..100
        if -40 < lat < -5 and 50 < lng < 100:
            return False
        return True

    def _extract_data_waypoints(self, snapshot) -> List[Dict[str, Any]]:
        """
        Mine ALL live data for geographic points to use as patrol targets.
        This is what makes Sable the Eye of God — it visits real data, not
        just the same 10 hardcoded coordinates.
        """
        waypoints: List[Dict[str, Any]] = []

        # Named datasets with coords
        dataset_configs = [
            (snapshot.cctv, "CCTV Camera", 12.0),
            (snapshot.conflicts, "Active Conflict", 8.0),
            (snapshot.carriers, "Carrier Group", 8.0),
            (snapshot.cyber_threats, "Cyber Threat", 6.0),
            (snapshot.ransomware, "Ransomware Activity", 7.0),
            (snapshot.gdelt_events, "GDELT Event", 6.0),
            (snapshot.news_feed, "News Event", 7.0),
            (snapshot.prediction_markets, "Prediction Signal", 5.0),
            # dark_intel is a Dict, handled separately below
            (snapshot.military_flights, "Military Flight", 9.0),
            (snapshot.ships, "Maritime Vessel", 9.0),
            (snapshot.earthquakes, "Seismic Event", 8.0),
            (snapshot.fires, "Fire Hotspot", 7.0),
            (snapshot.gps_jamming, "GPS Jamming", 8.0),
            (snapshot.internet_outages, "Internet Outage", 6.0),
            (snapshot.nuclear_facilities, "Nuclear Facility", 10.0),
            (snapshot.satellites, "Satellite Track", 5.0),
            (snapshot.flights, "Air Traffic", 8.0),
        ]

        for dataset, name_prefix, default_zoom in dataset_configs:
            if not dataset:
                continue
            # Sample up to 5 random items from each dataset
            sample_size = min(5, len(dataset))
            sample = random.sample(dataset, sample_size)
            for item in sample:
                coords = self._extract_coords(item)
                if coords:
                    item_name = (item.get("name") or item.get("title") or
                                 item.get("location") or item.get("country") or
                                 item.get("callsign") or name_prefix)
                    if len(str(item_name)) > 40:
                        item_name = str(item_name)[:37] + "..."
                    waypoints.append({
                        "name": f"{name_prefix}: {item_name}",
                        "lat": coords[0], "lng": coords[1],
                        "zoom": default_zoom,
                    })

        # Handle dark_intel dict separately
        if snapshot.dark_intel and isinstance(snapshot.dark_intel, dict):
            for _dk, _dv in snapshot.dark_intel.items():
                if isinstance(_dv, list):
                    for item in _dv[:5]:
                        if isinstance(item, dict):
                            coords = self._extract_coords(item)
                            if coords:
                                waypoints.append({
                                    "name": f"Dark Intel: {item.get('name', _dk)}",
                                    "lat": coords[0], "lng": coords[1],
                                    "zoom": 7.0,
                                })

        # Also mine snapshot.raw for any datasets we haven't covered
        if hasattr(snapshot, "raw") and isinstance(snapshot.raw, dict):
            known_keys = {
                "flights", "military", "ships", "satellites", "earthquakes",
                "fires", "gdelt", "cyber", "gps_jamming", "carriers",
                "conflicts", "dark_intel", "internet_outages", "nuclear",
                "ransomware", "prediction_markets", "news", "cctv",
            }
            for key, value in snapshot.raw.items():
                # Skip keys we already process
                if any(k in key.lower() for k in known_keys):
                    continue
                if isinstance(value, list) and len(value) > 0:
                    sample = random.sample(value, min(3, len(value)))
                    for item in sample:
                        if isinstance(item, dict):
                            coords = self._extract_coords(item)
                            if coords:
                                label = item.get("name") or item.get("title") or key
                                waypoints.append({
                                    "name": f"Data/{key}: {str(label)[:30]}",
                                    "lat": coords[0], "lng": coords[1],
                                    "zoom": 7.0,
                                })

        return waypoints

    def _mine_raw_for_coords(self, snapshot, max_extra: int = 3) -> List[CameraMove]:
        """
        Scan snapshot.raw for datasets NOT covered by the typed fields.
        This catches everything — radiation monitors, datacenters, fleet tracker,
        weather, etc. Sable sees it ALL.
        """
        moves: List[CameraMove] = []
        if not hasattr(snapshot, "raw") or not isinstance(snapshot.raw, dict):
            return moves

        covered_prefixes = {
            "flight", "military", "ship", "satellite", "earthquake", "fire",
            "gdelt", "cyber", "gps", "carrier", "conflict", "dark", "internet",
            "nuclear", "ransomware", "prediction", "news", "cctv", "economic",
        }

        for key, value in snapshot.raw.items():
            if any(p in key.lower() for p in covered_prefixes):
                continue
            if not isinstance(value, list) or len(value) == 0:
                continue

            items_with_coords = []
            for item in value[:50]:  # Don't scan huge datasets entirely
                if isinstance(item, dict):
                    coords = self._extract_coords(item)
                    if coords:
                        items_with_coords.append((item, coords))

            if items_with_coords:
                item, (lat, lng) = random.choice(items_with_coords)
                label = item.get("name") or item.get("title") or key
                if len(str(label)) > 40:
                    label = str(label)[:37] + "..."
                moves.append(CameraMove(
                    lat=lat, lng=lng, zoom=7.0,
                    style="SATELLITE",
                    label=f"Raw/{key}: {label}",
                    domain="default",
                    severity="info",
                    dwell_seconds=3.0,
                    toast_message=f"Scanning {key} — {label}",
                ))
                if len(moves) >= max_extra:
                    break

        return moves

    @staticmethod
    def _interleave(a: list, b: list) -> list:
        """Interleave two lists, putting items from 'a' first."""
        result = []
        ia, ib = 0, 0
        toggle = True
        while ia < len(a) or ib < len(b):
            if toggle and ia < len(a):
                result.append(a[ia])
                ia += 1
            elif ib < len(b):
                result.append(b[ib])
                ib += 1
            else:
                break
            toggle = not toggle
        # Append remaining
        result.extend(a[ia:])
        result.extend(b[ib:])
        return result

    # ── Execute the move sequence on the dashboard ───────────────────

    async def _execute_moves(self, moves: List[CameraMove]) -> int:
        """Execute a sequence of camera moves on the live dashboard."""
        executed = 0

        for i, move in enumerate(moves):
            try:
                # Build command sequence for this move
                commands: List[tuple] = []

                # Style change (only if different from current)
                if self.enable_styles and move.style != self._current_style:
                    commands.append(("setStyle", {"style": move.style}))
                    self._current_style = move.style

                # Fly to location
                commands.append(("flyTo", {
                    "lat": move.lat, "lng": move.lng,
                    "zoom": move.zoom,
                    "duration": 2000,  # 2s animation
                }))

                # Toast notification
                if self.enable_toasts and move.toast_message:
                    commands.append(("toast", {
                        "message": move.toast_message,
                        "severity": move.severity,
                    }))

                # Send as batch for low latency
                if len(commands) > 1:
                    await self.remote.sequence(commands)
                else:
                    cmd_type, payload = commands[0]
                    await self.remote._send(cmd_type, payload)

                # Highlight ring (sent separately after fly completes)
                if self.enable_highlights and move.highlight_radius > 0:
                    await asyncio.sleep(0.5)  # Let fly animation start
                    await self.remote.highlight(
                        lat=move.lat,
                        lng=move.lng,
                        radius=move.highlight_radius,
                        color=move.highlight_color,
                        label=move.label,
                        duration=int(move.dwell_seconds * 1000),
                    )

                self._current_lat = move.lat
                self._current_lng = move.lng
                self._total_moves += 1
                executed += 1

                self._move_history.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "lat": move.lat, "lng": move.lng,
                    "zoom": move.zoom, "style": move.style,
                    "label": move.label, "domain": move.domain,
                })

                logger.info(
                    "Camera move %d/%d: [%s] %s → (%.2f, %.2f) z=%.1f style=%s",
                    i + 1, len(moves), move.severity.upper(),
                    move.label, move.lat, move.lng, move.zoom, move.style,
                )

                # ── Live camera check — open a nearby stream if available ──
                await self._try_open_live_camera(move)

                # ── Operation diary — record what we're seeing ──
                await self._record_diary_entry(move)

                # ── Dossier capture — log structured intel ──
                await self._record_dossier_entry(move)

                # ── Push live intel event for ticker ──
                if move.domain and move.domain not in ("default", "threat_fusion") and move.label:
                    try:
                        sev = "info"
                        if move.severity in ("error", "critical"):
                            sev = "critical"
                        elif move.severity == "warning":
                            sev = "warning"
                        await self.remote.push_intel_event({
                            "id": f"live-{uuid.uuid4().hex[:8]}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "domain": move.domain,
                            "severity": sev,
                            "title": move.label,
                            "detail": move.toast_message or "",
                            "lat": move.lat,
                            "lng": move.lng,
                        })
                    except Exception:
                        pass  # Non-critical

                # Dwell — hold position before next move
                await asyncio.sleep(move.dwell_seconds)

                # Minimum delay between moves
                await asyncio.sleep(self.move_delay)

            except Exception as e:
                logger.warning("Camera move %d failed: %s", i + 1, e)
                await asyncio.sleep(1.0)

        return executed

    # ── Live camera integration ──────────────────────────────────────

    async def _try_open_live_camera(self, move) -> None:
        """Open a live camera/CCTV feed when the move carries a media_url.

        If the CameraMove has a `media_url` (CCTV or DOT camera), open it in the
        dashboard's LiveStreamViewer via WebSocket. Avoids showing the same URL
        twice in a single cycle and respects a 10s cooldown between opens.
        """
        url = getattr(move, "media_url", "") or ""
        if not url:
            return

        # Deduplicate
        if url in self._shown_live_cameras:
            return

        # Cooldown — don't spam the viewer
        now = time.time()
        if now - self._live_camera_cooldown < 10.0:
            return

        try:
            # Determine platform hint from URL
            platform = "cctv"
            if "youtube" in url:
                platform = "youtube"
            elif "dot" in url.lower() or "wsdot" in url.lower() or "caltrans" in url.lower():
                platform = "dot_camera"

            await self.remote.open_live_stream(
                name=move.label or "Camera Feed",
                url=url,
                lat=move.lat,
                lng=move.lng,
                platform=platform,
            )
            self._shown_live_cameras.add(url)
            self._live_camera_cooldown = now
            logger.info("Opened camera feed: %s → %s", move.label, url[:80])
        except Exception as e:
            logger.debug("Failed to open camera feed: %s", e)

    # ── Operation diary — automatic observation logging ──────────────

    async def _record_diary_entry(self, move) -> None:
        """Record what Sable is observing at this location in the operation diary."""
        # Only log medium+ severity or every 3rd move to avoid spam
        if move.severity == "info" and self._total_moves % 3 != 0:
            return

        severity_map = {
            "info": "info",
            "warning": "warning",
            "error": "critical",
            "critical": "critical",
            "success": "success",
        }
        diary_severity = severity_map.get(move.severity, "info")

        entry = {
            "title": move.label,
            "content": move.toast_message or move.label,
            "severity": diary_severity,
            "lat": move.lat,
            "lng": move.lng,
            "domain": move.domain,
            "tags": [move.domain, move.style],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "zoom": move.zoom,
        }
        self._diary_entries.append(entry)

        # Persist to disk via OperationDiary (survives restarts)
        try:
            self._op_diary.add_entry(
                title=entry["title"],
                content=entry["content"],
                severity=entry["severity"],
                lat=entry["lat"],
                lng=entry["lng"],
                domain=entry["domain"],
                tags=entry["tags"],
                cycle=self._cycle_count,
            )
        except Exception as e:
            logger.debug("Diary disk persist failed: %s", e)

        # Push live to dashboard via WebSocket
        try:
            await self.remote.add_diary_entry(
                title=entry["title"],
                content=entry["content"],
                severity=entry["severity"],
                lat=entry["lat"],
                lng=entry["lng"],
                domain=entry["domain"],
                tags=entry["tags"],
            )
        except Exception as e:
            logger.debug("Diary entry send failed: %s", e)

    # ── Dossier capture — structured intel collection ────────────────

    async def _record_dossier_entry(self, move) -> None:
        """Capture structured dossier data for critical/warning findings."""
        # Only capture for warning+ severity — dossier is for notable items
        if move.severity not in ("warning", "error", "critical"):
            return

        # Determine category from domain
        domain_to_cat = {
            "military": "aircraft",
            "kill_chain": "attack",
            "force_projection": "ship",
            "counter_surveillance": "surveillance",
            "cyber": "cyber",
            "sigint_ew": "surveillance",
            "infrastructure": "infrastructure",
            "narrative_warfare": "narrative",
            "nuclear": "nuclear",
            "ships": "ship",
            "piracy": "ship",
            "fires": "environmental",
            "environmental": "environmental",
            "space": "satellite",
            "satellites": "satellite",
            "finint": "financial",
        }
        category = domain_to_cat.get(move.domain, "custom")

        entry = {
            "category": category,
            "title": move.label,
            "data": {
                "description": move.toast_message or move.label,
                "severity": move.severity,
                "domain": move.domain,
                "zoom": move.zoom,
                "style": move.style,
                "highlight_radius": move.highlight_radius,
                "move_index": self._total_moves,
                "cycle": self._cycle_count,
            },
            "severity": move.severity,
            "lat": move.lat,
            "lng": move.lng,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._dossier_entries.append(entry)

        # Persist to disk via OperationDiary (survives restarts)
        try:
            self._op_diary.add_dossier(
                category=entry["category"],
                title=entry["title"],
                severity=entry["severity"],
                lat=entry["lat"],
                lng=entry["lng"],
                data=entry["data"],
            )
        except Exception as e:
            logger.debug("Dossier disk persist failed: %s", e)

        # Push live to dashboard via WebSocket
        try:
            await self.remote.add_dossier_entry(
                category=entry["category"],
                title=entry["title"],
                data=entry["data"],
                severity=entry["severity"],
                lat=entry["lat"],
                lng=entry["lng"],
            )
        except Exception as e:
            logger.debug("Dossier entry send failed: %s", e)

    # ── Helper — cluster entities by geographic grid ─────────────────

    def _cluster_entities(
        self,
        entities: list,
        label_prefix: str = "Cluster",
        grid_size: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """Simple grid-based clustering: group entities into grid cells."""
        grid: Dict[Tuple[int, int], List[Dict]] = {}
        for e in entities:
            lat = e.get("lat", e.get("latitude"))
            lng = e.get("lon", e.get("lng", e.get("longitude")))
            if lat is None or lng is None:
                continue
            try:
                flat, flng = float(lat), float(lng)
            except (ValueError, TypeError):
                continue
            cell = (int(flat / grid_size), int(flng / grid_size))
            grid.setdefault(cell, []).append({"lat": flat, "lng": flng})

        clusters = []
        for cell, items in grid.items():
            if len(items) >= 2:
                avg_lat = sum(i["lat"] for i in items) / len(items)
                avg_lng = sum(i["lng"] for i in items) / len(items)
                clusters.append({
                    "lat": avg_lat, "lng": avg_lng,
                    "count": len(items),
                    "label": f"{label_prefix} ({len(items)} entities)",
                })

        # Sort by count descending
        clusters.sort(key=lambda c: c["count"], reverse=True)
        return clusters

    # ── Build intelligence brief for dashboard overlay ───────────────

    def _build_intel_brief(
        self,
        results: Dict[str, Any],
        snapshot,
        moves: list,
        module_alerts: Optional[Dict[str, list]] = None,
    ) -> dict:
        """
        Construct a full intelligence brief from analysis results, snapshot,
        and planned camera moves. This brief feeds the IntelOverlay panel
        in the dashboard — threat matrix, wargame scenarios, intel events,
        and sensor stats.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        threat_level = results.get("threat_level", 0)
        threat_name = results.get("threat_name", "UNKNOWN")

        # ── Domain threat assessment ──
        domains = self._assess_domains(results, snapshot)

        # ── Wargame scenarios ──
        wargame_scenarios = self._extract_wargame_scenarios(results)

        # ── Intel events from moves + snapshot anomalies ──
        intel_events = self._extract_intel_events(results, snapshot, moves, module_alerts)

        # ── Sensor stats ──
        stats = {
            "militaryFlights": len(getattr(snapshot, "military_flights", []) or []),
            "activeConflicts": len(getattr(snapshot, "conflicts", []) or []),
            "gpsJamming": len(getattr(snapshot, "gps_jamming", []) or []),
            "cyberThreats": len(getattr(snapshot, "cyber_threats", []) or []),
            "nuclearFacilities": len(getattr(snapshot, "nuclear_facilities", []) or []),
            "earthquakes": len(getattr(snapshot, "earthquakes", []) or []),
            "fires": len(getattr(snapshot, "fires", []) or []),
            "ransomware": len(getattr(snapshot, "ransomware", []) or []),
            "ships": len(getattr(snapshot, "ships", []) or []),
            "carriers": len(getattr(snapshot, "carriers", []) or []),
            "internetOutages": len(getattr(snapshot, "internet_outages", []) or []),
        }

        return {
            "timestamp": now_iso,
            "overallThreat": threat_level,
            "threatName": threat_name,
            "cycleNumber": self._cycle_count,
            "domains": domains,
            "wargameScenarios": wargame_scenarios,
            "intelEvents": intel_events,
            "stats": stats,
        }

    def _assess_domains(self, results: Dict[str, Any], snapshot) -> list:
        """Build per-domain threat levels from analysis results."""
        domain_data = []

        # Helpers
        def _count(attr, default=0):
            lst = getattr(snapshot, attr, None)
            return len(lst) if lst else default

        def _level(count, thresholds=(2, 5, 15, 30)):
            """Map count to 0-5 threat level using thresholds."""
            if count == 0: return 0.0
            for i, t in enumerate(thresholds):
                if count <= t:
                    return (i + 1) * (count / t)
            return min(5.0, 4.0 + count / (thresholds[-1] * 2))

        def _trend(module_key):
            """Determine trend based on whether module reported changes."""
            val = results.get(module_key, 0)
            if isinstance(val, (int, float)):
                if val > 10: return "up"
                if val > 0: return "stable"
            return "stable"

        mil_count = _count("military_flights")
        mil_level = min(5.0, _level(mil_count, (10, 30, 80, 150)))
        # Boost if force projection or kill chain is active
        if results.get("kill_chain_alerts", 0) > 0:
            mil_level = min(5.0, mil_level + 1.0)
        domain_data.append({
            "domain": "military",
            "level": round(mil_level, 1),
            "label": self._level_label(mil_level),
            "color": self._level_color(mil_level),
            "activeThreats": mil_count + results.get("kill_chain_alerts", 0),
            "trend": "up" if results.get("kill_chain_alerts", 0) > 0 else _trend("force_projection_movements"),
            "details": f"{mil_count} military flights tracked",
        })

        cyber_count = _count("cyber_threats") + len(getattr(snapshot, "ransomware", []) or [])
        cyber_level = _level(cyber_count, (3, 10, 25, 50))
        domain_data.append({
            "domain": "cyber",
            "level": round(cyber_level, 1),
            "label": self._level_label(cyber_level),
            "color": self._level_color(cyber_level),
            "activeThreats": cyber_count,
            "trend": _trend("cyber_severity"),
        })

        conflict_count = _count("conflicts")
        conflict_level = _level(conflict_count, (2, 8, 20, 40))
        domain_data.append({
            "domain": "humanitarian",
            "level": round(conflict_level, 1),
            "label": self._level_label(conflict_level),
            "color": self._level_color(conflict_level),
            "activeThreats": conflict_count,
            "trend": _trend("active_conflicts"),
        })

        jam_count = _count("gps_jamming")
        jam_level = _level(jam_count, (2, 5, 10, 20))
        domain_data.append({
            "domain": "sigint_ew",
            "level": round(jam_level, 1),
            "label": self._level_label(jam_level),
            "color": self._level_color(jam_level),
            "activeThreats": jam_count,
            "trend": "up" if jam_count > 5 else "stable",
        })

        nuke_count = _count("nuclear_facilities")
        nuke_level = _level(nuke_count, (1, 3, 5, 10))
        domain_data.append({
            "domain": "nuclear",
            "level": round(nuke_level, 1),
            "label": self._level_label(nuke_level),
            "color": self._level_color(nuke_level),
            "activeThreats": nuke_count,
            "trend": "stable",
        })

        ship_count = _count("ships")
        carrier_count = _count("carriers")
        maritime_level = _level(carrier_count, (1, 3, 5, 8))
        domain_data.append({
            "domain": "maritime",
            "level": round(maritime_level, 1),
            "label": self._level_label(maritime_level),
            "color": self._level_color(maritime_level),
            "activeThreats": carrier_count,
            "trend": "up" if carrier_count > 3 else "stable",
            "details": f"{ship_count} vessels, {carrier_count} carriers",
        })

        infra_count = _count("internet_outages") + results.get("infrastructure_alerts", 0)
        infra_level = _level(infra_count, (2, 5, 10, 20))
        domain_data.append({
            "domain": "infrastructure",
            "level": round(infra_level, 1),
            "label": self._level_label(infra_level),
            "color": self._level_color(infra_level),
            "activeThreats": infra_count,
            "trend": _trend("infrastructure_alerts"),
        })

        fire_count = _count("fires")
        eq_count = _count("earthquakes")
        env_count = fire_count + eq_count
        env_level = _level(env_count, (5, 15, 30, 60))
        domain_data.append({
            "domain": "energy",
            "level": round(env_level, 1),
            "label": self._level_label(env_level),
            "color": self._level_color(env_level),
            "activeThreats": env_count,
            "trend": "up" if fire_count > 20 else "stable",
            "details": f"{fire_count} fires, {eq_count} earthquakes",
        })

        # Wargaming domain — based on active scenarios
        wg_count = results.get("wargame_live_scenarios", 0)
        wg_risks = results.get("wargame_escalation_risks", [])
        wg_level = 0.0
        for r in wg_risks:
            risk = r.get("risk", "low")
            if risk == "extreme": wg_level = max(wg_level, 5.0)
            elif risk == "high": wg_level = max(wg_level, 4.0)
            elif risk == "moderate": wg_level = max(wg_level, 2.5)
            else: wg_level = max(wg_level, 1.0)
        domain_data.append({
            "domain": "wargaming",
            "level": round(wg_level, 1),
            "label": self._level_label(wg_level),
            "color": self._level_color(wg_level),
            "activeThreats": wg_count,
            "trend": "up" if wg_count > 0 else "stable",
        })

        # Sort by threat level descending
        domain_data.sort(key=lambda d: d["level"], reverse=True)
        return domain_data

    def _extract_wargame_scenarios(self, results: Dict[str, Any]) -> list:
        """Extract wargame scenario data from results for the overlay."""
        scenarios = []
        wg_risks = results.get("wargame_escalation_risks", [])
        wg_details = results.get("wargame_scenario_details", [])

        for i, risk_info in enumerate(wg_risks[:4]):
            trigger = risk_info.get("trigger", "Unknown scenario")
            region = risk_info.get("region", "Unknown")
            risk = risk_info.get("risk", "moderate")
            score = risk_info.get("severity_score", 0.0)

            # Try to get full consequence data from wargame_scenario_details
            consequences = []
            if i < len(wg_details):
                detail = wg_details[i]
                for c in detail.get("consequences", []):
                    consequences.append({
                        "order": c.get("order", 1),
                        "domain": c.get("domain", "unknown"),
                        "description": c.get("description", c.get("desc", "")),
                        "probability": c.get("probability", c.get("prob", 0.5)),
                        "severity": c.get("severity", c.get("sev", "medium")),
                        "timeHorizon": c.get("time_horizon", c.get("time", "days")),
                    })

            scenarios.append({
                "scenarioId": f"wg-{self._cycle_count}-{i}",
                "trigger": trigger,
                "region": region,
                "escalationRisk": risk,
                "severityScore": score,
                "consequences": consequences,
                "summary": risk_info.get("summary", ""),
            })

        return scenarios

    def _extract_intel_events(
        self,
        results: Dict[str, Any],
        snapshot,
        moves: list,
        module_alerts: Optional[Dict[str, list]] = None,
    ) -> list:
        """Extract intelligence events from analysis results and snapshot for the live feed."""
        events = []
        now_iso = datetime.now(timezone.utc).isoformat()

        # From camera moves — each significant move becomes an event
        for move in moves:
            if move.domain and move.domain != "default" and move.label:
                sev = "info"
                if move.severity in ("error", "critical"):
                    sev = "critical"
                elif move.severity == "warning":
                    sev = "warning"
                events.append({
                    "id": f"ev-{uuid.uuid4().hex[:8]}",
                    "timestamp": now_iso,
                    "domain": move.domain,
                    "severity": sev,
                    "title": move.label,
                    "detail": move.toast_message or "",
                    "lat": move.lat,
                    "lng": move.lng,
                })

        # From module alerts — handle both dicts and dataclass objects
        if module_alerts:
            for module_name, alerts in module_alerts.items():
                for alert in (alerts or [])[:3]:
                    _g = (lambda a, k, d=None: a.get(k, d) if isinstance(a, dict)
                          else getattr(a, k, d))
                    # Coords: try lat/lng attributes, then coordinates tuple
                    a_lat = float(_g(alert, "lat", 0) or 0)
                    a_lng = float(_g(alert, "lng", 0) or _g(alert, "lon", 0) or 0)
                    if a_lat == 0 and a_lng == 0:
                        coords = _g(alert, "coordinates", None)
                        if coords and isinstance(coords, (tuple, list)) and len(coords) >= 2:
                            a_lat, a_lng = float(coords[0]), float(coords[1])
                    events.append({
                        "id": f"ev-{uuid.uuid4().hex[:8]}",
                        "timestamp": now_iso,
                        "domain": module_name,
                        "severity": _g(alert, "severity", "info"),
                        "title": _g(alert, "title", None) or _g(alert, "message", module_name),
                        "detail": _g(alert, "detail", None) or _g(alert, "description", ""),
                        "lat": a_lat,
                        "lng": a_lng,
                    })

        # From snapshot — significant items
        if snapshot.gps_jamming:
            for zone in snapshot.gps_jamming[:3]:
                lat = zone.get("lat", zone.get("latitude", 0))
                lng = zone.get("lon", zone.get("lng", zone.get("longitude", 0)))
                events.append({
                    "id": f"ev-jam-{uuid.uuid4().hex[:6]}",
                    "timestamp": now_iso,
                    "domain": "sigint_ew",
                    "severity": "warning",
                    "title": f"GPS Jamming Detected",
                    "detail": f"Active jamming zone near {lat:.1f}°, {lng:.1f}°",
                    "lat": float(lat) if lat else 0,
                    "lng": float(lng) if lng else 0,
                })

        if snapshot.earthquakes:
            for eq in snapshot.earthquakes[:2]:
                mag = eq.get("magnitude", eq.get("mag", 0))
                if mag and float(mag) >= 4.5:
                    events.append({
                        "id": f"ev-eq-{uuid.uuid4().hex[:6]}",
                        "timestamp": now_iso,
                        "domain": "energy",
                        "severity": "warning" if float(mag) < 6.0 else "critical",
                        "title": f"Earthquake M{mag}",
                        "detail": eq.get("place", eq.get("title", "")),
                        "lat": float(eq.get("lat", eq.get("latitude", 0))),
                        "lng": float(eq.get("lon", eq.get("lng", eq.get("longitude", 0)))),
                    })

        if snapshot.conflicts:
            for c in snapshot.conflicts[:2]:
                events.append({
                    "id": f"ev-conf-{uuid.uuid4().hex[:6]}",
                    "timestamp": now_iso,
                    "domain": "humanitarian",
                    "severity": "critical",
                    "title": f"Active Conflict: {c.get('country', c.get('name', 'Unknown'))}",
                    "detail": c.get("notes", c.get("description", "")),
                    "lat": float(c.get("lat", c.get("latitude", 0))),
                    "lng": float(c.get("lon", c.get("lng", c.get("longitude", 0)))),
                })

        if getattr(snapshot, "ransomware", None):
            for r in snapshot.ransomware[:2]:
                events.append({
                    "id": f"ev-ransom-{uuid.uuid4().hex[:6]}",
                    "timestamp": now_iso,
                    "domain": "cyber",
                    "severity": "critical",
                    "title": f"Ransomware: {r.get('group_name', r.get('name', 'Unknown'))}",
                    "detail": r.get("victim", r.get("description", "")),
                })

        # Deduplicate by title (keep first)
        seen_titles: Set[str] = set()
        unique_events = []
        for ev in events:
            if ev["title"] not in seen_titles:
                seen_titles.add(ev["title"])
                unique_events.append(ev)

        # Sort by severity (critical first), limit to 50
        sev_order = {"critical": 0, "warning": 1, "info": 2}
        unique_events.sort(key=lambda e: sev_order.get(e.get("severity", "info"), 2))
        unique_events = unique_events[:50]

        # ── Enrich: video URLs from news feeds ──
        unique_events = self._enrich_with_videos(unique_events, snapshot)

        # ── Enrich: missile/kinetic trajectory detection ──
        unique_events = self._extract_trajectories(unique_events, snapshot)

        # ── Enrich: story tracking / follow-up system ──
        unique_events = self._update_story_tracking(unique_events)

        return unique_events

    @staticmethod
    def _level_label(n: float) -> str:
        if n <= 0: return "MINIMAL"
        if n <= 1: return "LOW"
        if n <= 2: return "GUARDED"
        if n <= 3: return "ELEVATED"
        if n <= 4: return "HIGH"
        return "SEVERE"

    @staticmethod
    def _level_color(n: float) -> str:
        if n <= 1: return "#22c55e"
        if n <= 2: return "#3b82f6"
        if n <= 3: return "#eab308"
        if n <= 4: return "#f97316"
        return "#ef4444"

    # ── Helper — match scenario trigger text to coordinates ──────────

    @staticmethod
    def _match_scenario_to_coords(trigger: str) -> Optional[Tuple[float, float]]:
        """Map wargame scenario trigger text to approximate coordinates."""
        trigger_lower = trigger.lower()
        mappings = [
            (["hormuz", "iran", "persian gulf"],    (26.6, 56.3)),
            (["taiwan", "adiz", "china strait"],    (24.0, 121.0)),
            (["baltic", "cable", "nord stream"],    (58.0, 20.0)),
            (["suez", "canal"],                     (30.5, 32.3)),
            (["south china sea", "spratly"],        (12.0, 114.5)),
            (["houthi", "red sea", "bab"],          (13.0, 43.0)),
            (["korea", "dmz", "pyongyang"],         (38.0, 127.0)),
            (["ukraine", "crimea", "black sea"],    (44.5, 34.0)),
            (["arctic", "giuk"],                    (64.0, -20.0)),
            (["multi-domain", "escalation"],        (35.0, 45.0)),
        ]
        for keywords, coords in mappings:
            if any(kw in trigger_lower for kw in keywords):
                return coords
        return None

    # ── Story tracking — persistent follow-up system ───────────────

    def _load_tracked_stories(self):
        """Load tracked stories from disk."""
        try:
            if self._tracked_stories_path.exists():
                with open(self._tracked_stories_path, "r") as f:
                    self._tracked_stories = _json.load(f)
                    # Expire stories older than 72h
                    now = datetime.now(timezone.utc).isoformat()
                    expired = []
                    for sid, s in self._tracked_stories.items():
                        last = s.get("last_seen", "")
                        if last and (datetime.fromisoformat(last.replace("Z", "+00:00")) <
                                     datetime.now(timezone.utc) - timedelta(hours=72)):
                            expired.append(sid)
                    for sid in expired:
                        del self._tracked_stories[sid]
        except Exception:
            self._tracked_stories = {}

    def _save_tracked_stories(self):
        """Persist tracked stories to disk."""
        try:
            self._tracked_stories_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._tracked_stories_path, "w") as f:
                _json.dump(self._tracked_stories, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Could not save tracked stories: %s", e)

    def _update_story_tracking(self, events: list) -> list:
        """Register high-severity events as tracked stories and mark follow-ups.

        Returns:
            Updated events list with followUp/storyId fields set.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        for ev in events:
            if ev.get("severity") not in ("critical", "warning"):
                continue

            title_key = ev["title"].lower().strip()
            # Generate a stable story_id from title keywords
            words = [w for w in title_key.split() if len(w) > 3][:5]
            story_id = "story-" + "-".join(words) if words else f"story-{uuid.uuid4().hex[:8]}"

            # Check if we're already tracking a similar story
            matched_sid = None
            for sid, story in self._tracked_stories.items():
                # Match if >50% of keywords overlap
                story_words = set(story.get("keywords", []))
                overlap = len(set(words) & story_words)
                if overlap >= max(1, len(story_words) * 0.5):
                    matched_sid = sid
                    break

            if matched_sid:
                # Follow-up: update the existing story
                story = self._tracked_stories[matched_sid]
                story["last_seen"] = now_iso
                story["cycle_count"] = story.get("cycle_count", 1) + 1
                story["last_detail"] = ev.get("detail", "")
                ev["followUp"] = True
                ev["storyId"] = matched_sid
            else:
                # New story — start tracking
                self._tracked_stories[story_id] = {
                    "title": ev["title"],
                    "domain": ev.get("domain", ""),
                    "severity": ev.get("severity", "info"),
                    "first_seen": now_iso,
                    "last_seen": now_iso,
                    "cycle_count": 1,
                    "lat": ev.get("lat", 0),
                    "lng": ev.get("lng", 0),
                    "keywords": words,
                    "last_detail": ev.get("detail", ""),
                }
                ev["storyId"] = story_id

        # Trim to most recent 100 stories
        if len(self._tracked_stories) > 100:
            sorted_stories = sorted(
                self._tracked_stories.items(),
                key=lambda x: x[1].get("last_seen", ""),
                reverse=True,
            )
            self._tracked_stories = dict(sorted_stories[:100])

        self._save_tracked_stories()
        return events

    # ── Video enrichment — find related video URLs from news feeds ───

    def _enrich_with_videos(self, events: list, snapshot) -> list:
        """Try to attach related video URLs to intel events from news feeds.

        Phase 1 (passive): Scans snapshot.news_feed and snapshot.gdelt_events
        for items with video URLs that match event keywords / coordinates.
        Phase 2 (active web research) runs separately via _enrich_with_web_research().
        """
        if not events:
            return events

        # Build a pool of video-bearing items from news
        video_pool: List[Dict[str, Any]] = []
        for source in [
            getattr(snapshot, "news_feed", []) or [],
            getattr(snapshot, "gdelt_events", []) or [],
            getattr(snapshot, "trending", []) or [],
        ]:
            for item in source:
                url = (item.get("video_url") or item.get("media_url")
                       or item.get("url") or item.get("link") or "")
                if not url:
                    continue
                # Keep items that look like they have video content
                is_video = any(k in url.lower() for k in
                               ["youtube", "youtu.be", "video", "embed",
                                "watch", "vimeo", "rumble", "bitchute",
                                "tiktok", "twitter.com/i/status",
                                "reuters.com/video", "bbc.com/news/av"])
                has_source = bool(url)
                title = (item.get("title") or item.get("headline")
                         or item.get("name") or "")
                if is_video or has_source:
                    video_pool.append({
                        "url": url,
                        "is_video": is_video,
                        "title": title.lower(),
                        "lat": float(item.get("lat", 0) or 0),
                        "lng": float(item.get("lng", item.get("lon", 0)) or 0),
                    })

        if not video_pool:
            return events

        # Match events to videos by keyword overlap or proximity
        for ev in events:
            if ev.get("videoUrl"):
                continue  # Already has one
            ev_words = set(ev.get("title", "").lower().split())
            ev_lat = ev.get("lat", 0)
            ev_lng = ev.get("lng", 0)
            best_match = None
            best_score = 0

            for vp in video_pool:
                score = 0
                vp_words = set(vp["title"].split())
                overlap = len(ev_words & vp_words)
                score += overlap * 2

                # Proximity bonus (within ~2 degrees)
                if ev_lat and ev_lng and vp["lat"] and vp["lng"]:
                    dist = abs(ev_lat - vp["lat"]) + abs(ev_lng - vp["lng"])
                    if dist < 2:
                        score += 3
                    elif dist < 5:
                        score += 1

                if score > best_score:
                    best_score = score
                    best_match = vp

            if best_match and best_score >= 2:
                if best_match["is_video"]:
                    ev["videoUrl"] = best_match["url"]
                else:
                    ev["sourceUrl"] = best_match["url"]

        return events

    # ── Active web research — YouTube + news search for intel events ─

    # Rate limits: max searches per cycle to protect API quotas
    _MAX_VIDEO_SEARCHES_PER_CYCLE = 5
    _MAX_NEWS_SEARCHES_PER_CYCLE = 5

    async def _enrich_with_web_research(self, events: list) -> list:
        """Phase 2: Active web research for critical/warning intel events.

        For high-severity events still missing video/source URLs, actively
        search YouTube (via YouTubeSkill) and news (via NewsReaderSkill)
        to find relevant content. This gives the intel brief richer,
        real-world context that passive snapshot matching can't provide.

        Rate-limited to avoid API quota exhaustion.
        """
        if not self._video_searcher and not self._news_searcher:
            return events

        # Only research events that need enrichment (critical first)
        needs_video = [e for e in events if not e.get("videoUrl")
                       and e.get("severity") in ("critical", "warning")]
        needs_source = [e for e in events if not e.get("sourceUrl")
                        and not e.get("videoUrl")
                        and e.get("severity") in ("critical", "warning")]

        video_searches = 0
        news_searches = 0

        # ── YouTube video search ──
        if self._video_searcher and needs_video:
            for ev in needs_video:
                if video_searches >= self._MAX_VIDEO_SEARCHES_PER_CYCLE:
                    break
                query = self._build_search_query(ev)
                if not query:
                    continue
                try:
                    results = await self._video_searcher(query, 3)
                    video_searches += 1
                    if results:
                        # Pick the best match (first result is usually most relevant)
                        best = results[0]
                        ev["videoUrl"] = best.get("url", "")
                        if not ev.get("sourceUrl"):
                            ev["sourceUrl"] = best.get("url", "")
                        logger.info("Web research: YouTube match for '%s' → %s",
                                    ev.get("title", "")[:50], best.get("url", "")[:80])
                except Exception as e:
                    logger.debug("YouTube search failed for '%s': %s",
                                 ev.get("title", "")[:40], e)

        # ── News article search ──
        if self._news_searcher and needs_source:
            for ev in needs_source:
                if news_searches >= self._MAX_NEWS_SEARCHES_PER_CYCLE:
                    break
                if ev.get("sourceUrl"):
                    continue  # Already enriched by video search
                query = self._build_search_query(ev)
                if not query:
                    continue
                try:
                    articles = await self._news_searcher(query, 3)
                    news_searches += 1
                    if articles:
                        best = articles[0]
                        ev["sourceUrl"] = best.get("link", best.get("url", ""))
                        logger.info("Web research: News match for '%s' → %s",
                                    ev.get("title", "")[:50],
                                    best.get("link", "")[:80])
                except Exception as e:
                    logger.debug("News search failed for '%s': %s",
                                 ev.get("title", "")[:40], e)

        if video_searches or news_searches:
            logger.info("Web research enrichment: %d YouTube + %d news searches",
                        video_searches, news_searches)

        return events

    @staticmethod
    def _build_search_query(event: dict) -> str:
        """Build an effective search query from an intel event.

        Combines title keywords with domain context to produce
        a focused query that finds relevant video/news content.
        """
        title = event.get("title", "")
        domain = event.get("domain", "")

        # Strip common generic prefixes
        for prefix in ["alert:", "warning:", "critical:", "⚠", "🔴", "⭐"]:
            title = title.replace(prefix, "").strip()

        # If title is too short, combine with domain
        if len(title.split()) < 3 and domain:
            title = f"{domain} {title}"

        # Cap query length — search engines work best with ~5-8 words
        words = title.split()[:8]
        query = " ".join(words)

        return query if len(query) > 5 else ""

    # ── Missile / kinetic trajectory extraction ──────────────────────

    _MISSILE_KEYWORDS = frozenset([
        "missile", "intercept", "ballistic", "cruise", "shaheed",
        "shahed", "patriot", "iron dome", "s-300", "s-400",
        "thaad", "arrow", "david's sling", "kinzhal", "iskander",
        "kalibr", "harpoon", "scud", "drone strike", "drone attack",
        "uav strike", "artillery", "rocket attack", "launch",
    ])

    _TRAJECTORY_REGIONS = {
        # Known conflict corridors: name → (approx origin, approx target)
        "ukraine_east": {"from": (48.0, 38.0), "to": (50.4, 30.5)},      # Donbas → Kyiv
        "ukraine_south": {"from": (44.5, 33.5), "to": (46.5, 36.0)},     # Crimea → Zaporizhzhia
        "iran_israel": {"from": (32.5, 51.5), "to": (32.0, 34.8)},       # Iran → Israel
        "yemen_israel": {"from": (15.5, 44.2), "to": (31.8, 34.8)},      # Houthi → Israel
        "yemen_redSea": {"from": (15.0, 43.0), "to": (13.5, 42.5)},      # Houthi → Red Sea shipping
        "gaza_israel": {"from": (31.4, 34.4), "to": (31.8, 34.8)},       # Gaza → Israel
        "lebanon_israel": {"from": (33.9, 35.5), "to": (33.0, 35.2)},    # Hezbollah → Israel
        "north_korea": {"from": (39.0, 125.8), "to": (38.5, 131.0)},     # DPRK test launches
        "armenia_azerbaijan": {"from": (40.0, 44.5), "to": (39.8, 47.0)}, # Caucasus
    }

    def _extract_trajectories(self, events: list, snapshot) -> list:
        """Detect missile/strike events and attach trajectory data.

        Scans events and snapshot conflicts/liveuamap for missile keywords,
        then estimates launch → impact/intercept arcs using known corridors.
        Also triggers drawTrajectory commands to animate arcs on the map.
        """
        # Collect all text sources for missile detection
        text_items: List[Tuple[str, Dict[str, Any]]] = []
        for ev in events:
            text_items.append((
                f"{ev.get('title', '')} {ev.get('detail', '')}".lower(),
                ev,
            ))
        for item in (getattr(snapshot, "liveuamap", []) or []):
            text_items.append((
                str(item.get("title", "") or item.get("description", "")).lower(),
                item,
            ))
        for item in (getattr(snapshot, "conflicts", []) or []):
            text_items.append((
                str(item.get("notes", "") or item.get("description", "")).lower(),
                item,
            ))
        for item in (getattr(snapshot, "news_feed", []) or []):
            text_items.append((
                str(item.get("title", "") or item.get("headline", "")).lower(),
                item,
            ))

        trajectories_to_draw: List[Dict[str, Any]] = []

        for text, source in text_items:
            matching_keywords = [k for k in self._MISSILE_KEYWORDS if k in text]
            if not matching_keywords:
                continue

            # Determine trajectory type
            if any(k in text for k in ("intercept", "iron dome", "patriot",
                                        "s-300", "s-400", "thaad", "arrow",
                                        "david's sling", "shot down")):
                traj_type = "intercept"
            elif any(k in text for k in ("drone", "uav", "shaheed", "shahed")):
                traj_type = "drone"
            elif any(k in text for k in ("artillery", "rocket attack")):
                traj_type = "artillery"
            elif any(k in text for k in ("strike", "kinetic")):
                traj_type = "strike"
            else:
                traj_type = "missile"

            is_intercepted = any(k in text for k in (
                "intercept", "shot down", "neutralized", "destroyed",
                "iron dome", "patriot", "thaad", "arrow",
            ))

            # Try to match to a known corridor
            best_corridor = None
            for corridor_name, coords in self._TRAJECTORY_REGIONS.items():
                corridor_kw = corridor_name.replace("_", " ").split()
                if any(kw in text for kw in corridor_kw):
                    best_corridor = coords
                    break

            # Fallback: check if source has coordinates
            src_lat = float(source.get("lat", source.get("latitude", 0)) or 0)
            src_lng = float(source.get("lng", source.get("lon",
                           source.get("longitude", 0))) or 0)

            if best_corridor:
                from_lat, from_lng = best_corridor["from"]
                to_lat, to_lng = best_corridor["to"]
                # If source has coords, use those as target
                if src_lat and src_lng:
                    to_lat, to_lng = src_lat, src_lng
            elif src_lat and src_lng:
                # We know where it landed but not where it came from — estimate
                from_lat = src_lat + random.uniform(1.0, 3.0) * random.choice([-1, 1])
                from_lng = src_lng + random.uniform(1.0, 3.0) * random.choice([-1, 1])
                to_lat, to_lng = src_lat, src_lng
            else:
                continue  # Can't determine trajectory without coordinates

            # Build intercept point — slightly before target if intercepted
            intercept_lat = intercept_lng = None
            if is_intercepted:
                t = random.uniform(0.6, 0.85)
                intercept_lat = from_lat + (to_lat - from_lat) * t
                intercept_lng = from_lng + (to_lng - from_lng) * t

            traj_data = {
                "fromLat": round(from_lat, 4),
                "fromLng": round(from_lng, 4),
                "toLat": round(to_lat, 4),
                "toLng": round(to_lng, 4),
                "type": traj_type,
                "label": (source.get("title", "") or source.get("headline", ""))[:60] if isinstance(source, dict) else "",
                "intercepted": is_intercepted,
            }
            if intercept_lat is not None:
                traj_data["interceptLat"] = round(intercept_lat, 4)
                traj_data["interceptLng"] = round(intercept_lng, 4)

            # Attach to matching event
            if source in [ev for _, ev in text_items[:len(events)]]:
                for ev in events:
                    if ev.get("title", "").lower() in text or ev.get("detail", "").lower() in text:
                        if not ev.get("trajectory"):
                            ev["trajectory"] = traj_data
                            break

            # Queue for drawTrajectory command
            trajectories_to_draw.append(traj_data)

        # Store for later sending via remote control (narrate_cycle will call them)
        self._pending_trajectories = trajectories_to_draw[:10]
        return events

    async def _send_trajectory_arcs(self):
        """Send pending trajectory arcs to the dashboard for animation."""
        for traj in getattr(self, "_pending_trajectories", []):
            try:
                await self.remote.draw_trajectory(
                    from_lat=traj["fromLat"],
                    from_lng=traj["fromLng"],
                    to_lat=traj["toLat"],
                    to_lng=traj["toLng"],
                    trajectory_type=traj.get("type", "missile"),
                    label=traj.get("label", ""),
                    intercepted=traj.get("intercepted", False),
                    intercept_lat=traj.get("interceptLat"),
                    intercept_lng=traj.get("interceptLng"),
                    duration=6000.0,
                )
                await asyncio.sleep(0.5)  # Stagger arcs
            except Exception as e:
                logger.warning("Failed to send trajectory arc: %s", e)
        self._pending_trajectories = []

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_moves": self._total_moves,
            "current_position": {"lat": self._current_lat, "lng": self._current_lng},
            "current_style": self._current_style,
            "patrol_index": self._patrol_index,
            "history_length": len(self._move_history),
            "last_move": self._move_history[-1] if self._move_history else None,
            "cycle_count": self._cycle_count,
            "unique_cells_visited": len(self._visited_cells),
            "total_waypoints_available": len(PATROL_WAYPOINTS),
            "diary_entries": len(self._diary_entries),
            "dossier_entries": len(self._dossier_entries),
            "live_cameras_shown": len(self._shown_live_cameras),
            "tracked_stories": len(self._tracked_stories),
            "pending_trajectories": len(getattr(self, "_pending_trajectories", [])),
        }
