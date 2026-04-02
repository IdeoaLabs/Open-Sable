"""
#16 — Wargaming / What-If Simulator

"What happens if Iran blocks the Strait of Hormuz?"
"What if China establishes an Air Defense Identification Zone over Taiwan?"
"What if Russia cuts the Baltic undersea cables?"

Uses current world state (from snapshots + world model) and
historical conflict patterns to simulate multi-order consequences.

This is something Palantir sells to DOD for millions.
We give it away as an OpenSable skill.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


@dataclass
class Consequence:
    """A single consequence in a simulation cascade."""
    order: int  # 1st order, 2nd order, 3rd order
    domain: str  # military, economic, diplomatic, humanitarian, cyber, energy
    description: str
    probability: float  # 0.0-1.0
    severity: str  # low, medium, high, critical
    time_horizon: str  # immediate, days, weeks, months
    entities_affected: List[str] = field(default_factory=list)


@dataclass
class WargameScenario:
    """A what-if scenario with simulated consequences."""
    scenario_id: str
    trigger: str  # The "what if" statement
    region: str
    created_at: str
    consequences: List[Consequence] = field(default_factory=list)
    total_severity_score: float = 0.0
    escalation_risk: str = "low"  # low, moderate, high, extreme
    summary: str = ""
    world_state_at_creation: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Pre-built scenario templates (historical precedent based)
# ---------------------------------------------------------------------------

SCENARIO_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "strait_hormuz_blockade": {
        "trigger": "Iran blocks the Strait of Hormuz",
        "region": "Middle East",
        "consequences": [
            {"order": 1, "domain": "energy", "desc": "21% of global oil supply disrupted; oil price spikes 40-80%",
             "prob": 0.95, "sev": "critical", "time": "immediate"},
            {"order": 1, "domain": "military", "desc": "US 5th Fleet activates mine countermeasures; carrier strike group surges",
             "prob": 0.90, "sev": "high", "time": "immediate"},
            {"order": 2, "domain": "economic", "desc": "Global shipping insurance rates triple; LNG futures surge",
             "prob": 0.85, "sev": "high", "time": "days"},
            {"order": 2, "domain": "diplomatic", "desc": "UN Security Council emergency session; Gulf states invoke defense pacts",
             "prob": 0.80, "sev": "medium", "time": "days"},
            {"order": 3, "domain": "humanitarian", "desc": "Food price inflation in import-dependent nations (Egypt, Philippines)",
             "prob": 0.70, "sev": "high", "time": "weeks"},
            {"order": 3, "domain": "military", "desc": "Risk of miscalculation → direct US-Iran naval confrontation",
             "prob": 0.45, "sev": "critical", "time": "weeks"},
        ],
        "escalation_risk": "extreme",
    },
    "taiwan_adiz": {
        "trigger": "China declares ADIZ over Taiwan Strait",
        "region": "Indo-Pacific",
        "consequences": [
            {"order": 1, "domain": "military", "desc": "PLA Air Force surge; 100+ sorties daily in ADIZ enforcement",
             "prob": 0.90, "sev": "high", "time": "immediate"},
            {"order": 1, "domain": "diplomatic", "desc": "US/Japan/Australia joint statement; AUKUS activation likely",
             "prob": 0.85, "sev": "high", "time": "immediate"},
            {"order": 2, "domain": "economic", "desc": "Taiwan semiconductor exports disrupted; global chip shortage escalates",
             "prob": 0.75, "sev": "critical", "time": "days"},
            {"order": 2, "domain": "military", "desc": "US carrier groups reposition; Japan activates southwest island defense",
             "prob": 0.80, "sev": "high", "time": "days"},
            {"order": 3, "domain": "economic", "desc": "Stock market crash in Asia; capital flight from Chinese markets",
             "prob": 0.65, "sev": "high", "time": "weeks"},
            {"order": 3, "domain": "cyber", "desc": "Chinese APT groups target Taiwan critical infrastructure",
             "prob": 0.70, "sev": "critical", "time": "weeks"},
        ],
        "escalation_risk": "extreme",
    },
    "baltic_cable_cut": {
        "trigger": "Russia cuts Baltic undersea cables (Nord Stream style)",
        "region": "Europe / Baltic",
        "consequences": [
            {"order": 1, "domain": "cyber", "desc": "Internet connectivity disrupted for Baltic states and Scandinavia",
             "prob": 0.90, "sev": "high", "time": "immediate"},
            {"order": 1, "domain": "military", "desc": "NATO activates Article 5 consultation; P-8 ASW surge in Baltic",
             "prob": 0.75, "sev": "high", "time": "immediate"},
            {"order": 2, "domain": "economic", "desc": "Financial transaction disruption; Nordic banking system stress",
             "prob": 0.70, "sev": "high", "time": "days"},
            {"order": 2, "domain": "diplomatic", "desc": "EU emergency summit; new Russia sanctions package",
             "prob": 0.80, "sev": "medium", "time": "days"},
            {"order": 3, "domain": "military", "desc": "NATO permanent submarine patrol in Baltic; undersea surveillance network deployed",
             "prob": 0.60, "sev": "medium", "time": "months"},
        ],
        "escalation_risk": "high",
    },
    "suez_blockage": {
        "trigger": "Suez Canal blocked (deliberate or accidental)",
        "region": "Middle East / Global",
        "consequences": [
            {"order": 1, "domain": "economic", "desc": "12% of global trade rerouting via Cape of Good Hope; +10-14 days transit",
             "prob": 0.95, "sev": "high", "time": "immediate"},
            {"order": 1, "domain": "energy", "desc": "European LNG deliveries delayed; gas prices spike 15-25%",
             "prob": 0.85, "sev": "high", "time": "immediate"},
            {"order": 2, "domain": "economic", "desc": "Container shipping rates surge 300-500%; supply chain disruption",
             "prob": 0.80, "sev": "high", "time": "days"},
            {"order": 3, "domain": "humanitarian", "desc": "Delayed grain shipments to East Africa; food security crisis",
             "prob": 0.65, "sev": "high", "time": "weeks"},
        ],
        "escalation_risk": "moderate",
    },
    "south_china_sea_incident": {
        "trigger": "Chinese coast guard rams Philippine vessel at Second Thomas Shoal",
        "region": "Indo-Pacific",
        "consequences": [
            {"order": 1, "domain": "military", "desc": "Philippines activates Mutual Defense Treaty with US",
             "prob": 0.70, "sev": "high", "time": "immediate"},
            {"order": 1, "domain": "diplomatic", "desc": "ASEAN emergency meeting; China isolated diplomatically",
             "prob": 0.65, "sev": "medium", "time": "days"},
            {"order": 2, "domain": "military", "desc": "US amphibious group repositions to Philippine Sea",
             "prob": 0.60, "sev": "high", "time": "days"},
            {"order": 3, "domain": "economic", "desc": "Shipping reroutes from SCS; Southeast Asian trade disrupted",
             "prob": 0.50, "sev": "medium", "time": "weeks"},
        ],
        "escalation_risk": "high",
    },
    "houthi_chokepoint_escalation": {
        "trigger": "Houthis sink a commercial vessel in Bab el-Mandeb",
        "region": "Red Sea / Horn of Africa",
        "consequences": [
            {"order": 1, "domain": "economic", "desc": "All major shipping lines halt Red Sea transit",
             "prob": 0.90, "sev": "critical", "time": "immediate"},
            {"order": 1, "domain": "military", "desc": "US/UK expand naval operations; broadened strike campaign on Houthi targets",
             "prob": 0.85, "sev": "high", "time": "immediate"},
            {"order": 2, "domain": "energy", "desc": "European energy prices spike as LNG reroutes",
             "prob": 0.75, "sev": "high", "time": "days"},
            {"order": 2, "domain": "diplomatic", "desc": "Iran escalation risk rises; Gulf states increase defense posture",
             "prob": 0.65, "sev": "high", "time": "days"},
            {"order": 3, "domain": "humanitarian", "desc": "East Africa food aid delayed; humanitarian crisis in Yemen worsens",
             "prob": 0.70, "sev": "high", "time": "weeks"},
        ],
        "escalation_risk": "extreme",
    },
}


class WargamingSimulator:
    """
    What-if scenario simulator for geopolitical wargaming.

    Combines pre-built templates, current world state from snapshots,
    and LLM reasoning to simulate multi-order consequences.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "wargame_state.json"

        self.scenarios: Dict[str, WargameScenario] = {}
        self.total_simulations = 0
        self._world_state: Dict[str, Any] = {}

        self._load_state()

    # ── world state update ────────────────────────────────────────────

    def update_world_state(self, snapshot: IntelSnapshot):
        """Update current world state from latest snapshot — deep entity-level extraction."""
        import math

        mil_regions = self._count_military_regions(snapshot)
        ship_regions = self._count_ship_regions(snapshot)
        carrier_positions = self._extract_carrier_positions(snapshot)
        jamming_zones = self._extract_jamming_zones(snapshot)
        conflict_regions = self._extract_conflict_regions(snapshot)
        cyber_severity = self._assess_cyber_severity(snapshot)
        nuclear_status = self._assess_nuclear_status(snapshot)
        outage_countries = self._count_outage_countries(snapshot)
        recon_flights = self._detect_recon_flights(snapshot)

        self._world_state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_flights": len(snapshot.flights),
            "military_flights": len(snapshot.military_flights),
            "active_ships": len(snapshot.ships),
            "active_conflicts": len(snapshot.conflicts),
            "gps_jamming_zones": len(snapshot.gps_jamming),
            "cyber_threats": len(snapshot.cyber_threats),
            "nuclear_facilities": len(snapshot.nuclear_facilities),
            "internet_outages": len(snapshot.internet_outages),
            "carriers_tracked": len(snapshot.carriers),
            "earthquakes": len(snapshot.earthquakes),
            "fires": len(snapshot.fires),
            "ransomware_events": len(snapshot.ransomware),
            # Deep analysis
            "regions_with_military": mil_regions,
            "regions_with_ships": ship_regions,
            "carrier_positions": carrier_positions,
            "jamming_hotspots": jamming_zones,
            "conflict_regions": conflict_regions,
            "cyber_severity": cyber_severity,
            "nuclear_status": nuclear_status,
            "outage_countries": outage_countries,
            "recon_flights": recon_flights,
        }

        # Auto-detect live scenarios from current data
        self._live_scenario_triggers = self._detect_live_triggers(snapshot)

    def auto_wargame(self, snapshot: IntelSnapshot) -> List[WargameScenario]:
        """
        Automatically detect and simulate scenarios that MATCH CURRENT LIVE DATA.
        This is the key upgrade — the simulator now reacts to what's actually
        happening in the world, not just pre-built templates.
        """
        self.update_world_state(snapshot)
        results: List[WargameScenario] = []

        for trigger_info in self._live_scenario_triggers:
            scenario = self.simulate(
                trigger=trigger_info["trigger"],
                region=trigger_info["region"],
            )
            # Adjust probabilities based on live evidence
            scenario = self._adjust_probabilities_from_live_data(
                scenario, trigger_info
            )
            scenario.summary = self._generate_summary(scenario)
            results.append(scenario)

        return results

    def _detect_live_triggers(self, snapshot: IntelSnapshot) -> List[Dict[str, Any]]:
        """Detect which geopolitical scenarios are ACTIVE based on real data."""
        import math
        triggers: List[Dict[str, Any]] = []

        # --- Strait of Hormuz ---
        hormuz_lat, hormuz_lon = 26.56, 56.25
        hormuz_mil = sum(1 for f in snapshot.military_flights
                         if self._haversine(f.get("lat", 0), f.get("lon", 0),
                                            hormuz_lat, hormuz_lon) < 500)
        hormuz_ships = sum(1 for s in snapshot.ships
                           if self._haversine(s.get("lat", 0), s.get("lon", 0),
                                              hormuz_lat, hormuz_lon) < 300)
        hormuz_jam = sum(1 for j in snapshot.gps_jamming
                         if self._haversine(j.get("lat", 0), j.get("lon", 0),
                                            hormuz_lat, hormuz_lon) < 500)
        if hormuz_mil > 5 or hormuz_ships > 20 or hormuz_jam > 0:
            triggers.append({
                "trigger": "Strait of Hormuz escalation — elevated military activity detected",
                "region": "Middle East",
                "template_key": "strait_hormuz_blockade",
                "evidence": {"military_flights": hormuz_mil, "ships": hormuz_ships,
                             "gps_jamming": hormuz_jam},
                "intensity": min(1.0, (hormuz_mil / 10 + hormuz_ships / 40 + hormuz_jam) / 3),
            })

        # --- Taiwan Strait ---
        taiwan_lat, taiwan_lon = 24.0, 121.0
        taiwan_mil = sum(1 for f in snapshot.military_flights
                         if self._haversine(f.get("lat", 0), f.get("lon", 0),
                                            taiwan_lat, taiwan_lon) < 600)
        taiwan_carriers = sum(1 for c in snapshot.carriers
                              if self._haversine(c.get("lat", 0), c.get("lon", 0),
                                                 taiwan_lat, taiwan_lon) < 1000)
        if taiwan_mil > 8 or taiwan_carriers > 0:
            triggers.append({
                "trigger": "Taiwan Strait tension — military buildup detected",
                "region": "Indo-Pacific",
                "template_key": "taiwan_adiz",
                "evidence": {"military_flights": taiwan_mil, "carriers": taiwan_carriers},
                "intensity": min(1.0, (taiwan_mil / 15 + taiwan_carriers * 0.5) / 2),
            })

        # --- Baltic / North Sea cables ---
        baltic_lat, baltic_lon = 58.0, 20.0
        baltic_ships = sum(1 for s in snapshot.ships
                           if self._haversine(s.get("lat", 0), s.get("lon", 0),
                                              baltic_lat, baltic_lon) < 500)
        baltic_outages = sum(1 for o in snapshot.internet_outages
                             if any(c in str(o).lower() for c in
                                    ["estonia", "latvia", "lithuania", "finland", "sweden", "denmark"]))
        if baltic_outages > 0:
            triggers.append({
                "trigger": "Baltic infrastructure concern — internet outages in Northern Europe",
                "region": "Europe / Baltic",
                "template_key": "baltic_cable_cut",
                "evidence": {"outages": baltic_outages, "ships_in_area": baltic_ships},
                "intensity": min(1.0, baltic_outages / 3),
            })

        # --- Suez Canal ---
        suez_lat, suez_lon = 30.0, 32.5
        suez_ships = sum(1 for s in snapshot.ships
                         if self._haversine(s.get("lat", 0), s.get("lon", 0),
                                            suez_lat, suez_lon) < 200)
        if suez_ships > 30:
            triggers.append({
                "trigger": "Suez Canal congestion — abnormal vessel concentration detected",
                "region": "Middle East / Global",
                "template_key": "suez_blockage",
                "evidence": {"ships_near_suez": suez_ships},
                "intensity": min(1.0, suez_ships / 60),
            })

        # --- South China Sea ---
        scs_lat, scs_lon = 12.0, 114.5
        scs_mil = sum(1 for f in snapshot.military_flights
                      if self._haversine(f.get("lat", 0), f.get("lon", 0),
                                         scs_lat, scs_lon) < 800)
        scs_ships = sum(1 for s in snapshot.ships
                        if self._haversine(s.get("lat", 0), s.get("lon", 0),
                                           scs_lat, scs_lon) < 500)
        if scs_mil > 5 or scs_ships > 50:
            triggers.append({
                "trigger": "South China Sea incident risk — elevated military presence",
                "region": "Indo-Pacific",
                "template_key": "south_china_sea_incident",
                "evidence": {"military_flights": scs_mil, "ships": scs_ships},
                "intensity": min(1.0, (scs_mil / 10 + scs_ships / 80) / 2),
            })

        # --- Red Sea / Houthis ---
        redsea_lat, redsea_lon = 13.5, 42.5
        redsea_mil = sum(1 for f in snapshot.military_flights
                         if self._haversine(f.get("lat", 0), f.get("lon", 0),
                                            redsea_lat, redsea_lon) < 500)
        redsea_conflicts = sum(1 for c in snapshot.conflicts
                               if any(kw in str(c).lower() for kw in
                                      ["houthi", "yemen", "red sea", "bab"]))
        if redsea_mil > 3 or redsea_conflicts > 0:
            triggers.append({
                "trigger": "Red Sea / Houthi escalation — active threat environment",
                "region": "Red Sea / Horn of Africa",
                "template_key": "houthi_chokepoint_escalation",
                "evidence": {"military_flights": redsea_mil, "conflicts": redsea_conflicts},
                "intensity": min(1.0, (redsea_mil / 8 + min(redsea_conflicts, 3) / 3) / 2),
            })

        # --- Generic: high military + GPS jamming = unknown conflict ---
        total_mil = len(snapshot.military_flights)
        total_jam = len(snapshot.gps_jamming)
        total_cyber = len(snapshot.cyber_threats)
        if total_mil > 30 and total_jam > 2 and total_cyber > 5:
            triggers.append({
                "trigger": "Multi-domain escalation — concurrent military surge / EW / cyber activity",
                "region": "Global",
                "template_key": None,
                "evidence": {"military_flights": total_mil, "gps_jamming": total_jam,
                             "cyber_threats": total_cyber},
                "intensity": min(1.0, (total_mil / 50 + total_jam / 5 + total_cyber / 15) / 3),
            })

        return triggers

    def _adjust_probabilities_from_live_data(
        self, scenario: WargameScenario, trigger_info: Dict[str, Any]
    ) -> WargameScenario:
        """Scale consequence probabilities up/down based on live evidence intensity."""
        intensity = trigger_info.get("intensity", 0.5)
        # Scale: intensity 0 → probabilities * 0.5, intensity 1 → probabilities * 1.2
        scale = 0.5 + intensity * 0.7
        for c in scenario.consequences:
            c.probability = min(0.99, c.probability * scale)
        scenario.total_severity_score = self._score_scenario(scenario)
        scenario.escalation_risk = self._compute_escalation_risk(scenario, trigger_info)
        return scenario

    def _compute_escalation_risk(
        self, scenario: WargameScenario, trigger_info: Dict[str, Any]
    ) -> str:
        """Compute escalation risk combining score and live evidence."""
        score = scenario.total_severity_score
        intensity = trigger_info.get("intensity", 0.0)
        combined = score * 0.6 + intensity * 100 * 0.4
        if combined >= 75:
            return "extreme"
        elif combined >= 55:
            return "high"
        elif combined >= 35:
            return "moderate"
        return "low"

    def _count_military_regions(self, snapshot: IntelSnapshot) -> Dict[str, int]:
        """Count military entities per broad region."""
        regions: Dict[str, int] = {}
        for flight in snapshot.military_flights:
            lat = flight.get("lat")
            lon = flight.get("lon")
            if lat is not None and lon is not None:
                region = self._classify_region(float(lat), float(lon))
                regions[region] = regions.get(region, 0) + 1
        return regions

    def _count_ship_regions(self, snapshot: IntelSnapshot) -> Dict[str, int]:
        """Count ships per broad region."""
        regions: Dict[str, int] = {}
        for ship in snapshot.ships:
            lat = ship.get("lat")
            lon = ship.get("lon")
            if lat is not None and lon is not None:
                region = self._classify_region(float(lat), float(lon))
                regions[region] = regions.get(region, 0) + 1
        return regions

    def _extract_carrier_positions(self, snapshot: IntelSnapshot) -> List[Dict[str, Any]]:
        """Extract carrier strike group positions."""
        carriers = []
        for c in snapshot.carriers:
            carriers.append({
                "name": c.get("name", "unknown"),
                "lat": c.get("lat"), "lon": c.get("lon"),
                "region": self._classify_region(float(c.get("lat", 0)), float(c.get("lon", 0))),
            })
        return carriers

    def _extract_jamming_zones(self, snapshot: IntelSnapshot) -> List[Dict[str, Any]]:
        """Extract GPS jamming zones with regional classification."""
        zones = []
        for j in snapshot.gps_jamming:
            zones.append({
                "lat": j.get("lat"), "lon": j.get("lon"),
                "region": self._classify_region(float(j.get("lat", 0)), float(j.get("lon", 0))),
                "strength": j.get("strength", j.get("radius", "unknown")),
            })
        return zones

    def _extract_conflict_regions(self, snapshot: IntelSnapshot) -> List[str]:
        """Extract active conflict regions."""
        regions = set()
        for c in snapshot.conflicts:
            loc = c.get("location", c.get("country", c.get("region", "")))
            if loc:
                regions.add(str(loc))
        return list(regions)

    def _assess_cyber_severity(self, snapshot: IntelSnapshot) -> Dict[str, Any]:
        """Assess overall cyber threat severity."""
        high_sev = sum(1 for t in snapshot.cyber_threats
                       if str(t.get("severity", "")).lower() in ("high", "critical"))
        apt_count = sum(1 for t in snapshot.cyber_threats
                        if "apt" in str(t).lower())
        return {
            "total": len(snapshot.cyber_threats),
            "high_severity": high_sev,
            "apt_attributed": apt_count,
            "ransomware_active": len(snapshot.ransomware),
        }

    def _assess_nuclear_status(self, snapshot: IntelSnapshot) -> Dict[str, Any]:
        """Assess nuclear domain status."""
        return {
            "facilities_tracked": len(snapshot.nuclear_facilities),
            "near_test_quakes": sum(1 for q in snapshot.earthquakes
                                    if float(q.get("magnitude", 0)) >= 4.0
                                    and float(q.get("depth", 999)) < 10),
        }

    def _count_outage_countries(self, snapshot: IntelSnapshot) -> List[str]:
        """List countries with active internet outages."""
        countries = set()
        for o in snapshot.internet_outages:
            country = o.get("country", o.get("location", ""))
            if country:
                countries.add(str(country))
        return list(countries)

    def _detect_recon_flights(self, snapshot: IntelSnapshot) -> List[Dict[str, Any]]:
        """Detect reconnaissance/surveillance aircraft in the data."""
        recon_types = {"rc135", "rc-135", "p-8", "p8", "e-3", "e3", "rivet", "cobra",
                       "u-2", "u2", "rq-4", "rq4", "global hawk", "poseidon", "sentry"}
        recon = []
        for f in snapshot.military_flights:
            callsign = str(f.get("callsign", "")).lower()
            ftype = str(f.get("type", "")).lower()
            if any(rt in callsign or rt in ftype for rt in recon_types):
                recon.append({
                    "callsign": f.get("callsign"),
                    "type": f.get("type"),
                    "lat": f.get("lat"), "lon": f.get("lon"),
                    "region": self._classify_region(float(f.get("lat", 0)), float(f.get("lon", 0))),
                })
        return recon

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in km."""
        import math
        try:
            lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
        except (TypeError, ValueError):
            return 99999.0
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _classify_region(lat: float, lon: float) -> str:
        if 20 < lat < 45 and 25 < lon < 65:
            return "middle_east"
        elif 0 < lat < 50 and 70 < lon < 140:
            return "indo_pacific"
        elif 45 < lat < 72 and -10 < lon < 40:
            return "europe"
        elif 25 < lat < 50 and -130 < lon < -65:
            return "north_america"
        elif -35 < lat < 15 and -20 < lon < 55:
            return "africa"
        elif -55 < lat < 15 and -90 < lon < -30:
            return "south_america"
        else:
            return "other"

    # ── scenario simulation ───────────────────────────────────────────

    def simulate(self, trigger: str, region: str = "",
                 llm: Any = None) -> WargameScenario:
        """
        Simulate a what-if scenario.

        First checks templates, then falls back to rule-based generation.
        LLM enrichment optional.
        """
        sid = hashlib.md5(f"{trigger}_{time.time()}".encode()).hexdigest()[:10]
        now = datetime.now(timezone.utc).isoformat()

        # Check templates
        template = self._match_template(trigger)

        if template:
            consequences = [
                Consequence(
                    order=c["order"],
                    domain=c["domain"],
                    description=c["desc"],
                    probability=c["prob"],
                    severity=c["sev"],
                    time_horizon=c["time"],
                )
                for c in template["consequences"]
            ]

            scenario = WargameScenario(
                scenario_id=sid,
                trigger=trigger,
                region=template.get("region", region),
                created_at=now,
                consequences=consequences,
                escalation_risk=template.get("escalation_risk", "moderate"),
                world_state_at_creation=self._world_state.copy() if self._world_state else None,
            )
        else:
            # Rule-based generic simulation
            scenario = self._generic_simulate(sid, trigger, region, now)

        # Score
        scenario.total_severity_score = self._score_scenario(scenario)
        scenario.summary = self._generate_summary(scenario)

        self.scenarios[sid] = scenario
        self.total_simulations += 1
        self._save_state()

        return scenario

    async def simulate_with_llm(self, trigger: str, region: str,
                                 llm: Any) -> WargameScenario:
        """Use LLM for deep what-if analysis."""
        # Start with rule-based
        scenario = self.simulate(trigger, region)

        if not llm:
            return scenario

        world_ctx = json.dumps(self._world_state, default=str) if self._world_state else "No current data"

        prompt = (
            "You are a senior intelligence analyst conducting a wargame.\n\n"
            f"SCENARIO: {trigger}\n"
            f"REGION: {region}\n"
            f"CURRENT WORLD STATE: {world_ctx}\n\n"
            "Generate 6-8 consequences across 3 orders of effect.\n"
            "Reply ONLY with JSON array:\n"
            '[{"order": 1, "domain": "military|economic|diplomatic|cyber|energy|humanitarian", '
            '"description": "...", "probability": 0.X, "severity": "low|medium|high|critical", '
            '"time_horizon": "immediate|days|weeks|months"}]\n'
        )

        try:
            resp = await llm.ask(prompt)
            data = json.loads(resp)
            if isinstance(data, list):
                scenario.consequences = [
                    Consequence(
                        order=c.get("order", 1),
                        domain=c.get("domain", "unknown"),
                        description=c.get("description", ""),
                        probability=float(c.get("probability", 0.5)),
                        severity=c.get("severity", "medium"),
                        time_horizon=c.get("time_horizon", "days"),
                    )
                    for c in data
                ]
                scenario.total_severity_score = self._score_scenario(scenario)

                # Also ask for summary
                summary_prompt = (
                    f"Write a 200-word intelligence brief summarizing the likely "
                    f"consequences if: {trigger}\n"
                    f"Based on consequences: {resp}\n"
                    f"Focus on escalation dynamics and strategic implications."
                )
                scenario.summary = await llm.ask(summary_prompt)
        except Exception as e:
            logger.warning("LLM wargame simulation failed: %s", e)

        self.scenarios[scenario.scenario_id] = scenario
        self._save_state()
        return scenario

    def _match_template(self, trigger: str) -> Optional[Dict[str, Any]]:
        """Match trigger text to a pre-built template."""
        trigger_lower = trigger.lower()
        keywords_map = {
            "strait_hormuz_blockade": ["hormuz", "iran block", "persian gulf block"],
            "taiwan_adiz": ["taiwan", "adiz", "china air defense", "taiwan strait"],
            "baltic_cable_cut": ["baltic cable", "undersea cable", "nord stream", "cable cut"],
            "suez_blockage": ["suez", "canal block"],
            "south_china_sea_incident": ["south china sea", "spratly", "second thomas", "scarborough"],
            "houthi_chokepoint_escalation": ["houthi", "bab el-mandeb", "red sea attack"],
        }

        for template_key, keywords in keywords_map.items():
            if any(kw in trigger_lower for kw in keywords):
                return SCENARIO_TEMPLATES.get(template_key)
        return None

    def _generic_simulate(self, sid: str, trigger: str,
                          region: str, now: str) -> WargameScenario:
        """Generate generic consequences for unknown scenarios."""
        consequences = [
            Consequence(
                order=1, domain="military",
                description=f"Armed forces in region activate heightened readiness",
                probability=0.70, severity="high", time_horizon="immediate",
            ),
            Consequence(
                order=1, domain="diplomatic",
                description="Emergency diplomatic consultations initiated",
                probability=0.80, severity="medium", time_horizon="immediate",
            ),
            Consequence(
                order=2, domain="economic",
                description="Regional markets react negatively; risk premiums increase",
                probability=0.65, severity="medium", time_horizon="days",
            ),
            Consequence(
                order=2, domain="cyber",
                description="State-affiliated APT groups increase activity",
                probability=0.50, severity="high", time_horizon="days",
            ),
            Consequence(
                order=3, domain="humanitarian",
                description="Civilian population displacement possible if escalation continues",
                probability=0.40, severity="high", time_horizon="weeks",
            ),
        ]

        return WargameScenario(
            scenario_id=sid,
            trigger=trigger,
            region=region or "unknown",
            created_at=now,
            consequences=consequences,
            escalation_risk="moderate",
            world_state_at_creation=self._world_state.copy() if self._world_state else None,
        )

    @staticmethod
    def _score_scenario(scenario: WargameScenario) -> float:
        """Compute aggregate severity score 0-100."""
        severity_weights = {"low": 1, "medium": 2, "high": 4, "critical": 8}
        total = 0.0
        for c in scenario.consequences:
            weight = severity_weights.get(c.severity, 1)
            total += weight * c.probability
        # Normalize to 0-100
        max_possible = len(scenario.consequences) * 8 if scenario.consequences else 1
        return min(100.0, (total / max_possible) * 100)

    @staticmethod
    def _generate_summary(scenario: WargameScenario) -> str:
        """Rule-based summary generation."""
        lines = [
            f"WARGAME: {scenario.trigger}",
            f"Region: {scenario.region}",
            f"Escalation Risk: {scenario.escalation_risk.upper()}",
            f"Severity Score: {scenario.total_severity_score:.0f}/100",
            "",
        ]
        for order in [1, 2, 3]:
            order_consequences = [c for c in scenario.consequences if c.order == order]
            if order_consequences:
                lines.append(f"{'='*3} {order}{'st' if order==1 else 'nd' if order==2 else 'rd'} Order Effects {'='*3}")
                for c in order_consequences:
                    lines.append(f"  [{c.domain.upper()}] (p={c.probability:.0%}, {c.time_horizon}) {c.description}")
        return "\n".join(lines)

    # ── queries ───────────────────────────────────────────────────────

    def get_scenario(self, scenario_id: str) -> Optional[WargameScenario]:
        return self.scenarios.get(scenario_id)

    def get_all_scenarios(self) -> List[WargameScenario]:
        return list(self.scenarios.values())

    def list_templates(self) -> List[Dict[str, str]]:
        """List available built-in scenario templates."""
        return [
            {"id": k, "trigger": v["trigger"], "region": v.get("region", ""),
             "escalation_risk": v.get("escalation_risk", "")}
            for k, v in SCENARIO_TEMPLATES.items()
        ]

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "total_simulations": self.total_simulations,
                "scenario_count": len(self.scenarios),
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save wargame state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_simulations = state.get("total_simulations", 0)
        except Exception as e:
            logger.warning("Failed to load wargame state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_simulations": self.total_simulations,
            "scenarios_stored": len(self.scenarios),
            "templates_available": len(SCENARIO_TEMPLATES),
            "world_state_available": bool(self._world_state),
        }
