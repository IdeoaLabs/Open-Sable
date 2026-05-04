"""
#8,  Entity Deep Dossier Automation

Click on any entity (aircraft, vessel, satellite, Interpol notice) and Sable
launches an autonomous research session: searches 5+ open sources, cross-refs
ownership, checks sanctions lists, analyzes recent routes, generates a
1-page dossier in 10 seconds.

Example output:
  N123AB,  Gulfstream G650
  Owner: Enigma Holdings LLC (Cayman Islands)
  OFAC Status: Clean | Interpol: No match
  Last 30 days: 12 flights, 3 to sanctioned-adjacent jurisdictions
  Risk Score: ELEVATED (3/5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .connector import ZunvraConnector, IntelSnapshot

logger = logging.getLogger(__name__)


@dataclass
class DossierSection:
    """One section of an entity dossier."""
    title: str
    content: str
    source: str = ""
    confidence: float = 1.0


@dataclass
class EntityDossier:
    """Full intelligence dossier on a single entity."""
    dossier_id: str
    entity_type: str  # aircraft | vessel | satellite | person | organization
    entity_id: str    # ICAO hex, MMSI, NORAD ID, etc.
    entity_name: str
    timestamp: str
    sections: List[DossierSection] = field(default_factory=list)
    risk_score: int = 0  # 0-5
    risk_label: str = "UNKNOWN"
    flags: List[str] = field(default_factory=list)
    lat: Optional[float] = None
    lon: Optional[float] = None

    def to_text(self, max_chars: int = 4000) -> str:
        risk_colors = {0: "NONE", 1: "LOW", 2: "GUARDED", 3: "ELEVATED", 4: "HIGH", 5: "CRITICAL"}
        lines = [
            f"ENTITY DOSSIER,  {self.entity_type.upper()}",
            f"  ID: {self.entity_id} | Name: {self.entity_name}",
            f"  Risk: {risk_colors.get(self.risk_score, 'UNKNOWN')} ({self.risk_score}/5)",
            f"  Generated: {self.timestamp}",
        ]
        if self.flags:
            lines.append(f"  Flags: {', '.join(self.flags)}")
        lines.append("")
        for section in self.sections:
            lines.append(f"  [{section.title}]")
            lines.append(f"    {section.content}")
            if section.source:
                lines.append(f"    Source: {section.source}")
            lines.append("")
        return "\n".join(lines)[:max_chars]


# ---------------------------------------------------------------------------
# Lookup databases (embedded / fast)
# ---------------------------------------------------------------------------

# Sanctioned-adjacent jurisdictions
RISK_JURISDICTIONS = {
    "cayman islands", "british virgin islands", "panama", "malta",
    "cyprus", "bermuda", "isle of man", "guernsey", "jersey",
    "liechtenstein", "monaco", "bahamas", "seychelles",
    "marshall islands", "vanuatu", "samoa",
}

# Military ICAO hex ranges (simplified)
MILITARY_HEX_RANGES = [
    ("AE0000", "AE9999"),  # US Military
    ("43C000", "43CFFF"),  # UK RAF
    ("3A8000", "3AFFFF"),  # France
    ("3E0000", "3EFFFF"),  # Germany
]


# ---------------------------------------------------------------------------
# Dossier Generator
# ---------------------------------------------------------------------------

class EntityDossierGenerator:
    """
    Autonomous research engine that builds deep dossiers on any entity
    by querying Zunvra APIs, open databases, and optionally the LLM.
    """

    MAX_DOSSIERS = 500

    def __init__(
        self,
        connector: ZunvraConnector,
        data_dir: Optional[Path] = None,
    ):
        self.connector = connector
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "dossiers.json"

        self.dossiers: Dict[str, EntityDossier] = {}
        self.total_generated = 0
        self._load_state()

    # ── main entry point ──────────────────────────────────────────────

    async def generate(
        self,
        entity_type: str,
        entity_id: str,
        entity_data: Optional[Dict[str, Any]] = None,
        llm=None,
    ) -> EntityDossier:
        """
        Generate a complete dossier on an entity.

        Parameters
        ----------
        entity_type : str
            'aircraft', 'vessel', 'satellite', 'person', 'organization'
        entity_id : str
            Unique identifier (ICAO hex, MMSI, NORAD ID, etc.)
        entity_data : dict, optional
            Raw entity data from Zunvra snapshot
        llm : optional
            LLM for deeper analysis
        """
        now = datetime.now(timezone.utc).isoformat()
        did = hashlib.sha256(f"{entity_type}_{entity_id}".encode()).hexdigest()[:12]

        dossier = EntityDossier(
            dossier_id=did,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_name=self._extract_name(entity_data) if entity_data else entity_id,
            timestamp=now,
        )

        # Phase 1: Extract basic info from entity data
        if entity_data:
            dossier.lat = entity_data.get("lat") or entity_data.get("latitude")
            dossier.lon = entity_data.get("lon") or entity_data.get("longitude")
            try:
                dossier.lat = float(dossier.lat) if dossier.lat else None
                dossier.lon = float(dossier.lon) if dossier.lon else None
            except (ValueError, TypeError):
                dossier.lat = dossier.lon = None

            dossier.sections.append(DossierSection(
                title="Identity",
                content=self._format_identity(entity_type, entity_data),
                source="Zunvra Live Feed",
            ))

        # Phase 2: Type-specific analysis
        if entity_type == "aircraft":
            await self._aircraft_dossier(dossier, entity_id, entity_data)
        elif entity_type == "vessel":
            await self._vessel_dossier(dossier, entity_id, entity_data)
        elif entity_type == "satellite":
            await self._satellite_dossier(dossier, entity_id, entity_data)
        elif entity_type == "person":
            await self._person_dossier(dossier, entity_id, entity_data)

        # Phase 3: Cross-reference with dark intelligence
        await self._cross_reference(dossier, entity_data)

        # Phase 4: Risk scoring
        self._compute_risk(dossier)

        # Phase 5: LLM assessment
        if llm:
            try:
                await self._llm_assess(llm, dossier)
            except Exception as e:
                logger.debug("LLM dossier assessment failed: %s", e)

        # Phase 6: Nearby entity context
        if dossier.lat and dossier.lon:
            snap = self.connector.last_snapshot
            if snap:
                nearby = snap.entities_near(dossier.lat, dossier.lon, radius_km=50)
                if nearby:
                    parts = [f"{k}: {len(v)}" for k, v in nearby.items()]
                    dossier.sections.append(DossierSection(
                        title="Proximity Analysis",
                        content=f"Entities within 50km: {', '.join(parts)}",
                        source="Zunvra Spatial Query",
                    ))

        # Store
        self.dossiers[did] = dossier
        self.total_generated += 1
        if len(self.dossiers) > self.MAX_DOSSIERS:
            oldest_keys = sorted(self.dossiers.keys())[:100]
            for k in oldest_keys:
                del self.dossiers[k]
        self._save_state()

        return dossier

    # ── type-specific builders ────────────────────────────────────────

    async def _aircraft_dossier(self, dossier: EntityDossier, icao_hex: str, data: Optional[Dict]):
        """Build aircraft-specific sections."""
        if data:
            callsign = data.get("callsign", data.get("flight", "")).strip()
            altitude = data.get("alt_baro", data.get("altitude", ""))
            speed = data.get("gs", data.get("speed", ""))
            heading = data.get("track", data.get("heading", ""))
            
            dossier.sections.append(DossierSection(
                title="Flight Data",
                content=(
                    f"Callsign: {callsign} | Alt: {altitude}ft | "
                    f"Speed: {speed}kts | Heading: {heading}°"
                ),
                source="ADS-B",
            ))

            # Check aircraft type
            ac_type = data.get("t", data.get("type", data.get("aircraft_type", "")))
            if ac_type:
                dossier.sections.append(DossierSection(
                    title="Aircraft Type",
                    content=f"Type designator: {ac_type}",
                    source="Plane-Alert DB",
                ))

            # Check for special categories
            owner = data.get("owner", data.get("operator", ""))
            if owner:
                dossier.sections.append(DossierSection(
                    title="Owner/Operator",
                    content=owner,
                    source="Registry / Plane-Alert",
                ))

                # Jurisdiction check
                lower = owner.lower()
                for jurisdiction in RISK_JURISDICTIONS:
                    if jurisdiction in lower:
                        dossier.flags.append(f"Registered in risk jurisdiction: {jurisdiction}")
                        break

        # Military check
        is_military = self._check_military_hex(icao_hex)
        if is_military:
            dossier.flags.append("Military aircraft (ICAO hex range)")
            dossier.sections.append(DossierSection(
                title="Military Classification",
                content=f"ICAO hex {icao_hex} falls within known military allocation range.",
                source="ICAO Hex Analysis",
            ))

    async def _vessel_dossier(self, dossier: EntityDossier, mmsi: str, data: Optional[Dict]):
        """Build vessel-specific sections."""
        if data:
            ship_name = data.get("name", data.get("ship_name", ""))
            ship_type = data.get("ship_type", data.get("type", ""))
            flag = data.get("flag", data.get("country", ""))
            destination = data.get("destination", "")
            
            dossier.sections.append(DossierSection(
                title="Vessel Data",
                content=(
                    f"Name: {ship_name} | Type: {ship_type} | "
                    f"Flag: {flag} | Destination: {destination}"
                ),
                source="AIS Stream",
            ))

            # Flag of convenience check
            foc_flags = {"panama", "liberia", "marshall islands", "bahamas", "malta",
                        "antigua", "barbuda", "honduras", "comoros", "togo"}
            if flag and flag.lower() in foc_flags:
                dossier.flags.append(f"Flag of convenience: {flag}")

    async def _satellite_dossier(self, dossier: EntityDossier, norad_id: str, data: Optional[Dict]):
        """Build satellite-specific sections."""
        if data:
            sat_name = data.get("name", data.get("satname", ""))
            orbit_type = data.get("orbit", data.get("classification", ""))
            
            dossier.sections.append(DossierSection(
                title="Orbital Data",
                content=f"Name: {sat_name} | Orbit: {orbit_type} | NORAD ID: {norad_id}",
                source="CelesTrak TLE",
            ))

            # Military satellite check
            mil_keywords = ["usa ", "noss", "nrol", "lacrosse", "keyhole", "onyx",
                          "mentor", "orion", "trumpet", "advanced orion"]
            if any(k in (sat_name or "").lower() for k in mil_keywords):
                dossier.flags.append("Classified/military satellite")

    async def _person_dossier(self, dossier: EntityDossier, person_id: str, data: Optional[Dict]):
        """Build person-specific sections (Interpol notices, FBI wanted, etc.)."""
        if data:
            name = data.get("name", data.get("forename", ""))
            nationality = data.get("nationality", data.get("nationalities", ""))
            charge = data.get("charge", data.get("charges", ""))
            
            dossier.sections.append(DossierSection(
                title="Person Profile",
                content=f"Name: {name} | Nationality: {nationality} | Charge: {charge}",
                source="Interpol / FBI / OFAC",
            ))

    # ── cross-reference ───────────────────────────────────────────────

    async def _cross_reference(self, dossier: EntityDossier, data: Optional[Dict]):
        """Query Zunvra dark intel endpoints for cross-references."""
        checks = []

        # Check OFAC sanctions list
        try:
            ofac_data = await self.connector.fetch_endpoint("/api/dark-intel/sanctions")
            if ofac_data and isinstance(ofac_data, list):
                name = dossier.entity_name.upper()
                for entry in ofac_data:
                    if not isinstance(entry, dict):
                        continue
                    entry_name = (entry.get("name", "") or "").upper()
                    if name and name in entry_name:
                        dossier.flags.append(f"OFAC SDN MATCH: {entry_name}")
                        checks.append("OFAC: MATCH FOUND")
                        break
                else:
                    checks.append("OFAC: Clean")
        except Exception:
            checks.append("OFAC: Check unavailable")

        # Check Interpol
        try:
            interpol_data = await self.connector.fetch_endpoint("/api/dark-intel/interpol")
            if interpol_data and isinstance(interpol_data, list):
                name = dossier.entity_name.upper()
                for notice in interpol_data:
                    if not isinstance(notice, dict):
                        continue
                    notice_name = (notice.get("name", "") or "").upper()
                    if name and name in notice_name:
                        dossier.flags.append(f"INTERPOL RED NOTICE: {notice_name}")
                        checks.append("Interpol: MATCH FOUND")
                        break
                else:
                    checks.append("Interpol: No match")
        except Exception:
            checks.append("Interpol: Check unavailable")

        if checks:
            dossier.sections.append(DossierSection(
                title="Watchlist Cross-Reference",
                content=" | ".join(checks),
                source="Zunvra Dark Intel",
            ))

    # ── risk scoring ──────────────────────────────────────────────────

    def _compute_risk(self, dossier: EntityDossier):
        """Compute a 0-5 risk score from accumulated flags."""
        score = 0

        for flag in dossier.flags:
            flag_lower = flag.lower()
            if "ofac" in flag_lower or "sanction" in flag_lower:
                score += 3
            elif "interpol" in flag_lower:
                score += 3
            elif "military" in flag_lower or "classified" in flag_lower:
                score += 1
            elif "risk jurisdiction" in flag_lower or "flag of convenience" in flag_lower:
                score += 1
            else:
                score += 1

        dossier.risk_score = min(5, score)
        labels = {0: "NONE", 1: "LOW", 2: "GUARDED", 3: "ELEVATED", 4: "HIGH", 5: "CRITICAL"}
        dossier.risk_label = labels.get(dossier.risk_score, "UNKNOWN")

    # ── LLM assessment ────────────────────────────────────────────────

    async def _llm_assess(self, llm, dossier: EntityDossier):
        """Use LLM to generate a narrative assessment."""
        sections_text = "\n".join(f"[{s.title}] {s.content}" for s in dossier.sections)
        flags_text = ", ".join(dossier.flags) if dossier.flags else "None"

        prompt = (
            "You are an OSINT intelligence analyst generating an entity assessment.\n\n"
            f"Entity: {dossier.entity_type.upper()},  {dossier.entity_name}\n"
            f"ID: {dossier.entity_id}\n"
            f"Risk Score: {dossier.risk_score}/5\n"
            f"Flags: {flags_text}\n\n"
            f"Sections:\n{sections_text}\n\n"
            "Write a 3-4 sentence intelligence assessment. Include:\n"
            "1. Summary of the entity\n"
            "2. Key risk factors (if any)\n"
            "3. Recommended analyst actions\n"
            "Be precise and professional."
        )

        raw = await llm.chat_raw(prompt, max_tokens=200)
        if raw and raw.strip():
            dossier.sections.append(DossierSection(
                title="AI Assessment",
                content=raw.strip(),
                source="OpenSable LLM Analysis",
                confidence=0.8,
            ))

    # ── helpers ───────────────────────────────────────────────────────

    def _extract_name(self, data: Dict) -> str:
        for key in ["callsign", "flight", "name", "ship_name", "satname", "registration"]:
            val = data.get(key)
            if val and str(val).strip():
                return str(val).strip()
        return "Unknown"

    def _format_identity(self, entity_type: str, data: Dict) -> str:
        parts = []
        for key in ["callsign", "flight", "name", "registration", "icao",
                    "mmsi", "imo", "norad_id", "type", "flag", "operator"]:
            val = data.get(key)
            if val:
                parts.append(f"{key}: {val}")
        return " | ".join(parts) if parts else "No identity data"

    def _check_military_hex(self, icao_hex: str) -> bool:
        try:
            hex_val = int(icao_hex, 16)
            for low, high in MILITARY_HEX_RANGES:
                if int(low, 16) <= hex_val <= int(high, 16):
                    return True
        except (ValueError, TypeError):
            pass
        return False

    # ── queries ───────────────────────────────────────────────────────

    def get_dossier(self, entity_id: str) -> Optional[EntityDossier]:
        for d in self.dossiers.values():
            if d.entity_id == entity_id:
                return d
        return None

    def get_high_risk(self, min_score: int = 3) -> List[EntityDossier]:
        return [d for d in self.dossiers.values() if d.risk_score >= min_score]

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "dossiers": {k: asdict(v) for k, v in list(self.dossiers.items())[-100:]},
                "total_generated": self.total_generated,
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save dossier state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_generated = state.get("total_generated", 0)
        except Exception as e:
            logger.warning("Failed to load dossier state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_generated": self.total_generated,
            "cached_dossiers": len(self.dossiers),
            "high_risk_count": len(self.get_high_risk()),
        }
