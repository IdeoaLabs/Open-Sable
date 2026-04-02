"""
#20 — Financial Intelligence (FININT) Tracker

Treasury/FinCEN/OFAC capability.  Analyzes economic data, prediction
markets, and dark-intel for sanctions evasion, economic warfare,
dark fleet operations, and financial anomalies.

Data sources: snapshot.economics, snapshot.prediction_markets,
              snapshot.dark_intel, snapshot.ships (dark fleet detection)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


@dataclass
class FinintAlert:
    """Financial intelligence alert."""
    alert_id: str
    alert_type: str  # sanctions_evasion, dark_fleet, market_anomaly,
                     # economic_warfare, trade_disruption, crypto_indicator
    severity: str
    timestamp: str
    title: str
    description: str
    entities: List[str] = field(default_factory=list)
    value_usd: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DarkFleetVessel:
    """A vessel suspected of sanctions evasion / dark fleet operations."""
    vessel_id: str
    name: str
    mmsi: str
    flags: List[str] = field(default_factory=list)  # Suspected flag-hopping
    suspicious_behaviors: List[str] = field(default_factory=list)
    first_detected: str = ""
    last_seen: str = ""
    risk_score: float = 0.0  # 0-10


@dataclass
class EconomicIndicator:
    """Tracked economic metric for warfare detection."""
    name: str
    current_value: float = 0.0
    previous_value: float = 0.0
    change_pct: float = 0.0
    anomaly: bool = False
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Sanctioned flag states & suspicious patterns
# ---------------------------------------------------------------------------

SANCTIONED_FLAG_STATES = {
    "DPRK", "North Korea", "Iran", "Syria", "Cuba", "Venezuela",
    "Russia",  # Partial
}

DARK_FLEET_INDICATORS = [
    "ais_gap",           # AIS transponder off
    "flag_change",       # Changed flag state
    "name_change",       # Changed vessel name
    "sts_transfer",      # Ship-to-ship transfer at sea
    "old_vessel",        # >15 years old single-hull
    "unknown_owner",     # Opaque ownership structure
    "sanctioned_port",   # Visited sanctioned port
]

# Prediction market topics that signal economic warfare
ECON_WARFARE_TOPICS = [
    "sanctions", "embargo", "trade war", "tariff", "blockade",
    "swift", "freeze", "seizure", "default", "currency crisis",
    "oil price", "gas price", "commodity", "inflation",
    "bank collapse", "debt crisis",
]


class FinintTracker:
    """
    Financial intelligence analysis engine.

    Detects sanctions evasion, dark fleet operations, economic warfare
    indicators, and market manipulation signals.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.alerts: List[FinintAlert] = []
        self.dark_fleet: Dict[str, DarkFleetVessel] = {}
        self.indicators: Dict[str, EconomicIndicator] = {}
        self._prev_economics: Dict[str, Any] = {}
        self._prev_predictions: List[Dict[str, Any]] = []
        self.total_detections = 0

    # ── main observation ──────────────────────────────────────────────

    def observe(self, snapshot: IntelSnapshot) -> List[FinintAlert]:
        """Analyze snapshot for financial intelligence indicators."""
        now_str = datetime.now(timezone.utc).isoformat()
        new_alerts: List[FinintAlert] = []

        # 1. Dark fleet detection (maritime sanctions evasion)
        new_alerts.extend(self._detect_dark_fleet(snapshot, now_str))

        # 2. Economic warfare indicators
        new_alerts.extend(self._analyze_economics(snapshot, now_str))

        # 3. Prediction market signals
        new_alerts.extend(self._analyze_predictions(snapshot, now_str))

        # 4. Dark intel financial indicators
        new_alerts.extend(self._analyze_dark_intel(snapshot, now_str))

        self.total_detections += len(new_alerts)
        self.alerts.extend(new_alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-250:]

        self._prev_economics = snapshot.economics.copy() if snapshot.economics else {}
        self._prev_predictions = list(snapshot.prediction_markets)

        return new_alerts

    # ── dark fleet detection ──────────────────────────────────────────

    def _detect_dark_fleet(self, snapshot: IntelSnapshot,
                           now_str: str) -> List[FinintAlert]:
        """Identify vessels engaged in sanctions evasion."""
        alerts: List[FinintAlert] = []

        for ship in snapshot.ships:
            mmsi = str(ship.get("mmsi", ship.get("MMSI", "")))
            name = ship.get("name", ship.get("ship_name", ""))
            flag = ship.get("flag", ship.get("flag_state", ""))
            speed = ship.get("speed", ship.get("sog"))
            ship_type = ship.get("type", ship.get("ship_type", ""))

            vid = f"vessel_{mmsi}"
            suspicious = []
            risk = 0.0

            # No AIS / missing MMSI
            if not mmsi or mmsi in ("0", "000000000"):
                suspicious.append("ais_gap")
                risk += 3.0

            # Sanctioned flag state
            if flag and any(sf.lower() in flag.lower() for sf in SANCTIONED_FLAG_STATES):
                suspicious.append("sanctioned_flag")
                risk += 2.5

            # Very slow speed at sea (possible STS transfer)
            if speed is not None:
                try:
                    spd = float(speed)
                    if 0.5 < spd < 3.0 and ship_type and "tanker" in ship_type.lower():
                        suspicious.append("sts_transfer_speed")
                        risk += 2.0
                except (ValueError, TypeError):
                    pass

            # Check for name/MMSI changes
            existing = self.dark_fleet.get(vid)
            if existing:
                if name and existing.name and name != existing.name:
                    suspicious.append("name_change")
                    risk += 3.0

            if risk >= 3.0:
                vessel = DarkFleetVessel(
                    vessel_id=vid,
                    name=name or "UNKNOWN",
                    mmsi=mmsi,
                    flags=[flag] if flag else [],
                    suspicious_behaviors=suspicious,
                    first_detected=existing.first_detected if existing else now_str,
                    last_seen=now_str,
                    risk_score=min(10.0, risk),
                )
                self.dark_fleet[vid] = vessel

                if not existing:  # Only alert on first detection
                    alerts.append(FinintAlert(
                        alert_id=hashlib.md5(f"dark_{vid}_{time.time()}".encode()).hexdigest()[:10],
                        alert_type="dark_fleet",
                        severity="critical" if risk >= 5.0 else "high",
                        timestamp=now_str,
                        title=f"Dark fleet vessel: {name or 'UNKNOWN'} (MMSI: {mmsi})",
                        description=(f"Suspicion: {', '.join(suspicious)}. "
                                    f"Risk: {risk:.1f}/10. Flag: {flag or 'none'}"),
                        entities=[vid],
                        evidence={"risk_score": risk, "behaviors": suspicious,
                                  "flag": flag, "ship_type": ship_type},
                    ))

        return alerts

    # ── economic warfare ──────────────────────────────────────────────

    def _analyze_economics(self, snapshot: IntelSnapshot,
                            now_str: str) -> List[FinintAlert]:
        """Detect economic warfare indicators from economics data."""
        alerts: List[FinintAlert] = []
        econ = snapshot.economics

        if not econ or not isinstance(econ, dict):
            return alerts

        # Track key metrics
        for key, value in econ.items():
            if not isinstance(value, (int, float)):
                continue

            prev = self._prev_economics.get(key)
            if prev is not None and isinstance(prev, (int, float)) and prev != 0:
                change_pct = ((value - prev) / abs(prev)) * 100

                indicator = EconomicIndicator(
                    name=key,
                    current_value=float(value),
                    previous_value=float(prev),
                    change_pct=change_pct,
                    anomaly=abs(change_pct) > 10,
                    timestamp=now_str,
                )
                self.indicators[key] = indicator

                if abs(change_pct) > 10:
                    direction = "surged" if change_pct > 0 else "crashed"
                    alerts.append(FinintAlert(
                        alert_id=hashlib.md5(f"econ_{key}_{time.time()}".encode()).hexdigest()[:10],
                        alert_type="economic_warfare",
                        severity="high" if abs(change_pct) > 25 else "medium",
                        timestamp=now_str,
                        title=f"Economic anomaly: {key} {direction} {abs(change_pct):.1f}%",
                        description=(f"{key}: {prev} → {value} ({change_pct:+.1f}%). "
                                    f"Possible economic warfare or market stress indicator."),
                        evidence={"metric": key, "previous": prev, "current": value,
                                  "change_pct": change_pct},
                    ))

        return alerts

    # ── prediction market analysis ────────────────────────────────────

    def _analyze_predictions(self, snapshot: IntelSnapshot,
                             now_str: str) -> List[FinintAlert]:
        """Detect market manipulation signals from prediction markets."""
        alerts: List[FinintAlert] = []

        for pred in snapshot.prediction_markets:
            title = (pred.get("title") or pred.get("question") or "").lower()
            prob = pred.get("probability", pred.get("yes_price"))

            if prob is None:
                continue
            try:
                prob = float(prob)
            except (ValueError, TypeError):
                continue

            # Check if this is an economically relevant market
            is_econ = any(kw in title for kw in ECON_WARFARE_TOPICS)
            if not is_econ:
                continue

            # High probability on destabilizing events
            if prob > 0.7:
                alerts.append(FinintAlert(
                    alert_id=hashlib.md5(f"pred_{title[:30]}_{time.time()}".encode()).hexdigest()[:10],
                    alert_type="market_anomaly",
                    severity="high" if prob > 0.85 else "medium",
                    timestamp=now_str,
                    title=f"High-probability economic threat: {title[:80]}",
                    description=(f"Prediction market signals {prob:.0%} probability. "
                                f"Markets are pricing in: {title[:120]}"),
                    value_usd=pred.get("volume"),
                    evidence={"probability": prob, "title": title,
                              "volume": pred.get("volume")},
                ))

            # Check for sudden probability shifts
            for prev in self._prev_predictions:
                prev_title = (prev.get("title") or prev.get("question") or "").lower()
                if prev_title == title:
                    prev_prob = prev.get("probability", prev.get("yes_price"))
                    if prev_prob is not None:
                        try:
                            shift = prob - float(prev_prob)
                            if abs(shift) > 0.15:
                                alerts.append(FinintAlert(
                                    alert_id=hashlib.md5(f"shift_{title[:30]}_{time.time()}".encode()).hexdigest()[:10],
                                    alert_type="market_anomaly",
                                    severity="high",
                                    timestamp=now_str,
                                    title=f"Prediction market shift: {title[:60]}",
                                    description=(f"Probability shifted {shift:+.0%} "
                                                f"({float(prev_prob):.0%} → {prob:.0%}). "
                                                f"Possible insider information or manipulation."),
                                    evidence={"previous_prob": float(prev_prob),
                                              "current_prob": prob, "shift": shift},
                                ))
                        except (ValueError, TypeError):
                            pass
                    break

        return alerts

    # ── dark intel analysis ───────────────────────────────────────────

    def _analyze_dark_intel(self, snapshot: IntelSnapshot,
                             now_str: str) -> List[FinintAlert]:
        """Extract financial signals from dark intel data."""
        alerts: List[FinintAlert] = []
        dark = snapshot.dark_intel

        if not dark or not isinstance(dark, dict):
            return alerts

        # Check for sanctions-related intelligence
        sanctions = dark.get("sanctions", [])
        if isinstance(sanctions, list) and sanctions:
            for entry in sanctions[:10]:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name", entry.get("entity", ""))
                program = entry.get("program", entry.get("list", ""))
                if name:
                    alerts.append(FinintAlert(
                        alert_id=hashlib.md5(f"sanction_{name}_{time.time()}".encode()).hexdigest()[:10],
                        alert_type="sanctions_evasion",
                        severity="medium",
                        timestamp=now_str,
                        title=f"Sanctions entity active: {name}",
                        description=f"Entity {name} on {program} sanctions list detected in data",
                        entities=[name],
                        evidence={"program": program, "raw": entry},
                    ))

        # Cryptocurrency indicators
        crypto = dark.get("crypto", dark.get("cryptocurrency", []))
        if isinstance(crypto, list):
            for tx in crypto[:5]:
                if not isinstance(tx, dict):
                    continue
                amount = tx.get("amount", tx.get("value"))
                if amount and float(amount) > 1000000:
                    alerts.append(FinintAlert(
                        alert_id=hashlib.md5(f"crypto_{time.time()}".encode()).hexdigest()[:10],
                        alert_type="crypto_indicator",
                        severity="medium",
                        timestamp=now_str,
                        title="Large cryptocurrency transaction detected",
                        description=f"Transaction value: ${float(amount):,.0f}",
                        value_usd=float(amount),
                        evidence=tx,
                    ))

        return alerts

    # ── queries ───────────────────────────────────────────────────────

    def get_dark_fleet(self) -> List[DarkFleetVessel]:
        return sorted(self.dark_fleet.values(), key=lambda v: v.risk_score, reverse=True)

    def get_recent_alerts(self, limit: int = 30) -> List[FinintAlert]:
        return self.alerts[-limit:]

    def get_indicators(self) -> Dict[str, EconomicIndicator]:
        return self.indicators

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_detections": self.total_detections,
            "dark_fleet_vessels": len(self.dark_fleet),
            "high_risk_vessels": sum(1 for v in self.dark_fleet.values() if v.risk_score >= 5),
            "economic_indicators": len(self.indicators),
            "anomalous_indicators": sum(1 for i in self.indicators.values() if i.anomaly),
            "total_alerts": len(self.alerts),
        }
