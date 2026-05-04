"""
#5,  Predictive World Model

Builds a living model of global state from sequential Zunvra snapshots.
Tracks entities, relationships, and movement patterns. Uses statistical
trend analysis + LLM forecasting to predict future state.

Output: FORECAST panel with probability timelines.
  "Probability of increased South China Sea activity within 72h: 67%"
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


@dataclass
class WorldState:
    """Aggregated state at one point in time."""
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    # metrics: flights_count, military_count, ships_count, earthquakes_count,
    # fires_count, gdelt_count, cyber_count, jamming_count, carriers_count

    @classmethod
    def from_snapshot(cls, snap: IntelSnapshot) -> "WorldState":
        return cls(
            timestamp=snap.timestamp or datetime.now(timezone.utc).isoformat(),
            metrics={
                "flights": len(snap.flights),
                "military": len(snap.military_flights),
                "ships": len(snap.ships),
                "satellites": len(snap.satellites),
                "earthquakes": len(snap.earthquakes),
                "fires": len(snap.fires),
                "gdelt_events": len(snap.gdelt_events),
                "cyber_threats": len(snap.cyber_threats),
                "gps_jamming": len(snap.gps_jamming),
                "carriers": len(snap.carriers),
                "conflicts": len(snap.conflicts),
                "internet_outages": len(snap.internet_outages),
                "ransomware": len(snap.ransomware),
            },
        )


@dataclass
class Trend:
    """Statistical trend for a single metric."""
    metric: str
    direction: str  # rising | falling | stable
    slope: float  # change per observation
    current_value: float
    mean_value: float
    std_dev: float
    observations: int
    anomaly: bool = False  # > 2 sigma from mean

    def to_text(self) -> str:
        icon = {"rising": "↑", "falling": "↓", "stable": "→"}.get(self.direction, "?")
        anom = " ⚠ ANOMALY" if self.anomaly else ""
        return f"{self.metric}: {self.current_value:.0f} {icon} (mean {self.mean_value:.1f} ±{self.std_dev:.1f}){anom}"


@dataclass
class Forecast:
    """A predictive forecast for one dimension of the world."""
    forecast_id: str
    metric: str
    timeframe: str  # next_hour | next_6h | next_24h | next_72h
    predicted_value: float
    predicted_direction: str
    confidence: float
    reasoning: str
    timestamp: str
    current_value: float = 0.0

    def to_text(self) -> str:
        delta = self.predicted_value - self.current_value
        sign = "+" if delta >= 0 else ""
        return (
            f"FORECAST [{self.timeframe}] {self.metric}: "
            f"{self.current_value:.0f} → {self.predicted_value:.0f} ({sign}{delta:.0f}) "
            f"[{self.confidence:.0%} confidence]\n"
            f"  {self.reasoning}"
        )


# ---------------------------------------------------------------------------
# Predictive World Model
# ---------------------------------------------------------------------------

class PredictiveWorldModel:
    """
    Builds a living model of global state from Zunvra snapshot history.
    Computes trends, detects anomalies, and generates forecasts.
    """

    MAX_HISTORY = 500  # keep last N states

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "world_model_state.json"

        self.history: List[WorldState] = []
        self.trends: Dict[str, Trend] = {}
        self.forecasts: List[Forecast] = []
        self.total_observations = 0
        self._load_state()

    # ── ingest ────────────────────────────────────────────────────────

    def observe(self, snapshot: IntelSnapshot):
        """Record a new observation from a Zunvra snapshot."""
        state = WorldState.from_snapshot(snapshot)
        self.history.append(state)
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]
        self.total_observations += 1
        self._compute_trends()
        self._save_state()

    # ── trend computation ─────────────────────────────────────────────

    def _compute_trends(self):
        """Recompute trends from history."""
        if len(self.history) < 3:
            return

        all_metrics = self.history[-1].metrics.keys()

        for metric in all_metrics:
            values = [s.metrics.get(metric, 0) for s in self.history[-50:]]  # window

            if len(values) < 3:
                continue

            current = values[-1]
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            std_dev = math.sqrt(variance) if variance > 0 else 0.01

            # Linear slope (simple)
            n = len(values)
            x_mean = (n - 1) / 2
            numerator = sum((i - x_mean) * (v - mean_val) for i, v in enumerate(values))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator != 0 else 0

            if abs(slope) < 0.5:
                direction = "stable"
            elif slope > 0:
                direction = "rising"
            else:
                direction = "falling"

            anomaly = abs(current - mean_val) > 2 * std_dev

            self.trends[metric] = Trend(
                metric=metric,
                direction=direction,
                slope=slope,
                current_value=current,
                mean_value=mean_val,
                std_dev=std_dev,
                observations=len(values),
                anomaly=anomaly,
            )

    # ── forecasting ───────────────────────────────────────────────────

    async def forecast(self, llm=None) -> List[Forecast]:
        """Generate forecasts for all tracked metrics."""
        if len(self.history) < 5:
            return []

        now = datetime.now(timezone.utc).isoformat()
        new_forecasts: List[Forecast] = []

        for metric, trend in self.trends.items():
            # Statistical forecasts (extrapolation)
            for timeframe, steps in [("next_hour", 1), ("next_6h", 6), ("next_24h", 24), ("next_72h", 72)]:
                predicted = max(0, trend.current_value + trend.slope * steps)
                confidence = max(0.1, 0.8 - (steps * 0.01) - (trend.std_dev / max(trend.mean_value, 1)) * 0.3)

                forecast = Forecast(
                    forecast_id=f"{metric}_{timeframe}_{self.total_observations}",
                    metric=metric,
                    timeframe=timeframe,
                    predicted_value=predicted,
                    predicted_direction=trend.direction,
                    confidence=min(0.95, confidence),
                    reasoning=f"Linear extrapolation: slope={trend.slope:.2f}/obs, current={trend.current_value:.0f}",
                    timestamp=now,
                    current_value=trend.current_value,
                )
                new_forecasts.append(forecast)

        # LLM-enhanced forecasts for anomalous metrics
        if llm:
            anomalous = [m for m, t in self.trends.items() if t.anomaly]
            if anomalous:
                try:
                    llm_forecasts = await self._llm_forecast(llm, anomalous)
                    new_forecasts.extend(llm_forecasts)
                except Exception as e:
                    logger.warning("LLM forecast failed: %s", e)

        self.forecasts = new_forecasts
        self._save_state()
        return new_forecasts

    async def _llm_forecast(self, llm, anomalous_metrics: List[str]) -> List[Forecast]:
        """Use LLM to reason about anomalous trends."""
        trend_text = "\n".join(
            self.trends[m].to_text() for m in anomalous_metrics if m in self.trends
        )
        history_summary = []
        for s in self.history[-5:]:
            history_summary.append(json.dumps(s.metrics))

        prompt = (
            "You are a geopolitical intelligence forecaster for the Zunvra OSINT platform.\n\n"
            "ANOMALOUS TRENDS DETECTED:\n"
            f"{trend_text}\n\n"
            "Recent state history (last 5 observations):\n"
            + "\n".join(history_summary) + "\n\n"
            "For each anomalous metric, provide a 72-hour forecast.\n"
            "Return JSON array:\n"
            "[\n"
            '  {"metric": "...", "predicted_direction": "rising|falling|stable",\n'
            '   "confidence": 0.0-1.0, "reasoning": "brief explanation"}\n'
            "]"
        )

        now = datetime.now(timezone.utc).isoformat()
        try:
            raw = await llm.chat_raw(prompt, max_tokens=400)
            import re
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if not m:
                return []
            items = json.loads(m.group())
            forecasts = []
            for item in items[:len(anomalous_metrics)]:
                metric = item.get("metric", "")
                if metric in self.trends:
                    t = self.trends[metric]
                    forecasts.append(Forecast(
                        forecast_id=f"llm_{metric}_72h_{self.total_observations}",
                        metric=metric,
                        timeframe="next_72h",
                        predicted_value=t.current_value,  # LLM gives direction, not value
                        predicted_direction=item.get("predicted_direction", "stable"),
                        confidence=float(item.get("confidence", 0.5)),
                        reasoning=f"[LLM] {item.get('reasoning', '')}",
                        timestamp=now,
                        current_value=t.current_value,
                    ))
            return forecasts
        except Exception:
            return []

    # ── queries ───────────────────────────────────────────────────────

    def get_anomalies(self) -> List[Trend]:
        return [t for t in self.trends.values() if t.anomaly]

    def get_rising_threats(self) -> List[Trend]:
        return [t for t in self.trends.values() if t.direction == "rising" and t.slope > 1.0]

    def get_trend_report(self) -> str:
        """Human-readable trend report."""
        lines = [f"WORLD MODEL,  {self.total_observations} observations, {len(self.trends)} metrics"]
        for t in sorted(self.trends.values(), key=lambda x: abs(x.slope), reverse=True):
            lines.append(f"  {t.to_text()}")
        if self.forecasts:
            lines.append("")
            lines.append("FORECASTS:")
            # Only show 72h forecasts
            for f in self.forecasts:
                if f.timeframe == "next_72h":
                    lines.append(f"  {f.to_text()}")
        return "\n".join(lines)

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "history": [asdict(s) for s in self.history[-self.MAX_HISTORY:]],
                "total_observations": self.total_observations,
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save world model state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                for sd in state.get("history", []):
                    self.history.append(WorldState(**sd))
                self.total_observations = state.get("total_observations", 0)
                self._compute_trends()
                logger.info("Loaded %d world states, %d observations",
                           len(self.history), self.total_observations)
        except Exception as e:
            logger.warning("Failed to load world model state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_observations": self.total_observations,
            "history_length": len(self.history),
            "metrics_tracked": len(self.trends),
            "anomalies": len(self.get_anomalies()),
            "rising_threats": len(self.get_rising_threats()),
        }
