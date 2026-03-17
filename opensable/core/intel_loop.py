"""
Zunvra Intelligence Loop — Background task that runs the full 26-module
analysis pipeline + autonomous camera pilot on a recurring schedule.

This is the bridge between SableCore boot and the ZunvraIntelSkill.
When Sable starts, this loop:
  1. Initializes ZunvraIntelSkill (26 modules + remote control + camera pilot + broadcaster)
  2. Runs run_cycle() every ZUNVRA_INTEL_INTERVAL seconds (default 300 = 5 min)
  3. Each cycle: fetches live OSINT → runs 26 analysis modules → threat fusion → camera pilot
  4. The camera pilot autonomously drives the Central Intelligence dashboard map

Config (profile.env):
  ZUNVRA_INTEL_ENABLED=true          # Enable the intelligence loop
  ZUNVRA_INTEL_INTERVAL=300          # Seconds between cycles (default 5 min)
  ZUNVRA_BACKEND_URL=http://...      # Central Intelligence backend
  ZUNVRA_API_KEY=<key>               # API key for authenticated endpoints
  ZUNVRA_CAMERA_ENABLED=true         # Enable autonomous camera pilot
  ZUNVRA_CAMERA_DWELL=4.0            # Seconds to dwell on each finding
  ZUNVRA_CAMERA_DELAY=2.5            # Seconds between camera moves
  ZUNVRA_CAMERA_MAX_MOVES=15         # Max camera moves per cycle
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class IntelLoop:
    """Background loop that runs ZunvraIntelSkill.run_cycle() periodically."""

    def __init__(self, agent=None, config=None):
        self.agent = agent
        self.config = config
        self.skill = None
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._last_results: Optional[Dict[str, Any]] = None
        self._last_cycle_time: float = 0.0
        self._total_findings: int = 0

        # Config from env
        self.interval = float(os.environ.get("ZUNVRA_INTEL_INTERVAL", "300"))
        self.backend_url = os.environ.get("ZUNVRA_BACKEND_URL", "http://localhost:5580")
        self.api_key = os.environ.get("ZUNVRA_API_KEY", "")
        self.camera_enabled = os.environ.get("ZUNVRA_CAMERA_ENABLED", "true").lower() in ("true", "1", "yes")
        self.camera_dwell = float(os.environ.get("ZUNVRA_CAMERA_DWELL", "4.0"))
        self.camera_delay = float(os.environ.get("ZUNVRA_CAMERA_DELAY", "2.5"))
        self.camera_max_moves = int(os.environ.get("ZUNVRA_CAMERA_MAX_MOVES", "15"))

    async def start(self):
        """Initialize the skill and start the recurring loop."""
        self.running = True
        logger.info("🛰️  Intel Loop starting — initializing ZunvraIntelSkill...")

        try:
            from opensable.skills.zunvra_intel import ZunvraIntelSkill

            self.skill = ZunvraIntelSkill(
                base_url=self.backend_url,
                data_dir=os.environ.get("ZUNVRA_INTEL_DATA_DIR", "data/zunvra_intel"),
                enable_llm=False,  # Pure algorithmic — no LLM needed for analysis
            )
            await self.skill.initialize()

            # Configure camera pilot
            if self.skill.camera_pilot and self.camera_enabled:
                self.skill.camera_pilot.move_delay = self.camera_delay
                self.skill.camera_pilot.dwell_default = self.camera_dwell
                self.skill.camera_pilot.max_moves = self.camera_max_moves
                logger.info(
                    "🎥 Camera pilot armed (dwell=%.1fs, delay=%.1fs, max=%d moves)",
                    self.camera_dwell, self.camera_delay, self.camera_max_moves,
                )
            elif not self.camera_enabled:
                # Disable camera by removing the pilot reference
                self.skill.camera_pilot = None
                logger.info("🎥 Camera pilot disabled (ZUNVRA_CAMERA_ENABLED=false)")

            logger.info(
                "🛰️  ZunvraIntelSkill v%s initialized — 26 modules + remote + camera + broadcaster",
                self.skill.__class__.__module__.rsplit(".", 1)[0]
                and getattr(__import__("opensable.skills.zunvra_intel", fromlist=["__version__"]), "__version__", "?"),
            )

        except Exception as e:
            logger.error("🛰️  Intel Loop failed to initialize: %s", e, exc_info=True)
            self.running = False
            return

        # Run first cycle immediately, then loop
        await self._loop()

    async def _loop(self):
        """Main loop: run_cycle() → sleep → repeat."""
        while self.running:
            try:
                t0 = time.time()
                logger.info("🛰️  Intel cycle #%d starting...", self._cycle_count)

                results = await self.skill.run_cycle()
                elapsed = time.time() - t0
                self._last_results = results
                self._last_cycle_time = elapsed
                self._cycle_count += 1

                # Count findings
                findings = sum(
                    v for k, v in results.items()
                    if isinstance(v, int) and k != "cycle"
                )
                self._total_findings += findings

                # Log summary
                threat_level = results.get("threat_level", "?")
                threat_name = results.get("threat_name", "?")
                camera_moves = results.get("camera_moves", 0)

                logger.info(
                    "🛰️  Intel cycle #%d complete in %.1fs — "
                    "%d findings, DEFCON %s %s, %d camera moves",
                    self._cycle_count - 1, elapsed, findings,
                    threat_level, threat_name, camera_moves,
                )

                # Expose results to agent memory if available
                if self.agent and hasattr(self.agent, "memory"):
                    try:
                        self.agent.memory.store(
                            "intel_cycle",
                            {
                                "cycle": self._cycle_count - 1,
                                "findings": findings,
                                "threat_level": threat_level,
                                "threat_name": threat_name,
                                "elapsed": round(elapsed, 2),
                                "camera_moves": camera_moves,
                                "regions": results.get("camera_regions", []),
                            },
                        )
                    except Exception:
                        pass  # Memory not critical

            except Exception as e:
                logger.error("🛰️  Intel cycle #%d failed: %s", self._cycle_count, e, exc_info=True)

            # Wait for next cycle
            if self.running:
                logger.info("🛰️  Next intel cycle in %.0fs", self.interval)
                try:
                    await asyncio.sleep(self.interval)
                except asyncio.CancelledError:
                    break

    async def stop(self):
        """Gracefully stop the loop and shut down the skill."""
        self.running = False
        if self.skill:
            try:
                await self.skill.shutdown()
            except Exception:
                pass
        logger.info(
            "🛰️  Intel Loop stopped — %d cycles, %d total findings",
            self._cycle_count, self._total_findings,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return current status for monitoring/API."""
        return {
            "running": self.running,
            "cycles_completed": self._cycle_count,
            "total_findings": self._total_findings,
            "last_cycle_time": self._last_cycle_time,
            "interval": self.interval,
            "camera_enabled": self.camera_enabled,
            "last_threat_level": self._last_results.get("threat_level") if self._last_results else None,
            "last_threat_name": self._last_results.get("threat_name") if self._last_results else None,
            "last_camera_moves": self._last_results.get("camera_moves", 0) if self._last_results else 0,
            "last_camera_regions": self._last_results.get("camera_regions", []) if self._last_results else [],
        }
