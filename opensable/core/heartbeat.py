"""
Proactive Heartbeat System

Periodic checks that run autonomously:
- Email scanning
- Calendar reminders
- System health checks
- Proactive suggestions
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """
    Manages periodic proactive checks.
    """

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.running = False
        self.interval = getattr(config, "heartbeat_interval", 1800)  # 30 min default
        self.heartbeat_file = Path.home() / ".opensable" / "HEARTBEAT.md"
        self.last_check = datetime.now()

        # Heartbeat checklist
        self.checks: List = []
        self.active_hours_start = getattr(config, "heartbeat_active_start", "08:00")
        self.active_hours_end = getattr(config, "heartbeat_active_end", "23:00")

    def register_check(self, check_func, name: str):
        """Register a proactive check function"""
        self.checks.append({"func": check_func, "name": name})
        logger.info(f"✅ Registered heartbeat check: {name}")

    async def start(self):
        """Start heartbeat runner"""
        logger.info(f"💓 Starting heartbeat (interval: {self.interval}s)")
        self.running = True

        # Start heartbeat loop
        asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """Stop heartbeat"""
        logger.info("Stopping heartbeat...")
        self.running = False

    def _is_within_active_hours(self) -> bool:
        """Check if current time is within active hours"""
        now = datetime.now().time()
        start = datetime.strptime(self.active_hours_start, "%H:%M").time()
        end = datetime.strptime(self.active_hours_end, "%H:%M").time()
        return start <= now <= end

    async def _heartbeat_loop(self):
        """Main heartbeat loop - runs every interval"""
        while self.running:
            try:
                # Skip if outside active hours
                if not self._is_within_active_hours():
                    logger.debug("Outside active hours, skipping heartbeat")
                    await asyncio.sleep(60)  # Check again in 1 minute
                    continue

                # Run heartbeat check
                await self._run_heartbeat()

                # Wait for next interval
                await asyncio.sleep(self.interval)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(self.interval)

    async def _run_heartbeat(self):
        """Execute heartbeat check"""
        logger.info("💓 Running heartbeat check...")

        # Read HEARTBEAT.md if exists
        checklist = await self._read_heartbeat_file()

        # Run all registered checks
        alerts = []
        for check in self.checks:
            try:
                result = await check["func"]()
                if result and result.get("alert"):
                    alerts.append(
                        {
                            "check": check["name"],
                            "message": result["message"],
                            "priority": result.get("priority", "normal"),
                        }
                    )
            except Exception as e:
                logger.error(f"Check '{check['name']}' failed: {e}")

        # If there are alerts, send proactive message
        if alerts:
            await self._send_heartbeat_alerts(alerts)
        else:
            logger.info("✅ Heartbeat: All checks OK (HEARTBEAT_OK)")

        self.last_check = datetime.now()

    async def _read_heartbeat_file(self) -> Optional[str]:
        """Read HEARTBEAT.md checklist"""
        if not self.heartbeat_file.exists():
            return None

        try:
            content = self.heartbeat_file.read_text()
            # Parse markdown checklist
            return content
        except Exception as e:
            logger.error(f"Failed to read HEARTBEAT.md: {e}")
            return None

    async def _send_heartbeat_alerts(self, alerts: List[Dict]):
        """Send proactive alerts to user via active interfaces"""
        logger.info(f"🔔 Heartbeat: {len(alerts)} alerts")

        # Format alert message
        message = "💓 **Heartbeat Check**\n\n"
        for alert in alerts:
            priority_emoji = "🔴" if alert["priority"] == "urgent" else "🟡"
            message += f"{priority_emoji} **{alert['check']}**: {alert['message']}\n"

        # Try to send via the agent's active interfaces
        sent = False
        # Check if agent has a Telegram bot reference
        if hasattr(self.agent, "_telegram_notify"):
            try:
                await self.agent._telegram_notify(message)
                sent = True
            except Exception as e:
                logger.debug(f"Telegram notify failed: {e}")

        if not sent:
            logger.info(f"Heartbeat alert (no active notification channel):\n{message}")


# ── Built-in Heartbeat Checks ──────────────────────────────────────


async def check_system_health() -> Dict[str, Any]:
    """Check system health (CPU, memory, disk)"""
    import psutil

    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    if cpu > 90 or memory > 90 or disk > 90:
        return {
            "alert": True,
            "message": f"System resources high: CPU {cpu}%, RAM {memory}%, Disk {disk}%",
            "priority": "urgent",
        }

    return {"alert": False}


async def check_pending_tasks() -> Dict[str, Any]:
    """Check for pending goals nearing their deadline"""
    try:
        state_file = Path.home() / ".opensable" / "goals.json"
        if not state_file.exists():
            return {"alert": False}
        data = json.loads(state_file.read_text())
        active = [g for g in data.get("goals", []) if g.get("status") == "active"]
        overdue = []
        now = datetime.now().isoformat()
        for g in active:
            dl = g.get("deadline")
            if dl and dl < now:
                overdue.append(g.get("description", g.get("goal_id", "?")))
        if overdue:
            return {
                "alert": True,
                "message": f"{len(overdue)} overdue goal(s): {', '.join(overdue[:3])}",
                "priority": "urgent",
            }
        if len(active) > 5:
            return {
                "alert": True,
                "message": f"{len(active)} active goals — consider prioritizing",
                "priority": "normal",
            }
        return {"alert": False}
    except Exception:
        return {"alert": False}


async def check_idle_time() -> Dict[str, Any]:
    """Check if no user interaction for a long time"""
    try:
        session_dir = Path.home() / ".opensable" / "sessions"
        if not session_dir.exists():
            return {"alert": False}
        latest = None
        for f in session_dir.glob("*.json"):
            data = json.loads(f.read_text())
            msgs = data.get("messages", [])
            if msgs:
                ts = msgs[-1].get("timestamp")
                if ts and (latest is None or ts > latest):
                    latest = ts
        if latest:
            from datetime import datetime as dt

            last = dt.fromisoformat(latest)
            hours = (datetime.now() - last).total_seconds() / 3600
            if hours > 12:
                return {
                    "alert": True,
                    "message": f"No interaction in {int(hours)}h — everything ok?",
                    "priority": "normal",
                }
        return {"alert": False}
    except Exception:
        return {"alert": False}
