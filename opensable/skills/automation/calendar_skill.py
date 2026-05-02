"""
Calendar skill for Open-Sable,  local JSON-based calendar (no Google API dependency)
"""

import json
import logging
import os
import uuid
from typing import List, Dict, Any
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CALENDAR_FILE = Path(os.environ.get("_SABLE_DATA_DIR", "data")) / "calendar.json"


def _normalize_event_datetime(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_event_datetime(value: str) -> datetime:
    return _normalize_event_datetime(datetime.fromisoformat(value))


class CalendarSkill:
    """Local calendar stored in data/calendar.json"""

    def __init__(self, config):
        self.config = config
        self._ready = False

    # ── init ──────────────────────────────────────────────────────────────

    async def initialize(self):
        """Ensure the calendar file exists."""
        if not self.config.calendar_enabled:
            logger.info("Calendar skill disabled")
            return

        try:
            _CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)
            if not _CALENDAR_FILE.exists():
                _CALENDAR_FILE.write_text("[]", encoding="utf-8")
            self._ready = True
            logger.info("Calendar skill ready (local JSON)")
        except Exception as e:
            logger.error(f"Failed to initialize Calendar: {e}")
            logger.info("Calendar skill will run in demo mode")

    # ── helpers ───────────────────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(_CALENDAR_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save(self, events: List[Dict[str, Any]]):
        _CALENDAR_FILE.write_text(
            json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── public API ────────────────────────────────────────────────────────

    async def list_events(
        self, days_ahead: int = 7, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """List upcoming calendar events."""
        if not self._ready:
            return self._demo_events()

        try:
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(days=days_ahead)
            events = self._load()

            upcoming = []
            for ev in events:
                try:
                    start_dt = _parse_event_datetime(ev["start"])
                except (KeyError, ValueError):
                    continue
                if now <= start_dt <= cutoff:
                    upcoming.append((start_dt, ev))

            upcoming.sort(key=lambda item: item[0])
            return [event for _, event in upcoming[:max_results]]

        except Exception as e:
            logger.error(f"Failed to list events: {e}")
            return self._demo_events()

    async def add_event(
        self,
        summary: str,
        start_time: str,
        duration_minutes: int = 60,
        description: str = "",
        location: str = "",
    ) -> bool:
        """Add a new calendar event."""
        if not self._ready:
            logger.info(f"[DEMO] Would add event: {summary} at {start_time}")
            return True

        try:
            start_dt = _parse_event_datetime(start_time)
            end_dt = start_dt + timedelta(minutes=duration_minutes)

            event = {
                "id": uuid.uuid4().hex[:12],
                "summary": summary,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "description": description,
                "location": location,
            }

            events = self._load()
            events.append(event)
            self._save(events)

            logger.info(f"Added event: {summary}")
            return True

        except Exception as e:
            logger.error(f"Failed to add event: {e}")
            return False

    async def delete_event(self, event_id: str) -> bool:
        """Delete a calendar event by id."""
        if not self._ready:
            return True

        try:
            events = self._load()
            events = [e for e in events if e.get("id") != event_id]
            self._save(events)
            logger.info(f"Deleted event: {event_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete event: {e}")
            return False

    # ── demo fallback ─────────────────────────────────────────────────────

    def _demo_events(self) -> List[Dict[str, Any]]:
        """Return demo events when not ready."""
        now = datetime.now(timezone.utc)
        return [
            {
                "id": "demo1",
                "summary": "Team Meeting",
                "start": (now + timedelta(hours=2)).isoformat(),
                "description": "Weekly sync",
                "location": "Zoom",
            },
            {
                "id": "demo2",
                "summary": "Lunch with Sarah",
                "start": (now + timedelta(days=1, hours=12)).isoformat(),
                "description": "",
                "location": "Downtown Cafe",
            },
        ]
