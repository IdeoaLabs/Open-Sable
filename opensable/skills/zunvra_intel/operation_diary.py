"""
Zunvra Intelligence,  Operation Diary

Sable's autonomous operation journal.  Every analysis cycle, Sable writes
what it observed, its assessment, predictions, and threat opinions.

Entries are:
 • Pushed live via WebSocket to the dashboard
 • Persisted to a JSON file so they survive restarts
 • Available via API for the frontend diary panel

The diary is NOT just a log,  it's Sable's *opinion*.  The LLM generates
analytical commentary based on what the camera pilot found.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Persistent diary storage path
DIARY_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DIARY_FILE = DIARY_DIR / "operation_diary.json"
DOSSIER_FILE = DIARY_DIR / "operation_dossier.json"


# ── Data models ──────────────────────────────────────────────────────

@dataclass
class DiaryEntry:
    """A single diary entry,  Sable's autonomous analysis note."""
    timestamp: str                # ISO 8601
    title: str                    # Short heading
    content: str                  # Full analysis / opinion / observation
    severity: str = "info"        # info | warning | critical | success
    lat: float = 0.0
    lng: float = 0.0
    domain: str = ""              # Intelligence domain
    tags: List[str] = field(default_factory=list)
    cycle: int = 0                # Which cycle generated this
    opinion: str = ""             # Sable's personal take
    prediction: str = ""          # What Sable expects next


@dataclass
class DossierEntry:
    """A dossier capture,  structured intel snapshot."""
    timestamp: str
    category: str                 # cctv | aircraft | ship | attack | cyber | nuclear | custom
    title: str
    severity: str = "info"
    lat: float = 0.0
    lng: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    capture_url: str = ""         # Optional screenshot / stream URL
    entities: List[Dict] = field(default_factory=list)  # Related entities


# ── Diary Manager ────────────────────────────────────────────────────

class OperationDiary:
    """Manages the persistent operation diary and dossier."""

    def __init__(self, max_entries: int = 500, max_dossier: int = 200):
        self.max_entries = max_entries
        self.max_dossier = max_dossier
        self._diary: List[DiaryEntry] = []
        self._dossier: List[DossierEntry] = []
        self._load()

    # ── Diary operations ─────────────────────────────────────────────

    def add_entry(
        self,
        title: str,
        content: str,
        severity: str = "info",
        lat: float = 0.0,
        lng: float = 0.0,
        domain: str = "",
        tags: Optional[List[str]] = None,
        cycle: int = 0,
        opinion: str = "",
        prediction: str = "",
    ) -> DiaryEntry:
        """Add a new diary entry and persist."""
        entry = DiaryEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            title=title,
            content=content,
            severity=severity,
            lat=lat,
            lng=lng,
            domain=domain,
            tags=tags or [],
            cycle=cycle,
            opinion=opinion,
            prediction=prediction,
        )
        self._diary.append(entry)

        # Trim old entries
        if len(self._diary) > self.max_entries:
            self._diary = self._diary[-self.max_entries:]

        self._save_diary()
        return entry

    def get_entries(self, limit: int = 50, severity: Optional[str] = None) -> List[Dict]:
        """Get recent diary entries as dicts."""
        entries = self._diary
        if severity:
            entries = [e for e in entries if e.severity == severity]
        return [asdict(e) for e in entries[-limit:]]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the diary."""
        if not self._diary:
            return {"total": 0, "domains": {}, "severities": {}}

        domains: Dict[str, int] = {}
        severities: Dict[str, int] = {}
        for e in self._diary:
            domains[e.domain] = domains.get(e.domain, 0) + 1
            severities[e.severity] = severities.get(e.severity, 0) + 1

        return {
            "total": len(self._diary),
            "first": self._diary[0].timestamp,
            "last": self._diary[-1].timestamp,
            "domains": domains,
            "severities": severities,
        }

    # ── Dossier operations ───────────────────────────────────────────

    def add_dossier(
        self,
        category: str,
        title: str,
        severity: str = "info",
        lat: float = 0.0,
        lng: float = 0.0,
        data: Optional[Dict] = None,
        capture_url: str = "",
        entities: Optional[List[Dict]] = None,
    ) -> DossierEntry:
        """Add a new dossier capture entry."""
        entry = DossierEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
            title=title,
            severity=severity,
            lat=lat,
            lng=lng,
            data=data or {},
            capture_url=capture_url,
            entities=entities or [],
        )
        self._dossier.append(entry)

        if len(self._dossier) > self.max_dossier:
            self._dossier = self._dossier[-self.max_dossier:]

        self._save_dossier()
        return entry

    def get_dossier(self, limit: int = 50, category: Optional[str] = None) -> List[Dict]:
        """Get recent dossier entries."""
        entries = self._dossier
        if category:
            entries = [e for e in entries if e.category == category]
        return [asdict(e) for e in entries[-limit:]]

    def get_dossier_summary(self) -> Dict[str, Any]:
        """Get dossier statistics."""
        if not self._dossier:
            return {"total": 0, "categories": {}}

        cats: Dict[str, int] = {}
        for e in self._dossier:
            cats[e.category] = cats.get(e.category, 0) + 1

        return {
            "total": len(self._dossier),
            "categories": cats,
            "first": self._dossier[0].timestamp,
            "last": self._dossier[-1].timestamp,
        }

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self):
        """Load diary and dossier from disk."""
        try:
            if DIARY_FILE.exists():
                with open(DIARY_FILE, "r") as f:
                    data = json.load(f)
                self._diary = [DiaryEntry(**e) for e in data[-self.max_entries:]]
                logger.info("Loaded %d diary entries from disk", len(self._diary))
        except Exception as e:
            logger.warning("Failed to load diary: %s", e)
            self._diary = []

        try:
            if DOSSIER_FILE.exists():
                with open(DOSSIER_FILE, "r") as f:
                    data = json.load(f)
                self._dossier = [DossierEntry(**e) for e in data[-self.max_dossier:]]
                logger.info("Loaded %d dossier entries from disk", len(self._dossier))
        except Exception as e:
            logger.warning("Failed to load dossier: %s", e)
            self._dossier = []

    def _save_diary(self):
        """Persist diary to disk."""
        try:
            DIARY_DIR.mkdir(parents=True, exist_ok=True)
            with open(DIARY_FILE, "w") as f:
                json.dump([asdict(e) for e in self._diary], f, indent=2)
        except Exception as e:
            logger.warning("Failed to save diary: %s", e)

    def _save_dossier(self):
        """Persist dossier to disk."""
        try:
            DIARY_DIR.mkdir(parents=True, exist_ok=True)
            with open(DOSSIER_FILE, "w") as f:
                json.dump([asdict(e) for e in self._dossier], f, indent=2)
        except Exception as e:
            logger.warning("Failed to save dossier: %s", e)


# ── Singleton ────────────────────────────────────────────────────────

_instance: Optional[OperationDiary] = None

def get_diary() -> OperationDiary:
    """Get the global OperationDiary instance."""
    global _instance
    if _instance is None:
        _instance = OperationDiary()
    return _instance
