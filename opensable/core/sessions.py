"""
Session Management System

Persistent per-channel conversation history that survives restarts.
Each (channel, user_id) pair gets its own isolated session on disk.

Design:
- No open ports, no external servers — pure local file I/O
- Sessions auto-saved after every message
- Sessions auto-loaded on startup
- Each channel is truly isolated (telegram:123 ≠ discord:123)
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Single message in conversation"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class SessionConfig:
    """Session-specific configuration — per-session overrides"""
    model: Optional[str] = None
    thinking_level: str = "medium"   # off | minimal | low | medium | high | xhigh
    verbose: bool = False
    use_voice: bool = False           # kept as use_voice for compat
    auto_compact: bool = True
    max_history: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


class Session:
    """User conversation session — isolated per (channel, user_id)"""

    def __init__(
        self,
        session_id: str,
        user_id: str,
        channel: str,
        config: Optional[SessionConfig] = None
    ):
        self.id = session_id
        self.user_id = user_id
        self.channel = channel
        self.config = config or SessionConfig()

        self.messages: List[Message] = []
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}

        # Stats
        self.message_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    # ──────────────────────────────────────────────────────
    # Message helpers
    # ──────────────────────────────────────────────────────

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """Append a message and auto-compact when limit is reached."""
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        self.message_count += 1
        self.updated_at = datetime.utcnow()

        if self.config.auto_compact and len(self.messages) > self.config.max_history:
            self._trim_history()

    def clear_messages(self):
        """Clear all messages (keep system ones)."""
        self.messages = [m for m in self.messages if m.role == "system"]
        self.message_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.updated_at = datetime.utcnow()

    def compact_messages(self, keep_recent: int = 20):
        """Keep system messages + last *keep_recent* messages."""
        system = [m for m in self.messages if m.role == "system"]
        recent = [m for m in self.messages if m.role != "system"][-keep_recent:]
        self.messages = system + recent
        self.updated_at = datetime.utcnow()

    def _trim_history(self):
        """Internal trim when auto_compact kicks in."""
        system = [m for m in self.messages if m.role == "system"]
        recent = [m for m in self.messages if m.role != "system"][-(self.config.max_history - len(system)):]
        self.messages = system + recent

    def get_history(self, limit: Optional[int] = None) -> List[dict]:
        """Return conversation history as a list of dicts."""
        messages = self.messages[-limit:] if limit else self.messages
        return [m.to_dict() for m in messages]

    def get_llm_messages(self, limit: int = 20) -> List[dict]:
        """Return recent history in Ollama chat format (role + content only)."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages[-limit:]
            if m.role in ("user", "assistant", "system")
        ]

    def reset(self):
        """Alias kept for back-compat."""
        self.clear_messages()

    async def compact(self) -> str:
        """Public compact method — summarises old messages."""
        if len(self.messages) < 10:
            return "Session too short to compact."
        original = len(self.messages)
        self.compact_messages(keep_recent=20)
        return f"Compacted {original} → {len(self.messages)} messages."

    def update_stats(self, tokens: int = 0, cost: float = 0.0):
        self.total_tokens += tokens
        self.total_cost += cost
        self.updated_at = datetime.utcnow()

    # ──────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "channel": self.channel,
            "config": self.config.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "stats": {
                "message_count": self.message_count,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        session = cls(
            session_id=data["id"],
            user_id=data["user_id"],
            channel=data["channel"],
            config=SessionConfig(**data.get("config", {})),
        )
        session.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.updated_at = datetime.fromisoformat(data["updated_at"])
        session.metadata = data.get("metadata", {})
        stats = data.get("stats", {})
        session.message_count = stats.get("message_count", len(session.messages))
        session.total_tokens = stats.get("total_tokens", 0)
        session.total_cost = stats.get("total_cost", 0.0)
        return session


class SessionManager:
    """
    Manages all active sessions.

    Sessions are stored at ~/.opensable/sessions/<session_id>.json
    Each (channel, user_id) always maps to the same session_id so history
    survives bot restarts.

    No ports, no servers — pure file I/O.
    """

    def __init__(self, config=None):
        self.config = config
        self.active: Dict[str, Session] = {}
        self.sessions_dir = Path.home() / ".opensable" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_timeout = timedelta(hours=48)
        self.total_sessions = 0

        # Load recently-active sessions from disk on start
        self._load_recent()

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def get_or_create_session(
        self,
        user_id: str,
        channel: str,
        session_id: Optional[str] = None,
        config: Optional[SessionConfig] = None,
    ) -> Session:
        """Return existing persistent session or create a new one."""
        sid = session_id or self._make_id(user_id, channel)

        if sid in self.active:
            return self.active[sid]

        # Try disk
        session = self._load(sid)
        if not session:
            session = Session(
                session_id=sid,
                user_id=user_id,
                channel=channel,
                config=config,
            )
            self.total_sessions += 1
            logger.info(f"New session {sid} ({channel}:{user_id})")
        else:
            # Honour caller's config overrides (e.g. default model)
            if config and config.model and not session.config.model:
                session.config.model = config.model
            logger.info(f"Loaded session {sid} ({channel}:{user_id}) — {len(session.messages)} msgs")

        self.active[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        if session_id in self.active:
            return self.active[session_id]
        s = self._load(session_id)
        if s:
            self.active[session_id] = s
        return s

    def get_all_sessions(self) -> List[Session]:
        return list(self.active.values())

    def remove_session(self, session_id: str):
        self.active.pop(session_id, None)

    def cleanup_old_sessions(self):
        """Evict sessions inactive for > session_timeout from memory (keep on disk)."""
        now = datetime.utcnow()
        evict = [sid for sid, s in self.active.items()
                 if now - s.updated_at > self.session_timeout]
        for sid in evict:
            self._save(self.active[sid])
            del self.active[sid]
            logger.debug(f"Evicted session {sid} from memory")

    def save_to_disk(self):
        for s in self.active.values():
            self._save(s)

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _make_id(self, user_id: str, channel: str) -> str:
        return hashlib.sha256(f"{channel}:{user_id}".encode()).hexdigest()[:16]

    def _save(self, session: Session):
        path = self.sessions_dir / f"{session.id}.json"
        try:
            path.write_text(json.dumps(session.to_dict(), indent=2))
        except Exception as e:
            logger.error(f"Failed to save session {session.id}: {e}")

    # Keep old name for compat
    def _save_session(self, session: Session):
        self._save(session)

    def _load(self, session_id: str) -> Optional[Session]:
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return Session.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def _load_recent(self):
        """Load sessions updated within the last 48 h on startup."""
        cutoff = datetime.utcnow() - self.session_timeout
        loaded = 0
        for path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                updated = datetime.fromisoformat(data.get("updated_at", "2000-01-01"))
                if updated > cutoff:
                    s = Session.from_dict(data)
                    self.active[s.id] = s
                    loaded += 1
            except Exception:
                pass
        if loaded:
            logger.info(f"SessionManager: restored {loaded} sessions from disk")

