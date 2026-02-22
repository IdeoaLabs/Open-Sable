"""
Session Management - Handle conversation sessions across channels
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Single message in a session"""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class SessionConfig:
    """Session configuration"""

    model: str = "llama3.1:8b"
    thinking_level: str = "medium"  # off, low, medium, high, xhigh
    verbose: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048
    use_voice: bool = False
    auto_respond: bool = True

    def to_dict(self):
        return asdict(self)


class Session:
    """Manages a conversation session"""

    def __init__(
        self, session_id: str, channel: str, user_id: str, config: Optional[SessionConfig] = None
    ):
        self.id = session_id
        self.channel = channel
        self.user_id = user_id
        self.config = config or SessionConfig()
        self.messages: List[Message] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.metadata: Dict[str, Any] = {}
        self.state = "active"  # active, paused, archived

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to session"""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
        logger.debug(f"Added {role} message to session {self.id}")

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from session"""
        if limit:
            return self.messages[-limit:]
        return self.messages

    def get_llm_messages(self, limit: Optional[int] = None) -> List[dict]:
        """Return messages as list of {role, content} dicts for LLM APIs."""
        msgs = self.get_messages(limit=limit)
        return [{"role": m.role, "content": m.content} for m in msgs]

    def clear_messages(self):
        """Clear all messages (reset session)"""
        self.messages = []
        self.updated_at = datetime.now().isoformat()
        logger.info(f"Cleared session {self.id}")

    def compact_messages(self, keep_recent: int = 10):
        """Compact old messages with summary"""
        if len(self.messages) <= keep_recent:
            return

        # Keep recent messages
        recent = self.messages[-keep_recent:]

        # Create summary
        summary = Message(
            role="system",
            content=f"[Previous conversation summary: {len(self.messages) - keep_recent} messages compacted]",
            metadata={"type": "summary", "original_count": len(self.messages) - keep_recent},
        )

        self.messages = [summary] + recent
        self.updated_at = datetime.now().isoformat()
        logger.info(f"Compacted session {self.id}: kept {keep_recent} recent messages")

    def update_config(self, **kwargs):
        """Update session configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "id": self.id,
            "channel": self.channel,
            "user_id": self.user_id,
            "config": self.config.to_dict(),
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "state": self.state,
            "message_count": len(self.messages),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Session":
        """Deserialize from dictionary"""
        session = cls(
            session_id=data["id"],
            channel=data["channel"],
            user_id=data["user_id"],
            config=SessionConfig(**data.get("config", {})),
        )
        session.messages = [Message(**msg) for msg in data.get("messages", [])]
        session.created_at = data.get("created_at", session.created_at)
        session.updated_at = data.get("updated_at", session.updated_at)
        session.metadata = data.get("metadata", {})
        session.state = data.get("state", "active")
        return session


class SessionManager:
    """Manages all sessions"""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or (Path.home() / ".opensable" / "sessions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, Session] = {}
        self._load_sessions()

    def _load_sessions(self):
        """Load sessions from disk"""
        try:
            for session_file in self.storage_dir.glob("*.json"):
                try:
                    data = json.loads(session_file.read_text())
                    session = Session.from_dict(data)
                    self.sessions[session.id] = session
                    logger.debug(f"Loaded session: {session.id}")
                except Exception as e:
                    logger.error(f"Error loading session {session_file}: {e}")

            logger.info(f"Loaded {len(self.sessions)} sessions from disk")
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")

    def _save_session(self, session: Session):
        """Save session to disk"""
        try:
            session_file = self.storage_dir / f"{session.id}.json"
            session_file.write_text(json.dumps(session.to_dict(), indent=2))
            logger.debug(f"Saved session: {session.id}")
        except Exception as e:
            logger.error(f"Error saving session {session.id}: {e}")

    def create_session(
        self,
        channel: str,
        user_id: str,
        session_id: Optional[str] = None,
        config: Optional[SessionConfig] = None,
    ) -> Session:
        """Create new session"""
        session_id = session_id or str(uuid.uuid4())

        session = Session(session_id=session_id, channel=channel, user_id=user_id, config=config)

        self.sessions[session_id] = session
        self._save_session(session)

        logger.info(f"Created session {session_id} for {channel}:{user_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        return self.sessions.get(session_id)

    def get_or_create_session(
        self, channel: str, user_id: str, config: Optional[SessionConfig] = None
    ) -> Session:
        """Get existing session or create new one"""
        # Try to find existing session
        for session in self.sessions.values():
            if (
                session.channel == channel
                and session.user_id == user_id
                and session.state == "active"
            ):
                return session

        # Create new session
        return self.create_session(channel, user_id, config=config)

    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if session_id in self.sessions:
            # Remove from memory
            del self.sessions[session_id]

            # Remove from disk
            session_file = self.storage_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()

            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def list_sessions(
        self,
        channel: Optional[str] = None,
        user_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[Session]:
        """List sessions with optional filters"""
        sessions = list(self.sessions.values())

        if channel:
            sessions = [s for s in sessions if s.channel == channel]
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if state:
            sessions = [s for s in sessions if s.state == state]

        return sessions

    def reset_session(self, session_id: str) -> bool:
        """Reset session (clear messages)"""
        session = self.get_session(session_id)
        if session:
            session.clear_messages()
            self._save_session(session)
            return True
        return False

    def compact_session(self, session_id: str, keep_recent: int = 10) -> bool:
        """Compact session messages"""
        session = self.get_session(session_id)
        if session:
            session.compact_messages(keep_recent)
            self._save_session(session)
            return True
        return False

    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session configuration"""
        session = self.get_session(session_id)
        if session:
            session.update_config(**kwargs)
            self._save_session(session)
            return True
        return False

    def add_message(
        self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None
    ) -> bool:
        """Add message to session"""
        session = self.get_session(session_id)
        if session:
            session.add_message(role, content, metadata)
            self._save_session(session)
            return True
        return False

    def cleanup_old_sessions(self, days: int = 30):
        """Archive or delete old sessions"""
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=days)

        for session in list(self.sessions.values()):
            updated = datetime.fromisoformat(session.updated_at)
            if updated < cutoff:
                session.state = "archived"
                self._save_session(session)
                logger.info(f"Archived old session: {session.id}")
