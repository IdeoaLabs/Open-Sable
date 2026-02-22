"""
Open-Sable Analytics and Monitoring

Tracks usage metrics, performance, errors, and user interactions.
Provides insights for improving the agent and understanding usage patterns.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
from collections import defaultdict, Counter
import time

from opensable.core.config import Config

logger = logging.getLogger(__name__)


class MetricEvent:
    """Represents a metric event"""

    def __init__(
        self, event_type: str, data: Dict[str, Any] = None, timestamp: Optional[datetime] = None
    ):
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = timestamp or datetime.utcnow()
        self.event_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique event ID"""
        import hashlib

        content = f"{self.event_type}{self.timestamp}{id(self)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricEvent":
        """Create from dictionary"""
        event = cls(
            event_type=data["event_type"],
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
        event.event_id = data.get("event_id", event.event_id)
        return event


class Analytics:
    """Analytics and monitoring system"""

    def __init__(self, config: Config):
        self.config = config
        self.storage_dir = Path.home() / ".opensable" / "analytics"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Event storage
        self.events: List[MetricEvent] = []
        self.max_events_in_memory = 1000

        # Aggregated metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "commands_executed": 0,
            "errors": 0,
            "sessions_created": 0,
            "total_tokens": 0,
            "total_response_time": 0.0,
        }

        # Channel usage
        self.channel_stats = Counter()

        # User activity
        self.user_activity: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"messages": 0, "sessions": 0, "first_seen": None, "last_seen": None}
        )

        # Performance tracking
        self.response_times: List[float] = []
        self.max_response_times = 1000

        # Error tracking
        self.errors: List[Dict[str, Any]] = []
        self.max_errors = 100

        # Start time
        self.start_time = datetime.utcnow()

        # Auto-save interval
        self.auto_save_interval = 300  # 5 minutes
        self._save_task = None

    async def start(self):
        """Start analytics system"""
        logger.info("Starting analytics system")

        # Load existing data
        self.load_from_disk()

        # Start auto-save task
        self._save_task = asyncio.create_task(self._auto_save_loop())

    async def stop(self):
        """Stop analytics system"""
        logger.info("Stopping analytics system")

        # Cancel auto-save task
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass

        # Final save
        self.save_to_disk()

    def track_event(self, event_type: str, data: Dict[str, Any] = None):
        """Track an event"""
        event = MetricEvent(event_type, data)
        self.events.append(event)

        # Keep only recent events in memory
        if len(self.events) > self.max_events_in_memory:
            self.events = self.events[-self.max_events_in_memory :]

        logger.debug(f"Tracked event: {event_type}")

    def track_message_received(self, channel: str, user_id: str, message: str):
        """Track received message"""
        self.metrics["messages_received"] += 1
        self.channel_stats[channel] += 1

        # Update user activity
        user = self.user_activity[user_id]
        user["messages"] += 1
        user["last_seen"] = datetime.utcnow()
        if not user["first_seen"]:
            user["first_seen"] = datetime.utcnow()

        self.track_event(
            "message_received",
            {"channel": channel, "user_id": user_id, "message_length": len(message)},
        )

    def track_message_sent(
        self, channel: str, user_id: str, message: str, tokens: Optional[int] = None
    ):
        """Track sent message"""
        self.metrics["messages_sent"] += 1

        if tokens:
            self.metrics["total_tokens"] += tokens

        self.track_event(
            "message_sent",
            {
                "channel": channel,
                "user_id": user_id,
                "message_length": len(message),
                "tokens": tokens,
            },
        )

    def track_command(self, command: str, user_id: str, success: bool):
        """Track command execution"""
        self.metrics["commands_executed"] += 1

        self.track_event(
            "command_executed", {"command": command, "user_id": user_id, "success": success}
        )

    def track_session_created(self, channel: str, user_id: str):
        """Track session creation"""
        self.metrics["sessions_created"] += 1

        user = self.user_activity[user_id]
        user["sessions"] += 1

        self.track_event("session_created", {"channel": channel, "user_id": user_id})

    def track_response_time(self, duration: float):
        """Track response time"""
        self.metrics["total_response_time"] += duration
        self.response_times.append(duration)

        # Keep only recent response times
        if len(self.response_times) > self.max_response_times:
            self.response_times = self.response_times[-self.max_response_times :]

        self.track_event("response_time", {"duration": duration})

    def track_error(
        self, error_type: str, error_message: str, context: Optional[Dict[str, Any]] = None
    ):
        """Track error"""
        self.metrics["errors"] += 1

        error_data = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.errors.append(error_data)

        # Keep only recent errors
        if len(self.errors) > self.max_errors:
            self.errors = self.errors[-self.max_errors :]

        self.track_event("error", error_data)

    def get_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        # Calculate average response time
        avg_response_time = 0.0
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)

        # Calculate messages per minute
        messages_per_minute = 0.0
        if uptime > 0:
            total_messages = self.metrics["messages_received"] + self.metrics["messages_sent"]
            messages_per_minute = (total_messages / uptime) * 60

        return {
            "uptime_seconds": uptime,
            "start_time": self.start_time.isoformat(),
            "metrics": self.metrics.copy(),
            "channel_stats": dict(self.channel_stats),
            "active_users": len(self.user_activity),
            "total_events": len(self.events),
            "performance": {
                "avg_response_time": avg_response_time,
                "min_response_time": min(self.response_times) if self.response_times else 0,
                "max_response_time": max(self.response_times) if self.response_times else 0,
                "messages_per_minute": messages_per_minute,
            },
            "errors": {
                "total": self.metrics["errors"],
                "recent": self.errors[-10:],  # Last 10 errors
            },
        }

    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get stats for specific user"""
        if user_id not in self.user_activity:
            return None

        user = self.user_activity[user_id]

        return {
            "user_id": user_id,
            "messages": user["messages"],
            "sessions": user["sessions"],
            "first_seen": user["first_seen"].isoformat() if user["first_seen"] else None,
            "last_seen": user["last_seen"].isoformat() if user["last_seen"] else None,
        }

    def get_channel_stats(self) -> Dict[str, int]:
        """Get usage stats by channel"""
        return dict(self.channel_stats)

    def get_top_users(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most active users"""
        users = []

        for user_id, user_data in self.user_activity.items():
            users.append(
                {
                    "user_id": user_id,
                    "messages": user_data["messages"],
                    "sessions": user_data["sessions"],
                }
            )

        # Sort by messages
        users.sort(key=lambda u: u["messages"], reverse=True)

        return users[:limit]

    def get_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[MetricEvent]:
        """Get events with optional filters"""
        events = self.events

        # Filter by type
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Filter by time range
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Limit results
        return events[-limit:]

    def save_to_disk(self):
        """Save analytics data to disk"""
        try:
            data = {
                "start_time": self.start_time.isoformat(),
                "metrics": self.metrics,
                "channel_stats": dict(self.channel_stats),
                "user_activity": {
                    user_id: {
                        "messages": user["messages"],
                        "sessions": user["sessions"],
                        "first_seen": (
                            user["first_seen"].isoformat() if user["first_seen"] else None
                        ),
                        "last_seen": user["last_seen"].isoformat() if user["last_seen"] else None,
                    }
                    for user_id, user in self.user_activity.items()
                },
                "errors": self.errors,
                "last_updated": datetime.utcnow().isoformat(),
            }

            # Save to file
            analytics_file = self.storage_dir / "analytics.json"
            with open(analytics_file, "w") as f:
                json.dump(data, f, indent=2)

            # Save events
            events_file = self.storage_dir / "events.json"
            with open(events_file, "w") as f:
                json.dump([e.to_dict() for e in self.events], f, indent=2)

            logger.debug("Analytics data saved to disk")

        except Exception as e:
            logger.error(f"Error saving analytics data: {e}", exc_info=True)

    def load_from_disk(self):
        """Load analytics data from disk"""
        try:
            analytics_file = self.storage_dir / "analytics.json"

            if not analytics_file.exists():
                return

            with open(analytics_file) as f:
                data = json.load(f)

            self.start_time = datetime.fromisoformat(
                data.get("start_time", self.start_time.isoformat())
            )
            self.metrics = data.get("metrics", self.metrics)
            self.channel_stats = Counter(data.get("channel_stats", {}))
            self.errors = data.get("errors", [])

            # Load user activity
            for user_id, user_data in data.get("user_activity", {}).items():
                self.user_activity[user_id] = {
                    "messages": user_data.get("messages", 0),
                    "sessions": user_data.get("sessions", 0),
                    "first_seen": (
                        datetime.fromisoformat(user_data["first_seen"])
                        if user_data.get("first_seen")
                        else None
                    ),
                    "last_seen": (
                        datetime.fromisoformat(user_data["last_seen"])
                        if user_data.get("last_seen")
                        else None
                    ),
                }

            # Load events
            events_file = self.storage_dir / "events.json"
            if events_file.exists():
                with open(events_file) as f:
                    event_data = json.load(f)

                self.events = [MetricEvent.from_dict(e) for e in event_data]

            logger.info(
                f"Loaded analytics data: {len(self.events)} events, {len(self.user_activity)} users"
            )

        except Exception as e:
            logger.error(f"Error loading analytics data: {e}", exc_info=True)

    async def _auto_save_loop(self):
        """Auto-save loop"""
        while True:
            try:
                await asyncio.sleep(self.auto_save_interval)
                self.save_to_disk()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-save loop: {e}", exc_info=True)


# Context manager for tracking response time
class ResponseTimer:
    """Context manager for tracking response time"""

    def __init__(self, analytics: Analytics):
        self.analytics = analytics
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.analytics.track_response_time(duration)


if __name__ == "__main__":
    from opensable.core.config import load_config

    config = load_config()
    analytics = Analytics(config)

    # Test
    asyncio.run(analytics.start())

    analytics.track_message_received("telegram", "user123", "Hello!")
    analytics.track_message_sent("telegram", "user123", "Hi there!", tokens=10)
    analytics.track_session_created("telegram", "user123")

    summary = analytics.get_summary()
    print(json.dumps(summary, indent=2))

    asyncio.run(analytics.stop())
