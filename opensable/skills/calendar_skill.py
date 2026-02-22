"""
Calendar skill for Open-Sable - Google Calendar integration
"""

import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class CalendarSkill:
    """Google Calendar integration skill"""

    def __init__(self, config):
        self.config = config
        self.service = None

    async def initialize(self):
        """Initialize Calendar API"""
        if not self.config.calendar_enabled:
            logger.info("Calendar skill disabled")
            return

        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            SCOPES = ["https://www.googleapis.com/auth/calendar"]
            creds = None

            # Load credentials
            token_path = Path("./config/calendar_token.json")
            if token_path.exists():
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif self.config.calendar_credentials_path.exists():
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.config.calendar_credentials_path), SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                else:
                    logger.warning("Calendar credentials not found. Skill will run in demo mode.")
                    return

                # Save credentials
                token_path.parent.mkdir(parents=True, exist_ok=True)
                with open(token_path, "w") as token:
                    token.write(creds.to_json())

            self.service = build("calendar", "v3", credentials=creds)
            logger.info("Calendar skill initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Calendar: {e}")
            logger.info("Calendar skill will run in demo mode")

    async def list_events(self, days_ahead: int = 7, max_results: int = 10) -> List[Dict[str, Any]]:
        """List upcoming calendar events"""
        if not self.service:
            return self._demo_events()

        try:
            now = datetime.utcnow().isoformat() + "Z"
            later = (datetime.utcnow() + timedelta(days=days_ahead)).isoformat() + "Z"

            events_result = (
                self.service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    timeMax=later,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])

            formatted_events = []
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                formatted_events.append(
                    {
                        "id": event["id"],
                        "summary": event.get("summary", "No title"),
                        "start": start,
                        "description": event.get("description", ""),
                        "location": event.get("location", ""),
                    }
                )

            return formatted_events

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
        """Add a new calendar event"""
        if not self.service:
            logger.info(f"[DEMO] Would add event: {summary} at {start_time}")
            return True

        try:
            # Parse start time
            start_dt = datetime.fromisoformat(start_time)
            end_dt = start_dt + timedelta(minutes=duration_minutes)

            event = {
                "summary": summary,
                "description": description,
                "location": location,
                "start": {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": "UTC",
                },
                "end": {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": "UTC",
                },
            }

            self.service.events().insert(calendarId="primary", body=event).execute()
            logger.info(f"Added event: {summary}")
            return True

        except Exception as e:
            logger.error(f"Failed to add event: {e}")
            return False

    async def delete_event(self, event_id: str) -> bool:
        """Delete a calendar event"""
        if not self.service:
            return True

        try:
            self.service.events().delete(calendarId="primary", eventId=event_id).execute()
            logger.info(f"Deleted event: {event_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete event: {e}")
            return False

    def _demo_events(self) -> List[Dict[str, Any]]:
        """Return demo events for testing"""
        now = datetime.now()
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
