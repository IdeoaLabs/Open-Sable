"""
Email skill for Open-Sable - Gmail integration
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
import base64
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


class EmailSkill:
    """Gmail integration skill"""

    def __init__(self, config):
        self.config = config
        self.service = None

    async def initialize(self):
        """Initialize Gmail API"""
        if not self.config.gmail_enabled:
            logger.info("Gmail skill disabled")
            return

        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
            creds = None

            # Load credentials
            token_path = Path("./config/gmail_token.json")
            if token_path.exists():
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif self.config.gmail_credentials_path.exists():
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.config.gmail_credentials_path), SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                else:
                    logger.warning(
                        "Gmail credentials not found. Email skill will run in demo mode."
                    )
                    return

                # Save credentials
                token_path.parent.mkdir(parents=True, exist_ok=True)
                with open(token_path, "w") as token:
                    token.write(creds.to_json())

            self.service = build("gmail", "v1", credentials=creds)
            logger.info("Gmail skill initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Gmail: {e}")
            logger.info("Email skill will run in demo mode")

    async def read_emails(
        self, max_results: int = 10, unread_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Read recent emails"""
        if not self.service:
            return self._demo_emails()

        try:
            query = "is:unread" if unread_only else ""
            results = (
                self.service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )

            messages = results.get("messages", [])
            emails = []

            for msg in messages:
                msg_data = self.service.users().messages().get(userId="me", id=msg["id"]).execute()

                headers = {h["name"]: h["value"] for h in msg_data["payload"]["headers"]}

                emails.append(
                    {
                        "id": msg["id"],
                        "from": headers.get("From", "Unknown"),
                        "subject": headers.get("Subject", "No subject"),
                        "date": headers.get("Date", ""),
                        "snippet": msg_data.get("snippet", ""),
                    }
                )

            return emails

        except Exception as e:
            logger.error(f"Failed to read emails: {e}")
            return self._demo_emails()

    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email"""
        if not self.service:
            logger.info(f"[DEMO] Would send email to {to}: {subject}")
            return True

        try:
            message = MIMEText(body)
            message["to"] = to
            message["subject"] = subject

            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

            self.service.users().messages().send(userId="me", body={"raw": raw}).execute()

            logger.info(f"Sent email to {to}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    async def mark_as_read(self, email_id: str) -> bool:
        """Mark email as read"""
        if not self.service:
            return True

        try:
            self.service.users().messages().modify(
                userId="me", id=email_id, body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to mark email as read: {e}")
            return False

    def _demo_emails(self) -> List[Dict[str, Any]]:
        """Return demo emails for testing"""
        return [
            {
                "id": "demo1",
                "from": "boss@company.com",
                "subject": "Q1 Report Due",
                "date": "2026-02-15",
                "snippet": "Please submit your Q1 report by end of week...",
            },
            {
                "id": "demo2",
                "from": "newsletter@tech.com",
                "subject": "This Week in AI",
                "date": "2026-02-16",
                "snippet": "Top 10 AI breakthroughs this week...",
            },
        ]
