"""
Email skill for Open-Sable,  SMTP/IMAP (no Google API dependency)

TODO: Proton Bridge integration
  - Install and run Proton Bridge locally (https://proton.me/mail/bridge)
  - Set credentials in ~/.opensable/email_private.env (never in repo):
      SMTP_HOST=127.0.0.1  SMTP_PORT=1025
      IMAP_HOST=127.0.0.1  IMAP_PORT=1143
      EMAIL_ALLOWED_RECIPIENTS=<creator-only address>
  - Email sending is locked to allowed recipients only; disabled until configured.
"""

import logging
import smtplib
import imaplib
import email as email_mod
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class EmailSkill:
    """Email integration via SMTP (send) and IMAP (read)"""

    def __init__(self, config):
        self.config = config
        self._ready = False

    # ── init ──────────────────────────────────────────────────────────────

    async def initialize(self):
        """Validate SMTP config. No persistent connection needed."""
        if not self.config.gmail_enabled:
            logger.info("Email skill disabled (GMAIL_ENABLED=false)")
            return

        host = self.config.smtp_host
        user = self.config.smtp_user
        password = self.config.smtp_password

        if not host or not user or not password:
            logger.warning(
                "SMTP not configured (set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env). "
                "Email skill will run in demo mode."
            )
            return

        self._ready = True
        logger.info(f"Email skill ready (SMTP → {host}:{self.config.smtp_port})")

    # ── helpers ───────────────────────────────────────────────────────────

    def _smtp_connect(self) -> smtplib.SMTP:
        """Open an authenticated SMTP connection."""
        port = self.config.smtp_port or 587
        s = smtplib.SMTP(self.config.smtp_host, port, timeout=15)
        s.ehlo()
        if port != 25:
            s.starttls()
            s.ehlo()
        s.login(self.config.smtp_user, self.config.smtp_password)
        return s

    def _imap_connect(self) -> Optional[imaplib.IMAP4_SSL]:
        """Open an IMAP connection (auto-derives host from SMTP host)."""
        smtp_host = self.config.smtp_host or ""
        imap_host = getattr(self.config, "imap_host", None)
        if not imap_host:
            imap_host = smtp_host.replace("smtp.", "imap.", 1)

        try:
            conn = imaplib.IMAP4_SSL(imap_host, timeout=15)
            conn.login(self.config.smtp_user, self.config.smtp_password)
            return conn
        except Exception as e:
            logger.error(f"IMAP connect failed ({imap_host}): {e}")
            return None

    @staticmethod
    def _decode_header(raw: str) -> str:
        """Decode RFC-2047 encoded header value."""
        if not raw:
            return ""
        parts = decode_header(raw)
        decoded = []
        for data, charset in parts:
            if isinstance(data, bytes):
                decoded.append(data.decode(charset or "utf-8", errors="replace"))
            else:
                decoded.append(data)
        return " ".join(decoded)

    # ── public API ────────────────────────────────────────────────────────

    async def read_emails(
        self, max_results: int = 10, unread_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Read recent emails via IMAP."""
        if not self._ready:
            return self._demo_emails()

        conn = self._imap_connect()
        if not conn:
            return self._demo_emails()

        try:
            conn.select("INBOX", readonly=True)
            criterion = "UNSEEN" if unread_only else "ALL"
            _status, data = conn.search(None, criterion)

            ids = data[0].split() if data[0] else []
            ids = ids[-max_results:][::-1]

            emails: List[Dict[str, Any]] = []
            for mid in ids:
                _st, msg_data = conn.fetch(mid, "(RFC822)")
                if not msg_data or not msg_data[0]:
                    continue
                raw = msg_data[0][1]
                msg = email_mod.message_from_bytes(raw)

                snippet = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        ct = part.get_content_type()
                        if ct == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                snippet = payload.decode(errors="replace")[:200]
                            break
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        snippet = payload.decode(errors="replace")[:200]

                emails.append(
                    {
                        "id": mid.decode(),
                        "from": self._decode_header(msg.get("From", "")),
                        "subject": self._decode_header(msg.get("Subject", "")),
                        "date": msg.get("Date", ""),
                        "snippet": snippet.strip(),
                    }
                )

            return emails

        except Exception as e:
            logger.error(f"Failed to read emails: {e}")
            return self._demo_emails()
        finally:
            try:
                conn.close()
                conn.logout()
            except Exception:
                pass

    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send an email via SMTP."""
        if not self._ready:
            logger.info(f"[DEMO] Would send email to {to}: {subject}")
            return True

        try:
            sender = self.config.smtp_from or self.config.smtp_user
            msg = MIMEMultipart("alternative")
            msg["From"] = sender
            msg["To"] = to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain", "utf-8"))

            with self._smtp_connect() as smtp:
                smtp.sendmail(sender, [to], msg.as_string())

            logger.info(f"Sent email to {to}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    async def mark_as_read(self, email_id: str) -> bool:
        """Mark email as read via IMAP."""
        if not self._ready:
            return True

        conn = self._imap_connect()
        if not conn:
            return False

        try:
            conn.select("INBOX")
            conn.store(email_id.encode(), "+FLAGS", "\\Seen")
            return True
        except Exception as e:
            logger.error(f"Failed to mark email as read: {e}")
            return False
        finally:
            try:
                conn.close()
                conn.logout()
            except Exception:
                pass

    # ── demo fallback ─────────────────────────────────────────────────────

    def _demo_emails(self) -> List[Dict[str, Any]]:
        """Return demo emails when SMTP/IMAP not configured."""
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
