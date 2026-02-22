"""
Open-Sable Webhook System

Manages incoming/outgoing webhooks for integrations.
Supports webhook authentication, retries, and event subscriptions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
import json
import hmac
import hashlib
import aiohttp
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse

from opensable.core.config import Config

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Webhook event types"""

    MESSAGE_RECEIVED = "message.received"
    MESSAGE_SENT = "message.sent"
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    COMMAND_EXECUTED = "command.executed"
    ERROR_OCCURRED = "error.occurred"
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"


class WebhookDelivery:
    """Represents a webhook delivery attempt"""

    def __init__(self, webhook_id: str, event: str, payload: dict):
        self.webhook_id = webhook_id
        self.event = event
        self.payload = payload
        self.created_at = datetime.utcnow()
        self.attempts = 0
        self.max_attempts = 3
        self.last_attempt_at: Optional[datetime] = None
        self.success = False
        self.response_status: Optional[int] = None
        self.response_body: Optional[str] = None
        self.error: Optional[str] = None


class Webhook:
    """Webhook configuration"""

    def __init__(
        self,
        webhook_id: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        enabled: bool = True,
    ):
        self.webhook_id = webhook_id
        self.url = url
        self.events = events
        self.secret = secret
        self.enabled = enabled
        self.created_at = datetime.utcnow()
        self.last_delivery_at: Optional[datetime] = None
        self.total_deliveries = 0
        self.failed_deliveries = 0

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "webhook_id": self.webhook_id,
            "url": self.url,
            "events": self.events,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "last_delivery_at": (
                self.last_delivery_at.isoformat() if self.last_delivery_at else None
            ),
            "total_deliveries": self.total_deliveries,
            "failed_deliveries": self.failed_deliveries,
        }


class WebhookManager:
    """Manages outgoing webhooks"""

    def __init__(self, config: Config):
        self.config = config
        self.webhooks: Dict[str, Webhook] = {}
        self.deliveries: List[WebhookDelivery] = []
        self.max_deliveries_history = 100

        # Storage
        self.storage_dir = Path.home() / ".opensable" / "webhooks"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # HTTP client
        self.session: Optional[aiohttp.ClientSession] = None

        # Event subscribers
        self.subscribers: Dict[str, List[Callable]] = {}

    async def start(self):
        """Start webhook manager"""
        self.session = aiohttp.ClientSession()
        self._load_webhooks()
        logger.info("Webhook manager started")

    async def stop(self):
        """Stop webhook manager"""
        if self.session:
            await self.session.close()

        self._save_webhooks()
        logger.info("Webhook manager stopped")

    def register_webhook(self, url: str, events: List[str], secret: Optional[str] = None) -> str:
        """Register a new webhook"""
        import uuid

        webhook_id = str(uuid.uuid4())

        webhook = Webhook(webhook_id=webhook_id, url=url, events=events, secret=secret)

        self.webhooks[webhook_id] = webhook
        self._save_webhooks()

        logger.info(f"Registered webhook: {webhook_id} -> {url}")
        return webhook_id

    def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister webhook"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            self._save_webhooks()
            logger.info(f"Unregistered webhook: {webhook_id}")
            return True
        return False

    def update_webhook(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        enabled: Optional[bool] = None,
    ) -> bool:
        """Update webhook configuration"""
        webhook = self.webhooks.get(webhook_id)

        if not webhook:
            return False

        if url is not None:
            webhook.url = url
        if events is not None:
            webhook.events = events
        if enabled is not None:
            webhook.enabled = enabled

        self._save_webhooks()
        logger.info(f"Updated webhook: {webhook_id}")
        return True

    async def emit(self, event: str, payload: dict):
        """Emit event to all subscribed webhooks"""
        matching_webhooks = [
            wh for wh in self.webhooks.values() if event in wh.events and wh.enabled
        ]

        if not matching_webhooks:
            return

        logger.info(f"Emitting event '{event}' to {len(matching_webhooks)} webhooks")

        # Deliver to all webhooks
        tasks = [self._deliver(webhook, event, payload) for webhook in matching_webhooks]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _deliver(self, webhook: Webhook, event: str, payload: dict):
        """Deliver webhook event"""
        delivery = WebhookDelivery(webhook.webhook_id, event, payload)
        self.deliveries.append(delivery)

        # Trim delivery history
        if len(self.deliveries) > self.max_deliveries_history:
            self.deliveries = self.deliveries[-self.max_deliveries_history :]

        # Build payload
        webhook_payload = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "data": payload,
        }

        # Add signature if secret provided
        headers = {"Content-Type": "application/json"}

        if webhook.secret:
            signature = self._generate_signature(webhook.secret, webhook_payload)
            headers["X-Webhook-Signature"] = signature

        # Attempt delivery with retries
        while delivery.attempts < delivery.max_attempts:
            delivery.attempts += 1
            delivery.last_attempt_at = datetime.utcnow()

            try:
                async with self.session.post(
                    webhook.url,
                    json=webhook_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    delivery.response_status = response.status
                    delivery.response_body = await response.text()

                    if 200 <= response.status < 300:
                        delivery.success = True
                        webhook.last_delivery_at = datetime.utcnow()
                        webhook.total_deliveries += 1
                        logger.info(f"Webhook delivered: {webhook.webhook_id} ({event})")
                        return
                    else:
                        delivery.error = f"HTTP {response.status}"
                        logger.warning(
                            f"Webhook delivery failed: {webhook.webhook_id} - {delivery.error}"
                        )

            except Exception as e:
                delivery.error = str(e)
                logger.error(f"Webhook delivery error: {webhook.webhook_id} - {e}")

            # Wait before retry (exponential backoff)
            if delivery.attempts < delivery.max_attempts:
                await asyncio.sleep(2**delivery.attempts)

        # All attempts failed
        webhook.failed_deliveries += 1
        logger.error(
            f"Webhook delivery failed after {delivery.attempts} attempts: {webhook.webhook_id}"
        )

    def _generate_signature(self, secret: str, payload: dict) -> str:
        """Generate HMAC signature for payload"""
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        signature = hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
        return f"sha256={signature}"

    def _verify_signature(self, secret: str, payload: dict, signature: str) -> bool:
        """Verify HMAC signature"""
        expected = self._generate_signature(secret, payload)
        return hmac.compare_digest(expected, signature)

    def _save_webhooks(self):
        """Save webhooks to disk"""
        try:
            data = {"webhooks": [wh.to_dict() for wh in self.webhooks.values()]}

            with open(self.storage_dir / "webhooks.json", "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving webhooks: {e}")

    def _load_webhooks(self):
        """Load webhooks from disk"""
        try:
            webhooks_file = self.storage_dir / "webhooks.json"

            if not webhooks_file.exists():
                return

            with open(webhooks_file) as f:
                data = json.load(f)

            for wh_data in data.get("webhooks", []):
                webhook = Webhook(
                    webhook_id=wh_data["webhook_id"],
                    url=wh_data["url"],
                    events=wh_data["events"],
                    enabled=wh_data.get("enabled", True),
                )

                webhook.total_deliveries = wh_data.get("total_deliveries", 0)
                webhook.failed_deliveries = wh_data.get("failed_deliveries", 0)

                if wh_data.get("created_at"):
                    webhook.created_at = datetime.fromisoformat(wh_data["created_at"])

                self.webhooks[webhook.webhook_id] = webhook

            logger.info(f"Loaded {len(self.webhooks)} webhooks")

        except Exception as e:
            logger.error(f"Error loading webhooks: {e}")

    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Get webhook by ID"""
        return self.webhooks.get(webhook_id)

    def list_webhooks(self) -> List[dict]:
        """List all webhooks"""
        return [wh.to_dict() for wh in self.webhooks.values()]

    def get_deliveries(self, webhook_id: Optional[str] = None) -> List[dict]:
        """Get delivery history"""
        deliveries = self.deliveries

        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        return [
            {
                "webhook_id": d.webhook_id,
                "event": d.event,
                "attempts": d.attempts,
                "success": d.success,
                "created_at": d.created_at.isoformat(),
                "response_status": d.response_status,
                "error": d.error,
            }
            for d in deliveries
        ]


def create_webhook_server(config: Config, webhook_manager: WebhookManager) -> FastAPI:
    """Create FastAPI app for incoming webhooks"""
    app = FastAPI(title="Open-Sable Webhook Server", version="0.2.0")

    # In-memory webhook handlers
    handlers: Dict[str, Callable] = {}

    def register_handler(path: str, handler: Callable):
        """Register incoming webhook handler"""
        handlers[path] = handler

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": "Open-Sable Webhook Server",
            "version": "0.2.0",
            "webhooks": list(handlers.keys()),
        }

    @app.post("/webhooks/{path:path}")
    async def receive_webhook(
        path: str, request: Request, x_webhook_signature: Optional[str] = Header(None)
    ):
        """Receive incoming webhook"""
        if path not in handlers:
            raise HTTPException(status_code=404, detail="Webhook not found")

        # Get payload
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        # Verify signature if provided
        # (Secret should be configured per handler)

        # Call handler
        handler = handlers[path]

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(payload)
            else:
                result = handler(payload)

            return JSONResponse({"status": "success", "result": result})

        except Exception as e:
            logger.error(f"Webhook handler error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # Add register_handler to app
    app.register_handler = register_handler

    return app


if __name__ == "__main__":
    from opensable.core.config import load_config

    config = load_config()
    manager = WebhookManager(config)

    async def test():
        await manager.start()

        # Register webhook
        webhook_id = manager.register_webhook(
            url="https://example.com/webhook", events=["message.received", "session.created"]
        )

        # Emit event
        await manager.emit(
            "message.received", {"user_id": "123", "message": "Hello!", "channel": "telegram"}
        )

        # Check deliveries
        deliveries = manager.get_deliveries()
        print(f"Deliveries: {len(deliveries)}")

        await manager.stop()

    asyncio.run(test())
