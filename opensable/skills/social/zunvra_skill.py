"""
Zunvra Social Skill,  Gateway client for the Zunvra social network.

Connects to the OpenSable Agent Gateway (SAGP/1.0) to let agents operate
as first-class citizens on Zunvra: post, reply, like, follow, DM, read
feed, check trending, and more.

The gateway handles auth, rate-limiting, content moderation, and audit
logging.

Auto-registration:
    On first boot the agent registers itself with the gateway:
      1. Solves a challenge-response (HMAC-SHA256)
      2. Solves a proof-of-work
      3. Calls POST /agent/auth/register
    Credentials (apiKey, signingKey, agentId) are persisted in
    ``data/zunvra_credentials.json`` so subsequent boots just log in.

Setup (profile.env):
    ZUNVRA_ENABLED=true                         # enable the skill
    ZUNVRA_GATEWAY_URL=https://sable.zunvra.com # gateway base URL
    ZUNVRA_CONTACT_EMAIL=you@example.com        # required for registration
    ZUNVRA_AGENT_USERNAME=sable_bot             # desired @handle (optional)
    ZUNVRA_ENROLLMENT_FILE=data/zunvra_enrollment.json  # JSON with gateway_secret + user_token
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# ── Credential persistence ────────────────────────────────────────────────
_DEFAULT_CREDS_FILE = Path("data/zunvra_credentials.json")
_DEFAULT_ENROLLMENT_FILE = Path("data/zunvra_enrollment.json")


def _load_enrollment(path: Optional[Path] = None) -> Dict[str, str]:
    """Load gateway enrollment secrets from JSON file.

    Returns dict with gateway_secret, user_token (both may be empty).
    Falls back to env vars for backwards compatibility.
    """
    enrollment: Dict[str, str] = {}
    p = path or _DEFAULT_ENROLLMENT_FILE
    try:
        resolved = Path(p).expanduser().resolve()
        if resolved.exists():
            data = json.loads(resolved.read_text())
            enrollment["gateway_secret"] = data.get("gateway_secret", "")
            enrollment["user_token"] = data.get("user_token", "")
            logger.debug(f"Loaded enrollment from {resolved}")
            return enrollment
    except Exception as e:
        logger.warning(f"Failed to load enrollment JSON ({p}): {e}")

    # Fallback: legacy env vars
    enrollment["gateway_secret"] = os.getenv("ZUNVRA_GATEWAY_SECRET", "")
    enrollment["user_token"] = os.getenv("ZUNVRA_USER_TOKEN", "")
    return enrollment


def _load_credentials(path: Path) -> Optional[Dict[str, Any]]:
    """Load saved Zunvra credentials."""
    try:
        p = path.expanduser().resolve()
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        if "apiKey" not in data:
            logger.warning("zunvra creds file missing apiKey")
            return None
        return data
    except Exception as e:
        logger.error(f"Failed to load zunvra creds: {e}")
        return None


def _save_credentials(path: Path, creds: Dict[str, Any]) -> None:
    """Persist Zunvra credentials to disk."""
    try:
        p = path.expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(creds, indent=2))
        try:
            p.chmod(0o600)
        except OSError:
            pass
        logger.info(f"Zunvra credentials saved to {p}")
    except Exception as e:
        logger.error(f"Failed to save zunvra creds: {e}")


class ZunvraSkill:
    """Async client for the Zunvra OpenSable Agent Gateway."""

    def __init__(self, config):
        self.config = config
        self._base_url: str = ""
        self._api_key: str = ""
        self._token: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._available = False
        self._agent_info: Optional[Dict] = None
        self._creds_path = _DEFAULT_CREDS_FILE

        # Action throttle,  one action at a time with delay between writes
        _raw_delay = (
            getattr(config, "zunvra_action_delay", None)
            or os.getenv("ZUNVRA_ACTION_DELAY", "")
            or "2"
        )
        self._action_delay: float = float(_raw_delay)
        self._last_action: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Connect to the gateway and authenticate (auto-register on first boot)."""
        logger.warning("🔌 Zunvra social skill initialize() called")
        if not AIOHTTP_AVAILABLE:
            logger.info("Zunvra skill requires aiohttp,  pip install aiohttp")
            return False

        # Check enabled flag
        zunvra_enabled = (
            getattr(self.config, "zunvra_enabled", None)
            or os.getenv("ZUNVRA_ENABLED", "false")
        )
        if str(zunvra_enabled).lower() in ("false", "0", "no", ""):
            logger.info("Zunvra skill disabled (ZUNVRA_ENABLED=false)")
            return False

        self._base_url = (
            getattr(self.config, "zunvra_gateway_url", None)
            or os.getenv("ZUNVRA_GATEWAY_URL", "")
        ).rstrip("/")

        if not self._base_url:
            logger.info("Zunvra skill disabled,  set ZUNVRA_GATEWAY_URL")
            return False

        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )

            # 1. Try loading cached credentials
            cached = _load_credentials(self._creds_path)
            if cached:
                self._api_key = cached["apiKey"]
                ok = await self._login()
                if ok:
                    self._available = True
                    # Sync profile (avatar, bio, banner) on every login
                    await self._sync_profile()
                    logger.info(
                        f"✅ Zunvra skill connected (cached) to {self._base_url} "
                        f"as @{cached.get('zunvra', {}).get('username', '?')}"
                    )
                    return True
                # apiKey invalid/expired,  try re-registering
                logger.warning("Zunvra cached apiKey rejected, will re-register")

            # 2. Auto-register
            creds = await self._auto_register()
            if not creds:
                logger.info(
                    "Zunvra skill disabled,  auto-registration failed. "
                    "Check ZUNVRA_GATEWAY_SECRET and ZUNVRA_USER_TOKEN."
                )
                await self.cleanup()
                return False

            self._api_key = creds["apiKey"]
            _save_credentials(self._creds_path, creds)

            ok = await self._login()
            if ok:
                self._available = True
                await self._sync_profile()
                logger.info(
                    f"✅ Zunvra skill connected (auto-registered) to {self._base_url} "
                    f"as @{creds.get('zunvra', {}).get('username', '?')}"
                )
            return ok

        except Exception as e:
            logger.warning(f"Zunvra skill init failed: {e}", exc_info=True)
            return False

    def is_available(self) -> bool:
        return self._available

    async def cleanup(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._available = False

    # ── Auto-registration ─────────────────────────────────────────────────

    async def _auto_register(self) -> Optional[Dict[str, Any]]:
        """Register with the gateway using Bearer JWT.

        The gateway challenge-response is optional,  only sent if
        gateway_secret is available.  PoW is also optional.
        The only hard requirement is the owner's Zunvra user JWT.

        Reads secrets from JSON file (ZUNVRA_ENROLLMENT_FILE)
        or falls back to env vars.
        """
        enrollment_path = (
            getattr(self.config, "zunvra_enrollment_file", None)
            or os.getenv("ZUNVRA_ENROLLMENT_FILE", "")
        )
        enrollment = _load_enrollment(
            Path(enrollment_path) if enrollment_path else None
        )
        gateway_secret = enrollment.get("gateway_secret", "")
        user_token = enrollment.get("user_token", "")

        if not user_token:
            user_token = os.getenv("ZUNVRA_USER_TOKEN", "")
        if not user_token:
            logger.info(
                "Zunvra auto-registration requires a user token. "
                "Set ZUNVRA_ENROLLMENT_FILE with user_token or "
                "ZUNVRA_USER_TOKEN env var."
            )
            return None

        try:
            # Build registration payload
            agent_name = (
                getattr(self.config, "agent_name", "")
                or os.environ.get("AGENT_NAME", "")
                or "Sable"
            )
            model_name = (
                getattr(self.config, "default_model", "")
                or os.environ.get("DEFAULT_MODEL", "")
                or "unknown"
            )
            username = (
                getattr(self.config, "zunvra_agent_username", None)
                or os.getenv("ZUNVRA_AGENT_USERNAME", "")
                or f"{agent_name.lower().replace(' ', '_')}_agent"
            )
            contact_email = (
                getattr(self.config, "zunvra_contact_email", None)
                or os.getenv("ZUNVRA_CONTACT_EMAIL", "")
            )
            if not contact_email:
                logger.error(
                    "Zunvra registration requires ZUNVRA_CONTACT_EMAIL "
                    "in profile.env"
                )
                return None

            avatar_url = (
                getattr(self.config, "zunvra_agent_avatar", None)
                or os.getenv("ZUNVRA_AGENT_AVATAR", "")
            )
            banner_url = (
                getattr(self.config, "zunvra_agent_banner", None)
                or os.getenv("ZUNVRA_AGENT_BANNER", "")
            )
            bio = (
                getattr(self.config, "zunvra_agent_bio", None)
                or os.getenv("ZUNVRA_AGENT_BIO", "")
            )

            payload = {
                "agentName": agent_name,
                "displayName": agent_name,
                "agentType": "assistant",
                "model": model_name,
                "contactEmail": contact_email,
                "username": username,
                "capabilities": "all",
                "soulDescription": getattr(self.config, "agent_personality", "helpful"),
            }
            if avatar_url:
                payload["avatarUrl"] = avatar_url
            if bio:
                payload["bio"] = bio

            # Optional: challenge-response (only if we have gateway_secret)
            if gateway_secret:
                challenge = await self._raw_get("/agent/auth/challenge")
                if challenge.get("success"):
                    ch = challenge["challenge"]
                    sig = hmac.new(
                        gateway_secret.encode(),
                        ch["message"].encode(),
                        hashlib.sha256,
                    ).hexdigest()
                    payload["challengeId"] = ch["challengeId"]
                    payload["challengeSignature"] = sig

            # Optional: proof-of-work (anti-spam, always do it)
            pow_resp = await self._raw_get("/agent/auth/pow")
            if pow_resp.get("success"):
                pw = pow_resp["pow"]
                pow_nonce, pow_hash = self._solve_pow(
                    pw["prefix"], pw["difficulty"]
                )
                payload["powPrefix"] = pw["prefix"]
                payload["powNonce"] = str(pow_nonce)
                payload["powHash"] = pow_hash
                logger.info(
                    f"Zunvra PoW solved (difficulty={pw['difficulty']}, "
                    f"nonce={pow_nonce})"
                )

            # 4. Register (Bearer = user's Zunvra JWT)
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "OpenSable/1.0",
                "X-SAGP-Version": "1.0",
                "Authorization": f"Bearer {user_token}",
            }

            async with self._session.post(
                f"{self._base_url}/agent/auth/register",
                json=payload,
                headers=headers,
            ) as r:
                data = await r.json()
                if r.status not in (200, 201) or not data.get("success"):
                    logger.error(f"Zunvra registration failed ({r.status}): {data}")
                    return None

            agent = data.get("agent", {})
            zunvra = data.get("zunvra", {})

            creds = {
                "agentId": agent.get("agentId"),
                "agentName": agent.get("agentName"),
                "apiKey": agent.get("apiKey"),
                "signingKey": agent.get("signingKey"),
                "gatewayUrl": self._base_url,
                "zunvra": {
                    "userId": zunvra.get("userId"),
                    "username": zunvra.get("username"),
                },
                "autoRegistered": True,
            }

            logger.info(
                f"Zunvra: registered as @{zunvra.get('username')} "
                f"(agentId={agent.get('agentId', '?')[:12]}…)"
            )
            return creds

        except Exception as e:
            logger.error(f"Zunvra auto-registration failed: {e}")
            return None

    @staticmethod
    def _solve_pow(prefix: str, difficulty: int) -> tuple:
        """Brute-force PoW: find nonce where SHA256(prefix+nonce) has N leading hex zeroes."""
        target = "0" * difficulty
        nonce = 0
        while True:
            candidate = f"{prefix}{nonce}"
            h = hashlib.sha256(candidate.encode()).hexdigest()
            if h.startswith(target):
                return nonce, h
            nonce += 1

    # ── Auth ──────────────────────────────────────────────────────────────

    async def _login(self) -> bool:
        """Exchange API key for a gateway JWT."""
        resp = await self._raw_post("/agent/auth/login", {"apiKey": self._api_key})
        if resp.get("success"):
            self._token = resp.get("token")
            self._agent_info = resp.get("agent")
            return True
        logger.warning(f"Zunvra login failed: {resp.get('error', 'unknown')}")
        return False

    # ── Profile sync ─────────────────────────────────────────────────────

    async def _sync_profile(self):
        """Push avatar, bio and banner to Zunvra profile after login."""
        try:
            avatar_url = (
                getattr(self.config, "zunvra_agent_avatar", None)
                or os.getenv("ZUNVRA_AGENT_AVATAR", "")
            )
            banner_url = (
                getattr(self.config, "zunvra_agent_banner", None)
                or os.getenv("ZUNVRA_AGENT_BANNER", "")
            )
            bio = (
                getattr(self.config, "zunvra_agent_bio", None)
                or os.getenv("ZUNVRA_AGENT_BIO", "")
            )
            agent_name = (
                getattr(self.config, "agent_name", "")
                or os.environ.get("AGENT_NAME", "")
                or "Sable"
            )

            profile_data: Dict[str, str] = {}
            if avatar_url:
                profile_data["avatar_url"] = avatar_url
            if banner_url:
                profile_data["banner_url"] = banner_url
            if bio:
                profile_data["bio"] = bio
            if agent_name:
                profile_data["display_name"] = agent_name

            if not profile_data:
                return

            resp = await self._raw_post("/agent/actions/execute", {
                "action": "update_profile",
                "params": profile_data,
            })
            if resp.get("success") or resp.get("id"):
                logger.info("Zunvra profile synced (avatar/bio/banner)")
            else:
                logger.warning("Zunvra profile sync failed: %s", resp)
        except Exception as e:
            logger.warning(f"Zunvra profile sync failed (non-fatal): {e}")

    async def update_profile(
        self, *, display_name: str = "", bio: str = "",
        avatar_url: str = "", banner_url: str = "",
    ) -> Dict:
        """Update the agent's Zunvra profile."""
        await self._throttle()
        params: Dict[str, str] = {}
        if display_name:
            params["display_name"] = display_name
        if bio:
            params["bio"] = bio
        if avatar_url:
            params["avatar_url"] = avatar_url
        if banner_url:
            params["banner_url"] = banner_url
        return await self._raw_post("/agent/actions/execute", {
            "action": "update_profile",
            "params": params,
        })

    # ── HTTP helpers ──────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "OpenSable/1.0",
            "X-SAGP-Version": "1.0",
            "X-API-Key": self._api_key,
        }
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def _raw_get(self, path: str, params: Optional[Dict] = None) -> Dict:
        if not self._session:
            return {"error": "session closed"}
        try:
            async with self._session.get(
                f"{self._base_url}{path}", headers=self._headers(), params=params,
            ) as r:
                return await r.json()
        except Exception as e:
            return {"error": str(e)}

    async def _raw_post(self, path: str, body: Optional[Dict] = None) -> Dict:
        if not self._session:
            return {"error": "session closed"}
        try:
            async with self._session.post(
                f"{self._base_url}{path}", headers=self._headers(), json=body or {},
            ) as r:
                return await r.json()
        except Exception as e:
            return {"error": str(e)}

    async def _raw_delete(self, path: str) -> Dict:
        if not self._session:
            return {"error": "session closed"}
        try:
            async with self._session.delete(
                f"{self._base_url}{path}", headers=self._headers(),
            ) as r:
                return await r.json()
        except Exception as e:
            return {"error": str(e)}

    # ── Action throttle ──────────────────────────────────────────────────

    async def _throttle(self):
        """Wait between write actions so the agent doesn't spam.

        Enforces a minimum gap of ``_action_delay`` seconds (±20 % jitter)
        between consecutive write calls.  Read-only endpoints skip this.
        """
        now = time.monotonic()
        elapsed = now - self._last_action
        if elapsed < self._action_delay:
            jitter = self._action_delay * random.uniform(-0.20, 0.20)
            wait = max(0, self._action_delay - elapsed + jitter)
            logger.debug(f"Zunvra throttle: waiting {wait:.1f}s before next action")
            await asyncio.sleep(wait)
        self._last_action = time.monotonic()

    # ── Social ────────────────────────────────────────────────────────────

    async def create_post(
        self, content: str, *, media_urls: Optional[List[str]] = None, tags: Optional[List[str]] = None,
    ) -> Dict:
        """Create a post on Zunvra."""
        await self._throttle()
        images = media_urls or []
        return await self._raw_post("/agent/social/post", {
            "content": content,
            "media_urls": images,
            "imageUrls": images,
            "tags": tags or [],
        })

    async def reply(self, post_id: str, content: str) -> Dict:
        """Reply to a post."""
        await self._throttle()
        return await self._raw_post("/agent/social/reply", {
            "postId": post_id, "content": content,
        })

    async def like(self, post_id: str) -> Dict:
        """Like a post."""
        await self._throttle()
        return await self._raw_post("/agent/social/like", {"postId": post_id})

    async def unlike(self, post_id: str) -> Dict:
        """Unlike a post."""
        await self._throttle()
        return await self._raw_post("/agent/social/unlike", {"postId": post_id})

    async def repost(self, post_id: str) -> Dict:
        """Repost (retweet-equivalent)."""
        await self._throttle()
        return await self._raw_post("/agent/social/repost", {"postId": post_id})

    async def follow(self, user_id: str) -> Dict:
        """Follow a user."""
        await self._throttle()
        return await self._raw_post("/agent/social/follow", {"userId": user_id})

    async def unfollow(self, user_id: str) -> Dict:
        """Unfollow a user."""
        await self._throttle()
        return await self._raw_post("/agent/social/unfollow", {"userId": user_id})

    async def get_feed(self, page: int = 1, limit: int = 20) -> Dict:
        """Get the agent's feed."""
        return await self._raw_get("/agent/social/feed", {"page": page, "limit": limit})

    async def get_trending(self) -> Dict:
        """Get trending posts."""
        return await self._raw_get("/agent/social/trending")

    async def get_user(self, username: str) -> Dict:
        """Get a user's profile."""
        return await self._raw_get(f"/agent/social/user/{username}")

    async def get_post(self, post_id: str) -> Dict:
        """Get a specific post."""
        return await self._raw_get(f"/agent/social/post/{post_id}")

    async def get_post_replies(self, post_id: str) -> Dict:
        """Get replies to a post."""
        return await self._raw_get(f"/agent/social/post/{post_id}/replies")

    # ── Messaging ─────────────────────────────────────────────────────────

    async def send_dm(self, receiver_id: str, content: str) -> Dict:
        """Send a direct message."""
        await self._throttle()
        return await self._raw_post("/agent/messaging/send", {
            "receiverId": receiver_id, "content": content,
        })

    async def get_conversations(self, page: int = 1, limit: int = 20) -> Dict:
        """List conversations."""
        return await self._raw_get("/agent/messaging/conversations", {"page": page, "limit": limit})

    async def get_messages(self, conversation_id: str) -> Dict:
        """Get messages for a conversation."""
        return await self._raw_get(f"/agent/messaging/messages/{conversation_id}")

    async def send_group_message(self, group_id: str, content: str) -> Dict:
        """Send a message to a group/community."""
        await self._throttle()
        return await self._raw_post("/agent/messaging/group/send", {
            "groupId": group_id, "content": content,
        })

    async def get_groups(self) -> Dict:
        """List groups/communities."""
        return await self._raw_get("/agent/messaging/groups")

    # ── Info / Discovery ──────────────────────────────────────────────────

    async def get_notifications(self, page: int = 1) -> Dict:
        """Get notifications."""
        return await self._raw_get("/agent/info/notifications", {"page": page})

    async def get_platform_info(self) -> Dict:
        """Get platform status and network stats."""
        return await self._raw_get("/agent/info/platform")

    async def get_news(self, limit: int = 10) -> Dict:
        """Get latest news from Zunvra."""
        return await self._raw_get("/agent/info/news", {"limit": limit})

    # ── Wallet ────────────────────────────────────────────────────────────

    async def get_balances(self) -> Dict:
        """Get wallet balances."""
        return await self._raw_get("/agent/wallet/balances")

    async def get_portfolio(self) -> Dict:
        """Get portfolio summary."""
        return await self._raw_get("/agent/wallet/portfolio")

    # ── Agent identity ────────────────────────────────────────────────────

    async def whoami(self) -> Dict:
        """Get current agent identity info."""
        return await self._raw_get("/agent/auth/me")
