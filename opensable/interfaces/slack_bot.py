"""
Slack Integration for Open-Sable

Features:
- Streaming responses: bot posts a message then edits it as tokens arrive
- Pairing / allowlist system: first user auto-approved as owner,
  subsequent users need a pairing code approved by owner
- Multimodal: image analysis (file_shared), voice note stubs
- Slash commands: /sable <text>
- Thread-aware replies in channels; DMs always processed
- Typing indicator while agent works
"""

import asyncio
import logging
import os
import secrets
import string
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _generate_pair_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ──────────────────────────────────────────────────────────────────────
# Pairing store (mirrors Telegram's PairingStore)
# ──────────────────────────────────────────────────────────────────────


class SlackPairingStore:
    """
    Allowlist + pairing codes for Slack users.

    - Owner is auto-approved on first message if SLACK_ALLOWED_USERS is empty
    - New users receive a pairing code; owner must /sable pair approve <code>
    """

    def __init__(self, config):
        self.config = config
        self._path = Path.home() / ".opensable" / "slack_pairing.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.allowlist: dict = {}
        self.pending: dict = {}
        self._load()

        # Seed from env
        for uid in getattr(config, "slack_allowed_users", None) or []:
            if uid and uid not in self.allowlist:
                self.allowlist[uid] = {
                    "username": "env-seeded",
                    "approved_at": datetime.utcnow().isoformat(),
                }

    def is_allowed(self, user_id: str) -> bool:
        return user_id in self.allowlist

    def has_owner(self) -> bool:
        return bool(self.allowlist)

    def approve_first(self, user_id: str, username: str):
        self.allowlist[user_id] = {
            "username": username,
            "approved_at": datetime.utcnow().isoformat(),
            "role": "owner",
        }
        self._save()
        logger.info(f"[Slack] Auto-approved first user {username} ({user_id}) as owner")

    def create_pairing_code(self, user_id: str, username: str) -> str:
        for code, info in list(self.pending.items()):
            if info["user_id"] == user_id:
                expires = datetime.fromisoformat(info["expires"])
                if expires > datetime.utcnow():
                    return code
                del self.pending[code]
        code = _generate_pair_code()
        expires = (datetime.utcnow() + timedelta(minutes=30)).isoformat()
        self.pending[code] = {"user_id": user_id, "username": username, "expires": expires}
        self._save()
        return code

    def approve_code(self, code: str) -> Optional[dict]:
        code = code.upper()
        if code not in self.pending:
            return None
        info = self.pending.pop(code)
        expires = datetime.fromisoformat(info["expires"])
        if expires < datetime.utcnow():
            self._save()
            return None
        self.allowlist[info["user_id"]] = {
            "username": info["username"],
            "approved_at": datetime.utcnow().isoformat(),
            "role": "user",
        }
        self._save()
        return info

    def revoke(self, user_id: str) -> bool:
        if user_id in self.allowlist:
            del self.allowlist[user_id]
            self._save()
            return True
        return False

    def owner_id(self) -> Optional[str]:
        for uid, info in self.allowlist.items():
            if info.get("role") == "owner":
                return uid
        return next(iter(self.allowlist), None)

    def _save(self):
        self._path.write_text(
            json.dumps({"allowlist": self.allowlist, "pending": self.pending}, indent=2)
        )

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self.allowlist = data.get("allowlist", {})
                self.pending = data.get("pending", {})
            except Exception as e:
                logger.warning(f"Could not load Slack pairing store: {e}")


# ──────────────────────────────────────────────────────────────────────
# Main Slack Interface
# ──────────────────────────────────────────────────────────────────────


class SlackInterface:
    """Slack bot with streaming, pairing, multimodal, and commands."""

    _STREAM_CHUNK = 100  # chars between edits
    _STREAM_INTERVAL = 1.5  # seconds between edits

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.app = None
        self.session_manager = None
        self.command_handler = None
        self.pairing: Optional[SlackPairingStore] = None
        self.image_analyzer = None
        self.voice_handler = None
        self._bot_user_id_cache: Optional[str] = None

    # ─────────────────────────────────── lifecycle ─────────────────────

    async def initialize(self):
        """Initialize Slack app + pairing + optional multimodal."""
        try:
            from slack_bolt.async_app import AsyncApp
            from opensable.core.session_manager import SessionManager
            from opensable.core.commands import CommandHandler

            self.session_manager = SessionManager()
            self.command_handler = CommandHandler(
                self.session_manager,
                plugin_manager=getattr(self.agent, "plugins", None),
            )
            self.pairing = SlackPairingStore(self.config)

            self.app = AsyncApp(
                token=self.config.slack_bot_token,
                signing_secret=self.config.slack_signing_secret,
            )

            # Event handlers
            self.app.message("")(self.handle_message)
            self.app.command("/sable")(self.handle_slash_command)
            self.app.event("file_shared")(self.handle_file_shared)

            # Optional multimodal
            try:
                from opensable.core.image_analyzer import ImageAnalyzer

                self.image_analyzer = ImageAnalyzer(self.config)
                await self.image_analyzer.initialize()
            except Exception:
                logger.debug("Image analyzer not available for Slack")

            try:
                from opensable.core.voice_handler import VoiceMessageHandler

                self.voice_handler = VoiceMessageHandler(self.config, self.agent)
                await self.voice_handler.initialize()
            except Exception:
                logger.debug("Voice handler not available for Slack")

            logger.info("Slack app initialized (streaming + pairing + multimodal)")
            return True

        except ImportError:
            logger.error("slack-bolt not installed. Install with: pip install slack-bolt")
            return False
        except Exception as e:
            logger.error(f"Error initializing Slack: {e}", exc_info=True)
            return False

    async def start(self):
        """Start Slack app in Socket Mode."""
        if not await self.initialize():
            logger.error("Failed to initialize Slack")
            return
        try:
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

            handler = AsyncSocketModeHandler(self.app, self.config.slack_app_token)
            logger.info("Starting Slack bot in Socket Mode...")
            await handler.start_async()
        except Exception as e:
            logger.error(f"Slack error: {e}", exc_info=True)

    async def stop(self):
        """Stop Slack app."""
        logger.info("Stopping Slack interface...")
        if self.session_manager:
            self.session_manager.save_to_disk()

    # ─────────────────────────────── auth helpers ─────────────────────

    async def _get_bot_user_id(self, client) -> str:
        if not self._bot_user_id_cache:
            response = await client.auth_test()
            self._bot_user_id_cache = response["user_id"]
        return self._bot_user_id_cache

    async def _check_auth(self, user_id: str, username: str, client, channel_id: str) -> bool:
        """Check if user is allowed. Handle pairing flow for new users."""
        if not self.pairing.has_owner():
            self.pairing.approve_first(user_id, username)
            await client.chat_postMessage(
                channel=channel_id,
                text=(
                    f":wave: *Welcome, {username}!*\n"
                    f"You're the first user — auto-approved as *owner*.\n"
                    f"Your ID: `{user_id}`\nUse `/sable help` to see commands."
                ),
            )
            return True

        if self.pairing.is_allowed(user_id):
            return True

        # Issue pairing code
        code = self.pairing.create_pairing_code(user_id, username)
        await client.chat_postMessage(
            channel=channel_id,
            text=(
                f":lock: *Pairing required*\n"
                f"Send this code to the bot owner:\n`{code}`\n"
                f"_Code expires in 30 minutes._"
            ),
        )

        # Notify owner
        owner = self.pairing.owner_id()
        if owner:
            try:
                await client.chat_postMessage(
                    channel=owner,
                    text=(
                        f":bell: *Pairing request*\n"
                        f"User *{username}* (`{user_id}`) wants access.\n"
                        f"Approve: `/sable pair approve {code}`\n"
                        f"Deny:    `/sable pair deny {code}`"
                    ),
                )
            except Exception:
                pass
        return False

    # ──────────────────────────── pairing commands ────────────────────

    async def _handle_pair_command(self, args: list, user_id: str, say):
        """Process pair approve|deny|revoke subcommands."""
        if not args:
            await say(
                ":key: *Pairing commands*\n"
                "`pair approve <CODE>` — approve a pending user\n"
                "`pair deny <CODE>` — deny a pending user\n"
                "`pair revoke <USER_ID>` — revoke access"
            )
            return

        action = args[0].lower()

        if action == "approve" and len(args) > 1:
            result = self.pairing.approve_code(args[1])
            if result:
                await say(
                    f":white_check_mark: User *{result['username']}* (`{result['user_id']}`) approved."
                )
            else:
                await say(":x: Code not found or expired.")

        elif action == "deny" and len(args) > 1:
            code = args[1].upper()
            if code in self.pairing.pending:
                info = self.pairing.pending.pop(code)
                self.pairing._save()
                await say(f":white_check_mark: Denied pairing from {info['username']}.")
            else:
                await say(":x: Code not found.")

        elif action == "revoke" and len(args) > 1:
            if self.pairing.revoke(args[1]):
                await say(f":white_check_mark: User `{args[1]}` revoked.")
            else:
                await say(":x: User not found in allowlist.")
        else:
            await say(":x: Unknown pair action. Use: approve | deny | revoke")

    # ──────────────────────────── message handler ─────────────────────

    async def handle_message(self, message, say, client):
        """Handle incoming Slack messages with streaming."""
        try:
            user_id = message.get("user")
            text = message.get("text", "")
            channel_id = message.get("channel")
            thread_ts = message.get("thread_ts", message.get("ts"))

            if message.get("bot_id"):
                return

            # In channels, require @mention
            bot_uid = await self._get_bot_user_id(client)
            is_dm = message.get("channel_type") == "im"
            if not is_dm and f"<@{bot_uid}>" not in text:
                return
            text = text.replace(f"<@{bot_uid}>", "").strip()

            if not text:
                return

            # Get username for pairing
            try:
                info = await client.users_info(user=user_id)
                username = info["user"]["profile"].get("display_name") or info["user"]["name"]
            except Exception:
                username = user_id

            # Auth gate
            if not await self._check_auth(user_id, username, client, channel_id):
                return

            logger.info(f"[Slack] {username}: {text[:80]}")

            # Session
            session = self.session_manager.get_or_create_session(
                user_id=user_id,
                channel="slack",
            )

            # Check if it's a built-in command (pair, help, etc.)
            if text.lower().startswith("pair "):
                if user_id == self.pairing.owner_id():
                    await self._handle_pair_command(text.split()[1:], user_id, say)
                else:
                    await say(":x: Only the owner can manage pairing.")
                return

            # Route through command handler for /commands
            if self.command_handler.is_command(text):
                result = await self.command_handler.handle_command(
                    text=text,
                    session_id=session.id,
                    user_id=user_id,
                    is_admin=(user_id == self.pairing.owner_id()),
                    is_group=not is_dm,
                )
                if result.message:
                    await say(text=result.message, thread_ts=thread_ts)
                return

            # ── Streaming response ────────────────────────────────
            await self._stream_reply(
                text=text,
                user_id=user_id,
                session=session,
                client=client,
                channel_id=channel_id,
                thread_ts=thread_ts,
            )

        except Exception as e:
            logger.error(f"Error handling Slack message: {e}", exc_info=True)
            try:
                await say(
                    text="Sorry, I encountered an error processing your message.",
                    thread_ts=message.get("thread_ts", message.get("ts")),
                )
            except Exception:
                pass

    # ──────────────────────────── streaming reply ─────────────────────

    async def _stream_reply(
        self,
        text: str,
        user_id: str,
        session,
        client,
        channel_id: str,
        thread_ts: str,
    ):
        """Post a placeholder, then edit as tokens arrive from Ollama streaming."""

        # Add user message to history
        session.add_message("user", text)
        self.session_manager._save_session(session)
        history = session.get_llm_messages(limit=20)

        # Post placeholder
        placeholder = await client.chat_postMessage(
            channel=channel_id,
            text=":thought_balloon: _thinking…_",
            thread_ts=thread_ts,
        )
        ph_ts = placeholder["ts"]

        full_text = ""
        try:
            import ollama as _ollama

            ol_client = _ollama.AsyncClient(host=self.config.ollama_base_url)

            # Build messages
            if hasattr(self.agent, "_get_personality_prompt"):
                system_prompt = self.agent._get_personality_prompt()
            else:
                system_prompt = "You are Sable, a helpful AI assistant."
            system_prompt += (
                "\n\nCRITICAL RULES:\n"
                "- NEVER invent, fabricate, or hallucinate facts\n"
                "- If you don't know something, say so\n"
                "- Use Slack-compatible markdown (*bold*, _italic_, `code`)"
            )

            msgs = [{"role": "system", "content": system_prompt}]
            msgs += history[:-1]
            msgs.append({"role": "user", "content": text})

            model = (
                self.agent.llm.current_model
                if hasattr(self.agent, "llm") and hasattr(self.agent.llm, "current_model")
                else self.config.default_model
            )

            buffer = ""
            last_edit = 0.0

            async for chunk in await ol_client.chat(
                model=model,
                messages=msgs,
                stream=True,
            ):
                delta = chunk.get("message", {}).get("content", "")
                full_text += delta
                buffer += delta
                now = asyncio.get_event_loop().time()

                if len(buffer) >= self._STREAM_CHUNK or (now - last_edit) >= self._STREAM_INTERVAL:
                    if full_text.strip():
                        try:
                            await client.chat_update(
                                channel=channel_id,
                                ts=ph_ts,
                                text=full_text + " :black_medium_small_square:",
                            )
                        except Exception:
                            pass
                        buffer = ""
                        last_edit = now

            # Final edit — remove cursor
            if full_text.strip():
                try:
                    await client.chat_update(channel=channel_id, ts=ph_ts, text=full_text)
                except Exception:
                    pass
            else:
                # Streaming produced nothing — fall back to agent
                full_text = await self.agent.process_message(user_id, text, history=history)
                await client.chat_update(channel=channel_id, ts=ph_ts, text=full_text)

        except Exception as e:
            logger.warning(f"[Slack] Stream failed ({e}), falling back to agent")
            try:
                full_text = await self.agent.process_message(user_id, text, history=history)
                await client.chat_update(channel=channel_id, ts=ph_ts, text=str(full_text))
            except Exception as e2:
                logger.error(f"[Slack] Fallback also failed: {e2}")
                await client.chat_update(
                    channel=channel_id,
                    ts=ph_ts,
                    text=":x: Sorry, I encountered an error.",
                )

        # Persist assistant reply
        if full_text:
            session.add_message("assistant", full_text)
            self.session_manager._save_session(session)

    # ─────────────────────────── slash command ────────────────────────

    async def handle_slash_command(self, ack, command, say, client):
        """Handle /sable slash command."""
        await ack()
        try:
            user_id = command.get("user_id")
            text = command.get("text", "").strip()
            channel_id = command.get("channel_id")

            # Auth check
            try:
                info = await client.users_info(user=user_id)
                username = info["user"]["profile"].get("display_name") or info["user"]["name"]
            except Exception:
                username = user_id

            if not await self._check_auth(user_id, username, client, channel_id):
                await say(":lock: Not authorized. A pairing code has been sent.")
                return

            if not text or text.lower() == "help":
                await say(
                    ":robot_face: *Sable Commands*\n\n"
                    "`/sable <message>` — chat with Sable\n"
                    "`/sable status` — session info\n"
                    "`/sable reset` — clear conversation\n"
                    "`/sable pair approve|deny|revoke` — manage users\n"
                    "`/sable help` — this message\n\n"
                    "Or just mention <@me> in any channel!"
                )
                return

            # Pair sub-command
            if text.lower().startswith("pair"):
                if user_id == self.pairing.owner_id():
                    await self._handle_pair_command(text.split()[1:], user_id, say)
                else:
                    await say(":x: Only the owner can manage pairing.")
                return

            # Pass through command handler first
            session = self.session_manager.get_or_create_session(user_id=user_id, channel="slack")
            if self.command_handler.is_command(text):
                result = await self.command_handler.handle_command(
                    text=text,
                    session_id=session.id,
                    user_id=user_id,
                    is_admin=(user_id == self.pairing.owner_id()),
                    is_group=False,
                )
                if result.message:
                    await say(result.message)
                return

            # Regular message — process through agent
            response = await self.agent.process_message(user_id, text)
            await say(str(response))

        except Exception as e:
            logger.error(f"Error handling slash command: {e}", exc_info=True)
            await say(":x: Sorry, I encountered an error.")

    # ────────────────────────── file / image handler ──────────────────

    async def handle_file_shared(self, event, client, say):
        """Handle shared files — analyse images, transcribe audio."""
        try:
            file_id = event.get("file_id")
            if not file_id:
                return

            file_info = await client.files_info(file=file_id)
            file_data = file_info.get("file", {})
            mimetype = file_data.get("mimetype", "")
            channel_id = event.get("channel_id", "")
            user_id = file_data.get("user", "")

            # Auth gate
            try:
                uinfo = await client.users_info(user=user_id)
                username = uinfo["user"]["profile"].get("display_name") or uinfo["user"]["name"]
            except Exception:
                username = user_id

            if not self.pairing or not self.pairing.is_allowed(user_id):
                return

            # ── Image ──
            if mimetype.startswith("image/") and self.image_analyzer:
                await client.chat_postMessage(
                    channel=channel_id, text=":frame_with_picture: Analysing image…"
                )

                # Download image
                import aiohttp

                url = file_data.get("url_private_download") or file_data.get("url_private")
                if not url:
                    return
                headers = {"Authorization": f"Bearer {self.config.slack_bot_token}"}
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(url, headers=headers) as resp:
                        image_bytes = await resp.read()

                # Analyse
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(image_bytes)
                    tmp_path = tmp.name

                description = await self.image_analyzer.analyze_image(
                    tmp_path, prompt=file_data.get("title", "Describe this image")
                )
                os.unlink(tmp_path)

                response = f":frame_with_picture: *Image analysis:*\n{description}"
                await client.chat_postMessage(channel=channel_id, text=response)
                return

            # ── Audio ──
            if mimetype.startswith("audio/") and self.voice_handler:
                await client.chat_postMessage(
                    channel=channel_id, text=":microphone: Processing voice…"
                )

                import aiohttp

                url = file_data.get("url_private_download") or file_data.get("url_private")
                if not url:
                    return
                headers = {"Authorization": f"Bearer {self.config.slack_bot_token}"}
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(url, headers=headers) as resp:
                        audio_bytes = await resp.read()

                result = await self.voice_handler.process_voice_message(audio_bytes, user_id)
                if result.get("success"):
                    text = result.get("transcription", "")
                    response = result.get("response_text", "")
                    await client.chat_postMessage(
                        channel=channel_id,
                        text=f":microphone: *You said:* {text}\n\n{response}",
                    )
                else:
                    await client.chat_postMessage(
                        channel=channel_id,
                        text=f":x: Voice processing failed: {result.get('error', 'unknown')}",
                    )
                return

        except Exception as e:
            logger.error(f"[Slack] file_shared error: {e}", exc_info=True)
