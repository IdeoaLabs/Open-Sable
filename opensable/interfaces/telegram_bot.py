"""
Telegram Bot Interface,  Sable

Features:
- Persistent conversation history per user (survives restarts)
- Streaming responses,  bot edits the message token-by-token via Ollama
- Slash commands: /status /reset /new /compact /think /verbose /voice /model /usage /help
- Pairing/allowlist system:
    * If TELEGRAM_ALLOWED_USERS is empty → first user is auto-authorized (owner)
    * New users receive a pairing code; owner must /pair approve <code>
    * No open ports,  pairing is pure in-bot DM exchange
- Group support with activation mode (mention|always)
- Markdown-safe replies (falls back to plain text on parse errors)
- NO external ports opened,  only Telegram long-polling
"""

import asyncio
import logging
import re
import secrets
import string
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from opensable.core.paths import opensable_home

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.enums import ParseMode, ChatType

from opensable.core.session_manager import SessionManager, SessionConfig
from opensable.core.commands import CommandHandler
from opensable.core.heartbeat import HeartbeatManager
from opensable.core.voice_handler import VoiceMessageHandler
from opensable.core.image_analyzer import ImageAnalyzer, handle_telegram_photo
from opensable.interfaces.telegram_group import (
    PromptPoisonFilter,
    GroupMemory,
    EngagementEngine,
    RateLimiter,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_MD2_CHARS = r"_*[]()~`>#+-=|{}.!"


def _escape_md2(text: str) -> str:
    return re.sub(r"([" + re.escape(_MD2_CHARS) + r"])", r"\\\1", text)


def _generate_pair_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


# ──────────────────────────────────────────────────────────────────────────────
# Pairing store  (in-memory + persisted to ~/.opensable/pairing.json)
# ──────────────────────────────────────────────────────────────────────────────


class PairingStore:
    """
    Manages who is allowed to talk to the bot.

    Security model:
      - Owner slot is filled automatically by the first /start
      - After that, new senders get a pairing code in their DM
      - Owner approves with /pair approve <code>
      - Approved users are added to the allowlist and persisted to disk
      - No ports, no HTTP,  all via Telegram DMs
    """

    def __init__(self, config):
        self.config = config
        self._path = opensable_home() / "pairing.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # {user_id: {"username": str, "approved_at": iso}}
        self.allowlist: dict = {}
        # {code: {"user_id": str, "username": str, "expires": iso}}
        self.pending: dict = {}

        self._load()

        # Seed allowlist from env if provided
        for uid in config.telegram_allowed_users:
            if uid and uid not in self.allowlist:
                self.allowlist[uid] = {
                    "username": "env-seeded",
                    "approved_at": datetime.now(timezone.utc).isoformat(),
                }

    # ------------------------------------------------------------------

    def is_allowed(self, user_id: str) -> bool:
        return user_id in self.allowlist

    def has_owner(self) -> bool:
        return bool(self.allowlist)

    def approve_first(self, user_id: str, username: str):
        """Auto-approve the very first user as owner."""
        self.allowlist[user_id] = {
            "username": username,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "role": "owner",
        }
        self._save()
        logger.info(f"Auto-approved first user {username} ({user_id}) as owner")

        # Also persist to .env
        try:
            env = Path(".env")
            if env.exists():
                lines = env.read_text().splitlines()
                updated = False
                for i, ln in enumerate(lines):
                    if ln.startswith("TELEGRAM_ALLOWED_USERS="):
                        lines[i] = f"TELEGRAM_ALLOWED_USERS={user_id}"
                        updated = True
                        break
                if not updated:
                    lines.append(f"TELEGRAM_ALLOWED_USERS={user_id}")
                env.write_text("\n".join(lines) + "\n")
        except Exception as e:
            logger.warning(f"Could not update .env: {e}")

    def create_pairing_code(self, user_id: str, username: str) -> str:
        """Create a pairing code for an unknown user. Expires in 30 min."""
        # Reuse existing pending code if not expired
        for code, info in list(self.pending.items()):
            if info["user_id"] == user_id:
                expires = datetime.fromisoformat(info["expires"])
                if expires > datetime.now(timezone.utc):
                    return code
                del self.pending[code]

        code = _generate_pair_code()
        expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        self.pending[code] = {"user_id": user_id, "username": username, "expires": expires}
        self._save()
        return code

    def approve_code(self, code: str) -> Optional[dict]:
        """
        Approve a pending code. Returns the approved user dict or None.
        Removes expired codes automatically.
        """
        code = code.upper()
        if code not in self.pending:
            return None
        info = self.pending.pop(code)
        expires = datetime.fromisoformat(info["expires"])
        if expires < datetime.now(timezone.utc):
            self._save()
            return None
        self.allowlist[info["user_id"]] = {
            "username": info["username"],
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "role": "user",
        }
        self._save()
        logger.info(f"Paired user {info['username']} ({info['user_id']}) via code {code}")
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
        # Fallback: first entry
        return next(iter(self.allowlist), None)

    def _save(self):
        self._path.write_text(
            __import__("json").dumps(
                {"allowlist": self.allowlist, "pending": self.pending}, indent=2
            )
        )

    def _load(self):
        import json

        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self.allowlist = data.get("allowlist", {})
                self.pending = data.get("pending", {})
            except Exception as e:
                logger.warning(f"Could not load pairing store: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Telegram Interface
# ──────────────────────────────────────────────────────────────────────────────


class TelegramInterface:
    """Telegram bot with streaming, pairing, persistent sessions, and commands."""

    # Streaming: edit the placeholder message every N chars or N seconds
    _STREAM_CHUNK = 80  # characters before editing
    _STREAM_INTERVAL = 1.5  # seconds between edits (Telegram rate-limit ~20 edits/min)

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.bot: Optional[Bot] = None
        self.dp: Optional[Dispatcher] = None
        self.session_manager = SessionManager()
        self.command_handler = CommandHandler(
            self.session_manager,
            plugin_manager=getattr(agent, "plugins", None),
        )
        self.pairing = PairingStore(config)

        # Heartbeat manager for proactive checking
        self.heartbeat = HeartbeatManager(agent, config)

        # Voice message handler
        self.voice_handler = VoiceMessageHandler(config, agent)

        # Image analyzer
        self.image_analyzer = ImageAnalyzer(config)

        # Group intelligence layer
        self._poison_filter = PromptPoisonFilter()
        self._group_memory = GroupMemory(
            max_messages=getattr(config, "telegram_group_context_size", 30)
        )
        self._engagement = None  # initialized after bot.get_me in start()
        self._rate_limiter = RateLimiter(max_requests=5, window_sec=120)
        self._group_only = getattr(config, "telegram_group_only", False)
        self._allowed_groups = set(getattr(config, "telegram_allowed_groups", []))
        self._anti_poison = getattr(config, "telegram_anti_poison", True)
        self._engagement_level = getattr(config, "telegram_group_engagement", "medium")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        if not self.config.telegram_bot_token:
            logger.error("TELEGRAM_BOT_TOKEN not set")
            return

        self.bot = Bot(token=self.config.telegram_bot_token)
        self.dp = Dispatcher()

        # Register handlers (order matters,  specific filters first, catch-all last)
        self.dp.message.register(self._h_start, CommandStart())
        self.dp.message.register(self._h_voice, F.voice)
        self.dp.message.register(self._h_photo, F.photo)
        self.dp.message.register(self._h_message)  # catch-all: slash commands + regular text

        # Register callback query handler for inline buttons
        self.dp.callback_query.register(self._h_callback)

        # Initialize engagement engine (needs bot username)
        bot_me = await self.bot.get_me()
        self._bot_username = bot_me.username or ""
        self._bot_id = bot_me.id
        self._engagement = EngagementEngine(
            bot_name=self._bot_username or self.config.agent_name,
            engagement=self._engagement_level,
        )
        logger.info(
            f"Group mode: only={self._group_only}  "
            f"engagement={self._engagement_level}  "
            f"anti_poison={self._anti_poison}"
        )

        # Start heartbeat for proactive checking
        from opensable.core.heartbeat import (
            check_system_health,
            check_pending_tasks,
            check_idle_time,
            check_supply_chain,
        )

        self.heartbeat.register_check(check_system_health, "System Health")
        self.heartbeat.register_check(check_pending_tasks, "Pending Goals")
        self.heartbeat.register_check(check_idle_time, "Idle Check")
        self.heartbeat.register_check(check_supply_chain, "Supply Chain")

        # Wire heartbeat alerts to Telegram
        owner_id = self.pairing.owner_id()
        if owner_id:

            async def _tg_notify(msg: str):
                try:
                    await self.bot.send_message(int(owner_id), msg)
                except Exception as e:
                    logger.debug(f"Heartbeat Telegram notify failed: {e}")

            self.agent._telegram_notify = _tg_notify

            # Wire photo sending for IG autoposter → Telegram forwarding
            _bot_ref = self.bot
            _owner_int = int(owner_id)

            async def _tg_send_photo(photo_path: str, caption: str = ""):
                try:
                    from aiogram.types import FSInputFile
                    photo = FSInputFile(photo_path)
                    await _bot_ref.send_photo(
                        chat_id=_owner_int,
                        photo=photo,
                        caption=caption[:1024] if caption else "",
                    )
                except Exception as e:
                    logger.debug(f"Telegram send_photo failed: {e}")

            self.agent._telegram_send_photo = _tg_send_photo

            # Also wire notifications to X API queue for error alerts
            try:
                from opensable.core.x_api_queue import XApiQueue
                XApiQueue.get_instance().set_notify(_tg_notify)
            except Exception:
                pass

        await self.heartbeat.start()

        logger.info("Telegram bot starting (long-polling, no open ports)...")
        try:
            await self.dp.start_polling(self.bot, allowed_updates=["message", "callback_query"])
        except Exception as e:
            logger.error(f"Telegram polling error: {e}", exc_info=True)

    async def stop(self):
        if self.heartbeat:
            await self.heartbeat.stop()
        if self.bot:
            await self.bot.session.close()
        self.session_manager.save_to_disk()

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    async def _check_auth(self, message: Message) -> bool:
        """Return True if user is allowed. Side-effect: handle pairing for new users."""
        user_id = str(message.from_user.id)
        username = message.from_user.username or message.from_user.first_name or user_id

        # First ever user → auto-approve as owner
        if not self.pairing.has_owner():
            self.pairing.approve_first(user_id, username)
            await message.answer(
                f"👋 **Welcome, {username}!**\n\n"
                f"You're the first user,  auto-approved as **owner**.\n"
                f"Your ID: `{user_id}`\n\n"
                f"Use /help to see available commands.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return True

        if self.pairing.is_allowed(user_id):
            return True

        # Unknown user,  issue pairing code
        code = self.pairing.create_pairing_code(user_id, username)
        await message.answer(
            f"🔐 **Pairing required**\n\n"
            f"You are not yet authorized. Send the following code to the bot owner:\n\n"
            f"`{code}`\n\n"
            f"_Code expires in 30 minutes._",
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info(f"Pairing code {code} issued to {username} ({user_id})")

        # Notify owner
        owner_id = self.pairing.owner_id()
        if owner_id:
            try:
                await self.bot.send_message(
                    owner_id,
                    f"🔔 **Pairing request**\n\n"
                    f"User **{username}** (`{user_id}`) wants access.\n"
                    f"Approve with: `/pair approve {code}`\n"
                    f"Deny with:    `/pair deny {code}`",
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                pass
        return False

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _h_start(self, message: Message):
        if not await self._check_auth(message):
            return
        user_id = str(message.from_user.id)
        session = self.session_manager.get_or_create_session(
            user_id=user_id,
            channel="telegram",
            config=SessionConfig(model=self.config.default_model),
        )
        name = message.from_user.first_name or "there"
        age_h = (
            datetime.now(timezone.utc) - session.created_at.replace(tzinfo=timezone.utc)
        ).total_seconds() / 3600
        is_new = age_h < 0.1
        if is_new:
            text = (
                f"👋 **Hello, {name}!**\n\n"
                f"I'm **{self.config.agent_name}**, your personal AI assistant.\n\n"
                f"I can browse the web, run commands, read files, check weather, "
                f"manage your calendar, and more.\n\n"
                f"Just chat naturally, or use /help to see commands."
            )
        else:
            text = (
                f"👋 **Welcome back, {name}!**\n\n"
                f"Session: `{session.id[:10]}...` · "
                f"{len(session.messages)} messages · {age_h:.1f} h old\n\n"
                f"Pick up where we left off, or /reset to start fresh."
            )
        await message.answer(text, parse_mode=ParseMode.MARKDOWN)

    async def _h_help(self, message: Message):
        if not await self._check_auth(message):
            return
        text = (
            "🤖 **Sable Commands**\n\n"
            "/status,  session info (model, tokens, uptime)\n"
            "/reset ,  clear conversation history\n"
            "/new   ,  same as /reset\n"
            "/compact,  summarise old messages\n"
            "/think `<level>`,  off|minimal|low|medium|high|xhigh\n"
            "/verbose `on|off`,  toggle detailed output\n"
            "/voice `on|off`,  toggle voice mode\n"
            "/model `<name>`,  switch AI model\n"
            "/usage `full|tokens|off`,  usage footer\n"
            "/pair `approve|deny <code>`,  manage pairing\n"
            "/help,  this message\n\n"
            "Just chat naturally for everything else!"
        )

        # Example inline buttons
        buttons = [
            [
                {"text": "📊 Status", "callback_data": "cmd:/status"},
                {"text": "🔄 Reset", "callback_data": "cmd:/reset"},
            ],
            [{"text": "🌐 Search Web", "callback_data": "cmd:search latest AI news"}],
        ]

        await self._safe_reply(message, text, buttons=buttons)

    async def _h_pair(self, message: Message):
        """Owner-only pairing management."""
        user_id = str(message.from_user.id)
        if not self.pairing.is_allowed(user_id):
            return

        args = (message.text or "").split()[1:]
        if not args:
            await message.answer(
                "Usage:\n"
                "`/pair approve <CODE>`,  approve a pending user\n"
                "`/pair deny <CODE>`,  deny a pending user\n"
                "`/pair revoke <USER_ID>`,  revoke existing user",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        action = args[0].lower()

        if action == "approve" and len(args) > 1:
            result = self.pairing.approve_code(args[1])
            if result:
                await message.answer(
                    f"✅ User **{result['username']}** (`{result['user_id']}`) approved.",
                    parse_mode=ParseMode.MARKDOWN,
                )
                try:
                    await self.bot.send_message(
                        result["user_id"],
                        "✅ You've been approved! You can now use Sable.",
                    )
                except Exception:
                    pass
            else:
                await message.answer("❌ Code not found or expired.")

        elif action == "deny" and len(args) > 1:
            code = args[1].upper()
            if code in self.pairing.pending:
                info = self.pairing.pending.pop(code)
                self.pairing._save()
                await message.answer(f"✅ Denied pairing request from {info['username']}.")
                try:
                    await self.bot.send_message(
                        info["user_id"],
                        "❌ Your pairing request was denied.",
                    )
                except Exception:
                    pass
            else:
                await message.answer("❌ Code not found.")

        elif action == "revoke" and len(args) > 1:
            if self.pairing.revoke(args[1]):
                await message.answer(f"✅ User `{args[1]}` revoked.", parse_mode=ParseMode.MARKDOWN)
            else:
                await message.answer("❌ User not found in allowlist.")

        else:
            await message.answer("❌ Unknown pair action. Use: approve|deny|revoke")

    async def _h_callback(self, callback: CallbackQuery):
        """Handle inline button clicks."""
        user_id = str(callback.from_user.id)

        # Auth check
        if not self.pairing.is_allowed(user_id):
            await callback.answer("⛔ Not authorized", show_alert=True)
            return

        data = callback.data or ""

        # Parse callback data format: "cmd:action"
        if ":" in data:
            prefix, action = data.split(":", 1)

            if prefix == "cmd":
                # Execute command action
                session = self.session_manager.get_or_create_session(
                    user_id=user_id,
                    channel="telegram",
                    config=SessionConfig(model=self.config.default_model),
                )

                # Process the action through the agent
                history = session.get_llm_messages(limit=20)
                response = await self.agent.process_message(user_id, action, history=history)

                # Send response and close button
                await callback.message.answer(response, parse_mode=ParseMode.MARKDOWN)
                await callback.answer("✅ Executed")

                # Update session
                session.add_message("user", f"[Button: {action}]")
                session.add_message("assistant", response)
                self.session_manager._save_session(session)
            else:
                await callback.answer("Unknown action")

    async def _h_voice(self, message: Message):
        """Handle voice messages"""
        if not await self._check_auth(message):
            return

        user_id = str(message.from_user.id)
        session = self.session_manager.get_or_create_session(
            user_id=user_id,
            channel="telegram",
            config=SessionConfig(model=self.config.default_model),
        )

        try:
            await message.bot.send_chat_action(message.chat.id, "typing")

            # Download voice file
            voice_file = await message.voice.download_to_drive()
            audio_bytes = Path(voice_file.name).read_bytes()

            # Check if user wants voice response
            voice_enabled = session.config.use_voice

            # Process voice message
            status_msg = await message.answer("🎙️ Processing voice message...")

            result = await self.voice_handler.process_voice_message(
                audio_bytes, user_id, respond_with_voice=voice_enabled
            )

            await status_msg.delete()

            if not result.get("success"):
                await message.answer(f"❌ {result.get('error', 'Voice processing failed')}")
                return

            # Send transcription + text response
            transcription = result.get("transcription", "")
            response_text = result.get("response_text", "")

            formatted_response = f"🎙️ *You said:* {transcription}\n\n{response_text}"
            await self._safe_reply(message, formatted_response)

            # Send voice response if enabled
            if voice_enabled and "voice_data" in result:
                voice_bytes = result["voice_data"]
                await message.answer_voice(voice=voice_bytes, caption="🔊 Voice response")

            # Update session history
            session.add_message("user", f"[Voice: {transcription}]")
            session.add_message("assistant", response_text)
            self.session_manager._save_session(session)

        except Exception as e:
            logger.error(f"Voice message error: {e}", exc_info=True)
            await message.answer(f"❌ Voice processing error: {str(e)}")

    async def _h_photo(self, message: Message):
        """Handle photo messages"""
        if not await self._check_auth(message):
            return

        user_id = str(message.from_user.id)

        try:
            await message.bot.send_chat_action(message.chat.id, "typing")

            status_msg = await message.answer("🖼️ Analyzing image...")

            response = await handle_telegram_photo(
                message, self.image_analyzer, self.agent, user_id
            )

            await status_msg.delete()
            await self._safe_reply(message, response)

        except Exception as e:
            logger.error(f"Photo handling error: {e}", exc_info=True)
            await message.answer(f"❌ Image analysis error: {str(e)}")

    async def _h_message(self, message: Message):
        """Main message handler with group intelligence layer."""
        if not message.text:
            return

        user_id = str(message.from_user.id)
        username = message.from_user.username or message.from_user.first_name or user_id
        text = message.text.strip()
        is_group = message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
        group_id = str(message.chat.id)

        # ── DM gate: group-only mode ───────────────────────────────────
        if self._group_only and not is_group:
            is_owner = user_id == self.pairing.owner_id()
            if not is_owner:
                await message.answer(
                    "Hey — I only chat inside the group. "
                    "Come talk to me there!"
                )
                return
            # Owner DMs always allowed (for /pair, admin, etc.)

        # ── Group allowlist ────────────────────────────────────────────
        if is_group and self._allowed_groups and group_id not in self._allowed_groups:
            return  # silently ignore groups we're not configured for

        # ── Auth gate (pairing) ────────────────────────────────────────
        if not is_group:
            # Private chats still use full pairing auth
            if not await self._check_auth(message):
                return

        # ── Anti-prompt-poisoning filter ───────────────────────────────
        if self._anti_poison and not self._poison_filter.is_safe(text):
            matched = self._poison_filter.matched_pattern(text)
            logger.warning(
                f"[Telegram] Prompt injection blocked from {username} "
                f"({user_id}): {matched}"
            )
            if is_group:
                return  # silently drop in groups
            await message.answer("Nice try. 🛡️")
            return

        # ── Group intelligence layer ───────────────────────────────────
        if is_group:
            # Always observe — even if we don't respond
            self._group_memory.observe(
                group_id, username, user_id, text, is_bot=False,
            )

            # Detect mention / reply-to-bot
            is_mentioned = (
                f"@{self._bot_username}" in text if self._bot_username else False
            )
            is_reply_to_bot = (
                message.reply_to_message is not None
                and message.reply_to_message.from_user is not None
                and message.reply_to_message.from_user.id == self._bot_id
            )

            # Strip @mention from text before processing
            if is_mentioned and self._bot_username:
                text = text.replace(f"@{self._bot_username}", "").strip()

            # Slash commands in groups always handled if directed at bot
            if text.startswith("/"):
                pass  # fall through to command handling below
            else:
                # Engagement decision
                score = self._engagement.score(
                    text, self._group_memory, group_id,
                    is_reply_to_bot=is_reply_to_bot,
                    is_mentioned=is_mentioned,
                )
                decision = self._engagement.decide(score)

                if decision == "no":
                    return  # stay silent, we observed already

                if decision == "maybe":
                    # Tier 2: lightweight LLM tiebreak
                    llm = getattr(self.agent, "llm", None)
                    if llm and hasattr(llm, "acomplete"):
                        ctx = self._group_memory.format_context(group_id, 15)
                        should = await EngagementEngine.llm_tiebreak(
                            llm, ctx, username, text,
                        )
                        if not should:
                            logger.debug(
                                f"[Telegram] LLM tiebreak → skip "
                                f"(score={score}, user={username})"
                            )
                            return
                    else:
                        # No LLM available for tiebreak → skip borderline
                        return

                # Rate limit per user in groups
                if not self._rate_limiter.allow(user_id):
                    logger.debug(f"[Telegram] Rate-limited {username} in group")
                    return

                logger.info(
                    f"[Telegram] Engaging in group (score={score}, "
                    f"mention={is_mentioned}, reply={is_reply_to_bot})"
                )

        # ── Session — per user, per channel ────────────────────────────
        channel_key = f"telegram_group_{group_id}" if is_group else "telegram"
        session = self.session_manager.get_or_create_session(
            user_id=user_id,
            channel=channel_key,
            config=SessionConfig(model=self.config.default_model),
        )

        # ── Slash command handling ──────────────────────────────────────
        if text.startswith("/"):
            is_admin = user_id == self.pairing.owner_id()
            result = await self.command_handler.handle_command(
                text, session.id, user_id, is_admin=is_admin, is_group=is_group
            )
            if result.message:
                await self._safe_reply(message, result.message)
            if not result.should_continue:
                return

        # ── Inject group context so agent understands the conversation ─
        agent_text = text
        if is_group:
            ctx = self._group_memory.format_context(group_id, 12)
            if ctx:
                agent_text = (
                    f"[Recent group conversation:\n{ctx}\n]\n\n"
                    f"{username} says: {text}"
                )
        await message.bot.send_chat_action(message.chat.id, "typing")

        # Add user message to session history
        session.add_message("user", text)
        self.session_manager._save_session(session)

        # Build conversation history to pass to agent
        history = session.get_llm_messages(limit=20)

        response = await self._stream_to_telegram(
            message, user_id, agent_text, history,
        )

        # Persist assistant reply + group memory bookkeeping
        if response:
            session.add_message("assistant", response)
            self.session_manager._save_session(session)
            if is_group:
                self._group_memory.observe(
                    group_id, self._bot_username or "sable",
                    str(self._bot_id), response, is_bot=True,
                )
                self._group_memory.log_response(group_id)

        # ── Follow-through: if the agent promised to investigate, do it ──
        if response and self._promises_followup(response):
            logger.info("[Telegram] Agent promised a follow-up,  auto-continuing")
            await asyncio.sleep(1.5)
            await message.bot.send_chat_action(message.chat.id, "typing")

            followup_prompt = (
                "You just said you would investigate / look into something. "
                "Do it NOW,  use your tools (web search, etc.) and report "
                "the results. Do NOT say you will do it later. Act immediately."
            )
            session.add_message("user", followup_prompt)
            self.session_manager._save_session(session)
            followup_history = session.get_llm_messages(limit=20)

            try:
                followup = await asyncio.wait_for(
                    self._stream_to_telegram(
                        message, user_id, followup_prompt, followup_history
                    ),
                    timeout=120,  # 2 min max for follow-up
                )
                if followup:
                    session.add_message("assistant", followup)
                    self.session_manager._save_session(session)
            except asyncio.TimeoutError:
                logger.warning("[Telegram] Follow-up timed out after 120s")
            except Exception as e:
                logger.warning(f"[Telegram] Follow-up failed: {e}")

    # ------------------------------------------------------------------
    # Streaming + Follow-through
    # ------------------------------------------------------------------

    # Regex patterns that indicate the agent is promising to do something next
    _FOLLOWUP_PATTERNS = re.compile(
        r"(?:let me (?:investigate|check|look into|find out|search|verify|dig into|research|explore))"
        r"|(?:i(?:'ll| will) (?:investigate|check|look into|find out|search|verify|dig into|research|explore|look up))"
        r"|(?:(?:voy a|déjame|dejame|permíteme|permiteme) (?:investigar|verificar|buscar|revisar|explorar|comprobar))"
        r"|(?:let me (?:do|run|conduct) (?:a |an |the )?(?:investigation|search|check|analysis|audit|scan))"
        r"|(?:i(?:'ll| will) (?:do|run|conduct) (?:a |an |the )?(?:investigation|search|check|analysis))"
        r"|(?:we can investigate)"
        r"|(?:i want to understand)"
        r"|(?:let me map)"
        r"|(?:i tried to use a tool but)",
        re.IGNORECASE,
    )

    def _promises_followup(self, text: str) -> bool:
        """Return True if the agent's response ends with a promise to do more work.

        Only checks the LAST ~200 characters of the response.  If the promise
        phrase appears earlier (e.g. "Let me check … here are the results"),
        the agent already followed through and we should NOT re-trigger.
        """
        tail = text[-200:] if len(text) > 200 else text
        return bool(self._FOLLOWUP_PATTERNS.search(tail))

    async def _stream_to_telegram(
        self, message: Message, user_id: str, text: str, history: list
    ) -> str:
        """
        Two-phase streaming:
          Phase 1,  Run the full agent pipeline (``process_message``) with a
                    live ``progress_callback`` that edits the placeholder to
                    show tool steps ("🔍 Searching…", "📄 Reading…" etc.).
          Phase 2,  Re-stream the final response token-by-token via
                    ``llm.astream`` so the text appears progressively in chat.

        Falls back to a single send if streaming is unavailable.
        """
        placeholder = await message.answer(
            "💭 _thinking…_", parse_mode=ParseMode.MARKDOWN
        )
        _last_progress = {"text": ""}

        logger.info(f"[Telegram] → agent (stream): {text[:80]}")

        # ── Phase 1: agent pipeline with live progress ──────────────────
        async def _tg_progress(status: str):
            try:
                new_text = f"⏳ {status}"
                if new_text != _last_progress["text"]:
                    _last_progress["text"] = new_text
                    await placeholder.edit_text(new_text, parse_mode=None)
            except Exception:
                pass

        try:
            response = await self.agent.process_message(
                user_id, text, history=history, progress_callback=_tg_progress
            )
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            try:
                await placeholder.edit_text(
                    f"❌ Something went wrong: {e}", parse_mode=None
                )
            except Exception:
                pass
            return ""

        if not response or not response.strip():
            try:
                await placeholder.delete()
            except Exception:
                pass
            return ""

        # ── Phase 2: stream the final text token-by-token ───────────────
        has_astream = (
            hasattr(self.agent, "llm")
            and hasattr(self.agent.llm, "astream")
        )

        if has_astream:
            try:
                full_text = ""
                last_edit = asyncio.get_event_loop().time()
                buffer = ""

                # Show first bit of response immediately
                try:
                    await placeholder.edit_text("▌", parse_mode=None)
                except Exception:
                    pass

                system_prompt = "You are a helpful assistant."
                if hasattr(self.agent, "_get_personality_prompt"):
                    system_prompt = self.agent._get_personality_prompt()

                async for token in self.agent.llm.astream([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": (
                        "Repeat the following text EXACTLY as written, "
                        "character for character. Do NOT add or remove anything:\n\n"
                        + response
                    )},
                ]):
                    full_text += token
                    buffer += token
                    now = asyncio.get_event_loop().time()

                    if (
                        len(buffer) >= self._STREAM_CHUNK
                        or (now - last_edit) >= self._STREAM_INTERVAL
                    ):
                        if full_text.strip():
                            try:
                                await placeholder.edit_text(
                                    full_text + " ▌", parse_mode=None
                                )
                            except Exception:
                                pass
                            buffer = ""
                            last_edit = now

                # Final edit,  use the REAL agent response (not the astream
                # replay which might drift), with Markdown formatting
                await self._safe_edit(placeholder, response)
                return response.strip()

            except Exception as e:
                logger.debug(f"astream replay failed ({e}), sending final directly")
                # Fall through to direct send below

        # ── No streaming available,  just show the final response ───────
        try:
            await placeholder.delete()
        except Exception:
            pass
        await self._safe_reply(message, response)
        return response.strip()

    # ------------------------------------------------------------------
    # Safe send helpers
    # ------------------------------------------------------------------

    def _build_inline_keyboard(self, buttons: list) -> InlineKeyboardMarkup:
        """
        Build inline keyboard from button data.

        Format: [[{"text": "Button 1", "callback_data": "cmd:action1"}], ...]
        Each inner list is a row of buttons.
        """
        keyboard = []
        for row in buttons:
            button_row = []
            for btn in row:
                if isinstance(btn, dict) and "text" in btn and "callback_data" in btn:
                    # Security: limit callback_data to 64 chars (Telegram limit)
                    callback_data = btn["callback_data"][:64]
                    button_row.append(
                        InlineKeyboardButton(text=btn["text"], callback_data=callback_data)
                    )
            if button_row:
                keyboard.append(button_row)

        return InlineKeyboardMarkup(inline_keyboard=keyboard)

    async def _safe_reply(self, message: Message, text: str, buttons: list = None):
        """Send reply,  try Markdown, fall back to plain text. Optionally add inline buttons."""
        reply_markup = self._build_inline_keyboard(buttons) if buttons else None

        try:
            await message.answer(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        except Exception:
            try:
                await message.answer(
                    _escape_md2(text), parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup
                )
            except Exception:
                await message.answer(text, reply_markup=reply_markup)

    async def _safe_edit(self, msg: Message, text: str):
        """Edit existing message,  try Markdown, fall back to plain text."""
        try:
            await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            try:
                await msg.edit_text(text)
            except Exception:
                pass
