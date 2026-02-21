"""
Telegram Bot Interface — Sable

Features:
- Persistent conversation history per user (survives restarts)
- Streaming responses — bot edits the message token-by-token via Ollama
- Slash commands: /status /reset /new /compact /think /verbose /voice /model /usage /help
- Pairing/allowlist system:
    * If TELEGRAM_ALLOWED_USERS is empty → first user is auto-authorized (owner)
    * New users receive a pairing code; owner must /pair approve <code>
    * No open ports — pairing is pure in-bot DM exchange
- Group support with activation mode (mention|always)
- Markdown-safe replies (falls back to plain text on parse errors)
- NO external ports opened — only Telegram long-polling
"""

import asyncio
import logging
import os
import re
import secrets
import string
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.enums import ParseMode, ChatType

from opensable.core.session_manager import SessionManager, SessionConfig
from opensable.core.commands import CommandHandler
from opensable.core.heartbeat import HeartbeatManager
from opensable.core.voice_handler import VoiceMessageHandler
from opensable.core.image_analyzer import ImageAnalyzer, handle_telegram_photo

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
      - No ports, no HTTP — all via Telegram DMs
    """

    def __init__(self, config):
        self.config = config
        self._path = Path.home() / ".opensable" / "pairing.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # {user_id: {"username": str, "approved_at": iso}}
        self.allowlist: dict = {}
        # {code: {"user_id": str, "username": str, "expires": iso}}
        self.pending: dict = {}

        self._load()

        # Seed allowlist from env if provided
        for uid in config.telegram_allowed_users:
            if uid and uid not in self.allowlist:
                self.allowlist[uid] = {"username": "env-seeded", "approved_at": datetime.utcnow().isoformat()}

    # ------------------------------------------------------------------

    def is_allowed(self, user_id: str) -> bool:
        return user_id in self.allowlist

    def has_owner(self) -> bool:
        return bool(self.allowlist)

    def approve_first(self, user_id: str, username: str):
        """Auto-approve the very first user as owner."""
        self.allowlist[user_id] = {
            "username": username,
            "approved_at": datetime.utcnow().isoformat(),
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
                if expires > datetime.utcnow():
                    return code
                del self.pending[code]

        code = _generate_pair_code()
        expires = (datetime.utcnow() + timedelta(minutes=30)).isoformat()
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
        if expires < datetime.utcnow():
            self._save()
            return None
        self.allowlist[info["user_id"]] = {
            "username": info["username"],
            "approved_at": datetime.utcnow().isoformat(),
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
        import json
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
    _STREAM_CHUNK = 80      # characters before editing
    _STREAM_INTERVAL = 1.5  # seconds between edits (Telegram rate-limit ~20 edits/min)

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.bot: Optional[Bot] = None
        self.dp: Optional[Dispatcher] = None
        self.session_manager = SessionManager()
        self.command_handler = CommandHandler(self.session_manager)
        self.pairing = PairingStore(config)
        
        # Heartbeat manager for proactive checking
        self.heartbeat = HeartbeatManager(agent, config)
        
        # Voice message handler
        self.voice_handler = VoiceMessageHandler(config, agent)
        
        # Image analyzer
        self.image_analyzer = ImageAnalyzer(config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        if not self.config.telegram_bot_token:
            logger.error("TELEGRAM_BOT_TOKEN not set")
            return

        self.bot = Bot(token=self.config.telegram_bot_token)
        self.dp = Dispatcher()

        # Register handlers
        self.dp.message.register(self._h_start,   CommandStart())
        
        # Register callback query handler for inline buttons
        self.dp.callback_query.register(self._h_callback)
        
        # Start heartbeat for proactive checking
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
                f"You're the first user — auto-approved as **owner**.\n"
                f"Your ID: `{user_id}`\n\n"
                f"Use /help to see available commands.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return True

        if self.pairing.is_allowed(user_id):
            return True

        # Unknown user — issue pairing code
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
            user_id=user_id, channel="telegram",
            config=SessionConfig(model=self.config.default_model),
        )
        name = message.from_user.first_name or "there"
        age_h = (datetime.now(timezone.utc) - session.created_at.replace(tzinfo=timezone.utc)).total_seconds() / 3600
        is_new = age_h < 0.1
        if is_new:
            text = (
                f"👋 **Hello, {name}!**\n\n"
                f"I'm **{self.config.agent_name}** — your personal AI assistant.\n\n"
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
            "/status — session info (model, tokens, uptime)\n"
            "/reset  — clear conversation history\n"
            "/new    — same as /reset\n"
            "/compact — summarise old messages\n"
            "/think `<level>` — off|minimal|low|medium|high|xhigh\n"
            "/verbose `on|off` — toggle detailed output\n"
            "/voice `on|off` — toggle voice mode\n"
            "/model `<name>` — switch AI model\n"
            "/usage `full|tokens|off` — usage footer\n"
            "/pair `approve|deny <code>` — manage pairing\n"
            "/help — this message\n\n"
            "Just chat naturally for everything else!"
        )
        
        # Example inline buttons
        buttons = [
            [
                {"text": "📊 Status", "callback_data": "cmd:/status"},
                {"text": "🔄 Reset", "callback_data": "cmd:/reset"}
            ],
            [
                {"text": "🌐 Search Web", "callback_data": "cmd:search latest AI news"}
            ]
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
                "`/pair approve <CODE>` — approve a pending user\n"
                "`/pair deny <CODE>` — deny a pending user\n"
                "`/pair revoke <USER_ID>` — revoke existing user",
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
                        f"✅ You've been approved! You can now use Sable.",
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
                    user_id=user_id, channel="telegram",
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
            user_id=user_id, channel="telegram",
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
                audio_bytes,
                user_id,
                respond_with_voice=voice_enabled
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
                await message.answer_voice(
                    voice=voice_bytes,
                    caption="🔊 Voice response"
                )
            
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
                message,
                self.image_analyzer,
                self.agent,
                user_id
            )
            
            await status_msg.delete()
            await self._safe_reply(message, response)
            
        except Exception as e:
            logger.error(f"Photo handling error: {e}", exc_info=True)
            await message.answer(f"❌ Image analysis error: {str(e)}")

    async def _h_message(self, message: Message):
        """Main message handler — slash commands + regular chat with streaming."""
        if not message.text:
            return

        user_id = str(message.from_user.id)

        # Auth gate
        if not await self._check_auth(message):
            return

        # Group activation check
        is_group = message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
        if is_group:
            session_tmp = self.session_manager.get_or_create_session(
                user_id=user_id, channel=f"telegram_group_{message.chat.id}",
                config=SessionConfig(model=self.config.default_model),
            )
            activation = session_tmp.metadata.get("activation_mode", "mention")
            if activation == "mention":
                bot_info = await self.bot.get_me()
                bot_username = bot_info.username or ""
                if f"@{bot_username}" not in (message.text or ""):
                    return

        # Session — per user, per channel
        channel_key = f"telegram_group_{message.chat.id}" if is_group else "telegram"
        session = self.session_manager.get_or_create_session(
            user_id=user_id, channel=channel_key,
            config=SessionConfig(model=self.config.default_model),
        )

        text = message.text.strip()

        # ── Slash command handling ──────────────────────────────────────
        if text.startswith("/"):
            is_admin = (user_id == self.pairing.owner_id())
            result = await self.command_handler.handle_command(
                text, session.id, user_id, is_admin=is_admin, is_group=is_group
            )
            if result.message:
                await self._safe_reply(message, result.message)
            if not result.should_continue:
                return

        # ── Regular message — stream response ──────────────────────────
        await message.bot.send_chat_action(message.chat.id, "typing")

        # Add user message to session history
        session.add_message("user", text)
        self.session_manager._save_session(session)

        # Build conversation history to pass to agent
        history = session.get_llm_messages(limit=20)

        try:
            # Always route through the full agentic loop — the agent decides
            # internally whether to call tools (browser, weather, calendar…)
            # or just answer from context. The old split path caused the bot
            # to refuse tool use for messages that didn't match a keyword list.
            logger.info(f"[Telegram] → agent: {text[:80]}")
            response = await self.agent.process_message(user_id, text, history=history)
            await self._safe_reply(message, response)
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            try:
                await message.answer(f"❌ Something went wrong: {e}")
            except Exception:
                pass
            return

        # Persist assistant reply
        if response:
            session.add_message("assistant", response)
            self.session_manager._save_session(session)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    # Keywords that signal the user wants the agent to use a tool (web search, etc.)
    _TOOL_TRIGGERS = (
        "search", "busca", "buscar", "look up", "lookup", "find", "encuentra",
        "google", "bing", "what is", "who is", "quien es", "que es",
        "latest", "current", "news", "noticias", "weather", "clima",
        "scrape", "fetch", "download", "url", "http",
    )

    def _needs_tools(self, text: str) -> bool:
        """Return True when the message likely requires a tool call (e.g. web search)."""
        lower = text.lower()
        return any(kw in lower for kw in self._TOOL_TRIGGERS)

    async def _stream_response(
        self, message: Message, user_id: str, text: str, history: list
    ) -> str:
        """
        Stream Ollama response by editing a placeholder message progressively.
        Falls back to a single final reply if streaming is unavailable.
        """
        # Send typing placeholder
        placeholder = await message.answer("💭 _thinking…_", parse_mode=ParseMode.MARKDOWN)

        full_text = ""
        last_edit = 0.0
        buffer = ""

        try:
            import ollama
            client = ollama.AsyncClient(host=self.config.ollama_base_url)

            # Prepare messages for Ollama
            # Use the same system prompt as the agent graph (with CRITICAL RULES)
            if hasattr(self.agent, "_get_personality_prompt"):
                system_content = self.agent._get_personality_prompt()
            else:
                system_content = "You are Sable, a helpful AI assistant."

            system_content += (
                "\n\nCRITICAL RULES -- follow these exactly, no exceptions:\n"
                "- NEVER invent, fabricate, or hallucinate facts, names, dates, or any data\n"
                "- NEVER simulate search results, tool output, or API responses\n"
                "- If you don't know something, say 'I don't know' -- do NOT guess\n"
                "- Do not make up people, organizations, events, or URLs under any circumstance"
            )

            ollama_messages = [{"role": "system", "content": system_content}]
            ollama_messages += history[:-1]  # history already includes the user msg as last
            ollama_messages.append({"role": "user", "content": text})

            async for chunk in await client.chat(
                model=self.agent.llm.current_model if hasattr(self.agent.llm, "current_model") else "llama3.1:8b",
                messages=ollama_messages,
                stream=True,
            ):
                delta = chunk.get("message", {}).get("content", "")
                full_text += delta
                buffer += delta
                now = asyncio.get_event_loop().time()

                # Edit the message periodically
                if len(buffer) >= self._STREAM_CHUNK or (now - last_edit) >= self._STREAM_INTERVAL:
                    if full_text.strip():
                        try:
                            await placeholder.edit_text(
                                full_text + " ▌",
                                parse_mode=None,
                            )
                        except Exception:
                            pass
                        buffer = ""
                        last_edit = now

            # Final edit — remove cursor
            if full_text.strip():
                await self._safe_edit(placeholder, full_text)
            else:
                await placeholder.delete()
                full_text = await self.agent.process_message(user_id, text, history=history)
                await self._safe_reply(message, full_text)

        except Exception as e:
            logger.warning(f"Stream failed ({e}), using non-streaming agent")
            await placeholder.delete()
            full_text = await self.agent.process_message(user_id, text, history=history)
            await self._safe_reply(message, full_text)

        return full_text

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
                        InlineKeyboardButton(
                            text=btn["text"],
                            callback_data=callback_data
                        )
                    )
            if button_row:
                keyboard.append(button_row)
        
        return InlineKeyboardMarkup(inline_keyboard=keyboard)

    async def _safe_reply(self, message: Message, text: str, buttons: list = None):
        """Send reply — try Markdown, fall back to plain text. Optionally add inline buttons."""
        reply_markup = self._build_inline_keyboard(buttons) if buttons else None
        
        try:
            await message.answer(text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
        except Exception:
            try:
                await message.answer(_escape_md2(text), parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup)
            except Exception:
                await message.answer(text, reply_markup=reply_markup)

    async def _safe_edit(self, msg: Message, text: str):
        """Edit existing message — try Markdown, fall back to plain text."""
        try:
            await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            try:
                await msg.edit_text(text)
            except Exception:
                pass
