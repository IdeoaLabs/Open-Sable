"""
Chat Commands Handler

Slash-command surface for Sable — works across all platforms
(Telegram, Discord, WhatsApp, Slack, WebChat …).

Available commands:
  /status     — compact session status (model, tokens, uptime)
  /new        — reset conversation (alias: /reset)
  /reset      — reset conversation
  /compact    — compress old messages into a summary
  /think      — set thinking level: off|minimal|low|medium|high|xhigh
  /verbose    — toggle verbose mode: on|off
  /usage      — show token/cost usage: off|tokens|full
  /voice      — toggle voice mode: on|off
  /model      — switch AI model
  /help       — show this list
  /restart    — restart the gateway (owner-only)
  /activation — group activation: mention|always  (groups only)

All commands are platform-agnostic — only the interface layer knows
how to send the reply.
"""

import logging
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result returned to the interface layer."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    should_continue: bool = False   # True = also pass message to agent


class CommandHandler:
    """Platform-agnostic slash-command handler."""

    COMMANDS: Dict[str, str] = {
        "status":     "Show session status and statistics",
        "reset":      "Clear conversation history",
        "new":        "Start a new conversation (alias for /reset)",
        "compact":    "Compress old messages into a summary",
        "think":      "Set thinking depth: off|minimal|low|medium|high|xhigh",
        "verbose":    "Toggle verbose output: on|off",
        "usage":      "Usage footer: off|tokens|full",
        "voice":      "Toggle voice mode: on|off",
        "model":      "Switch AI model — /model llama3.1:8b",
        "help":       "Show this command list",
        "restart":    "Restart the gateway (owner only)",
        "activation": "Group activation mode: mention|always  (groups only)",
        "plugin":     "Run a plugin command — /plugin <command> [args]",
        "plugins":    "List loaded plugins and their commands",
    }

    THINKING_LEVELS: List[str] = ["off", "minimal", "low", "medium", "high", "xhigh"]
    USAGE_MODES:    List[str] = ["off", "tokens", "full"]

    def __init__(self, session_manager, gateway=None, plugin_manager=None):
        self.session_manager = session_manager
        self.gateway = gateway
        self.plugin_manager = plugin_manager

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_command(self, text: str) -> bool:
        """Return True if *text* starts with a recognised slash command."""
        if not text or not text.strip().startswith("/"):
            return False
        cmd, _ = self._parse(text)
        return cmd in self.COMMANDS

    def _parse(self, text: str) -> Tuple[str, List[str]]:
        """Split '/cmd arg1 arg2' -> ('cmd', ['arg1', 'arg2'])."""
        parts = text.strip().lstrip("/").split()
        cmd = parts[0].lower() if parts else ""
        return cmd, parts[1:]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def handle_command(
        self,
        text: str,
        session_id: str,
        user_id: str,
        is_admin: bool = False,
        is_group: bool = False,
    ) -> CommandResult:
        if not self.is_command(text):
            return CommandResult(success=False, message="", should_continue=True)

        cmd, args = self._parse(text)

        if cmd not in self.COMMANDS:
            return CommandResult(
                success=False,
                message=f"Unknown command: /{cmd}\nUse /help for available commands.",
            )

        session = self.session_manager.get_session(session_id)
        if not session:
            return CommandResult(
                success=False,
                message="Session not found — send a message first.",
            )

        handler = getattr(self, f"_cmd_{cmd}", None)
        if not handler:
            return CommandResult(
                success=False,
                message=f"/{cmd} is not yet implemented.",
            )

        try:
            return await handler(session, args, user_id, is_admin, is_group)
        except Exception as e:
            logger.error(f"Command /{cmd} error: {e}", exc_info=True)
            return CommandResult(success=False, message=f"Error: {e}")

    # ------------------------------------------------------------------
    # Individual command handlers
    # ------------------------------------------------------------------

    async def _cmd_status(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        now = datetime.now(timezone.utc)

        def _dt(val) -> datetime:
            if isinstance(val, datetime):
                return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
            return datetime.fromisoformat(str(val)).replace(tzinfo=timezone.utc)

        age_h  = (now - _dt(session.created_at)).total_seconds() / 3600
        idle_s = (now - _dt(session.updated_at)).total_seconds()

        lines = [
            "📊 **Session Status**",
            "",
            f"**ID:** `{session.id[:12]}...`",
            f"**Channel:** {session.channel}",
            f"**Model:** {session.config.model or 'default'}",
            f"**Messages:** {len(session.messages)}",
            f"**Tokens used:** {session.total_tokens}",
            f"**Cost:** ${session.total_cost:.4f}",
            f"**Session age:** {age_h:.1f} h",
            f"**Last message:** {idle_s:.0f} s ago",
            "",
            "**Settings:**",
            f"• Thinking: `{session.config.thinking_level}`",
            f"• Verbose: {'on' if session.config.verbose else 'off'}",
            f"• Voice: {'on' if session.config.use_voice else 'off'}",
            f"• Auto-compact: {'on' if session.config.auto_compact else 'off'}",
        ]
        return CommandResult(success=True, message="\n".join(lines), data=session.to_dict())

    async def _cmd_reset(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        count = len(session.messages)
        session.clear_messages()
        self.session_manager._save_session(session)
        return CommandResult(
            success=True,
            message=f"✅ Conversation reset — {count} messages cleared.",
        )

    async def _cmd_new(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        return await self._cmd_reset(session, args, user_id, is_admin, is_group)

    async def _cmd_compact(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        if len(session.messages) < 10:
            return CommandResult(
                success=False,
                message="Session too short to compact (need >= 10 messages).",
            )
        keep = int(args[0]) if args and args[0].isdigit() else 20
        original = len(session.messages)
        session.compact_messages(keep_recent=keep)
        self.session_manager._save_session(session)
        return CommandResult(
            success=True,
            message=f"✅ Compacted {original} -> {len(session.messages)} messages (kept last {keep}).",
        )

    async def _cmd_think(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        if not args:
            return CommandResult(
                success=True,
                message=(
                    f"Current thinking level: **{session.config.thinking_level}**\n\n"
                    f"Available: {' | '.join(self.THINKING_LEVELS)}\n"
                    "Usage: `/think <level>`"
                ),
            )
        level = args[0].lower()
        if level not in self.THINKING_LEVELS:
            return CommandResult(
                success=False,
                message=f"Invalid level: `{level}`\nAvailable: {' | '.join(self.THINKING_LEVELS)}",
            )
        session.config.thinking_level = level
        self.session_manager._save_session(session)
        return CommandResult(success=True, message=f"✅ Thinking level -> **{level}**")

    async def _cmd_verbose(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        if not args:
            state = "on" if session.config.verbose else "off"
            return CommandResult(
                success=True,
                message=f"Verbose is **{state}**. Usage: `/verbose on|off`",
            )
        mode = args[0].lower()
        if mode not in ("on", "off"):
            return CommandResult(success=False, message="Usage: `/verbose on|off`")
        session.config.verbose = mode == "on"
        self.session_manager._save_session(session)
        return CommandResult(success=True, message=f"✅ Verbose -> **{mode}**")

    async def _cmd_usage(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        mode = (args[0].lower() if args else "full")
        if mode not in self.USAGE_MODES:
            return CommandResult(
                success=False,
                message=f"Usage: `/usage {'|'.join(self.USAGE_MODES)}`",
            )
        if mode == "off":
            return CommandResult(success=True, message="✅ Usage footer disabled.")
        lines = [
            "📈 **Usage Statistics**",
            f"**Messages:** {len(session.messages)}",
        ]
        if mode == "full":
            lines += [
                f"**Tokens:** {session.total_tokens}",
                f"**Cost:** ${session.total_cost:.6f}",
                f"**Session started:** {session.created_at.isoformat()[:19]}",
                f"**Last updated:** {session.updated_at.isoformat()[:19]}",
            ]
        return CommandResult(success=True, message="\n".join(lines))

    async def _cmd_voice(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        if not args:
            state = "on" if session.config.use_voice else "off"
            return CommandResult(
                success=True,
                message=f"Voice is **{state}**. Usage: `/voice on|off`",
            )
        mode = args[0].lower()
        if mode not in ("on", "off"):
            return CommandResult(success=False, message="Usage: `/voice on|off`")
        session.config.use_voice = mode == "on"
        self.session_manager._save_session(session)
        return CommandResult(success=True, message=f"✅ Voice -> **{mode}**")

    async def _cmd_model(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        if not args:
            return CommandResult(
                success=True,
                message=(
                    f"Current model: **{session.config.model or 'default'}**\n"
                    "Usage: `/model <model_name>`\n"
                    "Examples: `/model llama3.1:8b` | `/model mistral:7b`"
                ),
            )
        session.config.model = args[0]
        self.session_manager._save_session(session)
        return CommandResult(success=True, message=f"✅ Model -> **{args[0]}**")

    async def _cmd_help(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        lines = ["🤖 **Sable Commands**", ""]
        for cmd, desc in self.COMMANDS.items():
            lines.append(f"**/{cmd}** — {desc}")
        lines += [
            "",
            "💡 **Examples:**",
            "`/think high` — enable deep thinking",
            "`/verbose on` — show detailed output",
            "`/model llama3.1:8b` — switch model",
            "`/compact 15` — keep last 15 messages",
        ]
        return CommandResult(success=True, message="\n".join(lines))

    async def _cmd_restart(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        if not is_admin:
            return CommandResult(success=False, message="❌ Owner permission required.")
        return CommandResult(
            success=True,
            message="⚠️ Restart requested — the gateway will restart shortly.",
        )

    async def _cmd_activation(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        if not is_group:
            return CommandResult(success=False, message="This command only works in groups.")
        if not is_admin:
            return CommandResult(success=False, message="❌ Group admin permission required.")
        if not args or args[0].lower() not in ("mention", "always"):
            current = session.metadata.get("activation_mode", "mention")
            return CommandResult(
                success=True,
                message=(
                    f"Current activation: **{current}**\n"
                    "Usage: `/activation mention|always`\n\n"
                    "• **mention** — respond only when @mentioned\n"
                    "• **always** — respond to every message"
                ),
            )
        mode = args[0].lower()
        session.metadata["activation_mode"] = mode
        self.session_manager._save_session(session)
        return CommandResult(success=True, message=f"✅ Group activation -> **{mode}**")

    # ── plugins ──────────────────────────────────────────────

    async def _cmd_plugins(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        """List loaded plugins and their commands."""
        pm = self.plugin_manager
        if not pm or not pm.plugins:
            return CommandResult(success=True, message="📦 No plugins loaded.")

        lines = ["📦 **Loaded Plugins**", ""]
        for info in pm.list_plugins():
            lines.append(f"**{info['name']}** v{info['version']}")
            if info.get("description"):
                lines.append(f"  _{info['description']}_")
            for cmd in info.get("commands", []):
                lines.append(f"  • `{cmd}`")
            lines.append("")
        return CommandResult(success=True, message="\n".join(lines))

    async def _cmd_plugin(self, session, args, user_id, is_admin, is_group) -> CommandResult:
        """Execute a plugin command: /plugin <command> [args...]"""
        pm = self.plugin_manager
        if not pm:
            return CommandResult(success=False, message="❌ Plugin system not available.")

        if not args:
            return CommandResult(
                success=False,
                message="Usage: `/plugin <command> [args...]`\nUse `/plugins` to see available commands.",
            )

        cmd_name = args[0]
        cmd_args = args[1:]

        try:
            result = await pm.execute_command(cmd_name, *cmd_args)
            return CommandResult(success=True, message=str(result))
        except ValueError:
            return CommandResult(success=False, message=f"❌ Unknown plugin command: `{cmd_name}`\nUse `/plugins` to see available commands.")
        except Exception as e:
            logger.error("Plugin command '%s' failed: %s", cmd_name, e)
            return CommandResult(success=False, message=f"❌ Plugin error: {e}")
