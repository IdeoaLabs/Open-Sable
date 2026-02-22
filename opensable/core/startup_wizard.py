"""
Startup Wizard — runs automatically when critical config is missing.

Unlike the full OnboardingWizard (7 interactive steps for first-time setup),
this is a lightweight checker that:
  1. Detects what's missing (Ollama, chat tokens, etc.)
  2. Only asks for what's actually needed
  3. Saves answers to .env
  4. Returns True if startup can proceed
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

ENV_PATH = Path(__file__).parent.parent.parent / ".env"


# ── Checks ─────────────────────────────────────────────────────────


class Issue:
    """A missing or broken config item."""

    __slots__ = ("key", "label", "hint", "critical", "secret")

    def __init__(self, key: str, label: str, hint: str, critical: bool = True, secret: bool = True):
        self.key = key
        self.label = label
        self.hint = hint
        self.critical = critical  # can't start without it
        self.secret = secret  # mask when printing


async def _check_ollama(config) -> Optional[Issue]:
    """Is Ollama reachable?"""
    url = config.ollama_base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{url}/api/tags")
            if r.status_code == 200:
                data = r.json()
                models = data.get("models", [])
                if models:
                    return None  # all good
                return Issue(
                    key="__ollama_no_models__",
                    label="No Ollama models installed",
                    hint="Run:  ollama pull qwen2.5:14b  (or any model you want)",
                    critical=True,
                    secret=False,
                )
    except Exception:
        pass

    return Issue(
        key="OLLAMA_BASE_URL",
        label="Ollama not reachable",
        hint=(
            f"Ollama is not running at {url}\n"
            "  Install:  curl -fsSL https://ollama.ai/install.sh | sh\n"
            "  Start:    ollama serve\n"
            "  Or set OLLAMA_BASE_URL in .env to point at a remote Ollama."
        ),
        critical=True,
        secret=False,
    )


def _check_chat_interfaces(config) -> List[Issue]:
    """At least one chat interface should be configured."""
    has_any = bool(
        config.telegram_bot_token
        or config.discord_bot_token
        or getattr(config, "slack_bot_token", None)
        or getattr(config, "whatsapp_enabled", False)
        or getattr(config, "cli_enabled", False)
    )
    if has_any:
        return []

    return [
        Issue(
            key="TELEGRAM_BOT_TOKEN",
            label="Telegram Bot Token",
            hint=(
                "Get one from @BotFather on Telegram:\n"
                "  1. Open Telegram → search @BotFather\n"
                "  2. Send /newbot, follow the steps\n"
                "  3. Copy the token here"
            ),
            critical=False,
            secret=True,
        ),
    ]


def _check_telegram_users(config) -> Optional[Issue]:
    """If Telegram is configured, allowed_users should be set."""
    if not config.telegram_bot_token:
        return None
    if config.telegram_allowed_users:
        return None
    return Issue(
        key="TELEGRAM_ALLOWED_USERS",
        label="Telegram Allowed Users",
        hint=(
            "Your Telegram user ID (so only you can use the bot).\n"
            "  Find it: open Telegram → search @userinfobot → send /start\n"
            "  Paste the numeric ID (e.g. 123456789)"
        ),
        critical=False,
        secret=False,
    )


# ── Core logic ─────────────────────────────────────────────────────


async def gather_issues(config) -> List[Issue]:
    """Run all checks, return list of issues."""
    issues: List[Issue] = []

    # Ollama
    ollama_issue = await _check_ollama(config)
    if ollama_issue:
        issues.append(ollama_issue)

    # Chat interfaces
    issues.extend(_check_chat_interfaces(config))

    # Telegram users
    tg_issue = _check_telegram_users(config)
    if tg_issue:
        issues.append(tg_issue)

    return issues


def _set_env_value(key: str, value: str):
    """Write or update a key in .env file."""
    env_path = ENV_PATH
    if not env_path.exists():
        env_path.write_text(f"# Open-Sable configuration\n{key}={value}\n")
        return

    content = env_path.read_text()
    # Try to replace existing key
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    if pattern.search(content):
        content = pattern.sub(f"{key}={value}", content)
    else:
        # Append
        if not content.endswith("\n"):
            content += "\n"
        content += f"{key}={value}\n"
    env_path.write_text(content)


# ── Interactive wizard ─────────────────────────────────────────────


async def run_startup_wizard(config) -> bool:
    """
    Check for missing config.  If everything is fine, return True silently.
    If something is missing, show a friendly wizard and ask the user.
    Returns True if startup can proceed, False if critical items are still missing.
    """
    issues = await gather_issues(config)
    if not issues:
        return True  # nothing missing — carry on

    # ── Show what's missing ────────────────────────────────────────
    critical = [i for i in issues if i.critical]
    optional = [i for i in issues if not i.critical]

    console.print()
    console.print(
        Panel(
            "[bold yellow]⚡ Startup Wizard[/bold yellow]\n"
            "[dim]Some configuration is missing. Let's fix it.[/dim]",
            border_style="yellow",
        )
    )
    console.print()

    # Table of issues
    table = Table(show_header=True, header_style="bold")
    table.add_column("Status", width=3)
    table.add_column("Item")
    table.add_column("Required?")
    for i in issues:
        status = "🔴" if i.critical else "🟡"
        req = "Yes" if i.critical else "No"
        table.add_row(status, i.label, req)
    console.print(table)
    console.print()

    # ── Handle critical issues ─────────────────────────────────────
    for issue in critical:
        if issue.key.startswith("__"):
            # Info-only issue (like "no models installed") — just show the hint
            console.print(f"[bold red]❌ {issue.label}[/bold red]")
            console.print(f"[dim]{issue.hint}[/dim]")
            console.print()
            console.print("[yellow]Fix this and restart Open-Sable.[/yellow]")
            return False

        console.print(f"[bold red]❌ {issue.label}[/bold red]")
        console.print(f"[dim]{issue.hint}[/dim]")
        console.print()

        value = Prompt.ask(
            f"[bold]{issue.label}[/bold]",
            password=issue.secret,
            default="",
        )
        if value.strip():
            _set_env_value(issue.key, value.strip())
            os.environ[issue.key] = value.strip()
            console.print(f"[green]✅ Saved {issue.key}[/green]\n")
        else:
            console.print("[red]⏭  Skipped (still required for startup)[/red]\n")
            return False

    # ── Handle optional issues ─────────────────────────────────────
    if optional:
        configure_now = Confirm.ask(
            "[yellow]Configure optional items now?[/yellow]",
            default=True,
        )
        if configure_now:
            for issue in optional:
                console.print(f"\n[bold yellow]🟡 {issue.label}[/bold yellow]")
                console.print(f"[dim]{issue.hint}[/dim]")
                console.print()

                value = Prompt.ask(
                    f"[bold]{issue.label}[/bold]",
                    password=issue.secret,
                    default="",
                )
                if value.strip():
                    _set_env_value(issue.key, value.strip())
                    os.environ[issue.key] = value.strip()
                    console.print(f"[green]✅ Saved {issue.key}[/green]")
                else:
                    console.print("[dim]Skipped — you can add it later in .env[/dim]")
        else:
            console.print("[dim]You can configure them later in .env[/dim]")

    console.print()
    console.print(
        Panel(
            "[bold green]✅ Configuration updated![/bold green]\n"
            "[dim]Changes saved to .env — reloading config...[/dim]",
            border_style="green",
        )
    )
    console.print()

    return True
