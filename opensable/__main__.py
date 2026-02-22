"""
Open-Sable - Autonomous AI Agent Framework
Entry point for running via: python -m opensable
"""

import asyncio
import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

console = Console(width=120)


def setup_logging(log_level: str = "INFO"):
    """Configure logging with rich output"""
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


async def async_main():
    """Main async entry point"""
    # Kill any existing bot instances first (only 1 agent per PC)
    import subprocess
    import os

    # Get current process PID to exclude it
    current_pid = os.getpid()

    # Check and kill OLD opensable processes (not this one)
    try:
        # Get all opensable process PIDs
        result = subprocess.run(
            ["pgrep", "-f", "python -m opensable"], capture_output=True, text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid and int(pid) != current_pid:
                    subprocess.run(
                        ["kill", "-9", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
        await asyncio.sleep(1)
    except:
        pass

    console.print("[bold cyan]🚀 Starting Open-Sable...[/bold cyan]")

    logger = logging.getLogger("opensable")  # defined before try so except can use it

    try:
        # Import here to allow for proper package structure
        from opensable.core.config import load_config
        from opensable.core.agent import SableAgent

        # Load configuration
        config = load_config()
        setup_logging(config.log_level)

        logger.info("Configuration loaded successfully")

        # Initialize core agent
        agent = SableAgent(config)
        await agent.initialize()
        logger.info("Core agent initialized")

        # ── Internal Gateway (Unix socket, zero TCP ports) ────────────────────
        gateway = None
        local_node = None
        if getattr(config, "gateway_enabled", True):
            try:
                from opensable.core.gateway import Gateway

                gateway = Gateway(agent, config)
                await gateway.start()
                logger.info("Internal Gateway started on /tmp/sable.sock")
                webchat_port = int(getattr(config, "webchat_port", 8789))
                console.print("[bold green]🔌 Gateway running[/bold green]")
                console.print(
                    f"[bold cyan]🌐 WebChat → http://127.0.0.1:{webchat_port}[/bold cyan]"
                )
                console.print(
                    f"[dim]Remote: ssh -L {webchat_port}:127.0.0.1:{webchat_port} user@host[/dim]"
                )

                # Start the built-in local node (system.run, fs.*, etc.)
                if getattr(config, "local_node_enabled", True):
                    from opensable.core.nodes import LocalNode

                    local_node = LocalNode(config)
                    await local_node.start()
                    logger.info("Local node started (system.run, fs.*, system.info)")
            except Exception as e:
                logger.warning(f"Gateway failed to start: {e}")

        # ── Mobile Relay (optional — needs Tailscale or Tor for remote access) ─
        mobile_relay = None
        if getattr(config, "mobile_relay_enabled", False):
            try:
                from opensable.core.mobile_relay import MobileRelay

                mobile_relay = MobileRelay(config)
                await mobile_relay.start()
                logger.info(
                    f"Mobile relay started on "
                    f"{getattr(config, 'mobile_relay_host', '127.0.0.1')}:"
                    f"{getattr(config, 'mobile_relay_port', 7891)}"
                )
            except Exception as e:
                logger.warning(f"Mobile relay failed to start: {e}")

        # Check if autonomous mode is enabled
        autonomous_enabled = getattr(config, "autonomous_mode", False)

        if autonomous_enabled:
            console.print("[bold yellow]🤖 AUTONOMOUS MODE ENABLED[/bold yellow]")
            console.print("[dim]Agent will run continuously and take actions independently[/dim]")

            from opensable.core.autonomous_mode import AutonomousMode

            autonomous = AutonomousMode(agent, config)

            # Start autonomous operation
            await autonomous.start()
            return

        # Start interfaces
        interfaces = []

        # CLI (terminal chat)
        if getattr(config, "cli_enabled", False):
            from opensable.interfaces.cli_interface import CLIInterface

            cli = CLIInterface(agent, config)
            logger.info("CLI interface enabled")
            # CLI runs solo - don't start other interfaces
            console.print("[bold green]✅ Starting CLI mode[/bold green]")
            await cli.start()
            return

        # Telegram (bot and/or userbot)
        if config.telegram_bot_token or getattr(config, "telegram_userbot_enabled", False):
            from opensable.interfaces.telegram_userbot import HybridTelegramInterface

            telegram = HybridTelegramInterface(agent, config)
            interfaces.append(telegram)
            if config.telegram_bot_token:
                logger.info("Telegram bot enabled")
            if getattr(config, "telegram_userbot_enabled", False):
                logger.info("Telegram userbot enabled")

        # Discord
        if getattr(config, "discord_bot_token", None):
            from opensable.interfaces.discord_bot import DiscordInterface

            discord = DiscordInterface(agent, config)
            interfaces.append(discord)
            logger.info("Discord interface enabled")

        # WhatsApp
        if getattr(config, "whatsapp_enabled", False):
            from opensable.interfaces.whatsapp_bot import WhatsAppBot

            whatsapp = WhatsAppBot(config, agent)
            interfaces.append(whatsapp)
            logger.info("WhatsApp interface enabled")

        # Slack
        if getattr(config, "slack_bot_token", None) and getattr(config, "slack_app_token", None):
            from opensable.interfaces.slack_bot import SlackInterface

            slack = SlackInterface(agent, config)
            interfaces.append(slack)
            logger.info("Slack interface enabled")

        if not interfaces and not gateway:
            logger.warning("No chat interfaces configured. Check your .env file.")
            console.print("[yellow]⚠️  No chat interfaces or gateway running.[/yellow]")
            console.print("[dim]Available: Telegram, Discord, WhatsApp, Slack[/dim]")
            console.print("[dim]To enable CLI mode, set CLI_ENABLED=true in .env[/dim]")
            return

        if not interfaces and gateway:
            console.print("[bold green]✅ Open-Sable is running with WebChat only[/bold green]")
            console.print(
                f"[dim]Agent: {config.agent_name} | Personality: {config.agent_personality}[/dim]"
            )
            console.print("[dim]Add tokens to .env for Telegram, Discord, WhatsApp, Slack[/dim]")
            console.print("[dim]Type Ctrl+C to stop[/dim]")
            try:
                await asyncio.Event().wait()
            finally:
                if gateway:
                    await gateway.stop()
            return

        console.print(
            f"[bold green]✅ Open-Sable is running with {len(interfaces)} interface(s)[/bold green]"
        )
        console.print(
            f"[dim]Agent: {config.agent_name} | Personality: {config.agent_personality}[/dim]"
        )
        console.print("[dim]Type Ctrl+C to stop[/dim]")

        try:
            await asyncio.gather(*[interface.start() for interface in interfaces])
        finally:
            if mobile_relay:
                await mobile_relay.stop()
            if local_node:
                await local_node.stop()
            if gateway:
                await gateway.stop()

    except KeyboardInterrupt:
        console.print("\n[bold yellow]⏸️  Shutting down gracefully...[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        logger.exception("Fatal error")
        sys.exit(1)


def main():
    """Synchronous entry point"""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        console.print("\n[bold cyan]👋 Goodbye![/bold cyan]")
        sys.exit(0)


if __name__ == "__main__":
    main()
