"""
Open-Sable - Autonomous AI Agent Framework
Entry point for running via: python -m opensable
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# ── Patch the `overrides` library BEFORE chromadb is imported ────────────────
# chromadb 0.5.x uses @override from the `overrides` package, which performs
# strict signature validation at class-definition time. The Posthog telemetry
# subclass has a signature mismatch that crashes on import.
# Fix: disable signature checking in overrides.
try:
    import overrides.signature as _ovr_sig
    _ovr_sig.ensure_all_kwargs_defined_in_sub = lambda *a, **kw: None
    _ovr_sig.ensure_all_positional_args_defined_in_sub = lambda *a, **kw: None
except Exception:
    pass
# Also disable telemetry at the Settings level so no posthog calls are made
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
# ─────────────────────────────────────────────────────────────────────────────

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


def _launch_sable_dev():
    """Start the Sable Dev Next.js server (port 5700) as a background process."""
    import subprocess
    import threading

    def _start():
        try:
            base = Path(__file__).resolve().parent.parent
            sable_dev_dir = base / "sable_dev"
            if not (sable_dev_dir / "package.json").exists():
                return  # not installed, skip silently
            next_build = sable_dev_dir / ".next"
            if not next_build.exists():
                return  # not built yet, skip silently
            _log = logging.getLogger("opensable")

            # Log stderr to a file for diagnostics
            log_dir = Path(os.environ.get("APPDATA", base)) / "opensable" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(log_dir / "sable_dev.log", "w", encoding="utf-8", errors="replace")

            # Prefer calling next directly (more reliable arg passing on Windows)
            if sys.platform == "win32":
                next_bin = sable_dev_dir / "node_modules" / ".bin" / "next.cmd"
                cmd = [str(next_bin), "start", "-p", "5700", "-H", "0.0.0.0"] if next_bin.exists() else [
                    "npm.cmd", "start", "--", "-p", "5700", "-H", "0.0.0.0"
                ]
            else:
                next_bin = sable_dev_dir / "node_modules" / ".bin" / "next"
                cmd = [str(next_bin), "start", "-p", "5700", "-H", "0.0.0.0"] if next_bin.exists() else [
                    "npm", "start", "--", "-p", "5700", "-H", "0.0.0.0"
                ]

            subprocess.Popen(
                cmd,
                cwd=str(sable_dev_dir),
                stdout=log_file,
                stderr=log_file,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            _log.info(f"🛠️  Sable Dev started on http://localhost:5700 (log: {log_dir / 'sable_dev.log'})")
        except Exception as e:
            logging.getLogger("opensable").warning(f"Sable Dev launch failed: {e}")

    threading.Thread(target=_start, daemon=True).start()


def _launch_desktop():
    """Launch Electron desktop app if available, otherwise open browser."""
    import subprocess
    import threading

    # Skip if already running inside our Electron .app (profile.env may override DESKTOP_ENABLED)
    if os.environ.get("_SABLE_ELECTRON_APP") == "1":
        return
    if os.environ.get("DESKTOP_ENABLED", "").lower() == "false":
        return

    # Skip if start.sh already launched the desktop (pidfile exists and process is alive)
    base = Path(__file__).resolve().parent.parent
    desktop_pidfile = base / ".desktop.pid"
    if desktop_pidfile.exists():
        try:
            existing_pid = int(desktop_pidfile.read_text().strip())
            import signal
            os.kill(existing_pid, 0)  # raises if not alive
            return  # already running, don't spawn another
        except (ValueError, OSError):
            pass  # pidfile stale, fall through and launch

    _log = logging.getLogger("opensable")

    def _open():
        try:
            base = Path(__file__).resolve().parent.parent
            desktop_dir = base / "desktop"
            # On Windows, .bin/electron is a Unix symlink — use .cmd or the .exe instead
            if sys.platform == "win32":
                electron_candidates = [
                    desktop_dir / "node_modules" / ".bin" / "electron.cmd",
                    desktop_dir / "node_modules" / "electron" / "dist" / "electron.exe",
                ]
                electron = next((p for p in electron_candidates if p.exists()), None)
            else:
                _e = desktop_dir / "node_modules" / ".bin" / "electron"
                electron = _e if _e.exists() else None
            if electron:
                subprocess.Popen(
                    [str(electron), str(desktop_dir)],
                    cwd=str(base),
                    env={**os.environ, "WEBCHAT_PORT": "8789", "WEBCHAT_HOST": "localhost"},
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                _log.info("🖥️  Desktop app launched")
            else:
                import webbrowser
                webbrowser.open("http://127.0.0.1:8789")
                _log.info("🌐 Opened browser to gateway")
        except Exception as e:
            _log.warning(f"Desktop launch failed: {e}")

    threading.Thread(target=_open, daemon=True).start()


async def async_main():
    """Main async entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Open-Sable Autonomous AI Agent")
    parser.add_argument(
        "--profile", "-p",
        default=None,
        help="Agent profile name (from agents/ directory). Each profile has its own soul, config, tools, and data.",
    )
    args, _unknown = parser.parse_known_args()

    # ── Load agent profile (before anything else) ─────────────────────────
    from opensable.core.profile import load_profile, DEFAULT_PROFILE
    profile_name = args.profile or os.environ.get("SABLE_PROFILE") or DEFAULT_PROFILE
    profile = load_profile(profile_name)
    profile.apply_env()  # merge profile.env into os.environ before load_config()

    # ── Profile isolation for Zunvra Intel/Social (prevent env bleed from root .env) ──
    # Only profile.env should opt-in these capabilities per agent.
    bool_isolated_keys = (
        "ZUNVRA_INTEL_ENABLED",
        "ZUNVRA_SOCIAL_ENABLED",
        "ZUNVRA_ENABLED",
    )
    sensitive_isolated_keys = (
        "ZUNVRA_API_KEY",
        "ZUNVRA_BACKEND_URL",
        "ZUNVRA_INTEL_DATA_DIR",
        "ZUNVRA_INTEL_INTERVAL",
    )

    for key in bool_isolated_keys:
        if key not in profile.env_overrides:
            os.environ[key] = "false"

    for key in sensitive_isolated_keys:
        if key not in profile.env_overrides:
            os.environ.pop(key, None)

    console.print(f"[bold magenta]👤 Profile: {profile_name}[/bold magenta]")

    # Kill any existing bot instances of the SAME profile (not other profiles)
    import subprocess

    # Get current process PID to exclude it
    current_pid = os.getpid()

    # Kill old instance of the SAME profile only (never touch other profiles)
    _pgrep_pattern = f"--profile {profile_name}"
    try:
        result = subprocess.run(
            ["pgrep", "-f", _pgrep_pattern], capture_output=True, text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid and int(pid) != current_pid:
                    subprocess.run(
                        ["kill", "-9", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
        await asyncio.sleep(2)
    except Exception:
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
                from opensable.core.gateway import Gateway, SOCKET_PATH as _gw_sock

                gateway = Gateway(agent, config)
                await gateway.start()
                logger.info(f"Internal Gateway started on {_gw_sock}")
                webchat_port = gateway._webchat_port  # may differ from config if auto-assigned
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

        # ── Auto-start child agents ({profile_name}-*) ───────────────────────
        _child_procs: list = []
        try:
            from opensable.core.agent_manager import AgentManager
            _agent_mgr = AgentManager(profile_name, gateway)
            agent._agent_manager = _agent_mgr   # expose to tools
            if gateway:
                gateway._agent_manager = _agent_mgr
            # Children are not auto-started; use the dashboard to start them.
        except Exception as e:
            logger.warning(f"Agent manager init failed: {e}")

        # ── Pixel-Bridge (Pixel Agents VS Code extension) ─────────────────────
        _bridge_proc = None
        if getattr(config, "pixel_bridge_enabled", False):
            try:
                import uuid as _uuid
                _bridge_script = Path(__file__).resolve().parent.parent / "scripts" / "pixel-bridge.py"
                if _bridge_script.exists():
                    _sid = f"sable-{_uuid.uuid4().hex[:8]}"
                    _ws_url = f"ws://127.0.0.1:{getattr(config, 'webchat_port', 8789)}"
                    _ws_token = getattr(config, 'webchat_token', '') or ''
                    _bridge_args = [
                        sys.executable, str(_bridge_script),
                        "--session-id", _sid,
                        "--gateway-url", _ws_url,
                    ]
                    _bridge_env = {**os.environ}
                    if _ws_token:
                        _bridge_env["WEBCHAT_TOKEN"] = _ws_token
                    _bridge_proc = await asyncio.create_subprocess_exec(
                        *_bridge_args,
                        env=_bridge_env,
                    )
                    logger.info(f"Pixel-Bridge started (session: {_sid}, pid: {_bridge_proc.pid})")
                    console.print(f"[bold magenta]🎮 Pixel-Bridge running (session: {_sid})[/bold magenta]")
                else:
                    logger.warning(f"Pixel-Bridge script not found at {_bridge_script}")
            except Exception as e:
                logger.warning(f"Pixel-Bridge failed to start: {e}")

        # ── Mobile Relay (optional,  needs Tailscale or Tor for remote access) ─
        mobile_relay = None
        if getattr(config, "mobile_relay_enabled", False):
            try:
                from opensable.interfaces.mobile_relay import MobileRelay

                relay_host = getattr(config, "mobile_relay_host", "0.0.0.0")
                relay_port = getattr(config, "mobile_relay_port", 4810)
                mobile_relay = MobileRelay(config, agent)
                await mobile_relay.start(host=relay_host, port=relay_port)
                logger.info(
                    f"Mobile relay started on "
                    f"{relay_host}:{relay_port}"
                )
            except Exception as e:
                logger.warning(f"Mobile relay failed to start: {e}")

        # ── Autonomous Mode (runs as background task alongside interfaces) ────
        autonomous = None
        autonomous_enabled = getattr(config, "autonomous_mode", False)

        if autonomous_enabled:
            console.print("[bold yellow]🤖 AUTONOMOUS MODE ENABLED[/bold yellow]")
            console.print("[dim]Agent will run the autonomous tick loop in the background[/dim]")

            from opensable.core.autonomous_mode import AutonomousMode

            autonomous = AutonomousMode(agent, config)

            async def _run_autonomous():
                try:
                    await autonomous.start()
                except Exception as exc:
                    logger.error(f"🤖 Autonomous mode crashed: {exc}", exc_info=True)

            asyncio.create_task(_run_autonomous())
            logger.info("🤖 Autonomous tick loop running in background")

        # ── X Autoposter (runs alongside any interface mode) ──────────────────
        x_autoposter = None
        if getattr(config, "x_autoposter_enabled", False) and getattr(
            config, "x_enabled", False
        ):
            try:
                from opensable.core.x_autoposter import XAutoposter

                x_autoposter = XAutoposter(agent, config)
                agent.x_autoposter = x_autoposter  # expose to gateway for mood reporting

                async def _run_autoposter():
                    try:
                        await x_autoposter.start()
                    except Exception as exc:
                        logger.error(f"🐦 X Autoposter crashed: {exc}", exc_info=True)

                asyncio.create_task(_run_autoposter())
                logger.info("🐦 X Autoposter task created")
                console.print("[bold blue]🐦 X Autoposter running in background[/bold blue]")
            except Exception as e:
                logger.warning(f"X Autoposter failed to start: {e}", exc_info=True)

        # ── Zunvra Intelligence Loop (26-module analysis + camera pilot) ───────
        intel_loop = None
        zunvra_intel_enabled = (
            os.environ.get("ZUNVRA_INTEL_ENABLED", "false").lower() in ("true", "1", "yes")
            and os.environ.get("ZUNVRA_BACKEND_URL", "")
        )
        if zunvra_intel_enabled:
            try:
                from opensable.core.intel_loop import IntelLoop

                intel_loop = IntelLoop(agent, config)

                async def _run_intel_loop():
                    try:
                        await intel_loop.start()
                    except Exception as exc:
                        logger.error(f"🛰️  Intel Loop crashed: {exc}", exc_info=True)

                asyncio.create_task(_run_intel_loop())
                interval = intel_loop.interval
                logger.info(f"🛰️  Intel Loop task created (every {interval:.0f}s)")
                console.print(
                    f"[bold cyan]🛰️  Intelligence Loop running in background "
                    f"(26 modules + camera pilot, every {interval:.0f}s)[/bold cyan]"
                )
            except Exception as e:
                logger.warning(f"Intel Loop failed to start: {e}", exc_info=True)

        # ── Zunvra Social Loop (autonomous posting, liking, replying) ──────────
        social_loop = None
        zunvra_social_enabled = (
            os.environ.get("ZUNVRA_SOCIAL_ENABLED", "false").lower() in ("true", "1", "yes")
        )
        if zunvra_social_enabled:
            try:
                from opensable.core.social_loop import SocialLoop

                social_loop = SocialLoop(agent, config)

                async def _run_social_loop():
                    try:
                        await social_loop.start()
                    except Exception as exc:
                        logger.error(f"🌐 Social Loop crashed: {exc}", exc_info=True)

                asyncio.create_task(_run_social_loop())
                interval_s = social_loop.interval
                logger.info(f"🌐 Social Loop task created (every {interval_s:.0f}s)")
                console.print(
                    f"[bold green]🌐 Zunvra Social Loop running in background "
                    f"(autonomous posts/replies/likes, every {interval_s:.0f}s)[/bold green]"
                )
            except Exception as e:
                logger.warning(f"Social Loop failed to start: {e}", exc_info=True)

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
            try:
                from opensable.interfaces.telegram_userbot import HybridTelegramInterface

                telegram = HybridTelegramInterface(agent, config)
                interfaces.append(telegram)
                if config.telegram_bot_token:
                    logger.info("Telegram bot enabled")
                if getattr(config, "telegram_userbot_enabled", False):
                    logger.info("Telegram userbot enabled")
            except ImportError as e:
                logger.warning(f"Telegram interface unavailable: {e}")

        # Discord
        if getattr(config, "discord_bot_token", None):
            try:
                from opensable.interfaces.discord_bot import DiscordInterface

                discord = DiscordInterface(agent, config)
                interfaces.append(discord)
                logger.info("Discord interface enabled")
            except ImportError as e:
                logger.warning(f"Discord interface unavailable: {e}")

        # WhatsApp
        if getattr(config, "whatsapp_enabled", False):
            try:
                from opensable.interfaces.whatsapp_bot import WhatsAppBot

                whatsapp = WhatsAppBot(config, agent)
                interfaces.append(whatsapp)
                logger.info("WhatsApp interface enabled")
            except ImportError as e:
                logger.warning(f"WhatsApp interface unavailable: {e}")

        # Slack
        if getattr(config, "slack_bot_token", None) and getattr(config, "slack_app_token", None):
            try:
                from opensable.interfaces.slack_bot import SlackInterface

                slack = SlackInterface(agent, config)
                interfaces.append(slack)
                logger.info("Slack interface enabled")
            except ImportError as e:
                logger.warning(f"Slack interface unavailable: {e}")

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
            _launch_sable_dev()
            _launch_desktop()
            try:
                await asyncio.Event().wait()
            finally:
                # Stop child agents first
                if _child_procs and '_agent_mgr' in dir():
                    await _agent_mgr.stop_all_children()
                if _bridge_proc and _bridge_proc.returncode is None:
                    _bridge_proc.terminate()
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
        _launch_sable_dev()
        _launch_desktop()

        try:
            await asyncio.gather(
                *[interface.start() for interface in interfaces],
                asyncio.Event().wait(),  # keep gateway alive even if interfaces exit
            )
        finally:
            # Stop child agents first
            if _child_procs and '_agent_mgr' in dir():
                await _agent_mgr.stop_all_children()
            if _bridge_proc and _bridge_proc.returncode is None:
                _bridge_proc.terminate()
            if autonomous:
                await autonomous.stop()
            if intel_loop:
                await intel_loop.stop()
            if social_loop:
                await social_loop.stop()
            if x_autoposter:
                await x_autoposter.stop()
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
