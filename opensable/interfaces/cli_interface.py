"""
CLI Interface for Open-Sable
Interactive terminal REPL with streaming progress indicators
"""

import logging
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.table import Table

logger = logging.getLogger(__name__)


class CLIInterface:
    """Interactive CLI with real-time progress and streaming output."""

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.console = Console()
        self.user_id = "cli_user"
        self.history = []
        self._live = None
        self._status_lines: list[str] = []

    async def _progress_callback(self, message: str):
        """Display progress updates inline using Rich Live."""
        self._status_lines.append(message)
        if self._live:
            renderable = Text()
            for line in self._status_lines[-6:]:
                renderable.append(f"  {line}\n", style="dim cyan")
            self._live.update(renderable)

    async def start(self):
        """Start the interactive REPL."""
        model_name = (
            self.agent.llm.current_model if hasattr(self.agent.llm, "current_model") else "unknown"
        )
        self.console.print(
            Panel.fit(
                "[bold cyan]Open-Sable CLI[/bold cyan]\n"
                f"Agent: {self.config.agent_name}  •  Model: {model_name}\n"
                "Type [bold]/help[/bold] for commands, [bold]exit[/bold] to quit",
                title="🤖 Welcome",
                border_style="cyan",
            )
        )

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
                if not user_input.strip():
                    continue

                if user_input.lower().strip() in ("exit", "quit", "bye", "goodbye"):
                    self.console.print("[bold yellow]👋 Goodbye![/bold yellow]")
                    break

                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # Process with progress
                self._status_lines = []
                with Live(
                    Spinner("dots", text="Thinking..."),
                    console=self.console,
                    refresh_per_second=8,
                    transient=True,
                ) as live:
                    self._live = live
                    try:
                        response = await self.agent.process_message(
                            user_id=self.user_id,
                            message=user_input,
                            history=self.history,
                            progress_callback=self._progress_callback,
                        )
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        self.console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
                        continue
                    finally:
                        self._live = None

                # Update history
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": response})
                if len(self.history) > 40:
                    self.history = self.history[-40:]

                # Display response
                self.console.print(
                    Panel(
                        Markdown(response),
                        title="[bold cyan]🤖 Sable[/bold cyan]",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                )

            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]👋 Goodbye![/bold yellow]")
                break
            except EOFError:
                break

    async def _handle_command(self, command: str):
        """Handle /commands"""
        cmd = command.lower().strip()

        if cmd == "/help":
            self.console.print(
                Panel(
                    "[bold]Commands:[/bold]\n\n"
                    "/help    — Show this help\n"
                    "/clear   — Clear conversation history\n"
                    "/model   — Show current model info\n"
                    "/stats   — Show agent statistics\n"
                    "/tools   — List available tools\n"
                    "/memory  — Show memory stats\n"
                    "/exit    — Exit CLI\n",
                    title="📖 Help",
                    border_style="blue",
                )
            )

        elif cmd == "/clear":
            self.history = []
            self.console.print("[yellow]🗑️  History cleared[/yellow]")

        elif cmd == "/model":
            if hasattr(self.agent.llm, "current_model"):
                model = self.agent.llm.current_model
                available = getattr(self.agent.llm, "available_models", [])
                self.console.print(
                    Panel(
                        f"[bold]Current:[/bold] {model}\n"
                        f"[bold]Available:[/bold] {', '.join(available[:10]) if available else 'Unknown'}",
                        title="🤖 Model",
                        border_style="cyan",
                    )
                )
            else:
                self.console.print("[yellow]Model info not available[/yellow]")

        elif cmd == "/tools":
            if self.agent.tools:
                schemas = self.agent.tools.get_tool_schemas()
                table = Table(title="🔧 Available Tools", border_style="dim")
                table.add_column("Tool", style="cyan")
                table.add_column("Description", style="dim")
                for s in schemas:
                    fn = s.get("function", {})
                    table.add_row(fn.get("name", "?"), fn.get("description", "")[:60])
                self.console.print(table)
            else:
                self.console.print("[yellow]Tools not loaded[/yellow]")

        elif cmd == "/memory":
            info_parts = [f"[bold]History length:[/bold] {len(self.history)} messages"]
            if self.agent.advanced_memory:
                info_parts.append("[bold]Advanced memory:[/bold] ✅ active")
            else:
                info_parts.append("[bold]Advanced memory:[/bold] ❌ not loaded")
            self.console.print(
                Panel("\n".join(info_parts), title="🧠 Memory", border_style="green")
            )

        elif cmd == "/stats":
            self.console.print(
                Panel(
                    f"[bold]Messages:[/bold] {len(self.history)}\n"
                    f"[bold]User ID:[/bold] {self.user_id}\n"
                    f"[bold]Agent:[/bold] {self.config.agent_name}",
                    title="📊 Stats",
                    border_style="green",
                )
            )

        elif cmd in ("/exit", "/quit"):
            raise KeyboardInterrupt

        else:
            self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
            self.console.print("[dim]Type /help for commands[/dim]")
