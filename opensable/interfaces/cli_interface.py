"""
CLI Interface for Open-Sable
Interactive terminal chat with the agent
"""

import logging
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

logger = logging.getLogger(__name__)


class CLIInterface:
    """Command-line interface for chatting with the agent"""

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.console = Console()
        self.user_id = "cli_user"
        self.history = []

    async def start(self):
        """Start the CLI interface"""
        self.console.print(
            Panel.fit(
                "[bold cyan]Open-Sable CLI Mode[/bold cyan]\n"
                f"Agent: {self.config.agent_name}\n"
                f"Model: {self.agent.llm.current_model if hasattr(self.agent.llm, 'current_model') else 'unknown'}\n"
                "Type 'exit', 'quit', or Ctrl+C to stop",
                title="🤖 Welcome",
                border_style="cyan",
            )
        )

        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold green]You[/bold green]")

                if not user_input.strip():
                    continue

                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    self.console.print("[bold yellow]👋 Goodbye![/bold yellow]")
                    break

                # Special commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # Process message through agent
                self.console.print("\n[bold cyan]Sable[/bold cyan] (thinking...)", end="")

                try:
                    response = await self.agent.process_message(
                        user_id=self.user_id, message=user_input, history=self.history
                    )

                    # Update history
                    self.history.append({"role": "user", "content": user_input})
                    self.history.append({"role": "assistant", "content": response})

                    # Keep only last 20 messages
                    if len(self.history) > 20:
                        self.history = self.history[-20:]

                    # Display response
                    self.console.print("\r" + " " * 50 + "\r", end="")  # Clear "thinking..."
                    self.console.print(
                        Panel(
                            Markdown(response),
                            title="[bold cyan]🤖 Sable[/bold cyan]",
                            border_style="cyan",
                            padding=(1, 2),
                        )
                    )

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self.console.print(f"\n[bold red]❌ Error: {e}[/bold red]")

            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]👋 Goodbye![/bold yellow]")
                break
            except EOFError:
                break

    async def _handle_command(self, command: str):
        """Handle special commands"""
        cmd = command.lower().strip()

        if cmd == "/help":
            self.console.print(
                Panel(
                    "[bold]Available Commands:[/bold]\n\n"
                    "/help - Show this help\n"
                    "/clear - Clear conversation history\n"
                    "/model - Show current model info\n"
                    "/stats - Show agent statistics\n"
                    "/exit or /quit - Exit CLI\n",
                    title="📖 Help",
                    border_style="blue",
                )
            )

        elif cmd == "/clear":
            self.history = []
            self.console.print("[yellow]🗑️  Conversation history cleared[/yellow]")

        elif cmd == "/model":
            if hasattr(self.agent.llm, "current_model"):
                model = self.agent.llm.current_model
                available = (
                    self.agent.llm.available_models
                    if hasattr(self.agent.llm, "available_models")
                    else []
                )
                self.console.print(
                    Panel(
                        f"[bold]Current Model:[/bold] {model}\n"
                        f"[bold]Available Models:[/bold] {', '.join(available) if available else 'Unknown'}",
                        title="🤖 Model Info",
                        border_style="cyan",
                    )
                )
            else:
                self.console.print("[yellow]Model information not available[/yellow]")

        elif cmd == "/stats":
            self.console.print(
                Panel(
                    f"[bold]Messages in History:[/bold] {len(self.history)}\n"
                    f"[bold]User ID:[/bold] {self.user_id}\n"
                    f"[bold]Agent Name:[/bold] {self.config.agent_name}",
                    title="📊 Statistics",
                    border_style="green",
                )
            )

        elif cmd in ["/exit", "/quit"]:
            raise KeyboardInterrupt

        else:
            self.console.print(f"[yellow]Unknown command: {command}[/yellow]")
            self.console.print("[dim]Type /help for available commands[/dim]")
