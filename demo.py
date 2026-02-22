#!/usr/bin/env python3
"""
Quick demo/test script for Open-Sable
Run this to test the agent without setting up chat interfaces
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from opensable.core.agent import SableAgent
from opensable.core.config import load_config
from rich.console import Console
from rich.panel import Panel

console = Console()


async def demo():
    """Run a quick demo of Open-Sable"""

    console.print(
        Panel.fit(
            "[bold cyan]Open-Sable Demo[/bold cyan]\n" "Testing the agent without chat interfaces",
            border_style="cyan",
        )
    )

    # Load config
    console.print("\n[yellow]Loading configuration...[/yellow]")
    config = load_config()

    # Initialize agent
    console.print("[yellow]Initializing agent...[/yellow]")
    agent = SableAgent(config)

    try:
        await agent.initialize()
        console.print("[green]✅ Agent initialized successfully![/green]\n")
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize agent: {e}[/red]")
        console.print("\n[yellow]Make sure Ollama is running: ollama serve[/yellow]")
        return

    # Test messages
    test_messages = [
        "Hello! What can you do?",
        "Check my emails",
        "What's on my calendar today?",
        "Search for flights to Paris",
    ]

    console.print(Panel.fit("[bold]Demo Conversations[/bold]", border_style="blue"))

    for i, message in enumerate(test_messages, 1):
        console.print(f"\n[bold cyan]Message {i}:[/bold cyan] {message}")
        console.print("[dim]Processing...[/dim]")

        try:
            response = await agent.process_message("demo_user", message)
            console.print(f"[bold green]Sable:[/bold green] {response}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        if i < len(test_messages):
            await asyncio.sleep(1)

    # Show memory
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]Testing Memory System[/bold yellow]")

    memories = await agent.memory.recall("demo_user", "emails")
    console.print(f"[dim]Found {len(memories)} relevant memories[/dim]")

    # Shutdown
    console.print("\n[yellow]Shutting down agent...[/yellow]")
    await agent.shutdown()

    console.print(
        Panel.fit(
            "[bold green]✅ Demo Complete![/bold green]\n\n"
            "Next steps:\n"
            "1. Configure .env with your bot tokens\n"
            "2. Run: python main.py\n"
            "3. Chat with Sable on Telegram/Discord",
            border_style="green",
        )
    )


async def quick_test():
    """Quick smoke test"""
    console.print("[bold]Running quick smoke test...[/bold]\n")

    from opensable.core.config import OpenSableConfig
    from opensable.core.memory import MemoryManager

    # Test config
    console.print("Testing config... ", end="")
    config = OpenSableConfig()
    assert config.agent_name == "Sable"
    console.print("[green]✓[/green]")

    # Test memory
    console.print("Testing memory... ", end="")
    memory = MemoryManager(config)
    await memory.initialize()
    await memory.store("test", "test message", {})
    await memory.close()
    console.print("[green]✓[/green]")

    console.print("\n[green]✅ All tests passed![/green]\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Open-Sable Demo & Test")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")

    args = parser.parse_args()

    if args.quick:
        asyncio.run(quick_test())
    else:
        asyncio.run(demo())
