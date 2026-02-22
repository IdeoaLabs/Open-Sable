"""
Open-Sable Autonomous Agent Demo
Shows the agent executing commands, modifying files, and running autonomously
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


async def demo_computer_control():
    """Demo: Computer control capabilities (execute commands, modify files)"""
    console.print("\n" + "=" * 60)
    console.print(Panel.fit("🖥️  COMPUTER CONTROL - Full System Access", style="bold cyan"))
    console.print("=" * 60 + "\n")

    from core.computer_tools import ComputerTools

    class FakeConfig:
        pass

    computer = ComputerTools(FakeConfig(), sandbox_mode=False)

    # 1. Execute commands
    console.print("[bold]1. Execute Shell Commands[/bold]")

    commands = ["ls -la /home", "whoami", "uname -a", "df -h", "free -h"]

    for cmd in commands:
        console.print(f"\n  💻 Executing: [cyan]{cmd}[/cyan]")
        result = await computer.execute_command(cmd)

        if result["success"]:
            output = result["stdout"][:500]  # Truncate
            console.print(f"  ✅ Success (exit code: {result['exit_code']})")
            console.print(f"  Output:\n{output}")
        else:
            console.print(f"  ❌ Failed: {result['stderr']}")

    # 2. File operations
    console.print("\n[bold]2. File Operations[/bold]")

    test_file = "/tmp/opensable_test.txt"

    # Write file
    console.print(f"\n  📝 Writing file: {test_file}")
    result = await computer.write_file(
        test_file, "Hello from Open-Sable!\nThis is an autonomous AI agent.\n"
    )
    console.print(f"  ✅ Wrote {result['bytes_written']} bytes")

    # Read file
    console.print(f"\n  📖 Reading file: {test_file}")
    result = await computer.read_file(test_file)
    console.print(f"  Content:\n{result['content']}")

    # Edit file
    console.print("\n  ✏️  Editing file...")
    result = await computer.edit_file(test_file, "autonomous AI agent", "SUPER AUTONOMOUS AI AGENT")
    console.print(f"  ✅ Made {result['replacements']} replacement(s)")

    # Read again
    result = await computer.read_file(test_file)
    console.print(f"  New content:\n{result['content']}")

    # 3. Directory operations
    console.print("\n[bold]3. Directory Operations[/bold]")

    test_dir = "/tmp/opensable_test_dir"

    # Create directory
    console.print(f"\n  📁 Creating directory: {test_dir}")
    result = await computer.create_directory(test_dir)
    console.print(f"  ✅ Created: {result['path']}")

    # List directory
    console.print("\n  📂 Listing /tmp")
    result = await computer.list_directory("/tmp", include_hidden=False)
    files = result["files"][:10]  # Show first 10
    for f in files:
        icon = "📁" if f["type"] == "directory" else "📄"
        console.print(f"  {icon} {f['name']}")

    # Search files
    console.print("\n  🔍 Searching for 'opensable' in /tmp")
    result = await computer.search_files("/tmp", "opensable", content_search=False)
    for match in result["matches"][:5]:
        console.print(f"  • {match['path']}")

    # 4. System info
    console.print("\n[bold]4. System Information[/bold]")
    result = await computer.get_system_info()

    console.print(f"""
  💻 Platform: {result['system']} ({result['platform']})
  🐍 Python: {result['python_version']}
  🔧 CPU: {result['cpu_count']} cores @ {result['cpu_percent']}% usage
  💾 Memory: {result['memory_available'] / (1024**3):.2f} GB available ({result['memory_percent']}% used)
  💿 Disk: {result['disk_usage']['free'] / (1024**3):.2f} GB free ({result['disk_usage']['percent']}% used)
""")

    # Cleanup
    console.print("\n[bold]5. Cleanup[/bold]")
    await computer.delete_file(test_file)
    await computer.delete_file(test_dir)
    console.print("  ✅ Cleaned up test files")


async def demo_autonomous_workflow():
    """Demo: Autonomous workflow"""
    console.print("\n" + "=" * 60)
    console.print(
        Panel.fit("🤖 AUTONOMOUS WORKFLOW - Self-Directed Operation", style="bold yellow")
    )
    console.print("=" * 60 + "\n")

    console.print("""
[bold]What This Agent Can Do Autonomously:[/bold]

1. 📧 Monitor your email and take action
   • "Meeting request from Bob" → Add to calendar
   • "Server alert: disk 90% full" → Run cleanup commands
   • "PR review needed" → Clone repo, review code

2. 📅 Proactive calendar management
   • Meeting in 15 mins → Send reminder
   • Travel event tomorrow → Check traffic, suggest departure time
   • Recurring task → Execute automatically

3. 🖥️  System maintenance
   • Disk space low → Clean temp files, suggest backups
   • High CPU usage → Identify process, suggest action
   • Security update available → Download and notify

4. 💡 Self-improvement
   • Learns from every task execution
   • Synthesizes new tools for recurring patterns
   • Improves strategies over time

5. 🔄 Multi-step workflows
   • "Deploy to production" →
     - Run tests
     - Build Docker image
     - Push to registry
     - Update k8s deployment
     - Verify health checks
     - Send notification
     
[bold cyan]Core Features:[/bold cyan]
✅ Execute shell commands
✅ Read/write/edit files
✅ File system operations
✅ System monitoring
✅ Autonomous goal setting
✅ Meta-learning & self-improvement
✅ Tool synthesis
✅ Memory system (episodic, semantic, working)

[bold yellow]Running Example:[/bold yellow]
""")

    # Simulate autonomous operation
    from core.computer_tools import ComputerTools

    class FakeConfig:
        pass

    computer = ComputerTools(FakeConfig())

    console.print("🤖 Agent: I'll check system health and take action...\n")

    # Step 1: Check system
    console.print("  [1/4] Getting system info...")
    await asyncio.sleep(1)
    result = await computer.get_system_info()

    disk_usage = result["disk_usage"]["percent"]
    console.print(f"  ✅ Disk usage: {disk_usage}%")

    # Step 2: Conditional action
    if disk_usage > 70:
        console.print("  ⚠️  Disk usage high! Taking action...")

        # Execute cleanup
        console.print("  [2/4] Finding large files...")
        await asyncio.sleep(1)
        cmd_result = await computer.execute_command(
            "find /tmp -type f -size +10M 2>/dev/null | head -5"
        )
        console.print(f"  Found: {len(cmd_result['stdout'].splitlines())} large files")

        # Create report
        console.print("  [3/4] Creating cleanup report...")
        await asyncio.sleep(1)
        report = f"""Disk Cleanup Report
Generated: {result}

Disk Usage: {disk_usage}%
Large files found: {cmd_result['stdout']}

Recommendation: Clean /tmp directory
"""
        await computer.write_file("/tmp/cleanup_report.txt", report)
        console.print("  ✅ Report saved to /tmp/cleanup_report.txt")

        # Send notification (simulated)
        console.print("  [4/4] Sending notification...")
        await asyncio.sleep(1)
        console.print("  📬 Notification sent to Telegram")

        console.print("\n✅ Autonomous task completed successfully!")
    else:
        console.print("  ✅ System healthy, no action needed")

    console.print("""
[bold green]Full autonomous agent capabilities![/bold green]

🚀 To enable autonomous mode:
   Set AUTONOMOUS_MODE=true in .env
   
🎯 To use computer control in chat:
   You: "List files in /home"
   Agent: *executes command and shows results*
   
   You: "Create a Python script that..."
   Agent: *writes file with code*
   
   You: "Find all .log files larger than 100MB"
   Agent: *searches filesystem and reports*
""")


async def demo_integration_with_llm():
    """Demo: Integration with LLM for intelligent automation"""
    console.print("\n" + "=" * 60)
    console.print(Panel.fit("🧠 LLM + COMPUTER CONTROL = Autonomous Agent", style="bold magenta"))
    console.print("=" * 60 + "\n")

    console.print("""
[bold]How It Works:[/bold]

1. User request (Telegram/Discord/Voice):
   "Analyze the logs and fix the issue"

2. LLM reasoning (using LangGraph):
   PLAN:
   - Step 1: Find log files → execute_command("find /var/log -name '*.log'")
   - Step 2: Read latest log → read_file("/var/log/app.log")
   - Step 3: Analyze errors → LLM identifies "Connection timeout"
   - Step 4: Check config → read_file("/etc/app/config.json")
   - Step 5: Fix config → edit_file(..., old="timeout: 5", new="timeout: 30")
   - Step 6: Restart service → execute_command("systemctl restart app")
   - Step 7: Verify → execute_command("systemctl status app")
   - Step 8: Report → Send result to user

3. Execution:
   Agent executes each step, adapts if errors occur

4. Result:
   "✅ Fixed timeout issue. Service restarted and running normally."

[bold cyan]Example Workflow Code:[/bold cyan]
""")

    code = '''async def autonomous_debug_workflow(user_request: str):
    """Example: Agent debugging a service issue"""
    
    # Step 1: Understand task
    plan = await llm.plan(user_request)
    # Output: ["find logs", "analyze errors", "fix config", "restart"]
    
    # Step 2: Execute plan
    for step in plan:
        if step == "find logs":
            result = await tools.execute_command("find /var/log -name 'app.log'")
            log_file = result['stdout'].strip()
            
        elif step == "analyze errors":
            content = await tools.read_file(log_file)
            errors = await llm.extract_errors(content['content'])
            
        elif step == "fix config":
            # LLM determines the fix
            fix = await llm.generate_fix(errors)
            await tools.edit_file(
                "/etc/app/config.json",
                old=fix['old_config'],
                new=fix['new_config']
            )
            
        elif step == "restart":
            await tools.execute_command("systemctl restart app")
    
    return "✅ Issue fixed and service restarted"'''

    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

    console.print("""
[bold green]The secret sauce:[/bold green]
LLM brain + Computer hands = Autonomous problem solver

[bold yellow]Open-Sable has all the tools you need:[/bold yellow]
✅ Command execution (subprocess)
✅ File operations (read/write/edit/search)
✅ Directory management
✅ System monitoring
✅ Autonomous goal setting
✅ Self-improvement
✅ Memory system
✅ Multi-step reasoning (LangGraph)

[bold red]What you were missing before:[/bold red]
❌ Just a chatbot waiting for commands
❌ No computer control
❌ No autonomous operation

[bold green]What you have NOW:[/bold green]
✅ Full computer control
✅ Autonomous operation mode
✅ Self-directed task execution
✅ Multi-step workflows
""")


async def main():
    """Run all demos"""
    console.print("""
[bold cyan]
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║       🚀 SABLECORE - Autonomous AI Agent                   ║
║                                                            ║
║     Now with FULL computer control & autonomous mode      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
[/bold cyan]
""")

    # Run demos
    await demo_computer_control()
    await asyncio.sleep(2)

    await demo_autonomous_workflow()
    await asyncio.sleep(2)

    await demo_integration_with_llm()

    console.print("\n" + "=" * 60)
    console.print(
        Panel.fit(
            "✅ Demo Complete - Open-Sable is Ready for Autonomous Operation!", style="bold green"
        )
    )
    console.print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
