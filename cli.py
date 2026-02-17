"""
CLI Interface - Command-line tools for Open-Sable
"""
import click
import asyncio
import logging
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from pathlib import Path
import json

from opensable.core.gateway import GatewayServer
from opensable.core.session_manager import SessionManager
from opensable.core.config import Config
from opensable.skills.code_executor import CodeExecutor, ExecutionConfig
from opensable.skills.file_manager import FileManager
from opensable.skills.database_skill import DatabaseManager, DatabaseConfig
from opensable.skills.api_client import APIClient, APIAuth, AuthType
from opensable.skills.rag_skill import RAGSystem, Document
from opensable.core.monitoring import MetricsCollector, HealthChecker

try:
    from opensable.skills.advanced_scraper import AdvancedScraper, ScrapingRecipe, ExtractionRule, ScrapingAction
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

from opensable.skills.image_skill import ImageGenerator, OCREngine, ImageAnalyzer
from opensable.core.advanced_ai import PromptLibrary, ChainOfThought, SelfReflection
from opensable.core.enterprise import MultiTenancy, RBAC, AuditLogger, SSOProvider
from opensable.core.observability import DistributedTracer, LogAggregator
from opensable.core.workflow_persistence import WorkflowEngine, WorkflowLibrary
from opensable.core.interface_sdk import InterfaceRegistry, InterfaceConfig, InterfaceType

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🚀 Open-Sable - Personal AI Assistant (100% Feature Parity)"""
    pass


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=18789, help="Port to listen on")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def gateway(host, port, verbose):
    """Start the Gateway control plane"""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    console.print(f"[bold green]Starting Gateway on {host}:{port}[/bold green]")
    
    try:
        server = GatewayServer(host=host, port=port)
        asyncio.run(server.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Gateway stopped[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


@cli.command()
@click.option("--message", "-m", required=True, help="Message to send")
@click.option("--channel", "-c", default="cli", help="Channel (cli, telegram, discord)")
@click.option("--user", "-u", default="cli-user", help="User ID")
@click.option("--model", default=None, help="Model to use")
def agent(message, channel, user, model):
    """Send a message directly to the agent"""
    from opensable.core.agent import SableAgent
    from opensable.core.llm import initialize_llm
    
    console.print(f"[cyan]Processing: {message}[/cyan]")
    
    try:
        config = Config()
        if model:
            config.default_model = model
        
        llm = initialize_llm(config)
        agent_instance = SableAgent(llm)
        
        # Create session
        session_manager = SessionManager()
        session = session_manager.get_or_create_session(channel, user)
        session.add_message("user", message)
        
        # Process
        response = asyncio.run(agent_instance.process_message(user, message))
        
        session.add_message("assistant", response)
        session_manager._save_session(session)
        
        console.print(f"\n[green]{response}[/green]\n")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise


@cli.command()
def sessions():
    """List all sessions"""
    session_manager = SessionManager()
    all_sessions = session_manager.list_sessions()
    
    if not all_sessions:
        console.print("[yellow]No sessions found[/yellow]")
        return
    
    table = Table(title="Active Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Channel", style="magenta")
    table.add_column("User", style="green")
    table.add_column("Messages", justify="right")
    table.add_column("Created", style="dim")
    table.add_column("Updated", style="dim")
    
    for session in all_sessions:
        table.add_row(
            session.id[:8] + "...",
            session.channel,
            session.user_id,
            str(len(session.messages)),
            session.created_at[:19],
            session.updated_at[:19]
        )
    
    console.print(table)


@cli.command()
@click.argument("session_id")
def session_info(session_id):
    """Show detailed session information"""
    session_manager = SessionManager()
    
    # Try to find session by partial ID
    for sess in session_manager.sessions.values():
        if sess.id.startswith(session_id):
            session = sess
            break
    else:
        console.print(f"[red]Session not found: {session_id}[/red]")
        return
    
    console.print(f"\n[bold]Session: {session.id}[/bold]\n")
    console.print(f"Channel: {session.channel}")
    console.print(f"User: {session.user_id}")
    console.print(f"Model: {session.config.model}")
    console.print(f"Messages: {len(session.messages)}")
    console.print(f"State: {session.state}")
    console.print(f"Created: {session.created_at}")
    console.print(f"Updated: {session.updated_at}")
    
    if session.messages:
        console.print("\n[bold]Recent Messages:[/bold]")
        for msg in session.messages[-5:]:
            role_color = "cyan" if msg.role == "user" else "green"
            console.print(f"[{role_color}]{msg.role}:[/{role_color}] {msg.content[:100]}...")


@cli.command()
@click.argument("session_id")
def reset_session(session_id):
    """Reset a session (clear messages)"""
    session_manager = SessionManager()
    
    # Find session
    for sess in session_manager.sessions.values():
        if sess.id.startswith(session_id):
            if session_manager.reset_session(sess.id):
                console.print(f"[green]✅ Session {sess.id[:8]}... reset[/green]")
            else:
                console.print(f"[red]Failed to reset session[/red]")
            return
    
    console.print(f"[red]Session not found: {session_id}[/red]")


@cli.command()
def doctor():
    """Run system diagnostics"""
    console.print("[bold]🔍 Running Open-Sable Diagnostics[/bold]\n")
    
    # Check config
    try:
        config = Config()
        console.print("[green]✅ Configuration loaded[/green]")
        console.print(f"   Agent: {config.agent_name}")
        console.print(f"   Model: {config.default_model}")
    except Exception as e:
        console.print(f"[red]❌ Config error: {e}[/red]")
    
    # Check sessions directory
    try:
        session_manager = SessionManager()
        session_count = len(session_manager.sessions)
        console.print(f"[green]✅ Sessions: {session_count} loaded[/green]")
    except Exception as e:
        console.print(f"[red]❌ Sessions error: {e}[/red]")
    
    # Check gateway lock
    lock_file = Path.home() / ".opensable" / "gateway.lock"
    if lock_file.exists():
        console.print(f"[yellow]⚠️  Gateway lock exists: {lock_file}[/yellow]")
        pid = lock_file.read_text().strip()
        console.print(f"   PID: {pid}")
    else:
        console.print("[green]✅ No gateway lock (not running)[/green]")
    
    # Check Ollama
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            console.print("[green]✅ Ollama is available[/green]")
            models = [line.split()[0] for line in result.stdout.split("\n")[1:] if line]
            if models:
                console.print(f"   Models: {', '.join(models[:3])}" + ("..." if len(models) > 3 else ""))
        else:
            console.print("[yellow]⚠️  Ollama not responding[/yellow]")
    except FileNotFoundError:
        console.print("[red]❌ Ollama not installed[/red]")
    
    console.print("\n[bold green]Diagnostics complete![/bold green]")


@cli.command()
def onboard():
    """Interactive setup wizard"""
    console.print("[bold cyan]🚀 Welcome to Open-Sable Setup![/bold cyan]\n")
    
    env_file = Path(".env")
    
    if env_file.exists():
        overwrite = click.confirm("⚠️  .env already exists. Overwrite?", default=False)
        if not overwrite:
            console.print("[yellow]Setup cancelled[/yellow]")
            return
    
    console.print("Let's configure your AI assistant...\n")
    
    # Agent name
    agent_name = click.prompt("Agent name", default="Open-Sable")
    
    # Model
    console.print("\n[bold]Choose default model:[/bold]")
    console.print("1. llama3.1:8b (recommended)")
    console.print("2. llama3.1:70b (high-end)")
    console.print("3. custom")
    
    model_choice = click.prompt("Choice", type=int, default=1)
    
    if model_choice == 1:
        model = "llama3.1:8b"
    elif model_choice == 2:
        model = "llama3.1:70b"
    else:
        model = click.prompt("Model name")
    
    # Telegram
    setup_telegram = click.confirm("\nSetup Telegram bot?", default=True)
    telegram_token = ""
    if setup_telegram:
        telegram_token = click.prompt("Telegram bot token (from @BotFather)")
    
    # Discord
    setup_discord = click.confirm("\nSetup Discord bot?", default=False)
    discord_token = ""
    if setup_discord:
        discord_token = click.prompt("Discord bot token")
    
    # Write .env
    env_content = f"""# Open-Sable Configuration

# Agent settings
AGENT_NAME={agent_name}
DEFAULT_MODEL={model}
AUTO_SELECT_MODEL=true

# Telegram
TELEGRAM_BOT_ENABLED={'true' if setup_telegram else 'false'}
TELEGRAM_BOT_TOKEN={telegram_token}

# Discord
DISCORD_BOT_ENABLED={'true' if setup_discord else 'false'}
DISCORD_BOT_TOKEN={discord_token}

# Gateway
GATEWAY_HOST=127.0.0.1
GATEWAY_PORT=18789
"""
    
    env_file.write_text(env_content)
    
    console.print("\n[bold green]✅ Configuration saved to .env[/bold green]")
    console.print("\nNext steps:")
    console.print("1. Install dependencies: [cyan]pip install -r requirements.txt[/cyan]")
    console.print("2. Start gateway: [cyan]sable gateway[/cyan]")
    console.print("3. Start agent: [cyan]python main.py[/cyan]")
    console.print("\n[bold]Happy building! 🚀[/bold]")

@cli.command()
@click.argument('url')
@click.option('--recipe', '-r', help='Recipe name to use')
@click.option('--save-recipe', '-s', help='Save as recipe')
@click.option('--headless/--no-headless', default=True, help='Run in headless mode')
def scrape(url, recipe, save_recipe, headless):
    """Advanced web scraping with recording"""
    if not SCRAPER_AVAILABLE:
        console.print("[red]Advanced scraper not available. Install: pip install playwright fake-useragent[/red]")
        return
    
    import asyncio
    
    async def run_scraper():
        scraper = AdvancedScraper()
        
        if recipe:
            # Load and execute existing recipe
            console.print(f"[cyan]Loading recipe: {recipe}[/cyan]")
            loaded_recipe = scraper.load_recipe(recipe)
            results = await scraper.execute_recipe(loaded_recipe)
            
            console.print(f"\n[green]✓ Extracted {len(results)} items[/green]")
            if results:
                console.print_json(data=results[0])
        else:
            # Interactive scraping
            console.print(f"[cyan]Opening browser for: {url}[/cyan]")
            console.print("[yellow]Tip: Actions will be recorded. Close browser when done.[/yellow]")
            
            scraper.start_recording()
            await scraper.start_browser(headless=headless)
            
            page = await scraper.navigate(url)
            
            # Wait for user to close browser
            console.print("[yellow]Browser ready. Perform your scraping actions...[/yellow]")
            
            # Keep browser open until user closes it
            try:
                await asyncio.sleep(300)  # 5 minutes max
            except KeyboardInterrupt:
                pass
            
            actions = scraper.stop_recording()
            await scraper.stop_browser()
            
            console.print(f"\n[green]✓ Recorded {len(actions)} actions[/green]")
            
            # Display recorded actions
            table = Table(title="Recorded Actions")
            table.add_column("#", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Selector", style="green")
            table.add_column("Value", style="yellow")
            
            for i, action in enumerate(actions, 1):
                table.add_row(
                    str(i),
                    action.action_type,
                    action.selector,
                    action.value or ""
                )
            
            console.print(table)
            
            # Save recipe if requested
            if save_recipe:
                new_recipe = ScrapingRecipe(
                    name=save_recipe,
                    start_url=url,
                    actions=actions
                )
                scraper.save_recipe(new_recipe)
                console.print(f"\n[green]✓ Recipe saved as: {save_recipe}[/green]")
    
    asyncio.run(run_scraper())


@cli.command()
def scrape_recipes():
    """List all scraping recipes"""
    if not SCRAPER_AVAILABLE:
        console.print("[red]Advanced scraper not available[/red]")
        return
    
    scraper = AdvancedScraper()
    recipes = scraper.list_recipes()
    
    if not recipes:
        console.print("[yellow]No recipes found[/yellow]")
        return
    
    table = Table(title="Scraping Recipes")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    
    for recipe_name in recipes:
        recipe = scraper.load_recipe(recipe_name)
        table.add_row(
            recipe_name,
            f"{len(recipe.actions)} actions, {len(recipe.extraction_rules)} rules"
        )
    
    console.print(table)
    console.print(f"\n[cyan]Total: {len(recipes)} recipes[/cyan]")


@cli.command()
@click.argument("code")
@click.option("--language", "-l", default="python", help="Language (python, javascript, bash)")
@click.option("--timeout", "-t", default=30, help="Timeout in seconds")
@click.option("--docker/--no-docker", default=False, help="Use Docker isolation")
def execute(code, language, timeout, docker):
    """Execute code in sandboxed environment"""
    console.print(f"[cyan]Executing {language} code...[/cyan]\n")
    
    executor = CodeExecutor(use_docker=docker)
    config = ExecutionConfig(timeout=timeout)
    
    result = asyncio.run(executor.execute(code, language, config))
    
    if result.success:
        console.print("[green]✅ Execution successful[/green]")
        console.print(f"[bold]Output:[/bold]\n{result.output}")
        console.print(f"\n[dim]Execution time: {result.execution_time:.3f}s[/dim]")
    else:
        console.print("[red]❌ Execution failed[/red]")
        console.print(f"[bold red]Error:[/bold red]\n{result.error}")


@cli.command()
@click.option("--list-files", "-l", is_flag=True, help="List all files")
@click.option("--stats", "-s", is_flag=True, help="Show storage stats")
@click.option("--search", help="Search for files")
def files(list_files, stats, search):
    """Manage files in storage"""
    fm = FileManager()
    
    if stats:
        storage_stats = asyncio.run(fm.get_storage_stats())
        
        table = Table(title="Storage Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Files", str(storage_stats["total_files"]))
        table.add_row("Total Directories", str(storage_stats["total_directories"]))
        table.add_row("Total Size", storage_stats["total_size_human"])
        table.add_row("Disk Usage", f"{storage_stats['disk_usage_percent']:.1f}%")
        table.add_row("Storage Dir", storage_stats["storage_dir"])
        
        console.print(table)
    
    elif search:
        results = asyncio.run(fm.search_files(search))
        
        table = Table(title=f"Search Results: '{search}'")
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="yellow")
        
        for file_info in results:
            table.add_row(
                file_info.name,
                file_info.human_size(),
                file_info.modified.strftime("%Y-%m-%d %H:%M")
            )
        
        console.print(table)
    
    elif list_files:
        file_list = asyncio.run(fm.list_files(recursive=True))
        
        table = Table(title="Files in Storage")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="yellow")
        
        for file_info in file_list[:50]:  # Limit to 50
            table.add_row(
                file_info.name,
                "DIR" if file_info.is_directory else file_info.mime_type or "FILE",
                file_info.human_size() if not file_info.is_directory else "-",
                file_info.modified.strftime("%Y-%m-%d %H:%M")
            )
        
        if len(file_list) > 50:
            console.print(f"\n[dim]Showing 50 of {len(file_list)} files[/dim]")
        
        console.print(table)
    else:
        console.print("[yellow]Use --list-files, --stats, or --search[/yellow]")


@cli.command()
@click.argument("database_type")
@click.argument("query")
@click.option("--database", "-d", help="Database name")
@click.option("--host", default="localhost", help="Database host")
@click.option("--port", type=int, help="Database port")
def db(database_type, query, database, host, port):
    """Execute database query"""
    console.print(f"[cyan]Connecting to {database_type} database...[/cyan]\n")
    
    config = DatabaseConfig(
        type=database_type,
        database=database,
        host=host,
        port=port
    )
    
    db_manager = DatabaseManager(config)
    
    try:
        result = asyncio.run(db_manager.execute(query))
        
        if result.success:
            console.print("[green]✅ Query successful[/green]")
            
            if result.rows:
                table = Table(title="Query Results")
                
                # Add columns
                if result.rows:
                    for col in result.rows[0].keys():
                        table.add_column(col, style="cyan")
                    
                    # Add rows
                    for row in result.rows[:20]:  # Limit to 20
                        table.add_row(*[str(v) for v in row.values()])
                    
                    console.print(table)
                    
                    if len(result.rows) > 20:
                        console.print(f"\n[dim]Showing 20 of {result.row_count} rows[/dim]")
            else:
                console.print(f"Affected rows: {result.affected_rows}")
            
            console.print(f"[dim]Execution time: {result.execution_time:.3f}s[/dim]")
        else:
            console.print(f"[red]❌ Query failed: {result.error}[/red]")
    
    finally:
        asyncio.run(db_manager.disconnect())


@cli.command()
@click.argument("url")
@click.argument("endpoint")
@click.option("--method", "-m", default="GET", help="HTTP method")
@click.option("--data", "-d", help="JSON data for request")
@click.option("--api-key", help="API key for authentication")
def api(url, endpoint, method, data, api_key):
    """Call external REST API"""
    console.print(f"[cyan]{method} {url}{endpoint}[/cyan]\n")
    
    # Setup auth
    auth = None
    if api_key:
        auth = APIAuth(type=AuthType.API_KEY, api_key=api_key)
    
    async def make_request():
        async with APIClient(url, auth=auth) as client:
            json_data = json.loads(data) if data else None
            response = await client.request(method, endpoint, json_data=json_data)
            return response
    
    response = asyncio.run(make_request())
    
    if response.success:
        console.print("[green]✅ Request successful[/green]")
        console.print(f"Status: {response.status_code}")
        console.print(f"\n[bold]Response:[/bold]")
        
        if isinstance(response.data, dict) or isinstance(response.data, list):
            console.print(json.dumps(response.data, indent=2))
        else:
            console.print(response.text)
        
        console.print(f"\n[dim]Response time: {response.response_time:.3f}s[/dim]")
    else:
        console.print(f"[red]❌ Request failed: {response.error}[/red]")


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results")
@click.option("--collection", "-c", default="documents", help="Collection name")
def search(query, top_k, collection):
    """Search documents with RAG"""
    console.print(f"[cyan]Searching for: {query}[/cyan]\n")
    
    rag = RAGSystem(collection_name=collection)
    
    results = asyncio.run(rag.search(query, top_k=top_k))
    
    if results:
        table = Table(title=f"Search Results ({len(results)} found)")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Score", style="green", width=8)
        table.add_column("Content", style="white")
        
        for result in results:
            content_preview = result.document.content[:100] + "..."
            table.add_row(
                str(result.rank),
                f"{result.score:.3f}",
                content_preview
            )
        
        console.print(table)
    else:
        console.print("[yellow]No results found[/yellow]")


@cli.command()
@click.option("--port", "-p", default=9090, help="Metrics port")
@click.option("--health", is_flag=True, help="Show health status")
def metrics(port, health):
    """Start metrics server or show health"""
    
    if health:
        # Show health status
        health_checker = HealthChecker()
        
        # Register basic checks
        health_checker.register_check("system", lambda: True)
        
        status = asyncio.run(health_checker.check_health(use_cache=False))
        
        console.print(f"\n[bold]Health Status:[/bold] ", end="")
        if status.status == "healthy":
            console.print("[green]✅ Healthy[/green]")
        elif status.status == "degraded":
            console.print("[yellow]⚠️  Degraded[/yellow]")
        else:
            console.print("[red]❌ Unhealthy[/red]")
        
        table = Table(title="Health Checks")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="white")
        
        for name, result in status.checks.items():
            status_icon = "✅" if result["healthy"] else "❌"
            table.add_row(name, f"{status_icon} {result['status']}")
        
        console.print(table)
    else:
        # Start metrics server
        console.print(f"[green]Starting metrics server on port {port}...[/green]")
        console.print(f"[dim]Metrics available at http://localhost:{port}/metrics[/dim]\n")
        
        metrics_collector = MetricsCollector()
        
        try:
            metrics_collector.start_server(port)
            console.print("[green]✅ Metrics server running[/green]")
            console.print("[yellow]Press Ctrl+C to stop[/yellow]")
            
            # Keep running
            import signal
            signal.pause()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down metrics server...[/yellow]")


@cli.command()
@click.argument("action", type=click.Choice(["generate", "ocr", "analyze"]))
@click.option("--prompt", help="Image generation prompt")
@click.option("--image", type=click.Path(exists=True), help="Image file path")
def image(action, prompt, image):
    """Image generation, OCR, and analysis"""
    
    async def run_image():
        if action == "generate":
            if not prompt:
                console.print("[red]Error: --prompt required for generation[/red]")
                return
            
            console.print(f"[green]Generating image: {prompt}[/green]")
            generator = ImageGenerator()
            result = await generator.generate(prompt)
            console.print(f"[green]✅ Image saved: {result.image_path}[/green]")
            
        elif action == "ocr":
            if not image:
                console.print("[red]Error: --image required for OCR[/red]")
                return
            
            console.print(f"[green]Extracting text from: {image}[/green]")
            ocr = OCREngine()
            result = await ocr.extract_text(image)
            console.print(f"[cyan]Extracted text ({result.confidence:.2f} confidence):[/cyan]")
            console.print(result.text)
            
        elif action == "analyze":
            if not image:
                console.print("[red]Error: --image required for analysis[/red]")
                return
            
            console.print(f"[green]Analyzing image: {image}[/green]")
            analyzer = ImageAnalyzer()
            result = await analyzer.analyze(image)
            console.print(f"[cyan]Faces detected: {len(result.faces)}[/cyan]")
            console.print(f"[cyan]Dominant colors: {', '.join(result.colors[:5])}[/cyan]")
    
    asyncio.run(run_image())


@cli.command()
@click.argument("action", type=click.Choice(["list", "render", "add"]))
@click.option("--name", help="Template name")
@click.option("--vars", help="Template variables (JSON)")
def prompts(action, name, vars):
    """Manage prompt templates"""
    
    library = PromptLibrary()
    
    if action == "list":
        templates = library.list_templates()
        console.print(f"[green]Available templates: {len(templates)}[/green]")
        for t in templates:
            template = library.get(t)
            console.print(f"  • {t}: {template.description if template else ''}")
    
    elif action == "render":
        if not name:
            console.print("[red]Error: --name required[/red]")
            return
        
        template_vars = json.loads(vars) if vars else {}
        result = library.render(name, **template_vars)
        if result:
            console.print(f"[cyan]Rendered prompt:[/cyan]\n{result}")
        else:
            console.print("[red]Template not found or invalid variables[/red]")


@cli.command()
@click.argument("action", type=click.Choice(["create", "list", "users"]))
@click.option("--tenant", help="Tenant name/ID")
@click.option("--plan", default="free", help="Tenant plan (free/pro/enterprise)")
def tenants(action, tenant, plan):
    """Manage multi-tenancy"""
    
    tenancy = MultiTenancy()
    
    if action == "create":
        if not tenant:
            console.print("[red]Error: --tenant required[/red]")
            return
        
        t = tenancy.create_tenant(tenant, plan)
        console.print(f"[green]✅ Created tenant: {t.name} (ID: {t.id})[/green]")
        console.print(f"[cyan]Plan: {t.plan}[/cyan]")
        console.print(f"[cyan]Limits: {t.limits}[/cyan]")
    
    elif action == "list":
        console.print(f"[green]Tenants: {len(tenancy.tenants)}[/green]")
        for tid, t in tenancy.tenants.items():
            console.print(f"  • {t.name} ({t.plan}) - {tid}")


@cli.command()
@click.argument("workflow_id")
@click.option("--resume", help="Resume from checkpoint ID")
def workflow(workflow_id, resume):
    """Execute workflows"""
    
    async def run_workflow():
        library = WorkflowLibrary()
        engine = WorkflowEngine()
        
        wf = library.get(workflow_id)
        if not wf:
            console.print(f"[red]Workflow not found: {workflow_id}[/red]")
            return
        
        console.print(f"[green]Executing workflow: {wf.name}[/green]")
        result = await engine.execute(wf, resume_from=resume)
        
        console.print(f"[cyan]Status: {result['status']}[/cyan]")
        console.print(f"[cyan]Completed: {len(result['completed_steps'])}/{len(wf.steps)}[/cyan]")
    
    asyncio.run(run_workflow())


@cli.command()
def interfaces():
    """List available custom interfaces"""
    
    registry = InterfaceRegistry()
    available = registry.list_interfaces()
    
    console.print(f"[green]Available interfaces: {len(available)}[/green]")
    for name in available:
        console.print(f"  • {name}")


if __name__ == "__main__":
    cli()
