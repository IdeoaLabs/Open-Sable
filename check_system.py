"""
System info utility - Display detected specs and recommended configuration
"""

import asyncio
from core.system_detector import (
    SystemDetector,
    ModelSelector,
    ResourceMonitor,
    auto_configure_system,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def main():
    """Display system information and recommendations"""
    console = Console()

    console.print(
        Panel.fit(
            "[bold cyan]Open-Sable System Detector[/bold cyan]\n"
            "Analyzing your device for optimal configuration",
            border_style="cyan",
        )
    )

    # Detect specs
    console.print("\n[yellow]Detecting hardware...[/yellow]")
    specs = SystemDetector.detect()

    # Create specs table
    specs_table = Table(title="System Specifications", show_header=True)
    specs_table.add_column("Component", style="cyan")
    specs_table.add_column("Value", style="green")

    specs_table.add_row("RAM", f"{specs.ram_gb:.1f} GB")
    specs_table.add_row("CPU Cores", str(specs.cpu_cores))
    specs_table.add_row("CPU Frequency", f"{specs.cpu_freq_ghz:.2f} GHz")
    specs_table.add_row("GPU Available", "Yes" if specs.gpu_available else "No")
    if specs.gpu_available:
        specs_table.add_row("GPU Memory", f"{specs.gpu_memory_gb:.1f} GB")
    specs_table.add_row("Free Storage", f"{specs.storage_free_gb:.1f} GB")
    specs_table.add_row("OS", specs.system)

    console.print(specs_table)

    # Get tier
    tier = SystemDetector.get_device_tier(specs)
    tier_colors = {
        "high-end": "green",
        "mid-range": "yellow",
        "low-end": "orange",
        "minimal": "red",
    }
    console.print(
        f"\n[bold]Device Tier:[/bold] [{tier_colors[tier]}]{tier.upper()}[/{tier_colors[tier]}]"
    )

    # Model recommendations
    recommended_models = ModelSelector.MODEL_RECOMMENDATIONS[tier]
    console.print(f"\n[bold]Recommended Models:[/bold]")
    for i, model in enumerate(recommended_models, 1):
        reqs = ModelSelector.get_memory_requirements(model)
        console.print(f"  {i}. {model} (min RAM: {reqs['min_ram']}GB)")

    # Auto configuration
    console.print("\n[yellow]Running auto-configuration...[/yellow]")
    auto_config = auto_configure_system()

    # Config table
    config_table = Table(title="Recommended Configuration", show_header=True)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Model", auto_config["recommended_model"])
    config_table.add_row("Batch Size", str(auto_config["batch_size"]))
    config_table.add_row("Use GPU", "Yes" if auto_config["use_gpu"] else "No")
    config_table.add_row("Max Workers", str(auto_config["max_workers"]))

    console.print(config_table)

    # Current usage
    console.print("\n[yellow]Current resource usage...[/yellow]")
    usage = ResourceMonitor.get_current_usage()

    usage_table = Table(show_header=True)
    usage_table.add_column("Resource", style="cyan")
    usage_table.add_column("Usage", style="yellow")

    usage_table.add_row(
        "RAM",
        f"{usage['ram_used_percent']:.1f}% ({usage['ram_used_gb']:.1f}GB / {usage['ram_available_gb']:.1f}GB available)",
    )
    usage_table.add_row("CPU", f"{usage['cpu_percent']:.1f}%")
    usage_table.add_row(
        "Disk", f"{usage['disk_used_percent']:.1f}% ({usage['disk_free_gb']:.1f}GB free)"
    )

    console.print(usage_table)

    # Recommendations
    console.print(
        Panel.fit(
            f"[bold green]✅ Recommended .env settings:[/bold green]\n\n"
            f"DEFAULT_MODEL={auto_config['recommended_model']}\n"
            f"AUTO_SELECT_MODEL=true\n\n"
            f"[dim]Open-Sable will automatically use the best model for your device.[/dim]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
