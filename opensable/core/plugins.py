"""
Open-Sable Plugin System

Dynamic plugin loading and management.
Allows extending Open-Sable with custom skills, interfaces, and tools.
"""

import asyncio
import logging
import importlib
import inspect
from typing import Dict, List, Any, Callable
from pathlib import Path
import json

from opensable.core.config import Config

logger = logging.getLogger(__name__)


class Plugin:
    """Base plugin class"""

    name: str = "UnnamedPlugin"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""

    def __init__(self, config: Config):
        self.config = config
        self.enabled = True

    async def initialize(self):
        """Initialize plugin (called on load)"""
        pass

    async def cleanup(self):
        """Cleanup plugin resources (called on unload)"""
        pass

    def get_commands(self) -> Dict[str, Callable]:
        """Return dict of command_name -> handler_function"""
        return {}

    def get_hooks(self) -> Dict[str, Callable]:
        """Return dict of hook_name -> hook_function"""
        return {}


class PluginMetadata:
    """Plugin metadata"""

    def __init__(self, path: Path):
        self.path = path
        self.manifest_path = path / "plugin.json"
        self.manifest = self._load_manifest()

        self.name = self.manifest.get("name", path.name)
        self.version = self.manifest.get("version", "0.1.0")
        self.description = self.manifest.get("description", "")
        self.author = self.manifest.get("author", "")
        self.dependencies = self.manifest.get("dependencies", [])
        self.entry_point = self.manifest.get("entry_point", "plugin.py")

    def _load_manifest(self) -> dict:
        """Load plugin manifest"""
        if not self.manifest_path.exists():
            return {}

        try:
            with open(self.manifest_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading plugin manifest: {e}")
            return {}


class PluginManager:
    """Manages plugin lifecycle and execution"""

    def __init__(self, config: Config):
        self.config = config
        self.plugins_dir = Path.home() / ".opensable" / "plugins"
        self.plugins_dir.mkdir(parents=True, exist_ok=True)

        # Loaded plugins
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}

        # Command registry
        self.commands: Dict[str, Callable] = {}

        # Hook registry
        self.hooks: Dict[str, List[Callable]] = {}

    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins"""
        plugins = []

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            if plugin_dir.name.startswith(".") or plugin_dir.name.startswith("__"):
                continue

            try:
                metadata = PluginMetadata(plugin_dir)
                plugins.append(metadata)
                logger.info(f"Discovered plugin: {metadata.name} v{metadata.version}")
            except Exception as e:
                logger.error(f"Error discovering plugin {plugin_dir.name}: {e}")

        return plugins

    async def load_plugin(self, metadata: PluginMetadata) -> bool:
        """Load and initialize a plugin"""
        try:
            logger.info(f"Loading plugin: {metadata.name}")

            # Add plugin directory to path
            import sys

            plugin_path = str(metadata.path)
            if plugin_path not in sys.path:
                sys.path.insert(0, plugin_path)

            # Import plugin module
            entry_module = metadata.entry_point.replace(".py", "")
            module = importlib.import_module(entry_module)

            # Find Plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Plugin) and obj is not Plugin:
                    plugin_class = obj
                    break

            if not plugin_class:
                logger.error(f"No Plugin class found in {metadata.name}")
                return False

            # Instantiate plugin
            plugin = plugin_class(self.config)

            # Initialize
            await plugin.initialize()

            # Store plugin
            self.plugins[metadata.name] = plugin
            self.plugin_metadata[metadata.name] = metadata

            # Register commands
            for cmd_name, cmd_handler in plugin.get_commands().items():
                self.commands[cmd_name] = cmd_handler
                logger.debug(f"Registered command: {cmd_name}")

            # Register hooks
            for hook_name, hook_handler in plugin.get_hooks().items():
                if hook_name not in self.hooks:
                    self.hooks[hook_name] = []
                self.hooks[hook_name].append(hook_handler)
                logger.debug(f"Registered hook: {hook_name}")

            logger.info(f"✅ Loaded plugin: {metadata.name}")
            return True

        except Exception as e:
            logger.error(f"Error loading plugin {metadata.name}: {e}", exc_info=True)
            return False

    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin not loaded: {plugin_name}")
            return False

        try:
            plugin = self.plugins[plugin_name]

            # Cleanup
            await plugin.cleanup()

            # Unregister commands
            for cmd_name, cmd_handler in plugin.get_commands().items():
                if cmd_name in self.commands:
                    del self.commands[cmd_name]

            # Unregister hooks
            for hook_name, hook_handler in plugin.get_hooks().items():
                if hook_name in self.hooks:
                    self.hooks[hook_name].remove(hook_handler)

            # Remove plugin
            del self.plugins[plugin_name]
            del self.plugin_metadata[plugin_name]

            logger.info(f"Unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}", exc_info=True)
            return False

    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin"""
        if plugin_name not in self.plugin_metadata:
            return False

        metadata = self.plugin_metadata[plugin_name]

        # Unload
        await self.unload_plugin(plugin_name)

        # Load
        return await self.load_plugin(metadata)

    async def load_all_plugins(self):
        """Discover and load all plugins"""
        plugins = self.discover_plugins()

        logger.info(f"Loading {len(plugins)} plugins...")

        for metadata in plugins:
            await self.load_plugin(metadata)

        logger.info(f"Loaded {len(self.plugins)}/{len(plugins)} plugins")

    async def execute_command(self, command_name: str, *args, **kwargs) -> Any:
        """Execute a plugin command"""
        if command_name not in self.commands:
            raise ValueError(f"Unknown command: {command_name}")

        handler = self.commands[command_name]

        # Check if async
        if asyncio.iscoroutinefunction(handler):
            return await handler(*args, **kwargs)
        else:
            return handler(*args, **kwargs)

    async def execute_hook(self, hook_name: str, *args, **kwargs):
        """Execute all handlers for a hook"""
        if hook_name not in self.hooks:
            return

        for handler in self.hooks[hook_name]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error executing hook {hook_name}: {e}", exc_info=True)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins"""
        return [
            {
                "name": metadata.name,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "enabled": self.plugins[name].enabled,
                "commands": list(self.plugins[name].get_commands().keys()),
                "hooks": list(self.plugins[name].get_hooks().keys()),
            }
            for name, metadata in self.plugin_metadata.items()
        ]


# Example plugin
class ExamplePlugin(Plugin):
    """Example plugin demonstrating the plugin system"""

    name = "Example Plugin"
    version = "1.0.0"
    description = "Example plugin showing how to extend Open-Sable"
    author = "Open-Sable Team"

    def __init__(self, config: Config):
        super().__init__(config)
        self.counter = 0

    async def initialize(self):
        """Initialize plugin"""
        logger.info(f"Initializing {self.name}")
        self.counter = 0

    async def cleanup(self):
        """Cleanup plugin"""
        logger.info(f"Cleaning up {self.name}")

    def get_commands(self) -> Dict[str, Callable]:
        """Register commands"""
        return {"example_hello": self.cmd_hello, "example_count": self.cmd_count}

    def get_hooks(self) -> Dict[str, Callable]:
        """Register hooks"""
        return {"message_received": self.on_message_received, "message_sent": self.on_message_sent}

    async def cmd_hello(self, name: str = "World") -> str:
        """Example command: say hello"""
        return f"Hello, {name}! This is {self.name} v{self.version}"

    async def cmd_count(self) -> str:
        """Example command: increment counter"""
        self.counter += 1
        return f"Counter: {self.counter}"

    async def on_message_received(self, message: str, user_id: str):
        """Hook: called when message is received"""
        logger.debug(f"[{self.name}] Message received from {user_id}: {message}")

    async def on_message_sent(self, message: str, user_id: str):
        """Hook: called when message is sent"""
        logger.debug(f"[{self.name}] Message sent to {user_id}: {message}")


if __name__ == "__main__":
    from opensable.core.config import load_config

    config = load_config()
    manager = PluginManager(config)

    # Test
    asyncio.run(manager.load_all_plugins())

    print("\nLoaded plugins:")
    for plugin in manager.list_plugins():
        print(f"  - {plugin['name']} v{plugin['version']}")
        print(f"    Commands: {', '.join(plugin['commands'])}")
        print(f"    Hooks: {', '.join(plugin['hooks'])}")
