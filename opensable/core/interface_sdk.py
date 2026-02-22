"""
Custom Interface SDK - Build custom chat interfaces.

Features:
- Interface base class and lifecycle
- Event handling
- Message formatting
- Plugin system
- Interface discovery
- Example implementations
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import importlib
import inspect
from pathlib import Path


class InterfaceType(Enum):
    """Interface types."""

    CHAT = "chat"
    VOICE = "voice"
    API = "api"
    WEBHOOK = "webhook"
    CUSTOM = "custom"


class MessageType(Enum):
    """Message types."""

    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    LOCATION = "location"
    COMMAND = "command"


@dataclass
class Message:
    """Interface message."""

    id: str
    type: MessageType
    content: Any
    sender_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class InterfaceConfig:
    """Interface configuration."""

    name: str
    type: InterfaceType
    enabled: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "enabled": self.enabled,
            "settings": self.settings,
        }


class InterfaceLifecycle(ABC):
    """
    Base class for custom interfaces.

    Implement this class to create a custom chat interface.
    """

    def __init__(self, config: InterfaceConfig):
        """
        Initialize interface.

        Args:
            config: Interface configuration
        """
        self.config = config
        self.is_running = False
        self.message_handlers: List[Callable] = []
        self.event_handlers: Dict[str, List[Callable]] = {}

    @abstractmethod
    async def start(self):
        """
        Start the interface.

        Called when interface is initialized.
        Setup connections, listeners, etc.
        """
        pass

    @abstractmethod
    async def stop(self):
        """
        Stop the interface.

        Called when interface is shutting down.
        Cleanup connections, close resources, etc.
        """
        pass

    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """
        Send message through interface.

        Args:
            message: Message to send

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    async def receive_message(self) -> Optional[Message]:
        """
        Receive message from interface.

        Returns:
            Message if available, None otherwise
        """
        pass

    async def on_message(self, message: Message):
        """
        Handle incoming message.

        Args:
            message: Received message
        """
        for handler in self.message_handlers:
            await handler(message)

    async def emit_event(self, event_name: str, data: Any = None):
        """
        Emit event.

        Args:
            event_name: Event name
            data: Event data
        """
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                await handler(data)

    def register_message_handler(self, handler: Callable):
        """Register message handler."""
        self.message_handlers.append(handler)

    def register_event_handler(self, event_name: str, handler: Callable):
        """Register event handler."""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)

    def get_info(self) -> Dict[str, Any]:
        """Get interface information."""
        return {
            "name": self.config.name,
            "type": self.config.type.value,
            "enabled": self.config.enabled,
            "running": self.is_running,
        }


class WebSocketInterface(InterfaceLifecycle):
    """
    WebSocket interface implementation.

    Example custom interface using WebSockets.
    """

    def __init__(self, config: InterfaceConfig):
        """Initialize WebSocket interface."""
        super().__init__(config)
        self.host = config.settings.get("host", "0.0.0.0")
        self.port = config.settings.get("port", 8765)
        self.clients: List[Any] = []
        self.message_queue: asyncio.Queue = asyncio.Queue()

    async def start(self):
        """Start WebSocket server."""
        try:
            # Simulated WebSocket server start
            self.is_running = True
            await self.emit_event("started")
            print(f"WebSocket interface started on {self.host}:{self.port}")
        except Exception as e:
            await self.emit_event("error", {"error": str(e)})
            raise

    async def stop(self):
        """Stop WebSocket server."""
        self.is_running = False
        self.clients.clear()
        await self.emit_event("stopped")

    async def send_message(self, message: Message) -> bool:
        """Send message to WebSocket clients."""
        if not self.is_running:
            return False

        try:
            # Broadcast to all clients
            for client in self.clients:
                # Simulated send
                await self.emit_event("message_sent", message.to_dict())
            return True
        except Exception:
            return False

    async def receive_message(self) -> Optional[Message]:
        """Receive message from WebSocket clients."""
        if not self.is_running:
            return None

        try:
            # Get from queue (non-blocking)
            message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
            return message
        except asyncio.TimeoutError:
            return None

    async def handle_client_connection(self, websocket):
        """Handle new client connection."""
        self.clients.append(websocket)
        await self.emit_event("client_connected", {"client_id": id(websocket)})


class HTTPWebhookInterface(InterfaceLifecycle):
    """
    HTTP Webhook interface implementation.

    Example custom interface using HTTP webhooks.
    """

    def __init__(self, config: InterfaceConfig):
        """Initialize HTTP webhook interface."""
        super().__init__(config)
        self.endpoint = config.settings.get("endpoint", "/webhook")
        self.secret = config.settings.get("secret")
        self.incoming_queue: asyncio.Queue = asyncio.Queue()

    async def start(self):
        """Start HTTP webhook listener."""
        self.is_running = True
        await self.emit_event("started")
        print(f"HTTP webhook interface started at {self.endpoint}")

    async def stop(self):
        """Stop HTTP webhook listener."""
        self.is_running = False
        await self.emit_event("stopped")

    async def send_message(self, message: Message) -> bool:
        """Send message via HTTP POST."""
        if not self.is_running:
            return False

        try:
            # Simulated HTTP POST
            await self.emit_event("message_sent", message.to_dict())
            return True
        except Exception:
            return False

    async def receive_message(self) -> Optional[Message]:
        """Receive message from webhook."""
        if not self.is_running:
            return None

        try:
            message = await asyncio.wait_for(self.incoming_queue.get(), timeout=0.1)
            return message
        except asyncio.TimeoutError:
            return None

    async def handle_webhook_request(self, payload: Dict[str, Any]):
        """Handle incoming webhook request."""
        # Verify signature if secret is set
        if self.secret:
            # Signature verification logic
            pass

        # Convert payload to Message
        message = Message(
            id=payload.get("id", ""),
            type=MessageType.TEXT,
            content=payload.get("content"),
            sender_id=payload.get("sender_id", "unknown"),
        )

        await self.incoming_queue.put(message)
        await self.emit_event("webhook_received", payload)


class CustomCLIInterface(InterfaceLifecycle):
    """
    Custom CLI interface implementation.

    Example custom interface for command-line interaction.
    """

    def __init__(self, config: InterfaceConfig):
        """Initialize CLI interface."""
        super().__init__(config)
        self.prompt = config.settings.get("prompt", "> ")
        self.history: List[Message] = []
        self.input_queue: asyncio.Queue = asyncio.Queue()

    async def start(self):
        """Start CLI interface."""
        self.is_running = True
        await self.emit_event("started")
        print("CLI interface started. Type 'exit' to quit.")

        # Start input listener
        asyncio.create_task(self._listen_for_input())

    async def stop(self):
        """Stop CLI interface."""
        self.is_running = False
        await self.emit_event("stopped")
        print("CLI interface stopped.")

    async def send_message(self, message: Message) -> bool:
        """Send message to CLI (print)."""
        if not self.is_running:
            return False

        try:
            print(f"\nAgent: {message.content}")
            self.history.append(message)
            return True
        except Exception:
            return False

    async def receive_message(self) -> Optional[Message]:
        """Receive message from CLI (user input)."""
        if not self.is_running:
            return None

        try:
            text = await asyncio.wait_for(self.input_queue.get(), timeout=0.1)

            if text.lower() == "exit":
                await self.stop()
                return None

            message = Message(
                id=f"cli_{len(self.history)}", type=MessageType.TEXT, content=text, sender_id="user"
            )

            self.history.append(message)
            return message

        except asyncio.TimeoutError:
            return None

    async def _listen_for_input(self):
        """Listen for user input."""
        while self.is_running:
            try:
                # In real implementation, use aioconsole or similar
                # For now, simulate with asyncio.sleep
                await asyncio.sleep(0.1)
            except Exception:
                break


class InterfaceRegistry:
    """
    Registry for discovering and managing interfaces.

    Features:
    - Register custom interfaces
    - Discover interfaces from plugins
    - Load/unload interfaces
    - Interface lifecycle management
    """

    def __init__(self):
        """Initialize interface registry."""
        self.interfaces: Dict[str, Type[InterfaceLifecycle]] = {}
        self.active_interfaces: Dict[str, InterfaceLifecycle] = {}
        self._register_builtin_interfaces()

    def _register_builtin_interfaces(self):
        """Register built-in interfaces."""
        self.register("websocket", WebSocketInterface)
        self.register("webhook", HTTPWebhookInterface)
        self.register("cli", CustomCLIInterface)

    def register(self, name: str, interface_class: Type[InterfaceLifecycle]):
        """
        Register interface class.

        Args:
            name: Interface name
            interface_class: Interface class
        """
        if not issubclass(interface_class, InterfaceLifecycle):
            raise ValueError("Interface must inherit from InterfaceLifecycle")

        self.interfaces[name] = interface_class

    def unregister(self, name: str):
        """Unregister interface."""
        if name in self.interfaces:
            del self.interfaces[name]

    async def create_interface(
        self, name: str, config: InterfaceConfig
    ) -> Optional[InterfaceLifecycle]:
        """
        Create and start interface instance.

        Args:
            name: Interface name
            config: Interface configuration

        Returns:
            Interface instance
        """
        interface_class = self.interfaces.get(name)
        if not interface_class:
            return None

        try:
            interface = interface_class(config)
            await interface.start()
            self.active_interfaces[name] = interface
            return interface
        except Exception as e:
            print(f"Failed to create interface {name}: {e}")
            return None

    async def destroy_interface(self, name: str):
        """Destroy interface instance."""
        if name in self.active_interfaces:
            interface = self.active_interfaces[name]
            await interface.stop()
            del self.active_interfaces[name]

    def get_interface(self, name: str) -> Optional[InterfaceLifecycle]:
        """Get active interface."""
        return self.active_interfaces.get(name)

    def list_interfaces(self) -> List[str]:
        """List all registered interface names."""
        return list(self.interfaces.keys())

    def list_active_interfaces(self) -> List[str]:
        """List all active interface names."""
        return list(self.active_interfaces.keys())

    def discover_plugins(self, plugin_dir: str):
        """
        Discover interface plugins from directory.

        Args:
            plugin_dir: Directory containing plugins
        """
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            return

        # Find Python files
        for file in plugin_path.glob("*.py"):
            if file.stem.startswith("_"):
                continue

            try:
                # Import module
                spec = importlib.util.spec_from_file_location(file.stem, file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find InterfaceLifecycle subclasses
                    for name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, InterfaceLifecycle)
                            and obj != InterfaceLifecycle
                        ):

                            self.register(name.lower(), obj)
            except Exception as e:
                print(f"Failed to load plugin {file}: {e}")


class InterfaceBuilder:
    """
    Helper for building custom interfaces.

    Provides utilities for common interface operations.
    """

    @staticmethod
    def create_message(
        content: Any,
        sender_id: str,
        message_type: MessageType = MessageType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Create a message."""
        import secrets

        return Message(
            id=secrets.token_hex(8),
            type=message_type,
            content=content,
            sender_id=sender_id,
            metadata=metadata or {},
        )

    @staticmethod
    def format_text_message(text: str, **kwargs) -> str:
        """Format text message with template."""
        return text.format(**kwargs)

    @staticmethod
    def parse_command(text: str) -> Optional[Dict[str, Any]]:
        """Parse command from text."""
        if not text.startswith("/"):
            return None

        parts = text[1:].split()
        if not parts:
            return None

        return {"command": parts[0], "args": parts[1:] if len(parts) > 1 else []}

    @staticmethod
    async def validate_message(message: Message, schema: Dict[str, Any]) -> bool:
        """Validate message against schema."""
        # Simple validation
        if "type" in schema and message.type.value != schema["type"]:
            return False

        if "required_fields" in schema:
            for field in schema["required_fields"]:
                if field not in message.metadata:
                    return False

        return True


# Example usage
async def main():
    """Example custom interface SDK."""

    print("=" * 50)
    print("Custom Interface SDK Examples")
    print("=" * 50)

    # Interface Registry
    print("\n1. Interface Registry")
    registry = InterfaceRegistry()

    available = registry.list_interfaces()
    print(f"  Available interfaces: {', '.join(available)}")

    # Create WebSocket interface
    print("\n2. Create WebSocket Interface")
    ws_config = InterfaceConfig(
        name="websocket", type=InterfaceType.CHAT, settings={"host": "0.0.0.0", "port": 8765}
    )

    ws_interface = await registry.create_interface("websocket", ws_config)
    if ws_interface:
        print(f"  Created: {ws_interface.get_info()}")

        # Register message handler
        async def handle_message(msg: Message):
            print(f"  Received message: {msg.content}")

        ws_interface.register_message_handler(handle_message)

        # Send test message
        test_msg = InterfaceBuilder.create_message("Hello from WebSocket!", sender_id="system")
        await ws_interface.send_message(test_msg)

    # Create Webhook interface
    print("\n3. Create Webhook Interface")
    webhook_config = InterfaceConfig(
        name="webhook",
        type=InterfaceType.WEBHOOK,
        settings={"endpoint": "/webhook", "secret": "my-secret"},
    )

    webhook_interface = await registry.create_interface("webhook", webhook_config)
    if webhook_interface:
        print(f"  Created: {webhook_interface.get_info()}")

    # Create CLI interface
    print("\n4. Create CLI Interface")
    cli_config = InterfaceConfig(
        name="cli", type=InterfaceType.CHAT, settings={"prompt": "Open-Sable> "}
    )

    cli_interface = await registry.create_interface("cli", cli_config)
    if cli_interface:
        print(f"  Created: {cli_interface.get_info()}")

    # List active interfaces
    print("\n5. Active Interfaces")
    active = registry.list_active_interfaces()
    print(f"  Active: {', '.join(active)}")

    for name in active:
        interface = registry.get_interface(name)
        if interface:
            info = interface.get_info()
            print(f"    - {info['name']}: {info['type']} (running: {info['running']})")

    # Interface Builder
    print("\n6. Interface Builder")

    # Create message
    msg = InterfaceBuilder.create_message(
        "Test message",
        sender_id="user123",
        message_type=MessageType.TEXT,
        metadata={"priority": "high"},
    )
    print(f"  Created message: {msg.id}")

    # Parse command
    command = InterfaceBuilder.parse_command("/help search")
    if command:
        print(f"  Parsed command: {command['command']} with args {command['args']}")

    # Cleanup
    print("\n7. Cleanup")
    for name in active:
        await registry.destroy_interface(name)
    print("  All interfaces stopped")

    print("\n✅ Custom interface SDK examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
