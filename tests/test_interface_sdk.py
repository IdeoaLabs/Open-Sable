"""
Tests for Interface SDK - Custom chat interface framework.
"""

import pytest
from opensable.core.interface_sdk import (
    InterfaceType,
    MessageType,
    InterfaceConfig,
    Message,
    InterfaceLifecycle,
    InterfaceRegistry,
    InterfaceBuilder,
    WebSocketInterface,
    HTTPWebhookInterface,
    CustomCLIInterface,
)


class TestInterfaceType:
    """Test InterfaceType enum."""

    def test_enum_members(self):
        assert InterfaceType.CHAT.value == "chat"
        assert InterfaceType.VOICE.value == "voice"
        assert InterfaceType.API.value == "api"
        assert InterfaceType.WEBHOOK.value == "webhook"
        assert InterfaceType.CUSTOM.value == "custom"


class TestMessageType:
    """Test MessageType enum."""

    def test_enum_members(self):
        assert MessageType.TEXT.value == "text"
        assert MessageType.IMAGE.value == "image"
        assert MessageType.FILE.value == "file"
        assert MessageType.COMMAND.value == "command"


class TestInterfaceConfig:
    """Test InterfaceConfig dataclass."""

    def test_required_fields(self):
        cfg = InterfaceConfig(name="test", type=InterfaceType.CHAT)
        assert cfg.name == "test"
        assert cfg.type == InterfaceType.CHAT
        assert cfg.enabled is True
        assert cfg.settings == {}

    def test_custom_settings(self):
        cfg = InterfaceConfig(
            name="ws",
            type=InterfaceType.CHAT,
            enabled=False,
            settings={"host": "0.0.0.0", "port": 8765},
        )
        assert cfg.enabled is False
        assert cfg.settings["port"] == 8765

    def test_to_dict(self):
        cfg = InterfaceConfig(name="x", type=InterfaceType.API)
        d = cfg.to_dict()
        assert d["name"] == "x"
        assert d["type"] == "api"


class TestMessage:
    """Test Message dataclass."""

    def test_create(self):
        msg = Message(id="m1", type=MessageType.TEXT, content="hello", sender_id="user1")
        assert msg.id == "m1"
        assert msg.content == "hello"
        assert msg.sender_id == "user1"

    def test_to_dict(self):
        msg = Message(id="m2", type=MessageType.IMAGE, content="img.png", sender_id="bot")
        d = msg.to_dict()
        assert d["id"] == "m2"
        assert d["type"] == "image"
        assert d["sender_id"] == "bot"

    def test_metadata(self):
        msg = Message(
            id="m3", type=MessageType.TEXT, content="hi", sender_id="u", metadata={"key": "val"}
        )
        assert msg.metadata["key"] == "val"


class TestInterfaceRegistry:
    """Test interface registry."""

    @pytest.fixture
    def registry(self):
        return InterfaceRegistry()

    def test_builtin_interfaces(self, registry):
        names = registry.list_interfaces()
        assert "websocket" in names
        assert "webhook" in names
        assert "cli" in names

    def test_register_custom(self, registry):
        class Dummy(InterfaceLifecycle):
            async def start(self):
                pass

            async def stop(self):
                pass

            async def send_message(self, m):
                return True

            async def receive_message(self):
                return None

        registry.register("dummy", Dummy)
        assert "dummy" in registry.list_interfaces()

    def test_unregister(self, registry):
        registry.unregister("cli")
        assert "cli" not in registry.list_interfaces()

    @pytest.mark.asyncio
    async def test_create_and_destroy(self, registry):
        cfg = InterfaceConfig(name="ws", type=InterfaceType.CHAT, settings={"port": 9999})
        iface = await registry.create_interface("websocket", cfg)
        assert iface is not None
        assert iface.is_running is True
        assert "websocket" in registry.list_active_interfaces()

        await registry.destroy_interface("websocket")
        assert "websocket" not in registry.list_active_interfaces()

    def test_get_interface_none(self, registry):
        assert registry.get_interface("nonexistent") is None


class TestWebSocketInterface:
    """Test WebSocket interface."""

    @pytest.fixture
    def ws(self):
        cfg = InterfaceConfig(
            name="ws", type=InterfaceType.CHAT, settings={"host": "127.0.0.1", "port": 9000}
        )
        return WebSocketInterface(cfg)

    @pytest.mark.asyncio
    async def test_start_stop(self, ws):
        await ws.start()
        assert ws.is_running is True
        await ws.stop()
        assert ws.is_running is False

    @pytest.mark.asyncio
    async def test_send_when_stopped(self, ws):
        msg = Message(id="m", type=MessageType.TEXT, content="x", sender_id="u")
        result = await ws.send_message(msg)
        assert result is False

    def test_host_port(self, ws):
        assert ws.host == "127.0.0.1"
        assert ws.port == 9000


class TestHTTPWebhookInterface:
    """Test HTTP webhook interface."""

    def test_init(self):
        cfg = InterfaceConfig(
            name="wh", type=InterfaceType.WEBHOOK, settings={"endpoint": "/hook", "secret": "s"}
        )
        wh = HTTPWebhookInterface(cfg)
        assert wh.endpoint == "/hook"
        assert wh.secret == "s"


class TestCustomCLIInterface:
    """Test CLI interface."""

    def test_init(self):
        cfg = InterfaceConfig(name="cli", type=InterfaceType.CHAT, settings={"prompt": ">> "})
        cli = CustomCLIInterface(cfg)
        assert cli.prompt == ">> "


class TestInterfaceBuilder:
    """Test interface builder utilities."""

    def test_create_message(self):
        msg = InterfaceBuilder.create_message("hello", sender_id="bot")
        assert isinstance(msg, Message)
        assert msg.content == "hello"
        assert msg.sender_id == "bot"
        assert msg.type == MessageType.TEXT

    def test_parse_command(self):
        cmd = InterfaceBuilder.parse_command("/help search")
        assert cmd is not None
        assert cmd["command"] == "help"
        assert cmd["args"] == ["search"]

    def test_parse_non_command(self):
        assert InterfaceBuilder.parse_command("hello") is None

    def test_format_text(self):
        result = InterfaceBuilder.format_text_message("Hello {name}", name="World")
        assert result == "Hello World"
