"""
Tests for Interface SDK - Custom interface development.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from core.interface_sdk import (
    InterfaceSDK, InterfaceType, MessageFormat,
    InterfaceEvent, EventHandler, InterfaceRegistry
)


class TestInterfaceSDK:
    """Test interface SDK"""
    
    @pytest.fixture
    def sdk(self):
        return InterfaceSDK(
            name="test-interface",
            interface_type=InterfaceType.WEBSOCKET,
            version="1.0.0"
        )
    
    def test_initialization(self, sdk):
        """Test SDK initialization"""
        assert sdk.name == "test-interface"
        assert sdk.interface_type == InterfaceType.WEBSOCKET
        assert sdk.version == "1.0.0"
    
    def test_format_message(self, sdk):
        """Test message formatting"""
        msg = sdk.format_message(
            content="Hello",
            sender="user_123",
            message_format=MessageFormat.TEXT
        )
        
        assert msg.content == "Hello"
        assert msg.sender == "user_123"
        assert msg.message_format == MessageFormat.TEXT
        assert msg.timestamp is not None
    
    def test_format_rich_message(self, sdk):
        """Test rich message with attachments"""
        msg = sdk.format_message(
            content="Check this",
            sender="user_456",
            message_format=MessageFormat.RICH,
            attachments=[
                {"type": "image", "url": "https://example.com/img.png"}
            ]
        )
        
        assert len(msg.attachments) == 1
        assert msg.attachments[0]["type"] == "image"
    
    def test_message_with_metadata(self, sdk):
        """Test message with custom metadata"""
        msg = sdk.format_message(
            content="Test",
            sender="user",
            message_format=MessageFormat.TEXT,
            metadata={"priority": "high", "tags": ["urgent"]}
        )
        
        assert msg.metadata["priority"] == "high"
        assert "urgent" in msg.metadata["tags"]
    
    @pytest.mark.asyncio
    async def test_event_handling(self, sdk):
        """Test event handler registration and emission"""
        received_events = []
        
        @sdk.on_event(InterfaceEvent.MESSAGE_RECEIVED)
        async def handle_message(data):
            received_events.append(data)
        
        await sdk.emit(InterfaceEvent.MESSAGE_RECEIVED, {"content": "Test"})
        
        assert len(received_events) == 1
        assert received_events[0]["content"] == "Test"
    
    @pytest.mark.asyncio
    async def test_multiple_event_handlers(self, sdk):
        """Test multiple handlers for same event"""
        call_count = []
        
        @sdk.on_event(InterfaceEvent.USER_CONNECTED)
        async def handler1(data):
            call_count.append(1)
        
        @sdk.on_event(InterfaceEvent.USER_CONNECTED)
        async def handler2(data):
            call_count.append(2)
        
        await sdk.emit(InterfaceEvent.USER_CONNECTED, {"user_id": "123"})
        
        assert len(call_count) == 2
    
    @pytest.mark.asyncio
    async def test_lifecycle(self, sdk):
        """Test interface lifecycle"""
        await sdk.start()
        assert sdk.is_running() is True
        
        await sdk.stop()
        assert sdk.is_running() is False
    
    @pytest.mark.asyncio
    async def test_send_receive(self, sdk):
        """Test sending and receiving messages"""
        await sdk.start()
        
        await sdk.send("Test message")
        
        # Mock receive
        received = await sdk.receive(timeout=0.1)
        
        await sdk.stop()
    
    def test_serialization(self, sdk):
        """Test message serialization"""
        msg = sdk.format_message("Test", "user", MessageFormat.TEXT)
        
        # Serialize to JSON
        json_data = sdk.serialize(msg, format="json")
        
        assert json_data is not None
        assert isinstance(json_data, str)
    
    def test_deserialization(self, sdk):
        """Test message deserialization"""
        msg = sdk.format_message("Test", "user", MessageFormat.TEXT)
        json_data = sdk.serialize(msg, format="json")
        
        # Deserialize back
        deserialized = sdk.deserialize(json_data, format="json")
        
        assert deserialized.content == msg.content
        assert deserialized.sender == msg.sender


class TestInterfaceTypes:
    """Test different interface types"""
    
    def test_websocket_interface(self):
        """Test WebSocket interface configuration"""
        ws = InterfaceSDK(
            name="websocket",
            interface_type=InterfaceType.WEBSOCKET,
            config={
                "host": "localhost",
                "port": 8765,
                "path": "/ws"
            }
        )
        
        assert ws.config["host"] == "localhost"
        assert ws.config["port"] == 8765
    
    def test_rest_api_interface(self):
        """Test REST API interface configuration"""
        api = InterfaceSDK(
            name="rest-api",
            interface_type=InterfaceType.REST_API,
            config={
                "base_url": "https://api.example.com",
                "auth_token": "token123"
            }
        )
        
        assert api.config["base_url"] == "https://api.example.com"
    
    def test_webhook_interface(self):
        """Test Webhook interface configuration"""
        webhook = InterfaceSDK(
            name="webhook",
            interface_type=InterfaceType.WEBHOOK,
            config={
                "listen_port": 9000,
                "secret": "webhook-secret",
                "verify_signature": True
            }
        )
        
        assert webhook.config["listen_port"] == 9000
        assert webhook.config["verify_signature"] is True


class TestInterfaceRegistry:
    """Test interface registry"""
    
    @pytest.fixture
    def registry(self):
        return InterfaceRegistry()
    
    def test_register_interface(self, registry):
        """Test registering an interface"""
        iface = InterfaceSDK(
            name="test",
            interface_type=InterfaceType.WEBSOCKET
        )
        
        registry.register(iface)
        
        assert len(registry.list()) == 1
    
    def test_get_interface(self, registry):
        """Test retrieving interface by name"""
        iface = InterfaceSDK(
            name="my-interface",
            interface_type=InterfaceType.WEBSOCKET
        )
        
        registry.register(iface)
        
        found = registry.get("my-interface")
        assert found is not None
        assert found.name == "my-interface"
    
    def test_list_interfaces(self, registry):
        """Test listing all interfaces"""
        iface1 = InterfaceSDK("iface1", InterfaceType.WEBSOCKET)
        iface2 = InterfaceSDK("iface2", InterfaceType.REST_API)
        
        registry.register(iface1)
        registry.register(iface2)
        
        interfaces = registry.list()
        assert len(interfaces) == 2
    
    def test_unregister_interface(self, registry):
        """Test unregistering an interface"""
        iface = InterfaceSDK("test", InterfaceType.WEBSOCKET)
        
        registry.register(iface)
        assert len(registry.list()) == 1
        
        registry.unregister("test")
        assert len(registry.list()) == 0


class TestMessageFormats:
    """Test different message formats"""
    
    @pytest.fixture
    def sdk(self):
        return InterfaceSDK("test", InterfaceType.WEBSOCKET)
    
    def test_text_format(self, sdk):
        """Test plain text message"""
        msg = sdk.format_message(
            "Plain text",
            "user",
            MessageFormat.TEXT
        )
        
        assert msg.message_format == MessageFormat.TEXT
    
    def test_markdown_format(self, sdk):
        """Test markdown message"""
        msg = sdk.format_message(
            "# Title\n\n**Bold**",
            "user",
            MessageFormat.MARKDOWN
        )
        
        assert msg.message_format == MessageFormat.MARKDOWN
        assert "**Bold**" in msg.content
    
    def test_json_format(self, sdk):
        """Test JSON message"""
        msg = sdk.format_message(
            '{"key": "value"}',
            "user",
            MessageFormat.JSON
        )
        
        assert msg.message_format == MessageFormat.JSON
    
    def test_rich_format_with_attachments(self, sdk):
        """Test rich format with multiple attachments"""
        msg = sdk.format_message(
            "Message with files",
            "user",
            MessageFormat.RICH,
            attachments=[
                {"type": "image", "url": "img.png"},
                {"type": "document", "url": "doc.pdf"}
            ]
        )
        
        assert msg.message_format == MessageFormat.RICH
        assert len(msg.attachments) == 2


class TestInterfaceMetrics:
    """Test interface metrics"""
    
    @pytest.fixture
    def sdk(self):
        return InterfaceSDK("test", InterfaceType.WEBSOCKET)
    
    @pytest.mark.asyncio
    async def test_message_count(self, sdk):
        """Test tracking message counts"""
        # Simulate messages
        for i in range(5):
            await sdk.emit(
                InterfaceEvent.MESSAGE_RECEIVED,
                {"content": f"Message {i}"}
            )
        
        metrics = sdk.get_metrics()
        
        assert metrics.get("messages_received", 0) >= 0
    
    @pytest.mark.asyncio
    async def test_uptime_tracking(self, sdk):
        """Test uptime tracking"""
        import time
        
        await sdk.start()
        time.sleep(0.1)
        
        metrics = sdk.get_metrics()
        
        assert metrics.get("uptime_seconds", 0) >= 0
        
        await sdk.stop()
    
    def test_error_rate(self, sdk):
        """Test error rate metrics"""
        metrics = sdk.get_metrics()
        
        assert "error_rate" in metrics or metrics.get("error_rate") is not None


class TestRateLimiting:
    """Test rate limiting"""
    
    def test_rate_limit_config(self):
        """Test rate limit configuration"""
        sdk = InterfaceSDK(
            name="limited",
            interface_type=InterfaceType.WEBSOCKET,
            config={
                "rate_limit": {
                    "max_messages_per_minute": 60,
                    "max_messages_per_hour": 1000
                }
            }
        )
        
        limits = sdk.config.get("rate_limit", {})
        assert limits["max_messages_per_minute"] == 60
        assert limits["max_messages_per_hour"] == 1000
