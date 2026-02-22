"""
Interface SDK Examples - Custom interface development.

Demonstrates creating custom chat interfaces, event handling, and message routing.
"""

import asyncio
from opensable.core.interface_sdk import (
    InterfaceSDK,
    InterfaceType,
    MessageFormat,
    InterfaceEvent,
    InterfaceRegistry,
)


async def main():
    """Run interface SDK examples."""

    print("=" * 60)
    print("Interface SDK Examples")
    print("=" * 60)

    # Example 1: Interface SDK initialization
    print("\n1. Interface SDK Initialization")
    print("-" * 40)

    sdk = InterfaceSDK(
        name="custom-chat-interface", interface_type=InterfaceType.WEBSOCKET, version="1.0.0"
    )

    print("Initialized SDK:")
    print(f"  Name: {sdk.name}")
    print(f"  Type: {sdk.interface_type.value}")
    print(f"  Version: {sdk.version}")

    # Example 2: Message formatting
    print("\n2. Message Formatting")
    print("-" * 40)

    # Format user message
    user_msg = sdk.format_message(
        content="Hello, Open-Sable!", sender="user_123", message_format=MessageFormat.TEXT
    )

    print("User message:")
    print(f"  Content: {user_msg.content}")
    print(f"  Sender: {user_msg.sender}")
    print(f"  Format: {user_msg.message_format.value}")
    print(f"  Timestamp: {user_msg.timestamp}")

    # Format assistant message
    assistant_msg = sdk.format_message(
        content="Hello! How can I help you today?",
        sender="assistant",
        message_format=MessageFormat.TEXT,
        metadata={"model": "gpt-4", "tokens": 12},
    )

    print("\nAssistant message:")
    print(f"  Content: {assistant_msg.content}")
    print(f"  Metadata: {assistant_msg.metadata}")

    # Example 3: Event handling
    print("\n3. Event Handling")
    print("-" * 40)

    events_received = []

    # Define event handlers
    @sdk.on_event(InterfaceEvent.MESSAGE_RECEIVED)
    async def handle_message(data):
        events_received.append(("message", data))
        print(f"  📨 Message received: {data.get('content', '')[:50]}")

    @sdk.on_event(InterfaceEvent.USER_CONNECTED)
    async def handle_connect(data):
        events_received.append(("connect", data))
        print(f"  ✅ User connected: {data.get('user_id')}")

    @sdk.on_event(InterfaceEvent.USER_DISCONNECTED)
    async def handle_disconnect(data):
        events_received.append(("disconnect", data))
        print(f"  ❌ User disconnected: {data.get('user_id')}")

    @sdk.on_event(InterfaceEvent.ERROR)
    async def handle_error(data):
        events_received.append(("error", data))
        print(f"  ⚠️  Error: {data.get('error')}")

    print("Registered event handlers:")
    print("  - MESSAGE_RECEIVED")
    print("  - USER_CONNECTED")
    print("  - USER_DISCONNECTED")
    print("  - ERROR")

    # Example 4: Emit events
    print("\n4. Emit Events")
    print("-" * 40)

    # Simulate events
    await sdk.emit(InterfaceEvent.USER_CONNECTED, {"user_id": "user_123", "ip": "192.168.1.100"})
    await sdk.emit(
        InterfaceEvent.MESSAGE_RECEIVED, {"content": "Test message", "sender": "user_123"}
    )
    await sdk.emit(InterfaceEvent.USER_DISCONNECTED, {"user_id": "user_123", "reason": "timeout"})

    print(f"\nTotal events processed: {len(events_received)}")

    # Example 5: Interface lifecycle
    print("\n5. Interface Lifecycle")
    print("-" * 40)

    # Start interface
    await sdk.start()
    print("✅ Interface started")

    # Check status
    is_running = sdk.is_running()
    print(f"Running: {is_running}")

    # Send message
    await sdk.send("Hello from interface!")
    print("📤 Sent message")

    # Receive message (simulated)
    received = await sdk.receive(timeout=1.0)
    if received:
        print(f"📥 Received: {received}")

    # Stop interface
    await sdk.stop()
    print("⛔ Interface stopped")

    # Example 6: WebSocket interface
    print("\n6. WebSocket Interface")
    print("-" * 40)

    ws_interface = InterfaceSDK(
        name="websocket-chat",
        interface_type=InterfaceType.WEBSOCKET,
        config={"host": "localhost", "port": 8765, "path": "/ws/chat"},
    )

    print("WebSocket configuration:")
    print(f"  Host: {ws_interface.config.get('host')}")
    print(f"  Port: {ws_interface.config.get('port')}")
    print(f"  Path: {ws_interface.config.get('path')}")

    # Example 7: REST API interface
    print("\n7. REST API Interface")
    print("-" * 40)

    api_interface = InterfaceSDK(
        name="rest-api",
        interface_type=InterfaceType.REST_API,
        config={
            "base_url": "https://api.example.com",
            "auth_token": "sk-test-1234567890",
            "endpoints": {"send": "/api/v1/messages", "receive": "/api/v1/messages/{id}"},
        },
    )

    print("REST API configuration:")
    print(f"  Base URL: {api_interface.config.get('base_url')}")
    print(f"  Endpoints: {list(api_interface.config.get('endpoints', {}).keys())}")

    # Example 8: Webhook interface
    print("\n8. Webhook Interface")
    print("-" * 40)

    webhook_interface = InterfaceSDK(
        name="webhook-receiver",
        interface_type=InterfaceType.WEBHOOK,
        config={"listen_port": 9000, "secret": "webhook-secret-key", "verify_signature": True},
    )

    print("Webhook configuration:")
    print(f"  Port: {webhook_interface.config.get('listen_port')}")
    print(f"  Signature verification: {webhook_interface.config.get('verify_signature')}")

    # Example 9: Interface registry
    print("\n9. Interface Registry")
    print("-" * 40)

    registry = InterfaceRegistry()

    # Register interfaces
    registry.register(ws_interface)
    registry.register(api_interface)
    registry.register(webhook_interface)

    print(f"Registered {len(registry.list())} interfaces:")
    for iface in registry.list():
        print(f"  - {iface.name} ({iface.interface_type.value})")

    # Get interface by name
    found = registry.get("websocket-chat")
    if found:
        print(f"\nFound interface: {found.name}")

    # Example 10: Message serialization
    print("\n10. Message Serialization")
    print("-" * 40)

    # Serialize to JSON
    msg = sdk.format_message("Test content", "user_1", MessageFormat.TEXT)
    json_data = sdk.serialize(msg, format="json")
    print("JSON serialization:")
    print(f"  {json_data[:100]}...")

    # Deserialize
    deserialized = sdk.deserialize(json_data, format="json")
    print("\nDeserialized:")
    print(f"  Content: {deserialized.content}")
    print(f"  Sender: {deserialized.sender}")

    # Example 11: Custom message types
    print("\n11. Custom Message Types")
    print("-" * 40)

    # Rich message with attachments
    rich_msg = sdk.format_message(
        content="Check out these files",
        sender="user_456",
        message_format=MessageFormat.RICH,
        attachments=[
            {"type": "image", "url": "https://example.com/image.png"},
            {"type": "document", "url": "https://example.com/report.pdf"},
        ],
        metadata={"priority": "high", "tags": ["important", "files"]},
    )

    print("Rich message:")
    print(f"  Attachments: {len(rich_msg.attachments)}")
    print(f"  Metadata tags: {rich_msg.metadata.get('tags')}")

    # Markdown message
    markdown_msg = sdk.format_message(
        content="# Title\n\n**Bold text** and *italic*\n\n- List item 1\n- List item 2",
        sender="assistant",
        message_format=MessageFormat.MARKDOWN,
    )

    print("\nMarkdown message:")
    print(f"  Format: {markdown_msg.message_format.value}")
    print(f"  Content length: {len(markdown_msg.content)} chars")

    # Example 12: Interface metrics
    print("\n12. Interface Metrics")
    print("-" * 40)

    # Simulate activity
    for i in range(10):
        await sdk.emit(
            InterfaceEvent.MESSAGE_RECEIVED, {"content": f"Message {i}", "sender": f"user_{i % 3}"}
        )

    metrics = sdk.get_metrics()

    print("Interface metrics:")
    print(f"  Messages received: {metrics.get('messages_received', 0)}")
    print(f"  Messages sent: {metrics.get('messages_sent', 0)}")
    print(f"  Active users: {metrics.get('active_users', 0)}")
    print(f"  Uptime: {metrics.get('uptime_seconds', 0):.2f}s")
    print(f"  Error rate: {metrics.get('error_rate', 0):.2%}")

    # Example 13: Rate limiting
    print("\n13. Rate Limiting")
    print("-" * 40)

    rate_limited_sdk = InterfaceSDK(
        name="rate-limited-chat",
        interface_type=InterfaceType.WEBSOCKET,
        config={"rate_limit": {"max_messages_per_minute": 60, "max_messages_per_hour": 1000}},
    )

    print("Rate limits configured:")
    limits = rate_limited_sdk.config.get("rate_limit", {})
    print(f"  Per minute: {limits.get('max_messages_per_minute')}")
    print(f"  Per hour: {limits.get('max_messages_per_hour')}")

    # Example 14: Interface plugin
    print("\n14. Custom Interface Plugin")
    print("-" * 40)

    class SlackInterfacePlugin:
        """Example plugin for Slack integration"""

        def __init__(self, bot_token, signing_secret):
            self.bot_token = bot_token
            self.signing_secret = signing_secret

        async def send_message(self, channel, text):
            print(f"  [Slack] Sending to #{channel}: {text[:50]}...")
            return {"ok": True, "ts": "1234567890.123456"}

        async def receive_event(self, event_data):
            print(f"  [Slack] Event received: {event_data.get('type')}")
            return event_data

    # Initialize plugin
    slack = SlackInterfacePlugin(
        bot_token="xoxb-slack-token", signing_secret="slack-signing-secret"
    )

    # Use plugin
    await slack.send_message("general", "Hello from Open-Sable!")
    await slack.receive_event({"type": "message", "text": "Hi!"})

    print("\nCustom Slack plugin initialized")

    print("\n" + "=" * 60)
    print("✅ Interface SDK examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
