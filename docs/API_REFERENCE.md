"""
Open-Sable API Documentation

Complete API reference for all Open-Sable components.
"""

# Open-Sable API Documentation

## Table of Contents

1. [Gateway API](#gateway-api)
2. [Session Manager API](#session-manager-api)
3. [Command Handler API](#command-handler-api)
4. [Multi-Agent Orchestration API](#multi-agent-api)
5. [Task Queue API](#task-queue-api)
6. [Cache API](#cache-api)
7. [Webhook API](#webhook-api)
8. [Plugin API](#plugin-api)
9. [Mobile REST API](#mobile-rest-api)
10. [Voice API](#voice-api)

---

## Gateway API

### WebSocket Connection

**Endpoint**: `ws://localhost:18789/ws`

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:18789/ws');

ws.onopen = () => {
    console.log('Connected to Open-Sable gateway');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

### RPC Methods

All WebSocket messages follow this format:
```json
{
    "method": "method_name",
    "params": {},
    "id": 1
}
```

Response format:
```json
{
    "jsonrpc": "2.0",
    "result": {},
    "id": 1
}
```

#### 1. send_message

Send message to agent and get response.

**Request**:
```json
{
    "method": "send_message",
    "params": {
        "session_id": "telegram_user123",
        "message": "Hello, how are you?",
        "channel": "telegram"
    },
    "id": 1
}
```

**Response**:
```json
{
    "jsonrpc": "2.0",
    "result": {
        "content": "I'm doing well, thank you! How can I help you today?",
        "tokens": 15,
        "session_id": "telegram_user123"
    },
    "id": 1
}
```

#### 2. create_session

Create new session.

**Request**:
```json
{
    "method": "create_session",
    "params": {
        "channel": "discord",
        "user_id": "user456"
    },
    "id": 2
}
```

**Response**:
```json
{
    "jsonrpc": "2.0",
    "result": {
        "session_id": "discord_user456_abc123",
        "channel": "discord",
        "user_id": "user456",
        "created_at": "2024-01-15T10:30:00Z"
    },
    "id": 2
}
```

#### 3. get_session

Retrieve session data.

**Request**:
```json
{
    "method": "get_session",
    "params": {
        "session_id": "telegram_user123"
    },
    "id": 3
}
```

**Response**:
```json
{
    "jsonrpc": "2.0",
    "result": {
        "session_id": "telegram_user123",
        "channel": "telegram",
        "user_id": "user123",
        "messages": [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": "2024-01-15T10:00:00Z"
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "timestamp": "2024-01-15T10:00:02Z"
            }
        ],
        "message_count": 2,
        "state": "active"
    },
    "id": 3
}
```

#### 4. list_sessions

List all active sessions.

**Request**:
```json
{
    "method": "list_sessions",
    "params": {},
    "id": 4
}
```

**Response**:
```json
{
    "jsonrpc": "2.0",
    "result": {
        "sessions": [
            {
                "session_id": "telegram_user123",
                "channel": "telegram",
                "user_id": "user123",
                "message_count": 5,
                "last_activity": "2024-01-15T10:30:00Z"
            },
            {
                "session_id": "discord_user456",
                "channel": "discord",
                "user_id": "user456",
                "message_count": 3,
                "last_activity": "2024-01-15T10:25:00Z"
            }
        ],
        "total": 2
    },
    "id": 4
}
```

#### 5. reset_session

Reset session (clear messages).

**Request**:
```json
{
    "method": "reset_session",
    "params": {
        "session_id": "telegram_user123"
    },
    "id": 5
}
```

**Response**:
```json
{
    "jsonrpc": "2.0",
    "result": {
        "success": true,
        "session_id": "telegram_user123",
        "message": "Session reset successfully"
    },
    "id": 5
}
```

#### 6. get_stats

Get gateway statistics.

**Request**:
```json
{
    "method": "get_stats",
    "params": {},
    "id": 6
}
```

**Response**:
```json
{
    "jsonrpc": "2.0",
    "result": {
        "active_sessions": 15,
        "total_messages": 342,
        "uptime_seconds": 86400,
        "connected_clients": 3,
        "channels": {
            "telegram": 8,
            "discord": 5,
            "whatsapp": 2
        }
    },
    "id": 6
}
```

---

## Session Manager API

### Python API

```python
from core.session_manager import SessionManager

# Initialize
session_manager = SessionManager()

# Create/get session
session = session_manager.get_or_create_session('telegram', 'user123')

# Add messages
session.add_message('user', 'Hello!')
session.add_message('assistant', 'Hi there!')

# Get messages
messages = session.get_messages()

# Reset session
session_manager.reset_session(session.session_id)

# Get all sessions
all_sessions = session_manager.get_all_sessions()

# Compact messages (summarize old messages)
session.compact_messages(keep_recent=10)
```

### Session Object

```python
@dataclass
class Session:
    session_id: str
    channel: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    messages: List[Message]
    config: SessionConfig
    state: str = 'active'
    metadata: Dict = field(default_factory=dict)
```

### Message Object

```python
@dataclass
class Message:
    role: str  # 'user', 'assistant', or 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tokens: int = 0
    metadata: Dict = field(default_factory=dict)
```

---

## Command Handler API

### Available Commands

```python
from core.commands import CommandHandler

handler = CommandHandler(session_manager)

# Execute command
result = handler.handle_command('/help', session)
```

### Command List

1. **/help** - Show available commands
2. **/status** - Show session status
3. **/reset** - Reset session (clear messages)
4. **/compact** - Compact old messages
5. **/think** - Toggle thinking mode
6. **/verbose** - Toggle verbose mode
7. **/voice** - Toggle voice mode
8. **/model <name>** - Change model
9. **/usage** - Show token usage
10. **/restart** - Restart conversation
11. **/activation <phrase>** - Set activation phrase

### CommandResult Object

```python
@dataclass
class CommandResult:
    success: bool
    message: str
    data: Optional[Dict] = None
    should_continue: bool = True  # Continue to agent?
```

---

## Multi-Agent API

### Python API

```python
from core.multi_agent import (
    MultiAgentOrchestrator,
    AgentRole,
    AgentTask,
    WorkflowBuilder
)

# Initialize orchestrator
orchestrator = MultiAgentOrchestrator(config)

# Delegate to single agent
result = await orchestrator.delegate_task(
    task_description="Analyze this data",
    role=AgentRole.ANALYST
)

# Execute workflow
task1 = AgentTask(
    task_id="1",
    role=AgentRole.RESEARCHER,
    description="Research topic",
    input_data={'topic': 'AI'}
)

task2 = AgentTask(
    task_id="2",
    role=AgentRole.WRITER,
    description="Write article",
    dependencies=["1"]
)

result = await orchestrator.execute_workflow([task1, task2])

# Use workflow builder
builder = WorkflowBuilder(orchestrator)

result = await builder.research_and_write(
    topic="Machine Learning",
    audience="beginners"
)
```

### Agent Roles

- **COORDINATOR** - Coordinates tasks between agents
- **RESEARCHER** - Gathers information and research
- **ANALYST** - Analyzes data and patterns
- **WRITER** - Creates written content
- **CODER** - Writes and reviews code
- **REVIEWER** - Reviews and validates work
- **EXECUTOR** - Executes tasks and actions

### Workflow Result

```python
{
    'success': True,
    'total_tasks': 3,
    'completed_tasks': 3,
    'failed_tasks': 0,
    'task_results': {
        '1': 'Research result...',
        '2': 'Analysis result...',
        '3': 'Final article...'
    },
    'final_result': 'Synthesized final output...',
    'execution_time': 45.2
}
```

---

## Task Queue API

### Python API

```python
from core.task_queue import TaskQueue, TaskPriority

# Initialize
queue = TaskQueue(config)

# Register handler
async def send_email(to, subject, body):
    # Send email
    return f"Sent to {to}"

queue.register_handler('send_email', send_email)

# Start queue
await queue.start()

# Enqueue tasks
task_id = await queue.enqueue(
    'send_email',
    'user@example.com',
    'Hello',
    'Message body',
    priority=TaskPriority.HIGH
)

# Get task status
task = await queue.get_task(task_id)
print(task.status)  # PENDING, RUNNING, COMPLETED, FAILED

# Wait for completion
await queue.wait_for_task(task_id)

# Stop queue
await queue.stop()
```

### Task Priority Levels

```python
class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
```

### Task Status

```python
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

---

## Cache API

### Python API

```python
from core.cache import MultiLayerCache

# Initialize
cache = MultiLayerCache(config)

# Set value
await cache.set('key', 'value', ttl=3600)

# Get value
value = await cache.get('key')

# Get or compute
def expensive_computation():
    return "computed value"

value = await cache.get_or_compute(
    'key',
    expensive_computation,
    ttl=3600
)

# Delete
await cache.delete('key')

# Clear all
await cache.clear()

# Get stats
stats = cache.get_stats()
# {
#     'hits': 100,
#     'misses': 20,
#     'hit_rate': 0.83,
#     'size': 50,
#     'utilization': 0.25
# }
```

### Decorator Usage

```python
from core.cache import cached

@cached(ttl=3600, key_prefix='myfunction')
async def my_expensive_function(arg1, arg2):
    # Expensive computation
    return result
```

---

## Webhook API

### Python API

```python
from core.webhooks import WebhookManager

# Initialize
webhook_manager = WebhookManager(config)
await webhook_manager.start()

# Register webhook
webhook_id = webhook_manager.register_webhook(
    url="https://example.com/webhook",
    events=["message.received", "session.created"],
    secret="my_secret_key"
)

# Emit event
await webhook_manager.emit('message.received', {
    'session_id': 'abc123',
    'user_id': 'user456',
    'message': 'Hello'
})

# Update webhook
webhook_manager.update_webhook(
    webhook_id,
    enabled=False
)

# Unregister
webhook_manager.unregister_webhook(webhook_id)

# Stop
await webhook_manager.stop()
```

### Webhook Events

- `message.received` - Message received from user
- `message.sent` - Message sent to user
- `session.created` - New session created
- `session.updated` - Session updated
- `command.executed` - Command executed
- `error.occurred` - Error occurred
- `agent.started` - Agent started
- `agent.stopped` - Agent stopped

### Webhook Payload

```json
{
    "event": "message.received",
    "timestamp": "2024-01-15T10:30:00Z",
    "data": {
        "session_id": "telegram_user123",
        "user_id": "user123",
        "message": "Hello!",
        "channel": "telegram"
    },
    "signature": "sha256=abcdef123456..."
}
```

### Signature Verification

```python
import hmac
import hashlib

def verify_signature(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(
        f"sha256={expected}",
        signature
    )
```

---

## Plugin API

### Creating a Plugin

```python
from core.plugins import Plugin, PluginMetadata

class MyPlugin(Plugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my-plugin",
            version="1.0.0",
            description="My awesome plugin",
            author="Your Name"
        )
    
    def initialize(self, config):
        """Initialize plugin"""
        print("Plugin initialized!")
    
    def cleanup(self):
        """Cleanup resources"""
        print("Plugin cleaned up!")
    
    def get_commands(self):
        """Register custom commands"""
        return {
            '/myplugin': self.handle_command
        }
    
    def get_hooks(self):
        """Register hooks"""
        return {
            'before_message': self.before_message_hook,
            'after_message': self.after_message_hook
        }
    
    def handle_command(self, message, session):
        """Handle custom command"""
        return "Plugin command executed!"
    
    def before_message_hook(self, message, session):
        """Called before processing message"""
        print(f"Processing: {message}")
        return message
    
    def after_message_hook(self, response, session):
        """Called after generating response"""
        print(f"Generated: {response}")
        return response
```

### Plugin Manager

```python
from core.plugins import PluginManager

# Initialize
plugin_manager = PluginManager(config)

# Discover plugins
plugin_manager.discover_plugins()

# Load plugin
plugin_manager.load_plugin("my-plugin")

# Reload plugin (hot-reload)
plugin_manager.reload_plugin("my-plugin")

# Unload plugin
plugin_manager.unload_plugin("my-plugin")

# Get loaded plugins
plugins = plugin_manager.get_plugins()
```

---

## Mobile REST API

### Authentication

#### POST /auth/login

Login and get JWT tokens.

**Request**:
```json
{
    "username": "user123",
    "password": "secure_password"
}
```

**Response**:
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "bearer",
    "expires_in": 3600
}
```

#### POST /auth/refresh

Refresh access token.

**Request**:
```json
{
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**Response**:
```json
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "bearer",
    "expires_in": 3600
}
```

### Messages

#### POST /messages

Send message to agent.

**Headers**:
```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

**Request**:
```json
{
    "message": "Hello, how are you?",
    "session_id": "mobile_user123"
}
```

**Response**:
```json
{
    "response": "I'm doing well, thank you! How can I help you today?",
    "session_id": "mobile_user123",
    "tokens": 15,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

#### GET /messages/{session_id}

Get message history.

**Response**:
```json
{
    "session_id": "mobile_user123",
    "messages": [
        {
            "role": "user",
            "content": "Hello",
            "timestamp": "2024-01-15T10:00:00Z"
        },
        {
            "role": "assistant",
            "content": "Hi there!",
            "timestamp": "2024-01-15T10:00:02Z"
        }
    ],
    "total": 2
}
```

### Sessions

#### GET /sessions

Get all user sessions.

**Response**:
```json
{
    "sessions": [
        {
            "session_id": "mobile_user123_1",
            "created_at": "2024-01-15T09:00:00Z",
            "last_activity": "2024-01-15T10:30:00Z",
            "message_count": 15
        }
    ],
    "total": 1
}
```

#### DELETE /sessions/{session_id}

Delete session.

**Response**:
```json
{
    "success": true,
    "message": "Session deleted successfully"
}
```

### WebSocket

#### ws://localhost:8000/ws

Real-time messaging WebSocket.

**Connect with JWT**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws?token=' + accessToken);

// Send message
ws.send(JSON.stringify({
    type: 'message',
    content: 'Hello!',
    session_id: 'mobile_user123'
}));

// Receive message
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Response:', data.content);
};
```

---

## Voice API

### Text-to-Speech

```python
from core.voice import VoiceManager

voice_manager = VoiceManager(config)

# Generate speech (returns audio file path)
audio_file = voice_manager.text_to_speech(
    "Hello, how can I help you?",
    engine='pyttsx3'  # or 'elevenlabs', 'openai'
)

# Async version
audio_file = await voice_manager.text_to_speech_async(
    "Hello world!",
    engine='elevenlabs',
    voice_id='rachel'  # ElevenLabs voice
)
```

### Speech-to-Text

```python
# Transcribe audio file
text = voice_manager.speech_to_text(
    audio_file="recording.wav",
    engine='whisper_local'  # or 'openai'
)

# Async version
text = await voice_manager.speech_to_text_async(
    audio_file="recording.wav",
    engine='openai',
    language='en'
)
```

### TTS Engines

- **pyttsx3** - Local TTS (offline, fast)
- **elevenlabs** - High-quality cloud TTS
- **openai** - OpenAI TTS (natural voices)

### STT Engines

- **whisper_local** - Local Whisper model (offline)
- **openai** - OpenAI Whisper API (cloud)

---

## Error Handling

### Error Response Format

```json
{
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "Too many requests. Please wait before trying again.",
        "details": {
            "retry_after": 60
        }
    }
}
```

### Error Codes

- `INVALID_REQUEST` - Malformed request
- `UNAUTHORIZED` - Authentication required
- `FORBIDDEN` - Insufficient permissions
- `NOT_FOUND` - Resource not found
- `RATE_LIMIT_EXCEEDED` - Rate limit exceeded
- `INTERNAL_ERROR` - Internal server error
- `SERVICE_UNAVAILABLE` - Service temporarily unavailable

---

## Rate Limiting

### Headers

Responses include rate limit information:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642252800
```

### Limits

- **Global**: 10,000 requests/hour
- **Per-user**: 60 messages/minute
- **Websocket**: 120 messages/minute

---

## Best Practices

1. **Authentication**: Always use HTTPS in production
2. **Rate Limits**: Implement exponential backoff
3. **Webhooks**: Verify signatures
4. **Errors**: Handle gracefully with retries
5. **Sessions**: Reuse sessions for conversations
6. **Caching**: Use cache for expensive operations
7. **Monitoring**: Track API usage and errors

---

For more information, visit: https://github.com/nexland/Open-Sable
