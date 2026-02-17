# Self-Modification & Advanced Features

OpenSable now includes advanced capabilities that rival and surpass commercial AI agent frameworks.

## 🔁 Heartbeats (Proactive Checking)

Sable runs periodic health checks every 30 minutes (configurable) to proactively detect issues and notify you.

### Features

- **Customizable Checks**: Define what to monitor via `~/.opensable/HEARTBEAT.md`
- **Active Hours**: Only runs checks during configured hours (default: 08:00-23:00)
- **Smart Alerts**: Only notifies when something needs attention
- **Built-in Checks**:
  - System health (CPU, RAM, disk)
  - Pending tasks and calendar events
  - Idle time detection (8+ hours)

---

## 🛠️ Dynamic Skill Creation

Sable can create new skills/extensions on-the-fly through natural language commands.

### Features

- **Natural Language**: "Create a skill that checks BTC price"
- **Auto-Validation**: Syntax and security checks via AST parsing
- **Hot-Loading**: Skills are loaded immediately without restart
- **Security**: Blocks dangerous operations (eval, os.system, etc.)
- **Persistence**: Skills saved to `~/.opensable/dynamic_skills/`

---

## 🎛️ Inline Buttons (Telegram)

Interactive inline buttons for quick actions without typing.

### Features

- **Scope Control**: Buttons can be enabled per-chat or globally
- **Security**: 64-char callback data limit (Telegram requirement)
- **Action Types**: Execute commands, search, or custom actions
- **Context Preservation**: Button clicks maintain conversation history

---

## 🎙️ Voice Integration

Full voice message support with transcription and synthesis.

### Features

- **Voice Messages**: Telegram voice → Whisper transcription
- **TTS Responses**: Optional voice replies (enable with `/voice on`)
- **Multi-Provider**: Local (pyttsx3, Whisper) or cloud (OpenAI, ElevenLabs)
- **Automatic Processing**: Just send voice, get text + response

### Usage

Send voice message → Sable transcribes → processes → responds in text/voice

```
User: [🎙️ Voice message]
Sable: 🎙️ *You said:* What's the weather?

Weather in London: 18°C, partly cloudy

🔊 [Voice response if enabled]
```

---

## 🖼️ Image Analysis

Vision capabilities with local model support.

### Features

- **Image Description**: Automatic captioning with LLaVA
- **Visual Q&A**: Ask questions about images
- **OCR**: Extract text from photos (Tesseract)
- **Object Detection**: Identify objects in images (planned)

### Usage

Send photo with optional caption → Sable analyzes → describes content

```
User: [📷 Photo of restaurant menu]
Sable: 🖼️ **Image Analysis:**

**Description:** This is a restaurant menu featuring Italian cuisine...

📝 **Text detected:**
PIZZA MARGHERITA - $12
PASTA CARBONARA - $14
...
```

---

## 📱 Multi-Messenger Router

Unified message routing across platforms.

### Architecture

```
telegram_bot.py ──┐
whatsapp_bot.py ──┼─→ MultiMessengerRouter ─→ agent.process_message()
discord_bot.py  ──┘
```

### Features

- **Platform-Agnostic**: Unified `UnifiedMessage` format
- **Smart Formatting**: Platform-specific markdown/buttons
- **Broadcast**: Send to all platforms at once
- **Preprocessors**: Normalize platform differences

---

## 📊 Statistics

Track all implemented features:

### Configuration

Edit `~/.opensable/HEARTBEAT.md` to customize checks. Example:

```markdown
# Heartbeat Checklist

## System Health
- [ ] CPU usage > 90%
- [ ] RAM usage > 80%
- [ ] Disk space < 10GB

## Tasks
- [ ] Calendar events in next 2 hours
- [ ] High-priority tasks
```

### Usage

Heartbeats start automatically when you run the Telegram bot. To customize:

```python
from opensable.core.heartbeat import HeartbeatManager

heartbeat = HeartbeatManager(agent, config)
heartbeat.interval = 1800  # 30 minutes
heartbeat.active_hours_start = "08:00"
heartbeat.active_hours_end = "23:00"

await heartbeat.start()
```

### Architecture

Based on the proactive checking pattern used by advanced AI agents:
- Main loop runs at configured interval
- Executes all registered checks
- Aggregates alerts and sends single notification
- Logs "HEARTBEAT_OK" when everything is normal

---

## 🛠️ Dynamic Skill Creation

Sable can now create new skills/extensions on-the-fly through natural language commands.

### Features

- **Natural Language**: "Create a skill that checks BTC price"
- **Auto-Validation**: Syntax and security checks via AST parsing
- **Hot-Loading**: Skills are loaded immediately without restart
- **Security**: Blocks dangerous operations (eval, os.system, etc.)
- **Persistence**: Skills saved to `~/.opensable/dynamic_skills/`

### Usage

Via Telegram:

```
User: Create a skill called 'bitcoin_price' that fetches BTC/USD from CoinGecko API

Sable: [Uses create_skill tool]
✅ Skill 'bitcoin_price' created successfully!

Path: ~/.opensable/dynamic_skills/bitcoin_price.py
```

Via Python:

```python
from opensable.core.skill_creator import SkillCreator

creator = SkillCreator(config)

result = await creator.create_skill(
    name="weather_check",
    description="Check weather using wttr.in API",
    code="""
import aiohttp

async def execute(location):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://wttr.in/{location}?format=3") as resp:
            return await resp.text()
""",
    metadata={"author": "sable", "version": "1.0"}
)
```

### Skill Registry

Skills are tracked in `~/.opensable/dynamic_skills/registry.json`:

```json
[
  {
    "name": "bitcoin_price",
    "description": "Fetch BTC price from CoinGecko",
    "enabled": true,
    "path": "/home/user/.opensable/dynamic_skills/bitcoin_price.py",
    "metadata": {
      "author": "sable",
      "created_at": "2024-01-15T10:30:00"
    }
  }
]
```

### Security

All skill code is validated before execution:
1. **Syntax Check**: AST parsing ensures valid Python
2. **Security Scan**: Blocks:
   - `eval()`, `exec()`, `compile()`
   - `os.system()`, `subprocess.run(shell=True)`
   - `__import__`, `importlib` misuse
3. **Sandboxing**: Skills run in restricted context

### Tools Available

- `create_skill(name, description, code, author)` - Create new skill
- `list_skills()` - Show all custom skills
- `enable_skill(name)` - Enable disabled skill
- `disable_skill(name)` - Temporarily disable skill
- `delete_skill(name)` - Permanently remove skill

---

## 🎛️ Inline Buttons (Telegram)

Interactive inline buttons for quick actions without typing.

### Features

- **Scope Control**: Buttons can be enabled per-chat or globally
- **Security**: 64-char callback data limit (Telegram requirement)
- **Action Types**: Execute commands, search, or custom actions
- **Context Preservation**: Button clicks maintain conversation history

### Format

```python
buttons = [
    [
        {"text": "📊 Status", "callback_data": "cmd:/status"},
        {"text": "🔄 Reset", "callback_data": "cmd:/reset"}
    ],
    [
        {"text": "🌐 Search", "callback_data": "cmd:search latest news"}
    ]
]

await bot._safe_reply(message, "Choose an action:", buttons=buttons)
```

### Callback Data Format

Use `prefix:action` pattern:
- `cmd:/status` - Execute /status command
- `cmd:search AI news` - Execute "search AI news"
- Custom prefixes supported via handler extension

### Example

When user clicks "🌐 Search" button:
1. Telegram sends callback query with data `"cmd:search latest news"`
2. Bot parses prefix and action
3. Processes action through agent (like normal message)
4. Sends response and closes button

### In /help Command

The `/help` command now includes example buttons:

```
📊 Status | 🔄 Reset
    🌐 Search Web
```

---

## Architecture Overview

### Heartbeat System

```
HeartbeatManager
├── _heartbeat_loop()          # Main async loop
├── _is_within_active_hours()  # Time gating
├── _run_heartbeat()            # Execute checks
├── _read_heartbeat_file()     # Parse HEARTBEAT.md
└── _send_heartbeat_alerts()   # Notify user

Built-in Checks:
- check_system_health()
- check_pending_tasks()
- check_idle_time()
```

### Skill Creation

```
SkillCreator
├── create_skill()      # Main entry point
├── _validate_syntax()  # AST parsing
├── _check_security()   # Regex-based scan
├── list_skills()       # Show all
├── enable_skill()      # Enable by name
├── disable_skill()     # Disable by name
└── delete_skill()      # Remove permanently

Storage:
~/.opensable/dynamic_skills/
├── registry.json
├── bitcoin_price.py
├── weather_check.py
└── custom_skill.py
```

### Inline Buttons

```
telegram_bot.py
├── _build_inline_keyboard()  # Construct markup
├── _h_callback()             # Handle clicks
└── _safe_reply()             # Send with buttons

Telegram Flow:
1. Send message with inline_keyboard
2. User clicks button
3. Telegram sends callback_query
4. Bot processes action
5. Bot responds and acknowledges
```

---

## Comparison with Other Agents

| Feature | OpenSable | Other Agents |
|---------|-----------|--------|
| Proactive Checks | ✅ Heartbeats | ❌ Most passive |
| Self-Modification | ✅ Dynamic skills | ⚠️ Limited |
| Local-First | ✅ 100% Ollama | ⚠️ Mixed |
| Multi-Messenger | ✅ Telegram, planned WhatsApp/Discord | ❌ Usually single |
| Inline Buttons | ✅ Full support | ⚠️ Limited |
| Persistent Memory | ✅ Session-based | ⚠️ Token-limited |

---

## Testing

### Test Heartbeat

1. Create `~/.opensable/HEARTBEAT.md` with checks
2. Start bot: `python -m opensable telegram`
3. Wait 30 minutes or modify interval for testing
4. Check logs for "💓 Running heartbeat check..."
5. Trigger alert (e.g., fill disk to 95%)
6. Verify proactive notification

### Test Skill Creation

Via Telegram:

```
You: Create a skill called 'btc_price' that uses coingecko API

Sable: ✅ Skill 'btc_price' created successfully!

You: list skills

Sable: 📦 Custom Skills (1):
• btc_price - Fetch BTC price from CoinGecko
  Status: ✅ Enabled
  Author: sable
```

### Test Inline Buttons

```
You: /help

Sable: [Shows help text with buttons]
📊 Status | 🔄 Reset
    🌐 Search Web

[Click "📊 Status"]

Sable: Session: abc123... · 42 messages · 2.5 h old
Model: llama3.1:8b
```

---

## Roadmap

- [ ] Multi-messenger support (WhatsApp, Discord, Signal)
- [ ] Advanced memory categorization (long-term, episodic, semantic)
- [ ] Voice/multimodal integration
- [ ] Skills marketplace (share/download community skills)
- [ ] Docker orchestration for multi-agent deployments
- [ ] Mobile app relay
