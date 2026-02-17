# 🤖 Telegram Userbot Setup Guide

## What is a Userbot?

A **Telegram Userbot** allows Open-Sable to run as your own Telegram account (not a bot). This means:

- ✅ Can respond in any chat (groups, DMs, channels)
- ✅ Auto-respond when someone mentions you
- ✅ Use commands in any conversation (`.ask`, `.email`, etc.)
- ✅ No need for special permissions or adding a bot
- ⚠️ Uses your personal Telegram account

## Setup Instructions

### Step 1: Get Telegram API Credentials

1. Go to [my.telegram.org](https://my.telegram.org)
2. Log in with your phone number
3. Click "API development tools"
4. Create a new application:
   - App title: `Open-Sable`
   - Short name: `opensable`
   - Platform: `Other`
5. Copy your **API ID** and **API Hash**

### Step 2: Configure Open-Sable

Edit `.env` file:

```bash
# Enable userbot
TELEGRAM_USERBOT_ENABLED=true

# Add your credentials
TELEGRAM_API_ID=12345678
TELEGRAM_API_HASH=abcdef1234567890abcdef1234567890
TELEGRAM_PHONE_NUMBER=+1234567890

# Session name (for saving login)
TELEGRAM_SESSION_NAME=opensable_session

# Auto-respond when mentioned
USERBOT_AUTO_RESPOND=true
```

### Step 3: First Run

When you first start Open-Sable with userbot enabled:

```bash
python3 main.py
```

It will ask for:
1. **Phone number** (already in .env)
2. **Login code** (sent to your Telegram)
3. **2FA password** (if you have 2FA enabled)

After first login, it saves the session so you won't need to log in again.

## Userbot Commands

All commands start with `.` (dot):

### Basic Commands
```
.ask <question>      - Ask the AI
.email               - Check your emails
.calendar            - Check your calendar
.search <query>      - Web search
.status              - System status
.help                - Show help
```

### Examples

**In any chat:**
```
You: .ask what's the weather like?
Sable: [responds with weather info]

You: .email
Sable: You have 2 unread emails...

You: .search best pizza in NYC
Sable: [shows search results]
```

**Auto-response when mentioned:**
```
Friend: Hey @YourName, what time is the meeting?
Sable: [auto-responds] Checking your calendar...
```

## Using Both Bot and Userbot

You can run **both** simultaneously:

```bash
# .env file
TELEGRAM_BOT_TOKEN=123456:ABC...    # For bot
TELEGRAM_USERBOT_ENABLED=true       # For userbot
```

**Bot**: Use in channels you control, shared chats
**Userbot**: Use in your personal chats, groups

## Security Notes

⚠️ **IMPORTANT**:
- Your userbot uses YOUR Telegram account
- Be careful what commands you run
- Don't share your API credentials
- Session file (`opensable_session.session`) contains your login - keep it safe
- Telegram may ban accounts that spam or violate ToS

### Permissions

The userbot has access to:
- All your chats
- Your contacts
- Messages you send/receive

It will only:
- Respond to `.` commands YOU send
- Auto-respond when someone mentions you (if enabled)

## Troubleshooting

### "Invalid phone number"
- Make sure format is: `+1234567890` (with country code)

### "Session file error"
- Delete `opensable_session.session` and restart
- You'll need to log in again

### "API ID/Hash invalid"
- Double-check credentials from my.telegram.org
- Make sure no extra spaces in .env

### "Flood wait"
- Telegram rate limiting
- Wait a few minutes before retrying

### Userbot not responding
- Check logs: `tail -f logs/opensable.log`
- Verify userbot is enabled in .env
- Make sure you're using `.` prefix

## Advanced Usage

### Custom Trigger Prefix

Edit `interfaces/telegram_userbot.py`:
```python
self.trigger_prefix = "!"  # Use ! instead of .
```

### Disable Auto-Response

```bash
# .env
USERBOT_AUTO_RESPOND=false
```

### Add Custom Commands

Edit `handle_command()` in `interfaces/telegram_userbot.py`:

```python
elif command == "mycommand":
    await event.reply("Custom response!")
    return
```

## Legal & Safety

- ✅ Personal use is fine
- ❌ Don't spam or violate Telegram ToS
- ❌ Don't use for mass messaging
- ✅ Keep your session file secure
- ✅ Use for automation of YOUR account only

## Uninstalling Userbot

1. Set `TELEGRAM_USERBOT_ENABLED=false` in .env
2. Delete session file: `rm opensable_session.session`
3. Revoke app access: Telegram → Settings → Privacy → Active Sessions

---

**Enjoy your AI-powered Telegram account! 🚀**
