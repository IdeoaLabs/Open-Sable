"""
Telegram Userbot interface for Open-Sable
Allows agent to run as your own Telegram account
"""
import logging
import asyncio
from telethon import TelegramClient, events
from telethon.tl.types import User

logger = logging.getLogger(__name__)


class TelegramUserbot:
    """Telegram userbot using Telethon"""
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.client = None
        self.trigger_prefix = "."  # Commands start with .
    
    async def start(self):
        """Start the userbot"""
        if not self.config.telegram_userbot_enabled:
            logger.info("Telegram userbot disabled")
            return
        
        if not all([
            self.config.telegram_api_id,
            self.config.telegram_api_hash,
            self.config.telegram_phone_number
        ]):
            logger.error("Telegram userbot credentials incomplete")
            return
        
        try:
            self.client = TelegramClient(
                self.config.telegram_session_name,
                self.config.telegram_api_id,
                self.config.telegram_api_hash
            )
            
            # Register handlers
            self.client.add_event_handler(
                self.handle_command,
                events.NewMessage(pattern=r'^\.', outgoing=True)
            )
            
            self.client.add_event_handler(
                self.handle_mention,
                events.NewMessage(incoming=True)
            )
            
            logger.info("Starting Telegram userbot...")
            
            await self.client.start(phone=self.config.telegram_phone_number)
            
            me = await self.client.get_me()
            logger.info(f"Userbot started as: {me.first_name} (@{me.username})")
            
            # Keep running
            await self.client.run_until_disconnected()
            
        except Exception as e:
            logger.error(f"Userbot error: {e}")
    
    async def stop(self):
        """Stop the userbot"""
        if self.client:
            await self.client.disconnect()
    
    async def handle_command(self, event):
        """Handle commands sent by user (messages starting with .)"""
        message = event.message.message
        
        # Remove trigger prefix
        command = message[1:].strip()
        
        if not command:
            return
        
        logger.info(f"Userbot command: {command}")
        
        # Built-in commands
        if command == "help":
            help_text = """
**Open-Sable Userbot Commands:**

`.ask <question>` - Ask the AI
`.email` - Check emails
`.calendar` - Check calendar
`.search <query>` - Web search
`.status` - Agent status
`.help` - This message

**Auto-response:**
When someone mentions your name in a message, Open-Sable will auto-respond.
            """
            await event.reply(help_text)
            return
        
        elif command == "status":
            from opensable.core.system_detector import ResourceMonitor
            usage = ResourceMonitor.get_current_usage()
            status = f"""
**Open-Sable Status:**
RAM: {usage['ram_used_percent']:.1f}%
CPU: {usage['cpu_percent']:.1f}%
Disk: {usage['disk_used_percent']:.1f}%
            """
            await event.reply(status)
            return
        
        # Process with agent
        try:
            user_id = str(event.sender_id)
            
            # Parse command
            if command.startswith("ask "):
                query = command[4:]
            elif command.startswith("search "):
                query = f"Search for: {command[7:]}"
            elif command == "email":
                query = "Check my emails"
            elif command == "calendar":
                query = "What's on my calendar?"
            else:
                query = command
            
            # Mark as "typing"
            async with event.client.action(event.chat_id, 'typing'):
                response = await self.agent.process_message(user_id, query)
            
            await event.reply(response)
            
        except Exception as e:
            logger.error(f"Error processing userbot command: {e}")
            await event.reply(f"❌ Error: {str(e)}")
    
    async def handle_mention(self, event):
        """Handle when someone mentions the user in a chat"""
        # Only respond if configured to auto-respond
        if not self.config.userbot_auto_respond:
            return
        
        message = event.message
        
        # Get user's name
        me = await self.client.get_me()
        my_names = [me.first_name.lower(), me.username.lower() if me.username else ""]
        
        # Check if message mentions user
        text = message.message.lower()
        mentioned = any(name in text for name in my_names if name)
        
        if not mentioned:
            return
        
        logger.info(f"Mentioned in chat: {message.chat_id}")
        
        try:
            user_id = str(event.sender_id)
            
            # Process with agent
            async with event.client.action(event.chat_id, 'typing'):
                response = await self.agent.process_message(user_id, message.message)
            
            await event.reply(response)
            
        except Exception as e:
            logger.error(f"Error handling mention: {e}")


class HybridTelegramInterface:
    """Runs both bot and userbot simultaneously"""
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.bot_interface = None
        self.userbot_interface = None
    
    async def start(self):
        """Start both interfaces"""
        tasks = []
        
        # Start bot if configured
        if self.config.telegram_bot_token:
            from opensable.interfaces.telegram_bot import TelegramInterface
            self.bot_interface = TelegramInterface(self.agent, self.config)
            tasks.append(self.bot_interface.start())
            logger.info("Starting Telegram bot...")
        
        # Start userbot if configured
        if self.config.telegram_userbot_enabled:
            self.userbot_interface = TelegramUserbot(self.agent, self.config)
            tasks.append(self.userbot_interface.start())
            logger.info("Starting Telegram userbot...")
        
        if not tasks:
            logger.warning("No Telegram interfaces configured")
            return
        
        # Run both concurrently
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop both interfaces"""
        if self.bot_interface:
            await self.bot_interface.stop()
        if self.userbot_interface:
            await self.userbot_interface.stop()
