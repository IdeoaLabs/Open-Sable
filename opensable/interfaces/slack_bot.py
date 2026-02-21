"""
Slack Integration for Open-Sable

Uses Slack Bolt SDK for app integration.
"""

import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


class SlackInterface:
    """Slack bot interface using Bolt SDK"""
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.app = None
        self.session_manager = None
        self.command_handler = None
        
    async def initialize(self):
        """Initialize Slack app"""
        try:
            from slack_bolt.async_app import AsyncApp
            from opensable.core.session_manager import SessionManager
            from opensable.core.commands import CommandHandler
            
            self.session_manager = SessionManager()
            self.command_handler = CommandHandler(
                self.session_manager,
                plugin_manager=getattr(self.agent, 'plugins', None),
            )
            
            self.app = AsyncApp(
                token=self.config.slack_bot_token,
                signing_secret=self.config.slack_signing_secret
            )
            
            # Register event handlers
            self.app.message("")(self.handle_message)
            self.app.command("/sable")(self.handle_slash_command)
            
            logger.info("Slack app initialized")
            return True
            
        except ImportError:
            logger.error("slack-bolt not installed. Install with: pip install slack-bolt")
            return False
        except Exception as e:
            logger.error(f"Error initializing Slack: {e}", exc_info=True)
            return False
    
    async def start(self):
        """Start Slack app"""
        if not await self.initialize():
            logger.error("Failed to initialize Slack")
            return
        
        try:
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
            
            handler = AsyncSocketModeHandler(
                self.app,
                self.config.slack_app_token
            )
            
            logger.info("Starting Slack bot in Socket Mode...")
            await handler.start_async()
            
        except Exception as e:
            logger.error(f"Slack error: {e}", exc_info=True)
    
    async def handle_message(self, message, say, client):
        """Handle incoming Slack messages"""
        try:
            user_id = message.get("user")
            text = message.get("text", "")
            channel_id = message.get("channel")
            thread_ts = message.get("thread_ts", message.get("ts"))
            
            # Ignore bot messages
            if message.get("bot_id"):
                return
            
            # Check if bot was mentioned
            bot_user_id = await self._get_bot_user_id(client)
            if f"<@{bot_user_id}>" not in text and message.get("channel_type") != "im":
                return
            
            # Remove bot mention
            text = text.replace(f"<@{bot_user_id}>", "").strip()
            
            logger.info(f"Slack message from {user_id}: {text}")
            
            # Get or create session
            session = self.session_manager.get_or_create_session(
                user_id=user_id,
                channel="slack"
            )
            
            # Check if it's a command
            if self.command_handler.is_command(text):
                result = await self.command_handler.handle_command(
                    text=text,
                    session_id=session.id,
                    user_id=user_id,
                    is_admin=False,
                    is_group=message.get("channel_type") != "im"
                )
                response = result.message
            else:
                # Process through agent
                response = await self.agent.process_message(user_id, text)
            
            # Send response in thread
            reply_text = response if isinstance(response, str) else str(response)
            await say(
                text=reply_text,
                thread_ts=thread_ts
            )
            
        except Exception as e:
            logger.error(f"Error handling Slack message: {e}", exc_info=True)
            await say(
                text="Sorry, I encountered an error processing your message.",
                thread_ts=message.get("thread_ts", message.get("ts"))
            )
    
    async def handle_slash_command(self, ack, command, say):
        """Handle /sable slash command"""
        await ack()
        
        try:
            user_id = command.get("user_id")
            text = command.get("text", "")
            
            # Process through agent
            response = await self.agent.process_message(user_id, text)
            
            await say(str(response))
            
        except Exception as e:
            logger.error(f"Error handling slash command: {e}", exc_info=True)
            await say("Sorry, I encountered an error.")
    
    async def _get_bot_user_id(self, client) -> str:
        """Get bot's user ID"""
        if not hasattr(self, '_bot_user_id'):
            response = await client.auth_test()
            self._bot_user_id = response["user_id"]
        return self._bot_user_id
    
    async def stop(self):
        """Stop Slack app"""
        logger.info("Stopping Slack interface...")
