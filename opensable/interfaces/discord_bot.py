"""
Discord Bot Interface for SableCore
Full-featured Discord integration with voice, images, and embeds
"""
import asyncio
import logging
import discord
from discord.ext import commands
from typing import Optional
from pathlib import Path

from opensable.core.session_manager import SessionManager, SessionConfig
from opensable.core.commands import CommandHandler

logger = logging.getLogger(__name__)

MULTIMODAL_ENABLED = False
try:
    from opensable.core.multi_messenger import (
        MultiMessengerRouter,
        MessengerPlatform,
        UnifiedMessage,
    )
    from opensable.core.image_analyzer import ImageAnalyzer
    from opensable.core.voice_handler import VoiceMessageHandler
    import aiohttp
    MULTIMODAL_ENABLED = True
except ImportError:
    pass


class DiscordInterface:
    """
    Full-featured Discord bot with multimodal support.
    
    Features:
    - Text chat with AI
    - Image analysis (if enabled)
    - Voice message transcription (if enabled)
    - Embeds with rich formatting
    - Session management
    - Command handling
    """
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.bot = None
        self.session_manager = SessionManager()
        self.command_handler = CommandHandler(self.session_manager)
        
        # Initialize multimodal handlers if available
        if MULTIMODAL_ENABLED:
            try:
                self.router = MultiMessengerRouter(self.agent, self.config)
                self.router.register_platform(MessengerPlatform.DISCORD, self._handle_message)
                self.image_analyzer = ImageAnalyzer(config)
                self.voice_handler = VoiceMessageHandler(config, agent)
            except Exception as e:
                logger.warning(f"Could not initialize multimodal features: {e}")
        
        # Conversation history per user
        self.conversations = {}
        
    async def start(self):
        """Start the Discord bot"""
        if not self.config.discord_bot_token:
            logger.error("Discord bot token not configured. Set DISCORD_BOT_TOKEN in .env")
            return
        
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        
        self.bot = commands.Bot(
            command_prefix="!",
            intents=intents,
            help_command=None
        )
        
        # Register events
        @self.bot.event
        async def on_ready():
            logger.info(f"✅ Discord bot logged in as {self.bot.user}")
            logger.info(f"   Connected to {len(self.bot.guilds)} servers")
            
            activity = discord.Activity(
                type=discord.ActivityType.listening,
                name=f"{self.config.agent_name} | !help"
            )
            await self.bot.change_presence(activity=activity)
        
        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return
            
            await self.bot.process_commands(message)
            
            if self.bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
                await self.handle_message(message)
        
        @self.bot.event
        async def on_message_edit(before, after):
            if after.author != self.bot.user:
                await self.bot.process_commands(after)
        
        # Register commands
        @self.bot.command(name="help")
        async def help_command(ctx):
            await self.handle_help(ctx)
        
        @self.bot.command(name="status")
        async def status_command(ctx):
            embed = discord.Embed(
                title="📊 Bot Status",
                color=discord.Color.green()
            )
            
            embed.add_field(name="Status", value="🟢 Online", inline=True)
            embed.add_field(name="Servers", value=str(len(self.bot.guilds)), inline=True)
            embed.add_field(name="Users", value=str(len(self.bot.users)), inline=True)
            embed.add_field(name="Model", value=self.config.default_model, inline=True)
            
            await ctx.send(embed=embed)
        
        @self.bot.command(name="clear")
        async def clear_command(ctx):
            user_id = str(ctx.author.id)
            
            session = self.session_manager.get_session_by_user(user_id, "discord")
            if session:
                session.messages = []
                self.session_manager._save_session(session)
                
            if user_id in self.conversations:
                del self.conversations[user_id]
                
            await ctx.send("✅ Conversation history cleared!")
        
        @self.bot.command(name="skills")
        async def skills_command(ctx):
            try:
                from opensable.core.skills_hub import SkillsHub
                
                hub = SkillsHub(self.config)
                await hub.initialize()
                
                skills = await hub.browse_skills(limit=10)
                
                embed = discord.Embed(
                    title="🛠️ Available Skills",
                    description=f"Top {len(skills)} skills",
                    color=discord.Color.blue()
                )
                
                for skill in skills[:5]:
                    embed.add_field(
                        name=f"{skill.name} ⭐ {skill.rating}",
                        value=f"{skill.description[:100]}...\n📥 {skill.downloads} downloads",
                        inline=False
                    )
                
                await ctx.send(embed=embed)
                
            except Exception as e:
                await ctx.send(f"❌ Error loading skills: {e}")
        
        logger.info("🎮 Starting Discord bot...")
        
        try:
            await self.bot.start(self.config.discord_bot_token)
        except Exception as e:
            logger.error(f"Discord bot failed to start: {e}")
    
    async def stop(self):
        """Stop the Discord bot"""
        if self.bot:
            await self.bot.close()
            logger.info("Discord bot stopped")
    
    async def handle_help(self, ctx):
        """Handle help command"""
        embed = discord.Embed(
            title=f"🤖 {self.config.agent_name} - AI Assistant",
            description="Your intelligent Discord companion",
            color=discord.Color.purple()
        )
        
        embed.add_field(
            name="💬 Chat",
            value="Mention me or DM me to chat!\nExample: `@Sable what's the weather?`",
            inline=False
        )
        
        if MULTIMODAL_ENABLED:
            embed.add_field(
                name="🖼️ Image Analysis",
                value="Upload an image and mention me\nI'll describe it and extract text (OCR)",
                inline=False
            )
            
            embed.add_field(
                name="🎤 Voice Messages",
                value="Send voice messages and I'll transcribe them",
                inline=False
            )
        
        embed.add_field(
            name="⚙️ Commands",
            value="`!help` - This message\n`!status` - Bot status\n`!clear` - Clear your history\n`!skills` - List available skills",
            inline=False
        )
        
        embed.add_field(
            name="Examples",
            value="• Check my emails\n• Add meeting tomorrow\n• What's the weather?",
            inline=False
        )
        
        embed.set_footer(text=f"Powered by {self.config.agent_name}")
        
        await ctx.send(embed=embed)
    
    async def handle_message(self, message):
        """Handle incoming Discord messages"""
        user_id = str(message.author.id)
        
        content = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
        
        if not content and not message.attachments:
            await message.channel.send("How can I help you?")
            return
        
        async with message.channel.typing():
            try:
                if MULTIMODAL_ENABLED and message.attachments:
                    for attachment in message.attachments:
                        if attachment.content_type and attachment.content_type.startswith('image/'):
                            await self._handle_image(message, attachment, content)
                            return
                        
                        elif attachment.content_type and 'audio' in attachment.content_type:
                            await self._handle_voice(message, attachment)
                            return
                
                logger.info(f"Received message from {message.author}: {content}")
                
                session = self.session_manager.get_or_create_session(
                    channel="discord",
                    user_id=user_id,
                    config=SessionConfig(model=self.config.default_model)
                )
                
                if self.command_handler.is_command(content):
                    result = await self.command_handler.handle_command(
                        text=content,
                        session_id=session.id,
                        user_id=user_id,
                        is_admin=message.author.guild_permissions.administrator if message.guild else False,
                        is_group=message.guild is not None
                    )
                    response = result.message
                else:
                    session.add_message("user", content)
                    self.session_manager._save_session(session)
                    
                    response = await self.agent.process_message(user_id, content)
                    
                    session.add_message("assistant", response)
                    self.session_manager._save_session(session)
                
                await self._send_long_message(message.channel, response)
                    
            except Exception as e:
                logger.error(f"Error processing Discord message: {e}", exc_info=True)
                await message.channel.send(f"❌ Sorry, I encountered an error: {str(e)[:100]}")
    
    async def _handle_message(self, msg):
        """Handle message routing (called by MultiMessengerRouter)"""
        if MULTIMODAL_ENABLED:
            history = self.conversations.get(msg.user_id, [])
            response = await self.agent.process_message(msg.user_id, msg.text, history)
            return response
        return ""
    
    async def _handle_image(self, message, attachment, caption):
        """Handle image attachments"""
        if not MULTIMODAL_ENABLED:
            await message.channel.send("❌ Image analysis not available")
            return
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as response:
                    image_data = await response.read()
            
            query = caption if caption else None
            result = await self.image_analyzer.analyze_image(image_data, query)
            
            embed = discord.Embed(
                title="🖼️ Image Analysis",
                color=discord.Color.blue()
            )
            
            embed.add_field(
                name="📝 Description",
                value=result.get('description', 'No description available'),
                inline=False
            )
            
            if result.get('text'):
                embed.add_field(
                    name="📄 Extracted Text (OCR)",
                    value=result['text'][:1000],
                    inline=False
                )
            
            if result.get('objects'):
                embed.add_field(
                    name="🎯 Detected Objects",
                    value=", ".join(result['objects'][:10]),
                    inline=False
                )
            
            embed.set_image(url=attachment.url)
            
            await message.channel.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            await message.channel.send(f"❌ Failed to analyze image: {e}")
    
    async def _handle_voice(self, message, attachment):
        """Handle voice message attachments"""
        if not MULTIMODAL_ENABLED:
            await message.channel.send("❌ Voice transcription not available")
            return
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as response:
                    audio_data = await response.read()
            
            result = await self.voice_handler.process_voice_message(
                audio_data,
                message.author.id,
                respond_with_voice=False
            )
            
            embed = discord.Embed(
                title="🎤 Voice Message Transcription",
                description=result.get('transcription', 'Could not transcribe'),
                color=discord.Color.green()
            )
            
            if result.get('response'):
                embed.add_field(
                    name="💬 Response",
                    value=result['response'],
                    inline=False
                )
            
            await message.channel.send(embed=embed)
            
        except Exception as e:
            logger.error(f"Error processing voice: {e}")
            await message.channel.send(f"❌ Failed to process voice message: {e}")
    
    async def _send_long_message(self, channel, text, max_length=2000):
        """Send long messages by splitting them"""
        if len(text) <= max_length:
            await channel.send(text)
            return
        
        parts = text.split("\n\n")
        current = ""
        
        for part in parts:
            if len(current) + len(part) + 2 <= max_length:
                current += part + "\n\n"
            else:
                if current:
                    await channel.send(current.strip())
                current = part + "\n\n"
        
        if current:
            await channel.send(current.strip())
