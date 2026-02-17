"""
Multi-Messenger Router - Unified Message Handling

Routes messages from multiple platforms (Telegram, WhatsApp, Discord, Signal)
to a single agent instance, with platform-specific formatting and features.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MessengerPlatform(Enum):
    """Supported messaging platforms"""
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    DISCORD = "discord"
    SIGNAL = "signal"
    SMS = "sms"
    WEB = "web"


@dataclass
class UnifiedMessage:
    """Platform-agnostic message format"""
    platform: MessengerPlatform
    user_id: str
    chat_id: str
    text: Optional[str] = None
    voice_data: Optional[bytes] = None
    image_data: Optional[bytes] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UnifiedResponse:
    """Platform-agnostic response format"""
    text: Optional[str] = None
    voice_data: Optional[bytes] = None
    image_data: Optional[bytes] = None
    buttons: Optional[list] = None
    formatting: str = "markdown"  # markdown, html, plain
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MultiMessengerRouter:
    """Routes messages between platforms and agent"""
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        
        # Platform handlers registry
        self.platforms: Dict[MessengerPlatform, Any] = {}
        
        # Message preprocessors (platform-specific normalization)
        self.preprocessors: Dict[MessengerPlatform, Callable] = {}
        
        # Response formatters (platform-specific formatting)
        self.formatters: Dict[MessengerPlatform, Callable] = {}
        
        # Statistics
        self.stats = {
            "messages_routed": 0,
            "by_platform": {p.value: 0 for p in MessengerPlatform}
        }
    
    def register_platform(
        self,
        platform: MessengerPlatform,
        handler: Any,
        preprocessor: Optional[Callable] = None,
        formatter: Optional[Callable] = None
    ):
        """
        Register a messaging platform
        
        Args:
            platform: Platform identifier
            handler: Platform-specific bot/client instance
            preprocessor: Function to normalize incoming messages
            formatter: Function to format outgoing responses
        """
        self.platforms[platform] = handler
        
        if preprocessor:
            self.preprocessors[platform] = preprocessor
        if formatter:
            self.formatters[platform] = formatter
        
        logger.info(f"✅ Registered platform: {platform.value}")
    
    async def route_message(self, message: UnifiedMessage) -> UnifiedResponse:
        """
        Route message through agent and return formatted response
        
        Args:
            message: Unified message from any platform
            
        Returns:
            Formatted response for the source platform
        """
        try:
            # Update stats
            self.stats["messages_routed"] += 1
            self.stats["by_platform"][message.platform.value] += 1
            
            # Preprocess if handler exists
            if message.platform in self.preprocessors:
                message = await self.preprocessors[message.platform](message)
            
            # Handle voice
            if message.voice_data:
                logger.info(f"🎙️ Voice message from {message.platform.value}")
                # Voice processing would go here
                # For now, just indicate voice was received
                message.text = "[Voice message received]"
            
            # Handle images
            if message.image_data:
                logger.info(f"🖼️ Image from {message.platform.value}")
                # Image analysis would go here
                message.text = (message.text or "") + " [Image attached]"
            
            # Route to agent
            logger.info(f"📨 Routing message from {message.platform.value}: {message.text[:50]}...")
            
            response_text = await self.agent.process_message(
                message.user_id,
                message.text or "",
                metadata=message.metadata
            )
            
            # Create response
            response = UnifiedResponse(text=response_text)
            
            # Format for specific platform
            if message.platform in self.formatters:
                response = await self.formatters[message.platform](response, message)
            
            return response
            
        except Exception as e:
            logger.error(f"Routing error: {e}", exc_info=True)
            return UnifiedResponse(
                text=f"❌ Error processing message: {str(e)}",
                formatting="plain"
            )
    
    async def broadcast(
        self,
        message: str,
        platforms: Optional[List[MessengerPlatform]] = None,
        user_filter: Optional[Callable] = None
    ):
        """
        Broadcast message to multiple platforms
        
        Args:
            message: Message text to broadcast
            platforms: Target platforms (None = all)
            user_filter: Optional function to filter recipients
        """
        if platforms is None:
            platforms = list(self.platforms.keys())
        
        logger.info(f"📡 Broadcasting to {len(platforms)} platforms")
        
        tasks = []
        for platform in platforms:
            if platform not in self.platforms:
                continue
            
            handler = self.platforms[platform]
            
            # Platform-specific broadcast
            # (Implementation depends on each platform's API)
            task = self._broadcast_to_platform(handler, platform, message, user_filter)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _broadcast_to_platform(
        self,
        handler,
        platform: MessengerPlatform,
        message: str,
        user_filter: Optional[Callable]
    ):
        """Broadcast to a specific platform"""
        try:
            if platform == MessengerPlatform.TELEGRAM:
                # Telegram broadcast
                if hasattr(handler, 'broadcast_message'):
                    await handler.broadcast_message(message, user_filter)
            
            elif platform == MessengerPlatform.DISCORD:
                # Discord broadcast
                if hasattr(handler, 'send_to_all_channels'):
                    await handler.send_to_all_channels(message)
            
            # Add more platforms as needed
            
        except Exception as e:
            logger.error(f"Broadcast error ({platform.value}): {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            **self.stats,
            "platforms_active": list(self.platforms.keys()),
            "total_platforms": len(self.platforms)
        }


# Platform-specific preprocessors and formatters

async def telegram_preprocessor(message: UnifiedMessage) -> UnifiedMessage:
    """Normalize Telegram messages"""
    # Remove bot mentions
    if message.text and "@" in message.text:
        message.text = message.text.replace(f"@{message.metadata.get('bot_username', '')}", "").strip()
    
    return message


async def telegram_formatter(response: UnifiedResponse, original: UnifiedMessage) -> UnifiedResponse:
    """Format response for Telegram"""
    # Telegram supports markdown
    if response.formatting == "markdown":
        # Ensure markdown is valid for Telegram
        pass
    
    # Add inline buttons if present
    if response.buttons:
        # Format buttons for Telegram InlineKeyboard
        response.metadata["reply_markup"] = _build_telegram_keyboard(response.buttons)
    
    return response


def _build_telegram_keyboard(buttons: list) -> Dict:
    """Build Telegram inline keyboard"""
    return {
        "inline_keyboard": [
            [{"text": btn["text"], "callback_data": btn["callback_data"]} for btn in row]
            for row in buttons
        ]
    }


async def discord_preprocessor(message: UnifiedMessage) -> UnifiedMessage:
    """Normalize Discord messages"""
    # Remove bot mentions
    if message.text and "<@" in message.text:
        # Discord mentions are in format <@USER_ID>
        import re
        message.text = re.sub(r"<@!?\d+>", "", message.text).strip()
    
    return message


async def discord_formatter(response: UnifiedResponse, original: UnifiedMessage) -> UnifiedResponse:
    """Format response for Discord"""
    # Discord supports markdown but with some differences
    if response.formatting == "markdown":
        # Convert Telegram markdown to Discord markdown
        # ** for bold, * for italic (same as Telegram actually)
        pass
    
    return response


async def whatsapp_preprocessor(message: UnifiedMessage) -> UnifiedMessage:
    """Normalize WhatsApp messages"""
    # WhatsApp-specific normalization
    return message


async def whatsapp_formatter(response: UnifiedResponse, original: UnifiedMessage) -> UnifiedResponse:
    """Format response for WhatsApp"""
    # WhatsApp has limited formatting support
    if response.formatting == "markdown":
        # Convert to WhatsApp's simple formatting
        # *bold*, _italic_, ~strikethrough~
        pass
    
    return response
