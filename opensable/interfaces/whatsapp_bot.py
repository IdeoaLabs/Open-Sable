"""
WhatsApp Bot Interface using Venom Bot
Provides full WhatsApp Web automation through Node.js bridge
"""
import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess
import signal
import sys
import base64

from opensable.core.agent import SableAgent
from opensable.core.config import Config
from opensable.core.multi_messenger import (
    MultiMessengerRouter,
    MessengerPlatform,
    UnifiedMessage,
)
from opensable.core.image_analyzer import ImageAnalyzer
from opensable.core.voice_handler import VoiceMessageHandler

logger = logging.getLogger(__name__)


class WhatsAppBot:
    """
    WhatsApp bot using Venom Bot for complete WhatsApp Web control.
    
    Features:
    - QR code authentication
    - Send/receive messages
    - Media handling (images, voice, documents)
    - Group management
    - Contact sync
    - Status updates
    """
    
    def __init__(self, config: Config, agent):
        self.config = config
        self.agent = agent
        self.running = False
        self.venom_process = None
        self.session_name = getattr(config, 'whatsapp_session_name', 'opensable')
        
        # Initialize handlers
        self.router = MultiMessengerRouter(self.agent, self.config)
        self.router.register_platform(MessengerPlatform.WHATSAPP, self._handle_message)
        self.image_analyzer = ImageAnalyzer(config)
        self.voice_handler = VoiceMessageHandler(config, agent)
        
        # Venom bot paths
        self.venom_dir = Path(__file__).parent.parent.parent / "venom-bot"
        self.bridge_script = self.venom_dir / "bridge.js"
        
        logger.info(f"WhatsApp bot initialized (session: {self.session_name})")
    
    async def start(self):
        """Start WhatsApp bot with QR authentication"""
        if self.running:
            logger.warning("WhatsApp bot already running")
            return
        
        # Check Venom Bot installation
        if not await self._check_venom_installation():
            logger.error("Venom Bot not installed. Run: python install.py --setup-whatsapp")
            return
        
        logger.info("🚀 Starting WhatsApp bot...")
        logger.info("📱 Scan the QR code with your phone to authenticate")
        
        self.running = True
        
        # Start Venom Bot bridge
        await self._start_venom_bridge()
        
        # Start message listener
        await self._listen_messages()
    
    async def _check_venom_installation(self) -> bool:
        """Check if Venom Bot is installed"""
        if not self.venom_dir.exists():
            return False
        
        if not self.bridge_script.exists():
            return False
        
        # Check node_modules
        node_modules = self.venom_dir / "node_modules"
        if not node_modules.exists():
            return False
        
        return True
    
    async def _start_venom_bridge(self):
        """Start the Node.js Venom Bot bridge"""
        try:
            # Start bridge process
            self.venom_process = await asyncio.create_subprocess_exec(
                "node",
                str(self.bridge_script),
                "--session", self.session_name,
                "--port", str(getattr(self.config, 'whatsapp_bridge_port', 3333)),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.venom_dir),
            )
            
            logger.info("✅ Venom Bot bridge started")
            
            # Monitor bridge output
            asyncio.create_task(self._monitor_bridge_output())
            
        except Exception as e:
            logger.error(f"Failed to start Venom bridge: {e}")
            raise
    
    async def _monitor_bridge_output(self):
        """Monitor Venom Bot bridge output for QR codes and events"""
        if not self.venom_process:
            return
        
        async for line in self.venom_process.stdout:
            decoded = line.decode().strip()
            
            if not decoded:
                continue
            
            # Parse JSON events from bridge
            try:
                event = json.loads(decoded)
                await self._handle_bridge_event(event)
            except json.JSONDecodeError:
                # Raw log output
                logger.info(f"[Venom] {decoded}")
    
    async def _handle_bridge_event(self, event: Dict[str, Any]):
        """Handle events from Venom Bot bridge"""
        event_type = event.get("type")
        
        if event_type == "qr":
            # QR code for authentication
            qr_code = event.get("qr")
            logger.info("\n" + "="*50)
            logger.info("📱 SCAN THIS QR CODE WITH WHATSAPP:")
            logger.info("="*50)
            logger.info(qr_code)
            logger.info("="*50 + "\n")
            
        elif event_type == "authenticated":
            logger.info("✅ WhatsApp authenticated successfully!")
            
        elif event_type == "ready":
            logger.info("✅ WhatsApp bot ready to receive messages")
            
        elif event_type == "message":
            # New message received
            await self._process_message(event.get("data"))
            
        elif event_type == "disconnected":
            logger.warning("⚠️ WhatsApp disconnected")
            
        elif event_type == "error":
            logger.error(f"Venom error: {event.get('error')}")
    
    async def _listen_messages(self):
        """Listen for incoming WhatsApp messages"""
        while self.running:
            try:
                await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                break
    
    async def _process_message(self, msg_data: Dict[str, Any]):
        """Process incoming WhatsApp message"""
        try:
            # Extract message details
            sender = msg_data.get("from", "")
            sender_name = msg_data.get("notifyName", sender)
            text = msg_data.get("body", "")
            is_group = msg_data.get("isGroupMsg", False)
            msg_type = msg_data.get("type", "chat")
            
            # Skip messages from self
            if msg_data.get("fromMe"):
                return
            
            logger.info(f"📩 Message from {sender_name}: {text[:50]}...")
            
            # Create unified message
            unified_msg = UnifiedMessage(
                platform=MessengerPlatform.WHATSAPP,
                user_id=sender,
                username=sender_name,
                text=text,
                is_group=is_group,
                raw_data=msg_data,
            )
            
            # Handle media messages
            if msg_type == "image":
                await self._handle_image(unified_msg, msg_data)
            elif msg_type == "ptt" or msg_type == "audio":
                await self._handle_voice(unified_msg, msg_data)
            else:
                # Text message
                response = await self.router.route_message(unified_msg)
                await self._send_message(sender, response.text)
                
        except Exception as e:
            logger.error(f"Error processing WhatsApp message: {e}")
    
    async def _handle_message(self, msg: UnifiedMessage) -> str:
        """Handle message routing (called by MultiMessengerRouter)"""
        return await self.agent.process(msg.text, msg.user_id)
    
    async def _handle_image(self, msg: UnifiedMessage, msg_data: Dict[str, Any]):
        """Handle image messages"""
        try:
            # Download image via bridge
            image_data = await self._download_media(msg_data.get("id"))
            
            if image_data:
                # Analyze image
                caption = msg_data.get("caption", "")
                query = caption if caption else None
                
                result = await self.image_analyzer.analyze_image(image_data, query)
                
                response = f"🖼️ Image Analysis:\n\n"
                response += f"📝 {result.get('description', 'No description')}\n\n"
                
                if result.get("text"):
                    response += f"📄 Text detected:\n{result['text']}\n\n"
                
                if result.get("objects"):
                    response += f"🎯 Objects: {', '.join(result['objects'])}"
                
                await self._send_message(msg.user_id, response)
                
        except Exception as e:
            logger.error(f"Error handling image: {e}")
            await self._send_message(msg.user_id, "Sorry, I couldn't analyze that image.")
    
    async def _handle_voice(self, msg: UnifiedMessage, msg_data: Dict[str, Any]):
        """Handle voice messages"""
        try:
            # Download audio via bridge
            audio_data = await self._download_media(msg_data.get("id"))
            
            if audio_data:
                # Process voice
                result = await self.voice_handler.process_voice_message(
                    audio_data,
                    msg.user_id,
                    respond_with_voice=False  # WhatsApp typically doesn't auto-respond with voice
                )
                
                response = f"🎙️ You said: {result['transcription']}\n\n"
                response += result['response_text']
                
                await self._send_message(msg.user_id, response)
                
        except Exception as e:
            logger.error(f"Error handling voice: {e}")
            await self._send_message(msg.user_id, "Sorry, I couldn't process that voice message.")
    
    async def _download_media(self, msg_id: str) -> Optional[bytes]:
        """Download media from WhatsApp via Venom bridge"""
        try:
            bridge_port = getattr(self.config, 'whatsapp_bridge_port', 3333)
            url = f"http://localhost:{bridge_port}/download"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"messageId": msg_id}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return base64.b64decode(data.get("media", ""))
        except Exception as e:
            logger.error(f"Error downloading media: {e}")
        return None
    
    async def _send_message(self, to: str, text: str):
        """Send message via Venom bridge"""
        try:
            bridge_port = getattr(self.config, 'whatsapp_bridge_port', 3333)
            url = f"http://localhost:{bridge_port}/send"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"to": to, "message": text}) as resp:
                    if resp.status != 200:
                        logger.error(f"Bridge send failed: {resp.status}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def stop(self):
        """Stop WhatsApp bot"""
        logger.info("Stopping WhatsApp bot...")
        self.running = False
        
        if self.venom_process:
            self.venom_process.terminate()
            await self.venom_process.wait()
        
        logger.info("✅ WhatsApp bot stopped")


async def main():
    """Standalone WhatsApp bot runner"""
    from opensable.core.config import load_config
    
    config = load_config()
    agent = Agent(config)
    
    bot = WhatsAppBot(config, agent)
    
    # Handle shutdown
    def signal_handler(sig, frame):
        asyncio.create_task(bot.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())

