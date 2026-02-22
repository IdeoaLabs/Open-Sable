"""WhatsApp Bot Interface using whatsapp-web.js (wwebjs)
Provides full WhatsApp Web automation through Node.js bridge
"""

import asyncio
import aiohttp
import aiohttp.web
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import signal
import sys
import base64

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
    WhatsApp bot using whatsapp-web.js for WhatsApp Web control.

    Architecture:
    - bridge.js (Node.js) runs wwebjs, posts events to Python webhook
    - Python webhook on :3334 receives messages
    - Python sends outbound via bridge REST API on :3333
    """

    def __init__(self, config: Config, agent):
        self.config = config
        self.agent = agent
        self.running = False
        self.bridge_process = None
        self.session_name = getattr(config, "whatsapp_session_name", "opensable")

        # Initialize handlers
        self.router = MultiMessengerRouter(self.agent, self.config)
        self.router.register_platform(MessengerPlatform.WHATSAPP, self._handle_message)
        self.image_analyzer = ImageAnalyzer(config)
        self.voice_handler = VoiceMessageHandler(config, agent)

        self.callback_port = getattr(config, "whatsapp_callback_port", 3334)
        self._webhook_runner = None

        # Bridge paths
        self.bridge_dir = Path(__file__).parent.parent.parent / "whatsapp-bridge"
        self.bridge_script = self.bridge_dir / "bridge.js"

        logger.info(f"WhatsApp bot initialized (session: {self.session_name})")

    async def start(self):
        """Start WhatsApp bot with QR authentication"""
        if self.running:
            logger.warning("WhatsApp bot already running")
            return

        # Check bridge installation
        if not await self._check_bridge_installation():
            logger.error("WhatsApp bridge not installed. Run: cd whatsapp-bridge && npm install")
            return

        logger.info("🚀 Starting WhatsApp bot...")
        logger.info("📱 Scan the QR code with your phone to authenticate")

        self.running = True

        # Start webhook server FIRST so bridge can POST to us
        await self._start_webhook_server()

        # Start wwebjs bridge
        await self._start_bridge()

        # Keep alive
        await self._listen_messages()

    async def _check_bridge_installation(self) -> bool:
        """Check if WhatsApp bridge (wwebjs) is installed"""
        if not self.bridge_dir.exists():
            return False
        if not self.bridge_script.exists():
            return False
        node_modules = self.bridge_dir / "node_modules" / "whatsapp-web.js"
        if not node_modules.exists():
            return False
        return True

    async def _start_webhook_server(self):
        """Start local HTTP server to receive events from bridge via POST"""
        app = aiohttp.web.Application()
        app.router.add_post("/whatsapp-event", self._webhook_handler)

        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, "127.0.0.1", self.callback_port)
        await site.start()
        self._webhook_runner = runner
        logger.info(f"📡 WhatsApp webhook listening on 127.0.0.1:{self.callback_port}")

    async def _webhook_handler(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Handle POST events from bridge.js"""
        try:
            event = await request.json()
            await self._handle_bridge_event(event)
        except Exception as e:
            logger.error(f"Webhook handler error: {e}")
        return aiohttp.web.Response(text="ok")

    async def _start_bridge(self):
        """Start the Node.js wwebjs bridge"""
        try:
            bridge_port = str(getattr(self.config, "whatsapp_bridge_port", 3333))
            self.bridge_process = await asyncio.create_subprocess_exec(
                "node",
                str(self.bridge_script),
                "--session",
                self.session_name,
                "--port",
                bridge_port,
                "--callback-port",
                str(self.callback_port),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.bridge_dir),
            )

            logger.info("✅ WhatsApp bridge started (wwebjs)")

            # Monitor stderr for debug logs + stdout for early boot events
            asyncio.create_task(self._monitor_bridge_stderr())
            asyncio.create_task(self._monitor_bridge_stdout())

        except Exception as e:
            logger.error(f"Failed to start WhatsApp bridge: {e}")
            raise

    async def _monitor_bridge_stderr(self):
        """Log bridge stderr for debugging"""
        if not self.bridge_process or not self.bridge_process.stderr:
            return
        async for line in self.bridge_process.stderr:
            decoded = line.decode().strip()
            if decoded:
                logger.info(f"[wwebjs] {decoded}")

    async def _monitor_bridge_stdout(self):
        """Monitor bridge stdout for early events (QR code before webhook ready)"""
        if not self.bridge_process or not self.bridge_process.stdout:
            return
        async for line in self.bridge_process.stdout:
            decoded = line.decode().strip()
            if not decoded:
                continue
            try:
                event = json.loads(decoded)
                await self._handle_bridge_event(event)
            except json.JSONDecodeError:
                logger.info(f"[wwebjs] {decoded}")

    async def _handle_bridge_event(self, event: Dict[str, Any]):
        """Handle events from WhatsApp bridge"""
        event_type = event.get("type")

        if event_type == "qr":
            # QR code for authentication
            qr_code = event.get("qr")
            logger.info("\n" + "=" * 50)
            logger.info("📱 SCAN THIS QR CODE WITH WHATSAPP:")
            logger.info("=" * 50)
            logger.info(qr_code)
            logger.info("=" * 50 + "\n")

        elif event_type == "authenticated":
            logger.info("✅ WhatsApp authenticated successfully!")

        elif event_type == "ready":
            logger.info("✅ WhatsApp bot ready to receive messages")

        elif event_type == "message":
            # New message received
            msg_data = event.get("data", {})
            # Handle both {data: message} (old) and message directly (new)
            if isinstance(msg_data, dict) and "data" in msg_data and "from" not in msg_data:
                msg_data = msg_data["data"]
            logger.warning(
                f"📨 WhatsApp event: keys={list(msg_data.keys()) if isinstance(msg_data, dict) else type(msg_data)}"
            )
            await self._process_message(msg_data)

        elif event_type == "heartbeat":
            logger.debug("💚 WhatsApp bridge heartbeat OK")

        elif event_type == "disconnected":
            logger.warning("⚠️ WhatsApp disconnected")

        elif event_type == "error":
            logger.error(f"WhatsApp bridge error: {event.get('data', {}).get('error', event)}")

        else:
            logger.debug(f"[wwebjs] Unhandled event: {event_type}")

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
                logger.debug(f"Skipping own message: {text[:30]}")
                return

            logger.info(f"📩 WhatsApp from {sender_name} ({sender}): {text[:80]}")

            # Create unified message
            unified_msg = UnifiedMessage(
                platform=MessengerPlatform.WHATSAPP,
                user_id=sender,
                chat_id=msg_data.get("chatId", sender),
                text=text,
                metadata={
                    "username": sender_name,
                    "is_group": is_group,
                    "msg_type": msg_type,
                },
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
        return await self.agent.process_message(msg.user_id, msg.text)

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

                response = "🖼️ Image Analysis:\n\n"
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
                    respond_with_voice=False,  # WhatsApp typically doesn't auto-respond with voice
                )

                response = f"🎙️ You said: {result['transcription']}\n\n"
                response += result["response_text"]

                await self._send_message(msg.user_id, response)

        except Exception as e:
            logger.error(f"Error handling voice: {e}")
            await self._send_message(msg.user_id, "Sorry, I couldn't process that voice message.")

    async def _download_media(self, msg_id: str) -> Optional[bytes]:
        """Download media from WhatsApp via wwebjs bridge"""
        try:
            bridge_port = getattr(self.config, "whatsapp_bridge_port", 3333)
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
        """Send message via wwebjs bridge"""
        try:
            bridge_port = getattr(self.config, "whatsapp_bridge_port", 3333)
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

        if self._webhook_runner:
            await self._webhook_runner.cleanup()

        if self.bridge_process:
            self.bridge_process.terminate()
            await self.bridge_process.wait()

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
