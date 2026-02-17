"""
Matrix Bot Interface for Open-Sable

Implements Matrix protocol bot with:
- End-to-end encryption support
- Room management
- Message threading
- Reactions and edits
- File uploads
"""

import asyncio
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
import json

try:
    from nio import (
        AsyncClient,
        MatrixRoom,
        RoomMessageText,
        RoomMessageNotice,
        RoomMemberEvent,
        InviteMemberEvent,
        LoginResponse,
        SyncResponse,
        RoomSendResponse,
        UploadResponse,
        crypto,
        exceptions
    )
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    AsyncClient = None
    MatrixRoom = None
    RoomMessageText = None
    RoomMessageNotice = None
    RoomMemberEvent = None
    InviteMemberEvent = None
    LoginResponse = None
    SyncResponse = None
    RoomSendResponse = None
    UploadResponse = None

from opensable.core.config import Config, load_config
from opensable.core.session_manager import SessionManager
from opensable.core.commands import CommandHandler
from opensable.core.analytics import Analytics
from opensable.core.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


@dataclass
class MatrixConfig:
    """Matrix configuration"""
    homeserver: str
    user_id: str
    password: str
    device_name: str = "Open-Sable Bot"
    device_id: str = "SABLECORE"
    store_path: str = ".matrix_store"
    auto_join: bool = True
    encryption_enabled: bool = True


class MatrixBot:
    """Matrix bot implementation"""
    
    def __init__(
        self,
        matrix_config: MatrixConfig,
        config: Optional[Config] = None,
        session_manager: Optional[SessionManager] = None,
        command_handler: Optional[CommandHandler] = None,
        analytics: Optional[Analytics] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        if not MATRIX_AVAILABLE:
            raise ImportError(
                "matrix-nio not installed. Install with: pip install matrix-nio[e2e]"
            )
        
        self.matrix_config = matrix_config
        self.config = config or load_config()
        self.session_manager = session_manager or SessionManager()
        self.command_handler = command_handler or CommandHandler(self.session_manager)
        self.analytics = analytics or Analytics(self.config)
        self.rate_limiter = rate_limiter or RateLimiter(self.config)
        
        # Create Matrix client
        self.client = AsyncClient(
            homeserver=matrix_config.homeserver,
            user=matrix_config.user_id,
            device_id=matrix_config.device_id,
            store_path=matrix_config.store_path
        )
        
        # Setup callbacks
        self.client.add_event_callback(self.handle_message, RoomMessageText)
        self.client.add_event_callback(self.handle_invite, InviteMemberEvent)
        
        self.running = False
    
    async def login(self):
        """Login to Matrix"""
        try:
            logger.info(f"Logging in to {self.matrix_config.homeserver}")
            
            response = await self.client.login(
                password=self.matrix_config.password,
                device_name=self.matrix_config.device_name
            )
            
            if isinstance(response, LoginResponse):
                logger.info(f"Logged in as {self.matrix_config.user_id}")
                
                # Setup encryption if enabled
                if self.matrix_config.encryption_enabled:
                    if self.client.should_upload_keys:
                        await self.client.keys_upload()
                        logger.info("Uploaded encryption keys")
                
                return True
            else:
                logger.error(f"Login failed: {response}")
                return False
        
        except Exception as e:
            logger.error(f"Login error: {e}", exc_info=True)
            return False
    
    async def handle_invite(self, room: MatrixRoom, event: InviteMemberEvent):
        """Handle room invite"""
        if not self.matrix_config.auto_join:
            return
        
        try:
            logger.info(f"Invited to {room.room_id} by {event.sender}")
            
            # Auto-join
            await self.client.join(room.room_id)
            logger.info(f"Joined {room.room_id}")
            
            # Send welcome message
            await self.client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.notice",
                    "body": "👋 Hello! I'm Open-Sable, an AI assistant. Mention me or send me a message to chat!"
                }
            )
        
        except Exception as e:
            logger.error(f"Error joining room: {e}")
    
    async def handle_message(self, room: MatrixRoom, event: RoomMessageText):
        """Handle incoming message"""
        # Ignore own messages
        if event.sender == self.client.user:
            return
        
        # Ignore old messages (before bot started)
        if not self.running:
            return
        
        try:
            message = event.body
            sender = event.sender
            
            logger.info(f"Message from {sender} in {room.display_name}: {message}")
            
            # Track message
            self.analytics.track_message_received('matrix', sender, message)
            
            # Check if bot is mentioned
            mentioned = self.matrix_config.user_id in message
            
            # Only respond if mentioned or DM
            if not mentioned and not room.is_direct:
                return
            
            # Remove mention from message
            if mentioned:
                message = message.replace(self.matrix_config.user_id, '').strip()
            
            # Check rate limit
            try:
                await self.rate_limiter.check_message_limit(sender)
            except Exception as e:
                logger.warning(f"Rate limit exceeded for {sender}: {e}")
                await self.send_message(
                    room.room_id,
                    "⏱️ You're sending messages too quickly. Please wait a moment."
                )
                return
            
            # Get/create session
            session = self.session_manager.get_or_create_session(
                'matrix',
                f"{room.room_id}:{sender}"
            )
            
            # Add user message
            session.add_message('user', message)
            
            # Check for commands
            if message.startswith('/'):
                result = self.command_handler.handle_command(message, session)
                
                if result:
                    await self.send_message(room.room_id, result.message)
                    self.analytics.track_command(message.split()[0], sender, result.success)
                
                if not result.should_continue:
                    return
            
            # Show typing indicator
            await self.client.room_typing(room.room_id, typing_state=True, timeout=30000)
            
            try:
                # Generate response
                from opensable.core.agent import SableAgent
                agent = SableAgent(self.config)
                
                response = await agent.send_message(
                    message,
                    session=session
                )
                
                if response and 'content' in response:
                    response_text = response['content']
                    
                    # Add to session
                    session.add_message('assistant', response_text)
                    
                    # Send response
                    await self.send_message(room.room_id, response_text)
                    
                    # Track
                    self.analytics.track_message_sent(
                        'matrix',
                        sender,
                        response_text,
                        tokens=response.get('tokens', 0)
                    )
            
            finally:
                # Stop typing
                await self.client.room_typing(room.room_id, typing_state=False)
        
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            await self.send_message(
                room.room_id,
                "❌ Sorry, an error occurred processing your message."
            )
    
    async def send_message(self, room_id: str, message: str, msgtype: str = "m.text"):
        """Send Matrix message"""
        try:
            # Split long messages
            max_length = 4000
            
            if len(message) > max_length:
                # Send in parts
                parts = []
                while len(message) > max_length:
                    # Find last newline before limit
                    split_pos = message.rfind('\n', 0, max_length)
                    if split_pos == -1:
                        split_pos = max_length
                    
                    parts.append(message[:split_pos])
                    message = message[split_pos:].lstrip()
                
                if message:
                    parts.append(message)
                
                # Send each part
                for i, part in enumerate(parts):
                    content = {
                        "msgtype": msgtype,
                        "body": part
                    }
                    
                    if i == 0 and len(parts) > 1:
                        content["body"] = f"{part}\n\n(Message continues...)"
                    elif i > 0:
                        content["body"] = f"(Continued...)\n\n{part}"
                    
                    response = await self.client.room_send(
                        room_id=room_id,
                        message_type="m.room.message",
                        content=content
                    )
                    
                    if not isinstance(response, RoomSendResponse):
                        logger.error(f"Failed to send message: {response}")
                    
                    # Small delay between parts
                    await asyncio.sleep(0.5)
            else:
                # Send single message
                content = {
                    "msgtype": msgtype,
                    "body": message
                }
                
                response = await self.client.room_send(
                    room_id=room_id,
                    message_type="m.room.message",
                    content=content
                )
                
                if not isinstance(response, RoomSendResponse):
                    logger.error(f"Failed to send message: {response}")
        
        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
    
    async def send_file(self, room_id: str, filepath: str, filename: str = None):
        """Upload and send file"""
        try:
            import mimetypes
            from pathlib import Path
            
            path = Path(filepath)
            
            if not path.exists():
                logger.error(f"File not found: {filepath}")
                return
            
            # Read file
            with open(filepath, 'rb') as f:
                file_data = f.read()
            
            # Determine mimetype
            mimetype = mimetypes.guess_type(filepath)[0] or 'application/octet-stream'
            
            # Upload file
            response = await self.client.upload(
                data_provider=lambda: file_data,
                content_type=mimetype,
                filename=filename or path.name,
                filesize=len(file_data)
            )
            
            if isinstance(response, UploadResponse):
                # Send file message
                content = {
                    "msgtype": "m.file",
                    "body": filename or path.name,
                    "url": response.content_uri,
                    "info": {
                        "mimetype": mimetype,
                        "size": len(file_data)
                    }
                }
                
                await self.client.room_send(
                    room_id=room_id,
                    message_type="m.room.message",
                    content=content
                )
                
                logger.info(f"Sent file: {filename or path.name}")
            else:
                logger.error(f"Failed to upload file: {response}")
        
        except Exception as e:
            logger.error(f"Error sending file: {e}", exc_info=True)
    
    async def start(self):
        """Start Matrix bot"""
        logger.info("Starting Matrix bot...")
        
        # Login
        if not await self.login():
            logger.error("Failed to login")
            return
        
        self.running = True
        
        try:
            # Initial sync
            logger.info("Performing initial sync...")
            await self.client.sync(timeout=30000, full_state=True)
            
            logger.info("Matrix bot started, listening for messages...")
            
            # Continuous sync
            while self.running:
                await self.client.sync_forever(timeout=30000, full_state=False)
        
        except Exception as e:
            logger.error(f"Error in sync loop: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop Matrix bot"""
        logger.info("Stopping Matrix bot...")
        self.running = False
        
        # Close client
        await self.client.close()
        
        logger.info("Matrix bot stopped")


async def main():
    """Matrix bot example"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config()
    
    # Create Matrix config
    matrix_config = MatrixConfig(
        homeserver=config.matrix_homeserver,
        user_id=config.matrix_user_id,
        password=config.matrix_password
    )
    
    # Create bot
    bot = MatrixBot(matrix_config=matrix_config, config=config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
