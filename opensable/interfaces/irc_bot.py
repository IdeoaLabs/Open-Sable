"""
IRC Bot Interface for Open-Sable

Implements IRC bot functionality with:
- Multi-server support
- Channel management
- Private messages
- Rate limiting
- Reconnection handling
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import re

from opensable.core.config import Config, load_config
from opensable.core.session_manager import SessionManager
from opensable.core.commands import CommandHandler
from opensable.core.analytics import Analytics
from opensable.core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class IRCServer:
    """IRC server configuration"""

    host: str
    port: int = 6667
    ssl: bool = False
    nick: str = "SableBot"
    username: str = "opensable"
    realname: str = "Open-Sable AI Bot"
    channels: List[str] = None
    password: Optional[str] = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = []


class IRCProtocol(asyncio.Protocol):
    """IRC protocol handler"""

    def __init__(self, server: IRCServer, bot: "IRCBot"):
        self.server = server
        self.bot = bot
        self.transport = None
        self.buffer = ""
        self.identified = False

    def connection_made(self, transport):
        """Handle connection established"""
        self.transport = transport
        logger.info(f"Connected to {self.server.host}:{self.server.port}")

        # Send registration
        if self.server.password:
            self.send(f"PASS {self.server.password}")

        self.send(f"NICK {self.server.nick}")
        self.send(f"USER {self.server.username} 0 * :{self.server.realname}")

    def data_received(self, data: bytes):
        """Handle received data"""
        try:
            self.buffer += data.decode("utf-8", errors="ignore")

            # Process complete lines
            while "\r\n" in self.buffer:
                line, self.buffer = self.buffer.split("\r\n", 1)
                asyncio.create_task(self.handle_line(line))

        except Exception as e:
            logger.error(f"Error processing data: {e}")

    def connection_lost(self, exc):
        """Handle connection lost"""
        logger.warning(f"Connection lost: {exc}")
        self.identified = False

        # Schedule reconnection
        asyncio.create_task(self.bot.reconnect(self.server))

    def send(self, message: str):
        """Send IRC message"""
        if self.transport:
            logger.debug(f">>> {message}")
            self.transport.write(f"{message}\r\n".encode("utf-8"))

    async def handle_line(self, line: str):
        """Handle IRC line"""
        logger.debug(f"<<< {line}")

        # PING/PONG
        if line.startswith("PING"):
            pong = line.replace("PING", "PONG", 1)
            self.send(pong)
            return

        # Parse IRC message
        prefix = ""
        if line.startswith(":"):
            prefix, line = line[1:].split(" ", 1)

        parts = line.split(" ")
        command = parts[0]
        params = parts[1:] if len(parts) > 1 else []

        # Handle different commands
        if command == "001":  # RPL_WELCOME
            await self.handle_welcome()
        elif command == "PRIVMSG":
            await self.handle_privmsg(prefix, params)
        elif command == "JOIN":
            await self.handle_join(prefix, params)
        elif command == "PART":
            await self.handle_part(prefix, params)

    async def handle_welcome(self):
        """Handle successful registration"""
        logger.info(f"Registered on {self.server.host}")
        self.identified = True

        # Join channels
        for channel in self.server.channels:
            self.send(f"JOIN {channel}")

    async def handle_privmsg(self, prefix: str, params: List[str]):
        """Handle PRIVMSG"""
        if len(params) < 2:
            return

        target = params[0]

        # Extract message (remove leading :)
        message = " ".join(params[1:])
        if message.startswith(":"):
            message = message[1:]

        # Extract sender nick
        sender = prefix.split("!")[0] if "!" in prefix else prefix

        # Determine if channel or private message
        is_channel = target.startswith("#")

        # Check if bot is mentioned
        mentioned = self.server.nick.lower() in message.lower()

        # Only respond if mentioned in channel, or if private message
        if is_channel and not mentioned:
            return

        # Remove bot nick from message if mentioned
        if mentioned:
            message = re.sub(
                f"@?{re.escape(self.server.nick)}[,:;]?", "", message, flags=re.IGNORECASE
            ).strip()

        # Process message
        await self.bot.handle_message(
            server=self.server.host,
            channel=target,
            sender=sender,
            message=message,
            is_channel=is_channel,
        )

    async def handle_join(self, prefix: str, params: List[str]):
        """Handle JOIN"""
        if len(params) < 1:
            return

        channel = params[0]
        if channel.startswith(":"):
            channel = channel[1:]

        nick = prefix.split("!")[0] if "!" in prefix else prefix

        if nick == self.server.nick:
            logger.info(f"Joined {channel} on {self.server.host}")

    async def handle_part(self, prefix: str, params: List[str]):
        """Handle PART"""
        if len(params) < 1:
            return

        channel = params[0]
        if channel.startswith(":"):
            channel = channel[1:]

        nick = prefix.split("!")[0] if "!" in prefix else prefix

        if nick == self.server.nick:
            logger.info(f"Left {channel} on {self.server.host}")


class IRCBot:
    """IRC bot implementation"""

    def __init__(
        self,
        config: Optional[Config] = None,
        session_manager: Optional[SessionManager] = None,
        command_handler: Optional[CommandHandler] = None,
        analytics: Optional[Analytics] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.config = config or load_config()
        self.session_manager = session_manager or SessionManager()
        self.command_handler = command_handler or CommandHandler(self.session_manager)
        self.analytics = analytics or Analytics(self.config)
        self.rate_limiter = rate_limiter or RateLimiter(self.config)

        self.servers: Dict[str, IRCServer] = {}
        self.protocols: Dict[str, IRCProtocol] = {}
        self.running = False

    def add_server(self, server: IRCServer):
        """Add IRC server"""
        self.servers[server.host] = server
        logger.info(f"Added server: {server.host}")

    async def connect(self, server: IRCServer):
        """Connect to IRC server"""
        try:
            logger.info(f"Connecting to {server.host}:{server.port} (SSL: {server.ssl})")

            # Create protocol
            protocol = IRCProtocol(server, self)

            # Connect
            if server.ssl:
                import ssl

                ssl_context = ssl.create_default_context()

                transport, _ = await asyncio.get_event_loop().create_connection(
                    lambda: protocol, server.host, server.port, ssl=ssl_context
                )
            else:
                transport, _ = await asyncio.get_event_loop().create_connection(
                    lambda: protocol, server.host, server.port
                )

            self.protocols[server.host] = protocol

        except Exception as e:
            logger.error(f"Failed to connect to {server.host}: {e}")
            # Schedule reconnection
            await asyncio.sleep(30)
            await self.reconnect(server)

    async def reconnect(self, server: IRCServer):
        """Reconnect to server"""
        if not self.running:
            return

        logger.info(f"Reconnecting to {server.host} in 30 seconds...")
        await asyncio.sleep(30)

        # Remove old protocol
        if server.host in self.protocols:
            del self.protocols[server.host]

        await self.connect(server)

    async def handle_message(
        self, server: str, channel: str, sender: str, message: str, is_channel: bool
    ):
        """Handle incoming IRC message"""
        try:
            # Track message
            self.analytics.track_message_received("irc", f"{server}:{sender}", message)

            # Check rate limit
            try:
                await self.rate_limiter.check_message_limit(f"{server}:{sender}")
            except Exception as e:
                logger.warning(f"Rate limit exceeded for {sender}: {e}")
                return

            # Get/create session
            session = self.session_manager.get_or_create_session(
                "irc", f"{server}:{channel}:{sender}"
            )

            # Add user message
            session.add_message("user", message)

            # Check for commands
            if message.startswith("/"):
                result = self.command_handler.handle_command(message, session)

                if result:
                    await self.send_message(server, channel, sender, result.message, is_channel)
                    self.analytics.track_command(message.split()[0], sender, result.success)

                if not result.should_continue:
                    return

            # Generate response
            from opensable.core.agent import SableAgent

            agent = SableAgent(self.config)

            response = await agent.send_message(message, session=session)

            if response and "content" in response:
                response_text = response["content"]

                # Add to session
                session.add_message("assistant", response_text)

                # Send response
                await self.send_message(server, channel, sender, response_text, is_channel)

                # Track
                self.analytics.track_message_sent(
                    "irc", f"{server}:{sender}", response_text, tokens=response.get("tokens", 0)
                )

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            error_msg = "Sorry, an error occurred processing your message."
            await self.send_message(server, channel, sender, error_msg, is_channel)

    async def send_message(
        self, server: str, channel: str, sender: str, message: str, is_channel: bool
    ):
        """Send IRC message"""
        if server not in self.protocols:
            logger.error(f"No protocol for server {server}")
            return

        protocol = self.protocols[server]

        # Split long messages
        max_length = 400
        lines = []

        for line in message.split("\n"):
            while len(line) > max_length:
                lines.append(line[:max_length])
                line = line[max_length:]
            if line:
                lines.append(line)

        # Send each line
        for line in lines:
            if is_channel:
                # Mention sender in channel
                protocol.send(f"PRIVMSG {channel} :{sender}: {line}")
            else:
                # Private message
                protocol.send(f"PRIVMSG {sender} :{line}")

            # Rate limit sends
            await asyncio.sleep(1)

    async def start(self):
        """Start IRC bot"""
        logger.info("Starting IRC bot...")
        self.running = True

        # Connect to all servers
        tasks = []
        for server in self.servers.values():
            task = asyncio.create_task(self.connect(server))
            tasks.append(task)

        # Wait for all connections
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"IRC bot started with {len(self.protocols)} connections")

    async def stop(self):
        """Stop IRC bot"""
        logger.info("Stopping IRC bot...")
        self.running = False

        # Close all connections
        for protocol in self.protocols.values():
            if protocol.transport:
                # Send QUIT
                protocol.send("QUIT :Open-Sable shutting down")
                protocol.transport.close()

        self.protocols.clear()
        logger.info("IRC bot stopped")


async def main():
    """IRC bot example"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config = load_config()
    bot = IRCBot(config=config)

    # Add servers from config
    if hasattr(config, "irc_servers") and config.irc_servers:
        for server_config in config.irc_servers:
            server = IRCServer(**server_config)
            bot.add_server(server)
    else:
        # Example server
        server = IRCServer(
            host="irc.libera.chat",
            port=6697,
            ssl=True,
            nick="SableBot",
            channels=["#opensable-test"],
        )
        bot.add_server(server)

    try:
        await bot.start()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
