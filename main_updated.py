#!/usr/bin/env python3
"""
Updated main entry point with gateway mode support
"""

import asyncio
import logging
import sys
from pathlib import Path

from core.config import load_config
from core.agent import SableAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path.home() / ".opensable" / "logs" / "opensable.log")
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point"""
    try:
        # Load configuration
        config = load_config()
        
        # Check if running in gateway mode
        if "--gateway" in sys.argv or config.gateway_mode:
            logger.info("Starting in Gateway mode")
            from core.gateway import Gateway
            
            gateway = Gateway(config)
            await gateway.start(
                host=config.gateway_host or "127.0.0.1",
                port=config.gateway_port or 18789
            )
            
        else:
            logger.info("Starting in Standalone mode")
            
            # Initialize agent
            agent = SableAgent(config)
            await agent.initialize()
            
            # Start interfaces
            interfaces = []
            
            # Telegram
            if config.telegram_bot_token:
                if config.telegram_userbot_enabled:
                    from interfaces.telegram_userbot import HybridTelegramInterface
                    telegram = HybridTelegramInterface(agent, config)
                else:
                    from interfaces.telegram_bot import TelegramInterface
                    telegram = TelegramInterface(agent, config)
                interfaces.append(("Telegram", telegram))
            
            # Discord
            if config.discord_bot_token:
                from interfaces.discord_bot import DiscordInterface
                discord_bot = DiscordInterface(agent, config)
                interfaces.append(("Discord", discord_bot))
            
            # WhatsApp
            if config.whatsapp_enabled:
                from interfaces.whatsapp_bot import WhatsAppInterface
                whatsapp = WhatsAppInterface(agent, config)
                interfaces.append(("WhatsApp", whatsapp))
            
            # Slack
            if hasattr(config, 'slack_bot_token') and config.slack_bot_token:
                from interfaces.slack_bot import SlackInterface
                slack = SlackInterface(agent, config)
                interfaces.append(("Slack", slack))
            
            if not interfaces:
                logger.error("No interfaces configured!")
                logger.info("Add bot tokens to .env file or run 'sable onboard'")
                sys.exit(1)
            
            logger.info(f"Starting {len(interfaces)} interface(s)...")
            
            # Start all interfaces concurrently
            tasks = [
                asyncio.create_task(interface.start(), name=name)
                for name, interface in interfaces
            ]
            
            # Wait for all
            await asyncio.gather(*tasks)
    
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Create necessary directories
    (Path.home() / ".opensable" / "logs").mkdir(parents=True, exist_ok=True)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print("Open-Sable - Lightweight AI Agent Platform")
            print()
            print("Usage:")
            print("  python main.py                 # Start in standalone mode")
            print("  python main.py --gateway       # Start in gateway mode")
            print("  sable gateway                  # Start gateway (if installed)")
            print("  sable agent                    # Interactive agent")
            print("  sable onboard                  # Setup wizard")
            print()
            print("For more commands, run: sable --help")
            sys.exit(0)
    
    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
