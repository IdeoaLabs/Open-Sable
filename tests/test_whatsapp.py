"""
Tests for WhatsApp Bot Interface (mocked, no live bridge needed).
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from opensable.interfaces.whatsapp_bot import WhatsAppBot


class TestWhatsAppBotInit:
    """Test WhatsApp bot initialization"""

    def test_init(self):
        config = Mock()
        config.whatsapp_session_name = "test_session"
        config.whatsapp_callback_port = 3334
        agent = Mock()
        bot = WhatsAppBot(config, agent)
        assert bot.config is config
        assert bot.agent is agent
        assert bot.session_name == "test_session"
        assert bot.running is False

    def test_default_session_name(self):
        config = Mock(spec=[])  # no whatsapp_session_name attribute
        agent = Mock()
        bot = WhatsAppBot(config, agent)
        assert bot.session_name == "opensable"

    def test_bridge_path(self):
        config = Mock()
        config.whatsapp_session_name = "s"
        config.whatsapp_callback_port = 3334
        agent = Mock()
        bot = WhatsAppBot(config, agent)
        assert bot.bridge_dir.name == "whatsapp-bridge"
        assert bot.bridge_script.name == "bridge.js"


class TestWhatsAppBotStart:
    """Test start / stop lifecycle"""

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        config = Mock()
        config.whatsapp_session_name = "s"
        config.whatsapp_callback_port = 3334
        agent = Mock()
        bot = WhatsAppBot(config, agent)
        bot.running = True
        # Should return early without error
        await bot.start()
        assert bot.running is True

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        config = Mock()
        config.whatsapp_session_name = "s"
        config.whatsapp_callback_port = 3334
        agent = Mock()
        bot = WhatsAppBot(config, agent)
        bot.running = False
        await bot.stop()
        assert bot.running is False
