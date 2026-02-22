"""
Core configuration management for Open-Sable
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class OpenSableConfig(BaseModel):
    """Main configuration for Open-Sable"""

    # Interface Mode
    cli_enabled: bool = False

    # Disable external APIs and web servers
    enable_gateway: bool = False  # Disable FastAPI/uvicorn web server
    enable_api: bool = False  # Disable REST API endpoints
    enable_websocket: bool = False  # Disable WebSocket server

    # LLM Settings
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3.1:8b"
    auto_select_model: bool = True
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Chat Platforms
    telegram_bot_token: Optional[str] = None
    telegram_allowed_users: List[str] = Field(default_factory=list)

    # Telegram Userbot
    telegram_userbot_enabled: bool = False
    telegram_api_id: Optional[int] = None
    telegram_api_hash: Optional[str] = None
    telegram_phone_number: Optional[str] = None
    telegram_session_name: str = "opensable_session"
    userbot_auto_respond: bool = True

    discord_bot_token: Optional[str] = None
    discord_guild_id: Optional[str] = None
    whatsapp_enabled: bool = False

    slack_bot_token: Optional[str] = None
    slack_app_token: Optional[str] = None
    slack_signing_secret: Optional[str] = None
    slack_allowed_users: Optional[list] = None

    # Email & Calendar
    gmail_enabled: bool = True
    gmail_credentials_path: Path = Path("./config/gmail_credentials.json")
    calendar_enabled: bool = True
    calendar_credentials_path: Path = Path("./config/calendar_credentials.json")
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: Optional[str] = None
    imap_host: Optional[str] = None
    imap_port: int = 993
    imap_user: Optional[str] = None
    imap_password: Optional[str] = None

    # Security
    enable_sandbox: bool = True
    allowed_domains: List[str] = Field(
        default_factory=lambda: ["*.google.com", "*.booking.com", "*.airbnb.com"]
    )
    max_file_size_mb: int = 100

    # Agent Behavior
    agent_name: str = "Sable"
    agent_personality: str = "helpful"  # professional, sarcastic, meme-aware, helpful
    heartbeat_interval: int = 300  # seconds
    max_retries: int = 3

    # Memory
    memory_retention_days: int = 90
    vector_db_path: Path = Path("./data/vectordb")

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("./logs/opensable.log")

    # Voice
    tts_provider: str = "local"
    stt_provider: str = "local"
    tts_voice_gender: str = "female"
    tts_rate: int = 150
    tts_volume: float = 0.9
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_voice_id: Optional[str] = None
    whisper_model_size: str = "base"

    # Image
    image_provider: str = "none"
    openai_image_api_key: Optional[str] = None

    # Mobile relay
    # Sable generates a QR code on first run; the mobile app scans it once.
    # The relay listens on 127.0.0.1 ONLY (never 0.0.0.0) — safe on a VPS.
    # For remote access: either use Tailscale, or expose via Tor hidden service.
    mobile_relay_enabled: bool = False  # disabled until mobile app exists
    mobile_relay_host: str = "127.0.0.1"  # loopback only — never 0.0.0.0
    mobile_relay_port: int = 7891  # arbitrary high port
    mobile_relay_secret: Optional[str] = None  # auto-generated if None
    mobile_relay_tor_enabled: bool = False  # expose via Tor hidden service
    mobile_relay_tailscale: bool = False  # log Tailscale IP on startup

    # Browser WebChat (served by Gateway on loopback TCP)
    webchat_host: str = "127.0.0.1"  # loopback only
    webchat_port: int = 8789  # open http://127.0.0.1:8789
    webchat_token: Optional[str] = None  # if set, URL must include ?token=<value>
    webchat_tailscale: bool = False  # also bind on Tailscale IP (100.x.x.x)

    # Misc compat
    Config_alias: Optional[str] = None

    def exists(self) -> bool:
        """Check if config file exists (compat)"""
        return True

    class Config:
        extra = "ignore"  # silently drop unknown fields (catches env var typos)


def load_config() -> OpenSableConfig:
    """Load configuration from environment variables"""
    load_dotenv()

    config_data = {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "default_model": os.getenv("DEFAULT_MODEL", "llama3.1:8b"),
        "auto_select_model": os.getenv("AUTO_SELECT_MODEL", "true").lower() == "true",
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "telegram_allowed_users": [
            u.strip() for u in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",") if u.strip()
        ],
        "telegram_userbot_enabled": os.getenv("TELEGRAM_USERBOT_ENABLED", "false").lower()
        == "true",
        "telegram_api_id": (
            int(os.getenv("TELEGRAM_API_ID", "0")) if os.getenv("TELEGRAM_API_ID") else None
        ),
        "telegram_api_hash": os.getenv("TELEGRAM_API_HASH"),
        "telegram_phone_number": os.getenv("TELEGRAM_PHONE_NUMBER"),
        "telegram_session_name": os.getenv("TELEGRAM_SESSION_NAME", "opensable_session"),
        "userbot_auto_respond": os.getenv("USERBOT_AUTO_RESPOND", "true").lower() == "true",
        "discord_bot_token": os.getenv("DISCORD_BOT_TOKEN"),
        "discord_guild_id": os.getenv("DISCORD_GUILD_ID"),
        "whatsapp_enabled": os.getenv("WHATSAPP_ENABLED", "false").lower() == "true",
        "slack_bot_token": os.getenv("SLACK_BOT_TOKEN"),
        "slack_app_token": os.getenv("SLACK_APP_TOKEN"),
        "slack_signing_secret": os.getenv("SLACK_SIGNING_SECRET"),
        "slack_allowed_users": [
            u.strip() for u in os.getenv("SLACK_ALLOWED_USERS", "").split(",") if u.strip()
        ],
        "gmail_enabled": os.getenv("GMAIL_ENABLED", "true").lower() == "true",
        "gmail_credentials_path": Path(
            os.getenv("GMAIL_CREDENTIALS_PATH", "./config/gmail_credentials.json")
        ),
        "smtp_host": os.getenv("SMTP_HOST"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "smtp_user": os.getenv("SMTP_USER"),
        "smtp_password": os.getenv("SMTP_PASSWORD"),
        "smtp_from": os.getenv("SMTP_FROM") or os.getenv("SMTP_USER"),
        "imap_host": os.getenv("IMAP_HOST"),
        "imap_port": int(os.getenv("IMAP_PORT", "993")),
        "imap_user": os.getenv("IMAP_USER") or os.getenv("SMTP_USER"),
        "imap_password": os.getenv("IMAP_PASSWORD") or os.getenv("SMTP_PASSWORD"),
        "calendar_enabled": os.getenv("CALENDAR_ENABLED", "true").lower() == "true",
        "calendar_credentials_path": Path(
            os.getenv("CALENDAR_CREDENTIALS_PATH", "./config/calendar_credentials.json")
        ),
        "enable_sandbox": os.getenv("ENABLE_SANDBOX", "true").lower() == "true",
        "allowed_domains": [
            d.strip()
            for d in os.getenv("ALLOWED_DOMAINS", "*.google.com,*.booking.com,*.airbnb.com").split(
                ","
            )
        ],
        "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "100")),
        "agent_name": os.getenv("AGENT_NAME", "Sable"),
        "agent_personality": os.getenv("AGENT_PERSONALITY", "helpful"),
        "heartbeat_interval": int(os.getenv("HEARTBEAT_INTERVAL", "300")),
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),
        "memory_retention_days": int(os.getenv("MEMORY_RETENTION_DAYS", "90")),
        "vector_db_path": Path(os.getenv("VECTOR_DB_PATH", "./data/vectordb")),
        "cli_enabled": os.getenv("CLI_ENABLED", "false").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "log_file": Path(os.getenv("LOG_FILE", "./logs/opensable.log")),
        # Mobile relay
        "mobile_relay_enabled": os.getenv("MOBILE_RELAY_ENABLED", "false").lower() == "true",
        "mobile_relay_host": os.getenv("MOBILE_RELAY_HOST", "127.0.0.1"),
        "mobile_relay_port": int(os.getenv("MOBILE_RELAY_PORT", "7891")),
        "mobile_relay_secret": os.getenv("MOBILE_RELAY_SECRET"),
        "mobile_relay_tor_enabled": os.getenv("MOBILE_RELAY_TOR", "false").lower() == "true",
        "mobile_relay_tailscale": os.getenv("MOBILE_RELAY_TAILSCALE", "false").lower() == "true",
        # Browser WebChat
        "webchat_host": os.getenv("WEBCHAT_HOST", "127.0.0.1"),
        "webchat_port": int(os.getenv("WEBCHAT_PORT", "8789")),
        "webchat_token": os.getenv("WEBCHAT_TOKEN") or None,
        "webchat_tailscale": os.getenv("WEBCHAT_TAILSCALE", "false").lower() == "true",
    }

    return OpenSableConfig(**config_data)


# Alias so older modules that import 'Config' still work
Config = OpenSableConfig
