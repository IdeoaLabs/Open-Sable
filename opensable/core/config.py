"""
Core configuration management for Open-Sable

Uses Pydantic v2 with field validators, env-var aliases, and grouped settings.
Env vars are loaded automatically via pydantic-settings when available,
with a manual fallback for backward compat.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class OpenSableConfig(BaseModel):
    """Main configuration for Open-Sable.

    All fields support env-var loading via ``load_config()``.
    Validators enforce URL formats, port ranges, probability bounds, and more.
    """

    # Interface Mode
    cli_enabled: bool = False

    # Disable external APIs and web servers
    enable_gateway: bool = False
    enable_api: bool = False
    enable_websocket: bool = False

    # LLM Settings
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "qwen3.5:0.8b"
    auto_select_model: bool = True
    low_vram_mode: bool = False
    low_vram_preferred_model: str = "qwen3.5:0.8b"
    low_vram_max_model_b: float = 3.0
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    together_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    kimi_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    fm_api_key: Optional[str] = None
    fm_api_url: Optional[str] = None
    fm_model: Optional[str] = None
    openwebui_api_key: Optional[str] = None
    openwebui_api_url: Optional[str] = None
    openwebui_model: Optional[str] = None

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

    # Telegram Group Mode
    telegram_group_only: bool = False
    telegram_allowed_groups: List[str] = Field(default_factory=list)
    telegram_group_engagement: str = "medium"
    telegram_group_context_size: int = 30
    telegram_anti_poison: bool = True

    discord_bot_token: Optional[str] = None
    discord_guild_id: Optional[str] = None
    whatsapp_enabled: bool = False
    whatsapp_callback_port: int = 3334
    whatsapp_bridge_port: int = 3333

    # X (Twitter) / Grok
    x_username: Optional[str] = None
    x_email: Optional[str] = None
    x_password: Optional[str] = None
    x_enabled: bool = False
    x_language: str = "en-US"
    x_action_delay: int = 2  # seconds between actions (rate limit safety)
    grok_enabled: bool = False

    # X Autoposter
    x_autoposter_enabled: bool = False
    x_post_interval: int = 1800  # seconds between posts (default 30 min)
    x_topics: str = "geopolitics,tech,ai"
    x_style: str = "analyst"  # analyst, news, meme, thread
    x_max_daily_posts: int = 5
    x_max_daily_engagements: int = 100
    x_dry_run: bool = False
    x_custom_feeds: str = ""  # comma-separated RSS URLs
    x_engage_interval: int = 300  # seconds between engagement sessions
    x_accounts_to_watch: str = ""  # comma-separated usernames
    x_reply_probability: float = 0.3
    x_like_probability: float = 0.6
    x_retweet_probability: float = 0.2
    x_follow_probability: float = 0.1
    x_quote_probability: float = 0.1
    x_bookmark_probability: float = 0.15

    slack_bot_token: Optional[str] = None
    slack_app_token: Optional[str] = None
    slack_signing_secret: Optional[str] = None
    slack_allowed_users: Optional[list] = None

    # News Reader (WorldMonitor)
    news_enabled: bool = True
    news_cache_ttl: int = 1800  # seconds (30 min)

    # Arena Fighter (fighting-game)
    arena_url: Optional[str] = None  # arena server URL (auto-provisions)

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
    vector_db_path: Path = Path(os.getenv("_SABLE_DATA_DIR", "data")) / "vectordb"

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
    elevenlabs_model: str = "eleven_multilingual_v2"
    whisper_model_size: str = "base"
    # Piper TTS (local, fast, works on Raspberry Pi)
    piper_binary: str = "piper"
    piper_voice: str = "en_US-lessac-medium"
    piper_model_dir: str = str(Path.home() / ".local" / "share" / "piper" / "voices")
    piper_aplay_device: str = "plug:dmix"
    piper_auto_play: bool = True

    @property
    def tts_engine(self) -> str:
        """Alias so core/voice.py can read config.tts_engine."""
        return self.tts_provider

    # Image
    image_provider: str = "none"
    openai_image_api_key: Optional[str] = None

    # Mobile relay
    # Sable generates a QR code on first run; the mobile app scans it once.
    # The relay listens on 127.0.0.1 ONLY (never 0.0.0.0),  safe on a VPS.
    # For remote access: either use Tailscale, or expose via Tor hidden service.
    mobile_relay_enabled: bool = False  # disabled until mobile app exists
    mobile_relay_host: str = "127.0.0.1"  # loopback only,  never 0.0.0.0
    mobile_relay_port: int = 7891  # arbitrary high port
    mobile_relay_secret: Optional[str] = None  # auto-generated if None
    mobile_relay_tor_enabled: bool = False  # expose via Tor hidden service
    mobile_relay_tailscale: bool = False  # log Tailscale IP on startup

    # Browser WebChat (served by Gateway on loopback TCP)
    webchat_host: str = "127.0.0.1"  # loopback only
    webchat_port: int = 8789  # open http://127.0.0.1:8789
    webchat_token: Optional[str] = None  # if set, URL must include ?token=<value>
    webchat_tailscale: bool = False  # also bind on Tailscale IP (100.x.x.x)

    # ── Trading Bot ──────────────────────────────────────────────
    trading_enabled: bool = False
    trading_paper_mode: bool = True  # SAFETY: paper trade by default
    trading_auto_trade: bool = False  # autonomous trading in autonomous mode
    trading_scan_interval: int = 60  # seconds between market scans

    # Exchange API keys
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    binance_testnet: bool = True

    coinbase_api_key: Optional[str] = None
    coinbase_api_secret: Optional[str] = None

    alpaca_api_key: Optional[str] = None
    alpaca_api_secret: Optional[str] = None
    alpaca_paper: bool = True

    polymarket_private_key: Optional[str] = None
    polymarket_funder: Optional[str] = None

    hyperliquid_private_key: Optional[str] = None
    hyperliquid_testnet: bool = True

    jupiter_private_key: Optional[str] = None
    jupiter_rpc_url: str = "https://api.mainnet-beta.solana.com"

    # Risk limits
    trading_max_position_pct: float = 5.0
    trading_max_daily_loss_pct: float = 2.0
    trading_max_drawdown_pct: float = 10.0
    trading_max_open_positions: int = 10
    trading_max_order_usd: float = 10000.0
    trading_require_approval_above_usd: float = 100.0
    trading_banned_assets: str = ""  # comma-separated

    # ── Skills Marketplace ───────────────────────────────────────
    skill_install_auto_approve: bool = False  # If True, skip user approval for marketplace installs

    # Strategy settings
    trading_strategies: str = "momentum,mean_reversion,sentiment"  # comma-separated
    trading_watchlist: str = "BTC/USDT,ETH/USDT,SOL/USDT"  # comma-separated

    # ── Autonomous Mode ──────────────────────────────────────────
    autonomous_mode: bool = False  # Enable autonomous tick loop
    autonomous_check_interval: int = 600  # Seconds between ticks (10 min)
    autonomous_max_tasks: int = 2  # Max tasks per tick
    autonomous_sources: str = "calendar,email,system_monitoring,news"  # comma-separated

    # ── Proactive Reasoning ─────────────────────────────────────
    proactive_think_every_n_ticks: int = 3  # LLM reasoning every N ticks (with 180s interval = every ~9 min)
    proactive_max_risk: str = "medium"  # low / medium / high

    # ── GitHub Integration ──────────────────────────────────────
    github_token: Optional[str] = None  # Personal access token (ghp_...)
    github_default_repo: Optional[str] = None  # owner/repo

    # ── Pixel-Bridge (Pixel Agents VS Code extension integration) ─
    # When True, pixel-bridge.py is auto-launched alongside the agent so the
    # Pixel Agents extension can show an animated character for this agent.
    pixel_bridge_enabled: bool = False

    # Misc compat
    Config_alias: Optional[str] = None

    # ── Validators ────────────────────────────────────────────────

    @field_validator("ollama_base_url", "fm_api_url", "openwebui_api_url", "jupiter_rpc_url", mode="before")
    @classmethod
    def _validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().rstrip("/")
        if v and not v.startswith(("http://", "https://")):
            raise ValueError(f"URL must start with http:// or https://, got: {v!r}")
        return v

    @field_validator(
        "webchat_port", "mobile_relay_port", "smtp_port", "imap_port", mode="before"
    )
    @classmethod
    def _validate_port(cls, v: int, info) -> int:
        v = int(v)
        # webchat_port=0 means "auto-assign a free port" (used by child agents)
        if v == 0 and info.field_name == "webchat_port":
            return v
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be 1-65535, got {v}")
        return v

    @field_validator(
        "x_reply_probability",
        "x_like_probability",
        "x_retweet_probability",
        "x_follow_probability",
        "x_quote_probability",
        "x_bookmark_probability",
        mode="before",
    )
    @classmethod
    def _validate_probability(cls, v: float) -> float:
        v = float(v)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Probability must be 0.0-1.0, got {v}")
        return v

    @field_validator(
        "trading_max_position_pct",
        "trading_max_daily_loss_pct",
        "trading_max_drawdown_pct",
        mode="before",
    )
    @classmethod
    def _validate_pct(cls, v: float) -> float:
        v = float(v)
        if not (0.0 <= v <= 100.0):
            raise ValueError(f"Percentage must be 0-100, got {v}")
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = str(v).upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return v

    @field_validator("agent_personality", mode="before")
    @classmethod
    def _validate_personality(cls, v: str) -> str:
        v = str(v).lower()
        valid = {"professional", "sarcastic", "meme-aware", "helpful"}
        if v not in valid:
            logger.warning(f"Unknown agent_personality {v!r}, defaulting to 'helpful'")
            return "helpful"
        return v

    @field_validator("tts_volume", mode="before")
    @classmethod
    def _validate_volume(cls, v: float) -> float:
        v = float(v)
        return max(0.0, min(1.0, v))

    @field_validator("webchat_host", "mobile_relay_host", mode="before")
    @classmethod
    def _validate_bind_host(cls, v: str) -> str:
        v = str(v).strip()
        if v == "0.0.0.0":
            logger.warning(
                f"Binding to 0.0.0.0 exposes port to the entire network. "
                f"Consider 127.0.0.1 for local-only access."
            )
        return v

    @field_validator("heartbeat_interval", "trading_scan_interval", mode="before")
    @classmethod
    def _validate_positive_int(cls, v: int) -> int:
        v = int(v)
        if v < 1:
            raise ValueError(f"Value must be >= 1, got {v}")
        return v

    @field_validator("memory_retention_days", mode="before")
    @classmethod
    def _validate_retention_days(cls, v: int) -> int:
        v = int(v)
        if v < 1:
            raise ValueError(f"Retention days must be >= 1, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_trading_safety(self) -> "OpenSableConfig":
        """Warn when live trading is enabled without paper mode."""
        if self.trading_enabled and not self.trading_paper_mode:
            logger.warning(
                "⚠️  LIVE TRADING is enabled (trading_paper_mode=False). "
                "Real money will be used. Set TRADING_PAPER_MODE=true to paper trade."
            )
        if self.trading_auto_trade and not self.trading_enabled:
            logger.warning(
                "trading_auto_trade is True but trading_enabled is False,  "
                "auto-trading will not activate."
            )
        return self

    def exists(self) -> bool:
        """Check if config file exists (compat)"""
        return True

    model_config = ConfigDict(extra="ignore", validate_default=True)


def load_config() -> OpenSableConfig:
    """Load configuration from environment variables"""
    load_dotenv()

    sofia_api_url = os.getenv("SOFIA_API_URL")
    sofia_api_key = os.getenv("SOFIA_API_KEY")
    sofia_model = os.getenv("SOFIA_MODEL")

    legacy_openwebui_api_url = os.getenv("OPENWEBUI_API_URL") or sofia_api_url
    legacy_openwebui_api_key = os.getenv("OPENWEBUI_API_KEY") or sofia_api_key
    legacy_openwebui_model = os.getenv("OPENWEBUI_MODEL") or sofia_model
    legacy_fm_endpoint = bool(
        legacy_openwebui_api_url and "fm.opensable.com" in legacy_openwebui_api_url.lower()
    )

    fm_api_url = os.getenv("FM_API_URL") or (legacy_openwebui_api_url if legacy_fm_endpoint else None)
    fm_api_key = os.getenv("FM_API_KEY") or (legacy_openwebui_api_key if legacy_fm_endpoint else None)
    fm_model = os.getenv("FM_MODEL") or (legacy_openwebui_model if legacy_fm_endpoint else None)

    openwebui_api_url = None if legacy_fm_endpoint else legacy_openwebui_api_url
    openwebui_api_key = None if legacy_fm_endpoint else legacy_openwebui_api_key
    openwebui_model = None if legacy_fm_endpoint else legacy_openwebui_model

    config_data = {
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "default_model": os.getenv("DEFAULT_MODEL", "qwen3.5:0.8b"),
        "auto_select_model": os.getenv("AUTO_SELECT_MODEL", "true").lower() == "true",
        "low_vram_mode": os.getenv("LOW_VRAM_MODE", "false").lower() == "true",
        "low_vram_preferred_model": os.getenv("LOW_VRAM_PREFERRED_MODEL", "qwen3.5:0.8b"),
        "low_vram_max_model_b": float(os.getenv("LOW_VRAM_MAX_MODEL_B", "3.0")),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "together_api_key": os.getenv("TOGETHER_API_KEY"),
        "xai_api_key": os.getenv("XAI_API_KEY"),
        "mistral_api_key": os.getenv("MISTRAL_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "cohere_api_key": os.getenv("COHERE_API_KEY"),
        "kimi_api_key": os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"),
        "qwen_api_key": os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "fm_api_key": fm_api_key,
        "fm_api_url": fm_api_url,
        "fm_model": fm_model,
        "openwebui_api_key": openwebui_api_key,
        "openwebui_api_url": openwebui_api_url,
        "openwebui_model": openwebui_model,
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
        "telegram_group_only": os.getenv("TELEGRAM_GROUP_ONLY", "false").lower() == "true",
        "telegram_allowed_groups": [
            g.strip() for g in os.getenv("TELEGRAM_ALLOWED_GROUPS", "").split(",") if g.strip()
        ],
        "telegram_group_engagement": os.getenv("TELEGRAM_GROUP_ENGAGEMENT", "medium"),
        "telegram_group_context_size": int(os.getenv("TELEGRAM_GROUP_CONTEXT_SIZE", "30")),
        "telegram_anti_poison": os.getenv("TELEGRAM_ANTI_POISON", "true").lower() == "true",
        "discord_bot_token": os.getenv("DISCORD_BOT_TOKEN"),
        "discord_guild_id": os.getenv("DISCORD_GUILD_ID"),
        "whatsapp_enabled": os.getenv("WHATSAPP_ENABLED", "false").lower() == "true",
        "whatsapp_callback_port": int(os.getenv("WHATSAPP_CALLBACK_PORT", "3334")),
        "whatsapp_bridge_port": int(os.getenv("WHATSAPP_BRIDGE_PORT", "3333")),
        # X (Twitter) / Grok
        "x_username": os.getenv("X_USERNAME"),
        "x_email": os.getenv("X_EMAIL"),
        "x_password": os.getenv("X_PASSWORD"),
        "x_enabled": os.getenv("X_ENABLED", "false").lower() == "true",
        "x_language": os.getenv("X_LANGUAGE", "en-US"),
        "x_action_delay": int(os.getenv("X_ACTION_DELAY", "2")),
        "grok_enabled": os.getenv("GROK_ENABLED", "false").lower() == "true",
        # X Autoposter
        "x_autoposter_enabled": os.getenv("X_AUTOPOSTER_ENABLED", "false").lower() == "true",
        "x_post_interval": int(os.getenv("X_POST_INTERVAL", "1800")),
        "x_topics": os.getenv("X_TOPICS", "geopolitics,tech,ai"),
        "x_style": os.getenv("X_STYLE", "analyst"),
        "x_max_daily_posts": int(os.getenv("X_MAX_DAILY_POSTS", "5")),
        "x_max_daily_engagements": int(os.getenv("X_MAX_DAILY_ENGAGEMENTS", "100")),
        "x_dry_run": os.getenv("X_DRY_RUN", "false").lower() == "true",
        "x_custom_feeds": os.getenv("X_CUSTOM_FEEDS", ""),
        "x_engage_interval": int(os.getenv("X_ENGAGE_INTERVAL", "300")),
        "x_accounts_to_watch": os.getenv("X_ACCOUNTS_TO_WATCH", ""),
        "x_reply_probability": float(os.getenv("X_REPLY_PROBABILITY", "0.3")),
        "x_like_probability": float(os.getenv("X_LIKE_PROBABILITY", "0.6")),
        "x_retweet_probability": float(os.getenv("X_RETWEET_PROBABILITY", "0.2")),
        "x_follow_probability": float(os.getenv("X_FOLLOW_PROBABILITY", "0.1")),
        "x_quote_probability": float(os.getenv("X_QUOTE_PROBABILITY", "0.1")),
        "x_bookmark_probability": float(os.getenv("X_BOOKMARK_PROBABILITY", "0.15")),
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
        # News Reader
        "news_enabled": os.getenv("NEWS_ENABLED", "true").lower() == "true",
        "news_cache_ttl": int(os.getenv("NEWS_CACHE_TTL", "1800")),
        # Arena Fighter
        "arena_url": os.getenv("ARENA_URL") or None,
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
        "vector_db_path": Path(os.getenv("VECTOR_DB_PATH") or os.path.join(os.getenv("_SABLE_DATA_DIR", "data"), "vectordb")),
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
        # Voice / TTS / STT
        "tts_provider": os.getenv("TTS_PROVIDER", "local"),
        "stt_provider": os.getenv("STT_PROVIDER", "local"),
        "tts_voice_gender": os.getenv("TTS_VOICE_GENDER", "female"),
        "tts_rate": int(os.getenv("TTS_RATE", "150")),
        "tts_volume": float(os.getenv("TTS_VOLUME", "0.9")),
        "elevenlabs_api_key": os.getenv("ELEVENLABS_API_KEY"),
        "elevenlabs_voice_id": os.getenv("ELEVENLABS_VOICE_ID"),
        "elevenlabs_model": os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
        "whisper_model_size": os.getenv("WHISPER_MODEL_SIZE", "base"),
        # Piper TTS
        "piper_binary": os.getenv("PIPER_BINARY", "piper"),
        "piper_voice": os.getenv("PIPER_VOICE", "en_US-lessac-medium"),
        "piper_model_dir": os.getenv("PIPER_MODEL_DIR", str(Path.home() / ".local" / "share" / "piper" / "voices")),
        "piper_aplay_device": os.getenv("PIPER_APLAY_DEVICE", "plug:dmix"),
        "piper_auto_play": os.getenv("PIPER_AUTO_PLAY", "true").lower() == "true",
        # ── Trading Bot ──────────────────────────────────────────
        "trading_enabled": os.getenv("TRADING_ENABLED", "false").lower() == "true",
        "trading_paper_mode": os.getenv("TRADING_PAPER_MODE", "true").lower() == "true",
        "trading_auto_trade": os.getenv("TRADING_AUTO_TRADE", "false").lower() == "true",
        "trading_scan_interval": int(os.getenv("TRADING_SCAN_INTERVAL", "60")),
        # Exchange keys
        "binance_api_key": os.getenv("BINANCE_API_KEY"),
        "binance_api_secret": os.getenv("BINANCE_API_SECRET"),
        "binance_testnet": os.getenv("BINANCE_TESTNET", "true").lower() == "true",
        "coinbase_api_key": os.getenv("COINBASE_API_KEY"),
        "coinbase_api_secret": os.getenv("COINBASE_API_SECRET"),
        "alpaca_api_key": os.getenv("ALPACA_API_KEY"),
        "alpaca_api_secret": os.getenv("ALPACA_API_SECRET"),
        "alpaca_paper": os.getenv("ALPACA_PAPER", "true").lower() == "true",
        "polymarket_private_key": os.getenv("POLYMARKET_PRIVATE_KEY"),
        "polymarket_funder": os.getenv("POLYMARKET_FUNDER"),
        "hyperliquid_private_key": os.getenv("HYPERLIQUID_PRIVATE_KEY"),
        "hyperliquid_testnet": os.getenv("HYPERLIQUID_TESTNET", "true").lower() == "true",
        "jupiter_private_key": os.getenv("JUPITER_PRIVATE_KEY"),
        "jupiter_rpc_url": os.getenv("JUPITER_RPC_URL", "https://api.mainnet-beta.solana.com"),
        # Risk limits
        "trading_max_position_pct": float(os.getenv("TRADING_MAX_POSITION_PCT", "5.0")),
        "trading_max_daily_loss_pct": float(os.getenv("TRADING_MAX_DAILY_LOSS_PCT", "2.0")),
        "trading_max_drawdown_pct": float(os.getenv("TRADING_MAX_DRAWDOWN_PCT", "10.0")),
        "trading_max_open_positions": int(os.getenv("TRADING_MAX_OPEN_POSITIONS", "10")),
        "trading_max_order_usd": float(os.getenv("TRADING_MAX_ORDER_USD", "10000")),
        "trading_require_approval_above_usd": float(os.getenv("TRADING_REQUIRE_APPROVAL_ABOVE_USD", "100")),
        "trading_banned_assets": os.getenv("TRADING_BANNED_ASSETS", ""),
        # Strategy & watchlist
        "trading_strategies": os.getenv("TRADING_STRATEGIES", "momentum,mean_reversion,sentiment"),
        "trading_watchlist": os.getenv("TRADING_WATCHLIST", "BTC/USDT,ETH/USDT,SOL/USDT"),
        # ── Skills Marketplace ──────────────────────────────────
        "skill_install_auto_approve": os.getenv("SKILL_INSTALL_AUTO_APPROVE", "false").lower() == "true",
        # ── Autonomous Mode ─────────────────────────────────────
        "autonomous_mode": os.getenv("AUTONOMOUS_MODE", "false").lower() == "true",
        "autonomous_check_interval": int(os.getenv("AUTONOMOUS_CHECK_INTERVAL", "600")),
        "autonomous_max_tasks": int(os.getenv("AUTONOMOUS_MAX_TASKS", "2")),
        "autonomous_sources": os.getenv("AUTONOMOUS_SOURCES", "calendar,email,system_monitoring,news"),
        # ── Proactive Reasoning ─────────────────────────────────
        "proactive_think_every_n_ticks": int(os.getenv("PROACTIVE_THINK_EVERY_N_TICKS", "5")),
        "proactive_max_risk": os.getenv("PROACTIVE_MAX_RISK", "medium"),
        # ── GitHub Integration ──────────────────────────────────
        "github_token": os.getenv("GITHUB_TOKEN"),
        "github_default_repo": os.getenv("GITHUB_DEFAULT_REPO"),
        # ── Pixel-Bridge ────────────────────────────────────────
        "pixel_bridge_enabled": os.getenv("PIXEL_BRIDGE_ENABLED", "false").lower() == "true",
    }

    return OpenSableConfig(**config_data)


# Alias so older modules that import 'Config' still work
Config = OpenSableConfig
