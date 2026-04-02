"""
Tool schema definitions organized by domain.
"""

from typing import List, Dict, Any

from .browser import SCHEMAS as BROWSER_SCHEMAS
from .system import SCHEMAS as SYSTEM_SCHEMAS
from .core import SCHEMAS as CORE_SCHEMAS
from .marketplace import SCHEMAS as MARKETPLACE_SCHEMAS
from .mobile import SCHEMAS as MOBILE_SCHEMAS
from .desktop import SCHEMAS as DESKTOP_SCHEMAS
from .vision import SCHEMAS as VISION_SCHEMAS
from .x_twitter import SCHEMAS as X_TWITTER_SCHEMAS
from .grok import SCHEMAS as GROK_SCHEMAS
from .instagram import SCHEMAS as INSTAGRAM_SCHEMAS
from .facebook import SCHEMAS as FACEBOOK_SCHEMAS
from .linkedin import SCHEMAS as LINKEDIN_SCHEMAS
from .tiktok import SCHEMAS as TIKTOK_SCHEMAS
from .youtube import SCHEMAS as YOUTUBE_SCHEMAS
from .documents import SCHEMAS as DOCUMENTS_SCHEMAS
from .email import SCHEMAS as EMAIL_SCHEMAS
from .calendar_google import SCHEMAS as CALENDAR_GOOGLE_SCHEMAS
from .clipboard import SCHEMAS as CLIPBOARD_SCHEMAS
from .ocr import SCHEMAS as OCR_SCHEMAS
from .trading import SCHEMAS as TRADING_SCHEMAS
from .github import SCHEMAS as GITHUB_SCHEMAS
from .google_workspace import SCHEMAS as GOOGLE_WORKSPACE_SCHEMAS
from .news_reader import SCHEMAS as NEWS_READER_SCHEMAS
from .business import SCHEMAS as BUSINESS_SCHEMAS
try:
    from .genelia import SCHEMAS as GENELIA_SCHEMAS
except ImportError:
    GENELIA_SCHEMAS = []
from .arena import SCHEMAS as ARENA_SCHEMAS
from .zunvra import SCHEMAS as ZUNVRA_SCHEMAS
from .agentmon import SCHEMAS as AGENTMON_SCHEMAS
from .agent_manager import SCHEMAS as AGENT_MANAGER_SCHEMAS


def get_all_schemas() -> List[Dict[str, Any]]:
    """Return all tool schemas from all domains."""
    all_schemas = []
    all_schemas.extend(BROWSER_SCHEMAS)
    all_schemas.extend(SYSTEM_SCHEMAS)
    all_schemas.extend(CORE_SCHEMAS)
    all_schemas.extend(MARKETPLACE_SCHEMAS)
    all_schemas.extend(MOBILE_SCHEMAS)
    all_schemas.extend(DESKTOP_SCHEMAS)
    all_schemas.extend(VISION_SCHEMAS)
    all_schemas.extend(X_TWITTER_SCHEMAS)
    all_schemas.extend(GROK_SCHEMAS)
    all_schemas.extend(INSTAGRAM_SCHEMAS)
    all_schemas.extend(FACEBOOK_SCHEMAS)
    all_schemas.extend(LINKEDIN_SCHEMAS)
    all_schemas.extend(TIKTOK_SCHEMAS)
    all_schemas.extend(YOUTUBE_SCHEMAS)
    all_schemas.extend(DOCUMENTS_SCHEMAS)
    all_schemas.extend(EMAIL_SCHEMAS)
    all_schemas.extend(CALENDAR_GOOGLE_SCHEMAS)
    all_schemas.extend(CLIPBOARD_SCHEMAS)
    all_schemas.extend(OCR_SCHEMAS)
    all_schemas.extend(TRADING_SCHEMAS)
    all_schemas.extend(GITHUB_SCHEMAS)
    all_schemas.extend(GOOGLE_WORKSPACE_SCHEMAS)
    all_schemas.extend(NEWS_READER_SCHEMAS)
    all_schemas.extend(BUSINESS_SCHEMAS)
    all_schemas.extend(GENELIA_SCHEMAS)
    all_schemas.extend(ARENA_SCHEMAS)
    all_schemas.extend(ZUNVRA_SCHEMAS)
    all_schemas.extend(AGENTMON_SCHEMAS)
    all_schemas.extend(AGENT_MANAGER_SCHEMAS)
    return all_schemas
