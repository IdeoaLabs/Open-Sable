"""
X Self-Healing System,  The bot watches its own console, detects errors,
and uses Grok to diagnose & fix them automatically.

Features:
  - LogBuffer: Custom logging handler that captures recent log entries
  - ErrorDetector: Pattern-based + AI-powered error classification
  - RemedyEngine: Safe, predefined fix actions the bot can apply
  - SelfHealLoop: Periodic monitor that ties it all together

The bot becomes self-aware of its own failures and adapts in real-time.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  LOG BUFFER,  Captures console output in a ring buffer
# ══════════════════════════════════════════════════════════════════════════════

class LogBuffer(logging.Handler):
    """
    Custom logging handler that keeps the last N log entries in memory.
    The bot reads this buffer to "see" its own console.
    """

    def __init__(self, max_entries: int = 500, error_max: int = 100):
        super().__init__()
        self.all_entries: deque = deque(maxlen=max_entries)
        self.errors: deque = deque(maxlen=error_max)
        self.warnings: deque = deque(maxlen=error_max)
        self._error_count = 0
        self._warning_count = 0
        self._last_check_idx = 0  # Track what we've already analyzed
        self._entry_counter = 0

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "idx": self._entry_counter,
                "ts": datetime.now().isoformat(),
                "level": record.levelname,
                "msg": self.format(record),
                "name": record.name,
                "raw": record.getMessage(),
            }
            self._entry_counter += 1
            self.all_entries.append(entry)

            if record.levelno >= logging.ERROR:
                self.errors.append(entry)
                self._error_count += 1
            elif record.levelno >= logging.WARNING:
                self.warnings.append(entry)
                self._warning_count += 1
        except Exception:
            pass  # Never crash the logging system

    def get_new_errors(self) -> List[Dict]:
        """Get errors that haven't been checked yet."""
        new_errors = [e for e in self.errors if e["idx"] >= self._last_check_idx]
        if self.all_entries:
            self._last_check_idx = self._entry_counter
        return new_errors

    def get_new_warnings(self) -> List[Dict]:
        """Get warnings that haven't been checked yet."""
        return [w for w in self.warnings if w["idx"] >= self._last_check_idx - len(self.errors)]

    def get_recent_context(self, n: int = 30) -> str:
        """Get the last N log lines as a string (what the bot 'sees')."""
        entries = list(self.all_entries)[-n:]
        lines = []
        for e in entries:
            level_marker = "❌" if e["level"] == "ERROR" else "⚠️" if e["level"] == "WARNING" else "  "
            lines.append(f"{level_marker} [{e['level']}] {e['raw'][:200]}")
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        return {
            "total_entries": self._entry_counter,
            "errors": self._error_count,
            "warnings": self._warning_count,
            "buffer_size": len(self.all_entries),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR PATTERNS,  Known errors and their automatic remedies
# ══════════════════════════════════════════════════════════════════════════════

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorPattern:
    """A known error pattern with its automatic remedy."""
    name: str
    pattern: str  # regex to match in log output
    severity: Severity
    remedy: str  # remedy action name
    params: Dict = field(default_factory=dict)
    cooldown: int = 300  # seconds before applying same remedy again
    description: str = ""


# Pre-defined known errors and their remedies
KNOWN_ERRORS: List[ErrorPattern] = [
    ErrorPattern(
        name="automated_detection_226",
        pattern=r"code.*226|might be automated|protect our users from spam",
        severity=Severity.CRITICAL,
        remedy="emergency_stealth",
        params={"pause_minutes": 30, "reduce_activity": 0.3},
        cooldown=1800,
        description="X detected us as automated (Error 226). Need to pause and reduce activity drastically.",
    ),
    ErrorPattern(
        name="rate_limit_429",
        pattern=r"status[:\s]*429|rate.?limit|too many requests",
        severity=Severity.HIGH,
        remedy="rate_limit_backoff",
        params={"backoff_minutes": 15},
        cooldown=900,
        description="Hit rate limit. Need exponential backoff.",
    ),
    ErrorPattern(
        name="auth_expired_401",
        pattern=r"status[:\s]*401|unauthorized|authentication.*fail",
        severity=Severity.HIGH,
        remedy="refresh_auth",
        params={},
        cooldown=600,
        description="Authentication expired. Need to refresh cookies/session.",
    ),
    ErrorPattern(
        name="forbidden_403",
        pattern=r"status[:\s]*403|forbidden|cloudflare",
        severity=Severity.HIGH,
        remedy="rotate_identity",
        params={"pause_minutes": 10},
        cooldown=600,
        description="Blocked by Cloudflare/X. Need to rotate UA and pause.",
    ),
    ErrorPattern(
        name="search_404",
        pattern=r"search.*(?:404|not.?found)|SearchTimeline.*404|status[:\s]*404.*search",
        severity=Severity.MEDIUM,
        remedy="disable_search_temp",
        params={"disable_minutes": 60},
        cooldown=3600,
        description="Search endpoint broken (stale GraphQL hash). Skip search, use timeline instead.",
    ),
    ErrorPattern(
        name="network_error",
        pattern=r"connect(ion)?.*error|timeout|connectionreset|dns.*fail",
        severity=Severity.MEDIUM,
        remedy="network_backoff",
        params={"backoff_minutes": 5},
        cooldown=300,
        description="Network connectivity issue. Brief pause and retry.",
    ),
    ErrorPattern(
        name="grok_fail",
        pattern=r"grok.*fail|grok.*error|KEY_BYTE|add_response.*[45]\d\d",
        severity=Severity.LOW,
        remedy="fallback_llm",
        params={},
        cooldown=120,
        description="Grok API failed. Fallback to LLM.",
    ),
    ErrorPattern(
        name="calendar_datetime_error",
        pattern=r"Calendar error|offset-naive and offset-aware",
        severity=Severity.LOW,
        remedy="skip",
        params={},
        cooldown=60,
        description="Calendar datetime comparison error. Already handled in code.",
    ),
    ErrorPattern(
        name="search_regex_error",
        pattern=r"Search failed: nothing to repeat|Search failed: .*regex|Search failed: .*re\.error",
        severity=Severity.LOW,
        remedy="skip",
        params={},
        cooldown=60,
        description="Search regex compilation error. Already handled in code.",
    ),
    ErrorPattern(
        name="daily_limit",
        pattern=r"reached your daily limit|daily limit for sending Tweets|\b344\b",
        severity=Severity.MEDIUM,
        remedy="pause_until_midnight",
        params={},
        cooldown=86400,
        description="Twitter daily post limit (344). Block posting until midnight, no emergency pause.",
    ),
    ErrorPattern(
        name="permission_error",
        pattern=r"Permissions.*Error|AuthorizationError|suspended|locked",
        severity=Severity.CRITICAL,
        remedy="emergency_pause",
        params={"pause_minutes": 120},
        cooldown=7200,
        description="Account permission issue. Long pause to avoid permanent suspension.",
    ),
    # ── Instagram errors (handled by IG autoposter,  self-heal should IGNORE) ──
    ErrorPattern(
        name="ig_challenge_required",
        pattern=r"challenge_required|Instagram.*upload.*error|IG upload blocked|IG posting blocked|Photo Upload failed|Response \[412\]",
        severity=Severity.LOW,
        remedy="ig_ignore",
        params={},
        cooldown=300,
        description="Instagram challenge,  handled by IG autoposter's own backoff. Do NOT pause X.",
    ),
    ErrorPattern(
        name="ig_login_required",
        pattern=r"Instagram.*login.*required|InstaLogin|instagrapi.*login",
        severity=Severity.LOW,
        remedy="ig_ignore",
        params={},
        cooldown=300,
        description="Instagram login issue,  handled by IG skill. Do NOT pause X.",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  MOBILE USER AGENTS,  Rotate through realistic mobile UAs
# ══════════════════════════════════════════════════════════════════════════════
MOBILE_USER_AGENTS = [
    # Android Chrome (most common)
    "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.200 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.200 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-A546B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.6723.102 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; Pixel 7a) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.200 Mobile Safari/537.36",
    # iOS Safari
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.7 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1",
]

# Desktop fallbacks (when mobile doesn't work)
DESKTOP_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]


def pick_user_agent(prefer_mobile: bool = True) -> str:
    """Pick a realistic user agent, preferring mobile."""
    if prefer_mobile:
        return random.choice(MOBILE_USER_AGENTS)
    return random.choice(DESKTOP_USER_AGENTS)


# ══════════════════════════════════════════════════════════════════════════════
#  REMEDY ENGINE,  Safe pre-defined fix actions
# ══════════════════════════════════════════════════════════════════════════════

class RemedyEngine:
    """
    Applies safe, pre-defined fixes. The bot can only execute these known remedies,
    not arbitrary code. Each remedy modifies agent state in a controlled way.
    """

    def __init__(self, agent_ref):
        """
        Args:
            agent_ref: Reference to XAutonomousAgent instance
        """
        self._agent = agent_ref
        self._applied: Dict[str, datetime] = {}  # remedy_name -> last applied
        self._heal_log: List[Dict] = []
        _data = os.environ.get("_SABLE_DATA_DIR", "data")
        self._log_file = Path(_data) / "x_consciousness" / "heal_log.jsonl"
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        # Track original values for restoration
        self._original_values: Dict[str, Any] = {}
        self._paused_loops: Dict[str, datetime] = {}  # loop_name -> resume_at
        self._search_disabled_until: Optional[datetime] = None

    def can_apply(self, remedy_name: str, cooldown: int) -> bool:
        """Check if enough time has passed since last application."""
        last = self._applied.get(remedy_name)
        if last is None:
            return True
        return (datetime.now() - last).total_seconds() > cooldown

    async def apply(self, error_pattern: ErrorPattern, error_context: str) -> Dict:
        """Apply a remedy. Returns result dict."""
        remedy = error_pattern.remedy
        params = error_pattern.params

        if not self.can_apply(remedy, error_pattern.cooldown):
            return {"applied": False, "reason": "cooldown active"}

        self._applied[remedy] = datetime.now()

        result = {"applied": True, "remedy": remedy, "ts": datetime.now().isoformat()}

        try:
            if remedy == "emergency_stealth":
                result.update(await self._emergency_stealth(params, error_context))
            elif remedy == "rate_limit_backoff":
                result.update(await self._rate_limit_backoff(params))
            elif remedy == "refresh_auth":
                result.update(await self._refresh_auth())
            elif remedy == "rotate_identity":
                result.update(await self._rotate_identity(params))
            elif remedy == "disable_search_temp":
                result.update(self._disable_search(params))
            elif remedy == "network_backoff":
                result.update(await self._network_backoff(params))
            elif remedy == "fallback_llm":
                result.update(self._fallback_llm())
            elif remedy == "ig_ignore":
                # Instagram errors are handled by the IG autoposter itself.
                # Do NOT apply any X remedy,  just log and move on.
                logger.info(f"🩺 Instagram error detected,  skipping X self-heal (IG handles its own backoff)")
                result.update({"action": "ig_ignore", "applied": True})
            elif remedy == "emergency_pause":
                result.update(await self._emergency_pause(params))
            elif remedy == "pause_until_midnight":
                result.update(self._pause_until_midnight())
            elif remedy == "skip":
                # Known non-critical error, already handled in code,  just log it
                logger.debug(f"🩺 Known benign error '{error_pattern.name}', skipping remedy")
                result.update({"action": "skip", "applied": True})
            elif remedy == "grok_custom_fix":
                result.update(await self._grok_custom_fix(error_context))
            else:
                result = {"applied": False, "reason": f"unknown remedy: {remedy}"}
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"🔧 Remedy {remedy} failed: {e}")

        # Log it
        self._log_heal(error_pattern.name, remedy, result, error_context)
        return result

    async def _emergency_stealth(self, params: Dict, error_context: str) -> Dict:
        """
        CRITICAL: Error 226 detected. Go into stealth mode:
        1. Pause ALL posting loops for N minutes
        2. Reduce all activity probabilities drastically
        3. Increase all delays significantly
        4. Switch to mobile UA
        5. Only do passive browsing (no writes) for a while
        """
        agent = self._agent
        pause_min = params.get("pause_minutes", 30)
        reduce = params.get("reduce_activity", 0.3)

        # Save originals if not saved
        if "post_interval" not in self._original_values:
            self._original_values.update({
                "post_interval": agent.post_interval,
                "engage_interval": agent.engage_interval,
                "p_reply": agent.p_reply,
                "p_like": agent.p_like,
                "p_retweet": agent.p_retweet,
                "p_quote": agent.p_quote,
                "p_follow": agent.p_follow,
            })

        # 1. Massively increase intervals
        agent.post_interval = max(agent.post_interval * 3, 5400)  # Min 1.5 hours
        agent.engage_interval = max(agent.engage_interval * 3, 900)  # Min 15 min

        # 2. Slash all action probabilities
        agent.p_reply *= reduce
        agent.p_like *= reduce
        agent.p_retweet *= reduce
        agent.p_quote *= reduce
        agent.p_follow *= (reduce * 0.5)  # Extra cautious with follows

        # 3. Pause posting loop
        resume_at = datetime.now() + timedelta(minutes=pause_min)
        self._paused_loops["post"] = resume_at
        self._paused_loops["trend"] = resume_at
        self._paused_loops["mention"] = resume_at

        # 4. Switch UA
        new_ua = pick_user_agent(prefer_mobile=True)
        self._apply_user_agent(new_ua)

        # 5. Increase action delays
        x_skill = agent._x()
        impl = getattr(x_skill, '_impl', x_skill) if x_skill else None
        if impl and hasattr(impl, '_action_delay'):
            if "_action_delay" not in self._original_values:
                self._original_values["_action_delay"] = impl._action_delay
            impl._action_delay = max(impl._action_delay * 3, 8)

        logger.warning(
            f"🛡️ STEALTH MODE activated,  pausing posts for {pause_min}min, "
            f"reducing activity to {reduce*100:.0f}%, intervals tripled"
        )

        # Ask Grok for additional insight
        grok_advice = await self._consult_grok_on_error(error_context, "emergency_stealth")

        return {
            "action": "emergency_stealth",
            "pause_until": resume_at.isoformat(),
            "intervals": {"post": agent.post_interval, "engage": agent.engage_interval},
            "probability_multiplier": reduce,
            "new_ua": new_ua[:50],
            "grok_advice": grok_advice,
        }

    async def _rate_limit_backoff(self, params: Dict) -> Dict:
        """Exponential backoff on rate limits."""
        agent = self._agent
        backoff_min = params.get("backoff_minutes", 15)

        # Increase delays
        agent.engage_interval = min(agent.engage_interval * 2, 3600)
        x_skill = agent._x()
        impl = getattr(x_skill, '_impl', x_skill) if x_skill else None
        if impl and hasattr(impl, '_action_delay'):
            impl._action_delay = min(impl._action_delay * 2, 15)

        resume_at = datetime.now() + timedelta(minutes=backoff_min)
        self._paused_loops["engage"] = resume_at

        logger.warning(f"⏸️ Rate limit backoff,  engage paused for {backoff_min}min")
        return {"action": "rate_limit_backoff", "resume_at": resume_at.isoformat()}

    async def _refresh_auth(self) -> Dict:
        """Try to re-authenticate by reloading cookies."""
        x_skill = self._agent._x()
        if not x_skill:
            return {"action": "refresh_auth", "success": False}

        _profile = os.environ.get("_SABLE_PROFILE", "")
        _cookie_name = f"x_cookies_{_profile}.json" if _profile else "x_cookies.json"
        cookies_path = Path.home() / ".opensable" / _cookie_name
        if cookies_path.exists():
            try:
                impl = getattr(x_skill, '_impl', x_skill)
                client = getattr(impl, '_client', None)
                if client:
                    client.load_cookies(str(cookies_path))
                    logger.info("🔧 Reloaded cookies from disk")
                    return {"action": "refresh_auth", "success": True}
            except Exception as e:
                logger.error(f"Cookie reload failed: {e}")

        return {"action": "refresh_auth", "success": False, "note": "manual cookie update needed"}

    async def _rotate_identity(self, params: Dict) -> Dict:
        """Change user agent and pause briefly."""
        pause_min = params.get("pause_minutes", 10)
        new_ua = pick_user_agent(prefer_mobile=True)
        self._apply_user_agent(new_ua)

        resume_at = datetime.now() + timedelta(minutes=pause_min)
        self._paused_loops["post"] = resume_at
        self._paused_loops["engage"] = resume_at

        logger.info(f"🔄 Rotated UA and pausing for {pause_min}min")
        return {"action": "rotate_identity", "new_ua": new_ua[:50], "resume_at": resume_at.isoformat()}

    def _disable_search(self, params: Dict) -> Dict:
        """Temporarily disable search (stale GraphQL hash)."""
        disable_min = params.get("disable_minutes", 60)
        self._search_disabled_until = datetime.now() + timedelta(minutes=disable_min)
        logger.info(f"🔍 Search disabled for {disable_min}min (404 endpoint)")
        return {"action": "disable_search", "until": self._search_disabled_until.isoformat()}

    async def _network_backoff(self, params: Dict) -> Dict:
        """Short pause on network errors."""
        backoff_min = params.get("backoff_minutes", 5)
        await asyncio.sleep(backoff_min * 60)
        logger.info(f"🌐 Network backoff complete ({backoff_min}min)")
        return {"action": "network_backoff", "waited_minutes": backoff_min}

    def _fallback_llm(self) -> Dict:
        """Note that Grok failed,  LLM fallback is already built-in."""
        logger.info("🔧 Grok unavailable,  using LLM fallback")
        return {"action": "fallback_llm", "note": "LLM fallback active"}

    async def _emergency_pause(self, params: Dict) -> Dict:
        """Full stop,  serious account issue."""
        pause_min = params.get("pause_minutes", 120)
        platform = params.get("platform", "X/Twitter")
        resume_at = datetime.now() + timedelta(minutes=pause_min)

        for loop in ["post", "engage", "trend", "mention"]:
            self._paused_loops[loop] = resume_at

        logger.critical(
            f"🚨 EMERGENCY PAUSE [{platform}],  ALL loops stopped for {pause_min}min. "
            f"Account may be flagged. Check manually."
        )

        # Ask Grok for advice
        grok_advice = await self._consult_grok_on_error(
            "Account permissions error / possible suspension", "emergency_pause"
        )

        return {
            "action": "emergency_pause",
            "resume_at": resume_at.isoformat(),
            "grok_advice": grok_advice,
        }

    def _pause_until_midnight(self) -> Dict:
        """Twitter 344: daily post limit hit.
        Pause post/trend loops for _limit_retry_hours (default 3h) so that
        the self-heal pause and the autoposter's _daily_limit_hit cooldown
        expire at the same time and don't fight each other.
        """
        agent = self._agent
        retry_hours = getattr(agent, '_limit_retry_hours', 3.0)
        resume_at = datetime.now() + timedelta(hours=retry_hours)
        for loop in ["post", "trend"]:
            self._paused_loops[loop] = resume_at
        # Set the flag + timestamp on the agent so _should_post_now also tracks it
        if hasattr(agent, '_daily_limit_hit'):
            agent._daily_limit_hit = True
        if hasattr(agent, '_daily_limit_hit_at'):
            agent._daily_limit_hit_at = datetime.now()
        logger.warning(
            f"\U0001f6ab Daily post limit (344),  posting paused for {retry_hours:.0f}h "
            f"(resumes ~{resume_at.strftime('%H:%M')}). Engagement continues normally."
        )
        return {
            "action": "pause_until_retry",
            "resume_at": resume_at.isoformat(),
            "loops_paused": ["post", "trend"],
            "retry_hours": retry_hours,
        }

    async def _grok_custom_fix(self, error_context: str) -> Dict:
        """For unknown errors,  ask Grok, then parse and execute concrete actions."""
        advice = await self._consult_grok_on_error(error_context, "unknown_error")
        actions_taken = []

        if not advice or "failed" in advice.lower():
            return {"action": "grok_custom_fix", "advice": advice, "actions_taken": []}

        advice_lower = advice.lower()

        # Parse Grok's advice and execute matching safe actions

        # 1. Pause/wait recommendation
        pause_match = re.search(r'(?:pause|wait|stop|back.?off|cool.?down).*?(\d+)\s*(?:min|hour|hr)',
                                advice_lower)
        if pause_match:
            pause_min = int(pause_match.group(1))
            if 'hour' in pause_match.group(0) or 'hr' in pause_match.group(0):
                pause_min *= 60
            pause_min = min(pause_min, 240)  # Cap at 4 hours
            resume_at = datetime.now() + timedelta(minutes=pause_min)
            for loop in ["post", "engage", "trend", "mention"]:
                self._paused_loops[loop] = resume_at
            actions_taken.append(f"paused_all_loops_{pause_min}min")
            logger.info(f"🧠 Grok says pause → pausing all loops for {pause_min}min")

        # 2. Reduce activity recommendation
        if re.search(r'reduce|slow\s*down|less.?frequent|lower.?rate|decrease', advice_lower):
            agent = self._agent
            for attr in ('p_post', 'p_engage', 'p_follow'):
                val = getattr(agent, attr, None)
                if val is not None:
                    setattr(agent, attr, val * 0.5)
            actions_taken.append("reduced_activity_50%")
            logger.info("🧠 Grok says reduce activity → halved all probabilities")

        # 3. Disable search recommendation
        if re.search(r'disable.?search|skip.?search|stop.?search|avoid.?search|don.?t.?search',
                      advice_lower):
            self._search_disabled_until = datetime.now() + timedelta(minutes=60)
            actions_taken.append("disabled_search_60min")
            logger.info("🧠 Grok says disable search → search off for 60min")

        # 4. Rotate UA recommendation
        if re.search(r'change.*user.?agent|rotate.*ua|switch.*ua|new.*user.?agent', advice_lower):
            new_ua = pick_user_agent(prefer_mobile=True)
            self._apply_user_agent(new_ua)
            actions_taken.append("rotated_ua")
            logger.info("🧠 Grok says rotate UA → done")

        # 5. Increase delays recommendation
        if re.search(r'increase.*delay|longer.*delay|more.*time.*between|slow.*request', advice_lower):
            x_skill = self._agent._x()
            impl = getattr(x_skill, '_impl', x_skill) if x_skill else None
            if impl and hasattr(impl, '_action_delay'):
                old_delay = impl._action_delay
                impl._action_delay = min(old_delay * 2, 30)
                actions_taken.append(f"doubled_delay_{old_delay:.1f}s→{impl._action_delay:.1f}s")
                logger.info(f"🧠 Grok says increase delays → {old_delay:.1f}s→{impl._action_delay:.1f}s")

        # If we couldn't parse any concrete action, apply a safe default
        if not actions_taken:
            # Default: short pause + minor rate reduction
            resume_at = datetime.now() + timedelta(minutes=10)
            self._paused_loops["post"] = resume_at
            self._paused_loops["engage"] = resume_at
            actions_taken.append("default_pause_10min")
            logger.info("🧠 Grok advice unclear → applying safe default: 10min pause")

        logger.info(f"🛠️ Self-heal actions from Grok advice: {', '.join(actions_taken)}")

        return {
            "action": "grok_custom_fix",
            "advice": advice,
            "actions_taken": actions_taken,
        }

    # ── Helpers ───────────────────────────────────────────────────────

    def _apply_user_agent(self, ua: str):
        """Apply a new user agent to the twikit client."""
        x_skill = self._agent._x()
        if not x_skill:
            return
        # Navigate through wrapper: tools.x_skill._impl._client
        client = getattr(x_skill, '_client', None) or getattr(getattr(x_skill, '_impl', None), '_client', None)
        if client:
            client._user_agent = ua
            # Works with both httpx and our TwikitCurlSession wrapper
            http = getattr(client, 'http', None)
            if http:
                if hasattr(http, 'headers') and isinstance(http.headers, dict):
                    http.headers["user-agent"] = ua
                elif hasattr(http, '_session') and hasattr(http._session, 'headers'):
                    http._session.headers["user-agent"] = ua
            logger.info(f"🔧 UA changed to: {ua[:60]}...")

    async def _consult_grok_on_error(self, error_context: str, remedy_applied: str) -> str:
        """Ask Grok for diagnosis of an error."""
        try:
            agent = self._agent
            system = (
                "You are an expert at X/Twitter automation debugging. "
                "You're monitoring a bot that posts on X using twikit. "
                "Analyze this error and suggest what to do. Be CONCISE (max 200 words). "
                "Focus on: Is this a temporary issue or permanent? What should the bot adjust? "
                "Any specific timing/behavior changes needed?"
            )
            user_msg = (
                f"Error detected in my X bot console:\n\n{error_context[:1500]}\n\n"
                f"Remedy already applied: {remedy_applied}\n\n"
                f"Current state: posts today={agent._posts_today}, "
                f"engagements today={agent._engagements_today}, "
                f"post_interval={agent.post_interval}s, "
                f"engage_interval={agent.engage_interval}s\n\n"
                f"What else should I do? Should I change my approach?"
            )
            # Try Grok first
            text = await agent._ask_grok(system, user_msg)
            if not text:
                text = await agent._ask_llm(system, user_msg)
            return text[:500] if text else "No AI advice available"
        except Exception as e:
            return f"Grok consultation failed: {e}"

    def is_loop_paused(self, loop_name: str) -> bool:
        """Check if a loop is currently paused by a remedy."""
        resume_at = self._paused_loops.get(loop_name)
        if resume_at is None:
            return False
        if datetime.now() >= resume_at:
            del self._paused_loops[loop_name]
            logger.info(f"▶️ Loop '{loop_name}' resumed after pause")
            return False
        return True

    def is_search_disabled(self) -> bool:
        """Check if search is temporarily disabled."""
        if self._search_disabled_until is None:
            return False
        if datetime.now() >= self._search_disabled_until:
            self._search_disabled_until = None
            logger.info("🔍 Search re-enabled")
            return False
        return True

    def get_pause_status(self) -> Dict:
        """Get current pause status of all loops."""
        now = datetime.now()
        status = {}
        for loop, resume_at in self._paused_loops.items():
            remaining = (resume_at - now).total_seconds()
            status[loop] = {
                "paused": remaining > 0,
                "resume_at": resume_at.isoformat(),
                "remaining_seconds": max(0, remaining),
            }
        return status

    async def restore_originals(self):
        """Restore original values after stealth period ends."""
        agent = self._agent
        restored = []
        for key, val in self._original_values.items():
            if key == "_action_delay":
                x_skill = agent._x()
                impl = getattr(x_skill, '_impl', x_skill) if x_skill else None
                if impl:
                    impl._action_delay = val
                    restored.append(key)
            elif hasattr(agent, key):
                setattr(agent, key, val)
                restored.append(key)
        if restored:
            self._original_values.clear()
            logger.info(f"🔧 Restored original values: {', '.join(restored)}")
        return restored

    def _log_heal(self, error_name: str, remedy: str, result: Dict, context: str):
        """Persist healing action to disk."""
        entry = {
            "ts": datetime.now().isoformat(),
            "error": error_name,
            "remedy": remedy,
            "result": {k: v for k, v in result.items() if k != "grok_advice"},
            "grok_advice": result.get("grok_advice", "")[:300],
            "context_snippet": context[:200],
        }
        self._heal_log.append(entry)
        try:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    def get_heal_stats(self) -> Dict:
        return {
            "total_heals": len(self._heal_log),
            "active_pauses": {k: v.isoformat() for k, v in self._paused_loops.items()
                              if datetime.now() < v},
            "search_disabled": self.is_search_disabled(),
            "remedies_applied": {name: ts.isoformat() for name, ts in self._applied.items()},
            "original_values_saved": bool(self._original_values),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-HEAL MONITOR,  The main loop that ties everything together
# ══════════════════════════════════════════════════════════════════════════════

class SelfHealMonitor:
    """
    The bot's self-awareness system. Watches its own console output,
    detects errors, classifies them, and orchestrates fixes.
    """

    def __init__(self, agent_ref, log_buffer: LogBuffer):
        self._agent = agent_ref
        self._buffer = log_buffer
        self._remedy = RemedyEngine(agent_ref)
        self._check_interval = 30  # Check console every 30 seconds
        self._consecutive_clean = 0  # Track clean checks for restoration
        self._grok_consult_count = 0
        self._max_grok_consults_per_hour = 5
        self._grok_consult_reset = datetime.now()

    @property
    def remedy(self) -> RemedyEngine:
        return self._remedy

    def _is_x_relevant_error(self, error_source: str, error_msg: str) -> bool:
        """Return True only for errors that plausibly belong to the X/Twitter stack."""
        text = f"{error_source} {error_msg}".lower()

        if re.search(
            r"instagram|instagrapi|ig_autoposter|whatsapp|wwebjs|telegram|arena|agentmon|"
            r"gdelt|zunvra|camera move|voice skill|pyttsx3|crm|pipeline|emailtemplates|"
            r"youtube|ollama|fm\.opensable|news\.zunvra|whatsapp-event|linkedin|facebook|tiktok",
            text,
            re.IGNORECASE,
        ):
            return False

        return bool(re.search(
            r"opensable\.core\.x_|opensable\.skills\.social|twikit|twitter|x\.com|grok\.x\.com|"
            r"create(tweet|grokconversation)|searchtimeline|hometimeline|tweet|retweet|reply|quote|"
            r"follow|cookie|user.?agent|cloudflare|might be automated|protect our users from spam|"
            r"daily limit|rate.?limit|unauthorized|forbidden|status[:\s]*(401|403|429)",
            text,
            re.IGNORECASE,
        ))

    async def run(self):
        """Main self-healing loop,  runs forever alongside other loops."""
        logger.info("🩺 Self-heal monitor active,  watching console for errors")
        await asyncio.sleep(60)  # Let things warm up

        while self._agent.running:
            try:
                await self._check_cycle()
            except Exception as e:
                logger.debug(f"Self-heal check error: {e}")

            await asyncio.sleep(self._check_interval)

    async def _check_cycle(self):
        """One monitoring cycle."""
        new_errors = self._buffer.get_new_errors()

        if not new_errors:
            self._consecutive_clean += 1
            # After 10 clean checks (~5 min), try restoring original values
            if self._consecutive_clean >= 10 and self._remedy._original_values:
                restored = await self._remedy.restore_originals()
                if restored:
                    logger.info(f"🩺 {self._consecutive_clean} clean checks,  restored normal operation")
                    self._consecutive_clean = 0
            return

        self._consecutive_clean = 0

        # Classify and handle each error
        for error_entry in new_errors:
            error_msg = error_entry.get("raw", "") + " " + error_entry.get("msg", "")
            error_source = error_entry.get("name", "")

            if not self._is_x_relevant_error(error_source, error_msg):
                logger.debug(f"🩺 Skipping non-X error from logger '{error_source}'")
                continue

            # Skip ALL errors from Instagram/instagrapi loggers,  IG autoposter
            # handles its own backoff.  These must NEVER trigger X remedies.
            if re.search(r"instagrapi|instagram|ig_autoposter", error_source, re.IGNORECASE):
                logger.debug(f"🩺 Skipping non-X error from logger '{error_source}'")
                continue

            matched = False

            for pattern in KNOWN_ERRORS:
                if re.search(pattern.pattern, error_msg, re.IGNORECASE):
                    matched = True
                    logger.info(
                        f"🩺 Detected: [{pattern.severity.value}] {pattern.name} → "
                        f"applying remedy: {pattern.remedy}"
                    )

                    # Get context around the error
                    context = self._buffer.get_recent_context(20)

                    # Apply the remedy
                    result = await self._remedy.apply(pattern, context)

                    if result.get("applied"):
                        # Think about what happened
                        if hasattr(self._agent, 'mind'):
                            self._agent.mind.remember("self_healed", {
                                "error": pattern.name,
                                "severity": pattern.severity.value,
                                "remedy": pattern.remedy,
                                "result": str(result)[:200],
                            })
                    break  # One remedy per error per cycle

            # Unknown error,  consult Grok if we haven't done too many
            # BUT skip non-X errors (Instagram, etc.),  those platforms handle their own recovery
            if not matched and self._can_consult_grok():
                # Skip errors from non-X platforms (they handle their own recovery)
                if re.search(r"Instagram|📸 IG|instagrapi|ig_autoposter|Photo Upload failed|challenge_required", error_msg, re.IGNORECASE):
                    logger.debug(f"🩺 Skipping non-X error (Instagram),  IG handles its own recovery")
                    continue
                context = self._buffer.get_recent_context(20)
                logger.info(f"🩺 Unknown error,  consulting Grok for diagnosis")
                result = await self._remedy.apply(
                    ErrorPattern(
                        name="unknown",
                        pattern="",
                        severity=Severity.MEDIUM,
                        remedy="grok_custom_fix",
                        cooldown=300,
                    ),
                    context,
                )
                self._grok_consult_count += 1

    def _can_consult_grok(self) -> bool:
        """Rate limit Grok consultations."""
        now = datetime.now()
        if (now - self._grok_consult_reset).total_seconds() > 3600:
            self._grok_consult_count = 0
            self._grok_consult_reset = now
        return self._grok_consult_count < self._max_grok_consults_per_hour

    def get_status(self) -> Dict:
        return {
            "log_stats": self._buffer.get_stats(),
            "heal_stats": self._remedy.get_heal_stats(),
            "consecutive_clean_checks": self._consecutive_clean,
            "grok_consults_this_hour": self._grok_consult_count,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SETUP HELPER,  Call this once at startup
# ══════════════════════════════════════════════════════════════════════════════

_global_log_buffer: Optional[LogBuffer] = None


def install_log_buffer() -> LogBuffer:
    """
    Install the LogBuffer as a handler on the root logger.
    Call this once at startup, before the agent starts.
    Returns the buffer for the SelfHealMonitor to read.
    """
    global _global_log_buffer
    if _global_log_buffer is not None:
        return _global_log_buffer

    buf = LogBuffer(max_entries=500, error_max=100)
    buf.setLevel(logging.WARNING)  # Only capture warnings and errors
    formatter = logging.Formatter("%(name)s - %(message)s")
    buf.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(buf)

    _global_log_buffer = buf
    logger.info("🩺 LogBuffer installed,  bot can see its own console")
    return buf


def get_log_buffer() -> Optional[LogBuffer]:
    return _global_log_buffer
