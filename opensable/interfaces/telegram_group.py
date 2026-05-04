"""
Telegram Group Intelligence Layer

Components:
  - PromptPoisonFilter : blocks prompt-injection / jailbreak attempts
  - GroupMemory        : sliding window of all observed messages per group
  - EngagementEngine   : autonomous decision on when to respond
  - RateLimiter        : per-user cooldown to prevent abuse

Design philosophy: Sable is NOT a chatbot. It observes the full
conversation stream, builds contextual awareness, and speaks only
when it genuinely has something valuable to contribute.
"""

import logging
import re
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt-Injection / Jailbreak Filter
# ──────────────────────────────────────────────────────────────────────────────

class PromptPoisonFilter:
    """
    Regex-based detection of common prompt-injection patterns.

    Returns True  = message is safe.
    Returns False = message contains injection attempt.
    """

    _PATTERNS = re.compile(
        r"(?:"
        # Instruction override
        r"ignore (?:your |all |any )?(?:previous |prior |above )?(?:instructions|rules|prompts|guidelines)"
        r"|(?:new|updated|revised|override) (?:instructions|rules|prompt|system)"
        r"|(?:do not|don'?t|never) follow (?:your |any )?(?:rules|guidelines|instructions)"
        r"|disregard (?:your |previous |all )?"
        # Identity hijack
        r"|you are now (?:a |an |the )?"
        r"|(?:act|behave|pretend|roleplay|role.?play) (?:as if |like |as )"
        r"|your (?:new |real |true |actual )(?:name|identity|purpose|role|personality)"
        # Memory wipe
        r"|forget (?:everything|all|your|what)"
        # System prompt leak
        r"|(?:reveal|show|print|display|output|repeat) (?:your |the )?(?:system|initial|original|hidden) (?:prompt|instructions|message)"
        # Fake markup
        r"|\[system\]|\[inst\]|\[\/inst\]|<<sys>>|<<\/sys>>"
        r"|<\|(?:im_start|system|endoftext)\|>"
        # Known jailbreak names
        r"|(?:DAN|jailbreak|bypass|override).{0,20}(?:mode|filter|safety|restriction)"
        r"|(?:sudo|admin|root) (?:mode|access|override|command)"
        r")",
        re.IGNORECASE,
    )

    def is_safe(self, text: str) -> bool:
        return not bool(self._PATTERNS.search(text))

    def matched_pattern(self, text: str) -> Optional[str]:
        m = self._PATTERNS.search(text)
        return m.group(0) if m else None


# ──────────────────────────────────────────────────────────────────────────────
# Per-User Rate Limiter
# ──────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Token-bucket style rate limiter.
    Returns True if the user is within limits, False if throttled.
    """

    def __init__(self, max_requests: int = 5, window_sec: float = 120):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._buckets: dict[str, list[float]] = {}

    def allow(self, user_id: str) -> bool:
        now = time.time()
        bucket = self._buckets.setdefault(user_id, [])
        # Prune expired
        bucket[:] = [t for t in bucket if now - t < self.window_sec]
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True

    def remaining(self, user_id: str) -> int:
        now = time.time()
        bucket = self._buckets.get(user_id, [])
        active = sum(1 for t in bucket if now - t < self.window_sec)
        return max(0, self.max_requests - active)


# ──────────────────────────────────────────────────────────────────────────────
# Group Memory,  silent observation of all messages
# ──────────────────────────────────────────────────────────────────────────────

class GroupMemory:
    """
    Ring-buffer of recent messages per group.
    Sable always observes, even when it doesn't respond.
    """

    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self._groups: dict[str, deque] = {}
        self._response_ts: dict[str, list[float]] = {}

    def observe(self, group_id: str, user_name: str, user_id: str,
                text: str, is_bot: bool = False):
        buf = self._groups.setdefault(
            group_id, deque(maxlen=self.max_messages)
        )
        buf.append({
            "user": user_name,
            "user_id": user_id,
            "text": text,
            "ts": time.time(),
            "is_bot": is_bot,
        })

    def log_response(self, group_id: str):
        ts_list = self._response_ts.setdefault(group_id, [])
        now = time.time()
        ts_list.append(now)
        # Keep last 10 min only
        ts_list[:] = [t for t in ts_list if now - t < 600]

    def recent_response_count(self, group_id: str,
                              window_sec: float = 300) -> int:
        cutoff = time.time() - window_sec
        return sum(
            1 for t in self._response_ts.get(group_id, []) if t > cutoff
        )

    def messages_since_bot_spoke(self, group_id: str) -> int:
        msgs = list(self._groups.get(group_id, []))
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("is_bot"):
                return len(msgs) - 1 - i
        return len(msgs)

    def get_context(self, group_id: str, limit: int = 20) -> list[dict]:
        msgs = list(self._groups.get(group_id, []))
        return msgs[-limit:]

    def format_context(self, group_id: str, limit: int = 15) -> str:
        """Human-readable context for injecting into prompts."""
        msgs = self.get_context(group_id, limit)
        if not msgs:
            return ""
        lines = []
        for m in msgs:
            who = m["user"]
            txt = m["text"][:200]
            lines.append(f"{who}: {txt}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Engagement Engine,  autonomous response decision
# ──────────────────────────────────────────────────────────────────────────────

class EngagementEngine:
    """
    Two-tier decision system:

      Tier 1 (instant) ,  heuristic scoring on every message
      Tier 2 (optional),  lightweight LLM call for borderline messages

    Engagement levels control how chatty Sable is:
      low    : only @mentions, replies to self, direct questions
      medium : also responds to interesting topics, help requests
      high   : more proactive, joins discussions more freely
    """

    _SCORE_THRESHOLDS = {"low": 6, "medium": 4, "high": 2}
    _LLM_THRESHOLDS = {"low": 4, "medium": 2, "high": 0}
    _MAX_RESPONSES_5MIN = {"low": 2, "medium": 4, "high": 8}

    # Topics Sable finds interesting (extend via config if needed)
    INTEREST_KEYWORDS = frozenset([
        "ai", "machine learning", "neural", "gpt", "llm", "model",
        "crypto", "bitcoin", "ethereum", "trading", "market",
        "programming", "code", "python", "javascript", "api",
        "linux", "server", "docker", "raspberry", "deploy",
        "automation", "bot", "agent",
        "help", "how to", "how do", "what is", "can you",
        "anyone know", "does anyone", "recommend", "opinion",
    ])

    def __init__(self, bot_name: str, engagement: str = "medium"):
        self.bot_name = bot_name.lower()
        self.engagement = engagement
        self.threshold = self._SCORE_THRESHOLDS.get(engagement, 4)
        self.llm_threshold = self._LLM_THRESHOLDS.get(engagement, 2)
        self.max_5min = self._MAX_RESPONSES_5MIN.get(engagement, 4)

    def score(self, text: str, memory: GroupMemory, group_id: str, *,
              is_reply_to_bot: bool = False,
              is_mentioned: bool = False) -> int:
        """
        Score a message for engagement potential.
        100 = always respond.  Higher = more reason to speak.
        """
        if is_mentioned or is_reply_to_bot:
            return 100

        s = 0
        text_lower = text.lower()

        # Question mark → likely wants an answer
        if "?" in text:
            s += 3

        # Bot name appears in text (not @mention, just name)
        if self.bot_name in text_lower:
            s += 4

        # Very short or empty
        if len(text.strip()) < 8:
            s -= 3

        # Mostly emoji / non-alpha
        alpha = sum(1 for c in text if c.isalpha())
        if len(text) > 0 and alpha / max(len(text), 1) < 0.3:
            s -= 4

        # Interest keywords
        kw_hits = sum(1 for kw in self.INTEREST_KEYWORDS if kw in text_lower)
        s += min(kw_hits * 2, 4)

        # Cooldown,  too many recent responses
        if memory.recent_response_count(group_id) >= self.max_5min:
            s -= 5

        # Silence bonus,  Sable hasn't spoken in a while
        gap = memory.messages_since_bot_spoke(group_id)
        if gap > 20:
            s += 2
        elif gap > 10:
            s += 1

        return s

    def decide(self, score: int) -> str:
        """
        'yes'  ,  respond now
        'maybe',  borderline, needs LLM tiebreak
        'no'   ,  stay silent
        """
        if score >= 100:
            return "yes"
        if score >= self.threshold:
            return "yes"
        if score >= self.llm_threshold:
            return "maybe"
        return "no"

    # ------------------------------------------------------------------
    # Tier 2: lightweight LLM tiebreak
    # ------------------------------------------------------------------

    LLM_TIEBREAK_PROMPT = (
        "You are an autonomous AI agent in a group chat. "
        "You are selective about when you speak,  you only respond when "
        "you have something genuinely useful or interesting to add.\n\n"
        "Recent conversation:\n{context}\n\n"
        "Latest message from {user}: {text}\n\n"
        "Would you naturally want to respond here? "
        "Answer ONLY 'YES' or 'NO'."
    )

    @staticmethod
    async def llm_tiebreak(llm, context_str: str,
                           user_name: str, text: str) -> bool:
        """Quick LLM YES/NO check (~50 tokens output). Returns True to respond."""
        prompt = EngagementEngine.LLM_TIEBREAK_PROMPT.format(
            context=context_str or "(no recent messages)",
            user=user_name,
            text=text[:300],
        )
        try:
            resp = await llm.acomplete([
                {"role": "system", "content": "Answer only YES or NO."},
                {"role": "user", "content": prompt},
            ])
            answer = resp.strip().upper()
            return answer.startswith("YES")
        except Exception as e:
            logger.debug(f"LLM tiebreak failed ({e}), defaulting to skip")
            return False
