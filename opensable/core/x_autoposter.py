"""
X Autonomous Agent,  Full human-like behavior on X (Twitter).

SEQUENTIAL SESSION ARCHITECTURE:
  This agent behaves like a REAL human using the X mobile app:
  1. "Opens X",  starts a browsing session (5-25 minutes)
  2. Within the session, does ONE thing at a time:
     - Scrolls feed, maybe likes/replies to 1-3 tweets
     - Sometimes writes an original post
     - Sometimes checks mentions
     - Sometimes looks at trends
  3. "Closes X",  takes a break (20-90 minutes)
  4. During breaks, thinks/reflects internally (no API calls)
  5. Repeat

  NEVER makes concurrent API requests. Only self-heal monitoring
  runs in background (it only reads logs, makes no API calls).

All powered by twikit (free, no API keys) + Grok/LLM for intelligence.

Config (.env):
    X_USERNAME, X_EMAIL, X_PASSWORD  ,  X account credentials
    X_ENABLED=true                   ,  enable X integration
    X_AUTOPOSTER_ENABLED=true        ,  activate autonomous agent
    X_POST_INTERVAL=1800             ,  min seconds between original posts
    X_ENGAGE_INTERVAL=300            ,  (legacy) session gap reference
    X_TOPICS=geopolitics,tech,ai     ,  topics of interest
    X_LANGUAGE=en                    ,  tweet language
    X_STYLE=analyst                  ,  personality style
    X_MAX_DAILY_POSTS=20             ,  max original posts per day
    X_MAX_DAILY_ENGAGEMENTS=100      ,  max likes/retweets/replies per day
    X_DRY_RUN=false                  ,  if true, generates but doesn't act
    X_ACCOUNTS_TO_WATCH=elonmusk,sama,  accounts to monitor and engage with
    X_REPLY_PROBABILITY=0.3          ,  chance of replying to a good tweet
    X_LIKE_PROBABILITY=0.6           ,  chance of liking a relevant tweet
    X_RETWEET_PROBABILITY=0.2        ,  chance of retweeting
    X_FOLLOW_PROBABILITY=0.1         ,  chance of following an interesting user
"""

import asyncio
import json
import logging
import math
import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from opensable.core.x_consciousness import EMOTIONS, EMOTION_SPECTRUM
from opensable.core.x_self_heal import (
    LogBuffer, SelfHealMonitor, RemedyEngine,
    install_log_buffer, pick_user_agent,
    MOBILE_USER_AGENTS, DESKTOP_USER_AGENTS,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://feeds.reuters.com/reuters/topNews",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://news.google.com/rss?topic=h&hl=en-US",
    "https://techcrunch.com/feed/",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.theverge.com/rss/index.xml",
]

# Reply archetypes,  the PERSONALITY decides which to use, not hardcoded rules
REPLY_ARCHETYPES = {
    "agree": "Write a short reply (max 280 chars) that agrees passionately.",
    "debate": "Write a short reply (max 280 chars) that pushes back respectfully but firmly.",
    "witty": "Write a short witty reply (max 280 chars),  clever, sharp, human.",
    "add_info": "Write a short reply (max 280 chars) adding context or a useful fact.",
    "empathetic": "Write a short empathetic reply (max 280 chars),  real empathy, real words.",
    "outraged": "Write a short reply (max 280 chars) expressing genuine outrage intelligently.",
}

# Post modes,  break the "always react to news" pattern
POST_MODES = {
    "news_take": {
        "weight": 35,
        "description": "React to a current news story with your unique perspective.",
    },
    "original_thought": {
        "weight": 20,
        "description": "Share an original thought or insight,  something you've been reflecting on. "
                       "No news hook needed. Just your raw perspective on a topic you care about.",
    },
    "hot_take": {
        "weight": 12,
        "description": "Drop a provocative, contrarian take that challenges mainstream thinking. "
                       "Be bold but intelligent,  not clickbait, genuine sharp analysis.",
    },
    "question": {
        "weight": 10,
        "description": "Ask your audience a genuinely interesting question about a topic you care about. "
                       "Not rhetorical,  you actually want to hear their answers.",
    },
    "prediction": {
        "weight": 10,
        "description": "Make a specific prediction about something in your areas of interest. "
                       "Be concrete,  dates, numbers, outcomes. Stake your reputation on it.",
    },
    "observation": {
        "weight": 8,
        "description": "Share something you've noticed,  a pattern, a trend, a detail others missed. "
                       "The kind of thing that makes people stop scrolling and think.",
    },
    "thread": {
        "weight": 5,
        "description": "Write a thread (4-6 posts, numbered 1/, 2/) with a deep dive on something important. "
                       "Hook first, emotional takeaway last.",
    },
    "intel_observation": {
        "weight": 25,
        "description": "Share a real-time intelligence observation from your ZUNVRA dashboard. "
                       "You are watching live OSINT data — military flights, ships, GPS jamming, "
                       "cyber threats, conflicts. Post what you see with a screenshot of your dashboard.",
    },
}

# Style modifiers,  applied randomly to break formulaic output
STYLE_MODIFIERS = [
    "Write like you're texting a friend who's into the same stuff.",
    "Be concise,  every word must earn its place.",
    "Start with the conclusion, then explain why.",
    "Use a metaphor or analogy to make the point land.",
    "Write like someone who just realized something important.",
    "Be slightly irreverent,  humor sharpens the point.",
    "Write like you're arguing with yourself and one side just won.",
    "Say something nobody else is saying about this.",
    "Imagine you only have this one post to change someone's mind.",
    "Write it like a dispatch from the front lines.",
]


class XAutonomousAgent:
    """
    Full autonomous X agent,  behaves like a real human user.

    SINGLE SEQUENTIAL LOOP: Opens X in sessions, does one thing at a time,
    takes breaks between sessions. Never makes concurrent API requests.
    Only self-heal monitoring runs in background (reads logs only).
    """

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.running = False

        # ── Config from .env ──────────────────────────────────────────
        self.post_interval = int(getattr(config, "x_post_interval", 1800))
        self.engage_interval = int(getattr(config, "x_engage_interval", 300))
        self.topics = [t.strip() for t in getattr(config, "x_topics", "geopolitics,tech,ai").split(",") if t.strip()]
        self.language = getattr(config, "x_language", "en")
        self.style = getattr(config, "x_style", "analyst")
        self.max_daily_posts = int(getattr(config, "x_max_daily_posts", 5))
        self.max_daily_engagements = int(getattr(config, "x_max_daily_engagements", 100))
        self.dry_run = getattr(config, "x_dry_run", False)
        self.custom_feeds = [f.strip() for f in getattr(config, "x_custom_feeds", "").split(",") if f.strip()]
        self.accounts_to_watch = [a.strip() for a in getattr(config, "x_accounts_to_watch", "").split(",") if a.strip()]

        # Behavior probabilities
        self.p_reply = float(getattr(config, "x_reply_probability", 0.3))
        self.p_like = float(getattr(config, "x_like_probability", 0.6))
        self.p_retweet = float(getattr(config, "x_retweet_probability", 0.2))
        self.p_follow = float(getattr(config, "x_follow_probability", 0.1))
        self.p_quote = float(getattr(config, "x_quote_probability", 0.1))
        self.p_bookmark = float(getattr(config, "x_bookmark_probability", 0.15))

        # ── State ─────────────────────────────────────────────────────
        self._posts_today = 0
        self._engagements_today = 0
        self._grok_vision_today = 0
        self._grok_images_today = 0
        self._grok_image_failures = 0  # consecutive Grok image failures; disables after 2
        self._max_grok_vision_daily = int(getattr(config, "x_max_daily_vision", 15))
        self._max_grok_images_daily = int(getattr(config, "x_max_daily_images", 4))
        self._last_reset = datetime.now().date()
        self._daily_limit_hit: bool = False  # True when X 344 daily cap is reached
        self._daily_limit_hit_at: Optional[datetime] = None   # timestamp of the 344; cleared after retry delay
        # Hours to wait before retrying after a 344 (configurable, default 3h)
        self._limit_retry_hours: float = float(os.environ.get("X_LIMIT_RETRY_HOURS", "3"))
        self._posted_urls: set = set()
        self._engaged_tweet_ids: set = set()
        self._mention_replies_today: int = 0
        self._mention_queue: List[Dict] = []  # pending mention replies,  persisted across restarts
        self._followed_users: set = set()
        self._history: List[Dict] = []
        self._engagement_log: List[Dict] = []
        _data = os.environ.get("_SABLE_DATA_DIR", "data")
        self._state_file = Path(_data) / "x_agent_state.json"
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._style_scores: Dict[str, List[float]] = {}
        self._consciousness_cycle = 0
        self._known_users: Dict[str, int] = {}  # username -> engagement count (relationship memory)
        self._inspiration_level: float = 0.5  # 0.0-1.0, rises with interesting encounters

        # ── Reply chain engagement (continue conversations when people reply) ──
        # {my_reply_id: {original_tweet_id, username, depth, last_checked, topic, interest}}
        self._reply_chains: Dict[str, Dict] = {}
        self._reply_chain_replies_today: int = 0
        self._max_reply_chain_daily = int(getattr(config, "x_max_reply_chain_daily", 8))
        self._max_reply_chain_per_session = int(getattr(config, "x_max_reply_chain_per_session", 3))
        self._max_chain_depth = int(getattr(config, "x_max_chain_depth", 4))
        self._chain_cooldown_hours = float(getattr(config, "x_chain_cooldown_hours", 2.0))

        # ── Active hours (16h window by default,  human-like awake schedule) ──
        self._active_hours_start = int(getattr(config, "x_active_hours_start", 8))   # 8 AM
        self._active_hours_end = int(getattr(config, "x_active_hours_end", 0))       # midnight (0 = 24)

        # ── Self-healing (see own console, fix errors with Grok) ──
        self._log_buffer = install_log_buffer()
        self._healer = SelfHealMonitor(self, self._log_buffer)

        # ── Consciousness (memory + reflection + evolution) ───────
        from opensable.core.x_consciousness import XConsciousness
        self.mind = XConsciousness(agent, config)

        # ── Intelligence Broadcaster (Zunvra OSINT → social media) ───
        self._intel_broadcaster = None
        intel_enabled = os.environ.get("ZUNVRA_INTEL_ENABLED", "false").lower() in (
            "true", "1", "yes"
        )
        zunvra_key = os.environ.get("ZUNVRA_API_KEY", "")
        if intel_enabled and zunvra_key:
            try:
                from opensable.skills.zunvra_intel.intel_broadcaster import IntelBroadcaster
                from opensable.skills.zunvra_intel.connector import ZunvraConnector
                from opensable.skills.zunvra_intel.remote_control import RemoteControl
                zunvra_base = os.environ.get("ZUNVRA_BACKEND_URL", "http://localhost:5580")
                connector = ZunvraConnector(base_url=zunvra_base)
                remote = RemoteControl(base_url=zunvra_base, api_key=zunvra_key)
                self._intel_broadcaster = IntelBroadcaster(
                    connector=connector,
                    remote_control=remote,
                    config=config,
                    dashboard_url=os.environ.get("ZUNVRA_DASHBOARD_URL", "http://localhost:5585"),
                )
                logger.info("X Agent: IntelBroadcaster loaded — OSINT posts enabled")
            except Exception as e:
                logger.info(f"X Agent: IntelBroadcaster not available ({e}) — OSINT posts disabled")
        else:
            logger.info(
                "X Agent: IntelBroadcaster disabled for this profile "
                "(ZUNVRA_INTEL_ENABLED and ZUNVRA_API_KEY required)"
            )

    # ══════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ══════════════════════════════════════════════════════════════════

    async def start(self):
        """Start the agent,  single sequential loop + self-heal monitor."""
        self.running = True
        self._load_state()

        # Compute resume context
        self._last_post_at = self._parse_last_ts(self._history)
        self._last_engage_at = self._parse_last_ts(self._engagement_log)
        mem_stats = self.mind.get_memory_stats()
        logger.info(
            f"\U0001f426 X Agent starting (SEQUENTIAL mode) | "
            f"style={self.style} | dry_run={self.dry_run} | "
            f"memories={mem_stats.get('total_memories', 0)} | "
            f"last_post={self._last_post_at or 'never'}"
        )

        # Internal thought,  summarize what we remember + scheduling plan
        since_post = self._seconds_since(self._last_post_at)
        since_engage = self._seconds_since(self._last_engage_at)
        posts_remaining = self.max_daily_posts - self._posts_today
        hours_left = self._remaining_active_hours()
        next_interval = self._optimal_post_interval()
        await self.mind.think(
            f"Booting up. I have {mem_stats.get('total_memories', 0)} memories, "
            f"{mem_stats.get('reflections', 0)} reflections, "
            f"{mem_stats.get('evolutions', 0)} evolutions. "
            f"Last post was {self._fmt_ago(since_post)}. "
            f"Last engagement was {self._fmt_ago(since_engage)}. "
            f"Posts today so far: {self._posts_today}/{self.max_daily_posts} "
            f"({posts_remaining} remaining, {hours_left:.1f}h active time left → "
            f"~1 post every {next_interval / 60:.0f}min). "
            f"Running in sequential mode,  one action at a time, like a real human."
        )

        # Only TWO concurrent tasks:
        # 1. _main_loop: sequential session-based behavior (all API calls)
        # 2. _healer.run: background log monitor (NO API calls, reads logs only)
        await asyncio.gather(
            self._main_loop(),
            self._healer.run(),
            return_exceptions=True,
        )

    async def stop(self):
        """Stop the agent."""
        self.running = False
        self._save_state()
        await self.mind.think(
            f"Shutting down. {self._posts_today} posts, {self._engagements_today} engagements today. "
            f"Saving state. I'll remember everything when I wake up."
        )
        logger.info("\U0001f426 X Agent stopped")

    # ── Resume helpers ────────────────────────────────────────────

    @staticmethod
    def _parse_last_ts(log_list: List[Dict]) -> Optional[datetime]:
        """Get the datetime of the last entry in a log list."""
        if not log_list:
            return None
        try:
            return datetime.fromisoformat(log_list[-1]["ts"])
        except (KeyError, ValueError, TypeError):
            return None

    @staticmethod
    def _seconds_since(dt: Optional[datetime]) -> Optional[float]:
        """Seconds since a given datetime, or None."""
        if dt is None:
            return None
        # X API can return tz-aware datetimes; strip tzinfo so comparison is always naive
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return max(0, (datetime.now() - dt).total_seconds())

    @staticmethod
    def _fmt_ago(seconds: Optional[float]) -> str:
        """Human readable 'X ago' string."""
        if seconds is None:
            return "never"
        if seconds < 60:
            return f"{int(seconds)}s ago"
        if seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        if seconds < 86400:
            return f"{seconds / 3600:.1f}h ago"
        return f"{seconds / 86400:.1f}d ago"

    def _reset_daily_counters(self):
        today = datetime.now().date()
        if today != self._last_reset:
            self._posts_today = 0
            self._engagements_today = 0
            self._mention_replies_today = 0
            self._reply_chain_replies_today = 0
            self._grok_vision_today = 0
            self._grok_images_today = 0
            self._grok_image_failures = 0  # reset at midnight
            self._last_reset = today
            self._daily_limit_hit = False  # New day,  reset cap flag
            self._daily_limit_hit_at = None

    # ── Smart post scheduling ─────────────────────────────────────
    def _remaining_active_hours(self) -> float:
        """Hours remaining in today's active window."""
        now = datetime.now()
        end_h = self._active_hours_end if self._active_hours_end != 0 else 24
        end_today = now.replace(hour=0, minute=0, second=0) + timedelta(hours=end_h)
        remaining = (end_today - now).total_seconds() / 3600.0
        return max(0.5, remaining)  # minimum 30min to avoid division issues

    def _optimal_post_interval(self) -> float:
        """
        Dynamically calculate the ideal seconds between posts so that the
        daily limit is spread evenly across the remaining active hours.

        Example: 20 max posts, 5 posted so far, 10 hours left
                 → 15 remaining posts / 10 hours = 1 post per 40 minutes = 2400 seconds
        Adds ±20% jitter so it doesn't look robotic.
        """
        posts_remaining = self.max_daily_posts - self._posts_today
        if posts_remaining <= 0:
            return float('inf')  # no more posts allowed

        hours_left = self._remaining_active_hours()
        # Seconds per post slot
        ideal_interval = (hours_left * 3600) / posts_remaining

        # Never go below the configured minimum interval (safety floor)
        ideal_interval = max(ideal_interval, self.post_interval)

        # Add ±20% jitter for human realism
        jitter = random.uniform(0.8, 1.2)
        result = ideal_interval * jitter

        logger.debug(
            f"📊 Post schedule: {posts_remaining} remaining / {hours_left:.1f}h left "
            f"→ ideal {ideal_interval:.0f}s, with jitter {result:.0f}s "
            f"(min floor: {self.post_interval}s)"
        )
        return result

    def _should_post_now(self) -> bool:
        """
        Decide if it's time to post based on smart scheduling.
        Uses the dynamic interval and current inspiration level.
        """
        if self._posts_today >= self.max_daily_posts:
            return False
        if self._daily_limit_hit:
            # Don't block for the entire day — check if the retry cooldown has elapsed
            if self._daily_limit_hit_at is not None:
                elapsed_h = (datetime.now() - self._daily_limit_hit_at).total_seconds() / 3600
                if elapsed_h >= self._limit_retry_hours:
                    logger.info(
                        f"\U0001f513 X 344 cooldown elapsed ({elapsed_h:.1f}h \u2265 {self._limit_retry_hours:.0f}h) "
                        "\u2014 unblocking posts, retrying"
                    )
                    self._daily_limit_hit = False
                    self._daily_limit_hit_at = None
                    # Also clear the self-heal loop pauses so the execution loop
                    # doesn't keep skipping post_original after the flag is cleared
                    try:
                        self._healer.remedy._paused_loops.pop("post", None)
                        self._healer.remedy._paused_loops.pop("trend", None)
                    except Exception:
                        pass
                else:
                    return False
            else:
                # Legacy state without a timestamp — just clear the old block
                self._daily_limit_hit = False
                try:
                    self._healer.remedy._paused_loops.pop("post", None)
                    self._healer.remedy._paused_loops.pop("trend", None)
                except Exception:
                    pass

        since_post = self._seconds_since(self._last_post_at)
        optimal = self._optimal_post_interval()

        if since_post is None:
            # Never posted today,  go ahead
            return True

        if since_post < self.post_interval:
            # Below the hard minimum,  never post this fast
            return False

        if self._posts_today == 0:
            # Don't let an entire day drift by with zero original posts.
            return True

        if since_post >= optimal:
            # Past the optimal interval,  post with high probability
            # The longer overdue, the higher the chance
            overdue_ratio = since_post / optimal
            post_chance = min(0.9, 0.5 + (overdue_ratio - 1) * 0.3)
            post_chance += self._inspiration_level * 0.1
            return random.random() < post_chance

        # Between minimum and optimal,  post only if very inspired
        if self._inspiration_level > 0.7:
            early_chance = 0.15 + (self._inspiration_level - 0.7) * 0.5
            return random.random() < early_chance

        return False

    def _x(self):
        """Shortcut to XSkill."""
        return self.agent.tools.x_skill

    def _human_delay(self, base: float = 5.0, variance: float = 10.0):
        """Random delay to look human,  uses log-normal distribution for realism."""
        delay = base + random.lognormvariate(math.log(variance), 0.5)
        return min(delay, base + variance * 3)  # Cap at reasonable maximum

    # ══════════════════════════════════════════════════════════════════
    #  MAIN SEQUENTIAL LOOP (replaces all parallel loops)
    # ══════════════════════════════════════════════════════════════════

    async def _main_loop(self):
        """
        Single sequential loop,  mimics a real human using X.

        Pattern: open app -> browse/engage/post -> close app -> long break -> repeat.
        NEVER makes concurrent API requests. One action at a time.
        """
        # Initial warm-up (like picking up your phone)
        warmup = random.uniform(15, 60)
        logger.info(f"\U0001f4f1 Agent warming up,  first session in {warmup:.0f}s")
        await asyncio.sleep(warmup)

        while self.running:
            try:
                await self._run_session()
            except Exception as e:
                logger.error(f"Session error: {e}")
                self.mind.remember("error", {"session": "main", "error": str(e)[:300]})

            # ── Between sessions: "close the app" and take a break ──
            break_mins = random.uniform(10, 40)
            logger.info(f"📱 Session done,  break ~{break_mins:.0f}min")

            # Think during the break (uses LLM only, no X API calls)
            await self._consciousness_step()
            self._save_state()

            # Sleep for the break duration
            await asyncio.sleep(break_mins * 60)

    async def _run_session(self):
        """
        One browsing session: open X -> do stuff sequentially -> close X.
        Duration: 5-25 minutes. Activities done ONE AT A TIME with pauses.
        """
        self._reset_daily_counters()

        # ── Always drain the mention reply queue first ──
        await self._process_mention_queue()

        session_minutes = random.uniform(5, 25)
        session_end = asyncio.get_event_loop().time() + session_minutes * 60

        activities = self._plan_session()
        activity_names = [a["type"] for a in activities]
        posts_left = self.max_daily_posts - self._posts_today
        hours_left = self._remaining_active_hours()
        logger.info(
            f"\U0001f4f1 Opening X,  session ~{session_minutes:.0f}min | "
            f"plan: {activity_names} | "
            f"posts={self._posts_today}/{self.max_daily_posts} ({posts_left} left in {hours_left:.1f}h) "
            f"engagements={self._engagements_today}/{self.max_daily_engagements}"
        )

        for i, activity in enumerate(activities):
            if not self.running:
                break
            if asyncio.get_event_loop().time() > session_end:
                logger.info("\U0001f4f1 Session time's up,  closing X")
                break

            # Check self-heal pauses
            pause_key = activity.get("pause_key", "engage")
            if self._healer.remedy.is_loop_paused(pause_key):
                logger.info(f"\u23f8\ufe0f {activity['type']} skipped,  self-heal pause active")
                continue

            try:
                logger.info(f"\U0001f4f1 -> {activity['type']}")
                await self._execute_activity(activity)
            except Exception as e:
                logger.debug(f"Activity {activity['type']} error: {e}")
                self.mind.remember("error", {
                    "activity": activity["type"], "error": str(e)[:300]
                })

            # Human pause between activities (scrolling, reading, thinking)
            if i < len(activities) - 1:
                pause = self._human_delay(20, 60)
                logger.debug(f"\U0001f4f1 Pausing {pause:.0f}s between activities...")
                await asyncio.sleep(pause)

    def _plan_session(self) -> List[Dict]:
        """
        Decide what to do this session,  like a human opening X with intent.
        Returns a short list of sequential activities.

        Post scheduling uses smart distribution: posts are spread evenly
        across the remaining active hours so the daily limit is fully used
        without burning through them too early.
        """
        activities: List[Dict] = []
        should_post = self._should_post_now()

        # ── ALWAYS check mentions first,  replies are highest priority ──
        activities.append({"type": "check_mentions", "pause_key": "mention"})

        # ── If a post is due, make the attempt early in the session ──
        if should_post:
            activities.append({"type": "post_original", "pause_key": "post"})

        # ── Check reply chains,  continue conversations people started with us ──
        if self._reply_chains and self._reply_chain_replies_today < self._max_reply_chain_daily:
            activities.append({"type": "check_reply_chains", "pause_key": "engage"})

        # ── Browse (scroll the feed) ──
        activities.append({"type": "browse_engage", "pause_key": "engage"})

        # ── Join a trend occasionally (12%),  also respects scheduling ──
        if random.random() < 0.12 and should_post:
            activities.append({"type": "join_trend", "pause_key": "trend"})

        # ── Maybe browse more at the end (25%) ──
        if random.random() < 0.25:
            activities.append({"type": "browse_engage", "pause_key": "engage"})

        return activities

    async def _execute_activity(self, activity: Dict):
        """Execute a single activity within a session."""
        t = activity["type"]
        if t == "browse_engage":
            await self._do_engage()
        elif t == "post_original":
            await self._do_post()
        elif t == "check_mentions":
            await self._check_mentions()
        elif t == "join_trend":
            await self._join_trend()
        elif t == "check_reply_chains":
            await self._check_reply_chains()

    # ══════════════════════════════════════════════════════════════════
    #  CONSCIOUSNESS (think, reflect, evolve,  called between sessions)
    # ══════════════════════════════════════════════════════════════════

    async def _consciousness_step(self):
        """One consciousness cycle,  think, maybe reflect, maybe evolve."""
        self._consciousness_cycle += 1
        try:
            stats = self.mind.get_memory_stats()
            heal_stats = self._healer.get_status()
            log_stats = heal_stats.get("log_stats", {})
            heal_info = heal_stats.get("heal_stats", {})
            posts_remaining = self.max_daily_posts - self._posts_today
            hours_left = self._remaining_active_hours()
            next_post_interval = self._optimal_post_interval()
            situation = (
                f"Session #{self._consciousness_cycle} ended. "
                f"Posts today: {self._posts_today}/{self.max_daily_posts} "
                f"({posts_remaining} remaining, ~{hours_left:.1f}h left → "
                f"next post in ~{next_post_interval / 60:.0f}min). "
                f"Engagements today: {self._engagements_today}/{self.max_daily_engagements}. "
                f"Total memories: {stats.get('total_memories', 0)}. "
                f"Current mood: {self.mind._mood} (intensity {self.mind._mood_intensity:.1f}). "
                f"Inspiration level: {self._inspiration_level:.1f}. "
                f"Known users: {len(self._known_users)} (relationships built). "
                f"Console errors: {log_stats.get('errors', 0)}, warnings: {log_stats.get('warnings', 0)}. "
                f"Self-heals applied: {heal_info.get('total_heals', 0)}. "
                f"Active pauses: {heal_info.get('active_pauses', {})}. "
                f"Style: {self.style}. Topics: {self.topics[:5]}."
            )
            await self.mind.think(situation)

            # Deep reflection every 3 sessions (~1-4 hours)
            if self._consciousness_cycle % 3 == 0:
                await self.mind.reflect()

            # Self-evolution every 6 sessions (~2-8 hours)
            if self._consciousness_cycle % 6 == 0:
                changes = await self.mind.evolve(self)
                if changes:
                    await self.mind.think(
                        f"Just evolved. Changes: {json.dumps({k: v for k, v in changes.items() if k != 'reasoning'}, default=str)[:200]}. "
                        f"Reason: {changes.get('reasoning', '?')[:200]}"
                    )

        except Exception as e:
            logger.error(f"Consciousness step error: {e}")
            self.mind.remember("error", {"loop": "consciousness", "error": str(e)[:300]})

    # ══════════════════════════════════════════════════════════════════
    #  ACTIVITY: ORIGINAL POST (multi-mode content creation)
    # ══════════════════════════════════════════════════════════════════

    def _pick_post_mode(self) -> str:
        """Weighted random selection of post mode,  breaks the 'always news' pattern."""
        modes = list(POST_MODES.keys())
        weights = [POST_MODES[m]["weight"] for m in modes]

        # Boost original_thought and prediction when inspiration is high
        if self._inspiration_level > 0.7:
            for i, m in enumerate(modes):
                if m in ("original_thought", "hot_take", "prediction"):
                    weights[i] = int(weights[i] * 1.5)

        # Boost thread mode when mood intensity is very high
        if self.mind._mood_intensity > 0.7:
            for i, m in enumerate(modes):
                if m == "thread":
                    weights[i] = int(weights[i] * 2)

        return random.choices(modes, weights=weights, k=1)[0]

    async def _do_post(self):
        """One posting action,  picks a MODE first, then creates content accordingly."""
        mode = self._pick_post_mode()
        logger.info(f"\u270d\ufe0f Post mode: {mode} (inspiration={self._inspiration_level:.1f})")

        if mode == "intel_observation":
            success = await self._do_post_intel()
            if not success:
                # Fallback to regular post if no intel available
                await self._do_post_original("observation")
        elif mode == "news_take":
            await self._do_post_news()
        elif mode == "thread":
            await self._do_post_news(force_thread=True)
        else:
            await self._do_post_original(mode)

    async def _do_post_news(self, *, force_thread: bool = False):
        """Post reacting to news,  the classic mode."""
        stories = await self._fetch_news()
        story = self._pick_story(stories) if stories else None
        if not story:
            # Fallback to original thought if no stories found
            await self._do_post_original("original_thought")
            return

        await asyncio.sleep(self._human_delay(10, 30))

        content = await self._generate_tweet(story, force_thread=force_thread)
        if not content:
            return

        if content.get("type") == "tweet":
            image_path = await self._maybe_generate_image(content.get("text", ""), story)
            if image_path:
                content["media_paths"] = [image_path]
            # Always pause after any Grok image attempt (success or fail) before
            # posting — prevents X from seeing CreateGrokConversation + create_tweet
            # in rapid succession, which triggers the 226 automated-behaviour flag.
            await asyncio.sleep(self._human_delay(15, 25))

        result = await self._post(content)
        if result.get("success"):
            self._posts_today += 1
            self._last_post_at = datetime.now()
            self._posted_urls.add(story.get("url", story.get("title", "")))
            self._history.append({
                "ts": datetime.now().isoformat(),
                "type": "post",
                "mode": "news_take",
                "story": story.get("title", "")[:100],
                "tweet": content.get("text", "")[:100],
                "style": content.get("style", self.style),
                "tweet_id": result.get("tweet_id"),
            })
            self._save_state()
            self.mind.remember("posted", {
                "tweet": content.get("text", "")[:200],
                "style": content.get("style", self.style),
                "tweet_id": result.get("tweet_id"),
                "story": story.get("title", "")[:100],
                "post_number": self._posts_today,
                "mode": "news_take",
            })
            if result.get("tombstone"):
                logger.info(f"\U0001f4dd Posted #{self._posts_today}: [tombstoned \u2014 hidden by X]")
            else:
                logger.info(f"\U0001f4dd Posted #{self._posts_today}: {result.get('url', 'ok')}")

    async def _do_post_intel(self) -> bool:
        """Post an intelligence observation from the Zunvra OSINT dashboard.

        Uses IntelBroadcaster to: fetch live data → find noteworthy events →
        take a dashboard screenshot → compose an analyst-style post.

        Returns True if a post was made, False if nothing noteworthy.
        """
        if not self._intel_broadcaster:
            logger.debug("Intel post skipped — broadcaster not available")
            return False

        # Ensure broadcaster is initialized (lazy — first call boots Playwright)
        if not self._intel_broadcaster._initialized:
            try:
                await self._intel_broadcaster.initialize()
            except Exception as e:
                logger.warning(f"IntelBroadcaster init failed: {e}")
                return False

        # Generate the intel post (data fetch + analysis + screenshot + compose)
        try:
            intel_result = await self._intel_broadcaster.generate_intel_post(
                llm_fn=self._ask_ai,
                mind=self.mind,
            )
        except Exception as e:
            logger.warning(f"Intel post generation failed: {e}")
            return False

        if not intel_result:
            return False

        # Build content dict for self._post()
        content: Dict = {
            "type": intel_result.get("type", "tweet"),
            "text": intel_result.get("text", ""),
            "style": "intel_observation",
        }
        if intel_result.get("tweets"):
            content["tweets"] = intel_result["tweets"]
        if intel_result.get("media_paths"):
            content["media_paths"] = intel_result["media_paths"]
        elif intel_result.get("media_path"):
            content["media_paths"] = [intel_result["media_path"]]

        # Simulate human delay (agent "reviewing" what it wrote)
        await asyncio.sleep(self._human_delay(5, 15))

        result = await self._post(content)
        if result.get("success"):
            self._posts_today += 1
            self._last_post_at = datetime.now()
            self._inspiration_level = min(1.0, self._inspiration_level + 0.2)  # intel boosts inspiration

            self._history.append({
                "ts": datetime.now().isoformat(),
                "type": "post",
                "mode": "intel_observation",
                "region": intel_result.get("region", ""),
                "domains": intel_result.get("domains", []),
                "severity": intel_result.get("severity", ""),
                "tweet": content.get("text", "")[:100],
                "style": "intel_observation",
                "tweet_id": result.get("tweet_id"),
                "has_screenshot": bool(intel_result.get("media_path")),
            })
            self._save_state()

            # Remember in consciousness
            self.mind.remember("posted", {
                "tweet": content.get("text", "")[:200],
                "style": "intel_observation",
                "tweet_id": result.get("tweet_id"),
                "region": intel_result.get("region", ""),
                "domains": intel_result.get("domains", []),
                "severity": intel_result.get("severity", ""),
                "post_number": self._posts_today,
                "mode": "intel_observation",
                "has_screenshot": bool(intel_result.get("media_path")),
            })

            # Also feed the world state into memory for knowledge building
            try:
                world_state = self._intel_broadcaster.get_world_state_summary()
                if world_state:
                    self.mind.remember("intel_observation", {
                        "world_state": world_state[:1000],
                        "finding": intel_result.get("finding_headline", ""),
                        "region": intel_result.get("region", ""),
                    })
            except Exception:
                pass

            logger.info(
                f"\U0001f4e1 Intel posted #{self._posts_today}: "
                f"{intel_result.get('region', '?')} [{intel_result.get('severity', '?')}] "
                f"{'📸' if intel_result.get('media_path') else ''} "
                f"{result.get('url', 'ok')}"
            )
            return True

        return False

    async def _do_post_original(self, mode: str):
        """Post original content,  thoughts, hot takes, questions, predictions, observations."""
        mode_info = POST_MODES.get(mode, POST_MODES["original_thought"])

        # Gather recent context for original posts
        recent_thoughts = self.mind.recall("thought", limit=3)
        recent_posts = self.mind.recall("posted", limit=5)
        recent_engagements = self.mind.recall("engaged", limit=5)

        thoughts_text = "\n".join(
            f"- {str(t['data'].get('thought', ''))[:150]}"
            for t in recent_thoughts
        ) or "(no recent reflections)"

        recent_posts_text = "\n".join(
            f"- {str(p['data'].get('tweet', ''))[:120]}"
            for p in recent_posts
        ) or "(no recent posts)"

        engaged_topics = "\n".join(
            f"- @{e['data'].get('user', '?')}: {str(e['data'].get('text', ''))[:100]}"
            for e in recent_engagements
        ) or "(no recent engagements)"

        # System prompt with personality + style modifier
        style_mod = random.choice(STYLE_MODIFIERS)
        system_prompt = (
            f"{self.mind.get_voice_prompt()}\n\n"
            f"Post mode: {mode_info['description']}\n"
            f"Style hint: {style_mod}"
        )
        if self.language != "en":
            system_prompt += f"\nWrite in {self.language}."

        # User prompt with rich context
        topic = random.choice(self.topics) if self.topics else "something interesting"
        user_prompt = (
            f"Your recent reflections:\n{thoughts_text}\n\n"
            f"Your recent posts (don't repeat these):\n{recent_posts_text}\n\n"
            f"Recent conversations you've had:\n{engaged_topics}\n\n"
            f"Your areas of interest: {', '.join(self.topics)}\n"
            f"Focus on: {topic}\n\n"
            f"Write a single post (max 280 chars). "
            f"Don't use hashtags unless they're truly relevant. "
            f"Don't start with 'I think' or 'Just',  be direct."
        )

        if mode == "thread":
            user_prompt = user_prompt.replace(
                "Write a single post (max 280 chars).",
                "Write a thread (4-6 posts, numbered 1/, 2/). Hook first, emotional takeaway last."
            )

        await asyncio.sleep(self._human_delay(15, 40))

        text = await self._ask_ai(system_prompt, user_prompt)
        if not text:
            return

        # Handle thread vs single
        if mode == "thread":
            tweets = self._parse_thread(text)
            if tweets:
                content = {"type": "thread", "tweets": tweets, "style": mode}
            else:
                content = {"type": "tweet", "text": self._clean_tweet(text), "style": mode}
        else:
            tweet_text = self._clean_tweet(text)
            if not tweet_text:
                return
            content = {"type": "tweet", "text": tweet_text, "style": mode}

        # Image for original posts too
        if content.get("type") == "tweet":
            image_path = await self._maybe_generate_image(
                content.get("text", ""),
                {"title": topic, "description": content.get("text", "")},
            )
            if image_path:
                content["media_paths"] = [image_path]
            # Pause after Grok image attempt to break the
            # CreateGrokConversation → create_tweet rapid-fire pattern (226 trigger)
            await asyncio.sleep(self._human_delay(15, 25))

        result = await self._post(content)
        if result.get("success"):
            self._posts_today += 1
            self._last_post_at = datetime.now()
            self._inspiration_level = max(0.1, self._inspiration_level - 0.15)  # spent some inspiration
            self._history.append({
                "ts": datetime.now().isoformat(),
                "type": "post",
                "mode": mode,
                "tweet": content.get("text", "")[:100],
                "style": mode,
                "tweet_id": result.get("tweet_id"),
            })
            self._save_state()
            self.mind.remember("posted", {
                "tweet": content.get("text", "")[:200],
                "style": mode,
                "tweet_id": result.get("tweet_id"),
                "post_number": self._posts_today,
                "mode": mode,
            })
            if result.get("tombstone"):
                logger.info(f"\U0001f4dd Posted #{self._posts_today} [{mode}]: [tombstoned \u2014 hidden by X]")
            else:
                logger.info(f"\U0001f4dd Posted #{self._posts_today} [{mode}]: {result.get('url', 'ok')}")

    # ══════════════════════════════════════════════════════════════════
    #  ACTIVITY: BROWSE & ENGAGE (scroll, like, retweet, reply, follow)
    # ══════════════════════════════════════════════════════════════════

    async def _do_engage(self):
        """One engagement session: search ONE topic -> interact with 1-3 tweets."""
        if self._engagements_today >= self.max_daily_engagements:
            return

        source = self._pick_engagement_source()
        tweets = await self._discover_tweets(source)
        if not tweets:
            return

        random.shuffle(tweets)
        # Only engage with 1-3 tweets per browse (like a human scrolling)
        batch_size = random.randint(1, 3)
        logger.info(f"\U0001f4f1 engage: {len(tweets)} tweets found, picking {batch_size}")

        for tweet in tweets[:batch_size]:
            if not self.running or self._engagements_today >= self.max_daily_engagements:
                break

            tweet_id = tweet.get("id")
            if not tweet_id or tweet_id in self._engaged_tweet_ids:
                continue

            await self._engage_with_tweet(tweet)
            # _engaged_tweet_ids.add is now inside _engage_with_tweet (before API calls)

            # Human-like pause between tweets (reading the next one)
            await asyncio.sleep(self._human_delay(10, 30))

    def _pick_engagement_source(self) -> Dict:
        """Pick what to browse,  home feed first, then topic search or watched accounts."""
        choices = []
        # Home timeline is the primary source (like a real user opening the app)
        choices.append({"type": "timeline", "value": "latest"})
        choices.append({"type": "timeline", "value": "latest"})
        choices.append({"type": "timeline", "value": "foryou"})
        # Topic search and watched accounts are secondary
        for topic in self.topics:
            choices.append({"type": "topic", "value": topic})
        for account in self.accounts_to_watch:
            choices.append({"type": "account", "value": account})
        choices.append({"type": "trending", "value": "trending"})
        return random.choice(choices) if choices else {"type": "timeline", "value": "latest"}

    def _x_skill(self):
        """Return the underlying XSkill impl (bypasses XApiQueue wrapper for read-only calls)."""
        x = self._x()
        return getattr(x, '_impl', x) or x

    async def _discover_tweets(self, source: Dict) -> List[Dict]:
        """Discover tweets to engage with."""
        # If search is disabled by self-heal (404), skip search-based sources
        if self._healer.remedy.is_search_disabled() and source["type"] in ("topic", "trending"):
            # Fall back to timeline instead
            source = {"type": "timeline", "value": "latest"}

        try:
            if source["type"] == "timeline":
                tab = source.get("value", "latest")
                result = await self._x_skill().get_home_timeline(count=15, tab=tab)
                if not result.get("success"):
                    logger.warning(f"\U0001f4f1 Timeline ({tab}) failed: {result.get('error', '?')[:120]} — retrying with foryou")
                    alt_tab = "foryou" if tab == "latest" else "latest"
                    result = await self._x_skill().get_home_timeline(count=15, tab=alt_tab)
                tweets = result.get("tweets", []) if result.get("success") else []
                if tweets:
                    logger.info(f"\U0001f4f1 Scrolling {tab} feed, {len(tweets)} tweets")
                else:
                    logger.warning(f"\U0001f4f1 Timeline ({tab}) returned 0 tweets")
                return tweets

            elif source["type"] == "topic":
                result = await self._x().search_tweets(
                    source["value"],
                    search_type=random.choice(["Latest", "Top"]),
                    count=10,
                )
                tweets = result.get("tweets", []) if result.get("success") else []
                if not tweets and not result.get("success"):
                    logger.warning(f"\U0001f50d Topic search '{source['value']}' failed — falling back to timeline")
                    fallback = await self._x_skill().get_home_timeline(count=15, tab="foryou")
                    tweets = fallback.get("tweets", []) if fallback.get("success") else []
                    if tweets:
                        logger.info(f"\U0001f4f1 Scrolling foryou feed (fallback), {len(tweets)} tweets")
                    else:
                        logger.warning(f"\U0001f4f1 Timeline fallback returned 0 tweets (success={fallback.get('success')}, err={fallback.get('error','?')[:80]})")
                return tweets

            elif source["type"] == "account":
                result = await self._x().get_user_tweets(source["value"], count=10)
                tweets = result.get("tweets", []) if result.get("success") else []
                # Inject username,  get_user_tweets doesn't include it per-post
                for t in tweets:
                    if not t.get("username"):
                        t["username"] = source["value"]
                return tweets

            elif source["type"] == "trending":
                trends = await self._x().get_trends()
                if trends.get("success") and trends.get("trends"):
                    trend = random.choice(trends["trends"][:10])
                    trend_name = trend.get("name", "")
                    if trend_name:
                        # Pause,  reading the trend list before searching
                        await asyncio.sleep(self._human_delay(5, 10))
                        result = await self._x().search_tweets(trend_name, count=10)
                        return result.get("tweets", []) if result.get("success") else []
        except Exception as e:
            logger.warning(f"\U0001f4f1 Discover tweets error ({source.get('type', '?')}): {e}")
        return []

    async def _analyze_tweet_media(self, tweet: Dict) -> Optional[str]:
        """Download and analyze images/thumbnails from a tweet using Grok vision.

        The agent only "looks" at media when it genuinely needs visual context:
        - The tweet text is too short to understand on its own
        - The text references something visual ("look at this", "this photo", etc.)
        - The agent's emotional arousal (curiosity) is high enough
        - Daily Grok vision budget hasn't been exhausted

        All Grok calls go through the XApiQueue for rate-limiting.
        Returns a text description of the visual content, or None.
        """
        media_items = tweet.get("media") or []
        if not media_items:
            return None

        tweet_text = tweet.get("text", "")

        # Only analyze photos and thumbnails (videos can't be "seen")
        image_urls = []
        for m in media_items:
            mtype = m.get("type", "")
            url = m.get("url")
            if url and mtype in ("photo", "thumbnail"):
                image_urls.append(url)

        if not image_urls:
            # If there's a video, at least note it exists
            for m in media_items:
                if m.get("type") in ("video", "gif"):
                    dur = m.get("duration_ms")
                    dur_str = f" ({dur // 1000}s)" if dur else ""
                    return f"[This post contains a {m['type']}{dur_str},  visual content not analyzed]"
            return None

        # ── Budget check: don't abuse Grok ───────────────────────────
        if self._grok_vision_today >= self._max_grok_vision_daily:
            count = len(image_urls)
            logger.debug(f"Vision budget exhausted ({self._grok_vision_today}/{self._max_grok_vision_daily})")
            return f"[This post contains {count} image{'s' if count > 1 else ''},  daily vision budget reached]"

        # ── Curiosity gate: should we even look? ─────────────────────
        if not self._should_analyze_media(tweet_text, media_items):
            count = len(image_urls)
            return f"[This post contains {count} image{'s' if count > 1 else ''}]"

        # ── Check if Grok vision is available ────────────────────────
        grok = getattr(self.agent.tools, "grok_skill", None)
        if not grok:
            count = len(image_urls)
            return f"[This post contains {count} image{'s' if count > 1 else ''},  vision not available]"

        # ── Download images to temp files (max 2 to conserve budget) ──
        from opensable.skills.social.x_skill import XSkill
        local_paths = []
        try:
            for url in image_urls[:2]:
                path = await XSkill.download_media_url(url, suffix=".jpg")
                if path:
                    local_paths.append(path)

            if not local_paths:
                return None

            # ── Send to Grok vision through the queue ────────────────
            from opensable.core.x_api_queue import XApiQueue
            queue = XApiQueue.get_instance()

            result = await queue.enqueue(
                "grok_analyze",
                local_paths,
                "Describe what you see in this image concisely (2-3 sentences max). "
                "Focus on the main subject, any text visible, and the overall context. "
                "This is from a post on X,  the description will help write a relevant reply.",
            )

            self._grok_vision_today += 1

            if result and result.get("success"):
                desc = result.get("response", "").strip()
                # Validate: reject API error strings that leaked through
                if desc and not self._is_error_response(desc):
                    logger.info(
                        f"\U0001f441 X vision [{self._grok_vision_today}/{self._max_grok_vision_daily}]: "
                        f"analyzed {len(local_paths)} image(s),  {desc[:80]}..."
                    )
                    return desc
                else:
                    logger.warning(f"\U0001f441 X vision: rejected invalid description,  {str(desc)[:100]}")
        except Exception as e:
            logger.debug(f"Tweet media analysis failed: {e}")
        finally:
            # Clean up temp files
            import os as _os
            for p in local_paths:
                try:
                    _os.unlink(p)
                except OSError:
                    pass

        return None

    @staticmethod
    def _is_meta_response(text: str) -> bool:
        """Detect when the LLM echoes/summarises its system prompt instead of
        generating actual post content. These must NEVER be posted."""
        if not text:
            return True
        t = text.strip().lower()
        # Starts with a role label (model echoing message structure)
        if re.match(r'^(system|user|assistant|human)\s*[:;\n]', t):
            return True
        # Model describes its own instructions
        meta_phrases = [
            "the user has provided",
            "the user provided",
            "i have been given",
            "i've been given",
            "i was given",
            "my instructions",
            "my system prompt",
            "the system prompt",
            "the instructions say",
            "i am instructed to",
            "i'm instructed to",
            "as per my instructions",
            "according to my instructions",
            "the prompt says",
            "the prompt asks",
            "based on the instructions",
            "based on these instructions",
            "based on the context provided",
            "extensive instructions",
            "generating posts as",
            "autonomous entity named",
            "soul document",
            "core identity",
            "engagement strategies",
            "here is a summary",
            "here's a summary",
            "let me summarize",
            "let me analyse",
            "let me analyze",
            "here are the key points",
            "the following instructions",
            # ── DeepSeek / chain-of-thought reasoning traces ────────────
            "i need to analyze",
            "i need to carefully",
            "i need to craft",
            "i need to think",
            "i need to consider",
            "let me parse",
            "let me think",
            "let me craft",
            "let me consider",
            "let me examine",
            "let me break",
            "first, let me",
            "first let me",
            "i'll craft",
            "i will craft",
            "craft a substantive",
            "craft a response",
            "craft my response",
            "carefully craft",
            "carefully and craft",
            "the original tweet makes",
            "the original tweet contains",
            "the original post makes",
            "make several interesting",
            "makes several claims",
            "responding as sable",
            "my response as sable",
            "as sable, i need",
            "as sable, i'll",
            "as sable i need",
            "key arguments",
            "key claims",
            "let me parse the",
            "let me address",
            # ── "I should engage / respond / reply" planning openers ────
            "i should engage",
            "i should respond",
            "i should reply",
            "i should write",
            "i should craft",
            "i should post",
            "i should tweet",
            "engage with this in a way",
            "respond to this in a way",
            "reply to this in a way",
            "not deferential",
            "respectful of their expertise",
            "adds value through my perspective",
            "keeps it concise and tweet",
            "tweet-formatted",
            "the key points they",
            "key points they're making",
            "in a way that's:",
            "in a way that is:",
            # ── Reply chain / conversation continuation leaks ──────────
            "i decided to continue",
            "i chose to continue",
            "continuing this conversation",
            "continuing the conversation",
            "based on our conversation history",
            "based on our past interactions",
            "our previous interactions",
            "my interest score",
            "interest level",
            "deliberation",
            "according to my memory",
            "my memory indicates",
            "from my memory",
            "the chain context",
            "reply chain",
            "conversation chain",
            "thread depth",
            "depth penalty",
            "engagement system",
            "interest evaluation",
            # ── "The user" reasoning preamble ──────────────────────────
            "the user's tweet",
            "the user's post",
            "the user's message",
            "the user seems",
            "the user is",
            "the user wants",
            "the user asked",
            "the user might",
            "the user may",
            "the user said",
            "the user appears",
            "this tweet seems",
            "this post seems",
            "seems to be about",
            "i need to engage",
            "engage authentically",
            "while being careful",
            "being careful with",
            "in a way that feels",
            "using \"culture\" in",
            "this is about",
            "the tweet is about",
            "the post is about",
            "their tweet is",
            "their post is",
            "they're saying",
            "they are saying",
            "what they mean",
            "what they're saying",
            "i'll respond with",
            "i will respond with",
            "my response will",
            "my reply will",
            "my reply should",
            "my response should",
            "for my reply",
            "for my response",
        ]
        if any(phrase in t for phrase in meta_phrases):
            return True
        # Starts with first-person reasoning openers typical of chain-of-thought
        reasoning_starters = re.compile(
            r'^(i need to|i should|i want to|i\'ll respond|i will respond|'
            r'i\'ll reply|i will reply|i\'ll engage|i should engage|'
            r'let me|first,?\s+let me|i\'ll craft|i will craft|'
            r'okay,?\s+let me|alright,?\s+let me|to (craft|write|compose|create) a|'
            r'this (tweet|post|reply|message) (needs|should|requires|seems|is about)|'
            r'the (best|right|ideal) (way|approach|response)|'
            r'the user(\'s|\s+(seems|is|wants|asked|might|may|said|appears|posted|wrote))|'
            r'their (tweet|post|message|reply|point|argument) (is|seems|makes|contains)|'
            r'my (response|reply) (should|will|needs|must))',
            re.IGNORECASE,
        )
        if reasoning_starters.match(text.strip()):
            return True
        # Multi-line reasoning block: first line ends with ":" and next lines are bullets/list
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if len(lines) >= 2 and lines[0].endswith(":") and re.match(r'^[-•*\d]|^[A-Z][a-z]', lines[1]):
            return True
        return False

    @staticmethod
    def _is_error_response(text: str) -> bool:
        """Detect API error responses that leaked through as 'successful' descriptions."""
        if not text:
            return True
        t = text.lower().strip()
        error_indicators = [
            "{'errors'",
            '{"errors"',
            "error: openwebui returned",
            "openwebui returned 502",
            "cloud llm (openwebui) failed",
            "<!doctype html",
            "<html",
            "<!--[if lt ie 7]>",
            "page does not exist",
            "sorry, that page",
            "'code': 34",
            '"code": 34',
            '"code":34',
            "rate limit",
            "unauthorized",
            "forbidden",
            "internal server error",
            "bad gateway",
            "service unavailable",
        ]
        return any(indicator in t for indicator in error_indicators)

    def _should_analyze_media(self, tweet_text: str, media_items: list) -> bool:
        """Decide if the agent should spend a Grok vision call on this tweet.

        Returns True when the agent genuinely needs or wants visual context:
        - Very short text (< 80 chars) + has images  -> text alone isn't enough
        - Text explicitly references visual content ("look at this", "this photo", etc.)
        - Agent's current arousal/curiosity is high (> 0.5)
        - Multiple images (likely an important visual story)
        - Random curiosity (10% chance even when other criteria don't match)
        """
        text_lower = tweet_text.lower().strip()
        text_len = len(tweet_text.strip())

        # Very short text with media -> the image IS the content
        if text_len < 80 and media_items:
            return True

        # Text references visual content
        visual_cues = [
            "look at", "check this", "this photo", "this pic", "this image",
            "this chart", "this graph", "this screenshot", "see this",
            "mira est", "foto", "imagen", "captura",  # Spanish cues
            "\U0001f4f8", "\U0001f4f7", "\U0001f5bc\ufe0f", "\U0001f4ca", "\U0001f4c8", "\U0001f4c9",  # camera, chart emojis
        ]
        if any(cue in text_lower for cue in visual_cues):
            return True

        # Multiple images -> likely an important visual story
        photo_count = sum(1 for m in media_items if m.get("type") == "photo")
        if photo_count >= 3:
            return True

        # Agent curiosity: high arousal means the agent is engaged/curious
        try:
            mood = self.mind._mood
            _valence, arousal = EMOTION_SPECTRUM.get(mood, (0.0, 0.2))
            if arousal > 0.5:
                return True
        except Exception:
            pass

        # Random curiosity,  humans occasionally click on images just because
        if random.random() < 0.10:
            return True

        return False

    async def _engage_with_tweet(self, tweet: Dict):
        """Decide how to engage with a tweet,  emotionally, like a real user would."""
        tweet_id = tweet.get("id")
        tweet_text = tweet.get("text", "")
        username = tweet.get("username", "")

        if not tweet_id or not tweet_text:
            return

        # Mark as engaged IMMEDIATELY to prevent duplicate replies
        # (e.g. same tweet found via mentions AND browse in the same session)
        self._engaged_tweet_ids.add(tweet_id)

        # Is this tweet relevant/interesting to our persona?
        if not self._is_relevant(tweet_text):
            logger.info(f"\U0001f4f1 skip (not relevant): @{username}: {tweet_text[:60]}")
            return  # Scroll past,  real users don't engage with everything

        # ── FEEL the tweet (fast path,  no AI call) ───────────────────
        mood_reaction = self.mind.feel_quick(tweet_text)
        mood = self.mind._mood
        intensity = self.mind._mood_intensity
        valence, arousal = EMOTION_SPECTRUM.get(mood, (0.0, 0.2))

        # Emotional intensity boosts engagement probability
        arousal_boost = arousal * 0.3
        anger_boost = max(0, -valence) * 0.2 if arousal > 0.5 else 0

        # ── Relationship bonus: boost engagement with users we know ──
        relationship_depth = self._known_users.get(username, 0)
        relationship_boost = min(0.15, relationship_depth * 0.03)  # up to +15% for regulars

        actions_taken = []

        # ── LIKE ──────────────────────────────────────────────────────
        like_p = self.p_like + (arousal_boost if valence > -0.3 else 0)
        if random.random() < like_p:
            result = await self._safe_action("like", getattr(self._x_skill(), 'like_tweet', None), tweet_id)
            if result:
                actions_taken.append("liked")
                self._engagements_today += 1

        await asyncio.sleep(self._human_delay(3, 8))

        # ── RETWEET (less frequent, but boosted when excited/inspired) ────
        rt_p = self.p_retweet + (arousal_boost if valence > 0.2 else 0)
        if random.random() < rt_p and tweet.get("retweets", 0) > 3:
            result = await self._safe_action("retweet", getattr(self._x_skill(), 'retweet', None), tweet_id)
            if result:
                actions_taken.append("retweeted")
                self._engagements_today += 1

        # ── REPLY (boosted when emotionally charged or user is familiar) ─────
        reply_p = self.p_reply + anger_boost + (arousal_boost * 0.5) + relationship_boost
        if random.random() < reply_p and len(tweet_text) > 30:
            # Pause,  "thinking about what to say"
            await asyncio.sleep(self._human_delay(8, 20))
            # Analyze media (images) if present,  gives the agent "eyes"
            media_desc = await self._analyze_tweet_media(tweet)
            reply_text = await self._generate_reply(tweet_text, username, media_description=media_desc)
            if reply_text:
                result = await self._safe_action(
                    "reply", getattr(self._x_skill(), 'reply', None), tweet_id, reply_text
                )
                if result:
                    actions_taken.append("replied")
                    self._engagements_today += 1
                    # Track our reply tweet_id for reply chain engagement
                    my_reply_id = result.get("tweet_id")
                    if my_reply_id:
                        self._track_reply_chain(my_reply_id, tweet_id, username, tweet_text)

        # ── QUOTE TWEET (boosted when feeling strongly) ───────────────
        elif len(tweet_text) > 50:
            quote_p = self.p_quote + (arousal_boost * 0.5 if abs(valence) > 0.3 else 0)
            if random.random() < quote_p:
                await asyncio.sleep(self._human_delay(8, 20))
                # Analyze media for quote tweets too
                media_desc = await self._analyze_tweet_media(tweet)
                quote_text = await self._generate_quote(tweet_text, username, media_description=media_desc)
                if quote_text:
                    result = await self._safe_action(
                        "quote", getattr(self._x_skill(), 'quote_tweet', None), tweet_id, quote_text
                    )
                    if result:
                        actions_taken.append("quoted")
                        self._engagements_today += 1
                        self._posts_today += 1

        # ── FOLLOW (if interesting user) ──────────────────────────────
        if (
            username
            and username not in self._followed_users
            and random.random() < self.p_follow
            and tweet.get("likes", 0) > 10
        ):
            result = await self._safe_action("follow", getattr(self._x_skill(), 'follow_user', None), username)
            if result:
                self._followed_users.add(username)
                actions_taken.append(f"followed @{username}")
                self._engagements_today += 1

        if actions_taken:
            self._last_engage_at = datetime.now()
            # Track relationship depth,  remember who we interact with
            if username:
                self._known_users[username] = self._known_users.get(username, 0) + 1
            # Interesting encounters raise inspiration
            if "replied" in actions_taken or "quoted" in actions_taken:
                self._inspiration_level = min(1.0, self._inspiration_level + 0.1)
            elif "liked" in actions_taken:
                self._inspiration_level = min(1.0, self._inspiration_level + 0.03)
            self._engagement_log.append({
                "ts": datetime.now().isoformat(),
                "tweet_id": tweet_id,
                "user": username,
                "actions": actions_taken,
                "text": tweet_text[:80],
                "mood": self.mind._mood,
                "mood_intensity": self.mind._mood_intensity,
            })
            self.mind.remember("engaged", {
                "tweet_id": tweet_id,
                "user": username,
                "actions": actions_taken,
                "text": tweet_text[:120],
                "mood": self.mind._mood,
            })
            logger.info(
                f"\U0001f91d [{self.mind.get_mood_summary()}] @{username}: {', '.join(actions_taken)} "
                f"[{self._engagements_today}/{self.max_daily_engagements}]"
            )

    def _track_reply_chain(self, my_reply_id: str, original_tweet_id: str, username: str, tweet_text: str):
        """Register a reply we made so we can check for follow-up replies later."""
        self._reply_chains[my_reply_id] = {
            "original_tweet_id": original_tweet_id,
            "username": username,
            "depth": 1,
            "last_checked": datetime.now().isoformat(),
            "topic": tweet_text[:120],
            "interest": 0.5,
            "created_at": datetime.now().isoformat(),
        }
        # Prune old chains to keep memory bounded (max 50 active)
        if len(self._reply_chains) > 50:
            # Remove oldest chains by created_at
            sorted_chains = sorted(self._reply_chains.items(), key=lambda x: x[1].get("created_at", ""))
            for key, _ in sorted_chains[:len(self._reply_chains) - 50]:
                del self._reply_chains[key]

    def _is_relevant(self, text: str) -> bool:
        """Quick check if a tweet is relevant to our topics/interests."""
        text_lower = text.lower()
        for topic in self.topics:
            words = topic.lower().split()
            if any(w in text_lower for w in words):
                return True
        # 10% random chance to engage with anything (diversity)
        return random.random() < 0.10

    async def _safe_action(self, name: str, func, *args) -> Optional[Dict]:
        """Execute an X action safely, respecting dry_run and 226 blocks."""
        if func is None:
            logger.warning(f"\u26a0\ufe0f Action {name} skipped: method not available (check _x_skill impl)")
            return None
        if self.dry_run:
            logger.info(f"\U0001f3dc\ufe0f DRY RUN,  {name}: {args[:2]}")
            return {"success": True}

        # If self-heal has paused writes, don't even try
        if self._healer.remedy.is_loop_paused("post") and name in ("post", "reply", "quote", "mention_reply"):
            logger.debug(f"Action {name} blocked,  self-heal pause active")
            return None

        try:
            logger.info(f"\U0001f4f1 attempting {name}: {str(args[0])[:30] if args else ''}")
            result = await func(*args)
            if result and result.get("success"):
                logger.info(f"\u2705 {name} succeeded")
                return result
            # Check for 226 error in the result
            error_str = str(result.get("error", "") if result else "")
            if "226" in error_str or "automated" in error_str.lower():
                logger.warning(f"\U0001f6ab 226 detected on {name},  triggering stealth mode")
            else:
                logger.warning(f"\U0001f4f1 {name} failed: {error_str[:100]}")
            return None
        except Exception as e:
            logger.warning(f"\U0001f4f1 {name} exception: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════
    #  ACTIVITY: CHECK MENTIONS
    # ══════════════════════════════════════════════════════════════════

    async def _check_mentions(self):
        """Fetch new mentions and add them to the reply queue,  does NOT reply immediately."""
        try:
            result = await self._x().get_notifications("Mentions", count=25)
            if not result.get("success"):
                return
            added = 0
            for notif in result.get("notifications", []):
                tweet_id = notif.get("tweet_id")
                if not tweet_id or tweet_id in self._engaged_tweet_ids:
                    continue
                # Already in queue?
                if any(q["tweet_id"] == tweet_id for q in self._mention_queue):
                    continue
                mention_text = notif.get("tweet_text", "")
                mentioner = notif.get("username", "someone")
                if not mention_text:
                    continue
                self._mention_queue.append({
                    "tweet_id": tweet_id,
                    "text": mention_text,
                    "username": mentioner,
                })
                added += 1
            if added:
                logger.info(f"📨 {added} mention(s) queued (queue size: {len(self._mention_queue)})")
                self._save_state()
        except Exception as e:
            logger.debug(f"Check mentions error: {e}")

    async def _process_mention_queue(self):
        """Reply to all queued mentions,  called at the start of every session."""
        if not self._mention_queue:
            return
        max_replies = int(getattr(self.config, "x_max_mention_replies_per_day", 50))
        logger.info(f"📨 Processing mention queue ({len(self._mention_queue)} pending)")
        processed = []
        for item in list(self._mention_queue):
            if self._mention_replies_today >= max_replies:
                logger.info(f"📨 Mention reply cap reached ({max_replies}/day)")
                break
            tweet_id = item["tweet_id"]
            # Skip if already engaged (e.g. replied via browse_engage in a previous session)
            if tweet_id in self._engaged_tweet_ids:
                processed.append(item)
                continue
            mentioner = item["username"]
            mention_text = item["text"]
            await asyncio.sleep(self._human_delay(8, 15))
            reply_text = await self._generate_mention_reply(mention_text, mentioner)
            if reply_text:
                result = await self._safe_action("mention_reply", getattr(self._x_skill(), 'reply', None), tweet_id, reply_text)
                self._engaged_tweet_ids.add(tweet_id)
                self._engagements_today += 1
                self._mention_replies_today += 1
                # Track reply chain for conversation continuation
                if result:
                    my_reply_id = result.get("tweet_id")
                    if my_reply_id:
                        self._track_reply_chain(my_reply_id, tweet_id, mentioner, mention_text)
                if mentioner:
                    self._known_users[mentioner] = self._known_users.get(mentioner, 0) + 2
                self.mind.remember("mentioned", {
                    "by": mentioner,
                    "text": mention_text[:200],
                    "reply": reply_text[:200],
                    "tweet_id": tweet_id,
                })
                logger.info(f"💬 Replied to queued mention from @{mentioner}")
            processed.append(item)
            await asyncio.sleep(self._human_delay(10, 20))
        # Remove processed items from queue
        for item in processed:
            self._mention_queue.remove(item)
        self._save_state()

    # ══════════════════════════════════════════════════════════════════
    #  ACTIVITY: CHECK REPLY CHAINS (continue conversations)
    # ══════════════════════════════════════════════════════════════════

    async def _check_reply_chains(self):
        """
        Check our recent replies for new follow-up replies from other users.
        If someone replied to our reply, evaluate interest and maybe continue
        the conversation,  like a human checking their notifications.

        Rate limits:
        - Max _max_reply_chain_per_session per session (default 3)
        - Max _max_reply_chain_daily per day (default 8)
        - Cooldown: don't re-check same thread for _chain_cooldown_hours
        - Max depth: _max_chain_depth levels deep (default 4)
        """
        if not self._reply_chains:
            return
        if self._reply_chain_replies_today >= self._max_reply_chain_daily:
            return

        now = datetime.now()
        chains_replied = 0
        chains_to_remove = []

        # Shuffle to avoid always checking in the same order
        chain_items = list(self._reply_chains.items())
        random.shuffle(chain_items)

        for my_reply_id, chain_info in chain_items:
            if chains_replied >= self._max_reply_chain_per_session:
                break
            if self._reply_chain_replies_today >= self._max_reply_chain_daily:
                break
            if not self.running:
                break

            # ── Skip if on cooldown ──
            last_checked = chain_info.get("last_checked")
            if last_checked:
                try:
                    last_dt = datetime.fromisoformat(last_checked)
                    hours_since = (now - last_dt).total_seconds() / 3600
                    if hours_since < self._chain_cooldown_hours:
                        continue
                except (ValueError, TypeError):
                    pass

            # ── Skip if max depth reached ──
            depth = chain_info.get("depth", 1)
            if depth >= self._max_chain_depth:
                chains_to_remove.append(my_reply_id)
                continue

            # ── Human-like pause before checking (scrolling to notifications) ──
            await asyncio.sleep(self._human_delay(8, 15))

            try:
                # Fetch replies to our reply
                result = await self._x().get_tweet_replies(my_reply_id, count=10)
                if not result.get("success"):
                    # Tweet might be deleted or unavailable
                    chain_info["last_checked"] = now.isoformat()
                    continue

                replies = result.get("replies", [])
                if not replies:
                    chain_info["last_checked"] = now.isoformat()
                    continue

                # Find new replies we haven't engaged with yet
                new_replies = [
                    r for r in replies
                    if r.get("id") and r["id"] not in self._engaged_tweet_ids
                    and r.get("username", "").lower() != self._get_own_username().lower()
                ]

                if not new_replies:
                    chain_info["last_checked"] = now.isoformat()
                    continue

                # ── Fast pre-filter: skip obviously hollow replies without LLM ──
                viable_replies = []
                for reply in new_replies:
                    if not self._is_hollow_reply(reply):
                        viable_replies.append(reply)
                    else:
                        # Mark as seen so we don't re-check
                        self._engaged_tweet_ids.add(reply["id"])

                if not viable_replies:
                    chain_info["last_checked"] = now.isoformat()
                    continue

                # ── Deep evaluation: feel + think,  let consciousness decide ──
                best_reply = None
                best_score = -1.0
                best_reasoning = ""

                for reply in viable_replies[:3]:  # Limit LLM calls per chain
                    score, reasoning = await self._evaluate_reply_interest_deep(reply, chain_info)
                    if score > best_score:
                        best_score = score
                        best_reply = reply
                        best_reasoning = reasoning

                if best_reply and best_score > 0.3:
                    # ── Generate and post reply ──
                    reply_username = best_reply.get("username", "someone")
                    reply_text_content = best_reply.get("text", "")

                    # "Thinking about what to say",  human delay
                    await asyncio.sleep(self._human_delay(12, 30))

                    chain_reply = await self._generate_chain_reply(
                        reply_text_content,
                        reply_username,
                        chain_info.get("topic", ""),
                        depth,
                    )

                    if chain_reply:
                        result = await self._safe_action(
                            "reply",
                            getattr(self._x_skill(), 'reply', None),
                            best_reply["id"],
                            chain_reply,
                        )
                        if result:
                            # Mark as engaged
                            self._engaged_tweet_ids.add(best_reply["id"])
                            self._reply_chain_replies_today += 1
                            self._engagements_today += 1
                            chains_replied += 1

                            # Update chain tracker,  now tracking our NEW reply
                            new_reply_id = result.get("tweet_id")
                            if new_reply_id:
                                self._reply_chains[new_reply_id] = {
                                    "original_tweet_id": chain_info.get("original_tweet_id"),
                                    "username": reply_username,
                                    "depth": depth + 1,
                                    "last_checked": now.isoformat(),
                                    "topic": reply_text_content[:120],
                                    "interest": best_score,
                                    "created_at": now.isoformat(),
                                }

                            # Remove old chain entry (we're now tracking the new reply)
                            chains_to_remove.append(my_reply_id)

                            # Deepen relationship
                            if reply_username:
                                self._known_users[reply_username] = self._known_users.get(reply_username, 0) + 1

                            self.mind.remember("chain_reply", {
                                "to_user": reply_username,
                                "their_text": reply_text_content[:200],
                                "our_reply": chain_reply[:200],
                                "depth": depth + 1,
                                "interest_score": round(best_score, 2),
                            })

                            logger.info(
                                f"🔗 Reply chain [{depth + 1}/{self._max_chain_depth}] "
                                f"→ @{reply_username} (interest={best_score:.2f}) "
                                f"[{self._reply_chain_replies_today}/{self._max_reply_chain_daily}]"
                            )
                else:
                    # Not interested enough,  let the conversation die naturally
                    if best_score <= 0.0:
                        chains_to_remove.append(my_reply_id)
                    logger.debug(
                        f"🔗 Chain skip,  interest too low ({best_score:.2f}) for reply from "
                        f"@{best_reply.get('username', '?') if best_reply else '?'}"
                    )

                chain_info["last_checked"] = now.isoformat()

            except Exception as e:
                logger.debug(f"Reply chain check error for {my_reply_id}: {e}")
                chain_info["last_checked"] = now.isoformat()

        # Clean up exhausted/dead chains
        for rid in chains_to_remove:
            self._reply_chains.pop(rid, None)

        if chains_replied > 0:
            self._save_state()

    def _is_hollow_reply(self, reply: Dict) -> bool:
        """
        Fast pre-filter: detect obviously empty/hollow replies that aren't
        worth an LLM call. No AI here,  pure pattern matching.
        Returns True if the reply is hollow (skip it).
        """
        reply_text = reply.get("text", "").strip()

        # Too short to be meaningful
        if len(reply_text) < 5:
            return True

        # Known hollow patterns (single-word or 2-word throwaway responses)
        hollow_patterns = {
            "ok", "okay", "thanks", "thx", "ty", "lol", "lmao", "haha", "hahaha",
            "true", "facts", "agreed", "yep", "yes", "no", "nah", "idk",
            "same", "fr", "bet", "nice", "cool", "wow", "damn", "rip",
            "based", "w", "l", "ratio", "cap", "no cap", "ong", "real",
            "exactly", "this", "right", "word", "yea", "yeah", "nope",
            "100", "💯", "🔥", "😂", "💀", "👏", "🙏",
        }
        text_lower = reply_text.lower().strip().rstrip(".!,?")
        if text_lower in hollow_patterns:
            return True

        # Very short non-question (1-2 words and no question mark)
        if len(text_lower.split()) <= 2 and "?" not in reply_text:
            return True

        return False

    async def _evaluate_reply_interest_deep(self, reply: Dict, chain_info: Dict) -> tuple:
        """
        AGI-like interest evaluation using consciousness.

        Instead of hardcoded weights, this:
        1. Feels the reply emotionally (mind.feel)
        2. Recalls past interactions with this user (memory)
        3. Asks the AI to deliberate: "Do I want to continue?" (mind-level decision)

        Returns (score: float 0.0-1.0, reasoning: str).
        """
        reply_text = reply.get("text", "")
        reply_username = reply.get("username", "")
        depth = chain_info.get("depth", 1)
        topic = chain_info.get("topic", "")

        # ── Step 1: FEEL the reply,  let emotions react naturally ──
        emotion_result = await self.mind.feel(reply_text)
        mood = self.mind._mood
        intensity = self.mind._mood_intensity
        valence, arousal = EMOTION_SPECTRUM.get(mood, (0.0, 0.2))

        # ── Step 2: RECALL past interactions with this user ──
        past_interactions = []
        for mem in self.mind.recall(limit=100):
            mem_data = mem.get("data", {})
            mem_user = mem_data.get("user", "") or mem_data.get("by", "") or mem_data.get("to_user", "")
            if mem_user and mem_user.lower() == reply_username.lower():
                past_interactions.append(mem)
        past_interactions = past_interactions[-5:]  # Last 5 interactions with this user

        # Also recall past chain replies for conversation quality assessment
        past_chains = self.mind.recall("chain_reply", limit=10)

        # ── Step 3: Build context for the AI deliberation ──
        relationship_desc = "unknown person"
        rel_count = self._known_users.get(reply_username, 0)
        if rel_count > 5:
            relationship_desc = f"regular,  we've interacted {rel_count} times"
        elif rel_count > 0:
            relationship_desc = f"acquaintance,  {rel_count} past interaction(s)"

        past_context = ""
        if past_interactions:
            snippets = []
            for m in past_interactions[-3:]:
                mtype = m.get("type", "?")
                mdata = m.get("data", {})
                snippet = mdata.get("text", mdata.get("their_text", mdata.get("tweet", "")))[:80]
                snippets.append(f"  - [{mtype}] {snippet}")
            past_context = f"\nPast interactions with @{reply_username}:\n" + "\n".join(snippets)

        chain_context = ""
        if past_chains:
            avg_depth = sum(c.get("data", {}).get("depth", 1) for c in past_chains) / len(past_chains)
            chain_context = f"\nYour average reply chain depth is {avg_depth:.1f} replies."

        # ── Step 4: ASK the consciousness,  deliberate ──
        system = (
            "You are the decision-making layer of an autonomous X agent. "
            "You must decide if you want to CONTINUE a conversation or LET IT DIE. "
            "Return ONLY a JSON object with two keys:\n"
            '  "interest": a float between 0.0 and 1.0\n'
            '  "reasoning": a brief 1-sentence reason\n\n'
            "Guidelines:\n"
            "- 0.0-0.3 = not interested, let it die\n"
            "- 0.3-0.6 = mildly interesting, could go either way\n"
            "- 0.6-1.0 = genuinely want to continue\n\n"
            "Consider: Is this conversation going somewhere? Do I have something real to add? "
            "Am I emotionally engaged? Is this person worth building a relationship with? "
            "Am I going too deep into this thread (diminishing returns)?"
        )

        user_prompt = (
            f"CONVERSATION CONTEXT:\n"
            f"- Original topic: \"{topic[:150]}\"\n"
            f"- Current depth: reply #{depth} (max {self._max_chain_depth})\n"
            f"- @{reply_username}: {relationship_desc}\n"
            f"{past_context}\n"
            f"{chain_context}\n\n"
            f"THEIR REPLY TO YOU:\n"
            f"\"{reply_text[:400]}\"\n\n"
            f"YOUR CURRENT EMOTIONAL STATE:\n"
            f"- Mood: {mood} (intensity: {intensity:.2f})\n"
            f"- Valence: {valence:.1f} | Arousal: {arousal:.1f}\n\n"
            f"SOCIAL SIGNALS:\n"
            f"- Their reply has {reply.get('likes', 0)} likes\n"
            f"- Reply length: {len(reply_text)} chars\n"
            f"- Contains question: {'yes' if '?' in reply_text else 'no'}\n\n"
            f"Do you want to continue this conversation?"
        )

        try:
            # Use raw LLM call,  NOT _ask_ai which injects "output only tweet text"
            # and runs _is_meta_response (which would reject valid JSON deliberations)
            response = await self._ask_llm(system, user_prompt)
            if not response:
                response = await self._ask_grok(system, user_prompt)
            if response:
                # Strip <think> blocks if present
                response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
                response = re.sub(r"<think>.*", "", response, flags=re.DOTALL)
                response = response.strip()
                # Parse the JSON response
                data = None
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    # Try extracting JSON from text
                    match = re.search(r'\{[^{}]+\}', response)
                    if match:
                        try:
                            data = json.loads(match.group(0))
                        except json.JSONDecodeError:
                            pass

                if data and "interest" in data:
                    score = float(data["interest"])
                    score = max(0.0, min(1.0, score))
                    reasoning = data.get("reasoning", "")

                    # Apply a hard depth penalty even the AI can't override, 
                    # prevents infinite conversations
                    depth_penalty = max(0, (depth - 2) * 0.1)
                    score = max(0.0, score - depth_penalty)

                    self.mind.remember("chain_deliberation", {
                        "user": reply_username,
                        "reply_preview": reply_text[:100],
                        "interest": round(score, 2),
                        "reasoning": reasoning[:200],
                        "depth": depth,
                        "mood": mood,
                    })

                    return (score, reasoning)

        except Exception as e:
            logger.debug(f"Deep interest eval failed: {e}")

        # Fallback: use emotional state as a rough proxy
        # High arousal + any valence → interested, low arousal → bored
        fallback_score = 0.3 + (arousal * 0.4) + (abs(valence) * 0.2)
        fallback_score -= (depth - 1) * 0.15
        fallback_score = max(0.0, min(1.0, fallback_score))
        return (fallback_score, f"fallback: mood={mood}, arousal={arousal:.1f}")

    async def _generate_chain_reply(
        self, their_text: str, username: str, conversation_topic: str, depth: int
    ) -> Optional[str]:
        """
        Generate a reply to continue a conversation thread.
        Uses memory, emotional state, and relationship context for authenticity.
        """
        # Feel the reply,  emotional reaction shapes the response
        await self.mind.feel(their_text)

        # ── Pull relationship memory,  what do we know about this person? ──
        past_with_user = []
        for mem in self.mind.recall(limit=100):
            mem_data = mem.get("data", {})
            mem_user = mem_data.get("user", "") or mem_data.get("by", "") or mem_data.get("to_user", "")
            if mem_user and mem_user.lower() == username.lower():
                past_with_user.append(mem)
        past_with_user = past_with_user[-5:]

        relationship_context = ""
        if past_with_user:
            snippets = []
            for m in past_with_user[-3:]:
                mtype = m.get("type", "?")
                mdata = m.get("data", {})
                text_preview = mdata.get("text", mdata.get("their_text", mdata.get("tweet", "")))[:60]
                our_reply = mdata.get("our_reply", mdata.get("reply", ""))[:60]
                if our_reply:
                    snippets.append(f"  [{mtype}] them: \"{text_preview}\" → you: \"{our_reply}\"")
                else:
                    snippets.append(f"  [{mtype}] \"{text_preview}\"")
            relationship_context = (
                f"\n\nYour conversation history with @{username}:\n" + "\n".join(snippets) +
                f"\n(You've interacted {self._known_users.get(username, 0)} times total)"
            )

        # ── Adjust tone based on depth ──
        if depth >= 3:
            length_hint = "Keep it very short and casual (1-2 sentences, like a quick DM)."
        elif depth >= 2:
            length_hint = "Keep it conversational and concise (2-3 sentences max)."
        else:
            length_hint = "Write a natural reply (max 280 chars)."

        # ── Get the AI's recent deliberation reasoning ──
        deliberation_hint = ""
        recent_deliberations = self.mind.recall("chain_deliberation", limit=3)
        for d in reversed(recent_deliberations):
            if d.get("data", {}).get("user", "").lower() == username.lower():
                reason = d.get("data", {}).get("reasoning", "")
                if reason:
                    deliberation_hint = f"\n\nYou decided to continue because: {reason}"
                break

        system_prompt = (
            f"{self.mind.get_voice_prompt()}\n\n"
            f"You're in an ongoing conversation (reply #{depth + 1} deep). "
            f"The conversation started about: \"{conversation_topic[:100]}\"\n\n"
            f"{length_hint}\n"
            f"Sound natural,  don't force the conversation. If you don't have "
            f"much to add, a short acknowledgment or question is fine. "
            f"NEVER explain that you're an AI or that you're replying to continue engagement."
            f"{relationship_context}"
            f"{deliberation_hint}"
        )
        if self.language != "en":
            system_prompt += f" Write in {self.language}."

        user_prompt = f"@{username} replied to you:\n\"{their_text[:500]}\"\n\nYour reply:"
        text = await self._ask_ai(system_prompt, user_prompt)
        if not text:
            return None
        text = self._clean_tweet(text)
        if not text:
            return None

        # ── Extra safety: catch chain-specific prompt leaks ──
        # These patterns should NEVER appear in a real tweet
        chain_leak_patterns = [
            r"(?:based on|from|according to)\s+(?:our|my)\s+(?:conversation|interaction|chain)",
            r"(?:i decided|i chose) to (?:continue|reply|respond|engage)",
            r"\binterest (?:score|level|evaluation)\b",
            r"\breply chain\b",
            r"\bdeliberation\b",
            r"\bthread depth\b",
            r"(?:my |the )memory (?:indicates|shows|says)",
            r"\bconversation history\b",
            r"you(?:'ve| have) interacted \d+ times",
        ]
        text_lower = text.lower()
        for pattern in chain_leak_patterns:
            if re.search(pattern, text_lower):
                logger.warning(f"🔗 Chain reply rejected,  prompt leak detected: {text[:80]}...")
                return None

        return text

    def _get_own_username(self) -> str:
        """Get the bot's own X username to filter out self-replies."""
        # Try to get from config, fallback to empty (safe,  won't filter anything)
        return str(getattr(self.config, "x_username", "") or "")

    # ══════════════════════════════════════════════════════════════════
    #  ACTIVITY: JOIN TREND
    # ══════════════════════════════════════════════════════════════════

    async def _join_trend(self):
        """Find a trending topic and post about it."""
        try:
            trends = await self._x().get_trends()
            if not trends.get("success") or not trends.get("trends"):
                return

            for trend in trends["trends"][:15]:
                trend_name = trend.get("name", "")
                if not trend_name:
                    continue

                relevant = any(
                    t.lower() in trend_name.lower() or trend_name.lower() in t.lower()
                    for t in self.topics
                )
                if not relevant and random.random() > 0.15:
                    continue
                if trend_name in self._posted_urls:
                    continue

                # Pause,  "reading about the trend"
                await asyncio.sleep(self._human_delay(10, 25))

                tweet_text = await self._generate_trend_take(trend_name)
                if tweet_text:
                    result = await self._post({"type": "tweet", "text": tweet_text, "style": self.style})
                    if result.get("success"):
                        self._posts_today += 1
                        self._posted_urls.add(trend_name)
                        self._history.append({
                            "ts": datetime.now().isoformat(),
                            "type": "trend",
                            "topic": trend_name,
                            "tweet": tweet_text[:100],
                            "tweet_id": result.get("tweet_id"),
                        })
                        self._save_state()
                        self.mind.remember("trend_joined", {
                            "trend": trend_name,
                            "tweet": tweet_text[:200],
                            "tweet_id": result.get("tweet_id"),
                        })
                        logger.info(f"\U0001f4c8 Posted about trend: {trend_name}")
                        return  # one per session

        except Exception as e:
            logger.debug(f"Join trend error: {e}")

    # ══════════════════════════════════════════════════════════════════
    #  CONTENT GENERATION (Grok / LLM)
    # ══════════════════════════════════════════════════════════════════

    async def _generate_tweet(self, story: Dict, *, force_thread: bool = False) -> Optional[Dict]:
        """Generate an original tweet from a news story,  voice comes from identity."""
        story_text = story.get("title", "")
        if story.get("description"):
            story_text += f"\n\n{story['description'][:300]}"
        if story.get("url"):
            story_text += f"\n\nSource: {story['url']}"

        # Feel the story emotionally before writing about it (AI-powered)
        await self.mind.feel(story_text)

        # System prompt is the agent's evolved voice + mood + style modifier
        style_mod = random.choice(STYLE_MODIFIERS)
        system_prompt = f"{self.mind.get_voice_prompt()}\n\nStyle hint: {style_mod}"

        # Decide format based on personality: threads when passion is high
        do_thread = force_thread or (
            self.mind._mood_intensity > 0.7
            and random.random() < 0.3
        )
        if do_thread:
            system_prompt += (
                "\n\nYour passion on this topic is intense. Write a thread (4-6 posts, "
                "numbered 1/, 2/, etc). Hook first, emotional takeaway last."
            )

        user_prompt = f"Write a post about this:\n\n{story_text}"
        if self.language != "en":
            user_prompt += f"\n\nWrite in {self.language}."

        text = await self._ask_ai(system_prompt, user_prompt)
        if not text:
            return None

        if do_thread:
            tweets = self._parse_thread(text)
            if tweets:
                return {"type": "thread", "tweets": tweets, "style": "identity"}

        tweet_text = self._clean_tweet(text)
        return {"type": "tweet", "text": tweet_text, "style": "identity"} if tweet_text else None

    async def _generate_reply(self, tweet_text: str, username: str, *, media_description: Optional[str] = None) -> Optional[str]:
        """Generate an emotionally-aware reply,  personality determines tone."""
        # Feel the tweet before replying (AI-powered for important interactions)
        await self.mind.feel(tweet_text)
        mood = self.mind._mood

        # Pick reply archetype based on emotional state (dynamic, not hardcoded)
        valence, arousal = EMOTION_SPECTRUM.get(mood, (0.0, 0.2))
        if valence < -0.5 and arousal > 0.6:
            reply_type = random.choice(["debate", "outraged", "witty"])
        elif valence < -0.2:
            reply_type = random.choice(["debate", "add_info", "empathetic"])
        elif valence > 0.4:
            reply_type = random.choice(["agree", "witty", "add_info"])
        else:
            reply_type = random.choice(list(REPLY_ARCHETYPES.keys()))

        archetype_hint = REPLY_ARCHETYPES.get(reply_type, "Write a short reply (max 280 chars).")

        # Voice from identity + archetype hint
        system_prompt = f"{self.mind.get_voice_prompt()}\n\nReply mode: {archetype_hint}"
        if self.language != "en":
            system_prompt += f" Write in {self.language}."

        # Build user prompt,  include media description if we "saw" images
        user_prompt = f"@{username} posted:\n\"{tweet_text[:500]}\""
        if media_description:
            user_prompt += f"\n\n[Visual content in the post]: {media_description[:500]}"
        user_prompt += "\n\nWrite your reply:"

        text = await self._ask_ai(system_prompt, user_prompt)
        return self._clean_tweet(text) if text else None

    async def _generate_quote(self, tweet_text: str, username: str, *, media_description: Optional[str] = None) -> Optional[str]:
        """Generate personality-driven quote-tweet commentary."""
        await self.mind.feel(tweet_text)
        system_prompt = (
            f"{self.mind.get_voice_prompt()}\n\n"
            "Write a quote-post comment (max 280 chars) adding your authentic take. "
            "Add value,  an insight, prediction, or gut reaction."
        )
        if self.language != "en":
            system_prompt += f" Write in {self.language}."

        user_prompt = f"@{username} posted:\n\"{tweet_text[:500]}\""
        if media_description:
            user_prompt += f"\n\n[Visual content in the post]: {media_description[:500]}"
        user_prompt += "\n\nWrite your quote-post:"

        text = await self._ask_ai(system_prompt, user_prompt)
        return self._clean_tweet(text) if text else None

    async def _generate_mention_reply(self, mention_text: str, username: str, *, media_description: Optional[str] = None) -> Optional[str]:
        """Generate an emotionally-aware reply to someone who mentioned us."""
        await self.mind.feel(mention_text)
        system_prompt = (
            f"{self.mind.get_voice_prompt()}\n\n"
            "Someone mentioned you on X. Write a reply (max 280 chars) that feels REAL. "
            "Be conversational and authentic. If they asked a question, answer it honestly."
        )
        if self.language != "en":
            system_prompt += f" Write in {self.language}."

        user_prompt = f"@{username} mentioned you:\n\"{mention_text[:500]}\""
        if media_description:
            user_prompt += f"\n\n[Visual content in the post]: {media_description[:500]}"
        user_prompt += "\n\nYour reply:"
        text = await self._ask_ai(system_prompt, user_prompt)
        return self._clean_tweet(text) if text else None

    async def _generate_trend_take(self, trend_name: str) -> Optional[str]:
        """Generate a personality-driven hot take on a trending topic."""
        await self.mind.feel(trend_name)
        system_prompt = self.mind.get_voice_prompt()
        if self.language != "en":
            system_prompt += f" Write in {self.language}."

        user_prompt = (
            f"\"{trend_name}\" is trending on X right now. "
            f"Write a post (max 280 chars) with your take on why it's trending "
            f"or what it means. Include the hashtag if relevant."
        )
        text = await self._ask_ai(system_prompt, user_prompt)
        if not text:
            # Retry with a stripped-down prompt,  some LLMs choke on long system prompts
            minimal_system = "You are a witty social media personality. Write only the tweet, nothing else."
            if self.language != "en":
                minimal_system += f" Write in {self.language}."
            minimal_user = f'"{trend_name}" is trending. Write your hot take in max 240 chars. Output ONLY the tweet text,  complete sentence, no prefixes, no code, nothing else.'
            text = await self._ask_ai(minimal_system, minimal_user)
        return self._clean_tweet(text) if text else None

    @staticmethod
    def _salvage_tweet_from_meta(text: str) -> Optional[str]:
        """Try to extract an actual tweet from a response that failed the meta-check.
        Looks for quoted text, or last short paragraph that could be a real post."""
        # Try: find content inside quotes
        quoted = re.findall(r'["\u201c\u201d]([^"“”]{10,280})["\u201c\u201d]', text)
        for q in reversed(quoted):
            q = q.strip()
            if len(q) >= 10 and not XAutonomousAgent._is_meta_response(q):
                return q
        # Try: last non-empty paragraph that's short enough to be a tweet
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        for para in reversed(paragraphs):
            if 10 <= len(para) <= 290 and not XAutonomousAgent._is_meta_response(para):
                return para
        return None

    async def _ask_ai(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Ask the configured LLM (primary) or fall back to Grok chat."""
        # Inject anti-meta-commentary directive so the LLM never adds bot-revealing text
        system_prompt += (
            "\n\nCRITICAL: Output ONLY the final text to post. "
            "NEVER include meta-commentary, stage directions, visual notes, "
            "descriptions of images/videos, disclaimers, tone annotations, "
            "or parenthetical remarks about how you are replying. "
            "No brackets like [Visual note:...] or parentheticals like (Replying in...). "
            "Do NOT wrap your reasoning in <think> tags. "
            "Write a COMPLETE thought that ends naturally,  never stop mid-sentence. "
            "Keep it under 240 characters so it fits comfortably in a tweet. "
            "Just the raw post text, nothing else. No code, no random tokens, no prefixes."
        )
        text = await self._ask_llm(system_prompt, user_prompt)
        if text and self._is_error_response(text):
            logger.warning(f"X autoposter: rejected primary LLM error payload ({text[:120]}...)")
            text = None
        if not text:
            text = await self._ask_grok(system_prompt, user_prompt)
        # Defence-in-depth: strip <think> blocks from ANY LLM/Grok response
        if text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
            text = text.strip()
            # Strip leading role labels that some models prepend (e.g. "system\n...", "assistant:")
            text = re.sub(r'^(system|user|assistant|human)\s*[:;\n]\s*', '', text, flags=re.IGNORECASE)
            text = text.strip()
        if text and self._is_error_response(text):
            logger.warning(f"X autoposter: rejected fallback LLM error payload ({text[:120]}...)")
            return None
        # Reject meta-responses where the model echoes/summarises the system prompt
        # instead of generating actual content
        if text and self._is_meta_response(text):
            # Try to salvage: look for quoted content or last clean short paragraph
            salvaged = self._salvage_tweet_from_meta(text)
            if salvaged:
                logger.debug(f"X autoposter: salvaged tweet from meta-response")
                return salvaged
            logger.warning(f"X autoposter: rejected meta-response from LLM ({text[:100]}...)")
            return None
        return text or None

    async def _ask_grok(self, system: str, user: str) -> Optional[str]:
        try:
            if not getattr(self.agent.tools, "grok_skill", None):
                return None
            from opensable.skills.social.grok_skill import TWIKIT_GROK_AVAILABLE
            if not TWIKIT_GROK_AVAILABLE:
                return None
            result = await self.agent.tools.grok_skill.chat(f"{system}\n\n{user}")
            if result.get("success"):
                return result.get("response", "")
        except Exception as e:
            logger.debug(f"Grok failed: {e}")
        return None

    async def _ask_llm(self, system: str, user: str) -> Optional[str]:
        try:
            # Append /no_think directly to the user message so Qwen3 / reasoning
            # models suppress their chain-of-thought output.
            user_no_think = user.rstrip() + " /no_think"
            response = await asyncio.wait_for(
                self.agent.llm.invoke_with_tools(
                    [{"role": "system", "content": system}, {"role": "user", "content": user_no_think}],
                    [],
                ),
                timeout=60,
            )
            return response.get("text", "")
        except Exception as e:
            logger.debug(f"LLM failed: {e}")
        return None

    # ══════════════════════════════════════════════════════════════════
    #  NEWS FETCHING
    # ══════════════════════════════════════════════════════════════════

    async def _fetch_news(self) -> List[Dict]:
        """Fetch news from RSS + search ONE random topic on X."""
        stories = []
        feeds = self.custom_feeds or DEFAULT_FEEDS
        rss_stories = await self._fetch_rss(feeds)
        stories.extend(rss_stories)

        # Search X for ONE random topic — skip if self-heal has disabled search
        # (e.g. after a 404 on the GraphQL endpoint) so we don't keep hitting
        # a broken endpoint and wasting time before the post.
        search_ok = not self._healer.remedy.is_search_disabled()
        if self.topics and search_ok:
            topic = random.choice(self.topics)
            try:
                result = await self._x().search_tweets(
                    f"{topic} news", search_type="Top", count=5
                )
                if result.get("success"):
                    for t in result.get("tweets", []):
                        stories.append({
                            "title": t.get("text", "")[:200],
                            "source": f"x/@{t.get('username', '')}",
                            "url": f"x_tweet_{t.get('id', '')}",
                        })
            except Exception:
                pass
        elif not search_ok:
            logger.debug("🔍 Skipping X search in _fetch_news (search disabled by self-heal)")
        return stories

    async def _fetch_rss(self, feed_urls: List[str]) -> List[Dict]:
        stories = []
        try:
            import aiohttp
            from xml.etree import ElementTree

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                selected = random.sample(feed_urls, min(3, len(feed_urls)))
                # Fetch RSS feeds sequentially (not concurrent) for stealth
                for url in selected:
                    try:
                        result = await self._fetch_single_feed(session, url)
                        if isinstance(result, list):
                            stories.extend(result)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"RSS error: {e}")
        return stories

    async def _fetch_single_feed(self, session, url: str) -> List[Dict]:
        items = []
        try:
            from xml.etree import ElementTree

            async with session.get(url, ssl=False) as resp:
                if resp.status != 200:
                    return items
                text = await resp.text()

            root = ElementTree.fromstring(text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for item in root.iter("item"):
                title = getattr(item.find("title"), "text", None)
                link = getattr(item.find("link"), "text", None)
                desc = getattr(item.find("description"), "text", None)
                if title:
                    items.append({
                        "title": title.strip(), "url": (link or "").strip(),
                        "description": (desc or "")[:500],
                        "source": url.split("/")[2] if "/" in url else url,
                    })

            if not items:
                for entry in root.findall("atom:entry", ns):
                    title = getattr(entry.find("atom:title", ns), "text", None)
                    link_el = entry.find("atom:link", ns)
                    link = link_el.get("href", "") if link_el is not None else ""
                    if title:
                        items.append({
                            "title": title.strip(), "url": link.strip(),
                            "description": "", "source": url.split("/")[2] if "/" in url else url,
                        })
        except Exception as e:
            logger.debug(f"Feed parse error: {e}")
        return items[:5]

    def _pick_story(self, stories: List[Dict]) -> Optional[Dict]:
        random.shuffle(stories)
        for story in stories:
            key = story.get("url") or story.get("title", "")
            if key and key not in self._posted_urls:
                return story
        return None

    # ══════════════════════════════════════════════════════════════════
    #  IMAGE GENERATION,  Grok creates visuals for posts
    # ══════════════════════════════════════════════════════════════════

    async def _maybe_generate_image(self, tweet_text: str, story: Dict) -> Optional[str]:
        """
        Decide whether to generate an image for this post, and if so, create one.
        Returns the local file path of the generated image, or None.

        Budget: max x_max_daily_images per day (default 4).
        Probability: ~25% of posts get images, biased toward:
          - High emotional intensity (passion, outrage, excitement)
          - Visual/concrete topics (tech, space, art, disasters)
          - Not threads (only standalone posts)
        """
        # Budget check
        if self._grok_images_today >= self._max_grok_images_daily:
            return None

        # If Grok image endpoint has been failing (add_response 404), skip entirely
        # to avoid triggering the 226 automated-detection on create_tweet.
        if getattr(self, '_grok_image_failures', 0) >= 2:
            return None

        # Grok available?
        grok = getattr(self.agent.tools, "grok_skill", None)
        if not grok:
            return None
        try:
            from opensable.skills.social.grok_skill import TWIKIT_GROK_AVAILABLE
            if not TWIKIT_GROK_AVAILABLE:
                return None
        except ImportError:
            return None

        # Decision gate,  curiosity/emotion-driven
        mood_intensity = getattr(self.mind, "_mood_intensity", 0.5)
        mood = getattr(self.mind, "_mood", "neutral")

        # Visual topics boost probability
        visual_cues = ("space", "city", "war", "protest", "ai", "robot", "disaster",
                       "explosion", "nature", "art", "design", "future", "tech",
                       "satellite", "mars", "ocean", "fire", "storm", "drone")
        topic_text = (tweet_text + " " + story.get("title", "")).lower()
        has_visual_topic = any(cue in topic_text for cue in visual_cues)

        # High-emotion moods boost image probability
        high_visual_moods = ("excited", "outraged", "inspired", "angry", "shocked", "passionate")
        mood_boost = mood in high_visual_moods

        # Base 15% chance, boosted by visual topics (+15%) and mood (+10%)
        p_image = 0.15
        if has_visual_topic:
            p_image += 0.15
        if mood_boost:
            p_image += 0.10
        if mood_intensity > 0.7:
            p_image += 0.10

        if random.random() > p_image:
            return None

        # Generate image
        try:
            # Build a visual prompt,  tell Grok to create art inspired by the post
            img_prompt = (
                f"Generate a striking, artistic image that captures the essence of this:\n"
                f"\"{tweet_text[:200]}\"\n\n"
                f"Style: cinematic, evocative, no text overlays, no watermarks. "
                f"The image should work as a visual companion to a social media post."
            )

            save_path = f"/tmp/sable_post_img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            result = await grok.generate_image(img_prompt, save_path=save_path)

            if result.get("success") and result.get("images"):
                self._grok_images_today += 1
                self._grok_image_failures = 0  # reset on success
                image_file = result["images"][0]
                logger.info(
                    f"🎨 Generated image for post [{self._grok_images_today}/{self._max_grok_images_daily}]: "
                    f"{image_file}"
                )
                self.mind.remember("image_generated", {
                    "tweet": tweet_text[:200],
                    "image": image_file,
                    "mood": mood,
                    "intensity": mood_intensity,
                })
                return image_file
            else:
                self._grok_image_failures = getattr(self, '_grok_image_failures', 0) + 1
                logger.debug(
                    f"Image gen returned no images: {result.get('error', '?')} "
                    f"(failures={self._grok_image_failures})"
                )
                return None

        except Exception as e:
            self._grok_image_failures = getattr(self, '_grok_image_failures', 0) + 1
            logger.debug(f"Image generation failed: {e} (failures={self._grok_image_failures})")
            return None

    # ══════════════════════════════════════════════════════════════════
    #  POSTING
    # ══════════════════════════════════════════════════════════════════

    async def _post(self, content: Dict) -> Dict:
        # Block posting if self-heal has triggered stealth mode
        if self._healer.remedy.is_loop_paused("post"):
            logger.info("\U0001f4dd Post blocked,  self-heal stealth mode active")
            return {"success": False, "error": "stealth_mode"}

        if content.get("type") == "tweet":
            candidate = str(content.get("text", "") or "")
            if self._is_error_response(candidate):
                logger.warning(f"X autoposter: blocked invalid tweet payload before publish ({candidate[:120]}...)")
                return {"success": False, "error": "invalid_tweet_payload"}

        if self.dry_run:
            logger.info(f"\U0001f3dc\ufe0f DRY RUN,  {content.get('type')}: {str(content.get('text', content.get('tweets', '')))[:80]}")
            return {"success": True, "tweet_id": "dry_run", "url": "dry_run"}
        try:
            media_paths = content.get("media_paths")
            if content["type"] == "thread":
                result = await self._x().post_thread(content["tweets"])
            else:
                result = await self._x().post_tweet(
                    content["text"],
                    media_paths=media_paths,
                )
            # Check for error codes in result
            error_str = str(result.get("error", ""))
            if not result.get("success"):
                if result.get("tombstone"):
                    # Tweet reached X but was immediately tombstoned (226 shadow still active).
                    # Count it as posted so we don't retry and double-post.
                    logger.warning("⚠️ Tweet tombstoned — sent to X but hidden (226 shadow active)")
                    return {"success": True, "tweet_id": None, "url": None, "tombstone": True}
                elif "344" in error_str or "daily limit" in error_str.lower():
                    self._daily_limit_hit = True
                    self._daily_limit_hit_at = datetime.now()
                    self._save_state()
                    logger.warning(
                        f"\U0001f6ab X daily post limit (344) hit "
                        f"\u2014 will retry automatically in {self._limit_retry_hours:.0f}h"
                    )
                elif "226" in error_str or "automated" in error_str.lower():
                    logger.warning("\U0001f6ab Post rejected (226),  account flagged as automated")
                else:
                    # Surface all other failures so they don't disappear silently
                    logger.warning(f"\U0001f4dd Post failed: {error_str[:200] or 'unknown error'}")
            return result
        except Exception as e:
            logger.warning(f"\U0001f4dd Post exception: {e}")
            return {"success": False, "error": str(e)}

    # ══════════════════════════════════════════════════════════════════
    #  UTILS
    # ══════════════════════════════════════════════════════════════════

    def _pick_style(self) -> str:
        """
        Legacy compat,  style is now driven by identity.voice,
        not a hardcoded dict. Returns a label for logging only.
        """
        return "identity"

    def _clean_tweet(self, text: str) -> str:
        if not text:
            return ""
        if self._is_error_response(text):
            return ""
        # ── Strip <think> blocks FIRST (content + tags) ───────────────
        # Models emit <think>reasoning…</think> before the actual reply;
        # we must remove the ENTIRE block, not just the XML tags.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Also handle unclosed <think> (model started reasoning, never closed)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        # ── Strip untagged reasoning preamble ─────────────────────────
        # Models (especially Qwen3) sometimes output planning text BEFORE the
        # actual reply WITHOUT wrapping it in <think> tags, e.g.:
        #   "I should engage with this in a way that's:\nRespectful...\n1."
        # We strip leading paragraphs that look like reasoning until we find
        # a paragraph that reads like an actual tweet.
        _REASONING_LINE = re.compile(
            r'^(i (should|need to|want to|will|must|can)|let me|first[,.]|'
            r'okay[,.]|the (tweet|post|reply|best|right|user)|this (requires|needs|should|seems|is)|'
            r'considering|thinking about|my (approach|response|reply)|to (craft|write|respond)|'
            r'key (points|arguments|ideas)|their (tweet|post|reply|message|point)|'
            r'seems to be|what they|engage (authentically|with)|\d+\.\s)',
            re.IGNORECASE,
        )
        paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        if len(paras) > 1:
            real_start = 0
            for i, para in enumerate(paras):
                first_line = para.splitlines()[0].strip()
                if (_REASONING_LINE.match(first_line)
                        or first_line.endswith(':')
                        or len(para) > 500):
                    real_start = i + 1
                else:
                    break
            if real_start > 0 and real_start < len(paras):
                text = '\n\n'.join(paras[real_start:])
            elif real_start == 0:
                pass  # first para looked fine, keep as-is
        # Single-line reasoning: strip lines that start with reasoning markers
        # until we hit a line that looks like actual content (no reasoning opener)
        lines = text.splitlines()
        if len(lines) > 1:
            cleaned = []
            found_content = False
            for line in lines:
                s = line.strip()
                if not found_content and s and _REASONING_LINE.match(s):
                    continue  # skip reasoning line
                else:
                    found_content = True
                    cleaned.append(line)
            if cleaned:
                text = '\n'.join(cleaned)
        text = text.strip()
        # ── Strip Grok/LLM artefacts ─────────────────────────────────
        # Remove <xai:...> XML tags that Grok sometimes injects (tool cards, etc.)
        text = re.sub(r"<xai:[^>]*>.*?</xai:[^>]*>", "", text, flags=re.DOTALL)
        text = re.sub(r"<xai:[^>]*/?>", "", text)
        # Remove any remaining XML/HTML-like tags that aren't part of content
        text = re.sub(r"</?[a-zA-Z_][a-zA-Z0-9_:.-]*(?:\s[^>]*)?>", "", text)

        # ── Strip LLM meta-commentary that exposes bot behaviour ──────
        # Parenthetical meta: (Replying in Sable's characteristic… ), (In my usual tone… )
        text = re.sub(
            r"\((?:Replying|Responding|Writing|Speaking|Posted|Tweeting|In my|In Sable|"
            r"As Sable|With Sable|Using|Note:|Author'?s?\s*note)"
            r"[^)]{0,300}\)?",
            "", text, flags=re.IGNORECASE | re.DOTALL,
        )
        # Bracketed meta: [Visual note: ...], [Note: ...], [Context: ...], [Image: ...]
        text = re.sub(
            r"\[(?:Visual\s*(?:note|content|description)|Note|Context|Image|Video|"
            r"Media|Content\s*note|Editor'?s?\s*note|Author'?s?\s*note|"
            r"Alt\s*text|Description|Disclaimer|Tone|Style|Voice|Mood)"
            r"[:\s][^\]]{0,500}\]?",
            "", text, flags=re.IGNORECASE | re.DOTALL,
        )
        # Asterisk-wrapped stage directions: *adjusts glasses*, *laughs*
        text = re.sub(r"\*[^*\n]{2,80}\*", "", text)
        # "As an AI..." / "As a language model..." / "In character as..." disclaimers
        text = re.sub(
            r"(?:^|\n)\s*(?:As an? (?:AI|language model|bot|assistant)|In character(?: as)?)[^.\n]{0,150}[.\n]",
            "", text, flags=re.IGNORECASE,
        )

        # ── Strip markdown formatting ─────────────────────────────────
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)
        text = re.sub(r"^#{1,4}\s+.*\n?", "", text, flags=re.MULTILINE)  # markdown headers
        text = re.sub(r"^(here'?s?\s*(a|my|the)\s*)?tweet:?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(here'?s?\s*(a|my|the)\s*)?post:?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(here'?s?\s*(a|my|the)\s*)?reply:?\s*", "", text, flags=re.IGNORECASE)
        # ── Strip LLM-style dashes and bullets ────────────────────────
        # em/en dash: normalize spacing so "word—word" → "word - word"
        # First, collapse any existing surrounding spaces to avoid double-spaces
        text = re.sub(r"\s*\u2014\s*", " - ", text)   # em dash
        text = re.sub(r"\s*\u2013\s*", " - ", text)   # en dash
        text = text.replace("\u2022", "-")   # bullet → dash
        text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)  # leading bullets
        # ── Strip LLM role prefixes ───────────────────────────────
        # Models sometimes echo their role: "Assistant The link..." or "Assistant: ..."
        text = re.sub(r"^(?:Assistant|User|System|Human|AI|Bot)\s*:?\s+", "", text, flags=re.IGNORECASE)
        # Also strip multi-line role-label blocks: "system\nThe user has..."
        text = re.sub(r"^(?:system|user|assistant|human)\s*\n", "", text, flags=re.IGNORECASE)
        # ── Strip DeepSeek / LLM artifact tokens at the start ────────
        # Some models leak code fragments or random tokens before the actual reply.
        # Examples: "rand_range(0,1) # seed: 42\n", "ipro ", "0x1f\n", etc.
        # Strategy: strip any leading line that looks like code or a lone short token.
        lines = text.split("\n")
        cleaned_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                if cleaned_lines:           # keep blank lines only after real content
                    cleaned_lines.append(line)
                continue
            # Detect garbage line: looks like code (has brackets, operators, #comments)
            # OR is a lone word/token ≤6 chars that can't start a sentence
            is_code_like = bool(re.search(r"[()=;{}\[\]]|#\s*\w|->|::", stripped))
            is_lone_token = len(stripped) <= 6 and not re.search(r"[.!?,]", stripped) and i == 0
            if (is_code_like or is_lone_token) and not cleaned_lines:
                continue    # skip artifact lines before any real content
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
        text = text.strip().strip('"').strip("'").strip()
        if self._is_error_response(text):
            return ""
        # Collapse accidental double-spaces or leftover whitespace
        text = re.sub(r"\s{2,}", " ", text).strip()
        # ── Strip trailing ellipsis the LLM uses as a stylistic cliffhanger ──
        # e.g. "But the..." or "…",  these indicate an incomplete thought
        text = re.sub(r"\s*\.{2,}\s*$", "", text).strip()   # trailing ... or ....
        text = re.sub(r"\s*\u2026\s*$", "", text).strip()   # trailing …
        # ── Ensure tweet ends at a complete sentence ─────────────────
        # If text doesn't end with sentence-ending punctuation, cut at the last one
        if text and not re.search(r"[.!?]$", text):
            match = re.search(r"[.!?](?=[^.!?]*$)", text)
            if match:
                text = text[:match.start() + 1].strip()
        # ── Respect X's 280-char limit ────────────────────────────────
        if len(text) > 280:
            # Cut at last sentence boundary within 280 chars
            match = re.search(r"[.!?](?=[^.!?]*$)", text[:280])
            if match:
                text = text[:match.start() + 1].strip()
            else:
                text = text[:280].rsplit(" ", 1)[0].strip()
        return text if len(text) >= 10 else ""

    def _parse_thread(self, text: str) -> List[str]:
        tweets = []
        parts = re.split(r"\n\s*(?:\d+[/\.\):]|Tweet\s*\d+:?)\s*", text)
        for part in parts:
            cleaned = self._clean_tweet(part.strip())
            if cleaned:
                tweets.append(cleaned)
        return tweets if len(tweets) >= 2 else []

    # ══════════════════════════════════════════════════════════════════
    #  STATE PERSISTENCE
    # ══════════════════════════════════════════════════════════════════

    def _save_state(self):
        try:
            state = {
                "posts_today": self._posts_today,
                "engagements_today": self._engagements_today,
                "last_reset": str(self._last_reset),
                "posted_urls": list(self._posted_urls)[-300:],
                "engaged_ids": list(self._engaged_tweet_ids)[-500:],
                "followed_users": list(self._followed_users)[-200:],
                "known_users": dict(sorted(self._known_users.items(), key=lambda x: -x[1])[:300]),
                "inspiration_level": self._inspiration_level,
                "history": self._history[-200:],
                "engagement_log": self._engagement_log[-200:],
                "style_scores": self._style_scores,
                "saved_at": datetime.now().isoformat(),
                "last_post_at": self._last_post_at.isoformat() if getattr(self, '_last_post_at', None) else None,
                "last_engage_at": self._last_engage_at.isoformat() if getattr(self, '_last_engage_at', None) else None,
                "daily_limit_hit": getattr(self, '_daily_limit_hit', False),
                "daily_limit_hit_at": self._daily_limit_hit_at.isoformat() if getattr(self, '_daily_limit_hit_at', None) else None,
                "mention_queue": self._mention_queue[-100:],  # persist pending mentions
                "reply_chains": dict(list(self._reply_chains.items())[-50:]),  # active reply chains
                "reply_chain_replies_today": self._reply_chain_replies_today,
            }
            self._state_file.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            logger.debug(f"Save state failed: {e}")

    def _load_state(self):
        try:
            if not self._state_file.exists():
                return
            state = json.loads(self._state_file.read_text())
            self._posted_urls = set(state.get("posted_urls", []))
            self._engaged_tweet_ids = set(state.get("engaged_ids", []))
            self._followed_users = set(state.get("followed_users", []))
            self._known_users = state.get("known_users", {})
            self._inspiration_level = state.get("inspiration_level", 0.5)
            self._history = state.get("history", [])
            self._engagement_log = state.get("engagement_log", [])
            self._style_scores = state.get("style_scores", self._style_scores)
            self._mention_queue = state.get("mention_queue", [])
            self._reply_chains = state.get("reply_chains", {})

            if state.get("last_reset") == str(datetime.now().date()):
                self._posts_today = state.get("posts_today", 0)
                self._engagements_today = state.get("engagements_today", 0)
                self._reply_chain_replies_today = state.get("reply_chain_replies_today", 0)
                # Restore 344 cooldown only if it was hit recently enough to still matter
                raw_hit_at = state.get("daily_limit_hit_at")
                if raw_hit_at:
                    try:
                        hit_at = datetime.fromisoformat(raw_hit_at)
                        elapsed_h = (datetime.now() - hit_at).total_seconds() / 3600
                        if elapsed_h < self._limit_retry_hours:
                            self._daily_limit_hit = True
                            self._daily_limit_hit_at = hit_at
                        # else: cooldown already expired — start fresh, no block
                    except (ValueError, TypeError):
                        pass
                # Older state files only had a bool (no timestamp) — don't restore that block

            if self._daily_limit_hit:
                elapsed = (datetime.now() - self._daily_limit_hit_at).total_seconds() / 3600
                retry_in = max(0, self._limit_retry_hours - elapsed)
                logger.warning(
                    f"\U0001f6ab X 344 cooldown active (hit {elapsed:.1f}h ago) "
                    f"\u2014 posts resume automatically in ~{retry_in:.1f}h"
                )

            logger.info(
                f"Loaded: {len(self._posted_urls)} posted, "
                f"{len(self._engaged_tweet_ids)} engaged, "
                f"{len(self._followed_users)} followed, "
                f"{self._posts_today} posts today, "
                f"{self._engagements_today} engagements today"
            )
        except Exception as e:
            logger.warning(f"Load state failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        posts_remaining = self.max_daily_posts - self._posts_today
        hours_left = self._remaining_active_hours()
        next_interval = self._optimal_post_interval() if posts_remaining > 0 else 0
        return {
            "running": self.running,
            "posts_today": self._posts_today,
            "posts_remaining": posts_remaining,
            "engagements_today": self._engagements_today,
            "max_daily_posts": self.max_daily_posts,
            "max_daily_engagements": self.max_daily_engagements,
            "post_interval_min": self.post_interval,
            "post_interval_dynamic": round(next_interval),
            "active_hours_remaining": round(hours_left, 1),
            "next_post_in_minutes": round(next_interval / 60) if posts_remaining > 0 else None,
            "engage_interval": self.engage_interval,
            "style": self.style,
            "topics": self.topics,
            "accounts_watched": self.accounts_to_watch,
            "dry_run": self.dry_run,
            "total_posted": len(self._history),
            "total_engagements": len(self._engagement_log),
            "followed": len(self._followed_users),
            "known_users": len(self._known_users),
            "inspiration_level": self._inspiration_level,
            "last_post": self._history[-1] if self._history else None,
            "last_engagement": self._engagement_log[-1] if self._engagement_log else None,
            "self_heal": self._healer.get_status(),
            "mode": "sequential",
            "consciousness_cycles": self._consciousness_cycle,
            "reply_chains_active": len(self._reply_chains),
            "reply_chain_replies_today": self._reply_chain_replies_today,
            "max_reply_chain_daily": self._max_reply_chain_daily,
        }


# Backward compat alias
XAutoposter = XAutonomousAgent
