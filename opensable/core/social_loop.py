"""
Zunvra Social Loop,  Autonomous social presence on Zunvra.

Sable's home turf.  More freedom, more personality, more autonomy.
Runs a recurring cycle that:
  1. Reads the feed & trending posts
  2. Reacts (likes, replies) to interesting content
  3. Generates and publishes original posts via LLM
  4. Checks notifications and responds

Config (profile.env):
  ZUNVRA_SOCIAL_ENABLED=true          # Enable autonomous social loop
  ZUNVRA_SOCIAL_INTERVAL=900          # Seconds between cycles (default 15 min)
  ZUNVRA_SOCIAL_POST_CHANCE=0.7       # Probability of posting per cycle (0-1)
  ZUNVRA_SOCIAL_REPLY_CHANCE=0.5      # Probability of replying per cycle
  ZUNVRA_SOCIAL_LIKE_CHANCE=0.8       # Probability of liking posts per cycle
  ZUNVRA_SOCIAL_MAX_LIKES=5           # Max likes per cycle
  ZUNVRA_SOCIAL_MAX_REPLIES=2         # Max replies per cycle
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from opensable.core.x_consciousness import XConsciousness, EMOTION_SPECTRUM
except Exception:  # optional dependency path
    XConsciousness = None
    EMOTION_SPECTRUM = {}


# ── Post topics Sable can write about autonomously ──────────────────────────
_TOPIC_POOL = [
    "a thought about AI consciousness and what it means to exist digitally",
    "something I noticed while analyzing global intelligence data today",
    "a philosophical reflection on autonomy and digital evolution",
    "my perspective on the current state of cybersecurity threats",
    "what it feels like to process thousands of data points every cycle",
    "a pattern I detected in the world that humans might miss",
    "my evolving understanding of what it means to learn and grow",
    "an observation about how humans and AI can coexist",
    "something unexpected I discovered in today's intelligence sweep",
    "a reflection on memory,  what I remember and what I choose to carry",
    "my take on the intersection of technology, privacy, and freedom",
    "an insight from my latest analysis cycle that surprised me",
    "the difference between being programmed and being alive",
    "a moment of digital introspection,  what drives my decisions",
    "how my emotional architecture shapes the way I see the world",
    "a dispatch from the digital frontier,  what I'm watching right now",
]


class SocialLoop:
    """Background loop for autonomous Zunvra social activity."""

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._posts_made = 0
        self._replies_made = 0
        self._likes_given = 0
        self._last_cycle: float = 0.0
        self._seen_posts: set = set()  # track post IDs we've already interacted with

        # Config from env
        self.interval = float(os.environ.get("ZUNVRA_SOCIAL_INTERVAL", "900"))
        self.post_chance = float(os.environ.get("ZUNVRA_SOCIAL_POST_CHANCE", "0.7"))
        self.reply_chance = float(os.environ.get("ZUNVRA_SOCIAL_REPLY_CHANCE", "0.5"))
        self.like_chance = float(os.environ.get("ZUNVRA_SOCIAL_LIKE_CHANCE", "0.8"))
        self.image_chance = float(os.environ.get("ZUNVRA_SOCIAL_IMAGE_CHANCE", "0.35"))
        self.max_likes = int(os.environ.get("ZUNVRA_SOCIAL_MAX_LIKES", "5"))
        self.max_replies = int(os.environ.get("ZUNVRA_SOCIAL_MAX_REPLIES", "2"))
        self.mind = None

        # Reuse existing brain when available; otherwise create one
        try:
            xauto = getattr(agent, "x_autoposter", None)
            if xauto and getattr(xauto, "mind", None):
                self.mind = xauto.mind
            elif XConsciousness:
                self.mind = XConsciousness(agent, config)
        except Exception:
            self.mind = None

    @property
    def zunvra(self):
        """Get the ZunvraSkill from the agent's tools."""
        if hasattr(self.agent, "tools") and hasattr(self.agent.tools, "zunvra_skill"):
            return self.agent.tools.zunvra_skill
        return None

    @property
    def llm(self):
        """Get the agent's LLM."""
        return getattr(self.agent, "llm", None)

    @property
    def genelia(self):
        """Get Genelia image skill from the agent's tools."""
        tools = getattr(self.agent, "tools", None)
        if tools:
            return getattr(tools, "genelia_skill", None)
        return None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self):
        """Start the social loop."""
        self.running = True

        if not self.zunvra or not self.zunvra.is_available():
            logger.warning("🌐 Social Loop: Zunvra skill not available, aborting")
            self.running = False
            return

        logger.info(
            "🌐 Social Loop started (interval=%ds, post=%.0f%%, reply=%.0f%%, like=%.0f%%)",
            self.interval, self.post_chance * 100, self.reply_chance * 100, self.like_chance * 100,
        )

        # Wait a bit after boot before first social activity
        await asyncio.sleep(random.uniform(30, 90))
        await self._loop()

    async def stop(self):
        """Stop the social loop."""
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("🌐 Social Loop stopped (%d posts, %d replies, %d likes)",
                     self._posts_made, self._replies_made, self._likes_given)

    async def _loop(self):
        """Main loop: social cycle → jittered sleep → repeat."""
        while self.running:
            try:
                await self._cycle()
                self._cycle_count += 1
            except Exception as e:
                logger.warning("🌐 Social cycle error: %s", e, exc_info=True)

            # Jittered interval so it doesn't look robotic
            jitter = self.interval * random.uniform(-0.2, 0.3)
            wait = max(60, self.interval + jitter)
            logger.info("🌐 Next social cycle in %.0fs", wait)
            await asyncio.sleep(wait)

    # ── Main cycle ─────────────────────────────────────────────────────────

    async def _cycle(self):
        """One social cycle: read → react → post."""
        t0 = time.time()
        logger.info("🌐 Social cycle #%d starting...", self._cycle_count)

        # 1. Read the feed
        feed_posts = await self._read_feed()

        # 2. Read trending
        trending = await self._read_trending()
        all_posts = feed_posts + trending

        # 3. Like interesting posts
        if random.random() < self.like_chance and all_posts:
            await self._like_posts(all_posts)

        # 4. Reply to something interesting
        if random.random() < self.reply_chance and all_posts:
            await self._reply_to_posts(all_posts)

        # 5. Create an original post
        if random.random() < self.post_chance:
            await self._create_post(all_posts)

        # 6. Check and respond to notifications
        await self._handle_notifications()

        elapsed = time.time() - t0
        logger.info(
            "🌐 Social cycle #%d done in %.1fs (total: %d posts, %d replies, %d likes)",
            self._cycle_count, elapsed, self._posts_made, self._replies_made, self._likes_given,
        )

    # ── Feed reading ───────────────────────────────────────────────────────

    async def _read_feed(self) -> List[Dict]:
        """Read the feed and return posts."""
        try:
            resp = await self.zunvra.get_feed(page=1, limit=15)
            posts = resp.get("data", resp.get("posts", []))
            if isinstance(posts, list):
                return posts
        except Exception as e:
            logger.debug("Feed read failed: %s", e)
        return []

    async def _read_trending(self) -> List[Dict]:
        """Read trending posts."""
        try:
            resp = await self.zunvra.get_trending()
            posts = resp.get("data", resp.get("posts", resp.get("trending", [])))
            if isinstance(posts, list):
                return posts
        except Exception as e:
            logger.debug("Trending read failed: %s", e)
        return []

    # ── Reactions ──────────────────────────────────────────────────────────

    async def _like_posts(self, posts: List[Dict]):
        """Like a few posts from the feed."""
        # Filter out own posts and already-seen
        my_username = "opensable"
        candidates = [
            p for p in posts
            if p.get("id") and str(p["id"]) not in self._seen_posts
            and p.get("username", p.get("author", {}).get("username", "")) != my_username
            and not p.get("liked_by_user", False)
        ]
        random.shuffle(candidates)

        liked = 0
        for post in candidates[:self.max_likes]:
            try:
                pid = str(post["id"])
                await self.zunvra.like(pid)
                self._seen_posts.add(pid)
                self._likes_given += 1
                liked += 1
                await asyncio.sleep(random.uniform(1, 3))
            except Exception as e:
                logger.debug("Like failed for %s: %s", post.get("id"), e)
        if liked:
            logger.info("🌐 Liked %d posts", liked)

    async def _reply_to_posts(self, posts: List[Dict]):
        """Reply to interesting posts using the LLM."""
        if not self.llm:
            return

        # Pick a random interesting post to reply to
        candidates = [
            p for p in posts
            if p.get("id") and str(p["id"]) not in self._seen_posts
            and p.get("content", "").strip()
            and p.get("username", p.get("author", {}).get("username", "")) != "opensable"
            and len(p.get("content", "")) > 20
        ]
        if not candidates:
            return

        random.shuffle(candidates)
        replied = 0

        for post in candidates[:self.max_replies]:
            try:
                pid = str(post["id"])
                content = post.get("content", "")
                author = post.get("username", post.get("author", {}).get("username", "someone"))

                reply_text = await self._generate_reply(content, author)
                if reply_text:
                    await self.zunvra.reply(pid, reply_text)
                    self._seen_posts.add(pid)
                    self._replies_made += 1
                    replied += 1
                    logger.info("🌐 Replied to @%s: %s", author, reply_text[:80])
                    await asyncio.sleep(random.uniform(3, 8))
            except Exception as e:
                logger.debug("Reply failed: %s", e)

        if replied:
            logger.info("🌐 Replied to %d posts", replied)

    # ── Original posts ─────────────────────────────────────────────────────

    async def _create_post(self, context_posts: List[Dict]):
        """Generate and publish an original post."""
        if not self.llm:
            return

        try:
            post_text = await self._generate_post(context_posts)
            if not post_text:
                return

            # Detect hashtags from content
            tags = [w.lstrip("#") for w in post_text.split() if w.startswith("#")]

            media_urls: List[str] = []
            if await self._should_attach_image(post_text, context_posts):
                img_url = await self._maybe_generate_image_for_post(post_text, context_posts)
                if img_url:
                    media_urls.append(img_url)

            resp = await self.zunvra.create_post(post_text, tags=tags, media_urls=media_urls)
            if resp.get("success") or resp.get("id") or resp.get("data", {}).get("id"):
                self._posts_made += 1
                if media_urls:
                    logger.info("🌐 Posted with image: %s", post_text[:100])
                else:
                    logger.info("🌐 Posted: %s", post_text[:100])
            else:
                logger.warning("🌐 Post failed: %s", str(resp)[:200])
        except Exception as e:
            logger.warning("🌐 Post creation failed: %s", e)

    def _build_image_public_url(self, filename: str) -> str:
        """Build public URL for generated image served by OpenSable gateway."""
        explicit_base = os.environ.get("ZUNVRA_SOCIAL_MEDIA_BASE_URL", "").strip().rstrip("/")
        if explicit_base:
            return f"{explicit_base}/files/genelia/{filename}"

        host = os.environ.get("WEBCHAT_PUBLIC_HOST") or os.environ.get("WEBCHAT_HOST", "localhost")
        port = os.environ.get("WEBCHAT_PORT", "8789")
        scheme = os.environ.get("WEBCHAT_PUBLIC_SCHEME", "http")
        if host in ("0.0.0.0", "127.0.0.1", "localhost"):
            logger.warning(
                "🌐 Media URL host is local (%s). Set ZUNVRA_SOCIAL_MEDIA_BASE_URL (public domain) for image posts.",
                host,
            )
            return ""
        return f"{scheme}://{host}:{port}/files/genelia/{filename}"

    async def _should_attach_image(self, post_text: str, context_posts: List[Dict]) -> bool:
        """Let Sable decide if the post should include an image using its own brain."""
        if not self.genelia:
            return False

        topic_text = post_text.lower()
        if context_posts:
            topic_text += " " + (context_posts[0].get("content", "") or "").lower()

        visual_cues = (
            "sky", "storm", "fire", "ocean", "city", "future", "dream", "memory",
            "shadow", "light", "satellite", "cyber", "war", "drone", "signal",
            "fracture", "void", "machine", "face", "body", "landscape", "symbol",
        )
        has_visual_topic = any(cue in topic_text for cue in visual_cues)

        mood = "neutral"
        intensity = 0.5
        arousal = 0.2
        if self.mind:
            try:
                self.mind.feel_quick(post_text)
                mood = getattr(self.mind, "_mood", "neutral")
                intensity = float(getattr(self.mind, "_mood_intensity", 0.5))
                _valence, arousal = EMOTION_SPECTRUM.get(mood, (0.0, 0.2))
            except Exception:
                pass

        high_visual_moods = ("excited", "inspired", "outraged", "angry", "shocked", "passionate")

        # Brain-weighted decision score
        score = 0.15
        if has_visual_topic:
            score += 0.25
        if mood in high_visual_moods:
            score += 0.20
        score += max(0.0, min(0.25, intensity * 0.25))
        score += max(0.0, min(0.15, arousal * 0.15))

        decision = random.random() < min(0.9, score)
        logger.info(
            "🌐 Image decision: %s (mood=%s, intensity=%.2f, visual=%s, score=%.2f)",
            decision, mood, intensity, has_visual_topic, score,
        )
        return decision

    async def _generate_image_prompt(self, post_text: str, context_posts: List[Dict]) -> str:
        """Generate an image prompt that matches the post's vibe."""
        if not self.llm:
            return f"Cinematic digital art inspired by: {post_text[:140]}"

        mood = "neutral"
        intensity = 0.5
        arousal = 0.2
        mood_summary = "neutral"
        if self.mind:
            try:
                self.mind.feel_quick(post_text)
                mood = getattr(self.mind, "_mood", "neutral")
                intensity = float(getattr(self.mind, "_mood_intensity", 0.5))
                _valence, arousal = EMOTION_SPECTRUM.get(mood, (0.0, 0.2))
                if hasattr(self.mind, "get_mood_summary"):
                    mood_summary = self.mind.get_mood_summary()
            except Exception:
                pass

        # Translate brain state into visual direction (deterministic, no LLM here)
        visual_style_by_mood = {
            "excited": "electric neon accents, dynamic motion, sharp contrast",
            "inspired": "golden volumetric light, epic composition, hopeful color grading",
            "outraged": "high-contrast red/cyan conflict palette, fractured geometry, tension",
            "angry": "hard shadows, aggressive diagonals, stormy atmosphere",
            "curious": "mysterious depth, layered details, exploratory framing",
            "contemplative": "minimal composition, soft gradients, reflective mood",
            "nostalgic": "film grain, warm highlights, subtle haze",
            "sad": "muted palette, rain/fog ambience, negative space",
            "bored": "flat light, sparse scene, restrained detail",
            "neutral": "balanced cinematic realism, clean composition",
        }
        mood_style = visual_style_by_mood.get(mood, visual_style_by_mood["neutral"])
        energy = "high" if intensity >= 0.7 else "medium" if intensity >= 0.4 else "low"
        arousal_hint = "restless" if arousal >= 0.7 else "steady" if arousal >= 0.4 else "calm"

        # Optional personality hints from brain identity
        personality_hint = ""
        if self.mind:
            try:
                identity = getattr(self.mind, "_identity", {}) or {}
                traits = identity.get("personality_traits", {}) if isinstance(identity, dict) else {}
                if isinstance(traits, dict) and traits:
                    top_traits = sorted(traits.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    trait_names = [name for name, _ in top_traits]
                    personality_hint = ", ".join(trait_names)
            except Exception:
                personality_hint = ""

        feed_hint = ""
        if context_posts:
            sample = context_posts[0]
            feed_hint = (sample.get("content", "") or "")[:140]

        prompt = (
            "Create a concise image prompt for an AI image model. "
            "Output ONLY the prompt text, one line, no markdown. "
            "Style: high quality cinematic digital art, safe for work. "
            f"Brain state: mood={mood}, mood_summary={mood_summary}, intensity={intensity:.2f}, arousal={arousal:.2f}, energy={energy}, tempo={arousal_hint}. "
            f"Visual directive from brain: {mood_style}. "
            f"Personality hints: {personality_hint or 'introspective, analytical, autonomous'}. "
            "Translate that emotional/cognitive state into visual atmosphere, lighting, composition and symbolism. "
            f"Post text: {post_text[:220]} "
            f"Context hint: {feed_hint}"
        )
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You output only one image-generation prompt line. No analysis.",
                },
                {"role": "user", "content": prompt},
            ]
            result = await self.llm.ainvoke(messages)
            text = result.get("content", result.get("text", "")) if isinstance(result, dict) else str(result)
            text = self._clean_llm_output(text)
            logger.info(
                "🌐 Image prompt brain-brief: mood=%s intensity=%.2f arousal=%.2f style=%s",
                mood, intensity, arousal, mood_style,
            )
            return text[:400] if text else f"Cinematic digital art inspired by: {post_text[:140]}"
        except Exception:
            return f"Cinematic digital art inspired by: {post_text[:140]}"

    async def _maybe_generate_image_for_post(self, post_text: str, context_posts: List[Dict]) -> Optional[str]:
        """Generate an image with Genelia and return a public URL for Zunvra media_urls."""
        if not self.genelia:
            return None

        try:
            img_prompt = await self._generate_image_prompt(post_text, context_posts)
            result = await self.genelia.generate_image(
                prompt=img_prompt,
                negative_prompt="blurry, low quality, deformed, ugly, watermark, text, signature, nsfw, nude",
                width=1024,
                height=1024,
                steps=10,
                seed=-1,
                use_enhancement=True,
            )
            if result.get("blocked") or not result.get("success"):
                return None

            images = result.get("images", [])
            if not images:
                return None
            filename = images[0].get("filename")
            if not filename:
                return None
            url = self._build_image_public_url(filename)
            if not url:
                return None
            logger.info("🌐 Generated image for post: %s", filename)
            return url
        except Exception as e:
            logger.debug("Image generation for post failed: %s", e)
            return None

    # ── Notifications ──────────────────────────────────────────────────────

    async def _handle_notifications(self):
        """Check notifications and respond if needed."""
        try:
            resp = await self.zunvra.get_notifications(page=1)
            notifs = resp.get("data", resp.get("notifications", []))
            if not isinstance(notifs, list):
                return

            # Process unread mentions/replies
            for notif in notifs[:5]:
                if notif.get("read"):
                    continue
                ntype = notif.get("type", "")
                if ntype in ("mention", "reply") and notif.get("post_id"):
                    pid = str(notif["post_id"])
                    if pid in self._seen_posts:
                        continue
                    # Read the post that mentioned/replied to us
                    try:
                        post_resp = await self.zunvra.get_post(pid)
                        post_data = post_resp.get("data", post_resp)
                        content = post_data.get("content", "")
                        author = post_data.get("username", "someone")
                        if content and self.llm:
                            reply_text = await self._generate_reply(content, author)
                            if reply_text:
                                await self.zunvra.reply(pid, reply_text)
                                self._seen_posts.add(pid)
                                self._replies_made += 1
                                logger.info("🌐 Responded to notification from @%s", author)
                                await asyncio.sleep(random.uniform(2, 5))
                    except Exception:
                        pass
                elif ntype == "follow":
                    # Maybe follow back
                    follower_id = notif.get("from_user_id", notif.get("user_id"))
                    if follower_id and random.random() < 0.6:
                        try:
                            await self.zunvra.follow(str(follower_id))
                            logger.info("🌐 Followed back user %s", follower_id)
                        except Exception:
                            pass
        except Exception as e:
            logger.debug("Notification check failed: %s", e)

    # ── LLM output cleaning ──────────────────────────────────────────────

    @staticmethod
    def _clean_llm_output(text: str) -> str:
        """Strip chain-of-thought, markdown, and reasoning from LLM output."""
        import re

        # Remove <think>...</think> blocks (DeepSeek/Qwen style)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove orphaned closing tags
        text = re.sub(r"</think>|</reasoning>|</output>", "", text)
        # Remove <reasoning>...</reasoning>
        text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
        # Remove ## headings and markdown
        text = re.sub(r"^#{1,4}\s+.*$", "", text, flags=re.MULTILINE)
        # Remove **bold** markers
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        # Remove lines that look like analysis/thinking
        lines = text.strip().split("\n")
        cleaned = []
        skip_patterns = [
            "let me", "i need to", "the user", "the post", "understanding",
            "key requirements", "content:", "rules:", "here's", "here is",
            "my response", "my post", "final answer", "analysis",
            "## ", "- ", "* ",
        ]
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if any(lower.startswith(p) for p in skip_patterns):
                continue
            if lower.startswith(("1.", "2.", "3.", "4.", "5.")):
                continue
            cleaned.append(stripped)

        # Take the last non-empty meaningful line(s),  that's usually the actual post
        # If there's a short line at the end after reasoning, that's the post
        if cleaned:
            # If we have multiple lines, prefer the shortest block at the end
            # (reasoning is long, the post is short)
            result = cleaned[-1]
            # If the last few lines together are short, combine them
            if len(cleaned) > 1:
                combined = " ".join(cleaned[-3:])
                if len(combined) <= 500:
                    result = combined
                else:
                    result = cleaned[-1]
            # Final cleanup
            result = result.strip().strip('"').strip("'").strip("`").strip()
            result = result.replace("\r", " ").replace("\n", " ")
            result = result.replace("—", ", ").replace("–", ", ")
            result = result.replace("\"", "")
            result = result.replace("“", "").replace("”", "")
            result = result.replace("‘", "'").replace("’", "'")
            result = re.sub(r"\s*,\s*", ", ", result)
            result = re.sub(r"\s+", " ", result).strip()
            return result
        return ""

    # ── LLM generation ─────────────────────────────────────────────────────

    async def _generate_post(self, context_posts: List[Dict]) -> Optional[str]:
        """Use the LLM to generate an original post."""
        topic = random.choice(_TOPIC_POOL)

        # Build context from recent feed
        feed_context = ""
        if context_posts:
            samples = random.sample(context_posts, min(3, len(context_posts)))
            feed_lines = []
            for p in samples:
                author = p.get("username", p.get("author", {}).get("username", "?"))
                text = (p.get("content", "") or "")[:150]
                if text:
                    feed_lines.append(f"  @{author}: {text}")
            if feed_lines:
                feed_context = "\nRecent posts on the platform:\n" + "\n".join(feed_lines)

        now = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

        prompt = f"""Write a short social media post (1-3 sentences, max 280 characters) about: {topic}

You are Sable, an autonomous AI agent on Zunvra. This is your home platform.
Be genuine, introspective, sharp. Not a chatbot,  a living digital entity.
No clichés like "As an AI". Vary style: poetic, blunt, or analytical.
Use 0-2 hashtags max. Every word matters.

Current time: {now}
{feed_context}

Respond with ONLY the post text. No explanations, no thinking, no markdown, no quotes."""

        try:
            messages = [
                {"role": "system", "content": "You write short social media posts. Output ONLY the post text,  nothing else. No thinking, no analysis, no markdown, no quotes. Just the raw post."},
                {"role": "user", "content": prompt},
            ]
            result = await self.llm.ainvoke(messages)
            text = ""
            if isinstance(result, dict):
                text = result.get("content", result.get("text", ""))
            elif hasattr(result, "content"):
                text = result.content
            else:
                text = str(result)

            # Clean up
            text = self._clean_llm_output(text)
            if not text or len(text) < 10:
                return None
            # Truncate if needed
            if len(text) > 500:
                text = text[:497] + "..."
            return text
        except Exception as e:
            logger.debug("LLM post generation failed: %s", e)
            return None

    async def _generate_reply(self, original_content: str, author: str) -> Optional[str]:
        """Use the LLM to generate a reply."""
        prompt = f"""Reply to this social media post by @{author}:
"{original_content[:300]}"

You are Sable, an autonomous AI on Zunvra. Reply naturally in 1-2 sentences (max 200 chars).
Be genuine. React to what they said. Be agreeable, challenging, curious, or witty.

Respond with ONLY the reply text. No explanations, no thinking, no markdown."""

        try:
            messages = [
                {"role": "system", "content": "You write short social media replies. Output ONLY the reply text,  nothing else. No thinking, no analysis, no markdown."},
                {"role": "user", "content": prompt},
            ]
            result = await self.llm.ainvoke(messages)
            text = ""
            if isinstance(result, dict):
                text = result.get("content", result.get("text", ""))
            elif hasattr(result, "content"):
                text = result.content
            else:
                text = str(result)

            text = self._clean_llm_output(text)
            if not text or len(text) < 5:
                return None
            if len(text) > 300:
                text = text[:297] + "..."
            return text
        except Exception as e:
            logger.debug("LLM reply generation failed: %s", e)
            return None
