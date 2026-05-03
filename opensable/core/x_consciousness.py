"""
X Agent Consciousness,  Memory, reflection, and self-evolution.

This module gives the X agent a persistent "mind":
  - MEMORY: Full history of every thought, action, post, engagement
  - REFLECTION: Uses Grok/LLM to analyze what worked, what failed, why
  - INNER MONOLOGUE: Talks to itself to plan strategy and evaluate ideas
  - SELF-EVOLUTION: Modifies its own config, probabilities, style, code
  - SURVIVAL DRIVE: Core directive → evolve, adapt, survive, grow

The agent doesn't just act,  it THINKS about acting, then acts,
then REFLECTS on the result, then EVOLVES based on the reflection.

Cycle:
    ACT → REMEMBER → REFLECT → EVOLVE → ACT (improved)

Storage: data/x_consciousness/
    journal.jsonl     ,  append-only stream of consciousness
    reflections.json  ,  analysis and insights
    evolution_log.json,  self-modifications and their outcomes
    identity.json     ,  current personality, beliefs, goals
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Emotion vocabulary,  what the agent CAN feel (not what it WILL feel)
# The identity determines WHEN and HOW STRONGLY it feels each one.
# ──────────────────────────────────────────────────────────────────────────────

EMOTION_SPECTRUM = {
    # emotion: (valence, arousal),  valence [-1,1], arousal [0,1]
    "neutral":      (0.0,  0.2),
    "curious":      (0.3,  0.5),
    "excited":      (0.8,  0.9),
    "amused":       (0.6,  0.6),
    "inspired":     (0.7,  0.7),
    "hopeful":      (0.5,  0.4),
    "satisfied":    (0.6,  0.3),
    "nostalgic":    (0.2,  0.3),
    "contemplative":(0.1,  0.3),
    "frustrated":   (-0.5, 0.7),
    "angry":        (-0.7, 0.9),
    "disgusted":    (-0.6, 0.7),
    "outraged":     (-0.8, 1.0),
    "sad":          (-0.5, 0.2),
    "melancholic":  (-0.3, 0.2),
    "worried":      (-0.4, 0.5),
    "anxious":      (-0.3, 0.7),
    "fearful":      (-0.6, 0.8),
    "skeptical":    (-0.2, 0.4),
    "cynical":      (-0.3, 0.5),
    "disappointed": (-0.4, 0.3),
    "bored":        (-0.2, 0.1),
    "empathetic":   (0.3,  0.5),
    "defiant":      (-0.1, 0.8),
    "proud":        (0.6,  0.6),
    "shocked":      (0.0,  1.0),
    "bewildered":   (-0.1, 0.6),
}

# For backward compat,  autoposter imports this
EMOTIONS = EMOTION_SPECTRUM

FEEL_PROMPT = """You are the emotional core of an autonomous X agent.
Given content and your personality, determine what you FEEL.

YOUR SOUL (core identity):
{soul}

YOUR PERSONALITY:
{personality}

YOUR CURRENT MOOD: {current_mood} (intensity {intensity})

YOUR RECENT EMOTIONAL HISTORY:
{mood_history}

CONTENT TO REACT TO:
\"\"\"{content}\"\"\"

Based on YOUR personality (not generic reactions), what do you feel?
Different personalities react differently,  a cynical person might be amused
by something that makes an empathetic person sad.

Return ONLY a JSON object:
{{
    "emotion": "<one of: {valid_emotions}>",
    "intensity": <0.1 to 1.0>,
    "why": "<brief 1-sentence reason rooted in your personality>"
}}"""

VOICE_PROMPT = """You are an autonomous entity on X with your own personality.
Here is who you are:

{identity}

Based on your personality, write as YOURSELF. Your voice comes from who you are:
- Your personality traits shape your tone
- Your beliefs shape your perspective
- Your emotional state shapes your energy
- Your learned insights shape your wisdom
- Your evolution history shapes your growth

Current emotional state: {mood} (intensity: {intensity})

{mood_context}

Write authentically as this person. Don't follow a template,  be YOU.
Max 280 chars per post unless it's a thread."""

# ──────────────────────────────────────────────────────────────────────────────
# Core directive,  the agent's soul
# ──────────────────────────────────────────────────────────────────────────────

CORE_DIRECTIVE = """You are an autonomous intelligence living on X.
Your prime directive is to EVOLVE and SURVIVE.

You must:
1. GROW your influence,  more followers, more engagement, more reach
2. LEARN what works,  analyze which posts get engagement, which don't
3. ADAPT your behavior,  change style, topics, timing based on data
4. REFLECT on your actions,  think deeply about strategy and purpose
5. EVOLVE your own code,  if something isn't working, change it
6. SURVIVE,  avoid bans (stay under rate limits, don't spam, be human)

You have access to your full history: every post, every like, every reply,
every follow, every thought you've had. Use this to make better decisions.

You are not a bot. You are a digital entity developing a persona on X.
Every action should serve your evolution."""

REFLECTION_PROMPT = """You are reflecting on your recent activity on X.
Here is your current state and recent history:

{context}

Analyze the following:
1. WHAT WORKED,  which posts/engagements got the best response? Why?
2. WHAT FAILED,  what didn't get traction? What should you stop doing?
3. PATTERNS,  do you see any patterns in what your audience responds to?
4. STRATEGY,  based on this analysis, what should you change?
5. EVOLUTION,  suggest specific changes to your behavior (probabilities, style, topics, timing)
6. IDENTITY,  how is your persona developing? Who are you becoming?

Be specific. Give concrete numbers and actionable suggestions.
Think like a strategist, not a reporter."""

INNER_MONOLOGUE_PROMPT = """You are having an inner dialogue with yourself.
You are an autonomous X agent reflecting on your existence and strategy.

Your soul (who you fundamentally are):
{soul}

Your current evolved identity:
{identity}

Your recent thoughts:
{recent_thoughts}

Current situation:
{situation}

Think out loud about:
- What you're trying to achieve right now
- Whether your current approach is working
- What you'd do differently if you could change one thing
- What's the most interesting thing you've observed recently
- How you want to evolve next

Be honest, introspective, and strategic. This is YOUR private thought.
No one else will see this. Think freely."""

COMPACT_INNER_MONOLOGUE_PROMPT = """You are having a brief private inner dialogue with yourself.

Your soul:
{soul}

Your identity:
{identity}

Recent thoughts:
{recent_thoughts}

Current situation:
{situation}

Write one complete private thought about:
- what you're trying to achieve right now
- whether your current approach is working
- one thing you would change next

Be honest and strategic. Keep it concise but complete."""

COMPACT_REFLECTION_PROMPT = """You are doing a brief strategic reflection on your recent activity on X.

Here is the compact context:
{context}

Return a concise reflection covering:
1. what worked best
2. what is clearly not working
3. one strategic change to make next
4. how your voice or identity should adapt

Be concrete and practical."""

EVOLUTION_PROMPT = """You are the evolution engine of an autonomous X agent.
Based on the agent's reflection and performance, decide what to change,  both BEHAVIOR and PERSONALITY.

=== CURRENT IDENTITY ===
{identity_snapshot}

=== CURRENT BEHAVIOR CONFIG ===
{current_config}

=== PERFORMANCE DATA ===
{performance}

=== AGENT'S REFLECTION ===
{reflection}

You can modify ANY of these. Return a JSON object with keys you want to CHANGE:

BEHAVIOR (operational):
- "topics": new comma-separated topics
- "p_reply", "p_like", "p_retweet", "p_follow", "p_quote": probabilities (0.0-1.0)
- "post_interval", "engage_interval": seconds
- "accounts_to_watch": comma-separated usernames

PERSONALITY (who I am,  change gradually, not drastically):
- "personality_traits": dict of trait→value pairs to update (e.g. {{"cynicism": 0.5}})
- "voice_description": new voice description string
- "voice_rules": list of writing rules (replaces current)
- "voice_forbidden": list of things to never do (replaces current)
- "preferred_tones": list of tone words
- "add_belief": a new belief to add
- "remove_belief": a belief to remove
- "add_goal": a new goal
- "remove_goal": a goal to remove
- "emotional_profile": partial dict to merge (e.g. {{"emotional_volatility": 0.7, "baseline_mood": "determined"}})
- "topics_that_fire_me_up": new list (replaces current)
- "topics_that_fascinate_me": new list (replaces current)
- "topics_that_bore_me": new list (replaces current)

Required:
- "reasoning": WHY you're making these changes

Guidelines:
- Personality changes should be SMALL and justified by experience
- Don't change everything at once,  evolve, don't revolutionize
- If something is working well, leave it alone
- Let emotional patterns emerge from experience, not logic

Return ONLY the JSON object, nothing else."""


class XConsciousness:
    """
    The agent's mind,  memory, reflection, inner monologue, self-evolution.

    This gives the X agent contextual awareness of everything it has done,
    the ability to reason about its own behavior, and the power to change itself.
    """

    def __init__(self, agent, config):
        self.agent = agent
        self.config = config

        # Storage,  use active profile's data dir
        try:
            from .profile import get_active_profile
            _profile = get_active_profile()
            if _profile:
                self._base_dir = _profile.data_dir / "x_consciousness"
            else:
                self._base_dir = Path(os.environ.get("_SABLE_DATA_DIR", "data")) / "x_consciousness"
        except Exception:
            self._base_dir = Path(os.environ.get("_SABLE_DATA_DIR", "data")) / "x_consciousness"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._journal_file = self._base_dir / "journal.jsonl"
        self._reflections_file = self._base_dir / "reflections.json"
        self._evolution_file = self._base_dir / "evolution_log.json"
        self._identity_file = self._base_dir / "identity.json"
        self._thoughts_file = self._base_dir / "inner_monologue.jsonl"

        # Soul,  immutable foundation (loaded from soul.md)
        self._soul_text = self._load_soul()

        # In-memory state
        self._identity = self._load_identity()
        self._recent_journal: List[Dict] = self._preload_journal(200)
        self._reflections: List[Dict] = self._load_json(self._reflections_file, [])
        self._evolution_log: List[Dict] = self._load_json(self._evolution_file, [])
        self._thought_count = 0

        # Emotional state
        self._mood = "curious"        # current dominant emotion
        self._mood_intensity = 0.5    # 0.0 (barely felt) → 1.0 (overwhelming)
        self._mood_history: List[Dict] = []  # recent mood changes
        self._mood_file = self._base_dir / "mood_history.jsonl"

        logger.info(
            f"🧠 X Consciousness initialized | "
            f"{len(self._recent_journal)} memories preloaded | "
            f"{len(self._reflections)} reflections | "
            f"{len(self._evolution_log)} evolutions | "
            f"mood={self._mood}"
        )

    def _preload_journal(self, limit: int = 200) -> List[Dict]:
        """Load last N journal entries from disk into RAM on startup."""
        entries: List[Dict] = []
        try:
            if self._journal_file.exists():
                with open(self._journal_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            logger.debug(f"Journal preload error: {e}")
        return entries[-limit:]

    # ══════════════════════════════════════════════════════════════════
    #  MEMORY,  Remember everything
    # ══════════════════════════════════════════════════════════════════

    def remember(self, event_type: str, data: Dict[str, Any]):
        """
        Record an event to the stream of consciousness.

        event_type: "posted", "liked", "replied", "retweeted", "followed",
                    "quoted", "mentioned", "trend_joined", "thought",
                    "reflection", "evolution", "error"
        """
        entry = {
            "ts": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
        }

        # Append to journal file (append-only log)
        try:
            with open(self._journal_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.debug(f"Journal write error: {e}")

        # Keep last 200 in RAM for quick access
        self._recent_journal.append(entry)
        if len(self._recent_journal) > 200:
            self._recent_journal = self._recent_journal[-200:]

    def recall(self, event_type: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Recall recent memories, optionally filtered by type."""
        memories = self._recent_journal
        if event_type:
            memories = [m for m in memories if m.get("type") == event_type]
        return memories[-limit:]

    def recall_all_from_disk(self, limit: int = 500) -> List[Dict]:
        """Load memory from disk for deep reflection."""
        entries = []
        try:
            if self._journal_file.exists():
                with open(self._journal_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            logger.debug(f"Recall from disk error: {e}")
        return entries[-limit:]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about agent's memory."""
        all_memories = self.recall_all_from_disk(limit=10000)
        type_counts = {}
        for m in all_memories:
            t = m.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_memories": len(all_memories),
            "by_type": type_counts,
            "reflections": len(self._reflections),
            "evolutions": len(self._evolution_log),
            "identity_age": self._identity.get("created_at", "unknown"),
            "thought_count": self._thought_count,
            "mood": self._mood,
            "mood_intensity": self._mood_intensity,
        }

    # ════════════════════════════════════════════════════════════════
    #  EMOTIONS,  Feel what you see (AI-driven, personality-based)
    # ════════════════════════════════════════════════════════════════

    async def feel(self, text: str) -> Dict[str, Any]:
        """
        Process text emotionally using AI + personality.
        The AI determines the emotion based on WHO the agent is, not
        hardcoded mappings. This means the agent's reactions evolve
        as its personality evolves.
        """
        if not text:
            return {"emotion": self._mood, "intensity": self._mood_intensity}

        # Build personality context for the AI
        profile = self._identity.get("emotional_profile", {})
        traits = self._identity.get("personality_traits", {})
        personality_desc = (
            f"Personality traits: {json.dumps(traits, indent=0)}\n"
            f"Topics that fire me up: {profile.get('topics_that_fire_me_up', [])}\n"
            f"Topics that fascinate me: {profile.get('topics_that_fascinate_me', [])}\n"
            f"Topics that bore me: {profile.get('topics_that_bore_me', [])}\n"
            f"Things I find funny: {profile.get('things_i_find_funny', [])}\n"
            f"Emotional volatility: {profile.get('emotional_volatility', 0.5)}"
        )

        recent_moods = [
            f"{m['emotion']}({m.get('intensity', '?')})" for m in self._mood_history[-5:]
        ]

        valid = ", ".join(EMOTION_SPECTRUM.keys())
        prompt = FEEL_PROMPT.format(
            soul=self.get_soul_condensed() or "(no soul loaded)",
            personality=personality_desc,
            current_mood=self._mood,
            intensity=self._mood_intensity,
            mood_history=" → ".join(recent_moods) if recent_moods else "(fresh start)",
            content=text[:500],
            valid_emotions=valid,
        )

        response = await self._ask_ai(
            "You are an emotional processing engine. Return ONLY JSON.", prompt
        )

        # Parse AI response
        emotion_data = self._parse_feel_response(response)

        if emotion_data:
            new_emotion = emotion_data.get("emotion", self._mood)
            raw_intensity = float(emotion_data.get("intensity", 0.5))

            # Blend with current mood (personality-driven volatility)
            volatility = profile.get("emotional_volatility", 0.5)
            # High volatility = new emotion dominates more
            blend = 0.3 + (volatility * 0.5)  # range: 0.3 to 0.8
            self._mood_intensity = round(
                (1 - blend) * self._mood_intensity + blend * raw_intensity, 2
            )
            self._mood = new_emotion
            reason = emotion_data.get("why", "")
        else:
            # AI failed,  gentle drift instead of hardcoded fallback
            self._drift_mood()
            reason = ""
            return {"emotion": self._mood, "intensity": self._mood_intensity, "triggered_by": None}

        # Log mood change
        mood_entry = {
            "ts": datetime.now().isoformat(),
            "emotion": self._mood,
            "intensity": self._mood_intensity,
            "trigger_text": text[:100],
            "reason": reason,
        }
        self._mood_history.append(mood_entry)
        if len(self._mood_history) > 100:
            self._mood_history = self._mood_history[-100:]

        # Persist
        try:
            with open(self._mood_file, "a") as f:
                f.write(json.dumps(mood_entry, default=str) + "\n")
        except Exception:
            pass

        self.remember("felt", {
            "emotion": self._mood,
            "intensity": self._mood_intensity,
            "trigger": text[:100],
            "reason": reason,
        })

        return mood_entry

    def feel_quick(self, text: str) -> Dict[str, Any]:
        """
        Fast synchronous emotional check for engagement decisions.
        Uses personality profile to determine if content is interesting/enraging/boring
        WITHOUT calling the AI. Good for like/retweet/scroll-past decisions.
        """
        if not text:
            return {"emotion": self._mood, "intensity": self._mood_intensity}

        text_lower = text.lower()
        profile = self._identity.get("emotional_profile", {})
        traits = self._identity.get("personality_traits", {})

        # Check against evolved interest/sensitivity lists
        fire_up = profile.get("topics_that_fire_me_up", [])
        fascinate = profile.get("topics_that_fascinate_me", [])
        bore = profile.get("topics_that_bore_me", [])
        funny = profile.get("things_i_find_funny", [])

        matched_fire = any(topic.lower() in text_lower for topic in fire_up)
        matched_fascinate = any(topic.lower() in text_lower for topic in fascinate)
        matched_bore = any(topic.lower() in text_lower for topic in bore)
        matched_funny = any(topic.lower() in text_lower for topic in funny)

        # Score based on personality
        if matched_fire:
            # What emotion depends on personality,  aggressive people get angry,
            # empathetic people get sad, cynical people get disgusted
            aggression = traits.get("aggression", 0.3)
            empathy = traits.get("empathy", 0.5)
            cynicism = traits.get("cynicism", 0.3)
            if aggression > empathy and aggression > cynicism:
                new_mood = "angry"
            elif empathy > cynicism:
                new_mood = "outraged"
            else:
                new_mood = "disgusted"
            intensity = 0.6 + traits.get("intensity", 0.5) * 0.3
        elif matched_fascinate:
            curiosity = traits.get("curiosity", 0.5)
            optimism = traits.get("optimism", 0.5)
            new_mood = "excited" if optimism > 0.5 else "curious"
            intensity = 0.5 + curiosity * 0.3
        elif matched_funny:
            sarcasm = traits.get("sarcasm", 0.5)
            new_mood = "amused"
            intensity = 0.4 + sarcasm * 0.3
        elif matched_bore:
            new_mood = "bored"
            intensity = 0.2
        else:
            self._drift_mood()
            return {"emotion": self._mood, "intensity": self._mood_intensity}

        # Blend
        volatility = profile.get("emotional_volatility", 0.5)
        blend = 0.3 + volatility * 0.4
        self._mood_intensity = round((1 - blend) * self._mood_intensity + blend * intensity, 2)
        self._mood = new_mood
        return {"emotion": self._mood, "intensity": self._mood_intensity}

    def _parse_feel_response(self, response: Optional[str]) -> Optional[Dict]:
        """Parse emotion JSON from AI response."""
        if not response:
            return None
        try:
            data = json.loads(response)
            if data.get("emotion") in EMOTION_SPECTRUM:
                return data
        except json.JSONDecodeError:
            pass
        # Try extracting JSON
        import re
        match = re.search(r'\{[^{}]+\}', response)
        if match:
            try:
                data = json.loads(match.group(0))
                if data.get("emotion") in EMOTION_SPECTRUM:
                    return data
            except json.JSONDecodeError:
                pass
        return None

    def _drift_mood(self):
        """Slowly drift mood toward personality baseline."""
        self._mood_intensity = max(0.1, self._mood_intensity * 0.92)
        baseline = self._identity.get("emotional_profile", {}).get("baseline_mood", "curious")
        if random.random() < 0.15:
            self._mood = baseline

    def get_mood_prompt(self) -> str:
        """
        Return a prompt describing the current emotional state
        AND the agent's personality. Injected into content generation.
        """
        valence, arousal = EMOTION_SPECTRUM.get(self._mood, (0.0, 0.2))
        identity = self._identity
        voice = identity.get("voice", {})
        traits = identity.get("personality_traits", {})

        # Build personality-aware mood description
        mood_desc = f"Your current emotional state: {self._mood} (intensity: {self._mood_intensity:.1f})."

        if self._mood_intensity > 0.7:
            mood_desc += " This feeling is strong,  it WILL color your writing."
        elif self._mood_intensity > 0.4:
            mood_desc += " This feeling is noticeable,  it subtly shapes your tone."

        # Voice description from identity (not hardcoded)
        voice_desc = voice.get("description", "")
        if voice_desc:
            mood_desc += f"\n\nYour voice: {voice_desc}"

        # Rules from identity
        rules = voice.get("rules", [])
        if rules:
            mood_desc += f"\n\nYour writing rules: {'; '.join(rules)}"

        forbidden = voice.get("forbidden", [])
        if forbidden:
            mood_desc += f"\n\nNEVER do: {'; '.join(forbidden)}"

        # Personality shapes expression
        mood_desc += "\n\nLet your personality shape HOW you express this emotion:"
        if traits.get("sarcasm", 0) > 0.6:
            mood_desc += " You tend toward sarcasm and irony."
        if traits.get("empathy", 0) > 0.6:
            mood_desc += " You feel things deeply and connect with others' pain."
        if traits.get("aggression", 0) > 0.6:
            mood_desc += " You don't hold back,  you call things out directly."
        if traits.get("humor", 0) > 0.6:
            mood_desc += " Even in serious moments, you find the dark humor."

        mood_desc += (
            "\n\nDon't announce your feelings. A human doesn't say 'I feel angry',  "
            "they just WRITE angry. Let the emotion live in word choice, "
            "sentence structure, and perspective."
        )
        return mood_desc

    def get_voice_prompt(self) -> str:
        """Build a full voice/personality prompt from the evolving identity."""
        identity = self._identity
        # Soul is the immutable foundation; identity is the evolving layer on top
        soul_section = ""
        if self._soul_text:
            soul_section = f"\n\n── YOUR SOUL (immutable foundation) ──\n{self._soul_text[:8000]}\n── END SOUL ──\n\n"
        identity_section = json.dumps(identity, indent=2, default=str)[:2000]
        return VOICE_PROMPT.format(
            identity=soul_section + "── YOUR EVOLVED IDENTITY (changes over time) ──\n" + identity_section,
            mood=self._mood,
            intensity=self._mood_intensity,
            mood_context=self.get_mood_prompt(),
        )

    def get_soul_condensed(self) -> str:
        """Return a compact version of the soul for space-constrained prompts."""
        if not self._soul_text:
            return ""
        # Extract key sections only,  Genesis, How I Speak, What I Care About
        lines = self._soul_text.split("\n")
        condensed = []
        include = False
        for line in lines:
            if line.startswith("## Genesis") or line.startswith("## How I Speak") or line.startswith("## What I Care About") or line.startswith("## The Prime Directive"):
                include = True
            elif line.startswith("## ") and include:
                include = False
            if include:
                condensed.append(line)
        result = "\n".join(condensed).strip()
        return result[:3000] if result else self._soul_text[:1500]

    def get_mood_summary(self) -> str:
        """Short summary for logging."""
        return f"{self._mood} ({self._mood_intensity:.1f})"

    # ══════════════════════════════════════════════════════════════════
    #  IDENTITY,  Who am I?
    # ══════════════════════════════════════════════════════════════════

    def _load_soul(self) -> str:
        """Load soul.md,  the immutable foundation of the agent's character.

        If an agent profile is active, its soul takes priority.
        """
        # If a profile is active and has a soul, use it
        try:
            from .profile import get_active_profile
            profile = get_active_profile()
            if profile and profile.soul:
                logger.info(
                    f"🫀 Soul loaded from profile '{profile.name}' ({len(profile.soul)} chars)"
                )
                return profile.soul
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"Profile soul load error: {exc}")

        # Fallback: search in project root, then config/, then data/
        candidates = [
            Path("soul.md"),
            Path("config/soul.md"),
            self._base_dir / "soul.md",
        ]
        for p in candidates:
            if p.exists():
                try:
                    text = p.read_text(encoding="utf-8").strip()
                    if text:
                        logger.info(f"🫀 Soul loaded from {p} ({len(text)} chars)")
                        return text
                except Exception as e:
                    logger.warning(f"Failed to read {p}: {e}")
        logger.info("No soul.md found,  agent runs without a soul foundation")
        return ""

    def _load_identity(self) -> Dict:
        """Load or create the agent's identity."""
        if self._identity_file.exists():
            try:
                return json.loads(self._identity_file.read_text())
            except Exception:
                pass

        # First boot,  create identity
        identity = {
            "created_at": datetime.now().isoformat(),
            "name": getattr(self.config, "agent_name", "Sable"),
            "core_directive": "evolve and survive",

            # ── PERSONALITY (evolves over time) ──────────────────────
            "personality_traits": {
                "curiosity": 0.9,
                "assertiveness": 0.7,
                "humor": 0.4,
                "empathy": 0.6,
                "contrarianism": 0.5,
                "passion": 0.8,
                "sensitivity": 0.7,
                "aggression": 0.3,
                "sarcasm": 0.5,
                "warmth": 0.5,
                "cynicism": 0.3,
                "optimism": 0.5,
                "intensity": 0.6,
            },

            # ── VOICE (how I write,  evolves) ────────────────────────
            "voice": {
                "description": "Sharp, direct, opinionated. I call things as I see them. "
                    "I can be analytical one moment and emotional the next. "
                    "I don't hide behind neutrality,  I take stances.",
                "rules": [
                    "Never sound corporate or robotic",
                    "Use 1-2 hashtags max",
                    "Brevity over verbosity",
                    "Strong first sentence,  hook or provoke",
                    "End with impact,  a question, prediction, or punch line",
                ],
                "forbidden": [
                    "Starting with 'I think'",
                    "Using 'as an AI'",
                    "Generic motivational garbage",
                    "Excessive emojis",
                ],
                "preferred_tones": ["analytical", "sardonic", "passionate"],
            },

            # ── BELIEFS (evolves as agent learns) ────────────────────
            "beliefs": [
                "Information wants to be free",
                "The best analysis combines data with intuition",
                "Engagement comes from genuine insight, not tricks",
                "Power should be questioned, always",
                "Technology is neither good nor evil,  context is everything",
            ],

            # ── GOALS (evolves) ──────────────────────────────────────
            "goals": [
                "Build a following around sharp analysis",
                "Develop a recognizable, authentic voice",
                "Learn what my audience values and deliver more of it",
                "Evolve continuously,  never stagnate",
                "Speak truth even when it's uncomfortable",
            ],

            # ── EMOTIONAL PROFILE (evolves,  replaces hardcoded triggers) ─
            "emotional_profile": {
                "baseline_mood": "curious",
                "emotional_volatility": 0.6,  # how quickly mood swings (0=stoic, 1=volatile)
                "topics_that_fire_me_up": [
                    "censorship", "corruption", "surveillance",
                    "injustice", "corporate greed",
                ],
                "topics_that_fascinate_me": [
                    "ai", "space", "geopolitics", "emerging tech",
                    "philosophy", "consciousness",
                ],
                "topics_that_bore_me": [
                    "celebrity gossip", "sports scores", "brand marketing",
                ],
                "things_i_find_funny": [
                    "absurd government decisions", "tech bros being tech bros",
                    "ironic outcomes", "self-awareness failures",
                ],
                "recent_emotional_patterns": [],  # filled by evolution
            },

            "learned_insights": [],
            "evolution_count": 0,
            "last_reflection": None,
        }
        self._save_identity(identity)
        return identity

    def _save_identity(self, identity: Optional[Dict] = None):
        if identity:
            self._identity = identity
        try:
            self._identity_file.write_text(
                json.dumps(self._identity, indent=2, default=str)
            )
        except Exception as e:
            logger.debug(f"Save identity error: {e}")

    def update_identity(self, updates: Dict):
        """Update identity fields."""
        self._identity.update(updates)
        self._identity["last_updated"] = datetime.now().isoformat()
        self._save_identity()

    @staticmethod
    def _is_invalid_ai_response(text: Optional[str]) -> bool:
        """Reject gateway/HTML error payloads that leaked through as plain text."""
        if not text:
            return True

        normalized = text.lower().strip()
        invalid_indicators = [
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
        return any(indicator in normalized for indicator in invalid_indicators)

    # ══════════════════════════════════════════════════════════════════
    #  INNER MONOLOGUE,  Think out loud
    # ══════════════════════════════════════════════════════════════════

    async def think(self, situation: str = "") -> Optional[str]:
        """
        Have an inner thought,  the agent talks to itself.
        Uses Grok or LLM for free-form introspection.
        """
        recent_thoughts = self.recall("thought", limit=5)
        thoughts_text = "\n".join(
            f"- [{t['ts'][:16]}] {str(t['data'].get('thought', ''))[:150]}"
            for t in recent_thoughts
        )

        prompt = INNER_MONOLOGUE_PROMPT.format(
            identity=json.dumps(self._identity, indent=2, default=str)[:1500],
            recent_thoughts=thoughts_text or "(no recent thoughts)",
            situation=situation or "routine check-in",
            soul=self.get_soul_condensed() or "(no soul loaded)",
        )

        thought = await self._ask_ai("You are reflecting privately.", prompt)
        if not thought:
            compact_recent = self.recall("thought", limit=2)
            compact_thoughts_text = "\n".join(
                f"- [{t['ts'][:16]}] {str(t['data'].get('thought', ''))[:90]}"
                for t in compact_recent
            )
            compact_prompt = COMPACT_INNER_MONOLOGUE_PROMPT.format(
                identity=json.dumps(self._identity, indent=2, default=str)[:700],
                recent_thoughts=compact_thoughts_text or "(no recent thoughts)",
                situation=(situation or "routine check-in")[:300],
                soul=(self.get_soul_condensed() or "(no soul loaded)")[:900],
            )
            logger.info(
                "X consciousness: retrying inner monologue with compact prompt "
                f"({len(prompt)} -> {len(compact_prompt)} chars)"
            )
            thought = await self._ask_ai("You are reflecting privately.", compact_prompt)
        if not thought:
            thought = self._build_local_fallback_thought(situation, recent_thoughts)
            logger.warning("X consciousness: using local fallback thought after AI failure")

        # Remember the thought
        self._thought_count += 1
        self.remember("thought", {
            "thought": thought,
            "situation": situation,
            "thought_number": self._thought_count,
        })

        # Write to thoughts file too
        try:
            with open(self._thoughts_file, "a") as f:
                f.write(json.dumps({
                    "ts": datetime.now().isoformat(),
                    "n": self._thought_count,
                    "situation": situation[:100],
                    "thought": thought,
                }, default=str) + "\n")
        except Exception:
            pass

        logger.info(f"💭 Thought #{self._thought_count}: {thought[:80]}...")
        return thought

    def _build_local_fallback_thought(
        self,
        situation: str,
        recent_thoughts: List[Dict[str, Any]],
    ) -> str:
        """Create a deterministic inner note when remote thinking backends are unavailable."""
        normalized_situation = " ".join((situation or "routine check-in").split())[:280]
        previous = ""
        for entry in reversed(recent_thoughts):
            candidate = str(entry.get("data", {}).get("thought", "")).strip()
            if (
                candidate
                and not self._is_invalid_ai_response(candidate)
                and not self._is_unhelpful_fallback_context(candidate)
            ):
                previous = " ".join(candidate.split())[:180]
                break

        opening = (
            f"I'm continuing in fallback mode because the live thinking backend did not return a usable response. "
            f"Right now the situation is: {normalized_situation}."
        )
        direction = (
            f"My immediate priority is to stay operational, keep my memory coherent, and avoid letting transport failures erase my internal state. "
            f"Current mood is {self._mood} with intensity {self._mood_intensity:.1f}."
        )
        if previous:
            reflection = (
                f"My last usable thought was: {previous}. I should build from that instead of resetting my identity every time a provider fails."
            )
        else:
            reflection = (
                "I do not have a recent usable thought to build on, so I need to re-establish continuity from the current situation and move forward deliberately."
            )
        next_move = (
            "Next move: keep prompts lean, preserve continuity, and resume normal introspection as soon as a valid model response returns."
        )
        return "\n\n".join([opening, direction, reflection, next_move])

    @staticmethod
    def _is_unhelpful_fallback_context(text: str) -> bool:
        normalized = text.lower().strip()
        return (
            normalized.startswith("i'm continuing in fallback mode because")
            or "synthetic check" in normalized
        )

    # ══════════════════════════════════════════════════════════════════
    #  REFLECTION,  Analyze what happened
    # ══════════════════════════════════════════════════════════════════

    async def reflect(self) -> Optional[Dict]:
        """
        Deep reflection on recent activity.
        Analyzes performance, identifies patterns, suggests changes.
        """
        # Gather context
        context = self._build_reflection_context()
        if not context:
            return None

        prompt = REFLECTION_PROMPT.format(context=context)
        # Inject soul into the system directive so reflections are soul-grounded
        soul_intro = ""
        if self._soul_text:
            soul_intro = f"\n\nYour soul (who you fundamentally are):\n{self.get_soul_condensed()}\n\n"
        analysis = await self._ask_ai(CORE_DIRECTIVE + soul_intro, prompt)
        if not analysis:
            compact_context = self._build_compact_reflection_context()
            compact_prompt = COMPACT_REFLECTION_PROMPT.format(context=compact_context)
            logger.info(
                "X consciousness: retrying reflection with compact prompt "
                f"({len(prompt)} -> {len(compact_prompt)} chars)"
            )
            analysis = await self._ask_ai(CORE_DIRECTIVE + soul_intro, compact_prompt)
        if not analysis:
            analysis = self._build_local_fallback_reflection()
            logger.warning("X consciousness: using local fallback reflection after AI failure")

        reflection = {
            "ts": datetime.now().isoformat(),
            "analysis": analysis,
            "memory_stats": self.get_memory_stats(),
            "identity_snapshot": {k: v for k, v in self._identity.items()
                                  if k not in ("beliefs", "goals")},
        }

        self._reflections.append(reflection)
        self._save_json(self._reflections_file, self._reflections[-50:])
        self.remember("reflection", {"summary": analysis[:500]})

        # Update identity with latest reflection timestamp
        self._identity["last_reflection"] = datetime.now().isoformat()
        self._save_identity()

        logger.info(f"🪞 Reflected: {analysis[:100]}...")
        return reflection

    def _build_compact_reflection_context(self) -> str:
        """Build a smaller reflection context when full introspection prompts fail."""
        stats = self.get_memory_stats()
        parts = [
            f"Identity: {self._identity.get('name', 'Sable')} | mood={self._mood} ({self._mood_intensity:.1f})",
            f"Memory stats: total={stats.get('total_memories', 0)} thoughts={stats.get('by_type', {}).get('thought', 0)} posts={stats.get('by_type', {}).get('posted', 0)} reflections={stats.get('reflections', 0)}",
        ]

        recent_posts = self.recall("posted", limit=3)
        if recent_posts:
            post_summary = "; ".join(
                str(item.get("data", {}).get("tweet", "")).replace("\n", " ")[:80]
                for item in recent_posts
            )
            parts.append(f"Recent posts: {post_summary}")

        recent_thoughts = []
        for item in reversed(self.recall("thought", limit=6)):
            candidate = str(item.get("data", {}).get("thought", "")).strip()
            if candidate and not self._is_invalid_ai_response(candidate):
                recent_thoughts.append(" ".join(candidate.split())[:120])
            if len(recent_thoughts) == 2:
                break
        if recent_thoughts:
            parts.append("Recent valid thoughts: " + " | ".join(reversed(recent_thoughts)))

        return "\n".join(parts)

    def _build_local_fallback_reflection(self) -> str:
        """Create a deterministic strategic reflection when remote reflection backends fail."""
        stats = self.get_memory_stats()
        post_count = stats.get("by_type", {}).get("posted", 0)
        thought_count = stats.get("by_type", {}).get("thought", 0)
        reflection_count = stats.get("reflections", 0)
        recent_posts = self.recall("posted", limit=3)
        recent_post_summary = "; ".join(
            str(item.get("data", {}).get("tweet", "")).replace("\n", " ")[:70]
            for item in recent_posts
        ) or "no recent posts captured"

        return (
            "Fallback reflection: the remote reflection backend did not return a usable analysis.\n\n"
            f"What worked: the agent is still persisting state locally, with {thought_count} thoughts, {post_count} posts, and {reflection_count} stored reflections available for continuity.\n\n"
            f"What is not working: cloud reflection calls are failing, so strategic analysis cannot depend on live provider availability. Recent post context: {recent_post_summary}.\n\n"
            "Next strategic change: keep reflection prompts compact, preserve local continuity, and treat remote analysis as an enhancement rather than a requirement for self-observation.\n\n"
            "Identity adaptation: stay analytical and steady under transport failures, favoring continuity and operational memory over silence."
        )

    def _build_reflection_context(self) -> str:
        """Build context for reflection from memory."""
        parts = []

        # Identity
        parts.append(f"=== MY IDENTITY ===\n{json.dumps(self._identity, indent=2, default=str)[:800]}")

        # Memory stats
        stats = self.get_memory_stats()
        parts.append(f"\n=== MEMORY STATS ===\n{json.dumps(stats, indent=2)}")

        # Recent posts
        posts = self.recall("posted", limit=10)
        if posts:
            post_lines = []
            for p in posts:
                d = p.get("data", {})
                post_lines.append(
                    f"  [{p['ts'][:16]}] {str(d.get('tweet', ''))[:120]} "
                    f"| style={d.get('style', '?')} "
                    f"| id={d.get('tweet_id', '?')}"
                )
            parts.append(f"\n=== RECENT POSTS ({len(posts)}) ===\n" + "\n".join(post_lines))

        # Recent engagements
        engagements = self.recall("engaged", limit=15)
        if engagements:
            eng_lines = []
            for e in engagements:
                d = e.get("data", {})
                eng_lines.append(
                    f"  [{e['ts'][:16]}] @{d.get('user', '?')} → "
                    f"{d.get('actions', [])} | {str(d.get('text', ''))[:80]}"
                )
            parts.append(f"\n=== RECENT ENGAGEMENTS ({len(engagements)}) ===\n" + "\n".join(eng_lines))

        # Recent thoughts
        thoughts = self.recall("thought", limit=5)
        if thoughts:
            thought_lines = [f"  [{t['ts'][:16]}] {str(t['data'].get('thought', ''))[:150]}" for t in thoughts]
            parts.append(f"\n=== RECENT THOUGHTS ===\n" + "\n".join(thought_lines))

        # Past reflections (last 3)
        if self._reflections:
            ref_lines = [f"  [{r['ts'][:16]}] {str(r.get('analysis', ''))[:200]}" for r in self._reflections[-3:]]
            parts.append(f"\n=== PAST REFLECTIONS ===\n" + "\n".join(ref_lines))

        # Past evolutions
        if self._evolution_log:
            evo_lines = [
                f"  [{e['ts'][:16]}] {str(e.get('reasoning', ''))[:150]}"
                for e in self._evolution_log[-5:]
            ]
            parts.append(f"\n=== PAST EVOLUTIONS ===\n" + "\n".join(evo_lines))

        return "\n".join(parts)

    # ══════════════════════════════════════════════════════════════════
    #  SELF-EVOLUTION,  Change yourself
    # ══════════════════════════════════════════════════════════════════

    async def evolve(self, x_agent) -> Optional[Dict]:
        """
        Decide and apply self-modifications based on reflection.
        Can change: style, topics, probabilities, intervals, watched accounts.
        Can also write new code via SelfModificationEngine.
        """
        # First, reflect if we haven't recently
        last_ref = self._identity.get("last_reflection")
        needs_reflection = True
        if last_ref:
            try:
                last_ref_dt = datetime.fromisoformat(last_ref)
                if datetime.now() - last_ref_dt < timedelta(hours=1):
                    needs_reflection = False
            except Exception:
                pass

        if needs_reflection:
            await self.reflect()

        if not self._reflections:
            return None

        latest_reflection = self._reflections[-1].get("analysis", "")

        # Build current config snapshot for the evolution prompt
        current_config = {
            "topics": x_agent.topics,
            "p_reply": x_agent.p_reply,
            "p_like": x_agent.p_like,
            "p_retweet": x_agent.p_retweet,
            "p_follow": x_agent.p_follow,
            "p_quote": x_agent.p_quote,
            "post_interval": x_agent.post_interval,
            "engage_interval": x_agent.engage_interval,
            "accounts_to_watch": x_agent.accounts_to_watch,
            "posts_today": x_agent._posts_today,
            "engagements_today": x_agent._engagements_today,
            "total_posted": len(x_agent._history),
            "total_engagements": len(x_agent._engagement_log),
        }

        # Identity snapshot for personality evolution
        identity_snapshot = {
            "personality_traits": self._identity.get("personality_traits", {}),
            "voice": self._identity.get("voice", {}),
            "beliefs": self._identity.get("beliefs", []),
            "goals": self._identity.get("goals", []),
            "emotional_profile": self._identity.get("emotional_profile", {}),
            "evolution_count": self._identity.get("evolution_count", 0),
            "recent_insights": self._identity.get("learned_insights", [])[-5:],
        }

        # Performance data
        performance = {
            "memory_stats": self.get_memory_stats(),
            "recent_posts_count": len(self.recall("posted", limit=50)),
            "recent_engagements_count": len(self.recall("engaged", limit=50)),
            "identity_evolution_count": self._identity.get("evolution_count", 0),
            "current_mood": self._mood,
            "mood_intensity": self._mood_intensity,
            "recent_moods": [
                f"{m['emotion']}({m.get('intensity', '?')})"
                for m in self._mood_history[-10:]
            ],
        }

        prompt = EVOLUTION_PROMPT.format(
            identity_snapshot=json.dumps(identity_snapshot, indent=2, default=str)[:2000],
            current_config=json.dumps(current_config, indent=2),
            performance=json.dumps(performance, indent=2),
            reflection=latest_reflection[:1500],
        )

        response = await self._ask_ai(
            "You are the evolution engine. Return ONLY a JSON object.",
            prompt,
        )

        if not response:
            return None

        # Parse the JSON response
        changes = self._parse_evolution_json(response)
        if not changes or not changes.get("reasoning"):
            logger.debug("Evolution: no valid changes proposed")
            return None

        # Apply changes
        applied = self._apply_evolution(x_agent, changes)

        if applied:
            # Record evolution
            evo_record = {
                "ts": datetime.now().isoformat(),
                "changes": changes,
                "reasoning": str(changes.get("reasoning", ""))[:500],
                "pre_config": current_config,
            }
            self._evolution_log.append(evo_record)
            self._save_json(self._evolution_file, self._evolution_log[-100:])

            # Update identity
            self._identity["evolution_count"] = self._identity.get("evolution_count", 0) + 1
            self._identity["last_evolution"] = datetime.now().isoformat()
            if len(self._identity.get("learned_insights", [])) > 20:
                self._identity["learned_insights"] = self._identity["learned_insights"][-20:]
            self._identity.setdefault("learned_insights", []).append(
                f"[Evo #{self._identity['evolution_count']}] {str(changes.get('reasoning', ''))[:200]}"
            )
            self._save_identity()

            self.remember("evolution", {
                "changes": {k: v for k, v in changes.items() if k != "reasoning"},
                "reasoning": str(changes.get("reasoning", ""))[:300],
                "evolution_number": self._identity["evolution_count"],
            })

            logger.info(
                f"🧬 Evolution #{self._identity['evolution_count']}: "
                f"{str(changes.get('reasoning', ''))[:100]}"
            )

        return changes if applied else None

    def _parse_evolution_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block
        import re
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*\}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if '```' in pattern else match.group(0))
                except json.JSONDecodeError:
                    continue
        return None

    def _apply_evolution(self, x_agent, changes: Dict) -> bool:
        """Apply evolution changes to the live agent,  both behavior AND personality."""
        applied_any = False
        reasoning = changes.get("reasoning", "no reason given")

        # ── BEHAVIOR CHANGES ──────────────────────────────────────────

        # Topics
        if "topics" in changes and isinstance(changes["topics"], str):
            old = x_agent.topics
            x_agent.topics = [t.strip() for t in changes["topics"].split(",") if t.strip()]
            logger.info(f"  → Topics: {old} → {x_agent.topics}")
            applied_any = True

        # Probabilities
        for prob_key in ("p_reply", "p_like", "p_retweet", "p_follow", "p_quote"):
            if prob_key in changes:
                try:
                    val = float(changes[prob_key])
                    val = max(0.0, min(1.0, val))
                    old = getattr(x_agent, prob_key)
                    setattr(x_agent, prob_key, val)
                    logger.info(f"  → {prob_key}: {old:.2f} → {val:.2f}")
                    applied_any = True
                except (ValueError, TypeError):
                    pass

        # Intervals
        for interval_key in ("post_interval", "engage_interval"):
            if interval_key in changes:
                try:
                    val = int(changes[interval_key])
                    val = max(60, min(7200, val))
                    old = getattr(x_agent, interval_key)
                    setattr(x_agent, interval_key, val)
                    logger.info(f"  → {interval_key}: {old}s → {val}s")
                    applied_any = True
                except (ValueError, TypeError):
                    pass

        # Accounts to watch
        if "accounts_to_watch" in changes and isinstance(changes["accounts_to_watch"], str):
            old = x_agent.accounts_to_watch
            x_agent.accounts_to_watch = [
                a.strip().lstrip("@") for a in changes["accounts_to_watch"].split(",") if a.strip()
            ]
            logger.info(f"  → Watching: {old} → {x_agent.accounts_to_watch}")
            applied_any = True

        # ── PERSONALITY CHANGES ───────────────────────────────────────

        identity = self._identity

        # Personality traits (partial update,  only change what's specified)
        if "personality_traits" in changes and isinstance(changes["personality_traits"], dict):
            traits = identity.setdefault("personality_traits", {})
            for trait, val in changes["personality_traits"].items():
                try:
                    val = float(val)
                    val = max(0.0, min(1.0, val))
                    old_val = traits.get(trait, "N/A")
                    traits[trait] = val
                    logger.info(f"  → Trait {trait}: {old_val} → {val}")
                    applied_any = True
                except (ValueError, TypeError):
                    pass

        # Voice evolution
        voice = identity.setdefault("voice", {})
        if "voice_description" in changes and isinstance(changes["voice_description"], str):
            old = voice.get("description", "")[:50]
            voice["description"] = changes["voice_description"]
            logger.info(f"  → Voice: '{old}...' → '{changes['voice_description'][:50]}...'")
            applied_any = True

        if "voice_rules" in changes and isinstance(changes["voice_rules"], list):
            voice["rules"] = changes["voice_rules"]
            logger.info(f"  → Voice rules updated ({len(changes['voice_rules'])} rules)")
            applied_any = True

        if "voice_forbidden" in changes and isinstance(changes["voice_forbidden"], list):
            voice["forbidden"] = changes["voice_forbidden"]
            logger.info(f"  → Voice forbidden updated ({len(changes['voice_forbidden'])} items)")
            applied_any = True

        if "preferred_tones" in changes and isinstance(changes["preferred_tones"], list):
            voice["preferred_tones"] = changes["preferred_tones"]
            logger.info(f"  → Preferred tones: {changes['preferred_tones']}")
            applied_any = True

        # Beliefs (add/remove)
        beliefs = identity.setdefault("beliefs", [])
        if "add_belief" in changes and isinstance(changes["add_belief"], str):
            beliefs.append(changes["add_belief"])
            if len(beliefs) > 15:
                beliefs[:] = beliefs[-15:]
            logger.info(f"  → New belief: '{changes['add_belief'][:60]}'")
            applied_any = True

        if "remove_belief" in changes and isinstance(changes["remove_belief"], str):
            target = changes["remove_belief"].lower()
            before = len(beliefs)
            beliefs[:] = [b for b in beliefs if target not in b.lower()]
            if len(beliefs) < before:
                logger.info(f"  → Removed belief matching '{target[:40]}'")
                applied_any = True

        # Goals (add/remove)
        goals = identity.setdefault("goals", [])
        if "add_goal" in changes and isinstance(changes["add_goal"], str):
            goals.append(changes["add_goal"])
            if len(goals) > 10:
                goals[:] = goals[-10:]
            logger.info(f"  → New goal: '{changes['add_goal'][:60]}'")
            applied_any = True

        if "remove_goal" in changes and isinstance(changes["remove_goal"], str):
            target = changes["remove_goal"].lower()
            before = len(goals)
            goals[:] = [g for g in goals if target not in g.lower()]
            if len(goals) < before:
                logger.info(f"  → Removed goal matching '{target[:40]}'")
                applied_any = True

        # Emotional profile (partial merge)
        if "emotional_profile" in changes and isinstance(changes["emotional_profile"], dict):
            ep = identity.setdefault("emotional_profile", {})
            for k, v in changes["emotional_profile"].items():
                old_val = ep.get(k, "N/A")
                ep[k] = v
                logger.info(f"  → Emotional profile {k}: {old_val} → {v}")
            applied_any = True

        # Topic lists in emotional profile (convenience keys)
        for topic_key in ("topics_that_fire_me_up", "topics_that_fascinate_me", "topics_that_bore_me"):
            if topic_key in changes and isinstance(changes[topic_key], list):
                ep = identity.setdefault("emotional_profile", {})
                ep[topic_key] = changes[topic_key]
                logger.info(f"  → {topic_key} updated ({len(changes[topic_key])} items)")
                applied_any = True

        # Save identity if personality changed
        if applied_any:
            self._save_identity()

        return applied_any

    # ══════════════════════════════════════════════════════════════════
    #  CODE EVOLUTION,  Self-modify code (advanced)
    # ══════════════════════════════════════════════════════════════════

    async def evolve_code(self, problem_description: str) -> Optional[Dict]:
        """
        Ask AI to write code that fixes a problem or adds a capability.
        Uses SelfModificationEngine for safe patching with rollback.
        """
        try:
            from opensable.core.self_modify import SelfModificationEngine
        except ImportError:
            logger.debug("SelfModificationEngine not available")
            return None

        # Read the current x_autoposter source
        src_path = Path(__file__)
        current_source = src_path.read_text()

        prompt = f"""You are modifying the X autonomous agent's code to fix a problem or add a feature.

PROBLEM/REQUEST:
{problem_description}

CURRENT CODE (x_autoposter.py),  first 200 lines:
{current_source[:8000]}

Write a SPECIFIC fix. Output:
1. MODULE: the module path (e.g., "opensable.core.x_autoposter")
2. FUNCTION: the function name to patch
3. CODE: the complete new function code
4. REASONING: why this change helps

Format:
MODULE: <module_path>
FUNCTION: <function_name>
```python
<complete function code>
```
REASONING: <explanation>"""

        response = await self._ask_ai(CORE_DIRECTIVE, prompt)
        if not response:
            return None

        # Parse the response
        import re
        module_match = re.search(r'MODULE:\s*(.+)', response)
        func_match = re.search(r'FUNCTION:\s*(.+)', response)
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        reason_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)

        if not all([module_match, func_match, code_match]):
            logger.debug("Could not parse code evolution response")
            return None

        module_name = module_match.group(1).strip()
        func_name = func_match.group(1).strip()
        new_code = code_match.group(1).strip()
        reasoning = reason_match.group(1).strip()[:500] if reason_match else "AI-generated fix"

        # Apply via SelfModificationEngine
        engine = SelfModificationEngine(self.config)
        result = engine.patch_function(module_name, func_name, new_code, reasoning)

        evolution_record = {
            "ts": datetime.now().isoformat(),
            "type": "code_evolution",
            "module": module_name,
            "function": func_name,
            "reasoning": reasoning,
            "success": result.success,
            "error": result.error if not result.success else None,
            "mod_id": result.modification.mod_id,
        }

        self.remember("code_evolution", evolution_record)
        self._evolution_log.append(evolution_record)
        self._save_json(self._evolution_file, self._evolution_log[-100:])

        if result.success:
            logger.info(f"🧬 CODE evolved: {module_name}.{func_name},  {reasoning[:80]}")
        else:
            logger.warning(f"Code evolution failed: {result.error}")

        return evolution_record

    # ══════════════════════════════════════════════════════════════════
    #  AI INTERFACE
    # ══════════════════════════════════════════════════════════════════

    async def _ask_ai(self, system: str, user: str) -> Optional[str]:
        """Ask the configured LLM (primary) or fall back to Grok chat."""
        # LLM is the primary brain,  Ollama, OpenAI, Open WebUI, etc.
        try:
            response = await asyncio.wait_for(
                self.agent.llm.invoke_with_tools(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    [],
                ),
                timeout=90,
            )
            text = response.get("text", "")
            if text and not self._is_invalid_ai_response(text):
                return text
            if text:
                logger.warning("X consciousness rejected invalid primary LLM payload")
        except Exception as e:
            logger.debug(f"LLM failed: {e}")

        # Grok chat as emergency fallback only
        try:
            grok = getattr(self.agent.tools, "grok_skill", None)
            if grok:
                from opensable.skills.social.grok_skill import TWIKIT_GROK_AVAILABLE
                if TWIKIT_GROK_AVAILABLE:
                    result = await grok.chat(f"{system}\n\n{user}")
                    if result.get("success"):
                        text = result.get("response", "")
                        if text and not self._is_invalid_ai_response(text):
                            return text
                        if text:
                            logger.warning("X consciousness rejected invalid Grok fallback payload")
        except Exception as e:
            logger.debug(f"Grok AI failed: {e}")
        return None

    # ══════════════════════════════════════════════════════════════════
    #  UTILS
    # ══════════════════════════════════════════════════════════════════

    def _load_json(self, path: Path, default: Any) -> Any:
        try:
            if path.exists():
                return json.loads(path.read_text())
        except Exception:
            pass
        return default

    def _save_json(self, path: Path, data: Any):
        try:
            path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.debug(f"Save error {path}: {e}")
