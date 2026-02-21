"""
Emotional Intelligence Layer

Real emotion detection, tracking, and response adaptation.

Architecture:
  Input text → EmotionDetector (lexicon + pattern analysis)
                    │
              EmotionState (rolling window per user)
                    │
              ResponseAdapter (adjusts tone, empathy, pacing)

Works fully offline — no external API needed.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Emotion taxonomy ─────────────────────────────────────────────────────────

class Emotion(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"
    EXCITEMENT = "excitement"
    GRATITUDE = "gratitude"


@dataclass
class EmotionScore:
    """Score for a single detected emotion."""
    emotion: Emotion
    confidence: float          # 0.0 – 1.0
    triggers: List[str]        # words/patterns that triggered this
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion.value,
            "confidence": round(self.confidence, 3),
            "triggers": self.triggers,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class UserEmotionState:
    """Tracks emotional state for one user over time."""
    user_id: str
    history: List[EmotionScore] = field(default_factory=list)
    dominant_emotion: Emotion = Emotion.NEUTRAL
    sentiment_trend: float = 0.0   # -1.0 (negative) to +1.0 (positive)
    interaction_count: int = 0

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "dominant_emotion": self.dominant_emotion.value,
            "sentiment_trend": round(self.sentiment_trend, 3),
            "interaction_count": self.interaction_count,
            "recent_emotions": [e.to_dict() for e in self.history[-5:]],
        }


# ─── Emotion Detection ───────────────────────────────────────────────────────

class EmotionDetector:
    """
    Detects emotions from text using multi-signal analysis:
      1. Lexicon matching (emotion word lists)
      2. Pattern/punctuation analysis (!! ?? CAPS)
      3. Emoji sentiment mapping
      4. Contextual intensifiers/negation
    """

    # Emotion lexicons (English + Spanish mixed for this project)
    LEXICON: Dict[Emotion, List[str]] = {
        Emotion.JOY: [
            "happy", "glad", "great", "wonderful", "amazing", "love", "excellent",
            "awesome", "fantastic", "perfect", "beautiful", "enjoy", "pleased",
            "delighted", "cheerful", "blessed", "grateful", "thankful", "yay",
            "feliz", "genial", "increíble", "maravilloso", "excelente", "perfecto",
            "me encanta", "qué bien", "buenísimo",
        ],
        Emotion.SADNESS: [
            "sad", "unhappy", "depressed", "miserable", "sorry", "miss", "lonely",
            "heartbroken", "disappointed", "crying", "tears", "grief", "mourn",
            "hopeless", "gloomy", "down", "blue",
            "triste", "deprimido", "solo", "llorar", "extraño", "decepcionado",
        ],
        Emotion.ANGER: [
            "angry", "furious", "mad", "hate", "annoyed", "irritated", "outraged",
            "pissed", "stupid", "idiot", "ridiculous", "terrible", "worst",
            "unacceptable", "rage", "hostile",
            "enojado", "furioso", "odio", "molesto", "irritado", "ridículo",
        ],
        Emotion.FEAR: [
            "afraid", "scared", "terrified", "anxious", "worried", "panic",
            "nervous", "frightened", "dread", "horror", "alarmed", "uneasy",
            "miedo", "asustado", "ansioso", "preocupado", "pánico", "nervioso",
        ],
        Emotion.SURPRISE: [
            "surprised", "shocked", "amazed", "astonished", "unexpected",
            "unbelievable", "wow", "whoa", "omg", "no way", "really",
            "sorprendido", "impactado", "increíble", "no puede ser",
        ],
        Emotion.FRUSTRATION: [
            "frustrated", "frustrating", "frustration", "stuck", "broken",
            "doesn't work", "doesnt work", "not working", "nothing works",
            "can't", "unable", "impossible", "ugh", "argh", "damn", "dammit",
            "why won't", "keeps failing", "still broken", "help me",
            "won't work", "wont work", "so annoying", "useless",
            "frustrado", "no funciona", "no puedo", "atascado", "roto",
            "no sirve", "no jala",
        ],
        Emotion.CONFUSION: [
            "confused", "don't understand", "what do you mean", "how does",
            "i'm lost", "makes no sense", "unclear", "huh", "what",
            "confundido", "no entiendo", "qué significa", "no me queda claro",
        ],
        Emotion.EXCITEMENT: [
            "excited", "can't wait", "thrilled", "pumped", "hyped", "stoked",
            "let's go", "finally", "yes!", "woohoo",
            "emocionado", "no puedo esperar", "por fin", "vamos",
        ],
        Emotion.GRATITUDE: [
            "thank you", "thanks", "appreciate", "grateful", "you're the best",
            "helpful", "lifesaver", "saved me", "perfect thanks",
            "gracias", "te agradezco", "eres el mejor", "me salvaste",
        ],
        Emotion.DISGUST: [
            "disgusting", "gross", "nasty", "eww", "yuck", "revolting",
            "horrible", "repulsive", "vile",
            "asqueroso", "horrible", "qué asco",
        ],
        Emotion.TRUST: [
            "trust", "reliable", "depend", "confident", "believe", "faith",
            "honest", "loyal", "safe", "secure",
            "confío", "confiable", "seguro",
        ],
        Emotion.ANTICIPATION: [
            "looking forward", "can't wait", "hoping", "planning", "soon",
            "expect", "about to", "ready for", "preparing",
            "esperando", "planeando", "listo para",
        ],
    }

    # Emoji → emotion mapping
    EMOJI_MAP: Dict[str, Emotion] = {
        "😊": Emotion.JOY, "😃": Emotion.JOY, "😁": Emotion.JOY,
        "❤️": Emotion.JOY, "🥰": Emotion.JOY, "😍": Emotion.JOY,
        "👍": Emotion.TRUST, "💪": Emotion.EXCITEMENT,
        "😢": Emotion.SADNESS, "😭": Emotion.SADNESS, "💔": Emotion.SADNESS,
        "😡": Emotion.ANGER, "🤬": Emotion.ANGER, "💢": Emotion.ANGER,
        "😰": Emotion.FEAR, "😨": Emotion.FEAR, "😱": Emotion.FEAR,
        "😲": Emotion.SURPRISE, "🤯": Emotion.SURPRISE,
        "😤": Emotion.FRUSTRATION, "🤦": Emotion.FRUSTRATION,
        "😕": Emotion.CONFUSION, "🤔": Emotion.CONFUSION,
        "🎉": Emotion.EXCITEMENT, "🚀": Emotion.EXCITEMENT,
        "🙏": Emotion.GRATITUDE, "🤝": Emotion.TRUST,
        "🤮": Emotion.DISGUST,
    }

    # Intensifiers multiply confidence
    INTENSIFIERS = [
        "very", "really", "extremely", "so", "incredibly", "absolutely",
        "totally", "completely", "super", "utterly",
        "muy", "demasiado", "súper", "bastante", "totalmente",
    ]

    # Negation words flip emotion
    NEGATORS = [
        "not", "no", "don't", "doesn't", "won't", "can't", "never",
        "neither", "nor", "hardly", "barely",
        "no", "nunca", "tampoco", "jamás",
    ]

    def detect(self, text: str) -> List[EmotionScore]:
        """
        Analyze text and return emotion scores (sorted by confidence desc).
        """
        if not text or not text.strip():
            return [EmotionScore(Emotion.NEUTRAL, 1.0, ["empty"])]

        text_lower = text.lower().strip()
        scores: Dict[Emotion, Tuple[float, List[str]]] = {}

        # ── Signal 1: Lexicon matching ────────────────────────────────
        for emotion, words in self.LEXICON.items():
            triggers = []
            for word in words:
                if word in text_lower:
                    triggers.append(word)
            if triggers:
                # Single-word exact matches score higher (direct emotion words)
                single_word_matches = sum(1 for t in triggers if " " not in t and t in text_lower.split())
                phrase_matches = len(triggers) - single_word_matches
                base = min((single_word_matches * 0.35 + phrase_matches * 0.2), 0.95)
                scores[emotion] = (base, triggers)

        # ── Signal 2: Punctuation / caps patterns ─────────────────────
        exclaim_count = text.count("!")
        question_count = text.count("?")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        if exclaim_count >= 3:
            # Many exclamation marks → boost existing negative emotions first,
            # only boost excitement if no negative emotion detected
            neg_emotions = {Emotion.ANGER, Emotion.FRUSTRATION, Emotion.SADNESS}
            has_negative = any(e in scores for e in neg_emotions)
            if has_negative:
                for e in neg_emotions:
                    if e in scores:
                        conf, triggers = scores[e]
                        scores[e] = (min(conf + 0.25, 1.0), triggers + ["!!!"])
            else:
                conf, triggers = scores.get(Emotion.EXCITEMENT, (0.0, []))
                scores[Emotion.EXCITEMENT] = (min(conf + 0.2, 1.0), triggers + ["!!!"])

        if question_count >= 2:
            conf, triggers = scores.get(Emotion.CONFUSION, (0.0, []))
            scores[Emotion.CONFUSION] = (min(conf + 0.15, 1.0), triggers + ["???"])

        if caps_ratio > 0.5 and len(text) > 5:
            # ALL CAPS → intensity boost (often anger/frustration)
            for e in [Emotion.ANGER, Emotion.FRUSTRATION]:
                conf, triggers = scores.get(e, (0.0, []))
                scores[e] = (min(conf + 0.25, 1.0), triggers + ["CAPS"])

        # ── Signal 3: Emoji detection ─────────────────────────────────
        for emoji, emotion in self.EMOJI_MAP.items():
            if emoji in text:
                conf, triggers = scores.get(emotion, (0.0, []))
                scores[emotion] = (min(conf + 0.3, 1.0), triggers + [emoji])

        # ── Signal 4: Intensifiers & negation ─────────────────────────
        has_intensifier = any(w in text_lower.split() for w in self.INTENSIFIERS)
        has_negation = any(w in text_lower.split() for w in self.NEGATORS)

        if has_intensifier:
            for e in scores:
                conf, triggers = scores[e]
                scores[e] = (min(conf * 1.3, 1.0), triggers)

        if has_negation:
            # Negation: reduce positive emotions, boost negative ones
            flip_map = {
                Emotion.JOY: Emotion.SADNESS,
                Emotion.EXCITEMENT: Emotion.FRUSTRATION,
                Emotion.TRUST: Emotion.FEAR,
                Emotion.GRATITUDE: Emotion.NEUTRAL,
            }
            new_scores = {}
            for e, (conf, triggers) in scores.items():
                if e in flip_map:
                    flipped = flip_map[e]
                    existing_conf, existing_triggers = new_scores.get(flipped, (0.0, []))
                    new_scores[flipped] = (
                        min(existing_conf + conf * 0.6, 1.0),
                        existing_triggers + triggers + ["negation"],
                    )
                else:
                    new_scores[e] = (conf, triggers)
            scores = new_scores

        # ── Build result ──────────────────────────────────────────────
        if not scores:
            return [EmotionScore(Emotion.NEUTRAL, 0.8, ["no_signal"])]

        result = [
            EmotionScore(emotion=e, confidence=conf, triggers=triggers)
            for e, (conf, triggers) in scores.items()
        ]
        result.sort(key=lambda s: s.confidence, reverse=True)
        return result


# ─── Emotion State Tracker ────────────────────────────────────────────────────

class EmotionTracker:
    """
    Maintains per-user emotional state over a rolling window.
    Tracks dominant emotion, sentiment trend, and emotional volatility.
    """

    def __init__(self, window_minutes: int = 30, max_history: int = 100):
        self.detector = EmotionDetector()
        self.window = timedelta(minutes=window_minutes)
        self.max_history = max_history
        self._users: Dict[str, UserEmotionState] = {}

    def analyze(self, user_id: str, text: str) -> UserEmotionState:
        """Analyze text, update user state, and return it."""
        scores = self.detector.detect(text)

        state = self._users.setdefault(
            user_id, UserEmotionState(user_id=user_id)
        )

        # Add scores to history
        state.history.extend(scores)
        state.interaction_count += 1

        # Prune old history
        cutoff = datetime.now(timezone.utc) - self.window
        state.history = [s for s in state.history if s.timestamp > cutoff]
        if len(state.history) > self.max_history:
            state.history = state.history[-self.max_history:]

        # Compute dominant emotion (weighted by recency + confidence)
        emotion_weights: Dict[Emotion, float] = defaultdict(float)
        now = datetime.now(timezone.utc)
        for s in state.history:
            age = (now - s.timestamp).total_seconds()
            recency = max(0.1, 1.0 - age / self.window.total_seconds())
            emotion_weights[s.emotion] += s.confidence * recency

        if emotion_weights:
            state.dominant_emotion = max(emotion_weights, key=emotion_weights.get)

        # Compute sentiment trend
        SENTIMENT_MAP = {
            Emotion.JOY: 1.0, Emotion.EXCITEMENT: 0.8,
            Emotion.GRATITUDE: 0.9, Emotion.TRUST: 0.6,
            Emotion.ANTICIPATION: 0.4, Emotion.SURPRISE: 0.2,
            Emotion.NEUTRAL: 0.0, Emotion.CONFUSION: -0.2,
            Emotion.FRUSTRATION: -0.5, Emotion.FEAR: -0.6,
            Emotion.SADNESS: -0.7, Emotion.DISGUST: -0.8,
            Emotion.ANGER: -0.9,
        }
        if state.history:
            recent = state.history[-10:]
            state.sentiment_trend = sum(
                SENTIMENT_MAP.get(s.emotion, 0) * s.confidence
                for s in recent
            ) / len(recent)

        return state

    def get_state(self, user_id: str) -> Optional[UserEmotionState]:
        return self._users.get(user_id)

    def get_all_states(self) -> Dict[str, Dict]:
        return {uid: s.to_dict() for uid, s in self._users.items()}


# ─── Response Adapter ─────────────────────────────────────────────────────────

class ResponseAdapter:
    """
    Adjusts the agent's response tone based on the user's emotional state.

    Instead of replacing the response, it provides:
      - A system prompt modifier for the LLM
      - Pre/post wrapping for the response text
      - Suggested response pacing (fast vs. gentle)
    """

    # Emotion → response strategy
    STRATEGIES: Dict[Emotion, Dict[str, Any]] = {
        Emotion.JOY: {
            "tone": "warm and enthusiastic",
            "prefix": "",
            "suffix": "",
            "system_addon": "The user is in a positive mood. Match their energy, be warm and encouraging.",
            "pacing": "normal",
        },
        Emotion.SADNESS: {
            "tone": "empathetic and gentle",
            "prefix": "",
            "suffix": "",
            "system_addon": "The user seems sad or down. Be empathetic, acknowledge their feelings gently, and offer support without being dismissive.",
            "pacing": "slow",
        },
        Emotion.ANGER: {
            "tone": "calm and validating",
            "prefix": "",
            "suffix": "",
            "system_addon": "The user seems frustrated or angry. Stay calm, validate their concern, and focus on being helpful and solution-oriented. Don't be defensive.",
            "pacing": "slow",
        },
        Emotion.FEAR: {
            "tone": "reassuring and steady",
            "prefix": "",
            "suffix": "",
            "system_addon": "The user seems anxious or worried. Be reassuring, break things into manageable steps, and provide clear guidance.",
            "pacing": "slow",
        },
        Emotion.FRUSTRATION: {
            "tone": "patient and solution-focused",
            "prefix": "",
            "suffix": "",
            "system_addon": "The user is frustrated (something isn't working or is confusing). Be extra patient, acknowledge the difficulty, and give clear step-by-step help.",
            "pacing": "slow",
        },
        Emotion.CONFUSION: {
            "tone": "clear and pedagogical",
            "prefix": "",
            "suffix": "",
            "system_addon": "The user is confused. Explain things simply and clearly. Use examples. Avoid jargon unless you explain it.",
            "pacing": "normal",
        },
        Emotion.EXCITEMENT: {
            "tone": "enthusiastic and supportive",
            "prefix": "",
            "suffix": "",
            "system_addon": "The user is excited! Match their enthusiasm and help them channel it productively.",
            "pacing": "fast",
        },
        Emotion.GRATITUDE: {
            "tone": "warm and humble",
            "prefix": "",
            "suffix": "",
            "system_addon": "The user is being grateful. Acknowledge it warmly but briefly, then stay helpful.",
            "pacing": "normal",
        },
        Emotion.NEUTRAL: {
            "tone": "balanced and helpful",
            "prefix": "",
            "suffix": "",
            "system_addon": "",
            "pacing": "normal",
        },
    }

    def __init__(self, tracker: EmotionTracker):
        self.tracker = tracker

    def adapt(self, user_id: str, text: str) -> Dict[str, Any]:
        """
        Analyze user input and return adaptation instructions.

        Returns:
            {
                "emotion": "frustration",
                "confidence": 0.75,
                "sentiment_trend": -0.3,
                "system_prompt_addon": "...",
                "tone": "patient and solution-focused",
                "pacing": "slow",
            }
        """
        state = self.tracker.analyze(user_id, text)
        strategy = self.STRATEGIES.get(
            state.dominant_emotion,
            self.STRATEGIES[Emotion.NEUTRAL],
        )

        return {
            "emotion": state.dominant_emotion.value,
            "confidence": state.history[-1].confidence if state.history else 0.0,
            "sentiment_trend": state.sentiment_trend,
            "interaction_count": state.interaction_count,
            "system_prompt_addon": strategy["system_addon"],
            "tone": strategy["tone"],
            "pacing": strategy["pacing"],
        }

    def get_system_prompt_modifier(self, user_id: str) -> str:
        """Get the system prompt addition based on user's current emotional state."""
        state = self.tracker.get_state(user_id)
        if not state:
            return ""
        strategy = self.STRATEGIES.get(
            state.dominant_emotion,
            self.STRATEGIES[Emotion.NEUTRAL],
        )
        return strategy.get("system_addon", "")


# ─── Integration point ────────────────────────────────────────────────────────

class EmotionalIntelligence:
    """
    Main entry point for the emotional intelligence layer.

    Usage in agent:
        ei = EmotionalIntelligence()
        adaptation = ei.process(user_id, user_text)
        # Use adaptation["system_prompt_addon"] in LLM call
    """

    def __init__(self, window_minutes: int = 30):
        self.tracker = EmotionTracker(window_minutes=window_minutes)
        self.adapter = ResponseAdapter(self.tracker)
        logger.info("Emotional intelligence layer initialized")

    def process(self, user_id: str, text: str) -> Dict[str, Any]:
        """Analyze text, update state, return adaptation instructions."""
        return self.adapter.adapt(user_id, text)

    def get_user_emotion(self, user_id: str) -> Optional[Dict]:
        """Get current emotional state for a user."""
        state = self.tracker.get_state(user_id)
        return state.to_dict() if state else None

    def get_all_emotions(self) -> Dict[str, Dict]:
        """Get emotional states for all users."""
        return self.tracker.get_all_states()

    def get_system_prompt_addon(self, user_id: str) -> str:
        """Convenience method for the agent's LLM call."""
        return self.adapter.get_system_prompt_modifier(user_id)
