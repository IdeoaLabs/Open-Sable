"""
#18 — Narrative Warfare Monitor

Track information operations, disinformation campaigns, and narrative
manipulation across open sources (GDELT, RSS feeds, prediction markets,
social media proxies).

Detects:
  - Coordinated narrative pushes (same story planted across multiple outlets)
  - Sentiment shifts (sudden polarity change on topic/region)
  - Bot-amplified narratives (velocity > organic growth)
  - State media vs independent media divergence
  - Prediction market manipulation signals
  - Counter-narrative timing (narrative deployed to counter real event)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .connector import IntelSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class NarrativeCluster:
    """A cluster of related narrative items (articles, social posts)."""
    cluster_id: str
    topic: str
    keywords: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    article_count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    velocity: float = 0.0  # articles per hour
    sentiment: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    is_coordinated: bool = False
    coordination_score: float = 0.0  # 0-1
    state_media_ratio: float = 0.0  # % from known state media
    regions: List[str] = field(default_factory=list)


@dataclass
class NarrativeAlert:
    """Alert for detected information operation."""
    alert_id: str
    alert_type: str  # coordinated_push, sentiment_shift, velocity_spike,
                     # state_media_divergence, counter_narrative, prediction_manipulation
    severity: str
    timestamp: str
    title: str
    description: str
    cluster_id: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeItem:
    """A single piece of narrative content (article, post, prediction)."""
    item_id: str
    source: str
    title: str
    content: str
    timestamp: str
    timestamp_epoch: float
    url: str = ""
    sentiment: float = 0.0
    topics: List[str] = field(default_factory=list)
    is_state_media: bool = False
    region: str = ""


# ---------------------------------------------------------------------------
# Known state media outlets (non-exhaustive but operationally useful)
# ---------------------------------------------------------------------------

STATE_MEDIA: Dict[str, str] = {
    # Russia
    "rt.com": "russia", "tass.com": "russia", "sputniknews.com": "russia",
    "ria.ru": "russia", "iz.ru": "russia",
    # China
    "xinhuanet.com": "china", "globaltimes.cn": "china", "chinadaily.com.cn": "china",
    "cgtn.com": "china", "en.people.cn": "china",
    # Iran
    "presstv.ir": "iran", "irna.ir": "iran", "tehrantimes.com": "iran",
    # DPRK
    "kcna.kp": "dprk",
    # Turkey
    "trtworld.com": "turkey", "aa.com.tr": "turkey",
    # Qatar
    "aljazeera.com": "qatar",
    # Saudi
    "arabnews.com": "saudi",
}

# Keywords for geopolitical topic classification
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "ukraine_conflict": ["ukraine", "kyiv", "zelensky", "donbas", "crimea", "kherson"],
    "taiwan_tension": ["taiwan", "taipei", "cross-strait", "pla navy", "tsai"],
    "middle_east": ["gaza", "hamas", "hezbollah", "iran", "israel", "houthi"],
    "nato_expansion": ["nato", "alliance", "article 5", "collective defense"],
    "energy_security": ["oil", "gas", "pipeline", "opec", "lng", "energy"],
    "cyber_warfare": ["cyber attack", "ransomware", "apt", "hack", "breach"],
    "nuclear": ["nuclear", "warhead", "intercontinental", "icbm", "enrichment"],
    "sanctions": ["sanctions", "export controls", "embargo", "frozen assets"],
    "election_interference": ["election", "disinformation", "voter", "ballot", "interference"],
    "climate_security": ["climate", "drought", "flood", "migration", "food security"],
}

# Simple lexicon-based sentiment
POSITIVE_WORDS = {
    "peace", "agreement", "deal", "ceasefire", "progress", "success",
    "growth", "cooperation", "alliance", "stabilize", "recovery",
    "de-escalation", "support", "aid", "rescue", "negotiate",
}
NEGATIVE_WORDS = {
    "war", "attack", "strike", "killed", "bomb", "missile", "threat",
    "crisis", "invasion", "conflict", "sanctions", "collapse", "explosion",
    "destroy", "escalation", "hostile", "provocation", "violation",
    "breach", "casualties", "terror", "danger", "emergency",
}


class NarrativeWarfareMonitor:
    """
    Monitor information operations and narrative warfare across
    open intelligence sources.
    """

    MAX_ITEMS = 10000
    COORDINATION_THRESHOLD = 0.6  # Above this = coordinated
    VELOCITY_SPIKE_FACTOR = 5.0   # 5x normal = spike

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/zunvra_intel")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "narrative_state.json"

        self.items: List[NarrativeItem] = []
        self.clusters: Dict[str, NarrativeCluster] = {}
        self.alerts: List[NarrativeAlert] = []
        self.total_items_ingested = 0
        self._baseline_velocity: Dict[str, float] = {}  # topic → avg articles/hour

        self._load_state()

    # ── main ingestion ────────────────────────────────────────────────

    def observe(self, snapshot: IntelSnapshot) -> List[NarrativeAlert]:
        """
        Ingest narrative data from snapshot (news, GDELT, prediction markets)
        and detect information operations.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        now_epoch = time.time()
        new_alerts: List[NarrativeAlert] = []
        new_items: List[NarrativeItem] = []

        # Process GDELT events
        for event in snapshot.gdelt_events:
            item = self._process_gdelt(event, now_epoch)
            if item:
                new_items.append(item)

        # Process news items
        for news in snapshot.news_feed:
            item = self._process_news(news, now_epoch)
            if item:
                new_items.append(item)

        # Process predictions/markets
        for pred in snapshot.prediction_markets:
            item = self._process_prediction(pred, now_epoch)
            if item:
                new_items.append(item)

        # Add to corpus
        self.items.extend(new_items)
        self.total_items_ingested += len(new_items)

        # Trim
        if len(self.items) > self.MAX_ITEMS:
            self.items = self.items[-self.MAX_ITEMS:]

        # Run detection algorithms
        if new_items:
            new_alerts.extend(self._detect_coordinated_push(new_items, now_str))
            new_alerts.extend(self._detect_velocity_spikes(new_items, now_str))
            new_alerts.extend(self._detect_sentiment_shifts(new_items, now_str))
            new_alerts.extend(self._detect_state_media_divergence(new_items, now_str))

        self.alerts.extend(new_alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-250:]

        self._save_state()
        return new_alerts

    # ── item processing ───────────────────────────────────────────────

    def _process_gdelt(self, event: Dict[str, Any],
                       now_epoch: float) -> Optional[NarrativeItem]:
        url = event.get("url") or event.get("sourceurl", "")
        title = event.get("title") or event.get("SOURCEURL", "")
        if not title and not url:
            return None

        source = self._extract_domain(url)
        text = f"{title} {event.get('description', '')}"

        return NarrativeItem(
            item_id=hashlib.md5(f"gdelt_{url}_{now_epoch}".encode()).hexdigest()[:12],
            source=source,
            title=title[:200],
            content=text[:500],
            timestamp=datetime.now(timezone.utc).isoformat(),
            timestamp_epoch=now_epoch,
            url=url,
            sentiment=self._compute_sentiment(text),
            topics=self._classify_topics(text),
            is_state_media=source.lower() in STATE_MEDIA,
            region=STATE_MEDIA.get(source.lower(), ""),
        )

    def _process_news(self, news: Dict[str, Any],
                      now_epoch: float) -> Optional[NarrativeItem]:
        title = news.get("title", "")
        if not title:
            return None

        source = news.get("source", self._extract_domain(news.get("url", "")))
        text = f"{title} {news.get('description', '')} {news.get('content', '')}"

        return NarrativeItem(
            item_id=hashlib.md5(f"news_{title[:50]}_{now_epoch}".encode()).hexdigest()[:12],
            source=source,
            title=title[:200],
            content=text[:500],
            timestamp=news.get("published", datetime.now(timezone.utc).isoformat()),
            timestamp_epoch=now_epoch,
            url=news.get("url", ""),
            sentiment=self._compute_sentiment(text),
            topics=self._classify_topics(text),
            is_state_media=source.lower() in STATE_MEDIA,
        )

    def _process_prediction(self, pred: Dict[str, Any],
                             now_epoch: float) -> Optional[NarrativeItem]:
        title = pred.get("title") or pred.get("question", "")
        if not title:
            return None

        return NarrativeItem(
            item_id=hashlib.md5(f"pred_{title[:50]}_{now_epoch}".encode()).hexdigest()[:12],
            source=pred.get("platform", "prediction_market"),
            title=title[:200],
            content=f"{title} — probability: {pred.get('probability', '?')}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            timestamp_epoch=now_epoch,
            url=pred.get("url", ""),
            sentiment=0.0,
            topics=self._classify_topics(title),
        )

    # ── detection algorithms ──────────────────────────────────────────

    def _detect_coordinated_push(self, new_items: List[NarrativeItem],
                                  now_str: str) -> List[NarrativeAlert]:
        """Detect when same narrative appears across multiple outlets rapidly."""
        alerts: List[NarrativeAlert] = []

        # Cluster by topic similarity
        topic_groups: Dict[str, List[NarrativeItem]] = {}
        for item in new_items:
            for topic in item.topics:
                topic_groups.setdefault(topic, []).append(item)

        for topic, items in topic_groups.items():
            if len(items) < 3:
                continue

            unique_sources = set(i.source for i in items)
            if len(unique_sources) < 3:
                continue

            state_count = sum(1 for i in items if i.is_state_media)
            state_ratio = state_count / len(items)

            # Time spread — all within tight window?
            epochs = [i.timestamp_epoch for i in items]
            time_spread = max(epochs) - min(epochs) if epochs else 0
            tight_window = time_spread < 3600  # All within 1 hour

            # Coordination score
            coord_score = 0.0
            coord_score += min(0.3, len(unique_sources) * 0.05)  # More sources = more suspicious
            coord_score += 0.3 if tight_window else 0.0
            coord_score += state_ratio * 0.4

            if coord_score >= self.COORDINATION_THRESHOLD:
                cid = hashlib.md5(f"cluster_{topic}_{now_str}".encode()).hexdigest()[:10]
                self.clusters[cid] = NarrativeCluster(
                    cluster_id=cid,
                    topic=topic,
                    sources=list(unique_sources),
                    article_count=len(items),
                    first_seen=items[0].timestamp,
                    last_seen=items[-1].timestamp,
                    is_coordinated=True,
                    coordination_score=coord_score,
                    state_media_ratio=state_ratio,
                )

                alerts.append(NarrativeAlert(
                    alert_id=hashlib.md5(f"coord_{topic}_{now_str}".encode()).hexdigest()[:10],
                    alert_type="coordinated_push",
                    severity="high" if state_ratio > 0.5 else "medium",
                    timestamp=now_str,
                    title=f"Coordinated narrative push: {topic}",
                    description=(
                        f"{len(items)} articles from {len(unique_sources)} sources "
                        f"within {time_spread/60:.0f}min. State media ratio: {state_ratio:.0%}. "
                        f"Coordination score: {coord_score:.2f}"
                    ),
                    cluster_id=cid,
                    evidence={
                        "sources": list(unique_sources),
                        "article_count": len(items),
                        "state_media_ratio": state_ratio,
                        "time_spread_min": time_spread / 60,
                    },
                ))

        return alerts

    def _detect_velocity_spikes(self, new_items: List[NarrativeItem],
                                 now_str: str) -> List[NarrativeAlert]:
        """Detect topics with abnormal article velocity."""
        alerts: List[NarrativeAlert] = []

        topic_counts: Dict[str, int] = Counter()
        for item in new_items:
            for topic in item.topics:
                topic_counts[topic] += 1

        for topic, count in topic_counts.items():
            baseline = self._baseline_velocity.get(topic, 2.0)
            current_velocity = count  # per cycle

            if current_velocity > baseline * self.VELOCITY_SPIKE_FACTOR:
                alerts.append(NarrativeAlert(
                    alert_id=hashlib.md5(f"velocity_{topic}_{now_str}".encode()).hexdigest()[:10],
                    alert_type="velocity_spike",
                    severity="medium",
                    timestamp=now_str,
                    title=f"Narrative velocity spike: {topic}",
                    description=(
                        f"{count} items this cycle vs baseline {baseline:.1f}. "
                        f"Spike factor: {current_velocity/baseline:.1f}x"
                    ),
                    evidence={"current": count, "baseline": baseline,
                              "spike_factor": current_velocity / baseline},
                ))

            # Update baseline (exponential moving average)
            self._baseline_velocity[topic] = baseline * 0.9 + count * 0.1

        return alerts

    def _detect_sentiment_shifts(self, new_items: List[NarrativeItem],
                                  now_str: str) -> List[NarrativeAlert]:
        """Detect sudden sentiment changes on a topic."""
        alerts: List[NarrativeAlert] = []

        topic_sentiments: Dict[str, List[float]] = {}
        for item in new_items:
            for topic in item.topics:
                topic_sentiments.setdefault(topic, []).append(item.sentiment)

        # Compare with historical
        historical: Dict[str, List[float]] = {}
        for item in self.items[:-len(new_items)] if len(self.items) > len(new_items) else []:
            for topic in item.topics:
                historical.setdefault(topic, []).append(item.sentiment)

        for topic, current_sents in topic_sentiments.items():
            hist = historical.get(topic, [])
            if len(hist) < 5:
                continue

            avg_current = sum(current_sents) / len(current_sents)
            avg_hist = sum(hist) / len(hist)
            shift = abs(avg_current - avg_hist)

            if shift > 0.4:  # Significant sentiment shift
                direction = "positive" if avg_current > avg_hist else "negative"
                alerts.append(NarrativeAlert(
                    alert_id=hashlib.md5(f"sentiment_{topic}_{now_str}".encode()).hexdigest()[:10],
                    alert_type="sentiment_shift",
                    severity="medium",
                    timestamp=now_str,
                    title=f"Sentiment shift on {topic}: → {direction}",
                    description=(
                        f"Sentiment shifted from {avg_hist:+.2f} to {avg_current:+.2f} "
                        f"(Δ{shift:.2f}). Based on {len(current_sents)} new items "
                        f"vs {len(hist)} historical."
                    ),
                    evidence={"avg_current": avg_current, "avg_historical": avg_hist,
                              "shift": shift, "direction": direction},
                ))

        return alerts

    def _detect_state_media_divergence(self, new_items: List[NarrativeItem],
                                        now_str: str) -> List[NarrativeAlert]:
        """Detect when state media narrative diverges from independent."""
        alerts: List[NarrativeAlert] = []

        state_items = [i for i in new_items if i.is_state_media]
        indie_items = [i for i in new_items if not i.is_state_media]

        if len(state_items) < 2 or len(indie_items) < 2:
            return alerts

        # Compare topic distributions
        state_topics: Counter = Counter()
        indie_topics: Counter = Counter()

        for item in state_items:
            for topic in item.topics:
                state_topics[topic] += 1
        for item in indie_items:
            for topic in item.topics:
                indie_topics[topic] += 1

        # Find topics heavily pushed by state media but not by independents
        for topic, state_count in state_topics.items():
            indie_count = indie_topics.get(topic, 0)
            if state_count >= 3 and indie_count <= 1:
                alerts.append(NarrativeAlert(
                    alert_id=hashlib.md5(f"diverge_{topic}_{now_str}".encode()).hexdigest()[:10],
                    alert_type="state_media_divergence",
                    severity="high",
                    timestamp=now_str,
                    title=f"State media narrative divergence: {topic}",
                    description=(
                        f"State media pushing '{topic}' ({state_count} items) while "
                        f"independent media has only {indie_count} items. "
                        f"Possible information operation."
                    ),
                    evidence={"state_count": state_count, "indie_count": indie_count,
                              "state_sources": [i.source for i in state_items if topic in i.topics]},
                ))

        # Compare sentiment divergence on same topic
        for topic in set(state_topics.keys()) & set(indie_topics.keys()):
            state_sent = [i.sentiment for i in state_items if topic in i.topics]
            indie_sent = [i.sentiment for i in indie_items if topic in i.topics]

            if state_sent and indie_sent:
                avg_state = sum(state_sent) / len(state_sent)
                avg_indie = sum(indie_sent) / len(indie_sent)
                divergence = abs(avg_state - avg_indie)

                if divergence > 0.5:
                    alerts.append(NarrativeAlert(
                        alert_id=hashlib.md5(f"sentdiv_{topic}_{now_str}".encode()).hexdigest()[:10],
                        alert_type="state_media_divergence",
                        severity="medium",
                        timestamp=now_str,
                        title=f"Sentiment divergence on {topic}: state vs independent",
                        description=(
                            f"State media sentiment: {avg_state:+.2f}, "
                            f"Independent: {avg_indie:+.2f} (Δ{divergence:.2f})"
                        ),
                        evidence={"state_sentiment": avg_state,
                                  "indie_sentiment": avg_indie,
                                  "divergence": divergence},
                    ))

        return alerts

    # ── NLP utilities (rule-based, no deps) ───────────────────────────

    @staticmethod
    def _compute_sentiment(text: str) -> float:
        """Simple lexicon-based sentiment (-1.0 to 1.0)."""
        words = set(re.findall(r'\b[a-z]+\b', text.lower()))
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    @staticmethod
    def _classify_topics(text: str) -> List[str]:
        """Classify text into geopolitical topics."""
        text_lower = text.lower()
        topics = []
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        return topics if topics else ["unclassified"]

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1) if match else url

    # ── LLM enrichment ────────────────────────────────────────────────

    async def analyze_narrative(self, topic: str, llm: Any) -> str:
        """Use LLM to provide deep analysis of a narrative cluster."""
        if not llm:
            return self._rule_analysis(topic)

        # Gather items on this topic
        relevant = [i for i in self.items if topic in i.topics][-20:]
        if not relevant:
            return f"No data on topic: {topic}"

        items_text = "\n".join(
            f"[{i.source}] ({i.sentiment:+.2f}) {i.title}"
            for i in relevant
        )

        prompt = (
            f"You are an information warfare analyst. Analyze these narratives "
            f"on the topic '{topic}' for signs of coordinated information operations.\n\n"
            f"ITEMS:\n{items_text}\n\n"
            f"Assess:\n1. Is there evidence of coordination?\n"
            f"2. Which state actors might be involved?\n"
            f"3. What is the likely objective?\n"
            f"4. Recommended counter-narrative strategy?\n"
        )

        try:
            return await llm.ask(prompt)
        except Exception as e:
            logger.warning("LLM narrative analysis failed: %s", e)
            return self._rule_analysis(topic)

    def _rule_analysis(self, topic: str) -> str:
        """Fallback rule-based analysis."""
        relevant = [i for i in self.items if topic in i.topics]
        if not relevant:
            return f"No data on topic: {topic}"

        sources = Counter(i.source for i in relevant)
        avg_sent = sum(i.sentiment for i in relevant) / len(relevant)
        state_count = sum(1 for i in relevant if i.is_state_media)

        lines = [
            f"Narrative Analysis: {topic}",
            f"Total items: {len(relevant)}",
            f"Unique sources: {len(sources)}",
            f"Average sentiment: {avg_sent:+.2f}",
            f"State media items: {state_count} ({state_count/len(relevant):.0%})",
            f"Top sources: {sources.most_common(5)}",
        ]
        return "\n".join(lines)

    # ── queries ───────────────────────────────────────────────────────

    def get_recent_alerts(self, limit: int = 30) -> List[NarrativeAlert]:
        return self.alerts[-limit:]

    def get_clusters(self) -> List[NarrativeCluster]:
        return list(self.clusters.values())

    def get_topic_summary(self) -> Dict[str, int]:
        topics: Counter = Counter()
        for item in self.items:
            for topic in item.topics:
                topics[topic] += 1
        return dict(topics.most_common(20))

    def search(self, query: str, limit: int = 50) -> List[NarrativeItem]:
        q = query.lower()
        return [i for i in self.items
                if q in i.title.lower() or q in i.content.lower()][-limit:]

    # ── persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "total_items_ingested": self.total_items_ingested,
                "current_items": len(self.items),
                "clusters": len(self.clusters),
                "alerts": len(self.alerts),
            }
            self.state_file.write_text(json.dumps(state, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save narrative state: %s", e)

    def _load_state(self):
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text("utf-8"))
                self.total_items_ingested = state.get("total_items_ingested", 0)
        except Exception as e:
            logger.warning("Failed to load narrative state: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_items_ingested": self.total_items_ingested,
            "current_items": len(self.items),
            "clusters_detected": len(self.clusters),
            "coordinated_clusters": sum(1 for c in self.clusters.values() if c.is_coordinated),
            "total_alerts": len(self.alerts),
            "topics_tracked": len(self.get_topic_summary()),
        }
