"""
Natural Language Automation Engine

IFTTT/Zapier-style automation built in plain English.
Users define persistent trigger→action rules via natural language:
  "When I receive an email from my boss, summarize it and send me a WhatsApp"
  "Every morning at 8am, check the weather and post it to my Telegram"
  "When Bitcoin drops below 60k, alert me and sell 10%"

The engine:
  1. Parses NL rules into structured trigger + condition + action triples
  2. Registers persistent watchers (email, cron, market, webhooks, events)
  3. Evaluates triggers continuously in the autonomous tick
  4. Executes multi-step actions via the agent's tool system
  5. Logs every activation with full audit trail
"""
import json
import logging
import re
import asyncio
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────

class TriggerType:
    CRON = "cron"            # Time-based: "every day at 8am", "every 30 minutes"
    EMAIL = "email"          # Email arrival: "when I get email from X"
    WEBHOOK = "webhook"      # External HTTP: "when webhook fires"
    MARKET = "market"        # Price condition: "when BTC drops below 60k"
    EVENT = "event"          # Internal event: "when a task completes"
    FILE = "file"            # File change: "when file X is modified"
    KEYWORD = "keyword"      # Message contains: "when someone mentions X"
    SYSTEM = "system"        # System state: "when CPU > 80%"


@dataclass
class AutomationRule:
    """A single NL automation rule."""
    rule_id: str
    raw_text: str                          # Original NL rule text
    name: str = ""                         # Short friendly name
    trigger_type: str = ""                  # TriggerType value
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    created_at: str = ""
    last_triggered: Optional[str] = None
    trigger_count: int = 0
    cooldown_seconds: int = 60             # Min time between activations
    max_triggers_per_day: int = 50
    daily_trigger_count: int = 0
    daily_reset_date: str = ""
    error_count: int = 0
    last_error: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AutomationRule":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ActivationLog:
    """Log of a single rule activation."""
    rule_id: str
    timestamp: str
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    actions_executed: List[str] = field(default_factory=list)
    success: bool = True
    error: str = ""
    duration_ms: int = 0


# ── NL Rule Parsing ──────────────────────────────────────────────────

# Patterns for extracting trigger/action from natural language
_TRIGGER_PATTERNS = [
    # Time-based
    (r"(?:every|each)\s+(day|morning|evening|night|hour|week)\s*(?:at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?))?",
     TriggerType.CRON),
    (r"every\s+(\d+)\s+(minutes?|hours?|days?|weeks?)",
     TriggerType.CRON),
    (r"(?:at|on)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+(?:every|each)\s+(day|weekday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
     TriggerType.CRON),
    # Email
    (r"(?:when|if)\s+(?:I\s+)?(?:get|receive)\s+(?:an?\s+)?email\s+(?:from\s+(.+?))?(?:\s+(?:about|with|containing)\s+(.+?))?(?:,|\s+then|\s+do)",
     TriggerType.EMAIL),
    # Market
    (r"(?:when|if)\s+(bitcoin|btc|eth|ethereum|sol|solana|[\w/]+)\s+(?:drops?|falls?|goes?)\s+(?:below|under)\s+\$?([\d,.]+)",
     TriggerType.MARKET),
    (r"(?:when|if)\s+(bitcoin|btc|eth|ethereum|sol|solana|[\w/]+)\s+(?:rises?|goes?|climbs?)\s+(?:above|over)\s+\$?([\d,.]+)",
     TriggerType.MARKET),
    # Webhook
    (r"(?:when|if)\s+(?:a\s+)?webhook\s+(?:fires|triggers|arrives)",
     TriggerType.WEBHOOK),
    # File
    (r"(?:when|if)\s+(?:the\s+)?file\s+(.+?)\s+(?:changes|is\s+modified|is\s+updated)",
     TriggerType.FILE),
    # System
    (r"(?:when|if)\s+(?:cpu|memory|disk|ram)\s+(?:usage\s+)?(?:is\s+)?(?:above|over|exceeds?)\s+(\d+)%?",
     TriggerType.SYSTEM),
    # Keyword/mention
    (r"(?:when|if)\s+(?:someone|anyone)\s+(?:mentions?|says?|writes?)\s+(.+?)(?:,|\s+then|\s+do)",
     TriggerType.KEYWORD),
    # Generic event
    (r"(?:when|if)\s+(?:a\s+)?task\s+(?:completes?|finishes?|fails?)",
     TriggerType.EVENT),
]

_ACTION_KEYWORDS = [
    "send", "email", "notify", "alert", "message", "whatsapp", "telegram",
    "summarize", "summary", "translate", "post", "tweet", "publish",
    "save", "write", "log", "record", "store",
    "search", "find", "look up", "check",
    "sell", "buy", "trade", "transfer",
    "run", "execute", "start", "stop", "restart",
    "create", "generate", "build", "make",
]


def _parse_nl_rule(text: str) -> Dict[str, Any]:
    """Parse a natural language rule into structured trigger + actions."""
    text_lower = text.lower().strip()
    result = {
        "trigger_type": TriggerType.EVENT,
        "trigger_config": {},
        "conditions": [],
        "actions": [],
        "name": text[:60],
    }

    # Try to match trigger patterns
    for pattern, ttype in _TRIGGER_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            result["trigger_type"] = ttype
            result["trigger_config"] = {
                "match_groups": list(m.groups()),
                "matched_text": m.group(0),
            }

            # Extract specific config based on type
            if ttype == TriggerType.CRON:
                groups = m.groups()
                if groups[0] and groups[0].isdigit():
                    # "every N minutes/hours"
                    result["trigger_config"]["interval_value"] = int(groups[0])
                    result["trigger_config"]["interval_unit"] = groups[1].rstrip("s") if len(groups) > 1 and groups[1] else "minute"
                else:
                    result["trigger_config"]["schedule"] = groups[0] if groups[0] else "day"
                    if len(groups) > 1 and groups[1]:
                        result["trigger_config"]["time"] = groups[1].strip()

            elif ttype == TriggerType.EMAIL:
                if m.groups()[0]:
                    result["trigger_config"]["from_filter"] = m.groups()[0].strip().rstrip(",")
                if len(m.groups()) > 1 and m.groups()[1]:
                    result["trigger_config"]["subject_filter"] = m.groups()[1].strip().rstrip(",")

            elif ttype == TriggerType.MARKET:
                asset = m.groups()[0].upper()
                price = m.groups()[1].replace(",", "")
                direction = "below" if any(w in m.group(0) for w in ["below", "under", "drop", "fall"]) else "above"
                result["trigger_config"]["asset"] = asset
                result["trigger_config"]["threshold"] = float(price)
                result["trigger_config"]["direction"] = direction

            elif ttype == TriggerType.FILE:
                result["trigger_config"]["path"] = m.groups()[0].strip()

            elif ttype == TriggerType.SYSTEM:
                metric = "cpu"
                for m2 in ["cpu", "memory", "ram", "disk"]:
                    if m2 in text_lower:
                        metric = m2
                result["trigger_config"]["metric"] = metric
                result["trigger_config"]["threshold"] = int(m.groups()[0])

            elif ttype == TriggerType.KEYWORD:
                result["trigger_config"]["keyword"] = m.groups()[0].strip().rstrip(",")

            break

    # Extract action part (everything after "then", "do", comma after trigger)
    action_text = text
    for delimiter in [" then ", ", then ", " do ", ", do ", ", "]:
        parts = text_lower.split(delimiter, 1)
        if len(parts) > 1:
            action_text = text[len(parts[0]) + len(delimiter):]
            break

    # Parse actions from the action text
    result["actions"] = [{"type": "agent_execute", "instruction": action_text.strip()}]

    return result


# ── Core Engine ───────────────────────────────────────────────────────

class NLAutomationEngine:
    """
    Natural language automation engine.
    Users define rules in plain English, engine evaluates triggers
    and executes actions via the agent's tool system.
    """

    MAX_RULES = 500
    MAX_LOGS = 5000

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "nl_automation_state.json"
        self.log_file = self.data_dir / "nl_automation_log.jsonl"
        self.llm = None

        self.rules: List[AutomationRule] = []
        self.activation_logs: List[ActivationLog] = []
        self._pending_activations: List[Dict[str, Any]] = []

        # Stats
        self.total_rules_created = 0
        self.total_activations = 0
        self.total_errors = 0
        self.total_actions_executed = 0

        self._load_state()

    def set_llm(self, llm):
        """Store the shared LLM handle injected by the core agent."""
        self.llm = llm

    # ── Rule Management ───────────────────────────────────────────────

    def create_rule(self, nl_text: str, tags: Optional[List[str]] = None) -> AutomationRule:
        """Create a new automation rule from natural language."""
        parsed = _parse_nl_rule(nl_text)

        rule_id = f"rule_{hashlib.sha256(nl_text.encode()).hexdigest()[:12]}"

        rule = AutomationRule(
            rule_id=rule_id,
            raw_text=nl_text,
            name=parsed.get("name", nl_text[:60]),
            trigger_type=parsed["trigger_type"],
            trigger_config=parsed["trigger_config"],
            conditions=parsed["conditions"],
            actions=parsed["actions"],
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=tags or [],
        )

        # Don't add duplicate
        existing_ids = {r.rule_id for r in self.rules}
        if rule_id in existing_ids:
            logger.info(f"Rule already exists: {rule_id}")
            return next(r for r in self.rules if r.rule_id == rule_id)

        self.rules.append(rule)
        if len(self.rules) > self.MAX_RULES:
            # Remove oldest disabled rules first
            disabled = [r for r in self.rules if not r.enabled]
            if disabled:
                self.rules.remove(disabled[0])
            else:
                self.rules.pop(0)

        self.total_rules_created += 1
        self._save_state()

        logger.info(f"[NLAutomation] Created rule '{rule.name}' (trigger={rule.trigger_type})")
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete rule by ID."""
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        if len(self.rules) < before:
            self._save_state()
            return True
        return False

    def toggle_rule(self, rule_id: str, enabled: bool) -> bool:
        """Enable or disable a rule."""
        for r in self.rules:
            if r.rule_id == rule_id:
                r.enabled = enabled
                self._save_state()
                return True
        return False

    def list_rules(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """List all rules."""
        rules = self.rules if not enabled_only else [r for r in self.rules if r.enabled]
        return [r.to_dict() for r in rules]

    # ── Trigger Evaluation ────────────────────────────────────────────

    async def evaluate_triggers(self, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate all active rules against current state.
        Called every autonomous tick.
        Returns list of rules that should fire.
        """
        context = context or {}
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        activations = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Reset daily counter if new day
            if rule.daily_reset_date != today:
                rule.daily_trigger_count = 0
                rule.daily_reset_date = today

            # Check daily limit
            if rule.daily_trigger_count >= rule.max_triggers_per_day:
                continue

            # Check cooldown
            if rule.last_triggered:
                try:
                    last = datetime.fromisoformat(rule.last_triggered)
                    if (now - last).total_seconds() < rule.cooldown_seconds:
                        continue
                except (ValueError, TypeError):
                    pass

            # Evaluate trigger
            should_fire = await self._evaluate_rule(rule, context, now)

            if should_fire:
                activations.append({
                    "rule": rule,
                    "context": context,
                    "timestamp": now.isoformat(),
                })

        return activations

    async def _evaluate_rule(self, rule: AutomationRule, context: Dict, now: datetime) -> bool:
        """Evaluate a single rule's trigger condition."""
        ttype = rule.trigger_type
        config = rule.trigger_config

        if ttype == TriggerType.CRON:
            return self._eval_cron(config, now)
        elif ttype == TriggerType.EMAIL:
            return self._eval_email(config, context)
        elif ttype == TriggerType.MARKET:
            return self._eval_market(config, context)
        elif ttype == TriggerType.SYSTEM:
            return self._eval_system(config, context)
        elif ttype == TriggerType.KEYWORD:
            return self._eval_keyword(config, context)
        elif ttype == TriggerType.FILE:
            return self._eval_file(config, context)
        elif ttype == TriggerType.WEBHOOK:
            return self._eval_webhook(config, context)
        elif ttype == TriggerType.EVENT:
            return self._eval_event(config, context)
        return False

    def _eval_cron(self, config: Dict, now: datetime) -> bool:
        """Evaluate time-based trigger."""
        if "interval_value" in config:
            # Interval-based: "every N minutes"
            interval = config["interval_value"]
            unit = config.get("interval_unit", "minute")
            if unit == "minute":
                return now.minute % max(interval, 1) == 0 and now.second < 30
            elif unit == "hour":
                return now.hour % max(interval, 1) == 0 and now.minute == 0
            elif unit == "day":
                return now.hour == 8 and now.minute == 0  # Default 8am
        elif "schedule" in config:
            schedule = config.get("schedule", "day")
            time_str = config.get("time", "")
            target_hour = 8  # default

            if time_str:
                try:
                    parts = time_str.lower().strip()
                    is_pm = "pm" in parts
                    parts = parts.replace("am", "").replace("pm", "").strip()
                    h = int(parts.split(":")[0])
                    if is_pm and h != 12:
                        h += 12
                    if not is_pm and h == 12:
                        h = 0
                    target_hour = h
                except (ValueError, IndexError):
                    pass

            if schedule in ("day", "morning", "evening", "night"):
                return now.hour == target_hour and now.minute == 0
            elif schedule == "hour":
                return now.minute == 0

        return False

    def _eval_email(self, config: Dict, context: Dict) -> bool:
        """Evaluate email trigger."""
        new_emails = context.get("new_emails", [])
        from_filter = config.get("from_filter", "").lower()
        subject_filter = config.get("subject_filter", "").lower()

        for email in new_emails:
            sender = str(email.get("from", "")).lower()
            subject = str(email.get("subject", "")).lower()
            if from_filter and from_filter not in sender:
                continue
            if subject_filter and subject_filter not in subject:
                continue
            return True
        return False

    def _eval_market(self, config: Dict, context: Dict) -> bool:
        """Evaluate market price trigger."""
        prices = context.get("market_prices", {})
        asset = config.get("asset", "").upper()
        threshold = config.get("threshold", 0)
        direction = config.get("direction", "below")

        # Normalize asset names
        asset_map = {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT",
                     "BITCOIN": "BTC/USDT", "ETHEREUM": "ETH/USDT", "SOLANA": "SOL/USDT"}
        asset_key = asset_map.get(asset, asset)
        price = prices.get(asset_key) or prices.get(asset, 0)

        if price and direction == "below":
            return float(price) < threshold
        elif price and direction == "above":
            return float(price) > threshold
        return False

    def _eval_system(self, config: Dict, context: Dict) -> bool:
        """Evaluate system metric trigger."""
        metric = config.get("metric", "cpu")
        threshold = config.get("threshold", 80)
        sys_metrics = context.get("system_metrics", {})

        current = sys_metrics.get(f"{metric}_percent", 0)
        return current > threshold

    def _eval_keyword(self, config: Dict, context: Dict) -> bool:
        """Evaluate keyword mention trigger."""
        keyword = config.get("keyword", "").lower()
        messages = context.get("recent_messages", [])
        for msg in messages:
            if keyword in str(msg.get("content", "")).lower():
                return True
        return False

    def _eval_file(self, config: Dict, context: Dict) -> bool:
        """Evaluate file change trigger."""
        path = config.get("path", "")
        changed_files = context.get("changed_files", [])
        return any(path in f for f in changed_files)

    def _eval_webhook(self, config: Dict, context: Dict) -> bool:
        """Evaluate webhook trigger."""
        return bool(context.get("webhook_fired", False))

    def _eval_event(self, config: Dict, context: Dict) -> bool:
        """Evaluate internal event trigger."""
        events = context.get("events", [])
        return len(events) > 0

    # ── Activation ────────────────────────────────────────────────────

    async def activate_rule(self, rule: AutomationRule, trigger_data: Dict) -> ActivationLog:
        """
        Record rule activation + queue actions for execution.
        The agent will execute the actions via its tool system.
        """
        import time as _time
        t0 = _time.monotonic()

        now = datetime.now(timezone.utc).isoformat()
        rule.last_triggered = now
        rule.trigger_count += 1
        rule.daily_trigger_count += 1

        actions_desc = [a.get("instruction", str(a)) for a in rule.actions]

        log = ActivationLog(
            rule_id=rule.rule_id,
            timestamp=now,
            trigger_data=trigger_data,
            actions_executed=actions_desc,
            success=True,
            duration_ms=int((_time.monotonic() - t0) * 1000),
        )

        self.activation_logs.append(log)
        if len(self.activation_logs) > self.MAX_LOGS:
            self.activation_logs = self.activation_logs[-self.MAX_LOGS:]

        self.total_activations += 1
        self.total_actions_executed += len(actions_desc)

        # Queue the instruction for the agent to execute
        self._pending_activations.append({
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "instruction": " ".join(actions_desc),
            "trigger_data": trigger_data,
        })

        self._save_state()
        logger.info(f"[NLAutomation] Rule '{rule.name}' activated (#{rule.trigger_count})")
        return log

    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """Get and clear pending automation actions for agent execution."""
        pending = list(self._pending_activations)
        self._pending_activations.clear()
        return pending

    # ── Persistence ───────────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "rules": [r.to_dict() for r in self.rules],
                "total_rules_created": self.total_rules_created,
                "total_activations": self.total_activations,
                "total_errors": self.total_errors,
                "total_actions_executed": self.total_actions_executed,
            }
            self.state_file.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.error(f"[NLAutomation] Failed to save state: {e}")

    def _load_state(self):
        if self.state_file.exists():
            try:
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.rules = [AutomationRule.from_dict(r) for r in state.get("rules", [])]
                self.total_rules_created = state.get("total_rules_created", 0)
                self.total_activations = state.get("total_activations", 0)
                self.total_errors = state.get("total_errors", 0)
                self.total_actions_executed = state.get("total_actions_executed", 0)
                logger.info(f"[NLAutomation] Loaded {len(self.rules)} rules")
            except Exception as e:
                logger.error(f"[NLAutomation] Failed to load state: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        enabled = sum(1 for r in self.rules if r.enabled)
        by_type = {}
        for r in self.rules:
            by_type[r.trigger_type] = by_type.get(r.trigger_type, 0) + 1

        return {
            "total_rules": len(self.rules),
            "enabled_rules": enabled,
            "disabled_rules": len(self.rules) - enabled,
            "total_rules_created": self.total_rules_created,
            "total_activations": self.total_activations,
            "total_actions_executed": self.total_actions_executed,
            "total_errors": self.total_errors,
            "pending_actions": len(self._pending_activations),
            "rules_by_type": by_type,
            "recent_activations": len(self.activation_logs),
        }
