"""
LLM integration for Open-Sable - Ollama with native tool calling
Dynamic model switching based on task requirements.
Supports: Ollama (local), OpenAI, Anthropic, DeepSeek, Groq, Together AI,
          xAI (Grok), Mistral, Google Gemini, Cohere, Kimi (Moonshot),
          Qwen (DashScope), OpenRouter,  all with full tool calling.
"""

import asyncio
import logging
import sys
import json
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Any, AsyncIterator, Optional, Tuple
try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ── Inter-process Ollama queue ───────────────────────────────────────
# When multiple agents share the same Ollama instance, concurrent requests
# force the model to context-switch or double-load, causing slowdowns and
# potential OOM.  This file lock serialises access so only one agent
# generates at a time.  The lock is acquired asynchronously (via executor)
# so the event loop stays responsive while waiting.

_OLLAMA_LOCK_PATH = os.environ.get("SABLE_OLLAMA_LOCK", "/tmp/sable-ollama.lock")


if sys.platform != "win32":
    import fcntl as _fcntl  # Unix only
else:
    _fcntl = None  # type: ignore[assignment]


@asynccontextmanager
async def _ollama_lock():
    """Async context-manager that acquires an inter-process file lock.

    Uses fcntl.flock on Unix/Mac. On Windows, file locking via fcntl is not
    available so this becomes a no-op (single-agent use is the norm on Windows).
    """
    if _fcntl is None:
        # Windows: no fcntl, just yield without locking
        yield
        return

    loop = asyncio.get_running_loop()
    fd = os.open(_OLLAMA_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    acquired = False
    try:
        # Try non-blocking first
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            # Another process holds the lock, wait up to 10 s then proceed anyway
            logger.warning("Ollama lock contended, waiting up to 10s...")
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, _fcntl.flock, fd, _fcntl.LOCK_EX),
                    timeout=10,
                )
                acquired = True
            except asyncio.TimeoutError:
                logger.warning("Ollama lock wait timed out, proceeding without lock")
        logger.debug("Ollama lock acquired" if acquired else "Ollama lock skipped")
        yield
    finally:
        if acquired:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        os.close(fd)
        logger.debug("Ollama lock released")


# ── DeepSeek <think> tag support ─────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_thinking(text: str) -> Tuple[str, Optional[str]]:
    """Extract DeepSeek-style <think>...</think> reasoning from response text.

    Returns:
        (clean_text, reasoning)  --  reasoning is None when no tags found.
    """
    if not text or "<think>" not in text:
        return text, None

    reasoning_parts = _THINK_RE.findall(text)
    clean = _THINK_RE.sub("", text).strip()
    reasoning = "\n".join(p.strip() for p in reasoning_parts if p.strip()) or None
    return clean, reasoning


# Models where /no_think kills output,  the distilled versions don't handle
# the Qwen3 /no_think token properly and produce empty responses.
_NO_THINK_BLOCKLIST_PATTERNS = ["distill", "distilled", "claude.*opus.*distill", "opus.*distill"]

def _should_skip_no_think(model_name: str) -> bool:
    """Return True if the model is known to break with /no_think."""
    lower = (model_name or "").lower()
    for pat in _NO_THINK_BLOCKLIST_PATTERNS:
        if re.search(pat, lower):
            return True
    return False

def _inject_no_think(messages: List[Dict], model_name: str = "") -> List[Dict]:
    """For Qwen3 models: append /no_think to the last user message to disable
    chain-of-thought output. This is the official Qwen3 per-turn mechanism.
    Skips injection for distilled models that don't handle it properly.
    """
    if _should_skip_no_think(model_name):
        return messages
    messages = list(messages)  # shallow copy,  don't mutate caller's list
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            content = messages[i].get("content", "")
            if "/no_think" not in content:
                messages[i] = dict(messages[i])
                messages[i]["content"] = content + " /no_think"
            break
    return messages


# ── Token & cost tracking ────────────────────────────────────────────

# Approximate costs per 1M tokens (USD),  updated periodically
_COST_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "mistral-large": {"input": 2.00, "output": 6.00},
    "llama-3.1-70b": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b": {"input": 0.05, "output": 0.08},
}


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""
    estimated_cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)


class TokenTracker:
    """Cumulative token usage tracker across all LLM calls."""

    def __init__(self):
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.call_count: int = 0
        self.history: List[TokenUsage] = []

    def record(self, usage: TokenUsage):
        """Record a single LLM call's token usage."""
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost_usd += usage.estimated_cost_usd
        self.call_count += 1
        self.history.append(usage)
        # Cap history to last 10 000 entries to avoid unbounded memory growth
        if len(self.history) > 10_000:
            self.history = self.history[-10_000:]
        logger.debug(
            f"Tokens: +{usage.total_tokens} ({usage.prompt_tokens}in/{usage.completion_tokens}out) "
            f"| Total: {self.total_tokens} | Cost: ${self.total_cost_usd:.6f}"
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable summary of current usage with time breakdowns."""
        now = time.time()
        h1 = now - 3600
        h24 = now - 86400

        # Time-window aggregations
        tokens_1h = 0
        tokens_24h = 0
        cost_1h = 0.0
        cost_24h = 0.0
        calls_1h = 0
        calls_24h = 0
        by_model: Dict[str, Dict[str, Any]] = {}

        for u in self.history:
            # Per-model breakdown
            m = u.model or "unknown"
            if m not in by_model:
                by_model[m] = {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0, "calls": 0}
            by_model[m]["prompt"] += u.prompt_tokens
            by_model[m]["completion"] += u.completion_tokens
            by_model[m]["total"] += u.total_tokens
            by_model[m]["cost"] += u.estimated_cost_usd
            by_model[m]["calls"] += 1

            # Time windows
            if u.timestamp >= h24:
                tokens_24h += u.total_tokens
                cost_24h += u.estimated_cost_usd
                calls_24h += 1
            if u.timestamp >= h1:
                tokens_1h += u.total_tokens
                cost_1h += u.estimated_cost_usd
                calls_1h += 1

        # Round costs in per-model
        for v in by_model.values():
            v["cost"] = round(v["cost"], 6)

        # Compute tokens-per-minute rate (last hour)
        elapsed_min = max((now - self.history[0].timestamp) / 60, 1) if self.history else 1
        tokens_per_min = round(self.total_tokens / elapsed_min, 1)

        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "call_count": self.call_count,
            "tokens_1h": tokens_1h,
            "tokens_24h": tokens_24h,
            "cost_1h": round(cost_1h, 6),
            "cost_24h": round(cost_24h, 6),
            "calls_1h": calls_1h,
            "calls_24h": calls_24h,
            "tokens_per_min": tokens_per_min,
            "by_model": by_model,
        }

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        self.history.clear()

    @staticmethod
    def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD for a given model and token counts."""
        # Try exact match, then prefix match
        costs = _COST_PER_1M.get(model)
        if not costs:
            for key, val in _COST_PER_1M.items():
                if key in model or model in key:
                    costs = val
                    break
        if not costs:
            return 0.0  # Local model or unknown,  free
        return (prompt_tokens * costs["input"] + completion_tokens * costs["output"]) / 1_000_000


# Model capabilities database (using actually available Ollama models)
MODEL_CAPABILITIES = {
    # High-end GPU models (20GB+ VRAM)
    "llama3.1:70b": {"reasoning": 9, "vision": 0, "tools": 8, "speed": 4, "vram": 24},
    "qwen2.5:72b": {"reasoning": 9, "vision": 0, "tools": 8, "speed": 4, "vram": 24},
    "mixtral:8x7b": {"reasoning": 8, "vision": 0, "tools": 7, "speed": 5, "vram": 24},
    # Mid-range GPU models (8GB+ VRAM)
    "llama3.1:8b": {"reasoning": 7, "vision": 0, "tools": 7, "speed": 8, "vram": 5},
    "qwen2.5:7b": {"reasoning": 7, "vision": 0, "tools": 7, "speed": 8, "vram": 4},
    "gemma2:9b": {"reasoning": 7, "vision": 0, "tools": 6, "speed": 8, "vram": 6},
    "phi3:14b": {"reasoning": 8, "vision": 0, "tools": 7, "speed": 6, "vram": 8},
    "mistral:7b": {"reasoning": 7, "vision": 0, "tools": 6, "speed": 8, "vram": 4},
    # Gemma 3 models
    "gemma3:latest": {"reasoning": 7, "vision": 0, "tools": 5, "speed": 7, "vram": 5},
    "gemma3:4b": {"reasoning": 6, "vision": 0, "tools": 4, "speed": 8, "vram": 2},
    "gemma3:12b": {"reasoning": 7, "vision": 0, "tools": 5, "speed": 7, "vram": 8},
    "gemma3:27b": {"reasoning": 8, "vision": 0, "tools": 6, "speed": 5, "vram": 16},
    # CPU/Low RAM models
    "llama3.2:3b": {"reasoning": 6, "vision": 0, "tools": 6, "speed": 9, "vram": 0},
    "gemma2:2b": {"reasoning": 5, "vision": 0, "tools": 5, "speed": 9, "vram": 0},
    "qwen2.5:3b": {"reasoning": 6, "vision": 0, "tools": 6, "speed": 9, "vram": 0},
    "phi3:3.8b": {"reasoning": 6, "vision": 0, "tools": 6, "speed": 9, "vram": 0},
    "llama3.2:1b": {"reasoning": 4, "vision": 0, "tools": 4, "speed": 10, "vram": 0},
    "qwen2.5:0.5b": {"reasoning": 3, "vision": 0, "tools": 3, "speed": 10, "vram": 0},
}


def _normalize_model_list(models_obj) -> List[str]:
    """Normalize Ollama list() response into model name strings."""
    result: List[str] = []
    try:
        for m in models_obj.get("models", []):
            name = getattr(m, "model", None) or m.get("name") or m.get("model", "")
            if name:
                result.append(name)
    except Exception:
        pass
    return result


def _extract_model_size_b(model_name: str) -> float:
    """Extract model size in billions from names like 14b, 0.8b, 72b."""
    if not model_name:
        return 999.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", model_name.lower())
    if not m:
        return 999.0
    try:
        return float(m.group(1))
    except Exception:
        return 999.0


def _select_low_vram_model(available: List[str], preferred: str = "") -> Optional[str]:
    """Pick best low-VRAM model from locally available Ollama models."""
    if not available:
        return None

    # Explicit preference first (if installed)
    if preferred:
        for m in available:
            if preferred in m or m in preferred:
                return m

    # Curated low-vram priorities
    low_vram_priority = [
        "qwen3.5:0.8b",
        "qwen3:0.6b",
        "qwen2.5:0.5b",
        "llama3.2:1b",
        "llama3.2:3b",
        "qwen2.5:3b",
        "gemma2:2b",
    ]
    for pref in low_vram_priority:
        for m in available:
            if pref in m or m in pref:
                return m

    # Fallback: smallest non-embedding model installed
    candidates = [m for m in available if "embed" not in m.lower()]
    if not candidates:
        return None
    candidates.sort(key=_extract_model_size_b)
    return candidates[0]


def _resolve_start_model(
    *,
    config,
    available: List[str],
    auto_recommended: str,
) -> str:
    """Resolve initial model with full low-VRAM aware fallback logic."""
    default_model = getattr(config, "default_model", "qwen3.5:0.8b")
    low_vram_mode = bool(getattr(config, "low_vram_mode", False))
    preferred_low = getattr(config, "low_vram_preferred_model", "qwen3.5:0.8b")

    # 1) Low-VRAM mode has priority and always stays local/installed
    if low_vram_mode:
        low = _select_low_vram_model(available, preferred_low)
        if low:
            logger.info(f"Low-VRAM mode active. Selected: {low}")
            return low
        logger.warning(
            "Low-VRAM mode active but no suitable local low-vram model found. "
            f"Falling back to default: {default_model}"
        )

    # 2) If auto-selected recommendation exists and is installed, use it
    if auto_recommended:
        for a in available:
            if auto_recommended in a or a in auto_recommended:
                return a

    # 3) If configured default exists locally, use it
    for a in available:
        if default_model in a or a in default_model:
            return a

    # 4) Smart fallback: prefer qwen3.5:0.8b if present, then smallest model
    low = _select_low_vram_model(available, "qwen3.5:0.8b")
    if low:
        logger.warning(
            f"Configured default '{default_model}' not installed. "
            f"Using local fallback: {low}"
        )
        return low

    # 5) Last resort (should rarely happen)
    logger.warning(
        f"No known fallback found in installed models. Using configured default: {default_model}"
    )
    return default_model


class AdaptiveLLM:
    """LLM that can switch models based on task requirements"""

    # Shared across instances,  remember which models don't support native tool calling
    _MODELS_WITHOUT_NATIVE_TOOLS: set = set()

    def __init__(self, config, initial_model: str):
        self.config = config
        self.current_model = initial_model
        self.base_url = config.ollama_base_url
        self.available_models = []
        self.token_tracker = TokenTracker()
        self._update_available_models()

    def _create_llm(self, model: str):
        """Store model name (no external LLM wrapper needed)."""
        self.current_model = model
        return model

    def _update_available_models(self):
        """Update list of available local models"""
        try:
            client = ollama.Client(host=self.base_url)
            models = client.list()
            self.available_models = [
                m.get("name") or m.get("model") or getattr(m, "model", "")
                for m in models.get("models", [])
            ]
            logger.info(f"Available models: {', '.join(self.available_models)}")
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
            self.available_models = [self.current_model]

    async def auto_switch_model(self, task_type: str) -> bool:
        """
        Automatically switch to best model for task type
        task_type: 'vision', 'reasoning', 'tools', 'general'
        Returns True if switched, False if kept current
        """
        # Determine requirements based on task
        requirements = {
            "vision": {"vision": 7, "tools": 5},
            "reasoning": {"reasoning": 8, "tools": 6},
            "tools": {"tools": 7, "reasoning": 6},
            "general": {"reasoning": 6, "tools": 5, "speed": 7},
        }

        req = requirements.get(task_type, requirements["general"])

        # Find best model that meets requirements
        best_model = None
        best_score = -1

        low_vram_mode = bool(getattr(self.config, "low_vram_mode", False))
        low_vram_max_b = float(getattr(self.config, "low_vram_max_model_b", 3.0))

        for model in MODEL_CAPABILITIES:
            caps = MODEL_CAPABILITIES[model]

            # In low-vram mode, never switch above configured size cap
            if low_vram_mode and _extract_model_size_b(model) > low_vram_max_b:
                continue

            # Check if requirements are met
            meets_req = all(caps.get(k, 0) >= v for k, v in req.items())
            if not meets_req:
                continue

            # Calculate score (prefer speed if tied)
            score = sum(caps.get(k, 0) for k in req.keys()) + caps.get("speed", 0) * 0.1

            if score > best_score:
                best_score = score
                best_model = model

        if best_model and best_model != self.current_model:
            # Block downloading large models (70b+)
            if "70b" in best_model.lower() or "405b" in best_model.lower():
                logger.warning(
                    f"Blocked download of large model {best_model}. Using {self.current_model} instead."
                )
                return False

            # Check if model is available, if not pull it
            if best_model not in self.available_models:
                logger.info(f"Model {best_model} not available, pulling...")
                await self._pull_model(best_model)

            # Switch model
            logger.info(f"Switching from {self.current_model} to {best_model} for {task_type} task")
            self.current_model = best_model
            return True

        return False

    async def _pull_model(self, model_name: str):
        """Download model from Ollama"""
        try:
            client = ollama.Client(host=self.base_url)
            logger.info(f"Downloading {model_name}...")
            client.pull(model_name)
            self.available_models.append(model_name)
            logger.info(f"Model {model_name} downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to pull {model_name}: {e}")
            raise

    async def ainvoke(self, messages):
        """Invoke current LLM (pure ollama, no langchain)."""
        client = ollama.AsyncClient(host=self.base_url)
        plain_msgs = []
        for m in messages:
            role = getattr(m, "type", None) or (m["role"] if isinstance(m, dict) else "user")
            content = getattr(m, "content", None) or (m["content"] if isinstance(m, dict) else str(m))
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            plain_msgs.append({"role": role, "content": content})
        async with _ollama_lock():
            resp = await client.chat(model=self.current_model, messages=plain_msgs)
        from types import SimpleNamespace
        return SimpleNamespace(content=resp.get("message", {}).get("content", ""))

    def invoke(self, messages):
        """Sync invoke"""
        import asyncio
        return asyncio.run(self.ainvoke(messages))

    async def _invoke_with_text_tool_calling(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        """
        Text-based tool calling for models that don't support native Ollama tools API.
        Injects tool schemas into the system prompt and parses <tool_call> JSON blocks.
        Qwen3 understands this format natively.
        """
        # Filter: only include tools that have actual parameters (full schemas).
        # Compact stub schemas (name + description only) are useless in text
        # mode and overwhelm small models.  Cap at 30 tools max.
        _MAX_TEXT_TOOLS = 30
        full_tools = []
        compact_tools = []
        for t in tools:
            fn = t.get("function", t)
            params = fn.get("parameters", {})
            if params.get("properties"):
                full_tools.append(t)
            else:
                compact_tools.append(t)

        # Prefer full-schema tools; if we have room, add compact ones
        selected = full_tools[:_MAX_TEXT_TOOLS]
        remaining = _MAX_TEXT_TOOLS - len(selected)
        if remaining > 0:
            selected.extend(compact_tools[:remaining])

        # Build compact tool schema list
        tool_schemas = []
        for t in selected:
            fn = t.get("function", t)
            schema = {
                "name": fn.get("name", ""),
                "description": fn.get("description", "")[:200],
            }
            params = fn.get("parameters", {})
            if params.get("properties"):
                schema["parameters"] = {k: {"type": v.get("type", "string"), "description": v.get("description", "")[:80]}
                                         for k, v in params["properties"].items()}
            tool_schemas.append(schema)

        logger.info(f"📝 Text-based tool calling: {len(tool_schemas)} tools (from {len(tools)} total)")

        tools_json = json.dumps(tool_schemas, indent=2)
        tool_instruction = (
            "TOOL USE: To call a tool output EXACTLY this format and nothing else:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"arg": "value"}}\n'
            "</tool_call>\n\n"
            f"Available tools:\n{tools_json}\n\n"
            "Rules: If you need a tool, output the <tool_call> block first and stop. "
            "Never describe what you are about to do,  just output the tool call or the final answer."
        )

        # Inject tool instruction into system message
        new_messages = []
        system_injected = False
        for m in messages:
            if m.get("role") == "system" and not system_injected:
                new_messages.append({"role": "system", "content": m["content"] + "\n\n" + tool_instruction})
                system_injected = True
            else:
                new_messages.append(m)
        if not system_injected:
            new_messages.insert(0, {"role": "system", "content": tool_instruction})

        _skip_no_think = _should_skip_no_think(self.current_model)
        if not _skip_no_think:
            new_messages = _inject_no_think(new_messages, self.current_model)
        client = ollama.AsyncClient(host=self.base_url)
        _chat_kwargs = {"model": self.current_model, "messages": new_messages}
        _chat_kwargs["options"] = {"repeat_penalty": 1.3, "num_predict": 2048}
        if not _skip_no_think:
            _chat_kwargs["think"] = False
        try:
            async with _ollama_lock():
                resp = await asyncio.wait_for(client.chat(**_chat_kwargs), timeout=120)
        except asyncio.TimeoutError:
            logger.warning("Text-based tool calling timed out after 120s,  returning empty")
            return {"tool_call": None, "tool_calls": [], "text": ""}
        except Exception as _tbe:
            # If think:false is rejected, retry without it
            if "think" in str(_tbe).lower() or "400" in str(_tbe):
                logger.warning(f"Text-based fallback error ({_tbe}), retrying without think param")
                _chat_kwargs.pop("think", None)
                async with _ollama_lock():
                    resp = await asyncio.wait_for(client.chat(**_chat_kwargs), timeout=120)
            else:
                raise
        raw_text = resp.get("message", {}).get("content", "")
        thinking_field = resp.get("message", {}).get("thinking", "") or ""
        clean_text, reasoning_from_tags = parse_thinking(raw_text)
        reasoning = (thinking_field.strip() or reasoning_from_tags) or None

        # If content is empty but reasoning exists, the model put its
        # answer inside <think> tags,  extract usable text from reasoning.
        if not clean_text.strip() and reasoning:
            logger.info("[LLM] Content empty after think-stripping, using reasoning as response")
            clean_text = reasoning

        # Parse <tool_call>...</tool_call> blocks
        tool_call_re = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        matches = tool_call_re.findall(clean_text)
        if matches:
            parsed_calls = []
            for raw_call in matches:
                try:
                    call_data = json.loads(raw_call)
                    name = call_data.get("name", "")
                    args = call_data.get("arguments", call_data.get("parameters", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if name:
                        parsed_calls.append({"name": name, "arguments": args})
                except json.JSONDecodeError:
                    continue
            if parsed_calls:
                logger.info(f"🔧 Text-based tool call parsed: {[c['name'] for c in parsed_calls]}")
                return {
                    "tool_call": parsed_calls[0],
                    "tool_calls": parsed_calls,
                    "text": None,
                }

        # No tool call found,  plain text answer
        result = {"tool_call": None, "tool_calls": [], "text": clean_text}
        if reasoning:
            result["reasoning"] = reasoning
        return result

    async def invoke_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        """
        Call Ollama with native tool calling (structured JSON output).

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            tools: List of Ollama tool schemas (OpenAI function calling format)

        Returns:
            Dict with keys:
              - "text": plain text response (if no tool was called)
              - "tool_call": {"name": str, "arguments": dict} if a tool was invoked
        """
        # Fast-path: skip native API for models known not to support it
        if tools and self.current_model in self._MODELS_WITHOUT_NATIVE_TOOLS:
            return await self._invoke_with_text_tool_calling(messages, tools)

        try:
            client = ollama.AsyncClient(host=self.base_url)
            # /no_think injected into last user message: official Qwen3 mechanism to
            # suppress chain-of-thought output while keeping full response quality.
            _skip_no_think = _should_skip_no_think(self.current_model)
            if not _skip_no_think:
                messages = _inject_no_think(messages, self.current_model)
            _chat_kwargs = {
                "model": self.current_model,
                "messages": messages,
            }
            _chat_kwargs["options"] = {"repeat_penalty": 1.3, "num_predict": 4096}
            # Only include tools key when there are actual tools,  some models
            # reject even an empty list with HTTP 400.
            if tools:
                _chat_kwargs["tools"] = tools
            if not _skip_no_think:
                _chat_kwargs["think"] = False
            async with _ollama_lock():
                response = await asyncio.wait_for(
                    client.chat(**_chat_kwargs),
                    timeout=120,
                )
            msg = response.get("message", {})

            # Track tokens (Ollama provides prompt_eval_count / eval_count)
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
            self.token_tracker.record(TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=self.current_model,
                provider="ollama",
                estimated_cost_usd=0.0,  # local model,  free
            ))

            # Tool call path,  collect ALL tool calls for parallel execution
            raw_calls = msg.get("tool_calls") or []
            if raw_calls:
                parsed: list[dict] = []
                for call in raw_calls:
                    fn = call.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    parsed.append({"name": fn.get("name", ""), "arguments": args})

                return {
                    "tool_call": parsed[0],  # backward compat
                    "tool_calls": parsed,  # NEW: all tool calls
                    "text": None,
                }

            # Plain text path
            # Ollama places reasoning tokens in msg.thinking (separate from content)
            # when the model supports thinking. Just use content directly.
            raw_text = msg.get("content", "")
            thinking_field = msg.get("thinking", "") or ""
            # Also strip any <think> tags that leaked into content
            clean_text, reasoning_from_tags = parse_thinking(raw_text)
            reasoning = (thinking_field.strip() or reasoning_from_tags) or None
            # If content is empty but reasoning exists, the model put its
            # answer inside <think> tags,  extract usable text from reasoning.
            if not clean_text.strip() and reasoning:
                logger.info("[LLM] Content empty after think-stripping, using reasoning as response")
                clean_text = reasoning
            result = {"tool_call": None, "tool_calls": [], "text": clean_text}
            if reasoning:
                result["reasoning"] = reasoning
                logger.info(f"💭 Reasoning captured ({len(reasoning)} chars)")
            return result

        except Exception as e:
            error_str = str(e)
            # Detect models that don't support native tool calling (Ollama returns 400)
            if tools and ("400" in error_str or "does not support tools" in error_str
                          or "tool" in error_str.lower()):
                logger.warning(f"Tool calling failed ({e}), switching to text-based tool calling")
                AdaptiveLLM._MODELS_WITHOUT_NATIVE_TOOLS.add(self.current_model)
                try:
                    return await self._invoke_with_text_tool_calling(messages, tools)
                except Exception as e2:
                    logger.error(f"Text-based tool calling failed: {e2}")
            else:
                logger.warning(f"Tool calling failed ({e}), falling back to plain chat")

            # Last resort: plain chat without tools
            try:
                client = ollama.AsyncClient(host=self.base_url)
                plain_msgs = []
                for m in messages:
                    role = m.get("role", "user")
                    plain_msgs.append({"role": role, "content": m.get("content", "")})
                _skip_nt = _should_skip_no_think(self.current_model)
                if not _skip_nt:
                    plain_msgs = _inject_no_think(plain_msgs, self.current_model)
                _chat_kw = {"model": self.current_model, "messages": plain_msgs}
                _chat_kw["options"] = {"repeat_penalty": 1.3, "num_predict": 2048}
                if not _skip_nt:
                    _chat_kw["think"] = False
                async with _ollama_lock():
                    resp = await client.chat(**_chat_kw)
                raw_text = resp.get("message", {}).get("content", "")
                thinking_field = resp.get("message", {}).get("thinking", "") or ""
                clean_text, reasoning_from_tags = parse_thinking(raw_text)
                reasoning = (thinking_field.strip() or reasoning_from_tags) or None
                if not clean_text.strip() and reasoning:
                    logger.info("[LLM] Content empty after think-stripping, using reasoning as response")
                    clean_text = reasoning
                result = {"tool_call": None, "tool_calls": [], "text": clean_text}
                if reasoning:
                    result["reasoning"] = reasoning
                    logger.info(f"💭 Reasoning captured ({len(reasoning)} chars)")
                return result
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return {"tool_call": None, "tool_calls": [], "text": f"Error: {e}"}

    async def plain_chat(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Simple chat completion *without* tools,  used for the no-tools fast-path.

        Unlike ``invoke_with_tools(msgs, [])``, this never sends a ``tools``
        key at all, which avoids 400 errors from models that reject even an
        empty tools list.  Think-stripping and token-tracking are still applied.

        Returns:
            Dict with keys ``text`` (str), ``tool_call`` (None), ``tool_calls`` ([]).
        """
        client = ollama.AsyncClient(host=self.base_url)
        _skip_nt = _should_skip_no_think(self.current_model)
        if not _skip_nt:
            messages = _inject_no_think(messages, self.current_model)
        _chat_kw: dict = {"model": self.current_model, "messages": messages}
        _chat_kw["options"] = {"repeat_penalty": 1.3, "num_predict": 2048}
        if not _skip_nt:
            _chat_kw["think"] = False
        async with _ollama_lock():
            resp = await asyncio.wait_for(
                client.chat(**_chat_kw),
                timeout=120,
            )
        # Token tracking
        prompt_tokens = resp.get("prompt_eval_count", 0)
        completion_tokens = resp.get("eval_count", 0)
        self.token_tracker.record(TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=self.current_model,
            provider="ollama",
            estimated_cost_usd=0.0,
        ))
        raw_text = resp.get("message", {}).get("content", "")
        thinking_field = resp.get("message", {}).get("thinking", "") or ""
        clean_text, reasoning_from_tags = parse_thinking(raw_text)
        reasoning = (thinking_field.strip() or reasoning_from_tags) or None
        if not clean_text.strip() and reasoning:
            logger.info("[LLM] Content empty after think-stripping, using reasoning as response")
            clean_text = reasoning
        result: Dict[str, Any] = {"tool_call": None, "tool_calls": [], "text": clean_text}
        if reasoning:
            result["reasoning"] = reasoning
        return result

    async def astream(self, messages: List[Dict]) -> AsyncIterator[str]:
        """
        Yield text tokens one-by-one from Ollama streaming API.

        DeepSeek models: <think>...</think> blocks are silently consumed
        and emitted as a single ``[REASONING]`` event at the end.

        Usage:
            async for token in llm.astream(messages):
                print(token, end="", flush=True)
        """
        try:
            client = ollama.AsyncClient(host=self.base_url)
            _skip_nt = _should_skip_no_think(self.current_model)
            if not _skip_nt:
                messages = _inject_no_think(messages, self.current_model)
            # Acquire Ollama lock for the entire streaming duration
            self._stream_lock_cm = _ollama_lock()
            await self._stream_lock_cm.__aenter__()
            _stream_kw = {"model": self.current_model, "messages": messages, "stream": True}
            _stream_kw["options"] = {"repeat_penalty": 1.3, "num_predict": 2048}
            if not _skip_nt:
                _stream_kw["think"] = False
            try:
                stream = await client.chat(**_stream_kw)
            except Exception:
                await self._stream_lock_cm.__aexit__(None, None, None)
                raise

            # State machine for <think> block detection
            _buf = ""        # accumulates partial tag matches
            _inside = False  # currently inside <think>...</think>
            _reasoning = []  # collected reasoning lines
            _check_tags = "deepseek" in self.current_model.lower()

            async for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if not token:
                    continue

                if not _check_tags:
                    yield token
                    continue

                # Tag-aware streaming
                _buf += token
                while _buf:
                    if _inside:
                        end_idx = _buf.find("</think>")
                        if end_idx != -1:
                            _reasoning.append(_buf[:end_idx])
                            _buf = _buf[end_idx + 8:]  # skip </think>
                            _inside = False
                        else:
                            # Still inside thinking,  buffer everything
                            _reasoning.append(_buf)
                            _buf = ""
                    else:
                        start_idx = _buf.find("<think>")
                        if start_idx != -1:
                            # Yield text before the tag
                            if start_idx > 0:
                                yield _buf[:start_idx]
                            _buf = _buf[start_idx + 7:]  # skip <think>
                            _inside = True
                        else:
                            # Check for partial tag at end of buffer
                            safe = True
                            for i in range(1, min(len("<think>"), len(_buf) + 1)):
                                if _buf.endswith("<think>"[:i]):
                                    # Might be start of a tag,  hold back
                                    yield _buf[:-i]
                                    _buf = _buf[-i:]
                                    safe = False
                                    break
                            if safe:
                                yield _buf
                                _buf = ""

            # Flush remaining buffer
            if _buf:
                yield _buf

            # Emit reasoning summary if captured
            if _reasoning:
                reasoning_text = "".join(_reasoning).strip()
                if reasoning_text:
                    logger.info(f"💭 DeepSeek reasoning streamed ({len(reasoning_text)} chars)")
                    yield f"\n\n---\n💭 **Reasoning** ({len(reasoning_text)} chars captured)\n"

            # Release Ollama lock after streaming completes
            await self._stream_lock_cm.__aexit__(None, None, None)

        except Exception as e:
            # Release lock on error too
            if hasattr(self, '_stream_lock_cm'):
                try:
                    await self._stream_lock_cm.__aexit__(None, None, None)
                except Exception:
                    pass
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {e}"


def get_llm(config):
    """Get LLM instance based on configuration"""

    # ── First, check if Ollama is actually reachable ──────────────────────
    ollama_available = False
    ollama_url = getattr(config, "ollama_base_url", "") or ""
    if ollama_url:
        try:
            client = ollama.Client(host=ollama_url)
            models = client.list()
            ollama_available = True
            logger.info(f"Ollama is reachable at {ollama_url}")
        except Exception as e:
            logger.info(f"Ollama not reachable at {ollama_url}: {e}")
    else:
        logger.info("OLLAMA_BASE_URL is empty — skipping local Ollama")

    if ollama_available:
        try:
            # Auto-select model if enabled
            model_to_use = config.default_model
            available = _normalize_model_list(models)
            recommended = ""

            if config.auto_select_model:
                from opensable.core.system_detector import auto_configure_system

                auto_config = auto_configure_system()
                recommended = auto_config["recommended_model"]

                # Verify the recommended model is actually available before using it
                try:
                    if any(recommended in a or a in recommended for a in available):
                        model_to_use = _resolve_start_model(
                            config=config,
                            available=available,
                            auto_recommended=recommended,
                        )
                        logger.info(
                            f"Auto-selected model: {model_to_use} (tier: {auto_config['device_tier']})"
                        )
                    else:
                        logger.warning(
                            f"Auto-selected model '{recommended}' not available locally. Resolving best local fallback..."
                        )
                        model_to_use = _resolve_start_model(
                            config=config,
                            available=available,
                            auto_recommended="",
                        )
                except Exception:
                    logger.warning("Cannot verify model availability. Falling back to resolver.")
                    model_to_use = _resolve_start_model(
                        config=config,
                        available=available,
                        auto_recommended="",
                    )
            else:
                model_to_use = _resolve_start_model(
                    config=config,
                    available=available,
                    auto_recommended="",
                )

            # Ensure selected local model exists; pull automatically if missing
            if not any(model_to_use in a or a in model_to_use for a in available):
                logger.info(f"Selected model '{model_to_use}' not installed. Pulling from Ollama...")
                try:
                    client.pull(model_to_use)
                    logger.info(f"Model '{model_to_use}' downloaded successfully")
                except Exception as pull_err:
                    logger.warning(f"Failed to pull '{model_to_use}': {pull_err}")

            # Return adaptive LLM that can switch models
            adaptive_llm = AdaptiveLLM(config, model_to_use)
            logger.info(f"Using adaptive LLM (Ollama) starting with: {model_to_use}")
            return adaptive_llm

        except Exception as e:
            logger.warning(f"Failed to initialize Ollama LLM: {e}")

    # ── Ollama not available — fall back to cloud providers ───────────────
    logger.info("Looking for cloud/remote LLM providers...")
    _CLOUD_FALLBACK_ORDER = [
        ("openwebui", "openwebui_api_key"),
        ("openai", "openai_api_key"),
        ("anthropic", "anthropic_api_key"),
        ("gemini", "gemini_api_key"),
        ("deepseek", "deepseek_api_key"),
        ("groq", "groq_api_key"),
        ("mistral", "mistral_api_key"),
        ("together", "together_api_key"),
        ("xai", "xai_api_key"),
        ("cohere", "cohere_api_key"),
        ("kimi", "kimi_api_key"),
        ("qwen", "qwen_api_key"),
        ("openrouter", "openrouter_api_key"),
    ]
    for provider, key_attr in _CLOUD_FALLBACK_ORDER:
        if getattr(config, key_attr, None):
            logger.info(f"Using {provider} as LLM provider")
            return CloudLLM(provider=provider, config=config)

    raise Exception(
        "No LLM available. Install Ollama or set an API key env var: "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, "
        "DEEPSEEK_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, "
        "TOGETHER_API_KEY, XAI_API_KEY, COHERE_API_KEY, "
        "KIMI_API_KEY, QWEN_API_KEY, OPENROUTER_API_KEY, "
        "OPENWEBUI_API_KEY + OPENWEBUI_API_URL"
    )


class CloudLLM:
    """Cloud LLM provider with full tool calling parity.

    Supported providers:
      OpenAI-compatible : openai, deepseek, groq, together, xai, mistral,
                          kimi, qwen, openrouter
      Native SDK        : anthropic, gemini, cohere

    Implements the same interface as AdaptiveLLM so the agent loop can use
    it transparently: invoke_with_tools(), ainvoke(), current_model, etc.
    """

    # Provider → (base_url | None, config key name, default model)
    _PROVIDERS = {
        # OpenAI-compatible providers (reuse openai SDK)
        "openai": (None, "openai_api_key", "gpt-4o-mini"),
        "deepseek": ("https://api.deepseek.com", "deepseek_api_key", "deepseek-chat"),
        "groq": (
            "https://api.groq.com/openai/v1",
            "groq_api_key",
            "llama-3.3-70b-versatile",
        ),
        "together": (
            "https://api.together.xyz/v1",
            "together_api_key",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        ),
        "xai": ("https://api.x.ai/v1", "xai_api_key", "grok-3-mini-fast"),
        "mistral": (
            "https://api.mistral.ai/v1",
            "mistral_api_key",
            "mistral-small-latest",
        ),
        "kimi": (
            "https://api.moonshot.cn/v1",
            "kimi_api_key",
            "moonshot-v1-8k",
        ),
        "qwen": (
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen_api_key",
            "qwen-plus",
        ),
        "openrouter": (
            "https://openrouter.ai/api/v1",
            "openrouter_api_key",
            "openai/gpt-4o-mini",
        ),
        "openwebui": (
            None,  # user sets OPENWEBUI_API_URL in .env
            "openwebui_api_key",
            "llama3.2:latest",  # user sets OPENWEBUI_MODEL in .env
        ),
        # Native SDK providers
        "anthropic": (None, "anthropic_api_key", "claude-sonnet-4-20250514"),
        "gemini": (None, "gemini_api_key", "gemini-2.5-flash"),
        "cohere": (None, "cohere_api_key", "command-a-03-2025"),
    }

    def __init__(self, provider: str, config):
        if provider not in self._PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. " f"Supported: {', '.join(self._PROVIDERS)}"
            )
        self.provider = provider
        self.config = config
        _, _, default_model = self._PROVIDERS[provider]
        self.current_model = default_model
        self.available_models = [self.current_model]
        self._client = None
        self.token_tracker = TokenTracker()

        # ── Circuit breaker state (prevents hammering a stuck server) ──
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0        # epoch time
        self._backoff_base = 30               # seconds
        self._backoff_max = 300               # 5 minute max
        self._failure_threshold = 3           # open circuit after N failures

        # Open WebUI: user configures URL and model via .env
        if provider == "openwebui":
            custom_url = getattr(config, "openwebui_api_url", None)
            if not custom_url:
                raise ValueError(
                    "OPENWEBUI_API_URL is required for Open WebUI provider. "
                    "Set it in .env (e.g. https://your-server.com/api)"
                )
            # Ensure URL ends with OpenAI-compatible path
            # Open WebUI's OpenAI-compat endpoint lives at /api/chat/completions,
            # so the base_url the openai SDK needs is {host}/api.
            self._openwebui_base_url = custom_url.rstrip("/")
            if not self._openwebui_base_url.endswith(("/v1", "/api")):
                self._openwebui_base_url += "/api"
            custom_model = getattr(config, "openwebui_model", None)
            if custom_model:
                self.current_model = custom_model
                self.available_models = [custom_model]

    def _get_api_key(self) -> str:
        """Retrieve the API key for the current provider from config."""
        _, key_attr, _ = self._PROVIDERS[self.provider]
        key = getattr(self.config, key_attr, None)
        if not key:
            raise ValueError(
                f"No API key for {self.provider}. " f"Set {key_attr.upper()} environment variable."
            )
        return key

    # ── OpenAI-compatible providers ──────────────────────────────────────

    async def _openai_invoke(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        """Works for openai, deepseek, groq, together, xai, mistral, etc."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install openai ,  required for OpenAI-compatible providers")

        base_url, _, _ = self._PROVIDERS[self.provider]
        client = AsyncOpenAI(api_key=self._get_api_key(), base_url=base_url)

        kwargs: dict = {"model": self.current_model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        resp = await client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        # Track token usage
        usage = getattr(resp, "usage", None)
        if usage:
            pt = getattr(usage, "prompt_tokens", 0)
            ct = getattr(usage, "completion_tokens", 0)
            self.token_tracker.record(TokenUsage(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
                model=self.current_model,
                provider=self.provider,
                estimated_cost_usd=TokenTracker.estimate_cost(self.current_model, pt, ct),
            ))

        if msg.tool_calls:
            parsed = []
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                parsed.append({"name": tc.function.name, "arguments": args})
            return {
                "tool_call": parsed[0],
                "tool_calls": parsed,
                "text": None,
            }

        # Check for DeepSeek reasoning,  API may return reasoning_content or <think> tags
        raw_text = msg.content or ""
        reasoning = None

        # DeepSeek API returns reasoning_content as a separate field
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning = msg.reasoning_content
        else:
            raw_text, reasoning = parse_thinking(raw_text)

        result = {
            "tool_call": None,
            "tool_calls": [],
            "text": raw_text,
        }
        if reasoning:
            result["reasoning"] = reasoning
            logger.info(f"💭 DeepSeek reasoning captured ({len(reasoning)} chars)")
        return result

    # ── Anthropic ────────────────────────────────────────────────────────

    async def _anthropic_invoke(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("pip install anthropic ,  required for Anthropic provider")

        client = AsyncAnthropic(api_key=self._get_api_key())

        # Convert OpenAI-style tool schemas → Anthropic format
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", t)
            anthropic_tools.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )

        # Separate system message from conversation
        system_text = ""
        conv_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                conv_msgs.append({"role": m["role"], "content": m["content"]})

        kwargs: dict = {
            "model": self.current_model,
            "max_tokens": 4096,
            "messages": conv_msgs or [{"role": "user", "content": "Hello"}],
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        resp = await client.messages.create(**kwargs)

        text_parts = []
        parsed_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                parsed_calls.append({"name": block.name, "arguments": block.input or {}})

        if parsed_calls:
            return {
                "tool_call": parsed_calls[0],
                "tool_calls": parsed_calls,
                "text": None,
            }
        return {
            "tool_call": None,
            "tool_calls": [],
            "text": "\n".join(text_parts),
        }

    # ── Google Gemini ────────────────────────────────────────────────────

    async def _gemini_invoke(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("pip install google-genai ,  required for Gemini provider")

        client = genai.Client(api_key=self._get_api_key())

        # Build Gemini function declarations from OpenAI schemas
        fn_decls = []
        for t in tools:
            fn = t.get("function", t)
            fn_decls.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )

        gemini_tools = [types.Tool(function_declarations=fn_decls)] if fn_decls else []
        config = types.GenerateContentConfig(tools=gemini_tools or None)

        # Build Gemini contents from messages
        system_text = ""
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            elif m["role"] == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=m["content"])],
                    )
                )
            else:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=m["content"])],
                    )
                )

        if system_text.strip():
            config.system_instruction = system_text.strip()

        if not contents:
            contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]

        resp = await client.aio.models.generate_content(
            model=self.current_model,
            contents=contents,
            config=config,
        )

        # Parse function calls
        parsed_calls = []
        text_parts = []
        if resp.candidates:
            for part in resp.candidates[0].content.parts:
                if part.function_call:
                    parsed_calls.append(
                        {
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args or {}),
                        }
                    )
                elif part.text:
                    text_parts.append(part.text)

        if parsed_calls:
            return {
                "tool_call": parsed_calls[0],
                "tool_calls": parsed_calls,
                "text": None,
            }
        return {
            "tool_call": None,
            "tool_calls": [],
            "text": "\n".join(text_parts),
        }

    # ── Cohere ───────────────────────────────────────────────────────────

    async def _cohere_invoke(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        try:
            import cohere
        except ImportError:
            raise ImportError("pip install cohere ,  required for Cohere provider")

        co = cohere.AsyncClientV2(self._get_api_key())

        # Build Cohere-format tool list (same as OpenAI schema)
        cohere_tools = []
        for t in tools:
            fn = t.get("function", t)
            cohere_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get(
                            "parameters",
                            {"type": "object", "properties": {}},
                        ),
                    },
                }
            )

        # Cohere v2 uses the same message format
        cohere_msgs = []
        for m in messages:
            cohere_msgs.append({"role": m["role"], "content": m["content"]})

        kwargs: dict = {
            "model": self.current_model,
            "messages": cohere_msgs,
        }
        if cohere_tools:
            kwargs["tools"] = cohere_tools

        resp = await co.chat(**kwargs)

        parsed_calls = []
        if resp.message.tool_calls:
            for tc in resp.message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                parsed_calls.append({"name": tc.function.name, "arguments": args})

        if parsed_calls:
            return {
                "tool_call": parsed_calls[0],
                "tool_calls": parsed_calls,
                "text": None,
            }

        text = ""
        if resp.message.content:
            text = resp.message.content[0].text if resp.message.content else ""
        return {"tool_call": None, "tool_calls": [], "text": text}

    # ── Open WebUI (aiohttp — Cloudflare blocks openai SDK's httpx) ─────

    async def _openwebui_invoke(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        """Call Open WebUI via aiohttp (bypasses Cloudflare httpx block).

        Tools are intentionally NOT forwarded because most Open WebUI backends
        (Ollama, llama.cpp) reject the OpenAI ``tools`` parameter with 400.
        The agent loop already falls back to extracting tool calls from text.

        Circuit breaker: after repeated failures, short-circuits for a cooldown
        period to avoid hammering a stuck server.
        """
        import aiohttp

        # ── Circuit breaker check ──
        now = time.time()
        if self._consecutive_failures >= self._failure_threshold:
            if now < self._circuit_open_until:
                wait = int(self._circuit_open_until - now)
                raise RuntimeError(
                    f"OpenWebUI circuit breaker open — {self._consecutive_failures} "
                    f"consecutive failures. Retrying in {wait}s."
                )
            # Cooldown expired — allow a single probe
            logger.info("OpenWebUI circuit breaker: attempting probe request...")

        url = getattr(self, "_openwebui_base_url", "") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }
        body: dict = {"model": self.current_model, "messages": messages}
        # NOTE: tools deliberately omitted — Ollama/OWUI backends return 400

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body,
                                        timeout=aiohttp.ClientTimeout(total=90)) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        raise RuntimeError(f"OpenWebUI returned {resp.status}: {err[:300]}")
                    data = await resp.json()

            # ── Success: reset circuit breaker ──
            if self._consecutive_failures > 0:
                logger.info(f"OpenWebUI recovered after {self._consecutive_failures} failures")
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
        except Exception:
            # ── Failure: update circuit breaker ──
            self._consecutive_failures += 1
            backoff = min(
                self._backoff_base * (2 ** (self._consecutive_failures - self._failure_threshold)),
                self._backoff_max,
            )
            if self._consecutive_failures >= self._failure_threshold:
                self._circuit_open_until = time.time() + backoff
                logger.warning(
                    f"OpenWebUI circuit breaker: {self._consecutive_failures} failures, "
                    f"cooldown {backoff}s"
                )
            raise

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})

        # Token tracking
        usage = data.get("usage", {})
        if usage:
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            self.token_tracker.record(TokenUsage(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
                model=self.current_model,
                provider=self.provider,
                estimated_cost_usd=TokenTracker.estimate_cost(self.current_model, pt, ct),
            ))

        # Tool calls
        tool_calls_raw = msg.get("tool_calls")
        if tool_calls_raw:
            parsed = []
            for tc in tool_calls_raw:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                parsed.append({"name": fn.get("name", ""), "arguments": args})
            return {"tool_call": parsed[0], "tool_calls": parsed, "text": None}

        # Text response
        raw_text = msg.get("content", "") or ""
        raw_text, reasoning = parse_thinking(raw_text)
        result: dict = {"tool_call": None, "tool_calls": [], "text": raw_text}
        if reasoning:
            result["reasoning"] = reasoning
        return result

    # ── Public interface (same as AdaptiveLLM) ───────────────────────────

    # Map providers to their invoke method
    _OPENAI_COMPAT = {
        "openai",
        "deepseek",
        "groq",
        "together",
        "xai",
        "mistral",
        "kimi",
        "qwen",
        "openrouter",
        # NOTE: "openwebui" intentionally excluded — Cloudflare blocks the
        # openai SDK's httpx transport; we use aiohttp for OpenWebUI instead.
    }

    async def invoke_with_tools(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        """Call cloud LLM with native tool calling."""
        try:
            if self.provider == "openwebui":
                return await self._openwebui_invoke(messages, tools)
            elif self.provider in self._OPENAI_COMPAT:
                return await self._openai_invoke(messages, tools)
            elif self.provider == "anthropic":
                return await self._anthropic_invoke(messages, tools)
            elif self.provider == "gemini":
                return await self._gemini_invoke(messages, tools)
            elif self.provider == "cohere":
                return await self._cohere_invoke(messages, tools)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        except ImportError:
            raise
        except Exception as e:
            logger.error(f"Cloud LLM ({self.provider}) failed: {e}")
            return {"tool_call": None, "tool_calls": [], "text": f"Error: {e}"}

    async def plain_chat(self, messages: List[Dict]) -> Dict[str, Any]:
        """Simple chat without tools — used for the no-tools fast-path.

        Cloud providers handle an empty tools list fine, so we just
        call invoke_with_tools with no tools.
        """
        return await self.invoke_with_tools(messages, [])

    async def ainvoke(self, messages):
        """Plain text invoke (LangChain compat)."""
        result = await self.invoke_with_tools(
            [
                {
                    "role": m.type if hasattr(m, "type") else "user",
                    "content": m.content,
                }
                for m in messages
            ],
            [],
        )
        from types import SimpleNamespace

        return SimpleNamespace(content=result.get("text", ""))

    def invoke(self, messages):
        """Sync invoke,  uses asyncio.run for simple scripts."""
        import asyncio

        return asyncio.run(self.ainvoke(messages))

    async def auto_switch_model(self, task_type: str) -> bool:
        """Cloud models don't auto-switch (always use the configured one)."""
        return False

    async def astream(self, messages: List[Dict]) -> AsyncIterator[str]:
        """
        Yield text tokens from cloud LLM streaming.

        DeepSeek API: reasoning_content is captured from stream deltas;
        <think> tags in content are also handled as fallback.

        Supports OpenAI-compatible providers, Anthropic, and Open WebUI (via aiohttp).
        """
        try:
            if self.provider == "openwebui":
                # Open WebUI: use aiohttp SSE (Cloudflare blocks openai SDK's httpx)
                import aiohttp

                # Circuit breaker check (shared with _openwebui_invoke)
                now = time.time()
                if self._consecutive_failures >= self._failure_threshold:
                    if now < self._circuit_open_until:
                        wait = int(self._circuit_open_until - now)
                        yield f"Error: OpenWebUI circuit breaker open ({self._consecutive_failures} failures). Retrying in {wait}s."
                        return

                url = getattr(self, "_openwebui_base_url", "") + "/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self._get_api_key()}",
                    "Content-Type": "application/json",
                }
                body = {"model": self.current_model, "messages": messages, "stream": True}
                _buf = ""
                _inside_think = False
                _reasoning_parts: list[str] = []

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, json=body,
                                                timeout=aiohttp.ClientTimeout(total=180)) as resp:
                            if resp.status != 200:
                                err = await resp.text()
                                self._consecutive_failures += 1
                                yield f"Error: OpenWebUI returned {resp.status}"
                                return
                            # Reset circuit breaker on first successful byte
                            if self._consecutive_failures > 0:
                                logger.info(f"OpenWebUI stream recovered after {self._consecutive_failures} failures")
                            self._consecutive_failures = 0
                            self._circuit_open_until = 0.0
                            # Parse SSE stream
                            async for line in resp.content:
                                line = line.decode("utf-8", errors="replace").strip()
                                if not line.startswith("data: "):
                                    continue
                                payload = line[6:]
                                if payload == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue
                                choices = chunk.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if not content:
                                    continue

                                # Strip <think> tags from models that produce them
                                _buf += content
                                while _buf:
                                    if _inside_think:
                                        end_idx = _buf.find("</think>")
                                        if end_idx != -1:
                                            _reasoning_parts.append(_buf[:end_idx])
                                            _buf = _buf[end_idx + 8:]
                                            _inside_think = False
                                        else:
                                            _reasoning_parts.append(_buf)
                                            _buf = ""
                                    else:
                                        start_idx = _buf.find("<think>")
                                        if start_idx != -1:
                                            if start_idx > 0:
                                                yield _buf[:start_idx]
                                            _buf = _buf[start_idx + 7:]
                                            _inside_think = True
                                        else:
                                            safe = True
                                            for i in range(1, min(len("<think>"), len(_buf) + 1)):
                                                if _buf.endswith("<think>"[:i]):
                                                    yield _buf[:-i]
                                                    _buf = _buf[-i:]
                                                    safe = False
                                                    break
                                            if safe:
                                                yield _buf
                                                _buf = ""
                    # Flush remaining buffer
                    if _buf:
                        yield _buf
                except Exception as exc:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._failure_threshold:
                        backoff = min(
                            self._backoff_base * (2 ** (self._consecutive_failures - self._failure_threshold)),
                            self._backoff_max,
                        )
                        self._circuit_open_until = time.time() + backoff
                        logger.warning(
                            f"OpenWebUI stream circuit breaker: {self._consecutive_failures} failures, "
                            f"cooldown {backoff}s"
                        )
                    yield f"Error: {exc}"
                    return

            elif self.provider in self._OPENAI_COMPAT:
                from openai import AsyncOpenAI
                base_url, _, _ = self._PROVIDERS[self.provider]
                client = AsyncOpenAI(api_key=self._get_api_key(), base_url=base_url)
                stream = await client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    stream=True,
                )

                _is_deepseek = self.provider == "deepseek" or "deepseek" in self.current_model.lower()
                _reasoning_parts: list[str] = []
                _buf = ""
                _inside_think = False

                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # DeepSeek API: reasoning_content comes as a separate field
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        _reasoning_parts.append(rc)
                        continue

                    if not delta.content:
                        continue

                    if not _is_deepseek:
                        yield delta.content
                        continue

                    # Fallback: parse <think> tags from content stream
                    _buf += delta.content
                    while _buf:
                        if _inside_think:
                            end_idx = _buf.find("</think>")
                            if end_idx != -1:
                                _reasoning_parts.append(_buf[:end_idx])
                                _buf = _buf[end_idx + 8:]
                                _inside_think = False
                            else:
                                _reasoning_parts.append(_buf)
                                _buf = ""
                        else:
                            start_idx = _buf.find("<think>")
                            if start_idx != -1:
                                if start_idx > 0:
                                    yield _buf[:start_idx]
                                _buf = _buf[start_idx + 7:]
                                _inside_think = True
                            else:
                                safe = True
                                for i in range(1, min(len("<think>"), len(_buf) + 1)):
                                    if _buf.endswith("<think>"[:i]):
                                        yield _buf[:-i]
                                        _buf = _buf[-i:]
                                        safe = False
                                        break
                                if safe:
                                    yield _buf
                                    _buf = ""

                # Flush
                if _buf:
                    yield _buf
                if _reasoning_parts:
                    reasoning_text = "".join(_reasoning_parts).strip()
                    if reasoning_text:
                        logger.info(f"💭 DeepSeek reasoning streamed ({len(reasoning_text)} chars)")
                        yield f"\n\n---\n💭 **Reasoning** ({len(reasoning_text)} chars captured)\n"

            elif self.provider == "anthropic":
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(api_key=self._get_api_key())
                system_text = ""
                conv_msgs = []
                for m in messages:
                    if m["role"] == "system":
                        system_text += m["content"] + "\n"
                    else:
                        conv_msgs.append({"role": m["role"], "content": m["content"]})
                kwargs: dict = {
                    "model": self.current_model,
                    "max_tokens": 4096,
                    "messages": conv_msgs or [{"role": "user", "content": "Hello"}],
                }
                if system_text.strip():
                    kwargs["system"] = system_text.strip()
                async with client.messages.stream(**kwargs) as stream:
                    async for text in stream.text_stream:
                        yield text
            else:
                # Fallback: non-streaming invoke
                result = await self.invoke_with_tools(messages, [])
                yield result.get("text", "")
        except Exception as e:
            logger.error(f"Cloud streaming failed ({self.provider}): {e}")
            yield f"Error: {e}"


async def check_ollama_models(base_url: str = "http://localhost:11434") -> list:
    """Check which models are available in Ollama"""
    if not base_url:
        return []
    try:
        client = ollama.Client(host=base_url)
        models = client.list()
        return [
            getattr(m, "model", None) or m.get("name") or m.get("model", "")
            for m in models.get("models", [])
        ]
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        return []


async def pull_ollama_model(model_name: str, base_url: str = "http://localhost:11434"):
    """Pull a model from Ollama"""
    try:
        client = ollama.Client(host=base_url)
        logger.info(f"Pulling model {model_name}...")
        client.pull(model_name)
        logger.info(f"Model {model_name} pulled successfully")
    except Exception as e:
        logger.error(f"Failed to pull model {model_name}: {e}")
        raise


# ── OpenWebUI model discovery ───────────────────────────────────────

async def check_openwebui_models(base_url: str, api_key: str) -> list:
    """Fetch available models from an Open WebUI or OpenAI-compatible endpoint.

    Supported routes:
      - Open WebUI: GET /api/models
      - OpenAI-compatible gateways: GET /v1/models
    """
    import aiohttp

    url = base_url.rstrip("/")
    if url.endswith(("/api", "/v1")):
        candidate_urls = [url + "/models"]
    else:
        candidate_urls = [url + "/api/models", url + "/v1/models"]

    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {api_key}"}
            for models_url in candidate_urls:
                async with session.get(models_url, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status != 200:
                        logger.debug(f"Model listing endpoint returned {resp.status}: {models_url}")
                        continue
                    data = await resp.json()
                    # Open WebUI and OpenAI-compatible APIs both commonly return
                    # either a raw list or an object with data/models entries.
                    models_list = data if isinstance(data, list) else data.get("data", data.get("models", []))
                    result = []
                    for m in models_list:
                        if isinstance(m, str):
                            result.append(m)
                        elif isinstance(m, dict):
                            result.append(m.get("id") or m.get("name") or m.get("model", ""))
                    return [x for x in result if x]
            logger.warning(f"No model listing endpoint responded successfully for {base_url}")
            return []
    except Exception as e:
        logger.error(f"Failed to list OpenWebUI models: {e}")
        return []


# ── Runtime LLM switching ───────────────────────────────────────────

async def list_all_models(config) -> dict:
    """List all available models across all configured providers.

    Returns dict with provider name → list of model IDs.
    """
    result = {}

    # Ollama models
    try:
        if config.ollama_base_url:
            ollama_models = await check_ollama_models(config.ollama_base_url)
            if ollama_models:
                result["ollama"] = ollama_models
    except Exception:
        pass

    # Open WebUI models
    owui_url = getattr(config, "openwebui_api_url", None)
    owui_key = getattr(config, "openwebui_api_key", None)
    if owui_url and owui_key:
        try:
            owui_models = await check_openwebui_models(owui_url, owui_key)
            if owui_models:
                result["openwebui"] = owui_models
        except Exception:
            pass

    # Cloud providers,  list the default model for each configured provider
    for provider, (_, key_attr, default_model) in CloudLLM._PROVIDERS.items():
        if provider in ("openwebui",):
            continue  # already handled above
        if getattr(config, key_attr, None):
            result[provider] = [default_model]

    return result


def switch_llm(agent, provider: str, model: str | None = None) -> dict:
    """Switch the agent's LLM to a different provider/model at runtime.

    Args:
        agent: The agent instance (has .llm and .config)
        provider: 'ollama', 'openwebui', or any cloud provider name
        model: Optional model name (uses provider default if omitted)

    Returns:
        dict with status info
    """
    config = agent.config
    old_provider = "none" if agent.llm is None else (
        "ollama" if isinstance(agent.llm, AdaptiveLLM) else getattr(agent.llm, "provider", "unknown")
    )
    old_model = None if agent.llm is None else agent.llm.current_model

    try:
        if provider == "ollama":
            new_llm = AdaptiveLLM(config, model or config.default_model)
            if model:
                new_llm.current_model = model
        else:
            new_llm = CloudLLM(provider=provider, config=config)
            if model:
                new_llm.current_model = model

        # Preserve token tracker history if previous LLM existed
        if agent.llm is not None:
            new_llm.token_tracker = agent.llm.token_tracker

        agent.llm = new_llm
        logger.info(f"LLM switched: {old_provider}/{old_model} → {provider}/{new_llm.current_model}")
        return {
            "success": True,
            "previous": {"provider": old_provider, "model": old_model},
            "current": {"provider": provider, "model": new_llm.current_model},
        }
    except Exception as e:
        logger.error(f"Failed to switch LLM to {provider}/{model}: {e}")
        return {"success": False, "error": str(e)}

