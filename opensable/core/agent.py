"""
Core Open-Sable Agent - The brain of the operation
"""
import asyncio
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .llm import get_llm
from .memory import MemoryManager
from .tools import ToolRegistry
from .config import OpenSableConfig

logger = logging.getLogger(__name__)

# Default timeout for a single LLM call (seconds)
_LLM_TIMEOUT = 120


class AgentState(Dict):
    """State for the agent graph"""
    messages: List[Any]
    user_id: str
    task: str
    plan: List[str]
    current_step: int
    results: Dict[str, Any]
    error: Optional[str]


class SableAgent:
    """Main autonomous agent using LangGraph"""
    
    def __init__(self, config: OpenSableConfig):
        self.config = config
        self.llm = None
        self.memory = None
        self.tools = None
        self.graph = None
        self.heartbeat_task = None
        
        # Advanced AGI components
        self.advanced_memory = None
        self.goals = None
        self.plugins = None
        self.autonomous = None
        self.multi_agent = None
        self.tool_synthesizer = None
        self.metacognition = None
        self.world_model = None
        self.tracer = None
        
    async def initialize(self):
        """Initialize agent components"""
        logger.info("Initializing Open-Sable agent...")
        
        # Initialize LLM
        self.llm = get_llm(self.config)
        
        # Initialize memory
        self.memory = MemoryManager(self.config)
        await self.memory.initialize()
        
        # Initialize tools
        self.tools = ToolRegistry(self.config)
        await self.tools.initialize()
        
        # Initialize AGI systems
        await self._initialize_agi_systems()
        
        # Build the agent graph
        self.graph = self._build_graph()
        
        # Start heartbeat (periodic checks)
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info("Agent initialized successfully")
    
    async def _initialize_agi_systems(self):
        """Initialize advanced AGI components"""
        # Initialize Advanced Memory System
        try:
            from .advanced_memory import AdvancedMemorySystem
            self.advanced_memory = AdvancedMemorySystem(
                storage_path=getattr(self.config, 'vector_db_path', None) and Path(str(self.config.vector_db_path)).parent / "advanced_memory.json"
            )
            await self.advanced_memory.initialize()
            logger.info("✅ Advanced memory system initialized")
        except Exception as e:
            logger.warning(f"Advanced memory init failed: {e}")
            self.advanced_memory = None
        
        try:
            from .goal_system import GoalManager
            self.goals = GoalManager()
            await self.goals.initialize()
            logger.info("✅ Goal system initialized")
        except Exception as e:
            logger.warning(f"Goal system init failed: {e}")
        
        try:
            from .plugins import PluginManager
            self.plugins = PluginManager(self.config)
            await self.plugins.load_all_plugins()
            logger.info(f"✅ Plugin system initialized ({len(self.plugins.plugins)} plugins loaded)")
        except Exception as e:
            logger.warning(f"Plugin system init failed: {e}")
        
        try:
            from .tool_synthesis import ToolSynthesizer
            self.tool_synthesizer = ToolSynthesizer()
            logger.info("✅ Tool synthesis initialized")
        except Exception as e:
            logger.warning(f"Tool synthesis init failed: {e}")
        
        try:
            from .metacognition import MetacognitiveSystem
            self.metacognition = MetacognitiveSystem(self.config)
            await self.metacognition.initialize()
            logger.info("✅ Metacognition initialized")
        except Exception as e:
            logger.warning(f"Metacognition init failed: {e}")
        
        try:
            from .world_model import WorldModel
            self.world_model = WorldModel()
            await self.world_model.initialize()
            logger.info("✅ World model initialized")
        except Exception as e:
            logger.warning(f"World model init failed: {e}")
        
        try:
            from .multi_agent import AgentPool
            self.multi_agent = AgentPool(self.config)
            await self.multi_agent.initialize()
            logger.info("✅ Multi-agent system initialized")
        except Exception as e:
            logger.warning(f"Multi-agent init failed: {e}")

        try:
            from .emotional_intelligence import EmotionalIntelligence
            self.emotional_intelligence = EmotionalIntelligence()
            logger.info("✅ Emotional intelligence initialized")
        except Exception as e:
            logger.warning(f"Emotional intelligence init failed: {e}")

        try:
            from .observability import DistributedTracer
            self.tracer = DistributedTracer(service_name="opensable-agent")
            logger.info("✅ Distributed tracing initialized")
        except Exception as e:
            logger.warning(f"Distributed tracing init failed: {e}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("run", self._agentic_loop)
        
        # Single-node graph -- the loop handles everything internally
        workflow.set_entry_point("run")
        workflow.add_edge("run", END)
        
        return workflow.compile()

    # ──────────────────────────────────────────────
    #  NATIVE TOOL-CALLING AGENTIC LOOP
    # ──────────────────────────────────────────────

    async def _agentic_loop(self, state: AgentState) -> AgentState:
        """
        Full agentic loop with:
        1. Fast path: Forced browser_search for obvious search queries
        2. Native tool calling: LLM picks from all 28 tools via Ollama
        3. Direct response: No tools needed, LLM answers directly
        """
        task    = state["task"]
        user_id = state["user_id"]

        # ── Tracing: create a span for the full request ───────────────
        trace_id = span = None
        if self.tracer:
            trace_id = self.tracer.create_trace()
            span = self.tracer.start_span("agentic_loop", trace_id, attributes={
                "user_id": user_id, "task_length": len(task),
            })

        # Memory context
        memories = await self.memory.recall(user_id, task)
        memory_ctx = "\n".join([m["content"] for m in memories[:3]]) if memories else ""

        from datetime import date
        today = date.today().strftime("%B %d, %Y")

        # Build conversation history
        history_for_ollama = []
        for m in state.get("messages", []):
            role = m.get("role", "")
            if role == "user":
                history_for_ollama.append({"role": "user", "content": m.get("content", "")})
            elif role == "assistant":
                history_for_ollama.append({"role": "assistant", "content": m.get("content", "")})

        base_system = self._get_personality_prompt() + (
            f"\n\nRelevant context from memory:\n{memory_ctx}" if memory_ctx else ""
        ) + f"\n\nToday's date: {today}." + (
            "\n\nIMPORTANT: For general knowledge questions (explanations, definitions, "
            "opinions, coding help), answer directly from your knowledge. "
            "Only use tools when the task specifically requires reading files, "
            "executing code, searching the web, or interacting with the system."
        )

        # Emotional intelligence: adapt tone to user's emotional state
        ei = getattr(self, 'emotional_intelligence', None)
        if ei:
            adaptation = ei.process(user_id, task)
            ei_addon = adaptation.get("system_prompt_addon", "")
            if ei_addon:
                base_system += f"\n\n[Emotional context] {ei_addon}"

        # ── FAST PATH: Forced search for obvious queries ──────────────
        task_lower = task.lower().strip()
        # Only trigger forced search when task clearly IS a search query
        # (starts with search-intent phrases, not buried in middle)
        search_start = [
            "search ", "search for ", "busca ", "buscar ", "google ",
            "look up ", "lookup ", "find me ", "find out ",
            "what is ", "what are ", "who is ", "who are ",
            "que es ", "quien es ", "cuales son ",
            "weather in ", "weather for ", "climate in ",
            "price of ", "cost of ", "reviews of ", "reviews for ",
            "news about ", "noticias de ", "noticias sobre ",
            "latest news", "current news", "flights from ", "flights to ",
        ]
        # Personal/history questions should NOT go to web search
        personal_indicators = [" my ", " our ", " your ", " mi ", " tu ", " nuestro "]
        is_personal = any(p in f" {task_lower} " for p in personal_indicators)
        is_search = (not is_personal) and any(task_lower.startswith(p) for p in search_start)

        tool_results = []

        if is_search:
            logger.info(f"🔍 [FORCED] Search intent detected")
            tool_span = None
            if self.tracer and trace_id:
                tool_span = self.tracer.start_span("tool:browser_search", trace_id, span.span_id if span else None)
            try:
                query = task
                for filler in ["search for", "busca", "find", "look up", "google", "what is", "who is"]:
                    query = query.replace(filler, "", 1).strip()
                result = await asyncio.wait_for(
                    self.tools.execute_schema_tool(
                        "browser_search", {"query": query, "num_results": 5}
                    ),
                    timeout=_LLM_TIMEOUT,
                )
                tool_results.append(f"**browser_search:** {result}")
            except asyncio.TimeoutError:
                tool_results.append("**browser_search:** ⚠️ Timed out")
                logger.error("browser_search timed out")
            except Exception as e:
                tool_results.append(f"**browser_search:** ⚠️ Failed: {e}")
            finally:
                if tool_span:
                    self.tracer.end_span(tool_span.span_id)

        # ── NATIVE TOOL CALLING: LLM chooses tools ───────────────────
        if not tool_results:
            messages = [{"role": "system", "content": base_system}]
            messages += history_for_ollama[-8:]
            messages.append({"role": "user", "content": task})

            tool_schemas = self.tools.get_tool_schemas()
            _MAX_ROUNDS = 5  # allow more rounds for code iteration
            _last_tool_was_code_error = False

            for _round in range(_MAX_ROUNDS):
                # Offer tools on round 0, or if the last execute_code had an error
                offer_tools = (not tool_results) or _last_tool_was_code_error
                _last_tool_was_code_error = False

                try:
                    response = await asyncio.wait_for(
                        self.llm.invoke_with_tools(
                            messages, tool_schemas if offer_tools else []
                        ),
                        timeout=_LLM_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"LLM call timed out (round {_round})")
                    break
                except Exception as e:
                    logger.error(f"LLM call failed: {e}")
                    break

                tc = response.get("tool_call")

                # Fallback: LLM sometimes outputs tool calls as JSON in text
                if not tc and response.get("text"):
                    tc = self._extract_tool_call_from_text(response["text"])

                if tc:
                    logger.info(f"🔧 LLM chose tool: {tc['name']}")
                    tool_span = None
                    if self.tracer and trace_id:
                        tool_span = self.tracer.start_span(
                            f"tool:{tc['name']}", trace_id,
                            span.span_id if span else None,
                            attributes={"tool.args": str(tc['arguments'])[:200]},
                        )
                    try:
                        result = await asyncio.wait_for(
                            self.tools.execute_schema_tool(
                                tc["name"], tc["arguments"]
                            ),
                            timeout=_LLM_TIMEOUT,
                        )
                        tool_results.append(f"**{tc['name']}:** {result}")
                    except asyncio.TimeoutError:
                        tool_results.append(f"**{tc['name']}:** ❌ Timed out")
                        logger.error(f"Tool {tc['name']} timed out")
                    except Exception as e:
                        tool_results.append(f"**{tc['name']}:** ❌ {e}")
                    finally:
                        if tool_span:
                            self.tracer.end_span(tool_span.span_id)

                    last_result = tool_results[-1]

                    # ── Code feedback loop: if execute_code failed, let the
                    #    LLM see the error and try again with tools available ──
                    if tc["name"] == "execute_code" and "❌" in last_result:
                        _last_tool_was_code_error = True
                        messages.append({"role": "assistant", "content": f"Used tool: {tc['name']}"})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"The code execution failed with this error:\n{last_result}\n\n"
                                "Please fix the code and try again using execute_code."
                            ),
                        })
                        logger.info(f"🔄 Code feedback loop: round {_round + 1}, re-offering tools")
                        continue

                    # Normal path: LLM synthesizes results (no tools offered)
                    messages.append({"role": "assistant", "content": f"Used tool: {tc['name']}"})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Tool result:\n{last_result}\n\n"
                            f"Now answer the original question: {task}"
                        ),
                    })
                else:
                    # LLM responded with text directly — no tool needed
                    final_text = response.get("text", "")
                    break
            else:
                final_text = None  # max rounds reached

            # If LLM answered directly (no tools used), save and return
            if not tool_results and final_text:
                state["messages"].append({
                    "role": "final_response",
                    "content": final_text,
                    "timestamp": datetime.now().isoformat(),
                })
                try:
                    await self.memory.store(
                        user_id,
                        f"Task: {task}\nResponse: {final_text}",
                        {"type": "task_completion", "timestamp": datetime.now().isoformat()},
                    )
                except Exception as e:
                    logger.debug(f"Memory store failed: {e}")
                return state

        # ── SYNTHESIS: Format tool results into final answer ──────────
        synthesis_prompt = (
            base_system
            + "\n\nCRITICAL RULES:"
            "\n- Use ONLY information from the tool results"
            "\n- NEVER invent facts not present in the results"
            "\n- If no data found, say so honestly"
            "\n- Be concise and direct"
        )
        tool_context = "\n\n".join(tool_results)
        synth_messages = [{"role": "system", "content": synthesis_prompt}]
        synth_messages += history_for_ollama[-8:]
        synth_messages.append({
            "role": "user",
            "content": f"[TOOL RESULTS]\n{tool_context}\n\n[USER QUESTION]\n{task}",
        })

        try:
            resp = await self.llm.invoke_with_tools(synth_messages, [])
            final_text = resp.get("text", "")
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            final_text = f"I found some results but had trouble formatting them:\n\n{tool_context}"

        state["messages"].append({
            "role": "final_response",
            "content": final_text,
            "timestamp": datetime.now().isoformat(),
        })
        try:
            await self.memory.store(
                user_id,
                f"Task: {task}\nResponse: {final_text}",
                {"type": "task_completion", "timestamp": datetime.now().isoformat()},
            )
        except Exception as e:
            logger.debug(f"Memory store failed: {e}")

        # ── End tracing span ──────────────────────────────────────────
        if span:
            span.set_attribute("response_length", len(final_text or ""))
            span.set_attribute("tools_used", len(tool_results))
            self.tracer.end_span(span.span_id)
            logger.debug(f"📊 Trace {trace_id}: {span.duration_ms:.0f}ms, {len(tool_results)} tools")

        return state

    def _extract_tool_call_from_text(self, text: str) -> Optional[dict]:
        """Parse tool calls that the LLM outputs as JSON text instead of structured tool_calls."""
        import json as _json
        import re as _re
        # Look for JSON blocks like {"name": "tool_name", "parameters": {...}}
        patterns = [
            r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"(?:parameters|arguments)"\s*:\s*(\{[^{}]*\})',
        ]
        for pat in patterns:
            m = _re.search(pat, text, _re.DOTALL)
            if m:
                name = m.group(1)
                try:
                    args = _json.loads(m.group(2))
                except Exception:
                    args = {}
                # Verify it's a known tool
                known = [s["function"]["name"] for s in self.tools.get_tool_schemas()]
                if name in known:
                    logger.info(f"🔧 [FALLBACK] Parsed tool call from text: {name}")
                    return {"name": name, "arguments": args}
        return None

    def _get_personality_prompt(self) -> str:
        """Get system prompt based on personality"""
        personalities = {
            "helpful": "You are Sable, a helpful and friendly AI assistant. Be clear, concise, and supportive.",
            "professional": "You are Sable, a professional AI assistant. Be formal, precise, and efficient.",
            "sarcastic": "You are Sable, a witty AI assistant with a sarcastic edge. Be helpful but add some sass.",
            "meme-aware": "You are Sable, a culturally-aware AI assistant. Use memes and internet culture when appropriate."
        }
        return personalities.get(self.config.agent_personality, personalities["helpful"])

    async def process_message(self, user_id: str, message: str, history: Optional[List[dict]] = None) -> str:
        """Process a user message and return response"""
        # Store in advanced memory if available
        if self.advanced_memory:
            try:
                from .advanced_memory import MemoryType, MemoryImportance
                await self.advanced_memory.store_memory(
                    memory_type=MemoryType.EPISODIC,
                    content=message,
                    context={"user_id": user_id, "type": "user_message"},
                    importance=MemoryImportance.MEDIUM
                )
            except Exception as e:
                logger.debug(f"Failed to store in advanced memory: {e}")
        
        # ── Python-level pre-processing ────────────────────────────────────
        # 1. Resolve pronouns using history (don't rely on LLM to do this)
        # 2. Detect "search more / again" and reuse last query
        resolved_message = self._resolve_message(message, history or [])

        # ── Multi-agent routing for complex tasks ──────────────────────────
        if self.multi_agent:
            try:
                from .multi_agent import MultiAgentOrchestrator
                orchestrator = MultiAgentOrchestrator(self.config)
                orchestrator.agent_pool = self.multi_agent
                result = await orchestrator.route_complex_task(resolved_message, user_id)
                if result:
                    logger.info("🤝 Multi-agent handled this task")
                    return result
            except Exception as e:
                logger.debug(f"Multi-agent routing skipped: {e}")

        # ── Plugin hooks ───────────────────────────────────────────────────
        if self.plugins:
            try:
                await self.plugins.execute_hook("message_received", user_id, resolved_message)
            except Exception as e:
                logger.debug(f"Plugin hook failed: {e}")

        initial_state = {
            "messages": history or [],
            "user_id": user_id,
            "task": resolved_message,
            "original_task": message,
            "plan": [],
            "current_step": 0,
            "results": {},
            "error": None,
            "last_search_query": self._last_search_query(history or []),
        }

        final_state = await self.graph.ainvoke(initial_state)

        for msg in reversed(final_state["messages"]):
            if msg["role"] == "final_response":
                return msg["content"]

        return "I processed your request, but couldn't formulate a response."

    # ------------------------------------------------------------------
    # Pre-processing helpers
    # ------------------------------------------------------------------

    _FILLER_ONLY = re.compile(
        r"^(search more|more results|search again|look it up again|find more|"
        r"busca mas|busca mas resultados|otra vez|de nuevo|show more)[\s!?.]*$",
        re.IGNORECASE,
    )
    _PRONOUNS = re.compile(
        r"\b(that|it|this|him|her|them|those|these|el|ella|ese|esa|eso|esto)\b",
        re.IGNORECASE,
    )

    def _last_search_query(self, history: list) -> str:
        """Return the last browser search query from history assistant messages."""
        import re as _re
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                # Our reflect prompt always mentions what was searched
                m = _re.search(r"searched? (?:for )?[\"']?(.+?)[\"']?[\.\n]", msg.get("content", ""), _re.I)
                if m:
                    return m.group(1).strip()
        return ""

    def _extract_topic_from_history(self, history: list) -> str:
        """Extract the most recent concrete subject from conversation history."""
        import re
        # Look at last few user messages for nouns / quoted things
        candidates = []
        for msg in reversed(history[-8:]):
            if msg.get("role") not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            # Quoted strings are strong candidates
            quoted = re.findall(r'"([^"]{3,40})"', content)
            candidates.extend(quoted)
            # Capitalized multi-word sequences (proper nouns)
            proper = re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", content)
            candidates.extend(proper)
        return candidates[0] if candidates else ""

    def _resolve_message(self, message: str, history: list) -> str:
        """
        Python-level message normalization:
        - 'search more' / 'again' → repeat last search query
        - Pronouns (that/it/this) → replace with last known subject
        """
        from datetime import date
        today = date.today().strftime("%B %d, %Y")

        # 1. Pure filler → repeat last query
        if self._FILLER_ONLY.match(message.strip()):
            last = self._last_search_query(history)
            if last:
                logger.info(f"[resolve] 'search more' → reusing query: {last}")
                return f"search for more information about {last}"
            # No history → pass through unchanged
            return message

        # 2. Message contains only pronouns for the subject → resolve
        if self._PRONOUNS.search(message):
            topic = self._extract_topic_from_history(history)
            if topic:
                resolved = self._PRONOUNS.sub(topic, message)
                logger.info(f"[resolve] pronoun '{message}' → '{resolved}'")
                message = resolved

        # 3. Inject today's date for time-sensitive queries
        time_words = ["today", "tonight", "this week", "now", "current", "latest",
                      "hoy", "esta semana", "ahora", "noticias"]
        if any(w in message.lower() for w in time_words):
            if today not in message:
                message = f"{message} (today is {today})"

        return message

    async def run(self, message: str, history: Optional[List[dict]] = None) -> str:
        """Simplified run method"""
        return await self.process_message("default_user", message, history)

    async def _heartbeat_loop(self):
        """Periodic check for scheduled tasks"""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                logger.debug("Heartbeat: checking for scheduled tasks...")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def shutdown(self):
        """Cleanup on shutdown"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.memory:
            await self.memory.close()
        logger.info("Agent shutdown complete")

