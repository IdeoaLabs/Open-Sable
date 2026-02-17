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
        FORCED tool execution before LLM synthesis.
        Python detects search intent and runs browser_search immediately,
        then LLM just formats/summarizes the real results.
        """
        task    = state["task"]
        user_id = state["user_id"]

        # Memory context
        memories = await self.memory.recall(user_id, task)
        memory_ctx = "\n".join([m["content"] for m in memories[:3]]) if memories else ""

        from datetime import date
        today = date.today().strftime("%B %d, %Y")

        # ── PYTHON-LEVEL TOOL FORCING ─────────────────────────────────
        # Detect search intent via keywords and execute browser_search
        # BEFORE asking the LLM anything. LLM only synthesizes results.
        
        task_lower = task.lower()
        search_keywords = [
            "search", "busca", "find", "encuentra", "look up", "lookup",
            "google", "bing", "what is", "who is", "que es", "quien es",
            "is ", "are ", "reviews", "price", "flight", "movie", "weather",
            "news", "noticias", "latest", "current", "today", "hoy",
        ]
        
        forced_tool_results = []
        
        if any(kw in task_lower for kw in search_keywords):
            logger.info(f"🔍 [FORCED] Detected search intent, executing browser_search")
            try:
                # Extract clean query from the task
                query = task
                for filler in ["search for", "busca", "find", "look up", "google", "what is", "who is"]:
                    query = query.replace(filler, "", 1).strip()
                
                result = await self.tools.execute_schema_tool("browser_search", {"query": query, "num_results": 5})
                forced_tool_results.append(f"**Web Search Results:**\n{result}")
                logger.info(f"✅ [FORCED] browser_search completed")
            except Exception as e:
                forced_tool_results.append(f"⚠️ Search failed: {e}")
                logger.error(f"Forced search error: {e}")
        
        # ── LLM SYNTHESIS (only if we have tool results) ──────────────
        
        system_prompt = self._get_personality_prompt() + (
            f"\n\nRelevant context:\n{memory_ctx}" if memory_ctx else ""
        ) + (
            f"\n\nToday's date: {today}."
            "\n\nYour job: synthesize the tool results below into a clear, helpful answer."
            "\n\nCRITICAL RULES:"
            "\n- Use ONLY the information from the tool results"
            "\n- NEVER invent or add facts not present in the results"
            "\n- If the results say 'no data found', say that honestly"
            "\n- Be concise and direct"
        )

        # Build conversation history
        history_for_ollama = []
        for m in state.get("messages", []):
            role = m.get("role", "")
            if role == "user":
                history_for_ollama.append({"role": "user", "content": m.get("content", "")})
            elif role == "assistant":
                history_for_ollama.append({"role": "assistant", "content": m.get("content", "")})

        messages = [{"role": "system", "content": system_prompt}]
        messages += history_for_ollama[-8:]
        
        if forced_tool_results:
            # Add tool results as context, then the user query
            tool_context = "\n\n".join(forced_tool_results)
            messages.append({"role": "user", "content": f"[TOOL RESULTS]\n{tool_context}\n\n[USER QUESTION]\n{task}\n\nSynthesize the above tool results to answer the user's question."})
        else:
            messages.append({"role": "user", "content": task})

        # Call LLM for synthesis (no tools offered — just text generation)
        try:
            response = await self.llm.invoke_with_tools(messages, [])  # empty tools = text-only
            final_text = response.get("text", "")
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            final_text = f"I encountered an error: {e}"

        state["results"]["tool_calls"] = {
            "count": len(forced_tool_results),
            "summary": forced_tool_results,
            "success": True,
            "tool": "forced" if forced_tool_results else "none",
        }
        state["messages"].append({
            "role": "final_response",
            "content": final_text,
            "timestamp": datetime.now().isoformat()
        })

        # Save to memory
        try:
            await self.memory.store(
                user_id,
                f"Task: {task}\nResponse: {final_text}",
                {"type": "task_completion", "timestamp": datetime.now().isoformat()}
            )
        except Exception:
            pass

        return state

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

        """Understand what the user wants"""
        user_message = state["task"]
        user_id = state["user_id"]
        
        # Get relevant memories
        memories = await self.memory.recall(user_id, user_message)
        memory_context = "\n".join([m["content"] for m in memories[:3]]) if memories else ""
        
        # Build prompt
        system_prompt = self._get_personality_prompt()
        prompt = f"""
{system_prompt}

User request: {user_message}

Relevant context from memory:
{memory_context}

What is the user asking for? Break it down into clear objectives.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        state["messages"].append({
            "role": "understanding",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        return state
    
    async def _plan_execution(self, state: AgentState) -> AgentState:
        """Create a step-by-step plan"""
        task = state["task"]  # already resolved by _resolve_message

        prompt = f"""User request: {task}

Available tools: browser (search/scrape), calendar, weather, email, execute_command

RULES:
- If the user asks to search/find/look up/check something, or asks about a movie/person/place/event/review/opinion/news: plan ONLY 1 step => "search for <SPECIFIC TOPIC>"
- If the user asks for prices, tickets, flights, showtimes, or booking info: plan 2 steps => 1. search  2. scrape the most relevant result URL for details
- NEVER use vector_search -- use browser search instead
- NEVER add extra scrape steps after a plain search -- only add scrape for prices/bookings/showtimes
- For weather/calendar/email: use those tools directly
- "is X good?", "reviews of X" => search for "X review"
- If user asks to write a script AND run it: 2 steps => 1. write the python script to <filename>.py  2. run the script <filename>.py
- NEVER output raw JSON, code blocks, or function call syntax -- plain English steps only

Respond with ONLY the numbered steps.

Plan:
1."""

        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)

        plan_lines = [line.strip() for line in response.content.split("\n") if line.strip()]
        plan = [line for line in plan_lines if line and line[0].isdigit()]

        state["plan"] = plan
        state["current_step"] = 0
        state["messages"].append({
            "role": "planning",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })

        return state
    
    async def _execute_step(self, state: AgentState) -> AgentState:
        """Execute current step of the plan"""
        if state["current_step"] >= len(state["plan"]):
            return state
        
        current_plan = state["plan"][state["current_step"]]
        
        try:
            # Auto-detect if we need special model capabilities
            task_type = self._detect_task_type(current_plan)
            
            # Switch model if needed (AdaptiveLLM will handle this)
            if hasattr(self.llm, 'auto_switch_model'):
                switched = await self.llm.auto_switch_model(task_type)
                if switched:
                    logger.info(f"Switched to optimal model for {task_type} task")
            
            # Determine which tool to use
            tool_name, tool_input = await self._parse_tool_call(current_plan)

            # If write_file with no content yet: ask LLM to generate the code/content first
            if tool_name == "write_file" and not tool_input.get("content"):
                logger.info(f"Generating content for write_file: {tool_input.get('path')}")
                code_prompt = (
                    f"Write ONLY the complete, working Python code for this task:\n{state['task']}\n\n"
                    "Output ONLY the code. No explanation, no markdown, no ```python block. Just raw code."
                )
                code_response = await self.llm.ainvoke([HumanMessage(content=code_prompt)])
                tool_input["content"] = code_response.content.strip()

            # If execute_command with a plain-English step (no real command): generate the command
            if tool_name == "execute_command":
                cmd = tool_input.get("command", "")
                # If the "command" looks like English prose rather than a shell command
                if not any(c in cmd for c in ["python", "bash", "./", "sh ", "node ", "ruby ", "perl "]):
                    # Find a previously written file to run, or ask LLM to write+run inline
                    prev_files = [
                        r.get("tool_input", {}).get("path", "")
                        for r in state["results"].values()
                        if r.get("tool") == "write_file" and r.get("success")
                    ]
                    if prev_files and prev_files[-1]:
                        tool_input["command"] = f"python {prev_files[-1]}"
                    else:
                        # Generate and run inline without writing to a file
                        code_prompt = (
                            f"Write ONLY the complete working Python one-liner or short script (no markdown) for:\n{state['task']}"
                        )
                        code_response = await self.llm.ainvoke([HumanMessage(content=code_prompt)])
                        code = code_response.content.strip().strip("```python").strip("```").strip()
                        tool_input["command"] = f"python3 -c {repr(code)}"
            
            # Skip scrape step if a previous search already succeeded
            # BUT: if the task involves prices/booking/showtimes, scrape the top result
            if tool_name == "browser" and tool_input.get("action") == "scrape":
                prev_results = state["results"]
                last_search = next(
                    (r for r in reversed(list(prev_results.values()))
                     if r.get("tool") == "browser" and r.get("success")),
                    None,
                )
                needs_detail = any(w in state["task"].lower() for w in [
                    "price", "ticket", "flight", "book", "buy", "cost", "cheap",
                    "showtime", "schedule", "cartelera", "precio", "vuelo", "reserva",
                ])
                if last_search and not needs_detail:
                    logger.info("Skipping scrape -- search results are enough for this query")
                    state["current_step"] += 1
                    return state
                elif last_search and needs_detail and not tool_input.get("url"):
                    # Extract first real URL from last search result text
                    import re as _re
                    urls = _re.findall(r"https?://[^\s\)>\"']+", last_search.get("result", ""))
                    if urls:
                        tool_input["url"] = urls[0]
                        logger.info(f"Auto-scraping first result for price/booking: {tool_input['url']}")
                    else:
                        state["current_step"] += 1
                        return state
            
            # Execute tool only if one is needed
            if tool_name == "none":
                # No tool needed - this is just conversation
                result = "Understood."
            else:
                # Execute tool
                result = await self.tools.execute(tool_name, tool_input)
            
            state["results"][f"step_{state['current_step']}"] = {
                "plan": current_plan,
                "tool": tool_name,
                "tool_input": tool_input,
                "query": tool_input.get("query", ""),  # saved for 'search more' reuse
                "result": result,
                "success": True,
                "model_used": self.llm.current_model if hasattr(self.llm, 'current_model') else "unknown"
            }
            
            state["current_step"] += 1
            
        except Exception as e:
            logger.error(f"Error executing step {state['current_step']}: {e}")
            state["error"] = str(e)
            state["results"][f"step_{state['current_step']}"] = {
                "plan": current_plan,
                "error": str(e),
                "success": False
            }
        
        return state
    
    def _detect_task_type(self, plan_step: str) -> str:
        """Detect what type of task this is to choose optimal model"""
        plan_lower = plan_step.lower()
        
        # Vision tasks
        if any(word in plan_lower for word in ['image', 'picture', 'photo', 'screenshot', 'visual', 'see', 'look']):
            return 'vision'
        
        # Reasoning tasks
        if any(word in plan_lower for word in ['analyze', 'think', 'reason', 'deduce', 'calculate', 'solve', 'problem']):
            return 'reasoning'
        
        # Tool/action tasks
        if any(word in plan_lower for word in ['search', 'fetch', 'download', 'execute', 'run', 'call', 'api']):
            return 'tools'
        
        return 'general'
    
    async def _reflect_on_results(self, state: AgentState) -> AgentState:
        """Reflect on execution and formulate response"""
        task = state["task"]
        results = state["results"]
        
        # Check if any tools were actually executed
        tools_executed = [r for r in results.values() if r.get("tool") != "none"]
        
        if not tools_executed:
            # No tools executed - just respond directly
            prompt = f"{self._get_personality_prompt()}\n\nUser: {task}\n\nIMPORTANT: Answer directly. Do NOT describe actions you didn't take."
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            state["messages"].append({
                "role": "final_response",
                "content": response.content,
                "timestamp": datetime.now().isoformat()
            })
            return state
        
        # Build summary with ONLY real tool results
        summary = []
        for step_key, step_result in results.items():
            if step_result.get("tool") == "none":
                continue  # Skip non-tool steps
                
            if step_result.get("success"):
                summary.append(f"Tool: {step_result['tool']}\nResult: {step_result.get('result', 'No output')}")
            else:
                summary.append(f"Tool: {step_result['tool']}\nError: {step_result.get('error', 'Failed')}")
        
        summary_text = "\n\n".join(summary)        
        # 🧠 METACOGNITIVE SELF-EVALUATION
        if self.metacognition:
            try:
                # Check if response needs more depth
                confidence = await self.metacognition.assess_confidence(summary_text, task)
                
                if confidence < 0.7:  # Low confidence - needs more info
                    logger.info(f"🧠 Metacognition: Low confidence ({confidence:.0%}), response may need improvement")
                    # Auto-scraping disabled: it was grabbing unrelated URLs and causing timeouts
                    # The LLM should ask the user for clarification or use the search results as-is
            except Exception as e:
                logger.debug(f"Metacognition eval skipped: {e}")        
        # Generate final response using ONLY actual tool outputs
        rules = (
            "CRITICAL RULES:\n"
            "1. Answer using ONLY the information from the tool results above\n"
            "2. NEVER invent, guess, or make up facts, names, dates, or any data not present in the results\n"
            "3. If the search returned no results or failed, say: I could not find information about that -- do NOT fabricate an answer\n"
            "4. If results are present, summarize them accurately and cite sources\n"
            "5. Be direct and concise\n"
            "6. Do NOT describe actions you did not take\n"
        )
        prompt = (
            f"User asked: {task}\n\n"
            f"ACTUAL TOOL RESULTS (use ONLY this information):\n{summary_text}\n\n"
            f"{rules}\n"
            "Provide a helpful answer based STRICTLY on what the tools returned above."
        )

        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        
        state["messages"].append({
            "role": "final_response",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save to memory
        await self.memory.store(
            state["user_id"],
            f"Task: {task}\nResponse: {response.content}",
            {"type": "task_completion", "timestamp": datetime.now().isoformat()}
        )
        
        return state

    async def run(self, message: str, history: Optional[List[dict]] = None) -> str:
        """
        Simplified run method for direct agent execution
        
        Args:
            message: User message
            history: Optional conversation history
            
        Returns:
            Agent response
        """
        # For now, delegate to process_message with a default user_id
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
