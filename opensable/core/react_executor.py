"""
ReAct Executor,  Reasoning + Acting loop for autonomous task execution.

Implements the ReAct paradigm (Yao et al. 2022):
  Thought → Action → Observation → Thought → ...

Instead of executing a single tool per task, the agent can chain multiple
tool calls with intermediate reasoning until the task is complete or it
decides to stop.

This is the "hands" of the autonomous agent,  the proactive reasoning engine
decides WHAT to do, the ReAct executor figures out HOW to do it, step by step.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReActStep:
    """A single step in a ReAct execution chain."""
    step_num: int
    thought: str = ""
    action: str = ""           # Tool name
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""      # Tool result
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReActResult:
    """Final result of a ReAct execution."""
    success: bool
    task: str
    final_answer: str = ""
    steps: List[ReActStep] = field(default_factory=list)
    total_duration_ms: float = 0
    tools_used: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def summary(self) -> str:
        parts = [f"Task: {self.task[:100]}"]
        parts.append(f"Steps: {len(self.steps)}")
        parts.append(f"Duration: {self.total_duration_ms:.0f}ms")
        if self.tools_used:
            parts.append(f"Tools: {', '.join(set(self.tools_used))}")
        if self.success:
            parts.append(f"Result: {self.final_answer[:200]}")
        else:
            parts.append(f"Error: {self.error}")
        return "\n".join(parts)


# System prompt that guides the LLM through ReAct reasoning
REACT_SYSTEM_PROMPT = """You are an autonomous AI agent executing a task step by step.

For each step, output your response in EXACTLY this JSON format:
{
  "thought": "Your reasoning about what to do next",
  "action": "tool_name_to_call",
  "action_input": {"param1": "value1", "param2": "value2"}
}

When you have enough information to complete the task, output:
{
  "thought": "I have all the information needed",
  "action": "FINISH",
  "action_input": {"answer": "Your final answer/result"}
}

Rules:
1. Think step by step,  reason about what you know and what you need
2. Use one tool at a time
3. After each observation, reason about the result before acting again
4. If a tool fails, try a different approach
5. Always finish with action="FINISH" when done
6. If stuck after 3 attempts, FINISH with what you have
7. Keep your thoughts concise but informative

Available tools will be listed before each task."""


class ReActExecutor:
    """
    Execute tasks using the ReAct reasoning + acting loop.

    The executor takes a task description, available tools, and an LLM,
    then iteratively reasons and acts until the task is complete.
    """

    def __init__(
        self,
        max_steps: int = 8,
        timeout_s: float = 180.0,
        log_dir: Optional[Path] = None,
    ):
        self.max_steps = max_steps
        self.timeout_s = timeout_s
        self.log_dir = log_dir or Path("data/react_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self._total_executions = 0
        self._total_successes = 0
        self._total_steps = 0

    async def execute(
        self,
        task: str,
        llm,
        tool_executor: Callable[[str, Dict[str, Any]], Coroutine[Any, Any, str]],
        available_tools: List[Dict[str, Any]] = None,
        context: str = "",
    ) -> ReActResult:
        """
        Execute a task using the ReAct loop.

        Args:
            task: Description of the task to accomplish.
            llm: The LLM instance (must have invoke_with_tools method).
            tool_executor: Async function(tool_name, args) -> result_string.
            available_tools: List of tool schemas (for LLM context).
            context: Additional context for the task.

        Returns:
            ReActResult with all steps and final answer.
        """
        start_time = time.monotonic()
        self._total_executions += 1
        steps: List[ReActStep] = []
        tools_used: List[str] = []

        # Build tool list description for the LLM
        tool_desc = self._format_tool_list(available_tools or [])

        # Build initial messages
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_task_prompt(task, tool_desc, context)},
        ]

        try:
            for step_num in range(self.max_steps):
                step = ReActStep(step_num=step_num)

                # Ask LLM for next thought + action
                try:
                    response = await asyncio.wait_for(
                        llm.invoke_with_tools(messages, []),
                        timeout=self.timeout_s,
                    )
                except asyncio.TimeoutError:
                    return ReActResult(
                        success=False, task=task, steps=steps,
                        error="LLM call timed out",
                        total_duration_ms=(time.monotonic() - start_time) * 1000,
                        tools_used=tools_used,
                    )

                text = response.get("text", "").strip()
                parsed = self._parse_react_output(text)

                step.thought = parsed.get("thought", "")
                step.action = parsed.get("action", "")
                step.action_input = parsed.get("action_input", {})

                # Check for FINISH
                if step.action.upper() == "FINISH":
                    answer = step.action_input.get("answer", step.thought)
                    steps.append(step)
                    self._total_successes += 1
                    self._total_steps += len(steps)

                    result = ReActResult(
                        success=True, task=task, final_answer=str(answer),
                        steps=steps,
                        total_duration_ms=(time.monotonic() - start_time) * 1000,
                        tools_used=tools_used,
                    )
                    self._log_execution(result)
                    return result

                # Execute tool
                if step.action:
                    try:
                        observation = await asyncio.wait_for(
                            tool_executor(step.action, step.action_input),
                            timeout=60.0,
                        )
                        step.observation = str(observation)[:2000]
                        tools_used.append(step.action)
                    except asyncio.TimeoutError:
                        step.observation = f"ERROR: Tool '{step.action}' timed out after 60s"
                    except Exception as e:
                        step.observation = f"ERROR: Tool '{step.action}' failed: {str(e)[:500]}"
                else:
                    step.observation = "ERROR: No action specified. Use 'FINISH' when done."

                steps.append(step)

                # Append step to conversation for next iteration
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {step.observation}\n\nContinue with next thought and action.",
                })

            # Max steps reached
            self._total_steps += len(steps)
            result = ReActResult(
                success=False, task=task,
                final_answer="Max steps reached",
                steps=steps,
                error=f"Reached maximum of {self.max_steps} steps without finishing",
                total_duration_ms=(time.monotonic() - start_time) * 1000,
                tools_used=tools_used,
            )
            self._log_execution(result)
            return result

        except Exception as e:
            self._total_steps += len(steps)
            result = ReActResult(
                success=False, task=task, steps=steps,
                error=str(e),
                total_duration_ms=(time.monotonic() - start_time) * 1000,
                tools_used=tools_used,
            )
            self._log_execution(result)
            return result

    def _build_task_prompt(self, task: str, tool_desc: str, context: str) -> str:
        """Build the initial task prompt."""
        parts = [f"Task: {task}"]
        if context:
            parts.append(f"\nContext:\n{context}")
        if tool_desc:
            parts.append(f"\nAvailable tools:\n{tool_desc}")
        parts.append("\nBegin. Output your first thought and action as JSON.")
        return "\n".join(parts)

    def _format_tool_list(self, tools: List[Dict[str, Any]]) -> str:
        """Format tool schemas into a concise list for the LLM."""
        if not tools:
            return "(Use execute_command for shell commands, read_file/write_file for files, browser for web)"

        lines = []
        for t in tools[:30]:  # Cap at 30 to avoid context overflow
            fn = t.get("function", t)
            name = fn.get("name", "unknown")
            desc = fn.get("description", "")[:80]
            params = fn.get("parameters", {}).get("properties", {})
            param_names = list(params.keys())[:5]
            lines.append(f"  - {name}({', '.join(param_names)}): {desc}")
        return "\n".join(lines)

    def _parse_react_output(self, text: str) -> Dict[str, Any]:
        """Parse the LLM's ReAct output (JSON or freeform)."""
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding a JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Fallback: extract thought and action from freeform text
        thought_match = re.search(r"(?:thought|thinking|reason(?:ing)?)\s*[:：]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        action_match = re.search(r"(?:action)\s*[:：]\s*(\w+)", text, re.IGNORECASE)

        return {
            "thought": thought_match.group(1).strip() if thought_match else text[:200],
            "action": action_match.group(1) if action_match else "FINISH",
            "action_input": {"answer": text} if not action_match else {},
        }

    def _log_execution(self, result: ReActResult):
        """Log a completed execution to JSONL."""
        try:
            log_file = self.log_dir / "react_executions.jsonl"
            entry = {
                "ts": datetime.now().isoformat(),
                "task": result.task[:200],
                "success": result.success,
                "steps": len(result.steps),
                "duration_ms": result.total_duration_ms,
                "tools_used": list(set(result.tools_used)),
                "error": result.error,
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log ReAct execution: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return executor statistics."""
        return {
            "total_executions": self._total_executions,
            "total_successes": self._total_successes,
            "success_rate": (
                self._total_successes / max(1, self._total_executions)
            ),
            "total_steps": self._total_steps,
            "avg_steps_per_execution": (
                self._total_steps / max(1, self._total_executions)
            ),
        }
