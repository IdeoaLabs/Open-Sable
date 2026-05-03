"""
Mock LLM for testing,  deterministic responses without Ollama or cloud APIs.

Usage:
    from tests.mock_llm import MockLLM

    llm = MockLLM(responses=[
        {"text": "Hello!"},
        {"tool_call": {"name": "browser_search", "arguments": {"query": "python"}}, "tool_calls": [...]},
    ])
"""

from typing import Any, AsyncIterator, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class MockLLMCall:
    """Record of a single LLM call for assertions."""
    messages: List[Dict]
    tools: List[Dict]
    response: Dict[str, Any]


class MockLLM:
    """
    Drop-in replacement for AdaptiveLLM / CloudLLM.

    Supports:
     - Sequenced responses (returns next in queue on each call)
     - Call recording (for assertions)
     - Tool call simulation
     - Streaming simulation
    """

    def __init__(
        self,
        responses: Optional[List[Dict[str, Any]]] = None,
        default_response: str = "Mock response",
    ):
        self._responses = list(responses or [])
        self._default_response = default_response
        self._call_index = 0
        self.calls: List[MockLLMCall] = []
        self.current_model = "mock-model"
        self.available_models = ["mock-model"]
        self.config = None

        # Token tracking (mirrors real LLM)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

    def _next_response(self) -> Dict[str, Any]:
        """Get next queued response or default."""
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return {"tool_call": None, "tool_calls": [], "text": self._default_response}

    async def invoke_with_tools(
        self, messages: List[Dict], tools: List[Dict]
    ) -> Dict[str, Any]:
        """Simulate tool-calling LLM response."""
        resp = self._next_response()

        # Normalize response format
        if "tool_calls" not in resp:
            resp["tool_calls"] = [resp["tool_call"]] if resp.get("tool_call") else []
        if "tool_call" not in resp:
            resp["tool_call"] = resp["tool_calls"][0] if resp["tool_calls"] else None
        if "text" not in resp:
            resp["text"] = None

        # Simulate token usage
        input_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        output_tokens = len(str(resp.get("text") or resp.get("tool_call") or "")) // 4
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Record call
        self.calls.append(MockLLMCall(
            messages=list(messages),
            tools=list(tools),
            response=resp,
        ))

        return resp

    async def ainvoke(self, messages):
        """Plain text invoke."""
        result = await self.invoke_with_tools(
            [{"role": "user", "content": str(m)} for m in messages], []
        )
        from types import SimpleNamespace
        return SimpleNamespace(content=result.get("text", ""))

    async def plain_chat(self, messages: List[Dict]) -> Dict[str, Any]:
        """Compatibility helper for code paths that use the no-tools fast-path."""
        return await self.invoke_with_tools(messages, [])

    def invoke(self, messages):
        """Sync invoke."""
        import asyncio
        return asyncio.run(self.ainvoke(messages))

    async def astream(self, messages: List[Dict]) -> AsyncIterator[str]:
        """Simulate streaming by yielding words."""
        resp = self._next_response()
        text = resp.get("text", self._default_response) or self._default_response
        for word in text.split():
            yield word + " "

    async def auto_switch_model(self, task_type: str) -> bool:
        """Mock,  never switches."""
        return False

    # ── Assertions ──

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def assert_called(self):
        assert self.calls, "MockLLM was never called"

    def assert_called_n_times(self, n: int):
        assert len(self.calls) == n, f"Expected {n} calls, got {len(self.calls)}"

    def assert_tool_was_offered(self, tool_name: str):
        """Assert that a specific tool schema was passed in at least one call."""
        for call in self.calls:
            for tool in call.tools:
                fn = tool.get("function", tool)
                if fn.get("name") == tool_name:
                    return
        raise AssertionError(f"Tool '{tool_name}' was never offered to the LLM")

    def last_messages(self) -> List[Dict]:
        """Get messages from the last call."""
        assert self.calls, "No calls recorded"
        return self.calls[-1].messages

    def reset(self):
        """Clear recorded calls and reset response index."""
        self.calls.clear()
        self._call_index = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
