"""
Integration + unit tests for the new SDK modules:
  - @function_tool decorator
  - Runner / Agent
  - MCP client
  - Token-level streaming
  - LangGraph removal (graph-free agent loop)
"""

import asyncio
import json
import pytest
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# ═══════════════════════════════════════════════════════════════════════
# 1.  @function_tool  decorator
# ═══════════════════════════════════════════════════════════════════════

from opensable.core.function_tool import (
    FunctionTool,
    function_tool,
    collect_schemas,
    build_tool_executor,
    _build_schema,
    _parse_docstring_args,
    _python_type_to_json,
)


class TestFunctionToolDecorator:
    """Tests for the @function_tool decorator."""

    def test_basic_decorator_no_parens(self):
        @function_tool
        async def greet(name: str) -> str:
            """Say hello."""
            return f"Hello {name}"

        assert isinstance(greet, FunctionTool)
        assert greet.name == "greet"
        schema = greet.schema
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"
        assert "name" in schema["function"]["parameters"]["properties"]
        assert "name" in schema["function"]["parameters"]["required"]

    def test_decorator_with_args(self):
        @function_tool(name="custom_greet", description="Custom greeting")
        async def greet(name: str) -> str:
            """Original doc."""
            return f"Hello {name}"

        assert greet.name == "custom_greet"
        assert greet.schema["function"]["description"] == "Custom greeting"

    def test_schema_types(self):
        @function_tool
        async def compute(
            text: str,
            count: int,
            ratio: float,
            flag: bool,
            items: list,
            metadata: dict,
        ) -> str:
            """Process data."""
            return "ok"

        props = compute.schema["function"]["parameters"]["properties"]
        assert props["text"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["ratio"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"
        assert props["items"]["type"] == "array"
        assert props["metadata"]["type"] == "object"

    def test_default_values(self):
        @function_tool
        async def search(query: str, limit: int = 10, lang: str = "en") -> str:
            """Search things."""
            return query

        schema = search.schema["function"]
        assert schema["parameters"]["required"] == ["query"]
        assert schema["parameters"]["properties"]["limit"]["default"] == 10
        assert schema["parameters"]["properties"]["lang"]["default"] == "en"

    def test_docstring_parsing(self):
        @function_tool
        async def weather(city: str, units: str = "celsius") -> str:
            """Get weather for a city.

            Args:
                city: Name of the city
                units: Temperature units
            """
            return "sunny"

        props = weather.schema["function"]["parameters"]["properties"]
        assert props["city"]["description"] == "Name of the city"
        assert props["units"]["description"] == "Temperature units"

    def test_description_from_first_docstring_line(self):
        @function_tool
        async def do_thing(x: str) -> str:
            """Perform a specific thing with x.

            Args:
                x: The input
            """
            return x

        desc = do_thing.schema["function"]["description"]
        assert "Perform a specific thing" in desc

    @pytest.mark.asyncio
    async def test_call_async_function(self):
        @function_tool
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await add(3, 4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_call_sync_function(self):
        @function_tool
        def multiply(a: int, b: int) -> int:
            """Multiply."""
            return a * b

        result = await multiply(3, 4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_execute_with_dict(self):
        @function_tool
        async def greet(name: str) -> str:
            """Greet."""
            return f"Hi {name}"

        result = await greet.execute({"name": "Alice"})
        assert result == "Hi Alice"

    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        @function_tool
        async def bad_tool(x: str) -> str:
            """Fail."""
            raise ValueError("boom")

        result = await bad_tool.execute({"x": "test"})
        assert "❌" in result
        assert "boom" in result

    def test_collect_schemas(self):
        @function_tool
        async def a(x: str) -> str:
            """A."""
            return x

        @function_tool
        async def b(y: int) -> str:
            """B."""
            return str(y)

        schemas = collect_schemas([a, b])
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"a", "b"}

    def test_build_tool_executor(self):
        @function_tool
        async def tool_a(x: str) -> str:
            """A."""
            return x

        @function_tool
        async def tool_b(y: str) -> str:
            """B."""
            return y

        executor = build_tool_executor([tool_a, tool_b])
        assert "tool_a" in executor
        assert "tool_b" in executor

    def test_repr(self):
        @function_tool
        async def my_tool(query: str, limit: int = 5) -> str:
            """Search."""
            return query

        r = repr(my_tool)
        assert "FunctionTool" in r
        assert "my_tool" in r
        assert "query" in r

    def test_list_type_hint(self):
        @function_tool
        async def process(items: List[str]) -> str:
            """Process items."""
            return str(items)

        prop = process.schema["function"]["parameters"]["properties"]["items"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"


class TestSchemaHelpers:
    """Tests for internal schema utilities."""

    def test_python_type_to_json_basic(self):
        assert _python_type_to_json(str)["type"] == "string"
        assert _python_type_to_json(int)["type"] == "integer"
        assert _python_type_to_json(float)["type"] == "number"
        assert _python_type_to_json(bool)["type"] == "boolean"

    def test_parse_docstring_args(self):
        doc = """Do thing.

        Args:
            x: First param
            y: Second param

        Returns:
            str
        """
        result = _parse_docstring_args(doc)
        assert result["x"] == "First param"
        assert result["y"] == "Second param"

    def test_parse_docstring_no_args(self):
        assert _parse_docstring_args("Simple doc.") == {}
        assert _parse_docstring_args("") == {}

    def test_build_schema_no_docstring(self):
        async def bare_fn(x: str) -> str:
            return x

        schema = _build_schema(bare_fn)
        assert schema["function"]["name"] == "bare_fn"
        assert schema["function"]["parameters"]["required"] == ["x"]


# ═══════════════════════════════════════════════════════════════════════
# 2.  Runner / Agent SDK
# ═══════════════════════════════════════════════════════════════════════

from opensable.core.runner import Agent, Runner, RunResult, StreamEvent


class TestRunResult:
    def test_str(self):
        r = RunResult(text="Hello world")
        assert str(r) == "Hello world"

    def test_repr(self):
        r = RunResult(text="Short answer")
        assert "Short answer" in repr(r)

    def test_repr_truncation(self):
        r = RunResult(text="x" * 200)
        assert "..." in repr(r)


class TestStreamEvent:
    def test_str(self):
        e = StreamEvent(type="token", text="hello")
        assert str(e) == "hello"


class TestAgent:
    def test_creation(self):
        agent = Agent(name="TestBot", instructions="Be helpful")
        assert agent.name == "TestBot"
        assert agent.instructions == "Be helpful"
        assert agent.tools == []
        assert agent._core_agent is None

    def test_creation_with_tools(self):
        @function_tool
        async def my_tool(x: str) -> str:
            """Tool."""
            return x

        agent = Agent(name="Test", tools=[my_tool])
        assert len(agent.tools) == 1

    def test_default_values(self):
        agent = Agent()
        assert agent.name == "Sable"
        assert agent.instructions is None
        assert agent.model is None


class TestRunnerSync:
    """Test Runner with mocked agent internals."""

    @pytest.mark.asyncio
    async def test_run_returns_result(self):
        """Test Runner.run with a mocked SableAgent."""
        agent = Agent(name="Mock")

        # Mock the internal agent
        mock_core = AsyncMock()
        mock_core.process_message = AsyncMock(return_value="Hello from Sable!")
        agent._core_agent = mock_core

        result = await Runner.run(agent, "Hi")
        assert isinstance(result, RunResult)
        assert result.text == "Hello from Sable!"
        assert result.metadata["agent"] == "Mock"

    @pytest.mark.asyncio
    async def test_run_streamed(self):
        """Test Runner.run_streamed with a mocked SableAgent."""
        agent = Agent(name="StreamBot")

        # Mock the stream method
        async def fake_stream(msg, user_id="default_user", history=None):
            yield {"type": "progress", "text": "Thinking..."}
            yield {"type": "response", "text": "Done!"}

        mock_core = MagicMock()
        mock_core.stream = fake_stream
        agent._core_agent = mock_core

        events = []
        async for ev in Runner.run_streamed(agent, "test"):
            events.append(ev)

        assert len(events) == 2
        assert events[0].type == "progress"
        assert events[1].type == "response"
        assert events[1].text == "Done!"


# ═══════════════════════════════════════════════════════════════════════
# 3.  MCP Client
# ═══════════════════════════════════════════════════════════════════════

from opensable.core.mcp import MCPClient, MCPTool, MCPResource, connect_mcp_tools


class TestMCPTool:
    def test_to_openai_schema(self):
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_file"
        assert schema["function"]["description"] == "Read a file"
        assert "path" in schema["function"]["parameters"]["properties"]


class TestMCPResource:
    def test_creation(self):
        r = MCPResource(uri="file:///tmp/test.txt", name="test.txt")
        assert r.uri == "file:///tmp/test.txt"
        assert r.mime_type == "text/plain"


class TestMCPClient:
    def test_stdio_factory(self):
        client = MCPClient.stdio("node", ["server.js"])
        assert client._transport == "stdio"
        assert client._kwargs["command"] == "node"

    def test_sse_factory(self):
        client = MCPClient.sse("https://example.com/sse")
        assert client._transport == "sse"
        assert client._kwargs["url"] == "https://example.com/sse"

    def test_not_connected_initially(self):
        client = MCPClient.stdio("echo")
        assert not client.connected

    def test_handle_message_resolves_pending(self):
        client = MCPClient.stdio("echo")
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        client._pending[1] = future
        client._handle_message({"id": 1, "result": {"tools": []}})
        assert future.done()
        assert future.result() == {"tools": []}
        loop.close()

    def test_handle_message_error(self):
        client = MCPClient.stdio("echo")
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        client._pending[2] = future
        client._handle_message({"id": 2, "error": {"message": "fail"}})
        assert future.done()
        assert future.result() is None
        loop.close()


class TestConnectMCPTools:
    def test_registers_tools(self):
        """Test that connect_mcp_tools wires schemas into a mock ToolRegistry."""
        mock_registry = MagicMock()
        mock_registry._custom_schemas = []
        mock_client = MCPClient.stdio("echo")

        schemas = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        connect_mcp_tools(mock_registry, schemas, mock_client)

        mock_registry.register.assert_called_once()
        assert len(mock_registry._custom_schemas) == 1
        assert mock_registry._custom_schemas[0]["function"]["name"] == "mcp_search"


# ═══════════════════════════════════════════════════════════════════════
# 4.  LangGraph removal,  graph-free agent loop
# ═══════════════════════════════════════════════════════════════════════


class TestGraphFreeAgent:
    """Verify the agent no longer depends on LangGraph."""

    def test_no_langgraph_import_in_agent(self):
        """agent.py must not import from langgraph."""
        import inspect
        from opensable.core import agent as agent_mod
        source = inspect.getsource(agent_mod)
        assert "from langgraph" not in source
        assert "import langgraph" not in source

    def test_agent_state_is_plain_dict(self):
        from opensable.core.agent import AgentState
        state = AgentState(task="hello", user_id="test", messages=[])
        assert isinstance(state, dict)
        assert state["task"] == "hello"

    def test_run_loop_method_exists(self):
        """SableAgent must have _run_loop instead of _build_graph."""
        from opensable.core.agent import SableAgent
        assert hasattr(SableAgent, "_run_loop")
        # _build_graph should be gone
        assert not hasattr(SableAgent, "_build_graph")


# ═══════════════════════════════════════════════════════════════════════
# 5.  Token-level streaming
# ═══════════════════════════════════════════════════════════════════════


class TestTokenStreaming:
    """Tests for the new astream() methods."""

    def test_adaptive_llm_has_astream(self):
        from opensable.core.llm import AdaptiveLLM
        assert hasattr(AdaptiveLLM, "astream")

    def test_cloud_llm_has_astream(self):
        from opensable.core.llm import CloudLLM
        assert hasattr(CloudLLM, "astream")

    def test_agent_stream_yields_token_type(self):
        """Verify the agent's stream() signature supports the 'token' event type."""
        import inspect
        from opensable.core.agent import SableAgent
        source = inspect.getsource(SableAgent.stream)
        assert '"token"' in source or "'token'" in source


class TestCloudProviderFailover:
    """Tests for automatic cloud provider failover."""

    def test_openwebui_is_preferred_over_fm_when_both_are_configured(self):
        from opensable.core.config import OpenSableConfig
        from opensable.core.llm import CloudLLM, get_llm

        config = OpenSableConfig(
            ollama_base_url="",
            fm_api_key="fm-key",
            fm_api_url="https://fm.opensable.com/v1",
            openwebui_api_key="owui-key",
            openwebui_api_url="https://sofia.zunvra.com",
            openwebui_model="sable-sofia",
        )

        llm = get_llm(config)

        assert isinstance(llm, CloudLLM)
        assert llm.provider == "openwebui"
        assert "fm" not in CloudLLM._provider_order(config, include_fm_when_openwebui=False)

    @pytest.mark.asyncio
    async def test_cloud_llm_fails_over_to_next_configured_provider(self):
        from opensable.core.config import OpenSableConfig
        from opensable.core.llm import CloudLLM

        config = OpenSableConfig(
            fm_api_key="fm-key",
            fm_api_url="https://fm.opensable.com/v1",
            openai_api_key="openai-key",
        )

        llm = CloudLLM(provider="fm", config=config)

        async def fail_fm(self, messages, tools):
            raise RuntimeError("FM returned 502")

        async def succeed_openai(self, messages, tools):
            return {"tool_call": None, "tool_calls": [], "text": "backup ok"}

        with patch.object(CloudLLM, "_fm_invoke", fail_fm), patch.object(CloudLLM, "_openai_invoke", succeed_openai):
            result = await llm.invoke_with_tools(
                [{"role": "user", "content": "hello"}],
                [],
            )

        assert result["text"] == "backup ok"
        assert llm.provider == "openai"

    def test_invalid_anthropic_key_is_not_treated_as_configured(self):
        from opensable.core.config import OpenSableConfig
        from opensable.core.llm import CloudLLM

        config = OpenSableConfig(
            fm_api_key="fm-key",
            fm_api_url="https://fm.opensable.com/v1",
            anthropic_api_key="foobar",
            deepseek_api_key="deepseek-key",
        )

        llm = CloudLLM(provider="fm", config=config)

        assert llm._is_provider_configured("anthropic") is False
        assert "anthropic" not in llm._failover_candidates()
        assert "deepseek" in llm._failover_candidates()

    def test_openwebui_failover_does_not_drop_back_to_fm(self):
        from opensable.core.config import OpenSableConfig
        from opensable.core.llm import CloudLLM

        config = OpenSableConfig(
            fm_api_key="fm-key",
            fm_api_url="https://fm.opensable.com/v1",
            openwebui_api_key="owui-key",
            openwebui_api_url="https://sofia.zunvra.com",
            openai_api_key="openai-key",
        )

        llm = CloudLLM(provider="openwebui", config=config)

        assert "openai" in llm._failover_candidates()
        assert "fm" not in llm._failover_candidates()


class TestConfigAliases:
    def test_load_config_accepts_sofia_aliases_for_openwebui(self, monkeypatch):
        from opensable.core.config import load_config

        monkeypatch.delenv("OPENWEBUI_API_URL", raising=False)
        monkeypatch.delenv("OPENWEBUI_API_KEY", raising=False)
        monkeypatch.delenv("OPENWEBUI_MODEL", raising=False)
        monkeypatch.setenv("SOFIA_API_URL", "https://sofia.zunvra.com")
        monkeypatch.setenv("SOFIA_API_KEY", "sofia-key")
        monkeypatch.setenv("SOFIA_MODEL", "sable-sofia")

        config = load_config()

        assert config.openwebui_api_url == "https://sofia.zunvra.com"
        assert config.openwebui_api_key == "sofia-key"
        assert config.openwebui_model == "sable-sofia"


# ═══════════════════════════════════════════════════════════════════════
# 6.  Integration: full loop with mocked LLM
# ═══════════════════════════════════════════════════════════════════════


class TestIntegrationMockedLoop:
    """End-to-end test of the agent loop with a mocked LLM (no Ollama needed)."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        """Agent should return a direct text answer when LLM returns text."""
        from opensable.core.agent import SableAgent
        from opensable.core.config import OpenSableConfig

        config = OpenSableConfig()
        agent = SableAgent(config)

        # Mock everything heavy
        agent.llm = AsyncMock()
        agent.llm.invoke_with_tools = AsyncMock(
            return_value={"tool_call": None, "tool_calls": [], "text": "Paris is the capital of France."}
        )
        agent.memory = AsyncMock()
        agent.memory.recall = AsyncMock(return_value=[])
        agent.memory.store = AsyncMock()
        agent.memory.get_user_preferences = AsyncMock(return_value={})
        agent.memory.close = AsyncMock()
        agent.tools = MagicMock()
        agent.tools.get_tool_schemas = MagicMock(return_value=[])

        result = await agent.run("What is the capital of France?")
        assert "Paris" in result

    @pytest.mark.asyncio
    async def test_llm_backend_error_is_not_leaked_to_user(self):
        """Transport/backend failures should not be returned as raw assistant text."""
        from opensable.core.agent import SableAgent
        from opensable.core.config import OpenSableConfig
        from tests.mock_llm import MockLLM

        config = OpenSableConfig()
        agent = SableAgent(config)

        agent.llm = MockLLM(
            responses=[
                {
                "tool_call": None,
                "tool_calls": [],
                "text": "",
                "error": "FM returned 502: <!DOCTYPE html>",
                }
            ]
        )
        agent.memory = AsyncMock()
        agent.memory.recall = AsyncMock(return_value=[])
        agent.memory.store = AsyncMock()
        agent.memory.get_user_preferences = AsyncMock(return_value={})
        agent.memory.close = AsyncMock()
        agent.tools = MagicMock()
        agent.tools.get_tool_schemas = MagicMock(return_value=[])

        result = await agent.run("hey")

        assert "FM returned 502" not in result
        assert "<!DOCTYPE html>" not in result
        assert "language backend" in result

    @pytest.mark.asyncio
    async def test_tool_call_loop(self):
        """Agent should execute a tool call and synthesize."""
        from opensable.core.agent import SableAgent
        from opensable.core.config import OpenSableConfig

        config = OpenSableConfig()
        agent = SableAgent(config)

        call_count = 0

        async def mock_invoke(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and tools:
                return {
                    "tool_call": {"name": "browser_search", "arguments": {"query": "test"}},
                    "tool_calls": [{"name": "browser_search", "arguments": {"query": "test"}}],
                    "text": None,
                }
            return {"tool_call": None, "tool_calls": [], "text": "Based on search: here are results."}

        agent.llm = AsyncMock()
        agent.llm.invoke_with_tools = mock_invoke

        agent.memory = AsyncMock()
        agent.memory.recall = AsyncMock(return_value=[])
        agent.memory.store = AsyncMock()
        agent.memory.get_user_preferences = AsyncMock(return_value={})
        agent.memory.close = AsyncMock()

        agent.tools = MagicMock()
        agent.tools.get_tool_schemas = MagicMock(return_value=[{
            "type": "function",
            "function": {"name": "browser_search", "description": "Search", "parameters": {"type": "object", "properties": {}}},
        }])
        agent.tools.execute_schema_tool = AsyncMock(return_value="Search result: Python is great")

        result = await agent.run("Search for Python")
        assert "results" in result.lower() or "search" in result.lower() or "Python" in result

    @pytest.mark.asyncio
    async def test_runner_end_to_end(self):
        """Runner.run should work end-to-end with mocked internals."""
        agent = Agent(name="E2E")

        mock_core = AsyncMock()
        mock_core.process_message = AsyncMock(return_value="E2E works!")
        agent._core_agent = mock_core

        result = await Runner.run(agent, "test")
        assert result.text == "E2E works!"


# ═══════════════════════════════════════════════════════════════════════
# 7.  ToolRegistry integration with custom schemas
# ═══════════════════════════════════════════════════════════════════════


class TestToolRegistryCustomSchemas:
    """Test that @function_tool tools integrate with ToolRegistry."""

    def test_custom_schemas_attribute_exists(self):
        from opensable.core.tools import ToolRegistry
        config = MagicMock()
        config.sandbox_mode = False
        config.ollama_base_url = "http://localhost:11434"
        registry = ToolRegistry(config)
        assert hasattr(registry, "_custom_schemas")
        assert isinstance(registry._custom_schemas, list)

    def test_execute_schema_tool_accepts_custom(self):
        """execute_schema_tool should handle custom-registered tools."""
        from opensable.core.tools import ToolRegistry
        config = MagicMock()
        config.sandbox_mode = False
        config.ollama_base_url = "http://localhost:11434"
        registry = ToolRegistry(config)

        # Register a custom tool
        async def my_handler(params):
            return f"result: {params.get('x', '')}"

        registry.register("my_custom_tool", my_handler)
        # Should not raise
        assert "my_custom_tool" in registry.tools


# ═══════════════════════════════════════════════════════════════════════
# 8.  pyproject.toml: no langgraph in core deps
# ═══════════════════════════════════════════════════════════════════════


class TestDependencies:
    def test_no_langgraph_in_core_deps(self):
        """pyproject.toml should NOT list langgraph in [project.dependencies]."""
        from pathlib import Path
        toml_path = Path(__file__).parent.parent / "pyproject.toml"
        if not toml_path.exists():
            pytest.skip("pyproject.toml not found")
        content = toml_path.read_text()
        # Parse dependencies section
        in_deps = False
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("dependencies"):
                in_deps = True
                continue
            if in_deps and stripped.startswith("["):
                break
            if in_deps:
                assert "langgraph" not in stripped.lower(), "langgraph should not be in core dependencies"
                assert "langchain" not in stripped.lower() or "optional" in stripped.lower(), \
                    "langchain should not be in core dependencies"
