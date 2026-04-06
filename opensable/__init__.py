"""
Open-Sable - Autonomous AI Agent Framework
Agentic AI cognitive subsystems for autonomous agents

Version: 1.7.0
"""

__version__ = "1.7.0"
__author__ = "Ideoa Labs"
__license__ = "MIT"

# Core imports for convenience
try:
    from opensable.core.agent import SableAgent
    from opensable.core.config import load_config, OpenSableConfig

    # SDK essentials
    from opensable.core.function_tool import FunctionTool, function_tool
    from opensable.core.runner import Agent, Runner, RunResult, StreamEvent
    from opensable.core.mcp import MCPClient

    # Production primitives
    from opensable.core.guardrails import GuardrailsEngine
    from opensable.core.structured_output import StructuredOutputParser
    from opensable.core.hitl import ApprovalGate, RiskLevel
    from opensable.core.checkpointing import Checkpoint, CheckpointStore
    from opensable.core.handoffs import HandoffRouter, Handoff
    from opensable.core.flows import Flow, start, listen, router

    __all__ = [
        "SableAgent",
        "load_config",
        "OpenSableConfig",
        # SDK
        "Agent",
        "Runner",
        "RunResult",
        "StreamEvent",
        "FunctionTool",
        "function_tool",
        "MCPClient",
        # Primitives
        "GuardrailsEngine",
        "StructuredOutputParser",
        "ApprovalGate",
        "RiskLevel",
        "Checkpoint",
        "CheckpointStore",
        "HandoffRouter",
        "Handoff",
        "Flow",
        "start",
        "listen",
        "router",
        "__version__",
    ]
except ImportError:
    # If core dependencies aren't installed, just expose version
    __all__ = ["__version__"]
