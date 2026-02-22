"""
Open-Sable - Autonomous AI Agent Framework
AGI-inspired cognitive subsystems for autonomous agents

Version: 0.1.0-beta
"""

__version__ = "0.1.0-beta"
__author__ = "IdeoaLabs"
__license__ = "MIT"

# Core imports for convenience
try:
    from opensable.core.agent import SableAgent
    from opensable.core.config import load_config

    __all__ = [
        "SableAgent",
        "load_config",
        "__version__",
    ]
except ImportError:
    # If core dependencies aren't installed, just expose version
    __all__ = ["__version__"]
