"""
LLM integration for Open-Sable - Ollama with native tool calling
Dynamic model switching based on task requirements
"""

import logging
import json
from typing import Dict, List, Any
import ollama

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

logger = logging.getLogger(__name__)


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
    # CPU/Low RAM models
    "llama3.2:3b": {"reasoning": 6, "vision": 0, "tools": 6, "speed": 9, "vram": 0},
    "gemma2:2b": {"reasoning": 5, "vision": 0, "tools": 5, "speed": 9, "vram": 0},
    "qwen2.5:3b": {"reasoning": 6, "vision": 0, "tools": 6, "speed": 9, "vram": 0},
    "phi3:3.8b": {"reasoning": 6, "vision": 0, "tools": 6, "speed": 9, "vram": 0},
    "llama3.2:1b": {"reasoning": 4, "vision": 0, "tools": 4, "speed": 10, "vram": 0},
    "qwen2.5:0.5b": {"reasoning": 3, "vision": 0, "tools": 3, "speed": 10, "vram": 0},
}


class AdaptiveLLM:
    """LLM that can switch models based on task requirements"""

    def __init__(self, config, initial_model: str):
        self.config = config
        self.current_model = initial_model
        self.base_url = config.ollama_base_url
        self.llm = self._create_llm(initial_model)
        self.available_models = []
        self._update_available_models()

    def _create_llm(self, model: str):
        """Create LangChain LLM instance"""
        return ChatOllama(
            base_url=self.base_url,
            model=model,
            temperature=0.7,
        )

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

        for model in MODEL_CAPABILITIES:
            caps = MODEL_CAPABILITIES[model]

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
            self.llm = self._create_llm(best_model)
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
        """Invoke current LLM"""
        return await self.llm.ainvoke(messages)

    def invoke(self, messages):
        """Sync invoke"""
        return self.llm.invoke(messages)

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
        try:
            client = ollama.AsyncClient(host=self.base_url)
            response = await client.chat(
                model=self.current_model,
                messages=messages,
                tools=tools,
            )
            msg = response.get("message", {})

            # Tool call path — collect ALL tool calls for parallel execution
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
            return {"tool_call": None, "tool_calls": [], "text": msg.get("content", "")}

        except Exception as e:
            logger.warning(f"Tool calling failed ({e}), falling back to plain ainvoke")
            # Fallback: use langchain ainvoke without tools
            from langchain_core.messages import HumanMessage, SystemMessage

            lc_messages = []
            for m in messages:
                if m["role"] == "system":
                    lc_messages.append(SystemMessage(content=m["content"]))
                else:
                    lc_messages.append(HumanMessage(content=m["content"]))
            resp = await self.llm.ainvoke(lc_messages)
            return {"tool_call": None, "tool_calls": [], "text": resp.content}


def get_llm(config):
    """Get LLM instance based on configuration"""
    try:
        # Auto-select model if enabled
        model_to_use = config.default_model

        if config.auto_select_model:
            from opensable.core.system_detector import auto_configure_system

            auto_config = auto_configure_system()
            recommended = auto_config["recommended_model"]

            # Verify the recommended model is actually available before using it
            try:
                client = ollama.Client(host=config.ollama_base_url)
                models = client.list()
                available = [
                    getattr(m, "model", None) or m.get("name") or m.get("model", "")
                    for m in models.get("models", [])
                ]
                if any(recommended in a or a in recommended for a in available):
                    model_to_use = recommended
                    logger.info(
                        f"Auto-selected model: {model_to_use} (tier: {auto_config['device_tier']})"
                    )
                else:
                    logger.warning(
                        f"Auto-selected model '{recommended}' not available locally. Using default: {model_to_use}"
                    )
            except Exception:
                logger.warning(f"Cannot verify model availability. Using default: {model_to_use}")

        # Return adaptive LLM that can switch models
        adaptive_llm = AdaptiveLLM(config, model_to_use)
        logger.info(f"Using adaptive LLM starting with: {model_to_use}")
        return adaptive_llm

    except Exception as e:
        logger.warning(f"Ollama not available: {e}")

        # Fallback to cloud APIs
        if config.openai_api_key:
            from langchain_openai import ChatOpenAI

            logger.info("Falling back to OpenAI")
            return ChatOpenAI(api_key=config.openai_api_key, model="gpt-3.5-turbo", temperature=0.7)
        elif config.anthropic_api_key:
            from langchain_anthropic import ChatAnthropic

            logger.info("Falling back to Anthropic")
            return ChatAnthropic(
                api_key=config.anthropic_api_key, model="claude-3-haiku-20240307", temperature=0.7
            )
        else:
            raise Exception(
                "No LLM available. Install Ollama or provide OPENAI_API_KEY/ANTHROPIC_API_KEY"
            )


async def check_ollama_models(base_url: str = "http://localhost:11434") -> list:
    """Check which models are available in Ollama"""
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
