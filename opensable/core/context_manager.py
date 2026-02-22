"""
Open-Sable Context Manager

Advanced context window management with intelligent compression,
summarization, and context retrieval for large conversations.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from opensable.core.config import Config

logger = logging.getLogger(__name__)


class ContextWindow:
    """Manages conversation context and token limits"""

    def __init__(self, config: Config):
        self.config = config
        self.max_tokens = getattr(config, "max_context_tokens", 4096)
        self.compression_threshold = getattr(config, "compression_threshold", 0.8)
        self.summary_model = getattr(config, "summary_model", "mistral")

        # Context segments
        self.system_prompt = ""
        self.compressed_history = ""
        self.recent_messages: List[Dict[str, str]] = []

        # Metadata
        self.total_messages = 0
        self.compressions_count = 0

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Simple heuristic: ~4 characters per token for English
        return len(text) // 4

    def get_total_tokens(self) -> int:
        """Get estimated total tokens in context"""
        total = 0
        total += self.estimate_tokens(self.system_prompt)
        total += self.estimate_tokens(self.compressed_history)

        for msg in self.recent_messages:
            total += self.estimate_tokens(msg.get("content", ""))

        return total

    def set_system_prompt(self, prompt: str):
        """Set system prompt"""
        self.system_prompt = prompt

    def add_message(self, role: str, content: str):
        """Add message to context"""
        self.recent_messages.append(
            {"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}
        )
        self.total_messages += 1

        # Check if compression needed
        if self._should_compress():
            asyncio.create_task(self.compress())

    def _should_compress(self) -> bool:
        """Check if context should be compressed"""
        current_tokens = self.get_total_tokens()
        threshold = int(self.max_tokens * self.compression_threshold)

        return current_tokens >= threshold

    async def compress(self):
        """Compress older messages into summary"""
        if len(self.recent_messages) < 10:
            return  # Not enough messages to compress

        logger.info("Compressing context...")

        try:
            # Take older messages (leave last 5 as recent)
            messages_to_compress = self.recent_messages[:-5]
            self.recent_messages = self.recent_messages[-5:]

            # Build summary prompt
            conversation = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in messages_to_compress]
            )

            summary_prompt = f"""Summarize the following conversation concisely while preserving key information, decisions, and context:

{conversation}

Summary:"""

            # Generate summary using LLM directly (avoid reinitializing the agent)
            from opensable.core.llm import OllamaLLM

            llm = OllamaLLM(self.config)
            resp = await llm.invoke_with_tools(
                [{"role": "user", "content": summary_prompt}],
                [],
            )
            summary = resp.get("text", "")

            # Update compressed history
            if self.compressed_history:
                self.compressed_history += f"\n\nPrevious summary:\n{summary}"
            else:
                self.compressed_history = f"Conversation summary:\n{summary}"

            self.compressions_count += 1

            logger.info(f"Context compressed. Compressions: {self.compressions_count}")

        except Exception as e:
            logger.error(f"Error compressing context: {e}", exc_info=True)

    def build_context(self) -> List[Dict[str, str]]:
        """Build full context for LLM"""
        context = []

        # System prompt
        if self.system_prompt:
            context.append({"role": "system", "content": self.system_prompt})

        # Compressed history
        if self.compressed_history:
            context.append({"role": "system", "content": self.compressed_history})

        # Recent messages
        context.extend(self.recent_messages)

        return context

    def get_context_info(self) -> Dict[str, Any]:
        """Get context information"""
        return {
            "max_tokens": self.max_tokens,
            "current_tokens": self.get_total_tokens(),
            "compression_threshold": int(self.max_tokens * self.compression_threshold),
            "total_messages": self.total_messages,
            "recent_messages_count": len(self.recent_messages),
            "compressions_count": self.compressions_count,
            "has_compressed_history": bool(self.compressed_history),
        }

    def reset(self):
        """Reset context (keep system prompt)"""
        self.compressed_history = ""
        self.recent_messages = []
        self.total_messages = 0
        self.compressions_count = 0

    def to_dict(self) -> dict:
        """Serialize to dict"""
        return {
            "system_prompt": self.system_prompt,
            "compressed_history": self.compressed_history,
            "recent_messages": self.recent_messages,
            "total_messages": self.total_messages,
            "compressions_count": self.compressions_count,
        }

    @classmethod
    def from_dict(cls, config: Config, data: dict) -> "ContextWindow":
        """Deserialize from dict"""
        context = cls(config)
        context.system_prompt = data.get("system_prompt", "")
        context.compressed_history = data.get("compressed_history", "")
        context.recent_messages = data.get("recent_messages", [])
        context.total_messages = data.get("total_messages", 0)
        context.compressions_count = data.get("compressions_count", 0)
        return context


class SemanticMemory:
    """Semantic memory for retrieving relevant past conversations"""

    def __init__(self, config: Config):
        self.config = config
        self.embeddings_enabled = getattr(config, "embeddings_enabled", False)

        # Memory store
        self.memories: List[Dict[str, Any]] = []
        self.max_memories = getattr(config, "max_memories", 100)

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add memory"""
        memory = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
            "embedding": None,  # Will be computed if embeddings enabled
        }

        self.memories.append(memory)

        # Trim old memories
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories :]

    async def retrieve_relevant(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve most relevant memories"""
        if not self.memories:
            return []

        # Simple keyword matching (upgrade to embeddings for better results)
        query_lower = query.lower()

        scored_memories = []
        for memory in self.memories:
            content_lower = memory["content"].lower()

            # Simple scoring: count matching words
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())

            matches = len(query_words & content_words)

            if matches > 0:
                scored_memories.append((matches, memory))

        # Sort by score
        scored_memories.sort(reverse=True, key=lambda x: x[0])

        # Return top k
        return [memory for _, memory in scored_memories[:top_k]]


class HybridContextManager:
    """Combines context window and semantic memory"""

    def __init__(self, config: Config):
        self.config = config
        self.context_window = ContextWindow(config)
        self.semantic_memory = SemanticMemory(config)

    def add_message(self, role: str, content: str):
        """Add message to both context and memory"""
        # Add to context window
        self.context_window.add_message(role, content)

        # Add to semantic memory (user messages only)
        if role == "user":
            self.semantic_memory.add_memory(content)

    async def build_context_with_memory(self, query: Optional[str] = None) -> List[Dict[str, str]]:
        """Build context with relevant memories"""
        context = self.context_window.build_context()

        # Retrieve relevant memories if query provided
        if query:
            relevant_memories = await self.semantic_memory.retrieve_relevant(query)

            if relevant_memories:
                # Add memories as system message
                memory_text = "Relevant past context:\n" + "\n".join(
                    [f"- {mem['content'][:200]}..." for mem in relevant_memories]
                )

                context.insert(1, {"role": "system", "content": memory_text})

        return context

    def get_info(self) -> Dict[str, Any]:
        """Get combined context information"""
        return {
            "context_window": self.context_window.get_context_info(),
            "semantic_memory": {
                "total_memories": len(self.semantic_memory.memories),
                "max_memories": self.semantic_memory.max_memories,
            },
        }


if __name__ == "__main__":
    from opensable.core.config import load_config

    config = load_config()
    manager = HybridContextManager(config)

    # Test
    manager.add_message("user", "What is the capital of France?")
    manager.add_message("assistant", "The capital of France is Paris.")
    manager.add_message("user", "What about Germany?")
    manager.add_message("assistant", "The capital of Germany is Berlin.")

    context = asyncio.run(manager.build_context_with_memory("capitals of Europe"))

    print("Context:")
    for msg in context:
        print(f"{msg['role']}: {msg['content'][:100]}...")

    print(f"\nContext info: {json.dumps(manager.get_info(), indent=2)}")
