"""
Advanced Memory System - Episodic, Semantic, and Working Memory for AGI.

Features:
- Episodic memory (autobiographical experiences)
- Semantic memory (factual knowledge)
- Working memory (active context)
- Memory consolidation (short-term → long-term)
- Memory retrieval with context
- Forgetting and memory decay
- Memory importance scoring
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory."""
    EPISODIC = "episodic"      # Personal experiences
    SEMANTIC = "semantic"       # Factual knowledge
    PROCEDURAL = "procedural"   # How-to knowledge
    WORKING = "working"         # Active context


class MemoryCategory(Enum):
    """Memory content categories for auto-organization."""
    FACT = "fact"               # General facts and information
    PREFERENCE = "preference"   # User preferences and likes
    TASK = "task"               # To-dos and actions
    CONTACT = "contact"         # People and relationships
    LOCATION = "location"       # Places and addresses
    EVENT = "event"             # Appointments and meetings
    SKILL = "skill"             # Learned skills and procedures
    GOAL = "goal"               # Long-term goals and objectives
    CONVERSATION = "conversation"  # General chat history
    OTHER = "other"             # Uncategorized


class MemoryImportance(Enum):
    """Memory importance levels."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


@dataclass
class Memory:
    """Individual memory unit."""
    memory_id: str
    memory_type: MemoryType
    content: str
    context: Dict[str, Any]
    importance: MemoryImportance
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    decay_factor: float = 1.0  # 1.0 = fresh, 0.0 = forgotten
    embedding: Optional[List[float]] = None
    associations: List[str] = field(default_factory=list)  # Related memory IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    category: Optional[MemoryCategory] = None  # Auto-categorization
    
    def access(self):
        """Record memory access."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        # Strengthen memory on access
        self.decay_factor = min(1.0, self.decay_factor + 0.1)
    
    def categorize(self, llm_func=None) -> MemoryCategory:
        """Auto-categorize memory content.
        
        Args:
            llm_func: Optional LLM function for smart categorization
        
        Returns:
            Detected category
        """
        content_lower = self.content.lower()
        
        # Pattern-based categorization
        if any(word in content_lower for word in ['like', 'prefer', 'favorite', 'love', 'hate']):
            return MemoryCategory.PREFERENCE
        
        if any(word in content_lower for word in ['todo', 'task', 'need to', 'must', 'should', 'remember to']):
            return MemoryCategory.TASK
        
        if any(word in content_lower for word in ['meet', 'call', 'email', 'person', 'friend', 'colleague']):
            return MemoryCategory.CONTACT
        
        if any(word in content_lower for word in ['address', 'location', 'place', 'at ', 'street', 'city']):
            return MemoryCategory.LOCATION
        
        if any(word in content_lower for word in ['meeting', 'appointment', 'schedule', 'calendar', 'event', 'tomorrow', 'next week']):
            return MemoryCategory.EVENT
        
        if any(word in content_lower for word in ['how to', 'learn', 'skill', 'ability', 'can do']):
            return MemoryCategory.SKILL
        
        if any(word in content_lower for word in ['goal', 'want to', 'plan to', 'achieve', 'objective']):
            return MemoryCategory.GOAL
        
        if any(word in content_lower for word in ['fact', 'information', 'is', 'are', 'was', 'were']):
            return MemoryCategory.FACT
        
        # Default to conversation
        return MemoryCategory.CONVERSATION
    
    def decay(self, time_delta: timedelta):
        """Apply memory decay over time."""
        # Decay rate depends on importance
        decay_rate = {
            MemoryImportance.CRITICAL: 0.01,
            MemoryImportance.HIGH: 0.05,
            MemoryImportance.MEDIUM: 0.1,
            MemoryImportance.LOW: 0.2,
            MemoryImportance.TRIVIAL: 0.3
        }
        
        rate = decay_rate[self.importance]
        days = time_delta.total_seconds() / 86400
        self.decay_factor = max(0.0, self.decay_factor - (rate * days))
    
    def is_forgotten(self, threshold: float = 0.1) -> bool:
        """Check if memory is forgotten."""
        return self.decay_factor < threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory_id': self.memory_id,
            'memory_type': self.memory_type.value,
            'content': self.content,
            'context': self.context,
            'importance': self.importance.name,
            'timestamp': self.timestamp.isoformat(),
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'decay_factor': self.decay_factor,
            'associations': self.associations,
            'metadata': self.metadata,
            'category': self.category.value if self.category else None
        }


class EpisodicMemory:
    """
    Episodic memory - stores personal experiences and events.
    
    Autobiographical memory of specific events with temporal and spatial context.
    """
    
    def __init__(self, max_size: int = 10000):
        self.memories: Dict[str, Memory] = {}
        self.max_size = max_size
        self.timeline: List[str] = []  # Chronological order
        self.categories: Dict[MemoryCategory, List[str]] = {}  # Category index
    
    def store(
        self,
        event: str,
        context: Dict[str, Any],
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        auto_categorize: bool = True
    ) -> str:
        """
        Store an episodic memory.
        
        Args:
            event: Event description
            context: Event context (location, participants, etc.)
            importance: Memory importance
            auto_categorize: Automatically categorize the memory
        
        Returns:
            Memory ID
        """
        memory_id = self._generate_id(event, context)
        
        memory = Memory(
            memory_id=memory_id,
            memory_type=MemoryType.EPISODIC,
            content=event,
            context=context,
            importance=importance,
            timestamp=datetime.utcnow()
        )
        
        # Auto-categorize
        if auto_categorize:
            memory.category = memory.categorize()
            
            # Add to category index
            if memory.category not in self.categories:
                self.categories[memory.category] = []
            self.categories[memory.category].append(memory_id)
        
        self.memories[memory_id] = memory
        self.timeline.append(memory_id)
        
        # Evict if over size
        if len(self.memories) > self.max_size:
            self._evict_least_important()
        
        logger.debug(f"Stored episodic memory: {memory_id}")
        return memory_id
    
    def recall_by_category(self, category: MemoryCategory, limit: int = 10) -> List[Memory]:
        """Recall memories by category.
        
        Args:
            category: Memory category to filter by
            limit: Maximum number of memories to return
        
        Returns:
            List of memories in category
        """
        if category not in self.categories:
            return []
        
        memory_ids = self.categories[category][-limit:]
        memories = [self.memories[mid] for mid in reversed(memory_ids) if mid in self.memories]
        
        # Mark as accessed
        for memory in memories:
            memory.access()
        
        return memories
    
    def recall_recent(self, n: int = 10) -> List[Memory]:
        """Recall n most recent memories."""
        recent_ids = self.timeline[-n:]
        return [self.memories[mid] for mid in reversed(recent_ids) if mid in self.memories]
    
    def recall_by_timeframe(
        self,
        start: datetime,
        end: datetime
    ) -> List[Memory]:
        """Recall memories within timeframe."""
        return [
            m for m in self.memories.values()
            if start <= m.timestamp <= end
        ]
    
    def recall_by_context(
        self,
        context_key: str,
        context_value: Any
    ) -> List[Memory]:
        """Recall memories matching context."""
        return [
            m for m in self.memories.values()
            if m.context.get(context_key) == context_value
        ]
    
    def _generate_id(self, event: str, context: Dict[str, Any]) -> str:
        """Generate unique memory ID."""
        content = f"{event}_{json.dumps(context, sort_keys=True)}_{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _evict_least_important(self):
        """Evict least important/oldest memory."""
        if not self.memories:
            return
        
        # Find memory with lowest importance * decay_factor
        min_score = float('inf')
        min_id = None
        
        for mid, mem in self.memories.items():
            score = mem.importance.value * mem.decay_factor
            if score < min_score:
                min_score = score
                min_id = mid
        
        if min_id:
            del self.memories[min_id]
            if min_id in self.timeline:
                self.timeline.remove(min_id)


class SemanticMemory:
    """
    Semantic memory - stores factual knowledge and concepts.
    
    General world knowledge independent of personal experience.
    """
    
    def __init__(self, max_size: int = 50000):
        self.memories: Dict[str, Memory] = {}
        self.max_size = max_size
        self.concept_index: Dict[str, List[str]] = {}  # concept -> memory_ids
    
    def store(
        self,
        fact: str,
        concepts: List[str],
        context: Optional[Dict[str, Any]] = None,
        importance: MemoryImportance = MemoryImportance.MEDIUM
    ) -> str:
        """
        Store semantic knowledge.
        
        Args:
            fact: Factual statement
            concepts: Related concepts/topics
            context: Optional context
            importance: Knowledge importance
        
        Returns:
            Memory ID
        """
        memory_id = self._generate_id(fact)
        
        memory = Memory(
            memory_id=memory_id,
            memory_type=MemoryType.SEMANTIC,
            content=fact,
            context=context or {},
            importance=importance,
            timestamp=datetime.utcnow(),
            metadata={'concepts': concepts}
        )
        
        self.memories[memory_id] = memory
        
        # Index by concepts
        for concept in concepts:
            if concept not in self.concept_index:
                self.concept_index[concept] = []
            self.concept_index[concept].append(memory_id)
        
        # Evict if over size
        if len(self.memories) > self.max_size:
            self._evict_least_important()
        
        logger.debug(f"Stored semantic memory: {memory_id}")
        return memory_id
    
    def recall_by_concept(self, concept: str) -> List[Memory]:
        """Recall all knowledge about a concept."""
        memory_ids = self.concept_index.get(concept, [])
        return [self.memories[mid] for mid in memory_ids if mid in self.memories]
    
    def recall_by_query(self, query: str, top_k: int = 5) -> List[Memory]:
        """
        Recall knowledge relevant to query.
        
        Simple keyword matching - could be enhanced with embeddings.
        """
        query_lower = query.lower()
        scored_memories = []
        
        for memory in self.memories.values():
            content_lower = memory.content.lower()
            concepts = memory.metadata.get('concepts', [])
            
            # Score based on keyword overlap
            score = 0.0
            for word in query_lower.split():
                if word in content_lower:
                    score += 1.0
                if word in ' '.join(concepts).lower():
                    score += 0.5
            
            if score > 0:
                scored_memories.append((score, memory))
        
        # Sort by score
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        
        return [mem for _, mem in scored_memories[:top_k]]
    
    def update_knowledge(
        self,
        memory_id: str,
        new_content: str
    ):
        """Update existing knowledge."""
        if memory_id in self.memories:
            self.memories[memory_id].content = new_content
            self.memories[memory_id].timestamp = datetime.utcnow()
    
    def _generate_id(self, fact: str) -> str:
        """Generate unique memory ID."""
        return hashlib.sha256(fact.encode()).hexdigest()[:16]
    
    def _evict_least_important(self):
        """Evict least important knowledge."""
        if not self.memories:
            return
        
        min_score = float('inf')
        min_id = None
        
        for mid, mem in self.memories.items():
            score = mem.importance.value * mem.decay_factor * (1 + mem.access_count * 0.1)
            if score < min_score:
                min_score = score
                min_id = mid
        
        if min_id:
            # Remove from concept index
            mem = self.memories[min_id]
            for concept in mem.metadata.get('concepts', []):
                if concept in self.concept_index:
                    self.concept_index[concept].remove(min_id)
            
            del self.memories[min_id]


class WorkingMemory:
    """
    Working memory - active context and current task information.
    
    Limited capacity, fast access, volatile.
    """
    
    def __init__(self, capacity: int = 7):  # Miller's 7±2
        self.capacity = capacity
        self.active_items: List[Memory] = []
        self.context: Dict[str, Any] = {}
    
    def add(self, content: str, context: Optional[Dict[str, Any]] = None):
        """Add item to working memory."""
        memory = Memory(
            memory_id=f"wm_{len(self.active_items)}",
            memory_type=MemoryType.WORKING,
            content=content,
            context=context or {},
            importance=MemoryImportance.HIGH,
            timestamp=datetime.utcnow()
        )
        
        self.active_items.append(memory)
        
        # Evict oldest if over capacity
        if len(self.active_items) > self.capacity:
            self.active_items.pop(0)
    
    def get_all(self) -> List[Memory]:
        """Get all items in working memory."""
        return self.active_items
    
    def clear(self):
        """Clear working memory."""
        self.active_items.clear()
        self.context.clear()
    
    def update_context(self, key: str, value: Any):
        """Update context."""
        self.context[key] = value
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        return self.context.copy()


class MemoryConsolidator:
    """
    Consolidates short-term memories into long-term storage.
    
    Simulates sleep-like memory consolidation process.
    """
    
    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory
    ):
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
    
    async def consolidate(
        self,
        working_memory: WorkingMemory,
        consolidation_threshold: float = 0.5
    ):
        """
        Consolidate working memory into long-term memory.
        
        Args:
            working_memory: Working memory to consolidate
            consolidation_threshold: Importance threshold for consolidation
        """
        logger.info("Starting memory consolidation...")
        
        for memory in working_memory.get_all():
            # Determine if memory should be consolidated
            importance_score = self._assess_importance(memory)
            
            if importance_score >= consolidation_threshold:
                # Decide memory type
                if self._is_episodic(memory):
                    # Store as episodic
                    self.episodic_memory.store(
                        event=memory.content,
                        context=memory.context,
                        importance=self._importance_from_score(importance_score)
                    )
                    logger.debug(f"Consolidated episodic memory: {memory.content[:50]}...")
                else:
                    # Extract facts and store as semantic
                    facts = self._extract_facts(memory.content)
                    for fact in facts:
                        self.semantic_memory.store(
                            fact=fact,
                            concepts=self._extract_concepts(fact),
                            importance=self._importance_from_score(importance_score)
                        )
                    logger.debug(f"Consolidated {len(facts)} semantic facts")
        
        # Clear working memory after consolidation
        working_memory.clear()
        
        logger.info("Memory consolidation completed")
    
    def _assess_importance(self, memory: Memory) -> float:
        """Assess memory importance (0-1)."""
        # Simple heuristic - could be enhanced with LLM
        score = 0.5
        
        # Longer content tends to be more important
        if len(memory.content) > 100:
            score += 0.2
        
        # Has context
        if memory.context:
            score += 0.1
        
        # Recent
        age = datetime.utcnow() - memory.timestamp
        if age < timedelta(hours=1):
            score += 0.2
        
        return min(1.0, score)
    
    def _is_episodic(self, memory: Memory) -> bool:
        """Determine if memory is episodic."""
        # Check for personal/temporal indicators
        indicators = ['I ', 'we ', 'my ', 'our ', 'today', 'yesterday']
        content_lower = memory.content.lower()
        return any(ind in content_lower for ind in indicators)
    
    def _extract_facts(self, content: str) -> List[str]:
        """Extract factual statements."""
        # Simple sentence splitting - could be enhanced
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        return sentences
    
    def _extract_concepts(self, fact: str) -> List[str]:
        """Extract concepts from fact."""
        # Simple word extraction - could be enhanced with NER
        words = fact.split()
        # Extract capitalized words and important nouns (simplified)
        concepts = [w for w in words if w and (w[0].isupper() or len(w) > 8)]
        return concepts[:5]  # Limit to 5 concepts
    
    def _importance_from_score(self, score: float) -> MemoryImportance:
        """Convert score to importance level."""
        if score >= 0.9:
            return MemoryImportance.CRITICAL
        elif score >= 0.7:
            return MemoryImportance.HIGH
        elif score >= 0.5:
            return MemoryImportance.MEDIUM
        elif score >= 0.3:
            return MemoryImportance.LOW
        else:
            return MemoryImportance.TRIVIAL


class AdvancedMemorySystem:
    """
    Integrated memory system combining all memory types.
    
    Coordinates episodic, semantic, and working memory.
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        episodic_size: int = 10000,
        semantic_size: int = 50000,
        working_capacity: int = 7
    ):
        """
        Initialize advanced memory system.
        
        Args:
            storage_path: Path for persistent storage
            episodic_size: Episodic memory size
            semantic_size: Semantic memory size
            working_capacity: Working memory capacity
        """
        self.episodic = EpisodicMemory(max_size=episodic_size)
        self.semantic = SemanticMemory(max_size=semantic_size)
        self.working = WorkingMemory(capacity=working_capacity)
        self.consolidator = MemoryConsolidator(self.episodic, self.semantic)
        
        self.storage_path = storage_path or Path("./data/advanced_memory.json")
        self._load_memory()
        
        # Background consolidation task
        self._consolidation_task = None
    
    async def initialize(self):
        """Initialize the memory system (async compatibility)"""
        # Already initialized in __init__, this is for async compatibility
        logger.info("Advanced memory system initialized")
        return True
    
    async def store_memory(
        self,
        memory_type: MemoryType,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        importance: MemoryImportance = MemoryImportance.MEDIUM
    ) -> str:
        """
        Store a memory (async wrapper for compatibility).
        
        Args:
            memory_type: Type of memory (episodic, semantic, etc.)
            content: Memory content
            context: Context dictionary
            importance: Memory importance level
            
        Returns:
            Memory ID
        """
        context = context or {}
        
        if memory_type == MemoryType.EPISODIC:
            return self.store_experience(content, context, importance)
        elif memory_type == MemoryType.SEMANTIC:
            concepts = context.get('concepts', [])
            return self.store_knowledge(content, concepts, importance)
        elif memory_type == MemoryType.WORKING:
            self.add_to_working_memory(content, context)
            return "working_memory"
        else:
            # Default to episodic
            return self.store_experience(content, context, importance)
    
    async def retrieve_memories(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Memory]:
        """
        Retrieve memories (async wrapper).
        
        Args:
            query: Search query
            memory_type: Filter by memory type
            limit: Maximum results
            **kwargs: Additional filters
            
        Returns:
            List of memories
        """
        if memory_type == MemoryType.EPISODIC or memory_type is None:
            return self.recall_experiences(query=query, n=limit)
        elif memory_type == MemoryType.SEMANTIC:
            return self.recall_knowledge(concepts=[query] if query else [], n=limit)
        else:
            return []
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics (async wrapper)"""
        semantic_count = len(getattr(self.semantic, 'knowledge_base', {})) or len(getattr(self.semantic, 'memories', {}))
        return {
            'total_memories': len(self.episodic.memories) + semantic_count,
            'episodic_count': len(self.episodic.memories),
            'semantic_count': semantic_count,
            'working_count': len(self.working.active_items)
        }
    
    def store_experience(
        self,
        event: str,
        context: Dict[str, Any],
        importance: MemoryImportance = MemoryImportance.MEDIUM
    ) -> str:
        """Store an experience in episodic memory."""
        return self.episodic.store(event, context, importance)
    
    def store_knowledge(
        self,
        fact: str,
        concepts: List[str],
        importance: MemoryImportance = MemoryImportance.MEDIUM
    ) -> str:
        """Store knowledge in semantic memory."""
        return self.semantic.store(fact, concepts, None, importance)
    
    def add_to_working_memory(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Add item to working memory."""
        self.working.add(content, context)
    
    def recall_experiences(
        self,
        query: Optional[str] = None,
        timeframe: Optional[Tuple[datetime, datetime]] = None,
        context_filter: Optional[Dict[str, Any]] = None,
        n: int = 10
    ) -> List[Memory]:
        """
        Recall experiences from episodic memory.
        
        Args:
            query: Search query (not implemented in simple version)
            timeframe: (start, end) datetime tuple
            context_filter: Context key-value to filter
            n: Max results
        
        Returns:
            List of memories
        """
        if timeframe:
            return self.episodic.recall_by_timeframe(timeframe[0], timeframe[1])
        elif context_filter:
            key, value = list(context_filter.items())[0]
            return self.episodic.recall_by_context(key, value)
        else:
            return self.episodic.recall_recent(n)
    
    def recall_knowledge(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Memory]:
        """Recall knowledge from semantic memory."""
        return self.semantic.recall_by_query(query, top_k)
    
    def get_working_memory(self) -> List[Memory]:
        """Get current working memory contents."""
        return self.working.get_all()
    
    async def consolidate_memories(self):
        """Consolidate working memory into long-term memory."""
        await self.consolidator.consolidate(self.working)
        self._save_memory()
    
    def apply_decay(self):
        """Apply time-based decay to all memories."""
        current_time = datetime.utcnow()
        
        # Decay episodic memories
        for memory in self.episodic.memories.values():
            time_delta = current_time - memory.last_accessed if memory.last_accessed else current_time - memory.timestamp
            memory.decay(time_delta)
        
        # Decay semantic memories
        for memory in self.semantic.memories.values():
            time_delta = current_time - memory.last_accessed if memory.last_accessed else current_time - memory.timestamp
            memory.decay(time_delta)
        
        logger.debug("Applied memory decay")
    
    def forget_old_memories(self, threshold: float = 0.1):
        """Remove forgotten memories below threshold."""
        # Remove from episodic
        forgotten_episodic = [
            mid for mid, mem in self.episodic.memories.items()
            if mem.is_forgotten(threshold)
        ]
        for mid in forgotten_episodic:
            del self.episodic.memories[mid]
            if mid in self.episodic.timeline:
                self.episodic.timeline.remove(mid)
        
        # Remove from semantic
        forgotten_semantic = [
            mid for mid, mem in self.semantic.memories.items()
            if mem.is_forgotten(threshold)
        ]
        for mid in forgotten_semantic:
            del self.semantic.memories[mid]
        
        logger.info(f"Forgot {len(forgotten_episodic)} episodic and {len(forgotten_semantic)} semantic memories")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            'episodic_count': len(self.episodic.memories),
            'semantic_count': len(self.semantic.memories),
            'working_count': len(self.working.active_items),
            'total_concepts': len(self.semantic.concept_index),
            'avg_episodic_decay': np.mean([m.decay_factor for m in self.episodic.memories.values()]) if self.episodic.memories else 0,
            'avg_semantic_decay': np.mean([m.decay_factor for m in self.semantic.memories.values()]) if self.semantic.memories else 0
        }
    
    async def start_background_consolidation(self, interval_hours: int = 1):
        """Start background consolidation task."""
        async def consolidation_loop():
            while True:
                await asyncio.sleep(interval_hours * 3600)
                await self.consolidate_memories()
                self.apply_decay()
                self.forget_old_memories()
        
        self._consolidation_task = asyncio.create_task(consolidation_loop())
        logger.info(f"Started background consolidation (every {interval_hours}h)")
    
    async def stop_background_consolidation(self):
        """Stop background consolidation task."""
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped background consolidation")
    
    def _save_memory(self):
        """Save memories to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'episodic': {
                    mid: mem.to_dict()
                    for mid, mem in self.episodic.memories.items()
                },
                'semantic': {
                    mid: mem.to_dict()
                    for mid, mem in self.semantic.memories.items()
                },
                'episodic_timeline': self.episodic.timeline,
                'concept_index': self.semantic.concept_index
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Saved memory to disk")
        
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def _load_memory(self):
        """Load memories from disk."""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Load episodic
            for mid, mem_data in data.get('episodic', {}).items():
                memory = Memory(
                    memory_id=mem_data['memory_id'],
                    memory_type=MemoryType(mem_data['memory_type']),
                    content=mem_data['content'],
                    context=mem_data['context'],
                    importance=MemoryImportance[mem_data['importance']],
                    timestamp=datetime.fromisoformat(mem_data['timestamp']),
                    access_count=mem_data['access_count'],
                    last_accessed=datetime.fromisoformat(mem_data['last_accessed']) if mem_data.get('last_accessed') else None,
                    decay_factor=mem_data['decay_factor'],
                    associations=mem_data.get('associations', []),
                    metadata=mem_data.get('metadata', {})
                )
                self.episodic.memories[mid] = memory
            
            self.episodic.timeline = data.get('episodic_timeline', [])
            
            # Load semantic
            for mid, mem_data in data.get('semantic', {}).items():
                memory = Memory(
                    memory_id=mem_data['memory_id'],
                    memory_type=MemoryType(mem_data['memory_type']),
                    content=mem_data['content'],
                    context=mem_data['context'],
                    importance=MemoryImportance[mem_data['importance']],
                    timestamp=datetime.fromisoformat(mem_data['timestamp']),
                    access_count=mem_data['access_count'],
                    last_accessed=datetime.fromisoformat(mem_data['last_accessed']) if mem_data.get('last_accessed') else None,
                    decay_factor=mem_data['decay_factor'],
                    associations=mem_data.get('associations', []),
                    metadata=mem_data.get('metadata', {})
                )
                self.semantic.memories[mid] = memory
            
            self.semantic.concept_index = data.get('concept_index', {})
            
            logger.info(f"Loaded {len(self.episodic.memories)} episodic and {len(self.semantic.memories)} semantic memories")
        
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")


# Example usage
async def main():
    """Example advanced memory system usage."""
    
    print("=" * 50)
    print("Advanced Memory System Example")
    print("=" * 50)
    
    # Initialize memory system
    memory = AdvancedMemorySystem()
    
    # Store some episodic memories
    print("\n1. Storing episodic memories...")
    memory.store_experience(
        event="Met with the development team to discuss new features",
        context={'location': 'office', 'participants': ['Alice', 'Bob'], 'duration': '2 hours'},
        importance=MemoryImportance.HIGH
    )
    
    memory.store_experience(
        event="Had lunch at the new Italian restaurant downtown",
        context={'location': 'restaurant', 'time': 'noon', 'food': 'pasta'},
        importance=MemoryImportance.LOW
    )
    
    print("  Stored 2 episodic memories")
    
    # Store semantic knowledge
    print("\n2. Storing semantic knowledge...")
    memory.store_knowledge(
        fact="Python is a high-level programming language created by Guido van Rossum",
        concepts=['Python', 'programming', 'Guido van Rossum'],
        importance=MemoryImportance.MEDIUM
    )
    
    memory.store_knowledge(
        fact="Machine learning is a subset of artificial intelligence focused on data-driven learning",
        concepts=['machine learning', 'AI', 'data science'],
        importance=MemoryImportance.HIGH
    )
    
    print("  Stored 2 semantic facts")
    
    # Use working memory
    print("\n3. Using working memory...")
    memory.add_to_working_memory("Current task: Write documentation")
    memory.add_to_working_memory("Current context: Python project")
    memory.add_to_working_memory("Current goal: Complete by Friday")
    
    working_items = memory.get_working_memory()
    print(f"  Working memory items: {len(working_items)}")
    for item in working_items:
        print(f"    - {item.content}")
    
    # Recall experiences
    print("\n4. Recalling experiences...")
    experiences = memory.recall_experiences(n=5)
    print(f"  Found {len(experiences)} recent experiences:")
    for exp in experiences:
        print(f"    - {exp.content[:60]}...")
    
    # Recall knowledge
    print("\n5. Recalling knowledge...")
    knowledge = memory.recall_knowledge("Python programming")
    print(f"  Found {len(knowledge)} relevant facts:")
    for fact in knowledge:
        print(f"    - {fact.content[:60]}...")
    
    # Consolidate memories
    print("\n6. Consolidating working memory...")
    await memory.consolidate_memories()
    print("  Consolidation complete")
    
    # Get statistics
    print("\n7. Memory statistics...")
    stats = memory.get_stats()
    print(f"  Episodic memories: {stats['episodic_count']}")
    print(f"  Semantic memories: {stats['semantic_count']}")
    print(f"  Working memory: {stats['working_count']}")
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  Avg episodic decay: {stats['avg_episodic_decay']:.2f}")
    print(f"  Avg semantic decay: {stats['avg_semantic_decay']:.2f}")
    
    print("\n✅ Advanced memory system example completed!")


if __name__ == "__main__":
    asyncio.run(main())
