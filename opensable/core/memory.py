"""
Memory management for Open-Sable - ChromaDB for vectors + JSON for structured data
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages both vector and structured memory"""
    
    def __init__(self, config):
        self.config = config
        self.vector_db = None
        self.collection = None
        self.structured_memory_path = Path("./data/memory.json")
        self.structured_memory = {}
    
    async def initialize(self):
        """Initialize memory systems"""
        # Create data directory
        self.config.vector_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with new API
        self.vector_db = chromadb.PersistentClient(
            path=str(self.config.vector_db_path)
        )
        
        self.collection = self.vector_db.get_or_create_collection(
            name="opensable_memory",
            metadata={"description": "User interactions and context"}
        )
        
        # Load structured memory
        if self.structured_memory_path.exists():
            with open(self.structured_memory_path, 'r') as f:
                self.structured_memory = json.load(f)
        else:
            self.structured_memory = {}
            self._save_structured_memory()
        
        logger.info("Memory systems initialized")
    
    async def store(self, user_id: str, content: str, metadata: Optional[Dict] = None):
        """Store a memory"""
        memory_id = f"{user_id}_{datetime.now().timestamp()}"
        
        # Store in vector DB for semantic search
        self.collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[{
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }]
        )
        
        # Store in structured memory
        if user_id not in self.structured_memory:
            self.structured_memory[user_id] = {
                "preferences": {},
                "interactions": [],
                "metadata": {}
            }
        
        self.structured_memory[user_id]["interactions"].append({
            "id": memory_id,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
        # Limit interactions to prevent bloat
        max_interactions = 100
        if len(self.structured_memory[user_id]["interactions"]) > max_interactions:
            self.structured_memory[user_id]["interactions"] = \
                self.structured_memory[user_id]["interactions"][-max_interactions:]
        
        self._save_structured_memory()
        logger.debug(f"Stored memory for user {user_id}")
    
    async def recall(self, user_id: str, query: str, n_results: int = 5) -> List[Dict]:
        """Recall relevant memories using semantic search"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"user_id": user_id}
            )
            
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    memories.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results.get('distances') else None
                    })
            
            return memories
        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return []
    
    async def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        if user_id in self.structured_memory:
            return self.structured_memory[user_id].get("preferences", {})
        return {}
    
    async def set_user_preference(self, user_id: str, key: str, value: Any):
        """Set a user preference"""
        if user_id not in self.structured_memory:
            self.structured_memory[user_id] = {
                "preferences": {},
                "interactions": [],
                "metadata": {}
            }
        
        self.structured_memory[user_id]["preferences"][key] = value
        self._save_structured_memory()
    
    async def cleanup_old_memories(self):
        """Remove memories older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.config.memory_retention_days)
        
        for user_id in list(self.structured_memory.keys()):
            interactions = self.structured_memory[user_id]["interactions"]
            filtered = [
                i for i in interactions
                if datetime.fromisoformat(i["timestamp"]) > cutoff_date
            ]
            self.structured_memory[user_id]["interactions"] = filtered
        
        self._save_structured_memory()
        logger.info(f"Cleaned up memories older than {self.config.memory_retention_days} days")
    
    def _save_structured_memory(self):
        """Save structured memory to disk"""
        self.structured_memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.structured_memory_path, 'w') as f:
            json.dump(self.structured_memory, f, indent=2)
    
    async def close(self):
        """Cleanup on shutdown"""
        self._save_structured_memory()
        logger.info("Memory manager closed")
