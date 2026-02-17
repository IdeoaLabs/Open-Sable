"""
RAG Engine — Retrieval-Augmented Generation pipeline for SableCore.

Core module that wraps the RAG skill with a clean interface for the agent.

Features:
- Document ingestion (text, PDF, markdown, HTML)
- Chunking strategies (fixed, sentence, semantic)
- Vector embeddings via ChromaDB
- Semantic search with relevance scoring
- Context assembly for LLM prompts
- Collection management
"""
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.info("ChromaDB not installed. RAG will use in-memory fallback.")


@dataclass
class Document:
    """A document to be ingested into the RAG store."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Chunk:
    """A chunk of a document."""
    chunk_id: str
    doc_id: str
    content: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search result."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Chunker:
    """Split documents into chunks using various strategies."""

    @staticmethod
    def fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    @staticmethod
    def by_sentences(text: str, max_sentences: int = 5) -> List[str]:
        """Split text into chunks of N sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = " ".join(sentences[i:i + max_sentences])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    @staticmethod
    def by_paragraphs(text: str) -> List[str]:
        """Split text by paragraph boundaries."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Provides document storage, vector search, and context retrieval
    for grounding the agent's responses in factual data.
    """

    def __init__(self, config=None, collection_name: str = "sablecore_docs"):
        self.config = config
        self.collection_name = collection_name
        self.chunker = Chunker()
        self._documents: Dict[str, Document] = {}
        self._chunks: List[Chunk] = []

        # ChromaDB backend
        self._client = None
        self._collection = None

        if CHROMADB_AVAILABLE:
            try:
                persist_dir = str(
                    Path(__file__).parent.parent.parent / "data" / "vectordb"
                )
                self._client = chromadb.PersistentClient(path=persist_dir)
                self._collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    f"📚 RAG Engine initialized with ChromaDB "
                    f"({self._collection.count()} vectors)"
                )
            except Exception as e:
                logger.warning(f"ChromaDB init failed, using in-memory: {e}")
        else:
            logger.info("📚 RAG Engine initialized (in-memory mode)")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest(
        self,
        content: str,
        source: str = "",
        metadata: Optional[Dict] = None,
        chunk_strategy: str = "sentences",
        chunk_size: int = 500,
    ) -> Document:
        """Ingest a document: chunk it and store embeddings."""
        doc_id = hashlib.sha256(content[:500].encode()).hexdigest()[:16]
        doc = Document(
            doc_id=doc_id,
            content=content,
            source=source,
            metadata=metadata or {},
        )
        self._documents[doc_id] = doc

        # Chunk the document
        if chunk_strategy == "fixed":
            raw_chunks = self.chunker.fixed_size(content, chunk_size)
        elif chunk_strategy == "paragraphs":
            raw_chunks = self.chunker.by_paragraphs(content)
        else:
            raw_chunks = self.chunker.by_sentences(content)

        chunks = []
        for i, text in enumerate(raw_chunks):
            chunk = Chunk(
                chunk_id=f"{doc_id}_c{i}",
                doc_id=doc_id,
                content=text,
                index=i,
                metadata={**doc.metadata, "source": source},
            )
            chunks.append(chunk)
        self._chunks.extend(chunks)

        # Store in ChromaDB if available
        if self._collection is not None and chunks:
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=[c.content for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )

        logger.info(f"📄 Ingested '{source}': {len(chunks)} chunks from {len(content)} chars")
        return doc

    async def ingest_file(self, file_path: str, **kwargs) -> Optional[Document]:
        """Ingest a file from disk."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            content = path.read_text(errors="replace")
            return await self.ingest(
                content=content,
                source=str(path),
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self, query: str, top_k: int = 5
    ) -> List[SearchResult]:
        """Search for relevant chunks using vector similarity."""
        # ChromaDB search
        if self._collection is not None and self._collection.count() > 0:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self._collection.count()),
                )
                search_results = []
                for i, doc in enumerate(results["documents"][0]):
                    score = 1.0
                    if results.get("distances") and results["distances"][0]:
                        score = 1.0 - results["distances"][0][i]
                    search_results.append(SearchResult(
                        chunk_id=results["ids"][0][i],
                        content=doc,
                        score=score,
                        metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                    ))
                return search_results
            except Exception as e:
                logger.warning(f"ChromaDB search failed, using fallback: {e}")

        # In-memory keyword fallback
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Simple keyword-based search fallback."""
        query_words = set(query.lower().split())
        scored = []
        for chunk in self._chunks:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(
                chunk_id=c.chunk_id,
                content=c.content,
                score=s,
                metadata=c.metadata,
            )
            for c, s in scored[:top_k]
        ]

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    async def get_context(
        self, query: str, max_tokens: int = 2000, top_k: int = 5
    ) -> str:
        """
        Retrieve relevant context for an LLM prompt.

        Returns a formatted string of the top-k matching chunks,
        truncated to approximately max_tokens.
        """
        results = await self.search(query, top_k=top_k)
        if not results:
            return ""

        context_parts = []
        total_len = 0
        for r in results:
            # Rough token estimate: 1 token ≈ 4 chars
            chunk_tokens = len(r.content) // 4
            if total_len + chunk_tokens > max_tokens:
                break
            context_parts.append(r.content)
            total_len += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return RAG engine statistics."""
        return {
            "documents": len(self._documents),
            "chunks": len(self._chunks),
            "vector_count": self._collection.count() if self._collection else 0,
            "chromadb_available": CHROMADB_AVAILABLE,
            "collection": self.collection_name,
        }

    async def clear(self):
        """Clear all stored data."""
        self._documents.clear()
        self._chunks.clear()
        if self._client is not None:
            try:
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name
                )
            except Exception as e:
                logger.warning(f"Failed to clear ChromaDB collection: {e}")
        logger.info("🗑️ RAG engine cleared")
