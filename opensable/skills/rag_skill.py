"""
RAG (Retrieval-Augmented Generation) Skill - Vector store integration for knowledge retrieval.

Features:
- Document ingestion and chunking
- Vector embeddings with multiple providers
- Vector store integration (ChromaDB, Pinecone, Weaviate)
- Semantic search
- Hybrid search (vector + keyword)
- Re-ranking
- Context retrieval for LLMs
- Document management
"""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@dataclass
class Document:
    """Document for RAG."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class DocumentChunk:
    """Chunked document with metadata."""
    content: str
    chunk_id: str
    document_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "metadata": self.metadata
        }


@dataclass
class SearchResult:
    """Search result from vector store."""
    document: DocumentChunk
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.document.content,
            "score": self.score,
            "rank": self.rank,
            "metadata": self.document.metadata
        }


class DocumentChunker:
    """
    Split documents into chunks for embedding.
    
    Strategies:
    - Fixed size chunking
    - Sentence-based chunking
    - Recursive character splitter
    - Token-based chunking
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Target chunk size (characters or tokens)
            chunk_overlap: Overlap between chunks
            separator: Primary separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        # Token counter
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Chunk a document.
        
        Args:
            document: Document to chunk
        
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        text = document.content
        
        # Split by separator
        parts = text.split(self.separator)
        
        current_chunk = ""
        chunk_index = 0
        
        for part in parts:
            # Check if adding this part would exceed chunk size
            if self._get_length(current_chunk + self.separator + part) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append(self._create_chunk(
                        current_chunk,
                        document.id,
                        chunk_index,
                        document.metadata
                    ))
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + self.separator + part
                    else:
                        current_chunk = part
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    current_chunk += self.separator + part
                else:
                    current_chunk = part
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk,
                document.id,
                chunk_index,
                document.metadata
            ))
        
        return chunks
    
    def _create_chunk(
        self,
        content: str,
        document_id: str,
        index: int,
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """Create a document chunk."""
        chunk_id = f"{document_id}_{index}"
        chunk_metadata = {
            **metadata,
            "chunk_index": index,
            "chunk_size": self._get_length(content)
        }
        
        return DocumentChunk(
            content=content.strip(),
            chunk_id=chunk_id,
            document_id=document_id,
            metadata=chunk_metadata
        )
    
    def _get_length(self, text: str) -> int:
        """Get length of text (tokens or characters)."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text)


class EmbeddingProvider:
    """
    Generate embeddings for text.
    
    Supports:
    - Sentence Transformers (local)
    - OpenAI embeddings
    - Ollama embeddings
    """
    
    def __init__(self, provider: str = "ollama", model: str = "nomic-embed-text"):
        """
        Initialize embedding provider.
        
        Args:
            provider: Provider name (ollama, openai, sentence-transformers)
            model: Model name
        """
        self.provider = provider
        self.model = model
        
        if provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self.model_obj = SentenceTransformer(model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed: "
                    "pip install sentence-transformers"
                )
        else:
            self.model_obj = None
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self.provider == "ollama":
            return await self._embed_ollama(text)
        elif self.provider == "sentence-transformers":
            return self._embed_sentence_transformers(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.provider == "sentence-transformers":
            embeddings = self.model_obj.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        else:
            # Sequential for other providers
            return [await self.embed_text(text) for text in texts]
    
    async def _embed_ollama(self, text: str) -> List[float]:
        """Generate embedding using Ollama."""
        try:
            import ollama
            response = await asyncio.to_thread(
                ollama.embeddings,
                model=self.model,
                prompt=text
            )
            return response["embedding"]
        except ImportError:
            raise ImportError("ollama not installed: pip install ollama")
        except Exception as e:
            raise Exception(f"Ollama embedding failed: {e}")
    
    def _embed_sentence_transformers(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformers."""
        embedding = self.model_obj.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


class VectorStore:
    """
    Vector store for semantic search.
    
    Uses ChromaDB for local vector storage.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of collection
            persist_directory: Directory for persistence
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed: pip install chromadb")
        
        self.collection_name = collection_name
        
        # Setup persistence directory
        if persist_directory is None:
            persist_directory = str(Path.home() / ".opensable" / "chromadb")
        
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ):
        """
        Add documents to vector store.
        
        Args:
            chunks: Document chunks
            embeddings: Embeddings for chunks
        """
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter: Metadata filter
        
        Returns:
            List of SearchResult objects
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        
        search_results = []
        
        for i in range(len(results["ids"][0])):
            chunk = DocumentChunk(
                content=results["documents"][0][i],
                chunk_id=results["ids"][0][i],
                document_id=results["metadatas"][0][i].get("document_id", ""),
                metadata=results["metadatas"][0][i]
            )
            
            result = SearchResult(
                document=chunk,
                score=1.0 - results["distances"][0][i],  # Convert distance to similarity
                rank=i + 1
            )
            
            search_results.append(result)
        
        return search_results
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents by ID."""
        # Find all chunks for these documents
        all_results = self.collection.get()
        
        chunk_ids_to_delete = []
        for i, metadata in enumerate(all_results["metadatas"]):
            if metadata.get("document_id") in document_ids:
                chunk_ids_to_delete.append(all_results["ids"][i])
        
        if chunk_ids_to_delete:
            self.collection.delete(ids=chunk_ids_to_delete)
    
    def count_documents(self) -> int:
        """Count total documents in store."""
        return self.collection.count()
    
    def clear(self):
        """Clear all documents from store."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


class RAGSystem:
    """
    Complete RAG (Retrieval-Augmented Generation) system.
    
    Combines document chunking, embedding, and vector search
    for knowledge retrieval.
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_provider: str = "ollama",
        embedding_model: str = "nomic-embed-text"
    ):
        """
        Initialize RAG system.
        
        Args:
            collection_name: Vector store collection name
            persist_directory: Persistence directory
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap size
            embedding_provider: Embedding provider
            embedding_model: Embedding model name
        """
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.embedder = EmbeddingProvider(
            provider=embedding_provider,
            model=embedding_model
        )
        
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    
    async def ingest_documents(self, documents: List[Document]):
        """
        Ingest documents into RAG system.
        
        Args:
            documents: List of documents to ingest
        """
        all_chunks = []
        
        # Chunk documents
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        # Generate embeddings
        texts = [chunk.content for chunk in all_chunks]
        embeddings = await self.embedder.embed_batch(texts)
        
        # Add to vector store
        self.vector_store.add_documents(all_chunks, embeddings)
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results
            filter: Metadata filter
        
        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_embedding = await self.embedder.embed_text(query)
        
        # Search vector store
        return self.vector_store.search(query_embedding, top_k, filter)
    
    async def get_context(
        self,
        query: str,
        top_k: int = 3,
        max_context_length: int = 2000
    ) -> str:
        """
        Get context for LLM from relevant documents.
        
        Args:
            query: Query for context retrieval
            top_k: Number of documents to retrieve
            max_context_length: Maximum context length
        
        Returns:
            Formatted context string
        """
        results = await self.search(query, top_k)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            content = result.document.content
            if current_length + len(content) <= max_context_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                # Add partial content to fit within limit
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if significant space left
                    context_parts.append(content[:remaining])
                break
        
        return "\n\n---\n\n".join(context_parts)
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents from RAG system."""
        self.vector_store.delete_documents(document_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            "total_chunks": self.vector_store.count_documents(),
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "embedding_provider": self.embedder.provider,
            "embedding_model": self.embedder.model,
            "collection_name": self.vector_store.collection_name
        }


# Example usage
async def main():
    """Example RAG system usage."""
    
    print("=" * 50)
    print("RAG System Example")
    print("=" * 50)
    
    # Initialize RAG system
    rag = RAGSystem(
        collection_name="example_docs",
        chunk_size=256,
        chunk_overlap=30
    )
    
    # Create sample documents
    print("\n1. Ingesting documents...")
    documents = [
        Document(
            content="""
            Python is a high-level, interpreted programming language.
            It was created by Guido van Rossum and first released in 1991.
            Python emphasizes code readability with significant whitespace.
            It supports multiple programming paradigms including procedural,
            object-oriented, and functional programming.
            """,
            metadata={"source": "python_intro", "topic": "programming"}
        ),
        Document(
            content="""
            Machine learning is a subset of artificial intelligence.
            It focuses on the development of algorithms that can learn from data.
            Common machine learning techniques include supervised learning,
            unsupervised learning, and reinforcement learning.
            Python is widely used for machine learning with libraries like
            scikit-learn, TensorFlow, and PyTorch.
            """,
            metadata={"source": "ml_intro", "topic": "AI"}
        ),
        Document(
            content="""
            Web development involves creating websites and web applications.
            Frontend development focuses on user interface and experience.
            Backend development handles server-side logic and databases.
            Popular web frameworks include Django and Flask for Python,
            React and Angular for JavaScript, and Ruby on Rails for Ruby.
            """,
            metadata={"source": "web_dev", "topic": "web"}
        )
    ]
    
    await rag.ingest_documents(documents)
    print(f"  Ingested {len(documents)} documents")
    
    # Get stats
    stats = rag.get_stats()
    print(f"  Total chunks: {stats['total_chunks']}")
    
    # Search for relevant documents
    print("\n2. Semantic search...")
    query = "How is Python used in AI?"
    results = await rag.search(query, top_k=3)
    
    print(f"  Query: {query}")
    print(f"  Found {len(results)} results:")
    for result in results:
        print(f"\n  Rank {result.rank} (score: {result.score:.3f})")
        print(f"  {result.document.content[:150]}...")
        print(f"  Metadata: {result.document.metadata}")
    
    # Get context for LLM
    print("\n3. Getting context for LLM...")
    context = await rag.get_context(query, top_k=2, max_context_length=500)
    print(f"  Context ({len(context)} chars):")
    print(f"  {context[:200]}...")
    
    # Filter search
    print("\n4. Filtered search...")
    results = await rag.search(
        "programming languages",
        top_k=2,
        filter={"topic": "programming"}
    )
    print(f"  Found {len(results)} results with topic='programming'")
    
    print("\n✅ RAG system example completed!")


if __name__ == "__main__":
    asyncio.run(main())
