"""
RAG Examples - Retrieval-Augmented Generation with vector search.

Demonstrates document ingestion, embeddings, semantic search, and context retrieval.
"""

import asyncio
from skills.rag_skill import RAGSystem, Document


async def main():
    """Run RAG examples."""
    
    print("=" * 60)
    print("RAG (Retrieval-Augmented Generation) Examples")
    print("=" * 60)
    
    # Initialize RAG system
    rag = RAGSystem(collection_name="examples")
    
    # Example 1: Ingest documents
    print("\n1. Document Ingestion")
    print("-" * 40)
    
    documents = [
        Document(
            id="doc1",
            content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"category": "programming", "language": "python"}
        ),
        Document(
            id="doc2",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            metadata={"category": "ai", "topic": "machine learning"}
        ),
        Document(
            id="doc3",
            content="Docker is a platform for developing, shipping, and running applications in containers.",
            metadata={"category": "devops", "tool": "docker"}
        ),
        Document(
            id="doc4",
            content="FastAPI is a modern web framework for building APIs with Python based on standard type hints.",
            metadata={"category": "programming", "language": "python", "framework": "fastapi"}
        ),
        Document(
            id="doc5",
            content="Vector databases store and query high-dimensional vectors for semantic search applications.",
            metadata={"category": "databases", "type": "vector"}
        ),
    ]
    
    result = await rag.ingest_documents(documents)
    print(f"Ingested {result.documents_added} documents")
    print(f"Total chunks: {result.chunks_created}")
    
    # Example 2: Semantic search
    print("\n2. Semantic Search")
    print("-" * 40)
    
    query = "How do I build web APIs with Python?"
    results = await rag.search(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results.documents)} relevant documents:")
    for i, doc in enumerate(results.documents, 1):
        print(f"\n  {i}. Score: {doc.score:.3f}")
        print(f"     Content: {doc.content[:100]}...")
        print(f"     Metadata: {doc.metadata}")
    
    # Example 3: Filtered search
    print("\n3. Filtered Search")
    print("-" * 40)
    
    query = "programming languages"
    results = await rag.search(
        query,
        top_k=5,
        filters={"category": "programming"}
    )
    
    print(f"Query: '{query}' (filtered by category='programming')")
    print(f"Found {len(results.documents)} results:")
    for doc in results.documents:
        print(f"  - {doc.content[:80]}... (score: {doc.score:.3f})")
    
    # Example 4: Get context for LLM
    print("\n4. Context Retrieval for LLM")
    print("-" * 40)
    
    query = "What is Docker used for?"
    context = await rag.get_context(query, max_tokens=200)
    
    print(f"Query: '{query}'")
    print(f"Retrieved context:\n{context}")
    
    # Example 5: Hybrid search (vector + keyword)
    print("\n5. Hybrid Search")
    print("-" * 40)
    
    results = await rag.hybrid_search(
        query="Python framework",
        top_k=3,
        vector_weight=0.7,
        keyword_weight=0.3
    )
    
    print(f"Hybrid search results (70% vector, 30% keyword):")
    for doc in results.documents:
        print(f"  - {doc.content[:80]}... (score: {doc.score:.3f})")
    
    # Example 6: Update document
    print("\n6. Update Document")
    print("-" * 40)
    
    updated_doc = Document(
        id="doc1",
        content="Python is a versatile, high-level programming language with extensive libraries and frameworks.",
        metadata={"category": "programming", "language": "python", "updated": True}
    )
    
    await rag.update_document(updated_doc)
    print(f"Updated document: {updated_doc.id}")
    
    # Verify update
    results = await rag.search("Python libraries", top_k=1)
    print(f"Search result: {results.documents[0].content[:80]}...")
    
    # Example 7: Delete document
    print("\n7. Delete Document")
    print("-" * 40)
    
    await rag.delete_document("doc5")
    print("Deleted document: doc5")
    
    # Verify deletion
    all_docs = await rag.list_documents()
    print(f"Remaining documents: {len(all_docs)}")
    
    # Example 8: Batch ingestion
    print("\n8. Batch Document Ingestion")
    print("-" * 40)
    
    batch_docs = [
        Document(
            id=f"batch_{i}",
            content=f"This is batch document number {i} about various topics.",
            metadata={"batch": True, "number": i}
        )
        for i in range(10)
    ]
    
    result = await rag.ingest_documents(batch_docs)
    print(f"Batch ingested {result.documents_added} documents")
    
    # Example 9: Collection statistics
    print("\n9. Collection Statistics")
    print("-" * 40)
    
    stats = await rag.get_stats()
    print(f"Total documents: {stats.get('total_documents', 0)}")
    print(f"Total chunks: {stats.get('total_chunks', 0)}")
    print(f"Collection name: {rag.collection_name}")
    
    # Example 10: Clear collection
    print("\n10. Clear Collection")
    print("-" * 40)
    
    await rag.clear()
    print("Collection cleared")
    
    stats = await rag.get_stats()
    print(f"Documents after clear: {stats.get('total_documents', 0)}")
    
    print("\n" + "=" * 60)
    print("✅ RAG examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
