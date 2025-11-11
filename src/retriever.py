"""Query processing and context retrieval."""

import logging
from typing import List, Dict
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: VectorStore):
        self.embedder = embedding_generator
        self.vector_store = vector_store
        logger.info("Retriever initialized")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        logger.info(f"Query: {query[:100]}...")
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_vector=query_embedding, limit=top_k)
        logger.info(f"Retrieved {len(results)} chunks")
        return results
    
    def format_context(self, results: List[Dict]) -> str:
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Context {i} - from {result['source_file']}]\n{result['text']}")
        
        return "\n\n".join(context_parts)


