"""RAG System - Vector Search-Based Retrieval-Augmented Generation"""

from .config import settings
from .document_processor import DocumentProcessor
from .chunker import TextChunker
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator

# Ollama alternatives (for local/backup use)
from .embeddings_ollama import EmbeddingGenerator as OllamaEmbeddingGenerator
from .generator_ollama import Generator as OllamaGenerator

__version__ = "2.0.0"

__all__ = [
    'settings',
    'DocumentProcessor',
    'TextChunker',
    'EmbeddingGenerator',
    'VectorStore',
    'Retriever',
    'Generator',
    'OllamaEmbeddingGenerator',
    'OllamaGenerator',
]


