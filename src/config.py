"""Configuration for the RAG system."""

from pathlib import Path
import os


class Settings:
    project_root: Path = Path(__file__).parent.parent
    uploads_dir: Path = project_root / "uploads"
    processed_dir: Path = project_root / "processed"
    chroma_dir: Path = project_root / "chroma_db"
    
    # Gemini API (primary)
    gemini_embedding_model: str = "text-embedding-004"
    gemini_llm_model: str = "gemini-2.5-flash-lite"
    
    # Ollama (backup/local option)
    ollama_host: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text:v1.5"
    ollama_llm_model: str = "llama3.2:latest"
    
    # Vector store
    collection_name: str = "documents"
    vector_size: int = 768
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    def __init__(self):
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()


