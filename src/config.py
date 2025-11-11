"""Configuration for the RAG system."""

from pathlib import Path


class Settings:
    project_root: Path = Path(__file__).parent.parent
    uploads_dir: Path = project_root / "uploads"
    processed_dir: Path = project_root / "processed"
    
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text:v1.5"
    llm_model: str = "llama3.2:latest"
    
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"
    vector_size: int = 768
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    def __init__(self):
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()


