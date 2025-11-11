# Basic RAG System with LLMs

Vector Search-Based Retrieval-Augmented Generation system for document Q&A.

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Start Services
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull models
ollama pull nomic-embed-text:v1.5
ollama pull llama3.2:latest

# Terminal 3: Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Run Q&A Interface
```bash
uv run streamlit run app.py
```

Open browser to http://localhost:8501

- Upload PDFs using the sidebar
- Click "Reindex Documents" to process them
- Start asking questions!

## Project Structure

```
├── app.py                     # Streamlit interface (indexing + Q&A)
├── src/
│   ├── config.py              # Configuration (edit here)
│   ├── document_processor.py  # PDF → Markdown
│   ├── chunker.py             # Text chunking
│   ├── embeddings.py          # Vector generation
│   ├── vector_store.py        # Qdrant integration
│   ├── retriever.py           # Query & retrieval
│   └── generator.py           # Answer generation
├── uploads/                   # Input PDFs
└── processed/                 # Output markdown
```

## Configuration

Edit `src/config.py` to customize:
- `embedding_model` - Embedding model name
- `llm_model` - LLM for answer generation
- `chunk_size` / `chunk_overlap` - Chunking parameters
- `qdrant_host` / `qdrant_port` - Database connection

## Dependencies

- Python 3.12+
- Ollama (embedding + LLM)
- Qdrant (vector database)
- Streamlit (web interface)
- UV (package manager)

