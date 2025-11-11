# Basic RAG System with LLMs

Vector Search-Based Retrieval-Augmented Generation system for document Q&A.

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Start Ollama & Pull Models
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull models
ollama pull nomic-embed-text:v1.5
ollama pull llama3.2:latest
```

### 3. Run Q&A Interface
```bash
uv run streamlit run app.py
```

Open browser to http://localhost:8501

**Try the demo:**
- A sample PDF is automatically indexed on startup
- Try asking: "What is a RAG system?" or "What are the key components?"
- Upload your own PDFs using the sidebar
- Click "Reindex Documents" to process them

## Project Structure

```
├── app.py                     # Streamlit interface (indexing + Q&A)
├── src/
│   ├── config.py              # Configuration (edit here)
│   ├── document_processor.py  # PDF → Markdown
│   ├── chunker.py             # Text chunking
│   ├── embeddings.py          # Vector generation
│   ├── vector_store.py        # ChromaDB integration
│   ├── retriever.py           # Query & retrieval
│   └── generator.py           # Answer generation
├── uploads/                   # Input PDFs
│   └── sample.pdf             # Demo document (committed to repo)
├── processed/                 # Output markdown
└── chroma_db/                 # Local vector database (auto-created)
```

## Configuration

Edit `src/config.py` to customize:
- `embedding_model` - Embedding model name
- `llm_model` - LLM for answer generation
- `chunk_size` / `chunk_overlap` - Chunking parameters
- `chroma_dir` - Local database directory

## Dependencies

- Python 3.12+
- Ollama (embedding + LLM)
- ChromaDB (local vector database)
- Streamlit (web interface)
- UV (package manager)

