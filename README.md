# RAG System - Document Q&A

Vector search-based Retrieval-Augmented Generation system for PDF documents using Google Gemini API.

## Quick Start

### 1. Get Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key
3. Set environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

### 2. Install Dependencies

```bash
uv sync
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

## Deployment on Streamlit Cloud

1. Push to GitHub
2. Deploy on [Streamlit Cloud](https://streamlit.io/cloud)
3. Add secret: `Settings → Secrets → GEMINI_API_KEY = "your-key"`
4. Your sample PDF will be auto-indexed on first load!

## Project Structure

```
├── app.py                     # Streamlit interface (indexing + Q&A)
├── src/
│   ├── config.py              # Configuration
│   ├── document_processor.py  # PDF → Markdown
│   ├── chunker.py             # Text chunking
│   ├── embeddings.py   # Gemini embeddings (active)
│   ├── embeddings_ollama.py   # Ollama embeddings (backup)
│   ├── generator.py    # Gemini LLM (active)
│   ├── generator_ollama.py    # Ollama LLM (backup)
│   ├── vector_store.py        # ChromaDB integration
│   └── retriever.py           # Query & retrieval
├── uploads/                   # Input PDFs
│   └── sample.pdf             # Demo document (committed to repo)
├── processed/                 # Output markdown
└── chroma_db/                 # Local vector database (auto-created)
```

## Configuration

Edit `src/config.py` to customize:
- `gemini_embedding_model` - Gemini embedding model (default: text-embedding-004)
- `gemini_llm_model` - Gemini LLM (default: gemini-2.0-flash-exp)
- `chunk_size` / `chunk_overlap` - Chunking parameters
- `chroma_dir` - Local database directory

## Dependencies

- **docling**: PDF to markdown conversion
- **chromadb**: Local vector database
- **google-genai**: Gemini API for embeddings and LLM
- **streamlit**: Web interface
- **uv**: Package manager

## Alternative: Ollama (Local)

If you prefer to run locally without API costs, Ollama implementations are available:

**Files:**
- `src/embeddings_ollama.py`
- `src/generator_ollama.py`

**To switch to Ollama:**

1. Install Ollama:
```bash
brew install ollama
ollama serve
```

2. Pull models:
```bash
ollama pull nomic-embed-text:v1.5
ollama pull llama3.2:latest
```

3. Update `src/__init__.py`:
```python
# Change these imports:
from .embeddings_ollama import EmbeddingGenerator
from .generator_ollama import Generator
```

4. Update `app.py` to use Ollama parameters:
```python
# Line 39 & 46:
embedder = EmbeddingGenerator(settings.ollama_embedding_model, settings.ollama_host)
generator = Generator(settings.ollama_llm_model, settings.ollama_host)
```
