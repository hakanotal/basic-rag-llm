"""Vector embedding generation using Ollama."""

import logging
from typing import List
import time
import ollama

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, model_name: str = "nomic-embed-text:v1.5", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
        logger.info(f"EmbeddingGenerator initialized ({model_name})")
        self._verify_model()
    
    def _verify_model(self):
        try:
            models_response = self.client.list()
            model_names = []
            
            if hasattr(models_response, 'models'):
                for model in models_response.models:
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
            
            if self.model_name not in model_names:
                logger.warning(f"Model {self.model_name} not found. Run: ollama pull {self.model_name}")
            else:
                logger.info(f"Model {self.model_name} available")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            raise
    
    def embed_text(self, text: str, retries: int = 3) -> List[float]:
        max_chars = 8192
        if len(text) > max_chars:
            logger.warning(f"Truncating text from {len(text)} to {max_chars} chars")
            text = text[:max_chars]
        
        # Clean text - remove null bytes and control characters that might cause issues
        text = text.replace('\x00', '').strip()
        if not text:
            logger.warning("Empty text after cleaning, using placeholder")
            text = "empty"
        
        for attempt in range(retries):
            try:
                response = self.client.embeddings(model=self.model_name, prompt=text)
                return response['embedding']
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {retries} attempts: {str(e)}")
                    logger.error(f"Text length: {len(text)} chars, preview: {text[:100]}...")
                    raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = []
        failed_indices = []
        
        for i, text in enumerate(texts):
            try:
                embeddings.append(self.embed_text(text))
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(texts)}")
                # Small delay to prevent overwhelming Ollama
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Failed on chunk {i+1}/{len(texts)}")
                failed_indices.append(i)
                # Use zero vector as fallback to continue processing
                logger.warning(f"Using zero vector for failed chunk {i+1}")
                embeddings.append([0.0] * 768)  # nomic-embed-text is 768 dims
        
        if failed_indices:
            logger.warning(f"Failed to embed {len(failed_indices)} chunks: {failed_indices[:10]}...")
        
        logger.info(f"Generated {len(embeddings)} embeddings ({len(embeddings) - len(failed_indices)} successful)")
        return embeddings
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        return chunks


