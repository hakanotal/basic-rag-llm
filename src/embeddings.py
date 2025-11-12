"""Vector embedding generation using Google Gemini API."""

import logging
import os
from typing import List
from google import genai

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name
        
        # Check for API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        logger.info(f"EmbeddingGenerator initialized (Gemini {model_name})")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        max_chars = 10000  # Gemini's limit
        if len(text) > max_chars:
            logger.warning(f"Truncating text from {len(text)} to {max_chars} chars")
            text = text[:max_chars]
        
        text = text.strip()
        if not text:
            logger.warning("Empty text, using placeholder")
            text = "empty"
        
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return result.embeddings[0].values
            
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            logger.error(f"Text length: {len(text)} chars, preview: {text[:100]}...")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = []
        failed_indices = []
        
        for i, text in enumerate(texts):
            try:
                embeddings.append(self.embed_text(text))
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(texts)}")
                    
            except Exception as e:
                logger.error(f"Failed on chunk {i+1}/{len(texts)}: {str(e)}")
                failed_indices.append(i)
                # Use zero vector as fallback
                logger.warning(f"Using zero vector for failed chunk {i+1}")
                embeddings.append([0.0] * 768)  # text-embedding-004 is 768 dims
        
        if failed_indices:
            logger.warning(f"Failed to embed {len(failed_indices)} chunks: {failed_indices[:10]}...")
        
        logger.info(f"Generated {len(embeddings)} embeddings ({len(embeddings) - len(failed_indices)} successful)")
        return embeddings
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """Add embeddings to chunk dictionaries."""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        return chunks

