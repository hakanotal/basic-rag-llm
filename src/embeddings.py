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
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts using batch API calls.
        
        Args:
            texts: List of texts to embed
            batch_size: Maximum number of texts to send in a single API call (default 100)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
        
        # Preprocess all texts
        max_chars = 10000
        processed_texts = []
        for text in texts:
            text = text.strip()
            if not text:
                text = "empty"
            if len(text) > max_chars:
                text = text[:max_chars]
            processed_texts.append(text)
        
        all_embeddings = []
        failed_batches = []
        
        # Process in batches
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(processed_texts) + batch_size - 1) // batch_size
            
            try:
                # Make a single API call for the entire batch
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch
                )
                
                # Extract embeddings from batch result
                batch_embeddings = [emb.values for emb in result.embeddings]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Batch {batch_num}/{total_batches} completed: {len(batch_embeddings)} embeddings")
                
            except Exception as e:
                logger.error(f"Batch {batch_num}/{total_batches} failed: {str(e)}")
                failed_batches.append((i, i + len(batch)))
                
                # Fallback: try individual embeddings for failed batch
                logger.warning(f"Attempting individual embeddings for failed batch {batch_num}...")
                for j, text in enumerate(batch):
                    try:
                        all_embeddings.append(self.embed_text(text))
                    except Exception as fallback_error:
                        logger.error(f"Failed individual embedding at index {i+j}: {str(fallback_error)}")
                        # Use zero vector as last resort
                        all_embeddings.append([0.0] * 768)  # text-embedding-004 is 768 dims
        
        if failed_batches:
            logger.warning(f"Failed batches (recovered via fallback): {failed_batches}")
        
        logger.info(f"Generated {len(all_embeddings)} embeddings total")
        return all_embeddings
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """Add embeddings to chunk dictionaries."""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        return chunks

