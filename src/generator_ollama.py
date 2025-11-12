"""LLM-based answer generation."""

import logging
import ollama

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, model_name: str = "llama3.2:latest", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
        logger.info(f"Generator initialized ({model_name})")
    
    def generate_answer(self, query: str, context: str, max_tokens: int = 500) -> str:
        prompt = self._build_prompt(query, context)
        logger.info(f"Generating answer (max_tokens={max_tokens})...")
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 40,
                    'repeat_penalty': 1.1,
                },
                stream=False
            )
            
            answer = response['response']
            logger.info(f"Generated {len(answer)} chars (~{len(answer.split())} words)")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_answer_stream(self, query: str, context: str, max_tokens: int = 500):
        prompt = self._build_prompt(query, context)
        logger.info(f"Streaming answer (max_tokens={max_tokens})...")
        
        try:
            response_stream = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 40,
                    'repeat_penalty': 1.1,
                },
                stream=True
            )
            
            full_text = ""
            for chunk in response_stream:
                if 'response' in chunk:
                    text = chunk['response']
                    full_text += text
                    yield text
            
            logger.info(f"Stream complete ({len(full_text)} chars)")
            
        except Exception as e:
            logger.error(f"Stream failed: {str(e)}")
            yield f"Error: {str(e)}"
    
    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are a helpful assistant that provides detailed answers based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Provide a thorough answer based on the information in the context
- Include all relevant details and explanations
- If the context doesn't contain enough information, explain what's missing
- Cite which context section you're referencing when relevant
- Break down complex topics clearly

Answer:"""

