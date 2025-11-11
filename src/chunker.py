"""Text chunking with semantic boundaries."""

import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"TextChunker initialized (size={chunk_size}, overlap={chunk_overlap})")
    
    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        text = text.strip()
        logger.info(f"Splitting {len(text)} characters...")
        
        paragraphs = self._split_paragraphs(text)
        chunks = []
        previous_sentence = ""
        
        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                chunk_text = previous_sentence + " " + paragraph if previous_sentence else paragraph
                chunk_text = chunk_text.strip()
                
                if chunk_text:
                    chunks.append(chunk_text)
                    previous_sentence = self._get_last_sentence(paragraph)
            else:
                sentences = self._split_sentences(paragraph)
                current_chunk = previous_sentence if previous_sentence else ""
                
                for sentence in sentences:
                    test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    
                    if len(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        
                        if len(sentence) > self.chunk_size:
                            sentence_chunks = self._split_long_sentence(sentence)
                            for i, sent_chunk in enumerate(sentence_chunks):
                                if i == 0 and previous_sentence:
                                    chunks.append((previous_sentence + " " + sent_chunk).strip())
                                else:
                                    chunks.append(sent_chunk.strip())
                            current_chunk = ""
                            previous_sentence = sentence_chunks[-1] if sentence_chunks else ""
                        else:
                            current_chunk = sentence
                            previous_sentence = sentence
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    previous_sentence = self._get_last_sentence(current_chunk)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_last_sentence(self, text: str) -> str:
        sentences = self._split_sentences(text)
        if sentences:
            last = sentences[-1]
            return last if len(last) <= 200 else last[-200:]
        return ""
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        words = sentence.split()
        chunks = []
        current = []
        current_len = 0
        
        for word in words:
            word_len = len(word) + 1
            if current_len + word_len <= self.chunk_size:
                current.append(word)
                current_len += word_len
            else:
                if current:
                    chunks.append(' '.join(current))
                current = [word]
                current_len = word_len
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
    
    def chunk_document(self, document: Dict[str, str]) -> List[Dict]:
        text = document['content']
        chunks = self.split_text(text)
        
        chunk_dicts = []
        for i, chunk_text in enumerate(chunks):
            chunk_dicts.append({
                'text': chunk_text,
                'chunk_id': i,
                'total_chunks': len(chunks),
                'source_file': document['filename'],
                'source_path': document.get('source_path', ''),
            })
        
        logger.info(f"Created {len(chunk_dicts)} chunks from {document['filename']}")
        return chunk_dicts
    
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks


