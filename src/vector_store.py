"""Vector storage and retrieval using ChromaDB."""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        
        try:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            logger.info(f"Connected to ChromaDB ({persist_path})")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection ready: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create/get collection: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List[Dict]) -> List[str]:
        if not chunks:
            return []
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}_{chunk['source_file']}_{chunk['chunk_id']}"
            ids.append(chunk_id)
            embeddings.append(chunk['embedding'])
            documents.append(chunk['text'])
            metadatas.append({
                'chunk_id': chunk['chunk_id'],
                'total_chunks': chunk['total_chunks'],
                'source_file': chunk['source_file'],
                'source_path': chunk.get('source_path', ''),
            })
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            logger.info(f"Batch {i//batch_size + 1}: {batch_end - i} chunks")
        
        logger.info(f"Added {len(ids)} chunks")
        return ids
    
    def search(self, query_vector: List[float], limit: int = 5, source_filter: Optional[str] = None) -> List[Dict]:
        where_filter = {"source_file": source_filter} if source_filter else None
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results and results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                # ChromaDB returns distances (lower is better), convert to similarity score
                distance = results['distances'][0][i]
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'score': similarity,
                    'text': results['documents'][0][i],
                    'source_file': results['metadatas'][0][i]['source_file'],
                    'chunk_id': results['metadatas'][0][i]['chunk_id'],
                })
        
        logger.info(f"Found {len(formatted_results)} chunks")
        return formatted_results
    
    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection")
            self._ensure_collection()  # Recreate it
        except Exception as e:
            logger.error(f"Failed to delete: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict:
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'points_count': count,
            }
        except Exception as e:
            logger.error(f"Failed to get info: {str(e)}")
            raise


