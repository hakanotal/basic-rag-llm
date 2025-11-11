"""Vector storage and retrieval using Qdrant."""

import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, collection_name: str = "documents", host: str = "localhost", port: int = 6333, vector_size: int = 768):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        try:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant ({host}:{port})")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                logger.info(f"Collection created")
            else:
                logger.info(f"Collection exists")
                
        except Exception as e:
            logger.error(f"Failed to create/check collection: {str(e)}")
            raise
    
    def add_chunks(self, chunks: List[Dict]) -> List[str]:
        if not chunks:
            return []
        
        points = []
        point_ids = []
        
        for chunk in chunks:
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            payload = {
                'text': chunk['text'],
                'chunk_id': chunk['chunk_id'],
                'total_chunks': chunk['total_chunks'],
                'source_file': chunk['source_file'],
                'source_path': chunk.get('source_path', ''),
            }
            
            points.append(PointStruct(id=point_id, vector=chunk['embedding'], payload=payload))
        
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            logger.info(f"Batch {i//batch_size + 1}: {len(batch)} points")
        
        logger.info(f"Added {len(points)} chunks")
        return point_ids
    
    def search(self, query_vector: List[float], limit: int = 5, source_filter: Optional[str] = None) -> List[Dict]:
        query_filter = None
        if source_filter:
            query_filter = Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_filter))])
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter
        )
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.id,
                'score': result.score,
                'text': result.payload['text'],
                'source_file': result.payload['source_file'],
                'chunk_id': result.payload['chunk_id'],
            })
        
        logger.info(f"Found {len(formatted_results)} chunks")
        return formatted_results
    
    def delete_collection(self):
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection")
        except Exception as e:
            logger.error(f"Failed to delete: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict:
        try:
            collection = self.client.get_collection(collection_name=self.collection_name)
            return {
                'name': self.collection_name,
                'vectors_count': collection.vectors_count,
                'points_count': collection.points_count,
                'status': collection.status,
            }
        except Exception as e:
            logger.error(f"Failed to get info: {str(e)}")
            raise


