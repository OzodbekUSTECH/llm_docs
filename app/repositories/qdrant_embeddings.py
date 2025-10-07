from typing import List, Dict, Any, Optional
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, ScoredPoint, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
import numpy as np


class QdrantEmbeddingsRepository:
    """Репозиторий для работы с эмбеддингами документов в Qdrant"""
    
    COLLECTION_NAME = "document_embeddings"
    
    def __init__(self, qdrant_client: AsyncQdrantClient):
        self.client = qdrant_client
        self.vector_size = 1024  # Размер вектора для e5-large-v2
    
    async def ensure_collection_exists(self):
        """Создает коллекцию если она не существует"""
        try:
            await self.client.get_collection(self.COLLECTION_NAME)
        except Exception:
            await self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
    
    async def bulk_create_embeddings(
        self, 
        document_id: str, 
        chunks: List[str], 
        embeddings: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Сохраняет чанки с эмбеддингами в Qdrant
        
        Args:
            document_id: ID документа
            chunks: Список текстовых чанков
            embeddings: Список векторных представлений
            metadata: Дополнительные метаданные (filename, content_type, etc.)
        """
        await self.ensure_collection_exists()
        
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            
            # Базовый payload
            payload = {
                "document_id": document_id,
                "chunk_index": idx,
                "chunk_content": chunk,
                "chunk_length": len(chunk),
            }
            
            # Добавляем метаданные если есть
            if metadata:
                payload.update({
                    "filename": metadata.get("filename", ""),
                    "content_type": metadata.get("content_type", ""),
                })
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
        
        await self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points
        )
    
    async def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7,
        document_ids: Optional[List[str]] = None
    ) -> List[ScoredPoint]:
        """Поиск похожих чанков по вектору запроса"""
        await self.ensure_collection_exists()
        
        # Создаем фильтр если указаны document_ids
        search_filter = None
        if document_ids:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=document_ids)
                    )
                ]
            )
        
        search_result = await self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            score_threshold=similarity_threshold,
            query_filter=search_filter
        )
        
        return search_result
    
    async def delete_document_embeddings(self, document_id: str) -> None:
        """Удаляет все эмбеддинги документа"""
        await self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )
    
    async def get_document_stats(self, document_id: str) -> Dict[str, int]:
        """Получает статистику по документу"""
        await self.ensure_collection_exists()
        
        search_result = await self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=10000,  # Максимальное количество для получения всех чанков
            with_payload=True
        )
        
        chunks = search_result[0]  # Первый элемент - это точки
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk.payload.get("chunk_length", 0) for chunk in chunks)
        }
