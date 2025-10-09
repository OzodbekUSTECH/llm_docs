from typing import List, Dict, Any, Optional, Union
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, ScoredPoint, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny
import numpy as np

from app.dto.qdrant_filters import QdrantFilters


class QdrantEmbeddingsRepository:
    """Репозиторий для работы с эмбеддингами документов в Qdrant"""
    
    
    def __init__(self, qdrant_client: AsyncQdrantClient):
        self.client = qdrant_client
   
    
    async def create_embeddings(
        self,
        collection_name: str,
        payload: Dict[str, Any],
        embedding: List[float],
    ) -> str:
        """
        Сохраняет один эмбеддинг в Qdrant
        
        Args:
            collection_name: Название коллекции
            entity_id: ID сущности (документа, правила и т.д.)
            content: Текстовое содержимое
            embedding: Векторное представление
            metadata: Дополнительные метаданные
            
        Returns:
            point_id: ID созданной точки в Qdrant
        """
        point_id = str(uuid.uuid4())
        
        
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        
        await self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        
        return point_id
    
    async def bulk_create_embeddings(
        self, 
        collection_name: str,
        document_id: str, 
        chunks: List[str], 
        embeddings: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Сохраняет чанки с эмбеддингами в Qdrant
        
        Args:
            collection_name: Название коллекции
            document_id: ID документа
            chunks: Список текстовых чанков
            embeddings: Список векторных представлений
            metadata: Дополнительные метаданные (filename, content_type, etc.)
        """
        
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
                    "document_type": metadata.get("document_type", "OTHER"),
                })
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
        
        await self.client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    async def search_similar(
        self, 
        collection_name: str,
        query_vector: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[QdrantFilters] = None,
    ) -> List[ScoredPoint]:
        """
        Поиск похожих векторов по вектору запроса

        Args:
            collection_name: Название коллекции
            query_vector: Вектор запроса
            limit: Максимальное количество результатов
            similarity_threshold: Порог схожести (0-1)
            filters: Фильтры для поиска. Можно передать:
                - QdrantFilters объект (рекомендуется):
                    QdrantFilters(
                        must=[FieldCondition(key="status", match=MatchValue(value="active"))],
                        should=[FieldCondition(key="category", match=MatchAny(any=["cat1", "cat2"]))],
                        must_not=[FieldCondition(key="archived", match=MatchValue(value=True))],
                        min_should=1
                    )
                - Словарь (обратная совместимость):
                    {
                        "must": [...],
                        "should": [...],
                        "must_not": [...],
                        "min_should": 1
                    }

        Returns:
            Список найденных точек с оценками схожести
        """

        search_filter = None
        if filters:
            search_filter = filters.to_qdrant_filter()
        

        search_result = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            score_threshold=similarity_threshold,
            query_filter=search_filter
        )

        return search_result
    
    async def delete_embeddings(self, entity_id: str, collection_name: str, field_name: str = "entity_id") -> None:
        """
        Удаляет эмбеддинги сущности
        
        Args:
            entity_id: ID сущности
            collection_name: Название коллекции
            field_name: Имя поля для фильтрации (по умолчанию "entity_id")
        """
        await self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key=field_name,
                            match=MatchValue(value=entity_id)
                        )
                    ]
                )
            )
        )
    
    async def delete_document_embeddings(self, document_id: str, collection_name: str) -> None:
        """Удаляет все эмбеддинги документа (обратная совместимость)"""
        await self.delete_embeddings(
            entity_id=document_id,
            collection_name=collection_name,
            field_name="document_id"
        )
 