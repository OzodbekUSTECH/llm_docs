# qdrant_vectorstore.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
import uuid

from app.dto.schemas import DocumentWithEmbedding


class QdrantVectorStore:
    """Qdrant Vector Store implementation using AsyncQdrantClient"""

    def __init__(
        self,
        collection_name: str = "documents",
        url: str = "http://localhost:6333",
        vector_size: int = 3072,
        distance: Distance = Distance.COSINE,
        batch_size: int = 100,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self.url = url
        self.vector_size = vector_size
        self.distance = distance
        self.batch_size = batch_size
        self._kwargs = kwargs
        
        # Initialize async client
        self._client = AsyncQdrantClient(
            url=url,
            **kwargs
        )
        self._initialized = False

    async def _ensure_initialized(self):
        """Lazy initialization of collection"""
        if not self._initialized:
            try:
                await self._client.get_collection(self.collection_name)
            except Exception:
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                )
            self._initialized = True

    async def add(
        self,
        embeddings: list[list[float]] | list[DocumentWithEmbedding],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Add vector embeddings to vector stores"""
        await self._ensure_initialized()
        
        if isinstance(embeddings[0], list):
            # Convert list[list[float]] to list[DocumentWithEmbedding]
            nodes: list[DocumentWithEmbedding] = [
                DocumentWithEmbedding(embedding=embedding) for embedding in embeddings
            ]
        else:
            nodes = embeddings  # type: ignore

        if metadatas is not None:
            for node, metadata in zip(nodes, metadatas):
                node.metadata = metadata

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in nodes]
        
        for node, id in zip(nodes, ids):
            node.id_ = id
            node.relationships = {
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id=id)
            }

        # Process in batches
        for i in range(0, len(nodes), self.batch_size):
            batch_nodes = nodes[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]
            
            points = []
            for node, node_id in zip(batch_nodes, batch_ids):
                point = PointStruct(
                    id=node_id,
                    vector=node.embedding,
                    payload={
                        "text": node.text or "",
                        "metadata": node.metadata or {}
                    }
                )
                points.append(point)
            
            await self._client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        return ids

    async def delete(self, ids: list[str], **kwargs):
        """Delete vector embeddings from vector stores"""
        await self._ensure_initialized()
        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids)
        )

    async def query(
        self,
        embedding: list[float],
        top_k: int = 1,
        ids: Optional[list[str]] = None,
        **kwargs,
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """Return the top k most similar vector embeddings"""
        await self._ensure_initialized()
        
        # Build filter if ids provided
        query_filter = None
        if ids:
            query_filter = Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.file_id",
                        match=models.MatchAny(any=ids)
                    )
                ]
            )

        # Search
        search_result = await self._client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
            query_filter=query_filter,
            **kwargs
        )

        # Extract results
        embeddings = []
        similarities = []
        out_ids = []
        
        for hit in search_result:
            embeddings.append(hit.vector)
            similarities.append(hit.score)
            out_ids.append(hit.id)

        return embeddings, similarities, out_ids

    async def drop(self):
        """Drop the vector store"""
        await self._ensure_initialized()
        await self._client.delete_collection(self.collection_name)
        self._initialized = False

    async def count(self) -> int:
        """Get the number of vectors in the collection"""
        await self._ensure_initialized()
        collection_info = await self._client.get_collection(self.collection_name)
        return collection_info.points_count

    def __persist_flow__(self):
        """Persist flow configuration"""
        return {
            "collection_name": self.collection_name,
            "url": self.url,
            "vector_size": self.vector_size,
            "distance": self.distance,
            "batch_size": self.batch_size,
            **self._kwargs,
        }