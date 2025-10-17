from typing import List, Dict, Any
from app.repositories.uow import UnitOfWork
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.utils.collections import Collections


class GetDocumentChunksInteractor:
    def __init__(
        self,
        uow: UnitOfWork,
        qdrant_embeddings_repository: QdrantEmbeddingsRepository,
    ):
        self.uow = uow
        self.qdrant_embeddings_repository = qdrant_embeddings_repository

    async def execute(self, document_id: str) -> List[Dict[str, Any]]:
        """Возвращает все чанки документа отсортированные по chunk_index ASC."""
        chunks = await self.qdrant_embeddings_repository.get_chunks_by_document(
            collection_name=Collections.DOCUMENT_EMBEDDINGS,
            document_id=document_id,
            with_vectors=False,
        )
        # Вернём минимальный набор полей по умолчанию
        return [
            {
                "chunk_index": c["chunk_index"],
                "content": c["content"],
            }
            for c in chunks
        ]