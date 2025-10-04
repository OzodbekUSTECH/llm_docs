from app.repositories.uow import UnitOfWork
from uuid import UUID
from app.exceptions.app_error import AppError
from app.repositories.documents import DocumentsRepository
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository


class DeleteDocumentInteractor:
    def __init__(
        self,
        uow: UnitOfWork,
        documents_repository: DocumentsRepository, qdrant_embeddings_repository: QdrantEmbeddingsRepository):
        self.uow = uow
        self.documents_repository = documents_repository
        self.qdrant_embeddings_repository = qdrant_embeddings_repository

    async def execute(self, id: UUID):
        document = await self.documents_repository.get_one(id=id)
        if not document:
            raise AppError(status_code=404, message="Document not found")
        await self.documents_repository.delete(id)
        await self.qdrant_embeddings_repository.delete_document_embeddings(str(id))
        
        await self.uow.commit()
        return document