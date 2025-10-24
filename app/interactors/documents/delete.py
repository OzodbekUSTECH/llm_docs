import logging
from pathlib import Path
from uuid import UUID
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.repositories.uow import UnitOfWork
from app.exceptions.app_error import AppError
from app.repositories.sources import SourcesRepository
from app.repositories.indexes import IndexesRepository
from app.entities.sources import Source
from app.entities.indexes import Index
from app.utils.docs_store import LanceDBDocumentStore
from app.utils.vectors_store import QdrantVectorStore
from app.utils.enums import IndexType

logger = logging.getLogger(__name__)


class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion"""
    deleted_source: str
    deleted_chunks: int
    deleted_vectors: int
    deleted_indexes: int
    deleted_file: bool


class DocumentStatsResponse(BaseModel):
    """Response model for document statistics"""
    source_name: str
    source_size: int
    document_chunks: int
    vector_chunks: int
    total_indexes: int
    created_at: Optional[datetime] = None


class DeleteDocumentInteractor:
    def __init__(
        self,
        uow: UnitOfWork,
        sources_repository: SourcesRepository,
        indexes_repository: IndexesRepository,
    ):
        self.uow = uow
        self.sources_repository = sources_repository
        self.indexes_repository = indexes_repository
        self.storage_dir = Path("storage/documents")
        self.docs_store = LanceDBDocumentStore()
        self.vector_store = QdrantVectorStore()
    
    async def execute(self, source_id: str) -> DeleteDocumentResponse:
        """
        Delete document and all related data
        
        Args:
            source_id: ID of the source to delete
            
        Returns:
            Dict with deletion statistics
        """
        try:
            # 1. Get source
            source = await self.sources_repository.get_one(source_id)
            if not source:
                # Document already deleted or doesn't exist
                logger.info(f"Document with ID {source_id} not found - already deleted or doesn't exist")
                return DeleteDocumentResponse(
                    deleted_source="Unknown",
                    deleted_chunks=0,
                    deleted_vectors=0,
                    deleted_indexes=0,
                    deleted_file=False
                )
            
            logger.info(f"🗑️ Deleting document: {source.name} (ID: {source_id})")
            
            # 2. Get all indexes for this source
            indexes = await self.indexes_repository.get_all(
                where=[Index.source_id == source_id]
            )
            
            if not indexes:
                logger.warning(f"No indexes found for source {source_id}")
                # Still delete the source
                await self.sources_repository.delete(source_id)
                await self.uow.commit()
                return DeleteDocumentResponse(
                    deleted_source=source.name,
                    deleted_chunks=0,
                    deleted_vectors=0,
                    deleted_indexes=0,
                    deleted_file=False
                )
            
            # 3. Collect target IDs (chunk IDs)
            document_target_ids = []
            vector_target_ids = []
            
            for index in indexes:
                if index.relation_type == IndexType.DOCUMENT:
                    document_target_ids.append(index.target_id)
                elif index.relation_type == IndexType.VECTOR:
                    vector_target_ids.append(index.target_id)
            
            logger.info(f"Found {len(document_target_ids)} document chunks and {len(vector_target_ids)} vector chunks")
            
            # 4. Delete from LanceDB (document store)
            deleted_docs = 0
            if document_target_ids:
                try:
                    self.docs_store.delete(document_target_ids)
                    deleted_docs = len(document_target_ids)
                    logger.info(f"✅ Deleted {deleted_docs} chunks from LanceDB")
                except Exception as e:
                    logger.error(f"❌ Error deleting from LanceDB: {e}")
            
            # 5. Delete from Qdrant (vector store)
            deleted_vectors = 0
            if vector_target_ids:
                try:
                    await self.vector_store.delete(vector_target_ids)
                    deleted_vectors = len(vector_target_ids)
                    logger.info(f"✅ Deleted {deleted_vectors} vectors from Qdrant")
                except Exception as e:
                    logger.error(f"❌ Error deleting from Qdrant: {e}")
            
            # 6. Delete indexes from database
            for index in indexes:
                await self.indexes_repository.delete(index.id)
            
            logger.info(f"✅ Deleted {len(indexes)} indexes from database")
            
            # 7. Delete physical file
            deleted_file = False
            if source.path:
                file_path = self.storage_dir / source.path
                try:
                    if file_path.exists():
                        file_path.unlink()
                        deleted_file = True
                        logger.info(f"✅ Deleted physical file: {file_path}")
                    else:
                        logger.warning(f"Physical file not found: {file_path}")
                except Exception as e:
                    logger.error(f"❌ Error deleting physical file: {e}")
            
            # 8. Delete source from database
            await self.sources_repository.delete(source_id)
            logger.info(f"✅ Deleted source from database")
            
            # 9. Commit transaction
            await self.uow.commit()
            
            result = DeleteDocumentResponse(
                deleted_source=source.name,
                deleted_chunks=deleted_docs,
                deleted_vectors=deleted_vectors,
                deleted_indexes=len(indexes),
                deleted_file=deleted_file
            )
            
            logger.info(f"🎉 Successfully deleted document: {result}")
            return result
            
        except Exception as e:
            await self.uow.rollback()
            logger.error(f"❌ Error deleting document {source_id}: {e}", exc_info=True)
            raise AppError(status_code=500, message=f"Failed to delete document: {str(e)}")
    
    async def get_document_stats(self, source_id: str) -> DocumentStatsResponse:
        """
        Get statistics about document before deletion
        
        Args:
            source_id: ID of the source
            
        Returns:
            Dict with document statistics
        """
        try:
            source = await self.sources_repository.get_by_id(source_id)
            if not source:
                raise AppError(status_code=404, message=f"Document with ID {source_id} not found")
            
            indexes = await self.indexes_repository.get_many(
                where=[Index.source_id == source_id]
            )
            
            document_chunks = sum(1 for idx in indexes if idx.relation_type == IndexType.DOCUMENT)
            vector_chunks = sum(1 for idx in indexes if idx.relation_type == IndexType.VECTOR)
            
            return DocumentStatsResponse(
                source_name=source.name,
                source_size=source.size,
                document_chunks=document_chunks,
                vector_chunks=vector_chunks,
                total_indexes=len(indexes),
                created_at=source.created_at
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting document stats {source_id}: {e}", exc_info=True)
            raise AppError(status_code=500, message=f"Failed to get document stats: {str(e)}")