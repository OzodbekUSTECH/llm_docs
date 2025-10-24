import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

from app.repositories.sources import SourcesRepository
from app.repositories.indexes import IndexesRepository
from app.dto.pagination import PaginatedResponse, PaginationRequest
from app.dto.sources import SourcesListResponse, GetSourcesParams, SourceResponse
from app.exceptions.app_error import AppError
from app.entities.sources import Source
from app.entities.indexes import Index
from app.utils.docs_store import LanceDBDocumentStore
from app.utils.enums import IndexType

logger = logging.getLogger(__name__)


class DocumentChunkResponse(BaseModel):
    """Response model for document chunk"""
    id: str
    content: str
    content_preview: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunksResponse(BaseModel):
    """Response model for document chunks with pagination"""
    source_name: str
    chunks: List[DocumentChunkResponse] = Field(default_factory=list)
    total: int
    page: int
    size: int
    total_pages: int


class GetAllDocumentsInteractor:
    def __init__(
        self, 
        sources_repository: SourcesRepository,
        indexes_repository: IndexesRepository
    ):
        self.sources_repository = sources_repository
        self.indexes_repository = indexes_repository
        
    async def execute(self, request: GetSourcesParams) -> PaginatedResponse[SourcesListResponse]:
        """
        Get all documents (sources) with their statistics
        
        Args:
            request: Parameters for pagination and search
            
        Returns:
            Paginated response with document list
        """
        try:
           
            # Get sources and total count
            sources, total = await self.sources_repository.get_all_and_count(
                request_query=request,
            )
            
            
            return PaginatedResponse(
                items=[SourcesListResponse.model_validate(source) for source in sources],
                total=total,
                page=request.page,
                size=request.size
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting documents: {e}", exc_info=True)
            raise AppError(status_code=500, message=f"Failed to get documents: {str(e)}")


class GetDocumentByIdInteractor:
    def __init__(
        self, 
        sources_repository: SourcesRepository,
        indexes_repository: IndexesRepository
    ):
        self.sources_repository = sources_repository
        self.indexes_repository = indexes_repository
        self.docs_store = LanceDBDocumentStore()
        
    async def execute(self, source_id: str) -> SourceResponse:
        """
        Get document by ID with full details including chunks
        
        Args:
            source_id: ID of the source
            
        Returns:
            Document response with full details
        """
        try:
            # Get source
            source = await self.sources_repository.get_one(source_id)
            if not source:
                raise AppError(status_code=404, message=f"Document with ID {source_id} not found")
            
            # Get all indexes for this source
            indexes = await self.indexes_repository.get_all(
                where=[Index.source_id == source_id]
            )
            
            # Get document chunks from LanceDB
            document_target_ids = [
                idx.target_id for idx in indexes 
                if idx.relation_type == IndexType.DOCUMENT
            ]
            
            chunks = []
            if document_target_ids:
                try:
                    lance_docs = self.docs_store.get(document_target_ids)
                    chunks = [
                        {
                            "id": getattr(doc, "id_", None) or getattr(doc, "doc_id", None),
                            "content": getattr(doc, "text", None) or getattr(doc, "content", ""),
                            "metadata": getattr(doc, "metadata", {})
                        }
                        for doc in lance_docs
                    ]
                except Exception as e:
                    logger.error(f"❌ Error getting chunks from LanceDB: {e}")
                    chunks = []
            
            # Prepare indexes data with chunks from LanceDB
            indexes_data = {
                "total_indexes": len(indexes),
                "document_chunks": sum(1 for idx in indexes if idx.relation_type == IndexType.DOCUMENT),
                "vector_chunks": sum(1 for idx in indexes if idx.relation_type == IndexType.VECTOR),
                "chunks": chunks[:10],  # Limit to first 10 chunks for preview
                "total_chunks": len(chunks)
            }
            
            # Create SourceResponse with indexes data
            return SourceResponse(
                id=source.id,
                name=source.name,
                size=source.size,
                created_at=source.created_at,
                updated_at=source.updated_at,
                note=source.note or {},
                indexes=indexes_data
            )
            
        except AppError:
            raise
        except Exception as e:
            logger.error(f"❌ Error getting document {source_id}: {e}", exc_info=True)
            raise AppError(status_code=500, message=f"Failed to get document: {str(e)}")


class GetDocumentChunksInteractor:
    def __init__(
        self, 
        sources_repository: SourcesRepository,
        indexes_repository: IndexesRepository
    ):
        self.sources_repository = sources_repository
        self.indexes_repository = indexes_repository
        self.docs_store = LanceDBDocumentStore()
    
    async def execute(self, source_id: str, page: int = 1, size: int = 20) -> DocumentChunksResponse:
        """
        Get document chunks with pagination
        
        Args:
            source_id: ID of the source
            page: Page number
            size: Page size
            
        Returns:
            Dict with chunks and pagination info
        """
        try:
            # Verify source exists
            source = await self.sources_repository.get_one(source_id)
            if not source:
                raise AppError(status_code=404, message=f"Document with ID {source_id} not found")
            
            # Get document indexes
            indexes = await self.indexes_repository.get_all(
                where=[
                    Index.source_id == source_id,
                    Index.relation_type == IndexType.DOCUMENT
                ]
            )
            
            if not indexes:
                return DocumentChunksResponse(
                    source_name=source.name,
                    chunks=[],
                    total=0,
                    page=page,
                    size=size,
                    total_pages=0
                )
            
            # Get target IDs and apply pagination
            target_ids = [idx.target_id for idx in indexes]
            total = len(target_ids)
            
            # Calculate pagination
            offset = (page - 1) * size
            paginated_ids = target_ids[offset:offset + size]
            
            # Get chunks from LanceDB
            chunk_responses = []
            if paginated_ids:
                try:
                    lance_docs = self.docs_store.get(paginated_ids)
                    chunk_responses = [
                        DocumentChunkResponse(
                            id=getattr(doc, "id_", None) or getattr(doc, "doc_id", None) or "",
                            content=getattr(doc, "text", None) or getattr(doc, "content", ""),
                            content_preview=(getattr(doc, "text", None) or getattr(doc, "content", ""))[:200] + "..." if len(getattr(doc, "text", None) or getattr(doc, "content", "")) > 200 else getattr(doc, "text", None) or getattr(doc, "content", ""),
                            metadata=getattr(doc, "metadata", {})
                        )
                        for doc in lance_docs
                    ]
                except Exception as e:
                    logger.error(f"❌ Error getting chunks from LanceDB: {e}")
                    chunk_responses = []
            
            total_pages = (total + size - 1) // size
            
            return DocumentChunksResponse(
                source_name=source.name,
                chunks=chunk_responses,
                total=total,
                page=page,
                size=size,
                total_pages=total_pages
            )
            
        except AppError:
            raise
        except Exception as e:
            logger.error(f"❌ Error getting document chunks {source_id}: {e}", exc_info=True)
            raise AppError(status_code=500, message=f"Failed to get document chunks: {str(e)}")