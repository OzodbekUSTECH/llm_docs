from fastapi import APIRouter, status, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from dishka.integrations.fastapi import FromDishka, DishkaRoute
from typing import Annotated, List, Dict, Any
import json

from app.dto.sources import GetSourcesParams, SourceResponse
from app.interactors.documents.create_v2 import CreateDocumentV2Interactor
from app.interactors.documents.delete import DeleteDocumentInteractor
from app.interactors.documents.search import SearchDocumentsInteractor
from app.interactors.documents.get import (
    GetAllDocumentsInteractor, 
    GetDocumentByIdInteractor,
    GetDocumentChunksInteractor,
)
from app.dto.pagination import PaginatedResponse


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    route_class=DishkaRoute,
)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_document(
    create_document_interactor: FromDishka[CreateDocumentV2Interactor],
    file: UploadFile = File(...),
):
    async def generate():
        async for doc_schema in create_document_interactor.execute(file):
            # Convert DocumentSchema to JSON string
            data = {
                "content": doc_schema.text,  # Используем text вместо content
                "channel": doc_schema.channel,
                "source": doc_schema.source,
                "metadata": doc_schema.metadata
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # Send final success message
        final_data = {"message": "Document created successfully", "type": "success"}
        yield f"data: {json.dumps(final_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.get("/")
async def get_documents(
    request: Annotated[GetSourcesParams, Query()],
    get_documents_interactor: FromDishka[GetAllDocumentsInteractor],
):
    """Get all documents with pagination and search"""
    return await get_documents_interactor.execute(request)
    

@router.get("/search")
async def search_documents(
    search_documents_interactor: FromDishka[SearchDocumentsInteractor],
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    similarity_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    source_id: str = Query(None, description="Search in specific source/document"),
    use_hybrid_search: bool = Query(True, description="Use hybrid search (semantic + keyword)"),
):
    """
    Search documents using hybrid search (semantic + keyword) with new architecture
    """
    results = await search_documents_interactor.execute(
        query=query,
        limit=limit,
        similarity_threshold=similarity_threshold,
        source_id=source_id,
        use_hybrid_search=use_hybrid_search
    )
    
    # Group results by source for better display
    sources_map = {}
    for result in results:
        source_id = result.source_id
        if source_id not in sources_map:
            sources_map[source_id] = {
                "source_id": source_id,
                "filename": result.filename,
                "file_size": result.file_size,
                "created_at": result.created_at,
                "max_similarity": result.similarity,
                "chunks_count": 1,
                "chunks": [result.model_dump()]
            }
        else:
            # Update max similarity
            if result.similarity > sources_map[source_id]["max_similarity"]:
                sources_map[source_id]["max_similarity"] = result.similarity
            
            sources_map[source_id]["chunks"].append(result.model_dump())
            sources_map[source_id]["chunks_count"] += 1
    
    # Sort sources by max similarity
    sorted_sources = sorted(
        sources_map.values(),
        key=lambda x: x["max_similarity"],
        reverse=True
    )
    
    return sorted_sources

@router.get("/{source_id}")
async def get_document(
    get_document_interactor: FromDishka[GetDocumentByIdInteractor],
    source_id: str,
) -> SourceResponse:
    """Get document by source ID with full details"""
    return await get_document_interactor.execute(source_id)
    
    
@router.delete("/{source_id}")
async def delete_document(
    delete_document_interactor: FromDishka[DeleteDocumentInteractor],
    source_id: str,
):
    """Delete document and all related data"""
    result = await delete_document_interactor.execute(source_id)
    return {
        "message": f"Successfully deleted document: {result.deleted_source}",
        "details": result.model_dump()
    }

@router.get("/{source_id}/stats")
async def get_document_stats(
    delete_document_interactor: FromDishka[DeleteDocumentInteractor],
    source_id: str,
):
    """Get document statistics before deletion"""
    return await delete_document_interactor.get_document_stats(source_id)

@router.get("/{source_id}/chunks")
async def get_document_chunks(
    get_document_chunks_interactor: FromDishka[GetDocumentChunksInteractor],
    source_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
):
    """Get document chunks with pagination"""
    return await get_document_chunks_interactor.execute(source_id, page, size)

@router.get("/{source_id}/search")
async def search_in_document(
    search_documents_interactor: FromDishka[SearchDocumentsInteractor],
    source_id: str,
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    similarity_threshold: float = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold"),
):
    """Search within a specific document"""
    return await search_documents_interactor.search_in_source(
        source_id=source_id,
        query=query,
        limit=limit,
        similarity_threshold=similarity_threshold
    )