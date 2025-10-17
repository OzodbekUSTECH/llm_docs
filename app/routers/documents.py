
from uuid import UUID
from fastapi import APIRouter, status, UploadFile, File, Query
from dishka.integrations.fastapi import FromDishka, DishkaRoute
from typing import Annotated, List, Dict, Any


from app.interactors.documents.create import CreateDocumentInteractor
from app.interactors.documents.delete import DeleteDocumentInteractor
from app.interactors.documents.search import SearchDocumentsInteractor
from app.interactors.documents.get import GetAllDocumentsInteractor, GetDocumentByIdInteractor
from app.interactors.documents.get_chunks import GetDocumentChunksInteractor
from app.dto.documents import GetDocumentsParams, DocumentListResponse, DocumentResponse
from app.dto.pagination import PaginatedResponse


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    route_class=DishkaRoute,
)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_document(
    create_document_interactor: FromDishka[CreateDocumentInteractor],
    file: UploadFile = File(...),
):
    await create_document_interactor.execute(file)
    return {"message": "Document created successfully"}


@router.get("/", response_model=PaginatedResponse[DocumentListResponse])
async def get_documents(
    get_documents_interactor: FromDishka[GetAllDocumentsInteractor],
    request: Annotated[GetDocumentsParams, Query()],
):
    return await get_documents_interactor.execute(request)

@router.get("/search", response_model=List[Dict[str, Any]])
async def search_documents(
    search_documents_interactor: FromDishka[SearchDocumentsInteractor],
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    document_types: str = Query(None, description="Comma-separated list of document types to filter by"),
):
    """
    Поиск документов по текстовому запросу с использованием векторного поиска
    """
    # Парсим document_types из строки в список
    document_types_list = None
    if document_types:
        document_types_list = [dt.strip() for dt in document_types.split(',') if dt.strip()]
    
    chunks_results = await search_documents_interactor.execute(
        query=query,
        limit=limit,
        similarity_threshold=similarity_threshold,
        document_types=document_types_list
    )
    
    # Группируем результаты по документам
    documents_map = {}
    for chunk in chunks_results:
        doc_id = chunk["document_id"]
        if doc_id not in documents_map:
            documents_map[doc_id] = {
                "id": doc_id,  # Добавляем id для совместимости с фронтендом
                "document_id": doc_id,
                "filename": chunk["filename"],
                "original_filename": chunk["filename"],
                "content_type": chunk["content_type"],
                "created_at": chunk["created_at"],
                "max_similarity": chunk["similarity"],
                "chunks_count": 1,
                "best_chunks": [{
                    "content": chunk["full_chunk"],
                    "preview": chunk["chunk"],
                    "similarity": chunk["similarity"],
                    "chunk_index": chunk["chunk_index"]
                }]
            }
        else:
            # Обновляем максимальное сходство
            if chunk["similarity"] > documents_map[doc_id]["max_similarity"]:
                documents_map[doc_id]["max_similarity"] = chunk["similarity"]
            
            # Добавляем чанк
            documents_map[doc_id]["best_chunks"].append({
                "content": chunk["full_chunk"],
                "preview": chunk["chunk"],
                "similarity": chunk["similarity"],
                "chunk_index": chunk["chunk_index"]
            })
            documents_map[doc_id]["chunks_count"] += 1
    
    # Сортируем документы по максимальному сходству
    results = sorted(
        documents_map.values(),
        key=lambda x: x["max_similarity"],
        reverse=True
    )
    
    return results

@router.get("/{id}", response_model=DocumentResponse)
async def get_document(
    get_document_interactor: FromDishka[GetDocumentByIdInteractor],
    id: UUID,
):
    return await get_document_interactor.execute(id)





@router.delete("/{id}")
async def delete_document(
    delete_document_interactor: FromDishka[DeleteDocumentInteractor],
    id: UUID,
):
    return await delete_document_interactor.execute(id)

@router.get("/{id}/file-url")
async def get_document_file_url(
    get_document_interactor: FromDishka[GetDocumentByIdInteractor],
    id: UUID,
):
    """
    Получить URL файла документа для скачивания или открытия в браузере
    """
    document = await get_document_interactor.execute(id)
    return {
        "file_url": f"/storage/documents/{document.filename}",
        "filename": document.original_filename,
        "content_type": document.content_type
    }
    

@router.get("/{id}/chunks", response_model=List[Dict[str, Any]])
async def get_document_chunks(
    get_document_chunks_interactor: FromDishka[GetDocumentChunksInteractor],
    id: UUID,
):
    """
    Вернуть все чанки документа по порядку chunk_index ASC
    """
    return await get_document_chunks_interactor.execute(str(id))
    
