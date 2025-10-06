"""
Инструменты для работы с документами
"""
from typing import List, Dict, Any, Optional
from uuid import UUID
from app.repositories.documents import DocumentsRepository
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.entities.documents import Document
from app.di.containers import app_container
from app.repositories.uow import UnitOfWork
from sentence_transformers import SentenceTransformer


def _combine_sequential_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Объединяет последовательные чанки в единый текст
    
    Args:
        chunks: Список чанков с индексами
        
    Returns:
        Объединенный текст
    """
    if not chunks:
        return ""
    
    # Сортируем по chunk_index
    sorted_chunks = sorted(chunks, key=lambda x: x["chunk_index"])
    
    # Объединяем текст
    combined = []
    prev_index = None
    
    for chunk in sorted_chunks:
        current_index = chunk["chunk_index"]
        content = chunk["content"]
        
        # Если чанки последовательные, просто добавляем
        if prev_index is None or current_index == prev_index + 1:
            combined.append(content)
        else:
            # Если пропуск, добавляем разделитель
            combined.append("\n\n[...]\n\n")
            combined.append(content)
        
        prev_index = current_index
    
    return "\n\n".join(combined)


async def search_documents(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for relevant documents using semantic vector search. Returns document IDs and FULL relevant chunks.
    
    IMPORTANT: This returns COMPLETE CHUNKS without truncation for maximum information quality.
    To get FULL document content, use get_document_by_id with the 'id' field from search results.

    Use this tool to:
    - Find documents matching a query (e.g., company names, owners, directors, contracts, specifications, etc.)
    - Get comprehensive relevant excerpts from documents with full context
    - Obtain document IDs for full content retrieval if needed

    Arguments:
        query (str): Natural language search query. Be specific and descriptive. Examples:
            - "Floriana Impex owner and directors information"
            - "vessel technical specifications and capabilities"
            - "contract terms, conditions and payment details"
            - "company registration and legal entity information"
        limit (int, optional): Maximum number of documents to return. Default is 10 for comprehensive results.

    Returns:
        List[Dict]: Each result contains:
            - id: Document UUID (use with get_document_by_id for full content)
            - filename: Document filename
            - relevant_content: Combined FULL relevant chunks from the document (NO truncation)
            - best_chunks: Individual matching chunks with similarity scores (COMPLETE chunks)
            - max_similarity: Highest similarity score (0-1, higher is better)
            - chunks_count: Total number of relevant chunks found
    
    If no results found, returns empty list [].
    """
    try:
        print(f"[SEARCH] Query: '{query}', limit: {limit}")
        
        async with app_container() as container:
            # Получаем зависимости
            documents_repository: DocumentsRepository = await container.get(DocumentsRepository)
            qdrant_embeddings_repository: QdrantEmbeddingsRepository = await container.get(QdrantEmbeddingsRepository)
            sentence_transformer: SentenceTransformer = await container.get(SentenceTransformer)
            
            # Генерируем эмбеддинг для запроса с префиксом E5
            # ВАЖНО: E5 требует "query: " для запросов
            query_with_prefix = "query: " + query
            query_vector = sentence_transformer.encode(
                [query_with_prefix], 
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=1,
                show_progress_bar=False
            )[0].tolist()
            
            # Выполняем векторный поиск с улучшенными параметрами
            search_results = await qdrant_embeddings_repository.search_similar(
                query_vector=query_vector,
                limit=limit * 5,  # Получаем БОЛЬШЕ результатов для лучшего покрытия
                similarity_threshold=0.4,  # НИЖЕ порог для более широкого поиска
                document_id=None
            )
            
            print(f"[SEARCH] Found {len(search_results)} raw chunks")
            
            # Группируем результаты по документам
            documents_dict = {}
            for result in search_results:
                doc_id = result["document_id"]
                if doc_id not in documents_dict:
                    documents_dict[doc_id] = {
                        "document_id": doc_id,
                        "chunks": [],
                        "max_similarity": 0,
                        "filename": result.get("filename", ""),
                    }
                
                # ВАЖНО: Сохраняем ПОЛНЫЙ контент чанка БЕЗ ОБРЕЗАНИЯ
                documents_dict[doc_id]["chunks"].append({
                    "content": result["chunk_content"],  # ПОЛНЫЙ контент без truncation
                    "similarity": result["similarity"],
                    "chunk_index": result["chunk_index"]
                })
                
                # Обновляем максимальную схожесть
                if result["similarity"] > documents_dict[doc_id]["max_similarity"]:
                    documents_dict[doc_id]["max_similarity"] = result["similarity"]
            
            # Получаем информацию о документах
            document_ids = list(documents_dict.keys())
            if document_ids:
                documents = await documents_repository.get_all(
                    where=[Document.id.in_(document_ids)]
                )
                documents_by_id = {str(doc.id): doc for doc in documents}
            else:
                documents_by_id = {}
            
            # Формируем результат для LLM
            results = []
            for doc_id, doc_data in documents_dict.items():
                doc = documents_by_id.get(doc_id)
                if doc:
                    # Сортируем чанки: сначала по схожести, потом по индексу
                    doc_data["chunks"].sort(
                        key=lambda x: (-x["similarity"], x["chunk_index"])
                    )
                    
                    # Берем ТОП-5 чанков для одного документа (было 3, стало 5)
                    best_chunks = doc_data["chunks"][:5]
                    
                    # ВАЖНО: НЕ обрезаем контент - возвращаем ПОЛНОСТЬЮ
                    # Объединяем чанки в один текст если они последовательные
                    combined_content = _combine_sequential_chunks(best_chunks)
                    
                    # УБРАЛИ ограничение на 3000 символов - возвращаем ВСЁ
                    # LLM достаточно умная, чтобы обработать больше информации
                    # И это даст более полные и точные ответы
                    
                    results.append({
                        "id": doc_id,
                        "filename": doc.original_filename,
                        "content_type": doc.content_type,
                        "created_at": doc.created_at.isoformat() if hasattr(doc, 'created_at') else None,
                        "max_similarity": round(doc_data["max_similarity"], 3),
                        "chunks_count": len(doc_data["chunks"]),
                        "relevant_content": combined_content,  # ПОЛНЫЙ объединенный контент
                        "best_chunks": [
                            {
                                # УБРАЛИ обрезание до 1000 символов - возвращаем ПОЛНОСТЬЮ
                                "content": chunk["content"],  # ПОЛНЫЙ контент чанка
                                "similarity": round(chunk["similarity"], 3),
                                "chunk_index": chunk["chunk_index"]
                            }
                            for chunk in best_chunks
                        ]
                    })
            
            # Сортируем по максимальной схожести
            results.sort(key=lambda x: x["max_similarity"], reverse=True)
            
            final_results = results[:limit]
            print(f"[SEARCH] Returning {len(final_results)} documents with FULL content:")
            for r in final_results:
                print(f"  - {r['filename']} (similarity: {r['max_similarity']})")
                print(f"    Total content length: {len(r['relevant_content'])} chars")
                print(f"    Chunks: {len(r['best_chunks'])}")
            
            return final_results
            
    except Exception as e:
        print(f"[SEARCH ERROR] {str(e)}")
        return [{"error": f"Error searching documents: {str(e)}"}]


async def get_document_by_id(document_id: str) -> Dict[str, Any]:
    """
    Get full document content by document ID.
    
    Use this tool when you need:
    - Full document content (not just chunks)
    - Complete tables and structured data
    - All metadata of a specific document
    
    Args:
        document_id: The unique ID of the document (UUID string from search results)
        
    Returns:
        Complete document information including full content, tables, and metadata
    """
    try:
        async with app_container() as container:
            documents_repository: DocumentsRepository = await container.get(DocumentsRepository)
            
            # Получаем документ
            document = await documents_repository.get_one(
                where=[Document.id == document_id]
            )
            
            if not document:
                return {"error": f"Document with ID {document_id} not found"}
            
            return {
                "id": str(document.id),
                "filename": document.original_filename,
                "content_type": document.content_type,
                "status": document.status.value if hasattr(document.status, 'value') else str(document.status),
                "full_content": document.content,
                "tables": document.tables or [],
                "doc_metadata": {},
                "created_at": document.created_at.isoformat() if hasattr(document, 'created_at') else None,
                "content_length": len(document.content) if document.content else 0,
                "file_hash": document.file_hash,
                "description": f"Full content of '{document.original_filename}' ({len(document.content) if document.content else 0} characters)"
            }
            
    except Exception as e:
        return {"error": f"Error retrieving document: {str(e)}"}


async def list_documents(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Список всех документов
    
    Args:
        limit: Максимальное количество документов
        
    Returns:
        Список документов
    """
    try:
        async with app_container() as container:
            documents_repository: DocumentsRepository = await container.get(DocumentsRepository)
            documents = await documents_repository.get_all(limit=limit)
            
            results = []
            for doc in documents:
                results.append({
                    "id": str(doc.id),
                    "filename": doc.original_filename,
                    "content_type": doc.content_type,
                    "status": doc.status.value if hasattr(doc.status, 'value') else str(doc.status),
                    "created_at": doc.created_at.isoformat() if hasattr(doc, 'created_at') else None,
                    "content_length": len(doc.content) if doc.content else 0
                })
            
            return results
            
    except Exception as e:
        return [{"error": f"Ошибка при получении списка документов: {str(e)}"}]


async def upload_document(filename: str, content: str, content_type: str = "text/plain") -> Dict[str, Any]:
    """
    Загрузка нового документа (симуляция)
    
    Args:
        filename: Имя файла
        content: Содержимое файла
        content_type: Тип содержимого
        
    Returns:
        Результат загрузки
    """
    # Это симуляция - в реальности нужно использовать CreateDocumentInteractor
    return {
        "message": f"Документ '{filename}' успешно загружен",
        "filename": filename,
        "content_type": content_type,
        "content_length": len(content),
        "note": "Это симуляция загрузки. Для реальной загрузки используйте API endpoint /documents/upload"
    }

