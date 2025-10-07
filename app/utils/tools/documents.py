"""
Инструменты для работы с документами
"""
from typing import List, Dict, Any, Optional
from app.repositories.documents import DocumentsRepository
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.entities.documents import Document
from app.di.containers import app_container
from sentence_transformers import SentenceTransformer
from app.dto.ai_models import TextContent
from app.dto.pagination import InfiniteScrollRequest


async def search_documents(query: str, limit: int = 10) -> List[TextContent]:
    """
    Search for relevant documents using semantic vector search. Returns formatted text with document information.
    
    Use this tool to:
    - Find documents matching a query (e.g., company names, owners, directors, contracts, specifications, etc.)
    - Get relevant excerpts from documents with preview
    - Obtain document IDs for full content retrieval if needed

    Arguments:
        query (str): Natural language search query. Be specific and descriptive. Examples:
            - "Floriana Impex owner and directors information"
            - "vessel technical specifications and capabilities"
            - "contract terms, conditions and payment details"
            - "company registration and legal entity information"
        limit (int, optional): Maximum number of documents to return. Default is 10.

    Returns:
        List[TextContent]: Formatted text containing:
            - 📄 Found X documents matching 'query'
            - For each document:
                - **N. filename** - Document name
                - 🆔 ID: document_id | 📊 Size: file_size bytes
                - 📅 Uploaded: upload_date
                - 👁️ Preview: text preview with key fragments
                - ⭐ Relevance: score (0.0-1.0)
    
    If no results found, returns "No documents found matching query: 'query'".
    """
    try:
        print(f"[SEARCH] Query: '{query}', limit: {limit}")
        
        async with app_container() as container:
            # Получаем зависимости
            qdrant_embeddings_repository: QdrantEmbeddingsRepository = await container.get(QdrantEmbeddingsRepository)
            sentence_transformer: SentenceTransformer = await container.get(SentenceTransformer)
            
            # Генерируем эмбеддинг для запроса с префиксом E5
            # ВАЖНО: E5 требует "query: " для запросов
            query_with_prefix = "query: " + query
            query_vector = sentence_transformer.encode(
                [query_with_prefix], 
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].tolist()
            
            # Выполняем векторный поиск
            search_results = await qdrant_embeddings_repository.search_similar(
                query_vector=query_vector,
                limit=limit * 3,  # Получаем больше результатов для группировки
                similarity_threshold=0.4,
            )
            
            print(f"[SEARCH] Found {len(search_results)} raw chunks")
            
            if not search_results:
                return [TextContent(
                    type="text",
                    text=f"No documents found matching query: '{query}'"
                )]
            
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
                        "content_type": result.get("content_type", ""),
                    }
                
                # Сохраняем контент чанка
                chunk_content = result["chunk_content"]
                documents_dict[doc_id]["chunks"].append({
                    "content": chunk_content,
                    "similarity": result["similarity"],
                    "chunk_index": result["chunk_index"]
                })
                
                # Обновляем максимальную схожесть
                if result["similarity"] > documents_dict[doc_id]["max_similarity"]:
                    documents_dict[doc_id]["max_similarity"] = result["similarity"]
            
            # Формируем результат для LLM
            results = []
            for doc_id, doc_data in documents_dict.items():
                # Сортируем чанки: сначала по схожести, потом по индексу
                doc_data["chunks"].sort(
                    key=lambda x: (-x["similarity"], x["chunk_index"])
                )
                
                # Берем ТОП-3 чанка для превью
                best_chunks = doc_data["chunks"][:3]
                
                # Создаем превью из лучших чанков
                preview_parts = []
                for chunk in best_chunks:
                    content = chunk["content"]
                    if len(content) > 200:  # Ограничиваем превью
                        content = content[:200] + "..."
                    preview_parts.append(content)
                
                preview_text = " | ".join(preview_parts)
                
                # Подсчитываем примерный размер файла
                total_chunk_length = sum(len(chunk["content"]) for chunk in doc_data["chunks"])
                
                results.append({
                    "filename": doc_data["filename"],  # 📄 Название файла
                    "id": doc_id,  # 🆔 ID документа
                    "file_size": total_chunk_length,  # 📊 Размер файла в байтах
                    "uploaded_at": "N/A",  # 📅 Дата загрузки (не доступна из Qdrant)
                    "preview": preview_text,  # 👁️ Превью текста
                    "relevance_score": round(doc_data["max_similarity"], 3),  # ⭐ Релевантность
                    "content_type": doc_data["content_type"],  # Тип контента
                    "chunks_count": len(doc_data["chunks"]),  # Количество найденных чанков
                })
            
            # Сортируем по релевантности
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
          
            
            # Форматируем результат для LLM
            response = f"📄 Found {len(results)} documents matching '{query}':\n\n"
            
            for i, doc in enumerate(results, 1):
                response += f"**{i}. {doc['filename']}**\n"
                response += f"   🆔 ID: {doc['id']} | 📊 Size: {doc['file_size']:,} bytes\n"
                response += f"   📅 Uploaded: {doc['uploaded_at']}\n"
                response += f"   👁️ Preview: {doc['preview']}\n"
                response += f"   ⭐ Relevance: {doc['relevance_score']:.3f}\n\n"
            
            return [TextContent(type="text", text=response)]
            
    except Exception as e:
        print(f"[SEARCH ERROR] {str(e)}")
        return [TextContent(
            type="text",
            text=f"Error searching documents: {str(e)}"
        )]


async def get_document_by_id(document_id: str, include_content: bool = False) -> List[TextContent]:
    """
    Get document information by document ID.
    
    Use this tool when you need:
    - Document metadata and basic information
    - Truncated content preview (if include_content=True)
    - Document status and properties
    
    Args:
        document_id: The unique ID of the document (UUID string from search results)
        include_content: If True, includes truncated content preview (max 2000 chars)
        
    Returns:
        List[TextContent]: Formatted document information
    """
    try:
        async with app_container() as container:
            documents_repository: DocumentsRepository = await container.get(DocumentsRepository)
            
            # Получаем документ
            document = await documents_repository.get_one(
                where=[Document.id == document_id]
            )
            
            if not document:
                return [TextContent(
                    type="text",
                    text=f"❌ Document with ID {document_id} not found"
                )]
            
            # Формируем базовую информацию
            content_length = len(document.content) if document.content else 0
            status = document.status.value if hasattr(document.status, 'value') else str(document.status)
            created_at = document.created_at.isoformat() if hasattr(document, 'created_at') else "N/A"
            
            response = f"📄 **Document Information**\n\n"
            response += f"**Filename:** {document.original_filename}\n"
            response += f"**ID:** {document_id}\n"
            response += f"**Content Type:** {document.content_type}\n"
            response += f"**Status:** {status}\n"
            response += f"**Content Length:** {content_length:,} characters\n"
            response += f"**File Hash:** {document.file_hash}\n"
            response += f"**Created At:** {created_at}\n"
            
            if include_content and document.content:
                # Добавляем обрезанный контент
                truncated_content = document.content
                if len(truncated_content) > 2000:
                    truncated_content = truncated_content[:2000] + "..."
                
                response += f"\n**Content Preview:**\n"
                response += f"```\n{truncated_content}\n```\n"
                response += f"\n*Note: Content is truncated to 2000 characters. Use get_document_full_content for complete content.*"
            else:
                response += f"\n*Use get_document_full_content to retrieve the complete document content.*"
            
            return [TextContent(type="text", text=response)]
            
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"❌ Error retrieving document: {str(e)}"
        )]


async def get_document_full_content(document_id: str, chunk_size: int = 3000, chunk_index: int = 0) -> List[TextContent]:
    """
    Get full document content in chunks for LLM processing.
    
    Use this tool when you need:
    - Complete document content (not truncated)
    - Large documents that need to be processed in parts
    - Full text analysis and processing
    
    Args:
        document_id: The unique ID of the document (UUID string from search results)
        chunk_size: Size of each content chunk in characters (default: 3000)
        chunk_index: Which chunk to retrieve (0-based, default: 0)
        
    Returns:
        List[TextContent]: Document content chunk with pagination info
    """
    try:
        async with app_container() as container:
            documents_repository: DocumentsRepository = await container.get(DocumentsRepository)
            
            # Получаем документ
            document = await documents_repository.get_one(
                where=[Document.id == document_id]
            )
            
            if not document:
                return [TextContent(
                    type="text",
                    text=f"❌ Document with ID {document_id} not found"
                )]
            
            if not document.content:
                return [TextContent(
                    type="text",
                    text=f"❌ Document '{document.original_filename}' has no content"
                )]
            
            # Разбиваем контент на чанки
            content = document.content
            total_chunks = (len(content) + chunk_size - 1) // chunk_size  # Округление вверх
            
            if chunk_index >= total_chunks:
                return [TextContent(
                    type="text",
                    text=f"❌ Chunk index {chunk_index} is out of range. Document has {total_chunks} chunks (0-{total_chunks-1})"
                )]
            
            # Извлекаем нужный чанк
            start_pos = chunk_index * chunk_size
            end_pos = min(start_pos + chunk_size, len(content))
            chunk_content = content[start_pos:end_pos]
            
            # Формируем ответ
            response = f"📄 **Full Document Content - Chunk {chunk_index + 1}/{total_chunks}**\n\n"
            response += f"**Document:** {document.original_filename}\n"
            response += f"**ID:** {document_id}\n"
            response += f"**Chunk Size:** {chunk_size} characters\n"
            response += f"**Position:** {start_pos:,}-{end_pos:,} of {len(content):,} characters\n\n"
            response += f"**Content:**\n```\n{chunk_content}\n```\n\n"
            
            # Добавляем информацию о навигации
            if total_chunks > 1:
                response += f"**Navigation:**\n"
                if chunk_index > 0:
                    response += f"- Previous chunk: chunk_index={chunk_index-1}\n"
                if chunk_index < total_chunks - 1:
                    response += f"- Next chunk: chunk_index={chunk_index+1}\n"
                response += f"- All chunks: 0 to {total_chunks-1}\n"
            
            return [TextContent(type="text", text=response)]
            
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"❌ Error retrieving document content: {str(e)}"
        )]


async def query_documents(
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    count_only: bool = False,
    group_by: Optional[str] = None
) -> List[TextContent]:
    """
    Query documents with flexible filtering, sorting, and aggregation.
    
    Use this tool when you need:
    - Count documents (e.g., "how many documents do I have?")
    - Filter by status, content type, date range, etc.
    - Group documents by specific fields
    - Sort documents by various criteria
    - Get statistics about your document collection
    
    Args:
        filters: Dictionary of filters to apply. Available fields:
            - status: DocumentStatus (PENDING, PROCESSING, COMPLETED, FAILED)
            - content_type: str (e.g., "application/pdf", "text/plain")
            - filename: str (partial match)
            - original_filename: str (partial match)
            - created_after: str (ISO date, e.g., "2024-01-01")
            - created_before: str (ISO date, e.g., "2024-12-31")
            - min_content_length: int (minimum content length)
            - max_content_length: int (maximum content length)
        order_by: Field to sort by (e.g., "created_at", "filename", "content_length")
        limit: Maximum number of documents to return
        count_only: If True, returns only count without document details
        group_by: Field to group by (e.g., "status", "content_type")
        
    Returns:
        List[TextContent]: Query results with statistics and document information
    """
    try:
        async with app_container() as container:
            documents_repository: DocumentsRepository = await container.get(DocumentsRepository)
            
            # Строим условия WHERE
            where_conditions = []
            
            if filters:
                if "status" in filters:
                    where_conditions.append(Document.status == filters["status"])
                
                if "content_type" in filters:
                    where_conditions.append(Document.content_type == filters["content_type"])
                
                if "filename" in filters:
                    where_conditions.append(Document.filename.ilike(f"%{filters['filename']}%"))
                
                if "original_filename" in filters:
                    where_conditions.append(Document.original_filename.ilike(f"%{filters['original_filename']}%"))
                
                if "created_after" in filters:
                    from datetime import datetime
                    created_after = datetime.fromisoformat(filters["created_after"].replace('Z', '+00:00'))
                    where_conditions.append(Document.created_at >= created_after)
                
                if "created_before" in filters:
                    from datetime import datetime
                    created_before = datetime.fromisoformat(filters["created_before"].replace('Z', '+00:00'))
                    where_conditions.append(Document.created_at <= created_before)
                
                if "min_content_length" in filters:
                    where_conditions.append(Document.content.length() >= filters["min_content_length"])
                
                if "max_content_length" in filters:
                    where_conditions.append(Document.content.length() <= filters["max_content_length"])
            
            # Получаем документы
            
            # Создаем запрос с лимитом если нужно
            request_query = None
            if limit:
                request_query = InfiniteScrollRequest(limit=limit, offset=0)
            
            documents = await documents_repository.get_all(
                request_query=request_query,
                where=where_conditions if where_conditions else None
            )
            
            # Если нужен только счетчик
            if count_only:
                total_count = len(documents)
                response = f"📊 **Document Count**\n\n"
                response += f"**Total documents:** {total_count}\n"
                
                if filters:
                    response += f"\n**Applied filters:**\n"
                    for key, value in filters.items():
                        response += f"- {key}: {value}\n"
                
                return [TextContent(type="text", text=response)]
            
            # Группировка
            if group_by:
                groups = {}
                for doc in documents:
                    if group_by == "status":
                        key = doc.status.value if hasattr(doc.status, 'value') else str(doc.status)
                    elif group_by == "content_type":
                        key = doc.content_type
                    elif group_by == "filename":
                        key = doc.original_filename.split('.')[-1] if '.' in doc.original_filename else "unknown"
                    else:
                        key = "unknown"
                    
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(doc)
                
                response = f"📊 **Documents grouped by {group_by}**\n\n"
                for group_name, group_docs in groups.items():
                    response += f"**{group_name}:** {len(group_docs)} documents\n"
                    for doc in group_docs[:3]:  # Показываем первые 3 документа в группе
                        response += f"  - {doc.original_filename} ({doc.status.value if hasattr(doc.status, 'value') else doc.status})\n"
                    if len(group_docs) > 3:
                        response += f"  ... and {len(group_docs) - 3} more\n"
                    response += "\n"
                
                return [TextContent(type="text", text=response)]
            
            # Обычный список документов
            if not documents:
                response = f"📄 **No documents found**\n\n"
                if filters:
                    response += f"**Applied filters:**\n"
                    for key, value in filters.items():
                        response += f"- {key}: {value}\n"
                return [TextContent(type="text", text=response)]
            
            # Сортируем если нужно
            if order_by:
                if order_by == "created_at":
                    documents.sort(key=lambda x: x.created_at, reverse=True)
                elif order_by == "filename":
                    documents.sort(key=lambda x: x.original_filename.lower())
                elif order_by == "content_length":
                    documents.sort(key=lambda x: len(x.content) if x.content else 0, reverse=True)
                elif order_by == "status":
                    documents.sort(key=lambda x: x.status.value if hasattr(x.status, 'value') else str(x.status))
            
            # Формируем ответ
            response = f"📄 **Found {len(documents)} documents**\n\n"
            
            if filters:
                response += f"**Applied filters:**\n"
                for key, value in filters.items():
                    response += f"- {key}: {value}\n"
                response += "\n"
            
            if order_by:
                response += f"**Sorted by:** {order_by}\n\n"
            
            # Показываем документы
            for i, doc in enumerate(documents, 1):
                content_length = len(doc.content) if doc.content else 0
                status = doc.status.value if hasattr(doc.status, 'value') else str(doc.status)
                created_at = doc.created_at.isoformat() if hasattr(doc, 'created_at') else "N/A"
                
                response += f"**{i}. {doc.original_filename}**\n"
                response += f"   🆔 ID: {doc.id}\n"
                response += f"   📊 Status: {status}\n"
                response += f"   📄 Type: {doc.content_type}\n"
                response += f"   📏 Size: {content_length:,} characters\n"
                response += f"   📅 Created: {created_at}\n\n"
            
            return [TextContent(type="text", text=response)]
            
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"❌ Error querying documents: {str(e)}"
        )]


