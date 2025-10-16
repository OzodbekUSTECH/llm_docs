"""
Инструменты для работы с документами
"""
from typing import List, Dict, Any, Optional
from sqlalchemy import and_, func
from app.repositories.documents import DocumentsRepository
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.entities.documents import Document
from app.di.containers import app_container
from sentence_transformers import SentenceTransformer
from app.dto.ai_models import TextContent
from app.dto.pagination import InfiniteScrollRequest
from app.utils.collections import Collections
from qdrant_client.http.models import FieldCondition, MatchAny
from app.dto.qdrant_filters import QdrantFilters
from app.utils.enums import DocumentType

def format_datetime(date_obj, include_time=True):
    """Format datetime object to readable string"""
    if not date_obj:
        return "Unknown"
    
    try:
        if include_time:
            return date_obj.strftime("%B %d, %Y at %H:%M")
        else:
            return date_obj.strftime("%B %d, %Y")
    except Exception:
        return "Invalid Date"

async def search_documents(query: str, limit: int = 10, document_ids: Optional[List[str]] = None, document_types: Optional[List[str]] = None) -> List[TextContent]:
    """
    Search for relevant documents using semantic vector search. Returns document metadata and keywords.
    
    Use this tool to:
    - Find documents matching a query (e.g., company names, owners, directors, contracts, specifications, etc.)
    - Get document metadata and extracted keywords
    - Search within specific documents by providing document IDs
    - Obtain document IDs for further analysis if needed
    - Documents are automatically categorized by type (INVOICE, CONTRACT, COO, COA, COW, BL, LC, FINANCIAL, OTHER)

    Arguments:
        query (str): Natural language search query. Be specific and descriptive. Examples:
            - "Floriana Impex owner and directors information"
            - "vessel technical specifications and capabilities"
            - "contract terms, conditions and payment details"
            - "company registration and legal entity information"
            - "invoice payment terms" - will find INVOICE type documents
            - "certificate of origin" - will find COO type documents
        
        Usage with document_types parameter:
            - search_documents("payment terms", document_types=["INVOICE"]) - only invoices
            - search_documents("weight specifications", document_types=["COW", "COA"]) - certificates only
            - search_documents("contract details", document_types=["CONTRACT", "BL"]) - contracts and bills of lading
        limit (int, optional): Maximum number of documents to return. Default is 10.
        document_ids (List[str], optional): List of specific document IDs to search within. 
            If provided, search will be limited to these documents only. Useful for large document collections.
        document_types (List[str], optional): List of document types to filter by. Examples:
            - ["INVOICE"] - only invoices
            - ["CONTRACT", "INVOICE"] - only contracts and invoices
            - ["COO", "COA", "COW"] - only certificates

    Returns:
        List[TextContent]: Formatted text containing:
            - 📄 Found X documents matching 'query'
            - For each document:
                - **N. filename** - Document name
                - 🆔 ID: document_id | 📊 Size: file_size bytes
                - 📋 Type: document_type | 📄 Format: content_type
                - 🔑 Keywords: extracted key information
                - ⭐ Relevance: score (0.0-1.0)
    
    Document Types:
        - INVOICE: Invoices and bills
        - CONTRACT: Contracts and agreements
        - COO: Certificate of Origin
        - COA: Certificate of Analysis
        - COW: Certificate of Weight
        - BL: Bill of Lading
        - FINANCIAL: Financial reports and statements
        - OTHER: Other document types
    
    If no results found, returns "No documents found matching query: 'query'".
    """
    try:
        print(f"[SEARCH] Query: '{query}', limit: {limit}, document_ids: {document_ids}, document_types: {document_types}")
        
        async with app_container() as container:
            # Получаем зависимости
            qdrant_embeddings_repository: QdrantEmbeddingsRepository = await container.get(QdrantEmbeddingsRepository)
            sentence_transformer: SentenceTransformer = await container.get(SentenceTransformer)
            documents_repository: DocumentsRepository = await container.get(DocumentsRepository)
            
            # Генерируем эмбеддинг для запроса с префиксом E5
            # ВАЖНО: E5 требует "query: " для запросов
            query_with_prefix = "query: " + query
            query_vector = sentence_transformer.encode(
                [query_with_prefix], 
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].tolist()
            
            # Создаем фильтры для поиска
            filter_conditions = []
            
            # Фильтр по ID документов
            if document_ids:
                filter_conditions.append(
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=document_ids)
                    )
                )
            
            # Фильтр по типам документов
            if document_types:
                filter_conditions.append(
                    FieldCondition(
                        key="document_type",
                        match=MatchAny(any=document_types)
                    )
                )
            
            # Создаем объект фильтров если есть условия
            filters = None
            if filter_conditions:
                filters = QdrantFilters(must=filter_conditions)
            
            search_results = await qdrant_embeddings_repository.search_similar(
                collection_name=Collections.DOCUMENT_EMBEDDINGS,
                query_vector=query_vector,
                limit=limit,
                similarity_threshold=0.7,  # Только релевантные результаты
                filters=filters
            )
            
            print(f"[SEARCH] Found {len(search_results)} raw chunks")
            
            if not search_results:
                return [TextContent(
                    type="text",
                    text=f"No documents found matching query: '{query}'"
                )]
            
            # Группируем результаты по документам и получаем максимальную схожесть
            documents_scores = {}
            for result in search_results:
                doc_id = result.payload.get("document_id")
                if doc_id not in documents_scores:
                    documents_scores[doc_id] = {
                        "max_similarity": 0,
                        "filename": result.payload.get("filename", ""),
                        "content_type": result.payload.get("content_type", ""),
                        "document_type": result.payload.get("document_type", "OTHER"),
                        "chunks": []  # Добавляем список чанков
                    }
                
                # Обновляем максимальную схожесть и добавляем чанк
                if result.score > documents_scores[doc_id]["max_similarity"]:
                    documents_scores[doc_id]["max_similarity"] = result.score
                
                # Добавляем чанк с его содержимым и релевантностью
                documents_scores[doc_id]["chunks"].append({
                    "content": result.payload.get("chunk_content", ""),
                    "score": result.score,
                    "chunk_index": result.payload.get("chunk_index", 0)
                })
            
            # Сортируем чанки по релевантности для каждого документа
            for doc_id, doc_data in documents_scores.items():
                doc_data["chunks"].sort(key=lambda x: x["score"], reverse=True)
            
            # Получаем документы из БД одним запросом
            doc_ids = list(documents_scores.keys())
            
            documents = await documents_repository.get_all(
                where=[Document.id.in_(doc_ids)]
            )
            
            # Создаем мапу id -> document для быстрого доступа
            documents_map = {str(doc.id): doc for doc in documents}
            
            # Формируем результат для LLM
            results = []
            for doc_id, doc_data in documents_scores.items():
                doc = documents_map.get(doc_id)
                if not doc:
                    continue
                
                
                keywords_text = " | ".join(doc.keywords) if doc.keywords else "No keywords"
                
                # Формируем содержимое всех найденных чанков без обрезаний
                chunks_content = []
                for i, chunk in enumerate(doc_data["chunks"]):
                    chunks_content.append(f"Chunk {i+1} (score: {chunk['score']:.3f}): {chunk['content']}")
                
                results.append({
                    "filename": doc.original_filename,
                    "id": doc_id,
                    "file_size": len(doc.content) if doc.content else 0,
                    "keywords": keywords_text,
                    "relevance_score": round(doc_data["max_similarity"], 3),
                    "content_type": doc.content_type,
                    "document_type": doc.type.value,
                    "chunks": chunks_content,
                    "total_chunks": len(doc_data["chunks"])
                })
            
            # Сортируем по релевантности
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Форматируем результат для LLM
            filter_info = []
            if document_ids:
                filter_info.append(f"limited to {len(document_ids)} specified documents")
            if document_types:
                filter_info.append(f"filtered by types: {', '.join(document_types)}")
            
            if filter_info:
                response = f"📄 Found {len(results)} documents matching '{query}' ({' and '.join(filter_info)}):\n\n"
            else:
                response = f"📄 Found {len(results)} documents matching '{query}':\n\n"
            
            for i, doc in enumerate(results, 1):
                response += f"**{i}. {doc['filename']}**\n"
                response += f"   🆔 ID: {doc['id']} | 📊 Size: {doc['file_size']:,} characters\n"
                response += f"   📋 Type: {doc['document_type']} | 📄 Format: {doc['content_type']}\n"
                if doc['keywords']:
                    response += f"   🔑 Keywords: {doc['keywords']}\n"
                response += f"   ⭐ Relevance: {doc['relevance_score']:.3f}\n"
                
                # Добавляем все найденные чанки
                if doc['chunks']:
                    response += f"   📝 Found {doc['total_chunks']} relevant chunks:\n"
                    for chunk in doc['chunks']:
                        response += f"      • {chunk}\n"
                
                response += "\n"
            
            return [TextContent(type="text", text=response)]
            
    except Exception as e:
        print(f"[SEARCH ERROR] {str(e)}")
        return [TextContent(
            type="text",
            text=f"Error searching documents: {str(e)}"
        )]


async def get_document_by_id(document_id: str) -> List[TextContent]:
    """
    Get document information by document ID including metadata and keywords.
    
    Use this tool when you need:
    - Document metadata and basic information
    - Extracted keywords and their values
    - Document status and properties
    
    Args:
        document_id: The unique ID of the document (UUID string from search results)
        
    Returns:
        List[TextContent]: Formatted document information with keywords
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
            response += f"**Document Type:** {document.type.value}\n"
            response += f"**Status:** {status}\n"
            response += f"**Content Length:** {content_length:,} characters\n"
            response += f"**File Hash:** {document.file_hash}\n"
            response += f"**Created At:** {created_at}\n"
            
            # Добавляем ключевые слова если есть
            if document.keywords:
                response += f"\n**🔑 Extracted Keywords:**\n"
                for key, value in document.keywords.items():
                    if value:
                        response += f"- **{key}**: {value}\n"
            else:
                response += f"\n**🔑 Keywords:** No keywords extracted\n"
            return [TextContent(type="text", text=response)]
            
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"❌ Error retrieving document: {str(e)}"
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
    - Filter by status, content type, document type, date range, etc.
    - Group documents by specific fields
    - Sort documents by various criteria
    - Get statistics about your document collection
    
    Args:
        filters: Dictionary of filters to apply. Available fields:
            - status: DocumentStatus (PENDING, PROCESSING, COMPLETED, FAILED)
            - content_type: str (e.g., "application/pdf", "text/plain")
            - filename: str (partial match)
            - document_type: DocumentType (e.g., INVOICE, CONTRACT, COO, COA, COW, BL, LC, FINANCIAL, OTHER)
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
                    
                if "document_type" in filters:
                    where_conditions.append(Document.type == DocumentType(filters["document_type"]))
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
                response += f"   📄 Content Type: {doc.content_type}\n"
                response += f"   📄 Document Type: {doc.type}\n"
                response += f"   📏 Size: {content_length:,} characters\n"
                response += f"   📅 Created: {created_at}\n\n"
            
            return [TextContent(type="text", text=response)]
            
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"❌ Error querying documents: {str(e)}"
        )]


async def search_documents_by_keywords(
    keyword: str, 
    value: Optional[str] = None, 
    document_types: Optional[List[str]] = None, 
    limit: int = 10
) -> List[TextContent]:
    """
    Search for documents by specific keywords extracted from their content.
    
    This tool allows you to find documents containing specific information like vessel names,
    invoice numbers, contract details, etc. Use this when you need to find documents with
    specific data points or values.
    
    Arguments:
        keyword (str): The specific keyword to search for (e.g., 'vessel', 'invoice_number', 'contract_number', 'seller', 'buyer')
        value (Optional[str]): The value to search for within that keyword field. If not provided, finds all documents with this keyword.
        document_types (Optional[List[str]]): List of document types to search within. If not provided, searches all types.
        limit (int): Maximum number of documents to return. Default is 10.
    
    Returns:
        List[TextContent]: Formatted results with document information and matching keyword values.
    """
    try:
        # Get repositories from DI container
        async with app_container() as container:
            documents_repository = await container.get(DocumentsRepository)
            
            # Build query conditions for database search
            conditions = []
            
            # Add document type filter if specified
            if document_types:
                conditions.append(Document.type.in_(document_types))
            
            # Add keyword search condition using JSON operations
            if value:
                # Search for documents where the keyword exists and contains the value
                keyword_condition = and_(
                    func.jsonb_exists(Document.keywords, keyword),
                    func.jsonb_path_exists(
                        Document.keywords,
                        f'$."{keyword}" ? (@ like_regex "{value}" flag "i")'
                    )
                )
            else:
                # Search for documents where the keyword exists (any value)
                keyword_condition = func.jsonb_exists(Document.keywords, keyword)
            
            conditions.append(keyword_condition)
            
            # Get documents matching the conditions with limit
            from app.dto.pagination import InfiniteScrollRequest
            request = InfiniteScrollRequest(limit=limit, offset=0)
            documents = await documents_repository.get_all(
                request_query=request,
                where=conditions,
            )
            
            # Prepare matching documents with keyword values
            matching_documents = []
            for doc in documents:
                if not doc.keywords or keyword not in doc.keywords:
                    continue
                
                keyword_data = doc.keywords[keyword]
                if isinstance(keyword_data, dict):
                    keyword_value = keyword_data.get('value', '')
                else:
                    keyword_value = str(keyword_data)
                
                matching_documents.append((doc, keyword_value))
            
            if not matching_documents:
                if value:
                    return [TextContent(
                        type="text",
                        text=f"🔍 No documents found with keyword '{keyword}' containing value '{value}'"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"🔍 No documents found with keyword '{keyword}'"
                    )]
            
            # Format results
            if value:
                response = f"🔍 **Found {len(matching_documents)} documents with keyword '{keyword}' containing '{value}'**\n\n"
            else:
                response = f"🔍 **Found {len(matching_documents)} documents with keyword '{keyword}'**\n\n"
            
            for i, (doc, keyword_value) in enumerate(matching_documents, 1):
                # Format creation date
                created_at = format_datetime(doc.created_at, include_time=True)
                
                # Get document type icon and name
                type_icons = {
                    DocumentType.INVOICE: "📄",
                    DocumentType.CONTRACT: "📋", 
                    DocumentType.COO: "🌍",
                    DocumentType.COA: "🧪",
                    DocumentType.COW: "⚖️",
                    DocumentType.COQ: "🏆",
                    DocumentType.BL: "🚢",
                    DocumentType.LC: "💳",
                    DocumentType.FINANCIAL: "💰",
                    DocumentType.OTHER: "📁"
                }
                type_icon = type_icons.get(doc.type, "📄")
                
                response += f"**{i}. {type_icon} {doc.original_filename}**\n"
                response += f"   🆔 ID: `{doc.id}`\n"
                response += f"   📋 Type: {doc.type.value}\n"
                response += f"   🔑 **{keyword}**: {keyword_value}\n"
                response += f"   📅 Created: {created_at}\n"
                
                response += "\n"
            
            return [TextContent(type="text", text=response)]
        
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"❌ Error searching documents by keywords: {str(e)}"
        )]


