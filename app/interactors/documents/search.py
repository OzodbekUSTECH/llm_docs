
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

from app.repositories.sources import SourcesRepository
from app.repositories.indexes import IndexesRepository
from app.entities.sources import Source
from app.entities.indexes import Index
from app.utils.docs_store import LanceDBDocumentStore
from app.utils.vectors_store import QdrantVectorStore
from app.utils.embeddings import FastEmbedEmbeddings
from app.utils.enums import IndexType
from app.exceptions.app_error import AppError

logger = logging.getLogger(__name__)


class SearchResultResponse(BaseModel):
    """Response model for search results"""
    document_id: str
    source_id: str
    filename: str
    file_size: int
    chunk: str
    full_chunk: str
    similarity: float
    semantic_score: float
    keyword_score: float
    chunk_index: int
    chunk_length: int
    page_label: str
    created_at: Optional[datetime] = None
    text_matches: int
    query_words: List[str]
    has_text_match: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
    text_relevance_score: Optional[float] = None


class SearchDocumentsInteractor:
    
    def __init__(
        self, 
        sources_repository: SourcesRepository,
        indexes_repository: IndexesRepository
    ):
        self.sources_repository = sources_repository
        self.indexes_repository = indexes_repository
        self.docs_store = LanceDBDocumentStore()
        self.vector_store = QdrantVectorStore()
        self.embedding = FastEmbedEmbeddings()

    async def execute(
        self, 
        query: str, 
        limit: int = 10, 
        similarity_threshold: float = 0.5,
        source_id: Optional[str] = None,
        use_hybrid_search: bool = True
    ) -> List[SearchResultResponse]:
        """
        Search documents using hybrid search (semantic + keyword) with new architecture
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold (0-1)
            source_id: ID of specific source to search in (optional)
            use_hybrid_search: Whether to use hybrid search (semantic + keyword)
        """
        try:
            logger.info(f"🔍 Searching: '{query}' (limit: {limit}, threshold: {similarity_threshold})")
            
            # 1. Semantic Search (Qdrant)
            query_embedding_docs = self.embedding.invoke(query)
            if not query_embedding_docs or not getattr(query_embedding_docs[0], "embedding", None):
                logger.error("Failed to create query embedding")
                return []
            
            query_embedding = query_embedding_docs[0].embedding
            
            # Search in Qdrant
            embeddings, similarities, doc_ids = await self.vector_store.query(
                embedding=query_embedding,
                top_k=limit * 2  # Get more for filtering
            )
            
            if not doc_ids:
                logger.info("No documents found in vector search")
                return []
            
            logger.info(f"✅ Vector search found {len(doc_ids)} documents")
            
            # 2. Keyword Search (LanceDB FTS) if enabled
            keyword_scores_dict = {}
            all_doc_ids = set(doc_ids)
            
            if use_hybrid_search:
                try:
                    fts_results = self.docs_store.query(query, top_k=limit * 2)
                    logger.info(f"🔤 FTS found {len(fts_results)} results")
                    
                    for i, doc in enumerate(fts_results):
                        doc_id = getattr(doc, "id_", None) or getattr(doc, "doc_id", None)
                        if doc_id:
                            # Score decreases with rank
                            score = 1.0 - (i * 0.05)
                            score = max(score, 0.1)
                            keyword_scores_dict[doc_id] = score
                            all_doc_ids.add(doc_id)
                except Exception as e:
                    logger.error(f"❌ Error in FTS: {e}")
            
            # 3. Get full documents from LanceDB
            logger.info(f"📚 Retrieving {len(all_doc_ids)} unique documents")
            all_docs = self.docs_store.get(list(all_doc_ids))
            doc_dict = {}
            for doc in all_docs:
                doc_id = getattr(doc, "id_", None) or getattr(doc, "doc_id", None)
                if doc_id:
                    doc_dict[doc_id] = doc
            
            # 4. Get source information for each document
            semantic_scores_dict = dict(zip(doc_ids, similarities))
            matches = []
            
            # Group documents by source_id from metadata
            source_doc_map = {}
            for doc_id in all_doc_ids:
                if doc_id not in doc_dict:
                    continue
                    
                doc = doc_dict[doc_id]
                doc_metadata = getattr(doc, "metadata", {})
                doc_source_id = doc_metadata.get("file_id")
                
                if doc_source_id:
                    if doc_source_id not in source_doc_map:
                        source_doc_map[doc_source_id] = []
                    source_doc_map[doc_source_id].append((doc_id, doc))
            
            # Filter by source_id if specified
            if source_id:
                source_doc_map = {k: v for k, v in source_doc_map.items() if k == source_id}
            
            # 5. Get source information
            source_ids = list(source_doc_map.keys())
            sources = []
            if source_ids:
                sources = await self.sources_repository.get_all(
                    where=[Source.id.in_(source_ids)]
                )
            sources_by_id = {source.id: source for source in sources}
            
            # 6. Build results
            for source_id, doc_list in source_doc_map.items():
                source = sources_by_id.get(source_id)
                if not source:
                    continue
                
                for doc_id, doc in doc_list:
                    semantic_score = semantic_scores_dict.get(doc_id, 0.0)
                    keyword_score = keyword_scores_dict.get(doc_id, 0.0)
                    
                    # Combine scores
                    if use_hybrid_search:
                        combined_score = 0.7 * semantic_score + 0.3 * keyword_score
                    else:
                        combined_score = semantic_score
                    
                    # Filter by threshold
                    if combined_score < similarity_threshold:
                        continue
                    
                    doc_content = getattr(doc, "text", None) or getattr(doc, "content", "")
                    doc_metadata = getattr(doc, "metadata", {})
                    
                    # Create smart preview
                    preview = self._create_smart_preview(doc_content, query)
                    
                    # Calculate text relevance
                    query_words = [word.lower().strip() for word in query.split() if len(word.strip()) > 2]
                    chunk_lower = doc_content.lower()
                    text_matches = sum(1 for word in query_words if word in chunk_lower) if query_words else 0
                    
                    matches.append(SearchResultResponse(
                        document_id=doc_id,
                        source_id=source_id,
                        filename=source.name,
                        file_size=source.size,
                        chunk=preview,
                        full_chunk=doc_content,
                        similarity=round(combined_score, 3),
                        semantic_score=round(semantic_score, 3),
                        keyword_score=round(keyword_score, 3),
                        chunk_index=doc_metadata.get("chunk_index", 0),
                        chunk_length=len(doc_content),
                        page_label=str(doc_metadata.get("page_label", "N/A")),
                        created_at=source.created_at,
                        text_matches=text_matches,
                        query_words=query_words,
                        has_text_match=text_matches > 0,
                        metadata=doc_metadata
                    ))
            
            # 7. Sort and limit results
            sorted_results = self._sort_results_optimally(matches, query)
            final_results = sorted_results[:limit]
            
            logger.info(f"✅ Returning {len(final_results)} search results")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Error in search: {e}", exc_info=True)
            raise AppError(status_code=500, message=f"Search failed: {str(e)}")
    
    async def search_in_source(
        self,
        source_id: str,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[SearchResultResponse]:
        """
        Search within a specific source/document
        
        Args:
            source_id: ID of the source to search in
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results from the specific source
        """
        return await self.execute(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            source_id=source_id,
            use_hybrid_search=True
        )
    
    def _create_smart_preview(self, chunk_content: str, query: str, max_length: int = 300) -> str:
        """
        Создает умное превью чанка с выделением релевантных частей
        
        Args:
            chunk_content: Содержимое чанка
            query: Поисковый запрос
            max_length: Максимальная длина превью
            
        Returns:
            Превью чанка с выделением релевантных частей
        """
        if len(chunk_content) <= max_length:
            return chunk_content
        
        # Ищем слова из запроса в чанке (регистронезависимо)
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        chunk_lower = chunk_content.lower()
        
        # Находим позиции слов из запроса
        word_positions = []
        for word in query_words:
            pos = chunk_lower.find(word)
            if pos != -1:
                word_positions.append(pos)
        
        if word_positions:
            # Начинаем превью с первого найденного слова
            start_pos = max(0, min(word_positions) - 50)
            end_pos = min(len(chunk_content), start_pos + max_length)
            
            preview = chunk_content[start_pos:end_pos]
            
            # Добавляем многоточие если обрезали
            if start_pos > 0:
                preview = "..." + preview
            if end_pos < len(chunk_content):
                preview = preview + "..."
                
            return preview
        else:
            # Если не нашли слова запроса, берем начало чанка
            return chunk_content[:max_length] + "..."
    
    def _sort_results_optimally(self, matches: List[SearchResultResponse], query: str) -> List[SearchResultResponse]:
        """
        Сортирует результаты поиска оптимальным образом:
        1. По убыванию similarity (самые релевантные сначала)
        2. По возрастанию chunk_index (порядок в документе)
        3. Дополнительная фильтрация по релевантности текста
        
        Args:
            matches: Список найденных чанков
            query: Поисковый запрос для дополнительной фильтрации
            
        Returns:
            Отсортированный список результатов
        """
        if not matches:
            return matches
        
        # Фильтруем результаты по текстовой релевантности
        filtered_matches = self._filter_by_text_relevance(matches, query)
        
        # Группируем результаты по документам
        documents_groups = {}
        for match in filtered_matches:
            doc_id = match.document_id
            if doc_id not in documents_groups:
                documents_groups[doc_id] = []
            documents_groups[doc_id].append(match)
        
        # Сортируем чанки внутри каждого документа по chunk_index
        for doc_id in documents_groups:
            documents_groups[doc_id].sort(key=lambda x: x.chunk_index)
        
        # Собираем результаты: сначала самые релевантные документы
        sorted_matches = []
        
        # Сортируем документы по максимальной similarity среди их чанков
        doc_similarities = {}
        for doc_id, chunks in documents_groups.items():
            max_similarity = max(chunk.similarity for chunk in chunks)
            doc_similarities[doc_id] = max_similarity
        
        # Сортируем документы по убыванию максимальной similarity
        sorted_doc_ids = sorted(
            documents_groups.keys(), 
            key=lambda x: doc_similarities[x], 
            reverse=True
        )
        
        # Добавляем чанки в порядке: документ -> chunk_index
        for doc_id in sorted_doc_ids:
            sorted_matches.extend(documents_groups[doc_id])
        
        return sorted_matches
    
    def _filter_by_text_relevance(self, matches: List[SearchResultResponse], query: str) -> List[SearchResultResponse]:
        """
        Фильтрует результаты по текстовой релевантности
        
        Args:
            matches: Список найденных чанков
            query: Поисковый запрос
            
        Returns:
            Отфильтрованный список результатов
        """
        if not query or not matches:
            return matches
        
        query_words = [word.lower().strip() for word in query.split() if len(word.strip()) > 2]
        if not query_words:
            return matches
        
        filtered_matches = []
        
        for match in matches:
            chunk_text = match.full_chunk.lower()
            
            # Проверяем, содержит ли чанк хотя бы одно слово из запроса
            contains_query_word = any(word in chunk_text for word in query_words)
            
            # Если содержит слова из запроса, добавляем с дополнительным весом
            if contains_query_word:
                # Подсчитываем количество совпадающих слов
                word_matches = sum(1 for word in query_words if word in chunk_text)
                match.text_relevance_score = word_matches / len(query_words)
                filtered_matches.append(match)
            else:
                # Если не содержит, но similarity высокая, все равно добавляем
                # но с пониженным приоритетом
                if match.similarity > 0.8:  # Высокая similarity
                    match.text_relevance_score = 0.1
                    filtered_matches.append(match)
        
        # Сортируем по текстовой релевантности, затем по similarity
        filtered_matches.sort(
            key=lambda x: (x.text_relevance_score or 0, x.similarity), 
            reverse=True
        )
        
        return filtered_matches