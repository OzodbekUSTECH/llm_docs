
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import FieldCondition, MatchValue, MatchAny

from app.dto.qdrant_filters import QdrantFilters
from app.repositories.documents import DocumentsRepository
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.entities.documents import Document
from app.utils.collections import Collections


class SearchDocumentsInteractor:
    
    def __init__(
        self, 
        documents_repository: DocumentsRepository,
        qdrant_embeddings_repository: QdrantEmbeddingsRepository,
        sentence_transformer: SentenceTransformer
    ):
        self.documents_repository = documents_repository
        self.qdrant_embeddings_repository = qdrant_embeddings_repository
        self.sentence_transformer = sentence_transformer

    async def execute(
        self, 
        query: str, 
        limit: int = 10, 
        similarity_threshold: float = 0.7,
        document_id: str = None,
        document_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск документов по текстовому запросу с использованием векторного поиска в Qdrant
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            similarity_threshold: Минимальный порог схожести (0-1)
            document_id: ID конкретного документа для поиска (опционально)
            document_types: Список типов документов для фильтрации (опционально)
        """
        
        # 1. Генерируем embedding для запроса с префиксом "query: "
        # ВАЖНО: E5 модели требуют "query: " для запросов и "passage: " для документов
        query_with_prefix = "query: " + query
        query_vector = self.sentence_transformer.encode(
            [query_with_prefix], 
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].tolist()
        
        print(f"🔍 Поиск: '{query}' (с префиксом E5: 'query:')")

        # 2. Создаем фильтры для поиска
        filter_conditions = []
        
        # Фильтр по ID документа
        if document_id:
            filter_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
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
        
        search_results = await self.qdrant_embeddings_repository.search_similar(
            collection_name=Collections.DOCUMENT_EMBEDDINGS,
            query_vector=query_vector,
            limit=limit,
            similarity_threshold=similarity_threshold,
            filters=filters
        )

        # 3. получаем информацию о документах для найденных чанков
        matches: List[Dict[str, Any]] = []
        document_ids = list(set(result.payload.get("document_id") for result in search_results))
        
        if document_ids:
            # Получаем информацию о документах
            documents = await self.documents_repository.get_all(
                where=[Document.id.in_(document_ids)]
            )
            documents_by_id = {str(doc.id): doc for doc in documents}
            
            # Формируем результат с дополнительной информацией
            for result in search_results:
                doc_id = result.payload.get("document_id")
                doc = documents_by_id.get(doc_id)
                if doc:
                    chunk_content = result.payload.get("chunk_content", "")
                    
                    # Создаем умное превью чанка
                    preview = self._create_smart_preview(chunk_content, query)
                    
                    # Проверяем текстовую релевантность для отладки
                    query_words = [word.lower().strip() for word in query.split() if len(word.strip()) > 2]
                    chunk_lower = chunk_content.lower()
                    text_matches = sum(1 for word in query_words if word in chunk_lower) if query_words else 0
                    
                    matches.append(
                        {
                            "document_id": doc_id,
                            "filename": doc.original_filename,
                            "content_type": doc.content_type,
                            "chunk": preview,
                            "full_chunk": chunk_content,
                            "similarity": round(result.score, 3),
                            "chunk_index": result.payload.get("chunk_index", 0),
                            "chunk_length": len(chunk_content),
                            "created_at": doc.created_at.isoformat() if hasattr(doc, 'created_at') else None,
                            "text_matches": text_matches,  # Количество совпадающих слов
                            "query_words": query_words,  # Слова из запроса
                            "has_text_match": text_matches > 0  # Есть ли текстовые совпадения
                        }
                    )

        # 4. Логируем результаты для отладки
        print(f"DEBUG: Found {len(search_results)} raw results from Qdrant")
        print(f"DEBUG: Query words: {[word.lower().strip() for word in query.split() if len(word.strip()) > 2]}")
        print(f"DEBUG: Similarity range: {min(match['similarity'] for match in matches) if matches else 0:.3f} - {max(match['similarity'] for match in matches) if matches else 0:.3f}")
        
        # 5. Сортируем результаты для оптимального отображения
        sorted_results = self._sort_results_optimally(matches, query)
        
        print(f"DEBUG: Returning {len(sorted_results)} filtered results")
        return sorted_results
    
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
    
    def _sort_results_optimally(self, matches: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
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
            doc_id = match["document_id"]
            if doc_id not in documents_groups:
                documents_groups[doc_id] = []
            documents_groups[doc_id].append(match)
        
        # Сортируем чанки внутри каждого документа по chunk_index
        for doc_id in documents_groups:
            documents_groups[doc_id].sort(key=lambda x: x["chunk_index"])
        
        # Собираем результаты: сначала самые релевантные документы
        sorted_matches = []
        
        # Сортируем документы по максимальной similarity среди их чанков
        doc_similarities = {}
        for doc_id, chunks in documents_groups.items():
            max_similarity = max(chunk["similarity"] for chunk in chunks)
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
    
    def _filter_by_text_relevance(self, matches: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
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
            chunk_text = match.get("full_chunk", "").lower()
            
            # Проверяем, содержит ли чанк хотя бы одно слово из запроса
            contains_query_word = any(word in chunk_text for word in query_words)
            
            # Если содержит слова из запроса, добавляем с дополнительным весом
            if contains_query_word:
                # Подсчитываем количество совпадающих слов
                word_matches = sum(1 for word in query_words if word in chunk_text)
                match["text_relevance_score"] = word_matches / len(query_words)
                filtered_matches.append(match)
            else:
                # Если не содержит, но similarity высокая, все равно добавляем
                # но с пониженным приоритетом
                if match["similarity"] > 0.8:  # Высокая similarity
                    match["text_relevance_score"] = 0.1
                    filtered_matches.append(match)
        
        # Сортируем по текстовой релевантности, затем по similarity
        filtered_matches.sort(
            key=lambda x: (x.get("text_relevance_score", 0), x["similarity"]), 
            reverse=True
        )
        
        return filtered_matches