"""
Инструменты для работы с правилами
"""
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import FieldCondition, MatchAny

from app.di.containers import app_container
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.repositories.rules import RulesRepository
from app.utils.collections import Collections
from app.dto.ai_models import TextContent
from app.dto.qdrant_filters import QdrantFilters

async def search_rules(query: str, limit: int = 10, rule_ids: Optional[List[str]] = None, category_ids: Optional[List[str]] = None) -> List[TextContent]:
   
    try:
        print(f"[SEARCH RULES] Query: '{query}', limit: {limit}, rule_ids: {rule_ids}, category_ids: {category_ids}")
        
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
            filters = None
            must_conditions = []
            
            if rule_ids:
                must_conditions.append(
                    FieldCondition(
                        key="rule_id",
                        match=MatchAny(any=rule_ids)
                    )
                )
                
            if category_ids:
                must_conditions.append(
                    FieldCondition(
                        key="category_id",
                        match=MatchAny(any=category_ids)
                    )
                )
            
            if must_conditions:
                filters = QdrantFilters(must=must_conditions)
            
            search_results = await qdrant_embeddings_repository.search_similar(
                collection_name=Collections.RULES_EMBEDDINGS,
                query_vector=query_vector,
                limit=limit,
                similarity_threshold=0.7,
                filters=filters
            )
            
            print(f"[SEARCH RULES] Found {len(search_results)} raw results")
            
            if not search_results:
                return [TextContent(
                    type="text",
                    text=f"No rules found matching query: '{query}'"
                )]
            
            # Формируем результаты используя только данные из Qdrant payload
            results = []
            for result in search_results:
                payload = result.payload
                
                
                results.append({
                    "rule_id": payload.get("rule_id"),
                    "rule_title": payload.get("rule_title"),
                    "category_id": payload.get("category_id"),
                    "category_title": payload.get("category_title"),
                    "description": payload.get("content"),
                    "relevance_score": round(result.score, 3),
                })
            
            # Сортируем по релевантности
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            
            # Форматируем результат для LLM
            if rule_ids:
                response = f"📋 Found {len(results)} rules matching '{query}' (search limited to {len(rule_ids)} specified rules):\n\n"
            elif category_ids:
                response = f"📋 Found {len(results)} rules matching '{query}' in specified categories:\n\n"
            else:
                response = f"📋 Found {len(results)} rules matching '{query}':\n\n"
            
            for i, rule in enumerate(results, 1):
                response += f"**{i}. {rule['rule_title']}**\n"
                response += f"   🆔 Rule ID: {rule['rule_id']}\n"
                response += f"   📂 Category: {rule['category_title']} (ID: {rule['category_id']})\n"
                response += f"   ⭐ Relevance: {rule['relevance_score']:.3f}\n"
                response += f"   📝 Content: {rule['description']}\n"
                response += "\n"
            
            return [TextContent(type="text", text=response)]
            
    except Exception as e:
        print(f"[SEARCH RULES ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return [TextContent(
            type="text",
            text=f"Error searching rules: {str(e)}"
        )]


async def get_rule_by_id(rule_id: str) -> List[TextContent]:
    """
    Get full rule information by ID including complete description.
    
    Use this tool to:
    - Get complete rule details including full description
    - Retrieve rule information for analysis or reference
    - Get category information associated with the rule
    
    Arguments:
        rule_id (str): The unique ID of the rule to retrieve
        
    Returns:
        List[TextContent]: Formatted text containing:
            - 📋 Rule: rule_title
            - 🆔 ID: rule_id
            - 📂 Category: category_name (category_id)
            - 📝 Full Description: complete rule description
            - 📅 Created: creation_date
            - 📅 Updated: update_date
    
    If rule not found, returns "Rule not found with ID: 'rule_id'".
    """
    try:
        print(f"[GET RULE BY ID] Rule ID: '{rule_id}'")
        
        async with app_container() as container:
            # Получаем зависимости
            rules_repository: RulesRepository = await container.get(RulesRepository)
            
            # Получаем правило из базы данных с категорией
            from sqlalchemy.orm import joinedload
            from app.entities.rules import Rule
            
            rule = await rules_repository.get_one(
                id=rule_id,
                options=[joinedload(Rule.category)]
            )
            
            if not rule:
                return [TextContent(
                    type="text",
                    text=f"Rule not found with ID: '{rule_id}'"
                )]
            
            # Форматируем ответ с полной информацией
            response = f"📋 **Rule: {rule.title}**\n\n"
            response += f"🆔 **ID:** {rule.id}\n"
            response += f"📂 **Category:** {rule.category.title if rule.category else 'N/A'}"
            
            if rule.category:
                response += f" (ID: {rule.category.id})\n"
            else:
                response += "\n"
            
            response += f"📝 **Full Description:**\n{rule.description}\n\n"
            
            if hasattr(rule, 'created_at') and rule.created_at:
                response += f"📅 **Created:** {rule.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if hasattr(rule, 'updated_at') and rule.updated_at:
                response += f"📅 **Updated:** {rule.updated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            return [TextContent(type="text", text=response)]
            
    except Exception as e:
        print(f"[GET RULE BY ID ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return [TextContent(
            type="text",
            text=f"Error getting rule by ID: {str(e)}"
        )]

