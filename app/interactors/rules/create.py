from app.repositories.uow import UnitOfWork
from app.repositories.rules import RulesRepository, RuleCategoriesRepository
from app.dto.rules import CreateRuleRequest
from app.entities import Rule
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.dto.common import BaseModelResponse
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.utils.collections import Collections




class CreateRuleInteractor:
    """Интерactor для создания правила"""
    def __init__(
        self,
        uow: UnitOfWork,
        rules_repository: RulesRepository,
        rule_categories_repository: RuleCategoriesRepository,
        qdrant_client: AsyncQdrantClient,
        sentence_transformer: SentenceTransformer,
        qdrant_embeddings_repository: QdrantEmbeddingsRepository,
    ):
        self.uow = uow
        self.rules_repository = rules_repository
        self.rule_categories_repository = rule_categories_repository
        self.qdrant_client = qdrant_client
        self.sentence_transformer = sentence_transformer
        self.qdrant_embeddings_repository = qdrant_embeddings_repository

    async def execute(self, request: CreateRuleRequest) -> BaseModelResponse:
        
        
        category = await self.rule_categories_repository.get_one(id=request.category_id)
        if not category:
            raise AppError(404, ErrorMessages.RULE_CATEGORY_NOT_FOUND)
        
        rule = Rule(**request.model_dump())
        await self.rules_repository.create(rule)
        
        text_for_embedding = f"Category: {category.title}\nRule: {rule.title}\nDescription: {rule.description}"
        embedding = self.sentence_transformer.encode(
            [text_for_embedding],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=8,
        )
        await self.qdrant_embeddings_repository.create_embeddings(
            collection_name=Collections.RULES_EMBEDDINGS,
            payload={
                "rule_id": str(rule.id),
                "category_id": str(category.id),
                "category_title": category.title,
                "rule_title": rule.title,
                "content": text_for_embedding,
                "content_length": len(text_for_embedding),
            },
            embedding=embedding[0].tolist(),  # Берем первый (и единственный) эмбеддинг из результата
        )
        
        await self.uow.commit()
        return BaseModelResponse.model_validate(rule)