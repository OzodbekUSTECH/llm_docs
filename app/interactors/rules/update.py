from uuid import UUID
from app.repositories.uow import UnitOfWork
from app.repositories.rules import RulesRepository, RuleCategoriesRepository
from app.dto.rules import PartialUpdateRuleRequest
from app.dto.common import BaseModelResponse
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from sentence_transformers import SentenceTransformer
from app.utils.collections import Collections
from sqlalchemy.orm import joinedload
from app.entities import Rule


class UpdateRulePartiallyInteractor:
    
    def __init__(
        self,
        uow: UnitOfWork,
        rules_repository: RulesRepository,
        rule_categories_repository: RuleCategoriesRepository,
        qdrant_embeddings_repository: QdrantEmbeddingsRepository,
        sentence_transformer: SentenceTransformer,
    ):
        self.uow = uow
        self.rules_repository = rules_repository
        self.rule_categories_repository = rule_categories_repository
        self.qdrant_embeddings_repository = qdrant_embeddings_repository
        self.sentence_transformer = sentence_transformer
    async def execute(self, id: UUID, request: PartialUpdateRuleRequest) -> BaseModelResponse:
        rule = await self.rules_repository.get_one(id=id, options=[joinedload(Rule.category)])
        if not rule:
            raise AppError(404, ErrorMessages.RULE_NOT_FOUND)
        category = rule.category
        if request.category_id:
            category = await self.rule_categories_repository.get_one(id=request.category_id)
            if not category:
                raise AppError(404, ErrorMessages.RULE_CATEGORY_NOT_FOUND)
        await self.rules_repository.update(rule.id, request.model_dump(exclude_unset=True))
        
        await self.qdrant_embeddings_repository.delete_embeddings(str(rule.id), Collections.RULES_EMBEDDINGS, "rule_id")
        
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