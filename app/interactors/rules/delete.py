from app.dto.common import BaseResponse
from app.repositories.uow import UnitOfWork
from app.repositories.rules import RulesRepository
from uuid import UUID
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.utils.collections import Collections


class DeleteRuleInteractor:
    
    def __init__(
        self,
        uow: UnitOfWork,
        rules_repository: RulesRepository,
        qdrant_embeddings_repository: QdrantEmbeddingsRepository,
    ):
        self.uow = uow
        self.rules_repository = rules_repository
        self.qdrant_embeddings_repository = qdrant_embeddings_repository
    async def execute(self, id: UUID) -> BaseResponse:
        rule = await self.rules_repository.get_one(id=id)
        if not rule:
            raise AppError(404, ErrorMessages.RULE_NOT_FOUND)
        await self.rules_repository.delete(id)
        await self.qdrant_embeddings_repository.delete_embeddings(str(rule.id), Collections.RULES_EMBEDDINGS, "rule_id")
        await self.uow.commit()
        return BaseResponse()
        