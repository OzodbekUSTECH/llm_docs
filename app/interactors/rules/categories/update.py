from uuid import UUID
from app.repositories.uow import UnitOfWork
from app.repositories.rules import RuleCategoriesRepository
from app.dto.rules import PartialUpdateRuleCategoryRequest
from app.dto.common import BaseModelResponse
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages


class UpdateRuleCategoryPartiallyInteractor:
    
    def __init__(
        self,
        uow: UnitOfWork,
        rule_categories_repository: RuleCategoriesRepository,
    ):
        self.uow = uow
        self.rule_categories_repository = rule_categories_repository
        
    async def execute(self, id: UUID, request: PartialUpdateRuleCategoryRequest) -> BaseModelResponse:
        rule_category = await self.rule_categories_repository.get_one(id=id)
        if not rule_category:
            raise AppError(404, ErrorMessages.RULE_CATEGORY_NOT_FOUND)
        await self.rule_categories_repository.update(rule_category.id, request.model_dump(exclude_unset=True))
        await self.uow.commit()
        return BaseModelResponse.model_validate(rule_category)