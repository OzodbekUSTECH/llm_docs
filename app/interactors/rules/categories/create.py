from app.repositories.uow import UnitOfWork
from app.repositories.rules import RuleCategoriesRepository
from app.dto.rules import CreateRuleCategoryRequest
from app.dto.common import BaseModelResponse
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.entities import RuleCategory


class CreateRuleCategoryInteractor:
    
    def __init__(
        self,
        uow: UnitOfWork,
        rule_categories_repository: RuleCategoriesRepository,
    ):
        self.uow = uow
        self.rule_categories_repository = rule_categories_repository
        
    async def execute(self, request: CreateRuleCategoryRequest) -> BaseModelResponse:
        rule_category = RuleCategory(**request.model_dump())
        await self.rule_categories_repository.create(rule_category)
        await self.uow.commit()
        return BaseModelResponse.model_validate(rule_category)
        