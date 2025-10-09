from app.repositories.uow import UnitOfWork
from app.repositories.rules import RuleCategoriesRepository
from uuid import UUID
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from app.dto.common import BaseResponse



class DeleteRuleCategoryInteractor:
    
    def __init__(
        self,
        uow: UnitOfWork,
        rule_categories_repository: RuleCategoriesRepository,
    ):
        self.uow = uow
        self.rule_categories_repository = rule_categories_repository
        
    async def execute(self, id: UUID) -> BaseResponse:
        rule_category = await self.rule_categories_repository.get_one(id=id)
        if not rule_category:
            raise AppError(404, ErrorMessages.RULE_CATEGORY_NOT_FOUND)
        await self.rule_categories_repository.delete(id)
        await self.uow.commit()
        return BaseResponse()