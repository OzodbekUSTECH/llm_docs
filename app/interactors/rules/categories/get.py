from app.repositories.rules import RuleCategoriesRepository
from app.dto.rules import RuleCategoryResponse, GetRuleCategoriesParams
from app.dto.pagination import PaginatedResponse
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages
from uuid import UUID



class GetAllRuleCategoriesInteractor:
    
    def __init__(
        self,
        rule_categories_repository: RuleCategoriesRepository,
    ):
        self.rule_categories_repository = rule_categories_repository
        
    async def execute(self, request: GetRuleCategoriesParams) -> PaginatedResponse[RuleCategoryResponse]:
        rule_categories, total = await self.rule_categories_repository.get_all_and_count(request)
        return PaginatedResponse(
            items=[RuleCategoryResponse.model_validate(rule_category) for rule_category in rule_categories],
            total=total,
            page=request.page,
            size=request.size
        )
        
        
class GetRuleCategoryByIdInteractor:
    
    def __init__(
        self,
        rule_categories_repository: RuleCategoriesRepository,
    ):
        self.rule_categories_repository = rule_categories_repository
        
    async def execute(self, id: UUID) -> RuleCategoryResponse:
        rule_category = await self.rule_categories_repository.get_one(id=id)
        if not rule_category:
            raise AppError(404, ErrorMessages.RULE_CATEGORY_NOT_FOUND)
        return RuleCategoryResponse.model_validate(rule_category)
        