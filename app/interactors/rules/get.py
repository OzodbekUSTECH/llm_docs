from sqlalchemy.orm import joinedload
from app.repositories.rules import RulesRepository
from app.entities import Rule
from app.dto.rules import GetRulesParams
from app.dto.pagination import PaginatedResponse
from app.dto.rules import RuleListResponse, RuleResponse
from uuid import UUID
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages



class GetAllRulesInteractor:
    
    def __init__(
        self,
        rules_repository: RulesRepository,
    ):
        self.rules_repository = rules_repository
        
    async def execute(self, request: GetRulesParams) -> PaginatedResponse[RuleListResponse]:
        rules, total = await self.rules_repository.get_all_and_count(
            request,
            options=[joinedload(Rule.category)],
        )
        return PaginatedResponse(
            items=[RuleListResponse.model_validate(rule) for rule in rules],
            total=total,
            page=request.page,
            size=request.size
        )
        
        
class GetRuleByIdInteractor:
    
    def __init__(
        self,
        rules_repository: RulesRepository,
    ):
        self.rules_repository = rules_repository
        
    async def execute(self, id: UUID) -> RuleResponse:
        rule = await self.rules_repository.get_one(id=id, options=[joinedload(Rule.category)])
        if not rule:
            raise AppError(404, ErrorMessages.RULE_NOT_FOUND)
        return RuleResponse.model_validate(rule)
        
        