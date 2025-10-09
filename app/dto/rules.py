from pydantic import BaseModel
from uuid import UUID
from typing import Optional
from app.dto.common import BaseModelResponse, TimestampResponse
from app.dto.pagination import PaginationRequest
from app.entities import Rule, RuleCategory


# Rule Category
class CreateRuleCategoryRequest(BaseModel):
    title: str
    
    
class PartialUpdateRuleCategoryRequest(BaseModel):
    title: Optional[str] = None
    
class BaseRuleCategoryResponse(BaseModelResponse):
    title: str
    
class RuleCategoryListResponse(BaseRuleCategoryResponse, TimestampResponse):
    pass

class RuleCategoryResponse(RuleCategoryListResponse):
    pass


class GetRuleCategoriesParams(PaginationRequest):
    title: Optional[str] = None
    
    class Constants:
        filter_map = {
            "title": lambda value: RuleCategory.title.ilike(f"%{value}%"),
        }
        searchable_fields = [RuleCategory.title]
        orderable_fields = {
            "created_at": RuleCategory.created_at,
            "updated_at": RuleCategory.updated_at,
        }

# Rule
class CreateRuleRequest(BaseModel):
    title: str
    description: str
    category_id: UUID
    
    
class PartialUpdateRuleRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category_id: Optional[UUID] = None
    
class BaseRuleResponse(BaseModelResponse):
    title: str
    description: str
    category: BaseRuleCategoryResponse
    
class RuleListResponse(BaseRuleResponse, TimestampResponse):
    pass

class RuleResponse(RuleListResponse):
    pass



class GetRulesParams(PaginationRequest):
    category_id: Optional[UUID] = None
    title: Optional[str] = None
    description: Optional[str] = None
    
    class Constants:
        filter_map = {
            "category_id": lambda value: Rule.category_id == value,
            "title": lambda value: Rule.title.ilike(f"%{value}%"),
            "description": lambda value: Rule.description.ilike(f"%{value}%"),
        }
        searchable_fields = [Rule.title, Rule.description]
        orderable_fields = {
            "created_at": Rule.created_at,
            "updated_at": Rule.updated_at,
        }
    
    
    