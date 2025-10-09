
from uuid import UUID
from fastapi import APIRouter, status, Query
from typing import Annotated
from dishka.integrations.fastapi import FromDishka, DishkaRoute


from app.interactors.rules.create import CreateRuleInteractor
from app.interactors.rules.delete import DeleteRuleInteractor
from app.interactors.rules.update import UpdateRulePartiallyInteractor
from app.interactors.rules.get import GetAllRulesInteractor, GetRuleByIdInteractor
from app.dto.rules import CreateRuleRequest, GetRulesParams, RuleListResponse, RuleResponse, PartialUpdateRuleRequest
from app.dto.pagination import PaginatedResponse
from app.dto.common import BaseModelResponse, BaseResponse


router = APIRouter(
    prefix="/rules",
    tags=["Rules"],
    route_class=DishkaRoute,
)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_rule(
    create_rule_interactor: FromDishka[CreateRuleInteractor],
    request: CreateRuleRequest,
) -> BaseModelResponse:
    return await create_rule_interactor.execute(request)


@router.get("/")
async def get_rules(
    get_rules_interactor: FromDishka[GetAllRulesInteractor],
    request: Annotated[GetRulesParams, Query()],
) -> PaginatedResponse[RuleListResponse]:
    return await get_rules_interactor.execute(request)


@router.get("/{id}")
async def get_rule(
    get_rule_interactor: FromDishka[GetRuleByIdInteractor],
    id: UUID,
) -> RuleResponse:
    return await get_rule_interactor.execute(id)


@router.patch("/{id}")
async def update_rule(
    update_rule_interactor: FromDishka[UpdateRulePartiallyInteractor],
    id: UUID,
    request: PartialUpdateRuleRequest,
) -> BaseModelResponse:
    return await update_rule_interactor.execute(id, request)


@router.delete("/{id}")
async def delete_rule(
    delete_rule_interactor: FromDishka[DeleteRuleInteractor],
    id: UUID,
) -> BaseResponse:
    return await delete_rule_interactor.execute(id)