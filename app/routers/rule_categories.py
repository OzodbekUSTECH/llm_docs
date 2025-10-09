
from uuid import UUID
from fastapi import APIRouter, status, Query
from typing import Annotated
from dishka.integrations.fastapi import FromDishka, DishkaRoute


from app.interactors.rules.categories.create import CreateRuleCategoryInteractor
from app.interactors.rules.categories.delete import DeleteRuleCategoryInteractor
from app.interactors.rules.categories.update import UpdateRuleCategoryPartiallyInteractor
from app.interactors.rules.categories.get import GetAllRuleCategoriesInteractor, GetRuleCategoryByIdInteractor
from app.dto.rules import CreateRuleCategoryRequest, GetRuleCategoriesParams, RuleCategoryListResponse, RuleCategoryResponse, PartialUpdateRuleCategoryRequest
from app.dto.pagination import PaginatedResponse
from app.dto.common import BaseModelResponse, BaseResponse


router = APIRouter(
    prefix="/rule-categories",
    tags=["Rule Categories"],
    route_class=DishkaRoute,
)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_rule_category(
    create_rule_category_interactor: FromDishka[CreateRuleCategoryInteractor],
    request: CreateRuleCategoryRequest,
) -> BaseModelResponse:
    return await create_rule_category_interactor.execute(request)


@router.get("/")
async def get_rule_categories(
    request: Annotated[GetRuleCategoriesParams, Query()],
    get_rule_categories_interactor: FromDishka[GetAllRuleCategoriesInteractor],
) -> PaginatedResponse[RuleCategoryListResponse]:
    return await get_rule_categories_interactor.execute(request)


@router.get("/{id}")
async def get_rule_category(
    get_rule_category_interactor: FromDishka[GetRuleCategoryByIdInteractor],
    id: UUID,
) -> RuleCategoryResponse:
    return await get_rule_category_interactor.execute(id)


@router.patch("/{id}")
async def update_rule_category(
    update_rule_category_interactor: FromDishka[UpdateRuleCategoryPartiallyInteractor],
    id: UUID,
    request: PartialUpdateRuleCategoryRequest,
) -> BaseModelResponse:
    return await update_rule_category_interactor.execute(id, request)


@router.delete("/{id}")
async def delete_rule_category(
    delete_rule_category_interactor: FromDishka[DeleteRuleCategoryInteractor],
    id: UUID,
) -> BaseResponse:
    return await delete_rule_category_interactor.execute(id)