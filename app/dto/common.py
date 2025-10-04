from datetime import datetime

from pydantic import BaseModel, ConfigDict
from uuid import UUID


class BaseModelResponse(BaseModel):
    """
    Base response model for all entities.
    """

    id: UUID

    model_config = ConfigDict(from_attributes=True)


class TimestampResponse(BaseModel):
    """
    Mixin for entities that have created_at and updated_at timestamps.
    """

    created_at: datetime
    updated_at: datetime


class UserTrackableResponse(BaseModel):
    """
    Mixin for entities that track user actions.
    """

    created_by: UUID | None = None
    updated_by: UUID | None = None


class BaseResponse(BaseModel):
    """
    Base response for some endpoints.
    """

    success: bool = True
