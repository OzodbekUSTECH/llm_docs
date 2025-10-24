from datetime import datetime
from typing import Optional
from pydantic import BaseModel

from app.dto.pagination import PaginationRequest
from app.dto.common import BaseModelResponse, TimestampResponse
from app.entities.sources import Source



class SourcesListResponse(BaseModelResponse,TimestampResponse):
    name: str
    size: int
    note: dict
    
    
class SourceResponse(SourcesListResponse):
    indexes: dict
    
    
class GetSourcesParams(PaginationRequest):
    name: Optional[str] = None
    order_by: Optional[str] = "created_at"
    
    class Constants:
        filter_map = {
            "name": lambda value: Source.name.ilike(f"%{value}%"),
        }
        orderable_fields = {
            "created_at": Source.created_at,
            "updated_at": Source.updated_at,
        }