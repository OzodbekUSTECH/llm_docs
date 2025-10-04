from typing import Optional

from sqlalchemy import or_
from app.dto.common import BaseModelResponse, TimestampResponse
from app.entities.documents import Document
from app.utils.enums import DocumentStatus
from app.dto.pagination import PaginationRequest


class BaseDocumentResponse(BaseModelResponse):
    filename: str
    original_filename: str
    file_path: str
    content_type: str
    status: DocumentStatus
    
    
class DocumentListResponse(BaseDocumentResponse, TimestampResponse):
    pass
    
    
class DocumentResponse(DocumentListResponse):
    tables: Optional[list[dict]] = None
    content: str
    doc_metadata: Optional[dict] = None
    
    
    
class GetDocumentsParams(PaginationRequest):
    status: Optional[DocumentStatus] = None
    filename: Optional[str] = None
    
    
    class Constants:
        filter_map = {
            "status": lambda value: Document.status == value,
            "filename": lambda value: or_(
                Document.filename == value,
                Document.original_filename == value
            ),
        }
        orderable_fields = {
            "created_at": Document.created_at,
            "updated_at": Document.updated_at,
        }
        