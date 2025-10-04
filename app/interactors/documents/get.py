from app.repositories.documents import DocumentsRepository
from app.dto.pagination import PaginatedResponse
from app.dto.documents import DocumentListResponse, GetDocumentsParams, DocumentResponse
from uuid import UUID
from app.exceptions.app_error import AppError
from app.exceptions.messages import ErrorMessages


class GetAllDocumentsInteractor:
    def __init__(self, documents_repository: DocumentsRepository):
        self.documents_repository = documents_repository
        
        
    async def execute(self, request: GetDocumentsParams) -> PaginatedResponse[DocumentListResponse]:
        
        docs, total = await self.documents_repository.get_all_and_count(request)
        
        return PaginatedResponse(
            items=[DocumentListResponse.model_validate(doc) for doc in docs],
            total=total,
            page=request.page,
            size=request.size
        )
        
        
class GetDocumentByIdInteractor:
    def __init__(self, documents_repository: DocumentsRepository):
        self.documents_repository = documents_repository
        
    async def execute(self, id: UUID) -> DocumentResponse:
        doc = await self.documents_repository.get_one(id=id)
        if not doc:
            raise AppError(status_code=404, message=ErrorMessages.DOCUMENT_NOT_FOUND)
        return DocumentResponse.model_validate(doc)
        
        