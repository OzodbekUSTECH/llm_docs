from dishka import Provider, Scope, provide_all

from app.interactors.documents.create import CreateDocumentInteractor
from app.interactors.documents.delete import DeleteDocumentInteractor
from app.interactors.documents.search import SearchDocumentsInteractor
from app.interactors.chat.generate_new import GenerateOptimizedAnswerInteractor
from app.interactors.chat.openai_generate import OpenAIGenerateInteractor
from app.interactors.documents.get import (
    GetAllDocumentsInteractor, 
    GetDocumentByIdInteractor,
    GetDocumentChunksInteractor
)
from app.interactors.documents.create_new import CreateOptimizedDocumentInteractor


class DocumentsInteractorProvider(Provider):

    scope = Scope.REQUEST

    interactors = provide_all(
        CreateDocumentInteractor,
        SearchDocumentsInteractor,
        DeleteDocumentInteractor,
        GenerateOptimizedAnswerInteractor,
        OpenAIGenerateInteractor,
        GetAllDocumentsInteractor,
        GetDocumentByIdInteractor,
        GetDocumentChunksInteractor,
        CreateOptimizedDocumentInteractor,
    )
