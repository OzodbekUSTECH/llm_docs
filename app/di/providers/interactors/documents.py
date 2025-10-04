from dishka import Provider, Scope, provide_all

from app.interactors.documents.create import CreateDocumentInteractor
from app.interactors.documents.delete import DeleteDocumentInteractor
from app.interactors.documents.search import SearchDocumentsInteractor
from app.interactors.chat.generate import GenerateAnswerInteractor
from app.interactors.documents.get import GetAllDocumentsInteractor, GetDocumentByIdInteractor


class DocumentsInteractorProvider(Provider):

    scope = Scope.REQUEST

    interactors = provide_all(
        CreateDocumentInteractor,
        SearchDocumentsInteractor,
        DeleteDocumentInteractor,
        GenerateAnswerInteractor,
        GetAllDocumentsInteractor,
        GetDocumentByIdInteractor
    )
