from dishka import Provider, Scope, provide_all
from app.repositories.documents import DocumentsRepository
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.repositories.uow import UnitOfWork

class RepositoriesProvider(Provider):

    scope = Scope.REQUEST

    repositories = provide_all(
        UnitOfWork,
        
        DocumentsRepository,
        QdrantEmbeddingsRepository,
    )
