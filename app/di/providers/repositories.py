from dishka import Provider, Scope, provide_all
from app.repositories.documents import DocumentsRepository
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.repositories.uow import UnitOfWork
from app.repositories.rules import RulesRepository, RuleCategoriesRepository
from app.repositories.sources import SourcesRepository
from app.repositories.indexes import IndexesRepository

class RepositoriesProvider(Provider):

    scope = Scope.REQUEST

    repositories = provide_all(
        UnitOfWork,
        RulesRepository,
        RuleCategoriesRepository,
        DocumentsRepository,
        QdrantEmbeddingsRepository,
        SourcesRepository,
        IndexesRepository,
    )
