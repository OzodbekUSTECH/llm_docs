from app.di.providers.interactors.documents import DocumentsInteractorProvider
from app.di.providers.interactors.rules import RulesInteractorProvider


all_interactors = [
    DocumentsInteractorProvider(),
    RulesInteractorProvider(),
]
