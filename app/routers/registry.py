from fastapi import FastAPI

from app.routers.documents import router as documents_router
from app.routers.chat import router as chat_router
from app.routers.chats import router as chats_router
from app.routers.rules import router as rules_router
from app.routers.rule_categories import router as rule_categories_router

all_routers = [
    documents_router,
    chat_router,
    chats_router,
    rules_router,
    rule_categories_router,
]


def register_routers(app: FastAPI, prefix: str = ""):
    """
    Initialize all routers in the app.
    """
    for router in all_routers:
        app.include_router(router, prefix=prefix)
