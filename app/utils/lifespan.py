from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.utils.init_qdrant import init_qdrant_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация при запуске
    await init_qdrant_collection()
    yield
    # Очистка при завершении (если нужно)
