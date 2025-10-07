from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.utils.init_qdrant import init_qdrant_collection, warmup_dependencies


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация при запуске
    await warmup_dependencies(app.state.dishka_container)
    await init_qdrant_collection()
    yield
    await app.state.dishka_container.close()
    # Очистка при завершении (если нужно)
