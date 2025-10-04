from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.logger_conf import configure_logging
from app.di.containers import app_container
from app.exceptions.registery import register_exceptions
from app.utils.dependencies import get_current_user_for_docs
from app.utils.lifespan import lifespan
from app.middlewares.registery import register_middlewares
from app.middlewares.auth import AuthMiddleware
from app.routers import auth

from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI, Depends

from app.core.config import settings
from app.routers.registry import register_routers



configure_logging(level=settings.LOG_LEVEL)


def create_app():
    app = FastAPI(
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    setup_dishka(app_container, app)
    
    # Регистрируем auth router ДО middleware
    app.include_router(auth.router)
    
    # Добавляем Auth Middleware (использует sessions из auth.py)
    app.add_middleware(AuthMiddleware, sessions_store=auth.sessions)
    
    register_middlewares(app)
    register_exceptions(app)
    register_routers(app)
    
    # Добавляем маршруты для страниц
    @app.get("/login", include_in_schema=False)
    async def login_page():
        return FileResponse("templates/login.html")
    
    @app.get("/", include_in_schema=False)
    async def read_root():
        return FileResponse("templates/chat.html")
    
    @app.get("/documents", include_in_schema=False)
    async def documents_page():
        return FileResponse("templates/documents.html")
    
    # Подключаем статические файлы (если они есть)
    app.mount("/storage", StaticFiles(directory="storage"), name="storage")
   
    return app


app = create_app()


@app.get(
    "/api/docs",
    include_in_schema=False,
    dependencies=[Depends(get_current_user_for_docs)],
)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="LLM STARTAP API",
        swagger_ui_parameters={"docExpansion": "none"},
    )


@app.get(
    "/api/openapi.json",
    include_in_schema=False,
    dependencies=[Depends(get_current_user_for_docs)],
)
async def get_open_api_endpoint():
    openapi_schema = get_openapi(
        title="LLM STARTAP API", version="1.0.0", routes=app.routes
    )
    openapi_schema["servers"] = [
        {"url": "/", "description": "Base Path for API"},
    ]
    return openapi_schema
