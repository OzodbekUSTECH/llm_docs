
import asyncio
import json
from uuid import UUID
from fastapi import APIRouter, status, UploadFile, File, Query
from dishka.integrations.fastapi import FromDishka, DishkaRoute
from typing import List, Dict, Any

from fastapi.responses import StreamingResponse
from ollama import AsyncClient

from app.dto.chat import GenerateAnswerRequest
from app.interactors.documents.create import CreateDocumentInteractor
from app.interactors.documents.delete import DeleteDocumentInteractor
from app.interactors.documents.search import SearchDocumentsInteractor
from app.interactors.chat.generate import GenerateAnswerInteractor, clear_chat_history
from app.interactors.chat.openai_generate import OpenAIGenerateInteractor


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    route_class=DishkaRoute,
)


@router.post("/generate")
async def generate_answer(
    request: GenerateAnswerRequest,
    generate_answer_interactor: FromDishka[GenerateAnswerInteractor],
    chat_id: str = Query(..., description="ID чата"),
    stream: bool = Query(False, description="Использовать streaming")
):
    """Генерирует ответ с использованием Ollama в контексте конкретного чата"""
    if stream:
        return StreamingResponse(
            generate_answer_interactor.execute_stream(request, chat_id=chat_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    else:
        return await generate_answer_interactor.execute(request, chat_id=chat_id)


@router.post("/generate-openai")
async def generate_answer_openai(
    request: GenerateAnswerRequest,
    openai_generate_interactor: FromDishka[OpenAIGenerateInteractor],
    chat_id: str = Query(..., description="ID чата"),
    model: str = Query("gpt-4o-mini", description="Модель OpenAI (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)"),
    stream: bool = Query(False, description="Использовать streaming")
):
    """
    Генерирует ответ с использованием OpenAI API в контексте конкретного чата
    
    Доступные модели:
    - gpt-4o-mini (рекомендуется): быстрая, дешевая, хорошее качество
    - gpt-4o: лучшее качество, медленнее, дороже
    - gpt-3.5-turbo: самая быстрая и дешевая
    """
    if stream:
        return StreamingResponse(
            openai_generate_interactor.execute_stream(request, chat_id=chat_id, model=model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    else:
        return await openai_generate_interactor.execute(request, chat_id=chat_id, model=model)


@router.post("/clear-history/{chat_id}")
async def clear_history(chat_id: str):
    """Очищает историю чата (удаляет все сообщения кроме system prompt)
    
    Используйте это когда:
    - Модель начинает галлюцинировать
    - Контекст загрязнен нерелевантной информацией
    - Хотите начать новый разговор в том же чате
    """
    clear_chat_history(chat_id)
    return {
        "status": "success",
        "message": f"История чата {chat_id} очищена. Начните новый разговор."
    }