
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
from app.interactors.chat.generate import GenerateAnswerInteractor


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
    """Генерирует ответ в контексте конкретного чата"""
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