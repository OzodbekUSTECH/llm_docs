"""
Router для управления чатами
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from app.services.chat_storage import chat_storage


router = APIRouter(prefix="/chats", tags=["chats"])


class CreateChatRequest(BaseModel):
    title: Optional[str] = "Новый чат"


class UpdateChatTitleRequest(BaseModel):
    title: str


class ChatResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ChatWithMessagesResponse(ChatResponse):
    messages: List[dict]


@router.post("/", response_model=ChatResponse)
async def create_chat(request: CreateChatRequest):
    """Создает новый чат"""
    chat = chat_storage.create_chat(request.title)
    return chat.to_dict()


@router.get("/", response_model=List[ChatResponse])
async def get_all_chats():
    """Получает список всех чатов"""
    chats = chat_storage.get_all_chats()
    return [chat.to_dict() for chat in chats]


@router.get("/{chat_id}", response_model=ChatWithMessagesResponse)
async def get_chat(chat_id: str):
    """Получает чат с сообщениями"""
    chat = chat_storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Чат не найден")
    return chat.to_dict(include_messages=True)


@router.patch("/{chat_id}/title")
async def update_chat_title(chat_id: str, request: UpdateChatTitleRequest):
    """Обновляет название чата"""
    success = chat_storage.update_chat_title(chat_id, request.title)
    if not success:
        raise HTTPException(status_code=404, detail="Чат не найден")
    return {"success": True, "title": request.title}


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str):
    """Удаляет чат"""
    success = chat_storage.delete_chat(chat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Чат не найден")
    return {"success": True}


@router.get("/{chat_id}/messages")
async def get_chat_messages(chat_id: str):
    """Получает сообщения чата"""
    messages = chat_storage.get_messages(chat_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Чат не найден")
    return {"messages": messages}

