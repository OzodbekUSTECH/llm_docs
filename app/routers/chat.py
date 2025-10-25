import asyncio
import json
from datetime import datetime
from uuid import UUID
from fastapi import APIRouter, status, HTTPException, Query
from dishka.integrations.fastapi import FromDishka, DishkaRoute
from typing import List, Dict, Any, Optional

from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.dto.chat import GenerateAnswerRequest, GeneratedAnswerResponse
from app.interactors.chat.generate import GenerateAnswerInteractor
from app.services.chat_storage import chat_storage


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    route_class=DishkaRoute,
)


async def stream_generator(interactor: GenerateAnswerInteractor, message: str, chat_id: str, document_ids: Optional[List[str]] = None):
    """Generator for streaming response in SSE format"""
    try:
        # Debug: log document_ids
        if document_ids:
            print(f"🔍 Backend: Filtering by document IDs: {document_ids}")
        else:
            print(f"🔍 Backend: Searching in all documents")
            
        # Get chat history
        history = chat_storage.get_messages(chat_id)
        
        async for doc_schema in interactor.stream(
            message=message,
            conv_id=chat_id,
            history=history,
            top_k=5,
            document_ids=document_ids
        ):
            # Format as Server-Sent Events
            event_data = {
                "channel": doc_schema.channel or "chat",
                "content": doc_schema.content or str(doc_schema),
            }
            
            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
        
        # Send completion event
        yield f"data: {json.dumps({'channel': 'done', 'content': ''})}\n\n"
        
    except Exception as e:
        error_data = {
            "channel": "error",
            "content": f"Error: {str(e)}"
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@router.post("/generate")
async def generate_answer(
    request: GenerateAnswerRequest,
    generate_answer_interactor: FromDishka[GenerateAnswerInteractor],
    chat_id: str = Query(..., description="Chat ID"),
):
    """Generate answer using RAG with Qdrant vector search and OpenAI
    
    Args:
        request: Request with message and stream option
        chat_id: Unique chat identifier
        
    Returns:
        StreamingResponse if stream=True, else GeneratedAnswerResponse
    """
    # Create chat if it doesn't exist
    chat = chat_storage.get_chat(chat_id)
    if not chat:
        chat_storage.create_chat(title="New Chat")
        chat_storage._chats[chat_id] = chat_storage._chats.pop(list(chat_storage._chats.keys())[-1])
        chat = chat_storage.get_chat(chat_id)
        if chat:
            chat.id = chat_id
    
    if request.stream:
        return StreamingResponse(
            stream_generator(generate_answer_interactor, request.message, chat_id, request.document_ids),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream"
            }
        )
    else:
        # Non-streaming response
        history = chat_storage.get_messages(chat_id)
        response = await generate_answer_interactor.execute(
            request=request,
            conv_id=chat_id,
            history=history
        )
        return response


@router.get("/chats")
async def get_all_chats():
    """Get all chats with basic info"""
    chats = chat_storage.get_all_chats()
    return {
        "chats": [chat.to_dict(include_messages=False) for chat in chats],
        "total": len(chats)
    }


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get chat details with full message history"""
    chat = chat_storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    return chat.to_dict(include_messages=True)


@router.post("/chats")
async def create_chat(title: str = "New Chat"):
    """Create a new chat"""
    chat = chat_storage.create_chat(title=title)
    return {
        "status": "success",
        "chat": chat.to_dict(include_messages=False)
    }


@router.put("/chats/{chat_id}/title")
async def update_chat_title(chat_id: str, title: str):
    """Update chat title"""
    success = chat_storage.update_chat_title(chat_id, title)
    if not success:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    return {
        "status": "success",
        "message": f"Chat title updated to: {title}"
    }


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat"""
    success = chat_storage.delete_chat(chat_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    return {
        "status": "success",
        "message": f"Chat {chat_id} deleted"
    }


@router.post("/chats/{chat_id}/clear")
async def clear_chat_history(chat_id: str):
    """Clear chat history (removes all messages)
    
    Use this when:
    - Model starts hallucinating
    - Context is polluted with irrelevant information
    - You want to start a new conversation
    """
    chat = chat_storage.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    
    chat.messages = []
    chat.updated_at = datetime.now()
    
    return {
        "status": "success",
        "message": f"Chat {chat_id} history cleared. Start a new conversation."
    }