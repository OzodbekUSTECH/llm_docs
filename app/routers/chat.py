import asyncio
import json
from datetime import datetime
from uuid import UUID
from fastapi import APIRouter, status, HTTPException, Query
from dishka.integrations.fastapi import DishkaRoute
from typing import List, Dict, Any, Optional

from fastapi.responses import StreamingResponse
from app.dto.chat import GenerateAnswerRequest, GeneratedAnswerResponse
from app.interactors.chat.generate_agent import GenerateAgentAnswerInteractor
from app.services.chat_storage import chat_storage


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    route_class=DishkaRoute,
)


# Legacy RAG stream removed; Agent-only path retained


async def agent_stream_generator(interactor: GenerateAgentAnswerInteractor, message: str, chat_id: str):
    """Generator for streaming response using the agent tool-calling loop."""
    try:
        # Get chat history
        history = chat_storage.get_messages(chat_id)
        # Accumulate for persistence
        debug_trace: list[str] = []
        final_content_parts: list[str] = []

        async for doc_schema in interactor.stream(
            message=message,
            conv_id=chat_id,
            history=history,
        ):
            event_data = {
                "channel": doc_schema.channel or "chat",
                "content": doc_schema.content or str(doc_schema),
            }
            # Accumulate
            if event_data["channel"] == "debug":
                debug_trace.append(str(event_data["content"]))
            elif event_data["channel"] == "chat" and event_data["content"]:
                final_content_parts.append(str(event_data["content"]))
            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.01)

        # Persist messages
        chat_storage.add_message(chat_id=chat_id, role="user", content=message)
        chat_storage.add_message(
            chat_id=chat_id,
            role="assistant",
            content="".join(final_content_parts),
            metadata={"agent_trace": debug_trace},
        )

        yield f"data: {json.dumps({'channel': 'done', 'content': ''})}\n\n"
    except Exception as e:
        error_data = {"channel": "error", "content": f"Error: {str(e)}"}
        yield f"data: {json.dumps(error_data)}\n\n"


@router.post("/generate")
async def generate_answer(
    request: GenerateAnswerRequest,
    chat_id: str = Query(..., description="Chat ID"),
):
    """Generate answer using Agent (OpenAI Agents SDK tool-calling)
    
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
        agent = GenerateAgentAnswerInteractor()
        return StreamingResponse(
            agent_stream_generator(agent, request.message, chat_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream",
            },
        )
    else:
        # Non-streaming response
        history = chat_storage.get_messages(chat_id)
        # Aggregate agent stream into a single message for backward compatibility
        agent = GenerateAgentAnswerInteractor()
        content_acc = []
        async for doc_schema in agent.stream(message=request.message, conv_id=chat_id, history=history):
            if (doc_schema.channel or "chat") == "chat" and doc_schema.content:
                content_acc.append(str(doc_schema.content))
        return GeneratedAnswerResponse(
            message_id=str(UUID(int=0)),
            content="".join(content_acc) if content_acc else "",
            sources=[],
            processing_time=0.0,
            model_used="agent",
            timestamp=datetime.now().isoformat(),
        )


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