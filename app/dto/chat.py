


from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class GenerateAnswerRequest(BaseModel):
    message: str
    stream: bool = False


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    output: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class Source(BaseModel):
    filename: str
    content: str
    similarity: float
    chunk_index: int


class GeneratedAnswerResponse(BaseModel):
    message_id: str
    role: str = "assistant"
    content: str
    sources: List[Source] = []
    tool_calls: List[ToolCall] = []
    reasoning: Optional[str] = None
    processing_time: float
    model_used: str
    timestamp: str