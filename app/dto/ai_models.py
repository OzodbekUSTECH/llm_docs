from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class TextContent(BaseModel):
    """Text content for a message."""

    type: Literal["text"]
    text: str
    meta: dict[str, Any] | None = Field(alias="_meta", default=None)
    model_config = ConfigDict(extra="allow")
