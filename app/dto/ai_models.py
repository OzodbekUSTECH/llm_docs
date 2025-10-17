from typing import Any, Literal, List
from pydantic import BaseModel, Field, ConfigDict


class TextContent(BaseModel):
    """Text content for a message."""

    type: Literal["text"]
    text: str
    meta: dict[str, Any] | None = Field(alias="_meta", default=None)
    model_config = ConfigDict(extra="allow")


class ContractSection(BaseModel):
    """Single contract section with title and content."""
    title: str = Field(description="Section title")
    content: str = Field(description="Section content text, may include tables in markdown")


class ContractSectionsOutput(BaseModel):
    """Structured output from LLM for contract sections."""
    sections: List[ContractSection] = Field(default_factory=list)


class InvoiceField(BaseModel):
    """Single extracted invoice field as title/value pair."""
    title: str = Field(description="Human-readable field title, e.g., 'INVOICE NO', 'DATE', 'SELLER' ")
    value: str = Field(description="Verbatim or minimally normalized value text; may include markdown for tables")


class InvoiceFieldsOutput(BaseModel):
    """Structured output from LLM for invoice key-value fields."""
    fields: List[InvoiceField] = Field(default_factory=list)