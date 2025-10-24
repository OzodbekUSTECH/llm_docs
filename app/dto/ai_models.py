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


# ===== Parser (LLM OCR) unified output models =====

class DocTypeClassification(BaseModel):
    """LLM classification result for document type."""
    document_type: Literal[
        "INVOICE",
        "CONTRACT",
        "COO",
        "COA",
        "COW",
        "COQ",
        "BL",
        "FINANCIAL",
        "LC",
        "OTHER",
    ]
    confidence: float = Field(description="0..1 confidence score")


class ParsedChunk(BaseModel):
    """Single parsed chunk with clause and content."""
    title: str = Field(description="Human-friendly section or clause title")
    clause: str | None = Field(default=None, description="Canonical clause name if applicable")
    content: str = Field(description="Full text content including any tables in markdown form")


class ParsedDocumentOutput(BaseModel):
    """Unified parsed document output for embeddings and storage."""
    document_type: DocTypeClassification
    chunks: List[ParsedChunk] = Field(default_factory=list)