


from datetime import date, datetime
from uuid import UUID
from app.entities.base import Base
from sqlalchemy.orm import Mapped, relationship, mapped_column, column_property
from sqlalchemy import ForeignKey, BIGINT, Identity, Float, JSON
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from app.entities.mixins.timestamp_mixin import TimestampMixin
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from app.entities.mixins.id_mixin import IdMixin
from app.utils.enums import DocumentStatus
    
class Document(Base,IdMixin, TimestampMixin):
    __tablename__ = "documents"
    


    filename: Mapped[str]
    original_filename: Mapped[str]
    file_path: Mapped[str]
    content_type: Mapped[str]
    file_hash: Mapped[str] = mapped_column(index=True, unique=True)
    status: Mapped[DocumentStatus] = mapped_column(default=DocumentStatus.PENDING)
    content: Mapped[str]
    tables: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSONB, nullable=True)
    doc_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    
    
    
    