


from uuid import UUID

from sqlalchemy.dialects.postgresql import JSONB
from app.entities.base import Base
from sqlalchemy.orm import Mapped, relationship, mapped_column
from sqlalchemy import ForeignKey
from app.entities.mixins.timestamp_mixin import TimestampMixin
from typing import List
from app.entities.mixins.id_mixin import IdMixin
    
class Source(Base,IdMixin, TimestampMixin):
    __tablename__ = "sources"
    


    name: Mapped[str]
    path: Mapped[str]
    size: Mapped[int] = mapped_column(default=0)
    note: Mapped[dict] = mapped_column(JSONB, default=dict,server_default="{}")
    