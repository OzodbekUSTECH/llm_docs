


from uuid import UUID
from app.entities.base import Base
from sqlalchemy.orm import Mapped, relationship, mapped_column
from sqlalchemy import ForeignKey
from app.entities.mixins.timestamp_mixin import TimestampMixin
from typing import List
from app.entities.mixins.id_mixin import IdMixin
from app.utils.enums import IndexType
    
class Index(Base,IdMixin):
    __tablename__ = "indexes"
    

    source_id: Mapped[str]
    target_id: Mapped[str] # id in lanceDB/qdrant
    relation_type: Mapped[IndexType] = mapped_column(default=IndexType.VECTOR)