


from uuid import UUID
from app.entities.base import Base
from sqlalchemy.orm import Mapped, relationship, mapped_column
from sqlalchemy import ForeignKey
from app.entities.mixins.timestamp_mixin import TimestampMixin
from typing import List
from app.entities.mixins.id_mixin import IdMixin
    
class Rule(Base,IdMixin, TimestampMixin):
    __tablename__ = "rules"
    


    title: Mapped[str]
    description: Mapped[str]
    
    category_id: Mapped[UUID] = mapped_column(ForeignKey("rule_categories.id"))
    category: Mapped["RuleCategory"] = relationship(back_populates="rules")
    
    
class RuleCategory(Base,IdMixin, TimestampMixin):
    __tablename__ = "rule_categories"
    
    title: Mapped[str]
    
    rules: Mapped[List["Rule"]] = relationship(back_populates="category")
    
    