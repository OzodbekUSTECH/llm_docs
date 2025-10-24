from uuid import UUID, uuid4
from sqlalchemy import text
from sqlalchemy.orm import Mapped, mapped_column


class IdMixin:

    id: Mapped[str] = mapped_column(
        primary_key=True, default=lambda: str(uuid4())
    )
