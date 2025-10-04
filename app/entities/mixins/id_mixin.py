from uuid import UUID, uuid4
from sqlalchemy import text
from sqlalchemy.orm import Mapped, mapped_column


class IdMixin:

    id: Mapped[UUID] = mapped_column(
        primary_key=True, default=uuid4, server_default=text("gen_random_uuid()")
    )
