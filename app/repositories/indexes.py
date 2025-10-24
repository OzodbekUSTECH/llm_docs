from sqlalchemy.ext.asyncio import AsyncSession

from app.entities import Index
from app.repositories.base import BaseRepository


class IndexesRepository(BaseRepository[Index]):

    def __init__(self, session: AsyncSession):
        super().__init__(session, entity=Index)