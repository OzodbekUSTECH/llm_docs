from sqlalchemy.ext.asyncio import AsyncSession

from app.entities import Source
from app.repositories.base import BaseRepository


class SourcesRepository(BaseRepository[Source]):

    def __init__(self, session: AsyncSession):
        super().__init__(session, entity=Source)