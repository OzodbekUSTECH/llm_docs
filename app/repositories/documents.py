from sqlalchemy.ext.asyncio import AsyncSession

from app.entities import Document
from app.repositories.base import BaseRepository


class DocumentsRepository(BaseRepository[Document]):

    def __init__(self, session: AsyncSession):
        super().__init__(session, entity=Document)