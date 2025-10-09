from sqlalchemy.ext.asyncio import AsyncSession

from app.entities import Rule, RuleCategory
from app.repositories.base import BaseRepository


class RulesRepository(BaseRepository[Rule]):

    def __init__(self, session: AsyncSession):
        super().__init__(session, entity=Rule)
        
        
class RuleCategoriesRepository(BaseRepository[RuleCategory]):

    def __init__(self, session: AsyncSession):
        super().__init__(session, entity=RuleCategory)
        
        