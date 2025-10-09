"""
DTO для фильтров Qdrant
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from qdrant_client.http.models import FieldCondition


class QdrantFilters(BaseModel):
    """
    DTO для передачи фильтров в Qdrant search
    
    Соответствует структуре Qdrant Filter:
    https://qdrant.tech/documentation/concepts/filtering/
    
    Примеры использования:
    
    1. Простой MUST фильтр (логическое И):
        filters = QdrantFilters(
            must=[
                FieldCondition(key="status", match=MatchValue(value="active")),
                FieldCondition(key="type", match=MatchValue(value="pdf"))
            ]
        )
    
    2. SHOULD фильтр (логическое ИЛИ):
        filters = QdrantFilters(
            should=[
                FieldCondition(key="priority", match=MatchValue(value="high")),
                FieldCondition(key="urgent", match=MatchValue(value=True))
            ],
            min_should=1  # Хотя бы одно условие должно выполниться
        )
    
    3. MUST_NOT фильтр (исключение):
        filters = QdrantFilters(
            must_not=[
                FieldCondition(key="deleted", match=MatchValue(value=True))
            ]
        )
    
    4. Комбинированный фильтр:
        filters = QdrantFilters(
            must=[
                FieldCondition(key="status", match=MatchValue(value="active"))
            ],
            should=[
                FieldCondition(key="category", match=MatchAny(any=["cat1", "cat2"]))
            ],
            must_not=[
                FieldCondition(key="archived", match=MatchValue(value=True))
            ],
            min_should=1
        )
    """
    
    must: Optional[List[FieldCondition]] = Field(
        default=None,
        description="Все условия должны быть выполнены (логическое И)"
    )
    
    should: Optional[List[FieldCondition]] = Field(
        default=None,
        description="Хотя бы одно условие должно быть выполнено (логическое ИЛИ)"
    )
    
    must_not: Optional[List[FieldCondition]] = Field(
        default=None,
        description="Условия НЕ должны быть выполнены (логическое НЕ)"
    )
    
    min_should: Optional[int] = Field(
        default=None,
        ge=1,
        description="Минимальное количество условий SHOULD, которые должны выполниться. "
                    "Если не указано, достаточно одного."
    )
    
    class Config:
        arbitrary_types_allowed = True  # Для поддержки FieldCondition
        
    def has_filters(self) -> bool:
        """Проверяет, есть ли какие-либо фильтры"""
        return bool(self.must or self.should or self.must_not)
    
    def to_qdrant_filter(self):
        """
        Конвертирует в объект Filter для Qdrant
        
        Returns:
            Filter или None если фильтры не заданы
        """
        from qdrant_client.http.models import Filter
        
        if not self.has_filters():
            return None
        
        filter_params = {}
        
        if self.must:
            filter_params["must"] = self.must
        
        if self.should:
            filter_params["should"] = self.should
        
        if self.must_not:
            filter_params["must_not"] = self.must_not
        
        if self.min_should is not None:
            filter_params["min_should"] = self.min_should
        
        return Filter(**filter_params)


# Вспомогательные функции для создания распространенных фильтров

def create_must_filter(*conditions: FieldCondition) -> QdrantFilters:
    """
    Создает фильтр с условиями MUST (все должны выполниться)
    
    Пример:
        filters = create_must_filter(
            FieldCondition(key="status", match=MatchValue(value="active")),
            FieldCondition(key="type", match=MatchValue(value="pdf"))
        )
    """
    return QdrantFilters(must=list(conditions))


def create_should_filter(*conditions: FieldCondition, min_should: int = 1) -> QdrantFilters:
    """
    Создает фильтр с условиями SHOULD (хотя бы одно должно выполниться)
    
    Пример:
        filters = create_should_filter(
            FieldCondition(key="priority", match=MatchValue(value="high")),
            FieldCondition(key="urgent", match=MatchValue(value=True)),
            min_should=1
        )
    """
    return QdrantFilters(should=list(conditions), min_should=min_should)


def create_must_not_filter(*conditions: FieldCondition) -> QdrantFilters:
    """
    Создает фильтр с условиями MUST_NOT (не должны выполниться)
    
    Пример:
        filters = create_must_not_filter(
            FieldCondition(key="deleted", match=MatchValue(value=True))
        )
    """
    return QdrantFilters(must_not=list(conditions))


def create_combined_filter(
    must: Optional[List[FieldCondition]] = None,
    should: Optional[List[FieldCondition]] = None,
    must_not: Optional[List[FieldCondition]] = None,
    min_should: Optional[int] = None
) -> QdrantFilters:
    """
    Создает комбинированный фильтр со всеми типами условий
    
    Пример:
        filters = create_combined_filter(
            must=[
                FieldCondition(key="status", match=MatchValue(value="active"))
            ],
            should=[
                FieldCondition(key="category", match=MatchAny(any=["cat1", "cat2"]))
            ],
            must_not=[
                FieldCondition(key="archived", match=MatchValue(value=True))
            ],
            min_should=1
        )
    """
    return QdrantFilters(
        must=must,
        should=should,
        must_not=must_not,
        min_should=min_should
    )

