from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ExtractedKeywords(BaseModel):
    """
    Простая модель для извлечения ключевых слов из документов.
    GPT возвращает чистый JSON с key: value парами в поле keywords.
    """
    keywords: Optional[Dict[str, str]] = Field(
        default=None,
        description="Словарь ключевых слов, где ключ - название поля, значение - извлеченное значение"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в формат, ожидаемый приложением"""
        return self.keywords or {}