from app.utils.tools.documents import (
    search_documents,
    get_document_by_id,
    query_documents,
    search_documents_by_keywords
)
from app.utils.tools.rules import (
    search_rules,
    get_rule_by_id
)

# Для Ollama нужно передавать функции как объекты, а не как значения словаря
available_tools = [
    search_documents,
    query_documents,
    search_rules,
]

available_tools_dict = {
    "search_documents": search_documents,
    "query_documents": query_documents,
    "search_rules": search_rules,
}


