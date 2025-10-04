from app.utils.tools.documents import (
    search_documents,
    get_document_by_id,
    upload_document
)

# Для Ollama нужно передавать функции как объекты, а не как значения словаря
available_tools = [
    search_documents,
    get_document_by_id,
]

available_tools_dict = {
    "search_documents": search_documents,
    "get_document_by_id": get_document_by_id,
}