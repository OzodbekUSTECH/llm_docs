from app.utils.tools.documents import (
    search_documents,
    get_document_by_id,
    get_document_full_content,
    query_documents
)

# Для Ollama нужно передавать функции как объекты, а не как значения словаря
available_tools = [
    search_documents,
    get_document_by_id,
    get_document_full_content,
    query_documents
]

available_tools_dict = {
    "search_documents": search_documents,
    "get_document_by_id": get_document_by_id,
    "get_document_full_content": get_document_full_content,
    "query_documents": query_documents
}


