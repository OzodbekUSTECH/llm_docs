
# OpenAI API tools definition
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search for relevant documents using semantic vector search. Returns formatted text with document information including filename, ID, size, preview, and relevance score. This tool performs intelligent semantic search across all uploaded documents and returns the most relevant information. Use this tool to find any factual information, data, or content from documents (e.g., company names, owners, directors, contracts, specifications, reports, etc.). The tool returns formatted results ready for display to users.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query. Be specific and descriptive. Examples: 'owner of Floriana Impex company', 'vessel name and specifications', 'contract details and terms', 'company registration information', 'financial data for Q4 2023'. The more specific your query, the better the results."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return. Higher values provide more context but may include less relevant results. Recommended: 3-5 for focused searches, 5-10 for comprehensive searches. Default is 10 for maximum information coverage.",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_by_id",
            "description": "Get document information and metadata by document ID. Returns formatted document details including filename, ID, content type, status, file size, creation date, and optionally a content preview. Use this tool to get basic information about a specific document or a truncated content preview (max 2000 characters). For full document content, use get_document_full_content instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document UUID obtained from search_documents results. Each search result includes an 'id' field with the document UUID."
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "If true, includes a truncated content preview (max 2000 characters). If false, returns only metadata. Default is false.",
                        "default": False
                    }
                },
                "required": ["document_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_full_content",
            "description": "Get complete document content in chunks for processing large documents. This tool allows you to retrieve the full untruncated document content by breaking it into manageable chunks. Use this tool when you need to analyze the complete document content, not just excerpts. The tool provides pagination information to navigate through large documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "Document UUID obtained from search_documents or get_document_by_id results."
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Size of each content chunk in characters. Larger chunks provide more context but may be harder to process. Recommended: 3000-5000 characters. Default is 3000.",
                        "default": 3000,
                        "minimum": 1000,
                        "maximum": 10000
                    },
                    "chunk_index": {
                        "type": "integer",
                        "description": "Which chunk to retrieve (0-based index). Use 0 for the first chunk, 1 for the second, etc. The tool will tell you the total number of chunks available.",
                        "default": 0,
                        "minimum": 0
                    }
                },
                "required": ["document_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_documents",
            "description": "Query documents with flexible filtering, sorting, and aggregation. This powerful tool allows you to count documents, filter by various criteria (status, content type, date range, file size), sort results, and group documents. Use this tool to answer questions like 'How many documents do I have?', 'Show me all PDF files', 'Find documents uploaded this month', or 'Group documents by status'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "description": "Dictionary of filters to apply. Available filters: status (PENDING/PROCESSING/COMPLETED/FAILED), content_type (e.g., 'application/pdf'), filename (partial match), original_filename (partial match), created_after (ISO date), created_before (ISO date), min_content_length (integer), max_content_length (integer).",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["PENDING", "PROCESSING", "COMPLETED", "FAILED"],
                                "description": "Filter by document status"
                            },
                            "content_type": {
                                "type": "string",
                                "description": "Filter by content type (e.g., 'application/pdf', 'text/plain')"
                            },
                            "filename": {
                                "type": "string",
                                "description": "Filter by filename (partial match, case-insensitive)"
                            },
                            "original_filename": {
                                "type": "string",
                                "description": "Filter by original filename (partial match, case-insensitive)"
                            },
                            "created_after": {
                                "type": "string",
                                "description": "Filter by creation date after this date (ISO format: '2024-01-01')"
                            },
                            "created_before": {
                                "type": "string",
                                "description": "Filter by creation date before this date (ISO format: '2024-12-31')"
                            },
                            "min_content_length": {
                                "type": "integer",
                                "description": "Filter by minimum content length in characters"
                            },
                            "max_content_length": {
                                "type": "integer",
                                "description": "Filter by maximum content length in characters"
                            }
                        }
                    },
                    "order_by": {
                        "type": "string",
                        "enum": ["created_at", "filename", "content_length", "status"],
                        "description": "Field to sort results by. Options: created_at (newest first), filename (alphabetical), content_length (largest first), status (alphabetical)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return. Use null for no limit.",
                        "minimum": 1,
                        "maximum": 100
                    },
                    "count_only": {
                        "type": "boolean",
                        "description": "If true, returns only the count of documents without details. Use this for questions like 'How many documents do I have?'. Default is false.",
                        "default": False
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["status", "content_type", "filename"],
                        "description": "Group documents by this field. Options: status (group by document status), content_type (group by file type), filename (group by file extension)"
                    }
                },
                "required": []
            }
        }
    }
]