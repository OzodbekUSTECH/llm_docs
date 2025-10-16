
# OpenAI API tools definition
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search for relevant documents using semantic vector search. "
                "Always pass the user's full query (not just individual keywords) for the best results – do not extract or break down to only certain words. "
                "This tool performs intelligent semantic search across all uploaded documents and returns the most relevant information including filename, ID, size, document type, and extracted keywords with their values. "
                "Documents are automatically categorized by type (INVOICE, CONTRACT, COO, COA, COW, COQ, BL, FINANCIAL, LC, OTHER). "
                "Use this tool to find documents containing specific information (e.g., company names, owners, directors, contracts, specifications, reports, etc.) and get their key extracted data points. "
                "You can optionally limit the search to specific documents by providing their IDs or filter by document types. "
                "The tool returns formatted results with keywords for efficient processing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A natural language search query. "
                            "Always provide the full query exactly as entered by the user, not individual keywords or a processed/shortened form. "
                            "The more specific and descriptive the user's query is, the better the results. "
                            "You can also combine with document_types parameter to filter by specific document types. "
                            "Additionally, you can use the 'document_ids' parameter to filter by specific documents and find the most relevant chunk within a particular document."
                        )
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return. Higher values provide more context but may include less relevant results",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "document_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional list of specific document IDs to search within. If provided, search will be limited to these documents only. Useful for large document collections or when you want to focus on specific documents. Each ID should be a UUID string obtained from previous search results or query_documents."
                    },
                    "document_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["INVOICE", "CONTRACT", "COO", "COA", "COW", "COQ", "BL", "FINANCIAL", "LC", "OTHER"]
                        },
                        "description": (
                            "Optional list of document types to filter by. "
                            "Available types:\n"
                            "INVOICE (invoices and bills),\n"
                            "CONTRACT (contracts and agreements),\n"
                            "COO (Certificate of Origin),\n"
                            "COA (Certificate of Analysis),\n"
                            "COW (Certificate of Weight),\n"
                            "COQ (Certificate of Quality),\n"
                            "BL (Bill of Lading),\n"
                            "FINANCIAL (financial reports),\n"
                            "LC (Letter of Credit),\n"
                            "OTHER (other documents).\n"
                            "Examples: ['INVOICE'] for only invoices, ['CONTRACT', 'INVOICE'] for contracts and invoices, ['COO', 'COA', 'COW', 'COQ'] for certificates only."
                        )
                    }
                },
                "required": ["query"]
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
                            "document_type": {
                                "type": "string",
                                "enum": [
                                    "INVOICE",
                                    "CONTRACT",
                                    "COO",
                                    "COA",
                                    "COW",
                                    "COQ",
                                    "BL",
                                    "FINANCIAL",
                                    "LC",
                                    "OTHER"
                                ],
                                "description": (
                                    "Filter by document type. Available types:\n"
                                    "INVOICE (invoices and bills),\n"
                                    "CONTRACT (contracts and agreements),\n"
                                    "COO (Certificate of Origin),\n"
                                    "COA (Certificate of Analysis),\n"
                                    "COW (Certificate of Weight),\n"
                                    "COQ (Certificate of Quality),\n"
                                    "BL (Bill of Lading),\n"
                                    "FINANCIAL (financial reports),\n"
                                    "LC (Letter of Credit),\n"
                                    "OTHER (other documents)."
                                )
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_rules",
            "description": (
                "Search for business rules and policies that explain HOW TO WORK WITH DOCUMENTS. "
                "Returns step-by-step instructions, procedures, and guidelines for document processing. "
                "Use this tool when users ask 'How do I...', 'What is the process for...', 'What are the rules for...' "
                "or need guidance on document workflows. Perfect for finding: "
                "• Document verification procedures and validation rules "
                "• Document comparison methods and parameters to check "
                "• Document processing workflows and approval processes "
                "• Compliance requirements and quality standards "
                "• Security protocols and data handling procedures "
                "• Business policies and regulatory guidelines "
                "• Step-by-step instructions for working with different document types "
                "This tool provides the 'HOW TO' knowledge that complements document content search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language search query about HOW TO WORK WITH DOCUMENTS. Be specific and descriptive. "
                            "Examples: 'How to verify invoice accuracy', 'What parameters to compare when checking documents', "
                            "'Document validation procedures', 'Invoice processing workflow', 'Contract approval process', "
                            "'Document quality standards', 'How to handle different document types', "
                            "'Document security requirements', 'Data processing guidelines', "
                            "'Document comparison methods', 'Approval workflow steps'. "
                            "Focus on processes, procedures, and step-by-step instructions."
                        )
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rules to return. Higher values provide more context but may include less relevant results. Recommended: 3-5 for focused searches, 5-10 for comprehensive searches. Default is 10.",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "rule_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Optional list of specific rule IDs to search within. If provided, search will be limited to these rules only. Useful for large rule collections or when you want to focus on specific rules. Each ID should be a UUID string obtained from previous search results."
                    },
                    "category_id": {
                        "type": "string",
                        "description": "Optional category ID to filter rules by specific category. Use this to find rules within a particular category or domain (e.g., security rules, data handling rules, document verification, etc.)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_rule_by_id",
            "description": "Get complete rule information by ID including full description, category details, and metadata. Returns formatted rule details with title, ID, category information, complete description, and creation/update timestamps. Use this tool to get detailed information about a specific rule when you have its ID from search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {
                        "type": "string",
                        "description": "Rule UUID obtained from search_rules results. Each search result includes an 'id' field with the rule UUID."
                    }
                },
                "required": ["rule_id"]
            }
        }
    },
]