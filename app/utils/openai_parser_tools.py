from typing import List, Dict, Any


# OpenAI API tools definition dedicated for the document parser flow
# IMPORTANT: Do not mix with chat tools
OPENAI_PARSER_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_document_clauses",
            "description": (
                "Return the canonical clause titles for a specific document type. "
                "Use these clause titles as the primary segmentation anchors when splitting the document. "
                "If a clause title doesn't explicitly exist in the document, map semantically similar headings to the closest canonical clause."
            ),
            "parameters": {
                "type": "object",
                "properties": {
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
                            "OTHER",
                        ],
                        "description": "Strict document type identifier"
                    }
                },
                "required": ["document_type"]
            }
        }
    }
]


