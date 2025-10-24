from typing import List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import logging
from app.dto.ai_models import ContractSectionsOutput, ParsedDocumentOutput, DocTypeClassification
from app.utils.tools.parser import get_document_clauses
logger = logging.getLogger(__name__)


PARSER_SYSTEM_PROMPT = (
    "You are an expert Document Parser and Data Structurer equipped with OCR.\n"
    "Task: Parse the uploaded file into high-quality chunks for vector search.\n\n"
    "STRICT REQUIREMENTS:\n"
    "- Perform OCR on images and scanned pages automatically.\n"
    "- Preserve ALL information, including tables (emit tables as GitHub-flavored Markdown), bullet lists, footnotes, headers.\n"
    "- Detect the document type strictly from the allowed set: INVOICE, CONTRACT, COO, COA, COW, COQ, BL, FINANCIAL, LC, OTHER.\n"
    "- Call the tool get_document_clauses to fetch canonical clause titles for the detected type.\n"
    "- Segment the content by mapping document headings/sections to the canonical clauses. If a clause is absent, allocate related content to the closest clause.\n"
    "- If a section exceeds the token budget, split it into smaller coherent parts without losing context.\n"
    "- Keep tables within the same chunk as their surrounding text; do NOT separate tables into standalone chunks.\n"
    "- Output strictly structured JSON as specified. Do not summarize or invent content.\n"
)


class DocumentParserOpenAI:
    def __init__(self, client: AsyncOpenAI):
        self.openai_client = client

    async def parse_document(self, file_path: str) -> ParsedDocumentOutput:
        """Parse any file via OpenAI OCR with clause-aware segmentation and return structured chunks."""
        uploaded_file = await self.openai_client.files.create(
            file=open(file_path, "rb"),
            purpose="assistants"
        )

        # Step 1: classify document type
        classify_prompt = (
            "Classify the uploaded file into one of the allowed document types. "
            "Return only the JSON matching the schema."
        )
        classify_resp = await self.openai_client.responses.parse(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": [
                    {"type": "input_text", "text": PARSER_SYSTEM_PROMPT},
                ]},
                {"role": "user", "content": [
                    {"type": "input_text", "text": classify_prompt},
                    {"type": "input_file", "file_id": uploaded_file.id}
                ]},
            ],
            text_format=DocTypeClassification,
        )
        doc_type = classify_resp.output_parsed.document_type

        # Fetch canonical clauses via dedicated parser tool (local runtime)
        clauses_payload = get_document_clauses(doc_type)
        clauses_list = clauses_payload.get("clauses", [])

        # Step 2: segment document using canonical clauses
        segment_prompt = (
            "Use OCR to extract the full text and segment it according to these canonical clauses: "
            f"{clauses_list}. "
            "Map semantically similar headings to the closest clause. If a clause is not present, omit it. "
            "Keep tables inline with their surrounding text and output them as GitHub-flavored Markdown tables. "
            "Split oversized sections into smaller coherent parts as needed. "
            "Return only the structured JSON as specified."
        )

        segment_resp = await self.openai_client.responses.parse(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": [
                    {"type": "input_text", "text": PARSER_SYSTEM_PROMPT},
                ]},
                {"role": "user", "content": [
                    {"type": "input_text", "text": segment_prompt},
                    {"type": "input_file", "file_id": uploaded_file.id}
                ]},
            ],
            text_format=ParsedDocumentOutput,
        )

        output: ParsedDocumentOutput = segment_resp.output_parsed
        # Ensure classification present
        output.document_type = classify_resp.output_parsed
        return output