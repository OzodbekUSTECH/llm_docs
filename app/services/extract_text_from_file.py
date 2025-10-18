from typing import List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import logging
from app.dto.ai_models import ContractSectionsOutput
logger = logging.getLogger(__name__)


class DocumentParserOpenAI:
    def __init__(self,client: AsyncOpenAI):
        self.openai_client = client

    async def parse_contract(self, file_path: str) -> tuple[ContractSectionsOutput, str]:
        # Upload file (PDF, DOCX, image — OCR happens automatically)
        uploaded_file = await self.openai_client.files.create(
            file=open(file_path, "rb"),
            purpose="assistants"
        )

        system_prompt = (
            "You are an expert document parser and data structurer. "
            "Your goal is to extract and structure all text content from the provided file. "
            "Perform OCR if the file contains images or scanned pages. "
            "Preserve all data — including tables, bullet points, and formatting (use markdown for tables). "
            "The extracted output will be used for vector-based semantic search (e.g., in Qdrant). "
            "Therefore, divide the document into coherent, semantically complete chunks of text, "
            "suitable for embedding and retrieval. "
            "Avoid hallucinating or inventing new titles — if no title exists, use a short neutral label like 'Section 1', 'Section 2', etc. "
            "Ensure that no information is lost or summarized. Return a complete and lossless structured representation."
        )

        user_prompt = (
            "Analyze the uploaded file and split it into meaningful semantic chunks. "
            "Each chunk should have a title (use the existing section header if present; otherwise, use a simple label like 'Section 1', 'Section 2', etc.) "
            "and its full content text. "
            "Do NOT omit or summarize anything. "
            "The output will be used for embedding and semantic search, so make sure the text is clean, complete, and contextually independent."
        )

        response = await self.openai_client.responses.parse(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": [
                    {"type": "input_text", "text": system_prompt},
                ]},
                {"role": "user", "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_file", "file_id": uploaded_file.id}
                ]},
            ],
            text_format=ContractSectionsOutput,
            # temperature=0,
        )

        output: ContractSectionsOutput = response.output_parsed
        
        full_content = "\n\n".join(
            section.content.strip() for section in output.sections if section.content
        )
        
        return output, full_content