import asyncio
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from fastapi import UploadFile
from sentence_transformers import SentenceTransformer

from app.entities.documents import Document
from app.repositories.documents import DocumentsRepository
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.repositories.uow import UnitOfWork
from app.services.extract_text_from_file import DocumentParserOpenAI
from app.utils.enums import DocumentStatus, DocumentType
from app.exceptions.app_error import AppError
from qdrant_client import AsyncQdrantClient
from app.utils.collections import Collections
from app.services.keyword_extractor import KeywordExtractor
from app.services.contract_section_extractor import ContractSectionExtractor
from app.services.document_chunker import DocumentChunker, Chunk
import logging

logger = logging.getLogger(__name__)

class CreateDocumentInteractor:
    def __init__(
        self,
        uow: UnitOfWork,
        documents_repository: DocumentsRepository,
        qdrant_embeddings_repository: QdrantEmbeddingsRepository,
        sentence_transformer: SentenceTransformer,
        qdrant_client: AsyncQdrantClient,
        keyword_extractor: KeywordExtractor,
        contract_section_extractor: ContractSectionExtractor,
        document_parser: DocumentParserOpenAI,
        document_chunker: DocumentChunker
    ):
        self.uow = uow
        self.documents_repository = documents_repository
        self.qdrant_embeddings_repository = qdrant_embeddings_repository
        self.sentence_transformer = sentence_transformer
        self.qdrant_client = qdrant_client
        self.storage_dir = Path("storage/documents")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.keyword_extractor = keyword_extractor
        self.contract_section_extractor = contract_section_extractor
        self.document_parser = document_parser
        self.document_chunker = document_chunker
        
    def _detect_document_type(self, filename: str, content: str) -> DocumentType:
        """
        Определяет тип документа на основе имени файла и содержимого.
        
        Args:
            filename: Имя файла
            content: Содержимое документа
            
        Returns:
            DocumentType: Определенный тип документа
        """
        # Нормализуем текст для поиска
        filename_lower = filename.lower()
        content_lower = content.lower() if content else ""

        # Keyword dictionaries for each document type (all in English, including COQ - Certificate of Quality)
        type_keywords = {
            DocumentType.INVOICE: {
                'filename': ['invoice', 'bill', 'inv'],
                'content': [
                    'invoice', 'bill to', 'total amount', 'payment terms', 'due date',
                    'invoice number', 'amount due', 'payable to'
                ]
            },
            DocumentType.CONTRACT: {
                'filename': ['contract', 'agreement'],
                'content': [
                    'hereby agree', 'parties agree', 'contract', 'agreement', 'terms and conditions',
                    'this agreement', 'the parties', 'contract number'
                ]
            },
            DocumentType.COO: {
                'filename': ['coo', 'certificate of origin', 'origin'],
                'content': [
                    'certificate of origin', 'country of origin', 'goods originate', 'origin certificate',
                    'place of origin', 'originating from'
                ]
            },
            DocumentType.COA: {
                'filename': ['coa', 'certificate of analysis', 'analysis'],
                'content': [
                    'certificate of analysis', 'test results', 'analytical results', 'specifications met',
                    'analysis report', 'quality analysis'
                ]
            },
            DocumentType.COW: {
                'filename': ['cow', 'certificate of weight', 'weight certificate'],
                'content': [
                    'certificate of weight', 'gross weight', 'net weight', 'weight certificate',
                    'total weight', 'weighing'
                ]
            },
            DocumentType.COQ: {
                'filename': ['coq', 'certificate of quality', 'quality certificate'],
                'content': [
                    'certificate of quality', 'quality certificate', 'quality assurance', 'quality control',
                    'meets quality standards', 'quality test'
                ]
            },
            DocumentType.BL: {
                'filename': ['bl', 'bill of lading', 'lading'],
                'content': [
                    'bill of lading', 'consignee', 'shipper', 'vessel', 'port of loading', 'port of discharge',
                    'lading number', 'cargo manifest'
                ]
            },
            DocumentType.LC: {
                'filename': ['lc', 'letter of credit', 'swift'],
                'content': [
                    'letter of credit', 'swift', 'bank', 'beneficiary', 'applicant', 'issuing bank'
                ]
            },
            DocumentType.FINANCIAL: {
                'filename': ['financial', 'report', 'statement', 'balance'],
                'content': [
                    'financial statement', 'balance sheet', 'income statement', 'cash flow', 'assets', 'liabilities',
                    'profit and loss', 'statement of financial position'
                ]
            }
        }
        # Подсчитываем очки для каждого типа
        scores = {}
        
        for doc_type, keywords in type_keywords.items():
            score = 0
            
            # Проверяем ключевые слова в имени файла (больший вес)
            for keyword in keywords['filename']:
                if keyword in filename_lower:
                    score += 3  # Имя файла имеет больший вес
            
            # Проверяем ключевые слова в содержимом
            for keyword in keywords['content']:
                if keyword in content_lower:
                    score += 1
            
            scores[doc_type] = score
        
        # Находим тип с максимальным скором
        max_score = max(scores.values()) if scores else 0
        
        if max_score > 0:
            # Возвращаем тип с максимальным скором
            detected_type = max(scores, key=scores.get)
            logger.info(f"Detected document type: {detected_type.value} (score: {max_score})")
            return detected_type
        
        logger.info("Could not detect specific document type, using OTHER")
        return DocumentType.OTHER
        
    # Docling flow removed: all parsing handled by LLM OCR parser
    async def _extract_text_and_tables(self, file_path: str):
        raise NotImplementedError("Docling extraction is disabled; use LLM parser")

    async def execute(self, file: UploadFile) -> Document:
        """LLM-only parsing with OpenAI OCR, clause-aware chunking, and vector storage."""
        # 1. Читаем файл в память
        file_bytes = await file.read()

        # Проверка на пустой файл
        if not file_bytes or len(file_bytes) == 0:
            raise AppError(status_code=400, message="Файл пустой")

        # 2. Считаем хэш
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # 3. Проверяем, нет ли такого файла уже в БД
        existing = await self.documents_repository.get_one(
            where=[Document.file_hash == file_hash]
        )
        if existing:
            raise AppError(status_code=400, message="Этот файл уже загружен")

        # 4. Сохраняем файл на диск
        ext = Path(file.filename).suffix
        id = uuid.uuid4()
        stored_filename = f"{id}{ext}"
        file_path = self.storage_dir / stored_filename
        with open(file_path, "wb") as f:
            f.write(file_bytes)
            
        try:
            # LLM OCR parsing
            parsed = await self.document_parser.parse_document(str(file_path))

            detected_type_str = parsed.document_type.document_type if parsed and parsed.document_type else "OTHER"
            try:
                detected_type = DocumentType(detected_type_str)
            except Exception:
                detected_type = DocumentType.OTHER

            # Build full content as concatenation of chunk contents
            full_content = "\n\n".join([c.content.strip() for c in parsed.chunks if c.content])

            if not full_content:
                raise AppError(status_code=400, message="Не удалось извлечь текст из файла. Файл может быть пустым или содержать только изображения.")

            document = Document(
                id=id,
                filename=stored_filename,
                original_filename=file.filename,
                file_path=str(file_path),
                content_type=file.content_type or "application/octet-stream",
                file_hash=file_hash,
                status=DocumentStatus.COMPLETED,
                content=full_content,
                type=detected_type,
                keywords={},
            )
            await self.documents_repository.create(document)

            # Base metadata for all chunks
            base_metadata = {
                "filename": document.original_filename,
                "content_type": file.content_type or "application/octet-stream",
                "document_type": detected_type.value,
                "document_id": str(document.id),
            }

            # Convert LLM chunks to Chunk dataclass with rich metadata
            structured_chunks: List[Chunk] = []
            for idx, ch in enumerate(parsed.chunks):
                text = ""
                if ch.clause:
                    text += f"{ch.clause}\n"
                if ch.title:
                    text += f"{ch.title}\n\n"
                if ch.content:
                    text += f"{ch.content}"
                token_count = self.document_chunker._count_tokens(text)  # reuse tokenizer for consistency
                metadata = {
                    **base_metadata,
                    "section_title": ch.title,
                    "clause": ch.clause or None,
                    "chunk_type": "llm_clause" if ch.clause else "llm_section",
                }
                structured_chunks.append(Chunk(
                    text=text,
                    metadata=metadata,
                    index=idx,
                    token_count=token_count,
                ))

            logger.info(f"Created {len(structured_chunks)} LLM-structured chunks")

            # Prepare embeddings
            chunk_texts = [chunk.text for chunk in structured_chunks]
            embeddings = self.sentence_transformer.encode(
                chunk_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=16,
            )

            # Persist to Qdrant with metadata
            await self.qdrant_embeddings_repository.bulk_create_embeddings_with_metadata(
                collection_name=Collections.DOCUMENT_EMBEDDINGS,
                document_id=str(document.id),
                chunks=structured_chunks,
                embeddings=embeddings.tolist(),
            )

            await self.uow.commit()
            logger.info("✅ LLM parsing successful")
            return document

        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            if file_path.exists():
                file_path.unlink()
            raise AppError(status_code=400, message=f"Не удалось обработать файл {ext}. {str(e)}")
        
        

    # Docling chunking method removed