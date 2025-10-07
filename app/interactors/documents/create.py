import asyncio
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from docling_core.types.doc.document import DoclingDocument
from fastapi import UploadFile
from sentence_transformers import SentenceTransformer

from app.entities.documents import Document
from app.repositories.documents import DocumentsRepository
from app.repositories.qdrant_embeddings import QdrantEmbeddingsRepository
from app.repositories.uow import UnitOfWork
from app.utils.enums import DocumentStatus
from app.exceptions.app_error import AppError
from qdrant_client import AsyncQdrantClient

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
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
        document_converter: DocumentConverter,
        docling_chunker: HybridChunker
    ):
        self.uow = uow
        self.documents_repository = documents_repository
        self.qdrant_embeddings_repository = qdrant_embeddings_repository
        self.sentence_transformer = sentence_transformer
        self.qdrant_client = qdrant_client
        self.storage_dir = Path("storage/documents")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.document_converter = document_converter
        self.docling_chunker = docling_chunker
        
        
    async def _extract_text_and_tables(self, file_path: str) -> Tuple[Optional[str], List[Dict]]:
        """Extract text content and tables from file using Docling"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Handle plain text files directly (Docling doesn't support them)
        if file_path.suffix.lower() == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Read plain text file directly: {len(content)} characters")
                return content, []  # No tables in plain text
            except Exception as e:
                logger.error(f"Failed to read plain text file {file_path}: {e}")
                raise Exception(f"Plain text file reading failed: {str(e)}")
        
        try:
            # Run Docling conversion in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.document_converter.convert, 
                str(file_path)
            )
            
            
            # Export to markdown for rich text preservation
            # This includes tables, lists, headers, and other structure
            markdown_content = result.document.export_to_markdown()
            
            # If markdown is empty, try plain text export
            if not markdown_content or not markdown_content.strip():
                # Fall back to text representation
                text_content = str(result.document)
                content = text_content if text_content.strip() else None
            else:
                content = markdown_content
            
            # Extract tables from the document
            tables = []
            try:
                # Docling provides tables through the document structure
                if hasattr(result.document, 'tables'):
                    for idx, table in enumerate(result.document.tables):
                        # Convert table to a structured format
                        table_data = {
                            "index": idx,
                            "rows": [],
                            "headers": [],
                            "caption": getattr(table, 'caption', None)
                        }
                        
                        # Extract table content - convert TableData to JSON-serializable format
                        if hasattr(table, 'data') and table.data:
                            # table.data is a TableData object, extract its grid
                            if hasattr(table.data, 'grid'):
                                # Convert grid of TableCell objects to simple array of arrays
                                table_rows = []
                                for row in table.data.grid:
                                    row_data = []
                                    for cell in row:
                                        if hasattr(cell, 'text'):
                                            row_data.append(cell.text)
                                        else:
                                            row_data.append(str(cell))
                                    table_rows.append(row_data)
                                table_data["rows"] = table_rows
                                
                                # Try to extract headers from first row if they are marked as headers
                                if table_rows and hasattr(table.data, 'grid') and len(table.data.grid) > 0:
                                    first_row = table.data.grid[0]
                                    if any(hasattr(cell, 'column_header') and cell.column_header for cell in first_row):
                                        table_data["headers"] = table_rows[0]
                                        table_data["rows"] = table_rows[1:]  # Remove header row from data
                                    
                        # Try to get HTML representation if available
                        try:
                            if hasattr(table, 'to_html'):
                                table_data["html"] = table.to_html()
                        except:
                            pass
                        
                        # Try to get CSV representation if available
                        try:
                            if hasattr(table, 'to_csv'):
                                table_data["csv"] = table.to_csv()
                        except:
                            pass
                            
                        tables.append(table_data)
                        logger.info(f"Extracted table {idx} with {len(table_data.get('rows', []))} rows from document")
                
                # Also check for tables in the document elements
                if hasattr(result.document, 'elements'):
                    for element in result.document.elements:
                        if hasattr(element, 'type') and element.type == 'table':
                            table_data = {
                                "index": len(tables),
                                "content": str(element),
                                "type": "element_table"
                            }
                            tables.append(table_data)
                            
            except Exception as e:
                logger.warning(f"Failed to extract tables: {e}")
                # Continue processing even if table extraction fails
            
            logger.info(f"Extracted {len(tables)} tables from document")
            return content, tables, result.document
            
        except Exception as e:
            logger.error(f"Docling extraction failed for {file_path}: {e}")
            raise Exception(f"Document extraction failed: {str(e)}")

    async def execute(self, file: UploadFile) -> Document:
        """
        Парсинг файла с использованием Docling библиотеки.
        Docling обеспечивает более качественное извлечение текста с сохранением структуры документа.
        """
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
            content, tables, dl_doc = await self._extract_text_and_tables(file_path)
            print(f"✅ Docling документ извлечен: {content}")
            
            if not content:
                raise AppError(status_code=400, message="Не удалось извлечь текст из файла. Файл может быть пустым или содержать только изображения.")
            
            document = Document(
                id=id,
                filename=stored_filename,
                original_filename=file.filename,
                file_path=str(file_path),
                content_type=file.content_type or "application/octet-stream",
                file_hash=file_hash,
                status=DocumentStatus.COMPLETED,
                content=content,
                tables=tables,
            )
            await self.documents_repository.create(document)
            
            chunks = self._chunk_with_docling(dl_doc)
            
            embeddings = self.sentence_transformer.encode(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=8,
            )
            
            await self.qdrant_embeddings_repository.bulk_create_embeddings(
                document_id=str(document.id),
                chunks=chunks,  # Сохраняем БЕЗ префикса
                embeddings=embeddings.tolist(),
                # Добавляем метаданные для лучшей фильтрации
                metadata={
                    "filename": document.original_filename,
                    "content_type": file.content_type,
                }
            )
            
            
            await self.uow.commit()
            
            print(f"✅ Docling парсинг успешен:")
            
            return document
            
            
           
        except Exception as e:
            print(f"❌ Ошибка парсинга с Docling: {e}")
            # Удаляем файл при ошибке парсинга
            if file_path.exists():
                file_path.unlink()
            raise AppError(status_code=400, message=f"Не удалось обработать файл {ext}. {str(e)}")
        
        

    def _chunk_with_docling(self, docling_document: DoclingDocument) -> List[str]:
        """
        Разделяет контент на чанки с помощью Docling HybridChunker.
        
        HybridChunker:
        1. Уважает структуру документа (не разрывает семантические блоки)
        2. Добавляет контекст из заголовков через contextualize()
        3. Учитывает токены, а не символы
        4. Объединяет маленькие соседние чанки
        
        """
       
            
        # Получаем итератор чанков
        chunk_iter = self.docling_chunker.chunk(dl_doc=docling_document)
        
        # Обрабатываем чанки с контекстуализацией
        chunks = []
        
        for chunk in chunk_iter:
            
            # КЛЮЧЕВОЙ МОМЕНТ: используем contextualize() для добавления контекста
            # Это добавляет заголовки разделов к чанку для лучшего понимания
            enriched_text = self.docling_chunker.contextualize(chunk=chunk)
            
            chunks.append(enriched_text)
            
        
        return chunks