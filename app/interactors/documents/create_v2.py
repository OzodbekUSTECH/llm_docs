import asyncio
from collections import defaultdict
import hashlib
import shutil
import time
import uuid
from uuid import UUID
from pathlib import Path
from typing import Any, Generator, List, Dict, Literal, Optional, Tuple
from docling.document_converter import DocumentConverter
from fastapi import UploadFile
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.dto.schemas import DocumentSchema
from app.entities.sources import Source
from app.repositories.indexes import IndexesRepository
from app.repositories.sources import SourcesRepository
from app.repositories.uow import UnitOfWork
from app.services.extract_text_from_file import DocumentParserOpenAI
from app.utils.docs_store import LanceDBDocumentStore
from app.utils.enums import DocumentStatus, DocumentType, IndexType
from app.exceptions.app_error import AppError
from qdrant_client import AsyncQdrantClient
from app.utils.collections import Collections
from app.services.keyword_extractor import KeywordExtractor
from app.services.contract_section_extractor import ContractSectionExtractor
from app.services.document_chunker import DocumentChunker, Chunk
import logging
from PIL import Image
from io import BytesIO
import base64
from hashlib import sha256
from llama_index.core.readers.file.base import default_file_metadata_func

from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.schema import TextNode
from app.entities.indexes import Index
from app.utils.vectors_store import QdrantVectorStore
from app.utils.embeddings import OpenAIEmbeddings
from app.utils.collections import Collections
from app.core.config import settings
logger = logging.getLogger(__name__)




def crop_image(file_path: Path, bbox: list[float], page_number: int = 0) -> Image.Image:
    """Crop the image based on the bounding box

    Args:
        file_path (Path): path to the image file
        bbox (list[float]): bounding box of the image (in percentage [x0, y0, x1, y1])
        page_number (int, optional): page number of the image. Defaults to 0.

    Returns:
        Image.Image: cropped image
    """
    left, upper, right, lower = bbox

    left, right = min(left, right), max(left, right)
    upper, lower = min(upper, lower), max(upper, lower)

    img: Image.Image
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz
        except ImportError:
            raise ImportError("Please install PyMuPDF: 'pip install PyMuPDF'")

        doc = fitz.open(file_path)
        page = doc.load_page(page_number)
        pm = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
    elif suffix in [".tif", ".tiff"]:
        img = Image.open(file_path)
        img.seek(page_number)
    else:
        img = Image.open(file_path)

    return img.crop(
        (
            int(left * img.width),
            int(upper * img.height),
            int(right * img.width),
            int(lower * img.height),
        )
    )
    
    
def make_markdown_table(table_as_list: List[List[str]]) -> str:
    """
    Convert table from python list representation to markdown format.
    The input list consists of rows of tables, the first row is the header.

    Args:
        table_as_list: list of table rows
            Example: [["Name", "Age", "Height"],
                    ["Jake", 20, 5'10],
                    ["Mary", 21, 5'7]]
    Returns:
        markdown representation of the table
    """
    markdown = "\n" + str("| ")

    for e in table_as_list[0]:
        to_add = " " + str(e) + str(" |")
        markdown += to_add
    markdown += "\n"

    markdown += "| "
    for i in range(len(table_as_list[0])):
        markdown += str("--- | ")
    markdown += "\n"

    for entry in table_as_list[1:]:
        markdown += str("| ")
        for e in entry:
            to_add = str(e) + str(" | ")
            markdown += to_add
        markdown += "\n"

    return markdown + "\n"



class SimpleTokenSplitter:
    """Простой splitter без наследований"""
    
    def __init__(self, chunk_size=1024, chunk_overlap=256, separator="\n\n", 
                 backup_separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.backup_separators = backup_separators or ["\n", ".", "\u200B"]
        
        # Создаем LlamaIndex splitter
        self._splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            backup_separators=backup_separators
        )
    
    def __call__(self, documents: List[DocumentSchema]) -> List[DocumentSchema]:
        """Вызывается как self.splitter(text_docs)"""
        return self.split_documents(documents)
    
    def split_documents(self, documents: List[DocumentSchema]) -> List[DocumentSchema]:
        """Разделяет документы на чанки"""
        all_chunks = []
        
        for doc in documents:
            # Извлекаем текст из документа
            text = doc.text or ""
            metadata = doc.metadata or {}
            
            if not text:
                continue
            
            # НЕ разделяем таблицы - они должны оставаться целыми
            if metadata.get("type") == "table":
                # Таблицы не разделяем, оставляем как есть
                print(f"SPLITTER: Keeping table whole - type={metadata.get('type')}, table_index={metadata.get('table_index')}")
                all_chunks.append(doc)
                continue
            
            # Разделяем только текстовые документы
            chunks = self._splitter.split_text(text)
            
            # Создаем чанки с метаданными
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_doc_id": metadata.get("file_id", str(doc.doc_id)),
                }
                
                chunk = DocumentSchema(
                    text=chunk_text,
                    metadata=chunk_metadata
                )
                all_chunks.append(chunk)
        
        return all_chunks

class CreateDocumentV2Interactor:
    def __init__(
        self,
        uow: UnitOfWork,
        sources_repository: SourcesRepository,
        indexes_repository: IndexesRepository,
        converter: DocumentConverter,
    ):
        self.uow = uow
        self.sources_repository = sources_repository
        self.indexes_repository = indexes_repository
        self.storage_dir = Path("storage/documents")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.converter = converter
        self.collection_name = Collections.DOCUMENT_EMBEDDINGS  # Используем правильную коллекцию
        self.splitter = SimpleTokenSplitter(
            chunk_size=1024,
            chunk_overlap=256,
            separator="\n\n",
            backup_separators=["\n", ".", "\u200B"]
        )
        
        self.docs_store = LanceDBDocumentStore()
        self.chunk_batch_size = 200
        self.embedding = OpenAIEmbeddings(model_name="text-embedding-3-large", api_key=settings.OPENAI_API_KEY)
        self.vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            vector_size=3072
        )
                
 
    async def stream(
        self, file_path: str | Path
    ) -> Generator[DocumentSchema, None, None]:
        # check if the file is already indexed
        if isinstance(file_path, Path):
            file_path = file_path.resolve()
            
        file_id = await self.get_id_if_exists(file_path)
        
        if file_id:
            raise AppError(status_code=400, message=f"File {file_path.name} already indexed")
        
        
        file_id = await self.store_file(file_path)
        
        extra_info = default_file_metadata_func(str(file_path))
        file_name = file_path.name
        
        extra_info["file_id"] = str(file_id)
        extra_info["collection_name"] = self.collection_name
        
        yield DocumentSchema(f" => Converting {file_name} to text", channel="debug")
        docs = self.load_data(file_path, extra_info)
        
        # Подсчитываем типы документов
        text_count = sum(1 for doc in docs if doc.metadata.get("type") == "text")
        table_count = sum(1 for doc in docs if doc.metadata.get("type") == "table")
        other_count = len(docs) - text_count - table_count
        
        yield DocumentSchema(f" => Converted {file_name}: {text_count} text blocks, {table_count} tables, {other_count} other elements", channel="debug")
        
        # Convert handle_docs to async and iterate through results
        async for doc_schema in self.handle_docs(docs, file_id, file_name):
            yield doc_schema
        
        await self.uow.commit()
        
        yield DocumentSchema(f" => Finished indexing {file_name}", channel="debug")
        return  
    
  
    async def handle_docs(self, docs, file_id, file_name) -> Generator[DocumentSchema, None, None]:
        s_time = time.time()
        text_docs = []
        table_docs = []
        non_text_docs = []
        thumbnail_docs = []

        for doc in docs:
            doc_type = doc.metadata.get("type", "text")
            print(f"Processing doc with type={doc_type}, metadata keys={list(doc.metadata.keys())}")
            if doc_type == "text":
                text_docs.append(doc)
            elif doc_type == "table":
                table_docs.append(doc)
                print(f"Added table doc: {doc.metadata.get('table_index', 'unknown')} on page {doc.metadata.get('page_label', 'unknown')}")
            elif doc_type == "thumbnail":
                thumbnail_docs.append(doc)
            else:
                non_text_docs.append(doc)
                
        print(f"Got {len(thumbnail_docs)} page thumbnails, {len(table_docs)} tables")
        page_label_to_thumbnail = {
            doc.metadata["page_label"]: str(doc.doc_id) for doc in thumbnail_docs
        }
        
        # Сообщение о начале обработки таблиц
        if table_docs:
            yield DocumentSchema(f" => Processing {len(table_docs)} tables with context", channel="debug")

        # Обрабатываем таблицы с контекстом - не разделяем их
        table_chunks = []
        for i, table_doc in enumerate(table_docs):
            page_label = table_doc.metadata.get("page_label", 1)
            
            # Находим текстовые чанки с той же страницы для контекста
            page_text_docs = [doc for doc in text_docs if doc.metadata.get("page_label") == page_label]
            
            # Объединяем текст страницы с таблицей
            context_text = ""
            if page_text_docs:
                context_text = "\n\n".join([doc.text for doc in page_text_docs])
            
            # Создаем чанк с таблицей и контекстом
            combined_text = f"{context_text}\n\n{table_doc.text}" if context_text else table_doc.text
            
            # Создаем новый DocumentSchema с уникальным ID для таблицы
            table_chunk = DocumentSchema(
                text=combined_text,
                metadata={
                    **table_doc.metadata,
                    "is_table_chunk": True,
                    "original_table_text": table_doc.text,
                    "context_text": context_text,
                    "table_index": i,
                    "total_tables": len(table_docs),
                    "file_id": str(file_id),
                    "source_doc_id": str(file_id)
                }
            )
            table_chunks.append(table_chunk)
            
            # Логируем каждую таблицу
            yield DocumentSchema(f" => Table {i+1}/{len(table_docs)} processed (Page {page_label})", channel="debug")

        # Разделяем только текстовые документы (без таблиц)
        if self.splitter:
            text_chunks = self.splitter(text_docs)
        else:
            text_chunks = text_docs
            
        # Логируем разделение текста
        yield DocumentSchema(f" => Split text into {len(text_chunks)} chunks", channel="debug")

        # add the thumbnails doc_id to the chunks and file_id to all chunks
        for chunk in text_chunks:
            page_label = chunk.metadata.get("page_label", None)
            if page_label and page_label in page_label_to_thumbnail:
                chunk.metadata["thumbnail_doc_id"] = page_label_to_thumbnail[page_label]
            # Add file_id to metadata for filtering
            chunk.metadata["file_id"] = str(file_id)
            chunk.metadata["source_doc_id"] = str(file_id)

        # ВАЖНО: Объединяем все чанки, но таблицы НЕ проходят через splitter
        # Таблицы уже обработаны отдельно в table_chunks
        all_chunks = text_chunks + table_chunks
        
        # Add file_id to all other documents
        for doc in non_text_docs + thumbnail_docs:
            if not hasattr(doc.metadata, 'get') or not doc.metadata:
                doc.metadata = {}
            doc.metadata["file_id"] = str(file_id)
            doc.metadata["source_doc_id"] = str(file_id)
        
        to_index_chunks = all_chunks + non_text_docs + thumbnail_docs
        
        # КРИТИЧЕСКИ ВАЖНО: Проверяем, что таблицы не разделились
        # Если таблицы разделились, это означает, что они попали в splitter
        original_table_count = len(table_docs)
        final_table_count = len(table_chunks)
        
        if original_table_count != final_table_count:
            yield DocumentSchema(f" => ERROR: Table count mismatch! Original: {original_table_count}, Final: {final_table_count}", channel="debug")
            # Принудительно исправляем - берем только оригинальные таблицы
            table_chunks = []
            for i, table_doc in enumerate(table_docs):
                # Создаем новый чанк с таблицей и контекстом
                page_label = table_doc.metadata.get("page_label", 1)
                page_text_docs = [doc for doc in text_docs if doc.metadata.get("page_label") == page_label]
                context_text = ""
                if page_text_docs:
                    context_text = "\n\n".join([doc.text for doc in page_text_docs])
                
                combined_text = f"{context_text}\n\n{table_doc.text}" if context_text else table_doc.text
                
                table_chunk = DocumentSchema(
                    text=combined_text,
                    metadata={
                        **table_doc.metadata,
                        "is_table_chunk": True,
                        "original_table_text": table_doc.text,
                        "context_text": context_text,
                        "table_index": i,
                        "total_tables": len(table_docs)
                    }
                )
                table_chunks.append(table_chunk)
            
            # Пересоздаем все чанки
            all_chunks = text_chunks + table_chunks
            to_index_chunks = all_chunks + non_text_docs + thumbnail_docs
            
            yield DocumentSchema(f" => FIXED: Recreated {len(table_chunks)} table chunks", channel="debug")
        
        # Отладочная информация
        print(f"FINAL CHUNKS: {len(text_chunks)} text + {len(table_chunks)} tables = {len(all_chunks)} total")
        for i, chunk in enumerate(all_chunks):
            chunk_type = chunk.metadata.get("type", "unknown")
            is_table_chunk = chunk.metadata.get("is_table_chunk", False)
            table_index = chunk.metadata.get("table_index", "N/A")
            text_preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            print(f"Chunk {i}: type={chunk_type}, is_table_chunk={is_table_chunk}, table_index={table_index}")
            print(f"  Text preview: {text_preview}")
            
        # Проверяем на дублирование контента
        text_contents = [chunk.text for chunk in all_chunks]
        unique_contents = set(text_contents)
        if len(text_contents) != len(unique_contents):
            yield DocumentSchema(f" => WARNING: Found {len(text_contents) - len(unique_contents)} duplicate chunks!", channel="debug")
        
        # Логируем финальную статистику
        yield DocumentSchema(f" => Final chunks: {len(text_chunks)} text + {len(table_chunks)} tables = {len(all_chunks)} total", channel="debug")
        
        # Проверяем, что таблицы не разделились
        table_chunk_count = sum(1 for chunk in all_chunks if chunk.metadata.get("is_table_chunk", False))
        if table_chunk_count != len(table_docs):
            yield DocumentSchema(f" => WARNING: Expected {len(table_docs)} table chunks, got {table_chunk_count}", channel="debug")
        
        # add to doc store
        chunks = []
        n_chunks = 0
        chunk_size = self.chunk_batch_size * 4
        for start_idx in range(0, len(to_index_chunks), chunk_size):
            chunks = to_index_chunks[start_idx : start_idx + chunk_size]
            await self.handle_chunks_docstore(chunks, file_id)
            n_chunks += len(chunks)
            yield DocumentSchema(
                f" => [{file_name}] Processed {n_chunks} chunks",
                channel="debug",
            )
            
        # Process chunks for vector store
        chunks = []
        n_chunks = 0
        chunk_size = self.chunk_batch_size
        for start_idx in range(0, len(to_index_chunks), chunk_size):
            chunks = to_index_chunks[start_idx : start_idx + chunk_size]
            await self.handle_chunks_vectorstore(chunks, file_id)
            n_chunks += len(chunks)
            yield DocumentSchema(
                f" => [{file_name}] Created embedding for {n_chunks} chunks",
                channel="debug",
            )
        
        print("indexing step took", time.time() - s_time)
        return

    async def handle_chunks_vectorstore(self, chunks: list[DocumentSchema], file_id) -> None:
        
        # Логируем информацию о чанках
        table_chunks = [chunk for chunk in chunks if chunk.metadata.get("is_table_chunk", False)]
        text_chunks = [chunk for chunk in chunks if not chunk.metadata.get("is_table_chunk", False)]
        
        if table_chunks:
            print(f"Creating embeddings for {len(table_chunks)} table chunks (should be whole tables)")
        
        print(f"Getting embeddings for {len(chunks)} nodes")
        embeddings = await self.embedding.ainvoke(chunks)
        print("Adding embeddings to vector store")
        
        # Extract metadata from chunks
        metadatas = [chunk.metadata for chunk in chunks]
        
        await self.vector_store.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[str(t.doc_id) for t in chunks],
        )
        
        nodes = []
        for chunk in chunks:
            nodes.append(
                Index(
                    source_id=file_id,
                    target_id=str(chunk.doc_id),
                    relation_type=IndexType.VECTOR
                )
            )
        await self.indexes_repository.bulk_create(nodes)

    async def handle_chunks_docstore(self, chunks, file_id) -> None:
        
        # Логируем информацию о чанках
        table_chunks = [chunk for chunk in chunks if chunk.metadata.get("is_table_chunk", False)]
        text_chunks = [chunk for chunk in chunks if not chunk.metadata.get("is_table_chunk", False)]
        
        if table_chunks:
            print(f"Storing {len(table_chunks)} table chunks (should be whole tables)")
        
        self.docs_store.add(chunks)
        
        nodes = []
        for chunk in chunks:
            nodes.append(
                Index(
                    source_id=file_id,
                    target_id=str(chunk.doc_id),
                    relation_type=IndexType.DOCUMENT
                )
            )
        await self.indexes_repository.bulk_create(nodes)
        
        
    async def get_id_if_exists(self, file_path: str | Path) -> Optional[str]:
        
        file_name = file_path.name if isinstance(file_path, Path) else file_path

        existing = await self.sources_repository.get_one(
            where=[Source.name == file_name]
        )
        return existing.id if existing else None
    
    
    async def store_file(self, file_path: str | Path) -> str:
        
        with file_path.open("rb") as fi:
            file_hash = sha256(fi.read()).hexdigest()

        shutil.copy(file_path, self.storage_dir / file_hash)
        source = Source(
            name=file_path.name,
            path=file_hash,
            size=file_path.stat().st_size,
        )
        
        await self.sources_repository.create(source)
        
        return source.id
    
    
 
 
    async def execute(self, file: UploadFile) -> Generator[DocumentSchema, None, int]:
        # 1. Читаем файл в память
        file_bytes = await file.read()

        # Проверка на пустой файл
        if not file_bytes or len(file_bytes) == 0:
            raise AppError(status_code=400, message="Файл пустой")

        # 4. Сохраняем файл на диск без изменения названия
        file_path = self.storage_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(file_bytes)
            
        yield DocumentSchema(f" => Saved file {file.filename} to disk", channel="debug")
        
        # Convert stream to async iteration
        async for doc_schema in self.stream(file_path):
            yield doc_schema
            
            
    def load_data(
        self, file_path: str | Path, extra_info: Optional[dict] = None, **kwargs
    ) -> List[DocumentSchema]:
        """Extract the input file, allowing multi-modal extraction"""

        metadata = extra_info or {}

        result = self.converter.convert(file_path)
        result_dict = result.document.export_to_dict()

        file_path = Path(file_path)
        file_name = file_path.name

        # extract the figures
        figures = []
       

        # extract the tables
        tables = []
        for i, table_obj in enumerate(result_dict.get("tables", [])):
            # convert the tables into markdown format
            markdown_table = self._parse_table(table_obj)
            caption_refs = [caption["$ref"] for caption in table_obj["captions"]]

            extractive_captions = []
            for caption_ref in caption_refs:
                text_id = caption_ref.split("/")[-1]
                try:
                    caption_text = result_dict["texts"][int(text_id)]["text"]
                    extractive_captions.append(caption_text)
                except (ValueError, TypeError, IndexError) as e:
                    print(e)
                    continue
            # join the extractive and generative captions
            caption = "\n".join(extractive_captions)
            markdown_table = f"{caption}\n{markdown_table}"

            page_number = table_obj["prov"][0].get("page_no", 1)

            table_metadata = {
                "type": "table",
                "page_label": page_number,
                "table_origin": markdown_table,
                "file_name": file_name,
                "file_path": file_path,
                "table_index": i,
                "total_tables": len(result_dict.get("tables", []))
            }
            table_metadata.update(metadata)

            table_doc = DocumentSchema(
                text=markdown_table,
                metadata=table_metadata,
            )
            tables.append(table_doc)
            
            # Отладочная информация
            print(f"Created table {i+1} with type={table_doc.metadata.get('type')} on page {page_number}")

        # join plain text elements
        texts = []
        page_number_to_text = defaultdict(list)

        for text_obj in result_dict["texts"]:
            page_number = text_obj["prov"][0].get("page_no", 1)
            page_number_to_text[page_number].append(text_obj["text"])

        for page_number, txts in page_number_to_text.items():
            texts.append(
                DocumentSchema(
                    text="\n".join(txts),
                    metadata={
                        "page_label": page_number,
                        "file_name": file_name,
                        "file_path": file_path,
                        **metadata,
                    },
                )
            )

        return texts + tables + figures
    
    
    def _convert_bbox_bl_tl(
        self, bbox: list[float], page_width: int, page_height: int
    ) -> list[float]:
        """Convert bbox from bottom-left to top-left"""
        x0, y0, x1, y1 = bbox
        return [
            x0 / page_width,
            (page_height - y1) / page_height,
            x1 / page_width,
            (page_height - y0) / page_height,
        ]

    def _parse_table(self, table_obj: dict) -> str:
        """Convert docling table object to markdown table"""
        table_as_list: List[List[str]] = []
        grid = table_obj["data"]["grid"]
        for row in grid:
            table_as_list.append([])
            for cell in row:
                table_as_list[-1].append(cell["text"])

        return make_markdown_table(table_as_list)
