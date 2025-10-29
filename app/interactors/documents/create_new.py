"""
Optimized Document Creation Interactor based on Kotaemon architecture.

This implementation follows the Kotaemon VectorIndexing approach:
- Docling for document parsing (texts, tables, figures)
- TokenSplitter for text chunking
- OpenAI embeddings
- Parallel addition to Qdrant (vector) and LanceDB (document store)
"""
import asyncio
import hashlib
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any
from uuid import UUID

from docling.document_converter import DocumentConverter
from fastapi import UploadFile
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.readers.file.base import default_file_metadata_func

from app.core.config import settings
from app.dto.schemas import DocumentSchema
from app.entities.sources import Source
from app.entities.indexes import Index
from app.repositories.indexes import IndexesRepository
from app.repositories.sources import SourcesRepository
from app.repositories.uow import UnitOfWork
from app.utils.docs_store import LanceDBDocumentStore
from app.utils.vectors_store import QdrantVectorStore
from app.utils.embeddings import OpenAIEmbeddings
from app.utils.collections import Collections
from app.utils.enums import IndexType
from app.exceptions.app_error import AppError
import logging

logger = logging.getLogger(__name__)


def make_markdown_table(table_as_list: List[List[str]]) -> str:
    """
    Convert table from python list representation to markdown format.
    The input list consists of rows of tables, the first row is the header.

    Args:
        table_as_list: list of table rows
            Example: [["Name", "Age", "Height"],
                    ["Jake", 20, 5'10],
                    ["Amy", 25, 5'8]]

    Returns:
        markdown-formatted table string
    """
    if not table_as_list:
        return ""

    markdown_table = ""
    # Header row
    markdown_table += "| " + " | ".join(table_as_list[0]) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(table_as_list[0])) + " |\n"

    # Data rows
    for row in table_as_list[1:]:
        markdown_table += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    return markdown_table


class CreateOptimizedDocumentInteractor:
    """
    Optimized document creation following Kotaemon VectorIndexing pattern.
    
    Architecture:
    1. Parse file with Docling (extracts texts, tables, figures)
    2. Split text using TokenSplitter
    3. Generate embeddings with OpenAI
    4. Store in Qdrant (vector) and LanceDB (document store)
    """
    
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
        
        # Kotaemon-style components
        self.collection_name = Collections.DOCUMENT_EMBEDDINGS
        self.doc_store = LanceDBDocumentStore()
        self.embedding = OpenAIEmbeddings(
            model_name="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY
        )
        self.vector_store = QdrantVectorStore(
            collection_name=self.collection_name,
            vector_size=3072
        )
        
        # TokenSplitter configuration (Kotaemon style)
        self.text_splitter = TokenTextSplitter(
            chunk_size=1024,
            chunk_overlap=256,
            separator="\n\n",
            backup_separators=["\n", ".", " ", "\u200B"]
        )
        
        self.chunk_batch_size = 200

    def load_data(
        self,
        file_path: str | Path,
        extra_info: Optional[dict] = None,
        **kwargs
    ) -> List[DocumentSchema]:
        """
        Extract the input file using Docling, following Kotaemon DoclingReader pattern.
        
        Returns:
            List of DocumentSchema with texts, tables, and figures
        """
        metadata = extra_info or {}
        file_path = Path(file_path)
        file_name = file_path.name

        # Convert with Docling
        result = self.converter.convert(file_path)
        result_dict = result.document.export_to_dict()

        # Extract tables
        tables = []
        for table_obj in result_dict.get("tables", []):
            # Convert table to markdown
            table_as_list: List[List[str]] = []
            grid = table_obj["data"]["grid"]
            for row in grid:
                table_as_list.append([])
                for cell in row:
                    table_as_list[-1].append(cell.get("text", ""))

            markdown_table = make_markdown_table(table_as_list)

            # Get captions
            caption_refs = [caption["$ref"] for caption in table_obj.get("captions", [])]
            extractive_captions = []
            for caption_ref in caption_refs:
                text_id = caption_ref.split("/")[-1]
                try:
                    caption_text = result_dict["texts"][int(text_id)]["text"]
                    extractive_captions.append(caption_text)
                except (ValueError, TypeError, IndexError):
                    continue

            caption = "\n".join(extractive_captions)
            markdown_table = f"{caption}\n{markdown_table}" if caption else markdown_table

            page_number = table_obj["prov"][0].get("page_no", 1)

            table_metadata = {
                "type": "table",
                "page_label": page_number,
                "file_name": file_name,
                "file_path": str(file_path),
                **metadata
            }

            tables.append(
                DocumentSchema(
                    text=markdown_table,
                    metadata=table_metadata
                )
            )

        # Extract texts (group by page)
        texts = []
        page_number_to_text = defaultdict(list)

        for text_obj in result_dict.get("texts", []):
            page_number = text_obj["prov"][0].get("page_no", 1)
            page_number_to_text[page_number].append(text_obj["text"])

        for page_number, txts in page_number_to_text.items():
            texts.append(
                DocumentSchema(
                    text="\n".join(txts),
                    metadata={
                        "page_label": page_number,
                        "file_name": file_name,
                        "file_path": str(file_path),
                        **metadata
                    }
                )
            )

        # Note: Figures extraction can be added here if needed
        # For now, we focus on texts and tables like Kotaemon base implementation

        return texts + tables

    async def stream(self, file_path: str | Path) -> AsyncGenerator[DocumentSchema, None]:
        """
        Stream document processing following Kotaemon VectorIndexing pattern.
        
        Steps:
        1. Check if file already exists
        2. Store file and create Source
        3. Parse with Docling
        4. Split text chunks
        5. Add to vector store and doc store
        """
        if isinstance(file_path, Path):
            file_path = file_path.resolve()

        # Check if already indexed
        file_id = await self.get_id_if_exists(file_path)
        if file_id:
            raise AppError(status_code=400, message=f"File {file_path.name} already indexed")

        # Store file
        file_id = await self.store_file(file_path)
        file_name = file_path.name

        extra_info = default_file_metadata_func(str(file_path))
        extra_info["file_id"] = str(file_id)
        extra_info["collection_name"] = self.collection_name

        yield DocumentSchema(
            content=f"🔍 Converting {file_name} to structured format...",
            channel="debug"
        )

        # Load data with Docling
        docs = self.load_data(file_path, extra_info)

        # Count document types
        text_count = sum(1 for doc in docs if doc.metadata.get("type") != "table")
        table_count = sum(1 for doc in docs if doc.metadata.get("type") == "table")

        yield DocumentSchema(
            content=f"✅ Converted: {text_count} text blocks, {table_count} tables",
            channel="debug"
        )

        # Process documents
        async for doc_schema in self.handle_docs(docs, file_id, file_name):
            yield doc_schema

        await self.uow.commit()

        yield DocumentSchema(
            content=f"✅ Finished indexing {file_name}",
            channel="debug"
        )

    async def handle_docs(
        self,
        docs: List[DocumentSchema],
        file_id: str,
        file_name: str
    ) -> AsyncGenerator[DocumentSchema, None]:
        """
        Process documents following Kotaemon pattern:
        - Separate text and table documents
        - Split text documents
        - Keep tables intact
        - Add to both vector store and doc store
        """
        start_time = time.time()

        # Separate by type
        text_docs = [doc for doc in docs if doc.metadata.get("type") != "table"]
        table_docs = [doc for doc in docs if doc.metadata.get("type") == "table"]

        yield DocumentSchema(
            content=f"📄 Processing {len(text_docs)} text blocks and {len(table_docs)} tables...",
            channel="debug"
        )

        # Split text documents (following Kotaemon pattern)
        chunks = []
        if self.text_splitter and text_docs:
            # Split each text document individually
            for doc in text_docs:
                doc_text = doc.text or ""
                if not doc_text:
                    continue
                
                # Split text into chunks
                text_chunks = self.text_splitter.split_text(doc_text)
                
                # Create DocumentSchema for each chunk
                for i, chunk_text in enumerate(text_chunks):
                    chunks.append(
                        DocumentSchema(
                            text=chunk_text,
                            metadata={
                                **doc.metadata,
                                "file_id": str(file_id),
                                "source_doc_id": str(file_id),
                                "chunk_index": i,
                                "total_chunks": len(text_chunks)
                            }
                        )
                    )
        else:
            # No splitter, use documents as-is
            chunks = text_docs

        # Keep tables intact (don't split)
        for table_doc in table_docs:
            table_doc.metadata.update({
                "file_id": str(file_id),
                "source_doc_id": str(file_id),
                "is_table_chunk": True
            })
            chunks.append(table_doc)

        yield DocumentSchema(
            content=f"✂️ Split into {len(chunks)} chunks (texts + tables)",
            channel="debug"
        )

        # Add file_id to all chunks
        for chunk in chunks:
            if "file_id" not in chunk.metadata:
                chunk.metadata["file_id"] = str(file_id)
                chunk.metadata["source_doc_id"] = str(file_id)

        # Process in batches
        total_chunks = len(chunks)
        processed = 0

        for start_idx in range(0, total_chunks, self.chunk_batch_size):
            batch = chunks[start_idx:start_idx + self.chunk_batch_size]
            
            # Add to vector store and doc store (Kotaemon pattern)
            await self.add_to_vectorstore(batch, file_id)
            await self.add_to_docstore(batch, file_id)
            
            processed += len(batch)
            yield DocumentSchema(
                content=f"💾 Indexed {processed}/{total_chunks} chunks...",
                channel="debug"
            )

        elapsed = time.time() - start_time
        yield DocumentSchema(
            content=f"✅ Indexing completed in {elapsed:.2f}s",
            channel="debug"
        )

    async def add_to_vectorstore(self, docs: List[DocumentSchema], file_id: str) -> None:
        """
        Add documents to vector store following Kotaemon VectorIndexing pattern.
        """
        if not docs:
            return

        logger.info(f"Getting embeddings for {len(docs)} documents")
        
        # Generate embeddings (returns list[DocumentWithEmbedding])
        embeddings_result = await self.embedding.ainvoke(docs)
        
        if not embeddings_result:
            logger.warning("No embeddings generated")
            return
        
        # Map embeddings to documents
        # Note: embeddings_result[i] corresponds to docs[i]
        ids = [str(doc.doc_id) for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        logger.info(f"Adding {len(embeddings_result)} embeddings to vector store")
        
        # Add to Qdrant (vector_store.add accepts list[DocumentWithEmbedding])
        await self.vector_store.add(
            embeddings=embeddings_result,
            metadatas=metadatas,
            ids=ids
        )

        # Create index entries for vectors
        vector_indexes = []
        for doc_id in ids:
            vector_indexes.append(
                Index(
                    source_id=file_id,
                    target_id=doc_id,
                    relation_type=IndexType.VECTOR
                )
            )
        if vector_indexes:
            await self.indexes_repository.bulk_create(vector_indexes)

    async def add_to_docstore(self, docs: List[DocumentSchema], file_id: str) -> None:
        """
        Add documents to document store following Kotaemon VectorIndexing pattern.
        """
        if not docs:
            return

        logger.info(f"Adding {len(docs)} documents to doc store")
        
        # Add to LanceDB
        self.doc_store.add(docs)

        # Create index entries for documents
        doc_indexes = [
            Index(
                source_id=file_id,
                target_id=str(doc.doc_id),
                relation_type=IndexType.DOCUMENT
            )
            for doc in docs
        ]
        await self.indexes_repository.bulk_create(doc_indexes)

    async def get_id_if_exists(self, file_path: str | Path) -> Optional[str]:
        """Check if file already exists in database"""
        file_name = file_path.name if isinstance(file_path, Path) else Path(file_path).name

        existing = await self.sources_repository.get_one(
            where=[Source.name == file_name]
        )
        return existing.id if existing else None

    async def store_file(self, file_path: str | Path) -> str:
        """Store file and create Source entry"""
        file_path = Path(file_path)
        
        # Calculate file hash
        with file_path.open("rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Copy to storage
        stored_path = self.storage_dir / file_hash
        shutil.copy(file_path, stored_path)

        # Create source
        source = Source(
            name=file_path.name,
            path=file_hash,
            size=file_path.stat().st_size,
        )
        await self.sources_repository.create(source)

        return source.id

    async def execute(self, file: UploadFile) -> AsyncGenerator[DocumentSchema, None]:
        """Execute document creation from uploaded file"""
        # Read file
        file_bytes = await file.read()

        if not file_bytes or len(file_bytes) == 0:
            raise AppError(status_code=400, message="File is empty")

        # Save to disk
        file_path = self.storage_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        yield DocumentSchema(
            content=f"💾 Saved file {file.filename}",
            channel="debug"
        )

        # Process file
        async for doc_schema in self.stream(file_path):
            yield doc_schema

