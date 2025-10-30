from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from app.utils.vectors_store import QdrantVectorStore
from app.utils.docs_store import LanceDBDocumentStore
from app.utils.embeddings import OpenAIEmbeddings
from app.utils.collections import Collections
from app.core.config import settings


@dataclass
class SearchResult:
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class AgentTools:
    """Toolbox exposed to the LLM agent via tool-calling.

    Methods here should be lightweight, side-effect free, and return JSON-serializable data.
    """

    def __init__(self) -> None:
        self.vector_store = QdrantVectorStore(
            collection_name=Collections.DOCUMENT_EMBEDDINGS,
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            vector_size=3072,
        )
        self.doc_store = LanceDBDocumentStore()
        self.embedding = OpenAIEmbeddings(
            model_name="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY,
        )
        # Simple in-memory cache for search results to avoid repeated calls
        self._cache: Dict[Tuple[str, int, bool, bool, Optional[Tuple[str, ...]]], Dict[str, Any]] = {}
        self._cache_order: List[Tuple[str, int, bool, bool, Optional[Tuple[str, ...]]]] = []
        self._cache_cap = 50

    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        use_vector: bool = True,
        use_keyword: bool = True,
        file_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Hybrid search without LLM reranking.

        Args:
            query: user query
            limit: max results
            use_vector: include semantic search
            use_keyword: include FTS
            file_ids: optional filter by source file ids
        Returns:
            Dict with items: list of {doc_id, content, score, metadata}
        """
        # Normalize args for cache key
        key = (query.strip(), int(limit), bool(use_vector), bool(use_keyword), tuple(file_ids) if file_ids else None)
        if key in self._cache:
            return self._cache[key]

        vector_results: List[Tuple[str, float]] = []
        if use_vector:
            emb_docs = await self.embedding.ainvoke(query)
            if emb_docs and getattr(emb_docs[0], "embedding", None):
                embedding = emb_docs[0].embedding
                _, scores, ids = await self.vector_store.query(
                    embedding=embedding,
                    top_k=limit,
                    ids=file_ids,
                )
                vector_results = list(zip(ids or [], scores or []))

        keyword_docs = []
        if use_keyword:
            keyword_docs = (
                self.doc_store.query(query, top_k=limit, doc_ids=file_ids) if file_ids else self.doc_store.query(query, top_k=limit)
            )

        # Merge results by doc_id using max score
        merged: Dict[str, float] = {}
        for did, s in vector_results:
            merged[did] = max(merged.get(did, 0.0), float(s))
        for i, d in enumerate(keyword_docs):
            did = getattr(d, "doc_id", None) or getattr(d, "id_", None)
            if did:
                # keyword score as inverse rank
                kw = max(0.5, 1.0 - (i / max(1, len(keyword_docs)) * 0.5))
                merged[did] = max(merged.get(did, 0.0), kw)

        # Fetch full docs and return
        items: List[Dict[str, Any]] = []
        if merged:
            all_docs = self.doc_store.get(list(merged.keys()))
            by_id = {}
            for d in all_docs:
                did = getattr(d, "doc_id", None) or getattr(d, "id_", None)
                if did:
                    by_id[did] = d
            for did, score in sorted(merged.items(), key=lambda x: x[1], reverse=True)[:limit]:
                d = by_id.get(did)
                if not d:
                    continue
                raw_content = getattr(d, "text", None) or getattr(d, "content", None) or ""
                # Truncate content to reduce tokens fed back to the LLM
                content = raw_content[:400]
                metadata = getattr(d, "metadata", None) or {}
                items.append({"doc_id": did, "content": content, "score": float(score), "metadata": metadata})

        result = {"items": items}
        # Maintain simple LRU cache
        self._cache[key] = result
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_cap:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return result

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Return a single chunk/document by doc_id."""
        docs = self.doc_store.get([doc_id])
        if not docs:
            return {"item": None}
        d = docs[0]
        return {
            "item": {
                "doc_id": doc_id,
                "content": getattr(d, "text", None) or getattr(d, "content", None) or "",
                "metadata": getattr(d, "metadata", None) or {},
            }
        }


