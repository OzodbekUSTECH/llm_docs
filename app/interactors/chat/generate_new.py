"""
Optimized Answer Generation Interactor based on Kotaemon architecture.

This implementation follows the Kotaemon approach:
- Hybrid search (parallel vector + keyword search)
- LLM-based reranking with binary relevance (YES/NO)
- Citation extraction pipeline
- All using OpenAI models
"""
import asyncio
import time
import uuid
import json
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.dto.chat import GenerateAnswerRequest, GeneratedAnswerResponse, Source
from app.dto.schemas import DocumentSchema, RetrievedDocument
from app.services.chat_storage import chat_storage
from app.utils.vectors_store import QdrantVectorStore
from app.utils.embeddings import OpenAIEmbeddings
from app.utils.docs_store import LanceDBDocumentStore
from app.utils.collections import Collections
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CiteEvidence:
    """Evidence citation structure"""
    evidences: List[str]


class LLMReranker:
    """
    LLM-based reranking following Kotaemon approach.
    Uses binary relevance classification (YES/NO) instead of scoring.
    """
    
    RERANK_PROMPT_TEMPLATE = """Given the following question and context,
return YES if the context is relevant to the question and NO if it isn't.

> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant (YES / NO):"""

    def __init__(self, openai_client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model
        self.top_k: int = 10
        self.concurrent: bool = True
        self.max_concurrency: int = 4
        self.max_retries: int = 3
        self.retry_backoff_s: float = 0.8
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    async def rerank(
        self,
        documents: List[RetrievedDocument],
        query: str,
        top_k: Optional[int] = None,
        return_all: bool = False,
    ) -> List[RetrievedDocument]:
        """Filter down documents based on their relevance to the query."""
        if top_k is None:
            top_k = self.top_k

        if not documents:
            return []

        # Limit documents for reranking (LLM calls can be expensive)
        # But be less aggressive - rerank more documents
        docs_to_rerank = documents[:min(len(documents), max(top_k * 4, 20))]
        
        logger.info(f"🔄 LLM Reranking: evaluating {len(docs_to_rerank)} documents...")

        results: List[bool] = []
        try:
            if self.concurrent:
                async def worker(doc: RetrievedDocument) -> bool:
                    prompt = self.RERANK_PROMPT_TEMPLATE.format(
                        question=query,
                        context=(doc.content or "")[:700],
                    )
                    return await self._check_relevance(prompt)

                # Use bounded concurrency
                async def run_with_semaphore(doc: RetrievedDocument) -> bool:
                    async with self._semaphore:
                        return await worker(doc)

                tasks = [run_with_semaphore(doc) for doc in docs_to_rerank]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Normalize exceptions to True (keep rather than drop) to avoid over-filtering on failures
                results = [False if isinstance(r, Exception) else bool(r) for r in results]
            else:
                for doc in docs_to_rerank:
                    prompt = self.RERANK_PROMPT_TEMPLATE.format(
                        question=query,
                        context=(doc.content or "")[:700],
                    )
                    res = await self._check_relevance(prompt)
                    results.append(res)
        except Exception as e:
            logger.warning(f"LLM reranker failed, using heuristic fallback: {e}")
            results = [((d.semantic_score or 0) * 0.7 + (d.keyword_score or 0) * 0.3) >= 0.3 for d in docs_to_rerank]

        # Filter relevant documents
        filtered_docs = [
            doc for doc, is_relevant in zip(docs_to_rerank, results)
            if is_relevant
        ]

        # If reranking filtered out too many, be less strict
        # Return at least top_k documents, prioritize reranked but add high-scoring originals
        if len(filtered_docs) < top_k:
            # Add back high-scoring documents that were filtered out
            filtered_ids = {doc.doc_id for doc in filtered_docs}
            remaining = [doc for doc in documents if doc.doc_id not in filtered_ids]
            
            # Sort remaining by score and add top ones
            remaining.sort(key=lambda x: x.score, reverse=True)
            needed = top_k - len(filtered_docs)
            filtered_docs.extend(remaining[:needed])
            
            logger.info(f"📊 Reranking filtered {len(docs_to_rerank) - len([d for d in docs_to_rerank if d.doc_id in filtered_ids])} docs, "
                       f"added {needed} high-scoring docs to reach {top_k}")

        logger.info(f"✅ LLM Reranking: {len(filtered_docs)} relevant documents (from {len(docs_to_rerank)})")
        return filtered_docs if return_all else filtered_docs[:top_k]

    async def _check_relevance(self, prompt: str) -> bool:
        """Check if document is relevant using LLM"""
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a relevance classifier. Respond with ONLY YES or NO."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=5,
                )
                result_text = (response.choices[0].message.content or "").strip().upper()
                return "YES" in result_text or result_text.startswith("Y")
            except Exception as e:
                msg = str(e)
                is_429 = "429" in msg or "rate limit" in msg.lower()
                if attempt >= self.max_retries or not is_429:
                    logger.error(f"❌ Error checking relevance: {e}")
                    # Fail open to avoid losing recall
                    return True
                # Exponential backoff with jitter
                backoff = self.retry_backoff_s * (2 ** (attempt - 1))
                jitter = min(1.0, backoff * 0.25)
                await asyncio.sleep(backoff + (jitter))


class CitationPipeline:
    """
    Citation pipeline following Kotaemon approach.
    Extracts evidences using OpenAI function calling.
    """
    
    def __init__(self, openai_client: AsyncOpenAI, model: str = "gpt-4o"):
        self.client = openai_client
        self.model = model

    async def extract_evidences(
        self,
        context: str,
        question: str
    ) -> Optional[CiteEvidence]:
        """Extract cited evidences from context using function calling"""
        try:
            # Prepare function schema
            function_schema = {
                "name": "CiteEvidence",
                "description": "List of evidences (maximum 5) to support the answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "evidences": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Each source should be a direct quote from the context, "
                                "as a substring of the original content (max 15 words)."
                            )
                        }
                    },
                    "required": ["evidences"]
                }
            }

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a world class algorithm to answer "
                        "questions with correct and exact citations."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Answer question using the following context. "
                        "Use the provided function CiteEvidence() to cite your sources."
                    )
                },
                {"role": "user", "content": context[:4000]},  # Limit context size
                {"role": "user", "content": f"Question: {question}"},
                {
                    "role": "user",
                    "content": (
                        "Tips: Make sure to cite your sources, "
                        "and use the exact words from the context."
                    )
                }
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[{"type": "function", "function": function_schema}],
                tool_choice="required",
                temperature=0.1,
                max_tokens=200
            )

            # Extract function call result
            message = response.choices[0].message
            if message.tool_calls:
                function_call = message.tool_calls[0].function
                arguments = json.loads(function_call.arguments)
                return CiteEvidence(**arguments)

            return None
        except Exception as e:
            logger.error(f"❌ Error extracting citations: {e}")
            return None


class GenerateOptimizedAnswerInteractor:
    """
    Optimized answer generation based on Kotaemon architecture.
    
    Key features:
    - Parallel hybrid search (vector + keyword)
    - LLM-based reranking
    - Citation extraction
    - OpenAI-only implementation
    """

    def __init__(self):
        """Initialize the interactor with necessary components"""
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.vector_store = QdrantVectorStore(
            collection_name=Collections.DOCUMENT_EMBEDDINGS,
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            vector_size=3072,  # text-embedding-3-large dimension
        )
        self.doc_store = LanceDBDocumentStore()
        self.embedding = OpenAIEmbeddings(
            model_name="text-embedding-3-large",
            api_key=settings.OPENAI_API_KEY
        )
        self.reranker = LLMReranker(self.openai_client, model="gpt-4o")
        self.citation_pipeline = CitationPipeline(self.openai_client, model="gpt-4o")
        
        # Configuration
        self.model = "gpt-4o"
        self.max_tokens = 2000
        self.temperature = 0.3
        self.top_k: int = 10
        self.first_round_top_k_mult: int = 10  # Retrieve 10x initially, then rerank
        self.max_total_retrieved: int = 30
        self.num_reformulations: int = 3

    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        document_ids: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """
        Parallel hybrid search following Kotaemon approach.
        Executes vector and keyword searches simultaneously.
        """
        top_k_first_round = top_k * self.first_round_top_k_mult
        
        logger.info(f"🔍 Starting parallel hybrid search (top_k_first_round={top_k_first_round})...")
        
        # Parallel search execution
        vs_docs: List[RetrievedDocument] = []
        vs_scores: List[float] = []
        vs_ids: List[str] = []
        
        ds_docs: List[RetrievedDocument] = []

        async def query_vectorstore():
            """Vector search using Qdrant"""
            nonlocal vs_docs, vs_scores, vs_ids
            try:
                logger.info("🔍 Step 1: Vector search (Qdrant)...")
                query_embedding_docs = await self.embedding.ainvoke(query)
                if query_embedding_docs and getattr(query_embedding_docs[0], "embedding", None):
                    query_embedding = query_embedding_docs[0].embedding
                    _, scores, doc_ids = await self.vector_store.query(
                        embedding=query_embedding,
                        top_k=top_k_first_round,
                        ids=document_ids,
                    )
                    
                    if doc_ids:
                        # Get full documents from LanceDB
                        all_docs = self.doc_store.get(doc_ids)
                        # Create mapping by doc_id (can be id_ or doc_id attribute)
                        doc_dict = {}
                        for doc in all_docs:
                            doc_id_value = getattr(doc, "doc_id", None) or getattr(doc, "id_", None)
                            if doc_id_value:
                                doc_dict[doc_id_value] = doc
                        
                        vs_docs = []
                        vs_scores = []
                        vs_ids = []
                        
                        for doc_id, score in zip(doc_ids, scores):
                            if doc_id in doc_dict:
                                doc = doc_dict[doc_id]
                                doc_content = getattr(doc, "text", None) or getattr(doc, "content", None) or ""
                                doc_metadata = getattr(doc, "metadata", None) or {}
                                doc_docid = getattr(doc, "doc_id", None) or getattr(doc, "id_", None)
                                
                                retrieved_doc = RetrievedDocument(
                                    content=doc_content,
                                    metadata=doc_metadata,
                                    doc_id=doc_docid,
                                    score=float(score),
                                    semantic_score=float(score),
                                    keyword_score=0.0,
                                )
                                vs_docs.append(retrieved_doc)
                                vs_scores.append(float(score))
                                vs_ids.append(doc_id)
                        
                        logger.info(f"✅ Vector search: found {len(vs_docs)} documents")
            except Exception as e:
                logger.error(f"❌ Error in vector search: {e}")

        async def query_docstore():
            """Keyword search using LanceDB FTS"""
            nonlocal ds_docs
            try:
                logger.info("🔤 Step 2: Keyword search (LanceDB FTS)...")
                if document_ids:
                    # Filter by document IDs if provided
                    fts_results = self.doc_store.query(
                        query, top_k=top_k_first_round, doc_ids=document_ids
                    )
                else:
                    fts_results = self.doc_store.query(query, top_k=top_k_first_round)
                
                ds_docs = []
                for i, doc in enumerate(fts_results):
                    doc_content = getattr(doc, "text", None) or getattr(doc, "content", None) or ""
                    doc_metadata = getattr(doc, "metadata", None) or {}
                    doc_docid = getattr(doc, "doc_id", None) or getattr(doc, "id_", None)
                    
                    # Better keyword scoring: inverse rank with decay
                    # Top result gets highest score, with exponential decay
                    max_results = min(len(fts_results), top_k_first_round)
                    keyword_score = max(0.5, 1.0 - (i / max_results * 0.5))  # Range from 1.0 to 0.5
                    
                    retrieved_doc = RetrievedDocument(
                        content=doc_content,
                        metadata=doc_metadata,
                        doc_id=doc_docid,
                        score=keyword_score,  # Use keyword score as initial score
                        semantic_score=0.0,
                        keyword_score=keyword_score,
                    )
                    ds_docs.append(retrieved_doc)
                
                logger.info(f"✅ Keyword search: found {len(ds_docs)} documents")
            except Exception as e:
                logger.error(f"❌ Error in keyword search: {e}")

        # Execute both searches in parallel
        await asyncio.gather(
            query_vectorstore(),
            query_docstore()
        )

        # Combine results (following Kotaemon approach with proper score combination)
        # Create maps for efficient lookup
        vs_docs_dict = {doc.doc_id: doc for doc in vs_docs}
        ds_docs_dict = {doc.doc_id: doc for doc in ds_docs}
        
        # Combined result dictionary
        combined_docs: Dict[str, RetrievedDocument] = {}
        
        # Process vector search results
        for doc in vs_docs:
            doc_id = doc.doc_id
            if doc_id not in combined_docs:
                combined_docs[doc_id] = doc
        
        # Process keyword search results and combine scores if document exists in both
        for doc in ds_docs:
            doc_id = doc.doc_id
            if doc_id in combined_docs:
                # Document exists in both - combine scores (weighted hybrid)
                existing = combined_docs[doc_id]
                semantic_weight = 0.7
                keyword_weight = 0.3
                combined_score = (semantic_weight * existing.semantic_score + 
                                keyword_weight * doc.keyword_score)
                
                # Update with combined scores
                combined_docs[doc_id] = RetrievedDocument(
                    content=existing.content,
                    metadata=existing.metadata,
                    doc_id=existing.doc_id,
                    score=combined_score,
                    semantic_score=existing.semantic_score,
                    keyword_score=doc.keyword_score,
                )
            else:
                # Keyword-only document
                combined_docs[doc_id] = doc

        result = list(combined_docs.values())

        logger.info(f"✅ Hybrid search: {len(vs_docs)} from vector, {len(ds_docs)} from keyword, {len(result)} total (with score combination)")

        # Sort by combined score
        result.sort(key=lambda x: x.score, reverse=True)
        
        return result

    async def stream(
        self,
        message: str,
        conv_id: str,
        history: Optional[List[Dict]] = None,
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[DocumentSchema, None]:
        """
        Stream response with optimized retrieval and answer generation.
        """
        start_time = time.time()
        history = history or []

        try:
            yield DocumentSchema(
                content="🔍 Starting optimized hybrid search (Qdrant + LanceDB FTS)...",
                channel="debug"
            )

            # Step 1: Hybrid search (parallel)
            retrieved_docs = await self._hybrid_search(
                query=message,
                top_k=top_k,
                document_ids=document_ids
            )

            if not retrieved_docs:
                yield DocumentSchema(
                    content="⚠️ No relevant documents found",
                    channel="debug"
                )
                yield DocumentSchema(
                    content="I couldn't find any relevant information in the available documents. Could you please rephrase your question?",
                    channel="chat"
                )
                return

            # Show initial retrieval stats
            avg_semantic_initial = sum(doc.semantic_score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0
            avg_keyword_initial = sum(doc.keyword_score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0
            avg_final_initial = sum(doc.score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0
            
            yield DocumentSchema(
                content=f"✅ Found {len(retrieved_docs)} documents | Avg score: {avg_final_initial:.3f} | Semantic: {avg_semantic_initial:.3f} | Keyword: {avg_keyword_initial:.3f} | 🔄 Reranking...",
                channel="debug"
            )

            reranked_docs = await self.reranker.rerank(
                documents=retrieved_docs,
                query=message,
                top_k=top_k,
                return_all=True,
            )

            # Step 2b: Iterative refinement – generate reformulations and search again
            merged_docs: List[RetrievedDocument] = list(reranked_docs)
            seen_ids = {doc.doc_id for doc in merged_docs if doc.doc_id}

            reformulations = await self._plan_gap_reformulations(message, merged_docs)
            if reformulations:
                yield DocumentSchema(
                    content=f"🔁 Refining query with {len(reformulations)} reformulation(s) to cover gaps...",
                    channel="debug",
                )

            for i, reformulated in enumerate(reformulations, 1):
                if len(merged_docs) >= self.max_total_retrieved:
                    break
                try:
                    yield DocumentSchema(
                        content=f"🔎 Iteration {i}: searching for complementary evidence...",
                        channel="debug",
                    )
                    iter_docs = await self._hybrid_search(
                        query=reformulated,
                        top_k=top_k,
                        document_ids=document_ids,
                    )
                    iter_reranked = await self.reranker.rerank(
                        documents=iter_docs,
                        query=reformulated,
                        top_k=top_k,
                        return_all=True,
                    )
                    # Merge new docs by id, preserve order
                    for d in iter_reranked:
                        if d.doc_id and d.doc_id not in seen_ids:
                            merged_docs.append(d)
                            seen_ids.add(d.doc_id)
                    yield DocumentSchema(
                        content=f"➕ Added {len(seen_ids) - len({doc.doc_id for doc in reranked_docs if doc.doc_id})} new docs (total {len(merged_docs)})",
                        channel="debug",
                    )
                except Exception as e:
                    logger.warning(f"Refinement iteration failed: {e}")

            # Use merged docs as final retrieved set
            reranked_docs = merged_docs

            yield DocumentSchema(
                content=f"✅ Reranked to {len(reranked_docs)} documents",
                channel="debug"
            )

            # Calculate average scores for logging
            avg_semantic = sum(doc.semantic_score for doc in reranked_docs) / len(reranked_docs) if reranked_docs else 0.0
            avg_keyword = sum(doc.keyword_score for doc in reranked_docs) / len(reranked_docs) if reranked_docs else 0.0
            avg_final = sum(doc.score for doc in reranked_docs) / len(reranked_docs) if reranked_docs else 0.0
            
            yield DocumentSchema(
                content=f"✅ Found {len(reranked_docs)} documents | Avg score: {avg_final:.3f} | Semantic: {avg_semantic:.3f} | Keyword: {avg_keyword:.3f}",
                channel="debug"
            )
            
            # Show document information (send structured JSON with full content)
            for i, doc in enumerate(reranked_docs, 1):
                try:
                    import json as _json
                    file_name = (doc.metadata or {}).get("file_name", "Unknown")
                    page_label = (doc.metadata or {}).get("page_label", "N/A")
                    payload = {
                        "rank": i,
                        "file_name": file_name,
                        "page": page_label,
                        "semantic_score": float(doc.semantic_score or 0.0),
                        "keyword_score": float(doc.keyword_score or 0.0),
                        "content": doc.content or "",
                    }
                    yield DocumentSchema(
                        content=_json.dumps(payload),
                        channel="info",
                    )
                except Exception:
                    # Fallback to minimal text if JSON serialization fails
                    yield DocumentSchema(
                        content=f"Document {i} | {file_name} (p. {page_label})",
                        channel="info",
                    )

            # Step 3: Generate answer
            yield DocumentSchema(
                content="💬 Generating answer...",
                channel="debug"
            )

            # Format context
            context = "\n\n".join([
                f"[{i+1}]{doc.content}"
                for i, doc in enumerate(reranked_docs[: self.max_total_retrieved])
            ])

            # Format history
            formatted_history = []
            for msg in history[-5:]:
                if isinstance(msg, dict):
                    if "role" in msg and "content" in msg:
                        formatted_history.append({"role": msg["role"], "content": msg["content"]})
                    elif "user" in msg:
                        formatted_history.append({"role": "user", "content": msg["user"]})
                    elif "assistant" in msg:
                        formatted_history.append({"role": "assistant", "content": msg["assistant"]})

            system_prompt = """You are a helpful AI assistant with access to document knowledge base.

CRITICAL RULES:
1. **ONLY use information from the provided context** - Never invent or make up information
2. **Be precise and cite sources** - Reference specific documents using [1], [2] notation
3. **If information is not in context** - Clearly state "I don't have this information in the available documents"
4. **Be concise but complete** - Provide all relevant information from the context

RESPONSE FORMAT:
- Answer the question based ONLY on the retrieved documents
- Use [1], [2], etc. to cite specific sources
- If context doesn't contain the answer, say so clearly"""

            user_prompt = f"""Context documents:
{context}

Question: {message}

Please answer the question using ONLY the information from the context above. Cite sources using [1], [2] notation."""

            messages = [
                {"role": "system", "content": system_prompt},
                *formatted_history,
                {"role": "user", "content": user_prompt}
            ]

            full_response = ""
            stream_response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            async for chunk in stream_response:
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield DocumentSchema(
                        content=content,
                        channel="chat"
                    )

            # Step 4: Extract citations
            yield DocumentSchema(
                content="📝 Extracting citations...",
                channel="debug"
            )

            context_for_citation = "\n\n".join([
                f"Document {i+1}: {doc.content}"
                for i, doc in enumerate(reranked_docs[:5])  # Limit for citation extraction
            ])

            cite_evidence = await self.citation_pipeline.extract_evidences(
                context=context_for_citation,
                question=message
            )

            # Stream highlights to info panel by locating evidences in retrieved docs
            if cite_evidence and cite_evidence.evidences:
                evidences_by_doc: Dict[int, List[str]] = {}
                
                # Parse cited numbers from the final answer to prioritize mapping
                import re as _re
                cited_numbers: list[int] = []
                try:
                    cited_numbers = [int(n) for n in _re.findall(r"\[(\d+)\]", full_response)]
                except Exception:
                    cited_numbers = []
                
                for ev in cite_evidence.evidences:
                    ev_lower = (ev or "").lower()
                    matched_rank: Optional[int] = None
                    
                    # 1) Try match inside specifically cited docs first (keep order)
                    for n in cited_numbers:
                        if 1 <= n <= len(reranked_docs):
                            doc_text = (reranked_docs[n - 1].content or "")
                            lc = doc_text.lower()
                            if ev_lower and ev_lower[:10] in lc and ev_lower in lc:
                                matched_rank = n
                                break
                            if ev_lower and ev_lower in lc:
                                matched_rank = n
                                break
                    # 2) Otherwise, match against any doc
                    if matched_rank is None:
                        for rank, doc in enumerate(reranked_docs, start=1):
                            doc_text = (doc.content or "")
                            lc = doc_text.lower()
                            if ev_lower and ev_lower[:10] in lc and ev_lower in lc:
                                matched_rank = rank
                                break
                            if ev_lower and ev_lower in lc:
                                matched_rank = rank
                                break
                    
                    if matched_rank is not None:
                        evidences_by_doc.setdefault(matched_rank, []).append(ev)
                        try:
                            import json as _json
                            highlight_payload = {"doc_rank": matched_rank, "evidence": ev}
                            yield DocumentSchema(content=_json.dumps(highlight_payload), channel="highlight")
                        except Exception:
                            pass
            else:
                evidences_by_doc = {}

            # Step 5: Save to chat storage
            processing_time = time.time() - start_time
            
            yield DocumentSchema(
                content=f"⏱️ Processing time: {processing_time:.2f}s | Documents: {len(reranked_docs)} | Citations: {len(cite_evidence.evidences) if cite_evidence else 0}",
                channel="debug"
            )

            # Save chat messages
            chat_storage.add_message(
                chat_id=conv_id,
                role="user",
                content=message
            )

            retrieved_docs_data = []
            for i, doc in enumerate(reranked_docs, 1):
                doc_data = {
                    "rank": i,
                    "content": doc.content,
                    "score": doc.score,
                    "semantic_score": doc.semantic_score,
                    "keyword_score": doc.keyword_score,
                    "doc_id": doc.doc_id,
                    "metadata": doc.metadata or {}
                }
                retrieved_docs_data.append(doc_data)

            assistant_metadata = {
                "retrieved_docs": retrieved_docs_data,
                "evidences": cite_evidence.evidences if cite_evidence else [],
                "evidences_by_doc": evidences_by_doc,
                "processing_time": processing_time,
                "citations_count": len(cite_evidence.evidences) if cite_evidence else 0,
            }

            chat_storage.add_message(
                chat_id=conv_id,
                role="assistant",
                content=full_response,
                metadata=assistant_metadata
            )

            logger.info(f"💾 Saved chat with {len(retrieved_docs_data)} retrieved docs")

        except Exception as e:
            logger.error(f"❌ Error in stream: {e}", exc_info=True)
            yield DocumentSchema(
                content=f"❌ Error: {str(e)}",
                channel="debug"
            )
            yield DocumentSchema(
                content=f"I encountered an error while processing your request: {str(e)}",
                channel="chat"
            )

    async def _generate_reformulations(
        self,
        question: str,
        docs: List[RetrievedDocument],
        max_reformulations: Optional[int] = None,
    ) -> List[str]:
        """Use LLM to propose alternative phrasings to fill potential gaps.

        Returns up to N reformulations that are diverse and focused on missing details.
        """
        try:
            if max_reformulations is None:
                max_reformulations = self.num_reformulations

            context_preview = "\n\n".join([(d.content or "")[:400] for d in docs[:5]])
            prompt = (
                "You are refining a search query to cover information gaps.\n"
                "Given the original user question and some retrieved snippets, propose up to "
                f"{max_reformulations} diverse, short reformulations that could retrieve complementary evidence.\n"
                "- Keep each reformulation concise (<= 18 words).\n"
                "- Vary phrasing and include plausible synonyms or explicit entities/dates/units from context.\n"
                "- Output as a JSON array of strings only.\n\n"
                f"Question: {question}\n\nSnippets:\n{context_preview}\n\nJSON:"
            )
            resp = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            text = resp.choices[0].message.content or ""
            import json as _json
            candidates = []
            try:
                candidates = _json.loads(text)
            except Exception:
                # Fallback: split by newline or semicolon
                candidates = [s.strip(" -•\n") for s in text.split("\n") if s.strip()]

            # Normalize and cap
            unique: List[str] = []
            seen = set()
            for r in candidates:
                r_norm = (r or "").strip()
                if not r_norm:
                    continue
                if r_norm.lower() in seen:
                    continue
                seen.add(r_norm.lower())
                unique.append(r_norm)
                if len(unique) >= max_reformulations:
                    break
            return unique
        except Exception as e:
            logger.warning(f"Failed to generate reformulations: {e}")
            return []

    async def _plan_gap_reformulations(
        self,
        question: str,
        docs: List[RetrievedDocument],
    ) -> List[str]:
        """Detect gaps via lightweight heuristic, then optionally call LLM to propose targeted reformulations.

        Returns 0..N reformulations; empty list means no iteration needed.
        """
        # Heuristic: scan current docs for key field signals
        text = "\n".join([(d.content or "")[:800] for d in docs[:20]]).lower()
        signals = {
            "price": any(k in text for k in [" usd", " eur", "$", "eur ", "usd ", "price", "unit price", "/mt", "per mt", "per dmt"]),
            "payment": any(k in text for k in ["payment", "l/c", "letter of credit", "tt", "days", "upon presentation", "bank"]),
            "buyer": any(k in text for k in ["buyer:", "buyer ", "purchaser", "consignee"]),
            "seller": any(k in text for k in ["seller:", "seller ", "supplier"]),
            "commodity": any(k in text for k in ["commodity", "product", "goods", "corn", "ore", "ulsd", "diesel"]),
        }
        missing = [k for k, v in signals.items() if not v]
        if not missing:
            return []

        # Ask LLM for targeted reformulations for missing fields
        try:
            prompt = (
                "You are improving a retrieval query to cover missing fields in contract extraction.\n"
                f"Original question: {question}\n"
                f"Missing fields: {', '.join(missing)}\n"
                "Propose up to 3 short alternative queries that explicitly target these fields in contracts.\n"
                "Keep each <= 12 words. Return a JSON array of strings only."
            )
            resp = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=160,
            )
            import json as _json
            text_resp = resp.choices[0].message.content or "[]"
            reformulations = []
            try:
                reformulations = _json.loads(text_resp)
            except Exception:
                reformulations = [s.strip() for s in text_resp.split("\n") if s.strip()]
            # Limit and dedupe
            out: List[str] = []
            seen = set()
            for r in reformulations:
                r_norm = (r or "").strip()
                if not r_norm:
                    continue
                key = r_norm.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(r_norm)
                if len(out) >= min(self.num_reformulations, 3):
                    break
            return out
        except Exception as e:
            logger.warning(f"Gap planning failed, skipping iterations: {e}")
            return []

    async def execute(
        self,
        request: GenerateAnswerRequest,
        conv_id: str,
        history: Optional[List[Dict]] = None
    ) -> GeneratedAnswerResponse:
        """Execute non-streaming request (for compatibility)"""
        start_time = time.time()
        history = history or []
        
        try:
            retrieved_docs = await self._hybrid_search(
                query=request.message,
                top_k=self.top_k
            )
            
            if not retrieved_docs:
                return GeneratedAnswerResponse(
                    message_id=str(uuid.uuid4()),
                    content="I couldn't find any relevant information in the available documents.",
                    sources=[],
                    processing_time=time.time() - start_time,
                    model_used=self.model,
                    timestamp=datetime.now().isoformat()
                )

            # Rerank
            reranked_docs = await self.reranker.rerank(
                documents=retrieved_docs,
                query=request.message,
                top_k=self.top_k
            )

            # Generate answer
            context = "\n\n".join([
                f"[{i+1}]{doc.content}"
                for i, doc in enumerate(reranked_docs)
            ])

            formatted_history = []
            for msg in history[-5:]:
                if isinstance(msg, dict):
                    if "role" in msg and "content" in msg:
                        formatted_history.append({"role": msg["role"], "content": msg["content"]})
                    elif "user" in msg:
                        formatted_history.append({"role": "user", "content": msg["user"]})
                    elif "assistant" in msg:
                        formatted_history.append({"role": "assistant", "content": msg["assistant"]})

            user_prompt = f"""Context documents:
{context}

Question: {request.message}

Please answer the question using ONLY the information from the context above."""

            messages = [
                *formatted_history,
                {"role": "user", "content": user_prompt}
            ]

            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content
            sources = []

            for i, doc in enumerate(reranked_docs, 1):
                file_name = doc.metadata.get("file_name", "Unknown") if doc.metadata else "Unknown"
                sources.append(Source(
                    filename=file_name,
                    content=(doc.content or "")[:200] + "...",
                    similarity=doc.score,
                    chunk_index=i
                ))

            return GeneratedAnswerResponse(
                message_id=str(uuid.uuid4()),
                content=answer,
                sources=sources,
                processing_time=time.time() - start_time,
                model_used=self.model,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Error in execute: {e}", exc_info=True)
            return GeneratedAnswerResponse(
                message_id=str(uuid.uuid4()),
                content=f"Error: {str(e)}",
                sources=[],
                processing_time=time.time() - start_time,
                model_used=self.model,
                timestamp=datetime.now().isoformat()
            )

