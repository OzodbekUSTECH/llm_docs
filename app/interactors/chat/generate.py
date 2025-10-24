import asyncio
import time
import uuid
import json
import logging
import re
from datetime import datetime
from typing import AsyncGenerator, List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI

from app.dto.chat import GenerateAnswerRequest, GeneratedAnswerResponse, Source
from app.dto.schemas import DocumentSchema, RetrievedDocument
from app.services.chat_storage import chat_storage
from app.interactors.chat.system_prompts import STRICT_RAG_PROMPT
from app.utils.vectors_store import QdrantVectorStore
from app.utils.embeddings import FastEmbedEmbeddings
from app.utils.docs_store import LanceDBDocumentStore
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Citation data class for storing citation information"""
    id: int
    content: str
    source: str
    page: str
    score: float
    semantic_score: float
    keyword_score: float
    doc_id: str
    excerpt: Optional[str] = None


class CitationPipeline:
    """Pipeline для создания и обработки цитат с помощью LLM"""

    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client

    async def extract_citations(
        self,
        answer: str,
        documents: List[RetrievedDocument]
    ) -> List[Citation]:
        """Извлекает цитаты из ответа и связывает их с документами"""
        citations = []
        citation_pattern = r'\[(\d+)\]'
        matches = set(re.findall(citation_pattern, answer))
        logger.info(f"📝 Found citation references: {sorted(matches)}")

        for match in sorted(matches, key=lambda x: int(x)):
            citation_id = int(match)
            if 1 <= citation_id <= len(documents):
                doc = documents[citation_id - 1]
                source = "Unknown"
                page = "N/A"
                if doc.metadata:
                    source = doc.metadata.get("file_name", "Unknown")
                    page = doc.metadata.get("page_label", "N/A")
                citation = Citation(
                    id=citation_id,
                    content=doc.content,
                    source=source,
                    page=page,
                    score=doc.score,
                    semantic_score=getattr(doc, 'semantic_score', 0.0),
                    keyword_score=getattr(doc, 'keyword_score', 0.0),
                    doc_id=doc.doc_id
                )
                citations.append(citation)
        logger.info(f"✅ Extracted {len(citations)} citations")
        return citations

    async def enhance_citations_with_llm(
        self,
        answer: str,
        citations: List[Citation]
    ) -> List[Citation]:
        """Улучшает цитаты с помощью LLM, извлекая наиболее релевантные части"""
        if not citations:
            return citations
        try:
            logger.info(f"🤖 Enhancing {len(citations)} citations with LLM...")
            prompt = f"""Given the answer and the source documents, extract the most relevant excerpt (1-2 sentences) from each document that directly supports the claims in the answer.

Answer:
{answer}

For each source, return ONLY the most relevant excerpt that was used.

Sources:
"""
            for citation in citations:
                prompt += f"\n[{citation.id}] {citation.content[:500]}...\n"

            prompt += "\nReturn a JSON array with format: [{\"id\": 1, \"excerpt\": \"relevant excerpt\"}]"

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise citation extraction assistant. Extract only the most relevant parts that support the answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            result_text = response.choices[0].message.content.strip()
            try:
                excerpts = json.loads(result_text)
                for excerpt_data in excerpts:
                    citation_id = excerpt_data.get("id")
                    excerpt = excerpt_data.get("excerpt")
                    for citation in citations:
                        if citation.id == citation_id and excerpt:
                            citation.excerpt = excerpt
                logger.info(f"✅ Enhanced citations with LLM excerpts")
            except json.JSONDecodeError:
                logger.warning("⚠️ Could not parse LLM response as JSON, keeping original citations")
            return citations
        except Exception as e:
            logger.error(f"❌ Error enhancing citations: {e}", exc_info=True)
            return citations

    async def extract_evidence_for_document(
        self,
        answer: str,
        document: 'RetrievedDocument',
        citation_number: int
    ) -> Optional[str]:
        """
        Извлекает evidence для КОНКРЕТНОГО документа используя LLM.
        """
        system_prompt = """You are a world class algorithm to extract exact citations FROM DOCUMENT CONTENT.
Find the EXACT quote from the DOCUMENT CONTENT that was used to generate the part of the ANSWER that references this document.

CRITICAL RULES:
- Return ONLY exact quote FROM THE DOCUMENT CONTENT (never from the answer)
- Must be a substring that exists in the document content
- Maximum 15 words
- Must be the most relevant part that supports the answer
- Return as JSON: {"evidence": "exact quote from document"}
"""
        pattern = rf'\[{citation_number}\][^\[]*'
        match = re.search(pattern, answer)
        relevant_part = match.group(0) if match else answer[:200]
        user_prompt = f"""Answer excerpt referencing document [{citation_number}]:
{relevant_part}

DOCUMENT CONTENT:
{document.content[:800]}

Find the exact substring from the DOCUMENT CONTENT above that was used as source for the answer excerpt.
The evidence must be a direct quote that EXISTS in the document content.

Return as JSON: {{"evidence": "exact substring from document content"}}"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            result = response.choices[0].message.content.strip()
            parsed = json.loads(result)
            evidence = parsed.get("evidence", "")
            if evidence:
                words = evidence.split()
                if len(words) > 15:
                    evidence = " ".join(words[:15])
                if evidence.lower() in document.content.lower():
                    logger.info(f"📝 ✅ Valid evidence for doc [{citation_number}]: \"{evidence[:50]}...\"")
                    return evidence
                else:
                    logger.warning(f"📝 ❌ Evidence not found in document [{citation_number}]: \"{evidence[:50]}...\"")
                    fallback = " ".join(document.content.split()[:10])
                    logger.info(f"📝 🔄 Using fallback for doc [{citation_number}]: \"{fallback[:50]}...\"")
                    return fallback
            return None
        except Exception as e:
            logger.error(f"❌ Error extracting evidence for doc [{citation_number}]: {e}")
            return None

    def find_evidences_in_documents(
        self,
        evidences: List[str],
        documents: List[RetrievedDocument]
    ) -> dict:
        """
        Сопоставляет evidences с документами используя fuzzy matching.
        """
        doc_evidences = {}
        for evidence in evidences:
            evidence_lower = evidence.lower().strip()
            for i, doc in enumerate(documents, 1):
                doc_text_lower = doc.content.lower()
                if evidence_lower in doc_text_lower:
                    if i not in doc_evidences:
                        doc_evidences[i] = {"evidences": [], "doc": doc}
                    doc_evidences[i]["evidences"].append(evidence)
                    logger.info(f"✅ Found evidence in doc {i}: \"{evidence[:50]}...\"")
                    break
                else:
                    evidence_words = set(evidence_lower.split())
                    if len(evidence_words) == 0:
                        continue
                    doc_words = doc_text_lower.split()
                    window_size = len(evidence_words)
                    best_score = 0.0
                    for j in range(len(doc_words) - window_size + 1):
                        window = doc_words[j:j + window_size]
                        window_set = set(window)
                        intersection = evidence_words & window_set
                        union = evidence_words | window_set
                        score = len(intersection) / len(union) if union else 0
                        if score > best_score and score >= 0.6:
                            best_score = score
                    if best_score >= 0.6:
                        if i not in doc_evidences:
                            doc_evidences[i] = {"evidences": [], "doc": doc}
                        doc_evidences[i]["evidences"].append(evidence)
                        logger.info(f"✅ Found fuzzy match in doc {i} (score: {best_score:.2f})")
                        break
        return doc_evidences

    def format_citations_markdown(self, citations: List[Citation]) -> str:
        """Форматирует цитаты в markdown для отображения"""
        if not citations:
            return "**No citations found.**"
        md_parts = ["\n\n---\n\n### 📚 Sources & Citations\n"]
        for citation in citations:
            md_parts.append(f"\n**[{citation.id}]** {citation.source}, Page: {citation.page}")
            md_parts.append(f"\n*Score: {citation.score:.3f} (Semantic: {citation.semantic_score:.3f}, Keyword: {citation.keyword_score:.3f})*")
            if citation.excerpt:
                md_parts.append(f"\n> {citation.excerpt}")
            md_parts.append("\n")
        return "\n".join(md_parts)


class SimpleReranker:
    """Kotaemon-style LLM reranker with 0-10 relevance scoring"""
    def __init__(self, openai_client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model
        self.system_prompt = """You are a RELEVANCE grader; providing the relevance of the given CONTEXT to the given QUESTION.
Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.

A few additional scoring guidelines:
- CONTEXT that is RELEVANT to some of the QUESTION should score 2, 3 or 4
- CONTEXT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8
- CONTEXT that is RELEVANT to the entire QUESTION should get a score of 9 or 10
- CONTEXT must be relevant and helpful for answering the entire QUESTION to get a score of 10
- CONTEXT that is completely irrelevant should get a score of 0 or 1

Respond with ONLY a single number (0-10)."""

    async def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = 5
    ) -> tuple[List[RetrievedDocument], float]:
        """
        Rerank documents using LLM scoring (Kotaemon-style)
        """
        if not documents:
            return [], 0.0
        logger.info(f"🔄 Kotaemon Reranking: scoring {len(documents)} documents...")
        scored_docs = []
        for i, doc in enumerate(documents):
            content_preview = doc.content[:500] if len(doc.content) > 500 else doc.content
            user_prompt = f"""Question: {query}

Context:
{content_preview}

Relevance score (0-10):"""
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                result_text = response.choices[0].message.content.strip()
                score_match = re.search(r'(\d+(?:\.\d+)?)', result_text)
                if score_match:
                    llm_score_raw = float(score_match.group(1))
                    llm_score_raw = min(max(llm_score_raw, 0), 10)
                    llm_score = llm_score_raw / 10.0
                else:
                    logger.warning(f"⚠️ Could not parse LLM score from: {result_text}")
                    llm_score = 0.5
                    llm_score_raw = 5.0
                # Ensure doc.metadata is a dict
                if getattr(doc, 'metadata', None) is None:
                    doc.metadata = {}
                doc.metadata["llm_rerank_score"] = llm_score
                doc.metadata["llm_rerank_score_raw"] = llm_score_raw
                combined_score = 0.5 * doc.score + 0.5 * llm_score
                doc.metadata["combined_rerank_score"] = combined_score
                scored_docs.append((doc, combined_score, llm_score, llm_score_raw))
                logger.info(f"📊 Doc {i + 1}: Hybrid={doc.score:.3f} | LLM={llm_score_raw:.1f}/10 | Combined={combined_score:.3f}")
            except Exception as e:
                logger.error(f"❌ Error scoring doc {i+1}: {e}")
                scored_docs.append((doc, doc.score, 0.5, 5.0))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _, _, _ in scored_docs[:top_k]]
        if scored_docs[:top_k]:
            avg_llm_score = sum(llm_raw for _, _, _, llm_raw in scored_docs[:top_k]) / len(scored_docs[:top_k])
            max_llm_score = max(llm_raw for _, _, _, llm_raw in scored_docs[:top_k])
            avg_combined = sum(comb for _, comb, _, _ in scored_docs[:top_k]) / len(scored_docs[:top_k])
        else:
            avg_llm_score = 0.0
            max_llm_score = 0.0
            avg_combined = 0.0
        logger.info(f"✅ Reranked top {len(top_docs)}: Avg LLM={avg_llm_score:.1f}/10 | Max={max_llm_score:.1f}/10 | Avg Combined={avg_combined:.3f}")
        if max_llm_score < 5.0:
            logger.warning(f"⚠️ LOW RELEVANCE WARNING: Max LLM score is {max_llm_score:.1f}/10 (< 5/10)")
        return top_docs, max_llm_score


class GenerateAnswerInteractor:
    """Generate answers using RAG with Hybrid Search (Semantic + Keyword) + Reranking"""
    def __init__(self):
        """Initialize the interactor with necessary components"""
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.vector_store = QdrantVectorStore(
            collection_name="documents",
            url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
            vector_size=768,
        )
        self.doc_store = LanceDBDocumentStore()
        self.embedding = FastEmbedEmbeddings()
        self.reranker = SimpleReranker(self.openai_client)
        self.citation_pipeline = CitationPipeline(self.openai_client)
        self.model = "gpt-4o-mini"
        self.max_tokens = 2000
        self.temperature = 0.3
        self.use_reranking = True
        self.use_keyword_search = True

    async def keyword_search_lancedb(
        self,
        query: str,
        top_k: int = 15
    ) -> List[tuple[str, float]]:
        """Keyword search using LanceDB Full-Text Search (Tantivy)"""
        try:
            fts_results = self.doc_store.query(query, top_k=top_k)
            logger.info(f"🔤 LanceDB FTS found {len(fts_results)} results")
            results = []
            for i, doc in enumerate(fts_results):
                score = 1.0 - (i * 0.05)
                score = max(score, 0.1)
                # fix: LanceDB docs can have id_ or doc_id, ensure correct attr
                doc_id = getattr(doc, "id_", None)
                if doc_id is None:
                    doc_id = getattr(doc, "doc_id", None)
                if doc_id is not None:
                    results.append((doc_id, score))
            return results
        except Exception as e:
            logger.error(f"❌ Error in LanceDB FTS: {e}", exc_info=True)
            return []

    async def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[RetrievedDocument]:
        """Retrieve relevant documents using Hybrid Search (Semantic + Keyword) + Reranking"""
        try:
            logger.info(f"🔍 Step 1: Semantic search for: {query[:100]}...")
            query_embedding_docs = self.embedding.invoke(query)
            if not query_embedding_docs or not getattr(query_embedding_docs[0], "embedding", None):
                logger.error("Failed to create query embedding")
                return []
            query_embedding = query_embedding_docs[0].embedding
            initial_k = top_k * 3 if self.use_keyword_search or self.use_reranking else top_k * 2
            embeddings, similarities, doc_ids = await self.vector_store.query(
                embedding=query_embedding,
                top_k=initial_k,
            )
            if not doc_ids:
                logger.warning("No documents found in vector search")
                return []
            logger.info(f"✅ Semantic search: found {len(doc_ids)} documents")
            keyword_scores_dict = {}
            all_doc_ids = set(doc_ids)
            if self.use_keyword_search:
                logger.info(f"🔤 Step 2: LanceDB Full-Text Search...")
                fts_results = await self.keyword_search_lancedb(query, top_k=initial_k)
                for fts_doc_id, fts_score in fts_results:
                    keyword_scores_dict[fts_doc_id] = fts_score
                    all_doc_ids.add(fts_doc_id)
                logger.info(f"✅ FTS found {len(fts_results)} documents")
            logger.info(f"📚 Retrieving {len(all_doc_ids)} unique documents from LanceDB...")
            all_docs = self.doc_store.get(list(all_doc_ids))
            # fix: allow LanceDB docs to have either id_ or doc_id
            doc_dict = {}
            for doc in all_docs:
                doc_id = getattr(doc, "id_", None)
                if doc_id is None:
                    doc_id = getattr(doc, "doc_id", None)
                if doc_id is not None:
                    doc_dict[doc_id] = doc
            semantic_scores_dict = dict(zip(doc_ids, similarities))
            retrieved_docs = []
            for doc_id in all_doc_ids:
                if doc_id not in doc_dict:
                    continue
                doc = doc_dict[doc_id]
                semantic_score = semantic_scores_dict.get(doc_id, 0.0)
                keyword_score = keyword_scores_dict.get(doc_id, 0.0)
                if self.use_keyword_search:
                    combined_score = semantic_weight * semantic_score + keyword_weight * keyword_score
                else:
                    combined_score = semantic_score
                # fix: doc may have text or content or both
                doc_content = getattr(doc, "text", None) or getattr(doc, "content", None) or ""
                doc_metadata = getattr(doc, "metadata", None) or {}
                doc_docid = getattr(doc, "doc_id", None)
                retrieved_doc = RetrievedDocument(
                    content=doc_content,
                    metadata=doc_metadata,
                    doc_id=doc_docid,
                    score=float(combined_score),
                    semantic_score=float(semantic_score),
                    keyword_score=float(keyword_score),
                )
                retrieved_docs.append(retrieved_doc)
            retrieved_docs.sort(key=lambda x: x.score, reverse=True)
            if self.use_keyword_search and retrieved_docs:
                avg_semantic = sum(d.semantic_score for d in retrieved_docs) / len(retrieved_docs)
                avg_keyword = sum(d.keyword_score for d in retrieved_docs) / len(retrieved_docs)
                logger.info(f"✅ Hybrid scores: semantic={avg_semantic:.3f}, keyword={avg_keyword:.3f}")
            filtered_docs = [doc for doc in retrieved_docs if doc.score >= score_threshold]
            if not filtered_docs:
                logger.warning(f"⚠️ No documents above threshold {score_threshold}")
                filtered_docs = retrieved_docs[:top_k]
            logger.info(f"✅ After filtering: {len(filtered_docs)} documents")
            max_llm_score = 10.0
            if self.use_reranking and len(filtered_docs) > top_k:
                logger.info(f"🔄 Step 3: LLM-based reranking (Kotaemon-style)...")
                reranked_docs, max_llm_score = await self.reranker.rerank(query, filtered_docs, top_k)
                logger.info(f"✅ Reranked to {len(reranked_docs)} documents | Max LLM score: {max_llm_score:.1f}/10")
                filtered_docs = reranked_docs
            final_docs = filtered_docs[:top_k]
            self._last_max_llm_score = max_llm_score
            logger.info(f"✅ Final result: {len(final_docs)} documents")
            return final_docs
        except Exception as e:
            logger.error(f"❌ Error in retrieve_documents: {e}", exc_info=True)
            return []

    def format_context(self, documents: List[RetrievedDocument]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant documents found."
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata_str = ""
            if getattr(doc, "metadata", None):
                file_name = doc.metadata.get("file_name", "")
                page_label = doc.metadata.get("page_label", "")
                if file_name or page_label:
                    metadata_str = f" [Source: {file_name}, Page: {page_label}]"
            doc_content = getattr(doc, "content", "")
            context_parts.append(
                f"[{i}]{metadata_str}\n{doc_content}\n"
            )
        return "\n".join(context_parts)

    def format_history(self, history: List[Dict]) -> List[Dict[str, str]]:
        """Format chat history for OpenAI API"""
        formatted = []
        for msg in history[-5:]:
            if isinstance(msg, dict):
                if "role" in msg and "content" in msg:
                    formatted.append({"role": msg["role"], "content": msg["content"]})
                elif "user" in msg:
                    formatted.append({"role": "user", "content": msg["user"]})
                elif "assistant" in msg:
                    formatted.append({"role": "assistant", "content": msg["assistant"]})
        return formatted

    async def stream(
        self,
        message: str,
        conv_id: str,
        history: Optional[List[Dict]] = None,
        top_k: int = 5,
        **kwargs
    ) -> AsyncGenerator[DocumentSchema, None]:
        """
        Stream response with document retrieval and answer generation
        """
        start_time = time.time()
        history = history or []
        try:
            search_mode = "Hybrid Search (Qdrant + LanceDB FTS)" if self.use_keyword_search else "Semantic Search (Qdrant)"
            yield DocumentSchema(
                content=f"🔍 Starting {search_mode}...",
                channel="debug"
            )
            retrieved_docs = await self.retrieve_documents(
                query=message,
                top_k=top_k
            )
            if not retrieved_docs:
                yield DocumentSchema(
                    content="⚠️ No relevant documents found",
                    channel="debug"
                )
                yield DocumentSchema(
                    content="I couldn't find any relevant information in the available documents. Could you please rephrase your question or provide more details?",
                    channel="chat"
                )
                return
            avg_score = sum(d.score for d in retrieved_docs) / len(retrieved_docs)
            avg_semantic = sum(d.semantic_score for d in retrieved_docs) / len(retrieved_docs)
            avg_keyword = sum(d.keyword_score for d in retrieved_docs) / len(retrieved_docs)
            stats = f"✅ Found {len(retrieved_docs)} documents | Avg score: {avg_score:.3f}"
            if self.use_keyword_search:
                stats += f" | Semantic: {avg_semantic:.3f} | Keyword: {avg_keyword:.3f}"
            if self.use_reranking:
                stats += " | 🔄 Reranked"
            yield DocumentSchema(
                content=stats,
                channel="debug"
            )
            max_llm_score = getattr(self, '_last_max_llm_score', 10.0)
            if self.use_reranking and max_llm_score < 5.0:
                yield DocumentSchema(
                    content=f"⚠️ **LOW RELEVANCE WARNING**\n\nThe retrieved documents have low relevance scores (Max: {max_llm_score:.1f}/10). The answer may not be accurate. Please double-check the information.",
                    channel="info"
                )
            for i, doc in enumerate(retrieved_docs, 1):
                doc_info = f"**Document {i}**\n"
                doc_info += f"- **Final Score**: {doc.score:.3f}"
                if self.use_keyword_search:
                    doc_info += f" (Semantic: {doc.semantic_score:.3f}, Keyword: {doc.keyword_score:.3f})"
                if doc.metadata and "llm_rerank_score_raw" in doc.metadata:
                    llm_score = doc.metadata["llm_rerank_score_raw"]
                    doc_info += f"\n- **🤖 LLM Relevance**: {llm_score:.1f}/10"
                doc_info += "\n"
                if doc.metadata:
                    file_name = doc.metadata.get("file_name", "Unknown")
                    page = doc.metadata.get("page_label", "N/A")
                    doc_info += f"- **Source**: {file_name}, Page: {page}\n"
                doc_info += f"\n```\n{doc.content}\n```"
                yield DocumentSchema(
                    content=doc_info,
                    channel="info"
                )
            yield DocumentSchema(
                content="💬 Generating answer...",
                channel="debug"
            )
            context = self.format_context(retrieved_docs)
            formatted_history = self.format_history(history)
            system_prompt = """You are a helpful AI assistant with access to document knowledge base.

CRITICAL RULES:
1. **ONLY use information from the provided context** - Never invent or make up information
2. **Be precise and cite sources** - Reference specific documents using [1], [2] notation
3. **If information is not in context** - Clearly state "I don't have this information in the available documents"
4. **Use exact quotes when possible** - This ensures accuracy
5. **Be concise but complete** - Provide all relevant information from the context

RESPONSE FORMAT:
- Answer the question based ONLY on the retrieved documents
- Use [1], [2], etc. to cite specific sources
- If context doesn't contain the answer, say so clearly
- Be helpful and conversational while staying accurate"""
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
            yield DocumentSchema(
                content="📝 Extracting evidences for each citation...",
                channel="debug"
            )
            citations = await self.citation_pipeline.extract_citations(
                answer=full_response,
                documents=retrieved_docs
            )
            logger.info(f"📌 Found {len(citations)} citations in answer: {[c.id for c in citations]}")
            doc_evidences = {}
            for citation in citations:
                yield DocumentSchema(
                    content=f"📝 Extracting evidence for [{citation.id}]...",
                    channel="debug"
                )
                try:
                    idx = citation.id - 1
                    if idx < 0 or idx >= len(retrieved_docs):
                        raise IndexError("Citation document index out of range")
                    doc_for_evidence = retrieved_docs[idx]
                except Exception:
                    doc_for_evidence = None
                if doc_for_evidence:
                    evidence = await self.citation_pipeline.extract_evidence_for_document(
                        answer=full_response,
                        document=doc_for_evidence,
                        citation_number=citation.id
                    )
                    if evidence:
                        doc_rank = citation.id
                        doc_evidences[doc_rank] = {
                            "evidences": [evidence],
                            "doc": doc_for_evidence
                        }
                        highlight_data = {
                            "doc_rank": doc_rank,
                            "evidence": evidence,
                            "doc_id": doc_for_evidence.doc_id
                        }
                        yield DocumentSchema(
                            content=json.dumps(highlight_data),
                            channel="highlight"
                        )
                        logger.info(f"✅ Evidence for [{citation.id}]: \"{evidence[:50]}...\"")
                    else:
                        logger.warning(f"⚠️ No evidence found for [{citation.id}], using fallback")
                        doc = doc_for_evidence
                        fallback_evidence = " ".join(doc.content.split()[:12])
                        doc_rank = citation.id
                        doc_evidences[doc_rank] = {
                            "evidences": [fallback_evidence],
                            "doc": doc
                        }
                        highlight_data = {
                            "doc_rank": doc_rank,
                            "evidence": fallback_evidence,
                            "doc_id": doc.doc_id
                        }
                        yield DocumentSchema(
                            content=json.dumps(highlight_data),
                            channel="highlight"
                        )
                else:
                    logger.warning(f"⚠️ Could not extract evidence: missing document for citation [{citation.id}]")
            logger.info(f"✅ Extracted evidences for {len(doc_evidences)} cited documents")
            if citations and len(citations) > 0:
                citations_md = self.citation_pipeline.format_citations_markdown(citations)
                yield DocumentSchema(
                    content=citations_md,
                    channel="chat"
                )
            processing_time = time.time() - start_time
            yield DocumentSchema(
                content=f"⏱️ Processing time: {processing_time:.2f}s | Documents: {len(retrieved_docs)} | Citations: {len(citations)}",
                channel="debug"
            )
            chat_storage.add_message(
                chat_id=conv_id,
                role="user",
                content=message
            )
            retrieved_docs_data = []
            for i, doc in enumerate(retrieved_docs, 1):
                doc_data = {
                    "rank": i,
                    "content": doc.content,
                    "score": doc.score,
                    "semantic_score": getattr(doc, 'semantic_score', 0.0),
                    "keyword_score": getattr(doc, 'keyword_score', 0.0),
                    "doc_id": doc.doc_id,
                    "metadata": doc.metadata or {}
                }
                if doc.metadata and "llm_rerank_score_raw" in doc.metadata:
                    doc_data["llm_score"] = doc.metadata["llm_rerank_score_raw"]
                retrieved_docs_data.append(doc_data)
            evidences_data = {}
            if doc_evidences:
                for doc_rank, evidence_info in doc_evidences.items():
                    evidences_data[str(doc_rank)] = {
                        "evidences": evidence_info["evidences"],
                        "doc_id": evidence_info["doc"].doc_id
                    }
            assistant_metadata = {
                "retrieved_docs": retrieved_docs_data,
                "evidences": evidences_data,
                "processing_time": processing_time,
                "citations_count": len(citations)
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
            retrieved_docs = await self.retrieve_documents(
                query=request.message,
                top_k=5
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
            context = self.format_context(retrieved_docs)
            formatted_history = self.format_history(history)
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
            for i, doc in enumerate(retrieved_docs, 1):
                file_name = doc.metadata.get("file_name", "Unknown") if getattr(doc, "metadata", None) else "Unknown"
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