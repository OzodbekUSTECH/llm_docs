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
    """
    Оптимизированный Kotaemon-style LLM reranker с batch processing и условным применением.
    
    Оптимизации:
    1. Reranking только для топ-15 документов (не всех)
    2. Пропускает reranking если hybrid scores уже высокие (>= 0.8)
    3. Batch scoring нескольких документов одновременно
    4. Early stopping если нашли отличные документы
    """
    def __init__(self, openai_client: AsyncOpenAI, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model
        self.max_rerank_count = 15  # Максимум документов для reranking
        self.skip_rerank_threshold = 0.8  # Пропустить reranking если hybrid score >= 0.8
        self.high_quality_threshold = 8.0  # Порог высокого качества для early stopping
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
        Оптимизированный reranking с batch processing и условным применением
        
        Оптимизации:
        1. Берет только топ-15 документов для reranking (не 40+)
        2. Пропускает reranking если hybrid scores уже высокие
        3. Останавливается раньше если нашел отличные документы (LLM score >= 8)
        """
        if not documents:
            return [], 0.0
        
        # Проверяем нужно ли вообще reranking
        top_doc_score = documents[0].score if documents else 0.0
        if top_doc_score >= self.skip_rerank_threshold:
            logger.info(f"🚀 Skipping reranking: top hybrid score {top_doc_score:.3f} already high (>= {self.skip_rerank_threshold})")
            # Просто назначаем LLM scores равными hybrid scores для консистентности
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["llm_rerank_score"] = doc.score
                doc.metadata["llm_rerank_score_raw"] = doc.score * 10
                doc.metadata["combined_rerank_score"] = doc.score
            return documents[:top_k], top_doc_score * 10
        
        # Берем только топ-N документов для reranking
        docs_to_rerank = documents[:self.max_rerank_count]
        if len(docs_to_rerank) < len(documents):
            logger.info(f"🎯 Reranking only top {len(docs_to_rerank)} docs (of {len(documents)} total) for speed")
        
        logger.info(f"🔄 Optimized Reranking: scoring {len(docs_to_rerank)} documents...")
        scored_docs = []
        
        for i, doc in enumerate(docs_to_rerank):
            # Early stopping если уже нашли отличный документ
            if scored_docs:
                max_llm_score_found = max(llm_raw for _, _, _, llm_raw in scored_docs)
                if max_llm_score_found >= self.high_quality_threshold:
                    logger.info(f"✅ Early stopping: already found excellent documents (LLM score >= {max_llm_score_found:.1f})")
                    break
            
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
                
                # Логируем только каждые 5 документов для экономии
                if (i + 1) % 5 == 0 or i == len(docs_to_rerank) - 1:
                    logger.info(f"📊 Doc {i + 1}/{len(docs_to_rerank)}: Hybrid={doc.score:.3f} | LLM={llm_score_raw:.1f}/10 | Combined={combined_score:.3f}")
                    
            except Exception as e:
                logger.error(f"❌ Error scoring doc {i+1}: {e}")
                scored_docs.append((doc, doc.score, 0.5, 5.0))
        
        # Добавляем остальные документы без LLM scoring (они просто сохраняют hybrid scores)
        docs_without_reranking = documents[len(docs_to_rerank):]
        for doc in docs_without_reranking:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["llm_rerank_score"] = doc.score
            doc.metadata["llm_rerank_score_raw"] = doc.score * 10
            doc.metadata["combined_rerank_score"] = doc.score
            scored_docs.append((doc, doc.score, doc.score, doc.score * 10))
        
        # Сортируем по combined score
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
            
        logger.info(f"✅ Optimized Reranking: scored {len(docs_to_rerank)} docs, selected top {len(top_docs)}")
        logger.info(f"📊 Results: Avg LLM={avg_llm_score:.1f}/10 | Max={max_llm_score:.1f}/10 | Avg Combined={avg_combined:.3f}")
        
        if max_llm_score < 5.0:
            logger.warning(f"⚠️ LOW RELEVANCE WARNING: Max LLM score is {max_llm_score:.1f}/10 (< 5/10)")
            
        return top_docs, max_llm_score


class QueryRephraser:
    """
    Класс для интеллектуального перефразирования запросов с помощью LLM.
    
    Анализирует найденные документы и определяет недостающую информацию,
    чтобы сгенерировать новые запросы для поиска пропущенных данных.
    """
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
    
    async def analyze_and_find_missing_info(
        self,
        original_query: str,
        found_documents: List[RetrievedDocument],
        max_iterations: int = 3
    ) -> List[str]:
        """Анализирует найденные документы и определяет недостающую информацию"""
        try:
            logger.info(f"🔍 Analyzing found documents to identify missing information...")
            
            # Проверяем, есть ли вообще релевантные документы
            relevant_docs = [doc for doc in found_documents if doc.score > 0.3]
            if not relevant_docs:
                logger.warning("⚠️ No relevant documents found, generating broad search queries")
                return await self._generate_broad_search_queries(original_query, max_iterations)
            
            # Анализируем содержимое найденных документов
            found_content = "\n\n".join([
                f"[Document {i+1}] Score: {doc.score:.3f}\nContent: {doc.content[:300]}..."
                for i, doc in enumerate(relevant_docs[:5])
            ])
            
            # Формируем prompt для анализа
            analysis_prompt = f"""Analyze the following user question and the documents that were found.

USER QUESTION: {original_query}

FOUND DOCUMENTS:
{found_content}

Your task:
1. Identify which specific parts/aspects of the question are NOT adequately answered by the found documents
2. Generate {max_iterations-1} new search queries that target the MISSING information
3. Focus on aspects that are clearly NOT covered in the existing documents
4. If the documents don't contain relevant information, generate broader search queries

IMPORTANT:
- Each new query should target DIFFERENT missing information
- Use different keywords and phrases from the original query
- Be specific about what information is missing
- If no relevant info found, try broader terms and synonyms
- Return ONLY a JSON object with format: {{"missing_info": ["query1", "query2", "query3"]}}

Return the missing information as search queries:"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing search results to identify missing information. You help refine search queries to find specific information gaps."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.7,
                max_tokens=400,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(result_text)
                missing_queries = parsed.get("missing_info", [])
                
                # Возвращаем оригинальный запрос + запросы для недостающей информации
                all_queries = [original_query] + missing_queries[:max_iterations-1]
                logger.info(f"✅ Identified {len(missing_queries)} areas of missing information")
                logger.info(f"   Generated {len(all_queries)} total search queries")
                
                return all_queries
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"⚠️ Could not parse missing info analysis: {e}")
                # Fallback: используем простые вариации
                return await self._generate_simple_variations(original_query, max_iterations)
                
        except Exception as e:
            logger.error(f"❌ Error analyzing missing information: {e}")
            # Fallback: используем простые вариации
            return await self._generate_simple_variations(original_query, max_iterations)
    
    async def _generate_simple_variations(
        self,
        original_query: str,
        max_iterations: int = 3
    ) -> List[str]:
        """Генерирует простые вариации запроса (fallback метод)"""
        try:
            logger.info(f"🔄 Generating simple query variations for: {original_query[:100]}...")
            
            prompt = f"""Given the following user question, generate {max_iterations-1} different ways to ask the same question. Each variation should:
1. Use different keywords and phrases
2. Focus on different aspects of the question
3. Use synonyms and alternative expressions
4. Maintain the same core meaning

Original question: {original_query}

Return ONLY a JSON object with format: {{"variations": ["query1", "query2", "query3"]}}"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a query rephrasing expert. Generate diverse but semantically equivalent variations of user questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            parsed = json.loads(result_text)
            variations = parsed.get("variations", [])
            
            all_queries = [original_query] + variations[:max_iterations-1]
            logger.info(f"✅ Generated {len(all_queries)} query variations")
            return all_queries
                
        except Exception as e:
            logger.error(f"❌ Error generating simple variations: {e}")
            return [original_query]
    
    async def _generate_broad_search_queries(
        self,
        original_query: str,
        max_iterations: int = 3
    ) -> List[str]:
        """Генерирует широкие поисковые запросы когда релевантные документы не найдены"""
        try:
            logger.info(f"🔄 Generating broad search queries for: {original_query[:100]}...")
            
            prompt = f"""The user asked: "{original_query}"

No relevant documents were found for this query. Generate {max_iterations-1} broad search queries that might help find related information. Use:

1. Broader terms and synonyms
2. Related concepts and topics
3. Different ways to express the same question
4. General terms instead of specific ones

Examples:
- If asking about "price clause", try "pricing", "cost", "payment terms", "financial terms"
- If asking about "Glencore contract", try "agreement", "document", "terms", "conditions"

Return ONLY a JSON object with format: {{"broad_queries": ["query1", "query2", "query3"]}}"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at generating broad search queries when specific information is not found. You help expand search terms to find related information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            parsed = json.loads(result_text)
            broad_queries = parsed.get("broad_queries", [])
            
            all_queries = [original_query] + broad_queries[:max_iterations-1]
            logger.info(f"✅ Generated {len(all_queries)} broad search queries")
            return all_queries
                
        except Exception as e:
            logger.error(f"❌ Error generating broad search queries: {e}")
            return [original_query]
    
    async def decompose_complex_query(
        self,
        query: str,
        max_subqueries: int = 3
    ) -> List[str]:
        """
        Разбивает сложный вопрос на несколько простых подвопросов.
        Это полезно когда один вопрос содержит несколько аспектов.
        """
        try:
            logger.info(f"🔍 Decomposing complex query into sub-queries: {query[:100]}...")
            
            prompt = f"""The user asked a complex question that might contain multiple aspects or sub-questions.

Original Question: {query}

Your task:
1. Analyze if this question can be broken down into {max_subqueries} simpler sub-questions
2. Each sub-question should target ONE specific aspect or piece of information
3. The sub-questions together should cover all aspects of the original question
4. If the question is already simple, return variations of it

Examples:
- "What is the price and payment terms?" → ["What is the price?", "What are the payment terms?"]
- "How does the contract define delivery and warranty?" → ["How does the contract define delivery?", "What are the warranty terms?"]

Return ONLY a JSON object with format: {{"sub_queries": ["query1", "query2", "query3"]}}"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at breaking down complex questions into simpler sub-questions. Each sub-question should be independently searchable."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            parsed = json.loads(result_text)
            sub_queries = parsed.get("sub_queries", [])
            
            logger.info(f"✅ Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries[:max_subqueries]
                
        except Exception as e:
            logger.error(f"❌ Error decomposing query: {e}")
            return [query]
    
    async def extract_key_entities(
        self,
        query: str
    ) -> Dict[str, List[str]]:
        """
        Извлекает ключевые сущности из вопроса для таргетированного поиска.
        
        Returns:
            dict with keys:
            - entities: List[str] - имена, компании, даты и т.д.
            - concepts: List[str] - ключевые концепты и термины
            - actions: List[str] - действия или процессы
        """
        try:
            logger.info(f"🔍 Extracting key entities from query: {query[:100]}...")
            
            prompt = f"""Extract key information from this search query to help improve document search.

Query: {query}

Extract:
1. **Entities**: Names, companies, dates, locations, specific terms (e.g., "Glencore", "contract", "2023")
2. **Concepts**: Key concepts and topics (e.g., "pricing", "payment terms", "delivery")
3. **Actions**: Actions or processes mentioned (e.g., "define", "calculate", "determine")

Return ONLY a JSON object with format:
{{
    "entities": ["entity1", "entity2"],
    "concepts": ["concept1", "concept2"],
    "actions": ["action1", "action2"]
}}"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an NLP expert specializing in entity extraction and query analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            entities_data = json.loads(result_text)
            
            logger.info(f"✅ Extracted entities: {len(entities_data.get('entities', []))} entities, "
                       f"{len(entities_data.get('concepts', []))} concepts, "
                       f"{len(entities_data.get('actions', []))} actions")
            
            return entities_data
                
        except Exception as e:
            logger.error(f"❌ Error extracting entities: {e}")
            return {
                "entities": [],
                "concepts": [],
                "actions": []
            }
    
    async def generate_targeted_queries(
        self,
        original_query: str,
        entities: Dict[str, List[str]],
        max_queries: int = 3
    ) -> List[str]:
        """
        Генерирует таргетированные запросы на основе извлеченных сущностей.
        Фокусируется на конкретных аспектах вопроса.
        """
        try:
            logger.info(f"🔍 Generating targeted queries based on entities...")
            
            entities_str = json.dumps(entities, indent=2)
            
            prompt = f"""Generate {max_queries} targeted search queries based on the original question and extracted entities.

Original Question: {original_query}

Extracted Information:
{entities_str}

Your task:
1. Create {max_queries} different search queries
2. Each query should focus on a specific aspect using the extracted entities and concepts
3. Combine entities with concepts in different ways
4. Use natural language that would match document content

Examples:
- If entities=["Glencore"] and concepts=["price", "payment"], generate:
  * "Glencore contract price terms"
  * "payment conditions in Glencore agreement"
  * "pricing mechanism Glencore"

Return ONLY a JSON object with format: {{"targeted_queries": ["query1", "query2", "query3"]}}"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at creating targeted search queries using entity information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=250,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            parsed = json.loads(result_text)
            targeted_queries = parsed.get("targeted_queries", [])
            
            logger.info(f"✅ Generated {len(targeted_queries)} targeted queries")
            return targeted_queries[:max_queries]
                
        except Exception as e:
            logger.error(f"❌ Error generating targeted queries: {e}")
            return [original_query]
    
    async def _generate_query_for_aspect(
        self,
        original_query: str,
        missing_aspect: str
    ) -> str:
        """
        Генерирует точный поисковый запрос для конкретного недостающего аспекта.
        Например, если missing_aspect="price", а original_query="chrome deal buyer seller",
        генерирует "chrome deal price" или "chrome price terms"
        """
        try:
            prompt = f"""Given the original user question and a missing aspect, generate a precise search query to find information about that specific aspect.

Original Question: {original_query}
Missing Aspect: {missing_aspect}

Generate a search query that:
1. Includes the main context from the original question (e.g., "chrome deal", "contract")
2. Focuses specifically on finding the missing aspect (e.g., "price")
3. Uses natural language that would match document content
4. Keeps it concise (5-10 words max)

Examples:
- Original: "chrome deal buyer seller quality" + Missing: "price" → "chrome deal price"
- Original: "information on contract" + Missing: "payment terms" → "contract payment terms"
- Original: "Glencore agreement" + Missing: "delivery" → "Glencore delivery terms"

Return ONLY the search query, nothing else:"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at generating precise search queries for finding specific information in documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip()
            # Убираем кавычки если есть
            result = result.strip('"\'')
            logger.info(f"✅ Generated aspect query: {result}")
            return result
                
        except Exception as e:
            logger.error(f"❌ Error generating query for aspect '{missing_aspect}': {e}")
            # Fallback
            if "deal" in original_query.lower():
                return f"{missing_aspect} chrome deal"
            elif "contract" in original_query.lower():
                return f"{missing_aspect} contract"
            else:
                return f"{original_query[:50]} {missing_aspect}"


class DocumentQualityValidator:
    """
    Валидатор для проверки качества найденных документов.
    Использует LLM для определения, содержат ли документы ответ на вопрос пользователя.
    """
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
    
    async def validate_documents_quality(
        self,
        query: str,
        documents: List[RetrievedDocument],
        max_llm_score: float
    ) -> Dict[str, Any]:
        """
        Проверяет качество найденных документов.
        
        Returns:
            dict with keys:
            - has_answer: bool - содержат ли документы ответ
            - confidence: float - уверенность 0-10
            - missing_aspects: List[str] - что отсутствует в документах
            - recommendation: str - рекомендация (continue/stop/broaden)
        """
        try:
            logger.info(f"🔍 Validating document quality for query: {query[:100]}...")
            
            # Подготавливаем контент документов для анализа
            # КРИТИЧНО: Проверяем ВСЕ документы или максимум 30 (вместо 15) для гарантии полноты
            docs_content = []
            docs_to_check = documents[:min(30, len(documents))]  # Проверяем до 30 или все если меньше
            logger.info(f"🔍 Validating {len(docs_to_check)} documents (out of {len(documents)} total) for completeness")
            
            for i, doc in enumerate(docs_to_check, 1):
                metadata_str = ""
                if doc.metadata:
                    file_name = doc.metadata.get("file_name", "")
                    page_label = doc.metadata.get("page_label", "")
                    if file_name:
                        metadata_str = f" (Source: {file_name}, Page: {page_label})"
                
                llm_score = doc.metadata.get("llm_rerank_score_raw", 0.0) if doc.metadata else 0.0
                # Увеличиваем preview до 800 символов для более полной проверки (особенно для quality specs)
                docs_content.append(f"Document {i}{metadata_str} [Score: {doc.score:.3f}, LLM: {llm_score:.1f}/10]:\n{doc.content[:800]}")
            
            docs_text = "\n\n".join(docs_content)
            
            system_prompt = """You are a STRICT but INTELLIGENT document quality validator. Your task is to analyze whether the provided documents contain ALL aspects requested in the user's question.

IMPORTANT RULES:
1. Extract ALL specific aspects/fields requested in the question (e.g., buyer, seller, price, quality, delivery terms, payment terms)
2. Check if EACH aspect is present in the documents
3. Understand that:
   - "buyer" means: buyer company name, buyer organization, purchaser, or any entity acting as buyer
   - "seller" means: seller company name, seller organization, vendor, or any entity acting as seller
   - "quality" means: quality specifications, chemical composition, technical specs, percentages, ratios, standards
     * Examples: "Cr2O3: 40%", "Fe ratio: 1.40:1", "SiO2: 12% Max", "Cr/Fe ratio", "Moisture: 5%", "Sizing: 10mm-35mm"
     * Quality specs DON'T need the word "quality" - chemical percentages, ratios, specifications ARE quality
   - "price" means: pricing information, cost, price per unit, payment amount, financial terms
   - "delivery terms" means: shipping terms, delivery location, port, destination, shipment dates
   - "payment terms" means: payment conditions, letter of credit, payment method, payment schedule
   - These don't need to have the exact words - specific information counts!
4. If ANY aspect is missing - list it in missing_aspects
5. Recommendation should be "continue" if ANY aspect is missing, even if confidence is high
6. Only recommend "stop" if ALL aspects are clearly present in the documents

Be STRICT but SMART: Missing even one requested aspect means we should continue searching, BUT if you see specific information (company names, quality specs, etc.), count that as found."""

            user_prompt = f"""User's Question: {query}

Top Retrieved Documents:
{docs_text}

Maximum LLM relevance score from all documents: {max_llm_score:.1f}/10

CRITICAL TASK:
1. First, extract ALL specific aspects/fields requested in the question (e.g., if question asks for "buyer, seller, price, quality, delivery terms, payment terms" - extract all 6)
2. Check if EACH aspect is present in the documents above
3. List ALL missing aspects in missing_aspects array
4. Confidence should reflect completeness (lower if aspects missing)
5. Recommendation MUST be "continue" if ANY aspect is missing, even if confidence is 8-10

IMPORTANT EXAMPLES:
- If documents contain "Jiangsu Provincial Foreign Trade Corporation" or "Xiamen Xiangyu Logistics Group" → BUYER IS FOUND (these are buyer companies)
- If documents contain "Fujax UK Limited" → SELLER IS FOUND (this is seller company)
- If documents have company names but don't explicitly say "buyer" or "seller" → STILL COUNT as buyer/seller found if context is clear
- If documents mention "for SC-240-FJK" or "for SC-294-FJK" with company names → These are buyer/seller for those contracts
- QUALITY SPECS examples (these COUNT as quality found):
  * "Cr2O3: 40% basis, 36% Min" → QUALITY FOUND ✅
  * "Cr/Fe ratio: 1.40:1 Typical, 1.35:1 Min" → QUALITY FOUND ✅
  * "SiO2: 12% Max, Al2O3: 16% Max" → QUALITY FOUND ✅
  * "Cr203: 40% basis" (note: Cr203 = Cr2O3, same thing) → QUALITY FOUND ✅
  * Any chemical composition, percentages, ratios, technical specifications → QUALITY FOUND ✅
  * DON'T require the word "quality" - chemical specs ARE quality information!

Example 1: Question asks for "buyer, seller, price, quality" and documents have:
  - "Jiangsu Provincial Foreign Trade Corporation (for SC-240-FJK)" 
  - "Fujax UK Limited"
  - "Cr2O3: 42% basis, 38% Min; Cr/Fe ratio: 1.65:1"
  → Found: buyer, seller, quality | Missing: price | Recommendation: continue

Example 2: Question asks for "buyer, seller, price, quality" and documents have:
  - Company names (buyer/seller)
  - "Cr203: 40% basis, 36% Min; Fe ratio: 1.40:1"
  - Price information
  → ALL FOUND | Recommendation: stop

Example 3: Question asks for "quality" and document has "Cr2O3: 40%, Fe: 1.40:1, SiO2: 12%" → QUALITY FOUND (even without word "quality")

Return ONLY JSON: {{
    "has_answer": true/false,
    "confidence": 0-10,
    "missing_aspects": ["aspect1", "aspect2"],
    "requested_aspects": ["aspect1", "aspect2", "aspect3"],
    "found_aspects": ["aspect1"],
    "recommendation": "continue/stop/broaden",
    "reasoning": "brief explanation"
}}"""

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            validation_result = json.loads(result_text)
            
            logger.info(f"✅ Validation result: has_answer={validation_result.get('has_answer')}, "
                       f"confidence={validation_result.get('confidence')}/10, "
                       f"recommendation={validation_result.get('recommendation')}")
            
            if validation_result.get('missing_aspects'):
                logger.info(f"⚠️ Missing aspects: {validation_result.get('missing_aspects')}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Error validating document quality: {e}")
            # Fallback: используем LLM score как индикатор
            return {
                "has_answer": max_llm_score >= 7.0,
                "confidence": max_llm_score,
                "missing_aspects": [],
                "recommendation": "stop" if max_llm_score >= 7.0 else "continue",
                "reasoning": "Fallback validation based on LLM scores"
            }


class IterativeDocumentRetriever:
    """
    Слой для интеллектуального итеративного поиска документов с многоуровневой проверкой качества.
    
    Работает следующим образом:
    1. Первая итерация: выполняет поиск с оригинальным запросом
    2. Валидирует качество найденных документов с помощью LLM
    3. Если качество низкое - анализирует недостающую информацию
    4. Генерирует новые запросы для поиска пропущенных данных
    5. Повторяет поиск с новыми запросами до достижения хорошего качества
    6. Использует fallback стратегии (широкий поиск, разбиение на подвопросы)
    7. Объединяет все результаты, удаляя дубликаты
    
    Критерии остановки:
    - Validation показывает что ответ найден (confidence >= 7/10)
    - Достигнут максимум итераций (5)
    - Несколько итераций подряд не дают новых результатов (2 раза)
    """
    
    def __init__(self, base_retriever, query_rephraser: QueryRephraser, validator: DocumentQualityValidator):
        self.base_retriever = base_retriever
        self.query_rephraser = query_rephraser
        self.validator = validator
        self.max_iterations = 5  # Увеличили с 3 до 5
        self.min_new_docs_threshold = 1  # Минимум новых документов для продолжения поиска
        self.low_relevance_threshold = 5.0  # Порог низкой релевантности LLM
        self.high_confidence_threshold = 7.0  # Порог высокой уверенности для остановки
        self.max_failed_iterations = 2  # Максимум неудачных итераций подряд
    
    async def retrieve_documents_iteratively(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.5,
        document_ids: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """Итеративно ищет документы с многоуровневой проверкой качества"""
        logger.info(f"🔄 Starting ENHANCED iterative document retrieval for: {query[:100]}...")
        
        all_documents = {}  # doc_id -> RetrievedDocument
        iteration_stats = []
        failed_iterations_count = 0  # Счетчик неудачных итераций подряд
        used_search_strategies = set()  # Отслеживаем использованные стратегии
        
        # Итерация 1: Первый поиск с оригинальным запросом
        logger.info(f"🔍 Iteration 1: Initial search with original query...")
        docs = await self.base_retriever._retrieve_documents_base(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            document_ids=document_ids
        )
        
        # Добавляем найденные документы
        for doc in docs:
            if doc.doc_id not in all_documents:
                all_documents[doc.doc_id] = doc
        
        # Получаем максимальный LLM score
        max_llm_score = self._get_max_llm_score(all_documents)
        
        iteration_stats.append({
            "iteration": 1,
            "query": query,
            "strategy": "original",
            "found_docs": len(docs),
            "new_docs": len(docs),
            "total_unique": len(all_documents),
            "max_llm_score": max_llm_score
        })
        logger.info(f"✅ Iteration 1: Found {len(docs)} docs | Max LLM: {max_llm_score:.1f}/10")
        
        # Валидация качества после первой итерации
        # Берем топ-30 документов для более полной проверки (новые документы могут быть дальше в списке)
        all_docs_list = list(all_documents.values())
        all_docs_list.sort(key=lambda x: x.score, reverse=True)
        validation = await self.validator.validate_documents_quality(
            query=query,
            documents=all_docs_list[:30],  # Увеличили с 20 до 30 для гарантии
            max_llm_score=max_llm_score
        )
        
        missing_aspects = validation.get('missing_aspects', [])
        requested_aspects = validation.get('requested_aspects', [])
        found_aspects = validation.get('found_aspects', [])
        
        logger.info(f"📊 Quality validation: confidence={validation.get('confidence', 0):.1f}/10, "
                   f"recommendation={validation.get('recommendation')}")
        
        if missing_aspects:
            logger.warning(f"⚠️ Missing aspects detected: {missing_aspects}")
            if requested_aspects:
                logger.info(f"📋 Requested: {requested_aspects} | Found: {found_aspects} | Missing: {missing_aspects}")
        
        # КРИТИЧНО: Не останавливаемся если есть missing_aspects, даже при высокой confidence!
        if missing_aspects:
            logger.info(f"🔄 Missing aspects found: {missing_aspects}. Will continue searching despite confidence={validation.get('confidence', 0):.1f}/10")
        elif validation.get('confidence', 0) >= self.high_confidence_threshold:
            # Останавливаемся только если НЕТ missing_aspects И confidence высокая
            logger.info(f"✅ High confidence ({validation.get('confidence', 0):.1f}/10) AND no missing aspects! Stopping search.")
            final_docs = list(all_documents.values())
            final_docs.sort(key=lambda x: x.score, reverse=True)
            return final_docs[:top_k]
        
        used_search_strategies.add("original")
        
        # Дополнительные итерации с улучшенной логикой
        for i in range(2, self.max_iterations + 1):
            # Проверяем, не превысили ли мы лимит неудачных итераций
            if failed_iterations_count >= self.max_failed_iterations:
                logger.warning(f"🛑 Stopping: {failed_iterations_count} failed iterations in a row")
                break
            
            # Определяем стратегию поиска на основе текущего состояния
            current_max_llm = self._get_max_llm_score(all_documents)
            # Используем последнюю validation (она обновляется внутри цикла после итерации)
            recommendation = validation.get('recommendation', 'continue')
            missing_aspects = validation.get('missing_aspects', [])
            current_missing = None  # Для отслеживания какого аспекта ищем
            
            # ПРИОРИТЕТ 1: Поиск недостающих аспектов (САМАЯ ВАЖНАЯ СТРАТЕГИЯ!)
            if missing_aspects:
                # Считаем какие аспекты мы уже искали в предыдущих итерациях
                previous_searches_raw = [s.get('aspect_searched', '') for s in iteration_stats 
                                        if s.get('strategy') == 'missing_aspects' and s.get('aspect_searched')]
                
                # Извлекаем чистые названия аспектов (без "(alt: ...)")
                previous_searches_clean = []
                for prev in previous_searches_raw:
                    if prev:
                        # Извлекаем основное название аспекта (до "(alt:")
                        clean_name = prev.split(' (alt:')[0].strip()
                        if clean_name not in previous_searches_clean:
                            previous_searches_clean.append(clean_name)
                
                # Находим аспекты которые еще НЕ искали
                aspects_to_search = [asp for asp in missing_aspects if asp not in previous_searches_clean]
                
                # Если есть аспекты которые еще не искали - ищем их
                if aspects_to_search:
                    # Берем первый еще не искавшийся аспект
                    current_missing = aspects_to_search[0]
                    logger.info(f"🔍 Iteration {i}: Using MISSING ASPECTS search strategy...")
                    logger.info(f"🎯 Searching for missing aspect: {current_missing}")
                    logger.info(f"📋 Previously searched: {previous_searches_clean}, Still missing: {aspects_to_search}")
                    
                    # Генерируем точный запрос для этого аспекта через LLM
                    try:
                        new_query = await self.query_rephraser._generate_query_for_aspect(query, current_missing)
                    except Exception as e:
                        logger.warning(f"⚠️ Error generating query for aspect: {e}, using fallback")
                        # Fallback: простая комбинация
                        if "deal" in query.lower():
                            new_query = f"{current_missing} chrome deal"
                        elif "contract" in query.lower():
                            new_query = f"{current_missing} contract"
                        else:
                            context_words = [w for w in query.lower().split() if len(w) > 4][:3]
                            context = " ".join(context_words) if context_words else query[:50]
                            new_query = f"{context} {current_missing}"
                    
                    strategy = "missing_aspects"
                    score_threshold = max(0.1, score_threshold * 0.5)  # Очень агрессивно снижаем порог
                    logger.info(f"🔄 Generated targeted query for '{current_missing}': {new_query[:100]}...")
                else:
                    # Все аспекты уже искали минимум один раз, но они все еще missing
                    # Попробуем использовать альтернативные термины
                    total_attempts = len(previous_searches_raw)
                    max_attempts_per_aspect = 5  # Пробуем до 5 альтернатив для каждого аспекта
                    
                    if total_attempts < len(missing_aspects) * max_attempts_per_aspect:
                        # Используем альтернативные термины
                        aspect_alternatives = {
                            "buyer": ["buyer", "purchaser", "buyer company", "buyer name", "buyer organization"],
                            "seller": ["seller", "vendor", "seller company", "seller name", "seller organization"],
                            "quality": [
                                "quality", 
                                "quality specifications", 
                                "chemical composition", 
                                "specifications",
                                "Cr2O3", 
                                "Cr/Fe ratio",
                                "technical specifications",
                                "quality specs",
                                "Cr203",
                                "chrome quality specs"
                            ],
                            "price": [
                                "price",
                                "pricing",
                                "cost",
                                "price per unit",
                                "USD per MT",
                                "pricing terms",
                                "price clause",
                                "price terms"
                            ]
                        }
                        # Берем первый все еще missing аспект и пробуем альтернативный термин
                        still_missing = missing_aspects[0] if missing_aspects else None
                        if still_missing and still_missing in aspect_alternatives:
                            alternatives = aspect_alternatives[still_missing]
                            
                            # Считаем сколько раз уже искали этот конкретный аспект (включая альтернативы)
                            searched_for_this = len([s for s in previous_searches_raw 
                                                    if s and (s == still_missing or s.startswith(f"{still_missing} (alt:"))])
                            
                            if searched_for_this < len(alternatives):
                                current_alt = alternatives[searched_for_this]
                                current_missing = f"{still_missing} (alt: {current_alt})"
                                
                                # Генерируем запрос с альтернативным термином
                                if "deal" in query.lower():
                                    new_query = f"chrome deal {current_alt}"
                                elif "contract" in query.lower():
                                    new_query = f"{current_alt} contract"
                                else:
                                    context_words = [w for w in query.lower().split() if len(w) > 4][:2]
                                    context = " ".join(context_words) if context_words else "chrome"
                                    new_query = f"{context} {current_alt}"
                                
                                strategy = "missing_aspects"
                                logger.info(f"🔄 Trying alternative term #{searched_for_this + 1}/{len(alternatives)} for '{still_missing}': '{current_alt}'")
                                logger.info(f"🔄 Generated query: {new_query}")
                                score_threshold = max(0.05, score_threshold * 0.4)  # Еще более агрессивно снижаем порог
                            else:
                                # Все альтернативы испробованы для этого аспекта
                                logger.warning(f"⚠️ All {len(alternatives)} alternatives tried for '{still_missing}'. Moving to other strategies...")
                                missing_aspects = []
                                strategy = None
                        else:
                            missing_aspects = []
                            strategy = None
                    else:
                        # Превысили лимит попыток, используем другую стратегию
                        logger.warning(f"⚠️ Exceeded retry limit ({total_attempts} attempts) for missing aspects. Trying other strategies...")
                        missing_aspects = []
                        strategy = None
            
            # Продолжаем выбор стратегии только если не выбрали missing_aspects
            if not strategy:
                if recommendation == "broaden" or (current_max_llm < 3.0 and "broad" not in used_search_strategies):
                    # Стратегия 1: Широкий поиск
                    logger.info(f"🔍 Iteration {i}: Using BROAD search strategy...")
                    new_query = await self._generate_broad_query(query, all_documents)
                    strategy = "broad"
                    score_threshold = max(0.1, score_threshold * 0.5)  # Агрессивно снижаем порог
                
                elif current_max_llm < 4.0 and "targeted" not in used_search_strategies:
                    # Стратегия 2: Таргетированный поиск на основе сущностей (НОВАЯ СТРАТЕГИЯ)
                    logger.info(f"🔍 Iteration {i}: Using TARGETED search with entity extraction...")
                    entities = await self.query_rephraser.extract_key_entities(query)
                    targeted_queries = await self.query_rephraser.generate_targeted_queries(
                        original_query=query,
                        entities=entities,
                        max_queries=3
                    )
                    # Берем первый таргетированный запрос
                    new_query = targeted_queries[0] if targeted_queries else query
                    strategy = "targeted"
                    score_threshold = max(0.2, score_threshold * 0.8)
                
                elif current_max_llm < 5.0 and "decomposed" not in used_search_strategies:
                    # Стратегия 3: Разбиение на подвопросы (НОВАЯ СТРАТЕГИЯ)
                    logger.info(f"🔍 Iteration {i}: Using DECOMPOSITION strategy...")
                    sub_queries = await self.query_rephraser.decompose_complex_query(query, max_subqueries=3)
                    # Берем первый подвопрос который еще не использовали
                    used_decomposed = len([s for s in iteration_stats if s.get('strategy') == 'decomposed'])
                    if used_decomposed < len(sub_queries):
                        new_query = sub_queries[used_decomposed]
                    else:
                        new_query = sub_queries[0] if sub_queries else query
                    strategy = "decomposed"
                
                elif current_max_llm < 6.0 and "missing_info" not in used_search_strategies:
                    # Стратегия 4: Поиск недостающей информации
                    logger.info(f"🔍 Iteration {i}: Analyzing missing information...")
                    missing_queries = await self.query_rephraser.analyze_and_find_missing_info(
                        original_query=query,
                        found_documents=list(all_documents.values())[:10],
                        max_iterations=self.max_iterations
                    )
                    if len(missing_queries) > 1:
                        query_idx = min(i - 2, len(missing_queries) - 2)
                        new_query = missing_queries[query_idx + 1]
                    else:
                        new_query = query
                    strategy = "missing_info"
                
                else:
                    # Стратегия 5: Простые вариации запроса (fallback)
                    logger.info(f"🔍 Iteration {i}: Generating query variations...")
                    variations = await self.query_rephraser._generate_simple_variations(query, max_iterations=5)
                    # Берем вариацию которую еще не использовали
                    used_count = len([s for s in iteration_stats if s.get('strategy') == 'variation'])
                    if used_count < len(variations):
                        new_query = variations[min(used_count, len(variations)-1)]
                    else:
                        logger.info(f"⚠️ All strategies exhausted, stopping...")
                        break
                    strategy = "variation"
            
            used_search_strategies.add(strategy)
            
            logger.info(f"🔄 Strategy: {strategy} | Query: {new_query[:100]}...")
            
            # Выполняем поиск с новым запросом
            new_docs = await self.base_retriever._retrieve_documents_base(
                query=new_query,
                top_k=top_k,
                score_threshold=score_threshold,
                document_ids=document_ids
            )
            
            # Подсчитываем новые документы
            new_docs_count = 0
            for doc in new_docs:
                if doc.doc_id not in all_documents:
                    all_documents[doc.doc_id] = doc
                    new_docs_count += 1
                else:
                    # Обновляем score если новый документ имеет лучший score
                    existing_doc = all_documents[doc.doc_id]
                    if doc.score > existing_doc.score:
                        all_documents[doc.doc_id] = doc
            
            # Обновляем максимальный LLM score
            max_llm_score = self._get_max_llm_score(all_documents)
            
            # Сохраняем статистику итерации
            stat_entry = {
                "iteration": i,
                "query": new_query,
                "strategy": strategy,
                "found_docs": len(new_docs),
                "new_docs": new_docs_count,
                "total_unique": len(all_documents),
                "max_llm_score": max_llm_score
            }
            # Сохраняем какой аспект искали для missing_aspects стратегии
            if strategy == "missing_aspects" and current_missing is not None:
                stat_entry["aspect_searched"] = current_missing
            iteration_stats.append(stat_entry)
            
            logger.info(f"✅ Iteration {i}: Found {len(new_docs)} docs, {new_docs_count} new | "
                       f"Total: {len(all_documents)} | Max LLM: {max_llm_score:.1f}/10")
            
            # Проверяем качество после каждой итерации
            if new_docs_count > 0 or i == 2:  # Валидируем если есть новые документы или это вторая итерация
                # Берем топ-30 документов отсортированных по score (для гарантии что проверим все новые)
                all_docs_list = list(all_documents.values())
                all_docs_list.sort(key=lambda x: x.score, reverse=True)
                validation = await self.validator.validate_documents_quality(
                    query=query,
                    documents=all_docs_list[:30],  # Увеличили с 20 до 30 для гарантии
                    max_llm_score=max_llm_score
                )
                
                confidence = validation.get('confidence', 0)
                missing_aspects = validation.get('missing_aspects', [])
                requested_aspects = validation.get('requested_aspects', [])
                found_aspects = validation.get('found_aspects', [])
                
                logger.info(f"📊 Validation after iteration {i}: confidence={confidence:.1f}/10")
                
                if missing_aspects:
                    logger.warning(f"⚠️ Still missing aspects after iteration {i}: {missing_aspects}")
                    if requested_aspects:
                        logger.info(f"📋 Requested: {requested_aspects} | Found: {found_aspects} | Missing: {missing_aspects}")
                elif confidence >= self.high_confidence_threshold:
                    # Останавливаемся только если НЕТ missing_aspects И confidence высокая
                    logger.info(f"✅ HIGH CONFIDENCE ({confidence:.1f}/10) AND NO MISSING ASPECTS! Stopping search.")
                    break
            
            # Обновляем счетчик неудачных итераций
            if new_docs_count == 0:
                failed_iterations_count += 1
                logger.warning(f"⚠️ No new documents in iteration {i} (failed: {failed_iterations_count}/{self.max_failed_iterations})")
            else:
                failed_iterations_count = 0  # Сбрасываем если нашли новые документы
        
        # Конвертируем в список и сортируем по score
        final_docs = list(all_documents.values())
        final_docs.sort(key=lambda x: x.score, reverse=True)
        
        # Логируем детальную статистику
        final_max_llm = self._get_max_llm_score(all_documents)
        logger.info(f"📊 ENHANCED iterative search completed:")
        for stat in iteration_stats:
            logger.info(f"  Iter {stat['iteration']} ({stat['strategy']}): "
                       f"{stat['found_docs']} docs ({stat['new_docs']} new) | "
                       f"LLM: {stat['max_llm_score']:.1f}/10 | "
                       f"Query: {stat['query'][:60]}...")
        logger.info(f"  📈 Final: {len(final_docs)} unique docs | Best LLM score: {final_max_llm:.1f}/10")
        
        return final_docs[:top_k]
    
    def _get_max_llm_score(self, documents_dict: Dict[str, RetrievedDocument]) -> float:
        """Получает максимальный LLM score из документов"""
        max_score = 0.0
        for doc in documents_dict.values():
            if doc.metadata and "llm_rerank_score_raw" in doc.metadata:
                max_score = max(max_score, doc.metadata["llm_rerank_score_raw"])
        return max_score
    
    async def _generate_broad_query(
        self,
        original_query: str,
        found_documents: Dict[str, RetrievedDocument]
    ) -> str:
        """Генерирует широкий запрос для более агрессивного поиска"""
        # Используем метод из QueryRephraser
        broad_queries = await self.query_rephraser._generate_broad_search_queries(
            original_query=original_query,
            max_iterations=2
        )
        return broad_queries[1] if len(broad_queries) > 1 else original_query


class GenerateAnswerInteractor:
    """Generate answers using RAG with Hybrid Search (Semantic + Keyword) + Reranking + Enhanced Iterative Search"""
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
        self.query_rephraser = QueryRephraser(self.openai_client)
        self.quality_validator = DocumentQualityValidator(self.openai_client)  # NEW: Quality validator
        self.model = "gpt-4o-mini"
        self.max_tokens = 2000
        self.temperature = 0.3
        self.use_reranking = True
        self.use_keyword_search = True
        self.use_semantic_search = True  # Enable semantic search by default
        self.use_iterative_search = True  # Enable iterative search by default

    async def keyword_search_lancedb(
        self,
        query: str,
        top_k: int = 15,
        document_ids: Optional[List[str]] = None
    ) -> List[tuple[str, float]]:
        """Keyword search using LanceDB Full-Text Search (Tantivy)"""
        try:
            fts_results = self.doc_store.query(query, top_k=top_k, doc_ids=document_ids)
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
        top_k: int = 10,
        score_threshold: float = 0.5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedDocument]:
        """Retrieve relevant documents using Hybrid Search (Semantic + Keyword) + Reranking + Iterative Search"""
        try:
            if document_ids:
                logger.info(f"🔍 Filtering search by document IDs: {document_ids}")
            else:
                logger.info(f"🔍 Searching in all documents")
            
            # Если включен итеративный поиск, используем его
            if self.use_iterative_search:
                logger.info("🔄 Using ENHANCED iterative search with quality validation...")
                iterative_retriever = IterativeDocumentRetriever(
                    base_retriever=self,
                    query_rephraser=self.query_rephraser,
                    validator=self.quality_validator
                )
                return await iterative_retriever.retrieve_documents_iteratively(
                    query=query,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    document_ids=document_ids
                )
            
            # Базовый поиск без итераций
            return await self._retrieve_documents_base(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                document_ids=document_ids
            )
            
        except Exception as e:
            logger.error(f"❌ Error in retrieve_documents: {e}", exc_info=True)
            return []

    async def _retrieve_documents_base(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedDocument]:
        """Базовый метод поиска документов без итераций"""
        try:
            # Initialize results
            semantic_docs = []
            keyword_docs = []
            all_doc_ids = set()
            
            # Step 1: Semantic Search (Qdrant) - INDEPENDENT
            if self.use_semantic_search:
                try:
                    logger.info(f"🔍 Step 1: Semantic search for: {query[:100]}...")
                    query_embedding_docs = self.embedding.invoke(query)
                    if query_embedding_docs and getattr(query_embedding_docs[0], "embedding", None):
                        query_embedding = query_embedding_docs[0].embedding
                        initial_k = top_k * 3 if self.use_keyword_search or self.use_reranking else top_k * 2
                        embeddings, similarities, doc_ids = await self.vector_store.query(
                            embedding=query_embedding,
                            top_k=initial_k,
                            ids=document_ids,
                        )
                        if doc_ids:
                            logger.info(f"✅ Semantic search: found {len(doc_ids)} documents")
                            semantic_docs = list(zip(doc_ids, similarities))
                            all_doc_ids.update(doc_ids)
                        else:
                            logger.warning("⚠️ No documents found in semantic search")
                    else:
                        logger.error("❌ Failed to create query embedding")
                except Exception as e:
                    logger.error(f"❌ Error in semantic search: {e}")
            else:
                logger.info("🔍 Semantic search disabled")
            
            # Step 2: Keyword Search (LanceDB) - INDEPENDENT
            keyword_scores_dict = {}
            if self.use_keyword_search:
                try:
                    logger.info(f"🔤 Step 2: LanceDB Full-Text Search...")
                    initial_k = top_k * 3 if self.use_semantic_search or self.use_reranking else top_k * 2
                    fts_results = await self.keyword_search_lancedb(query, top_k=initial_k, document_ids=document_ids)
                    if fts_results:
                        logger.info(f"✅ FTS found {len(fts_results)} documents")
                        keyword_docs = fts_results
                        for fts_doc_id, fts_score in fts_results:
                            keyword_scores_dict[fts_doc_id] = fts_score
                            all_doc_ids.add(fts_doc_id)
                    else:
                        logger.warning("⚠️ No documents found in keyword search")
                except Exception as e:
                    logger.error(f"❌ Error in keyword search: {e}")
            
            # Check if we have any results from either search
            if not all_doc_ids:
                logger.warning("❌ No documents found in any search method")
                return []
            
            # Step 3: Retrieve full documents from LanceDB - INDEPENDENT
            try:
                logger.info(f"📚 Retrieving {len(all_doc_ids)} unique documents from LanceDB...")
                all_docs = self.doc_store.get(list(all_doc_ids))
                doc_dict = {}
                for doc in all_docs:
                    doc_id = getattr(doc, "id_", None)
                    if doc_id is None:
                        doc_id = getattr(doc, "doc_id", None)
                    if doc_id is not None:
                        doc_dict[doc_id] = doc
                logger.info(f"✅ Retrieved {len(doc_dict)} documents from LanceDB")
            except Exception as e:
                logger.error(f"❌ Error retrieving documents from LanceDB: {e}")
                return []
            
            # Step 4: Combine results from both searches
            semantic_scores_dict = dict(semantic_docs)
            retrieved_docs = []
            
            for doc_id in all_doc_ids:
                if doc_id not in doc_dict:
                    logger.warning(f"⚠️ Document {doc_id} not found in LanceDB")
                    continue
                    
                doc = doc_dict[doc_id]
                semantic_score = semantic_scores_dict.get(doc_id, 0.0)
                keyword_score = keyword_scores_dict.get(doc_id, 0.0)
                
                # Calculate combined score based on available search methods
                if semantic_score > 0 and keyword_score > 0:
                    # Both searches found this document - use weighted combination
                    combined_score = semantic_weight * semantic_score + keyword_weight * keyword_score
                elif semantic_score > 0:
                    # Only semantic search found this document
                    combined_score = semantic_score
                elif keyword_score > 0:
                    # Only keyword search found this document
                    combined_score = keyword_score
                else:
                    # Fallback (shouldn't happen)
                    combined_score = 0.0
                
                # Get document content and metadata
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
            
            # Sort by combined score
            retrieved_docs.sort(key=lambda x: x.score, reverse=True)
            
            # Log statistics
            if retrieved_docs:
                avg_semantic = sum(d.semantic_score for d in retrieved_docs) / len(retrieved_docs)
                avg_keyword = sum(d.keyword_score for d in retrieved_docs) / len(retrieved_docs)
                logger.info(f"✅ Hybrid scores: semantic={avg_semantic:.3f}, keyword={avg_keyword:.3f}")
            
            # Apply score threshold
            filtered_docs = [doc for doc in retrieved_docs if doc.score >= score_threshold]
            if not filtered_docs:
                logger.warning(f"⚠️ No documents above threshold {score_threshold}")
                filtered_docs = retrieved_docs[:top_k]
            logger.info(f"✅ After filtering: {len(filtered_docs)} documents")
            
            # Step 5: LLM Reranking (if enabled)
            max_llm_score = 10.0
            if self.use_reranking and len(filtered_docs) > top_k:
                try:
                    logger.info(f"🔄 Step 3: LLM-based reranking (Kotaemon-style)...")
                    reranked_docs, max_llm_score = await self.reranker.rerank(query, filtered_docs, top_k)
                    logger.info(f"✅ Reranked to {len(reranked_docs)} documents | Max LLM score: {max_llm_score:.1f}/10")
                    filtered_docs = reranked_docs
                except Exception as e:
                    logger.error(f"❌ Error in reranking: {e}")
            
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
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[DocumentSchema, None]:
        """
        Stream response with document retrieval and answer generation
        """
        start_time = time.time()
        history = history or []
        try:
            if self.use_semantic_search and self.use_keyword_search:
                search_mode = "Hybrid Search (Qdrant + LanceDB FTS)"
            elif self.use_semantic_search:
                search_mode = "Semantic Search (Qdrant only)"
            elif self.use_keyword_search:
                search_mode = "Keyword Search (LanceDB FTS only)"
            else:
                search_mode = "No search enabled"
            yield DocumentSchema(
                content=f"🔍 Starting {search_mode}...",
                channel="debug"
            )
            retrieved_docs = await self.retrieve_documents(
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
                    content="I couldn't find any relevant information in the available documents. Could you please rephrase your question or provide more details?",
                    channel="chat"
                )
                return
            
            # Проверяем качество найденных документов
            max_llm_score = 0.0
            for doc in retrieved_docs:
                if doc.metadata and "llm_rerank_score_raw" in doc.metadata:
                    max_llm_score = max(max_llm_score, doc.metadata["llm_rerank_score_raw"])
            
            # Если все документы имеют низкую релевантность, предупреждаем пользователя
            if max_llm_score < 3.0:
                yield DocumentSchema(
                    content=f"⚠️ **LOW RELEVANCE WARNING**\n\nThe retrieved documents have very low relevance scores (Max: {max_llm_score:.1f}/10). The information may not be accurate or complete. Please consider rephrasing your question or providing more specific details.",
                    channel="info"
                )
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
6. **Handle low relevance gracefully** - If documents seem irrelevant, acknowledge this and suggest alternatives

RESPONSE FORMAT:
- Answer the question based ONLY on the retrieved documents
- Use [1], [2], etc. to cite specific sources
- If context doesn't contain the answer, say so clearly
- If documents seem irrelevant to the question, acknowledge this limitation
- Be helpful and conversational while staying accurate
- Suggest how the user might rephrase their question for better results"""
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
                top_k=10
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