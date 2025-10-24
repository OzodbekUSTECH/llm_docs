"""
Структурированный чанкер для разных типов документов.
Не использует LLM, работает на основе паттернов и правил.
"""
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from transformers import AutoTokenizer
from app.utils.enums import DocumentType
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Структура чанка с метаданными"""
    text: str
    metadata: Dict[str, Any]
    index: int
    token_count: int


class DocumentChunker:
    """
    Умный чанкер, который делит документы по структуре в зависимости от типа.
    
    Стратегии:
    - CONTRACT: По clauses (Article 1., 1.1, ARTICLE 1, etc.)
    - INVOICE: По смысловым секциям (header, items, totals, footer)
    - BL/LC: По полям документа
    - OTHER: По параграфам с умным объединением
    """
    
    def __init__(self, tokenizer_model: str = "intfloat/e5-base-v2", max_tokens: int = 512, overlap_tokens: int = 50):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        # Паттерны для распознавания структуры контрактов
        self.contract_patterns = [
            # Article 1., Article 1.1, ARTICLE 1, Article I
            r'^(?:ARTICLE|Article|article)\s+(?:\d+\.?\d*|[IVXLCDM]+)\.?\s*[-:\.]?\s*',
            # 1. Title, 1.1 Title, 1.1.1 Title
            r'^\d+(?:\.\d+)*\.?\s+[A-Z]',
            # TITLE IN CAPS (как заголовок секции)
            r'^[A-Z][A-Z\s]{3,}:?\s*$',
            # Clause 1, Section 1
            r'^(?:Clause|Section|clause|section)\s+\d+\.?\d*\.?\s*[-:\.]?\s*',
        ]
        
        # Паттерны для Invoice
        self.invoice_sections = {
            'header': ['invoice', 'bill to', 'ship to', 'seller', 'buyer', 'invoice no', 'date'],
            'items': ['description', 'commodity', 'quantity', 'unit price', 'amount'],
            'totals': ['subtotal', 'total', 'amount due', 'vat', 'tax'],
            'banking': ['bank', 'swift', 'iban', 'account'],
            'terms': ['payment terms', 'delivery terms', 'incoterms'],
        }
        
    def chunk_document(self, content: str, document_type: DocumentType, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Главный метод - выбирает стратегию чанкинга в зависимости от типа документа
        """
        logger.info(f"Chunking document of type: {document_type.value}")
        
        if document_type == DocumentType.CONTRACT:
            return self._chunk_contract(content, metadata)
        elif document_type == DocumentType.INVOICE:
            return self._chunk_invoice(content, metadata)
        elif document_type in [DocumentType.BL, DocumentType.LC]:
            return self._chunk_structured_document(content, metadata, document_type)
        else:
            # Для остальных типов - умное деление по параграфам
            return self._chunk_by_paragraphs(content, metadata)
    
    def _chunk_contract(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Деление контракта по clauses/articles.
        
        Логика:
        1. Ищем все заголовки секций (Article 1, 1.1, etc.)
        2. Делим контент между заголовками
        3. Если секция слишком большая - делим дополнительно по токенам
        4. Добавляем контекст: заголовок секции в метаданные
        """
        chunks = []
        lines = content.split('\n')
        
        current_section = {
            'title': 'Preamble',
            'content': [],
            'level': 0,
            'number': '0',
        }
        sections = []
        
        for line in lines:
            # Проверяем, является ли строка заголовком секции
            is_heading = False
            heading_info = self._parse_contract_heading(line)
            
            if heading_info:
                # Сохраняем предыдущую секцию
                if current_section['content']:
                    sections.append(current_section)
                
                # Начинаем новую секцию
                current_section = {
                    'title': heading_info['title'],
                    'content': [],
                    'level': heading_info['level'],
                    'number': heading_info['number'],
                }
                is_heading = True
            
            if not is_heading and line.strip():
                current_section['content'].append(line)
        
        # Добавляем последнюю секцию
        if current_section['content']:
            sections.append(current_section)
        
        # Конвертируем секции в чанки
        chunk_index = 0
        for section in sections:
            section_text = '\n'.join(section['content'])
            section_tokens = self._count_tokens(section_text)
            
            # Если секция помещается в один чанк
            if section_tokens <= self.max_tokens:
                chunk_metadata = {
                    **metadata,
                    'section_title': section['title'],
                    'section_number': section['number'],
                    'section_level': section['level'],
                    'chunk_type': 'contract_clause',
                }
                
                chunks.append(Chunk(
                    text=f"{section['title']}\n\n{section_text}",
                    metadata=chunk_metadata,
                    index=chunk_index,
                    token_count=section_tokens,
                ))
                chunk_index += 1
            else:
                # Секция слишком большая - делим на подчанки по токенам
                sub_chunks = self._split_by_tokens(
                    text=section_text,
                    title=section['title'],
                    metadata={
                        **metadata,
                        'section_title': section['title'],
                        'section_number': section['number'],
                        'section_level': section['level'],
                        'chunk_type': 'contract_clause_part',
                    }
                )
                
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.metadata['part'] = f"{i+1}/{len(sub_chunks)}"
                    sub_chunk.index = chunk_index
                    chunks.append(sub_chunk)
                    chunk_index += 1
        
        logger.info(f"Contract chunked into {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def _parse_contract_heading(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Парсит заголовок контракта и возвращает его структуру.
        
        Примеры:
        - "Article 1. Payment Terms" -> {title: "Payment Terms", number: "1", level: 1}
        - "1.1 Price" -> {title: "Price", number: "1.1", level: 2}
        - "ARTICLE 5 - CONFIDENTIALITY" -> {title: "CONFIDENTIALITY", number: "5", level: 1}
        """
        line = line.strip()
        if not line:
            return None
        
        # Pattern 1: Article 1., ARTICLE 1, etc.
        match = re.match(r'^(?:ARTICLE|Article|article)\s+(\d+)\.?\s*[-:\.]?\s*(.*)$', line)
        if match:
            return {
                'number': match.group(1),
                'title': match.group(2).strip() or f"Article {match.group(1)}",
                'level': 1,
            }
        
        # Pattern 2: 1.1, 1.1.1, etc. (followed by text starting with capital letter)
        match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+([A-Z].+)$', line)
        if match:
            number = match.group(1)
            level = len(number.split('.'))
            return {
                'number': number,
                'title': match.group(2).strip(),
                'level': level,
            }
        
        # Pattern 3: ALL CAPS line (section heading)
        if len(line) > 3 and line.isupper() and not line.endswith('.'):
            # Убираем лишние пробелы и двоеточия
            title = re.sub(r':\s*$', '', line).strip()
            return {
                'number': '',
                'title': title,
                'level': 1,
            }
        
        # Pattern 4: Clause 1, Section 1
        match = re.match(r'^(?:Clause|Section|clause|section)\s+(\d+)\.?\s*[-:\.]?\s*(.*)$', line)
        if match:
            return {
                'number': match.group(1),
                'title': match.group(2).strip() or f"Clause {match.group(1)}",
                'level': 1,
            }
        
        return None
    
    def _chunk_invoice(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Деление Invoice по смысловым секциям.
        
        Секции:
        - Header (от/кому, дата, номер)
        - Line Items (товары/услуги)
        - Totals (суммы, налоги)
        - Banking details
        - Terms & Conditions
        """
        chunks = []
        lines = content.split('\n')
        
        current_section = {
            'type': 'header',
            'content': [],
        }
        sections = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Определяем тип секции по ключевым словам
            section_type = self._identify_invoice_section(line_lower)
            
            if section_type and section_type != current_section['type']:
                # Переход на новую секцию
                if current_section['content']:
                    sections.append(current_section)
                
                current_section = {
                    'type': section_type,
                    'content': [line],
                }
            else:
                if line.strip():
                    current_section['content'].append(line)
        
        # Добавляем последнюю секцию
        if current_section['content']:
            sections.append(current_section)
        
        # Конвертируем в чанки
        chunk_index = 0
        for section in sections:
            section_text = '\n'.join(section['content'])
            section_tokens = self._count_tokens(section_text)
            
            if section_tokens <= self.max_tokens:
                chunk_metadata = {
                    **metadata,
                    'invoice_section': section['type'],
                    'chunk_type': 'invoice_section',
                }
                
                chunks.append(Chunk(
                    text=section_text,
                    metadata=chunk_metadata,
                    index=chunk_index,
                    token_count=section_tokens,
                ))
                chunk_index += 1
            else:
                # Секция слишком большая
                sub_chunks = self._split_by_tokens(
                    text=section_text,
                    title=section['type'].replace('_', ' ').title(),
                    metadata={
                        **metadata,
                        'invoice_section': section['type'],
                        'chunk_type': 'invoice_section_part',
                    }
                )
                
                for sub_chunk in sub_chunks:
                    sub_chunk.index = chunk_index
                    chunks.append(sub_chunk)
                    chunk_index += 1
        
        logger.info(f"Invoice chunked into {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def _identify_invoice_section(self, line_lower: str) -> Optional[str]:
        """Определяет тип секции Invoice по ключевым словам"""
        for section_type, keywords in self.invoice_sections.items():
            for keyword in keywords:
                if keyword in line_lower:
                    return section_type
        return None
    
    def _chunk_structured_document(self, content: str, metadata: Dict[str, Any], doc_type: DocumentType) -> List[Chunk]:
        """
        Для структурированных документов (BL, LC, etc.) - деление по полям.
        
        Логика:
        - Определяем поля по паттернам "Field Name: Value"
        - Группируем связанные поля
        - Создаем чанки по группам
        """
        chunks = []
        lines = content.split('\n')
        
        current_field_group = []
        field_groups = []
        
        for line in lines:
            # Паттерн "KEY: VALUE" или "KEY VALUE"
            if re.match(r'^[A-Z][A-Za-z\s]+:', line) or re.match(r'^[A-Z][A-Z\s]{3,}', line):
                # Это поле документа
                if len(current_field_group) > 5:  # Группируем по 5-10 полей
                    field_groups.append(current_field_group)
                    current_field_group = [line]
                else:
                    current_field_group.append(line)
            else:
                if line.strip():
                    current_field_group.append(line)
        
        # Добавляем последнюю группу
        if current_field_group:
            field_groups.append(current_field_group)
        
        # Конвертируем в чанки
        chunk_index = 0
        for group in field_groups:
            group_text = '\n'.join(group)
            tokens = self._count_tokens(group_text)
            
            chunk_metadata = {
                **metadata,
                'chunk_type': f'{doc_type.value}_fields',
            }
            
            chunks.append(Chunk(
                text=group_text,
                metadata=chunk_metadata,
                index=chunk_index,
                token_count=tokens,
            ))
            chunk_index += 1
        
        logger.info(f"{doc_type.value} chunked into {len(chunks)} chunks")
        return chunks
    
    def _chunk_by_paragraphs(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Универсальное деление по параграфам для документов без явной структуры.
        
        Логика:
        - Делим по двойным переносам строк (параграфы)
        - Объединяем маленькие параграфы
        - Делим большие параграфы по токенам
        """
        chunks = []
        # Делим по двойным переносам или одинарным с отступом
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk_text = []
        current_tokens = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = self._count_tokens(para)
            
            # Если параграф сам по себе больше max_tokens
            if para_tokens > self.max_tokens:
                # Сохраняем текущий накопленный чанк
                if current_chunk_text:
                    chunk_text = '\n\n'.join(current_chunk_text)
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={**metadata, 'chunk_type': 'paragraph'},
                        index=chunk_index,
                        token_count=current_tokens,
                    ))
                    chunk_index += 1
                    current_chunk_text = []
                    current_tokens = 0
                
                # Делим большой параграф на части
                sub_chunks = self._split_by_tokens(
                    text=para,
                    title=None,
                    metadata={**metadata, 'chunk_type': 'paragraph_split'}
                )
                
                for sub_chunk in sub_chunks:
                    sub_chunk.index = chunk_index
                    chunks.append(sub_chunk)
                    chunk_index += 1
            
            # Если добавление параграфа превысит лимит
            elif current_tokens + para_tokens > self.max_tokens:
                # Сохраняем текущий чанк
                if current_chunk_text:
                    chunk_text = '\n\n'.join(current_chunk_text)
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={**metadata, 'chunk_type': 'paragraph'},
                        index=chunk_index,
                        token_count=current_tokens,
                    ))
                    chunk_index += 1
                
                # Начинаем новый чанк с этого параграфа
                current_chunk_text = [para]
                current_tokens = para_tokens
            
            else:
                # Добавляем параграф к текущему чанку
                current_chunk_text.append(para)
                current_tokens += para_tokens
        
        # Добавляем последний чанк
        if current_chunk_text:
            chunk_text = '\n\n'.join(current_chunk_text)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, 'chunk_type': 'paragraph'},
                index=chunk_index,
                token_count=current_tokens,
            ))
        
        logger.info(f"Document chunked into {len(chunks)} paragraph-based chunks")
        return chunks
    
    def _split_by_tokens(self, text: str, title: Optional[str], metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Делит большой текст на чанки по токенам с overlap.
        
        Используется как fallback когда секция слишком большая.
        """
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                # Сохраняем текущий чанк
                chunk_text = ' '.join(current_chunk)
                if title:
                    chunk_text = f"{title}\n\n{chunk_text}"
                
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=metadata.copy(),
                    index=0,  # Будет установлен позже
                    token_count=current_tokens,
                ))
                
                # Overlap: оставляем последние предложения для контекста
                overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) > 1 else ''
                overlap_tokens = self._count_tokens(overlap_text)
                
                current_chunk = current_chunk[-2:] if len(current_chunk) > 1 else []
                current_tokens = overlap_tokens
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Добавляем последний чанк
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if title:
                chunk_text = f"{title}\n\n{chunk_text}"
            
            chunks.append(Chunk(
                text=chunk_text,
                metadata=metadata.copy(),
                index=0,
                token_count=current_tokens,
            ))
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Подсчитывает количество токенов в тексте"""
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except:
            # Fallback: примерная оценка (1 токен ≈ 4 символа)
            return len(text) // 4


