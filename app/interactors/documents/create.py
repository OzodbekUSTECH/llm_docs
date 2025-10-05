import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Any
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

        # 5. Парсим файл с помощью Docling
        try:
            print(f"🔄 Начинаем парсинг файла: {file.filename}")
            
            # Конвертируем документ
            result = self.document_converter.convert(str(file_path))
            
            # Извлекаем структурированный контент
            structured_content = self._extract_structured_content(result)
            
            # Экспортируем в Markdown с правильным отображением таблиц
            # strict_text=False позволяет извлечь максимум текста из таблиц
            markdown_content = result.document.export_to_markdown(strict_text=False)
            
            # Извлекаем таблицы
            tables = self._extract_tables_from_docling_result(result)
            
            print(f"✅ Docling парсинг успешен:")
            print(f"   - Markdown контент: {len(markdown_content)} символов")
            print(f"   - Структурных элементов: {len(structured_content.get('elements', []))}")
            print(f"   - Таблиц: {len(tables)}")
            
            # Показываем первые 500 символов markdown для диагностики
            if markdown_content:
                preview = markdown_content[:500].replace('\n', '\\n')
                print(f"   📄 Превью контента: {preview}...")
            
        except Exception as e:
            print(f"❌ Ошибка парсинга с Docling: {e}")
            # Удаляем файл при ошибке парсинга
            if file_path.exists():
                file_path.unlink()
            
            error_msg = f"Не удалось обработать файл {ext}. "
            if ext.lower() == '.pdf':
                error_msg += "PDF может содержать только изображения или быть защищён паролем. "
                error_msg += "Попробуйте конвертировать PDF в текстовый формат."
            else:
                error_msg += "Файл может быть повреждён или в неподдерживаемом формате."
            raise AppError(status_code=400, message=error_msg)

        # Проверяем, что удалось извлечь контент
        if not markdown_content or not markdown_content.strip():
            if file_path.exists():
                file_path.unlink()
            raise AppError(
                status_code=400, 
                message="Не удалось извлечь текст из файла. Файл может быть пустым или содержать только изображения."
            )
        
        # Для PDF с очень коротким контентом выводим предупреждение
        if ext.lower() == '.pdf' and len(markdown_content.strip()) < 100:
            print(f"⚠️ Контент PDF очень короткий ({len(markdown_content)} символов). "
                  f"Возможно, это отсканированный документ.")

        print(f"📄 Информация о файле:")
        print(f"   - Тип: {file.content_type}")
        print(f"   - Расширение: {ext}")
        print(f"   - Размер контента: {len(markdown_content)} символов")

        # 6. Создаём Document с структурированным контентом
        document = Document(
            id=id,
            filename=stored_filename,
            original_filename=file.filename,
            file_path=str(file_path),
            content_type=file.content_type or "application/octet-stream",
            file_hash=file_hash,
            status=DocumentStatus.COMPLETED,
            content=markdown_content,
            tables=tables,
            # Сохраняем структурированный контент в doc_metadata
            doc_metadata={
                "structured_content": structured_content,
                "parsing_method": "docling",
                "has_tables": len(tables) > 0
            }
        )
        await self.documents_repository.create(document)

        # 7. Делим на чанки с помощью Docling chunker
        chunks = self._chunk_with_docling(result)
        
        print(f"📦 Chunking результаты:")
        print(f"   - Количество чанков: {len(chunks)}")
        if chunks:
            avg_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            print(f"   - Средний размер: {avg_size:.0f} символов")
            print(f"   - Размеры первых 5: {[len(c) for c in chunks[:5]]}")

        # 8. Получаем эмбеддинги с префиксом для E5 модели
        if chunks and any(chunk.strip() for chunk in chunks):
            print(f"🔄 Генерация эмбеддингов для {len(chunks)} чанков...")
            
            # ВАЖНО: E5 модели требуют префикс "passage: " для документов
            # Это улучшает качество эмбеддингов на 10-15%!
            chunks_with_prefix = ["passage: " + chunk for chunk in chunks]
            
            embeddings = self.sentence_transformer.encode(
                chunks_with_prefix,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True  # Нормализуем для косинусного расстояния
            )
            
            print(f"✅ Эмбеддинги сгенерированы: {embeddings.shape}")
            print(f"   📊 С префиксом 'passage:' для E5 модели")
            
            # 9. Сохраняем чанки + эмбеддинги в Qdrant с метаданными
            await self.qdrant_embeddings_repository.bulk_create_embeddings(
                document_id=str(document.id),
                chunks=chunks,  # Сохраняем БЕЗ префикса
                embeddings=embeddings.tolist(),
                # Добавляем метаданные для лучшей фильтрации
                metadata={
                    "filename": document.original_filename,
                    "content_type": file.content_type,
                    "has_tables": len(tables) > 0,
                }
            )
            print(f"✅ Сохранено {len(chunks)} чанков в Qdrant с метаданными")
        else:
            print("⚠️ Нет чанков для сохранения в Qdrant")

        # 10. Коммитим UoW
        await self.uow.commit()
        print(f"✅ Документ успешно создан и сохранён")

        return document

    def _extract_structured_content(self, result) -> Dict[str, Any]:
        """
        Извлекает структурированный контент из результата Docling.
        Сохраняет структуру документа: заголовки, параграфы, списки, таблицы.
        
        ВАЖНО: Для таблиц пытаемся извлечь максимум информации разными способами.
        """
        structured = {
            "elements": [],
            "metadata": {}
        }
        
        try:
            # Проверяем наличие метода iterate_items
            if not hasattr(result.document, 'iterate_items'):
                print(f"⚠️ Документ не имеет метода iterate_items")
                return structured
            
            # Итерируемся по элементам документа
            element_count = 0
            elements_by_type = {}
            
            for item in result.document.iterate_items():
                element_count += 1
                
                # Получаем тип элемента
                item_type = getattr(item, 'label', 'paragraph')
                elements_by_type[item_type] = elements_by_type.get(item_type, 0) + 1
                
                # Извлекаем текст разными способами
                text = ""
                
                # Способ 1: прямой атрибут text
                if hasattr(item, 'text') and item.text:
                    text = item.text
                
                # Способ 2: для таблиц пытаемся получить текст через export
                elif item_type == 'table' and hasattr(item, 'export_to_markdown'):
                    try:
                        text = item.export_to_markdown()
                    except:
                        pass
                
                # Способ 3: через str() для некоторых типов
                if not text and hasattr(item, '__str__'):
                    try:
                        text_candidate = str(item)
                        if text_candidate and len(text_candidate) < 10000:  # Разумный лимит
                            text = text_candidate
                    except:
                        pass
                
                element = {
                    "type": item_type,
                    "text": text,
                    "level": getattr(item, 'level', None),
                }
                
                # Добавляем дополнительные атрибуты если есть (но сериализуем их!)
                if hasattr(item, 'bbox'):
                    bbox = item.bbox
                    # Конвертируем bbox в dict если это объект
                    if hasattr(bbox, '__dict__'):
                        element["bbox"] = {
                            'l': getattr(bbox, 'l', 0),
                            't': getattr(bbox, 't', 0),
                            'r': getattr(bbox, 'r', 0),
                            'b': getattr(bbox, 'b', 0),
                        }
                    else:
                        element["bbox"] = bbox
                
                # Сохраняем элемент если есть текст
                if text.strip():
                    structured["elements"].append(element)
            
            print(f"📊 Обработано элементов: {element_count}, сохранено: {len(structured['elements'])}")
            print(f"📊 Типы элементов: {elements_by_type}")
            
            # Добавляем метаданные документа (сериализуем если это объект)
            if hasattr(result.document, 'metadata'):
                metadata = result.document.metadata
                if hasattr(metadata, '__dict__'):
                    # Конвертируем объект в dict
                    structured["metadata"] = {
                        k: v for k, v in metadata.__dict__.items() 
                        if not k.startswith('_') and isinstance(v, (str, int, float, bool, type(None)))
                    }
                elif isinstance(metadata, dict):
                    structured["metadata"] = metadata
                else:
                    structured["metadata"] = {}
                
        except Exception as e:
            print(f"⚠️ Ошибка извлечения структуры: {e}")
            import traceback
            traceback.print_exc()
        
        return structured

    def _extract_tables_from_docling_result(self, result) -> List[Dict[str, Any]]:
        """
        Извлекает таблицы из результата Docling с подробной информацией.
        Пытается извлечь максимум текста из таблиц разными способами.
        """
        tables = []
        table_count = 0
        
        try:
            for item in result.document.iterate_items():
                if hasattr(item, 'label') and item.label == 'table':
                    table_count += 1
                    
                    # Извлекаем текст таблицы разными способами
                    table_text = ""
                    
                    # Способ 1: прямой текст
                    if hasattr(item, 'text') and item.text:
                        table_text = item.text
                    
                    # Способ 2: export_to_markdown
                    if not table_text and hasattr(item, 'export_to_markdown'):
                        try:
                            table_text = item.export_to_markdown()
                        except:
                            pass
                    
                    # Способ 3: через data если есть
                    if not table_text and hasattr(item, 'data'):
                        try:
                            # Пытаемся сконвертировать data в текст
                            data = item.data
                            if isinstance(data, (list, dict)):
                                table_text = str(data)
                        except:
                            pass
                    
                    table_data = {
                        'text': table_text,
                        'label': item.label,
                        'rows': [],
                        'metadata': {}
                    }
                    
                    # Пытаемся извлечь структуру таблицы (но НЕ сохраняем сырые объекты!)
                    # data может быть очень большим и содержать несериализуемые объекты
                    # Вместо этого используем text который уже извлечен
                    
                    if hasattr(item, 'bbox'):
                        bbox = item.bbox
                        if hasattr(bbox, '__dict__'):
                            table_data['bbox'] = {
                                'l': getattr(bbox, 'l', 0),
                                't': getattr(bbox, 't', 0),
                                'r': getattr(bbox, 'r', 0),
                                'b': getattr(bbox, 'b', 0),
                            }
                        else:
                            table_data['bbox'] = bbox
                    
                    if table_text.strip():
                        tables.append(table_data)
                        print(f"   ✓ Таблица {table_count}: {len(table_text)} символов")
                    else:
                        print(f"   ⚠️ Таблица {table_count}: не удалось извлечь текст")
                        
        except Exception as e:
            print(f"⚠️ Ошибка извлечения таблиц: {e}")
            import traceback
            traceback.print_exc()
        
        if table_count > 0:
            print(f"📊 Найдено таблиц: {table_count}, извлечено с текстом: {len(tables)}")
        
        return tables

    def _chunk_with_docling(self, docling_result) -> List[str]:
        """
        Разделяет контент на чанки с помощью Docling HybridChunker.
        
        HybridChunker:
        1. Уважает структуру документа (не разрывает семантические блоки)
        2. Добавляет контекст из заголовков через contextualize()
        3. Учитывает токены, а не символы
        4. Объединяет маленькие соседние чанки
        
        """
       
        try:
            if not docling_result or not hasattr(docling_result, 'document'):
                raise ValueError("Некорректный результат Docling")
            
            print(f"🔄 Используем Docling HybridChunker...")
            
            # Получаем итератор чанков
            chunk_iter = self.docling_chunker.chunk(dl_doc=docling_result.document)
            
            # Обрабатываем чанки с контекстуализацией
            chunks = []
            chunk_stats = {
                'total': 0,
                'with_context': 0,
                'min_size': float('inf'),
                'max_size': 0,
            }
            
            for i, chunk in enumerate(chunk_iter):
                if not hasattr(chunk, 'text') or not chunk.text.strip():
                    continue
                
                chunk_stats['total'] += 1
                
                # КЛЮЧЕВОЙ МОМЕНТ: используем contextualize() для добавления контекста
                # Это добавляет заголовки разделов к чанку для лучшего понимания
                enriched_text = self.docling_chunker.contextualize(chunk=chunk)
                
                # Если контекст добавлен (текст изменился), отмечаем это
                if enriched_text != chunk.text:
                    chunk_stats['with_context'] += 1
                
                chunks.append(enriched_text.strip())
                
                # Статистика размеров
                text_len = len(enriched_text)
                chunk_stats['min_size'] = min(chunk_stats['min_size'], text_len)
                chunk_stats['max_size'] = max(chunk_stats['max_size'], text_len)
            
            if not chunks:
                raise ValueError("Docling chunker не создал чанки")
            
            # Красивый вывод статистики
            avg_size = sum(len(c) for c in chunks) / len(chunks)
            print(f"✅ Docling HybridChunker:")
            print(f"   ├─ Всего чанков: {chunk_stats['total']}")
            print(f"   ├─ С контекстом: {chunk_stats['with_context']} ({chunk_stats['with_context']/chunk_stats['total']*100:.1f}%)")
            print(f"   ├─ Размеры: min={chunk_stats['min_size']}, avg={avg_size:.0f}, max={chunk_stats['max_size']}")
            print(f"   └─ Первые 3 размера: {[len(c) for c in chunks[:3]]}")
            
            # Показываем пример контекстуализации
            if chunk_stats['with_context'] > 0:
                for i, chunk_text in enumerate(chunks[:2]):
                    if '\n' in chunk_text[:100]:  # Скорее всего есть контекст
                        lines = chunk_text.split('\n', 2)
                        if len(lines) >= 2:
                            print(f"   📌 Пример контекста чанка {i+1}: {lines[0][:80]}...")
                            break
            
            # Показываем содержимое первого чанка для диагностики
            if chunks:
                first_chunk_preview = chunks[0][:300].replace('\n', '\\n')
                print(f"   📝 Первый чанк: {first_chunk_preview}...")
            
            return chunks
            
        except Exception as e:
            raise AppError(400, "Не удалось разделить документ на чанки")
    
   