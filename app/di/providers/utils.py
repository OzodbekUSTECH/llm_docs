import base64
from dishka import Provider, Scope, provide, provide_all

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from transformers import AutoTokenizer
from app.core.config import settings
from ollama import AsyncClient
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions, RapidOcrOptions
from docling.document_converter import PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from openai import AsyncOpenAI
from app.services.keyword_extractor import KeywordExtractor

class UtilsProvider(Provider):
    """
    Provider для утилит: эмбеддинги, Qdrant, Docling, chunking.
    Все компоненты настроены для оптимальной работы вместе.
    
    ОСОБЕННОСТИ OCR НАСТРОЕК ДЛЯ ТАБЛИЦ:
    - Специальная настройка для химических формул и числовых данных
    - Распознавание формул (do_formula_enrichment) и кода (do_code_enrichment)
    - Увеличенный масштаб изображений (2x) для лучшего качества
    - Очень низкий порог площади (1%) для мелких символов в таблицах
    - Отключение cell_matching для лучшего распознавания структуры таблиц
    - Точный режим TableFormer для максимальной точности
    - Генерация изображений таблиц для анализа
    - Два варианта OCR: RapidOCR (по умолчанию) и Tesseract (альтернатива)
    - Оптимизация для распознавания химических формул и специальных символов
    """
    scope = Scope.APP
    
    utils = provide_all(
        KeywordExtractor,
    )
    
    # Константы для единообразия
    EMBEDDING_MODEL = "intfloat/e5-base-v2"
    EMBEDDING_DIMENSION = 768  # e5-base-v2 имеет 768 измерения
    MAX_CHUNK_TOKENS = 512  # Максимум токенов в чанке
    CHUNK_OVERLAP_TOKENS = 64  # Перекрытие для контекста
    
    @provide
    def provide_sentence_transformer(self) -> SentenceTransformer:
        """
        Загружает модель для генерации эмбеддингов.
        intfloat/e5-large-v2 - одна из лучших open-source моделей.
        """
        print(f"🔄 Загружаем модель эмбеддингов: {self.EMBEDDING_MODEL}")
        return SentenceTransformer(self.EMBEDDING_MODEL)
    
    @provide
    def provide_qdrant_client(self) -> AsyncQdrantClient:
        """Создает асинхронный клиент для Qdrant векторной БД."""
        print(f"🔄 Подключаемся к Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        return AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
   
    @provide
    def provide_ollama_client(self) -> AsyncClient:
        """Создает клиент для Ollama LLM."""
        userpass = "goodman:password4ollama"
        auth = base64.b64encode(userpass.encode()).decode()
        client = AsyncClient(
            host="https://ollama.technocrats.uz",
            headers={"Authorization": f"Basic {auth}"}
        )
        print(f"🔄 Ollama клиент создан: https://ollama.technocrats.uz")
        return client
    
    @provide
    def provide_accelerator_options(self) -> AcceleratorOptions:
        """
        Настраивает accelerator для Docling с оптимизацией для OCR.
        AUTO автоматически выберет лучший доступный (CUDA > MPS > CPU).
        """
        return AcceleratorOptions(
            num_threads=12,  # Увеличиваем количество потоков для OCR
            device=AcceleratorDevice.AUTO,  # Автовыбор: CUDA > MPS > CPU
            # Дополнительные настройки для OCR
            cuda_use_flash_attention2=True,
        )
    
    @provide
    def provide_pdf_pipeline_options(self) -> PdfPipelineOptions:
        """
        Настройки pipeline для обработки PDF с оптимизированным OCR.
        Специально настроено для лучшего распознавания таблиц с химическими формулами.
        """
        pipeline_options = PdfPipelineOptions()
        
        # Основные настройки OCR и таблиц
        pipeline_options.do_ocr = True  # Включаем OCR для отсканированных PDF
        pipeline_options.do_table_structure = True  # Извлекаем структуру таблиц
        pipeline_options.do_formula_enrichment = True  # Включаем распознавание формул
        pipeline_options.do_code_enrichment = True  # Включаем распознавание кода/символов
        
        # Настройки для лучшего распознавания таблиц
        pipeline_options.table_structure_options.do_cell_matching = False  # Отключаем для лучшего распознавания структуры
        pipeline_options.table_structure_options.mode = "accurate"  # Точный режим для таблиц
        
        # Настройки масштабирования для лучшего качества
        pipeline_options.images_scale = 2.0  # Увеличиваем масштаб для лучшего распознавания
        pipeline_options.generate_page_images = True  # Генерируем изображения страниц
        pipeline_options.generate_table_images = True  # Генерируем изображения таблиц
        
        # Специальные настройки OCR для химических формул и таблиц
        # Используем RapidOCR для лучшего распознавания таблиц
        ocr_options = RapidOcrOptions(
            force_full_page_ocr=True,  # Принудительное OCR всей страницы
            # Языки для распознавания (английский + китайский для лучшего распознавания символов)
            lang=["english", "chinese"],
            # Очень низкий порог для распознавания мелких символов в таблицах
            bitmap_area_threshold=0.01,  # 1% - очень низкий порог для мелких элементов
        )
        pipeline_options.ocr_options = ocr_options

        return pipeline_options
    
    @provide
    def provide_pdf_pipeline_options_tesseract(self) -> PdfPipelineOptions:
        """
        Альтернативная конфигурация с Tesseract для случаев, когда RapidOCR не работает.
        Специально настроено для лучшего распознавания таблиц с химическими формулами.
        """
        pipeline_options = PdfPipelineOptions()
        
        # Основные настройки OCR и таблиц
        pipeline_options.do_ocr = True  # Включаем OCR для отсканированных PDF
        pipeline_options.do_table_structure = True  # Извлекаем структуру таблиц
        pipeline_options.do_formula_enrichment = True  # Включаем распознавание формул
        pipeline_options.do_code_enrichment = True  # Включаем распознавание кода/символов
        
        # Настройки для лучшего распознавания таблиц
        pipeline_options.table_structure_options.do_cell_matching = False  # Отключаем для лучшего распознавания структуры
        # pipeline_options.table_structure_options.mode = "accurate"  # Точный режим для таблиц
        
        # Настройки масштабирования для лучшего качества
        pipeline_options.images_scale = 2.0  # Увеличиваем масштаб для лучшего распознавания
        pipeline_options.generate_page_images = True  # Генерируем изображения страниц
        pipeline_options.generate_table_images = True  # Генерируем изображения таблиц
        
        # Специальные настройки Tesseract OCR для химических формул и таблиц
        ocr_options = TesseractCliOcrOptions(
            force_full_page_ocr=True,  # Принудительное OCR всей страницы
            # Языки для распознавания (английский + русский для лучшего распознавания формул)
            lang=["eng", "rus"],
            # Очень низкий порог для распознавания мелких символов в таблицах
            bitmap_area_threshold=0.01,  # 1% - очень низкий порог для мелких элементов
            # Путь к исполняемому файлу Tesseract
            tesseract_cmd="tesseract",
            # Путь к данным Tesseract (если нужно)
            path=None
        )
        pipeline_options.ocr_options = ocr_options

        return pipeline_options
    
    @provide
    def provide_document_converter(self, pipeline_options: PdfPipelineOptions, accelerator_options: AcceleratorOptions) -> DocumentConverter:
        """
        Создает DocumentConverter с оптимизированными настройками.
        Специально настроено для лучшего распознавания таблиц и химических формул.
        """
        print(f"🔄 Настраиваем Docling DocumentConverter с accelerator: {accelerator_options.device}")
        
        # Применяем accelerator к pipeline_options
        pipeline_options.accelerator_options = accelerator_options
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            },
        )
        return converter
    
    @provide
    def provide_huggingface_tokenizer(self) -> HuggingFaceTokenizer:
        """
        Создает tokenizer для HybridChunker.
        ВАЖНО: использует ту же модель что и для эмбеддингов!
        """
        print(f"🔄 Загружаем tokenizer: {self.EMBEDDING_MODEL}")
        return HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL),
            max_tokens=self.MAX_CHUNK_TOKENS  # Максимум токенов в чанке
        )
    
    @provide
    def provide_docling_chunker(self, tokenizer: HuggingFaceTokenizer) -> HybridChunker:
        """
        Создает HybridChunker для интеллектуального разделения документов.
        
        HybridChunker:
        - Уважает структуру документа (заголовки, параграфы)
        - Добавляет контекст из заголовков к чанкам
        - Учитывает токены, а не символы
        - Объединяет маленькие чанки (merge_peers=True)
        """
        print(f"🔄 Настраиваем HybridChunker: max_tokens={self.MAX_CHUNK_TOKENS}, merge_peers=True")
        
        return HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,  # Объединяем соседние маленькие чанки
        )
    
    
    @provide
    def provide_openai_client(self) -> AsyncOpenAI:
        """Создает клиент для OpenAI."""
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    