import base64
import os
from dishka import Provider, Scope, provide, provide_all

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from transformers import AutoTokenizer
from app.core.config import settings
from ollama import AsyncClient
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions, RapidOcrOptions, EasyOcrOptions
from docling.document_converter import PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from openai import AsyncOpenAI
from app.services.extract_text_from_file import DocumentParserOpenAI
from app.services.keyword_extractor import KeywordExtractor
from app.services.contract_section_extractor import ContractSectionExtractor
from app.services.document_chunker import DocumentChunker
from huggingface_hub import snapshot_download

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
        ContractSectionExtractor,
        DocumentParserOpenAI,
    )
    
    # Константы для единообразия
    EMBEDDING_MODEL = "intfloat/e5-base-v2"
    EMBEDDING_DIMENSION = 3072  # e5-base-v2 имеет 768 измерения
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
        Настраивает accelerator для Docling с принудительным использованием CUDA.
        RTX 4060 с 8GB VRAM - отличная производительность для OCR.
        """
        return AcceleratorOptions(
            device=AcceleratorDevice.CUDA,  # Принудительно используем CUDA
            num_threads=8,  # Оптимально для RTX 4060
            cuda_use_flash_attention2=True,  # Используем Flash Attention для скорости
        )
    
    @provide
    def provide_pdf_pipeline_options(self) -> PdfPipelineOptions:
        """
        Настройки pipeline для обработки PDF с оптимизированным OCR.
        Специально настроено для лучшего распознавания таблиц с химическими формулами.
        """
        pipeline_options = PdfPipelineOptions()
        
        download_path = snapshot_download(repo_id="SWHL/RapidOCR")

        # Setup RapidOcrOptions for english detection
        det_model_path = os.path.join(
            download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"
        )
        rec_model_path = os.path.join(
            download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"
        )
        cls_model_path = os.path.join(
            download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx"
        )
        ocr_options = RapidOcrOptions(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            cls_model_path=cls_model_path,
        )
        
        pipeline_options.ocr_options = ocr_options
        
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.AUTO,
        )
        

        return pipeline_options
    
    
    @provide
    def provide_document_converter(
        self, 
    ) -> DocumentConverter:
        """
        Создает DocumentConverter с явным указанием OCR-движка RapidOCR.
        """
        print(f"🔄 Настраиваем Docling DocumentConverter")
        return DocumentConverter()
    
    
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
        Умный HybridChunker для Markdown-документов.
        
        Особенности:
        - Делит по заголовкам (#, ##, ###)
        - Делит жирные подзаголовки (**text**) как смысловые блоки
        - Добавляет родительский контекст заголовков к каждому чанку
        - Не ломает списки, таблицы и код-блоки
        - Разделяет длинные параграфы по предложениям
        - Объединяет мелкие чанки (<40 токенов) с соседними
        """

        print(f"⚙️ Настраиваем Markdown-aware HybridChunker: max_tokens={self.MAX_CHUNK_TOKENS}")

        chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,
            respect_hierarchy=True,          # Уважает иерархию заголовков
            add_parent_headings=True,        # Добавляет родительские заголовки в контекст чанка
            sentence_split=True,             # Делит длинные абзацы по предложениям
            max_tokens=self.MAX_CHUNK_TOKENS,
            min_chunk_tokens=40,
            overlap_tokens=self.CHUNK_OVERLAP_TOKENS,
            merge_small_chunks=True,
            weight_structure=True,
            include_inline_formatting=True,  # Учитывает **жирный**, _курсив_, `код`
        )

        # Конфиг специально под Markdown

        return chunker
    
    @provide
    def provide_openai_client(self) -> AsyncOpenAI:
        """Создает клиент для OpenAI."""
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    @provide
    def provide_document_chunker(self) -> DocumentChunker:
        """
        Создает кастомный чанкер для структурированного деления документов.
        Использует те же параметры что и эмбеддинги для согласованности.
        """
        print(f"🔄 Создаем DocumentChunker с max_tokens={self.MAX_CHUNK_TOKENS}")
        return DocumentChunker(
            tokenizer_model=self.EMBEDDING_MODEL,
            max_tokens=self.MAX_CHUNK_TOKENS,
            overlap_tokens=self.CHUNK_OVERLAP_TOKENS,
        )
    
    