import base64
from dishka import Provider, Scope, provide, provide_all

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from transformers import AutoTokenizer
from app.core.config import settings
from ollama import AsyncClient
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
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
        Настраивает accelerator для Docling.
        AUTO автоматически выберет лучший доступный (CUDA > MPS > CPU).
        """
        return AcceleratorOptions(
            num_threads=8,  # Количество потоков для CPU операций
            device=AcceleratorDevice.AUTO  # Автовыбор: CUDA > MPS > CPU
        )
    
    @provide
    def provide_pdf_pipeline_options(self) -> PdfPipelineOptions:
        """
        Настройки pipeline для обработки PDF.
        Включаем OCR и table structure для максимальной точности.
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Включаем OCR для отсканированных PDF
        pipeline_options.do_table_structure = True  # Извлекаем структуру таблиц
        pipeline_options.table_structure_options.do_cell_matching = True

        ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
        pipeline_options.ocr_options = ocr_options

        return pipeline_options
    
    @provide
    def provide_document_converter(self, pipeline_options: PdfPipelineOptions) -> DocumentConverter:
        """
        Создает DocumentConverter с оптимизированными настройками.
        Использует accelerator и продвинутый PDF pipeline.
        """
        print(f"🔄 Настраиваем Docling DocumentConverter с accelerator: {pipeline_options.accelerator_options.device}")
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
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
    
    