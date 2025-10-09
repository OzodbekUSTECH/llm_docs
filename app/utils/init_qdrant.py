from app.utils.collections import Collections
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import VectorParams, Distance
from app.core.config import settings
from dishka import AsyncContainer
from sentence_transformers import SentenceTransformer
from docling.chunking import HybridChunker


async def init_qdrant_collection():
    """Инициализирует коллекции в Qdrant: document_embeddings и rules_embeddings"""
    client = AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
    )
    
    try:
        # Проверяем, существуют ли коллекции
        collections = await client.get_collections()
        collection_names = [col.name for col in collections.collections]

        # Коллекция для документов
        if Collections.DOCUMENT_EMBEDDINGS not in collection_names:
            print(f"Creating Qdrant collection '{Collections.DOCUMENT_EMBEDDINGS}'...")
            await client.create_collection(
                collection_name=Collections.DOCUMENT_EMBEDDINGS,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE,
                ),
                on_disk_payload=True,
            )
            print(f"Collection '{Collections.DOCUMENT_EMBEDDINGS}' created successfully!")
        else:
            print(f"Collection '{Collections.DOCUMENT_EMBEDDINGS}' already exists")

        # Коллекция для правил
        if Collections.RULES_EMBEDDINGS not in collection_names:
            print(f"Creating Qdrant collection '{Collections.RULES_EMBEDDINGS}'...")
            await client.create_collection(
                collection_name=Collections.RULES_EMBEDDINGS,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE,
                ),
                on_disk_payload=True,
            )
            print(f"Collection '{Collections.RULES_EMBEDDINGS}' created successfully!")
        else:
            print(f"Collection '{Collections.RULES_EMBEDDINGS}' already exists")
            
    except Exception as e:
        print(f"Error initializing Qdrant collections: {e}")
    finally:
        await client.close()


async def warmup_dependencies(container: AsyncContainer):
    print("🚀 Прогреваем зависимости...")

    # Получаем зависимости из контейнера, чтобы заставить Dishka их создать
    sentence_transformer = await container.get(SentenceTransformer)
    chunker = await container.get(HybridChunker)
    _ = sentence_transformer.encode(["тестовая инициализация"])
    print("✅ Модель эмбеддингов и chunker прогреты!")