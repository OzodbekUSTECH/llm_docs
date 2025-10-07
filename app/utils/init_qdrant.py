
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import VectorParams, Distance
from app.core.config import settings
from dishka import AsyncContainer
from sentence_transformers import SentenceTransformer
from docling.chunking import HybridChunker


async def init_qdrant_collection():
    """Инициализирует коллекцию в Qdrant"""
    client = AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
    )
    
    try:
        # Проверяем, существует ли коллекция
        collections = await client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if "document_embeddings" not in collection_names:
            print("Creating Qdrant collection 'document_embeddings'...")
            await client.create_collection(
                collection_name="document_embeddings",
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE,
                ),
                on_disk_payload=True,
            )
            print("Collection created successfully!")
        else:
            print("Collection 'document_embeddings' already exists")
            
    except Exception as e:
        print(f"Error initializing Qdrant collection: {e}")
    finally:
        await client.close()


async def warmup_dependencies(container: AsyncContainer):
    print("🚀 Прогреваем зависимости...")

    # Получаем зависимости из контейнера, чтобы заставить Dishka их создать
    sentence_transformer = await container.get(SentenceTransformer)
    chunker = await container.get(HybridChunker)
    _ = sentence_transformer.encode(["тестовая инициализация"])
    print("✅ Модель эмбеддингов и chunker прогреты!")
    