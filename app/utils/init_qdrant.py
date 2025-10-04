
from qdrant_client import AsyncQdrantClient
from app.core.config import settings


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
                vectors_config={
                    "size": 1024,  # Размер вектора для e5-large-v2
                    "distance": "Cosine"
                }
            )
            print("Collection created successfully!")
        else:
            print("Collection 'document_embeddings' already exists")
            
    except Exception as e:
        print(f"Error initializing Qdrant collection: {e}")
    finally:
        await client.close()

