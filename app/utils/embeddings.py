# fastembed_bge_embeddings.py
from typing import TYPE_CHECKING, Optional
from app.dto.schemas import DocumentWithEmbedding, DocumentSchema
from openai import AsyncOpenAI
from app.core.config import settings

if TYPE_CHECKING:
    from fastembed import TextEmbedding


class FastEmbedEmbeddings:
    """Utilize fastembed library for embeddings locally without GPU.
    
    Using BAAI/bge-base-en-v1.5 model (768 dimensions)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 256,
        parallel: Optional[int] = None
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.parallel = parallel
        self._client = None

    @property
    def client_(self) -> "TextEmbedding":
        if self._client is None:
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise ImportError("Please install FastEmbed: `pip install fastembed`")

            self._client = TextEmbedding(model_name=self.model_name)
        return self._client
    
    def prepare_input(
        self, text: str | list[str] | DocumentSchema | list[DocumentSchema]
    ) -> list[DocumentSchema]:
        if isinstance(text, (str, DocumentSchema)):
            return [DocumentSchema(content=text)]
        elif isinstance(text, list):
            return [DocumentSchema(content=_) for _ in text]
        return text

    def invoke(
        self, text: str | list[str] | DocumentSchema | list[DocumentSchema], *args, **kwargs
    ) -> list[DocumentWithEmbedding]:
        input_ = self.prepare_input(text)
        embeddings = self.client_.embed(
            [_.content for _ in input_],
            batch_size=self.batch_size,
            parallel=self.parallel,
        )
        return [
            DocumentWithEmbedding(
                content=doc,
                embedding=list(embedding),
            )
            for doc, embedding in zip(input_, embeddings)
        ]

    async def ainvoke(
        self, text: str | list[str] | DocumentSchema | list[DocumentSchema], *args, **kwargs
    ) -> list[DocumentWithEmbedding]:
        """Fastembed does not support async API."""
        return self.invoke(text, *args, **kwargs)


class OpenAIEmbeddings:
    """Utilize OpenAI embeddings API for high-quality embeddings.
    
    Using text-embedding-3-large model (3072 dimensions)
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        request_timeout_s: float = 60.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
        max_batch_size: int = 32,
    ):
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.embedding_dimension = 3072  # text-embedding-3-large dimension
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.max_batch_size = max_batch_size

    def prepare_input(
        self, text: str | list[str] | DocumentSchema | list[DocumentSchema]
    ) -> list[str]:
        """Prepare input text for embedding"""
        if isinstance(text, str):
            return [text]
        elif isinstance(text, DocumentSchema):
            return [text.content if hasattr(text, 'content') else str(text)]
        elif isinstance(text, list):
            result = []
            for item in text:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, DocumentSchema):
                    result.append(item.content if hasattr(item, 'content') else str(item))
                else:
                    result.append(str(item))
            return result
        return [str(text)]

    async def ainvoke(
        self, text: str | list[str] | DocumentSchema | list[DocumentSchema], *args, **kwargs
    ) -> list[DocumentWithEmbedding]:
        """Generate embeddings using OpenAI API (async) with batching, timeouts, and retries."""
        input_texts = self.prepare_input(text)
        if not input_texts:
            return []

        results: list[DocumentWithEmbedding] = []

        # Process in sub-batches
        for start in range(0, len(input_texts), self.max_batch_size):
            batch = input_texts[start : start + self.max_batch_size]

            attempt = 0
            while True:
                attempt += 1
                try:
                    response = await self.client.embeddings.create(
                        model=self.model_name,
                        input=batch,
                        timeout=self.request_timeout_s,
                    )

                    for i, embedding_data in enumerate(response.data):
                        embedding = embedding_data.embedding
                        results.append(
                            DocumentWithEmbedding(
                                content=batch[i] if i < len(batch) else "",
                                embedding=embedding,
                            )
                        )
                    break
                except Exception as e:
                    if attempt >= self.max_retries:
                        raise Exception(f"Error generating OpenAI embeddings after {attempt} attempts: {e}")
                    # Exponential backoff
                    import asyncio as _asyncio
                    await _asyncio.sleep(self.retry_backoff_s * attempt)

        return results

    def invoke(
        self, text: str | list[str] | DocumentSchema | list[DocumentSchema], *args, **kwargs
    ) -> list[DocumentWithEmbedding]:
        """Synchronous wrapper for ainvoke (not recommended, use ainvoke)"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.ainvoke(text, *args, **kwargs))
        except RuntimeError:
            # Если нет event loop, создаем новый
            return asyncio.run(self.ainvoke(text, *args, **kwargs))