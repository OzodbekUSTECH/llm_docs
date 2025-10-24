# fastembed_bge_embeddings.py
from typing import TYPE_CHECKING, Optional
from app.dto.schemas import DocumentWithEmbedding, DocumentSchema

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