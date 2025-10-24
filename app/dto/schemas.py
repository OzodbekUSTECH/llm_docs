from typing import Any, Literal, Optional
from llama_index.core.schema import Document as BaseDocument
from pydantic import Field

class DocumentSchema(BaseDocument):
    """
    Base document class, mostly inherited from Document class from llama-index.

    This class accept one positional argument `content` of an arbitrary type, which will
        store the raw content of the document. If specified, the class will use
        `content` to initialize the base llama_index class.

    Attributes:
        content: raw content of the document, can be anything
        source: id of the source of the Document. Optional.
        channel: the channel to show the document. Optional.:
            - chat: show in chat message
            - info: show in information panel
            - index: show in index panel
            - debug: show in debug panel
    """

    content: Any = None
    source: Optional[str] = None
    channel: Optional[Literal["chat", "info", "index", "debug", "plot", "highlight"]] = None

    def __init__(self, content: Optional[Any] = None, *args, **kwargs):
        if content is None:
            if kwargs.get("text", None) is not None:
                kwargs["content"] = kwargs["text"]
            elif kwargs.get("embedding", None) is not None:
                kwargs["content"] = kwargs["embedding"]
                # default text indicating this document only contains embedding
                kwargs["text"] = "<EMBEDDING>"
        elif isinstance(content, DocumentSchema):
            # TODO: simplify the Document class
            temp_ = content.dict()
            temp_.update(kwargs)
            kwargs = temp_
        else:
            kwargs["content"] = content
            if content:
                kwargs["text"] = str(content)
            else:
                kwargs["text"] = ""
        super().__init__(*args, **kwargs)

    def __bool__(self):
        return bool(self.content)

    def __str__(self):
        return str(self.content)


class DocumentWithEmbedding(DocumentSchema):
    """Subclass of Document which must contains embedding

    Use this if you want to enforce component's IOs to must contain embedding.
    """

    def __init__(self, embedding: list[float], *args, **kwargs):
        kwargs["embedding"] = embedding
        super().__init__(*args, **kwargs)
        
        
class RetrievedDocument(DocumentSchema):
    """Subclass of Document with retrieval-related information

    Attributes:
        score (float): final hybrid score of the document (from 0.0 to 1.0)
        semantic_score (float): semantic similarity score from vector search
        keyword_score (float): keyword relevance score from BM25
        retrieval_metadata (dict): metadata from the retrieval process, can be used
            by different components in a retrieved pipeline to communicate with each
            other
    """

    score: float = Field(default=0.0)
    semantic_score: float = Field(default=0.0)
    keyword_score: float = Field(default=0.0)
    retrieval_metadata: dict = Field(default={})